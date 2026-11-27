"""Inter-phase patient motion QC for DCE-MRI breast exams.

WHY
---
A DCE acquisition is several 3D phases of the same breast minutes apart. Everything
downstream assumes those phases are spatially aligned: a kinetic curve applies ONE lesion
mask voxel-wise to the pre-contrast volume and to every post-contrast volume, so if the
patient shifted 5 mm between phase 0 and phase 3, the ROI samples different tissue at the
end of the curve than at the start and the enhancement it reports is quietly wrong.
Subtraction images degrade the same way.

This module measures that motion so callers can flag it. It ANNOTATES and never corrects:
it does not resample, does not modify images, and a `fail` verdict is a label, not a gate.

HOW
---
Each post-contrast phase is rigidly registered to the pre-contrast reference phase, and the
score is the displacement that transform induces over the breast ROI, in millimetres.

Rigid, not deformable: a deformable registration can hallucinate deformation to explain
contrast enhancement and report motion on a patient who never moved, which is exactly the
wrong failure mode for a QC signal. Mutual information, not least squares: the phases
differ in intensity BY DESIGN (that is what contrast enhancement is), so an intensity-
difference metric would chase the enhancement instead of the anatomy.

The score is the mean displacement over ROI voxels rather than the raw translation norm,
because a rotation about the patient centre translates the breast by millimetres while
contributing nothing to the translation vector.

See docs/adr/0001-rigid-registration-as-motion-metric.md.

WHAT THE CALLER OWNS
--------------------
This module never segments. The caller supplies the Registration ROI — the landmark
model's breast region at inference time, a dataset mask during curation — and the caller
decomposes an exam into Series:

* bilateral exam  -> one Series per breast side, each with that side's region label
* unilateral exam -> ONE Series over ALL region labels merged (`roi_label=None`)

That second rule is not cosmetic. A landmark model emits two "left"/"right" labels even on
a unilateral acquisition, where they are two fragments of the SAME breast; scoring a
fragment gives mutual information too little anatomy and the optimiser runs away, inventing
tens of millimetres of motion on a still patient. `series_units()` applies the rule.
See docs/adr/0004-merge-regions-on-unilateral-exams.md.

MEASUREMENT HERE, POLICY IN THE APPLICATION
-------------------------------------------
This module reports numbers and never verdicts. There is no pass/warn/fail in a
`SeriesScore` and no threshold applied to one: what counts as "too much motion" depends on
what the caller does with the exam, and a library that bakes in a cutoff forces every
consumer to live with one project's judgement. Applications threshold `score_mm`
themselves; `REFERENCE_WARN_MM` and friends are published as calibrated starting points,
never enforced.

`status` is therefore factual, not editorial: `ok` means a score was produced, `error`
means one could not be. QC must fail loudly — a registration that did not converge must
never be reported as a still patient — but "converged to 6 mm" is a measurement, not a
failure, and this module leaves the judging to the caller.

The one thing a caller MUST look at alongside `score_mm` is `roi_cm3`. Degenerate ROIs
(slivers, edge fragments) are the dominant source of large fictional scores, and they come
from upstream segmentation failure rather than from the registration. This module reports
the ROI size and lets the application decide what is too small.
See docs/adr/0002-roi-plausibility-gate.md.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)

# Reference values calibrated on MAMA-MIA, published for applications that want a starting
# point. NOTHING in this module applies them; see docs/adr/0003-calibration-on-mama-mia.md.
# WARN sits ~100x the measured noise floor (0.02 mm) and near slice thickness; FAIL sits at
# the scale of a small lesion's radius, past which a kinetic curve samples other tissue.
REFERENCE_WARN_MM = 2.0
REFERENCE_FAIL_MM = 5.0

# A real breast side is hundreds of cm3 (MAMA-MIA median 830). An ROI far below this is a
# segmentation failure rather than a breast, and its score will be large and fictional.
# Reported, never enforced — the application chooses where to draw the line.
REFERENCE_MIN_ROI_CM3 = 100.0

# Left-right field of view below which an exam is treated as unilateral. Mirrors the
# primary rule in `mri_transforms.utils.infer_bilateral_view`.
BILATERAL_FOV_MM = 260.0

STATUS_OK, STATUS_ERROR = "ok", "error"

_PHASE_RE = re.compile(r"_(\d{4})\.nii(\.gz)?$")

# Every knob the registration depends on, in one place and written into every result. A score
# is only reproducible if the settings that produced it travel with it.
REGISTRATION_SETTINGS: Dict[str, Any] = {
    "transform": "Euler3D (rigid: 3 rotations + 3 translations)",
    "metric": "MattesMutualInformation",
    "metric_histogram_bins": 50,
    "metric_sampling_strategy": "RANDOM",
    "metric_sampling_percentage": 0.10,
    "interpolator": "linear",
    "optimizer": "RegularStepGradientDescent",
    "optimizer_learning_rate": 1.0,
    "optimizer_min_step": 1e-4,
    "optimizer_max_iterations": 200,
    "optimizer_gradient_tolerance": 1e-6,
    "optimizer_scales": "from_physical_shift",
    "shrink_factors_per_level": [4, 2, 1],
    "smoothing_sigmas_per_level": [2, 1, 0],
    "smoothing_sigmas_in_physical_units": True,
    "initializer": "CenteredTransformInitializer(GEOMETRY)",
    "score_definition": "mean over ROI voxels of ||T(p) - p||, in mm; "
                        "series score = max over phase pairs",
}


@dataclass
class PhasePairScore:
    """Displacement of one post-contrast phase relative to the reference phase.

    Holds enough to reconstruct the registration: the transform itself (not merely its
    magnitudes), the optimizer's path to it, and why it stopped.
    """
    phase: int
    mean_mm: float
    p95_mm: float
    max_mm: float
    translation_mm: float
    rotation_deg: float
    metric: float
    converged: bool
    moving_file: str = ""
    # The transform itself. `translation_mm` and `rotation_deg` above are magnitudes and cannot
    # be inverted back into a transform; these can, so a result can be re-applied or checked.
    rotation_rad_xyz: List[float] = field(default_factory=list)
    translation_xyz_mm: List[float] = field(default_factory=list)
    rotation_center_xyz: List[float] = field(default_factory=list)
    rotation_matrix: List[float] = field(default_factory=list)
    # Metric at the very first iteration. NOT comparable with `metric` (the final value):
    # they come from different pyramid levels, and mutual information is scaled per level.
    # Use `metric_by_level` to judge whether the optimizer actually improved anything.
    initial_metric: Optional[float] = None
    # Per resolution level: {"level", "iterations", "metric_start", "metric_end", "improvement"}.
    # This is the honest view of "did it converge" — MI must fall WITHIN a level, and a level
    # that ran many iterations for no improvement is the signature of a registration that
    # wandered rather than locked on.
    metric_by_level: List[Dict[str, Any]] = field(default_factory=list)
    iterations: Optional[int] = None
    # (level, iteration, metric) per optimizer step. The single most diagnostic field here:
    # a runaway registration and a clean one look identical in the final number and completely
    # different in this trajectory.
    trajectory: List[List[float]] = field(default_factory=list)
    # Wall-clock for this one registration. Recorded per pair because cost scales with volume
    # size and iteration count, so a slow exam is diagnosable without re-running it.
    duration_seconds: float = 0.0
    stop_condition: str = ""


@dataclass
class SeriesScore:
    """The measurement for one Series (one ordered set of phases + one ROI).

    Carries no verdict. `score_mm` is the number; `roi_cm3` is the context the caller
    needs to judge whether to trust it; `status` says only whether a score exists.
    """
    status: str
    score_mm: Optional[float] = None
    n_phases: int = 0
    worst_phase: Optional[int] = None
    roi_provenance: str = "none"
    roi_cm3: Optional[float] = None
    reference_phase: Optional[str] = None
    reference_substituted: bool = False
    pairs: List[PhasePairScore] = field(default_factory=list)
    note: str = ""
    # ---- provenance: what was measured, over what, with which settings -------------------
    phase_files: List[str] = field(default_factory=list)
    roi_file: Optional[str] = None
    roi_label: Optional[int] = None
    roi_voxels: Optional[int] = None
    # Displacement is sampled over a subset of ROI voxels; both numbers are recorded because
    # the statistic is only reproducible if you know how many points it averaged over.
    roi_voxels_sampled: Optional[int] = None
    seed: Optional[int] = None
    # Geometry of the reference phase. Without spacing and direction, a displacement in mm
    # cannot be checked against the voxel grid it came from.
    reference_geometry: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_utc: Optional[str] = None
    total_duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pairs"] = [asdict(p) for p in self.pairs]
        return d


def discover_phases(exam_dir: str, case: Optional[str] = None) -> List[str]:
    """Phase files in acquisition order, by the nnU-Net `{case}_0000.nii.gz` convention."""
    files = [f for f in glob.glob(os.path.join(exam_dir, "*.nii*")) if _PHASE_RE.search(f)]
    if case:
        files = [f for f in files if os.path.basename(f).startswith(case + "_")]
    return sorted(files, key=lambda f: int(_PHASE_RE.search(f).group(1)))


def fov_lr_mm(image_path: str) -> float:
    """Left-right field of view in mm, whichever voxel axis carries it."""
    im = sitk.ReadImage(image_path)
    direction = np.array(im.GetDirection()).reshape(3, 3)
    size, spacing = np.array(im.GetSize()), np.array(im.GetSpacing())
    lr_axis = int(np.argmax(np.abs(direction[0, :])))  # world x row -> voxel axis
    return float(size[lr_axis] * spacing[lr_axis])


def is_bilateral(image_path: str, fov_min_mm: float = BILATERAL_FOV_MM) -> bool:
    """Whether the acquisition covers both breasts.

    Prefer `mri_transforms.utils.infer_bilateral_view` when its symmetry fallback matters;
    this is the FOV rule alone, kept dependency-light so motion QC can run standalone.
    Erring toward "unilateral" is the safe direction: merging a truly bilateral exam only
    costs per-side granularity, while splitting a unilateral one manufactures motion.
    """
    fov = fov_lr_mm(image_path)
    return fov >= max(fov_min_mm, 0.8 * fov)


def series_units(roi_path: str, side_labels: Dict[str, int],
                 bilateral: bool) -> List[Tuple[str, Optional[int]]]:
    """Decompose an exam into Series: `[(name, roi_label)]`.

    Bilateral -> one unit per side. Unilateral -> a single `merged` unit whose label is
    None, meaning "every non-zero label in the ROI volume".
    """
    if not bilateral:
        return [("merged", None)]
    return [(side, label) for side, label in sorted(side_labels.items()) if label is not None]


def roi_volume_cm3(roi_path: str, roi_label: Optional[int] = None) -> float:
    """Volume of the ROI, in cm3, honouring the image's real spacing."""
    im = sitk.ReadImage(roi_path)
    arr = sitk.GetArrayViewFromImage(im)
    selected = (arr > 0) if roi_label is None else (arr == roi_label)
    return float(selected.sum() * np.prod(im.GetSpacing()) / 1000.0)


def _rigid_register(fixed: sitk.Image, moving: sitk.Image, roi: Optional[sitk.Image],
                    seed: int, record_trajectory: bool = True) -> Dict[str, Any]:
    """Register `moving` onto `fixed` and return the transform WITH its full audit trail."""
    cfg = REGISTRATION_SETTINGS
    reg = sitk.ImageRegistrationMethod()
    # Mutual information: the phases differ in intensity by design (contrast enhancement),
    # so the metric must not treat that difference as misalignment.
    reg.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=cfg["metric_histogram_bins"])
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(cfg["metric_sampling_percentage"], seed)
    if roi is not None:
        # Sample only inside the breast. Otherwise the thorax and heart, which move on
        # their own schedule, dominate the metric.
        reg.SetMetricFixedMask(roi)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=cfg["optimizer_learning_rate"], minStep=cfg["optimizer_min_step"],
        numberOfIterations=cfg["optimizer_max_iterations"],
        gradientMagnitudeTolerance=cfg["optimizer_gradient_tolerance"])
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(cfg["shrink_factors_per_level"])
    reg.SetSmoothingSigmasPerLevel(cfg["smoothing_sigmas_per_level"])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(
        sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY),
        inPlace=False)
    # Watch the optimizer rather than only its verdict: the metric path separates a
    # registration that walked steadily downhill from one that bolted, and the two produce
    # indistinguishable final numbers.
    trajectory: List[List[float]] = []

    def _observe():
        trajectory.append([int(reg.GetCurrentLevel()), int(reg.GetOptimizerIteration()),
                           round(float(reg.GetMetricValue()), 6)])

    reg.AddCommand(sitk.sitkIterationEvent, _observe)
    transform = reg.Execute(fixed, moving)
    return {
        "transform": transform,
        "stop_condition": reg.GetOptimizerStopConditionDescription(),
        "metric": reg.GetMetricValue(),
        "iterations": int(reg.GetOptimizerIteration()),
        # The per-level summary is always kept; the full trace is optional because it
        # dominates the file size.
        "trajectory": trajectory if record_trajectory else [],
        "initial_metric": trajectory[0][2] if trajectory else None,
        "metric_by_level": _summarise_levels(trajectory),
    }


def _summarise_levels(trajectory: List[List[float]]) -> List[Dict[str, Any]]:
    """Collapse the optimizer trace to one row per resolution level.

    Mutual information is scaled per level, so first-vs-last across the whole run compares
    two different quantities. Improvement is only meaningful within a level.
    """
    by_level: Dict[int, List[float]] = {}
    for level, _iteration, metric in trajectory:
        by_level.setdefault(int(level), []).append(metric)
    out = []
    for level in sorted(by_level):
        values = by_level[level]
        out.append({"level": level, "iterations": len(values),
                    "metric_start": values[0], "metric_end": values[-1],
                    # Negative MI: falling is improving, so a positive number here is progress.
                    "improvement": round(values[0] - values[-1], 6)})
    return out


def _rigid_parts(transform: sitk.Transform) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap the registration result into (rotation matrix, centre, translation)."""
    if isinstance(transform, sitk.CompositeTransform):
        transform = transform.GetNthTransform(0)
    euler = sitk.Euler3DTransform(transform)
    return (np.array(euler.GetMatrix()).reshape(3, 3),
            np.array(euler.GetCenter()),
            np.array(euler.GetTranslation()))


def _physical_points(reference: sitk.Image, voxel_idx: Tuple[np.ndarray, ...]) -> np.ndarray:
    """Vectorised index->physical-point mapping for ROI voxels.

    `voxel_idx` comes from numpy (k, j, i); SimpleITK indexes (i, j, k).
    """
    ijk = np.stack([voxel_idx[2], voxel_idx[1], voxel_idx[0]], axis=1).astype(np.float64)
    origin = np.array(reference.GetOrigin())
    spacing = np.array(reference.GetSpacing())
    direction = np.array(reference.GetDirection()).reshape(3, 3)
    return origin + (direction @ (ijk * spacing).T).T


def _displacement_mm(transform: sitk.Transform, reference: sitk.Image,
                     voxel_idx: Tuple[np.ndarray, ...]) -> Tuple[float, float, float]:
    """Mean, p95 and max displacement the transform induces over the ROI, in mm."""
    points = _physical_points(reference, voxel_idx)
    rotation, centre, translation = _rigid_parts(transform)
    moved = (rotation @ (points - centre).T).T + centre + translation
    d = np.linalg.norm(moved - points, axis=1)
    return float(d.mean()), float(np.percentile(d, 95)), float(d.max())


def tier_for(score_mm: float,
             warn_mm: float = REFERENCE_WARN_MM,
             fail_mm: float = REFERENCE_FAIL_MM) -> str:
    """Optional convenience for applications that want the calibrated cutoffs.

    Nothing in this module calls it. It exists so a caller can reuse the MAMA-MIA
    reference values without copying magic numbers, and it takes both cutoffs as
    arguments precisely so a caller with different needs can pass its own.
    """
    if score_mm >= fail_mm:
        return "fail"
    if score_mm >= warn_mm:
        return "warn"
    return "pass"


def score_series(phase_files: Sequence[str],
                 roi_path: Optional[str] = None,
                 roi_label: Optional[int] = None,
                 max_roi_voxels: int = 50000,
                 seed: int = 42,
                 record_trajectory: bool = True) -> SeriesScore:
    """Score one Series: phases in acquisition order, plus one Registration ROI.

    `phase_files[0]` is the reference. When no pre-contrast volume exists the caller
    passes the earliest available phase; `reference_substituted` records that, mirroring
    how the inference pipeline already substitutes the first post-contrast volume.

    `roi_label=None` with an ROI path means "all non-zero labels" — the merged ROI a
    unilateral exam requires.

    Every ROI is measured, however small, and the result always reports `roi_cm3` beside
    `score_mm`. Judging a tiny ROI's score untrustworthy is the application's call, and it
    needs the number to make it.

    `record_trajectory` keeps the optimizer's metric at every iteration. It is the most
    diagnostic field in the result and roughly 80% of its serialised size; turn it off for
    large batches, where `metric_by_level` still summarises convergence per level.
    """
    started_all = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if len(phase_files) < 2:
        return SeriesScore(status=STATUS_ERROR, note="fewer than two phases",
                           phase_files=list(phase_files), created_utc=stamp)

    reference = sitk.ReadImage(phase_files[0], sitk.sitkFloat32)
    roi_image, provenance, roi_cm3 = None, "none", None

    if roi_path:
        roi_raw = sitk.ReadImage(roi_path)
        roi_arr = sitk.GetArrayFromImage(roi_raw)
        selected = (roi_arr > 0) if roi_label is None else (roi_arr == roi_label)
        roi_cm3 = float(selected.sum() * np.prod(roi_raw.GetSpacing()) / 1000.0)
        if not selected.any():
            return SeriesScore(status=STATUS_ERROR, roi_provenance="empty", roi_cm3=0.0,
                               roi_file=roi_path, roi_label=roi_label, roi_voxels=0,
                               phase_files=list(phase_files), created_utc=stamp,
                               note=f"ROI label {roi_label} has no voxels")
        if roi_arr.shape != sitk.GetArrayViewFromImage(reference).shape:
            return SeriesScore(status=STATUS_ERROR, roi_provenance="breast_region",
                               roi_cm3=roi_cm3, roi_file=roi_path, roi_label=roi_label,
                               phase_files=list(phase_files), created_utc=stamp,
                               note=(f"ROI grid {roi_arr.shape} does not match the reference "
                                     f"phase {sitk.GetArrayViewFromImage(reference).shape}"))
        roi_image = sitk.GetImageFromArray(selected.astype(np.uint8))
        roi_image.CopyInformation(reference)
        provenance = "breast_region"
        voxel_idx = np.nonzero(selected)
    else:
        voxel_idx = np.nonzero(np.ones(sitk.GetArrayViewFromImage(reference).shape, bool))

    # Displacement varies smoothly across a rigid field, so a subsample of ROI voxels
    # gives the same statistics for a fraction of the memory.
    n_voxels = len(voxel_idx[0])
    roi_voxels_total = n_voxels
    if n_voxels > max_roi_voxels:
        keep = np.random.default_rng(seed).choice(n_voxels, max_roi_voxels, replace=False)
        voxel_idx = tuple(axis[keep] for axis in voxel_idx)

    pairs: List[PhasePairScore] = []
    failures: List[str] = []
    for phase, moving_path in enumerate(phase_files[1:], start=1):
        try:
            pair_started = time.perf_counter()
            moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
            res = _rigid_register(reference, moving, roi_image, seed, record_trajectory)
            transform = res["transform"]
            mean_mm, p95_mm, max_mm = _displacement_mm(transform, reference, voxel_idx)
            rotation, centre, translation = _rigid_parts(transform)
            angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
            euler = transform.GetNthTransform(0) if isinstance(
                transform, sitk.CompositeTransform) else transform
            euler_params = list(sitk.Euler3DTransform(euler).GetParameters())
            pairs.append(PhasePairScore(
                phase=phase, mean_mm=mean_mm, p95_mm=p95_mm, max_mm=max_mm,
                translation_mm=float(np.linalg.norm(translation)),
                rotation_deg=float(angle), metric=float(res["metric"]), converged=True,
                moving_file=os.path.basename(moving_path),
                rotation_rad_xyz=[float(v) for v in euler_params[0:3]],
                translation_xyz_mm=[float(v) for v in translation],
                rotation_center_xyz=[float(v) for v in centre],
                rotation_matrix=[float(v) for v in rotation.ravel()],
                initial_metric=res["initial_metric"], iterations=res["iterations"],
                metric_by_level=res["metric_by_level"], trajectory=res["trajectory"],
                duration_seconds=round(time.perf_counter() - pair_started, 3),
                stop_condition=res["stop_condition"]))
        except Exception as exc:                                  # noqa: BLE001
            log.warning("motion_qc: phase %d failed to register: %s", phase, exc)
            failures.append(f"phase {phase}: {type(exc).__name__}")

    geometry = {
        "size": list(reference.GetSize()),
        "spacing_mm": [round(v, 6) for v in reference.GetSpacing()],
        "origin": [round(v, 6) for v in reference.GetOrigin()],
        "direction": [round(v, 6) for v in reference.GetDirection()],
    }
    common = dict(roi_provenance=provenance, roi_cm3=roi_cm3, roi_file=roi_path,
                  roi_label=roi_label, roi_voxels=roi_voxels_total,
                  roi_voxels_sampled=len(voxel_idx[0]), seed=seed,
                  phase_files=[os.path.abspath(f) for f in phase_files],
                  reference_geometry=geometry, settings=dict(REGISTRATION_SETTINGS),
                  created_utc=stamp,
                  total_duration_seconds=round(time.perf_counter() - started_all, 3))

    if not pairs:
        return SeriesScore(status=STATUS_ERROR, n_phases=len(phase_files),
                           note="; ".join(failures), **common)

    worst = max(pairs, key=lambda p: p.mean_mm)
    # A partial result is not a measurement: a phase we could not register might be the one
    # that moved, so the Series is an error even though the rest scored.
    return SeriesScore(
        status=STATUS_ERROR if failures else STATUS_OK,
        score_mm=worst.mean_mm, n_phases=len(phase_files), worst_phase=worst.phase,
        reference_phase=os.path.basename(phase_files[0]),
        pairs=pairs, note="; ".join(failures), **common)


def score_exam(phase_files: Sequence[str],
               roi_path: Optional[str] = None,
               side_labels: Optional[Dict[str, int]] = None,
               bilateral: Optional[bool] = None,
               **kwargs) -> Dict[str, SeriesScore]:
    """Score a whole exam, decomposing it into Series by laterality.

    Returns `{series_name: SeriesScore}` — one entry per side for a bilateral exam, a
    single `merged` entry for a unilateral one. Pass the result to `exam_motion_score`
    to reduce it to one number.
    """
    if not phase_files:
        return {}
    if roi_path is None or not side_labels:
        return {"merged": score_series(phase_files, roi_path, None, **kwargs)}
    if bilateral is None:
        bilateral = is_bilateral(phase_files[0])
    return {name: score_series(phase_files, roi_path, label, **kwargs)
            for name, label in series_units(roi_path, side_labels, bilateral)}


def exam_motion_score(series: Dict[str, SeriesScore]) -> Tuple[Optional[float], int]:
    """Reduce an exam's Series to `(score_mm, n_unscored)`. Still no verdict.

    Worst-wins: the exam is as moved as its worst-measured side.

    `n_unscored` is not an afterthought. A bilateral exam whose left side scored while its
    right errored is NOT a fully checked exam, and a bare score would overstate what was
    measured — so the count comes back with it and the application decides what a
    partially measured exam is worth. `None` means nothing could be measured at all.
    """
    scored = [s for s in series.values() if s.score_mm is not None]
    unscored = len(series) - len(scored)
    if not scored:
        return None, unscored
    return max(s.score_mm for s in scored), unscored
