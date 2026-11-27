"""Measure MRI acquisition geometry from image headers."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

# Anatomical axis code -> the axis it names. `nib.aff2axcodes` returns the code for the
# direction of INCREASING index, so 'L' and 'R' are the same physical axis.
_AXIS_OF_CODE = {"L": "LR", "R": "LR", "A": "AP", "P": "AP", "S": "SI", "I": "SI"}

# The plane an acquisition is named after is the one its slices LIE IN, which is the plane
# perpendicular to the through-plane axis.
_PLANE_OF_THROUGH_AXIS = {"SI": "axial", "AP": "coronal", "LR": "sagittal"}

REFERENCE_BILATERAL_FOV_MM = 260.0

REFERENCE_MIN_SIDE_MM = 98.0

REFERENCE_ABSTAIN_MM = 20.0

# Tilt from the anatomical frame beyond which no axis-aligned extent is a true extent: every
# FOV above becomes the width of a bounding box around a tilted volume, which overstates it.
# Not calibrated -- it is the angle at which `aff2axcodes` is rounding rather than reporting.
# Published here because three decisions already read it (scope certainty, the per-breast split's
# structural-safety test, and stage-1's review flag) and they must not be able to disagree.
REFERENCE_OBLIQUE_DEG = 5.0


@dataclass
class AxisGeometry:
    """What one volume's header says about its geometry."""

    path: str
    shape: Tuple[int, int, int]
    zooms: Tuple[float, float, float]
    axcodes: str                      # e.g. "LAI"
    plane: str                        # axial | coronal | sagittal | unknown (axcodes[2])
    plane_by_thickest_axis: str       # cross-check only; loses 34/1506 on MAMA-MIA
    plane_agrees: bool                # False => near-isotropic, no dominant slice axis
    through_axis: str                 # LR | AP | SI, from the PRIMARY rule
    through_axis_index: int
    n_slices: int
    n_volumes: int                    # 4th dimension; >1 means the file holds several volumes
    slice_spacing_mm: float
    fov_lr_mm: float
    fov_ap_mm: float
    fov_si_mm: float
    lr_min_mm: float                  # world left-right bounds, midline (isocentre) = 0
    lr_max_mm: float
    min_side_mm: float                # coverage on the WEAKER side of the midline
    axes_touching_zero: int           # axes with a bound on 0 -- a corner origin, not a patient
    ap_straddles_zero: bool           # a real patient frame is centred front-to-back
    frame_usable: bool                # is world x = 0 credibly the patient midline?
    oblique_deg: float                # how far the axes tilt from the anatomical frame
    error: Optional[str] = None

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["shape"] = "x".join(str(s) for s in self.shape)
        d["zooms"] = "x".join(f"{z:.3f}" for z in self.zooms)
        return d


def _oblique_degrees(affine: np.ndarray) -> float:
    """The largest angle between a voxel axis and the anatomical axis it maps onto.

    Zero for an axis-aligned volume. A large value means the axis codes are a rounding of
    a tilted acquisition, so every FOV below is a bounding box rather than a true extent.
    """
    direction = affine[:3, :3]
    norms = np.linalg.norm(direction, axis=0)
    norms[norms == 0] = 1.0
    unit = direction / norms
    worst = 0.0
    for col in range(3):
        dominant = float(np.max(np.abs(unit[:, col])))
        worst = max(worst, math.degrees(math.acos(min(1.0, dominant))))
    return round(worst, 2)


def axis_geometry(nifti_path: str) -> AxisGeometry:
    """Read one volume's header and describe its geometry. Never raises."""
    blank = dict(shape=(0, 0, 0), zooms=(0.0, 0.0, 0.0), axcodes="", plane="unknown",
                 plane_by_thickest_axis="unknown", plane_agrees=False, through_axis="",
                 through_axis_index=-1, n_slices=0, n_volumes=1, slice_spacing_mm=0.0,
                 fov_lr_mm=0.0, fov_ap_mm=0.0, fov_si_mm=0.0, lr_min_mm=0.0,
                 lr_max_mm=0.0, min_side_mm=0.0, axes_touching_zero=0,
                 ap_straddles_zero=False, frame_usable=False, oblique_deg=0.0)
    try:
        img = nib.load(nifti_path)
    except Exception as exc:                                    # unreadable file
        return AxisGeometry(path=nifti_path, error=f"load failed: {exc}", **blank)

    try:
        shape = tuple(int(s) for s in img.shape[:3])
        # A converter can emit 4D. Usually the 4th axis is TIME and the spatial extent is
        # intact, but dicom2nifti sometimes splits one slice stack across it instead --
        # ACRIN-182 "stir sag" has 68 sagittal slices spanning 306 mm written as
        # (512, 512, 34, 2), so shape[:3] sees half the anatomy. Record it so the FOV can
        # be distrusted where it matters; the header alone cannot say which case this is.
        n_volumes = int(img.shape[3]) if len(img.shape) > 3 else 1
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        codes = nib.aff2axcodes(img.affine)
        axcodes = "".join(codes)

        # Physical extent along each anatomical axis. A voxel axis contributes its whole
        # extent to the anatomical axis its code names.
        fov = {"LR": 0.0, "AP": 0.0, "SI": 0.0}
        for i, code in enumerate(codes):
            named = _AXIS_OF_CODE.get(code)
            if named:
                fov[named] = round(shape[i] * zooms[i], 1)

        # Where the left-right window SITS, not just how wide it is. The affine maps voxel
        # indices to world mm, and world x = 0 is the scanner isocentre, which is the
        # patient midline for a breast exam. Take all 8 corners so an oblique volume is
        # bounded correctly rather than measured along one edge.
        corners = np.array([img.affine @ np.array(
            [i * (shape[0] - 1), j * (shape[1] - 1), k * (shape[2] - 1), 1.0])
            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        xs = corners[:, 0]
        lr_min_mm, lr_max_mm = round(float(xs.min()), 1), round(float(xs.max()), 1)
        # Reach on each side of the midline; the smaller one is what decides bilateral.
        min_side_mm = round(min(max(lr_max_mm, 0.0), max(-lr_min_mm, 0.0)), 1)

        axes = (xs, corners[:, 1], corners[:, 2])
        axes_touching_zero = sum(
            1 for ax in axes if min(abs(float(ax.min())), abs(float(ax.max()))) <= 1.0)
        ap = corners[:, 1]
        ap_straddles_zero = bool(ap.min() < 0.0 < ap.max())
        frame_usable = bool(ap_straddles_zero and axes_touching_zero < 2)

        # Primary rule: the slices were stored last. See the module docstring for the
        # measurement that put this rule first.
        through_idx = 2
        through_axis = _AXIS_OF_CODE.get(codes[through_idx], "")
        plane = _PLANE_OF_THROUGH_AXIS.get(through_axis, "unknown")

        # Cross-check: thickest voxel wins, ties broken by fewest samples then last axis.
        thickest = sorted(range(3), key=lambda i: (-zooms[i], shape[i], -i))[0]
        plane_by_thickest_axis = _PLANE_OF_THROUGH_AXIS.get(
            _AXIS_OF_CODE.get(codes[thickest], ""), "unknown")

        return AxisGeometry(
            path=nifti_path, shape=shape, zooms=zooms, axcodes=axcodes,
            plane=plane, plane_by_thickest_axis=plane_by_thickest_axis,
            plane_agrees=(plane == plane_by_thickest_axis),
            through_axis=through_axis, through_axis_index=through_idx,
            n_slices=shape[through_idx], n_volumes=n_volumes,
            slice_spacing_mm=round(zooms[through_idx], 3),
            fov_lr_mm=fov["LR"], fov_ap_mm=fov["AP"], fov_si_mm=fov["SI"],
            lr_min_mm=lr_min_mm, lr_max_mm=lr_max_mm, min_side_mm=min_side_mm,
            axes_touching_zero=axes_touching_zero, ap_straddles_zero=ap_straddles_zero,
            frame_usable=frame_usable,
            oblique_deg=_oblique_degrees(img.affine),
        )
    except Exception as exc:
        return AxisGeometry(path=nifti_path, error=f"header parse failed: {exc}", **blank)


def scope_for(geom: AxisGeometry,
              fov_cut_mm: float = REFERENCE_BILATERAL_FOV_MM,
              abstain_mm: float = REFERENCE_ABSTAIN_MM,
              min_side_mm: float = REFERENCE_MIN_SIDE_MM) -> Tuple[str, float, str, bool]:
    """Bilateral or unilateral, from the left-right field of view.

    Returns `(scope, margin_mm, reason, certain)`. `margin_mm` is `fov_lr_mm - fov_cut_mm`:
    positive for bilateral, and its ABSOLUTE VALUE is the distance to the decision.

    `certain` is False inside the abstention band, and a caller must not route an uncertain
    exam automatically. The verdict is still returned — abstention is a request for a second
    source, not a refusal to answer. Pass `abstain_mm=0` to disable the band and get the
    bare threshold behaviour.

    A field of view says how much of the patient the scanner covered, which is what
    "bilateral acquisition" means. It does not say a breast is present in the coverage,
    so a wide but empty volume still reads bilateral here. That is deliberate: the header
    rule is a screen, and `mri_analysis.symmetry` is what looks at anatomy.

    NOTE on `utils.infer_bilateral_view`: its cut is
    `max(260, 0.8 * fov_lr)`, and since `fov_lr >= 0.8 * fov_lr` always holds, the second
    term never binds — the effective rule there is the plain 260 mm used here.
    """
    if geom.error or geom.fov_lr_mm <= 0:
        return "unknown", 0.0, geom.error or "no left-right extent in the header", False
    margin = round(geom.fov_lr_mm - fov_cut_mm, 1)
    # TWO conditions, both necessary. Width says the window is big enough for two breasts;
    # centring says the window actually straddles the midline instead of sitting over one
    # side plus an axilla. Width alone got ACRIN-153 "AX FSPGR THRU AXILLA" wrong: 260 mm
    # across, but 242 mm of it on one side of the midline and 17 mm on the other.
    wide_enough = margin >= 0
    # The centring test only means anything if world x = 0 really is the patient midline,
    # and NOTHING IN A NIfTI HEADER STATES THAT. `frame_usable` tests the frame for
    # plausibility instead of assuming it (corner origin, AP centring) -- see
    # `axis_geometry`. Where it fails, fall back to width alone rather than assert: the
    # same abstention the pipeliner's `_coverage_from_geometry` makes when it requires
    # x_min <= -20.
    #
    # KNOWN RESIDUAL: a pure LEFT-RIGHT translation of an otherwise valid affine is
    # undetectable from the header. It would shift `min_side_mm` and could flip a wide
    # exam to unilateral while every plausibility signal still passes. That is why a
    # verdict which DEPENDS on the centring test never comes back certain (below).
    origin_usable = geom.frame_usable
    centred_enough = geom.min_side_mm >= min_side_mm
    scope = ("bilateral" if (wide_enough and (centred_enough or not origin_usable))
             else "unilateral")
    reason = (f"fov_lr={geom.fov_lr_mm:.1f}mm vs cut={fov_cut_mm:.1f}mm; "
              f"min_side={geom.min_side_mm:.1f}mm vs {min_side_mm:.0f}mm")
    if not origin_usable:
        reason += ("; affine origin sits on the FOV edge, midline unknown -> "
                   "centring test skipped")
    elif wide_enough and not centred_enough:
        reason += "; wide but off-centre -> one breast plus axilla"
    certain = abs(margin) > abstain_mm
    if not certain:
        reason += f"; within +/-{abstain_mm:.0f}mm abstention band"
    # The centring cut sits in an 8.6 mm gap calibrated on two positive cases, so an exam
    # landing near it is exactly as untrustworthy as one near the width cut.
    # A 4D file holds several volumes, and the header cannot say WHICH kind:
    #   * TIME  -- the same slices imaged repeatedly (a DCE series fused into one file).
    #     shape[:3] is the true anatomy and fov_lr is CORRECT.
    #   * SPLIT -- one slice stack divided across the 4th axis. shape[:3] is HALF the
    #     anatomy and fov_lr is understated (ACRIN-182 "stir sag": 149.6 mm, really 306 mm).
    # Telling them apart needs the DICOM slice positions (do they repeat?), which this
    # module does not read. Measured on the pipeliner output: all 9 4D volumes were TIME,
    # and the one SPLIT case came from a different converter path -- so the common case is
    # the harmless one. We still abstain, because being wrong here is silent.
    if geom.n_volumes > 1:
        certain = False
        reason += (f"; 4D file ({geom.n_volumes} volumes) -- cannot tell repeated phases "
                   f"from a split slice stack without the DICOM slice positions")
    if not origin_usable:
        certain = False
    elif abs(geom.min_side_mm - min_side_mm) <= abstain_mm:
        certain = False
        reason += f"; min_side within +/-{abstain_mm:.0f}mm of the centring cut"
    # A verdict the centring test CHANGED rests on the midline assumption, which the header
    # cannot establish. Report it, never trust it unattended.
    if wide_enough and not centred_enough and origin_usable:
        certain = False
        reason += "; verdict depends on the midline assumption"
    # An oblique acquisition has no true left-right extent — the FOV above is the extent of
    # a bounding box around a tilted volume, which overstates it. Never call that certain.
    if geom.oblique_deg > REFERENCE_OBLIQUE_DEG:
        certain = False
        reason += f"; oblique {geom.oblique_deg:.1f}deg, FOV is a bounding box"
    return scope, margin, reason, certain


def calibrate_fov_cut(fovs, labels) -> Dict[str, float]:
    """The left-right FOV cut that best separates labelled bilateral from unilateral.

    `labels` is truthy for bilateral. Returns the best cut, its accuracy, and the width of
    the gap it sits in — a wide gap means the constant is not delicate, a narrow one means
    the rule is riding on this dataset and needs the pixel fallback behind it.
    """
    pairs = sorted(zip(fovs, [bool(x) for x in labels]))
    if not pairs:
        return {}
    values = [v for v, _ in pairs]
    best = {"cut_mm": 0.0, "accuracy": 0.0, "gap_mm": 0.0}
    candidates = sorted({round((a + b) / 2, 1)
                         for a, b in zip(values, values[1:])} | {values[0] - 1, values[-1] + 1})
    for cut in candidates:
        correct = sum(1 for v, is_bi in pairs if (v >= cut) == is_bi)
        acc = correct / len(pairs)
        if acc > best["accuracy"]:
            below = max((v for v in values if v < cut), default=cut)
            above = min((v for v in values if v >= cut), default=cut)
            best = {"cut_mm": cut, "accuracy": round(acc, 4),
                    "gap_mm": round(above - below, 1)}
    return best
