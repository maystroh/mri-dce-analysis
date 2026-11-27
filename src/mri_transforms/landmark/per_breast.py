"""Per-breast landmark inference: cut an exam into Breast Halves, run each, fuse the results.

The model this serves (`3D_landmarks_per_breast`) consumes ONE breast per forward pass and its
heads are binary — they make no left/right claim. Sidedness is therefore not predicted but
RESTORED, from which half was sent. See `CONTEXT.md` (Breast Half, Single-Breast Formulation)
and `docs/adr/0003-share-the-split-not-the-fusion.md`.

Two rules this module exists to enforce:

1. **The cut is shared with training.** `mri_transforms.breast_geometry.split_image` is the only
   definition of it, called by `prepare_per_breast.py` at training time and here at inference
   time. A cut that disagrees with training is invisible in the outputs — the masks still look
   like masks — so it is imported, never reimplemented.
2. **The fusion is not shared.** Each half's decided mask is put back on the parent grid with
   SimpleITK's `resample_like`, in one physical-space resample, rather than via the library's
   `restore_half` + `reconstruct_parent`. The ADR has the reasoning; the short version is that
   `reconstruct_parent` raises on an incomplete set, which is right for a training metric and
   wrong for inference, where one failed side should degrade rather than abort the exam.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from mri_transforms.utils import log

from .geometry import resample_like

COVERAGE_PER_BREAST = "per_breast"
COVERAGE_WHOLE_EXAM = "whole_exam"

SCOPE_BILATERAL = "bilateral"
SCOPE_UNILATERAL = "unilateral"

#: Label values the fused mask carries, matching what the whole-exam model emits so that
#: `assign_sides`, `region_boxes` and `split_by_regions` see exactly what they see today.
LABEL_FOR_SIDE = {"left": 1, "right": 2, "unilateral": 1}

FOLDER_HALVES = "per_breast_halves"

#: The head whose fused mask carries sidedness. Named once here rather than inline at the call
#: site: `registry.decide()` is explicit that behaviour is chosen by a head's own config "never by
#: the head's name", so the one place this code DOES depend on a name should be visible and should
#: fail with a diagnostic rather than a bare KeyError.
REGION_HEAD = "breast_region"

# Bump whenever fusion changes in a way that makes an existing on-disk per-breast mask stale.
# The runner records and checks this value so `reuse_existing=true` cannot bypass a fusion fix.
PER_BREAST_FUSION_VERSION = 2


@dataclass
class CoverageVerdict:
    """The per-breast input contract, kept separate from its supporting measurement.

    `scope` is retained as the model INPUT CONTRACT for manifest/cache compatibility: bilateral
    means two Breast Halves and unilateral means one full-width pass. `measured_scope` is what
    geometry said about Acquisition Scope. They differ when geometry abstains and no agreeing
    caller hint makes a split safe enough to execute.
    """

    scope: str
    certain: bool
    margin_mm: Optional[float]
    reason: str
    source: str                      #: "scope_for", "consensus", "abstention", or "override"
    measured_scope: Optional[str] = None
    hint_scope: Optional[str] = None

    @property
    def is_bilateral(self) -> bool:
        return self.scope == SCOPE_BILATERAL

    @property
    def execution(self) -> str:
        return "two_halves" if self.is_bilateral else "full_width"

    def as_dict(self) -> Dict[str, Any]:
        return {"scope": self.scope, "certain": self.certain, "margin_mm": self.margin_mm,
                "reason": self.reason, "source": self.source,
                "measured_scope": self.measured_scope, "hint_scope": self.hint_scope,
                "execution": self.execution}


def resolve_coverage(source_path: str, override: Optional[str] = None,
                     scope_hint: Optional[str] = None) -> CoverageVerdict:
    """Choose the landmark execution contract from geometry and an optional caller hint.

    Uses `mri_transforms.geometry.scope_for` rather than the older `utils.infer_bilateral_view`:
    it measures centring as well as width, checks that world x = 0 is credibly the patient
    midline before believing any centring measurement, and — the reason it matters here — it
    reports whether it is `certain`. The midline is exactly where this module is about to cut,
    so a volume whose affine origin was reset by de-identification is a volume we must not
    silently halve.

    A certain geometry verdict decides the contract. An uncertain bilateral verdict may still be
    acted on when it agrees with the caller's independently-owned pipeline laterality and the
    header is structurally safe to split. Structural uncertainty (4D, oblique or unusable affine
    frame), a disagreement or a missing hint abstains from the destructive action and runs one
    full-width pass. The raw measurement remains in `measured_scope`; abstention is never
    presented as a measured unilateral acquisition.
    """
    if scope_hint not in (None, SCOPE_BILATERAL, SCOPE_UNILATERAL):
        raise ValueError(f"scope hint must be {SCOPE_BILATERAL!r}, {SCOPE_UNILATERAL!r}, "
                         f"or None, got {scope_hint!r}")
    if override:
        if override not in (SCOPE_BILATERAL, SCOPE_UNILATERAL):
            raise ValueError(f"coverage override must be {SCOPE_BILATERAL!r} or "
                             f"{SCOPE_UNILATERAL!r}, got {override!r}")
        log.info(f"      [landmark] coverage={override!r} (explicit override)")
        return CoverageVerdict(override, True, None, "explicit override", "override",
                               measured_scope=None, hint_scope=scope_hint)

    try:
        from mri_transforms.geometry import (REFERENCE_OBLIQUE_DEG, axis_geometry,
                                              scope_for)
    except ImportError as exc:                                  # pragma: no cover - env guard
        raise ImportError(
            "per-breast landmark inference needs `mri_transforms.geometry` (axis_geometry, "
            "scope_for) from mri-transforms-latest. An older installed mri_transforms only has "
            "utils.infer_bilateral_view, which reports no confidence and cannot be used to "
            "decide whether to cut a volume in half.") from exc

    geom = axis_geometry(source_path)
    scope, margin_mm, reason, certain = scope_for(geom)

    # Threshold-band uncertainty (width or centring) may be resolved by an agreeing bilateral
    # caller, but uncertainty about the header structure itself must never be waived. This keeps
    # wide-but-off-centre acquisitions unilateral because `scope_for` measures those as
    # unilateral, while allowing a measured-bilateral exam near either threshold to use the same
    # midpoint split the downstream lesion pipeline already performs.
    structurally_safe_to_split = (
        not geom.error
        and geom.n_volumes <= 1
        and geom.frame_usable
        and geom.oblique_deg <= REFERENCE_OBLIQUE_DEG
    )
    consensus_can_split = (not certain and scope == SCOPE_BILATERAL
                           and scope_hint == SCOPE_BILATERAL
                           and structurally_safe_to_split)
    if consensus_can_split:
        consensus_reason = f"{reason}; uncertain geometry agrees with caller scope {scope_hint!r}"
        log.info(f"      [landmark] coverage={scope!r} by consensus (geometry is uncertain: "
                 f"{reason}; margin {margin_mm:.1f} mm; caller agrees); execution="
                 f"{'TWO half passes' if scope == SCOPE_BILATERAL else 'ONE full-width pass'}")
        return CoverageVerdict(scope, False, margin_mm, consensus_reason, "consensus",
                               measured_scope=scope, hint_scope=scope_hint)

    if not certain:
        abstain_reason = (f"{reason}; geometry measured {scope!r} but caller scope is "
                          f"{scope_hint!r}; abstained from landmark split")
        log.warning(f"   [!] [landmark] Acquisition Scope is UNCERTAIN ({reason}; margin "
                    f"{margin_mm:.1f} mm) — geometry measured {scope!r}, caller scope is "
                    f"{scope_hint!r}; abstaining from the split and running ONE full-width "
                    "pass. Acquisition Scope remains disputed/unknown; 'unilateral' below is "
                    "only the backward-compatible model input contract.")
        return CoverageVerdict(SCOPE_UNILATERAL, False, margin_mm, abstain_reason,
                               "abstention", measured_scope=scope, hint_scope=scope_hint)

    log.info(f"      [landmark] coverage={scope!r} (scope_for: {reason}, margin "
             f"{margin_mm:.1f} mm)")
    return CoverageVerdict(scope, True, margin_mm, reason, "scope_for",
                           measured_scope=scope, hint_scope=scope_hint)


def split_to_halves(source_path: str, verdict: CoverageVerdict,
                    work_dir: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Write one NIfTI per Breast Half and return `(side, path, geometry)` for each.

    Going through a file rather than converting `nib.Nifti1Image` to `sitk.Image` in memory is
    deliberate. NIfTI round-tripping defines the affine-to-origin/direction/spacing translation
    unambiguously, whereas hand-converting it is easy to get subtly wrong — and a sign error
    there puts every landmark on the wrong side of the volume, which is the precise failure
    `regions.SideRegion` already warns about for non-canonical exams. The halves are also then
    openable in a viewer when a result looks wrong.

    The halves are transient: they exist only to feed the model and are not the exam's output.
    """
    from mri_transforms.breast_geometry import split_image

    os.makedirs(work_dir, exist_ok=True)
    stem = os.path.basename(source_path)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    parts = split_image(nib.load(source_path), coverage=verdict.scope)
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for side, half, geometry in parts:
        path = os.path.join(work_dir, f"{stem}__{side}.nii.gz")
        nib.save(half, path)
        log.info(f"      [landmark] Breast Half {side!r}: {tuple(half.shape)} -> "
                 f"{os.path.basename(path)}")
        out.append((side, path, geometry))
    return out


def _ownership_slice_zyx(parent: sitk.Image,
                         split_geometry: Dict[str, Any]) -> Tuple[slice, slice, slice]:
    """The exact parent-array slice owned by one canonical-RAS Breast Half.

    ``split_image`` records its cut in canonical RAS x, while SimpleITK exposes the parent array
    in reversed native-axis order (zyx). Nibabel's orientation transform tells us which native
    axis carries canonical RAS x and whether it is flipped. No interpolation is involved: this is
    the original integer split expressed on the parent array.
    """
    parent_shape = tuple(int(v) for v in split_geometry.get("parent_shape", []))
    ras_shape = tuple(int(v) for v in split_geometry.get("ras_shape", []))
    x_slice = tuple(int(v) for v in split_geometry.get("x_slice", []))
    if len(parent_shape) != 3 or len(ras_shape) != 3 or len(x_slice) != 2:
        raise ValueError(f"invalid Breast Half split geometry: {split_geometry!r}")
    if tuple(int(v) for v in parent.GetSize()) != parent_shape:
        raise ValueError(
            "Breast Half split geometry parent shape "
            f"{parent_shape} does not match fusion parent {tuple(parent.GetSize())}")

    start_ras, stop_ras = x_slice
    if not (0 <= start_ras < stop_ras <= ras_shape[0]):
        raise ValueError(f"invalid canonical-RAS x_slice {x_slice} for shape {ras_shape}")

    parent_affine = np.asarray(split_geometry.get("parent_affine"), dtype=float)
    if parent_affine.shape != (4, 4):
        raise ValueError("Breast Half split geometry has no valid parent_affine")
    parent_orientation = nib.orientations.io_orientation(parent_affine)
    native_axes = np.flatnonzero(parent_orientation[:, 0] == 0)
    if len(native_axes) != 1:
        raise ValueError(
            "Breast Half parent affine does not identify exactly one canonical RAS x axis: "
            f"{parent_orientation.tolist()}")

    native_axis = int(native_axes[0])
    if ras_shape[0] != parent_shape[native_axis]:
        raise ValueError(
            f"canonical RAS width {ras_shape[0]} does not match native LR-axis length "
            f"{parent_shape[native_axis]}")
    direction = int(parent_orientation[native_axis, 1])
    if direction == 1:
        start_native, stop_native = start_ras, stop_ras
    elif direction == -1:
        start_native = parent_shape[native_axis] - stop_ras
        stop_native = parent_shape[native_axis] - start_ras
    else:                                                       # pragma: no cover - nib guard
        raise ValueError(f"invalid orientation direction {direction} for native LR axis")

    # sitk.GetArrayFromImage reverses the native image axes: native xyz -> array zyx.
    array_axis = 2 - native_axis
    owned = [slice(None), slice(None), slice(None)]
    owned[array_axis] = slice(start_native, stop_native)
    return tuple(owned)


def _clip_to_half_ownership(array_zyx: np.ndarray, parent: sitk.Image,
                            split_geometry: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """Remove inverse-resampling spill outside the Breast Half's original integer cut."""
    owned = _ownership_slice_zyx(parent, split_geometry)
    before = int(np.count_nonzero(array_zyx))
    ownership_axes = [axis for axis, part in enumerate(owned)
                      if part.start is not None or part.stop is not None]
    if len(ownership_axes) != 1:                                # pragma: no cover - helper guard
        raise ValueError(f"expected one Breast Half ownership axis, got {owned!r}")
    axis = ownership_axes[0]
    start = int(owned[axis].start)
    stop = int(owned[axis].stop)
    if start:
        before_slice = [slice(None), slice(None), slice(None)]
        before_slice[axis] = slice(0, start)
        array_zyx[tuple(before_slice)] = False
    if stop < array_zyx.shape[axis]:
        after_slice = [slice(None), slice(None), slice(None)]
        after_slice[axis] = slice(stop, None)
        array_zyx[tuple(after_slice)] = False
    return array_zyx, before - int(np.count_nonzero(array_zyx))


def fuse_halves(decided: List[Tuple[str, Dict[str, np.ndarray], Any, Dict[str, Any]]],
                parent: sitk.Image,
                head_names: List[str]) -> Dict[str, sitk.Image]:
    """Put each half's decided masks back on the parent grid and merge them.

    The side is written in here rather than re-derived afterwards, because it is KNOWN: the left
    half's foreground is the left breast, by construction of the cut. `assign_sides` stays
    reachable for the whole-exam path and as a fallback, but on this path it would be
    re-deriving from position something we were told outright.

    Each half is resampled onto the FULL parent grid, not onto its own. A coarse model-grid
    boundary voxel can extend by nearest-neighbour support across the fine-grid cut even though
    the source halves were disjoint. The restored mask is therefore clipped to the exact integer
    ownership slice recorded by `split_image` before it is merged. A half that is missing simply
    contributes nothing, which is what lets one failed side degrade instead of aborting the exam.
    """
    fused: Dict[str, sitk.Image] = {}
    shape = sitk.GetArrayViewFromImage(parent).shape
    voxel_mm3 = float(np.prod(parent.GetSpacing()))

    for head in head_names:
        acc = np.zeros(shape, dtype=np.uint8)
        for side, masks, model_grid, split_geometry in decided:
            label = LABEL_FOR_SIDE[side]
            on_parent = resample_like((masks[head] > 0).astype(np.uint8),
                                      reference=parent, source_geometry=model_grid)
            arr = sitk.GetArrayFromImage(on_parent).astype(bool)
            raw_overlap = int(np.count_nonzero((acc > 0) & arr))
            arr, clipped_count = _clip_to_half_ownership(arr, parent, split_geometry)
            if clipped_count:
                log.info(
                    f"      [landmark] head {head!r}: clipped {clipped_count} restored voxel(s) "
                    f"outside Breast Half {side!r} ownership"
                    + (f" ({raw_overlap} overlapped the previously placed half before clipping)"
                       if raw_overlap else ""))
            post_clip_overlap = int(np.count_nonzero((acc > 0) & arr))
            if post_clip_overlap:
                raise ValueError(
                    f"head {head!r}: Breast Half {side!r} still overlaps the previously placed "
                    f"half by {post_clip_overlap} parent voxel(s) after ownership clipping")
            acc[arr] = label

        out = sitk.GetImageFromArray(acc)
        out.CopyInformation(parent)
        n = int((acc > 0).sum())
        log.info(f"      [landmark] fused head {head!r}: labels "
                 f"{sorted(int(v) for v in np.unique(acc) if v > 0)}, {n} vox "
                 f"({n * voxel_mm3 / 1000:.1f} mL) on the parent grid")
        fused[head] = out

    return fused


def sides_from_fused(region: sitk.Image, verdict: CoverageVerdict) -> Dict[str, Optional[int]]:
    """`{side: label}` for the sides that actually CARRY voxels in the fused region.

    A unilateral exam keeps the `MERGED_LABEL` bucket, exactly as the whole-exam path does: the
    Breast Half IS the whole volume there, so there is no position along the left-right axis to
    restore a side from, and geometry cannot say which breast the acquisition holds.
    """
    from .regions import LEFT, MERGED_LABEL, RIGHT, UNILATERAL

    arr = sitk.GetArrayViewFromImage(region)
    if not verdict.is_bilateral:
        if not (arr > 0).any():
            log.warning("   [!] [landmark] breast_region is empty — no side to assign.")
            return {UNILATERAL: None}
        return {UNILATERAL: MERGED_LABEL}

    sides: Dict[str, Optional[int]] = {}
    for side, key in (("left", LEFT), ("right", RIGHT)):
        label = LABEL_FOR_SIDE[side]
        present = bool((arr == label).any())
        sides[key] = label if present else None
        if not present:
            log.warning(f"   [!] [landmark] Breast Half {side!r} ran but predicted no breast "
                        "region; reporting that side as missing.")
    return sides


def cleanup_halves(paths: List[str]) -> None:
    """Remove the transient Breast Half files once fusion has succeeded.

    Called only after fusion, never before: on a failure the halves are exactly what someone
    needs to see in a viewer to work out what went wrong. They are two full-resolution volumes
    per exam, and `inference_registry` already treats volumes this size as a memory and disk
    concern, so keeping every one of them forever is not a neutral default.
    """
    for path in paths:
        try:
            os.remove(path)
        except OSError as exc:
            log.warning(f"   [!] [landmark] could not remove transient half {path}: {exc}")


def halves_work_dir(outdir: str) -> str:
    """Where the transient Breast Half NIfTIs live — private to the landmark stage.

    Deliberately NOT the pipeline's `split_bilateral_exams` folder. These halves feed the model
    and are discarded after fusion; the downstream split is a different cut, computed from the
    landmark output, with margins and per-axis cropping. Mixing the two in one directory would
    invite reading one for the other.
    """
    return os.path.join(outdir, FOLDER_HALVES)
