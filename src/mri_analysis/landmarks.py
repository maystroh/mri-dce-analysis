"""Landmark plausibility policy and measurements for saved predictions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

REGION_VOLUME_MIN_MM3 = 279_000.0
REGION_VOLUME_MAX_MM3 = 2_635_000.0

# The same volume below which `motion_qc` says a score over the region should not be believed.
# Expressed in mm3 here so it can be compared with the band above without a unit change.
ROI_PLAUSIBILITY_MIN_MM3 = 100_000.0

PLAUSIBLE, TOO_SMALL, TOO_LARGE, ABSENT = "plausible", "too_small", "too_large", "absent"


def region_plausibility(volume_mm3: Optional[float],
                        lo: float = REGION_VOLUME_MIN_MM3,
                        hi: float = REGION_VOLUME_MAX_MM3) -> str:
    """Where one side's region volume falls against the reference band.

    `None` means the side reported no region at all, which is a different finding from a
    region of implausible size: one is the model predicting nothing, the other is the model
    predicting something wrong. They are never collapsed into a single "bad" bucket because
    they have different causes and different fixes.
    """
    if volume_mm3 is None:
        return ABSENT
    if volume_mm3 < lo:
        return TOO_SMALL
    if volume_mm3 > hi:
        return TOO_LARGE
    return PLAUSIBLE




def midline_cut_offset_mm(nifti_path: str) -> Tuple[Optional[float], str]:
    """`(offset_mm, reason)` -- how far `split_image`'s cut sits from world x = 0.

    Positive means the cut falls on the patient's LEFT of the midline (the left Breast Half
    is given more than half the anatomy), negative to the right. `None` means the offset
    could not be measured and the reason says why.

    The offset is computed on the CANONICAL RAS grid, because that is the grid `split_image`
    slices -- measuring it on the source grid would answer a different question for any
    volume the converter did not write in RAS.

    The cut plane is the boundary BETWEEN voxels `width // 2 - 1` and `width // 2`. An affine
    maps an index to a voxel's CENTRE, so that boundary sits at continuous index
    `width // 2 - 0.5`, half a voxel below the first voxel of the right half. Dropping the
    half-voxel reports the cut at a voxel centre instead of at the cut, which is a systematic
    error of `spacing / 2` -- 0.4 mm on a DUKE exam, and always in the same direction.

    It is evaluated at the volume's CENTRE in the other two axes, so an oblique affine gives
    the offset where the anatomy is rather than at a corner. On an oblique volume the cut is a
    tilted plane and no single number describes it: MAMA-MIA `DUKE_001` (2.1 deg) reads 9.1 mm
    at the centre against 5.9 mm at the corner. The centre is the honest summary; the obliquity
    is reported beside it so a reader can tell when one number is not enough.

    This is a MEASUREMENT with no cut attached. A large offset does not invalidate a result:
    it says the two Breast Halves were unequal shares of the patient, which is a thing to
    know when reading a per-side volume, not a thing to reject an exam for.
    """
    try:
        image = nib.load(nifti_path)
    except Exception as exc:                                     # unreadable is not a crash
        return None, f"could not load: {exc}"
    if len(image.shape) != 3:
        # split_image raises on this, so an exam here never reaches the cut at all. Reported
        # rather than silently skipped: a 4D file is the case `scope_for` also abstains on.
        return None, f"{len(image.shape)}D file -- split_image accepts 3D only"
    try:
        ras = nib.as_closest_canonical(image)
    except Exception as exc:
        return None, f"could not reorient to canonical RAS: {exc}"
    if nib.aff2axcodes(ras.affine) != ("R", "A", "S"):
        return None, "cannot establish RAS orientation"

    width = ras.shape[0]
    if width < 2:
        return None, "fewer than two left-right voxels"
    centre = [(width // 2) - 0.5, (ras.shape[1] - 1) / 2.0, (ras.shape[2] - 1) / 2.0, 1.0]
    world_x = float(np.asarray(ras.affine).dot(centre)[0])
    if not math.isfinite(world_x):
        return None, "affine produced a non-finite world coordinate"
    return round(world_x, 2), f"cut at RAS voxel x={width // 2} of {width}"


# --------------------------------------------------------------------------------------
# Gate 2 -- is this mask fit to be CONSUMED?
# --------------------------------------------------------------------------------------

@dataclass
class SideVerdict:
    """What stage 2 concluded about one side of one exam."""

    side: str
    volume_mm3: Optional[float]
    plausibility: str
    n_nipples: int
    roi_big_enough: bool
    reasons: List[str]

    @property
    def fit_to_consume(self) -> bool:
        return not self.reasons

    def as_row(self) -> Dict[str, Any]:
        return {"side": self.side, "region_volume_mm3": self.volume_mm3,
                "region_plausibility": self.plausibility, "n_nipples": self.n_nipples,
                "roi_big_enough": self.roi_big_enough,
                "fit_to_consume": self.fit_to_consume,
                "unfit_reason": "; ".join(self.reasons)}


def judge_side(side: str, region: Optional[Dict[str, Any]], nipples: List[Dict[str, Any]],
               lo: float = REGION_VOLUME_MIN_MM3, hi: float = REGION_VOLUME_MAX_MM3,
               roi_min_mm3: float = ROI_PLAUSIBILITY_MIN_MM3) -> SideVerdict:
    """Apply Gate 2 to one `landmarks.json` side entry.

    Gate 2 asks whether a mask is fit to be USED as a Registration ROI, not whether it is
    accurate -- accuracy needs an annotation and is scored by the shared landmark evaluator against the
    benchmark cohorts. What is checked here is what can be checked WITHOUT ground truth on
    any dataset, which is the only kind of check a dataset sweep can make:

    * a region exists;
    * its volume is inside the reference band;
    * its volume clears the ROI Plausibility Gate, because a region that passes the band but
      not the gate would be handed to motion QC and score against too little anatomy.

    The count of regions per side is NOT checked here and cannot be: under the Single-Breast
    Formulation each side's mask comes from one Breast Half and carries one label by
    construction. The "spurious second region" failure is now a connected-component question
    inside a half, which needs the mask itself -- `predicted_region_components` in the
    benchmark CSV is where that is measured.
    """
    volume = None if region is None else float(region.get("volume_mm3") or 0.0)
    plausibility = region_plausibility(volume, lo, hi)
    roi_ok = volume is not None and volume >= roi_min_mm3
    reasons: List[str] = []
    if plausibility == ABSENT:
        reasons.append("no breast region predicted for this side")
    elif plausibility == TOO_SMALL:
        reasons.append(f"region {volume:.0f} mm3 below the {lo:.0f} mm3 reference band")
    elif plausibility == TOO_LARGE:
        reasons.append(f"region {volume:.0f} mm3 above the {hi:.0f} mm3 reference band")
    if volume is not None and not roi_ok:
        reasons.append(f"region {volume/1000:.1f} cm3 below the "
                       f"{roi_min_mm3/1000:.0f} cm3 ROI Plausibility Gate")
    return SideVerdict(side=side, volume_mm3=volume, plausibility=plausibility,
                       n_nipples=len([n for n in nipples if not n.get("filtered_out")]),
                       roi_big_enough=roi_ok, reasons=reasons)


def judge_entry(entry: Dict[str, Any], **cuts) -> List[SideVerdict]:
    """Gate 2 over every side of one `landmarks.json` entry, in a stable order.

    Iterates the entry's OWN side keys rather than a fixed `("left", "right")` pair. A
    unilateral exam produces a single `unilateral` key under the per-breast model and a
    hard-coded pair would raise on it -- the failure `run_landmarks.py` still has.
    """
    sides = entry.get("sides") or {}
    order = [s for s in ("left", "right", "unilateral") if s in sides]
    order += [s for s in sorted(sides) if s not in order]
    return [judge_side(side, sides[side].get("breast_region"),
                       sides[side].get("nipples") or [], **cuts)
            for side in order]


class StoredLandmarkRunner:
    """Read existing outputs for analysis without loading a model or rewriting its manifest.

    The source filename, grid shape and affine fingerprint must match the selected exam.
    Historical fusion versions remain analysable and are reported rather than upgraded.
    """

    def run(self, source_path: str, output_path: str, is_bilateral=None) -> Dict[str, Any]:
        with open(Path(output_path) / "landmarks.json") as fh:
            manifest = json.load(fh)
        image = nib.load(source_path)
        fingerprint = hashlib.sha1(
            np.ascontiguousarray(image.affine, dtype=np.float64).tobytes()).hexdigest()[:16]
        entries = [e for e in manifest.get("entries", [])
                   if e.get("source_file") == Path(source_path).name
                   and e.get("source_shape") == list(image.shape)
                   and e.get("source_affine_sha1") == fingerprint]
        if len(entries) != 1:
            raise ValueError("stored landmarks must contain exactly one entry matching source "
                             "filename, shape and affine")
        entry = entries[0]
        for key in ("breast_region_file", "nipples_file"):
            if not entry.get(key) or not (Path(output_path) / entry[key]).is_file():
                raise FileNotFoundError(f"stored landmark mask missing: {entry.get(key)!r}")
        return {"entry": entry, "coverage": entry.get("coverage"),
                "model": manifest.get("model", {}), "reused": True}


def region_mask_measurements(path: str, entry: Dict[str, Any],
                             source_path: str) -> Dict[str, Dict[str, Any]]:
    """Measure connected components and LR edge contact per reported side on the parent grid.

    Edge contact is a review signal, not proof of truncation. Volumes use the affine
    determinant so oblique grids do not change the physical-volume definition.
    """
    from scipy import ndimage

    image = nib.load(path)
    source = nib.load(source_path)
    if image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-3):
        raise ValueError("breast-region mask does not match the source grid")
    codes = nib.aff2axcodes(image.affine)
    lr_axis = next(i for i, code in enumerate(codes) if code in ("L", "R"))
    array = np.asarray(image.dataobj)
    voxel_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
    measurements = {}
    for side, record in (entry.get("sides") or {}).items():
        region = record.get("breast_region") or {}
        value = region.get("label_value")
        if value is None:
            continue
        binary = array > 0 if value == -1 else array == value
        labelled, count = ndimage.label(binary)  # six-neighbour connectivity
        volumes = np.bincount(labelled.ravel())[1:] * voxel_mm3
        measurements[side] = {
            "region_components": int(count),
            "region_mask_volume_mm3": round(float(volumes.sum()), 2),
            "largest_component_mm3": round(float(volumes.max()), 2) if count else 0.0,
            "region_touches_lr_border": bool(np.take(binary, 0, axis=lr_axis).any()
                                              or np.take(binary, -1, axis=lr_axis).any()),
        }
    return measurements
