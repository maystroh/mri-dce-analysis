"""Sides, nipple components, and the per-side regions that replace the midline cut.

Three jobs, all of them downstream of `LandmarkInferenceModel.predict`:

2. **Nipple components.** Every surviving component is reported with `biggest` and
   `intersect_with_br` flags rather than filtered. The model's own metrics say false positives
   happen (`nipples_fpi_total` 0.167) and misses happen (`nipples_tpr` 0.960); dropping the
   evidence would hide exactly the cases worth looking at. `distance_to_anterior_mm` is reported
   alongside `volume_mm3` for the same reason `biggest` exists — as a second, independent ranking
   a consumer (e.g. the review viewer) can pick "the" nipple by, without the pipeline baking that
   choice in.

3. **Per-side regions.** The bbox of each side's label, cropped on all three axes and with the
   contralateral label's voxels zeroed inside the box. Two boxes may OVERLAP in x, which is the
   point: a single cut plane can trim a breast on an asymmetric exam, two boxes cannot.

Coordinate spaces (all three are recorded; see the plan, section 5):
  * `position_source_voxel` — index into the saved mask and into the input NIfTI: the same grid.
    This is the one surfaced in `run_details.json`, matching `original_boundingBox3D`.
  * `position_ras_voxel`    — index in the RAS-oriented volume, i.e. the space the 2D pipeline
    slabs and reports lesion `boundingBox3D` in.
  * `position_mm`           — world/patient coordinates in RAS convention (+x = patient Right),
    the only space comparable across exams.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.measure import label as cc_label

from mri_transforms.preprocessing import reorient_to_ras
from mri_transforms.utils import log

LEFT, RIGHT = "left", "right"
# The third acquisition scope, matching `laterality_status.lower()` from `infer_bilateral_view`
# and the `left`/`right`/`unilateral` directory convention the rest of the pipeline already uses
# for a non-split exam (see inference_registry.py's `list_of_niftis_to_process`).
UNILATERAL = "unilateral"
# Sentinel `label_value` meaning "every labelled voxel, regardless of how many distinct model
# classes it split into" — used only under `is_bilateral=False`. Not a real class value (labels
# are 0/1/2), so it can't collide with one; every `arr == value` / `reg == label_value` comparison
# below has a matching `== MERGED_LABEL -> > 0` branch.
MERGED_LABEL = -1


# --------------------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------------------

class CoordinateFrames:
    """Maps a source-grid voxel index into the three spaces we report.

    Built from the source NIfTI's own affine rather than from the SimpleITK image, so the numbers
    line up with `get_ras_to_original_transform` and the lesion coordinates already in
    `run_details.json`.
    """

    def __init__(self, source_path: str):
        self.source_path = str(source_path)
        img = nib.load(self.source_path)
        ras = nib.as_closest_canonical(img)
        self.affine = img.affine                       # source voxel -> RAS mm
        self.shape = img.shape
        # source voxel -> RAS-oriented voxel
        self.source_to_ras_voxel = np.linalg.inv(ras.affine) @ img.affine
        self.affine_sha1 = hashlib.sha1(
            np.ascontiguousarray(img.affine, dtype=np.float64).tobytes()).hexdigest()[:16]

    def describe(self, ijk_source) -> Dict[str, Dict[str, float]]:
        v = np.asarray(ijk_source, dtype=float)
        ras_vox = (self.source_to_ras_voxel @ np.append(v, 1.0))[:3]
        mm = nib.affines.apply_affine(self.affine, v)
        r = lambda x: round(float(x), 2)
        return {
            "position_source_voxel": {"w": r(v[0]), "h": r(v[1]), "d": r(v[2])},
            "position_ras_voxel": {"w": r(ras_vox[0]), "h": r(ras_vox[1]), "d": r(ras_vox[2])},
            "position_mm": {"x": r(mm[0]), "y": r(mm[1]), "z": r(mm[2])},
        }


def _lr_axis_and_sign(img: sitk.Image) -> Tuple[int, float]:
    """Which array axis is left-right, and which way patient-RIGHT lies along it.

    Returns `(numpy_axis, sign)` for a `(z, y, x)` array, where `sign * index` increases toward
    the patient's RIGHT. Read from the direction matrix rather than assumed: SimpleITK stores
    LPS, so patient-Right is -x.
    """
    d = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
    lr_col = int(np.argmax(np.abs(d[0, :])))          # image axis most aligned with LPS x
    sign = -1.0 if d[0, lr_col] > 0 else 1.0          # +LPS x is patient LEFT
    return 2 - lr_col, sign                            # sitk axis (x,y,z) -> numpy axis (z,y,x)


def _ap_axis_and_sign(img: sitk.Image) -> Tuple[int, float]:
    """Which array axis is anterior-posterior, and which way patient-ANTERIOR lies along it.

    Same derivation as `_lr_axis_and_sign`, one row down in the direction matrix: SimpleITK
    stores LPS, so +LPS-y is POSTERIOR and patient-Anterior is the opposite sign.
    """
    d = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
    ap_col = int(np.argmax(np.abs(d[1, :])))          # image axis most aligned with LPS y
    sign = -1.0 if d[1, ap_col] > 0 else 1.0          # +LPS y is patient POSTERIOR
    return 2 - ap_col, sign


# --------------------------------------------------------------------------------------
# sides
# --------------------------------------------------------------------------------------

def assign_sides(region_img: sitk.Image, is_bilateral: bool = True) -> Dict[str, Optional[int]]:
    """`{'left': label_value, 'right': label_value}` decided by position, not by label value —
    or, for a verified-unilateral acquisition, `{'unilateral': MERGED_LABEL}`.
    """
    arr = sitk.GetArrayViewFromImage(region_img)
    values = [int(v) for v in np.unique(arr) if v > 0]

    if not is_bilateral:
        if not values:
            log.warning("   [!] breast_region is empty — no side to assign.")
            return {UNILATERAL: None}
        log.info(f"      [landmark] unilateral exam: merging label(s) {values} into one side "
                 f"— sidedness not attempted on this acquisition (see assign_sides docstring)")
        return {UNILATERAL: MERGED_LABEL}

    axis, sign = _lr_axis_and_sign(region_img)
    if not values:
        log.warning("   [!] breast_region is empty — no sides to assign.")
        return {LEFT: None, RIGHT: None}

    scored = []
    for v in values:
        idx = np.argwhere(arr == v)
        scored.append((float(idx[:, axis].mean()) * sign, v))   # larger = more toward RIGHT
    scored.sort()

    sides: Dict[str, Optional[int]] = {LEFT: None, RIGHT: None}
    if len(scored) == 1:
        # one region only: it is whichever side it sits on relative to the volume centre
        centre = arr.shape[axis] / 2.0
        idx = np.argwhere(arr == scored[0][1])
        is_right = (float(idx[:, axis].mean()) - centre) * sign > 0
        sides[RIGHT if is_right else LEFT] = scored[0][1]
        log.info(f"      [landmark] one breast region only -> {'right' if is_right else 'left'} "
                 f"(label {scored[0][1]})")
    else:
        sides[LEFT], sides[RIGHT] = scored[0][1], scored[-1][1]
        if len(scored) > 2:
            log.warning(f"   [!] breast_region has {len(scored)} labels {values}; using the two "
                        f"outermost as left/right.")
    log.info(f"      [landmark] sides by geometry: left=label {sides[LEFT]}, right=label {sides[RIGHT]}")
    return sides


# --------------------------------------------------------------------------------------
# nipples
# --------------------------------------------------------------------------------------

NIPPLE_FILTER_NONE = "none"
NIPPLE_FILTER_INTERSECT = "intersect_with_br"
FILTER_SCOPE_ANY = "any_region"
FILTER_SCOPE_OWN = "own_side"


def analyse_nipples(nipples_img: sitk.Image, region_img: sitk.Image,
                    sides: Dict[str, Optional[int]], frames: CoordinateFrames,
                    nipple_filter: str = NIPPLE_FILTER_NONE,
                    filter_scope: str = FILTER_SCOPE_ANY,
                    is_bilateral: bool = True) -> Dict[str, List[Dict]]:
    """Every nipple component, grouped by side, flagged rather than removed.

    `is_bilateral=False` must match what `sides` (from `assign_sides`) was built with: every
    component goes to `sides`' one key instead of being split by centroid position. Splitting a
    unilateral acquisition's nipple candidates left/right by geometry is exactly how a model
    artifact on the empty side of the volume gets reported as a second breast's nipple — see
    `assign_sides` for why that split is refused here.

    `nipple_filter="intersect_with_br"` uses the breast region to suppress false positives: a
    component that touches no breast voxel is marked `filtered_out` and cannot be `biggest`, so it
    stops being the reported nipple. It is still WRITTEN, in both the mask and the manifest —
    deleting it would remove the evidence for the decision the filter just made.

    Two properties of this filter are worth keeping in mind when reading its output:

    * The heads share an encoder AND a decoder, so a nipple false positive is computed from the
      same features that decide where the breast is. The errors are correlated by construction,
      and a false positive inside the breast is exactly the kind this cannot catch.
    * The nipple sits on the anterior skin surface, which is where a region segmentation's
      boundary error lives. `filtered_out` on a plausible nipple is therefore as likely to be a
      REGION error as a nipple error — which is why the default scope is the union of both
      regions (`any_region`) rather than the component's own side, and why nothing is deleted.

    `intersect_with_br` keeps its own meaning throughout: the component's own side's region. It is
    a description of the anatomy, not the filter's verdict; `filtered_out` is the verdict.
    """
    nip = sitk.GetArrayViewFromImage(nipples_img)
    reg = sitk.GetArrayViewFromImage(region_img)
    axis, sign = _lr_axis_and_sign(nipples_img)
    voxel_mm3 = float(np.prod(nipples_img.GetSpacing()))
    out: Dict[str, List[Dict]] = {side: [] for side in sides}

    labelled, n = cc_label(nip > 0, return_num=True)
    if n == 0:
        log.warning("   [!] no nipple components found.")
        return out

    # Anterior-most (skin) voxel of each side's own region, in physical mm — the reference point
    # for `distance_to_anterior_mm` below. Computed once per side, not per component: it is a
    # property of the region, and every nipple candidate on that side is measured against the same
    # point so the numbers are comparable to each other.
    ap_axis, ap_sign = _ap_axis_and_sign(region_img)
    anterior_tip_mm: Dict[str, Optional[np.ndarray]] = {}
    for side, label_value in sides.items():
        if label_value is None:
            anterior_tip_mm[side] = None
            continue
        side_idx = (np.argwhere(reg > 0) if label_value == MERGED_LABEL
                    else np.argwhere(reg == label_value))
        if side_idx.size == 0:
            anterior_tip_mm[side] = None
            continue
        tip_zyx = side_idx[(side_idx[:, ap_axis] * ap_sign).argmax()]
        anterior_tip_mm[side] = np.asarray(
            region_img.TransformContinuousIndexToPhysicalPoint(
                tuple(float(v) for v in tip_zyx[::-1])))

    centre = nip.shape[axis] / 2.0
    # Unilateral: `sides` has exactly one key (see assign_sides) — every component goes there,
    # with no centroid-position vote. Bilateral: unchanged, position decides left vs right.
    only_side = next(iter(sides)) if not is_bilateral else None
    for i in range(1, n + 1):
        component = labelled == i
        idx = np.argwhere(component)
        centroid_zyx = idx.mean(axis=0)
        side = only_side if only_side is not None else (
            RIGHT if (centroid_zyx[axis] - centre) * sign > 0 else LEFT)
        own_label = sides.get(side)
        region_here = reg[component]
        intersects_own = bool(own_label is not None and (
            np.any(region_here > 0) if own_label == MERGED_LABEL
            else np.any(region_here == own_label)))
        intersects_any = bool(np.any(region_here > 0))

        keep = True
        if nipple_filter == NIPPLE_FILTER_INTERSECT:
            keep = intersects_own if filter_scope == FILTER_SCOPE_OWN else intersects_any

        tip = anterior_tip_mm.get(side)
        distance_to_anterior_mm = None
        if tip is not None:
            centroid_mm = np.asarray(
                nipples_img.TransformContinuousIndexToPhysicalPoint(
                    tuple(float(v) for v in centroid_zyx[::-1])))
            distance_to_anterior_mm = round(float(np.linalg.norm(centroid_mm - tip)), 2)

        entry = {
            **frames.describe(centroid_zyx[::-1]),      # (z,y,x) -> (x,y,z) source index
            "volume_mm3": round(len(idx) * voxel_mm3, 2),
            "volume_voxels": int(len(idx)),
            "distance_to_anterior_mm": distance_to_anterior_mm,
            "biggest": False,
            "intersect_with_br": intersects_own,
            "intersect_with_any_br": intersects_any,
            "filtered_out": not keep,
        }
        out[side].append(entry)

    for side, entries in out.items():
        if entries:
            entries.sort(key=lambda e: e["volume_voxels"], reverse=True)
            kept = [e for e in entries if not e["filtered_out"]]
            if kept:
                kept[0]["biggest"] = True
            elif nipple_filter != NIPPLE_FILTER_NONE:
                log.warning(f"   [!] nipples {side}: all {len(entries)} component(s) were filtered "
                            f"out by {nipple_filter}/{filter_scope} — this side reports no nipple.")
        dropped = sum(1 for e in entries if e["filtered_out"])
        log.info(f"      [landmark] nipples {side}: {len(entries)} component(s)"
                 + (f", {dropped} filtered out" if dropped else "")
                 + (f", biggest {next((e['volume_mm3'] for e in entries if e['biggest']), None)} mm3"
                    if any(e["biggest"] for e in entries) else " — NONE reported"))
    return out


# --------------------------------------------------------------------------------------
# per-side regions
# --------------------------------------------------------------------------------------

@dataclass
class SideRegion:
    """One side's crop box, **in RAS voxel indices**.

    RAS, not the source grid, because the 2D pipeline slabs volumes loaded through
    `load_nifti_ras`. On a source that is not already canonical the two differ by a flip: for an
    LAS exam of width 864, source x in [430, 863] is RAS x in [0, 433]. Recording the source
    index would put every reconstructed lesion coordinate on the wrong side of the volume.
    """

    side: str
    label_value: int
    bbox_zyx: List[List[int]]                 #: [[z0,z1],[y0,y1],[x0,x1]], stop-exclusive
    volume_mm3: float

    @property
    def crop_origin_xyz(self) -> List[int]:
        """`[x0, y0, z0]` — the offset every coordinate from this side must be shifted by."""
        (z0, _), (y0, _), (x0, _) = self.bbox_zyx
        return [x0, y0, z0]

    @property
    def size_xyz(self) -> List[int]:
        return [b - a for a, b in reversed(self.bbox_zyx)]


def region_boxes(region_ras: sitk.Image, sides: Dict[str, Optional[int]],
                 margin_mm: float = 5.0) -> Dict[str, SideRegion]:
    """Per-side bounding box of the label, padded by `margin_mm` and clipped to the volume.

    Takes the **RAS-oriented** region mask — see `SideRegion` for why the space matters.

    Cropped on ALL THREE axes: inference then runs only on slabs that contain breast. The full
    z extent is restored later by zero-filling the OUTPUT slabs, so the output contract does not
    change (see the plan, section 6.2).
    """
    arr = sitk.GetArrayViewFromImage(region_ras)
    spacing_zyx = list(reversed(region_ras.GetSpacing()))
    voxel_mm3 = float(np.prod(region_ras.GetSpacing()))
    margin_vox = [int(np.ceil(margin_mm / s)) for s in spacing_zyx]

    boxes: Dict[str, SideRegion] = {}
    for side, value in sides.items():
        if value is None:
            continue
        idx = np.argwhere(arr > 0) if value == MERGED_LABEL else np.argwhere(arr == value)
        if idx.size == 0:
            continue
        bbox = []
        for ax in range(3):
            lo = max(0, int(idx[:, ax].min()) - margin_vox[ax])
            hi = min(int(arr.shape[ax]), int(idx[:, ax].max()) + 1 + margin_vox[ax])
            bbox.append([lo, hi])
        boxes[side] = SideRegion(side=side, label_value=int(value), bbox_zyx=bbox,
                                 volume_mm3=round(len(idx) * voxel_mm3, 2))
        log.info(f"      [landmark] region {side}: label {value}, box zyx {bbox}, "
                 f"size xyz {boxes[side].size_xyz}, {boxes[side].volume_mm3/1000:.1f} mL")
    return boxes


def extract_side_volume(image: sitk.Image, region_img: sitk.Image,
                        box: SideRegion, sides: Dict[str, Optional[int]]) -> sitk.Image:
    """Crop `image` to this side's box and zero the CONTRALATERAL label inside it.

    Background, skin and air inside the box are kept: the slabs must still look like an MRI, not
    a breast silhouette, because the 2D lesion model never saw silhouettes. Zeroing only the other
    side's labelled voxels is what makes "which breast is this lesion in?" well defined where the
    two boxes overlap.
    """
    (z0, z1), (y0, y1), (x0, x1) = box.bbox_zyx
    cropped = image[x0:x1, y0:y1, z0:z1]
    other = next((v for s, v in sides.items() if v is not None and s != box.side), None)
    if other is None:
        return cropped
    reg_crop = sitk.GetArrayViewFromImage(region_img)[z0:z1, y0:y1, x0:x1]
    arr = sitk.GetArrayFromImage(cropped)
    arr[reg_crop == other] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(cropped)
    return out


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------

def write_landmark_outputs(result: Dict[str, Any], outdir: str, source_path: str,
                           margin_mm: float = 5.0,
                           nipple_filter: str = NIPPLE_FILTER_NONE,
                           filter_scope: str = FILTER_SCOPE_ANY,
                           write_masks: bool = True,
                           is_bilateral: bool = True,
                           sides: Optional[Dict[str, Optional[int]]] = None,
                           coverage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the masks and the `landmarks.json` entry for one source volume.

    Names are phase-addressable (`<mask>__<source stem>.nii.gz`) and the manifest is a LIST of
    entries, so generating masks for every phase later adds entries instead of changing format.

    `is_bilateral` (from `infer_bilateral_view`, upstream and independent of this model) decides
    whether `sides` — and therefore `entry["sides"]` below — comes out keyed `left`/`right` or as
    one `unilateral` bucket. See `assign_sides`.
    """
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(source_path)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    region_img, nipples_img = result["masks"]["breast_region"], result["masks"]["nipples"]
    frames = CoordinateFrames(source_path)
    if sides is None:
        sides = assign_sides(region_img, is_bilateral=is_bilateral)
    else:
        # Per-breast: the side is not inferred from position, it is known from which Breast Half
        # was sent. `assign_sides` stays the default and the fallback; it is simply not asked to
        # re-derive something the caller was told outright.
        log.info(f"      [landmark] sides supplied by the caller: {sides}")
    nipples = analyse_nipples(nipples_img, region_img, sides, frames,
                              nipple_filter=nipple_filter, filter_scope=filter_scope,
                              is_bilateral=is_bilateral)
    # Boxes are computed on the RAS view: that is the space the per-side volumes are cropped in
    # and the space `split_info.crop_origin` has to be expressed in. The masks themselves stay on
    # the source grid on disk, so they overlay the input exam directly in a viewer.
    boxes = region_boxes(reorient_to_ras(region_img), sides, margin_mm)

    files = {}
    for name, img in result["masks"].items():
        path = os.path.join(outdir, f"{name}__{stem}.nii.gz")
        if write_masks:
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), path)
        files[name] = os.path.basename(path)

    entry = {
        "source_file": os.path.basename(source_path),
        "source_path": source_path,
        "source_shape": [int(s) for s in frames.shape],
        "source_affine_sha1": frames.affine_sha1,
        "breast_region_file": files.get("breast_region"),
        "nipples_file": files.get("nipples"),
        "crop_margin_mm": margin_mm,
        "nipple_filter": nipple_filter,
        "nipple_filter_scope": filter_scope if nipple_filter != NIPPLE_FILTER_NONE else None,
        "is_bilateral": is_bilateral,
        # The per-breast input contract plus its Acquisition Scope evidence, or None for a
        # whole-exam model. `scope`/`execution` say how the model ran; `measured_scope` says what
        # geometry measured. They may differ when an uncertain disagreement abstains from a split.
        "coverage": coverage,
        "sides": {
            side: {
                "breast_region": (
                    None if side not in boxes else {
                        "label_value": boxes[side].label_value,
                        "volume_mm3": boxes[side].volume_mm3,
                        "crop_origin_xyz_ras": boxes[side].crop_origin_xyz,
                        "crop_size_xyz": boxes[side].size_xyz,
                        "bbox_zyx_ras": boxes[side].bbox_zyx,
                    }),
                "nipples": nipples.get(side, []),
            }
            for side in sides
        },
    }

    manifest_path = os.path.join(outdir, "landmarks.json")
    manifest = {"model": result["model"], "entries": []}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["model"] = result["model"]
    manifest["entries"] = [e for e in manifest.get("entries", [])
                           if e.get("source_file") != entry["source_file"]] + [entry]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"      [landmark] wrote {manifest_path}")

    return {"entry": entry, "sides": sides, "boxes": boxes, "manifest_path": manifest_path}


# --------------------------------------------------------------------------------------
# the split itself
# --------------------------------------------------------------------------------------

def split_by_regions(nifti_files: List[str], output_split_path: str, region_img: sitk.Image,
                     sides: Dict[str, Optional[int]], boxes: Dict[str, SideRegion],
                     breast_mask_name: str = "breast_mask") -> Tuple[Dict[str, List[str]], Dict]:
    """Write per-side volumes, replacing the midline cut with the label bounding boxes.

    Produces exactly the filenames `split_bilateral_exams` produces — `{base}_{side}.nii.gz` plus
    `{breast_mask_name}_{side}.nii.gz` — so `voxel_resampling`, `generate_slabs` and everything
    downstream are untouched.

    Everything happens in RAS: the 2D pipeline reloads these files through `load_nifti_ras`, so
    writing them already-canonical makes that a no-op and keeps `crop_origin` in the same space
    the lesion coordinates are reconstructed in.

    Returns `(files_by_side, split_info)`, where `split_info` carries the per-side origin that
    `reconstruct_split_coordinates` needs. That origin is recorded rather than re-derived: a
    bounding box cannot be recovered from the volume width the way a midpoint can.
    """
    os.makedirs(output_split_path, exist_ok=True)
    region_ras = reorient_to_ras(region_img)
    ras_shape_xyz = list(region_ras.GetSize())

    processed: Dict[str, List[str]] = {}
    split_info: Dict[str, Any] = {
        "was_split": True,
        "original_x_dim": int(ras_shape_xyz[0]),
        "split_method": "landmark_regions",
        # full RAS extent: the z component is what the output slabs are zero-filled back up to,
        # so the pmap stack keeps covering the whole volume even though inference did not.
        "full_shape_xyz": [int(v) for v in ras_shape_xyz],
        "sides": {},
    }

    for side, box in boxes.items():
        out_files: List[str] = []
        for nifti_path in nifti_files:
            img_ras = reorient_to_ras(sitk.ReadImage(str(nifti_path)))
            if list(img_ras.GetSize()) != ras_shape_xyz:
                raise ValueError(
                    f"{os.path.basename(nifti_path)} is {img_ras.GetSize()} but the breast-region "
                    f"mask is {tuple(ras_shape_xyz)}; the box cannot be applied to a different grid.")
            cropped = extract_side_volume(img_ras, region_ras, box, sides)
            base = os.path.basename(nifti_path)
            for ext in (".nii.gz", ".nii"):
                if base.endswith(ext):
                    base = base[: -len(ext)]
                    break
            out_path = os.path.join(output_split_path, f"{base}_{side}.nii.gz")
            sitk.WriteImage(cropped, out_path)
            out_files.append(out_path)

        # this side's own region, cropped to the same box, is the breast mask for this side
        (z0, z1), (y0, y1), (x0, x1) = box.bbox_zyx
        mask = sitk.Cast(region_ras == box.label_value, sitk.sitkUInt8)[x0:x1, y0:y1, z0:z1]
        sitk.WriteImage(mask, os.path.join(output_split_path, f"{breast_mask_name}_{side}.nii.gz"))

        processed[side] = out_files
        split_info["sides"][side] = {
            "crop_origin": box.crop_origin_xyz,
            "crop_size": box.size_xyz,
            "label_value": box.label_value,
            "bbox_zyx": box.bbox_zyx,
        }
        log.info(f"      [landmark] split {side}: {len(out_files)} volume(s) at "
                 f"{box.size_xyz} from origin {box.crop_origin_xyz} (RAS)")

    return processed, split_info
