from typing import Iterable, List, Tuple, Optional

import math

import os

import re

import logging

import cv2

import json

from dataclasses import dataclass

import numpy as np

import nibabel as nib

from PIL import Image

from skimage.io import imsave

from skimage import measure, img_as_ubyte

from scipy import ndimage

name_to_filter = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX
}

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

log = logging.getLogger()

@dataclass
class SeriesMeta:
    original: str          # full path, not just basename
    series_id: str
    date: Optional[int]    # e.g., 19440301, or None
    time_seconds: float    # seconds since midnight
    series_desc: str       # the "SeriesDescription" tail

def find_nifti_files_by_metadata(exam_path: str) -> Tuple[Optional[str], List[str]]:
    """
    High-precision extractor that understands filenames like:
      {seriesid}-{YYYYMMDD}-{HHMMSS[.fff]}-{SeriesDescription}.nii.gz
      {seriesid}--{HHMMSS[.fff]}-{SeriesDescription}.nii.gz

    Rules:
      - PRE: series description contains a token-ish 'pre' (case-insensitive).
      - POST: series description contains 'post' (case-insensitive).
      - Exclude anything with 'SUB' anywhere in the filename.
      - Sort posts by (date, time). Dated items come before undated.
    """
    all_paths: List[str] = []
    for root, _, files in os.walk(exam_path):
        for f in files:
            if f.lower().endswith(('.nii', '.nii.gz')):
                all_paths.append(os.path.join(root, f))

    cand_paths = [
        p for p in all_paths if "sub" not in os.path.basename(p).lower()]
    metas: List[SeriesMeta] = []
    for p in cand_paths:
        base = os.path.basename(p).strip().strip("'").strip('"')
        meta = _parse_series_meta(base)
        if meta:
            # keep the full path as the "original"
            metas.append(SeriesMeta(
                original=p,
                series_id=meta.series_id,
                date=meta.date,
                time_seconds=meta.time_seconds,
                series_desc=meta.series_desc,
            ))

    # PRE and POST detection (from series description only)
    pre_candidates = [m for m in metas if _has_token_pre(m.series_desc)]
    post_candidates = [m for m in metas if "post" in m.series_desc.lower()]

    pre_file = None
    if pre_candidates:
        pre_file = sorted(pre_candidates, key=_sort_key)[0].original

    post_files = [m.original for m in sorted(post_candidates, key=_sort_key)]
    return pre_file, post_files

@dataclass
class _Parsed:
    series_id: str
    date: Optional[int]
    time_seconds: float
    series_desc: str

def _sort_key(m: SeriesMeta):
    # Sort dated first; then by date; then by time
    return (m.date is None, m.date or 0, m.time_seconds)

def _parse_series_meta(stem_with_ext_removed_or_basename: str) -> Optional[_Parsed]:
    """
    Accepts a basename without caring about directory path.
    Expects a stem like "...-YYYYMMDD-HHMMSS(.frac)-DESC" or "...--HHMMSS(.frac)-DESC".
    Returns None if pattern not recognized.
    """
    stem = stem_with_ext_removed_or_basename
    if stem.lower().endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.lower().endswith(".nii"):
        stem = stem[:-4]

    # Pattern with date + time
    m = re.search(
        r"^(?P<sid>.+)-(?P<date>\d{8})-(?P<time>\d{6}(?:\.\d+)?)-(?P<desc>.+)$", stem)
    if m:
        return _Parsed(
            series_id=m.group("sid"),
            date=int(m.group("date")),
            time_seconds=_time_to_seconds(m.group("time")),
            series_desc=m.group("desc"),
        )

    # Pattern with only time (double hyphen)
    m = re.search(
        r"^(?P<sid>.+)--(?P<time>\d{6}(?:\.\d+)?)-(?P<desc>.+)$", stem)
    if m:
        return _Parsed(
            series_id=m.group("sid"),
            date=None,
            time_seconds=_time_to_seconds(m.group("time")),
            series_desc=m.group("desc"),
        )

    # Fallback: single hyphen before time, no date
    m = re.search(
        r"^(?P<sid>.+)-(?P<time>\d{6}(?:\.\d+)?)-(?P<desc>.+)$", stem)
    if m:
        return _Parsed(
            series_id=m.group("sid"),
            date=None,
            time_seconds=_time_to_seconds(m.group("time")),
            series_desc=m.group("desc"),
        )
    return None

def _time_to_seconds(timestr: str) -> float:
    mm = re.fullmatch(
        r"(?P<hh>\d{2})(?P<mm>\d{2})(?P<ss>\d{2})(?:\.(?P<frac>\d+))?", timestr)
    if not mm:
        return 0.0
    hh = int(mm.group("hh"))
    mn = int(mm.group("mm"))
    ss = int(mm.group("ss"))
    frac = mm.group("frac")
    f = 0.0 if not frac else float("0." + frac)
    return hh * 3600 + mn * 60 + ss + f

def _has_token_pre(series_desc: str) -> bool:
    # whole-token-ish "pre" (avoids 'pressure')
    return re.search(r"(?i)(^|[^a-z])pre([^a-z]|$)", series_desc) is not None

def find_nifti_files(exam_path: str) -> Tuple[str, List[str]]:
    """
    Runs BOTH strategies and merges the results:
      1) metadata-aware (precise) -> preferred
      2) your simple heuristic -> fallback/augmentation

    - Excludes any path with 'SUB' (case-insensitive).
    - De-duplicates posts.
    - Re-sorts posts using metadata ordering (date → time; undated last).
    """
    pre_simple = None
    posts_simple = []
    try:
        for root, _, files in os.walk(exam_path):
            for f in files:
                if f.endswith(('.nii', '.nii.gz')):
                    f_lower = f.lower()
                    full_path = os.path.join(root, f)

                    if (
                        'pre' in f_lower
                        # or ('t1' in f_lower and 'ce' not in f_lower)
                        or '000.nii' in f_lower
                    ):
                        pre_simple = full_path
                    elif (
                        'post' in f_lower
                        # or 't1ce' in f_lower
                        or any(f"{i:03}.nii" in f_lower for i in range(1, 8))
                    ):
                        posts_simple.append(full_path)
    except FileNotFoundError:
        return None, []

    pre_meta, posts_meta = find_nifti_files_by_metadata(exam_path)
    pre = pre_meta or pre_simple
    merged_posts = []
    seen = set()

    def _add(paths: Iterable[str]):
        for p in paths:
            if p and "sub" not in os.path.basename(p).lower() and p not in seen:
                seen.add(p)
                merged_posts.append(p)

    _add(posts_meta)
    _add(posts_simple)

    # Re-sort merged posts using metadata parsing (best effort)
    metas_for_sort: List[SeriesMeta] = []
    unsortable: List[str] = []
    for p in merged_posts:
        base = os.path.basename(p)
        parsed = _parse_series_meta(base)
        if parsed:
            metas_for_sort.append(SeriesMeta(
                original=p,
                series_id=parsed.series_id,
                date=parsed.date,
                time_seconds=parsed.time_seconds,
                series_desc=parsed.series_desc,
            ))
        else:
            unsortable.append(p)

    # Sort parseable ones (dated first), then append lexicographically the rest
    sorted_posts = [m.original for m in sorted(metas_for_sort, key=_sort_key)]
    sorted_posts += sorted(unsortable)

    return pre, sorted_posts

def infer_acquisition_plane(nifti_file_path: str, output_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Infers the acquisition plane (Axial, Coronal, Sagittal) of a NIfTI file.
    It uses the orientation codes from the NIfTI affine matrix.

    Returns:
        A tuple containing the inferred plane and a message with the axis codes.
    """
    try:
        img = nib.load(nifti_file_path)
        # canonical_img = nib.as_closest_canonical(img)
        axis_codes = nib.aff2axcodes(img.affine)
        # The third code (axis_codes[2]) tells us the slice direction
        slice_direction = axis_codes[2]
        if slice_direction in ('S', 'I'):  # Superior / Inferior
            plane = "Axial"
        elif slice_direction in ('A', 'P'):  # Anterior / Posterior
            plane = "Coronal"
        elif slice_direction in ('R', 'L'):  # Right / Left
            plane = "Sagittal"
        else:
            plane = "Unknown"

        reason = f"Slice direction is '{slice_direction}' from codes {axis_codes}"
        return plane, reason
    except Exception as e:
        log.error(f"Could not infer plane for {nifti_file_path}: {e}")
        return "Error", str(e)

def fov_lr_mm(nifti_path: str) -> float:

    try:
        img = nib.load(nifti_path)
        hdr = img.header
        affine = img.affine
        shape = img.shape

        lr_axis = None

        # --- Primary Method: Use nibabel's anatomical axis codes ---
        # This is the most reliable and direct way to find the L-R axis.
        axis_codes = nib.aff2axcodes(img.affine)
        try:
            if 'R' in axis_codes:
                lr_axis = axis_codes.index('R')
            elif 'L' in axis_codes:
                lr_axis = axis_codes.index('L')
        except ValueError:
            # This case should not happen if 'R' or 'L' is present.
            lr_axis = None

        # --- Fallback Method: For oblique images without a clear L-R axis code ---
        if lr_axis is None:
            log.warning(
                f"Could not determine L-R axis from axis codes {axis_codes} for "
                f"{os.path.basename(nifti_path)}. Falling back to affine matrix analysis. "
                "This may occur with highly oblique acquisitions."
            )
            # v_image = M_inv * v_world. The first column of M_inv gives the change
            # in image coordinates for a unit step in the world's X (L-R) direction.
            # The largest component of this vector tells us the dominant image axis.
            rotation_scaling_matrix_inv = np.linalg.inv(affine[:3, :3])
            lr_axis_vector = rotation_scaling_matrix_inv[:, 0]
            lr_axis = np.argmax(np.abs(lr_axis_vector))

        # --- Calculation ---
        # Get voxel spacing for the identified L-R axis.
        # NIfTI header's pixdim is 1-indexed, with pixdim[0] for qform/sform.
        lr_voxel_spacing = hdr['pixdim'][lr_axis + 1]

        # Get the number of voxels along the L-R axis.
        lr_num_voxels = shape[lr_axis]

        fov = math.ceil(lr_num_voxels * lr_voxel_spacing)

        log.info(
            f"  -> Axis Codes: {axis_codes}, Identified L-R Axis: {lr_axis}")
        log.info(
            f"  -> L-R Voxel Spacing: {lr_voxel_spacing:.4f} mm, Voxel Count: {lr_num_voxels}")
        log.info(f"  -> Calculated L-R FOV: {fov:.2f} mm")

        return fov

    except Exception as e:
        log.error(
            f"Failed to calculate L-R FOV for {nifti_path}: {e}", exc_info=True)
        return 0.0                  # physical FoV

def _to_uint8(vol_f32: np.ndarray) -> np.ndarray:
    lo, hi = np.min(vol_f32), np.max(vol_f32)
    if hi - lo < 1e-6:
        return np.zeros_like(vol_f32, dtype=np.uint8)
    vol = np.clip((vol_f32 - lo) / (hi - lo), 0, 1)
    return (vol * 255).astype(np.uint8)

def bilateral_by_symmetry(
    nifti_path: str,
    threshold_symmetry=0.7,
    output_path: str = None
) -> dict:
    img = nib.load(nifti_path)

    original_plane, reason = infer_acquisition_plane(nifti_path)
    log.info(
        f"  -> Original acquisition plane detected as: {original_plane}")

    axis_codes = nib.aff2axcodes(img.affine)
    try:
        lr_axis = axis_codes.index(
            'R') if 'R' in axis_codes else axis_codes.index('L')
        ap_axis = axis_codes.index(
            'A') if 'A' in axis_codes else axis_codes.index('P')
        si_axis = axis_codes.index(
            'S') if 'S' in axis_codes else axis_codes.index('I')
    except ValueError:
        log.error(
            f"Could not determine standard axes from {axis_codes}. Cannot process profile.")
        return False

    # --- 2. Orient Data Consistently and Get Voxel Spacing ---
    # Transpose the data so that the axes are always in (LR, AP, SI) order
    data = img.get_fdata().transpose(lr_axis, ap_axis, si_axis)

    # Get the voxel spacing for the Left-Right axis (now axis 0)
    lr_voxel_spacing = img.header.get_zooms()[lr_axis]
    if lr_voxel_spacing <= 0:
        log.error("Invalid Left-Right voxel spacing. Cannot process profile.")
        return False

    # --- 3. Create 1D Profile from a Single Central 2D Slice ---
    # For an axial acquisition, we analyze the central axial slice.
    # After transpose to (LR, AP, SI), this is the middle slice along the SI axis (axis 2).
    middle_slice_idx = data.shape[2] // 2
    central_slice_2d = data[:, :, middle_slice_idx]

    if central_slice_2d.size == 0:
        return False

    central_data_u8 = np.flipud(_to_uint8(central_slice_2d).T)
    if output_path:
        imsave(f"{output_path}_middle_slice.png",
               img_as_ubyte(central_data_u8))
    # Preprocessing
    # 1. Denoise
    img_denoised = cv2.bilateralFilter(central_data_u8, 9, 75, 75)

    # 2. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_denoised)

    # 3. Find the brain/tissue region (exclude background)
    _, binary = cv2.threshold(img_enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Find the largest connected component (main brain/tissue region)
    labels = measure.label(binary)

    # After obtaining `binary` and `labels`
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    # Collect areas of all components except the background
    areas = [(i, stats[i, 4]) for i in range(1, num_labels)]
    # Keep components whose area is at least 10 % of the largest
    main_components = [i for i, a in areas if a > 0.1*max(stats[1:, 4])]
    if len(main_components) < 2:
        output_dict = {'classification': 'unilateral', 'symmetry_score': 0.0,
                       'confidence': 1.0, 'affected_side': 'left/right'}
        return False, json.dumps(output_dict)

    regions = measure.regionprops(labels)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        mask = labels == largest_region.label
    else:
        mask = binary > 0

    # 6. Apply mask to enhanced image
    img_masked = img_enhanced * mask.astype(np.uint8)

    # 7. Find center of mass to determine midline
    center_y, center_x = ndimage.center_of_mass(mask)
    midline_x = int(center_x)

    # 8. Split image into left and right halves
    height, width = img_masked.shape
    left_half = img_masked[:, :midline_x]
    right_half = img_masked[:, midline_x:]

    # 9. Flip right half for comparison
    right_half_flipped = cv2.flip(right_half, 1)

    # 10. Ensure both halves have same width
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, -min_width:]
    right_half_flipped = right_half_flipped[:, :min_width]

    # 11. Calculate symmetry metrics
    # Metric 1: Pixel-wise difference
    diff = cv2.absdiff(left_half, right_half_flipped)
    diff_normalized = diff / 255.0

    # Only consider non-zero pixels (within the mask)
    mask_left = left_half > 0
    mask_right = right_half_flipped > 0
    mask_combined = mask_left | mask_right

    if np.sum(mask_combined) == 0:
        output_dict = {'classification': 'unknown',
                       'symmetry_score': 0,
                       'confidence': 0,
                       'affected_side': 'unknown'
                       }
        return False, json.dumps(output_dict)

    # Calculate mean absolute difference
    mean_diff = np.mean(diff_normalized[mask_combined])
    symmetry_score = 1 - mean_diff

    # Metric 2: Structural similarity
    hist_left = cv2.calcHist(
        [left_half], [0], mask_left.astype(np.uint8), [256], [0, 256])
    hist_right = cv2.calcHist([right_half_flipped], [
                              0], mask_right.astype(np.uint8), [256], [0, 256])

    hist_left = hist_left / (np.sum(hist_left) + 1e-7)
    hist_right = hist_right / (np.sum(hist_right) + 1e-7)

    # Calculate histogram correlation
    hist_correlation = cv2.compareHist(
        hist_left, hist_right, cv2.HISTCMP_CORREL)

    combined_symmetry = (symmetry_score + hist_correlation) / 2

    # Determine affected side for unilateral cases
    if combined_symmetry < threshold_symmetry:
        # Calculate mean intensity for each side
        left_intensity = np.mean(left_half[mask_left])
        right_intensity = np.mean(right_half_flipped[mask_right])

        # Calculate area of abnormality (high intensity regions)
        high_threshold = np.percentile(img_masked[img_masked > 0], 75)
        left_abnormal = np.sum(left_half > high_threshold)
        right_abnormal = np.sum(right_half > high_threshold)

        if left_abnormal > right_abnormal * 1.5:
            affected_side = 'left'
        elif right_abnormal > left_abnormal * 1.5:
            affected_side = 'right'
        else:
            affected_side = 'both' if left_abnormal > 0 and right_abnormal > 0 else 'unknown'
    else:
        affected_side = 'both'

    classification = 'bilateral' if round(
        combined_symmetry, 2) > threshold_symmetry else 'unilateral'

    if classification == 'bilateral':
        confidence = combined_symmetry
    else:
        confidence = 1 - combined_symmetry

    output_dict = {
        'classification': classification,
        'symmetry_score': combined_symmetry,
        'confidence': confidence,
        'affected_side': affected_side
    }

    return True if classification == 'bilateral' else False, json.dumps(output_dict)

def infer_bilateral_view(
    nifti_file_path: str,
    fov_interval: List[int] = (260, 280),
    output_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Infers if a NIfTI scan is bilateral or unilateral based on a set of rules.
    This is a refactoring of the original is_scan_bilateral method.

    Returns:
        A tuple containing the inferred laterality ('bilateral' or 'unilateral')
        and a message explaining the decision.
    """
    # 2) Header FOV check
    try:
        acqui_plane = infer_acquisition_plane(nifti_file_path=nifti_file_path)
        log.info(f"  -> Acquisition plane: {acqui_plane}")
        lr_fov = fov_lr_mm(nifti_file_path)
        patient_width = 0.8 * lr_fov
        adaptive_fov_cut = max(fov_interval[0], patient_width)
        # if fov_interval[0] <= lr_fov <= fov_interval[1]:
        #     log.info(
        #         f"  -> FOV check: {lr_fov:.2f}mm is within the interval {fov_interval}")
        # elif lr_fov >= adaptive_fov_cut:
        if lr_fov >= adaptive_fov_cut:
            return "bilateral", f"FOV={lr_fov:.1f}mm >= adaptive_cut={adaptive_fov_cut:.1f}mm"
    except Exception as e:
        log.warning(
            f"Could not perform FOV check on {nifti_file_path}: {e}")

    # 3) Symmetry fallback
    bilateral, msg = bilateral_by_symmetry(
        nifti_file_path, threshold_symmetry=0.9, output_path=output_path)
    if bilateral:
        return "bilateral", msg

    return "unilateral", "Unilateral by combined rules"
