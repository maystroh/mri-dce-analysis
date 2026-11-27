"""Serializable per-breast geometry. Arrays at this API boundary use XYZ order.

No training-framework dependencies. Preserve the returned record with every prepared
sample; reconstruction uses physical affines, not a resize to an approximate shape.
"""
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def split_image(image, coverage="bilateral"):
    """Return (side, RAS half, geometry) tuples without mirroring anatomy."""
    if coverage not in ("bilateral", "unilateral"):
        raise ValueError("coverage must be bilateral or unilateral")
    if len(image.shape) != 3:
        raise ValueError("Expected a three-dimensional image")
    ras = nib.as_closest_canonical(image)
    if nib.aff2axcodes(ras.affine) != ("R", "A", "S"):
        raise ValueError("Cannot establish RAS orientation")
    width = ras.shape[0]
    if coverage == "bilateral" and width < 2:
        raise ValueError("Bilateral image must have at least two LR voxels")
    spans = [("left", 0, width // 2), ("right", width // 2, width)]
    if coverage == "unilateral":
        spans = [("unilateral", 0, width)]
    result = []
    for side, start, stop in spans:
        half = ras.slicer[start:stop, :, :]
        geometry = {
            "version": 1, "side": side, "coverage": coverage,
            "parent_shape": list(image.shape), "parent_affine": image.affine.tolist(),
            "ras_shape": list(ras.shape), "ras_affine": ras.affine.tolist(),
            "half_shape": list(half.shape), "half_affine": half.affine.tolist(),
            "x_slice": [start, stop],
        }
        result.append((side, half, geometry))
    return result


def restore_half(array_xyz, prepared_affine, geometry, order=1, crop_bbox_xyz=None,
                 prepared_shape=None, fill_value=0):
    """Undo an optional crop then resample to the exact source RAS half grid."""
    array_xyz = np.asarray(array_xyz)
    if array_xyz.ndim != 3:
        raise ValueError("Expected one XYZ map")
    if crop_bbox_xyz is not None:
        full = np.full(prepared_shape, fill_value, dtype=array_xyz.dtype)
        slices = tuple(slice(*bounds) for bounds in crop_bbox_xyz)
        if full[slices].shape != array_xyz.shape:
            raise ValueError("Prediction shape does not match recorded crop")
        full[slices] = array_xyz
        array_xyz = full
    image = nib.Nifti1Image(array_xyz.astype(np.float32), np.asarray(prepared_affine))
    return resample_from_to(image, (tuple(geometry["half_shape"]),
                                    np.asarray(geometry["half_affine"])),
                            order=order, mode="nearest").get_fdata(dtype=np.float32)


def reconstruct_parent(halves, order=1):
    """Combine (restored XYZ half, geometry) pairs into the native parent image.

    Require exactly one complete set. Never invent a missing side as background.
    """
    if not halves:
        raise ValueError("No halves supplied")
    reference = halves[0][1]
    expected = {"left", "right"} if reference["coverage"] == "bilateral" else {"unilateral"}
    sides = [g["side"] for _, g in halves]
    if len(sides) != len(expected) or set(sides) != expected:
        raise ValueError(f"Incomplete or duplicate parent parts: {sides}, expected {expected}")
    array = np.zeros(reference["ras_shape"], dtype=np.float32)
    occupied = np.zeros(reference["ras_shape"][0], dtype=bool)
    for half, geometry in halves:
        for key in ("parent_shape", "parent_affine", "ras_shape", "ras_affine", "coverage"):
            if geometry[key] != reference[key]:
                raise ValueError(f"Parts have inconsistent {key}")
        start, stop = geometry["x_slice"]
        if not 0 <= start < stop <= array.shape[0] or occupied[start:stop].any():
            raise ValueError("Invalid or overlapping half slices")
        if tuple(half.shape) != tuple(geometry["half_shape"]):
            raise ValueError("Half has not been restored to its source grid")
        array[start:stop] = half
        occupied[start:stop] = True
    if not occupied.all():
        raise ValueError("Parent coverage is incomplete")
    ras = nib.Nifti1Image(array, np.asarray(reference["ras_affine"]))
    return resample_from_to(ras, (tuple(reference["parent_shape"]),
                                 np.asarray(reference["parent_affine"])),
                            order=order, mode="nearest")
