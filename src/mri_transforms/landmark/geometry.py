"""Geometry for the 3D landmark model: source grid -> model grid -> source grid.

The model was trained on volumes that had been reoriented to RAS, resampled to 1.5 mm
isotropic, cropped to their non-zero bounding box and z-scored. Inference has to reproduce that
chain exactly, then put the prediction back on the grid the exam arrived on.

The single most useful property here is that **every step preserves physical space**:
reorientation permutes and flips axes, resampling regrids, and cropping only shifts the origin.
None of them moves the patient. So the inverse is not three operations undone in reverse — it is
ONE resample onto the source image as reference (`resample_like`), which is exact because the
geometry was carried along rather than recomputed.

Matching spacing alone is NOT enough to invert: the origin and direction have to come from the
source grid too, or the mask has the right voxel size and still fails to overlay.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from mri_transforms.preprocessing import normalize_intensity_zscore, reorient_to_ras
from mri_transforms.utils import log

DEFAULT_IMAGE_INTERPOLATOR = "bspline"

_INTERPOLATOR_BY_NAME = {
    "bspline": sitk.sitkBSpline,
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "nearestneighbor": sitk.sitkNearestNeighbor,
}


def resolve_interpolator(name: str) -> int:
    """Map a config's `image_interpolator`/`label_interpolator` string to its sitk constant."""
    key = str(name).strip().lower()
    try:
        return _INTERPOLATOR_BY_NAME[key]
    except KeyError:
        raise ValueError(f"unknown interpolator {name!r}; expected one of "
                         f"{sorted(_INTERPOLATOR_BY_NAME)}")


# --------------------------------------------------------------------------------------
# individual steps
# --------------------------------------------------------------------------------------

def read_image(path: str) -> sitk.Image:
    """Read a NIfTI as float32. 4D volumes keep their first component only."""
    img = sitk.ReadImage(str(path))
    if img.GetDimension() == 4:
        img = img[:, :, :, 0]
    return sitk.Cast(img, sitk.sitkFloat32)


def resampled_size(size_xyz: Sequence[int], spacing_xyz: Sequence[float],
                   target_xyz: Sequence[float]) -> Tuple[int, ...]:
    """Output size for a spacing change — same formula and rounding the dataset prep used.

    Lifted from `prepare_3d_pipeline/resample_size_report.py` so the inference grid matches the
    training grid voxel for voxel. Rounding matters: `int()` instead of `round()` here shifts the
    grid by up to one voxel against what the model saw.
    """
    return tuple(max(1, int(round(osz * ospc / tspc)))
                 for osz, ospc, tspc in zip(size_xyz, spacing_xyz, target_xyz))


def resample_to_spacing(img: sitk.Image, target_spacing_xyz: Sequence[float],
                        interpolator=sitk.sitkLinear) -> sitk.Image:
    """Regrid to a fixed voxel spacing, keeping origin and direction."""
    size = resampled_size(img.GetSize(), img.GetSpacing(), target_spacing_xyz)
    f = sitk.ResampleImageFilter()
    f.SetSize([int(s) for s in size])
    f.SetOutputSpacing([float(s) for s in target_spacing_xyz])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetTransform(sitk.Transform())
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(0)
    f.SetOutputPixelType(img.GetPixelID())
    return f.Execute(img)


def nonzero_bbox_zyx(volume_zyx: np.ndarray) -> List[List[int]]:
    """nnU-Net-style non-zero bounding box, as `[[z0,z1],[y0,y1],[x0,x1]]` (stop-exclusive).

    Reimplements `core.datasets.volume_3d._image_nonzero_bbox`. `binary_fill_holes` is not
    cosmetic: without it an interior zero-valued region can split the mask and shrink the box.
    An all-zero volume keeps its full field of view rather than collapsing to nothing.
    """
    nonzero = binary_fill_holes(volume_zyx != 0)
    coords = np.argwhere(nonzero)
    if coords.size == 0:
        return [[0, int(s)] for s in volume_zyx.shape]
    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    return [[int(a), int(b)] for a, b in zip(starts, stops)]


def crop_to_nonzero(img: sitk.Image) -> Tuple[sitk.Image, List[List[int]]]:
    """Crop to the non-zero box. SimpleITK slicing updates the origin, so geometry survives."""
    bbox = nonzero_bbox_zyx(sitk.GetArrayViewFromImage(img))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    return img[x0:x1, y0:y1, z0:z1], bbox


def resample_like(array_or_image, reference: sitk.Image, source_geometry: sitk.Image,
                  interpolator=sitk.sitkNearestNeighbor) -> sitk.Image:
    """Put a prediction back on `reference`'s grid with ONE physical-space resample.

    `source_geometry` supplies the grid the prediction currently lives on (the cropped, resampled
    volume the model saw). Because that geometry was carried through every forward step, this one
    call undoes the crop, the spacing change AND the reorientation together.
    """
    if isinstance(array_or_image, np.ndarray):
        pred = sitk.GetImageFromArray(array_or_image)
        pred.CopyInformation(source_geometry)
    else:
        pred = array_or_image
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(reference)
    f.SetTransform(sitk.Transform())
    f.SetInterpolator(interpolator)
    f.SetDefaultPixelValue(0)
    f.SetOutputPixelType(pred.GetPixelID())
    return f.Execute(pred)


# --------------------------------------------------------------------------------------
# the whole forward chain
# --------------------------------------------------------------------------------------

@dataclass
class PreparedVolume:
    """Everything needed to run the model and to invert the result."""

    source: sitk.Image                    #: as read from disk, untouched
    model_grid: sitk.Image                #: RAS + resampled + cropped — the grid the model sees
    volume: np.ndarray                    #: (1, Z, Y, X) float32, z-scored — the model input
    crop_bbox_zyx: List[List[int]]        #: box in the resampled grid, for the audit trail
    source_path: Optional[str] = None
    steps: List[str] = field(default_factory=list)

    @property
    def spacing_xyz(self) -> Tuple[float, ...]:
        return tuple(self.model_grid.GetSpacing())

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.model_grid.GetSpacing()))


def prepare_volume(path: str, spacing_zyx: Sequence[float],
                   online_ops: Optional[Sequence[Dict[str, Any]]] = None,
                   image_interpolator: str = DEFAULT_IMAGE_INTERPOLATOR) -> PreparedVolume:
    """Run the registered `preprocessing` recipe over one exam volume.

    `spacing_zyx` comes from the config in (z, y, x) order; SimpleITK wants (x, y, z), and getting
    that backwards is silent whenever the spacing is isotropic — which it is here (1.5 mm), so the
    reversal is written out explicitly rather than left to be noticed later.

    `image_interpolator` must match `preprocessing.offline_ops[].image_interpolator` from the
    registered config — training resampled with that interpolator before z-scoring, so using a
    different one here changes voxel intensities and creates a train/inference distribution shift
    that Dice on a validation set would not necessarily catch, since it's a smooth, small-per-voxel
    shift rather than a gross geometric error.
    """
    online_ops = list(online_ops or [])
    steps: List[str] = []

    source = read_image(path)
    steps.append(f"read {tuple(source.GetSize())} @ {tuple(round(s, 4) for s in source.GetSpacing())}")

    img = reorient_to_ras(source)
    steps.append("reorient RAS")

    target_xyz = tuple(float(s) for s in reversed(list(spacing_zyx)))
    img = resample_to_spacing(img, target_xyz, resolve_interpolator(image_interpolator))
    steps.append(f"resample -> {target_xyz} mm, size {tuple(img.GetSize())}, "
                 f"interpolator={image_interpolator}")

    bbox = [[0, int(s)] for s in sitk.GetArrayViewFromImage(img).shape]
    volume = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)

    for op in online_ops:
        kind = op.get("op")
        if kind == "crop_to_nonzero":
            img, bbox = crop_to_nonzero(img)
            volume = sitk.GetArrayFromImage(img).astype(np.float32)
            steps.append(f"crop_to_nonzero -> {volume.shape} (zyx box {bbox})")
        elif kind == "normalize":
            method = op.get("method")
            if method != "z_score":
                raise ValueError(f"unsupported normalize method {method!r} for the landmark model")
            volume = normalize_intensity_zscore(volume).astype(np.float32)
            steps.append("normalize z_score")
        else:
            raise ValueError(f"unknown online op {kind!r} in the registered preprocessing recipe")

    for line in steps:
        log.info(f"      [landmark] {line}")

    return PreparedVolume(source=source, model_grid=img, volume=volume[None].astype(np.float32),
                          crop_bbox_zyx=bbox, source_path=str(path), steps=steps)
