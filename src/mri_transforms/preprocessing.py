import SimpleITK as sitk

import numpy as np

def reorient_to_ras(image_itk: sitk.Image) -> sitk.Image:
    """
    Reorients a SimpleITK image to RAS (Right-Anterior-Superior) orientation.

    This is the SimpleITK equivalent of nibabel's `as_closest_canonical`.
    The filter uses nearest-neighbor interpolation by default, which is
    safe for both images and segmentation masks.

    Args:
        image_itk (sitk.Image): The input SimpleITK image.

    Returns:
        sitk.Image: The reoriented image.
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")
    return orient_filter.Execute(image_itk)

def normalize_intensity_zscore(image_np, mask_np=None):
    """Applies Z-score normalization. Uses mask if provided for statistics."""
    if image_np.size == 0:
        return image_np

    if mask_np is not None and np.any(mask_np):
        masked_pixels = image_np[mask_np.astype(bool)]
        if masked_pixels.size == 0:  # Mask is present but selects no pixels
            mean_val = np.mean(image_np)
            std_val = np.std(image_np)
        else:
            mean_val = np.mean(masked_pixels)
            std_val = np.std(masked_pixels)
    else:
        mean_val = np.mean(image_np)
        std_val = np.std(image_np)

    if std_val > 1e-6:  # Avoid division by zero
        return (image_np - mean_val) / std_val
    else:
        return image_np - mean_val
