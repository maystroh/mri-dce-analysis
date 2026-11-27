"""Acquisition geometry measurements and scope policy."""
from __future__ import annotations

from typing import Tuple

from mri_transforms.geometry import (  # noqa: F401  (re-exported for callers)
    AxisGeometry,
    axis_geometry,
    calibrate_fov_cut,
    scope_for,
    REFERENCE_ABSTAIN_MM,
    REFERENCE_BILATERAL_FOV_MM,
    REFERENCE_MIN_SIDE_MM,
    REFERENCE_OBLIQUE_DEG,
)
from mri_transforms.acquisition import (  # noqa: F401  (re-exported for callers)
    AGREE,
    CONFLICT,
    DICOM_ONLY,
    NIFTI_ONLY,
    GEOMETRY_SOURCE,
    AcquisitionVerdict,
    acquisition_flags,
    characterise_acquisition,
    resolve,
)

# The cuts this project applies. They are the library's calibrated reference values: an
# application is free to choose others, and this one has chosen these. Bound by name rather
# than repeated as literals so a re-calibration in the library cannot silently disagree
# with what stage 1 runs.
BILATERAL_FOV_MM = REFERENCE_BILATERAL_FOV_MM
MIN_SIDE_COVERAGE_MM = REFERENCE_MIN_SIDE_MM
ABSTAIN_MARGIN_MM = REFERENCE_ABSTAIN_MM
# Tilt beyond which stage 1 raises `flag_oblique`. The same angle the library uses to refuse
# certainty and the per-breast split uses to refuse a cut -- one number, three decisions.
OBLIQUE_MAX_DEG = REFERENCE_OBLIQUE_DEG


def scope_by_fov(geom: AxisGeometry, fov_cut_mm: float = BILATERAL_FOV_MM,
                 abstain_mm: float = ABSTAIN_MARGIN_MM,
                 min_side_mm: float = MIN_SIDE_COVERAGE_MM) -> Tuple[str, float, str, bool]:
    """Apply this project's cuts. See `mri_transforms.geometry.scope_for` for the rule.

    Kept as the name stage 1 calls so the application has one place to change policy.
    """
    return scope_for(geom, fov_cut_mm=fov_cut_mm, abstain_mm=abstain_mm,
                     min_side_mm=min_side_mm)
