"""Shared breast-region and nipple inference, fusion, cache and scoring.

Model execution requires the mri_transforms[landmarks] extra. Importing this
package or its evaluation module does not load PyTorch, MLflow or model weights.

The registered model defaults live in `mri_transforms.models`, outside this package, so a
consumer can still name the model it would run while reporting that this package is missing.
They are re-exported here for callers that already import from the landmark namespace.
"""
from __future__ import annotations

from ..models import (LEGACY_WHOLE_EXAM_MODEL_NAME, LEGACY_WHOLE_EXAM_MODEL_VERSION,
                      REFERENCE_MODEL_NAME, REFERENCE_MODEL_VERSION, REFERENCE_TRACKING_URI)

__all__ = ["LandmarkRunner", "REFERENCE_MODEL_NAME", "REFERENCE_MODEL_VERSION",
           "REFERENCE_TRACKING_URI", "LEGACY_WHOLE_EXAM_MODEL_NAME",
           "LEGACY_WHOLE_EXAM_MODEL_VERSION"]


def __getattr__(name: str):
    if name == "LandmarkRunner":
        from .runner import LandmarkRunner
        return LandmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
