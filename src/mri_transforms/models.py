"""Model identifiers and defaults; importing this module performs no inference."""
from __future__ import annotations

__all__ = ["REFERENCE_MODEL_NAME", "REFERENCE_MODEL_VERSION", "REFERENCE_TRACKING_URI",
           "LEGACY_WHOLE_EXAM_MODEL_NAME", "LEGACY_WHOLE_EXAM_MODEL_VERSION"]

#: The current registered default. `3D_landmarks_per_breast` v1 is the first model of this
#: family whose heads are 1-channel sigmoid and which consumes ONE Breast Half per forward
#: pass. Dispatch between the per-breast and whole-exam paths is driven by the registered
#: config's `preprocessing.input_coverage`, never by this name -- so pointing a consumer at
#: the predecessor keeps working.
REFERENCE_MODEL_NAME = "3D_landmarks_per_breast"
REFERENCE_MODEL_VERSION = "1"

#: The research MLflow registry. A deployment with its own registry passes `--tracking-uri`.
REFERENCE_TRACKING_URI = None

LEGACY_WHOLE_EXAM_MODEL_NAME = "nipples_breast_region_v2"
LEGACY_WHOLE_EXAM_MODEL_VERSION = "2"
