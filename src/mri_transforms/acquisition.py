"""Combine acquisition evidence and preserve confidence and disagreement reasons."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from .geometry import (REFERENCE_ABSTAIN_MM, REFERENCE_BILATERAL_FOV_MM, REFERENCE_MIN_SIDE_MM,
                       REFERENCE_OBLIQUE_DEG, AxisGeometry, scope_for)

# --------------------------------------------------------------------------------------
# Layer 1 / Layer 2 resolution
# --------------------------------------------------------------------------------------
# See mri-analysis/docs/DECISION_LADDERS.md. The DICOM's own statement outranks anything
# reconstructed from the NIfTI grid, because it is the acquisition's record rather than an
# inference from it. But the loser is never discarded: a conflict between the two layers is the
# highest-yield quality signal in the stack, and it is the only thing that can catch a
# mislabelled tag or a converter that reordered the axes.

AGREE, CONFLICT = "agree", "conflict"
DICOM_ONLY, NIFTI_ONLY = "dicom_only", "nifti_only"

#: The one `breast_coverage_source` value that means the DICOM layer MEASURED coverage rather
#: than reporting which breast the exam is about. Only a conflict against a measurement is a
#: laterality conflict; a conflict against `Laterality` or a description is the two layers
#: answering different questions, which is expected and not a defect.
GEOMETRY_SOURCE = "image_geometry"

#: Conversion statuses that are not a pipeliner complaint. `""` is "the column is absent",
#: which must not be read as a failure.
CLEAN_CONVERSION_STATUSES = ("", "converted")


def resolve(dicom_value: Optional[str], nifti_value: Optional[str],
            prefer: str = "nifti") -> Tuple[str, str, str]:
    """Combine DICOM and NIfTI evidence using the requested precedence.

    Return (verdict, source, agreement), retaining conflicts for review.
    """
    dicom = (dicom_value or "").strip().lower() or None
    nifti = (nifti_value or "").strip().lower() or None
    if dicom and nifti:
        agreement = AGREE if dicom == nifti else CONFLICT
        if agreement == AGREE:
            return dicom, "both", AGREE
        winner, source = ((dicom, "dicom") if prefer == "dicom" else (nifti, "nifti"))
        return winner, source, CONFLICT
    if nifti:
        return nifti, "nifti", NIFTI_ONLY
    if dicom:
        return dicom, "dicom", DICOM_ONLY
    return "unknown", "none", NIFTI_ONLY


def acquisition_flags(geom: AxisGeometry, *,
                      plane_agreement: str,
                      scope_agreement: str,
                      scope_dicom_source: str = "",
                      pipeliner_row_found: Optional[bool] = None,
                      conversion_status: str = "",
                      fov_cut_mm: float = REFERENCE_BILATERAL_FOV_MM,
                      abstain_mm: float = REFERENCE_ABSTAIN_MM,
                      min_side_mm: float = REFERENCE_MIN_SIDE_MM,
                      oblique_deg: float = REFERENCE_OBLIQUE_DEG) -> Dict[str, bool]:
    """Return review flags for geometry, metadata conflicts and conversion status.

    For pipeliner_row_found, True means the series has metadata, False means
    expected metadata is missing, and None means no DICOM metadata layer is available.
    """
    flags = {
        "flag_multi_volume": geom.n_volumes > 1,
        "flag_near_cut": (abs(geom.fov_lr_mm - fov_cut_mm) <= abstain_mm
                          or abs(geom.min_side_mm - min_side_mm) <= abstain_mm),
        "flag_midline_unknown": not geom.frame_usable,
        "flag_oblique": geom.oblique_deg > oblique_deg,
        "flag_read_error": bool(geom.error),
    }
    if pipeliner_row_found is not None:
        flags["flag_pipeliner_no_metadata"] = not pipeliner_row_found
        flags["flag_pipeliner_conversion"] = (
            pipeliner_row_found
            and (conversion_status or "") not in CLEAN_CONVERSION_STATUSES)
    # A plane conflict means a converter may have rewritten the grid: DICOM supplies the
    # verdict, but the exam still must not be acted on unattended.
    flags["flag_plane_conflict"] = plane_agreement == CONFLICT
    # Only a conflict against a MEASURED coverage is a laterality conflict -- see
    # GEOMETRY_SOURCE.
    flags["flag_laterality_conflict"] = (scope_agreement == CONFLICT
                                         and scope_dicom_source == GEOMETRY_SOURCE)
    return flags


@dataclass
class AcquisitionVerdict:
    """What one volume's Acquisition Plane and Acquisition Scope are, and whether to believe it.

    Both layers survive in the record (`plane_nifti`/`plane_dicom`, `scope_nifti`/`scope_dicom`)
    so a verdict can be re-derived or re-thresholded later without re-reading the volume.
    """

    plane: str
    plane_source: str                  #: both | dicom | nifti | none
    plane_agreement: str               #: agree | conflict | dicom_only | nifti_only
    scope: str
    scope_source: str
    scope_agreement: str
    plane_nifti: str
    scope_nifti: str
    plane_dicom: str
    scope_dicom: str
    scope_dicom_source: str            #: which rule decided Layer 1: tag, text, image_geometry
    margin_mm: float                   #: fov_lr_mm - fov_cut_mm; its absolute value is distance to the cut
    reason: str
    scope_certain: bool                #: `scope_for`'s abstention, before any flag is considered
    flags: Dict[str, bool] = field(default_factory=dict)
    trusted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """A flat provenance record: the verdicts, the sources, the flags and `trusted`.

        Flat rather than nested because this is written into run artifacts that are read with
        `dict.get`, and a nested block makes every consumer handle a missing level.
        """
        return {
            "plane": self.plane, "plane_source": self.plane_source,
            "plane_agreement": self.plane_agreement,
            "scope": self.scope, "scope_source": self.scope_source,
            "scope_agreement": self.scope_agreement,
            "plane_nifti": self.plane_nifti, "scope_nifti": self.scope_nifti,
            "plane_dicom": self.plane_dicom, "scope_dicom": self.scope_dicom,
            "scope_dicom_source": self.scope_dicom_source,
            "margin_mm": self.margin_mm, "reason": self.reason,
            "scope_certain": self.scope_certain,
            "flags": dict(self.flags), "trusted": self.trusted,
        }


def characterise_acquisition(geom: AxisGeometry, *,
                             plane_dicom: str = "",
                             scope_dicom: str = "",
                             scope_dicom_source: str = "",
                             pipeliner_row_found: Optional[bool] = None,
                             conversion_status: str = "",
                             fov_cut_mm: float = REFERENCE_BILATERAL_FOV_MM,
                             abstain_mm: float = REFERENCE_ABSTAIN_MM,
                             min_side_mm: float = REFERENCE_MIN_SIDE_MM,
                             oblique_deg: float = REFERENCE_OBLIQUE_DEG) -> AcquisitionVerdict:
    """Build one `AcquisitionVerdict` from a measured header and whatever DICOM stated.

    Applies this order, and the order matters: measure, fuse, flag, then trust. `scope_for`
    abstains on the geometry alone; the flags add everything geometry cannot see (a missing
    pipeliner row, a failed conversion, a cross-layer conflict); `trusted` is the conjunction
    and is therefore the only field a caller should gate on.

    `trusted` is `not (any flag raised or geometry abstained)`. It is a demand for attention,
    never a replacement verdict: nothing here reroutes an exam or rewrites a caller's own
    laterality decision.

    Pass `plane_dicom`/`scope_dicom` empty -- the default -- when there is no DICOM layer.
    The verdict then comes from the NIfTI alone with `nifti_only` agreement, which is a
    truthful single-layer record rather than a fabricated consensus.
    """
    scope_nifti, margin_mm, reason, certain = scope_for(
        geom, fov_cut_mm=fov_cut_mm, abstain_mm=abstain_mm, min_side_mm=min_side_mm)
    # PLANE: DICOM wins a conflict -- it states the orientation directly, while our
    # axcodes[2] rule is a proxy a converter can break. SCOPE: our measurement wins --
    # the DICOM tag mostly answers a different question. See `resolve`.
    plane, plane_src, plane_agree = resolve(plane_dicom, geom.plane, prefer="dicom")
    scope, scope_src, scope_agree = resolve(scope_dicom, scope_nifti, prefer="nifti")
    flags = acquisition_flags(
        geom, plane_agreement=plane_agree, scope_agreement=scope_agree,
        scope_dicom_source=scope_dicom_source, pipeliner_row_found=pipeliner_row_found,
        conversion_status=conversion_status, fov_cut_mm=fov_cut_mm, abstain_mm=abstain_mm,
        min_side_mm=min_side_mm, oblique_deg=oblique_deg)
    return AcquisitionVerdict(
        plane=plane, plane_source=plane_src, plane_agreement=plane_agree,
        scope=scope, scope_source=scope_src, scope_agreement=scope_agree,
        plane_nifti=geom.plane, scope_nifti=scope_nifti,
        plane_dicom=plane_dicom, scope_dicom=scope_dicom,
        scope_dicom_source=scope_dicom_source,
        margin_mm=margin_mm, reason=reason, scope_certain=certain,
        flags=flags, trusted=not (any(flags.values()) or not certain))



#: The landmark execution each scope implies when the two decisions agree.
_EXECUTION_OF_SCOPE = {"bilateral": "two_halves", "unilateral": "full_width"}

KNOWN_SCOPES = ("bilateral", "unilateral")
EXECUTIONS = ("two_halves", "full_width")


def pipeline_laterality(source_path: str) -> Dict[str, Any]:
    """Call the same shared classifier as lesion inference, retaining failures as unknown."""
    from mri_transforms.utils import infer_bilateral_view

    try:
        scope, reason = infer_bilateral_view(source_path)
        if scope not in KNOWN_SCOPES:
            raise ValueError(f"unexpected pipeline scope {scope!r}")
        return {"pipeline_scope": scope, "pipeline_reason": reason,
                "pipeline_source": "infer_bilateral_view", "pipeline_error": ""}
    except Exception as exc:
        return {"pipeline_scope": None, "pipeline_reason": None,
                "pipeline_source": "infer_bilateral_view", "pipeline_error": str(exc)}


def measured_scope_of(coverage: Dict[str, Any]) -> Optional[str]:
    """The landmark resolver's MEASUREMENT, or None when it did not make one.

    Tested with `in`, never with `or`. `resolve_coverage` writes `measured_scope: None` for an
    explicit override, which means "no measurement was taken" -- an `or` fallback there reports
    the forced contract in `scope` as if geometry had measured it, which is the one answer that
    must not be invented. A manifest predating the key has no `measured_scope` at all, and for
    those `scope` legitimately IS the measurement.
    """
    return coverage.get("measured_scope") if "measured_scope" in coverage else coverage.get("scope")


def execution_of(coverage: Dict[str, Any]) -> Optional[str]:
    """Read landmark execution, deriving it from scope when execution is absent."""
    execution = coverage.get("execution")
    if execution is None and coverage:
        execution = _EXECUTION_OF_SCOPE.get(coverage.get("scope"), "full_width")
    return execution


def compare_coverage(pipeline_scope: Optional[str], coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Separate measurement disagreement from a different execution contract.

    An explicit null measured_scope means an override, not a geometry measurement.
    Older manifests without that key use scope as their legacy measurement.
    """
    measured = measured_scope_of(coverage)
    execution = execution_of(coverage)
    known = pipeline_scope in KNOWN_SCOPES
    measured_agrees = (measured == pipeline_scope
                       if known and measured in KNOWN_SCOPES else None)
    execution_agrees = (execution == _EXECUTION_OF_SCOPE[pipeline_scope]
                        if known and execution in EXECUTIONS else None)
    if measured_agrees is None or execution_agrees is None:
        difference = "unavailable"
    elif measured_agrees and execution_agrees:
        difference = "none"
    elif measured_agrees:
        difference = "landmark_execution_abstention"
    elif execution_agrees:
        difference = "measurement_only"
    else:
        difference = "measurement_and_execution"
    return {"pipeline_measured_agreement": measured_agrees,
            "pipeline_execution_agreement": execution_agrees,
            "laterality_difference_type": difference}


def laterality_comparison(pipeline_scope: Optional[str], pipeline_reason: Optional[str],
                          coverage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a structured comparison of acquisition scope and landmark execution.

    Missing coverage returns None. Unavailable comparisons retain null agreement
    values and an explicit difference_type.
    """
    if not coverage:
        return None
    scope = str(pipeline_scope).lower() if pipeline_scope is not None else None
    compared = compare_coverage(scope, coverage)
    return {
        "lesion_pipeline": {
            "scope": scope,
            "execution": _EXECUTION_OF_SCOPE.get(scope, "full_width"),
            "source": "infer_bilateral_view",
            "reason": pipeline_reason,
        },
        "landmark": {
            "measured_scope": measured_scope_of(coverage),
            "execution_scope": coverage.get("scope"),
            "execution": execution_of(coverage),
            "certain": coverage.get("certain"),
            "source": coverage.get("source"),
            "margin_mm": coverage.get("margin_mm"),
            "reason": coverage.get("reason"),
        },
        "measured_agreement": compared["pipeline_measured_agreement"],
        "execution_agreement": compared["pipeline_execution_agreement"],
        "difference_type": compared["laterality_difference_type"],
    }


def summarize_comparisons(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Count each exam once, including failed inference and explicit overrides.

    The row contract is the shared one: `case`, the three keys `compare_coverage` returns,
    plus `pipeline_scope`, `coverage_measured_scope`, `landmark_execution` and
    `coverage_certain`. An exam missing either agreement is counted as unavailable rather
    than as agreement, so an abstention can never inflate the agreement rate.
    """
    exams = {r["case"]: r for r in rows}.values()
    exams = list(exams)
    comparable = [r for r in exams if r.get("pipeline_measured_agreement") is not None
                  and r.get("pipeline_execution_agreement") is not None]

    def counts(key):
        return dict(sorted(Counter(r[key] for r in exams if r.get(key) is not None).items()))

    def differences(key):
        cases = sorted(r["case"] for r in comparable if r.get(key) is False)
        return {"count": len(cases), "rate": len(cases) / len(comparable) if comparable else None,
                "exams": cases}

    return {
        "schema_version": 1,
        "exams": len(exams), "comparable_exams": len(comparable),
        "unavailable_exams": len(exams) - len(comparable),
        "lesion_pipeline_scope_counts": counts("pipeline_scope"),
        "landmark_measured_scope_counts": counts("coverage_measured_scope"),
        "landmark_execution_counts": counts("landmark_execution"),
        "difference_type_counts": counts("laterality_difference_type"),
        "measured_scope_differences": differences("pipeline_measured_agreement"),
        "execution_differences": differences("pipeline_execution_agreement"),
        "uncertain_exams": sorted(r["case"] for r in exams
                                  if r.get("coverage_certain") is False),
    }
