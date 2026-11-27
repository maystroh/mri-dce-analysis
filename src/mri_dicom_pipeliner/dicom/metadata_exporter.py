"""Extract one-row-per-series metadata from any MRI DICOM tree."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pydicom


# PHI such as PatientName, birth date, address, and referring physician is
# intentionally excluded. PatientID is retained so rows can be joined to output.
SERIES_FIELDS = {
    "patient_id": "PatientID",
    "patient_sex": "PatientSex",
    "patient_age": "PatientAge",
    "patient_size_m": "PatientSize",
    "patient_weight_kg": "PatientWeight",
    "patient_identity_removed": "PatientIdentityRemoved",
    "deidentification_method": "DeidentificationMethod",
    "study_instance_uid": "StudyInstanceUID",
    "series_instance_uid": "SeriesInstanceUID",
    "frame_of_reference_uid": "FrameOfReferenceUID",
    "study_date": "StudyDate",
    "study_time": "StudyTime",
    "series_date": "SeriesDate",
    "series_time": "SeriesTime",
    "study_description": "StudyDescription",
    "series_description": "SeriesDescription",
    "protocol_name": "ProtocolName",
    "modality": "Modality",
    "sop_class_uid": "SOPClassUID",
    "body_part_examined": "BodyPartExamined",
    "series_number": "SeriesNumber",
    "acquisition_number": "AcquisitionNumber",
    "temporal_position_identifier": "TemporalPositionIdentifier",
    "number_of_temporal_positions": "NumberOfTemporalPositions",
    "patient_position": "PatientPosition",
    "laterality": "Laterality",
    "image_laterality": "ImageLaterality",
    "manufacturer": "Manufacturer",
    "manufacturer_model_name": "ManufacturerModelName",
    "device_serial_number": "DeviceSerialNumber",
    "station_name": "StationName",
    "software_versions": "SoftwareVersions",
    "institution_name": "InstitutionName",
    "magnetic_field_strength_t": "MagneticFieldStrength",
    "receive_coil_name": "ReceiveCoilName",
    "transmit_coil_name": "TransmitCoilName",
    "scanning_sequence": "ScanningSequence",
    "sequence_variant": "SequenceVariant",
    "scan_options": "ScanOptions",
    "mr_acquisition_type": "MRAcquisitionType",
    "sequence_name": "SequenceName",
    "image_type": "ImageType",
    "repetition_time_ms": "RepetitionTime",
    "echo_time_ms": "EchoTime",
    "inversion_time_ms": "InversionTime",
    "flip_angle_deg": "FlipAngle",
    "number_of_averages": "NumberOfAverages",
    "echo_train_length": "EchoTrainLength",
    "echo_number": "EchoNumbers",
    "imaging_frequency_mhz": "ImagingFrequency",
    "imaged_nucleus": "ImagedNucleus",
    "pixel_bandwidth_hz": "PixelBandwidth",
    "number_of_phase_encoding_steps": "NumberOfPhaseEncodingSteps",
    "percent_sampling": "PercentSampling",
    "percent_phase_fov": "PercentPhaseFieldOfView",
    "parallel_acquisition": "ParallelAcquisition",
    "parallel_reduction_factor_in_plane": "ParallelReductionFactorInPlane",
    "acquisition_duration_s": "AcquisitionDuration",
    "sar_w_kg": "SAR",
    "db_dt_t_s": "dBdt",
    "b1_rms_ut": "B1rms",
    "diffusion_b_value_s_mm2": "DiffusionBValue",
    "diffusion_gradient_orientation": "DiffusionGradientOrientation",
    "contrast_bolus_agent": "ContrastBolusAgent",
    "contrast_bolus_route": "ContrastBolusRoute",
    "contrast_bolus_volume_ml": "ContrastBolusVolume",
    "contrast_bolus_total_dose": "ContrastBolusTotalDose",
    "contrast_bolus_ingredient_concentration": "ContrastBolusIngredientConcentration",
    "contrast_bolus_start_time": "ContrastBolusStartTime",
    "rows": "Rows",
    "columns": "Columns",
    "pixel_spacing_mm": "PixelSpacing",
    "slice_thickness_mm": "SliceThickness",
    "spacing_between_slices_mm": "SpacingBetweenSlices",
    "reconstruction_diameter_mm": "ReconstructionDiameter",
    "acquisition_matrix": "AcquisitionMatrix",
    "in_plane_phase_encoding_direction": "InPlanePhaseEncodingDirection",
    "image_orientation_patient": "ImageOrientationPatient",
    "photometric_interpretation": "PhotometricInterpretation",
    "bits_stored": "BitsStored",
    "bits_allocated": "BitsAllocated",
    "pixel_representation": "PixelRepresentation",
    "lossy_image_compression": "LossyImageCompression",
    "lossy_image_compression_method": "LossyImageCompressionMethod",
    "lossy_image_compression_ratio": "LossyImageCompressionRatio",
}

DERIVED_FIELDS = [
    "series_group_id", "acquisition_system", "acquisition_view",
    "acquisition_view_tilt_deg", "breast_coverage", "breast_coverage_source",
    "coverage_from_tags", "coverage_from_text", "coverage_from_geometry",
    "coverage_agreement", "sequence_weighting", "contrast_status",
    "contrast_phase", "contrast_administered",
    "is_derived", "is_subtraction", "is_mip", "instance_count",
    "sop_instance_count", "frame_count", "slice_position_count",
    "instance_number_min", "instance_number_max", "missing_instance_numbers",
    "acquisition_time_first", "acquisition_time_last", "field_of_view_row_mm",
    "field_of_view_column_mm", "position_x_min_mm", "position_x_max_mm",
    "position_y_min_mm", "position_y_max_mm", "position_z_min_mm",
    "position_z_max_mm", "coverage_x_min_mm", "coverage_x_max_mm",
    "fov_lr_mm", "fov_lr_center_offset_mm", "fov_lr_asymmetry_mm",
    "geometry_consistent", "source_directory",
    "nifti_path", "conversion_status", "conversion_error",
]

CSV_FIELDS = list(SERIES_FIELDS) + DERIVED_FIELDS

ENHANCED_MR_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.4.1",  # Enhanced MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4.3",  # Enhanced Color MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4.4",  # Legacy Converted Enhanced MR
}


def _find_value(dataset, keyword, depth=0):
    """Read a keyword from classic or nested enhanced-MR attributes."""
    value = dataset.get(keyword)
    if value not in (None, ""):
        return value
    if depth >= 4:
        return None
    if depth == 0 and str(dataset.get("SOPClassUID", "")) not in ENHANCED_MR_SOP_CLASS_UIDS:
        return None
    for element in dataset:
        if element.VR != "SQ":
            continue
        for item in element.value:
            value = _find_value(item, keyword, depth + 1)
            if value not in (None, ""):
                return value
    return None


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return ""
    if isinstance(value, (str, int, float)):
        return value
    try:
        return "\\".join(str(item) for item in value)
    except TypeError:
        return str(value)


def acquisition_orientation(dataset) -> tuple[str, float | str]:
    orientation = _find_value(dataset, "ImageOrientationPatient")
    try:
        values = [float(value) for value in orientation]
        if len(values) != 6:
            return "unknown", ""
        row, column = values[:3], values[3:]
        normal = (
            row[1] * column[2] - row[2] * column[1],
            row[2] * column[0] - row[0] * column[2],
            row[0] * column[1] - row[1] * column[0],
        )
        magnitude = math.sqrt(sum(component * component for component in normal))
        if magnitude == 0:
            return "unknown", ""
        normalized = tuple(component / magnitude for component in normal)
        axis = max(range(3), key=lambda index: abs(normal[index]))
        alignment = min(1.0, max(0.0, abs(normalized[axis])))
        tilt = round(math.degrees(math.acos(alignment)), 6)
        if alignment < 0.8:
            return "oblique", tilt
        return ("sagittal", "coronal", "axial")[axis], tilt
    except (TypeError, ValueError):
        return "unknown", ""


def acquisition_view(dataset) -> str:
    return acquisition_orientation(dataset)[0]


# DICOM series descriptions join their tokens with underscores as often as with spaces
# ("Ax 3D T1 FS Post_SUB_MIP_SAG", "ORIGINAL\\PRIMARY\\MIP_SAG\\NORM\\DIS2D\\SUB"). `\b` is
# the wrong boundary for them: `_` is a word character, so `\bMIP\b` cannot match `_MIP_` and
# `\bSUB\b` cannot match `Post_SUB`. Every text rule in this module wants a *token* boundary —
# alphanumeric on either side blocks the match, anything else (space, `_`, `-`, `\`, `/`, `:`,
# start/end of string) allows it.
_TOKEN_BEFORE = r"(?<![0-9A-Za-z])"
_TOKEN_AFTER = r"(?![0-9A-Za-z])"


def _token_search(pattern: str, text: str):
    """`re.search` with token boundaries that treat `_` and `-` as separators.

    `pattern` is the bare alternation, without anchors — it is wrapped in a non-capturing
    group so alternations bind correctly.
    """
    return re.search(f"{_TOKEN_BEFORE}(?:{pattern}){_TOKEN_AFTER}", text, re.I)


def _is_breast_dataset(dataset) -> bool:
    text = " ".join(str(_find_value(dataset, key) or "") for key in (
        "BodyPartExamined", "StudyDescription", "SeriesDescription",
        "ProtocolName", "ReceiveCoilName",
    ))
    return bool(_token_search(r"breast|brst", text))


def _coverage_from_tags(dataset) -> tuple[str, str]:
    if not _is_breast_dataset(dataset):
        return "unknown", ""
    for keyword in ("ImageLaterality", "Laterality"):
        raw = str(_find_value(dataset, keyword) or "").strip().upper()
        if raw in {"B", "BOTH", "BILATERAL"}:
            return "bilateral", keyword
        if raw in {"L", "R", "LEFT", "RIGHT"}:
            return "unilateral", keyword
    return "unknown", ""


def _coverage_from_text(dataset) -> str:
    if not _is_breast_dataset(dataset):
        return "unknown"
    text = " ".join(str(_find_value(dataset, key) or "") for key in (
        "StudyDescription", "SeriesDescription", "ProtocolName", "ReceiveCoilName"
    ))
    if _token_search(r"bil|bilat|bilateral|both[ _-]*breasts?", text):
        return "bilateral"
    if _token_search(r"unilat|unilateral", text):
        return "unilateral"
    if _token_search(r"(?:left|right)[ _-]*(?:breast|brst)", text):
        return "unilateral"
    if _token_search(r"rt|lt", text):
        return "unilateral"
    return "unknown"


def breast_coverage(dataset) -> tuple[str, str]:
    """Return bilateral/unilateral only when tags or text support it."""
    tag_coverage, tag_source = _coverage_from_tags(dataset)
    if tag_coverage != "unknown":
        return tag_coverage, tag_source
    text_coverage = _coverage_from_text(dataset)
    if text_coverage != "unknown":
        return text_coverage, "text"
    return "unknown", ""


def sequence_weighting(dataset) -> str:
    text = " ".join(str(_find_value(dataset, key) or "") for key in (
        "SeriesDescription", "ProtocolName", "SequenceName", "ImageType"
    ))
    if _token_search(r"dwi|diff|adc", text):
        return "DWI"
    if _token_search(r"t2", text):
        return "T2"
    if _token_search(r"t1", text):
        return "T1"
    try:
        repetition = float(_find_value(dataset, "RepetitionTime") or 0)
        echo = float(_find_value(dataset, "EchoTime") or 0)
        if repetition and repetition < 1000 and echo and echo < 30:
            return "T1"
        if repetition > 2000 and echo > 60:
            return "T2"
    except (TypeError, ValueError):
        pass
    return "Unknown"


def _description_flags(dataset) -> tuple[str, str, str, str, str, str]:
    image_type = str(_csv_value(_find_value(dataset, "ImageType")))
    description = " ".join(str(_find_value(dataset, key) or "") for key in (
        "SeriesDescription", "ProtocolName", "ImageType"
    ))
    is_subtraction = bool(_token_search(r"sub(?:traction)?", description))
    is_mip = bool(_token_search(r"mip", description))
    # `is_derived` answers "is this pixel data a source acquisition?", which the ImageType
    # value-1 tag alone does not: Siemens ships subtractions and projections as
    # `ORIGINAL\PRIMARY\MIP_SAG\...`. Anything the tag marks derived stays derived, and the
    # reformat/projection classes are added on top, so this flag alone is a safe filter.
    is_derived = bool(
        _token_search(r"derived|secondary|resampled|reformatted|projection|mpr", image_type)
        or is_mip
        or is_subtraction
    )
    agent = str(_find_value(dataset, "ContrastBolusAgent") or "").strip()
    try:
        volume = float(_find_value(dataset, "ContrastBolusVolume") or 0)
    except (TypeError, ValueError):
        volume = 0
    negative = bool(_token_search(
        r"pre(?:[ _-]*(?:contrast|gad))?|"
        r"(?:without|wo|non)[ _-]*contrast",
        description,
    ))
    positive = bool(_token_search(r"post|sub|subtraction|gad", description))
    if is_subtraction:
        contrast_phase = "subtraction"
    elif negative:
        contrast_phase = "pre"
    elif positive:
        contrast_phase = "post"
    else:
        contrast_phase = "unknown"

    dicom_evidence = bool(agent or volume > 0)
    if dicom_evidence:
        contrast_administered = "yes_dicom"
    elif positive or _token_search(r"contrast", description):
        contrast_administered = "yes_inferred"
    elif _token_search(r"(?:without|wo|non)[ _-]*contrast", description):
        contrast_administered = "no_inferred"
    else:
        contrast_administered = "unknown"

    if contrast_phase == "pre":
        contrast_status = "no_inferred"
    elif contrast_phase in {"post", "subtraction"}:
        contrast_status = "yes_dicom" if dicom_evidence else "yes_inferred"
    elif dicom_evidence:
        contrast_status = "yes_dicom"
    else:
        contrast_status = "unknown"
    return (
        str(is_derived).lower(), str(is_subtraction).lower(),
        str(is_mip).lower(), contrast_status, contrast_phase,
        contrast_administered,
    )


def _float_list(value):
    try:
        return tuple(round(float(item), 6) for item in value)
    except (TypeError, ValueError):
        return ()


def _image_x_bounds(dataset):
    """Approximate patient left-right coverage from DICOM image geometry."""
    position = _float_list(_find_value(dataset, "ImagePositionPatient"))
    orientation = _float_list(_find_value(dataset, "ImageOrientationPatient"))
    spacing = _float_list(_find_value(dataset, "PixelSpacing"))
    try:
        rows = int(_find_value(dataset, "Rows"))
        columns = int(_find_value(dataset, "Columns"))
    except (TypeError, ValueError):
        return ()
    if len(position) != 3 or len(orientation) != 6 or len(spacing) != 2:
        return ()
    column_extent = orientation[0] * spacing[1] * max(columns - 1, 0)
    row_extent = orientation[3] * spacing[0] * max(rows - 1, 0)
    corners = (
        position[0], position[0] + column_extent,
        position[0] + row_extent, position[0] + column_extent + row_extent,
    )
    return min(corners), max(corners)


def _coverage_from_geometry(dataset, x_min, x_max):
    if not _is_breast_dataset(dataset) or x_min == "" or x_max == "":
        return None
    # DICOM patient X is the left-right axis. Require substantial coverage on
    # both sides of zero so tiny obliquity/position errors do not imply bilateral.
    if x_min <= -20 and x_max >= 20 and (x_max - x_min) >= 140:
        return "bilateral", "image_geometry"
    return None


def _coverage_summary(dataset, x_min, x_max) -> dict:
    tag_coverage, tag_source = _coverage_from_tags(dataset)
    text_coverage = _coverage_from_text(dataset)
    geometry_result = _coverage_from_geometry(dataset, x_min, x_max)
    geometry_coverage = geometry_result[0] if geometry_result else "unknown"

    known = [value for value in (
        tag_coverage, text_coverage, geometry_coverage
    ) if value != "unknown"]
    if not known:
        agreement = "no_evidence"
    elif len(set(known)) == 1 and len(known) > 1:
        agreement = "agree"
    elif len(set(known)) == 1:
        agreement = "single_source"
    else:
        agreement = "conflict"

    if tag_coverage != "unknown":
        resolved, source = tag_coverage, tag_source
    elif text_coverage != "unknown":
        resolved, source = text_coverage, "text"
    elif geometry_coverage != "unknown":
        resolved, source = geometry_coverage, "image_geometry"
    else:
        resolved, source = "unknown", ""
    return {
        "breast_coverage": resolved,
        "breast_coverage_source": source,
        "coverage_from_tags": tag_coverage,
        "coverage_from_text": text_coverage,
        "coverage_from_geometry": geometry_coverage,
        "coverage_agreement": agreement,
    }


def _lr_geometry_fields(x_min, x_max) -> dict:
    if x_min == "" or x_max == "":
        return {
            "coverage_x_min_mm": "", "coverage_x_max_mm": "",
            "fov_lr_mm": "", "fov_lr_center_offset_mm": "",
            "fov_lr_asymmetry_mm": "",
        }
    return {
        "coverage_x_min_mm": x_min,
        "coverage_x_max_mm": x_max,
        "fov_lr_mm": round(x_max - x_min, 6),
        "fov_lr_center_offset_mm": round(abs(x_min + x_max) / 2, 6),
        "fov_lr_asymmetry_mm": round(abs(x_min + x_max), 6),
    }


def series_group_id(dataset) -> str:
    uid = str(_find_value(dataset, "SeriesInstanceUID") or "missing-series-uid")
    temporal = _find_value(dataset, "TemporalPositionIdentifier")
    acquisition = _find_value(dataset, "AcquisitionNumber")
    if temporal not in (None, ""):
        return f"{uid}:temporal-{temporal}"
    if acquisition not in (None, ""):
        return f"{uid}:acquisition-{acquisition}"
    return uid


def summarize_series(
    datasets: Sequence,
    source_paths: Sequence[os.PathLike | str] | None = None,
    nifti_path: str = "",
    conversion_status: str = "not_requested",
    conversion_error: str = "",
) -> dict:
    if not datasets:
        raise ValueError("At least one DICOM dataset is required")
    representative = datasets[0]
    row = {
        output: _csv_value(_find_value(representative, keyword))
        for output, keyword in SERIES_FIELDS.items()
    }
    derived, subtraction, mip, contrast, contrast_phase, contrast_administered = (
        _description_flags(representative)
    )

    sop_uids = {str(_find_value(ds, "SOPInstanceUID") or "") for ds in datasets}
    sop_uids.discard("")
    instance_numbers = []
    acquisition_times = []
    positions = set()
    x_bounds = []
    geometries = set()
    frame_count = 0
    for dataset in datasets:
        try:
            instance_numbers.append(int(_find_value(dataset, "InstanceNumber")))
        except (TypeError, ValueError):
            pass
        acquisition_time = str(_find_value(dataset, "AcquisitionTime") or "")
        if acquisition_time:
            acquisition_times.append(acquisition_time)
        position = _float_list(_find_value(dataset, "ImagePositionPatient"))
        if position:
            positions.add(position)
        bounds = _image_x_bounds(dataset)
        if bounds:
            x_bounds.append(bounds)
        geometries.add((
            _csv_value(_find_value(dataset, "Rows")),
            _csv_value(_find_value(dataset, "Columns")),
            _float_list(_find_value(dataset, "PixelSpacing")),
            _float_list(_find_value(dataset, "ImageOrientationPatient")),
        ))
        try:
            frame_count += int(_find_value(dataset, "NumberOfFrames") or 1)
        except (TypeError, ValueError):
            frame_count += 1

    missing_instances = ""
    if instance_numbers:
        expected = set(range(min(instance_numbers), max(instance_numbers) + 1))
        missing_instances = len(expected.difference(instance_numbers))

    spacing = _float_list(_find_value(representative, "PixelSpacing"))
    try:
        rows = float(_find_value(representative, "Rows"))
        columns = float(_find_value(representative, "Columns"))
        fov_row = round(rows * spacing[0], 3) if len(spacing) == 2 else ""
        fov_column = round(columns * spacing[1], 3) if len(spacing) == 2 else ""
    except (TypeError, ValueError):
        fov_row = fov_column = ""

    source_directory = ""
    if source_paths:
        parents = [str(Path(path).resolve().parent) for path in source_paths]
        source_directory = os.path.commonpath(parents)

    x_min = min((bounds[0] for bounds in x_bounds), default="")
    x_max = max((bounds[1] for bounds in x_bounds), default="")
    coverage_fields = _coverage_summary(representative, x_min, x_max)
    lr_geometry_fields = _lr_geometry_fields(x_min, x_max)
    view, view_tilt = acquisition_orientation(representative)

    row.update({
        "series_group_id": series_group_id(representative),
        "acquisition_system": " ".join(filter(None, (
            str(_find_value(representative, "Manufacturer") or "").strip(),
            str(_find_value(representative, "ManufacturerModelName") or "").strip(),
        ))),
        "acquisition_view": view,
        "acquisition_view_tilt_deg": view_tilt,
        **coverage_fields,
        "sequence_weighting": sequence_weighting(representative),
        "contrast_status": contrast,
        "contrast_phase": contrast_phase,
        "contrast_administered": contrast_administered,
        "is_derived": derived,
        "is_subtraction": subtraction,
        "is_mip": mip,
        "instance_count": len(datasets),
        "sop_instance_count": len(sop_uids),
        "frame_count": frame_count,
        "slice_position_count": len(positions),
        "instance_number_min": min(instance_numbers) if instance_numbers else "",
        "instance_number_max": max(instance_numbers) if instance_numbers else "",
        "missing_instance_numbers": missing_instances,
        "acquisition_time_first": min(acquisition_times) if acquisition_times else "",
        "acquisition_time_last": max(acquisition_times) if acquisition_times else "",
        "field_of_view_row_mm": fov_row,
        "field_of_view_column_mm": fov_column,
        "position_x_min_mm": min((position[0] for position in positions), default=""),
        "position_x_max_mm": max((position[0] for position in positions), default=""),
        "position_y_min_mm": min((position[1] for position in positions), default=""),
        "position_y_max_mm": max((position[1] for position in positions), default=""),
        "position_z_min_mm": min((position[2] for position in positions), default=""),
        "position_z_max_mm": max((position[2] for position in positions), default=""),
        **lr_geometry_fields,
        "geometry_consistent": str(len(geometries) <= 1).lower(),
        "source_directory": source_directory,
        "nifti_path": nifti_path,
        "conversion_status": conversion_status,
        "conversion_error": conversion_error,
    })
    return row


def write_csv(rows: Iterable[dict], output_path: os.PathLike | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(input_path: os.PathLike | str) -> list[dict]:
    """Read an exporter CSV while handling its Excel-compatible UTF-8 BOM."""
    with Path(input_path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _row_is_below_source(row: dict, source_path: os.PathLike | str | None) -> bool:
    if source_path is None:
        return True
    row_source = str(row.get("source_directory", "")).strip()
    if not row_source:
        return False
    try:
        root = os.path.normcase(os.path.abspath(source_path))
        candidate = os.path.normcase(os.path.abspath(row_source))
        return os.path.commonpath((root, candidate)) == root
    except (OSError, ValueError):
        return False


def reconcile_conversion_csv(
    metadata_csv_path: os.PathLike | str,
    conversion_csv_path: os.PathLike | str,
    source_path: os.PathLike | str | None = None,
) -> list[dict]:
    """Merge selected-series conversion results into a complete metadata CSV.

    Rows below ``source_path`` that are absent from the conversion CSV were filtered out before
    conversion. They remain useful acquisition records and are marked explicitly as not selected.
    """
    rows = read_csv(metadata_csv_path)
    conversion_path = Path(conversion_csv_path)
    converted_rows = read_csv(conversion_path) if conversion_path.is_file() else []

    converted_by_key = {
        (str(row.get("source_directory", "")), str(row.get("series_group_id", ""))): row
        for row in converted_rows
    }
    converted_by_group = {
        str(row.get("series_group_id", "")): row
        for row in converted_rows
        if row.get("series_group_id")
    }

    for row in rows:
        if not _row_is_below_source(row, source_path):
            continue
        key = (str(row.get("source_directory", "")), str(row.get("series_group_id", "")))
        converted = converted_by_key.get(key) or converted_by_group.get(key[1])
        if converted is None:
            row.update({
                "nifti_path": "NOT_GENERATED",
                "conversion_status": "not_selected",
                "conversion_error": "",
            })
            continue

        status = str(converted.get("conversion_status", "")) or "unknown"
        nifti_path = str(converted.get("nifti_path", ""))
        if status == "failed" or not nifti_path:
            nifti_path = "NOT_GENERATED"
        row.update({
            "nifti_path": nifti_path,
            "conversion_status": status,
            "conversion_error": converted.get("conversion_error", ""),
        })

    write_csv(rows, metadata_csv_path)
    return rows


class _SeriesAccumulator:
    """Keep aggregate values without retaining thousands of DICOM datasets."""

    def __init__(self, dataset, path):
        self.representative = dataset
        self.instance_count = 0
        self.sop_uids = set()
        self.instance_numbers = set()
        self.positions = set()
        self.x_min = ""
        self.x_max = ""
        self.geometries = set()
        self.frame_count = 0
        self.acquisition_time_first = ""
        self.acquisition_time_last = ""
        self.source_directory = str(Path(path).resolve().parent)
        self.add(dataset, path)

    def add(self, dataset, path):
        self.instance_count += 1
        sop_uid = str(_find_value(dataset, "SOPInstanceUID") or "")
        if sop_uid:
            self.sop_uids.add(sop_uid)
        try:
            self.instance_numbers.add(int(_find_value(dataset, "InstanceNumber")))
        except (TypeError, ValueError):
            pass
        position = _float_list(_find_value(dataset, "ImagePositionPatient"))
        if position:
            self.positions.add(position)
        bounds = _image_x_bounds(dataset)
        if bounds:
            self.x_min = bounds[0] if self.x_min == "" else min(self.x_min, bounds[0])
            self.x_max = bounds[1] if self.x_max == "" else max(self.x_max, bounds[1])
        self.geometries.add((
            _csv_value(_find_value(dataset, "Rows")),
            _csv_value(_find_value(dataset, "Columns")),
            _float_list(_find_value(dataset, "PixelSpacing")),
            _float_list(_find_value(dataset, "ImageOrientationPatient")),
        ))
        try:
            self.frame_count += int(_find_value(dataset, "NumberOfFrames") or 1)
        except (TypeError, ValueError):
            self.frame_count += 1
        acquisition_time = str(_find_value(dataset, "AcquisitionTime") or "")
        if acquisition_time:
            if not self.acquisition_time_first or acquisition_time < self.acquisition_time_first:
                self.acquisition_time_first = acquisition_time
            if not self.acquisition_time_last or acquisition_time > self.acquisition_time_last:
                self.acquisition_time_last = acquisition_time
        self.source_directory = os.path.commonpath((
            self.source_directory, str(Path(path).resolve().parent)
        ))

    def summary(self):
        row = summarize_series([self.representative])
        coverage_fields = _coverage_summary(
            self.representative, self.x_min, self.x_max
        )
        lr_geometry_fields = _lr_geometry_fields(self.x_min, self.x_max)
        missing = ""
        if self.instance_numbers:
            expected = set(range(min(self.instance_numbers), max(self.instance_numbers) + 1))
            missing = len(expected.difference(self.instance_numbers))
        row.update({
            "instance_count": self.instance_count,
            "sop_instance_count": len(self.sop_uids),
            "frame_count": self.frame_count,
            "slice_position_count": len(self.positions),
            "instance_number_min": min(self.instance_numbers) if self.instance_numbers else "",
            "instance_number_max": max(self.instance_numbers) if self.instance_numbers else "",
            "missing_instance_numbers": missing,
            "acquisition_time_first": self.acquisition_time_first,
            "acquisition_time_last": self.acquisition_time_last,
            "position_x_min_mm": min((position[0] for position in self.positions), default=""),
            "position_x_max_mm": max((position[0] for position in self.positions), default=""),
            "position_y_min_mm": min((position[1] for position in self.positions), default=""),
            "position_y_max_mm": max((position[1] for position in self.positions), default=""),
            "position_z_min_mm": min((position[2] for position in self.positions), default=""),
            "position_z_max_mm": max((position[2] for position in self.positions), default=""),
            **coverage_fields,
            **lr_geometry_fields,
            "geometry_consistent": str(len(self.geometries) <= 1).lower(),
            "source_directory": self.source_directory,
        })
        return row


def export_directory(
    source_path: os.PathLike | str,
    output_path: os.PathLike | str,
    progress_every: int = 0,
) -> list[dict]:
    """Scan a tree without loading pixels and export one row per conversion group."""
    grouped = {}
    files_seen = 0
    mr_instances = 0
    for path in Path(source_path).rglob("*"):
        if not path.is_file():
            continue
        files_seen += 1
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        if str(dataset.get("Modality", "")) != "MR":
            continue
        mr_instances += 1
        key = series_group_id(dataset)
        if key in grouped:
            grouped[key].add(dataset, path)
        else:
            grouped[key] = _SeriesAccumulator(dataset, path)
        if progress_every and files_seen % progress_every == 0:
            print(
                f"Scanned {files_seen} files; found {mr_instances} MR instances "
                f"in {len(grouped)} groups"
            )
    rows = [item.summary() for _, item in sorted(grouped.items())]
    write_csv(rows, output_path)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export MRI DICOM series metadata to CSV")
    parser.add_argument("source", help="DICOM file tree")
    parser.add_argument("output", help="Destination CSV")
    parser.add_argument("--quiet", action="store_true", help="Hide scan progress")
    args = parser.parse_args(argv)
    rows = export_directory(args.source, args.output, progress_every=0 if args.quiet else 1000)
    print(f"Wrote {len(rows)} series to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
