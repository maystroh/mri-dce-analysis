"""Scanner context per exam: field strength, manufacturer, model, site, fat suppression.

**None of this is in the NIfTI header.** Field strength and manufacturer are DICOM
attributes, and every NIfTI converter in use drops them — `descrip`, `aux_file` and
`intent_name` are empty on every volume in both challenge datasets. So a dataset either
ships a sidecar or the information is gone, and this module is the adapter layer over
"whatever that dataset ships".

That asymmetry is itself a finding, and the loaders below make it explicit rather than
filling gaps with defaults: `field_strength_t is None` means NOT RECORDED, never "assume
1.5T". A stratification built on invented values would hide exactly the population the
landmark model has not been tested on.

Sources, by dataset:

* `mama_mia` — `clinical_and_imaging_info.xlsx`, which carries all five attributes.
* `odelia`   — `data_bilateral/{INSTITUTION}/metadata_unilateral/annotation.csv`, which
  carries the INSTITUTION ONLY. Field strength, manufacturer and model are not in the
  release. Institution is a usable proxy: a site runs one or two scanners, so it bounds
  the scanner variation even though it does not name it.
* `series_csv` — the per-series CSV that `mri-dicom-pipeliner`'s `metadata_exporter`
  writes at conversion time. This is the **Layer 1** source (see docs/DECISION_LADDERS.md):
  it carries the DICOM's own verdict on plane and laterality, not a reconstruction of it.
* `dicom`    — read the tags directly, when no CSV was written.

`discover` finds one of these beside a dataset without being told, so the normal call is
"use the metadata if the dataset has any, otherwise fall back to the NIfTI header".
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

# The DICOM tags that carry scanner context. Keyword form, resolved by pydicom.
DICOM_TAGS = {
    "field_strength_t": "MagneticFieldStrength",     # (0018,0087)
    "manufacturer": "Manufacturer",                  # (0008,0070)
    "scanner_model": "ManufacturerModelName",        # (0008,1090)
    "site": "InstitutionName",                       # (0008,0080)
}

# Manufacturer names arrive in several spellings across sites and export tools. The
# stratification counts scanners, so "GE MEDICAL SYSTEMS" and "GE" must not split into two
# strata. Anything unrecognised is passed through unchanged and title-cased, never dropped.
_MANUFACTURER_ALIASES = {
    "ge": "GE", "ge medical systems": "GE", "ge healthcare": "GE",
    "siemens": "SIEMENS", "siemens healthineers": "SIEMENS",
    "siemens healthcare gmbh": "SIEMENS", "siemens medical solutions": "SIEMENS",
    "philips": "Philips", "philips medical systems": "Philips",
    "philips healthcare": "Philips",
    "toshiba": "Toshiba", "canon": "Canon", "canon medical systems": "Canon",
    "hitachi": "Hitachi",
}

UNKNOWN = "unknown"


def normalise_manufacturer(raw) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return _MANUFACTURER_ALIASES.get(text.lower(), text)


def normalise_field_strength(raw) -> Optional[float]:
    """Tesla, as a float. Handles the milli-Tesla some exports use.

    A scanner is 1.5 T or 3 T; a value in the thousands is milli-Tesla, and one under 0.05
    is nonsense rather than a real low-field scanner in this population.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value > 100:            # milli-Tesla
        value /= 1000.0
    if value <= 0.05:
        return None
    return round(value, 2)


def field_strength_bin(tesla: Optional[float]) -> str:
    """The stratum a field strength belongs to. `unknown` when it was not recorded."""
    if tesla is None:
        return UNKNOWN
    if tesla < 1.0:
        return "<1T"
    if tesla < 2.0:
        return "1.5T"
    if tesla < 4.0:
        return "3T"
    return ">=4T"


# What the DICOM itself said about geometry, when a Layer 1 source is present. These are
# kept separate from the NIfTI-derived verdicts so the two can be COMPARED rather than one
# silently overwriting the other — a disagreement is the highest-yield QC signal we have.
@dataclass
class AcquisitionContext:
    """Scanner context for one exam. Every field is optional — absent means NOT RECORDED."""

    case: str
    field_strength_t: Optional[float] = None
    manufacturer: Optional[str] = None
    scanner_model: Optional[str] = None
    site: Optional[str] = None
    fat_suppressed: Optional[int] = None
    # The DICOM slice THICKNESS, which is not the affine slice SPACING. Where spacing is
    # about half the thickness, the volume was reconstructed with overlapping slices: the
    # grid looks finer than the data really is. See `slice_overlap` in run_view_laterality.
    slice_thickness_mm: Optional[float] = None
    # Layer 1 verdicts. `None` means the source did not state one.
    plane_dicom: Optional[str] = None
    scope_dicom: Optional[str] = None
    scope_dicom_source: Optional[str] = None    # which tag/text/geometry rule decided it
    metadata_source: str = "none"

    def as_row(self) -> Dict[str, object]:
        row = asdict(self)
        row.pop("case")
        row["field_strength_bin"] = field_strength_bin(self.field_strength_t)
        for key in ("manufacturer", "scanner_model", "site"):
            row[key] = row[key] if row[key] else UNKNOWN
        for key in ("field_strength_t", "fat_suppressed", "slice_thickness_mm"):
            row[key] = "" if getattr(self, key) is None else getattr(self, key)
        return row


ROW_FIELDS = ["field_strength_t", "field_strength_bin", "manufacturer", "scanner_model",
              "site", "fat_suppressed", "slice_thickness_mm", "metadata_source"]

EMPTY_ROW = {**{k: UNKNOWN for k in ("field_strength_bin", "manufacturer",
                                     "scanner_model", "site")},
             "field_strength_t": "", "fat_suppressed": "", "slice_thickness_mm": "",
             "metadata_source": "none"}

# Filenames a dataset may carry its Layer 1 metadata under. `metadata_exporter` writes the
# first one; the rest are the spellings seen in practice.
METADATA_FILENAMES = ("dicom_series_info.csv", "series_info.csv", "dicom_metadata.csv")


def discover(dataset_root: str) -> Optional[str]:
    """Find a Layer 1 metadata file for a dataset, or None.

    Looks beside the dataset and one level up — a sweep is usually pointed at
    `.../images` while the CSV sits at `.../`. Returns a `load`-ready spec.
    """
    here = os.path.abspath(dataset_root)
    for directory in (here, os.path.dirname(here)):
        for name in METADATA_FILENAMES:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return f"series_csv:{candidate}"
    return None


def slice_overlap(spacing_mm, thickness_mm) -> str:
    """How the affine's slice SPACING relates to the DICOM slice THICKNESS.

    `contiguous` when they agree, `overlapping` when the volume was reconstructed on a
    finer grid than its slices are thick (spacing < 0.9 x thickness), `gapped` when slices
    are further apart than they are thick, `unknown` when no thickness was recorded.

    This matters because a landmark model resamples to a fixed isotropic grid. An exam with
    2.2 mm slices sampled every 1.1 mm has the same grid as one with true 1.1 mm slices and
    genuinely blurrier through-plane content. The header cannot tell them apart; only the
    thickness label can.
    """
    try:
        spacing, thickness = float(spacing_mm), float(thickness_mm)
    except (TypeError, ValueError):
        return UNKNOWN
    if thickness <= 0 or spacing <= 0:
        return UNKNOWN
    ratio = spacing / thickness
    if ratio < 0.9:
        return "overlapping"
    if ratio > 1.1:
        return "gapped"
    return "contiguous"


def _as_float(raw) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return None if value != value or value <= 0 else round(value, 4)


def load_mama_mia(xlsx_path: str, sheet: str = "dataset_info"
                  ) -> Dict[str, AcquisitionContext]:
    """MAMA-MIA's `clinical_and_imaging_info.xlsx`, keyed by `patient_id`."""
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype={"patient_id": str})
    out: Dict[str, AcquisitionContext] = {}
    for row in df.itertuples(index=False):
        fat = getattr(row, "fat_suppressed", None)
        out[str(row.patient_id)] = AcquisitionContext(
            case=str(row.patient_id),
            field_strength_t=normalise_field_strength(getattr(row, "field_strength", None)),
            manufacturer=normalise_manufacturer(getattr(row, "manufacturer", None)),
            scanner_model=(str(getattr(row, "scanner_model", "")).strip() or None),
            site=(str(getattr(row, "site", "")).strip() or None),
            fat_suppressed=(None if fat is None or fat != fat else int(fat)),
            slice_thickness_mm=_as_float(getattr(row, "slice_thickness", None)),
            metadata_source="mama_mia_xlsx",
        )
    return out


def load_odelia(challenge_root: str) -> Dict[str, AcquisitionContext]:
    """ODELIA's per-institution `annotation.csv` files.

    Only `site` comes out — the release does not carry field strength, manufacturer or
    model, so those stay `None` and every ODELIA exam lands in the `unknown` field-strength
    stratum. That is the honest result, not a gap to paper over.

    UIDs are read as STRINGS. Read as numbers, the 128 UMCU identifiers lose their leading
    zeros and 13 exams silently fail to join.
    """
    import pandas as pd

    pattern = os.path.join(challenge_root, "data_bilateral", "*",
                           "metadata_unilateral", "annotation.csv")
    out: Dict[str, AcquisitionContext] = {}
    for path in sorted(glob.glob(pattern)):
        institution = path.split(os.sep)[-3]
        df = pd.read_csv(path, dtype=str)
        for uid in df["UID"].astype(str):
            out[uid] = AcquisitionContext(case=uid, site=institution,
                                          metadata_source="odelia_annotation_csv")
    return out


def load_dicom(dicom_root: str, cases=None) -> Dict[str, AcquisitionContext]:
    """Read the tags from one DICOM file per exam directory under `dicom_root`.

    One file is enough: these tags are constant within a series, and reading every slice of
    every exam to re-confirm that would cost hours. `stop_before_pixels` keeps it to the
    header.
    """
    import pydicom

    out: Dict[str, AcquisitionContext] = {}
    for case in sorted(cases or os.listdir(dicom_root)):
        case_dir = os.path.join(dicom_root, case)
        if not os.path.isdir(case_dir):
            continue
        ctx = AcquisitionContext(case=case, metadata_source="dicom")
        for root, _, files in os.walk(case_dir):
            found = False
            for name in sorted(files):
                try:
                    ds = pydicom.dcmread(os.path.join(root, name),
                                         stop_before_pixels=True, force=True)
                except Exception:
                    continue
                if "Modality" not in ds:
                    continue
                ctx.field_strength_t = normalise_field_strength(
                    getattr(ds, DICOM_TAGS["field_strength_t"], None))
                ctx.manufacturer = normalise_manufacturer(
                    getattr(ds, DICOM_TAGS["manufacturer"], None))
                ctx.scanner_model = (str(getattr(ds, DICOM_TAGS["scanner_model"], "")).strip()
                                     or None)
                ctx.site = (str(getattr(ds, DICOM_TAGS["site"], "")).strip() or None)
                ctx.slice_thickness_mm = _as_float(getattr(ds, "SliceThickness", None))
                found = True
                break
            if found:
                break
        out[case] = ctx
    return out


# `metadata_exporter` emits these; a series that is one of them is a DERIVED product and
# must not decide the exam's geometry — a subtraction or a MIP can carry any orientation.
def _is_primary(row: Dict[str, str]) -> bool:
    return not any(str(row.get(key, "")).strip().lower() in ("true", "1", "yes")
                   for key in ("is_derived", "is_subtraction", "is_mip"))


def _modal(values) -> Optional[str]:
    """The most common non-empty, non-unknown value, or None."""
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in (UNKNOWN, "none", "nan"):
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], k))


def load_series_csv(csv_path: str, key: str = "patient_id"
                    ) -> Dict[str, AcquisitionContext]:
    """The per-series CSV from `mri-dicom-pipeliner`, reduced to one row per exam.

    The CSV is per SERIES and this analysis is per EXAM, so the rows are grouped by `key`
    and reduced. Two rules govern the reduction:

    * **Derived series do not vote.** A subtraction or a MIP inherits nothing reliable about
      acquisition geometry, so `is_derived` / `is_subtraction` / `is_mip` rows are dropped
      before the vote and only used if nothing else remains.
    * **The verdict is the mode, not the first row.** A study routinely mixes a sagittal
      localiser with axial DCE phases; taking row 0 would let the localiser decide.

    Vocabulary is lowercased on the way in. `metadata_exporter.acquisition_view` returns
    `"Axial"` while this package uses `"axial"`, and an unnormalised join silently produces
    zero matches (see docs/DICOM_METADATA_REVIEW.md, defect 6).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if key not in df.columns:
        raise SystemExit(f"{csv_path} has no {key!r} column; columns are {list(df.columns)}")

    out: Dict[str, AcquisitionContext] = {}
    for case, group in df.groupby(key):
        case = str(case).strip()
        if not case:
            continue
        primary = [r for _, r in group.iterrows() if _is_primary(r)] or list(
            r for _, r in group.iterrows())

        plane = _modal(r.get("acquisition_view") for r in primary)
        # `oblique` and `unknown` are NOT planes — they are the export telling us it could
        # not decide, which is exactly when Layer 2 should answer instead.
        if plane in ("oblique", "unknown"):
            plane = None
        scope = _modal(r.get("breast_coverage") for r in primary)
        if scope == "unknown":
            scope = None

        out[case] = AcquisitionContext(
            case=case,
            field_strength_t=normalise_field_strength(
                _modal(r.get("magnetic_field_strength_t") for r in primary)),
            manufacturer=normalise_manufacturer(
                _modal(r.get("manufacturer") for r in primary)),
            scanner_model=_modal(r.get("manufacturer_model_name") for r in primary),
            site=_modal(r.get("institution_name") for r in primary),
            slice_thickness_mm=_as_float(
                _modal(r.get("slice_thickness_mm") for r in primary)),
            plane_dicom=plane,
            scope_dicom=scope,
            scope_dicom_source=_modal(r.get("breast_coverage_source") for r in primary),
            metadata_source="dicom_series_csv",
        )
    return out


def load(spec: str) -> Dict[str, AcquisitionContext]:
    """Load a metadata source from a `kind:path` spec, e.g. `mama_mia:/data/info.xlsx`."""
    if ":" not in spec:
        raise SystemExit(f"metadata spec must be 'kind:path', got {spec!r}")
    kind, path = spec.split(":", 1)
    loaders = {"mama_mia": load_mama_mia, "odelia": load_odelia, "dicom": load_dicom,
               "series_csv": load_series_csv}
    if kind not in loaders:
        raise SystemExit(f"unknown metadata kind {kind!r}; pick one of {sorted(loaders)}")
    return loaders[kind](path)
