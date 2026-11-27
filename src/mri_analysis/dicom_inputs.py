"""Inference-compatible JSON exam inputs and their DICOM conversion provenance.

Input keys are exam directories; their values are arbitrary JSON metadata objects.
A DICOM exam directory is converted; a NIfTI exam directory (NIfTI files and no `.dcm`
files) is used in place, with its phases found as for `--input`.
Converted manifests keep routing separate from metadata and record every requested exam.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, TextIO

from mri_analysis.artifacts import file_path, path_ref, read_csv
from mri_analysis.discovery import Exam, find_exam

CONVERTED_FORMAT = "mri-analysis-converted-dicom"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        value[key] = item
    return value


def read_input_json(path: str | Path) -> dict[str, Any]:
    def invalid_constant(value: str):
        raise ValueError(f"Invalid JSON numeric value: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict) or not value:
        raise ValueError("JSON input must be a nonempty object mapping exam directories to metadata objects")
    return value


def dicom_inputs(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    records, seen = [], set()
    for supplied, metadata in read_input_json(path).items():
        if not supplied.strip() or not isinstance(metadata, dict):
            raise ValueError("Each JSON entry must have an exam directory key and a metadata object value")
        source = Path(supplied).expanduser()
        source = (source if source.is_absolute() else path.parent / source).resolve()
        if source in seen:
            raise ValueError(f"Exam directory appears more than once: {source}")
        seen.add(source)
        prefix = re.sub(r"[^A-Za-z0-9._-]", "_", source.name).strip(".")[:100] or "exam"
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        records.append({"case": f"{prefix}__{digest}", "input_path": supplied,
                        "dicom_source": str(source), "metadata": metadata})
    return records


def converted_exams(path: str | Path, limit: int = 0) -> list[Exam]:
    path = Path(path).resolve()
    value = read_input_json(path)
    if value.get("format") != CONVERTED_FORMAT or value.get("schema_version") != 1:
        raise ValueError("This JSON lists DICOM directories. Use the full run command or Stage 1 to convert it first")
    records = value.get("cases")
    if not isinstance(records, list):
        raise ValueError("Converted DICOM manifest requires a cases list")
    exams, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Converted cases must be objects")
        case = record.get("case")
        if (not isinstance(case, str) or not case or Path(case).is_absolute()
                or ".." in Path(case).parts or "\\" in case or case == "." or case in seen):
            raise ValueError(f"Unsafe or duplicate converted case ID: {case!r}")
        seen.add(case)
        metadata, posts = record.get("metadata", {}), record.get("posts", [])
        if not isinstance(metadata, dict) or not isinstance(posts, list) or not all(isinstance(p, str) for p in posts):
            raise ValueError("Converted metadata must be an object and posts an ordered path list")
        for key in ("pre", "nifti_directory", "dicom_source", "error"):
            if record.get(key) is not None and not isinstance(record[key], str):
                raise ValueError(f"Converted {key} must be a path or string, or null")
        pre = file_path(path.parent, record.get("pre"))
        directory = file_path(path.parent, record.get("nifti_directory"))
        exams.append(Exam(case, str(directory or path.parent), str(pre) if pre else None,
                          [str(file_path(path.parent, p)) for p in posts], metadata,
                          record.get("dicom_source"), record.get("error")))
        if limit and len(exams) >= limit:
            break
    return exams


def has_dicom_files(directory: Path) -> bool:
    for _, _, files in os.walk(directory):
        if any(name.lower().endswith(".dcm") for name in files):
            return True
    return False


def nifti_exam(directory: Path, case: str) -> Exam | None:
    """The exam of a NIfTI exam directory, or None for a DICOM (or empty) one."""
    if not any(n.endswith((".nii", ".nii.gz")) for n in os.listdir(directory)) or has_dicom_files(directory):
        return None
    exam = find_exam(str(directory), case, "auto")
    if not exam or not exam.pre:
        raise ValueError("NIfTI exam directory: no pre-contrast reference could be identified "
                         "(name the files Pre.nii.gz, Post_1.nii.gz, ...)")
    return exam


def convert_json_input(input_path: str, nifti_dir: str, outdir: str, name: str, fh: TextIO,
                       limit: int = 0, organized_root: str | None = None) -> dict[str, Any]:
    """Convert or adopt each JSON exam. With `organized_root` (`--organize`), exam folders
    inside it are link folders: records keep the original files and each folder is removed
    as soon as its exam is done."""
    from mri_analysis.commands.stage1 import convert, log
    from mri_analysis.organize import origin, release
    records = dicom_inputs(input_path)
    selected = records[:limit] if limit else records
    root, destination = Path(outdir).resolve(), Path(nifti_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{name}_dicom_exams.json"
    audit = {"format": CONVERTED_FORMAT, "schema_version": 1,
             "input_manifest": str(Path(input_path).resolve()), "cases": []}
    for index, original in enumerate(selected, 1):
        record = {**original, "pre": None, "posts": [], "status": "error", "error": None}
        case_directory = destination / record["case"]
        record["nifti_directory"] = path_ref(root, case_directory)
        log(fh, f"JSON exam [{index}/{len(selected)}]: {record['dicom_source']}")
        try:
            source = Path(record["dicom_source"])
            if not source.is_dir():
                raise FileNotFoundError(f"Exam directory is missing or inaccessible: {source}")
            exam = nifti_exam(source, record["case"])
            if exam:
                log(fh, "  NIfTI exam: used in place, no conversion")
                record["input_type"] = "nifti"
                # An organized folder holds links (or copies): record the original files.
                original = origin(source)
                real = lambda p: Path(original.get(Path(p).name, p))
                record["nifti_directory"] = path_ref(root, real(exam.pre).parent if original else source)
                record["pre"] = path_ref(root, real(exam.pre))
                record["posts"] = [path_ref(root, real(p)) for p in exam.posts]
                # Same shape as converted series, so geometry checks read these images too
                # (T2 included, as for a converted exam; masks and labels are not images).
                images = [n for n in sorted(os.listdir(source)) if n.endswith((".nii", ".nii.gz"))
                          and not re.search(r"(^|[_-])(seg|mask|label)", n, re.IGNORECASE)]
                record["series_conversion"] = [{"nifti_path": str(real(source / n).resolve()) if original else n,
                                                "conversion_status": "already_exists"} for n in images]
                record["status"] = "ok"
            else:
                record["input_type"] = "dicom"
                result = convert(str(source), str(case_directory), fh, studies=[str(source)],
                                 nifti_name_format="{SeriesInstanceUID}-{ContentDate}-{ContentTime:int}-{const:phase}-{const:series_label}")
                if result["failed"]:
                    raise ValueError("DICOM pipeliner failed; inspect the Stage 1 log")
                rows = read_csv(case_directory / "_series_metadata/series_00000.csv")
                if not rows:
                    raise ValueError("Converter produced no series metadata; no usable MR series were found")
                record["series_conversion"] = rows
                allowed = {file_path(case_directory, row.get("nifti_path")) for row in rows
                           if row.get("conversion_status") in ("converted", "already_exists")}
                exam = find_exam(str(case_directory), record["case"], "auto")
                if not exam or not exam.pre or Path(exam.pre).resolve() not in allowed:
                    raise ValueError("No current pre-contrast reference could be identified after conversion")
                record["pre"] = path_ref(root, Path(exam.pre))
                record["posts"] = [path_ref(root, Path(p)) for p in exam.posts if Path(p).resolve() in allowed]
                failures = [row.get("conversion_error") or "Series conversion failed" for row in rows
                            if row.get("conversion_status") == "failed"]
                record["status"] = "partial" if failures else "ok"
                record["error"] = "; ".join(failures) if failures else None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            log(fh, f"  JSON EXAM FAILED: {record['error']}")
        if organized_root and record["metadata"].get("organized_from"):
            release(record["dicom_source"], organized_root)
            record["organized_input"] = record["dicom_source"]
            record["dicom_source"] = record["metadata"]["organized_from"]
        audit["cases"].append(record)
        # Atomic updates keep completed exam records readable if conversion is interrupted.
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(manifest)
    return {"studies": len(selected), "converted": sum(r["pre"] is not None for r in audit["cases"]),
            "nifti_in_place": sum(r.get("input_type") == "nifti" for r in audit["cases"]),
            "failed": sum(r["status"] != "ok" for r in audit["cases"]),
            "input_manifest": audit["input_manifest"], "exam_manifest": str(manifest)}
