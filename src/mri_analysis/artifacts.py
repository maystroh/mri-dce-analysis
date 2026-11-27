"""Versioned viewer manifests and portable exports of saved analysis runs.

Inference manifests and metrics remain untouched. Viewer manifests only locate files;
relative paths resolve against the analysis folder, absolute paths remain external.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ROLES = ("source", "pred_region", "pred_nipple", "gt_region", "gt_nipple", "landmarks")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        try:
            return list(csv.DictReader(handle))
        except csv.Error as exc:
            raise ValueError(f"Cannot read {path.name}: {exc}") from exc


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def file_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def path_ref(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _case_files(root: Path, case: str, source: Path | None,
                predictions_root: Path) -> dict:
    # Case names can contain collection/study paths, but cannot escape predictions_root.
    directory = (predictions_root / case).resolve()
    if not directory.is_relative_to(predictions_root.resolve()):
        raise ValueError(f"Unsafe case identifier: {case}")
    manifest = directory / "landmarks.json"
    files = {"source": path_ref(root, source), "landmarks": None}
    if manifest.is_file():
        files["landmarks"] = path_ref(root, manifest)
        try:
            entries = read_json(manifest).get("entries", [])
        except (OSError, ValueError):
            # The failed case remains discoverable; its details report the corrupt manifest.
            return files
        matches = [e for e in entries if source and e.get("source_file") == source.name]
        if len(matches) == 1:
            for role, key in (("pred_region", "breast_region_file"),
                              ("pred_nipple", "nipples_file")):
                mask = file_path(directory, matches[0].get(key))
                if mask and not mask.is_relative_to(directory):
                    raise ValueError(f"Mask path escapes case directory: {case}")
                files[role] = path_ref(root, mask)
    return files


def write_viewer_manifest(outdir: str, name: str, exams: list, input_path: str,
                          predictions_root: str | None = None,
                          motion_enabled: bool | None = None) -> Path:
    """Record selected source and annotation paths after a stage-2 run."""
    root = Path(outdir).resolve()
    input_root = Path(input_path).resolve()
    if input_root.is_file():
        input_root = input_root.parent
    prediction_root = Path(predictions_root).resolve() if predictions_root else root / "predictions"
    cases = []
    for exam in exams:
        source = Path(exam.pre or exam.phases[0]).resolve() if exam.phases else None
        files = _case_files(root, exam.case, source, prediction_root)
        if exam.dicom_source is None:
            for role, key in (("gt_region", "breast_region"), ("gt_nipple", "nipples")):
                files[role] = path_ref(root, file_path(input_root, exam.metadata.get(key)))
        cases.append({"case": exam.case, "files": files,
                      "phases": [path_ref(root, Path(p)) for p in exam.phases],
                      "metadata": exam.metadata, "dicom_source": exam.dicom_source,
                      "input_error": exam.input_error})
    content = {"schema_version": SCHEMA_VERSION, "name": name,
               "portable": False, "cases": cases}
    if motion_enabled is not None:
        content["motion_enabled"] = motion_enabled
    path = root / f"{name}_viewer.json"
    path.write_text(json.dumps(content, indent=2) + "\n")
    return path


def discover_runs(root: Path) -> dict[str, dict]:
    """Read current manifests, or reconstruct older runs from their saved CSVs."""
    root = root.resolve()
    names = {p.name.removesuffix("_viewer.json") for p in root.glob("*_viewer.json")}
    names.update(p.name.removesuffix("_stage2_summary.json") for p in root.glob("*_stage2_summary.json"))
    names.update(p.name.removesuffix("_stage1.csv") for p in root.glob("*_stage1.csv"))
    names.update(p.name.removesuffix("_summary.json") for p in root.glob("*_summary.json")
                 if "volumes" in read_json(p))
    runs = {}
    for name in sorted(names):
        path = root / f"{name}_viewer.json"
        if path.is_file():
            manifest = read_json(path)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported viewer manifest version in {path.name}")
        else:
            rows = read_csv(root / f"{name}_exams.csv")
            if not rows:
                rows = read_csv(root / f"{name}_stage2.csv")
            if not rows:
                rows = read_csv(root / f"{name}_stage1.csv")
            cases = {}
            for row in rows:
                source = file_path(root, row.get("source_path") or row.get("path"))
                case = row.get("case") or (source.name if source else row.get("filename", "unknown"))
                if case in cases:
                    continue
                cases[case] = {"case": case, "files": _case_files(
                    root, case, source, root / "predictions")}
            manifest = {"schema_version": SCHEMA_VERSION, "name": name,
                        "portable": False, "cases": list(cases.values())}
        # A portable manifest is authoritative; historical evaluation paths may be stale.
        if not manifest.get("portable"):
            identities = read_json(root / f"{name}_annotation_manifest.json").get("cases", [])
            annotations = {row["case"]: row.get("files", {}) for row in identities}
            for case in manifest["cases"]:
                known = annotations.get(case["case"], {})
                for role, old in (("gt_region", "annotated_region"), ("gt_nipple", "annotated_nipple")):
                    if known.get(old, {}).get("path"):
                        case["files"][role] = known[old]["path"]
        runs[name] = manifest
    if not runs:
        raise ValueError("No analysis runs found: expected *_viewer.json, *_stage2_summary.json "
                         "or *_stage1.csv in the output folder")
    return runs


def export_bundle(run_dir: str, outdir: str) -> Path:
    """Export all runs with reference images, masks and reports using relative paths.

    The destination must be new. The source run and its production manifests are never
    rewritten. Model caches and raw/conversion trees are unnecessary for viewing.
    """
    root, destination = Path(run_dir).resolve(), Path(outdir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Bundle destination must be outside the source run folder")
    if destination.exists():
        raise FileExistsError("Bundle destination already exists; choose a new folder")
    runs = discover_runs(root)
    # Validate inputs before creating a partial export. Optional absent predictions on failed
    # cases stay absent; every selected reference image must exist for a portable review.
    for manifest in runs.values():
        for case in manifest["cases"]:
            source = file_path(root, case["files"].get("source"))
            if source is not None and not source.is_file():
                raise FileNotFoundError(f"Source image unavailable for {case['case']}: {source}")
            for role in ROLES[1:]:
                path = file_path(root, case["files"].get(role))
                if path is not None and not path.is_file():
                    raise FileNotFoundError(f"{role} unavailable for {case['case']}: {path}")
    destination.mkdir(parents=True)
    for path in root.iterdir():
        if path.is_file() and not path.name.endswith("_viewer.json"):
            shutil.copy2(path, destination / path.name)
    for name, manifest in runs.items():
        for case in manifest["cases"]:
            identity = hashlib.sha256(f"{name}/{case['case']}".encode()).hexdigest()[:20]
            directory = destination / "viewer_data" / identity
            directory.mkdir(parents=True)
            for role in ROLES:
                source = file_path(root, case["files"].get(role))
                if source is None:
                    continue
                # Keep each artifact's original name in a role directory; production JSON
                # remains evidence, while this manifest supplies the relocated file mapping.
                target = directory / role / source.name
                target.parent.mkdir()
                shutil.copy2(source, target)
                case["files"][role] = path_ref(destination, target)
            case["phases"] = [case["files"]["source"]] if case["files"].get("source") else []
        manifest["portable"] = True
        (destination / f"{name}_viewer.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "OPEN_VIEWER.txt").write_text(
        'Install the mri-analysis wheel, then run:\n\nmri-analysis viewer "' + str(destination.name) +
        '" --open\n\nRun from the parent folder, or replace the quoted name with the full path.\n'
        'Reference images and labels are included. Model caches are unnecessary.\n'
        'Historical CSV and model-manifest paths are provenance; *_viewer.json locates exported files.\n')
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a portable analysis folder including reference images")
    parser.add_argument("run_dir")
    parser.add_argument("--out", required=True, help="new destination folder")
    args = parser.parse_args()
    try:
        destination = export_bundle(args.run_dir, args.out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Portable run ready: {destination}\nmri-analysis viewer "{destination}" --open')
    return 0
