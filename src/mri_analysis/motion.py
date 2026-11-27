"""Optional inter-phase motion measurements using the production transform library.

Registration runs on original unsplit 3D phases. Saved results carry numbers and
registration evidence, without assigning clinical pass/fail thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from mri_analysis.artifacts import file_path
from mri_analysis.discovery import Exam
from mri_analysis.landmarks import ROI_PLAUSIBILITY_MIN_MM3, StoredLandmarkRunner


def reference_identity(path: str) -> dict:
    image = nib.load(path)
    return {"source_file": Path(path).name, "source_shape": list(image.shape),
            "source_affine_sha1": hashlib.sha1(
                np.ascontiguousarray(image.affine, dtype=np.float64).tobytes()).hexdigest()[:16]}


def _roi(exam: Exam, predictions: Path) -> tuple[str | None, dict, bool, list[str]]:
    """Trust side labels only for split execution and aligned, nonempty native masks."""
    directory = (predictions / exam.case).resolve()
    if not directory.is_relative_to(predictions.resolve()):
        raise ValueError("Case ID escapes the prediction directory")
    warnings = []
    try:
        result = StoredLandmarkRunner().run(exam.phases[0], str(directory))
        entry = result["entry"]
        path = file_path(directory, entry.get("breast_region_file"))
        if path is None or not path.is_relative_to(directory):
            raise ValueError("Region mask path escapes the prediction directory")
        reference, mask = nib.load(exam.phases[0]), nib.load(path)
        if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, atol=1e-3):
            raise ValueError("Region mask grid differs from the reference phase")
        data = np.asarray(mask.dataobj)
        if not np.isfinite(data).all() or not np.any(data > 0):
            raise ValueError("Region mask is empty or contains non-finite labels")
        coverage = result.get("coverage") or {}
        execution = coverage.get("execution")
        if execution is None and coverage:
            execution = "two_halves" if coverage.get("scope") == "bilateral" else "full_width"
        labels = {}
        for side, info in entry.get("sides", {}).items():
            label = (info.get("breast_region") or {}).get("label_value")
            if side in ("left", "right") and isinstance(label, (int, float)) and label > 0 and int(label) == label:
                labels[side] = int(label)
        from mri_analysis.laterality import pipeline_laterality
        scope = pipeline_laterality(exam.phases[0])["pipeline_scope"]
        bilateral = (scope == "bilateral" and execution != "full_width"
                     and set(labels) == {"left", "right"} and len(set(labels.values())) == 2)
        if scope == "bilateral" and not bilateral:
            warnings.append("Using one merged region: trustworthy left/right labels are unavailable")
        return str(path), labels if bilateral else {}, bilateral, warnings
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"Breast ROI unavailable ({exc}); measuring the whole volume")
        return None, {}, False, warnings


def score_case(exam: Exam, predictions: Path) -> dict:
    phases = [str(Path(p).resolve()) for p in exam.phases]
    detail = {"case": exam.case, "n_phases": len(phases), "phase_files": phases,
              "reference_phase": Path(phases[0]).name if phases else None,
              "reference_substituted": exam.pre is None, "score_mm": None,
              "series": {}, "warnings": [], "status": "skipped"}
    if len(phases) < 2:
        if phases:
            try:
                if len(nib.load(phases[0]).shape) != 3:
                    raise ValueError("Motion requires separate 3D phase files; split 4D acquisitions first")
            except Exception as exc:
                detail.update(status="error", needs_review=True, reason=f"{type(exc).__name__}: {exc}")
                return detail
        detail["reason"] = "Motion requires at least two phases; only one phase is available" if phases else "No phases available"
        return detail
    started = time.perf_counter()
    try:
        if len(set(phases)) != len(phases):
            raise ValueError("Duplicate phases: provide each DCE phase exactly once")
        for phase in phases:
            image = nib.load(phase)
            if len(image.shape) != 3:
                raise ValueError("Motion requires separate 3D phase files; split 4D acquisitions first")
            if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
                raise ValueError(f"Invalid phase geometry: {phase}")
        detail["reference_identity"] = reference_identity(phases[0])
        from mri_transforms import motion_qc as mq
        detail["engine"] = {"module": "mri_transforms.motion_qc", "sha256": hashlib.sha256(
            Path(mq.__file__).read_bytes()).hexdigest()}
        roi, labels, bilateral, warnings = _roi(exam, predictions)
        detail.update(roi_file=roi, roi_source="landmark_breast_region" if roi else "whole_volume",
                      series_scope="bilateral" if bilateral else "merged", warnings=warnings)
        scores = mq.score_exam(phases, roi_path=roi, side_labels=labels, bilateral=bilateral)
        if not scores:
            raise ValueError("No motion series returned")
        incomplete = 0
        for name, score in scores.items():
            record = score.to_dict()
            record["reference_substituted"] = exam.pre is None
            expected = set(range(1, len(phases)))
            measured = {p["phase"] for p in record.get("pairs", [])}
            record["missing_phases"] = sorted(expected - measured)
            if score.status != "ok" or score.score_mm is None or measured != expected:
                incomplete += 1
            if score.roi_cm3 is not None and score.roi_cm3 < ROI_PLAUSIBILITY_MIN_MM3 / 1000:
                detail["warnings"].append(f"{name}: ROI is only {score.roi_cm3:.1f} cm³; inspect registration evidence")
            detail["series"][name] = record
        # Keep partial raw numbers, while making the missing measurements explicit.
        detail["score_mm"], _ = mq.exam_motion_score(scores)
        if any(s.score_mm is not None and not math.isfinite(s.score_mm) for s in scores.values()):
            raise ValueError("Registration returned a non-finite displacement; measurement is unavailable")
        detail["n_unscored_series"] = incomplete
        detail["status"] = "ok" if incomplete == 0 else "partial" if detail["score_mm"] is not None else "error"
        detail["needs_review"] = bool(incomplete or detail["warnings"])
        if incomplete:
            detail["reason"] = "Some phases or breast series could not be measured; score covers only available measurements"
    except Exception as exc:
        detail.update(status="error", needs_review=True, reason=f"{type(exc).__name__}: {exc}")
    detail["duration_seconds"] = round(time.perf_counter() - started, 3)
    return detail


def _json_safe(value):
    """Keep failed numerical evidence readable without invalid JSON NaN/Infinity tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_motion(exams: list[Exam], outdir: str, name: str,
               predictions_root: str | None = None) -> dict:
    root = Path(outdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    predictions = Path(predictions_root).resolve() if predictions_root else root / "predictions"
    cases, rows, pairs = [], [], []
    for index, exam in enumerate(exams, 1):
        print(f"Motion [{index}/{len(exams)}]: {exam.case}", flush=True)
        detail = _json_safe(score_case(exam, predictions))
        cases.append(detail)
        for side, record in detail["series"].items():
            rows.append({**detail, **record, "series": side, "exam_status": detail["status"],
                         "warnings": "; ".join(detail["warnings"])})
            pairs.extend({"case": exam.case, "series": side, **pair}
                         for pair in record.get("pairs", []))
        if not detail["series"]:
            rows.append({**detail, "series": "-", "exam_status": detail["status"]})
        # Save progress after each exam; a stopped run still has reviewable evidence.
        (root / f"{name}_motion.json").write_text(json.dumps(
            {"schema_version": 1, "name": name, "cases": cases}, indent=2, allow_nan=False) + "\n")
    counts = Counter(case["status"] for case in cases)
    summary = {"exams": len(cases), "exams_ok": counts["ok"], "exams_skipped": counts["skipped"],
               "exams_failed": counts["error"] + counts["partial"], "status_counts": dict(counts),
               "measurement": "Worst phase mean rigid displacement in millimetres; no pass/fail thresholds",
               "engine": "mri_transforms.motion_qc", "phase_index": "0 = reference; 1 = first subsequent phase"}
    (root / f"{name}_motion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(root / f"{name}_motion.csv", rows,
               ["case", "series", "exam_status", "status", "score_mm", "n_phases", "worst_phase",
                "reference_phase", "reference_substituted", "roi_source", "roi_cm3", "reason", "note", "warnings"])
    _write_csv(root / f"{name}_motion_pairs.csv", pairs,
               ["case", "series", "phase", "moving_file", "mean_mm", "p95_mm", "max_mm", "translation_mm",
                "rotation_deg", "converged", "duration_seconds", "stop_condition"])
    if not cases:
        (root / f"{name}_motion.json").write_text(json.dumps({"schema_version": 1, "name": name, "cases": []}) + "\n")
    return summary


def main() -> int:
    from mri_analysis.commands.stage2 import resolve_exams
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--layout", choices=["auto", "mama_mia", "odelia", "generic"], default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--predictions-root", help="existing landmark predictions; otherwise <out>/predictions")
    args = parser.parse_args()
    if args.limit < 0 or not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use a non-negative limit and a filename prefix for --name")
    exams = resolve_exams(args.input, args.layout, args.limit,
                          exclude=[p for p in (args.out, args.predictions_root) if p])
    if not exams:
        parser.error("No exams found; check input layout or provide a CSV phase manifest")
    summary = run_motion(exams, args.out, args.name, args.predictions_root)
    from mri_analysis.artifacts import write_viewer_manifest
    write_viewer_manifest(args.out, args.name, exams, args.input, args.predictions_root, True)
    return int(bool(summary["exams_failed"]))
