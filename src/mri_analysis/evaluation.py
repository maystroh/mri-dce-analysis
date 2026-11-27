"""Score stored landmark predictions with the production unilateral-cohort evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np
import pandas as pd

from mri_transforms.landmark.evaluation import STATUS_OK, STATUS_SCORING_FAILED, is_scored

from .discovery import exams_from_manifest
from .landmarks import StoredLandmarkRunner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(frame: pd.DataFrame, column: str) -> Dict[str, Any]:
    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "mean": None, "p90": None, "min": None, "max": None}
    return {"n": len(values), "median": round(float(values.median()), 4),
            "mean": round(float(values.mean()), 4), "p90": round(float(values.quantile(.9)), 4),
            "min": round(float(values.min()), 4), "max": round(float(values.max()), 4)}


def _group_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    scored = frame[frame.status.map(is_scored)]
    return {
        "cases": len(frame), "scored": len(scored), "unscored": len(frame) - len(scored),
        "breast_region_dice": _stats(scored, "breast_region_dice"),
        "nipple_dice": _stats(scored, "nipple_dice"),
        "nipple_centroid_distance_mm": _stats(scored, "nipple_centroid_distance_mm"),
        "nipple_nearest_component_distance_mm": _stats(scored, "nipple_nearest_component_distance_mm"),
        "annotated_nipple_volume_mm3": _stats(scored, "annotated_nipple_volume_mm3"),
        "predicted_nipple_volume_mm3": _stats(scored, "predicted_nipple_volume_mm3"),
        "nipples_missing": int(scored.predicted_nipple_mm.isna().sum())
                           if "predicted_nipple_mm" in scored else 0,
        "annotations_multi_component": int(scored.annotated_nipple_components.gt(1).sum())
                                        if "annotated_nipple_components" in scored else 0,
        "predictions_multi_nipple_component": int(scored.predicted_nipple_components.gt(1).sum())
                                             if "predicted_nipple_components" in scored else 0,
    }


def evaluate_stored_landmarks(dataset: str, predictions_root: str,
                              outdir: str, name: str, limit: int = 0) -> Dict[str, Any]:
    """Evaluate a labelled unilateral package, preserving failed cases and annotation groups.

    The package's metadata.csv needs case, image (or source_path), breast_region and nipples.
    No model is loaded, no manifests are rewritten, and missing predictions remain unscored.
    The nipple reference is the largest GT component and the prediction is the production
    runner's selected component. Missing nipple predictions have null localisation error.
    """
    from mri_transforms.landmark import evaluation as evaluator
    root, output = Path(dataset).resolve(), Path(outdir)
    metadata = root / "metadata.csv"
    records = pd.read_csv(metadata, keep_default_na=False)
    if not {"case", "breast_region", "nipples"}.issubset(records.columns):
        raise ValueError("metadata.csv requires case, image/source_path, breast_region and nipples")
    if "laterality" in records and not records.laterality.str.lower().eq("unilateral").all():
        raise ValueError("this nipple localisation evaluator requires unilateral annotations")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    exams = exams_from_manifest(str(metadata), limit or None)
    records = records.set_index("case")
    reader, rows, identities, models = StoredLandmarkRunner(), [], [], []
    for exam in exams:
        row = {"case": exam.case, "status": STATUS_OK, "reason": "",
               "annotation_source": records.loc[exam.case].get("annotation_source", "unknown"),
               "annotation_review_flags": records.loc[exam.case].get("review_flags", "")}
        case_out = Path(predictions_root) / exam.case
        try:
            result = reader.run(exam.pre, str(case_out))
            entry = result["entry"]
            gt_region = root / records.loc[exam.case, "breast_region"]
            gt_nipple = root / records.loc[exam.case, "nipples"]
            source = nib.load(exam.pre)
            for label in (gt_region, gt_nipple):
                image = nib.load(label)
                if image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-4):
                    raise ValueError(f"annotation grid differs from source: {label.name}")
            pred_region = case_out / entry["breast_region_file"]
            pred_nipple = case_out / entry["nipples_file"]
            row.update(evaluator.score_case(str(pred_region), str(pred_nipple), str(gt_region),
                                           str(gt_nipple), *evaluator.predicted_nipples(entry), exam.case))
            lr_axis = next(i for i, code in enumerate(nib.aff2axcodes(source.affine))
                           if code in ("L", "R"))
            for key, path in (("annotated", gt_region), ("predicted", pred_region)):
                binary = np.asarray(nib.load(path).dataobj) > 0
                row[f"{key}_region_touches_lr_border"] = bool(
                    np.take(binary, 0, axis=lr_axis).any()
                    or np.take(binary, -1, axis=lr_axis).any())
            row.update(coverage_scope=(result.get("coverage") or {}).get("scope"),
                       coverage_certain=(result.get("coverage") or {}).get("certain"))
            models.append(result["model"])
            identities.append({"case": exam.case, "files": {
                role: {"path": str(path), "sha256": _sha256(path)} for role, path in (
                    ("image", Path(exam.pre)), ("annotated_region", gt_region),
                    ("annotated_nipple", gt_nipple), ("predicted_region", pred_region),
                    ("predicted_nipple", pred_nipple), ("landmarks", case_out / "landmarks.json"))}})
        except Exception as exc:
            row.update(status=STATUS_SCORING_FAILED, reason=str(exc))
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "dataset": str(root), "metric_definition_version": evaluator.METRIC_DEFINITION_VERSION,
        "headline": "median per-case breast-region union Dice",
        "nipple_reference": "largest annotated component versus selected predicted component",
        "localisation_denominator": "cases with an annotated and a selected predicted nipple",
        "models": list({json.dumps(m, sort_keys=True): m for m in models}.values()),
        **_group_summary(frame),
        "by_annotation_source": {str(key): _group_summary(group)
                                 for key, group in frame.groupby("annotation_source")},
        "annotation_flagged_cases": frame.loc[frame.annotation_review_flags.ne(""), "case"].tolist(),
        "failed_cases": frame.loc[frame.status.ne(STATUS_OK), ["case", "reason"]].to_dict("records"),
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{name}_annotation_scores.csv", index=False)
    for filename, content in ((f"{name}_annotation_summary.json", summary),
                              (f"{name}_annotation_manifest.json", {
                                  "metadata_sha256": _sha256(metadata), "cases": identities})):
        (output / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return summary
