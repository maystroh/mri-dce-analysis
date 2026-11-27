"""Shared landmark metrics, preserving the union-v3 definition and native grids.

Scores stored masks without importing the model runtime, MLflow or a consumer CLI.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from skimage.measure import label as cc_label

METRIC_DEFINITION_VERSION = "union-v5"

STATUS_OK = "ok"
STATUS_INFERENCE_FAILED = "inference_failed"
STATUS_EMPTY_REGION = "empty_region"
STATUS_MISSING_SOURCE = "missing_source"
#: `mri-analysis`'s stored-prediction evaluator does not separate a missing source volume from
#: an annotation/source grid mismatch, so it reports both as one not-scored status. Named here
#: rather than left a literal so `is_scored` covers every status either evaluator can write.
STATUS_SCORING_FAILED = "scoring_failed"

#: A model failure still SCORES, pinned at 0.0. The model was asked and produced nothing
#: usable, which is a result.
MODEL_FAILURE_STATUSES = (STATUS_INFERENCE_FAILED, STATUS_EMPTY_REGION)

UNSCORED_STATUSES = (STATUS_MISSING_SOURCE, STATUS_SCORING_FAILED)


def is_scored(status: str) -> bool:
    """Whether a case contributes to the metrics. See `UNSCORED_STATUSES` for why not."""
    return status not in UNSCORED_STATUSES


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice on two boolean volumes. Two empty masks agree perfectly; one empty does not."""
    a_sum, b_sum = int(a.sum()), int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    return round(2.0 * float(np.logical_and(a, b).sum()) / float(a_sum + b_sum), 6)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labelled, n = cc_label(mask, return_num=True)
    if n <= 1:
        return mask
    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    return labelled == int(counts.argmax())


def _centroid_mm(mask: np.ndarray, affine: np.ndarray) -> Optional[List[float]]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return [round(float(v), 2) for v in nib.affines.apply_affine(affine, idx.mean(axis=0))]


def _overlapping_nipple_mm(nipples_path: str, gt_reference: np.ndarray, gt_img,
                           case: str, min_inter: float = 0.1) -> Optional[List[float]]:
    """Centroid (RAS mm) of the LARGEST predicted nipple component that overlaps the reference.

    The scored nipple is chosen overlap first, then size, then the first one: a predicted
    component covering more than `min_inter` (10%) of the reference (largest annotated)
    nipple beats every component that does not, and size breaks ties inside each group.
    10% is the TP rule of the detection TPR, so the scored nipple is one the TPR counts. None when no
    component overlaps, and the caller then keeps the runner's `biggest` -- the largest of the
    non-overlapping group, so the order still holds. This matches `centroid_distance_metric`
    in deepbreath (`core/utils/metrics.py`), so the training and the cohort numbers use one rule.

    Read from the mask FILE, like every other number here, and over EVERY component: a
    component the nipple filter rejected but that sits on the annotated nipple is the model
    finding the nipple, and the filter's verdict is reported by the runner, not scored here.
    """
    if not os.path.exists(nipples_path):
        return None
    img = nib.load(nipples_path)
    _check_grid(img, gt_img, case)
    labelled, n = cc_label(np.asarray(img.dataobj) > 0, return_num=True)
    if n == 0:
        return None
    shared = np.bincount(labelled[gt_reference & (labelled > 0)], minlength=n + 1)
    hit = np.flatnonzero(shared > gt_reference.sum() * min_inter)
    hit = hit[hit > 0]
    if hit.size == 0:
        return None
    sizes = np.bincount(labelled.ravel(), minlength=n + 1)
    best = int(hit[np.argmax(sizes[hit])])
    return _centroid_mm(labelled == best, img.affine)


def _check_grid(pred_img, gt_img, case: str) -> None:
    """The written masks index the source grid (`src/landmark/regions.py:23`) and every dev
    annotation was verified to match its source volume — but a silent mismatch here would produce
    a plausible-looking, entirely wrong Dice, so it is checked per case.
    """
    if pred_img.shape != gt_img.shape:
        raise ValueError(f"{case}: prediction {pred_img.shape} vs annotation {gt_img.shape}")
    if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-4):
        raise ValueError(f"{case}: prediction and annotation affines differ")


def score_case(region_path: str, nipples_path: str, gt_region_path: str, gt_nipple_path: str,
               predicted_nipple_mm: Optional[List[float]],
               predicted_nipple_mms: List[List[float]], case: str) -> Dict[str, Any]:
    """Union Dice + its decomposition, and the nipple centroid distance, for one case."""
    gt_region_img = nib.load(gt_region_path)
    pred_region_img = nib.load(region_path)
    _check_grid(pred_region_img, gt_region_img, case)

    gt = np.asarray(gt_region_img.dataobj) > 0
    pred = np.asarray(pred_region_img.dataobj) > 0
    voxel_mm3 = float(np.abs(np.linalg.det(gt_region_img.affine[:3, :3])))

    largest = _largest_component(pred)
    row: Dict[str, Any] = {
        "breast_region_dice": _dice(pred, gt),
        "largest_component_dice": _dice(largest, gt),
        "predicted_volume_mm3": round(float(pred.sum()) * voxel_mm3, 2),
        "annotated_volume_mm3": round(float(gt.sum()) * voxel_mm3, 2),
        "spurious_volume_mm3": round(float(pred.sum() - largest.sum()) * voxel_mm3, 2),
        "predicted_region_components": int(cc_label(pred, return_num=True)[1]),
    }

    gt_nipple_img = nib.load(gt_nipple_path)
    gt_nipple = np.asarray(gt_nipple_img.dataobj) > 0
    # The reference is the LARGEST ground-truth component, mirroring `biggest` on the prediction
    # side. Taking the centre of mass of the whole mask made the two sides asymmetric, and on a
    # multi-component annotation it put the reference point BETWEEN the components, on no anatomy
    # at all — which is how QIN-BREAST-01-0013 read 27.5 mm while the model was 2.6 mm out.
    # Annotations should have exactly one component; the count is reported so that when they do
    # not, it is visible as an annotation problem rather than absorbed into the model's score.
    gt_components = int(cc_label(gt_nipple, return_num=True)[1])
    gt_reference = _largest_component(gt_nipple)
    gt_mm = _centroid_mm(gt_reference, gt_nipple_img.affine)
    # Overlap first, then size: a component on the annotated nipple beats a larger one elsewhere.
    # Without this, a large false positive took the score from the real (smaller) nipple.
    overlapping_mm = (_overlapping_nipple_mm(nipples_path, gt_reference, gt_nipple_img, case)
                      if gt_mm is not None else None)
    if overlapping_mm is not None:
        predicted_nipple_mm = overlapping_mm
    row["annotated_nipple_components"] = gt_components
    row["annotated_nipple_mm"] = gt_mm
    row["predicted_nipple_mm"] = predicted_nipple_mm
    # Null, never imputed: a large stand-in number would corrupt the median distance, which is a
    # statement about localisation on the cases where localisation happened (CONTEXT.md).
    row["nipple_centroid_distance_mm"] = (
        round(float(np.linalg.norm(np.asarray(gt_mm) - np.asarray(predicted_nipple_mm))), 2)
        if gt_mm is not None and predicted_nipple_mm is not None else None)
    # DIAGNOSTIC, never the headline: a model spraying components would drive this to ~0 while
    # getting worse. Read beside the scored distance, the pair separates "can it localise?" from
    # "can it pick the right blob?" — 40.68 mm scored against 0.94 mm nearest is a selection
    # failure, not a localisation failure, and points at a different fix.
    # The overlapping component joins the candidates so the selection cost cannot go negative
    # when it is one the nipple filter rejected (and so absent from `predicted_nipple_mms`).
    candidates = list(predicted_nipple_mms) + ([overlapping_mm] if overlapping_mm else [])
    dists = ([float(np.linalg.norm(np.asarray(gt_mm) - np.asarray(p)))
              for p in candidates] if gt_mm is not None else [])
    row["nipple_nearest_component_distance_mm"] = round(min(dists), 2) if dists else None
    row["nipple_selection_cost_mm"] = (
        round(row["nipple_centroid_distance_mm"] - row["nipple_nearest_component_distance_mm"], 2)
        if row["nipple_centroid_distance_mm"] is not None
        and row["nipple_nearest_component_distance_mm"] is not None else None)
    # Nipple Dice over the FULL predicted nipple mask, every component, not just `biggest`:
    # an overlap metric that quietly dropped the spurious components would be flattering in a way
    # the component count could not correct for. Reported BESIDE the centroid distance, never
    # instead of it — nipple marker size is not a settled convention (median annotated volume
    # varies about sixfold across collections), so this number partly measures whose convention
    # the annotation follows, and is not comparable across collections.
    if os.path.exists(nipples_path):
        pred_nip_img = nib.load(nipples_path)
        _check_grid(pred_nip_img, gt_nipple_img, case)
        pred_nip = np.asarray(pred_nip_img.dataobj) > 0
        row["nipple_dice"] = _dice(pred_nip, gt_nipple)
        row["predicted_nipple_components"] = int(cc_label(pred_nip, return_num=True)[1])
        row["predicted_nipple_volume_mm3"] = round(float(pred_nip.sum()) * voxel_mm3, 2)
    else:
        row["nipple_dice"] = 0.0
        row["predicted_nipple_components"] = 0
        row["predicted_nipple_volume_mm3"] = 0.0
    row["annotated_nipple_volume_mm3"] = round(float(gt_nipple.sum()) * voxel_mm3, 2)
    return row


def predicted_nipples(entry: Dict[str, Any]) -> Tuple[Optional[List[float]], List[List[float]]]:
    """`(biggest component centroid, every component centroid)`, in RAS mm.

    The `biggest` centroid is what gets SCORED unless a component overlaps the annotated
    nipple (see `_overlapping_nipple_mm`, overlap first, then size): not the union of components (one spurious
    component would drag a localisation measure off the nipple using a detection error) and not
    the nearest (a shotgun of components would always score well). Component COUNT carries the
    detection signal instead (CONTEXT.md).

    Every component is returned as well, for the nearest-component DIAGNOSTIC. Components the
    nipple filter rejected are excluded from both, matching how `analyse_nipples` picks `biggest`
    among kept components only.
    """
    biggest, every = None, []
    for side in entry.get("sides", {}).values():
        for nip in side.get("nipples", []):
            if nip.get("filtered_out"):
                continue
            mm = nip.get("position_mm") or {}
            pos = [mm.get("x"), mm.get("y"), mm.get("z")]
            if None in pos:
                continue
            every.append(pos)
            if nip.get("biggest"):
                biggest = pos
    return biggest, every
