"""Score `run_view_laterality.py` against a labelled dataset.

MAMA-MIA ships `view` (axial | sagittal) and `bilateral_mri` (0 | 1) per patient in
`clinical_and_imaging_info.xlsx`. This script joins them onto the CSV, prints the two
confusion matrices, lists every disagreeing case, and re-derives the best left-right FOV
cut from the labels — so the constant in `mri_analysis.geometry` is a measured number
rather than an inherited one.

    mri-analysis evaluate-laterality \
        --csv ./out/MAMA_MIA_view_laterality.csv \
        --xlsx /data/MAMMA_MIA/mama_mia_challenge_dataset/clinical_and_imaging_info.xlsx \
        --out ./out
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import os
import sys

# Source-tree execution adds the src directory; installed commands need no bootstrap.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mri_analysis.geometry import calibrate_fov_cut


def confusion(df: pd.DataFrame, truth: str, pred: str) -> pd.DataFrame:
    return pd.crosstab(df[truth], df[pred], dropna=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default="dataset_info")
    args = ap.parse_args()

    pred = pd.read_csv(args.csv)
    gt = pd.read_excel(args.xlsx, sheet_name=args.sheet)[
        ["patient_id", "dataset", "view", "bilateral_mri", "num_phases",
         "field_strength", "manufacturer", "slice_thickness"]]
    df = pred.merge(gt, left_on="case", right_on="patient_id", how="inner")
    df["gt_plane"] = df["view"].str.lower()
    df["gt_scope"] = df["bilateral_mri"].map({1: "bilateral", 0: "unilateral"})

    print(f"joined {len(df)} of {len(pred)} predicted rows against {len(gt)} labelled rows\n")

    report = {"n_joined": int(len(df))}
    for truth, pred_col in (("gt_plane", "plane"), ("gt_plane", "plane_by_thickest_axis"),
                            ("gt_scope", "scope")):
        cm = confusion(df, truth, pred_col)
        acc = float((df[truth] == df[pred_col]).mean())
        print(f"=== {truth} vs {pred_col}   accuracy {acc:.4f}")
        print(cm.to_string(), "\n")
        report[f"{truth}_vs_{pred_col}"] = {
            "accuracy": round(acc, 4), "matrix": json.loads(cm.to_json())}

    # Phase count is a third label, and a cheap integrity check on exam discovery.
    n_phase_match = int((df["n_phases"] == df["num_phases"]).sum())
    print(f"=== phase count: {n_phase_match}/{len(df)} exams match `num_phases`")
    report["n_phase_count_match"] = n_phase_match
    if n_phase_match < len(df):
        bad = df[df.n_phases != df.num_phases][["case", "n_phases", "num_phases"]]
        print(bad.head(20).to_string(index=False))
        report["phase_count_mismatch"] = json.loads(bad.to_json(orient="records"))

    # The labelled `slice_thickness` is a DICOM attribute; our slice spacing comes from the
    # affine. They should agree, and where they do not, the join or the conversion is off.
    if "slice_thickness" in df and "slice_spacing_mm" in df:
        delta = (df["slice_spacing_mm"] - df["slice_thickness"]).abs()
        agree = int((delta <= 0.05).sum())
        print(f"\n=== slice thickness: header spacing matches the DICOM label on "
              f"{agree}/{len(df)} exams (median |delta| {delta.median():.3f} mm)")
        report["slice_thickness_agreement"] = {
            "n_within_0.05mm": agree, "n": int(len(df)),
            "median_abs_delta_mm": round(float(delta.median()), 4),
            "p95_abs_delta_mm": round(float(delta.quantile(0.95)), 4)}

    # Where the header rule is wrong, and what the labels say the cut should have been.
    cal = calibrate_fov_cut(df["fov_lr_mm"], df["gt_scope"] == "bilateral")
    print(f"\n=== best left-right FOV cut from the labels: {cal}")
    report["fov_cut_calibration"] = cal

    for truth, pred_col in (("gt_plane", "plane"), ("gt_scope", "scope")):
        wrong = df[df[truth] != df[pred_col]]
        if wrong.empty:
            continue
        cols = ["case", "dataset", truth, pred_col, "plane_by_thickest_axis", "axcodes",
                "fov_lr_mm", "fov_ap_mm", "fov_si_mm", "shape", "zooms", "oblique_deg"]
        path = os.path.join(args.out, f"disagreements_{pred_col}.csv")
        wrong[cols].to_csv(path, index=False)
        print(f"\n=== {len(wrong)} disagreements on {pred_col} -> {path}")
        print(wrong[cols].head(15).to_string(index=False))
        print(wrong.groupby(["dataset", truth, pred_col]).size().to_string())

    path = os.path.join(args.out, "evaluation.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
