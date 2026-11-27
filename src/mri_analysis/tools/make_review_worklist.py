"""Rank the volumes worth checking by eye, from a mri-analysis stage1 CSV.

    mri-analysis worklist --csv out/ispy1/ispy1_stage1.csv --out out/ispy1

Nothing here is known to be wrong. These are the rows where the rule ran with the least
support: no independent reference, a check it could not apply, or a value close enough to a
cut that the verdict could go either way. Laterality is a property of the ACQUISITION, so
phases of one series collapse to a single row rather than being checked N times.
"""
import argparse, os, warnings
import pandas as pd
warnings.filterwarnings("ignore")

WIDTH_CUT, MIN_SIDE_CUT, BAND = 260.0, 98.0, 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="a *_stage1.csv from mri-analysis stage1")
    ap.add_argument("--out", default=".", help="directory for the two worklists")
    ap.add_argument("--per-group", type=int, default=4,
                    help="cap per (priority, dataset) so one group cannot fill the list")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)
    # run_stage1 writes an absolute `path`; group by whatever identifies a study here.
    for col, default in (("dataset", ""), ("patient", ""), ("study", "")):
        if col not in d.columns:
            d[col] = default

    # ---------- 1. the 4D volumes ----------
    four = d[d.n_volumes > 1].copy()
    four["why"] = [
        "phases stacked in the 4th dim; sagittal so fov_lr is understated"
        if r.view == "sagittal" else
        "phases stacked in the 4th dim; axial so fov_lr is unaffected"
        for r in four.itertuples()]
    four_out = os.path.join(args.out, "review_4d_volumes.csv")
    four[["dataset", "patient", "study", "label", "description", "view", "laterality",
          "n_volumes", "fov_lr_mm", "scope_certain", "why", "path"]].to_csv(
        four_out, index=False)
    print(f"wrote {four_out}  {len(four)} volumes")

    # ---------- 2. laterality to verify ----------
    # Laterality is a property of the ACQUISITION, not of each phase: post1..postN of one
    # series share a grid. Collapse to one row per series so a check is not repeated 34x.
    one = d.sort_values("label").groupby(
        ["dataset", "patient", "study", "description"], as_index=False).first()
    comparable = one.dicom_laterality_source.eq("image_geometry")

    def why(r, idx):
        # NOTE: index this row with [] , never r.view -- `view` is a pandas Series METHOD,
        # so attribute access silently returns the method and every test against it fails.
        out = []
        view, fov = r["view"], r["fov_lr_mm"]
        nvol, mside = r["n_volumes"], r["min_side_mm"]
        if comparable[idx] and r["laterality"] != r["dicom_laterality"]:
            out.append("A: disagrees with the DICOM geometry rule")
        if nvol > 1 and view == "sagittal":
            out.append(f"B: 4D ({int(nvol)} vols) on a sagittal grid, fov_lr may be halved")
        if abs(fov - WIDTH_CUT) <= BAND:
            out.append(f"C: {abs(fov-WIDTH_CUT):.0f}mm from the 260mm width cut")
        if abs(mside - MIN_SIDE_CUT) <= BAND:
            out.append(f"D: {abs(mside-MIN_SIDE_CUT):.0f}mm from the 98mm centring cut")
        if fov >= WIDTH_CUT and mside < MIN_SIDE_CUT:
            out.append("E: wide but off-centre -> called unilateral by the centring rule")
        if view == "sagittal" and not comparable[idx]:
            out.append("F: sagittal, no independent reference")
        return "; ".join(out)

    one["why_check"] = [why(r, i) for i, r in one.iterrows()]
    sel = one[one.why_check != ""].copy()
    sel["priority"] = sel.why_check.str[0]
    sel = sel.sort_values(["priority", "dataset"])
    sel = sel.groupby(["priority", "dataset"], sort=False).head(args.per_group)
    sel["verdict_after_review"] = ""
    cols = ["priority", "why_check", "dataset", "patient", "description", "label",
            "view", "laterality", "fov_lr_mm", "min_side_mm", "scope_certain",
            "dicom_laterality", "dicom_laterality_source", "verdict_after_review", "path"]
    lat_out = os.path.join(args.out, "review_laterality.csv")
    sel[cols].to_csv(lat_out, index=False)
    print(f"wrote {lat_out}  {len(sel)} series "
          f"(collapsed from {len(d)} volumes -> {len(one)} series)\n")
    MEAN = {"A": "disagrees with the DICOM geometry rule",
            "B": "4D on a sagittal grid (fov_lr may be halved)",
            "C": "near the 260mm width cut", "D": "near the 98mm centring cut",
            "E": "wide but off-centre", "F": "sagittal, no independent reference"}
    t = sel.groupby("priority").size().rename("n").to_frame()
    t["meaning"] = [MEAN[p] for p in t.index]
    print(t.to_string())


if __name__ == "__main__":
    main()
