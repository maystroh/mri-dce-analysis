"""Characterise a dataset: acquisition plane and acquisition scope, one row per exam.

Stage 1 of the analysis. It reads NIfTI HEADERS only, so a 1500-exam dataset takes
seconds. The CSV it writes is the artifact: it carries every measurement next to every
verdict, so a cut can be moved and the whole dataset re-decided without touching a file
again.

    mri-analysis view-laterality --dataset /data/MAMMA_MIA/images --out ./out/mama_mia
    mri-analysis view-laterality --dataset /data/ODELIA/ALL --layout odelia --out ./out/odelia

Phase consistency is checked, not assumed. An exam whose phases disagree on plane or on
field of view is reported in `phase_disagreement`: the pipeline registers phases to each
other, so two phases on different grids is a finding, not a detail.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import os
import sys

# Source-tree execution adds the src directory; installed commands need no bootstrap.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mri_analysis import acquisition
from mri_analysis.discovery import iter_exams
from mri_analysis.geometry import (ABSTAIN_MARGIN_MM, BILATERAL_FOV_MM, CONFLICT,
                                   axis_geometry, resolve, scope_by_fov)

FIELDS = (["case", "n_phases",
           # Fused verdicts, then the per-layer evidence behind each one.
           "plane", "plane_source", "plane_agreement", "plane_dicom", "plane_nifti",
           "scope", "scope_source", "scope_agreement", "scope_dicom", "scope_nifti",
           "scope_dicom_source", "scope_certain",
           "plane_needs_review", "scope_needs_review",
           "trusted", "flag_multi_volume", "flag_near_cut", "flag_midline_unknown",
           "flag_oblique", "flag_plane_conflict",
           "plane_by_thickest_axis", "plane_agrees", "axcodes",
           "fov_margin_mm", "fov_lr_mm", "fov_ap_mm", "fov_si_mm",
           "min_side_mm", "lr_min_mm", "lr_max_mm", "n_volumes",
           "frame_usable", "ap_straddles_zero", "axes_touching_zero"]
          + acquisition.ROW_FIELDS
          + ["through_axis", "n_slices", "slice_spacing_mm", "shape", "zooms",
             "slice_overlap", "oblique_deg", "phase_disagreement", "reason", "error",
             "pre_file"])


def characterise(exam, fov_cut_mm: float, abstain_mm: float) -> Dict[str, object]:
    """One row for one exam, decided on the pre-contrast phase and checked against the rest."""
    geoms = [axis_geometry(p) for p in exam.phases]
    head = geoms[0]
    scope, margin, reason, certain = scope_by_fov(head, fov_cut_mm, abstain_mm)

    disagree: List[str] = []
    for other in geoms[1:]:
        if other.error:
            disagree.append(f"unreadable:{os.path.basename(other.path)}")
            continue
        if other.plane != head.plane:
            disagree.append(f"plane:{other.plane}")
        if abs(other.fov_lr_mm - head.fov_lr_mm) > 1.0:
            disagree.append(f"fov_lr:{other.fov_lr_mm:.0f}")
        if other.shape != head.shape:
            disagree.append(f"shape:{'x'.join(map(str, other.shape))}")

    row = head.as_row()
    row["plane_nifti"] = head.plane
    row["scope_nifti"] = scope
    # A phase disagreement makes the exam-level verdict unsafe too: the verdict was decided
    # on phase 0 and the other phases are not on that grid.
    certain = certain and not disagree
    row.update(case=exam.case, n_phases=exam.n_phases, scope=scope, scope_certain=certain,
               fov_margin_mm=margin, reason=reason,
               phase_disagreement=";".join(sorted(set(disagree))),
               pre_file=os.path.basename(exam.phases[0]))
    return {k: row.get(k, "") for k in FIELDS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="root holding one directory per exam")
    ap.add_argument("--out", required=True, help="output directory for the CSV and summary")
    ap.add_argument("--layout", default="auto", choices=["auto", "mama_mia", "odelia", "generic"])
    ap.add_argument("--fov-cut-mm", type=float, default=BILATERAL_FOV_MM,
                    help="left-right FOV at or above which an exam reads bilateral")
    ap.add_argument("--abstain-mm", type=float, default=ABSTAIN_MARGIN_MM,
                    help="flag scope as uncertain within this margin of the cut; 0 disables")
    ap.add_argument("--metadata", default="auto",
                    help="Layer 1 metadata, as 'kind:path'. Default 'auto' looks for a "
                         "DICOM series CSV beside the dataset and falls back to the NIfTI "
                         "header alone when there is none. Kinds: series_csv:<csv> | "
                         "mama_mia:<xlsx> | odelia:<challenge root> | dicom:<dicom root> | "
                         "none")
    ap.add_argument("--limit", type=int, default=None, help="first N exams only, for a dry run")
    ap.add_argument("--name", default=None, help="basename of the CSV (default: the dataset dir)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(args.dataset))
    csv_path = os.path.join(args.out, f"{name}_view_laterality.csv")

    spec = args.metadata
    if spec == "auto":
        spec = acquisition.discover(args.dataset)
        print(f"metadata: {spec or 'none found — using the NIfTI header alone'}",
              file=sys.stderr)
    contexts = acquisition.load(spec) if spec and spec != "none" else {}

    rows: List[Dict[str, object]] = []
    n_joined = 0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, exam in enumerate(iter_exams(args.dataset, args.layout, args.limit), 1):
            row = characterise(exam, args.fov_cut_mm, args.abstain_mm)
            ctx = contexts.get(exam.case)
            row.update(ctx.as_row() if ctx else acquisition.EMPTY_ROW)

            row["plane_dicom"] = getattr(ctx, "plane_dicom", None) or ""
            row["scope_dicom"] = getattr(ctx, "scope_dicom", None) or ""
            row["scope_dicom_source"] = getattr(ctx, "scope_dicom_source", None) or ""
            # PLANE: DICOM wins a conflict. Both layers answer the same question and our
            # axcodes[2] rule is a proxy a converter can break (reorientation -> all LAS).
            row["plane"], row["plane_source"], row["plane_agreement"] = resolve(
                row["plane_dicom"], row["plane_nifti"], prefer="dicom")
            # SCOPE: our measurement wins. Layer 1 mostly answers a DIFFERENT question --
            # which breast the exam is about, not which breasts the image covers.
            row["scope"], row["scope_source"], row["scope_agreement"] = resolve(
                row["scope_dicom"], row["scope_nifti"], prefer="nifti")
            # A conflict no longer changes the verdict or the certainty of the FOV rule.
            # It raises a review flag, so the CSV can be filtered for manual checking.
            row["plane_needs_review"] = row["plane_agreement"] == CONFLICT
            row["scope_needs_review"] = row["scope_agreement"] == CONFLICT
            if row["plane_needs_review"] or row["scope_needs_review"]:
                row["reason"] += "; disagrees with DICOM (see *_dicom columns)"
            row["flag_multi_volume"] = bool(row.get("n_volumes", 1) > 1)
            row["flag_near_cut"] = bool(
                abs((row.get("fov_lr_mm") or 0) - args.fov_cut_mm) <= args.abstain_mm
                or abs((row.get("min_side_mm") or 0) - 98.0) <= args.abstain_mm)
            # The frame plausibility test (`geometry.axis_geometry`), not the old
            # LR-edge heuristic: an LR edge on the midline only matters when the centring
            # test actually decides, and on a narrow exam the width rule already has.
            row["flag_midline_unknown"] = not bool(row.get("frame_usable"))
            row["flag_oblique"] = bool((row.get("oblique_deg") or 0) > 5.0)
            # A plane conflict means a converter may have rewritten the grid, so the
            # geometry cannot be acted on unattended even though DICOM supplied the verdict.
            row["flag_plane_conflict"] = bool(row["plane_needs_review"])
            row["trusted"] = bool(row["scope_certain"] and not row["flag_multi_volume"]
                                  and not row["flag_plane_conflict"]
                                  and not row.get("error"))
            row["slice_overlap"] = acquisition.slice_overlap(
                row["slice_spacing_mm"], row["slice_thickness_mm"])
            n_joined += ctx is not None
            writer.writerow(row)
            rows.append(row)
            if i % 100 == 0:
                print(f"  {i} exams", file=sys.stderr, flush=True)

    def tally(key):
        out: Dict[str, int] = {}
        for r in rows:
            out[str(r[key])] = out.get(str(r[key]), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def cross(*keys):
        out: Dict[str, int] = {}
        for r in rows:
            out["/".join(str(r[k]) for k in keys)] = out.get(
                "/".join(str(r[k]) for k in keys), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    summary = {
        "dataset": args.dataset,
        "n_exams": len(rows),
        "fov_cut_mm": args.fov_cut_mm,
        "metadata_source": spec or "none",
        "n_metadata_joined": n_joined,
        "n_metadata_missing": len(rows) - n_joined,
        "plane": tally("plane"),
        "scope": tally("scope"),
        "plane_x_scope": {},
        "field_strength": tally("field_strength_bin"),
        "manufacturer": tally("manufacturer"),
        "site": tally("site"),
        "scanner_model": tally("scanner_model"),
        "fat_suppressed": tally("fat_suppressed"),
        "slice_overlap": tally("slice_overlap"),
        "geometry_x_slice_overlap": cross("plane", "scope", "slice_overlap"),
        # The stratification the landmark model is judged on: a geometry it was not built
        # for, on a scanner it has not been seen on, is where a failure would hide.
        "geometry_x_field": cross("plane", "scope", "field_strength_bin"),
        "geometry_x_manufacturer": cross("plane", "scope", "manufacturer"),
        "field_x_manufacturer": cross("field_strength_bin", "manufacturer"),
        "n_plane_rule_disagreement": sum(1 for r in rows if not r["plane_agrees"]),
        "n_oblique_gt_5deg": sum(1 for r in rows if float(r["oblique_deg"] or 0) > 5),
        "abstain_mm": args.abstain_mm,
        "n_trusted": sum(1 for r in rows if r.get("trusted")),
        "n_flag_multi_volume": sum(1 for r in rows if r.get("flag_multi_volume")),
        "n_flag_midline_unknown": sum(1 for r in rows if r.get("flag_midline_unknown")),
        "n_scope_uncertain": sum(1 for r in rows if not r["scope_certain"]),
        "n_plane_needs_review": sum(1 for r in rows if r.get("plane_needs_review")),
        "n_scope_needs_review": sum(1 for r in rows if r.get("scope_needs_review")),
        "plane_source": tally("plane_source"),
        "scope_source": tally("scope_source"),
        "plane_agreement": tally("plane_agreement"),
        "scope_agreement": tally("scope_agreement"),
        "conflicts": [r["case"] for r in rows
                      if CONFLICT in (r["plane_agreement"], r["scope_agreement"])][:40],
        "uncertain_cases": [r["case"] for r in rows if not r["scope_certain"]][:40],
        "n_phase_disagreement": sum(1 for r in rows if r["phase_disagreement"]),
        "n_error": sum(1 for r in rows if r["error"]),
        "closest_to_fov_cut": sorted(
            ({"case": r["case"], "fov_lr_mm": r["fov_lr_mm"], "scope": r["scope"]}
             for r in rows if r["scope"] != "unknown"),
            key=lambda d: abs(float(d["fov_lr_mm"]) - args.fov_cut_mm))[:10],
    }
    summary["plane_x_scope"] = cross("plane", "scope")

    summary_path = os.path.join(args.out, f"{name}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
