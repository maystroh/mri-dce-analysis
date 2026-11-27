"""Cross-tabulate geometry against scanner context, and name the gaps.

Stage 1 answers "what geometry is this exam". This script answers the question that
actually drives the landmark-model decision: **which combinations of geometry and scanner
do we have enough exams to measure anything on, and which can we never measure at all?**

Three things get flagged, because each one breaks a different conclusion:

* **empty cell** — no exam has this combination, so no result on this dataset can ever
  speak to it. A model shipped for it would be shipped untested.
* **thin cell** — populated but below `--min-exams`, so a measured rate there is noise.
* **confound** — a geometry that occurs at only ONE level of a scanner factor. Any
  difference measured on that geometry is the geometry AND the factor together, and this
  dataset cannot separate them.

    mri-analysis strata --csv ./out/MAMA_MIA_view_laterality.csv --out ./out --name MAMA_MIA
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

GEOMETRY = ["plane", "scope"]
FACTORS = ["field_strength_bin", "manufacturer"]


def geometry_label(row) -> str:
    return f"{row['plane']}/{row['scope']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="output of run_view_laterality.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--min-exams", type=int, default=20,
                    help="a cell below this is reported as thin")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["geometry"] = df.apply(geometry_label, axis=1)

    report = {"dataset": args.name, "n_exams": int(len(df)), "min_exams": args.min_exams}
    lines = [f"# Strata — {args.name}", "",
             f"{len(df)} exams. A cell below {args.min_exams} exams is marked thin; "
             f"an empty cell cannot be measured on this dataset at all.", ""]

    for factor in FACTORS:
        table = pd.crosstab(df["geometry"], df[factor])
        lines += [f"## geometry x {factor}", "", table.to_markdown(), ""]
        report[f"geometry_x_{factor}"] = json.loads(table.to_json())

        empty, thin = [], []
        for geom in table.index:
            for level in table.columns:
                n = int(table.loc[geom, level])
                if n == 0:
                    empty.append(f"{geom} @ {level}")
                elif n < args.min_exams:
                    thin.append(f"{geom} @ {level} (n={n})")
        report[f"empty_cells_{factor}"] = empty
        report[f"thin_cells_{factor}"] = thin
        if empty:
            lines += ["**Empty — never measurable here:** " + "; ".join(empty), ""]
        if thin:
            lines += ["**Thin — a rate here is noise:** " + "; ".join(thin), ""]

        # A geometry seen at only one level of this factor is confounded with it.
        confounded = []
        for geom, row in table.iterrows():
            present = [str(c) for c in table.columns if row[c] > 0]
            if len(present) == 1 and row.sum() > 0:
                # Is there a comparison geometry at the same level? If so the confound is
                # controllable: hold the factor fixed and vary the geometry instead.
                same_level = [str(g) for g in table.index
                              if g != geom and table.loc[g, present[0]] >= args.min_exams]
                confounded.append({"geometry": geom, "n": int(row.sum()),
                                   "only_level": present[0],
                                   "controllable_against": same_level})
        report[f"confounded_{factor}"] = confounded
        for c in confounded:
            control = (f"controllable — compare against {', '.join(c['controllable_against'])} "
                       f"at {c['only_level']}" if c["controllable_against"]
                       else "NOT controllable — no comparison geometry at this level")
            lines += [f"**Confound:** all {c['n']} `{c['geometry']}` exams are "
                      f"`{c['only_level']}`. {control}.", ""]

    table = pd.crosstab(df["field_strength_bin"], df["manufacturer"])
    lines += ["## field strength x manufacturer", "", table.to_markdown(), ""]
    report["field_x_manufacturer"] = json.loads(table.to_json())

    if "site" in df.columns:
        by_site = pd.crosstab(df["site"], df["geometry"])
        mixed = by_site[(by_site > 0).sum(axis=1) > 1]
        report["n_sites"] = int(df["site"].nunique())
        report["n_sites_with_mixed_geometry"] = int(len(mixed))
        lines += [f"## sites", "",
                  f"{df['site'].nunique()} sites; {len(mixed)} of them contribute more than "
                  f"one geometry. A single-geometry site cannot be separated from the "
                  f"geometry it contributes.", ""]
        if len(mixed):
            lines += [mixed.to_markdown(), ""]

    md_path = os.path.join(args.out, f"{args.name}_strata.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    json_path = os.path.join(args.out, f"{args.name}_strata.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n".join(lines))
    print(f"wrote {md_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
