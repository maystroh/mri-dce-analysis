"""Evaluate saved landmark masks against a labelled unilateral NIfTI package."""
from __future__ import annotations

import argparse
import json

from mri_analysis.evaluation import evaluate_stored_landmarks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="package with metadata.csv and label masks")
    parser.add_argument("--predictions-root", required=True, help="stage-2 predictions directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="cohort")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N manifest cases")
    args = parser.parse_args()
    summary = evaluate_stored_landmarks(args.dataset, args.predictions_root, args.out, args.name, args.limit)
    print(json.dumps(summary, indent=2))
    return 1 if summary["unscored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
