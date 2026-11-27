"""Installed entry point; production dependencies load only for analysis commands."""
from __future__ import annotations

import importlib
import sys


def main() -> int:
    commands = {
        "run": "mri_analysis.pipeline",
        "stage1": "mri_analysis.commands.stage1",
        "stage2": "mri_analysis.commands.stage2",
        "motion": "mri_analysis.motion",
        "evaluate": "mri_analysis.commands.evaluate",
        "viewer": "mri_analysis.viewer",
        "bundle": "mri_analysis.artifacts",
        "view-laterality": "mri_analysis.tools.run_view_laterality",
        "evaluate-laterality": "mri_analysis.tools.evaluate_view_laterality",
        "strata": "mri_analysis.tools.report_strata",
        "worklist": "mri_analysis.tools.make_review_worklist",
    }
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: mri-analysis COMMAND [options]\nCommands: " + ", ".join(commands) +
              "\nRun COMMAND --help for command options.")
        return 0
    command = sys.argv[1]
    if command not in commands:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 2
    sys.argv = [f"mri-analysis {command}", *sys.argv[2:]]
    return importlib.import_module(commands[command]).main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
