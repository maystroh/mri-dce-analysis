#!/usr/bin/env bash
# Open a completed output folder or portable export; pass the folder as the first argument.
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${task_repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${MRI_ANALYSIS_PYTHON:-python}" -m mri_analysis viewer "$@"
