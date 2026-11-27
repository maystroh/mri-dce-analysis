#!/usr/bin/env bash
# Geometry -> landmarks -> available unilateral annotation scores -> viewer.
# Works from any directory, including paths containing spaces; accepts `run --help` flags.
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${task_repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${MRI_ANALYSIS_PYTHON:-python}" -m mri_analysis run "$@"
