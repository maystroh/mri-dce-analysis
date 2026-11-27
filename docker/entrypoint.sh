#!/usr/bin/env bash
# Container entry point: adds container defaults to mri-dce-analysis commands.
#   run     -> --model-bundle "$MODEL_BUNDLE" and --no-viewer unless given
#   viewer  -> --host 0.0.0.0 unless given
#   example duke|odelia [run options] -> runs a bundled example into /out/example_<name>
# Any other command (bash, dicom-info, mri-viewer, ...) runs as given.
set -euo pipefail

has_flag() {
  local flag="$1" arg
  shift
  for arg in "$@"; do
    [[ "$arg" == "$flag" || "$arg" == "$flag="* ]] && return 0
  done
  return 1
}

usage() {
  cat <<EOF
mri-dce-analysis container

  example duke|odelia [options]    run a bundled example exam into /out/example_<name>
  run --input /data/... --out NAME [options]     (--organize for untidy input folders)
  viewer /out/NAME                 serve a result folder on port 8060
  <any mri-dce-analysis command>   e.g. stage1, stage2, motion, evaluate, bundle
  dicom-info | mri-viewer | bash

Model bundle: $MODEL_BUNDLE
Inputs are mounted at /data, results are written to /out.
EOF
  mri-dce-analysis --help
}

commands=" run stage1 stage2 motion evaluate viewer bundle view-laterality evaluate-laterality strata worklist "

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" || "$1" == "help" ]]; then
  usage
  exit 0
fi

command="$1"
shift

if [[ ! -w /out ]]; then
  echo "/out is not writable by user $(id -u):$(id -g)." >&2
  echo "On the host, run: mkdir -p results && sudo chown $(id -u):$(id -g) results" >&2
  exit 1
fi

case "$command" in
  example)
    name="${1:-}"
    [[ $# -gt 0 ]] && shift
    if [[ ! -f /app/examples/.verified ]]; then
      echo "The example images were not included in this image (Git LFS pointers)." >&2
      echo "Run 'git lfs pull' in the repository, then 'docker compose build'." >&2
      exit 1
    fi
    case "$name" in
      duke) example=(--input /app/examples/dicom/duke --dicom --name duke) ;;
      odelia) example=(--input /app/examples/nifti/odelia --layout odelia --name odelia) ;;
      *) echo "Usage: example duke|odelia [run options]" >&2; exit 2 ;;
    esac
    exec "$0" run "${example[@]}" --out "/out/example_${name}" --device cpu --motion "$@"
    ;;
esac

if [[ "$commands" != *" $command "* ]]; then
  exec "$command" "$@"
fi

args=("$@")
if [[ "$command" == "run" ]] && ! has_flag --help "${args[@]}" && ! has_flag -h "${args[@]}"; then
  if ! has_flag --model-bundle "${args[@]}" && ! has_flag --tracking-uri "${args[@]}" \
     && ! has_flag --analyse-only "${args[@]}"; then
    args+=(--model-bundle "$MODEL_BUNDLE")
  fi
  # The pipeline's own viewer binds 127.0.0.1, unreachable from outside the container.
  has_flag --no-viewer "${args[@]}" || args+=(--no-viewer)
fi
if [[ "$command" == "viewer" ]] && ! has_flag --host "${args[@]}"; then
  args+=(--host 0.0.0.0)
fi

exec mri-dce-analysis "$command" "${args[@]}"
