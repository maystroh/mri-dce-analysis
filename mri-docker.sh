#!/usr/bin/env bash
# Run mri-dce-analysis in Docker on folders of this computer.
#
#   bash mri-docker.sh run --input ~/exams --out ~/results/study --motion
#   bash mri-docker.sh viewer ~/results/study
#   bash mri-docker.sh example duke --out ~/results/example_duke --viewer
#
# Path options (--input, --out, ...) take ordinary host paths, relative or absolute.
# Each one is mounted into the container and rewritten; inputs are mounted read-only.
# Results belong to the user who runs the script. Needs only Docker and the image.
set -euo pipefail

image="${MRI_IMAGE:-mri-dce-analysis:0.2.5}"
port="${VIEWER_PORT:-8060}"
# All interfaces by default so the viewer on a server is reachable from other computers.
# VIEWER_BIND=127.0.0.1 keeps it on this machine only.
bind="${VIEWER_BIND:-0.0.0.0}"

usage() {
  cat <<EOF
Usage: bash mri-docker.sh COMMAND [options]

  run --input PATH --out PATH [options]   analyse exams (DICOM or NIfTI, detected)
  viewer PATH                             review a result folder at http://localhost:$port
  example duke|odelia --out PATH          analyse a bundled example exam

  Add --viewer to run or example to open the viewer on --out when the run succeeds.
  Add --organize to run to arrange an untidy --input folder into exams automatically,
  or --organize-check to only list the exams it would form.
  <other command> [options]               any mri-dce-analysis command, e.g. run --help

Host paths given to --input, --input-json, --out, --annotations, --predictions-root,
--model-bundle and --dataset are mounted automatically, and so are the exam folders
listed in an --input-json file.

Environment: MRI_IMAGE ($image), VIEWER_PORT ($port), VIEWER_BIND ($bind),
MRI_DOCKER_OPTIONS (extra docker run options, e.g. "--gpus all").
EOF
}

die() { echo "mri-docker: $*" >&2; exit 1; }

# Absolute host path; the parent folder must exist.
host_path() {
  local path="$1"
  [[ "$path" == "~"* ]] && path="$HOME${path:1}"
  if [[ -d "$path" ]]; then
    (cd -- "$path" && pwd -P)
  else
    local parent
    parent="$(cd -- "$(dirname -- "$path")" 2>/dev/null && pwd -P)" || die "folder not found: $(dirname -- "$path")"
    printf '%s/%s\n' "$parent" "$(basename -- "$path")"
  fi
}

mounts=()
host_dirs=()
host_modes=()
container_dirs=()
count=0
mounted=""
json_copy_dir=""   # temporary JSON copy with container paths, deleted after the run

# Folders are mounted at the same path inside the container, so messages show host
# paths; folders that would hide parts of the container go under /host/N instead.
container_dir() {
  case "$1" in
    /|/app|/app/*|/bin|/bin/*|/data|/data/*|/dev|/dev/*|/etc|/etc/*|/host|/host/*|/lib*|\
    /opt|/opt/*|/out|/out/*|/proc|/proc/*|/root|/root/*|/run|/run/*|/sbin|/sbin/*|\
    /sys|/sys/*|/tmp|/tmp/*|/usr|/usr/*|/var|/var/*)
      echo "/host/$count" ;;
    *) echo "$1" ;;
  esac
}

# Mount a host path and set $mounted to its container path.
# Modes: ro (must exist), rw (folder created if missing).
mount_path() {
  local path mode="$2" dir name="" target="" i
  if [[ "$mode" == "rw" ]] && ! error="$(LC_ALL=C mkdir -p -- "$1" 2>&1)"; then
    # On a Windows drive under WSL, a deleted folder still open in a program keeps its
    # name reserved: mkdir reports "File exists" although nothing is there.
    if [[ "$error" == *"File exists"* && ! -e "$1" ]]; then
      die "cannot create $1: the name is still in use by a deleted folder." \
        "Close programs that had it open (Explorer, editors, running viewers) or create it from Windows, then retry."
    fi
    die "cannot create $1: ${error#mkdir: }"
  fi
  path="$(host_path "$1")" || exit 1
  [[ -e "$path" ]] || die "not found: $1"
  [[ "$path" != *:* ]] || die "paths containing ':' cannot be mounted: $path"
  dir="$path"
  if [[ ! -d "$path" ]]; then
    dir="$(dirname -- "$path")"
    name="/$(basename -- "$path")"
  fi
  # Docker refuses two mounts on one container path; reuse an earlier mount.
  for ((i = 0; i < ${#host_dirs[@]}; i++)); do
    if [[ "${host_dirs[i]}" == "$dir" ]]; then
      target="${container_dirs[i]}"
    elif [[ -z "$target" && "$dir" == "${host_dirs[i]%/}/"* && ( "$mode" == ro || "${host_modes[i]}" == rw ) ]]; then
      # Inside an earlier mount; a folder to write never reuses a read-only one.
      target="${container_dirs[i]%/}${dir#"${host_dirs[i]%/}"}"
    fi
  done
  if [[ -z "$target" ]]; then
    target="$(container_dir "$dir")"
    count=$((count + 1))
    mounts+=(-v "$dir:$target:$mode")
    host_dirs+=("$dir")
    host_modes+=("$mode")
    container_dirs+=("$target")
  fi
  mounted="$target$name"
}

mode_for() {
  case "$1" in
    --out|--predictions-root) echo rw ;;
    --input|--input-json|--annotations|--model-bundle|--dataset) echo ro ;;
    *) return 1 ;;
  esac
}

# --input-json lists exam folders that can be anywhere on this computer: mount each one.
# If Docker sees any of them at another path (/host/N), pass a copy of the JSON with the
# container paths. Sets $mounted to the JSON path to give the container.
mount_json_exams() {
  local json json_dir json_mount="$mounted" source changed=false
  local -a exams=() targets=()
  json="$(host_path "$1")" || exit 1
  json_dir="$(dirname -- "$json")"
  mapfile -t exams < <(docker run --rm --entrypoint python -u "$(id -u):$(id -g)" \
    -v "$json_dir:/json:ro" "$image" -c '
import json, os, sys
try:
    data = json.load(open("/json/" + sys.argv[1]))
except (OSError, ValueError):
    sys.exit(0)  # the analysis reports the invalid JSON itself
for key in data if isinstance(data, dict) else []:
    if "\n" not in key:
        print(os.path.normpath(os.path.join(sys.argv[2], os.path.expanduser(key))))
' "$(basename -- "$json")" "$json_dir")
  for source in ${exams[@]+"${exams[@]}"}; do
    if [[ -d "$source" ]]; then
      mount_path "$source" ro
      targets+=("$mounted")
      [[ "$mounted" == "$source" ]] || changed=true
    else
      echo "Note: exam folder $source of the JSON was not found; it is reported as missing." >&2
      targets+=("$source")
    fi
  done
  mounted="$json_mount"
  $changed || return 0
  local copy_dir
  copy_dir="$(mktemp -d "${TMPDIR:-/tmp}/mri-docker-json.XXXXXX")"
  json_copy_dir="$copy_dir"
  paste -d '\n' <(printf '%s\n' "${exams[@]}") <(printf '%s\n' "${targets[@]}") |
    docker run --rm -i --entrypoint python -u "$(id -u):$(id -g)" \
      -v "$json_dir:/json:ro" -v "$copy_dir:/copy" "$image" -c '
import json, os, sys
lines = sys.stdin.read().splitlines()
moved = dict(zip(lines[0::2], lines[1::2]))
data = json.load(open("/json/" + sys.argv[1]))
copy = {moved.get(os.path.normpath(os.path.join(sys.argv[2], os.path.expanduser(key))), key): value
        for key, value in data.items()}
json.dump(copy, open("/copy/" + sys.argv[1], "w"), indent=2)
' "$(basename -- "$json")" "$json_dir" || die "cannot write the container copy of $1"
  mount_path "$copy_dir/$(basename -- "$json")" ro
}

# Print the --input of each NIfTI run in a result folder, one per line, and the NIfTI
# exam folders of --input-json runs. NIfTI images are read in place; DICOM runs keep
# converted copies in the result folder.
# Parsed with the image's Python so the host needs nothing besides Docker.
run_inputs() {
  docker run --rm --entrypoint python -u "$(id -u):$(id -g)" -v "$1:/results:ro" "$image" -c '
import glob, json
for path in sorted(glob.glob("/results/*_pipeline.json")):
    try:
        run = json.load(open(path))
    except (OSError, ValueError):
        continue
    # NIfTI runs, and --organize runs, whose NIfTI phases stay in the input folder.
    if run.get("input") and ("conversion" not in run or run.get("organize")):
        print(run["input"])
    manifest = (run.get("conversion") or {}).get("exam_manifest")
    try:
        cases = json.load(open("/results/" + manifest.rsplit("/", 1)[-1]))["cases"] if manifest else []
    except (OSError, ValueError, KeyError, TypeError):
        cases = []
    for case in cases:
        if case.get("input_type") == "nifti" and str(case.get("nifti_directory", "")).startswith("/"):
            print(case["nifti_directory"])
' | sort -u
}

[[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]] && { usage; exit 0; }
command -v docker >/dev/null || die "docker is not installed or not on PATH"
docker image inspect "$image" >/dev/null 2>&1 \
  || die "image $image not found; build it with 'docker compose build' in the project folder"

command="$1"
shift
args=("$command")
docker_options=()
has_out=false
out_path=""
open_viewer=false

if [[ "$command" == "viewer" ]]; then
  [[ $# -gt 0 && "$1" != -* ]] || die "usage: bash mri-docker.sh viewer RESULT_FOLDER"
  [[ -d "$1" ]] || die "result folder not found: $1"
  mount_path "$1" rw
  args+=("$mounted" --port 8060)
  results_dir="$(host_path "$1")"
  shift
  while IFS= read -r source; do
    if [[ -e "$source" ]]; then
      mount_path "$source" ro
    else
      echo "Note: input folder $source of a run here was not found; that run's images will not show." >&2
    fi
  done < <(run_inputs "$results_dir")
  docker_options+=(-p "$bind:$port:8060")
  echo "Viewer: http://localhost:$port  (Ctrl+C stops it)"
  if [[ "$bind" == "0.0.0.0" ]]; then
    for address in $(hostname -I 2>/dev/null); do
      [[ "$address" == *:* ]] || echo "  From another computer: http://$address:$port"
    done
    echo "  Open to the network (no login). Set VIEWER_BIND=127.0.0.1 to keep it on this machine."
  fi
fi

while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  [[ "$arg" == "--out" || "$arg" == --out=* ]] && has_out=true
  [[ "$arg" == --out=* ]] && out_path="${arg#*=}"
  [[ "$arg" == "--out" && $# -gt 0 ]] && out_path="$1"
  if [[ "$arg" == "--viewer" && ( "$command" == "run" || "$command" == "example" ) ]]; then
    open_viewer=true
  elif [[ "$arg" == --*=* ]] && mode="$(mode_for "${arg%%=*}")"; then
    mount_path "${arg#*=}" "$mode"
    [[ "${arg%%=*}" == "--input-json" ]] && mount_json_exams "${arg#*=}"
    args+=("${arg%%=*}=$mounted")
  elif mode="$(mode_for "$arg")"; then
    [[ $# -gt 0 ]] || die "$arg needs a path"
    mount_path "$1" "$mode"
    [[ "$arg" == "--input-json" ]] && mount_json_exams "$1"
    args+=("$arg" "$mounted")
    shift
  else
    args+=("$arg")
  fi
done

# Without --out, results would stay inside the removed container.
if [[ "$command" == "example" ]] && ! $has_out; then
  die "add --out PATH to choose where the results are written"
fi
if $open_viewer && ! $has_out; then
  die "--viewer needs --out PATH: the viewer opens on that folder"
fi

docker_options+=(--rm --init --shm-size 2g -u "$(id -u):$(id -g)")
[[ -t 0 && -t 1 ]] && docker_options+=(-it)
if [[ -n "${MRI_DOCKER_OPTIONS:-}" ]]; then
  read -r -a extra <<< "$MRI_DOCKER_OPTIONS"
  docker_options+=("${extra[@]}")
fi

if ! $open_viewer && [[ -z "$json_copy_dir" ]]; then
  exec docker run "${docker_options[@]}" ${mounts[@]+"${mounts[@]}"} "$image" "${args[@]}"
fi
[[ -n "$json_copy_dir" ]] && trap 'rm -rf -- "$json_copy_dir"' EXIT

# Analyse; with --viewer, then serve the result folder if the run succeeded.
status=0
docker run "${docker_options[@]}" ${mounts[@]+"${mounts[@]}"} "$image" "${args[@]}" || status=$?
$open_viewer || exit "$status"
[[ $status -eq 0 ]] || die "the run failed (exit $status); the viewer was not started"
[[ -n "$json_copy_dir" ]] && rm -rf -- "$json_copy_dir"
exec bash "${BASH_SOURCE[0]}" viewer "$out_path"
