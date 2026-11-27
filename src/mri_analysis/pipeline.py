"""Run geometry, landmarks, optional annotation scoring and the packaged viewer."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mri_analysis.commands.stage2 import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION


def annotation_package(path: Path) -> tuple[bool, str]:
    """Detect supported unilateral labels without mistaking unrelated metadata for masks."""
    metadata = path / "metadata.csv"
    if not metadata.is_file():
        return False, "no labelled metadata package"
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not {"case", "breast_region", "nipples"}.issubset(columns) or not columns.intersection({"image", "source_path"}):
            return False, "metadata does not specify reference images and annotation masks"
        rows = list(reader)
    if not rows:
        return False, "annotation metadata is empty"
    if "laterality" in columns and any(row["laterality"].strip().lower() != "unilateral" for row in rows):
        return False, "annotated nipple localisation currently supports unilateral packages"
    return True, "unilateral annotation masks available"


def looks_like_dicom(root: Path) -> bool:
    """True when a folder holds `.dcm` files and no NIfTI images.

    A folder with both stays NIfTI: converted copies often sit next to their source.
    """
    if not root.is_dir():
        return False
    found_dicom = False
    for _, _, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if lower.endswith((".nii", ".nii.gz")):
                return False
            if lower.endswith(".dcm"):
                found_dicom = True
    return found_dicom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="NIfTI dataset, package, CSV manifest, single image or DICOM root")
    source.add_argument("--input-json", help="JSON mapping exam directories (DICOM or NIfTI) to metadata")
    parser.add_argument("--out", required=True, help="one folder for reports, predictions, cache and logs")
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--dicom", action="store_true", help="convert raw DICOM through stage 1 first; "
                        "detected automatically when --input holds .dcm files and no NIfTI")
    parser.add_argument("--annotations", help="labelled package to score; default: detect labels in --input")
    parser.add_argument("--skip-annotation-evaluation", action="store_true")
    parser.add_argument("--analyse-only", action="store_true", help="reuse stored predictions without model inference")
    parser.add_argument("--predictions-root", help="existing predictions directory for --analyse-only")
    parser.add_argument("--limit", type=int, default=0, help="limit landmark and annotation evaluation cases")
    parser.add_argument("--limit-studies", type=int, default=0, help="limit DICOM conversion studies")
    parser.add_argument("--layout", default="auto", choices=["auto", "mama_mia", "odelia", "generic"])
    # Bound to stage 2's defaults, which are the shared library's registered reference values.
    # A wrapper repeating the literals is how the two came to disagree about the current model.
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--model-bundle", help="local landmark model bundle (model/ and config/)")
    parser.add_argument("--tracking-uri", help="your MLflow model tracking server")
    parser.add_argument("--device", help="cuda or cpu")
    parser.add_argument("--motion", action="store_true", help="measure inter-phase motion after landmarks")
    parser.add_argument("--no-viewer", action="store_true", help="finish after generating artifacts")
    parser.add_argument("--organize", action="store_true",
                        help="arrange an untidy --input folder into one exam folder each (DICOM and "
                             "NIfTI); stops before analysis if any exam cannot be formed")
    parser.add_argument("--organize-check", action="store_true",
                        help="only list the exams --organize would form, or why it cannot, then stop")
    parser.add_argument("--viewer-port", type=int, default=8060)
    args = parser.parse_args()
    if args.model_bundle:
        args.model_bundle = str(Path(args.model_bundle).expanduser().resolve())
    organize = args.organize or args.organize_check
    if organize and args.input_json:
        parser.error("--organize works with --input; an --input-json file already lists each exam folder")
    if organize and (args.dicom or args.analyse_only):
        parser.error("--organize finds DICOM and NIfTI exams itself; omit --dicom and --analyse-only")
    args.input = args.input_json or args.input
    if args.predictions_root and not args.analyse_only:
        parser.error("--predictions-root is for --analyse-only")
    if args.annotations and args.skip_annotation_evaluation:
        parser.error("choose --annotations or --skip-annotation-evaluation")
    if args.limit < 0 or args.limit_studies < 0:
        parser.error("limits must be non-negative")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a filename prefix without directory separators")
    dataset, output = Path(args.input).expanduser().resolve(), Path(args.out).expanduser().resolve()
    try:
        input_exists = dataset.exists()
    except OSError as exc:
        parser.error(f"Cannot access input {dataset}: {exc}")
    if not input_exists:
        parser.error(f"Input does not exist: {dataset}")
    organized_root = None
    if organize:
        from mri_analysis.organize import ORGANIZED_DIR, build, describe, plan_input
        if not dataset.is_dir():
            parser.error(f"--organize needs a folder as --input: {dataset}")
        plan = plan_input(dataset, skip=[output])
        output.mkdir(parents=True, exist_ok=True)
        report = output / f"{args.name}_organize.json"
        report.write_text(json.dumps(plan.report(), indent=2, ensure_ascii=False) + "\n")
        print(describe(plan), flush=True)
        print(f"Organize report: {report}", flush=True)
        if plan.errors:
            print("Organize failed: fix or remove the exams above, or arrange them as in "
                  "docs/INPUT_DATASETS.md. Nothing was analysed.", file=sys.stderr)
            return 1
        if args.organize_check:
            return 0
        args.input_json = str(build(plan, output, args.name))
        organized_root = output / ORGANIZED_DIR
    json_dicom = bool(args.input_json)
    if json_dicom:
        from mri_analysis.dicom_inputs import dicom_inputs
        try:
            dicom_inputs(Path(args.input_json).expanduser().resolve())
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif dataset.is_file() and dataset.suffix.lower() == ".json":
        from mri_analysis.dicom_inputs import converted_exams
        try:
            converted_exams(dataset)
        except (OSError, ValueError) as exc:
            parser.error(f"{exc}. For a DICOM exam mapping, use --input-json instead of --input")
        if args.dicom:
            parser.error("This JSON contains converted NIfTI paths; omit --dicom")
    args.dicom = args.dicom or json_dicom
    if not args.dicom and not args.analyse_only and looks_like_dicom(dataset):
        print(f"Detected DICOM input in {dataset}; converting it (as with --dicom).", flush=True)
        args.dicom = True
    if args.dicom and args.analyse_only:
        parser.error("--analyse-only needs converted NIfTI input or the saved *_dicom_exams.json, not DICOM source directories")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"{args.name}_pipeline.json"
    summary = {"name": args.name, "input": str(dataset), "output": str(output),
               "started_at": datetime.now(timezone.utc).isoformat(), "steps": [],
               "motion": "requested" if args.motion else "not requested"}
    if organized_root:
        summary["organize"] = {"report": f"{args.name}_organize.json", "exams_json": Path(args.input_json).name}

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    def step(command: str, options: list[str]) -> int:
        argv = [sys.executable, "-m", "mri_analysis", command, *options]
        console = output / f"{args.name}_{command}_console.log"
        print(f"Running {command}; console log: {console}", flush=True)
        environment = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = package_root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        with console.open("w") as log:
            result = subprocess.run(argv, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment, check=False)
        summary["steps"].append({"command": command, "argv": argv, "returncode": result.returncode,
                                  "console_log": console.name})
        save()
        return result.returncode

    geometry_input = Path(args.input_json).expanduser().resolve() if json_dicom else dataset
    if not args.dicom and dataset.is_dir():
        if (dataset / "metadata.csv").is_file():
            with (dataset / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
                columns = set(csv.DictReader(handle).fieldnames or [])
            if "case" in columns and columns.intersection({"image", "source_path"}):
                geometry_input = dataset / "metadata.csv"
        elif (dataset / "images").is_dir():
            geometry_input = dataset / "images"
    geometry_options = ["--input-json" if json_dicom else "--input", str(geometry_input),
                        "--out", str(output), "--name", args.name]
    geometry_options += (["--limit-studies", str(args.limit_studies)] if args.dicom else ["--no-convert"])
    if organized_root:
        geometry_options += ["--organized-root", str(organized_root)]
    stage1_status = step("stage1", geometry_options)
    if organized_root:
        # Each exam's folder goes as soon as it is done; this removes any left (crash, --limit-studies).
        import shutil
        shutil.rmtree(organized_root, ignore_errors=True)
    if stage1_status:
        print("Geometry command failed; inspect its console log.", file=sys.stderr)
        return 1
    landmark_input = output / "nifti" if args.dicom else dataset
    if json_dicom:
        landmark_input = output / f"{args.name}_dicom_exams.json"
    landmark_options = ["--input", str(landmark_input), "--out", str(output), "--name", args.name,
                        "--layout", args.layout, "--limit", str(args.limit),
                        "--model-name", args.model_name, "--version", args.version,
                        "--cache-root", str(output / "models_cache")]
    if args.analyse_only:
        landmark_options.append("--analyse-only")
    if args.motion:
        landmark_options.append("--motion")
    if args.predictions_root:
        landmark_options += ["--predictions-root", str(Path(args.predictions_root).expanduser().resolve())]
    for flag in ("tracking_uri", "device", "model_bundle"):
        if getattr(args, flag):
            landmark_options += ["--" + flag.replace("_", "-"), getattr(args, flag)]
    if step("stage2", landmark_options):
        print("Landmark command failed; inspect its console log.", file=sys.stderr)
        return 1
    evaluation = Path(args.annotations).expanduser().resolve() if args.annotations else dataset
    available, reason = annotation_package(evaluation)
    incomplete = False
    if not args.skip_annotation_evaluation and (available or args.annotations):
        prediction_root = Path(args.predictions_root).expanduser().resolve() if args.predictions_root else output / "predictions"
        incomplete = bool(step("evaluate", ["--dataset", str(evaluation), "--predictions-root", str(prediction_root),
                                           "--out", str(output), "--name", args.name, "--limit", str(args.limit)]))
        summary["annotation_evaluation"] = "incomplete" if incomplete else "complete"
    else:
        summary["annotation_evaluation"] = "skipped by request" if args.skip_annotation_evaluation else reason
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    landmark_summary = json.loads((output / f"{args.name}_stage2_summary.json").read_text())
    summary["exams_failed"] = landmark_summary.get("exams_failed", 0)
    conversion_failed = 0
    if args.dicom:
        geometry_summary = json.loads((output / f"{args.name}_summary.json").read_text())
        summary["conversion"] = geometry_summary.get("conversion", {})
        conversion_failed = summary["conversion"].get("failed", 0)
    motion_failed = False
    if args.motion:
        motion_summary = json.loads((output / f"{args.name}_motion_summary.json").read_text())
        summary["motion"] = motion_summary
        motion_failed = bool(motion_summary["exams_failed"])
    summary["status"] = "complete_with_failures" if incomplete or summary["exams_failed"] or motion_failed or conversion_failed else "complete"
    save()
    print(f"Analysis artifacts: {output}\nReopen: mri-analysis viewer \"{output}\" --open", flush=True)
    print(f"Annotation evaluation: {summary['annotation_evaluation']}", flush=True)
    if not args.no_viewer:
        from mri_analysis.viewer import serve
        serve(str(output), args.name, port=args.viewer_port, open_browser=True)
    return 1 if incomplete or summary["exams_failed"] or motion_failed or conversion_failed else 0
