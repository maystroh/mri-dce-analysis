"""Run landmark analysis and save per-side measurements and per-exam status."""
from __future__ import annotations

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse, csv, json, shlex, sys, traceback, warnings
import pandas as pd

warnings.filterwarnings("ignore")

import mri_analysis                                                          # noqa: E402,F401
from mri_analysis.discovery import Exam, exams_from_manifest, iter_exams      # noqa: E402
from mri_analysis.geometry import axis_geometry, scope_by_fov                 # noqa: E402
from mri_transforms.models import (REFERENCE_MODEL_NAME,                     # noqa: E402
                                   REFERENCE_MODEL_VERSION, REFERENCE_TRACKING_URI)
from mri_analysis.landmarks import (REGION_VOLUME_MAX_MM3, REGION_VOLUME_MIN_MM3,  # noqa: E402
                                    ROI_PLAUSIBILITY_MIN_MM3, judge_entry,
                                    midline_cut_offset_mm, StoredLandmarkRunner,
                                    region_mask_measurements)
from mri_analysis.laterality import (compare_coverage, pipeline_laterality,  # noqa: E402
                                    summarize_comparisons)

DEFAULT_MODEL_NAME = REFERENCE_MODEL_NAME
DEFAULT_MODEL_VERSION = REFERENCE_MODEL_VERSION
DEFAULT_TRACKING_URI = REFERENCE_TRACKING_URI


def log(fh, msg):
    print(msg, flush=True)
    fh.write(str(msg) + "\n")
    fh.flush()


def stage1_bilateral_hint(scope: str):
    """Translate Stage 1 scope without turning `unknown` into unilateral."""
    if scope == "bilateral":
        return True
    if scope == "unilateral":
        return False
    return None


def coverage_audit_fields(coverage: dict, stage1_scope: str) -> dict:
    """Separate geometry's measurement from the landmark execution contract."""
    if not coverage:
        return {
            "coverage_scope": None,
            "coverage_measured_scope": None,
            "coverage_hint_scope": None,
            "landmark_execution": None,
            "coverage_agrees": None,
        }
    measured_scope = (coverage.get("measured_scope")
                      if "measured_scope" in coverage else coverage.get("scope"))
    execution = coverage.get("execution")
    if execution is None:
        execution = "two_halves" if coverage.get("scope") == "bilateral" else "full_width"
    return {
        # Backward-compatible name: the input contract actually used by the model.
        "coverage_scope": coverage.get("scope"),
        # Acquisition Scope as measured by geometry, even when execution abstained from a split.
        "coverage_measured_scope": measured_scope,
        "coverage_hint_scope": coverage.get("hint_scope"),
        "landmark_execution": execution,
        "coverage_agrees": (None if measured_scope is None
                            else measured_scope == stage1_scope),
    }


def resolve_exams(path: str, layout: str, limit: int, exclude=None):
    """Everything to run, as `Exam` records -- a file, an exam directory, or a dataset root.

    The caller should not have to say which of the three they pointed at, for the same reason
    `run_stage1.find_studies` accepts any level.
    """
    if os.path.isfile(path):
        if path.lower().endswith(".csv"):
            return exams_from_manifest(path, limit or None)
        if path.lower().endswith(".json"):
            from mri_analysis.dicom_inputs import converted_exams
            return converted_exams(path, limit)
        if not path.endswith((".nii", ".nii.gz")):
            raise ValueError("--input file must be NIfTI, a CSV exam manifest or a converted DICOM JSON manifest")
        case = os.path.basename(path).removesuffix(".gz").removesuffix(".nii")
        return [Exam(case=case, directory=os.path.dirname(path), pre=path, posts=[])]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    metadata = os.path.join(path, "metadata.csv")
    if os.path.isfile(metadata):
        with open(metadata, newline="", encoding="utf-8-sig") as fh:
            columns = csv.DictReader(fh).fieldnames or []
        if "case" in columns and ("source_path" in columns or "image" in columns):
            return exams_from_manifest(metadata, limit or None)
    return list(iter_exams(path, layout, limit or None, recursive=True, exclude=exclude))


def build_runner(args):
    """The runner, configured exactly as the pipeline configures it.

    `output_subdir=""` writes straight into the per-case directory, matching `run_landmarks.py`.
    `on_landmark_failure="fallback"` makes a failed exam a recorded row rather than an abandoned
    sweep -- a dataset characterisation that stops at the first bad exam characterises nothing.
    """
    try:
        from mri_transforms.landmark.runner import LandmarkRunner
    except ImportError as exc:
        shared_path = getattr(sys.modules.get("mri_transforms"), "__file__", "not found")
        raise RuntimeError(
            "Landmark runtime is unavailable. "
            f"Python: {sys.executable}. Shared library: {shared_path}. "
            f"Install the landmark extra with {shlex.quote(sys.executable)} -m pip install "
            f"'mri-dce-analysis[analysis]'. See docs/INSTALL.md. Import error: {exc}") from exc
    if not (getattr(args, "model_bundle", None) or args.tracking_uri):
        raise ValueError("New inference requires --model-bundle /path/to/bundle or --tracking-uri URL")
    overrides = ({"breast_region": {"class_assignment": args.class_assignment}}
                 if args.class_assignment else {})
    return LandmarkRunner({
        "enabled": True,
        "mlflow_configs": {"tracking_uri": args.tracking_uri,
                           "registered_model_name": args.model_name,
                           "stage": None, "version": args.version,
                           "cache_root": args.cache_root,
                           "local_bundle": getattr(args, "model_bundle", None)},
        "head_overrides": overrides,
        "crop_margin_mm": args.margin_mm,
        "device": args.device,
        "image_interpolator": args.image_interpolator,
        "reuse_existing": not args.no_reuse,
        "output_subdir": "",
        # Forces the Acquisition Scope a per-breast model is fed under. Left unset, the model
        # resolves it itself with `scope_for`; set, it bypasses that measurement entirely. Only
        # ever pass this to answer "what would this exam look like under the other contract".
        "coverage_override": args.coverage,
        # The transient Breast Halves. Off by default -- two full-resolution volumes per exam --
        # but the only thing worth looking at when a fused mask is wrong on one side.
        "keep_halves": args.keep_halves,
        "on_landmark_failure": "fallback",
    })


def exam_context(exam: Exam) -> dict:
    """Preserve arbitrary metadata, including nested values, without overriding routing."""
    return {"case": exam.case,
            **{f"dataset_{key}": (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
               for key, value in exam.metadata.items()},
            "dataset_metadata_json": json.dumps(exam.metadata, ensure_ascii=False, allow_nan=False),
            "dicom_source": exam.dicom_source, "input_error": exam.input_error}


def characterise(exam: Exam, runner, outdir: str, fh, predictions_root=None) -> list:
    """Run one exam and turn its `landmarks.json` entry into one row per side."""
    source = exam.pre or (exam.phases[0] if exam.phases else None)
    if source is None:
        log(fh, f"   [!] {exam.case}: no phase volume found")
        return [{**exam_context(exam), "status": "no_source", "reason": exam.input_error or "no phase volume found"}]

    geom = axis_geometry(source)
    stage1_scope, stage1_margin, stage1_reason, stage1_certain = scope_by_fov(geom)
    pipeline = (pipeline_laterality(source) if not geom.error else {
        "pipeline_scope": None, "pipeline_reason": None,
        "pipeline_source": "infer_bilateral_view", "pipeline_error": geom.error})

    base = {
        **exam_context(exam),
        **pipeline,
        "source_file": os.path.basename(source),
        "source_path": source,
        "n_phases": exam.n_phases,
        # ---- stage 1, carried through so stage 2 can be read without re-reading stage 1 ----
        "plane": geom.plane,
        "stage1_scope": stage1_scope,
        "stage1_certain": stage1_certain,
        "stage1_margin_mm": stage1_margin,
        "stage1_reason": stage1_reason,
        "fov_lr_mm": geom.fov_lr_mm,
        "min_side_mm": geom.min_side_mm,
        "frame_usable": geom.frame_usable,
        "oblique_deg": geom.oblique_deg,
        # ---- where the model's cut falls relative to the patient midline ----
        # Filled in below, once the Coverage Verdict says whether a cut was made at all.
        "midline_cut_offset_mm": None,
        "midline_cut_reason": "",
        "geometry_error": geom.error or "",
    }

    if geom.error:
        return [dict(base, status="read_error", reason=geom.error)]

    case_out = os.path.join(predictions_root or os.path.join(outdir, "predictions"), exam.case)
    try:
        result = runner.run(source, case_out,
                            is_bilateral=stage1_bilateral_hint(pipeline["pipeline_scope"]))
    except Exception as exc:                          # on_failure=fallback covers most of these
        log(fh, f"   [!] {exam.case}: landmark inference raised: {exc}")
        log(fh, traceback.format_exc())
        return [dict(base, status="inference_raised", reason=str(exc))]
    if result is None:
        log(fh, f"   [!] {exam.case}: landmark inference failed")
        return [dict(base, status="inference_failed", reason="landmark inference returned None")]

    entry = result["entry"]
    model = result.get("model") or {}
    # Null for a whole-exam model, and that is the honest value: such a model makes no coverage
    # claim and was not fed under one. Never defaulted to the stage-1 verdict, which would make
    # the two models' rows look alike when they were produced under different contracts.
    coverage = result.get("coverage") or entry.get("coverage") or {}
    audit = coverage_audit_fields(coverage, stage1_scope)
    base.update({
        "model_name": model.get("name"), "model_version": model.get("version"),
        "model_run_id": model.get("run_id"),
        "fusion_version": model.get("fusion_version"),
        "reused": bool(result.get("reused")),
        **audit,
        **compare_coverage(pipeline["pipeline_scope"], coverage),
        "coverage_certain": coverage.get("certain"),
        "coverage_margin_mm": coverage.get("margin_mm"),
        "coverage_source": coverage.get("source"),
        "coverage_reason": coverage.get("reason"),
        "n_sides_reported": len(entry.get("sides") or {}),
    })

    # The cut offset only exists when there IS a cut. `split_image` gives a unilateral exam the
    # single span `(0, width)` and never divides it, so an offset reported there would describe
    # a cut that was not made -- and it would be LARGE and alarming, because a unilateral
    # sagittal volume sits ~80 mm off the midline by construction. Reported against the verdict
    # the exam was actually FED under. A whole-exam model makes no cut, regardless of scope.
    if audit["landmark_execution"] == "two_halves":
        cut_offset, cut_reason = midline_cut_offset_mm(source)
        base.update(midline_cut_offset_mm=cut_offset, midline_cut_reason=cut_reason)
    else:
        base.update(midline_cut_offset_mm=None,
                    midline_cut_reason="not cut -- one full-width pass")

    measurements, measurement_error = {}, ""
    if entry.get("breast_region_file"):
        try:
            measurements = region_mask_measurements(
                os.path.join(case_out, entry["breast_region_file"]), entry, source)
        except Exception as exc:
            measurement_error = str(exc)
    base["mask_measurement_error"] = measurement_error

    verdicts = judge_entry(entry)
    if not verdicts:
        return [dict(base, status="no_sides", reason="entry reported no sides")]
    rows = []
    for verdict in verdicts:
        nipples = entry["sides"][verdict.side].get("nipples") or []
        selected = next((n for n in nipples if n.get("biggest") and not n.get("filtered_out")), {})
        row = dict(base, status="ok", **verdict.as_row(),
                   **measurements.get(verdict.side, {}),
                   nipple_missing=verdict.n_nipples == 0,
                   nipple_multiple=verdict.n_nipples > 1,
                   selected_nipple_volume_mm3=selected.get("volume_mm3"))
        for axis in ("x", "y", "z"):
            row[f"selected_nipple_{axis}_mm"] = (selected.get("position_mm") or {}).get(axis)
        # A full-width fallback on measured-bilateral/unknown coverage may contain two breasts.
        # The per-breast band is then not evidence for a per-breast consumption decision.
        ambiguous = (verdict.side == "unilateral"
                     and (coverage.get("source") == "abstention"
                          or (coverage and audit["coverage_measured_scope"] == "bilateral")
                          or (coverage.get("source") == "override" and stage1_scope != "unilateral")))
        row["region_band_applicable"] = not ambiguous
        if ambiguous:
            row.update(fit_to_consume=False,
                       unfit_reason="full-width region has no verified per-breast interpretation; "
                                    + row["unfit_reason"])
        if measurement_error:
            row.update(fit_to_consume=False,
                       unfit_reason=(row["unfit_reason"] + "; " if row["unfit_reason"] else "")
                                    + "mask measurement failed: " + measurement_error)
        row["needs_review"] = bool(exam.input_error or not row["fit_to_consume"] or measurement_error
                                    or row["nipple_missing"] or row["nipple_multiple"]
                                    or row.get("region_touches_lr_border")
                                    or row.get("region_components", 0) > 1
                                    or coverage.get("certain") is False
                                    or row["pipeline_measured_agreement"] is False
                                    or row["pipeline_execution_agreement"] is False)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="NIfTI file, exam directory, nested dataset root, or CSV exam manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", action="store_true", help="open the packaged viewer after analysis")
    ap.add_argument("--motion", action="store_true", help="measure inter-phase motion with the shared production library")
    ap.add_argument("--viewer-port", type=int, default=8060)
    ap.add_argument("--name", default="", help="prefix for the outputs (default: input dir name)")
    ap.add_argument("--layout", default="auto", choices=["auto", "mama_mia", "odelia", "generic"])
    ap.add_argument("--limit", type=int, default=0, help="run only the first N exams")
    ap.add_argument("--model-bundle", help="local bundle containing model/ and config/; no registry required")
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--version", default=DEFAULT_MODEL_VERSION)
    ap.add_argument("--cache-root", default=None, help="model cache (default: <out>/models_cache)")
    ap.add_argument("--margin-mm", type=float, default=5.0)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: whatever is available)")
    ap.add_argument("--class-assignment", default=None,
                    help="override breast_region class_assignment (argmax|threshold); only "
                         "needed for older whole-exam versions whose config predates it")
    ap.add_argument("--image-interpolator", default=None,
                    choices=["bspline", "linear", "nearest"],
                    help="override the offline-resample interpolator; only needed when the "
                         "registered config's offline_ops is null")
    ap.add_argument("--coverage", default=None, choices=["bilateral", "unilateral"],
                    help="FORCE the Acquisition Scope a per-breast model is fed under, "
                         "bypassing scope_for. Ignored by a whole-exam model. Use it to ask "
                         "what an exam looks like under the other contract, not to fix data")
    ap.add_argument("--keep-halves", action="store_true",
                    help="keep the transient Breast Half NIfTIs for inspection")
    ap.add_argument("--analyse-only", action="store_true",
                    help="analyse stored landmarks.json and masks without loading a model")
    ap.add_argument("--predictions-root", default=None,
                    help="per-case landmark directories (default: <out>/predictions); "
                         "can point to existing predictions with --analyse-only")
    ap.add_argument("--no-reuse", action="store_true",
                    help="ignore masks already on disk and re-run the model. Rarely needed: the "
                         "cache is keyed on the source volume AND the full inference contract "
                         "(model name, version, run id, coverage verdict), so changing any of "
                         "them already misses. Use it when you suspect the cache itself")
    args = ap.parse_args()
    if args.model_bundle:
        args.model_bundle = os.path.abspath(os.path.expanduser(args.model_bundle))
    args.cache_root = args.cache_root or os.path.join(args.out, "models_cache")
    if args.limit < 0:
        ap.error("--limit must be non-negative")
    if args.analyse_only and (args.coverage or args.no_reuse or args.keep_halves):
        ap.error("--analyse-only reads the recorded contract; --coverage, --no-reuse and "
                 "--keep-halves require inference")

    name = args.name or os.path.basename(os.path.abspath(args.input).rstrip(os.sep))
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, f"{name}_stage2.log"), "w") as fh:
        log(fh, f"stage 2: {args.input}")
        log(fh, "mode   : stored outputs" if args.analyse_only
            else f"model  : {args.model_name} v{args.version}")
        log(fh, f"bands  : region {REGION_VOLUME_MIN_MM3:.0f}-{REGION_VOLUME_MAX_MM3:.0f} mm3, "
                f"ROI gate {ROI_PLAUSIBILITY_MIN_MM3:.0f} mm3")
        if args.coverage:
            log(fh, f"   [!] coverage FORCED to {args.coverage!r}; scope_for is bypassed")

        exams = resolve_exams(args.input, args.layout, args.limit,
                              exclude=[args.out, args.predictions_root] if args.predictions_root
                              else [args.out])
        log(fh, f"exams  : {len(exams)}")
        if not exams:
            log(fh, "NO EXAMS FOUND -- check --input and --layout")
            json.dump({"name": name, "exams": 0},
                      open(os.path.join(args.out, f"{name}_stage2_summary.json"), "w"), indent=2)
            from mri_analysis.artifacts import write_viewer_manifest
            write_viewer_manifest(args.out, name, [], args.input, args.predictions_root, args.motion)
            if args.motion:
                from mri_analysis.motion import run_motion
                run_motion([], args.out, name, args.predictions_root)
            return

        runner = StoredLandmarkRunner() if args.analyse_only or not any(e.phases for e in exams) else build_runner(args)
        rows = []
        for i, exam in enumerate(exams, 1):
            log(fh, f"[{i}/{len(exams)}] {exam.case}")
            try:
                rows.extend(characterise(exam, runner, args.out, fh, args.predictions_root))
            except Exception as exc:
                log(fh, f"   [!] {exam.case}: analysis raised: {exc}")
                log(fh, traceback.format_exc())
                rows.append({**exam_context(exam), "source_path": exam.pre,
                             "status": "analysis_raised", "reason": str(exc)})

        d = pd.DataFrame(rows)
        csv = os.path.join(args.out, f"{name}_stage2.csv")
        d.to_csv(csv, index=False)

        ok = d[d.status == "ok"] if "status" in d else d.iloc[0:0]
        unfit = ok[ok.fit_to_consume.eq(False)] if "fit_to_consume" in ok else ok.iloc[0:0]
        unfit_csv = os.path.join(args.out, f"{name}_unfit.csv")
        unfit.assign(verdict_after_review="").to_csv(unfit_csv, index=False)

        # All failures remain in the exam denominator. A per-side summary otherwise makes a
        # model that reports nothing look better by dropping its failed exams.
        side_columns = [c for c in d if c.startswith(("region_", "selected_nipple_"))]
        side_columns += ["side", "n_nipples", "roi_big_enough", "fit_to_consume", "unfit_reason",
                         "nipple_missing", "nipple_multiple", "largest_component_mm3", "needs_review"]
        exam_frame = d.drop_duplicates("case").drop(columns=side_columns, errors="ignore")
        if len(ok):
            side_totals = ok.groupby("case").agg(
                sides_reported=("side", "size"), sides_fit_to_consume=("fit_to_consume", "sum"),
                sides_needing_review=("needs_review", "sum"),
                sides_without_nipple=("nipple_missing", "sum"))
            exam_frame = exam_frame.join(side_totals, on="case")
        exam_frame.to_csv(os.path.join(args.out, f"{name}_exams.csv"), index=False)
        review = d[(d.status != "ok") | d.get("needs_review", pd.Series(False, index=d.index)).eq(True)]
        review.assign(verdict_after_review="").to_csv(
            os.path.join(args.out, f"{name}_review.csv"), index=False)

        def counts(frame, col):
            if col not in frame:
                return {}
            return {str(k): int(v) for k, v in
                    frame[col].fillna("unavailable").value_counts().items()}

        exams_ok = ok.drop_duplicates("case") if len(ok) else ok
        summary = {
            "name": name,
            "mode": "analyse_only" if args.analyse_only else "inference",
            "model": (None if args.analyse_only else {
                "name": args.model_name, "version": args.version,
                "coverage_override": args.coverage}),
            "observed_models": (exam_frame[["model_name", "model_version", "model_run_id", "fusion_version"]]
                                .loc[exam_frame.model_name.notna()]
                                .drop_duplicates().astype(object).where(lambda x: x.notna(), None)
                                .to_dict("records") if "model_name" in exam_frame else []),
            "cuts": {"region_volume_mm3": [REGION_VOLUME_MIN_MM3, REGION_VOLUME_MAX_MM3],
                     "roi_plausibility_mm3": ROI_PLAUSIBILITY_MIN_MM3},
            "exams": len(exams), "sides": len(ok),
            "exams_ok": len(exams_ok), "exams_failed": len(exams) - len(exams_ok),
            "review_rows": len(review),
            "laterality_comparison": summarize_comparisons(rows),
            "status": counts(d, "status"),
            "exam_status": counts(exam_frame, "status"),
            "plane": counts(exams_ok, "plane"),
            # How many exams the model actually SPLIT, and how many it abstained on. The
            # headline number of a per-breast run: an abstention means one full-width pass.
            "coverage_scope": counts(exams_ok, "coverage_scope"),
            "coverage_measured_scope": counts(exams_ok, "coverage_measured_scope"),
            "landmark_execution": counts(exams_ok, "landmark_execution"),
            "coverage_certain": {str(k): int(v) for k, v in
                                 counts(exams_ok, "coverage_certain").items()},
            "coverage_disagrees_with_stage1": (
                int((exams_ok.coverage_agrees == False).sum())           # noqa: E712
                if "coverage_agrees" in exams_ok else 0),
            "region_plausibility": counts(ok, "region_plausibility"),
            "fit_to_consume": {str(k): int(v) for k, v in counts(ok, "fit_to_consume").items()},
            "nipple_missing_sides": int(ok.nipple_missing.sum()) if "nipple_missing" in ok else 0,
            "nipple_multiple_sides": int(ok.nipple_multiple.sum()) if "nipple_multiple" in ok else 0,
            "region_border_contact_sides": (int(ok.region_touches_lr_border.sum())
                                            if "region_touches_lr_border" in ok else 0),
            "strata": (ok.groupby(["plane", "stage1_scope", "side"], dropna=False)
                       .agg(sides=("case", "size"), fit_to_consume=("fit_to_consume", "sum"),
                            nipple_missing=("nipple_missing", "sum"))
                       .reset_index().to_dict("records") if len(ok) else []),
            # |offset| over the exams that were actually SPLIT. Empty on a purely unilateral
            # dataset, which is correct rather than missing: nothing was cut there.
            "midline_cut_offset_mm": (
                {"exams_split": int(exams_ok.midline_cut_offset_mm.notna().sum()),
                 "median": round(float(exams_ok.midline_cut_offset_mm.abs().median()), 2),
                 "p90": round(float(exams_ok.midline_cut_offset_mm.abs().quantile(0.9)), 2),
                 "max": round(float(exams_ok.midline_cut_offset_mm.abs().max()), 2)}
                if len(exams_ok) and exams_ok.midline_cut_offset_mm.notna().any()
                else {"exams_split": 0}),
        }
        with open(os.path.join(args.out, f"{name}_stage2_summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=2, allow_nan=False)

        log(fh, "")
        log(fh, f"exams      : {summary['exams']}   sides: {summary['sides']}")
        log(fh, f"status     : {summary['status']}")
        log(fh, f"coverage   : measured={summary['coverage_measured_scope']}  "
                f"execution={summary['landmark_execution']}  "
                f"certain={summary['coverage_certain']}")
        log(fh, f"disagrees  : {summary['coverage_disagrees_with_stage1']} exams differ from stage 1")
        comparison = summary["laterality_comparison"]
        log(fh, f"pipeline   : {comparison['measured_scope_differences']['count']} measurement / "
                f"{comparison['execution_differences']['count']} execution differences "
                f"over {comparison['comparable_exams']} comparable exams")
        log(fh, f"plausible  : {summary['region_plausibility']}")
        log(fh, f"cut offset : {summary['midline_cut_offset_mm']}")
        log(fh, f"\nwrote {csv}")
        log(fh, f"wrote {unfit_csv}  ({len(unfit)} sides to check)")

    from mri_analysis.artifacts import write_viewer_manifest
    write_viewer_manifest(args.out, name, exams, args.input, args.predictions_root, args.motion)
    if args.motion:
        from mri_analysis.motion import run_motion
        run_motion(exams, args.out, name, args.predictions_root)
    log_command = f'mri-analysis viewer "{os.path.abspath(args.out)}" --name "{name}"'
    print(f"\nExplore this run: {log_command}", flush=True)
    if args.view:
        from mri_analysis.viewer import serve
        serve(args.out, name=name, port=args.viewer_port, open_browser=True)


if __name__ == "__main__":
    main()
