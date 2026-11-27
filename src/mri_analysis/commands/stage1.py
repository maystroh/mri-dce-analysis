"""Stage 1 for one exam or a whole dataset: acquisition view + laterality, with flags.

    mri-analysis stage1 --input /data/ispy1/MR/example_exam --out out/ispy1_1001
    mri-analysis stage1 --input /data/ispy1/MR          --out out/ispy1 --name ispy1
    mri-analysis stage1 --input ./already_converted     --out out/x --no-convert

DICOM in, CSV out. Conversion, series selection, phase labelling and file naming all come
from `mri_dicom_pipeliner` -- this script does not second-guess any of it. What the
pipeliner cannot resolve is FLAGGED, never silently repaired.

Writes, under --out:
  <name>_stage1.csv     one row per produced volume: the verdicts, the evidence, the flags
  <name>_uncertain.csv  only the rows needing a human -- the worklist
  <name>_summary.json   counts, so a run can be compared with the last one
  <name>_stage1.log     what the pipeliner did, per study
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys, time, traceback, warnings
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mri_analysis.runtime import add_checkout
add_checkout("mri-dicom-pipeliner", "MRI_DICOM_PIPELINER_ROOT")

from mri_analysis.geometry import (axis_geometry, characterise_acquisition,  # noqa: E402
                                   BILATERAL_FOV_MM, ABSTAIN_MARGIN_MM,
                                   MIN_SIDE_COVERAGE_MM, OBLIQUE_MAX_DEG)

# DCE phases plus T2 -- the series the pipeline consumes.
LABEL_REGEX = "pre|post|t2"
# {SeriesInstanceUID}-{ContentDate}-{ContentTime}-{SeriesDescription}-{label}
NAME_RE = re.compile(r"^(?P<uid>[0-9.]+)-(?P<date>\d*)-(?P<time>[\d.]*)-(?P<rest>.+)$")
LABEL_RE = re.compile(r"-(pre|post\d*(?:-post_last)?|post_last|t2|sub)$")


def log(fh, msg):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def find_studies(root):
    """Every directory that directly contains DICOM files, plus its patient label.

    Accepts a dataset root, a patient directory or a single exam directory -- the caller
    should not have to know which level they are pointing at.
    """
    studies = []
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in files):
            # feed the pipeliner the STUDY level (parent of the series directories)
            studies.append(os.path.dirname(dirpath))
    if not studies:                       # exam dir whose files sit one level down only
        return []
    uniq = sorted(set(studies))
    # drop any directory that is an ancestor of another: keep the deepest study dirs
    return [d for d in uniq if not any(o != d and o.startswith(d + os.sep) for o in uniq)]


def convert(root, dest, fh, limit=0, studies=None, nifti_name_format=None):
    """Convert every study under `root`.

    `studies` overrides the discovery. `find_studies` infers the study level from where the
    .dcm files sit, which is right whenever the layout is `study/series/*.dcm` -- but a
    SeriesDescription containing a slash makes TCIA write `study/Ph1/series/*.dcm`, and the
    inferred level is then one phase, converted as if it were a whole exam and labelled
    `pre` for want of anything to order it against. A caller that KNOWS the study level
    passes it; see `duke_supply/build_duke_dataset.py`.
    """
    from mri_dicom_pipeliner.dicom.dicom_pipelining import DicomPipelining
    studies = list(studies) if studies is not None else find_studies(root)
    log(fh, f"found {len(studies)} study directories under {root}")
    if limit and len(studies) > limit:
        studies = studies[:limit]
        log(fh, f"--limit-studies {limit}: converting the first {limit} only")
    ok = failed = 0
    meta_dir = os.path.join(dest, "_series_metadata")
    os.makedirs(meta_dir, exist_ok=True)
    for i, sdir in enumerate(studies):
        t0 = time.time()
        try:
            config = {
                "series_filters": {"generate_nifti": {"label": {"regex": LABEL_REGEX}}},
                "walkers": ["cases_mapper", "dicom2nifti"],
                "src_path": sdir,
                "dest_path": dest,
                # one CSV per study: the pipeliner writes `dicom_series_info.csv` into
                # dest_path, so a shared destination would leave only the last study's rows.
                "metadata_csv_path": os.path.join(meta_dir, f"series_{i:05d}.csv"),
            }
            if nifti_name_format:
                config["nifti_file_name_format"] = nifti_name_format
            DicomPipelining(config).process()
            ok += 1
            log(fh, f"  converted  {os.path.relpath(sdir, root)}  ({time.time()-t0:.0f}s)")
        except Exception as exc:
            failed += 1
            log(fh, f"  PIPELINER FAILED  {os.path.relpath(sdir, root)}: "
                    f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=open(os.devnull, "w"))
    return {"studies": len(studies), "converted": ok, "failed": failed}


def parse_name(basename):
    stem = basename[:-7] if basename.endswith(".nii.gz") else os.path.splitext(basename)[0]
    m = NAME_RE.match(stem)
    if not m:
        return {"uid": "", "description": stem, "label": ""}
    rest = m.group("rest")
    lab = LABEL_RE.search(rest)
    return {"uid": m.group("uid"),
            "description": rest[:lab.start()] if lab else rest,
            "label": lab.group(1) if lab else ""}


def load_layer1(root):
    """The pipeliner's own metadata CSV, keyed by series UID. Empty dict if absent."""
    out = {}
    csvs = (glob.glob(os.path.join(root, "**", "dicom_series_info.csv"), recursive=True)
            + glob.glob(os.path.join(root, "**", "_series_metadata", "*.csv"), recursive=True))
    for csv in csvs:
        try:
            d = pd.read_csv(csv, dtype=str, low_memory=False)
        except Exception:
            continue
        for _, r in d.iterrows():
            out[str(r.get("series_instance_uid"))] = r
    return out


def characterise(nifti_root, fh):
    layer1 = {}
    rows = []
    volumes = ([nifti_root] if os.path.isfile(nifti_root) else
               glob.glob(os.path.join(nifti_root, "**", "*.nii.gz"), recursive=True)
               + glob.glob(os.path.join(nifti_root, "**", "*.nii"), recursive=True))
    if os.path.isfile(nifti_root) and nifti_root.lower().endswith(".csv"):
        from mri_analysis.discovery import exams_from_manifest
        volumes = [phase for exam in exams_from_manifest(nifti_root) for phase in exam.phases]
    elif os.path.isfile(nifti_root) and nifti_root.lower().endswith(".json"):
        from pathlib import Path
        from mri_analysis.artifacts import file_path
        from mri_analysis.dicom_inputs import converted_exams, read_input_json
        exams = converted_exams(nifti_root)
        for exam in exams:
            layer1.update(load_layer1(exam.directory))
        # Use this conversion's series only, including T2 for geometry. Old outputs
        # from earlier runs or unselected exams must not enter the current analysis.
        volumes = list(dict.fromkeys(
            str(file_path(file_path(Path(nifti_root).resolve().parent, record["nifti_directory"]),
                          series["nifti_path"]))
            for record in read_input_json(nifti_root)["cases"]
            for series in record.get("series_conversion", [])
            if series.get("conversion_status") in ("converted", "already_exists")
            and series.get("nifti_path")))
    else:
        layer1 = load_layer1(nifti_root)
    log(fh, f"layer 1 metadata: {len(layer1)} series from the pipeliner's CSV")
    for path in sorted(volumes):
        meta = parse_name(os.path.basename(path))
        g = axis_geometry(path)
        l1 = layer1.get(meta["uid"])

        def l1get(key):
            if l1 is None:
                return ""
            v = str(l1.get(key) or "").strip()
            return "" if v.lower() in ("nan", "unknown", "oblique", "none") else v

        # One shared verdict builder: the fusion, the review flags and `trusted` are
        # `mri_transforms.acquisition`'s mechanism, applied here with THIS project's cuts.
        # `pipeliner_row_found` is a boolean rather than None because stage 1 always has a
        # pipeliner layer to report on -- a missing row is its uncertainty, not its absence.
        verdict = characterise_acquisition(
            g,
            plane_dicom=l1get("acquisition_view"),
            scope_dicom=l1get("breast_coverage"),
            scope_dicom_source=l1get("breast_coverage_source"),
            pipeliner_row_found=l1 is not None,
            conversion_status=l1get("conversion_status"),
            fov_cut_mm=BILATERAL_FOV_MM, abstain_mm=ABSTAIN_MARGIN_MM,
            min_side_mm=MIN_SIDE_COVERAGE_MM, oblique_deg=OBLIQUE_MAX_DEG)
        scope, certain, reason = verdict.scope_nifti, verdict.scope_certain, verdict.reason
        rows.append({
            "path": path, "filename": os.path.basename(path),
            "label": meta["label"], "description": meta["description"],
            "series_uid": meta["uid"],
            # ---- the verdicts ----
            "view": verdict.plane, "laterality": verdict.scope,
            "view_source": verdict.plane_source,
            "laterality_source": verdict.scope_source,
            "view_nifti": g.plane, "laterality_nifti": scope,
            # ---- trust ----
            "trusted": verdict.trusted,
            **verdict.flags,
            # ---- the evidence, so a verdict can be re-derived without re-reading ----
            "fov_lr_mm": g.fov_lr_mm, "min_side_mm": g.min_side_mm,
            "lr_min_mm": g.lr_min_mm, "lr_max_mm": g.lr_max_mm,
            "axcodes": g.axcodes, "n_volumes": g.n_volumes,
            "frame_usable": g.frame_usable, "axes_touching_zero": g.axes_touching_zero,
            "shape": "x".join(map(str, g.shape)),
            "zooms": "x".join(f"{z:.3f}" for z in g.zooms),
            "oblique_deg": g.oblique_deg, "scope_certain": certain,
            # ---- what the DICOM said, as a side note ----
            "dicom_view": verdict.plane_dicom, "dicom_laterality": verdict.scope_dicom,
            "dicom_laterality_source": verdict.scope_dicom_source,
            "dicom_fov_lr_mm": l1get("fov_lr_mm"),
            "reason": reason, "error": g.error or "",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="DICOM root (dataset, patient or exam); NIfTI with --no-convert")
    source.add_argument("--input-json", help="JSON mapping exam directories (DICOM or NIfTI) to metadata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="prefix for the outputs (default: input dir name)")
    ap.add_argument("--nifti-dir", default="", help="where NIfTIs go (default: <out>/nifti)")
    ap.add_argument("--limit-studies", type=int, default=0,
                    help="convert only the first N studies (smoke-test a large dataset)")
    ap.add_argument("--no-convert", action="store_true",
                    help="--input is already converted NIfTI; skip the pipeliner")
    ap.add_argument("--organized-root", help=argparse.SUPPRESS)  # set by `run --organize`
    args = ap.parse_args()
    args.input = args.input_json or args.input
    if args.limit_studies < 0:
        ap.error("--limit-studies must be non-negative")
    json_input = bool(args.input_json)
    if json_input:
        if args.no_convert:
            ap.error("--input-json converts its DICOM exams; use --input with the saved converted manifest for reanalysis")
        from mri_analysis.dicom_inputs import dicom_inputs
        try:
            dicom_inputs(args.input)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    elif os.path.isfile(args.input) and args.input.lower().endswith(".json"):
        from mri_analysis.dicom_inputs import converted_exams
        try:
            if not args.no_convert:
                ap.error("For a DICOM exam mapping, use --input-json instead of --input")
            converted_exams(args.input)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))

    name = args.name or os.path.basename(os.path.abspath(args.input).rstrip(os.sep))
    os.makedirs(args.out, exist_ok=True)
    nifti_dir = args.nifti_dir or (args.input if args.no_convert
                                   else os.path.join(args.out, "nifti"))
    if not args.no_convert:
        os.makedirs(nifti_dir, exist_ok=True)

    with open(os.path.join(args.out, f"{name}_stage1.log"), "w") as fh:
        log(fh, f"stage 1: {args.input}")
        conv = {"skipped": True}
        if not args.no_convert:
            if json_input:
                from mri_analysis.dicom_inputs import convert_json_input
                conv = convert_json_input(args.input, nifti_dir, args.out, name, fh, args.limit_studies,
                                          organized_root=args.organized_root)
            else:
                conv = convert(args.input, nifti_dir, fh, args.limit_studies)
            log(fh, f"conversion: {conv}")

        geometry_input = conv["exam_manifest"] if json_input else nifti_dir
        d = characterise(geometry_input, fh)
        if d.empty:
            log(fh, "NO VOLUMES PRODUCED -- nothing the pipeliner selected could be converted")
            json.dump({"name": name, "conversion": conv, "volumes": 0},
                      open(os.path.join(args.out, f"{name}_summary.json"), "w"), indent=2)
            return

        csv = os.path.join(args.out, f"{name}_stage1.csv")
        d.to_csv(csv, index=False)
        unc = d[~d.trusted]
        unc_csv = os.path.join(args.out, f"{name}_uncertain.csv")
        unc.assign(verdict_after_review="").to_csv(unc_csv, index=False)

        flagcols = [c for c in d.columns if c.startswith("flag_")]
        summary = {
            "name": name, "conversion": conv, "volumes": len(d),
            "trusted": int(d.trusted.sum()), "uncertain": int((~d.trusted).sum()),
            "view": d.view.value_counts().to_dict(),
            "laterality": d.laterality.value_counts().to_dict(),
            "label": d.label.value_counts().to_dict(),
            "flags": {c: int(d[c].sum()) for c in flagcols if d[c].sum()},
        }
        json.dump(summary, open(os.path.join(args.out, f"{name}_summary.json"), "w"), indent=2)

        log(fh, "")
        log(fh, f"volumes    : {len(d)}")
        log(fh, f"trusted    : {summary['trusted']}   uncertain: {summary['uncertain']}")
        log(fh, f"view       : {summary['view']}")
        log(fh, f"laterality : {summary['laterality']}")
        log(fh, f"flags      : {summary['flags']}")
        log(fh, f"\nwrote {csv}")
        log(fh, f"wrote {unc_csv}  ({len(unc)} rows to check)")


if __name__ == "__main__":
    main()
