# Running an analysis

[Install](INSTALL.md) the package and the model, and prepare a compatible
[input dataset](INPUT_DATASETS.md).

## Full NIfTI workflow

```bash
mri-dce-analysis run --input /data/exams --out out/study --name study \
  --model-bundle /path/to/model --motion --no-viewer
mri-dce-analysis viewer out/study --open
```

The workflow runs geometry and landmarks, then motion when requested. Available
annotations can also be scored. Add `--device cpu` or `--device cuda` to choose the
inference device. Use `--limit N` for a smaller landmark run.

## DICOM workflow

```bash
mri-dce-analysis run --input /data/dicoms --out out/study --name study \
  --model-bundle /path/to/model --no-viewer
```

For an explicit JSON exam mapping, use `--input-json` instead of `--input`:

```bash
mri-dce-analysis run --input-json /data/exams.json --out out/study --name study \
  --model-bundle /path/to/model --motion --no-viewer
```

JSON input enables conversion automatically. `--limit-studies N` converts the first N
JSON entries; `--limit N` limits landmark analysis. Inspect conversion status and
selected phases.

## Run the stages separately

For NIfTI input:

```bash
mri-dce-analysis stage1 --input /data/exams --out out/study --name study --no-convert
mri-dce-analysis stage2 --input /data/exams --out out/study --name study \
  --model-bundle /path/to/model --motion
```

For JSON-selected DICOM exams:

```bash
mri-dce-analysis stage1 --input-json /data/exams.json --out out/study --name study
mri-dce-analysis stage2 --input out/study/study_dicom_exams.json --out out/study \
  --name study --model-bundle /path/to/model --motion
```

Stage 2 reads the generated converted manifest; the original mapping cannot be passed
to Stage 2. For a raw DICOM directory converted
without a JSON mapping, its converted input is `out/study/nifti/`.

## Reanalyse existing predictions

```bash
mri-dce-analysis run --input /data/exams --out out/reanalysis --name study \
  --analyse-only --predictions-root /data/saved_predictions --no-viewer
```

Saved predictions must match the source image identity. For JSON-selected exams,
reuse the generated manifest as `--input` to retain case IDs and metadata.
See [saved-prediction checks](STAGE2_DATASET_ANALYSIS.md#existing-artifacts).

## Score available annotations

The full workflow detects supported unilateral annotation packages. Use
`--annotations /data/labels_package` to supply a separate package, or
`--skip-annotation-evaluation` to disable automatic scoring.

To evaluate saved predictions directly:

```bash
mri-dce-analysis evaluate --dataset /data/labels_package \
  --predictions-root out/study/predictions --out out/study --name study
```

See [annotation package requirements](INPUT_DATASETS.md#annotation-packages).

## What the output folder contains

| Artifact | Contents |
| --- | --- |
| `<name>_stage1.csv`, `<name>_uncertain.csv` | Geometry measurements and uncertainty worklist |
| `<name>_stage2.csv`, `<name>_exams.csv` | Landmark QC and per-exam status |
| `<name>_unfit.csv`, `<name>_review.csv` | Landmark review worklists |
| `<name>_stage2_summary.json` | Aggregate landmark results |
| `predictions/<case>/` | Region/nipple masks and `landmarks.json` |
| `<name>_motion.csv`, `<name>_motion_pairs.csv` | Per-region and per-phase motion measurements |
| `<name>_motion.json`, `<name>_motion_summary.json` | Motion evidence and summary |
| `<name>_viewer.json` | Image, mask and metadata references for review |
| `<name>_viewer_reviews.json`, `<name>_viewer_reviews.csv` | Reviewer decisions and notes from the viewer (written only after a review) |
| `nifti/`, `<name>_dicom_exams.json` | Converted images and JSON-selected exam manifest |
| `<name>_pipeline.json`, log files | Execution status, timing and errors |

[Analysis methods](ANALYSIS_WORKFLOW.md) explain the measurements and review flags.

## Exit status

The workflow retains failed cases in its reports and returns a nonzero exit status
for execution failures, after saving the artifacts and opening the viewer (immediately
with `--no-viewer`). Single-phase motion skips and anatomical review flags alone do not
make a run fail. Use `--no-viewer` for unattended execution.

Run `mri-dce-analysis COMMAND --help` for each command's complete option list.
