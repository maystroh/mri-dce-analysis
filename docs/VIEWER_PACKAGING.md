# Reviewing and sharing results

Open saved analysis results to inspect images, masks, measurements and review flags.
For new analyses, see [running an analysis](RUNNING_ANALYSIS.md).

## Open a saved run

```bash
mri-dce-analysis viewer out/my_analysis --open
```

The default address is **http://127.0.0.1:8060/**. `mri-viewer` is also an installed
alias. Stop the viewer with Ctrl+C. Restart it after adding or recalculating artifacts.
Stage-1-only folders can be opened through their saved CSVs.

Select a named run or a different port:

```bash
mri-dce-analysis viewer out/my_analysis --name study --port 8061 --open
```

With Docker (`mri-docker.sh viewer`), the viewer is reachable at
`http://<server address>:8060` from other computers; see [Viewer on a server](DOCKER.md#viewer-on-a-server).
Without Docker, the viewer listens on this machine only. Add `--host 0.0.0.0` to open it
to the network, or run the viewer on the server and forward its port from your local machine:

```bash
ssh -L 8060:127.0.0.1:8060 user@server
```

Then open http://127.0.0.1:8060/ in your local browser. `--open` attempts to open a
browser on the machine running the viewer; it is optional for a server.

## Review controls

Use the searchable case queue to select an exam. J/K move to the next/previous case.
The viewer shows one view at a time: **Axial** by default. Pick **Sagittal** or **Coronal**
in the **View** list to switch. The choice stays when you change case, until the page is
reloaded.

Click a slice to move the other planes. The toolbar controls overlays, structures,
opacity and intensity window. Nipple and disagreement buttons jump to findings.
Predictions are orange, annotations cyan, and overlap green. Error maps require both
prediction and annotation masks. Case evidence preserves QC, metadata, warnings and scores;
the viewer does not rerun inference or change analysis artifacts.

### Motion graphs and optional table

When requested during analysis, saved motion appears as interactive displacement and
rotation graphs. Mean and 95th-percentile displacement share a millimetre scale; rotation
uses degrees. Breast regions within an exam use matching graph scales. Click a point or
phase for exact values. Missing measurements remain gaps.

Expand **Show detailed measurements (table)** for filenames, measurements and registration
stop reasons. ROI warnings and skipped/incomplete status remain visible. The image panes
show the reference MRI. See [motion measurement definitions](ANALYSIS_WORKFLOW.md#motion-inside-the-breast-regions).
Runs that omit `--motion` hide motion results, even if older files remain in the folder.

### Review notes and artifacts

The **Review** bar stays at the bottom of the screen. For each case:

1. Check the overlays.
2. Pick **All masks**: `Accurate`, `Incorrect` or `Needs review` (keys `1`, `2`, `3`).
   Use this when the breast region and the nipple have the same quality.
3. If one mask differs, change only **Breast region** or **Nipple**. *Same* uses the
   **All masks** decision.
4. Add a note if you want.
5. Click **Save & next** (or press `Enter`). The viewer saves the review and opens the
   next case that is not reviewed yet.

The case list shows reviewed cases in color: green when both masks are `Accurate`,
red when a mask is `Incorrect`, orange when a mask needs review. **Not reviewed yet
only** hides the done cases, and the bar under it shows the progress. **Clear**
removes the saved review of the open case.

The viewer saves to `<name>_viewer_reviews.json` and `<name>_viewer_reviews.csv` in the
results folder and loads them again when it reopens. If the folder is read-only,
reviews stay in this browser only.

**Export reviews** downloads one zip file, `<results folder>_reviews_<date-time>.zip`,
with the saved `<name>_viewer_reviews.csv` and `<name>_viewer_reviews.json` of every run
in the folder and `mask_reviews.json`. `mask_reviews.json` has every review of every
run, also those kept only in this browser, one entry per case:

```json
{"run": "duke", "case": "Breast_MRI_237/19900101", "breast_region": "Accurate",
 "nipple": "Incorrect", "note": "nipple 5 mm too lateral", "reviewed_at": "2026-11-17T11:13:06+00:00"}
```

Reviews are never written into MRI volumes or inference artifacts.

### Quality control and downloads

**Quality control · this exam** lists the exam checks (coverage, field of view, phases,
model) and one row per breast (region volume, plausibility, nipples found, review flags).
Values that need attention are red. **Run summary** shows the counts for the whole run,
image conversion, annotation agreement and motion. When a run has no `--motion`, the
motion section says so.

**Downloads** gives the full data as files: the open case's QC and measurements, the run
summary, the saved reviews (CSV and JSON) and all reviews (zip). **Saved files** lists the
tables, logs and reports of the results folder.

The artifacts panel opens saved tables, figures, reports and logs. Download preserves
the original file. Artifact filenames are listed in the
[command guide](RUNNING_ANALYSIS.md#what-the-output-folder-contains).

## Artifact contract

`<name>_viewer.json` uses schema version 1: a run name, `portable`, and cases with a
`files` mapping of `source`, `pred_region`, `pred_nipple`, `gt_region`, `gt_nipple` and
`landmarks` to paths. Relative references resolve against the output folder; absolute
references locate external images or predictions. Optional layers are absent when they
were not generated. Case IDs may include collection/study paths.

Without a manifest, the reader reconstructs cases from `<name>_exams.csv`,
`<name>_stage2.csv` or `<name>_stage1.csv`. Predictions are resolved through saved
`landmarks.json`, not glob order. Annotations are joined from
`<name>_annotation_manifest.json`. Multiple named runs appear in the run selector;
failed cases remain visible with their reasons. Motion is loaded from
`<name>_motion.json` and its summary without importing the registration engine.

Native shape and affine must agree before a mask is overlaid. Where landmark identity
is available, source filename, shape and affine fingerprint must match too. Invalid
layers are disabled with an explanation. Images and masks are reoriented together to
canonical RAS without resampling. On oblique scans the panes remain voxel-plane views;
pixel display preserves physical aspect ratio. A 4D source displays frame zero explicitly.

## Portable export and distribution

An ordinary output folder can reference images outside it. Export a new review folder
to open the results elsewhere without the original dataset paths:

```bash
mri-dce-analysis bundle out/my_analysis --out out/my_analysis_review
# Move or zip the review folder, then run on the receiving machine:
mri-dce-analysis viewer /path/to/my_analysis_review --open
```

An export copies top-level evidence and each case's reference MRI, masks, available
annotations and production manifest into `viewer_data/`, then writes relative viewer
references. Historical CSV paths are retained as evidence; they are not used to resolve
relocated layers. Raw DICOM, conversion trees, model caches and additional DCE phases
are excluded. Saved motion JSON/CSV evidence remains reviewable without the post phases.

Missing required files fail export before creating the destination. Failed cases without
images remain listed. The destination must be new and outside the original analysis folder.
Ship the package wheel and the separate review folder; the review folder contains MRI and
annotation data.
