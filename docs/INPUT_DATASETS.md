# Dataset input requirements

Pass your dataset folder with `--input` and choose one results folder with `--out`.
Subfolders are searched. A folder with `.dcm` files and no NIfTI files is read as DICOM automatically
(`--dicom` forces it); to select DICOM
exams from a JSON file, use [`--input-json`](#json-list-of-exams). Commands are
in [running an analysis](RUNNING_ANALYSIS.md); real exams to try are in the
[examples](../examples/README.md).

## NIfTI: one folder per exam

The simplest layout is:

```text
dataset/
├── exam_001/
│   ├── Pre.nii.gz
│   ├── Post_1.nii.gz
│   └── Post_2.nii.gz
└── exam_002/
    └── Pre.nii.gz
```

- Files can be `.nii` or `.nii.gz`, one 3D file per phase, with valid spatial affines
  and voxel spacing.
- Keep phases from the same exam together. Additional collection or patient folders above
  the exam folder are supported.
- `Pre` must be the actual pre-contrast reference. Number `Post_1`, `Post_2`, etc. in
  acquisition order; numbers are sorted numerically, so `Post_10` comes after `Post_9`.
- One image per exam is enough for geometry and landmarks. A single NIfTI file can also
  be passed directly as `--input`.

Numbered files are also supported, either in exam folders or together in a flat folder:

```text
dataset/
├── exam_001_0000.nii.gz
├── exam_001_0001.nii.gz
└── exam_002_0000.nii.gz
```

Here, `_0000` is the reference and higher indices are later phases. These indices must
represent time phases, rather than different image sequences such as T2 or diffusion.
For other names, use a [CSV](#custom-nifti-layouts-specify-the-files-in-a-csv) instead of
relying on filename guesses.

NIfTI images are read from their original location and are not copied. Reports,
predicted masks and logs are saved under `--out`.

## DICOM: series folders inside each exam

Use this layout:

```text
dataset/
└── patient_001/
    └── exam_001/
        ├── series_001/
        │   ├── slice001.dcm
        │   └── slice002.dcm
        └── series_002/
            ├── slice001.dcm
            └── slice002.dcm
```

The script recursively finds folders containing `.dcm` files and treats their parent as
the exam folder. It passes each exam to the DICOM converter, which uses DICOM metadata,
not folder names, to group series and identify pre/post-contrast and T2 images. Keep all
slices with their original spatial, series and acquisition timing metadata, and keep the
pre-contrast and post-contrast series of one acquisition together. For a DCE-only test,
leave out localizers, subtractions, MIPs, diffusion and unrelated examinations.

Converted NIfTI files are kept in **`out/my_analysis/nifti/`**, and the NIfTI analysis
runs on that folder. Geometry checks cover converted images; landmarks and motion use the
discovered DCE reference and phases. T2 is not a motion phase.

Discovery requires `.dcm` filenames; files without extensions or with `.ima` extensions
are not found. Keep the `exam/series/files` structure: extra folders between the exam and
series can cause phases to be treated as separate exams. For other layouts, use a
[JSON list](#json-list-of-exams). Check the conversion log and discovered phase
counts before interpreting the results.

## JSON list of exams

Use `--input-json`, instead of `--input`, for an object mapping exam directories to
metadata. Each exam directory can be DICOM or NIfTI, and the directories can be anywhere:

- A DICOM exam (`.dcm` files) is converted automatically; `--dicom` is not needed.
- A NIfTI exam (NIfTI files and no `.dcm` files) is used in place, with no conversion. Its
  phases are found as in [one folder per exam](#nifti-one-folder-per-exam) (`Pre.nii.gz`,
  `Post_1.nii.gz`, ... or `{case}_0000.nii.gz`, ...).

```json
{
  "/data/collection/patient_001/exam_001": {
    "patient_id": "patient_001",
    "labels": "N/A"
  },
  "/data/collection/patient_002/exam_002": {
    "patient_id": "patient_002",
    "labels": "N/A",
    "site": "site_a",
    "extra": {"reviewed": false, "tags": ["example"]}
  }
}
```

- Each key must identify **one complete exam directory** accessible on the machine running
  analysis. The converter receives that exact root and groups the series below it, including
  nested series folders. It does not guess the exam root from the folder depth.
- Absolute paths are supported. Relative exam paths resolve beside the JSON file.
- Each value must be a JSON object. `patient_id` and `labels` are optional; additional
  fields, objects, lists, numbers, booleans and null values are preserved. This metadata
  is not an [annotation](#annotation-packages) mapping.
- Each exam gets a stable ID derived from its folder name and path, so repeated patient
  IDs or folder names do not overwrite another exam.

With `--name study`, `study_dicom_exams.json` records each exam's metadata, source root,
selected phases and conversion errors, and `nifti/<exam_id>/` holds the converted NIfTIs
with a `_series_metadata/` conversion CSV. The metadata appears as `dataset_*` columns in
`study_exams.csv` (nested values JSON-encoded, the complete object in
`dataset_metadata_json`) and in the viewer's evidence panel. Missing exam directories and
failed conversions remain listed with their reasons; other exams continue.

## Organize an untidy folder

Add `--organize` to `run` to let the tool arrange an `--input` folder that does not follow
the layouts above. It works the same in Docker (`bash mri-docker.sh run ... --organize`).
Add `--organize-check` instead to only list the exams it would form, and why it cannot,
without running any analysis.

```bash
mri-dce-analysis run --input ~/messy_data --out ~/results/study --organize-check
mri-dce-analysis run --input ~/messy_data --out ~/results/study --organize
```

- **DICOM** files are found by their content, whatever their names or extensions (no
  `.dcm` needed), and grouped into exams by `StudyInstanceUID`, whatever the folders. Each
  exam is named after the deepest folder holding all of its files.
- **NIfTI** phases are found by the [name rules above](#nifti-one-folder-per-exam) only;
  nothing is guessed. A `Pre` whose `Post_N` files sit in a sibling folder
  (`exam/pre/Pre.nii.gz`, `exam/post/Post_1.nii.gz`) becomes one exam named after `exam`.
- When one exam folder holds both DICOM and NIfTI, the DICOM is used and the NIfTI ignored.
  Without `--organize`, such a folder is still read as NIfTI.
- DICOM and NIfTI exams can be mixed in one input folder.

Every exam is checked before analysis. If one cannot be formed, nothing is analysed and the
reasons are printed: a `Post` file with no `Pre` beside it or in a sibling folder, a folder
where several exams could own the same `Post` files, a repeated `Post_N`, NIfTI names
that identify no phases, a DICOM file without `StudyInstanceUID`, or one study with
several `PatientID` values. Fix or remove those exams, or list exam folders in a
[JSON](#json-list-of-exams).

Your input folder is never changed. Each exam gets a temporary folder of links (copies where
links are not supported) under `--out/_organized/`, removed as soon as that exam is
processed. Results refer to the original files. `<name>_organize.json` records every exam,
its type and files, and any errors; `--organize` cannot be combined with `--input-json`,
`--dicom` or `--analyse-only`.

## Custom NIfTI layouts: specify the files in a CSV

Use a CSV when filenames do not identify exams and phase order clearly. One row describes
one exam:

```csv
case,source_path,posts
exam_001,images/reference.nii.gz,"[""images/phase1.nii.gz"", ""images/phase2.nii.gz""]"
exam_002,images/another_reference.nii.gz,[]
```

- `case`: a unique exam ID. Use a relative ID without `..` or backslashes.
- `source_path`: the reference image. The column can also be named `image`.
- `posts`: optional, ordered JSON list of later phase paths. Omit it or use `[]` for
  single-image exams. The doubled quotes in the example are standard CSV escaping.
- Additional columns preserve dataset provenance in analysis reports.
- Relative image paths resolve beside the CSV. Absolute paths are also accepted.

Pass the CSV file as `--input`, or name it **`metadata.csv`** inside your dataset and pass
the dataset folder. The images are then selected explicitly rather than scanned.

## Annotation packages

For supported unilateral annotation scoring, use a package with `metadata.csv`:

```csv
case,image,breast_region,nipples
exam_001,images/reference.nii.gz,labels/region/exam_001.nii.gz,labels/nipples/exam_001.nii.gz
```

`source_path` can replace `image`. Image and mask paths resolve beside `metadata.csv`;
case IDs must match the landmark predictions. Keep annotation masks, for example under
`labels/`, separate from the MRI inputs. `labels` in a DICOM JSON entry, including
`"N/A"`, is metadata: it neither replaces these masks nor enables scoring. See
[annotation scoring](RUNNING_ANALYSIS.md#score-available-annotations) for the commands.

## Optional motion analysis

Add `--motion` to a NIfTI or DICOM command. Motion needs **at least two ordered 3D DCE
phases from the same exam**. Do not include subtraction images, T2, diffusion, masks or
images from another exam in the phase list. Split a 4D acquisition into ordered 3D phase
files, without changing their spatial geometry, before running analysis.

Single-phase exams still receive geometry and landmark analysis; motion is marked as
skipped. The method and measurements are described in
[analysis methods](ANALYSIS_WORKFLOW.md#motion-inside-the-breast-regions).
