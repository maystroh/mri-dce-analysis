# mri-dce-analysis

Tools for breast dynamic contrast-enhanced (DCE) MRI analysis and quality control.

- DICOM and NIfTI input.
- Acquisition geometry and coverage assessment.
- Breast-region and nipple landmark detection.
- Motion measurement.
- A review app to accept or reject the breast-region and nipple masks, exam by exam.

## Run with Docker

The image contains all dependencies, the landmark model and the example exams.
There are two options:

- [Option 1: use the image from Docker Hub](#option-1-use-the-image-from-docker-hub),
  ready to use, no build.
- [Option 2: build the image yourself](#option-2-build-the-image-yourself) from the
  project.

Both options run the image with the `mri-docker.sh` script of the project folder.

### Option 1: use the image from Docker Hub

A ready-to-use image is available on Docker Hub at
[h2thez/mri-dce-analysis](https://hub.docker.com/r/h2thez/mri-dce-analysis).
The `latest` tag always points to the newest version. It has the review viewer described in
[Review the results in the viewer](VIEWER_REVIEW.md), with the axial,
sagittal and coronal views. Pull it and point the commands below at it:

```bash
docker pull h2thez/mri-dce-analysis:latest
export MRI_IMAGE=h2thez/mri-dce-analysis:latest
```

`export` sets the image for this terminal only. Run it again in a new terminal, or put
`MRI_IMAGE=h2thez/mri-dce-analysis:latest` before each command.

**Update to a new version.** Run `docker pull h2thez/mri-dce-analysis:latest` again.
Results and reviews stay in your results folders, and the new viewer opens them as they are.
To keep one fixed version, use its tag (for example `0.2.1`) in place of `latest`.

You still need the project folder for `mri-docker.sh`: get it as in
[Option 2](#option-2-build-the-image-yourself) and skip `docker compose build`.
Run every command below from the `mri-dce-analysis` folder.

**Input folder format.** See [input requirements](INPUT_DATASETS.md) for how to
arrange your exams: `exam/Pre.nii.gz, Post_1.nii.gz, ...` for NIfTI, or
`patient/exam/series/*.dcm` for DICOM. For exams in other folders, a JSON file that
lists the exam folders (DICOM or NIfTI) is enough: pass it with `--input-json` in place of
`--input` (see [JSON list of exams](INPUT_DATASETS.md#json-list-of-exams)).
`mri-docker.sh` shares each exam folder of the JSON with Docker, wherever it is.
Or add `--organize` to let the tool arrange an untidy `--input` folder itself: it finds
the DICOM and NIfTI exams, checks them all first, and stops with a list of problems if any
exam cannot be formed. `--organize-check` only prints that list (see
[organize an untidy folder](INPUT_DATASETS.md#organize-an-untidy-folder)).

**End to end in one command.** Add `--viewer` to analyse and then open the viewer on
the `--out` folder (Ctrl+C stops the viewer). Open http://localhost:8060 on the same
computer, or the "From another computer" address that the script prints on a server:

```bash
# Bundled example exam
bash mri-docker.sh example duke --out /path/to/results/example_duke --viewer

# Your exams (DICOM or NIfTI, detected automatically)
bash mri-docker.sh run --input /path/to/exams --name study --out /path/to/results/study --motion --viewer
```

**Or list your exams in a JSON file.** One key per exam folder, DICOM or NIfTI; the value can be `{}`.
Example `exams.json`:

```json
{
  "/path/to/patient_001/dicom_exam_001": {},
  "/other/disk/patient_002/nifti_exam_007": {"patient_id": "patient_002"}
}
```

Then run it:

```bash
bash mri-docker.sh run --input-json exams.json --name study --out /path/to/results/study --motion --viewer
```

Otherwise, run each step separately:

**On data inside the image.** Try the bundled example exam:

```bash
bash mri-docker.sh example duke --out /path/to/results/example_duke
```

**On external data from your disk.** Prepare your input folder with one subfolder per
exam, as described in [input requirements](INPUT_DATASETS.md). Then:

```bash
# Analyse your exams (DICOM or NIfTI, detected automatically)
bash mri-docker.sh run --input /path/to/exams --name study --out /path/to/results/study --motion
```

**Open the viewer.** The viewer does not open on its own after the run: launch it
yourself on the `--out` folder you set in the command above (here `/path/to/results/study`;
for the example, `/path/to/results/example_duke`):

```bash
# Open http://localhost:8060, or on a server the address it prints (Ctrl+C stops it)
bash mri-docker.sh viewer /path/to/results/study
```

Review the masks as in [Review the results in the viewer](VIEWER_REVIEW.md).
Your reviews are saved in the same results folder.

`--name` labels a run; without it, the run is called `dataset`. Output files start with
this name, and the viewer lists one entry per name found in the folder, so give each run
its own `--name`, ideally with its own `--out` folder. For NIfTI input the viewer shows
the images from the original `--input` folder, which `mri-docker.sh viewer` shares
automatically, so keep that folder in place.

### Option 2: build the image yourself

You need Git, [Git LFS](https://git-lfs.com/) and Docker with the Compose plugin.

Download or clone the public project, then build the image from its folder:

```bash
git lfs install
git lfs pull
docker compose build
```

Then run `mri-docker.sh` from the `mri-dce-analysis` folder, as in
[Option 1](#option-1-use-the-image-from-docker-hub).

**Input folder format.** See [input requirements](INPUT_DATASETS.md) for how to
arrange your exams: `exam/Pre.nii.gz, Post_1.nii.gz, ...` for NIfTI, or
`patient/exam/series/*.dcm` for DICOM. For exams in other folders, a JSON file that
lists the exam folders (DICOM or NIfTI) is enough: pass it with `--input-json` in place of
`--input` (see [JSON list of exams](INPUT_DATASETS.md#json-list-of-exams)).
`mri-docker.sh` shares each exam folder of the JSON with Docker, wherever it is.
Or add `--organize` to let the tool arrange an untidy `--input` folder itself: it finds
the DICOM and NIfTI exams, checks them all first, and stops with a list of problems if any
exam cannot be formed. `--organize-check` only prints that list (see
[organize an untidy folder](INPUT_DATASETS.md#organize-an-untidy-folder)).

**End to end in one command.** Add `--viewer` to analyse and then open the viewer on
the `--out` folder (Ctrl+C stops the viewer). Open http://localhost:8060 on the same
computer, or the "From another computer" address that the script prints on a server:

```bash
# Bundled example exam
bash mri-docker.sh example duke --out /path/to/results/example_duke --viewer

# Your exams (DICOM or NIfTI, detected automatically)
bash mri-docker.sh run --input /path/to/exams --name study --out /path/to/results/study --motion --viewer
```

**Or list your exams in a JSON file.** One key per exam folder, DICOM or NIfTI; the value can be `{}`.
Example `exams.json`:

```json
{
  "/path/to/patient_001/dicom_exam_001": {},
  "/other/disk/patient_002/nifti_exam_007": {"patient_id": "patient_002"}
}
```

Then run it:

```bash
bash mri-docker.sh run --input-json exams.json --name study --out /path/to/results/study --motion --viewer
```

Otherwise, run each step separately:

**On data inside the image.** Try the bundled example exam:

```bash
bash mri-docker.sh example duke --out /path/to/results/example_duke
```

**On external data from your disk.** Prepare your input folder as above. Then:

```bash
# Analyse your exams (DICOM or NIfTI, detected automatically)
bash mri-docker.sh run --input /path/to/exams --name study --out /path/to/results/study --motion
```

**Open the viewer.** The viewer does not open on its own after the run: launch it
yourself on the `--out` folder you set in the command above (here `/path/to/results/study`;
for the example, `/path/to/results/example_duke`):

```bash
# Open http://localhost:8060, or on a server the address it prints (Ctrl+C stops it)
bash mri-docker.sh viewer /path/to/results/study
```

Review the masks as in [Review the results in the viewer](VIEWER_REVIEW.md).
Your reviews are saved in the same results folder.

The notes on `--name`, `--out` and NIfTI input in Option 1 apply here too.
The example exams can also be passed from the repository, for example
`--input examples/dicom/duke`.

See [Docker](DOCKER.md) for all commands, servers and GPU use.

## Review the results in the viewer

The viewer shows each exam with its predicted masks, so you can mark them as accurate or not.
See [how to review the results in the viewer](VIEWER_REVIEW.md).

## Run it in Python

Install the package and download the model with [installation](INSTALL.md).
**Input folder format.** See [input requirements](INPUT_DATASETS.md) for how to
arrange your exams: `exam/Pre.nii.gz, Post_1.nii.gz, ...` for NIfTI, or
`patient/exam/series/*.dcm` for DICOM. For exams in other folders, a JSON file that
lists the exam folders (DICOM or NIfTI) is enough: pass it with `--input-json` in place of
`--input` (see [JSON list of exams](INPUT_DATASETS.md#json-list-of-exams)).
`mri-docker.sh` shares each exam folder of the JSON with Docker, wherever it is.
Or add `--organize` to let the tool arrange an untidy `--input` folder itself: it finds
the DICOM and NIfTI exams, checks them all first, and stops with a list of problems if any
exam cannot be formed. `--organize-check` only prints that list (see
[organize an untidy folder](INPUT_DATASETS.md#organize-an-untidy-folder)).

Then run:

```bash
mri-dce-analysis run --input /path/to/exams --out out/study \
  --model-bundle /path/to/model --motion --no-viewer
mri-dce-analysis viewer out/study --open
```

**Or list your exams in a JSON file.** One key per exam folder, DICOM or NIfTI; the value can be `{}`.
Example `exams.json`:

```json
{
  "/path/to/patient_001/dicom_exam_001": {},
  "/other/disk/patient_002/nifti_exam_007": {"patient_id": "patient_002"}
}
```

Then run it:

```bash
mri-dce-analysis run --input-json exams.json --out out/study \
  --model-bundle /path/to/model --motion --no-viewer
mri-dce-analysis viewer out/study --open
```

Use `mri-dce-analysis --help` to explore the available commands.
Try the [real DICOM and ODELIA NIfTI examples](../examples/README.md) for complete DCE workflows.

## License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([full text](../LICENSE)).
See [NOTICE.md](../NOTICE.md) for attribution.
