# Real DCE MRI examples

Two exams are included for testing the complete workflow. Images are stored with
Git LFS; the download is approximately 181 MiB. Files retain their original image
values, resolution and geometry.

| Example | Source exam | DCE phases | Native image size |
| --- | --- | --- | --- |
| [DICOM](dicom/duke/) | Duke `Breast_MRI_237` | Pre + 3 post-contrast phases | 320 × 320 × 144 per phase |
| [NIfTI](nifti/odelia/) | ODELIA / UMCU `0059120514` | Pre + 5 post-contrast phases | 384 × 384 × 200 per phase |

The ordered phase lists are [Duke phases](dicom/duke/phases.csv) and
[ODELIA phases](nifti/odelia/phases.csv). Only DCE inputs are included; subtraction
images, T2 images, clinical tables and annotation masks are omitted.

## Get the images

Install [Git LFS](https://git-lfs.com/), then run these commands from the repository root:

```bash
git lfs install
git lfs pull --include="examples/dicom/**,examples/nifti/**"
sha256sum --quiet --check examples/SHA256SUMS
```

A successful checksum check prints nothing. If an image is a short text file
starting with `version https://git-lfs.github.com/spec/v1`, its LFS content has not
been downloaded yet. Run `git lfs pull` before analysis.

## Run the examples with Docker

The Docker image contains both exams when it is built after `git lfs pull` (see
[Docker](../docs/DOCKER.md)). Run them from the image:

```bash
bash mri-docker.sh example duke --out out/example_duke
bash mri-docker.sh example odelia --out out/example_odelia
bash mri-docker.sh viewer out/example_duke
```

Or pass the images in this folder from your computer:

```bash
bash mri-docker.sh run --input examples/dicom/duke --dicom \
  --out out/example_duke --name duke --motion
bash mri-docker.sh run --input examples/nifti/odelia --layout odelia \
  --out out/example_odelia --name odelia --motion
```

Run these commands from the repository root. The model is included in the image.

## Run the examples without Docker

[Install](../docs/INSTALL.md) the project with its analysis, DICOM and motion
dependencies and [download the model](../docs/INSTALL.md#download-the-model), then set
its extracted folder:

```bash
MODEL_BUNDLE=/path/to/3D_landmarks_per_breast_v1
```

Run one complete DICOM exam, including conversion:

```bash
mri-dce-analysis run --input examples/dicom/duke --dicom \
  --out out/example_duke --name duke --model-bundle "$MODEL_BUNDLE" \
  --device cpu --motion --no-viewer
mri-dce-analysis viewer out/example_duke --open
```

Run the ODELIA NIfTI exam:

```bash
mri-dce-analysis run --input examples/nifti/odelia --layout odelia \
  --out out/example_odelia --name odelia --model-bundle "$MODEL_BUNDLE" \
  --device cpu --motion --no-viewer
mri-dce-analysis viewer out/example_odelia --open
```

Each example takes a few minutes on a CPU; remove `--motion` for a shorter
geometry-and-landmark run. Results are written under
`out/`; input images are left unchanged. Review breast-region and nipple overlays,
phase order, and the motion measurements in the viewer. These examples demonstrate
execution; they do not establish model accuracy or clinical suitability.

The selected exams produce both breast-region masks and both nipple masks without
landmark review flags. A complete motion run compares three post-contrast phases
per breast for Duke and five per breast for ODELIA. Pure NIfTI inputs do not carry
DICOM acquisition tags, so missing-metadata notices for ODELIA are expected.

To prepare your own DICOM or NIfTI data, follow the
[input requirements](../docs/INPUT_DATASETS.md).

## Sources and image licenses

The images retain their dataset licenses; the project's CC BY-NC-SA 4.0 license
does not replace them. Use de-identified data and retain the source dataset's
required notices when sharing examples.

**Duke.** Saha et al. (2021), [Dynamic contrast-enhanced magnetic resonance images
of breast cancer patients with tumor locations](https://doi.org/10.7937/TCIA.e3sv-re93),
The Cancer Imaging Archive. The [collection](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The included DICOM files are unchanged copies of the public de-identified DCE
series. TCIA notes that the collection's Frame of Reference UID was replaced
during curation and should not alone establish image alignment.

**ODELIA.** Müller-Franzes et al. (2025), [A European Multi-Center Breast Cancer
MRI Dataset](https://arxiv.org/abs/2506.00474). The [ODELIA Challenge dataset](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025)
is labelled [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and has
an additional data-usage agreement. This single-exam example is included with
explicit dataset-author consent. Access to, and redistribution of, the broader
dataset remain subject to that agreement. The six NIfTI files are unchanged
copies of the source DCE phases.
