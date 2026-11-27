# Breast DCE MRI landmark - per breast

The model predicts breast-region and nipple segmentation masks for breast DCE MRI
dataset characterization.

## Model

- Identifier: `3D_landmarks_per_breast`, version `1`.
- Architecture: 3D multi-head segmentation with an EfficientNet B0 encoder.
- Input: one MRI channel, one breast per forward pass, 128 × 128 × 128 patches.
- Outputs: breast-region and nipple masks restored to the input image grid.
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([full text](../LICENSE)).

The bundle's inference configuration specifies spacing, normalization, tiling and
postprocessing. The [release manifest](3D_landmarks_per_breast_v1.release.json)
records the archive size, SHA-256 and pinned download revision.

Download the [model archive](https://huggingface.co/h2thez3/breast-mri-landmarks-per-breast-v1/resolve/23403f6d00ef915544a4ea00a5f854e74ba70a12/3D_landmarks_per_breast_v1.tar.gz) or visit the
[Hugging Face model page](https://huggingface.co/h2thez3/breast-mri-landmarks-per-breast-v1) for the model files.

## Training dataset

Rokuss et al. (2025), [Divide and Conquer: A Large-Scale Dataset and Model for
Left-Right Breast MRI Segmentation](https://arxiv.org/abs/2507.13830).

## Evaluation

Evaluation results will be added in a future update.

## Usage

The Docker image includes the model. Without Docker, download, verify and extract
the bundle as described in [installation](../docs/INSTALL.md#download-the-model).
The bundle includes model code and weights. Inspect predicted masks and review flags
when interpreting results.
