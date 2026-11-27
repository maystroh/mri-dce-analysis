# Analysis steps and methods

This page explains how measurements are produced: image preparation, geometry checks,
per-breast landmarks, optional motion, then review. Commands are in
[running an analysis](RUNNING_ANALYSIS.md).

## Image preparation and geometry

NIfTI input is read from its existing location; DICOM input is converted first. Exam
discovery selects a reference volume and ordered later phases
([input requirements](INPUT_DATASETS.md)).

Stage 1 checks acquisition plane, spacing, field of view and scope, and saves uncertainty
and decision reasons beside the measurements. [Acquisition geometry](DECISION_LADDERS.md)
explains how plane and scope are decided and which evidence wins a conflict.

## Per-breast landmarks

Stage 2 runs **`3D_landmarks_per_breast`, version `1`** by default on the reference
volume; the model configuration selects its input contract. Bilateral exams are processed
as left and right [Breast Halves](CONTEXT.md) whose predictions are restored to the
original image grid; unilateral exams and full-width fallbacks use one full-width input.

The model predicts breast-region and nipple masks. Stage 2 measures region volume,
components, border contact and nipple availability, and keeps the reasons for review.
Supported annotations allow accuracy scoring; unlabelled datasets still get predictions
and plausibility QC. [Landmark analysis](STAGE2_DATASET_ANALYSIS.md) details the execution
choice, QC policy and saved-prediction rules.

## Motion inside the breast regions

When requested, motion analysis runs after landmark prediction.
The pre-contrast phase is the reference. If it is unavailable, the first available phase
is substituted and recorded. Every later phase is compared with that same reference;
these are not consecutive-phase differences. Input requirements are in
[optional motion analysis](INPUT_DATASETS.md#optional-motion-analysis).

### Registration method

The method uses **SimpleITK rigid registration**: three translations and three rotations,
fitted from coarse to fine. It uses **Mattes mutual information**, which allows image
brightness to change after contrast injection while finding spatial alignment.
Registration works on the original, unsplit phase volumes. The predicted breast region
restricts the reference voxels used to fit and measure motion.

- Trustworthy bilateral side labels produce separate left and right measurements.
- Unilateral exams and full-width execution use one merged breast region.
- Missing or invalid region masks trigger a whole-volume fallback, with a saved warning.

For each sampled point in the region, the fitted transform gives displacement in
millimetres: `distance = ||T(point) - point||`. This includes movement caused by both
translation and rotation. Large regions are sampled at up to 50,000 voxels with a fixed seed.

### Meaning of the measurements

| Measurement | Meaning |
|---|---|
| Mean displacement (mm) | Average displacement across the sampled region voxels. |
| 95th percentile (mm) | 95% of sampled voxel displacements are at or below this value. |
| Rotation (°) | Overall rotation angle of the fitted rigid transform. |
| Breast-series score (mm) | Largest mean displacement among its measured phases. |
| Exam score (mm) | Largest breast-series score among the available measurements. |

Results retain ROI size, registration settings and optimizer evidence, including stop
reasons. Missing measurements remain explicit; partial scores cover only measured phases.
`ok` means a measurement was produced. No motion pass/fail threshold is applied.
These are rigid-motion estimates; the original images are preserved and corrected phase
volumes are not saved.

## Reviewing results

Keep the raw values and warnings when comparing exams. Saved artifacts are listed in
[running an analysis](RUNNING_ANALYSIS.md#what-the-output-folder-contains); the
[viewer guide](VIEWER_PACKAGING.md) explains image overlays, phase graphs, the optional
table and portable review exports.
