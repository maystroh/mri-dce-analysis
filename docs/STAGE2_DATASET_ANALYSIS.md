# Landmark analysis

Stage 2 selects a reference volume per exam, predicts breast-region and nipple masks,
and saves measurements, review flags and prediction metadata. For commands, see
[running an analysis](RUNNING_ANALYSIS.md).

## Coverage and model execution

Geometric coverage and model execution are recorded separately. Bilateral execution
can use two per-breast inputs. Unilateral exams and uncertain cases can use a full-width
input, depending on the model contract and available evidence.

Uncertain measured-bilateral exams can split when the independent scope classifier
agrees and the image grid is structurally usable. Structural uncertainty requires
full-width execution. The saved evidence preserves the measured scope, classifier
hint, confidence, chosen execution and reason.

The per-breast split uses the middle of the canonical RAS array. Its offset from
world x = 0 is measured for two-half execution; reporting the offset does not move
the split.

## Dataset adapters

Input layouts and manifests are described in [input requirements](INPUT_DATASETS.md).
Stage 2 preserves recursive case IDs and phase order, excludes label directories
from image discovery, and retains extra manifest fields as `dataset_*` metadata.

## Measurements and review

Per-side results include region volume, connected components, largest-component
volume, left-right border contact, nipple availability, nipple volume and physical
coordinates. The configured per-breast volume band is 279,000–2,635,000 mm³, and the
ROI floor is 100,000 mm³. These are plausibility checks, not accuracy measurements.

A full-width record can contain both breasts. A per-breast volume band is not
applicable to that merged record. Missing nipples, multiple components and uncertain
coverage remain explicit review findings.

`<name>_exams.csv` includes successful and failed cases. The stage-2 summary aggregates
statuses, model provenance, geometry strata and scope disagreements. Incomparable
cases remain visible instead of being counted as agreement.

## Existing artifacts

`--analyse-only --predictions-root <directory>` reads saved masks and `landmarks.json`
without loading model weights or rewriting predictions. Source filename, shape and
affine must agree with the saved identity. Changing model content or inference
configuration changes prediction identity during normal inference.

For annotated unilateral packages, `mri-dce-analysis evaluate` scores saved predictions.
Reports include mask Dice, nipple detection and centroid error, scoring statuses and
file hashes. Source-data failures are distinguished from model failures.

## Limitations

Use annotated data to measure segmentation and localization accuracy. Provide separate 3D phase volumes; full 4D landmark execution
is not supported. Dataset-specific phase selection and annotation conventions may
require an explicit manifest.
