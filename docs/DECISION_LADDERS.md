# Acquisition geometry

Geometry analysis records acquisition plane, coverage, confidence and the evidence
used for each verdict. Measurements and disagreement flags accompany the result.

## Acquisition plane

DICOM orientation uses `ImageOrientationPatient`: the cross product of its row and
column directions gives the slice normal. The dominant anatomical axis identifies
the plane; tilt is retained as an obliquity measurement.

For NIfTI, plane is inferred from the third voxel axis in the image affine:

| Direction | Plane |
| --- | --- |
| Superior/inferior | Axial |
| Anterior/posterior | Coronal |
| Left/right | Sagittal |

This assumes that the third voxel axis represents the acquired slice direction.
Axis reordering during conversion can affect that interpretation. Spacing and phase
consistency provide additional review evidence.

## Acquisition scope

The geometric rule uses left-right extent and centring to estimate unilateral or
bilateral coverage. The default width cut is 260 mm, with a 20 mm uncertainty margin.
Obliquity, an unusable affine, uncertain centring and disagreement between phases
can reduce confidence.

DICOM laterality tags and descriptions can describe acquisition intent. Geometric
coverage describes the field of view. Predicted anatomy provides additional evidence
about completeness and border contact. These observations are retained separately.

## Evidence precedence

The two geometry commands serve different review workflows:

| Command | Plane | Scope |
| --- | --- | --- |
| `stage1` | Prefers available DICOM plane metadata | Prefers NIfTI geometric coverage |
| `view-laterality` | Uses available DICOM evidence, then NIfTI | Uses available DICOM evidence, then NIfTI |

Conflicts remain visible in the output and reduce trust. Unknown DICOM values fall
back to available NIfTI evidence.

For `view-laterality`, `--metadata auto` searches for `dicom_series_info.csv` beside
the dataset and one directory above it. Use `series_csv:<path>` for an explicit file
or `none` to use NIfTI evidence alone. Derived series do not determine the exam verdict;
usable series are aggregated rather than choosing the first row.

## Interpreting results

Review raw plane and scope measurements alongside the selected verdict, source and
confidence. Image geometry cannot establish that both breasts are fully contained.
Mask border contact can suggest clipping, but does not prove it.

The model's input split is a separate execution decision. See
[landmark analysis](STAGE2_DATASET_ANALYSIS.md) and [analysis methods](ANALYSIS_WORKFLOW.md).
