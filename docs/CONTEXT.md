# Terminology

| Term | Meaning |
| --- | --- |
| Exam | A reference MRI volume and its associated phases and metadata. |
| Exam folder | The folder that names an exam. DICOM: the deepest folder holding every file of one study. NIfTI: the nearest folder above the reference that also holds its later phases. |
| Organized exam | A temporary, standard-layout view of one exam built from an untidy input folder; it is removed once that exam is processed, and results refer to the original files. |
| Organize check | A pass over an input folder that lists every exam it can form, or why it cannot, before any analysis runs. |
| DCE phase | One 3D acquisition in an ordered contrast-enhanced MRI series. |
| Reference phase | The image used for landmark inference and as the motion reference. |
| Acquisition plane | The orientation of the acquired slice axis: axial, sagittal or coronal. |
| Acquisition scope | The geometric classification of coverage as unilateral or bilateral. |
| Per-breast input | The volume passed to a model for one breast; bilateral exams may require two passes. |
| Breast Half | A left or right input split along the left-right axis, retaining the full anterior-posterior and superior-inferior extent. |
| Coverage verdict | The measured coverage and confidence used to select model execution. |
| Breast-region mask | A predicted or annotated binary segmentation of the breast region. |
| Nipple target | A binary nipple segmentation. |
| Nipple centre | A location derived from a nipple mask, expressed in physical coordinates. |
| Region volume plausibility | A comparison of predicted volume with a configured reference band. |
| Midline cut offset | The distance between the model's actual split and world x = 0. |
| Review flag | A finding that needs inspection, supported by saved measurements and reasons. |
| Scored case | A case included in evaluation; missing or unscorable source data are excluded from the metric denominator. |

Acquisition scope, anatomical completeness and model execution describe different
properties. A bilateral field of view can still clip a breast. A plausible predicted
volume does not establish segmentation accuracy.
