# mri-dce-analysis

https://github.com/user-attachments/assets/3110b73d-a8eb-45fa-a76d-137671586d06

Tools for breast dynamic contrast-enhanced (DCE) MRI analysis and quality control.

- DICOM and NIfTI input.
- Acquisition geometry and coverage assessment.
- Breast-region and nipple landmark detection.
- Motion measurement.
- A review app to accept or reject the breast-region and nipple masks, exam by exam.

## For human

For human readers: [full installation, analysis and viewer guide](docs/USER_GUIDE.md).

The project paper link will be added soon.

## For agents

Copy the instructions below into any AI coding agent with terminal access to the
computer that will run the analysis. If Git is not
installed, help the user install it before cloning.

```text
Install mri-dce-analysis and guide me through breast DCE MRI inference and review.

1) Clone the project:
   git clone https://github.com/maystroh/mri-dce-analysis
   If it is already cloned, reuse that checkout after verifying its origin URL.

2) Go into the project folder:
   cd mri-dce-analysis
   Run all following commands from this folder. Read docs/USER_GUIDE.md,
   docs/INPUT_DATASETS.md and docs/VIEWER_REVIEW.md before proceeding.

3) Install dependencies and prepare the model:
   Check Git LFS, Docker and the Compose plugin:
   git lfs version
   docker info
   docker compose version
   Help me install missing prerequisites or start Docker as needed. On Windows,
   check that Docker is available inside WSL2. Use CPU by default; if I request
   GPU, follow the GPU setup in docs/DOCKER.md.

   Download the example exams and verify their checksums:
   git lfs install
   git lfs pull
   sha256sum -c examples/SHA256SUMS

   Build the image, which installs all Python runtime dependencies, downloads
   the landmark model, verifies its checksum and checks that it loads:
   docker compose build
   bash mri-docker.sh run --help
   Use the image built from this checkout for subsequent commands. Check for an
   MRI_IMAGE environment override and use the default local image unless I have
   chosen another image. For installation without Docker, follow docs/INSTALL.md.

4) Verify the installation on a real example exam:
   bash mri-docker.sh example duke --out ./results/example_duke
   Check results/example_duke/duke_pipeline.json for status "complete" and
   exams_failed = 0. Inspect the inference and motion summaries. Report any
   failure with the relevant log and resolve it before processing my dataset.

5) Get my run settings:
   Ask only for missing information: input exam folder or JSON list of exams,
   output folder, run name, and whether I will open the viewer locally or from
   another computer. Set INPUT_DIR (or EXAMS_JSON), OUTPUT_DIR and RUN_NAME to
   these agreed values in the shell that runs the commands below, or replace
   the variables with the agreed values. Quote paths containing spaces. On
   Windows, use WSL2 paths such as /mnt/c/... for files on the C: drive.

6) Check the input before inference:
   Follow docs/INPUT_DATASETS.md to identify each exam's reference and ordered
   DCE phases. DICOM and NIfTI are detected automatically. For an untidy input
   folder, first inspect what can be organized:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --organize-check
   Explain any problems and resolve them with me. Add --organize to the actual
   run only when the layout needs it. For a JSON list of exam folders, follow
   the JSON instructions in docs/INPUT_DATASETS.md.

7) Run inference and motion, then open the viewer on the results:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   For a JSON list of exams, use this command instead:
   bash mri-docker.sh run --input-json "$EXAMS_JSON" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   Run this in a persistent terminal/process session. The --viewer option starts
   the viewer after successful analysis and keeps the command running.

8) Check the results and guide me in the viewer:
   Verify "$OUTPUT_DIR/${RUN_NAME}_pipeline.json" reports completion. Inspect
   failed exams and review flags. Check that the viewer responds, then give me
   http://localhost:8060 if I am on the same computer, or the server address
   printed by the script if I connect remotely. See docs/DOCKER.md for another
   port, remote access or an SSH tunnel.
   Help me select the run and exam, move through axial, sagittal and coronal
   slices, adjust mask transparency, and save my accept/reject reviews for the
   breast-region and nipple masks. Keep the viewer session running.

9) Explain how to return to the results:
   Tell me where results and reviews were saved. Show how to stop the viewer
   with Ctrl+C and reopen the saved run without rerunning inference:
   bash mri-docker.sh viewer "$OUTPUT_DIR"
   Keep the original NIfTI input folder available so its images remain viewable.
```
