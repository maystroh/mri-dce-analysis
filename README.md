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
computer that will run the analysis.

```text
Install mri-dce-analysis and guide me through breast DCE MRI inference and review.

1) Check Git and clone the project:
   git --version
   If Git is missing, help me install it before cloning. Then run:
   git clone https://github.com/maystroh/mri-dce-analysis
   If it is already cloned, reuse that checkout after verifying its origin URL.

2) Go into the project folder:
   cd mri-dce-analysis
   Run all following commands from this folder. Read docs/USER_GUIDE.md,
   docs/INSTALL.md, docs/INPUT_DATASETS.md and docs/VIEWER_REVIEW.md.

3) Ask me how I want to install and run the project:
   First ask: "Would you like to use Docker or a Python virtual environment?"
   If I choose Docker, also ask: "Would you like to use the prebuilt image from
   Docker Hub, or build the image locally from this checkout?"
   Wait for my answers before installing dependencies, pulling or building an
   image, or downloading the model. Follow only the selected setup below and
   keep that choice for every later step. Detect my operating system and shell,
   check the selected setup's requirements, and adapt commands, environment
   activation and paths accordingly. The Docker launcher mri-docker.sh uses
   Bash; ensure Bash and Docker are available together. Suggest WSL2 on Windows
   only if needed for that setup.
   The commands below use CPU. If I request GPU, follow docs/DOCKER.md or
   docs/INSTALL.md and adapt the selected setup and inference commands.

   Docker with the prebuilt Docker Hub image:
   - Check docker info; help me install Docker or start its daemon if needed.
   - Pull the image and select it for all subsequent Docker commands:
     docker pull h2thez/mri-dce-analysis:latest
     export MRI_IMAGE=h2thez/mri-dce-analysis:latest
   - The image includes dependencies, the model and the example exams. Skip
     local image building and separate host model/example downloads.
   - Keep MRI_IMAGE set in every shell that runs mri-docker.sh, or prefix each
     command with MRI_IMAGE=h2thez/mri-dce-analysis:latest.
     bash mri-docker.sh run --help

   Docker with a local build:
   - Check docker info, docker compose version and git lfs version. Help me
     install missing prerequisites or start Docker if needed.
   - Download the example exams and verify them:
     git lfs install
     git lfs pull
     sha256sum -c examples/SHA256SUMS
   - Set MRI_IMAGE to the local image tag in compose.yaml and mri-docker.sh
     (currently mri-dce-analysis:0.2.5), then build and check it:
     export MRI_IMAGE=mri-dce-analysis:0.2.5
     docker compose build
     bash mri-docker.sh run --help
   - This build installs dependencies and downloads and verifies the model.
     Keep MRI_IMAGE set to the built image in every subsequent Docker shell.

   Python virtual environment:
   - Check for Python 3.10 and Git LFS; help me install missing prerequisites.
     Create a dedicated environment and install the CPU runtime dependencies:
     python3.10 -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip
     python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
     python -m pip install '.[all]'
   - Follow the model-download commands in docs/INSTALL.md: download the pinned
     release, verify its SHA-256 and extract it into models/. Then set:
     export MODEL_BUNDLE="$(pwd)/models/3D_landmarks_per_breast_v1"
     Check for model/MLmodel and config/inference_config.json in that folder.
   - Download and verify the example exams:
     git lfs install
     git lfs pull
     sha256sum -c examples/SHA256SUMS
   - Check the installed commands:
     mri-dce-analysis --help
     mri-viewer --help
     dicom-info --help
   - Activate this environment and set MODEL_BUNDLE in every subsequent shell.
     Use the Python commands below for this choice; skip Docker setup entirely.

4) Verify the chosen installation on a real example exam:
   For either Docker choice:
   bash mri-docker.sh example duke --out ./results/example_duke
   For the Python environment:
   mri-dce-analysis run --input examples/dicom/duke --dicom --out ./results/example_duke --name duke --model-bundle "$MODEL_BUNDLE" --device cpu --motion --no-viewer
   Check results/example_duke/duke_pipeline.json for status "complete" and
   exams_failed = 0. Inspect the inference and motion summaries. Report any
   failure with the relevant log and resolve it before processing my dataset.

5) Get my run settings:
   Ask only for missing information: input exam folder or JSON list of exams,
   output folder, run name, and whether I will open the viewer locally or from
   another computer. Set INPUT_DIR (or EXAMS_JSON), OUTPUT_DIR and RUN_NAME to
   these agreed values in the shell that runs the commands below, or replace
   the variables with the agreed values. Quote paths containing spaces and use
   paths valid for the selected shell. If using WSL2, Windows files on the C:
   drive are accessed through paths such as /mnt/c/data/exams.
   For the Python viewer, also set VIEWER_HOST=127.0.0.1 for local access or
   VIEWER_HOST=0.0.0.0 for access from another computer; see docs/DOCKER.md and
   docs/VIEWER_PACKAGING.md for remote access and SSH tunnels.

6) Check the input before inference:
   Follow docs/INPUT_DATASETS.md to identify each exam's reference and ordered
   DCE phases. DICOM and NIfTI are detected automatically. For an untidy input
   folder, first inspect what can be organized using the chosen installation:
   Docker:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --organize-check
   Python:
   mri-dce-analysis run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --organize-check
   Explain any problems and resolve them with me. Add --organize to the actual
   run only when the layout needs it. For a JSON list of exam folders, follow
   the JSON instructions in docs/INPUT_DATASETS.md.

7) Run inference and motion, then open the viewer on the results:
   For either Docker choice, use the selected MRI_IMAGE:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   For a JSON list of exams, use this Docker command instead:
   bash mri-docker.sh run --input-json "$EXAMS_JSON" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   The wrapper starts the viewer after successful analysis and keeps running.

   For the Python environment, run analysis first:
   mri-dce-analysis run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --name "$RUN_NAME" --model-bundle "$MODEL_BUNDLE" --device cpu --motion --no-viewer
   For a JSON list of exams, use this Python command instead:
   mri-dce-analysis run --input-json "$EXAMS_JSON" --out "$OUTPUT_DIR" --name "$RUN_NAME" --model-bundle "$MODEL_BUNDLE" --device cpu --motion --no-viewer
   After successful analysis, start the Python viewer:
   mri-dce-analysis viewer "$OUTPUT_DIR" --host "$VIEWER_HOST" --port 8060

   Use a persistent terminal/process session for the chosen viewer command.

8) Check the results and guide me in the viewer:
   Verify "$OUTPUT_DIR/${RUN_NAME}_pipeline.json" reports completion. Inspect
   failed exams and review flags. Check that the viewer responds, then give me
   http://localhost:8060 if I am on the same computer, or a usable server address
   or SSH tunnel if I connect remotely. For Docker, the script prints the server
   address. Help me select the run and exam, move through axial, sagittal and
   coronal slices, adjust mask transparency, and save my accept/reject reviews
   for the breast-region and nipple masks. Keep the viewer session running.

9) Explain how to return to the results:
   Tell me where results and reviews were saved. Show how to stop the viewer
   with Ctrl+C and reopen the saved run without rerunning inference.
   For Docker, with the same selected MRI_IMAGE:
   bash mri-docker.sh viewer "$OUTPUT_DIR"
   For Python, with the same environment active:
   mri-dce-analysis viewer "$OUTPUT_DIR" --host "$VIEWER_HOST" --port 8060
   Keep the original NIfTI input folder available so its images remain viewable.
```
