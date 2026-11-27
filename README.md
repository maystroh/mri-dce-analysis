# mri-dce-analysis

For human readers: [full installation, analysis and viewer guide](docs/USER_GUIDE.md).

## For agents

Copy the instructions below into any AI coding agent with terminal access to the
computer that will run the analysis. The agent should run commands from the project
root, using Linux or WSL2 on Windows.

```text
Install this project and help me run breast DCE MRI inference with its viewer.

1. Read docs/USER_GUIDE.md, docs/INPUT_DATASETS.md and docs/VIEWER_REVIEW.md.
   Find the project folder. Ask me for any missing information: input exam folder
   (or JSON list of exams), output folder, run name, and whether I will open the
   viewer on this computer or from another computer. Use CPU unless I request GPU.

2. Check the prerequisites:
   git --version
   git lfs version
   docker info
   docker compose version
   Help me install missing tools or start Docker if needed. On Windows, use WSL2
   paths and check that Docker is available inside WSL2. For installation without
   Docker, follow docs/INSTALL.md instead.

3. Prepare the example images and build the image, which installs dependencies
   and downloads and verifies the landmark model:
   If this is a Git clone and the example images are Git LFS pointers, run:
   git lfs install
   git lfs pull
   Then run:
   sha256sum -c examples/SHA256SUMS
   docker compose build
   bash mri-docker.sh run --help
   Use the image built from this checkout for the following commands; leave
   MRI_IMAGE unset unless I explicitly choose another image.

4. Validate the installation on the bundled DICOM exam:
   bash mri-docker.sh example duke --out ./results/example_duke
   Check that results/example_duke/duke_pipeline.json has status "complete" and
   exams_failed = 0. Inspect the inference and motion summaries for failures.
   Report any error with the relevant log before starting my dataset.

5. Inspect my input locally and check its layout against docs/INPUT_DATASETS.md.
   Identify each exam's reference and subsequent DCE phases. Before running the
   commands below, replace INPUT_DIR, OUTPUT_DIR, RUN_NAME and EXAMS_JSON with our
   agreed values, or set those variables in the shell that runs the command.
   Quote paths containing spaces. For an untidy input folder, first run:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --organize-check
   Explain any problems and resolve them with me. Add --organize to the actual
   run only when that layout needs it. For a JSON list, use --input-json instead
   of --input.

6. Run inference and motion, then launch the viewer on the same output folder:
   bash mri-docker.sh run --input "$INPUT_DIR" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   For a JSON list of exams, use this command instead:
   bash mri-docker.sh run --input-json "$EXAMS_JSON" --out "$OUTPUT_DIR" --name "$RUN_NAME" --motion --viewer
   Use a persistent terminal/process session: --viewer keeps running after the
   analysis. Check ${RUN_NAME}_pipeline.json for completion, inspect failed exams
   and review flags, and check that the viewer responds before giving me its URL.

7. Give me the viewer URL: http://localhost:8060 on this computer, or the server
   address printed by the script if I connect remotely. Follow docs/DOCKER.md for
   a different port or remote access. Help me select the run and exam, move
   through axial, sagittal and coronal slices, adjust mask transparency, and save
   my breast-region and nipple reviews. Keep the viewer session running.

8. Show me where results and reviews were saved, how to stop the viewer with
   Ctrl+C, and how to reopen these results later:
   bash mri-docker.sh viewer "$OUTPUT_DIR"
   Keep the original NIfTI input folder available so its images remain viewable.
```
