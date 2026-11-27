# Docker

The [README](USER_GUIDE.md#run-with-docker) shows how to get the image and run the
main commands. This page covers the details.

## The image

The image contains the code, every runtime dependency (Python 3.10, CPU PyTorch
2.6.0, MLflow, SimpleITK, DICOM conversion), the per-breast v1
[landmark model](../models/MODEL_CARD.md) and the two example exams. It is about
2.6 GB. `docker compose build` verifies the model's SHA-256 and loads it once, so a
finished image runs offline. Run the build again after changing the code; unchanged
dependency layers are reused.

The example exams are stored with Git LFS and are copied into the image only when
they were downloaded before the build (see [examples](../examples/README.md#get-the-images)).
If the build prints `WARNING: example images are Git LFS pointers`, run `git lfs pull`
and build again. The `example` command needs those images.

## mri-docker.sh

`mri-docker.sh` runs the image on folders of your computer. Give it ordinary paths,
relative or absolute. `bash mri-docker.sh --help` and `bash mri-docker.sh run --help`
list the commands and options.

The script mounts the paths given to `--input`, `--input-json`, `--out`,
`--annotations`, `--predictions-root`, `--model-bundle` and `--dataset`. Inputs are
read-only. A missing `--out` folder is created, and results belong to the user who
runs the script. Folders are mounted at the same path inside the container, so the
paths in messages are your own. Folders under system paths such as `/tmp` or `/usr`
appear as `/host/N` instead.

NIfTI runs read their images from the original `--input` folder; DICOM runs keep
converted copies under `--out`. `viewer` therefore also shares, read-only, the input
folder recorded in each NIfTI run's `*_pipeline.json`. If that folder has moved, the
viewer prints a note and shows the run's results without images. Runs whose inputs
were under a system path (`/host/N`) cannot be found again this way.

`--organize` works the same through the script: `bash mri-docker.sh run --input
~/messy_data --out ~/results/study --organize` (or `--organize-check` to only list the
exams). The input stays read-only. The temporary exam folders of links go under `--out`
and are removed as each exam is done. `viewer` shares the input folder of organized runs
too, because their NIfTI phases stay there. See
[organize an untidy folder](INPUT_DATASETS.md#organize-an-untidy-folder).

All `mri-dce-analysis` options work as described in
[running an analysis](RUNNING_ANALYSIS.md). The container adds these defaults:

- `run` adds `--model-bundle` (the bundled model) and `--no-viewer`.
- `viewer` adds `--host 0.0.0.0`.

`mri-docker.sh` itself also accepts `--viewer` on `run` and `example`: after a successful
run it starts `mri-docker.sh viewer` on the `--out` folder, so one command goes from
analysis to review. A failed run does not start the viewer.

To use a different model, pass its folder with `--model-bundle`.

Paths that contain `:` cannot be mounted by Docker. An input CSV or JSON manifest
is mounted with its folder; the image paths listed inside it must be in that folder.

## Viewer on a server

`mri-docker.sh viewer` and `docker compose up viewer` publish the port on all
interfaces (`0.0.0.0:8060`). On a server, start the viewer and open
`http://<server address>:8060` from your computer. The script prints the address:

```bash
bash mri-docker.sh viewer /path/to/results/study
# Viewer: http://localhost:8060  (Ctrl+C stops it)
#   From another computer: http://10.0.0.12:8060
```

Change the defaults with environment variables before the command:

| Variable | Default | Use |
|----------|---------|-----|
| `VIEWER_BIND` | `0.0.0.0` | `127.0.0.1` keeps the viewer on this machine only; then use an [SSH tunnel](VIEWER_PACKAGING.md#open-a-saved-run) |
| `VIEWER_PORT` | `8060` | another port, e.g. when 8060 is in use |

```bash
VIEWER_PORT=8070 bash mri-docker.sh viewer /path/to/results/study
VIEWER_BIND=127.0.0.1 bash mri-docker.sh viewer /path/to/results/study
```

If the page does not open from another computer, the server firewall can block the
port. Ask the server admin to allow it (for example `sudo ufw allow 8060/tcp`).

The viewer has no login. Anyone who can reach the port can see the patient images and
change reviews. Use it only on a trusted network.

If you start the container with `docker run` yourself, publish the port and make the
viewer listen inside the container on all interfaces:

```bash
docker run --rm -p 0.0.0.0:8060:8060 -v /path/to/results/study:/out/study \
  mri-dce-analysis:0.2.5 viewer /out/study --host 0.0.0.0 --port 8060
```

## GPU

On a host with the NVIDIA Container Toolkit, build the CUDA image and select it:

```bash
docker compose --profile gpu build mri-gpu
MRI_IMAGE=mri-dce-analysis:0.2.5-cu124 MRI_DOCKER_OPTIONS="--gpus all" \
  bash mri-docker.sh run --input ~/exams --out ~/results/study --device cuda --motion
```

## Another computer without internet access

`mri-docker.sh` needs only Docker and the image. Copy both:

```bash
docker save mri-dce-analysis:0.2.5 | gzip > mri-dce-analysis.tar.gz
# copy mri-dce-analysis.tar.gz and mri-docker.sh, then on the other computer:
docker load < mri-dce-analysis.tar.gz
```

## Docker Compose

Compose can also run the image on two fixed folders set in `.env` (see
`.env.example`): `DATA_DIR` is mounted at `/data` and `OUT_DIR` at `/out`.

```bash
docker compose run --rm mri run --input /data/exams --out study --motion
RUN_DIR=study docker compose up -d viewer
docker compose down
```
