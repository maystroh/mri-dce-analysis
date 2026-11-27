# Installation

Install in your own Python environment to change or extend the code. To run without
installing Python, use the [Docker image](USER_GUIDE.md#run-with-docker).

The Python setup has been validated on Linux with Python 3.10 in a dedicated
virtual environment. The commands below use Bash; adapt the Python executable
and environment activation command to your operating system and shell.

## Install the package

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[all]'
```

These commands install CPU inference support. To use the tested Linux CPU dependency
set, add `-c constraints/python310-linux-cpu.txt` to the last command.

For CUDA inference, install matching GPU builds of PyTorch 2.6.0 and torchvision
0.21.0 before installing this package. Select the device with `--device cpu` or
`--device cuda`.

## Optional dependencies

| Install | Use |
| --- | --- |
| `pip install .` | NIfTI geometry, saved-output evaluation and viewer |
| `pip install '.[analysis]'` | Landmark inference |
| `pip install '.[motion]'` | Motion measurement |
| `pip install '.[dicom]'` | DICOM conversion |
| `pip install '.[all]'` | All runtime features |

## Download the model

Download the per-breast v1 bundle, verify it and extract it into `models/`:

```bash
curl -fL -o 3D_landmarks_per_breast_v1.tar.gz \
  https://huggingface.co/h2thez3/breast-mri-landmarks-per-breast-v1/resolve/23403f6d00ef915544a4ea00a5f854e74ba70a12/3D_landmarks_per_breast_v1.tar.gz
echo "da4e285d4dc08b2b86a225f935541d952fb222884c6c83433a69ae3f75c5e6fa  3D_landmarks_per_breast_v1.tar.gz" | sha256sum -c -
mkdir -p models
tar -xzf 3D_landmarks_per_breast_v1.tar.gz -C models
```

Pass `models/3D_landmarks_per_breast_v1` with `--model-bundle`. The folder must
contain `model/MLmodel` and `config/inference_config.json`. Load bundles from a
trusted source: loading a model executes its packaged Python code. The
[model card](../models/MODEL_CARD.md) describes the model.

## Check the installation

```bash
mri-dce-analysis --help
mri-viewer --help
dicom-info --help
```

Next, prepare your [input dataset](INPUT_DATASETS.md) and
[run an analysis](RUNNING_ANALYSIS.md).
