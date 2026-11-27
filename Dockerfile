# Self-contained image: code, all runtime dependencies, the landmark model and the examples.
# Build and use it through compose.yaml; see docs/DOCKER.md.

# ---- Model: download the pinned release archive and verify its checksum -------------------
FROM python:3.10-slim-bookworm AS model

ARG MODEL_URL=https://huggingface.co/h2thez3/breast-mri-landmarks-per-breast-v1/resolve/23403f6d00ef915544a4ea00a5f854e74ba70a12/3D_landmarks_per_breast_v1.tar.gz
ARG MODEL_SHA256=da4e285d4dc08b2b86a225f935541d952fb222884c6c83433a69ae3f75c5e6fa

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fSL --retry 5 -o /tmp/model.tar.gz "${MODEL_URL}" \
    && echo "${MODEL_SHA256}  /tmp/model.tar.gz" | sha256sum --check - \
    && mkdir -p /opt/models \
    && tar xzf /tmp/model.tar.gz -C /opt/models \
    && test -f /opt/models/3D_landmarks_per_breast_v1/model/MLmodel \
    && test -f /opt/models/3D_landmarks_per_breast_v1/config/inference_config.json \
    && rm /tmp/model.tar.gz

# ---- Dependency list: pyproject.toml without its version -----------------------------------
# A version bump alone must not reinstall the 1.3 GB of dependencies, so the runtime stage
# installs them from this copy, whose content changes only when the dependencies change.
FROM python:3.10-slim-bookworm AS deps-spec
COPY pyproject.toml LICENSE NOTICE.md MANIFEST.in /spec/
RUN sed -i 's/^version = .*/version = "0.0.0"/' /spec/pyproject.toml

# ---- Runtime --------------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

# cpu (default) or a CUDA wheel tag such as cu124 for GPU inference.
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_TRACKING_URI=file:///tmp/mlruns \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    MODEL_BUNDLE=/opt/models/3D_landmarks_per_breast_v1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Large files that rarely change, first: a dependency change does not resend them.
COPY --from=model /opt/models /opt/models
COPY examples/ examples/
COPY models/ models/

# Examples are Git LFS files; report whether the build context contained the real images.
RUN if sha256sum --quiet --check examples/SHA256SUMS >/dev/null 2>&1; then \
        echo "Example images verified."; touch examples/.verified; \
    else \
        echo "WARNING: example images are Git LFS pointers; run 'git lfs pull' before building to include them."; \
    fi

# Dependencies next so code edits do not reinstall PyTorch.
COPY constraints/ constraints/
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.6.0 torchvision==0.21.0 \
         --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    && grep -v -E '^torch(vision)?==' constraints/python310-linux-cpu.txt > /tmp/constraints.txt

# Then the other dependencies, from pyproject.toml alone: an empty src/ and a stub README
# install no project code, so code and README edits do not rebuild this 1.3 GB layer.
COPY --from=deps-spec /spec/ ./
RUN mkdir -p src && touch README.md \
    && python -m pip install -c /tmp/constraints.txt ".[all]" \
    && python -c "import torch; assert torch.__version__.startswith('2.6.0'), torch.__version__" \
    && rm -rf src README.md build *.egg-info

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod 0755 /usr/local/bin/entrypoint.sh \
    && mkdir -p /data /out && chmod 0777 /out

# Code last: a code change rebuilds only these small layers (about 1 MB to push).
COPY docs/ docs/
COPY README.md pyproject.toml ./
COPY src/ src/
RUN python -m pip install --no-deps --force-reinstall . \
    && rm -rf build src/*.egg-info

# Fail the build now, not on the server later, if the model cannot load offline.
RUN python -c "import mlflow.pytorch, os; mlflow.pytorch.load_model(os.path.join(os.environ['MODEL_BUNDLE'], 'model'), map_location='cpu'); print('Model loads.')" \
    && mri-dce-analysis --help >/dev/null \
    && rm -rf /tmp/mlruns /tmp/matplotlib

WORKDIR /out
EXPOSE 8060
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
