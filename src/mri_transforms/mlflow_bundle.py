"""Shared MLflow model bundle: resolve a registered version, cache it locally, load it.

Split out of `RegistryInferenceModel` so a second model can reuse it without inheriting the
2D lesion pipeline. Everything here is model-agnostic — it knows how to pin a registered
version to `{cache_root}/{name}_v{version}/` and read the `inference_config` that ships beside
it, and nothing about slabs, postprocessing pipelines or head layouts.

Two consumers today:
  * `src.inference_registry.RegistryInferenceModel`  — the 2D lesion segmentation model
  * `src.landmark.registry.LandmarkInferenceModel`   — the 3D breast region + nipples model

Their `inference_config.json` files do NOT share a schema (the landmark model is
`schema_version: 3`), which is exactly why the split stops here: caching and loading are
common, interpreting the config is not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import torch
from mlflow.tracking import MlflowClient
from mri_transforms.utils import log

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_json_or_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(handle)
        import json
        return json.load(handle)


@dataclass
class RegistryInferenceConfig:
    tracking_uri: str
    registered_model_name: str
    stage: Optional[str] = "Production"   # "Production" | "Staging" | None
    version: Optional[Union[int, str]] = None  # if set, overrides stage
    local_bundle: Optional[str] = None
    cache_root: str = ".ml_infer_cache"   # local folder where we pin model+artifacts


class MlflowModelBundle:
    """A registered MLflow model pinned to a local cache directory, plus its config artifact.

    Layout under `{cache_root}/{name}_v{version}/`:
        model/    MLflow model directory (MLmodel, data/, and `code/` when the model was
                  registered with `code_paths`)
        config/   inference_config.json|yaml
        .complete sentinel written last, so a half-finished download is never reused

    `load()` is idempotent and safe to call from anywhere in a pipeline.
    """

    def __init__(self, cfg: RegistryInferenceConfig):
        self.cfg = cfg
        self.client = None
        if not cfg.local_bundle and cfg.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking_uri)
            self.client = MlflowClient()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version: Optional[str] = None
        self.run_id: Optional[str] = None
        self.cache_dir: Optional[str] = None

        self.model: Optional[torch.nn.Module] = None
        self.model_arch: Optional[str] = None
        self.model_in_channels: int = 1

        self.inf_cfg: Dict[str, Any] = {}
        self._loaded = False

    # --------------------------- loading & caching --------------------------- #
    def _resolve_version_and_run(self) -> Tuple[str, str]:
        """Return (version, run_id) for the registered model."""
        name = self.cfg.registered_model_name
        if self.cfg.version is not None:
            mv = self.client.get_model_version(
                name=name, version=str(self.cfg.version))
            return mv.version, mv.run_id
        # resolve stage → concrete version
        if not self.cfg.stage:
            raise ValueError("Either version or stage must be provided.")
        latest = self.client.get_latest_versions(
            name=name, stages=[self.cfg.stage])
        if not latest:
            raise RuntimeError(
                f"No versions found for model {name} at stage {self.cfg.stage}")
        mv = latest[0]
        return mv.version, mv.run_id

    def _materialize_cache(self):
        """Download model dir and inference_config into a stable local cache (idempotent)."""
        if self.cfg.local_bundle:
            self._materialize_local_bundle()
            return
        if not self.cfg.tracking_uri:
            raise ValueError("Specify a local model bundle or an explicit MLflow tracking URI")
        self.version, self.run_id = self._resolve_version_and_run()
        name = self.cfg.registered_model_name
        self.cache_dir = _ensure_dir(os.path.join(
            self.cfg.cache_root, f"{name}_v{self.version}"))
        sentinel = os.path.join(self.cache_dir, ".complete")

        if os.path.exists(sentinel):
            return

        if os.path.exists(self.cache_dir):
            for fname in os.listdir(self.cache_dir):
                fp = os.path.join(self.cache_dir, fname)
                if os.path.isdir(fp):
                    shutil.rmtree(fp)
                else:
                    os.remove(fp)

        model_dst = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{self.run_id}/model",
            dst_path=self.cache_dir,
        )

        try:
            cfg_dst = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{self.run_id}/config",
                dst_path=self.cache_dir,
            )
        except Exception:
            # Fall back if a single file was logged or path differs
            cfg_dst = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{self.run_id}/config/inference_config.json",
                dst_path=os.path.join(self.cache_dir, "config"),
            )

        with open(sentinel, "w") as f:
            f.write("ok")

    def _materialize_local_bundle(self) -> None:
        """Load an exported bundle in place, with content identity for prediction caches."""
        root = Path(self.cfg.local_bundle).expanduser().resolve()
        if not (root / "model" / "MLmodel").is_file():
            raise ValueError(f"Local bundle must contain model/MLmodel: {root}")
        config_names = ("inference_config.json", "inference_config.yaml", "inference_config.yml",
                        "config.yaml", "config.yml")
        if not any((root / "config" / name).is_file() for name in config_names):
            raise ValueError(f"Local bundle must contain config/inference_config.json or YAML: {root}")
        digest = hashlib.sha256()
        # Ignore Python caches generated while loading. Include shipped custom model code,
        # weights and config so replacing any of them invalidates saved predictions.
        for directory in (root / "model", root / "config"):
            for item in sorted(directory.rglob("*")):
                if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
                    continue
                if item.is_symlink():
                    raise ValueError(f"Local bundles must use regular files, not symlinks: {item}")
                if item.is_file():
                    file_digest = hashlib.sha256()
                    with item.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            file_digest.update(chunk)
                    digest.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
                    digest.update(file_digest.digest())
        self.cache_dir = str(root)
        self.version = str(self.cfg.version or "local")
        self.run_id = "local-sha256:" + digest.hexdigest()

    def _load_inference_config(self):
        """Load inference_config.(json|yaml) from cache — FIRST match wins.

        The candidate order is a priority order: a model that ships both an `inference_config.json`
        and the raw training `config.yaml` must be read from the former. Without the `break` the
        loop kept overwriting and the LAST existing candidate won, so `config.yaml` displaced the
        inference config and every schema-3 key (`heads`, `preprocessing`, `schema_version`)
        vanished. Models that ship only one candidate — every 2D lesion model — are unaffected
        either way.
        """
        cfg_dir = os.path.join(self.cache_dir, "config")
        self.inf_cfg = {}
        for fname in ("inference_config.json", "inference_config.yaml", "inference_config.yml", "config.yaml", "config.yml"):
            cand = os.path.join(cfg_dir, fname)
            if os.path.exists(cand):
                self.inf_cfg = _read_json_or_yaml(cand) or {}
                break

    def load(self):
        """Public: ensure cached + loaded into memory & ready to predict."""
        if self._loaded:
            return

        self._materialize_cache()
        self._load_inference_config()

        # Load the Pytorch model from local cached MLflow model dir.
        model_local_uri = os.path.join(self.cache_dir, "model")
        self.model = mlflow.pytorch.load_model(model_local_uri).to(self.device)
        self.model.eval()
        self._loaded = True
