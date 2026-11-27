"""The object the 2D pipeline holds: run the landmark model, cache it, hand back the pieces.

Caller-owned and passed in, the way `xai_runner` is, so it can be built and driven on its own —
`run_landmarks.py` uses the same object with no 2D pipeline involved.

The one thing this must never do is change the pipeline when it is absent. Every call site in
`inference_registry.py` is guarded on `landmark_runner is None`, and with the runner absent the
existing classical breast-mask path runs unchanged.
"""

import json
import os
from typing import Any, Dict, List, Optional

import SimpleITK as sitk

from mri_transforms.utils import log

from ..mlflow_bundle import RegistryInferenceConfig
from .regions import (FILTER_SCOPE_ANY, NIPPLE_FILTER_NONE, assign_sides, region_boxes,
                      split_by_regions, write_landmark_outputs)
from .per_breast import (COVERAGE_PER_BREAST, PER_BREAST_FUSION_VERSION, REGION_HEAD,
                         CoverageVerdict, halves_work_dir, resolve_coverage, sides_from_fused)
from .registry import LandmarkInferenceModel

FOLDER_LANDMARKS = "landmarks"


class LandmarkRunner:
    """Runs the landmark model for one exam and reuses the result when it is already on disk."""

    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs or {}
        mlflow_cfg = self.configs.get("mlflow_configs", {})
        self.cfg = RegistryInferenceConfig(
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            registered_model_name=mlflow_cfg.get("registered_model_name"),
            stage=mlflow_cfg.get("stage"),
            version=mlflow_cfg.get("version"),
            cache_root=mlflow_cfg.get("cache_root", "./models_cache"),
            local_bundle=mlflow_cfg.get("local_bundle"),
        )
        self.model = LandmarkInferenceModel(
            self.cfg,
            head_overrides=self.configs.get("head_overrides") or {},
            device=self.configs.get("device"),
            tile_step_size=self.configs.get("tile_step_size"),
            autocast_dtype=self.configs.get("autocast_dtype"),
            image_interpolator=self.configs.get("image_interpolator"),
        )
        self.use_as_breast_mask = bool(self.configs.get("use_as_breast_mask", True))
        self.use_for_bilateral_split = bool(self.configs.get("use_for_bilateral_split", True))
        self.on_failure = str(self.configs.get("on_landmark_failure", "fallback"))
        self.reuse_existing = bool(self.configs.get("reuse_existing", True))
        # Forces the Acquisition Scope a per-breast model is fed under, bypassing `scope_for`.
        # Present for the same reason `head_overrides` is — a registered config can be wrong —
        # and logged loudly when used. Ignored by whole-exam models.
        self.coverage_override = self.configs.get("coverage_override")
        # Retain the transient Breast Halves for inspection. Off by default: they are two
        # full-resolution volumes per exam and nothing downstream reads them.
        self.keep_halves = bool(self.configs.get("keep_halves", False))
        self.crop_margin_mm = float(self.configs.get("crop_margin_mm", 5.0))
        self.output_subdir = str(self.configs.get("output_subdir", FOLDER_LANDMARKS))
        # False-positive suppression for nipples, OFF by default. See regions.analyse_nipples for
        # why the default scope is the union of both regions and why nothing is deleted.
        # How the breast mask is applied in postprocessing. "component_intersect" keeps a lesion
        # touching the region whole, so boundary findings survive a tight mask.
        self.breast_mask_mode = str(self.configs.get("breast_mask_mode", "component_intersect"))
        self.nipple_filter = str(self.configs.get("nipple_filter", NIPPLE_FILTER_NONE))
        self.nipple_filter_scope = str(self.configs.get("nipple_filter_scope", FILTER_SCOPE_ANY))

        if self.use_for_bilateral_split and not self.use_as_breast_mask:
            raise ValueError(
                "landmark_configs: use_for_bilateral_split needs use_as_breast_mask — the split "
                "is driven by the same breast_region labels that produce the mask.")

    # ------------------------------------------------------------------ #
    def outdir(self, output_path: str) -> str:
        return os.path.join(output_path, self.output_subdir)

    def _cached(self, outdir: str, source_path: str,
                verdict: Optional[CoverageVerdict] = None) -> Optional[Dict[str, Any]]:
        """A previous run's masks, when they were made from this exact volume BY THIS MODEL UNDER
        THIS INPUT CONTRACT.

        Provenance is checked, not just presence. A cache hit has to mean "these masks would be
        produced again right now", because everything downstream reinterprets them as if it did:
        the analysis is redone on a hit, so a stale mask does not look stale — it looks like a
        fresh result with a current verdict written over it.

        These fields therefore have to agree, and each closes a reachable way of being wrong:

        * `source_file`, `source_shape`, `source_affine_sha1` — the same volume, not another
          phase of the same exam.
        * `model.name` — checking the VERSION alone is not enough, because versions are numbered
          per model and collide across them. `nipples_breast_region_v2` has versions 1-5 and
          `3D_landmarks_per_breast` has version 1, so "v1" names two different models, one
          whole-exam and one per-breast. Without this the two are interchangeable in the cache.
        * `model.run_id` — the exact registered artifact, when the cached manifest records one.
          A model version can be re-registered against a new run; name and version would still
          match. Only checked when present, so manifests written before run_id was recorded stay
          readable rather than being silently discarded.
        * the COVERAGE CONTRACT — the Acquisition Scope the masks were produced under. A
          bilateral mask is fused from two Breast Halves and carries two labels; a unilateral one
          is a single full-width pass. Reinterpreting one as the other is exactly what
          `--coverage-override` and a changed `scope_for` verdict would otherwise do, and the
          result is a plausible-looking mask describing a cut that never happened.
        * `fusion_version` for a per-breast model — model identity and coverage do not change
          when the inference-side restoration algorithm is corrected. Reusing a mask made by an
          older fusion would silently bypass that correction.
        """
        manifest_path = os.path.join(outdir, "landmarks.json")
        if not (self.reuse_existing and os.path.exists(manifest_path)):
            return None
        wanted_scope = verdict.scope if verdict is not None else None
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            from .regions import CoordinateFrames
            frames = CoordinateFrames(source_path)
            cached_model = manifest.get("model", {}) or {}
            if str(cached_model.get("name")) != str(self.cfg.registered_model_name):
                log.info(f"      [landmark] not reusing {manifest_path}: made by "
                         f"{cached_model.get('name')!r}, this run is "
                         f"{self.cfg.registered_model_name!r}")
                return None
            cached_run_id = cached_model.get("run_id")
            if cached_run_id and str(cached_run_id) != str(self.model.run_id):
                log.info(f"      [landmark] not reusing {manifest_path}: made by run "
                         f"{cached_run_id}, this run is {self.model.run_id}")
                return None
            if (verdict is not None
                    and cached_model.get("fusion_version") != PER_BREAST_FUSION_VERSION):
                log.info(f"      [landmark] not reusing {manifest_path}: made by fusion "
                         f"version {cached_model.get('fusion_version')!r}, this run needs "
                         f"{PER_BREAST_FUSION_VERSION}")
                return None
            for entry in manifest.get("entries", []):
                # The scope the masks were PRODUCED under. None on a whole-exam model, which
                # makes no coverage claim — and None == None is the correct match there.
                cached_scope = (entry.get("coverage") or {}).get("scope")
                if cached_scope != wanted_scope:
                    log.info(f"      [landmark] not reusing masks for "
                             f"{entry.get('source_file')}: produced under coverage "
                             f"{cached_scope!r}, this run is {wanted_scope!r}")
                    continue
                if (entry.get("source_file") == os.path.basename(source_path)
                        and entry.get("source_affine_sha1") == frames.affine_sha1
                        and list(entry.get("source_shape", [])) == list(frames.shape)
                        and str(cached_model.get("version")) == str(self.model.version)):
                    masks = {name: sitk.ReadImage(os.path.join(outdir, entry[key]))
                             for name, key in (("breast_region", "breast_region_file"),
                                               ("nipples", "nipples_file"))
                             if entry.get(key) and os.path.exists(os.path.join(outdir, entry[key]))}
                    if len(masks) == 2:
                        log.info(f"      [landmark] reusing masks from {manifest_path}")
                        return {"masks": masks, "entry": entry,
                                "model": manifest.get("model", {})}
        except Exception as exc:                      # a broken cache must not fail the run
            log.warning(f"   [!] could not reuse landmarks from {outdir}: {exc}")
        return None

    def run(self, source_path: str, output_path: str,
           is_bilateral: Optional[bool] = True) -> Optional[Dict[str, Any]]:
        """Masks + sides + boxes for one exam, or None when it failed and fallback is allowed.

        `is_bilateral` comes from the caller's own laterality check (`infer_bilateral_view` in
        `inference_registry.py`). For a per-breast model it is a hint that can confirm an
        uncertain geometry measurement; it never lets landmark inference rewrite the caller's
        pipeline laterality. `None` means the caller has no verdict. Defaults to `True` for
        standalone compatibility.
        """
        outdir = self.outdir(output_path)
        try:
            self.model.load()
            # Resolved BEFORE the cache is consulted, because the verdict is part of the cache
            # identity — masks produced under a different contract must miss, not be reinterpreted.
            # This is a header-only read, so it costs nothing against a model forward pass.
            scope_hint = (None if is_bilateral is None else
                          ("bilateral" if is_bilateral else "unilateral"))
            verdict = (resolve_coverage(source_path, self.coverage_override, scope_hint)
                       if self.model.input_coverage == COVERAGE_PER_BREAST else None)
            cached = self._cached(outdir, source_path, verdict)
            if cached is not None:
                # The masks are reused; the ANALYSIS is redone. `nipple_filter`,
                # `nipple_filter_scope`, `crop_margin_mm` and `is_bilateral` are read at run time,
                # so a cached entry can carry verdicts made under different settings — recomputing
                # costs seconds against the minutes the model would take, and keeps the manifest
                # honest.
                cached_sides, coverage = None, None
                if verdict is not None:
                    # Sides are part of the ANALYSIS for a per-breast model — they come from the
                    # coverage verdict and from which half each voxel came from, not from the
                    # mask's geometry. Derived here rather than letting `write_landmark_outputs`
                    # fall back to `assign_sides`, which would give a cached exam different
                    # semantics from a freshly-run one. The verdict is the SAME object the cache
                    # lookup matched on, so a hit cannot carry a different contract than the one
                    # reported.
                    coverage = verdict.as_dict()
                    is_bilateral = verdict.is_bilateral
                    cached_sides = sides_from_fused(cached["masks"][REGION_HEAD], verdict)
                written = write_landmark_outputs(
                    {"masks": cached["masks"], "model": cached.get("model", {})},
                    outdir, source_path, margin_mm=self.crop_margin_mm,
                    nipple_filter=self.nipple_filter, filter_scope=self.nipple_filter_scope,
                    write_masks=False, is_bilateral=is_bilateral, sides=cached_sides,
                    coverage=coverage)
                return {"masks": cached["masks"], "sides": written["sides"],
                        "boxes": written["boxes"], "entry": written["entry"],
                        "reused": True, "model": cached.get("model", {}),
                        "coverage": coverage}

            result = self.model.predict(source_path, work_dir=halves_work_dir(outdir),
                                        coverage_override=self.coverage_override,
                                        keep_halves=self.keep_halves,
                                        coverage_verdict=verdict)
            # The coverage verdict controls only the per-breast model's INPUT CONTRACT. It does
            # not replace the caller's pipeline laterality. Sides written below must nevertheless
            # describe the contract actually executed, so they follow coverage.scope.
            coverage = result.get("coverage")
            if coverage is not None:
                executed_bilateral = coverage["scope"] == "bilateral"
                measured_scope = coverage.get("measured_scope") or coverage["scope"]
                if is_bilateral is not None and executed_bilateral != is_bilateral:
                    log.warning(
                        f"   [!] [landmark] execution scope is {coverage['scope']!r}; geometry "
                        f"measured {measured_scope!r} "
                        f"({coverage['reason']}) but the caller passed "
                        f"is_bilateral={is_bilateral}; the landmark outputs follow the executed "
                        "execution contract without changing caller laterality.")
                is_bilateral = executed_bilateral
            written = write_landmark_outputs(result, outdir, source_path,
                                             margin_mm=self.crop_margin_mm,
                                             nipple_filter=self.nipple_filter,
                                             filter_scope=self.nipple_filter_scope,
                                             is_bilateral=is_bilateral,
                                             sides=result.get("sides"),
                                             coverage=coverage)
            return {"masks": result["masks"], "sides": written["sides"],
                    "boxes": written["boxes"], "entry": written["entry"],
                    "reused": False, "model": result["model"],
                    "coverage": coverage}
        except Exception as exc:
            log.error(f"   [!] landmark inference failed for {source_path}: {exc}", exc_info=True)
            if self.on_failure == "fallback":
                return None
            raise

    def write_binary_breast_mask(self, landmarks: Dict[str, Any], outdir: str,
                                 name: str = "breast_mask") -> Optional[str]:
        """Write `breast_mask.nii.gz` — the two regions binarised, on the SOURCE grid.

        Needed only for the `use_as_breast_mask=true, use_for_bilateral_split=false` combination,
        where the landmark mask feeds the pipeline but the historical midline cut still does the
        splitting. `split_bilateral_exams` expects that exact filename and splits it alongside the
        images, so it has to be on the same grid they are — the source grid, not RAS.
        """
        region = (landmarks.get("masks") or {}).get("breast_region")
        if region is None:
            return None
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{name}.nii.gz")
        sitk.WriteImage(sitk.Cast(region > 0, sitk.sitkUInt8), path)
        log.info(f"      [landmark] wrote {path} for the midline-split path")
        return path

    def split(self, nifti_files: List[str], output_split_path: str,
              landmarks: Dict[str, Any]) -> Optional[tuple]:
        """Per-side volumes from the region boxes, or None when there is nothing to split on."""
        boxes = landmarks.get("boxes") or {}
        if not boxes:
            log.warning("   [!] no breast regions found — not splitting.")
            return None
        if len(boxes) < 2:
            log.warning(f"   [!] only one breast region found ({list(boxes)}) — not splitting.")
            return None
        return split_by_regions(nifti_files, output_split_path,
                                landmarks["masks"]["breast_region"],
                                landmarks["sides"], boxes)
