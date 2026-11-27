"""The 3D breast-landmark model: breast region (left/right) + nipples.

The pipeline is: prepare the volume (geometry.py) -> tile it through the model -> turn each
head's logits into a mask AT THE MODEL'S OWN 1.5 mm SPACING -> put the masks back on the source
grid with one physical-space resample.

Deciding before inverting is deliberate: the config precomputes its volume floors as voxel counts
"at run spacing", and the checkpoint's published metrics were computed that way.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import torch
from skimage.morphology import remove_small_holes, remove_small_objects

from mri_transforms.inference import sliding_window_predict
from mri_transforms.utils import log

from ..mlflow_bundle import MlflowModelBundle
from .geometry import (DEFAULT_IMAGE_INTERPOLATOR, PreparedVolume, prepare_volume, read_image,
                       resample_like)
from .per_breast import (COVERAGE_PER_BREAST, COVERAGE_WHOLE_EXAM,
                         PER_BREAST_FUSION_VERSION, REGION_HEAD, CoverageVerdict, cleanup_halves,
                         fuse_halves, resolve_coverage, sides_from_fused, split_to_halves)

ACTIVATION_SIGMOID = "sigmoid"
ACTIVATION_SOFTMAX = "softmax"
ASSIGNMENT_ARGMAX = "argmax"
ASSIGNMENT_THRESHOLD = "threshold"


@dataclass
class HeadSpec:
    """One prediction target, as the registered config describes it."""

    name: str
    channels: int
    activation: str
    class_assignment: str
    background_channel: Optional[int]
    classes: List[Dict[str, Any]]
    postprocess: Dict[str, Any]

    @property
    def is_multiclass(self) -> bool:
        return self.activation == ACTIVATION_SOFTMAX and self.channels > 1

    def label_values(self) -> List[int]:
        """Label value per class row, in channel order. `[1]` for a plain binary head."""
        if not self.classes:
            return [1]
        return [int(c.get("label", c.get("channel", i + 1)))
                for i, c in enumerate(self.classes)]

    @property
    def applies_threshold(self) -> bool:
        """Whether a threshold is applied at all — the config states it; argmax implies no.

        Newer configs carry `postprocess.applies_threshold` explicitly. Older ones do not, so it
        is derived: under `argmax` ONE reduction replaces thresholding for every row of the head,
        and `technique`/`threshold`/`threshold_space` do nothing.
        """
        stated = self.postprocess.get("applies_threshold")
        if stated is not None:
            return bool(stated)
        return self.class_assignment != ASSIGNMENT_ARGMAX

    @property
    def threshold(self) -> float:
        pp = self.postprocess
        technique = str(pp.get("thresholding_technique", "manual"))
        if technique != "manual":
            raise ValueError(
                f"head '{self.name}': thresholding_technique={technique!r} is not supported by the "
                "landmark path yet — only 'manual'. A global technique would have to be computed "
                "over the volume, which is a decision, not a default.")
        space = str(pp.get("thresholding_threshold_space", "prob"))
        if space != "prob":
            raise ValueError(f"head '{self.name}': expected a probability-space threshold, "
                             f"got {space!r}.")
        return float(pp.get("thresholding_threshold", 0.5))

    def validate(self) -> None:
        """Reject head configurations this path cannot honour, naming what is wrong."""
        if self.activation not in (ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX):
            raise ValueError(f"head '{self.name}': unknown activation {self.activation!r}")
        if self.class_assignment == ASSIGNMENT_ARGMAX and self.activation != ACTIVATION_SOFTMAX:
            raise ValueError(f"head '{self.name}': class_assignment=argmax needs activation "
                             f"softmax, got {self.activation!r}")
        if not self.applies_threshold and self.activation == ACTIVATION_SIGMOID:
            raise ValueError(f"head '{self.name}': a sigmoid head with no threshold has no rule "
                             "left to turn probabilities into a mask")

    def describe_rule(self) -> str:
        """One line naming exactly how this head's logits become a mask, for the startup log."""
        if self.activation == ACTIVATION_SOFTMAX and self.class_assignment == ASSIGNMENT_ARGMAX:
            return (f"argmax over {self.channels} channels, no threshold "
                    f"(bg=ch{self.background_channel})")
        if self.is_multiclass:
            return (f"softmax, foreground = 1-p(bg) > {self.threshold}, "
                    f"side by argmax over class channels")
        if self.channels > 1:
            return f"{self.channels} independent sigmoid channels > {self.threshold}"
        return f"sigmoid > {self.threshold}"


class LandmarkInferenceModel(MlflowModelBundle):
    """Runs `MRI_breast_landmarks_v1` over one exam volume and returns masks on its own grid.

    `head_overrides` forces a per-head setting that the registered config does not carry yet —
    today only `class_assignment`, so v3 (registered with the inherited `threshold` default) can
    be read with `argmax`, which is what the head's own documentation prescribes for a partition
    target. Safe because `class_assignment` never entered the loss and `breast_region` metrics
    were not checkpoint-selection objectives. See section 3.1 of the plan.
    """

    def __init__(self, cfg, head_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 device: Optional[str] = None, tile_step_size: Optional[float] = None,
                 autocast_dtype: Optional[str] = None,
                 image_interpolator: Optional[str] = None):
        super().__init__(cfg)
        self.head_overrides = head_overrides or {}
        self._device_override = device
        self._tile_step_override = tile_step_size
        self._autocast_override = autocast_dtype
        self._image_interpolator_override = image_interpolator

    # ------------------------------- config reading ------------------------------- #
    def load(self):
        super().load()
        if self._device_override:
            self.device = self._device_override
            self.model.to(self.device)
        schema = self.inf_cfg.get("schema_version")
        if schema is None or int(schema) < 2:
            raise ValueError(
                f"{self.cfg.registered_model_name} v{self.version} has schema_version={schema!r}; "
                "the landmark path needs schema 2 or later (heads[]/preprocessing{}).")
        for head in self.heads:
            head.validate()
            log.info(f"      [landmark] head '{head.name}': {head.channels}ch {head.activation} -> "
                     f"{head.describe_rule()}"
                     + (" (class_assignment OVERRIDDEN)" if head.name in self.head_overrides else ""))

    @property
    def heads(self) -> List[HeadSpec]:
        specs = []
        for raw in self.inf_cfg.get("heads", []):
            name = raw["name"]
            override = self.head_overrides.get(name, {})
            specs.append(HeadSpec(
                name=name,
                channels=int(raw.get("channels", 1)),
                activation=str(raw.get("activation", ACTIVATION_SIGMOID)),
                class_assignment=str(override.get("class_assignment",
                                                  raw.get("class_assignment", ASSIGNMENT_THRESHOLD))),
                background_channel=raw.get("background_channel"),
                classes=list(raw.get("classes", []) or []),
                postprocess=dict(raw.get("postprocess", {}) or {}),
            ))
        return specs

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.inf_cfg.get("preprocessing", {}) or {}

    @property
    def input_coverage(self) -> str:
        """How this model wants to be FED: one breast per pass, or the whole exam.

        Read from the registered config, never from a pipeline flag and never inferred from the
        head shape. That is the same contract `decide()` already runs on — which rule fires is
        decided by the head's own config, "never by the head's name or by its position in the
        list" — lifted one level up: the model states how it must be preprocessed, and the runner
        obeys. A flag that can disagree with the model eventually will, and the failure is silent:
        whole exams fed to a per-breast model produce plausible-looking nonsense.

        Absent means whole-exam, because every model registered before this key existed was a
        whole-exam model. Logged rather than assumed quietly, for the same reason
        `image_interpolator` logs its fallback.
        """
        raw = self.preprocessing.get("input_coverage")
        if raw is None:
            log.info(f"      [landmark] preprocessing.input_coverage absent -> "
                     f"{COVERAGE_WHOLE_EXAM!r} (this model predates the key)")
            return COVERAGE_WHOLE_EXAM
        value = str(raw)
        if value not in (COVERAGE_WHOLE_EXAM, COVERAGE_PER_BREAST):
            raise ValueError(
                f"{self.cfg.registered_model_name} v{self.version} declares "
                f"preprocessing.input_coverage={value!r}, which this code does not know how to "
                f"feed. Refusing rather than defaulting to {COVERAGE_WHOLE_EXAM!r}: running a "
                "model on an input contract it did not ask for yields confident nonsense.")
        return value

    @property
    def image_interpolator(self) -> str:
        """The interpolator training resampled with, so inference regrids the same way.

        Override wins; otherwise it's read from `preprocessing.offline_ops[].image_interpolator`
        (the `voxel_resample` entry); otherwise it falls back to `DEFAULT_IMAGE_INTERPOLATOR`
        because some registrations (this model's v3 among them) have `offline_ops: null` — the
        resample manifest wasn't readable at registration time. The fallback is logged, never
        silent, since guessing wrong here shifts every voxel's intensity before z-scoring.
        """
        if self._image_interpolator_override:
            log.info(f"      [landmark] image_interpolator={self._image_interpolator_override!r} "
                     "(CLI/config override)")
            return self._image_interpolator_override
        for op in self.preprocessing.get("offline_ops") or []:
            if op.get("op") == "voxel_resample" and op.get("image_interpolator"):
                value = str(op["image_interpolator"])
                log.info(f"      [landmark] image_interpolator={value!r} (from registered "
                         "offline_ops)")
                return value
        log.warning(f"   [!] [landmark] preprocessing.offline_ops has no image_interpolator "
                    f"(commonly because offline_ops is null); assuming "
                    f"{DEFAULT_IMAGE_INTERPOLATOR!r} to match this model family's training "
                    "resample. Pass image_interpolator= explicitly if that's wrong.")
        return DEFAULT_IMAGE_INTERPOLATOR

    @property
    def patch_size(self) -> Sequence[int]:
        tiling = self.preprocessing.get("tiling", {}) or {}
        return [int(v) for v in tiling.get("patch_size", [128, 128, 128])]

    @property
    def tile_step_size(self) -> float:
        if self._tile_step_override is not None:
            return float(self._tile_step_override)
        tiling = self.preprocessing.get("tiling", {}) or {}
        return float(tiling.get("tile_step_size", 0.5))

    # ------------------------------- the forward pass ------------------------------ #
    def prepare(self, source_path: str) -> PreparedVolume:
        # the recipe lives in the registered config, so the config has to be on hand first;
        # without this the defaults below are used and the mismatch is silent when they happen
        # to agree with the config (they do, at 1.5 mm isotropic).
        self.load()
        pre = self.preprocessing
        return prepare_volume(source_path,
                              spacing_zyx=pre.get("spacing_zyx", [1.5, 1.5, 1.5]),
                              online_ops=pre.get("online_ops", []),
                              image_interpolator=self.image_interpolator)

    def predict_logits(self, prepared: PreparedVolume) -> Dict[str, np.ndarray]:
        """`{head: (C, Z, Y, X)}` blended logits on the model grid.

        `SMP3D_MultiHead.forward` already returns `{head_name: logits}`, so the model goes
        straight into `sliding_window_predict` with no adapter.
        """
        self.load()
        autocast = None
        if self._autocast_override:
            autocast = getattr(torch, str(self._autocast_override))
        n_vox = int(np.prod(prepared.volume.shape[1:]))
        log.info(f"      [landmark] tiling {prepared.volume.shape[1:]} ({n_vox/1e6:.1f} MVox) "
                 f"with {tuple(self.patch_size)} @ step {self.tile_step_size} on {self.device}")
        return sliding_window_predict(
            lambda x: self.model(x), prepared.volume,
            patch_size=self.patch_size, tile_step_size=self.tile_step_size,
            device=self.device, autocast_dtype=autocast)

    # ------------------------------- head decisions -------------------------------- #
    def _voxel_counts(self, head: HeadSpec, voxel_volume_mm3: float) -> Dict[str, int]:
        """mm3 floors -> voxel counts, cross-checked against the config's own precomputation.

        The config already stores `voxel_counts_at_run_spacing`. Recomputing and comparing is what
        turns "we assumed the same spacing" into "we verified it" — a mismatch means the volume
        did not land on the grid the config was written for, which would otherwise only show up
        as slightly wrong masks.
        """
        pp = head.postprocess
        counts = {
            "min_object": int(round(float(pp.get("min_object_volume_mm3", 0.0)) / voxel_volume_mm3)),
            "min_hole": int(round(float(pp.get("min_hole_volume_mm3", 0.0)) / voxel_volume_mm3)),
        }
        stated = pp.get("voxel_counts_at_run_spacing") or {}
        expect_obj = stated.get("th_restrain_small_objects_segmentation_output_3d")
        expect_hole = stated.get("th_restrain_small_holes_segmentation_output_3d")
        for got, want, what in ((counts["min_object"], expect_obj, "min_object"),
                                (counts["min_hole"], expect_hole, "min_hole")):
            if want is not None and int(want) != got:
                raise ValueError(
                    f"head '{head.name}': {what} recomputes to {got} voxels at "
                    f"{voxel_volume_mm3} mm3/voxel, but the registered config states {want}. "
                    "The prepared volume is not on the spacing this config was written for.")
        return counts

    def _clean(self, mask: np.ndarray, counts: Dict[str, int]) -> np.ndarray:
        out = mask.astype(bool)
        if counts["min_object"] > 1 and out.any():
            out = remove_small_objects(out, min_size=counts["min_object"])
        if counts["min_hole"] > 1 and out.any():
            out = remove_small_holes(out, area_threshold=counts["min_hole"])
        return out

    def decide(self, logits: Dict[str, np.ndarray], voxel_volume_mm3: float) -> Dict[str, np.ndarray]:
        """Logits -> label maps, at the model's own spacing. See plan section 4.

        Which rule runs is decided by the head's own config — `activation`, `class_assignment`
        and `applies_threshold` — never by the head's name or by its position in the list. A
        retrain that changes any of those changes the behaviour here without a code change.
        """
        out: Dict[str, np.ndarray] = {}
        for head in self.heads:
            head.validate()
            z = logits[head.name]
            counts = self._voxel_counts(head, voxel_volume_mm3)

            if head.class_assignment == ASSIGNMENT_ARGMAX:
                result = self._decide_argmax(head, z, counts)
            elif head.is_multiclass:
                result = self._decide_softmax_threshold(head, z, counts)
            else:
                result = self._decide_sigmoid(head, z, counts)

            out[head.name] = result
            n = int((result > 0).sum())
            log.info(f"      [landmark] head '{head.name}' [{head.describe_rule()}] -> "
                     f"labels {sorted(np.unique(result).tolist())}, {n} vox "
                     f"({n * voxel_volume_mm3 / 1000:.1f} mL)")
        return out

    def _decide_argmax(self, head: HeadSpec, z: np.ndarray, counts) -> np.ndarray:
        """ONE joint reduction over the channels; no threshold anywhere.

        The head mask is exactly the union of its classes, so no voxel is foreground without a
        class. Cleanup runs PER CLASS so a spurious blob on one side cannot be rescued by the
        other and the partition is preserved.
        """
        labels = np.argmax(z, axis=0)
        result = np.zeros(labels.shape, dtype=np.uint8)
        for ch, value in enumerate(head.label_values(), start=1):
            result[self._clean(labels == ch, counts)] = value
        return result

    def _decide_softmax_threshold(self, head: HeadSpec, z: np.ndarray, counts) -> np.ndarray:
        """Foreground from the head row (`1 - p(bg)`), class within it by argmax."""
        t = torch.from_numpy(z)
        fg = torch.sigmoid(torch.logsumexp(t[1:], dim=0) - t[0]).numpy() > head.threshold
        fg = self._clean(fg, counts)
        sides = np.argmax(z[1:], axis=0)
        result = np.zeros(fg.shape, dtype=np.uint8)
        for ch, value in enumerate(head.label_values()):
            result[fg & (sides == ch)] = value
        return result

    def _decide_sigmoid(self, head: HeadSpec, z: np.ndarray, counts) -> np.ndarray:
        """Independent binary channels. One channel is the ordinary binary head."""
        p = torch.sigmoid(torch.from_numpy(z)).numpy()
        if head.channels == 1:
            return self._clean(p[0] > head.threshold, counts).astype(np.uint8)
        result = np.zeros(p.shape[1:], dtype=np.uint8)
        overlap = np.zeros(p.shape[1:], dtype=bool)
        for ch, value in reversed(list(enumerate(head.label_values()))):
            mask = self._clean(p[ch] > head.threshold, counts)
            overlap |= (result > 0) & mask
            result[mask] = value            # earlier channels win: written last
        if overlap.any():
            log.warning(f"   [!] head '{head.name}': {int(overlap.sum())} voxels claimed by more "
                        "than one sigmoid channel; the earlier channel wins.")
        return result

    # ------------------------------- public entry point ---------------------------- #
    def predict(self, source_path: str, work_dir: Optional[str] = None,
                coverage_override: Optional[str] = None,
                keep_halves: bool = False,
                coverage_verdict: Optional[CoverageVerdict] = None) -> Dict[str, Any]:
        """Masks for one exam volume, on the SOURCE grid, plus the audit trail.

        Both paths return the SAME shape of result — masks as SimpleITK images on the source
        grid — so everything downstream of here is identical whichever model is loaded. That is
        the point of putting the per-breast machinery behind this call rather than in the runner.
        """
        self.load()
        if self.input_coverage == COVERAGE_PER_BREAST:
            if not work_dir:
                raise ValueError(
                    f"{self.cfg.registered_model_name} v{self.version} is a per-breast model and "
                    "needs `work_dir` to write its Breast Halves into. Pass one (the runner uses "
                    "`per_breast.halves_work_dir(outdir)`).")
            return self._predict_per_breast(source_path, work_dir, coverage_override,
                                           keep_halves, coverage_verdict)
        return self._predict_whole_exam(source_path)

    def _predict_per_breast(self, source_path: str, work_dir: str,
                            coverage_override: Optional[str],
                            keep_halves: bool = False,
                            coverage_verdict: Optional[CoverageVerdict] = None) -> Dict[str, Any]:
        """Split into Breast Halves, run each, fuse back onto the parent grid.

        An exception in EITHER half fails the whole exam. That is deliberate: the surviving
        half's mask is not "the breast mask for this exam", it is half of one, and an exam masked
        on one side and unmasked on the other is worse than a uniform classical fallback and much
        harder to notice downstream. An EMPTY half is a different thing — a model finding, not an
        infrastructure failure — and it flows through as a missing side, which
        `inference_registry`'s `sides_found < 2` check already reports.
        """
        verdict = coverage_verdict or resolve_coverage(source_path, coverage_override)
        halves = split_to_halves(source_path, verdict, work_dir)
        parent = read_image(source_path)

        decided_halves = []
        for side, half_path, split_geometry in halves:
            log.info(f"      [landmark] --- Breast Half {side!r} ---")
            prepared = self.prepare(half_path)
            logits = self.predict_logits(prepared)
            decided = self.decide(logits, prepared.voxel_volume_mm3)
            decided_halves.append((side, decided, prepared.model_grid, split_geometry))

        head_names = [h.name for h in self.heads]
        masks = fuse_halves(decided_halves, parent, head_names)
        if REGION_HEAD not in masks:
            raise ValueError(
                f"{self.cfg.registered_model_name} v{self.version} declares heads "
                f"{sorted(masks)} and no {REGION_HEAD!r}; per-breast side restoration needs the "
                "breast-region head to know which voxels belong to which Breast Half.")
        sides = sides_from_fused(masks[REGION_HEAD], verdict)

        # Only once fusion has succeeded — see cleanup_halves.
        if not keep_halves:
            cleanup_halves([p for _, p, _ in halves])

        return {
            "masks": masks,
            "sides": sides,
            "coverage": verdict.as_dict(),
            "halves": [{"side": s, "path": p} for s, p, _ in halves],
            "model": {"name": self.cfg.registered_model_name, "version": self.version,
                      "run_id": self.run_id,
                      "fusion_version": PER_BREAST_FUSION_VERSION},
        }

    def _predict_whole_exam(self, source_path: str) -> Dict[str, Any]:
        """The pre-existing path: one pass over the whole exam, masks back on the source grid."""
        prepared = self.prepare(source_path)
        logits = self.predict_logits(prepared)
        decided = self.decide(logits, prepared.voxel_volume_mm3)
        masks = {name: resample_like(arr.astype(np.uint8), reference=prepared.source,
                                     source_geometry=prepared.model_grid)
                 for name, arr in decided.items()}
        return {
            "masks": masks,
            "decided_model_grid": decided,
            "prepared": prepared,
            "model": {"name": self.cfg.registered_model_name, "version": self.version,
                      "run_id": self.run_id},
        }
