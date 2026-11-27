"""Run sliding-window segmentation and restore predicted masks."""

from functools import lru_cache

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "compute_sliding_window_steps",
    "sliding_window_slicers",
    "gaussian_importance_map",
    "pad_to_patch",
    "sliding_window_predict",
]

def compute_sliding_window_steps(
    image_shape_zyx: Sequence[int],
    patch_size: Sequence[int],
    tile_step_size: float = 0.5,
) -> List[List[int]]:
    """Per-axis start offsets for the tiles covering `image_shape_zyx`.

    Reproduces nnU-Net's `compute_steps_for_sliding_window`: the step is *at most*
    `patch * tile_step_size`, but is then shrunk so the tiles spread evenly and the last one ends
    exactly at the volume edge. That even spread is why overlap is often free — see the note in
    `sliding_window_predict`.
    """
    if not (0 < tile_step_size <= 1):
        raise ValueError(f"tile_step_size must be in (0, 1], got {tile_step_size}")
    if any(i < j for i, j in zip(image_shape_zyx, patch_size)):
        raise ValueError(
            f"volume {tuple(image_shape_zyx)} is smaller than patch {tuple(patch_size)} on some "
            f"axis — call pad_to_patch first")

    target_step = [p * tile_step_size for p in patch_size]
    num_steps = [int(np.ceil((i - p) / s)) + 1
                 for i, s, p in zip(image_shape_zyx, target_step, patch_size)]

    steps: List[List[int]] = []
    for dim in range(len(patch_size)):
        max_step_value = image_shape_zyx[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # only one step, at 0 — value is irrelevant
        steps.append([int(np.round(actual_step_size * i)) for i in range(num_steps[dim])])
    return steps

def sliding_window_slicers(
    image_shape_zyx: Sequence[int],
    patch_size: Sequence[int],
    tile_step_size: float = 0.5,
) -> List[Tuple[slice, slice, slice]]:
    """The `(z, y, x)` slices of every tile covering the volume."""
    steps = compute_sliding_window_steps(image_shape_zyx, patch_size, tile_step_size)
    return [(slice(z, z + patch_size[0]),
             slice(y, y + patch_size[1]),
             slice(x, x + patch_size[2]))
            for z in steps[0] for y in steps[1] for x in steps[2]]

@lru_cache(maxsize=8)
def _gaussian_importance_map_cached(patch_size: Tuple[int, ...], sigma_scale: float) -> np.ndarray:
    """The real computation, memoised. See `gaussian_importance_map` for the contract."""
    from scipy.ndimage import gaussian_filter

    tmp = np.zeros(tuple(patch_size), dtype=np.float32)
    centre = tuple(i // 2 for i in patch_size)
    tmp[centre] = 1.0
    sigmas = [i * sigma_scale for i in patch_size]

    g = gaussian_filter(tmp, sigmas, mode="constant", cval=0.0)
    g = g / g.max()
    g = g.astype(np.float32)
    # nnU-Net floors the map at its smallest nonzero value so no voxel gets zero weight — a
    # zero-weight voxel in a single-tile region would divide by zero in the accumulator.
    g[g == 0] = g[g != 0].min()
    g.flags.writeable = False  # the cache hands this out many times; nobody may edit it
    return g

def gaussian_importance_map(patch_size: Sequence[int], sigma_scale: float = 1. / 8) -> np.ndarray:
    """Gaussian tile weighting: centre voxels count more than edge voxels when tiles overlap.

    Edge voxels of a patch see less context and are systematically worse, so weighting them down
    is what makes overlapping tiles help rather than smear. Matches nnU-Net's `compute_gaussian`
    (float32 on CPU; nnU-Net returns float16 on CUDA).

    The map depends on the patch size and on `sigma_scale`, and on nothing else, but
    `sliding_window_predict` asked for it once per VOLUME. At a 128^3 patch the Gaussian filter
    costs about 0.7 s, so a 492-case validation epoch spent about 6 minutes rebuilding the same
    array. The result is now memoised and this function returns a fresh writeable copy of it.
    The copy costs about 2 ms and keeps the old contract: the caller owns what it receives.
    """
    return np.array(_gaussian_importance_map_cached(tuple(int(p) for p in patch_size),
                                                    float(sigma_scale)))

def pad_to_patch(
    data: np.ndarray, patch_size: Sequence[int]
) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """Centre-pad `(C, Z, Y, X)` up to at least `patch_size`.

    Returns the padded array and the slicer that crops a padded result back to the input shape.
    Needed because preprocessed breast volumes are often *smaller* than the patch on some axis.
    """
    zyx = data.shape[1:]
    pads = []
    for dim, p in zip(zyx, patch_size):
        need = max(0, p - dim)
        before = need // 2
        pads.append((before, need - before))
    padded = np.pad(data, [(0, 0)] + pads, mode="constant")
    crop_back = tuple([slice(None)] + [slice(b, b + d) for (b, _), d in zip(pads, zyx)])
    return padded, crop_back

def _as_named_outputs(out, output_names: Optional[Sequence[str]]) -> Dict[str, "object"]:
    """Normalise whatever the forward returned into an ordered `{name: tensor}` mapping."""
    if isinstance(out, dict):
        return dict(out)
    if isinstance(out, (list, tuple)):
        if output_names is None:
            names = [f"out{i}" for i in range(len(out))]
        elif len(output_names) != len(out):
            raise ValueError(
                f"forward returned {len(out)} outputs but output_names has "
                f"{len(output_names)}: {list(output_names)}")
        else:
            names = list(output_names)
        return dict(zip(names, out))
    name = output_names[0] if output_names else "out"
    return {name: out}

def sliding_window_predict(
    forward: Callable[["object"], "object"],
    volume: Union[np.ndarray, "object"],
    patch_size: Sequence[int],
    tile_step_size: float = 0.5,
    device=None,
    autocast_dtype=None,
    output_names: Optional[Sequence[str]] = None,
    return_numpy: bool = True,
) -> Dict[str, Union[np.ndarray, "object"]]:
    """Predict a whole volume by Gaussian-blended sliding window.

    Args:
        forward: callable taking a batched patch `(1, C, *patch_size)` and returning a tensor,
            a tuple/list of tensors, or a `{name: tensor}` dict — each `(1, C_out, *patch_size)`.
            For a model with an inference flag, pass a lambda: ``lambda x: model(x, inference_mode=True)``.
        volume: `(C, Z, Y, X)` preprocessed array (numpy or torch). Padded up to `patch_size`
            automatically, and the result is cropped back to this shape.
        patch_size: `(z, y, x)`. Must suit the architecture — nets that downsample 5× need
            multiples of 32. This is a *config* value; do not read it off a checkpoint's plan,
            or the patch silently changes when the checkpoint does.
        tile_step_size: fraction of the patch between tile starts. 0.5 is nnU-Net's default.
        device: where tiles are run. `None` = wherever the model already lives. Passing a device
            that differs from the model's gives a mixed-device conv3d, which silently falls back
            to `slow_conv3d` instead of erroring.
        autocast_dtype: `None` (default) runs in the model's own dtype — fp32 for a normal model.
            Pass `torch.float16` / `torch.bfloat16` to run tiles under autocast; accumulation
            stays fp32 regardless, so blending never loses precision.
        output_names: names for tuple/tensor returns; ignored when `forward` returns a dict.
        return_numpy: `True` returns numpy arrays; `False` keeps torch tensors on `device`.

    Returns:
        `{name: (C_out, Z, Y, X)}` of blended **logits**, cropped back to the input shape.

    Note:
        Overlap is frequently free. Because the steps are spread evenly (see
        `compute_sliding_window_steps`), a volume that only slightly exceeds the patch yields the
        same tile grid at `tile_step_size` 1.0 and 0.5 — identical cost. Check `len(slicers)`
        before assuming a smaller step buys anything.
    """
    import torch

    patch = tuple(int(p) for p in patch_size)
    if len(patch) != 3:
        raise ValueError(f"patch_size must be 3 values (z, y, x), got {patch}")

    if not isinstance(volume, np.ndarray):
        volume = volume.detach().cpu().numpy()
    if volume.ndim != 4:
        raise ValueError(f"volume must be (C, Z, Y, X), got shape {volume.shape}")

    padded, crop_back = pad_to_patch(volume, patch)
    zyx = padded.shape[1:]
    slicers = sliding_window_slicers(zyx, patch, tile_step_size)

    # A single tile covering the whole volume needs no blending: the weights would cancel in
    # `(x * g) / g`. Skipping is not just faster — that round trip perturbs every voxel by ~1e-7,
    # and a logit sitting exactly on the decision boundary can then threshold differently than a
    # direct forward pass would, changing a connected component and with it Dice/TPR. Skipping
    # makes the one-tile case EXACTLY equal to calling the model directly.
    single_full_tile = len(slicers) == 1 and all(
        sl.start == 0 and sl.stop >= dim for sl, dim in zip(slicers[0], zyx))

    gauss = None if single_full_tile else torch.as_tensor(
        gaussian_importance_map(patch), dtype=torch.float32, device=device)
    pv = torch.as_tensor(padded, dtype=torch.float32, device=device)

    # Accumulators are allocated lazily from the first tile: the number of outputs and their
    # channel counts are properties of the model, which this function deliberately does not know.
    accs: Dict[str, "torch.Tensor"] = {}
    wsum = torch.zeros((1, *zyx), dtype=torch.float32, device=device)

    use_autocast = autocast_dtype is not None
    dev_type = (pv.device.type if device is None else torch.device(device).type)

    with torch.no_grad():
        for sl in slicers:
            x = pv[(slice(None), *sl)][None]  # (1, C, *patch)
            if use_autocast:
                with torch.autocast(device_type=dev_type, dtype=autocast_dtype):
                    raw = forward(x)
            else:
                raw = forward(x)
            outputs = _as_named_outputs(raw, output_names)

            for name, tensor in outputs.items():
                t = tensor[0].to(torch.float32)  # drop batch dim; accumulate in fp32 always
                if t.shape[1:] != patch:
                    raise ValueError(
                        f"output '{name}' has spatial shape {tuple(t.shape[1:])} but the patch is "
                        f"{patch} — patch-based stitching needs the output on the input grid")
                if name not in accs:
                    accs[name] = torch.zeros((t.shape[0], *zyx), dtype=torch.float32,
                                             device=t.device)
                accs[name][(slice(None), *sl)] += t if gauss is None else t * gauss
            wsum[(slice(None), *sl)] += 1.0 if gauss is None else gauss

    spatial = crop_back[1:]  # the slices that undo the padding
    result = {}
    for name, acc in accs.items():
        blended = acc if gauss is None else acc / wsum.to(acc.device)
        blended = blended[(slice(None), *spatial)]
        result[name] = blended.cpu().numpy() if return_numpy else blended
    return result
