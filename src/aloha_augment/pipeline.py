"""LeRobot v3 dataset augmentation CLI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import v2

from .prefilter import filter_episodes, save_filter_scores
from .transforms import (
    DriftingBlob,
    FrameDecimator,
    FrameStride,
    HorizontalFlipWithActionMirror,
    ROBOT_PRESETS,
    StaticErasing,
)


def _walk_transforms(transform):
    if transform is None:
        return
    if hasattr(transform, "transforms"):
        for child in transform.transforms:
            yield from _walk_transforms(child)
        return
    yield transform


def reset_transform_state(transform):
    for t in _walk_transforms(transform):
        if hasattr(t, "reset_episode"):
            t.reset_episode()


def set_transform_camera(transform, camera_key: str):
    for t in _walk_transforms(transform):
        if hasattr(t, "set_camera_key"):
            t.set_camera_key(camera_key)


def build_color_jitter(args):
    return v2.ColorJitter(
        brightness=tuple(args.brightness),
        contrast=tuple(args.contrast),
        saturation=tuple(args.saturation),
        hue=tuple(args.hue),
    )


def build_gaussian_blur(args):
    return v2.GaussianBlur(kernel_size=args.blur_kernel, sigma=tuple(args.blur_sigma))


def build_sharpness(args):
    return v2.RandomAdjustSharpness(sharpness_factor=args.sharpness_factor, p=1.0)


def build_random_erasing(args):
    return v2.RandomErasing(p=args.erasing_p, scale=tuple(args.erasing_scale))


def build_static_erasing(args):
    return StaticErasing(scale=tuple(args.erasing_scale))


def build_frame_decimate(args):
    return FrameDecimator(remove_every_n=args.remove_every_n)


def build_drifting_blob(args):
    return DriftingBlob(
        radius=args.blob_radius,
        speed=args.blob_speed,
        softness=args.blob_softness,
        opacity=args.blob_opacity,
    )


def build_horizontal_flip(args):
    if args.robot_type and args.robot_type in ROBOT_PRESETS:
        preset = ROBOT_PRESETS[args.robot_type]
        return HorizontalFlipWithActionMirror(
            action_mirror_mask=args.action_mirror_mask or preset["action_mirror_mask"],
            state_mirror_mask=args.state_mirror_mask or preset["state_mirror_mask"],
            swap_action_ranges=preset.get("swap_action_ranges"),
            swap_state_ranges=preset.get("swap_state_ranges"),
        )
    if not args.action_mirror_mask or not args.state_mirror_mask:
        raise SystemExit("Error: --action-mirror-mask and --state-mirror-mask are required when --robot-type is not specified.")
    return HorizontalFlipWithActionMirror(args.action_mirror_mask, args.state_mirror_mask)


def build_sam3(args):
    from .sam3_augmentation import SAM3BackgroundCompositor

    return SAM3BackgroundCompositor(
        feather_radius=args.sam3_feather_radius,
        brightness_threshold=args.sam3_brightness_threshold,
        background_history=args.sam3_background_history,
        top_masks=getattr(args, "sam3_top_masks", 4),
        mask_iou_threshold=getattr(args, "sam3_mask_iou_threshold", 0.75),
        text_prompt=getattr(args, "sam3_text_prompt", "plastic cup, lid, robot hand"),
    )


AUGMENTATION_BUILDERS = {
    "color_jitter": build_color_jitter,
    "gaussian_blur": build_gaussian_blur,
    "sharpness": build_sharpness,
    "random_erasing": build_random_erasing,
    "static_erasing": build_static_erasing,
    "frame_decimate": build_frame_decimate,
    "drifting_blob": build_drifting_blob,
    "horizontal_flip": build_horizontal_flip,
    "sam3": build_sam3,
}


def build_transform(args, augmentations=None):
    if augmentations is None:
        augmentations = args.augmentations
    transforms = []
    for name in augmentations:
        if name not in AUGMENTATION_BUILDERS:
            raise SystemExit(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_BUILDERS.keys())}")
        transforms.append(AUGMENTATION_BUILDERS[name](args))

    if not transforms:
        return None
    if len(transforms) == 1:
        return transforms[0]
    return v2.Compose(transforms)


def _parser_defaults(parser):
    return {action.dest: action.default for action in parser._actions if action.dest != "help"}


TIER_PRESETS = {
    "tier1": {
        "include_originals": False,
        "num_passes": 1,
        "augmentations": [],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 4,
        "frame_stride_cycle": None,
        "skip_prefilter": True,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.02,
        "max_action_jerk": None,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.15,
        "tail_drop_max": 0,
    },
    "tier2": {
        "include_originals": True,
        "num_passes": 3,
        "augmentations": ["color_jitter", "gaussian_blur", "sharpness", "random_erasing", "horizontal_flip"],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 3,
        "frame_stride_cycle": None,
        "skip_prefilter": False,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.0188,
        "max_action_jerk": 0.0138,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.12,
        "tail_drop_max": 12,
    },
    "tier3": {
        "include_originals": False,
        "num_passes": 3,
        "augmentations": [
            "color_jitter",
            "gaussian_blur",
            "sharpness",
            "random_erasing",
            "drifting_blob",
            "static_erasing",
            "horizontal_flip",
        ],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 1,
        "frame_stride_cycle": [2, 3, 4],
        "skip_prefilter": True,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.03,
        "max_action_jerk": None,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.10,
        "tail_drop_max": 12,
    },
    # Canonical recipe for jgiegold/aloha_balanced_v7
    "v7": {
        "include_originals": True,
        "include_originals_decimated": True,
        "num_passes": 1,
        "augmentations": ["color_jitter", "gaussian_blur", "sharpness", "random_erasing", "horizontal_flip"],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 3,
        "frame_stride_cycle": None,
        "skip_prefilter": False,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.015,
        "max_action_jerk": 0.0145,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.12,
        "tail_drop_max": 8,
        "temporal_jitter_pct": 0.15,
        "action_noise_std": 0.003,
        "action_smoothing": "savgol",
        "savgol_window_length": 7,
        "savgol_polyorder": 2,
        "smooth_exclude_indices": [6, 13],
    },
    # v7 + SAM3 text-prompted background compositing (requires transformers with Sam3Model)
    "v8": {
        "include_originals": True,
        "include_originals_decimated": True,
        "num_passes": 1,
        "augmentations": ["color_jitter", "gaussian_blur", "sharpness", "random_erasing", "horizontal_flip", "sam3"],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 3,
        "frame_stride_cycle": None,
        "skip_prefilter": False,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.015,
        "max_action_jerk": 0.0145,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.12,
        "tail_drop_max": 8,
        "temporal_jitter_pct": 0.15,
        "action_noise_std": 0.003,
        "action_smoothing": "savgol",
        "savgol_window_length": 7,
        "savgol_polyorder": 2,
        "smooth_exclude_indices": [6, 13],
    },
    # v7 + SAM2 background compositing (requires sam2 package)
    "v7_sam": {
        "include_originals": True,
        "include_originals_decimated": True,
        "num_passes": 1,
        "augmentations": ["color_jitter", "gaussian_blur", "sharpness", "random_erasing", "horizontal_flip", "sam3"],
        "robot_type": "aloha",
        "action_shift": 4,
        "keep_every_n": 3,
        "frame_stride_cycle": None,
        "skip_prefilter": False,
        "prefilter_mode": "fast",
        "prefilter_sample_every_n": 8,
        "min_action_delta": 0.015,
        "max_action_jerk": 0.0145,
        "sparc_threshold": -10.0,
        "saturation_threshold": 0.12,
        "tail_drop_max": 8,
        "temporal_jitter_pct": 0.15,
        "action_noise_std": 0.003,
        "action_smoothing": "savgol",
        "savgol_window_length": 7,
        "savgol_polyorder": 2,
        "smooth_exclude_indices": [6, 13],
    },
}


def apply_tier_configuration(args, defaults):
    if not args.tier:
        return args

    preset = TIER_PRESETS[args.tier]
    for key, value in preset.items():
        setattr(args, key, value)

    return args


def get_temporal_selector(args, pass_idx: int):
    if args.frame_stride_cycle:
        stride = args.frame_stride_cycle[pass_idx % len(args.frame_stride_cycle)]
    else:
        stride = args.keep_every_n

    if stride is None or stride <= 1:
        return None

    offset = pass_idx % stride
    return FrameStride(keep_every_n=stride, start_offset=offset)


def compute_effective_action_shift(action_shift: int, temporal_selector) -> int:
    """Project action shift into selected-frame space."""
    if action_shift <= 0:
        return 0
    stride = 1
    if temporal_selector is not None and hasattr(temporal_selector, "keep_every_n"):
        stride = max(1, int(temporal_selector.keep_every_n))
    return int(round(action_shift / stride))


def _normalize_value(value):
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    return value


def _episode_record(episodes, ep_idx: int):
    if hasattr(episodes, "iloc"):
        return episodes.iloc[ep_idx]
    return episodes[ep_idx]


def get_episode_range(source, ep_idx):
    ep = _episode_record(source.meta.episodes, ep_idx)
    if hasattr(ep, "to_dict"):
        ep = ep.to_dict()
    return ep["dataset_from_index"], ep["dataset_to_index"]


def build_frame_dict(item, feature_keys, features_meta, action_override=None):
    frame = {"task": item.get("task", "robot manipulation")}
    for key in feature_keys:
        if key in {"timestamp", "index", "episode_index", "frame_index", "task_index"}:
            continue
        if key == "action" and action_override is not None:
            val = action_override
        elif key in item:
            val = item[key]
        else:
            continue

        val = _normalize_value(val)
        if hasattr(val, "ndim") and val.ndim == 3 and val.shape[0] == 3:
            val = np.transpose(val, (1, 2, 0))
        if key in features_meta:
            expected_shape = tuple(features_meta[key]["shape"])
            if getattr(val, "ndim", None) == 0 and expected_shape == (1,):
                val = val.reshape(1)
        frame[key] = val
    return frame


def _lerp_value(v0, v1, alpha: float):
    return v0 * (1.0 - alpha) + v1 * alpha


def _interp_key_at_pos(source, selected_indices, pos: float, key: str):
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    lo = max(0, min(lo, len(selected_indices) - 1))
    hi = max(0, min(hi, len(selected_indices) - 1))
    alpha = float(pos - lo)

    item_lo = source[selected_indices[lo]]
    if key not in item_lo:
        return None
    v0 = item_lo[key]
    if lo == hi or alpha <= 1e-8:
        return v0

    item_hi = source[selected_indices[hi]]
    if key not in item_hi:
        return v0
    v1 = item_hi[key]
    return _lerp_value(v0, v1, alpha)


def _add_action_noise(action, std: float):
    if std <= 0.0 or action is None:
        return action
    if isinstance(action, torch.Tensor):
        return action + torch.randn_like(action) * std
    return action + np.random.normal(0.0, std, size=action.shape).astype(action.dtype)


def _smooth_action_sequence(actions, method: str, window_length: int, polyorder: int, exclude_indices=None):
    if method == "none" or not actions:
        return actions

    if any(a is None for a in actions):
        return actions

    if method != "savgol":
        raise SystemExit(f"Error: unsupported action smoothing method: {method}")

    try:
        from scipy.signal import savgol_filter
    except Exception as exc:
        raise SystemExit("Error: scipy is required for --action-smoothing savgol. Install optional pipeline deps with scipy.") from exc

    exclude_indices = sorted(set(exclude_indices or []))
    first = actions[0]

    n_steps = len(actions)
    if n_steps < 3:
        return actions

    effective_window = min(int(window_length), n_steps)
    if effective_window % 2 == 0:
        effective_window -= 1
    if effective_window <= polyorder:
        effective_window = polyorder + 1
        if effective_window % 2 == 0:
            effective_window += 1
    if effective_window > n_steps:
        effective_window = n_steps if n_steps % 2 == 1 else n_steps - 1
    if effective_window <= polyorder or effective_window < 3:
        return actions

    if isinstance(first, torch.Tensor):
        stacked = torch.stack(actions, dim=0)
        dtype = stacked.dtype
        smoothed = stacked.to(dtype=torch.float32).cpu().numpy()
        dim = smoothed.shape[1] if smoothed.ndim >= 2 else 0
        smooth_mask = np.ones(dim, dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < dim:
                smooth_mask[idx] = False

        if np.any(smooth_mask):
            smoothed[:, smooth_mask] = savgol_filter(
                smoothed[:, smooth_mask],
                window_length=effective_window,
                polyorder=polyorder,
                axis=0,
                mode="interp",
            )

        smoothed_t = torch.from_numpy(smoothed).to(dtype=dtype)
        return [smoothed_t[t] for t in range(smoothed_t.shape[0])]

    stacked = np.stack(actions, axis=0).astype(np.float32, copy=False)
    smoothed = stacked.copy()
    dim = smoothed.shape[1] if smoothed.ndim >= 2 else 0
    smooth_mask = np.ones(dim, dtype=bool)
    for idx in exclude_indices:
        if 0 <= idx < dim:
            smooth_mask[idx] = False

    if np.any(smooth_mask):
        smoothed[:, smooth_mask] = savgol_filter(
            smoothed[:, smooth_mask],
            window_length=effective_window,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )

    smoothed = smoothed.astype(actions[0].dtype, copy=False)
    return [smoothed[t] for t in range(smoothed.shape[0])]


def resolve_smoothing_exclude_indices(args):
    if args.smooth_exclude_indices is not None:
        return args.smooth_exclude_indices
    if args.robot_type == "aloha":
        # ALOHA gripper dimensions are binary-like and should not be low-pass filtered.
        return [6, 13]
    return []


def write_episode(
    source,
    output,
    ep_idx,
    feature_keys,
    camera_keys,
    features_meta,
    action_shift=0,
    temporal_selector=None,
    transform=None,
    flip=None,
    tail_drop_max=0,
    temporal_jitter_pct=0.0,
    action_noise_std=0.0,
    action_smoothing="none",
    savgol_window_length=7,
    savgol_polyorder=2,
    smooth_exclude_indices=None,
):
    from_idx, to_idx = get_episode_range(source, ep_idx)
    tail_drop = int(np.random.randint(0, tail_drop_max + 1)) if tail_drop_max > 0 else 0
    effective_to_idx = to_idx - tail_drop

    if effective_to_idx <= from_idx:
        return

    selected_indices = []
    for local_idx, global_idx in enumerate(range(from_idx, effective_to_idx)):
        if temporal_selector is not None and not temporal_selector.should_keep(local_idx):
            continue
        selected_indices.append(global_idx)

    if not selected_indices:
        return

    effective_shift = compute_effective_action_shift(action_shift, temporal_selector)
    n_selected = len(selected_indices)

    positions = np.arange(n_selected, dtype=np.float32)
    if temporal_jitter_pct > 0.0 and n_selected > 1:
        speed_scale = float(np.random.uniform(1.0 - temporal_jitter_pct, 1.0 + temporal_jitter_pct))
        target_len = max(2, int(round(n_selected * speed_scale)))
        positions = np.linspace(0.0, float(n_selected - 1), num=target_len, dtype=np.float32)

    usable_len = len(positions) - effective_shift
    if usable_len <= 0:
        return

    planned_actions = []
    for out_idx in range(usable_len):
        obs_pos = float(positions[out_idx])
        obs_seq_idx = int(round(obs_pos))
        obs_seq_idx = max(0, min(obs_seq_idx, n_selected - 1))
        global_idx = selected_indices[obs_seq_idx]
        item = source[global_idx]

        action_override = None
        if effective_shift:
            action_pos = float(positions[out_idx + effective_shift])
            action_override = _interp_key_at_pos(source, selected_indices, action_pos, "action")
        elif "action" in item:
            action_override = item["action"]

        planned_actions.append(action_override)

    planned_actions = _smooth_action_sequence(
        planned_actions,
        method=action_smoothing,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
        exclude_indices=smooth_exclude_indices,
    )

    if transform is not None:
        reset_transform_state(transform)

    if isinstance(transform, (StaticErasing, DriftingBlob)):
        first = source[from_idx]
        for cam_key in camera_keys:
            if cam_key in first:
                _, h, w = first[cam_key].shape
                transform.resample(h, w)
                break

    for out_idx in range(usable_len):
        obs_pos = float(positions[out_idx])
        obs_seq_idx = int(round(obs_pos))
        obs_seq_idx = max(0, min(obs_seq_idx, n_selected - 1))
        global_idx = selected_indices[obs_seq_idx]

        item = dict(source[global_idx])
        action_override = planned_actions[out_idx]

        if temporal_jitter_pct > 0.0:
            interp_state = _interp_key_at_pos(source, selected_indices, obs_pos, "observation.state")
            if interp_state is not None:
                item["observation.state"] = interp_state

        action_override = _add_action_noise(action_override, action_noise_std)

        for cam_key in camera_keys:
            if cam_key in item:
                if flip is not None:
                    item[cam_key] = flip.flip_image(item[cam_key])
                if transform is not None:
                    set_transform_camera(transform, cam_key)
                    item[cam_key] = transform(item[cam_key])

        if flip is not None:
            if action_override is not None:
                action_override = flip.mirror_actions(action_override)
            elif "action" in item:
                item["action"] = flip.mirror_actions(item["action"])
            if "observation.state" in item:
                item["observation.state"] = flip.mirror_state(item["observation.state"])

        output.add_frame(build_frame_dict(item, feature_keys, features_meta, action_override=action_override))

    output.save_episode()


def build_parser():
    parser = argparse.ArgumentParser(description="Augment a LeRobot v3 dataset and push to Hugging Face Hub")

    parser.add_argument("--source", required=True, help="Source dataset repo_id (e.g. lerobot/aloha_static_cups_open)")
    parser.add_argument("--output", required=True, help="Output dataset repo_id (e.g. user/dataset_augmented)")
    parser.add_argument("--tier", choices=list(TIER_PRESETS.keys()), default=None, help="Tier preset: tier1, tier2, or tier3")
    parser.add_argument("--num-passes", type=int, default=2, help="Number of augmented copies per episode")
    parser.add_argument("--augmentations", nargs="*", default=None, choices=list(AUGMENTATION_BUILDERS.keys()))
    parser.add_argument("--include-originals", action="store_true")
    parser.add_argument(
        "--include-originals-decimated",
        action="store_true",
        help="When copying originals, apply the same temporal selector/action-shift as augmented passes",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--action-shift", type=int, default=0, help="Shift action labels forward by this many frames")
    parser.add_argument("--keep-every-n", type=int, default=1, help="Keep every Nth frame for temporal downsampling")
    parser.add_argument(
        "--tail-drop-max",
        type=int,
        default=0,
        help="Randomly drop up to this many trailing frames per output episode to preserve length diversity",
    )
    parser.add_argument(
        "--frame-stride-cycle",
        nargs="+",
        type=int,
        default=None,
        help="Cycle through different keep-every-N values across passes",
    )

    parser.add_argument("--brightness", nargs=2, type=float, default=[0.8, 1.2])
    parser.add_argument("--contrast", nargs=2, type=float, default=[0.8, 1.2])
    parser.add_argument("--saturation", nargs=2, type=float, default=[0.5, 1.5])
    parser.add_argument("--hue", nargs=2, type=float, default=[-0.05, 0.05])

    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--blur-sigma", nargs=2, type=float, default=[0.1, 2.0])
    parser.add_argument("--sharpness-factor", type=float, default=2.0)
    parser.add_argument("--erasing-p", type=float, default=0.5)
    parser.add_argument("--erasing-scale", nargs=2, type=float, default=[0.02, 0.15])
    parser.add_argument("--blob-radius", type=int, default=30)
    parser.add_argument("--blob-speed", type=float, default=2.0)
    parser.add_argument("--blob-softness", type=float, default=0.6)
    parser.add_argument("--blob-opacity", type=float, default=0.5)
    parser.add_argument("--remove-every-n", type=int, default=5)
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Gaussian std for additive motor noise on actions (e.g. 0.003)",
    )
    parser.add_argument(
        "--temporal-jitter-pct",
        type=float,
        default=0.0,
        help="Per-episode temporal length jitter fraction in [0,1] (e.g. 0.15 for +/-15%)",
    )
    parser.add_argument(
        "--action-smoothing",
        choices=["none", "savgol"],
        default="none",
        help="Optional low-pass smoothing for action trajectories",
    )
    parser.add_argument(
        "--savgol-window-length",
        type=int,
        default=7,
        help="Savitzky-Golay window length (odd integer, auto-clamped per episode)",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order",
    )
    parser.add_argument(
        "--smooth-exclude-indices",
        nargs="+",
        type=int,
        default=None,
        help="Action dimensions to exclude from smoothing (defaults include ALOHA grippers)",
    )

    parser.add_argument("--robot-type", type=str, default=None, choices=list(ROBOT_PRESETS.keys()))
    parser.add_argument("--action-mirror-mask", nargs="+", type=float, default=None)
    parser.add_argument("--state-mirror-mask", nargs="+", type=float, default=None)

    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "video_reader", "torchcodec"])
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    # Pre-filtering options
    parser.add_argument("--skip-prefilter", action="store_true", help="Skip pre-filtering (SPARC + saturation)")
    parser.add_argument("--prefilter-mode", choices=["full", "fast"], default="full", help="Use a fast sampled prefilter path")
    parser.add_argument("--prefilter-sample-every-n", type=int, default=1, help="Sample every Nth frame in fast prefilter mode")
    parser.add_argument("--sparc-threshold", type=float, default=-10.0, help="SPARC threshold for smoothness")
    parser.add_argument("--saturation-threshold", type=float, default=0.15, help="Max saturation fraction")
    parser.add_argument("--min-action-delta", type=float, default=0.02, help="Minimum mean absolute action delta to keep an episode")
    parser.add_argument(
        "--max-action-jerk",
        type=float,
        default=None,
        help="Maximum mean absolute second-order action difference to keep an episode",
    )
    
    # SAM3 augmentation options
    parser.add_argument("--skip-sam3", action="store_true", help="Skip SAM3 background augmentation")
    parser.add_argument("--sam3-feather-radius", type=int, default=10, help="Feather radius for SAM3 compositing edges")
    parser.add_argument("--sam3-brightness-threshold", type=int, default=100, help="Fallback mask threshold when SAM3 predictor is unavailable")
    parser.add_argument("--sam3-background-history", type=int, default=24, help="Number of recent frames to use as SAM3 background candidates")
    parser.add_argument("--sam3-top-masks", type=int, default=4, help="Number of largest SAM2 masks to union as foreground")
    parser.add_argument("--sam3-mask-iou-threshold", type=float, default=0.75, help="Minimum predicted_iou to include a SAM2 mask in the foreground")
    parser.add_argument("--sam3-text-prompt", type=str, default="plastic cup, lid, robot hand", help="Comma-separated text prompt for SAM3 object segmentation")

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    defaults = _parser_defaults(parser)
    if args.augmentations is None:
        args.augmentations = ["color_jitter"]
    if args.skip_sam3:
        args.augmentations = [a for a in args.augmentations if a != "sam3"]
    if args.temporal_jitter_pct < 0 or args.temporal_jitter_pct > 1:
        raise SystemExit("Error: --temporal-jitter-pct must be in [0, 1].")
    if args.savgol_window_length < 3:
        raise SystemExit("Error: --savgol-window-length must be >= 3.")
    if args.savgol_polyorder < 1:
        raise SystemExit("Error: --savgol-polyorder must be >= 1.")
    return args, defaults


def main():
    args, defaults = parse_args()
    apply_tier_configuration(args, defaults)
    t0 = time.time()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.force:
        import shutil

        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / args.output
        if cache_dir.exists():
            print(f"Removing existing cache: {cache_dir}")
            shutil.rmtree(cache_dir)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_FEATURES

    print(f"Loading source dataset: {args.source}")
    source = LeRobotDataset(args.source, video_backend=args.video_backend)

    camera_keys = source.meta.camera_keys
    feature_keys = list(source.meta.features.keys())
    episode_indices = args.episodes if args.episodes else list(range(source.meta.total_episodes))
    
    # Pre-filtering: SPARC smoothness + saturation check
    if not args.skip_prefilter:
        print("\n=== Pre-filtering episodes ===")
        kept_indices, filter_scores = filter_episodes(
            source,
            episode_indices=episode_indices,
            sparc_threshold=args.sparc_threshold,
            saturation_threshold_frac=args.saturation_threshold,
            min_action_delta=args.min_action_delta,
            max_action_jerk=args.max_action_jerk,
            mode=args.prefilter_mode,
            sample_every_n=args.prefilter_sample_every_n,
            fps=source.fps,
        )
        
        n_filtered = len(episode_indices) - len(kept_indices)
        print(f"Filtered: {n_filtered} episodes rejected, {len(kept_indices)} kept")
        
        # Save filter scores for inspection
        save_filter_scores(filter_scores, Path.home() / f"filter_scores_{args.output.replace('/', '_')}.json")
        
        episode_indices = kept_indices

    print(f"\n  Episodes: {source.meta.total_episodes} ({len(episode_indices)} selected)")
    print(f"  Frames: {source.meta.total_frames}, FPS: {source.fps}")
    print(f"  Cameras: {camera_keys}")
    print(f"  Robot: {source.meta.robot_type}")
    if args.tier:
        print(f"  Tier: {args.tier}")
        print(f"  Action shift: {args.action_shift}")
        print(f"  Keep every N: {args.keep_every_n}")
        print(f"  Tail drop max: {args.tail_drop_max}")
        print(f"  Prefilter mode: {args.prefilter_mode}")
        print(f"  Prefilter sample every N: {args.prefilter_sample_every_n}")
        print(f"  Min action delta: {args.min_action_delta}")
        print(f"  Max action jerk: {args.max_action_jerk}")
        print(f"  Action noise std: {args.action_noise_std}")
        print(f"  Action smoothing: {args.action_smoothing}")
        if args.action_smoothing == "savgol":
            print(f"  Savgol window length: {args.savgol_window_length}")
            print(f"  Savgol polyorder: {args.savgol_polyorder}")
        print(f"  Temporal jitter pct: {args.temporal_jitter_pct}")
        if args.frame_stride_cycle:
            print(f"  Frame stride cycle: {args.frame_stride_cycle}")

    n_original = len(episode_indices) if args.include_originals else 0
    n_augmented = len(episode_indices) * args.num_passes
    print("\nOutput plan:")
    print(f"  Original episodes: {n_original}")
    print(f"  Augmented episodes: {n_augmented} ({args.num_passes} passes)")
    print(f"  Total episodes: {n_original + n_augmented}")
    print(f"  Augmentations: {args.augmentations}")

    features_meta = source.meta.features
    user_features = {k: v for k, v in features_meta.items() if k not in set(DEFAULT_FEATURES.keys())}

    print(f"\nCreating output dataset: {args.output}")
    output = LeRobotDataset.create(
        repo_id=args.output,
        fps=source.fps,
        features=user_features,
        robot_type=source.meta.robot_type,
        use_videos=len(source.meta.camera_keys) > 0,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
    )

    try:
        if args.include_originals:
            print("\nCopying original episodes...")
            original_selector = get_temporal_selector(args, 0) if args.include_originals_decimated else None
            original_effective_shift = compute_effective_action_shift(args.action_shift, original_selector)
            if args.include_originals_decimated:
                print(f"Original temporal selector: {original_selector}")
                print(
                    f"Original action shift: raw={args.action_shift} frames, "
                    f"effective={original_effective_shift} selected-frame steps"
                )
            for ep_idx in tqdm(episode_indices, desc="Originals"):
                write_episode(
                    source,
                    output,
                    ep_idx,
                    feature_keys,
                    camera_keys,
                    features_meta,
                    action_shift=args.action_shift,
                    temporal_selector=original_selector,
                    tail_drop_max=args.tail_drop_max,
                    temporal_jitter_pct=0.0,
                    action_noise_std=0.0,
                    action_smoothing="none",
                    savgol_window_length=args.savgol_window_length,
                    savgol_polyorder=args.savgol_polyorder,
                    smooth_exclude_indices=resolve_smoothing_exclude_indices(args),
                )

        use_frame_decimate = "frame_decimate" in args.augmentations
        use_flip = "horizontal_flip" in args.augmentations
        image_augmentations = [a for a in args.augmentations if a not in ("frame_decimate", "horizontal_flip")]

        if use_frame_decimate:
            frame_decimator = build_frame_decimate(args)
            print(f"\nFrame decimation: {frame_decimator}")

        if use_flip:
            flip = build_horizontal_flip(args)
            print(f"Horizontal flip: {flip}")

        if image_augmentations:
            transform = build_transform(args, image_augmentations)
            print(f"Image transform: {transform}")
        else:
            transform = None

        for pass_idx in range(args.num_passes):
            if args.seed is not None:
                torch.manual_seed(args.seed + pass_idx + 1)

            temporal_selector = get_temporal_selector(args, pass_idx)
            if temporal_selector is None and use_frame_decimate:
                temporal_selector = frame_decimator
            if temporal_selector is not None:
                print(f"Temporal selector pass {pass_idx + 1}: {temporal_selector}")
            effective_shift = compute_effective_action_shift(args.action_shift, temporal_selector)
            print(
                f"Action shift pass {pass_idx + 1}: raw={args.action_shift} frames, "
                f"effective={effective_shift} selected-frame steps"
            )

            for ep_idx in tqdm(episode_indices, desc=f"Pass {pass_idx + 1}/{args.num_passes}"):
                write_episode(
                    source,
                    output,
                    ep_idx,
                    feature_keys,
                    camera_keys,
                    features_meta,
                    action_shift=args.action_shift,
                    temporal_selector=temporal_selector,
                    transform=transform,
                    flip=flip if use_flip else None,
                    tail_drop_max=args.tail_drop_max,
                    temporal_jitter_pct=args.temporal_jitter_pct,
                    action_noise_std=args.action_noise_std,
                    action_smoothing=args.action_smoothing,
                    savgol_window_length=args.savgol_window_length,
                    savgol_polyorder=args.savgol_polyorder,
                    smooth_exclude_indices=resolve_smoothing_exclude_indices(args),
                )

        print("\nFinalizing dataset...")
        output.finalize()

        if not args.no_push:
            print("Pushing to Hugging Face Hub...")
            output.push_to_hub()

    except Exception:
        output.finalize()
        raise

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Total episodes: {n_original + n_augmented}")
    encoded_path = quote(args.output, safe="")
    print(f"\nVisualizer link:\n  https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{encoded_path}%2Fepisode_0")


if __name__ == "__main__":
    main()
