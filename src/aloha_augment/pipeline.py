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


def build_transform(args):
    transforms = []
    for name in args.augmentations:
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
):
    from_idx, to_idx = get_episode_range(source, ep_idx)
    tail_drop = int(np.random.randint(0, tail_drop_max + 1)) if tail_drop_max > 0 else 0
    effective_to_idx = to_idx - action_shift - tail_drop

    if effective_to_idx <= from_idx:
        return

    if transform is not None:
        reset_transform_state(transform)

    if isinstance(transform, (StaticErasing, DriftingBlob)):
        first = source[from_idx]
        for cam_key in camera_keys:
            if cam_key in first:
                _, h, w = first[cam_key].shape
                transform.resample(h, w)
                break

    for local_idx, global_idx in enumerate(range(from_idx, effective_to_idx)):
        if temporal_selector is not None and not temporal_selector.should_keep(local_idx):
            continue

        item = dict(source[global_idx])
        action_override = None

        if action_shift:
            future_item = source[global_idx + action_shift]
            if "action" in future_item:
                action_override = future_item["action"]

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


def copy_episode(source, output, ep_idx, feature_keys, features_meta):
    from_idx, to_idx = get_episode_range(source, ep_idx)
    for global_idx in range(from_idx, to_idx):
        item = source[global_idx]
        output.add_frame(build_frame_dict(item, feature_keys, features_meta))
    output.save_episode()


def decimate_episode(source, output, ep_idx, decimator, feature_keys, features_meta):
    from_idx, to_idx = get_episode_range(source, ep_idx)
    for local_idx, global_idx in enumerate(range(from_idx, to_idx)):
        if not decimator.should_keep(local_idx):
            continue
        item = source[global_idx]
        output.add_frame(build_frame_dict(item, feature_keys, features_meta))
    output.save_episode()


def augment_episode(source, output, ep_idx, transform, feature_keys, camera_keys, features_meta):
    from_idx, to_idx = get_episode_range(source, ep_idx)

    if isinstance(transform, (StaticErasing, DriftingBlob)):
        first = source[from_idx]
        for cam_key in camera_keys:
            if cam_key in first:
                _, h, w = first[cam_key].shape
                transform.resample(h, w)
                break

    for global_idx in range(from_idx, to_idx):
        item = source[global_idx]
        for cam_key in camera_keys:
            if cam_key in item:
                item[cam_key] = transform(item[cam_key])
        output.add_frame(build_frame_dict(item, feature_keys, features_meta))
    output.save_episode()


def augment_episode_with_flip(source, output, ep_idx, flip, feature_keys, camera_keys, features_meta):
    from_idx, to_idx = get_episode_range(source, ep_idx)

    for global_idx in range(from_idx, to_idx):
        item = source[global_idx]
        for cam_key in camera_keys:
            if cam_key in item:
                item[cam_key] = flip.flip_image(item[cam_key])
        if "action" in item:
            item["action"] = flip.mirror_actions(item["action"])
        if "observation.state" in item:
            item["observation.state"] = flip.mirror_state(item["observation.state"])
        output.add_frame(build_frame_dict(item, feature_keys, features_meta))
    output.save_episode()


def build_parser():
    parser = argparse.ArgumentParser(description="Augment a LeRobot v3 dataset and push to Hugging Face Hub")

    parser.add_argument("--source", required=True, help="Source dataset repo_id (e.g. lerobot/aloha_static_cups_open)")
    parser.add_argument("--output", required=True, help="Output dataset repo_id (e.g. user/dataset_augmented)")
    parser.add_argument("--tier", choices=list(TIER_PRESETS.keys()), default=None, help="Tier preset: tier1, tier2, or tier3")
    parser.add_argument("--num-passes", type=int, default=2, help="Number of augmented copies per episode")
    parser.add_argument("--augmentations", nargs="*", default=None, choices=list(AUGMENTATION_BUILDERS.keys()))
    parser.add_argument("--include-originals", action="store_true")
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

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    defaults = _parser_defaults(parser)
    if args.augmentations is None:
        args.augmentations = ["color_jitter"]
    if args.skip_sam3:
        args.augmentations = [a for a in args.augmentations if a != "sam3"]
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
            for ep_idx in tqdm(episode_indices, desc="Originals"):
                write_episode(
                    source,
                    output,
                    ep_idx,
                    feature_keys,
                    camera_keys,
                    features_meta,
                    action_shift=args.action_shift,
                    tail_drop_max=args.tail_drop_max,
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
            saved = args.augmentations
            args.augmentations = image_augmentations
            transform = build_transform(args)
            args.augmentations = saved
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
