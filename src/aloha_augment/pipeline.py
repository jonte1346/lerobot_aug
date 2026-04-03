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

from .transforms import DriftingBlob, FrameDecimator, HorizontalFlipWithActionMirror, ROBOT_PRESETS, StaticErasing
from .prefilter import filter_episodes, save_filter_scores


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


class SAM3AugmentationMarker:
    """Marker to indicate SAM3 augmentation should be applied (requires frame data)."""
    def __call__(self, frame):
        # This is a no-op transform; actual SAM3 augmentation happens at episode level
        return frame


def build_sam3(args):
    return SAM3AugmentationMarker()


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

    if len(transforms) == 1:
        return transforms[0]
    return v2.Compose(transforms)


def _episode_record(episodes, ep_idx: int):
    if hasattr(episodes, "iloc"):
        return episodes.iloc[ep_idx]
    return episodes[ep_idx]


def get_episode_range(source, ep_idx):
    ep = _episode_record(source.meta.episodes, ep_idx)
    if hasattr(ep, "to_dict"):
        ep = ep.to_dict()
    return ep["dataset_from_index"], ep["dataset_to_index"]


def build_frame_dict(item, feature_keys, features_meta):
    frame = {"task": item.get("task", "robot manipulation")}
    for key in feature_keys:
        if key in {"timestamp", "index", "episode_index", "frame_index", "task_index"}:
            continue
        if key in item:
            val = item[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            if hasattr(val, "ndim") and val.ndim == 3 and val.shape[0] == 3:
                val = np.transpose(val, (1, 2, 0))
            if key in features_meta:
                expected_shape = tuple(features_meta[key]["shape"])
                if getattr(val, "ndim", None) == 0 and expected_shape == (1,):
                    val = val.reshape(1)
            frame[key] = val
    return frame


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


def parse_args():
    parser = argparse.ArgumentParser(description="Augment a LeRobot v3 dataset and push to Hugging Face Hub")

    parser.add_argument("--source", required=True, help="Source dataset repo_id (e.g. lerobot/aloha_static_cups_open)")
    parser.add_argument("--output", required=True, help="Output dataset repo_id (e.g. user/dataset_augmented)")
    parser.add_argument("--num-passes", type=int, default=2, help="Number of augmented copies per episode")
    parser.add_argument("--augmentations", nargs="+", default=["color_jitter"], choices=list(AUGMENTATION_BUILDERS.keys()))
    parser.add_argument("--include-originals", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)

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
    parser.add_argument("--sparc-threshold", type=float, default=-10.0, help="SPARC threshold for smoothness")
    parser.add_argument("--saturation-threshold", type=float, default=0.15, help="Max saturation fraction")
    
    # SAM3 augmentation options
    parser.add_argument("--skip-sam3", action="store_true", help="Skip SAM3 background augmentation")

    return parser.parse_args()


def main():
    args = parse_args()
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
                copy_episode(source, output, ep_idx, feature_keys, features_meta)

        use_decimation = "frame_decimate" in args.augmentations
        use_flip = "horizontal_flip" in args.augmentations
        image_augmentations = [a for a in args.augmentations if a not in ("frame_decimate", "horizontal_flip")]

        if use_decimation:
            decimator = build_frame_decimate(args)
            print(f"\nFrame decimation: {decimator}")

        if use_flip:
            flip = build_horizontal_flip(args)
            print(f"Horizontal flip: {flip}")

        if image_augmentations:
            saved = args.augmentations
            args.augmentations = image_augmentations
            transform = build_transform(args)
            args.augmentations = saved
            print(f"Image transform: {transform}")

        for pass_idx in range(args.num_passes):
            if args.seed is not None:
                torch.manual_seed(args.seed + pass_idx + 1)

            for ep_idx in tqdm(episode_indices, desc=f"Pass {pass_idx + 1}/{args.num_passes}"):
                if use_decimation and not image_augmentations and not use_flip:
                    decimate_episode(source, output, ep_idx, decimator, feature_keys, features_meta)
                elif use_flip and not use_decimation and not image_augmentations:
                    augment_episode_with_flip(source, output, ep_idx, flip, feature_keys, camera_keys, features_meta)
                elif not use_decimation and not use_flip and image_augmentations:
                    augment_episode(source, output, ep_idx, transform, feature_keys, camera_keys, features_meta)
                else:
                    from_idx, to_idx = get_episode_range(source, ep_idx)
                    for local_idx, global_idx in enumerate(range(from_idx, to_idx)):
                        if use_decimation and not decimator.should_keep(local_idx):
                            continue
                        item = source[global_idx]
                        for cam_key in camera_keys:
                            if cam_key in item:
                                if use_flip:
                                    item[cam_key] = flip.flip_image(item[cam_key])
                                if image_augmentations:
                                    item[cam_key] = transform(item[cam_key])
                        if use_flip:
                            if "action" in item:
                                item["action"] = flip.mirror_actions(item["action"])
                            if "observation.state" in item:
                                item["observation.state"] = flip.mirror_state(item["observation.state"])
                        output.add_frame(build_frame_dict(item, feature_keys, features_meta))
                    output.save_episode()

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
