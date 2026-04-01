"""Metadata synchronisation: Parquet reindexing, JSONL filtering, info.json updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def rewrite_episode_parquet(
    old_path: str | Path,
    new_path: str | Path,
    new_episode_idx: int,
    start_global_index: int,
) -> int:
    """Read an episode Parquet, reassign indices, and write to new_path.

    Args:
        old_path: Source Parquet file.
        new_path: Destination Parquet file.
        new_episode_idx: New episode_index value to assign to every row.
        start_global_index: The global frame index for the first frame of this episode.

    Returns:
        Number of frames written (i.e. length of the Parquet).
    """
    old_path = Path(old_path)
    new_path = Path(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(old_path)
    n_frames = len(df)

    df["episode_index"] = new_episode_idx
    df["frame_index"] = list(range(n_frames))
    df["index"] = list(range(start_global_index, start_global_index + n_frames))

    df.to_parquet(new_path, compression="zstd", index=False)
    return n_frames


def update_meta_jsonl(
    episodes_jsonl_path: str | Path,
    good_episodes: list[int],
    task_text: str,
    aug_params_per_episode: dict[int, Any] | None = None,
) -> None:
    """Filter and reindex an episodes JSONL file in-place.

    Args:
        episodes_jsonl_path: Path to the episodes.jsonl metadata file.
        good_episodes: Original episode indices to keep (in the desired output order).
        task_text: Task description string to write into each episode dict.
        aug_params_per_episode: Optional mapping from new episode index → aug params dict.
    """
    episodes_jsonl_path = Path(episodes_jsonl_path)
    if aug_params_per_episode is None:
        aug_params_per_episode = {}

    # Read existing episodes
    existing: dict[int, dict] = {}
    with open(episodes_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            existing[ep["episode_index"]] = ep

    # Build filtered + reindexed list
    out_lines = []
    for new_idx, orig_idx in enumerate(good_episodes):
        ep = dict(existing.get(orig_idx, {"episode_index": orig_idx}))
        ep["episode_index"] = new_idx
        ep["task"] = task_text
        if new_idx in aug_params_per_episode:
            ep["aug_params"] = aug_params_per_episode[new_idx]
        out_lines.append(json.dumps(ep))

    with open(episodes_jsonl_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")


def update_info_json(
    info_path: str | Path,
    new_total_episodes: int,
    new_total_frames: int,
    new_total_videos: int,
) -> None:
    """Update totals in a LeRobot info.json file.

    Args:
        info_path: Path to info.json.
        new_total_episodes: Updated episode count.
        new_total_frames: Updated frame count.
        new_total_videos: Updated video count.
    """
    info_path = Path(info_path)
    with open(info_path) as f:
        info = json.load(f)

    info["total_episodes"] = new_total_episodes
    info["total_frames"] = new_total_frames
    info["total_videos"] = new_total_videos

    # Update splits if present
    if "splits" in info:
        info["splits"]["train"] = f"0:{new_total_episodes}"

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
