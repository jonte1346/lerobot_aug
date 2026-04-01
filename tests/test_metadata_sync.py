"""Tests for metadata synchronisation utilities."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aloha_augment.metadata_sync import (
    rewrite_episode_parquet,
    update_info_json,
    update_meta_jsonl,
)


def _make_parquet(path: Path, episode_idx: int, n_frames: int = 10) -> None:
    df = pd.DataFrame(
        {
            "episode_index": [episode_idx] * n_frames,
            "frame_index": list(range(n_frames)),
            "index": list(range(episode_idx * n_frames, (episode_idx + 1) * n_frames)),
            "observation.state": [[0.0] * 6] * n_frames,
        }
    )
    df.to_parquet(path, index=False)


def test_rewrite_episode_parquet_contiguous():
    """After filtering 3→2 episodes, episode_index should be contiguous [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src0 = tmpdir / "ep0_src.parquet"
        src2 = tmpdir / "ep2_src.parquet"
        dst0 = tmpdir / "ep0_dst.parquet"
        dst1 = tmpdir / "ep1_dst.parquet"

        _make_parquet(src0, episode_idx=0, n_frames=5)
        _make_parquet(src2, episode_idx=2, n_frames=7)

        rewrite_episode_parquet(src0, dst0, new_episode_idx=0, start_global_index=0)
        rewrite_episode_parquet(src2, dst1, new_episode_idx=1, start_global_index=5)

        df0 = pd.read_parquet(dst0)
        df1 = pd.read_parquet(dst1)

        assert list(df0["episode_index"].unique()) == [0]
        assert list(df1["episode_index"].unique()) == [1]

        # frame_index should restart from 0 for each episode
        assert list(df0["frame_index"]) == list(range(5))
        assert list(df1["frame_index"]) == list(range(7))

        # global index should be contiguous across both
        combined_index = list(df0["index"]) + list(df1["index"])
        assert combined_index == list(range(12))


def test_update_meta_jsonl():
    """JSONL should only contain kept episodes, reindexed from 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "episodes.jsonl"
        lines = [
            json.dumps({"episode_index": i, "task": "old_task"}) for i in range(5)
        ]
        jsonl_path.write_text("\n".join(lines) + "\n")

        # Keep original episodes 1 and 3
        update_meta_jsonl(jsonl_path, good_episodes=[1, 3], task_text="new_task")

        result = []
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                result.append(json.loads(line))

        assert len(result) == 2
        assert result[0]["episode_index"] == 0
        assert result[1]["episode_index"] == 1
        assert all(r["task"] == "new_task" for r in result)


def test_update_info_json():
    """info.json totals and splits should be updated correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        info_path = Path(tmpdir) / "info.json"
        info = {
            "total_episodes": 100,
            "total_frames": 5000,
            "total_videos": 300,
            "splits": {"train": "0:100"},
        }
        info_path.write_text(json.dumps(info))

        update_info_json(info_path, new_total_episodes=40, new_total_frames=2000, new_total_videos=120)

        updated = json.loads(info_path.read_text())
        assert updated["total_episodes"] == 40
        assert updated["total_frames"] == 2000
        assert updated["total_videos"] == 120
        assert updated["splits"]["train"] == "0:40"
