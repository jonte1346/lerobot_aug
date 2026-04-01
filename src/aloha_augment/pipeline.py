"""Main CLI pipeline for ALOHA dataset augmentation (LeRobot v3 format)."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from aloha_augment.prefilter import score_sparc, score_actuator_saturation
from aloha_augment.visual_aug import apply_color_jitter, apply_occlusion_patch
from aloha_augment.upload import upload_dataset

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, video_backend="pyav")


def _status_markup(status: str) -> str:
    colours = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "N/A": "dim"}
    colour = colours.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


def _pass_warn_fail(val, pass_thresh, warn_thresh, higher_is_better=True) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    if higher_is_better:
        if val >= pass_thresh:
            return "PASS"
        elif val >= warn_thresh:
            return "WARN"
        return "FAIL"
    else:
        if val <= pass_thresh:
            return "PASS"
        elif val <= warn_thresh:
            return "WARN"
        return "FAIL"


def _get_camera_keys(info: dict) -> list[str]:
    """Return camera feature keys (those with dtype == 'video') from info.json."""
    features = info.get("features", {})
    cam_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    return cam_keys


def _get_fps(info: dict) -> float:
    return float(info.get("fps", 50.0))


def _read_info_json(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def _open_video_pyav(src_video: Path):
    """Open a video with pyav, returning (container, stream). Falls back to h264 temp file."""
    try:
        import av
        container = av.open(str(src_video))
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        return container, stream, None  # (container, stream, tmp_path)
    except Exception:
        pass

    # Fall back: transcode via ffmpeg to h264 temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src_video),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    import av
    container = av.open(tmp.name)
    stream = container.streams.video[0]
    stream.codec_context.thread_type = "AUTO"
    return container, stream, Path(tmp.name)


def _episode_arrays_from_parquet(src_df: pd.DataFrame, episode_idx: int):
    """Build state_list and actions array for one episode from the flat parquet."""
    ep_df = src_df[src_df["episode_index"] == episode_idx].sort_values("frame_index")
    if ep_df.empty:
        return [], np.empty((0,))

    state_col = "observation.state"
    action_col = "action"

    if state_col not in ep_df.columns or action_col not in ep_df.columns:
        # Try to find columns by partial match
        state_cols = [c for c in ep_df.columns if "state" in c.lower()]
        action_cols = [c for c in ep_df.columns if "action" in c.lower()]
        state_col = state_cols[0] if state_cols else None
        action_col = action_cols[0] if action_cols else None

    state_list = []
    action_rows = []
    for _, row in ep_df.iterrows():
        if state_col and state_col in row.index:
            q = np.array(row[state_col], dtype=float)
        else:
            q = np.zeros(6, dtype=float)
        state_list.append({"q": q})
        if action_col and action_col in ep_df.columns:
            action_rows.append(np.array(row[action_col], dtype=float))
        else:
            action_rows.append(np.zeros(6, dtype=float))

    actions = np.stack(action_rows) if action_rows else np.empty((0,))
    return state_list, actions


# ---------------------------------------------------------------------------
# Stage 2 helper: process one camera for all copies
# ---------------------------------------------------------------------------

def _process_camera(
    src_video: Path,
    cam_key: str,
    good_indices: list[int],
    frames_per_episode: int,
    n_aug_copies: int,
    fps: float,
    output_dataset_dir: Path,
    seed: int,
) -> None:
    """Read src_video once per aug copy, write filtered+augmented output videos."""
    import av
    import cv2

    for c in range(n_aug_copies):
        dst = output_dataset_dir / "videos" / cam_key / f"chunk-{c:03d}" / "file-000.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed + c)

        good_set = set(good_indices)

        # Open source video
        container, stream, tmp_path = _open_video_pyav(src_video)

        # We need width/height — peek at first decodable frame
        width = stream.codec_context.width
        height = stream.codec_context.height
        if width == 0 or height == 0:
            # Decode one frame to get size
            for packet in container.demux(stream):
                for frame in packet.decode():
                    arr = frame.to_ndarray(format="bgr24")
                    height, width = arr.shape[:2]
                    break
                if width != 0:
                    break
            container.seek(0)
            container, stream, tmp_path2 = _open_video_pyav(src_video)
            if tmp_path2 is not None:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                tmp_path = tmp_path2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

        frame_counter = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                episode_of_frame = frame_counter // frames_per_episode
                if episode_of_frame in good_set:
                    img = frame.to_ndarray(format="bgr24")
                    img, _ = apply_color_jitter(img, rng=rng)
                    img, _ = apply_occlusion_patch(img, rng=rng)
                    writer.write(img)
                frame_counter += 1

        container.close()
        writer.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

        console.print(
            f"  [dim]cam {cam_key} copy {c} → {dst.relative_to(output_dataset_dir)}[/]"
        )


# ---------------------------------------------------------------------------
# full_run  — 7-stage end-to-end pipeline
# ---------------------------------------------------------------------------

def full_run(
    repo_id: str,
    output_dir: str,
    hf_repo_id: str | None = None,
    n_aug_copies: int = 2,
    seed: int = 42,
    sparc_threshold: float = -10.0,
    saturation_threshold: float = 0.15,
    skip_sam3: bool = True,
    skip_text: bool = True,
    token: str | None = None,
    private: bool = False,
) -> None:
    """End-to-end 7-stage augmentation pipeline for LeRobot v3 datasets.

    Args:
        repo_id: Source HuggingFace dataset, e.g. 'lerobot/aloha_static_cups_open'.
        output_dir: Root directory for all outputs.
        hf_repo_id: If set, upload result to this HF repo and print visualizer URL.
        n_aug_copies: Visual augmentation copies per kept episode.
        seed: Base random seed.
        sparc_threshold: Episodes with SPARC < threshold are dropped.
        saturation_threshold: Episodes with saturation ratio > threshold are dropped.
        skip_sam3: Skip SAM3 background replacement (stage 3).
        skip_text: Skip text relabeling (stage 4).
        token: HuggingFace write token (or set HF_TOKEN env var).
        private: Create the HF repo as private.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset_dir = output_dir / "dataset"
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Pre-filter ──────────────────────────────────────────────────
    console.rule("[bold cyan]Stage 1 — Pre-filter[/]")
    console.print(f"Loading dataset [bold]{repo_id}[/] …")
    dataset = _load_dataset(repo_id)
    console.print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames total")

    dataset_root = Path(dataset.root)
    info = _read_info_json(dataset_root)
    fps = _get_fps(info)

    # Load flat parquet
    src_parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    console.print(f"  Reading parquet: {src_parquet_path}")
    src_df = pd.read_parquet(src_parquet_path)

    n_episodes = dataset.num_episodes
    n_total_frames = len(src_df)
    frames_per_episode = n_total_frames // n_episodes if n_episodes > 0 else 400

    # Score each episode
    scores = []
    good_indices = []

    for ep_idx in range(n_episodes):
        state_list, actions = _episode_arrays_from_parquet(src_df, ep_idx)
        sparc = score_sparc(state_list, actions)
        sat = score_actuator_saturation(state_list, actions)
        kept = sparc >= sparc_threshold and sat <= saturation_threshold
        scores.append({"episode_idx": ep_idx, "sparc": sparc, "saturation": sat, "kept": kept})
        if kept:
            good_indices.append(ep_idx)

    filter_table = Table(title="Pre-filter scores", header_style="bold")
    filter_table.add_column("Episode", justify="right")
    filter_table.add_column("SPARC", justify="right")
    filter_table.add_column("Saturation", justify="right")
    filter_table.add_column("Kept")
    for s in scores:
        kept_str = "[green]yes[/]" if s["kept"] else "[red]no[/]"
        filter_table.add_row(
            str(s["episode_idx"]),
            f"{s['sparc']:.3f}",
            f"{s['saturation']:.3f}",
            kept_str,
        )
    console.print(filter_table)
    console.print(f"Kept [bold]{len(good_indices)}[/] / {n_episodes} episodes")

    if not good_indices:
        console.print("[bold red]No episodes passed the filters. Exiting.[/]")
        return

    # ── Stage 2: Visual augmentation and output construction ─────────────────
    console.rule("[bold cyan]Stage 2 — Visual augmentation[/]")

    cam_keys = _get_camera_keys(info)
    if not cam_keys:
        console.print("[yellow]No video features found in info.json — skipping video processing.[/]")

    for cam_key in cam_keys:
        src_video = dataset_root / "videos" / cam_key / "chunk-000" / "file-000.mp4"
        if not src_video.exists():
            console.print(f"  [yellow]Source video not found: {src_video}[/]")
            continue
        console.print(f"  Processing camera [bold]{cam_key}[/] …")
        _process_camera(
            src_video=src_video,
            cam_key=cam_key,
            good_indices=good_indices,
            frames_per_episode=frames_per_episode,
            n_aug_copies=n_aug_copies,
            fps=fps,
            output_dataset_dir=output_dataset_dir,
            seed=seed,
        )

    # ── Stage 3: SAM3 (optional) ─────────────────────────────────────────────
    if skip_sam3:
        console.rule("[dim]Stage 3 — SAM3 background replacement (skipped)[/]")
    else:
        console.rule("[bold cyan]Stage 3 — SAM3 background replacement[/]")
        console.print("[yellow]SAM3 module not yet implemented — skipping.[/]")

    # ── Stage 4: Text relabeling (optional) ──────────────────────────────────
    if skip_text:
        console.rule("[dim]Stage 4 — Text relabeling (skipped)[/]")
    else:
        console.rule("[bold cyan]Stage 4 — Text relabeling[/]")
        console.print("[yellow]Text relabeling module not yet implemented — skipping.[/]")

    # ── Stage 5: Metadata sync ───────────────────────────────────────────────
    console.rule("[bold cyan]Stage 5 — Metadata sync[/]")

    # --- Build augmented data parquet ---
    pieces = []
    global_frame_counter = 0
    n_good = len(good_indices)

    for c in range(n_aug_copies):
        for local_idx, orig_ep in enumerate(good_indices):
            new_ep_idx = c * n_good + local_idx
            ep_rows = src_df[src_df["episode_index"] == orig_ep].copy()
            ep_rows = ep_rows.sort_values("frame_index").reset_index(drop=True)
            n_frames = len(ep_rows)
            ep_rows["episode_index"] = new_ep_idx
            ep_rows["frame_index"] = range(n_frames)
            ep_rows["index"] = range(global_frame_counter, global_frame_counter + n_frames)
            ep_rows["timestamp"] = [i / fps for i in range(n_frames)]
            pieces.append(ep_rows)
            global_frame_counter += n_frames

    aug_df = pd.concat(pieces, ignore_index=True)
    aug_data_dir = output_dataset_dir / "data" / "chunk-000"
    aug_data_dir.mkdir(parents=True, exist_ok=True)
    aug_parquet_path = aug_data_dir / "file-000.parquet"
    aug_df.to_parquet(aug_parquet_path, index=False, compression="zstd")
    console.print(f"  Data parquet written: {aug_parquet_path} ({len(aug_df)} rows)")

    # --- Build episodes meta parquet ---
    src_episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src_episodes_df = None
    if src_episodes_path.exists():
        src_episodes_df = pd.read_parquet(src_episodes_path)

    ep_duration = frames_per_episode / fps
    new_ep_rows = []
    new_global_from = 0

    for c in range(n_aug_copies):
        for local_idx, orig_ep in enumerate(good_indices):
            new_ep_idx = c * n_good + local_idx
            n_frames = frames_per_episode  # approximate; exact count per episode

            row: dict[str, Any] = {
                "episode_index": new_ep_idx,
                "dataset_from_index": new_global_from,
                "dataset_to_index": new_global_from + n_frames,
                "length": n_frames,
                "tasks": ["robot manipulation"],  # default; overridden below if we have meta
            }

            # Copy stats and other columns from original if available
            if src_episodes_df is not None and "episode_index" in src_episodes_df.columns:
                orig_rows = src_episodes_df[src_episodes_df["episode_index"] == orig_ep]
                if not orig_rows.empty:
                    orig_row = orig_rows.iloc[0].to_dict()
                    # Carry over stats columns
                    for col, val in orig_row.items():
                        if col not in row:
                            row[col] = val
                    # Override key columns
                    row["episode_index"] = new_ep_idx
                    row["dataset_from_index"] = new_global_from
                    row["dataset_to_index"] = new_global_from + n_frames
                    row["length"] = orig_row.get("length", n_frames)
                    # Update tasks if available
                    if "tasks" in orig_row:
                        row["tasks"] = orig_row["tasks"]

            # Update video timestamps and chunk pointers for each camera
            from_ts = new_ep_idx * ep_duration
            to_ts = from_ts + ep_duration
            for cam_key in cam_keys:
                ts_from_col = f"videos/{cam_key}/timestamp_from"
                ts_to_col = f"videos/{cam_key}/timestamp_to"
                chunk_col = f"videos/{cam_key}/chunk_index"
                file_col = f"videos/{cam_key}/file_index"
                row[ts_from_col] = from_ts
                row[ts_to_col] = to_ts
                row[chunk_col] = c
                row[file_col] = 0

            new_global_from += n_frames
            new_ep_rows.append(row)

    new_episodes_df = pd.DataFrame(new_ep_rows)
    aug_episodes_dir = output_dataset_dir / "meta" / "episodes" / "chunk-000"
    aug_episodes_dir.mkdir(parents=True, exist_ok=True)
    aug_episodes_path = aug_episodes_dir / "file-000.parquet"
    new_episodes_df.to_parquet(aug_episodes_path, index=False, compression="zstd")
    console.print(f"  Episodes meta written: {aug_episodes_path} ({len(new_episodes_df)} episodes)")

    # --- Copy tasks.parquet unchanged ---
    src_tasks_path = dataset_root / "meta" / "tasks.parquet"
    if src_tasks_path.exists():
        dst_tasks_dir = output_dataset_dir / "meta"
        dst_tasks_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tasks_path, dst_tasks_dir / "tasks.parquet")
        console.print("  tasks.parquet copied")

    # --- Write new info.json ---
    new_total_episodes = n_aug_copies * n_good
    new_total_frames = global_frame_counter

    new_info = dict(info)
    new_info["total_episodes"] = new_total_episodes
    new_info["total_frames"] = new_total_frames
    new_info["splits"] = {"train": f"0:{new_total_episodes}"}
    # Update data_path and video_path templates if present (keep same pattern)
    if "data_path" not in new_info:
        new_info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    if "video_path" not in new_info:
        new_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    aug_meta_dir = output_dataset_dir / "meta"
    aug_meta_dir.mkdir(parents=True, exist_ok=True)
    info_out_path = aug_meta_dir / "info.json"
    info_out_path.write_text(json.dumps(new_info, indent=2))
    console.print(f"  info.json written: {info_out_path}")
    console.print(f"  Synced: {new_total_episodes} episodes, {new_total_frames} frames")

    # ── Stage 6: Quality report ──────────────────────────────────────────────
    console.rule("[bold cyan]Stage 6 — Quality report[/]")

    sample_n = min(10, n_good)
    sampled_orig = good_indices[:sample_n]

    orig_sparc_scores = []
    for ep_idx in sampled_orig:
        sl, act = _episode_arrays_from_parquet(src_df, ep_idx)
        orig_sparc_scores.append(score_sparc(sl, act))

    orig_mean_sparc = float(np.mean(orig_sparc_scores)) if orig_sparc_scores else float("nan")
    # Augmented episodes have the same trajectories (only visual changes), so SPARC is the same
    aug_mean_sparc = orig_mean_sparc

    sparc_delta_pct = 0.0  # visual aug doesn't change trajectories

    report_table = Table(title="Quality Report", header_style="bold magenta")
    report_table.add_column("Metric")
    report_table.add_column("Before", justify="right")
    report_table.add_column("After", justify="right")

    report_table.add_row("n_episodes", str(n_episodes), str(new_total_episodes))
    report_table.add_row("n_frames", str(n_total_frames), str(new_total_frames))
    report_table.add_row("n_kept_episodes", str(n_good), str(n_good))
    report_table.add_row(
        "mean_SPARC (sample)",
        f"{orig_mean_sparc:.3f}",
        f"{aug_mean_sparc:.3f}",
    )
    console.print(report_table)

    report_dict: dict[str, Any] = {
        "source_repo": repo_id,
        "output_dir": str(output_dir),
        "n_original_episodes": n_episodes,
        "n_kept_episodes": n_good,
        "n_augmented_episodes": new_total_episodes,
        "n_original_frames": n_total_frames,
        "n_augmented_frames": new_total_frames,
        "original_mean_sparc": orig_mean_sparc,
        "augmented_mean_sparc": aug_mean_sparc,
        "sparc_delta_pct": sparc_delta_pct,
    }
    report_path = output_dir / "augmentation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    console.print(f"  Report saved → {report_path}")

    # ── Stage 7: Upload ──────────────────────────────────────────────────────
    if hf_repo_id:
        console.rule("[bold cyan]Stage 7 — Upload to HuggingFace Hub[/]")
        url = upload_dataset(output_dataset_dir, hf_repo_id, token=token, private=private)
        console.print(f"[bold green]Visualizer:[/] {url}")
    else:
        console.rule("[dim]Stage 7 — Upload (skipped — no --hf_repo_id provided)[/]")
        console.print("[dim]Pass --hf_repo_id=<your-user/dataset-name> to upload.[/]")

    console.print(f"\n[bold green]Pipeline complete.[/] Output: {output_dataset_dir}")


# ---------------------------------------------------------------------------
# run  — shortcut without upload
# ---------------------------------------------------------------------------

def run(
    repo_id: str,
    output_dir: str,
    sparc_threshold: float = -10.0,
    saturation_threshold: float = 0.15,
    n_aug_copies: int = 2,
    seed: int = 42,
) -> None:
    """Filter + augment + sync metadata for a LeRobot v3 dataset. Does not upload.

    For the end-to-end pipeline including upload, use `full_run`.
    """
    full_run(
        repo_id=repo_id,
        output_dir=output_dir,
        hf_repo_id=None,
        n_aug_copies=n_aug_copies,
        seed=seed,
        sparc_threshold=sparc_threshold,
        saturation_threshold=saturation_threshold,
        skip_sam3=True,
        skip_text=True,
        token=None,
        private=False,
    )


# ---------------------------------------------------------------------------
# report  — standalone quality comparison
# ---------------------------------------------------------------------------

def report(original_repo_id: str, augmented_output_dir: str) -> None:
    """Print a quality comparison table for an existing augmented dataset.

    Args:
        original_repo_id: Source HF dataset ID.
        augmented_output_dir: Directory produced by `run` or `full_run`.
    """
    augmented_output_dir = Path(augmented_output_dir)
    dst_root = augmented_output_dir / "dataset"

    info_path = dst_root / "meta" / "info.json"
    if not info_path.exists():
        console.print(f"[red]No meta/info.json found in {dst_root}[/]")
        return

    with open(info_path) as f:
        aug_info = json.load(f)

    dataset = _load_dataset(original_repo_id)
    dataset_root = Path(dataset.root)
    src_parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    src_df = pd.read_parquet(src_parquet_path)

    sample_n = min(10, dataset.num_episodes)
    orig_sparc_scores = []
    for ep_idx in range(sample_n):
        sl, act = _episode_arrays_from_parquet(src_df, ep_idx)
        orig_sparc_scores.append(score_sparc(sl, act))

    mean_orig = float(np.mean(orig_sparc_scores)) if orig_sparc_scores else float("nan")

    table = Table(title="Quality Comparison", header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Original", justify="right")
    table.add_column("Augmented", justify="right")

    table.add_row("episodes", str(dataset.num_episodes), str(aug_info.get("total_episodes", "?")))
    table.add_row("frames", str(len(dataset)), str(aug_info.get("total_frames", "?")))
    table.add_row("mean_SPARC (sample)", f"{mean_orig:.3f}", "—")
    console.print(table)


# ---------------------------------------------------------------------------
# extract_masks  — SAM3 stub
# ---------------------------------------------------------------------------

def extract_masks(repo_id: str, output_mask_dir: str) -> None:
    """Extract SAM3 robot masks. Not yet implemented."""
    console.print("[yellow]SAM3 not yet implemented.[/]")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fire.Fire(
        {
            "run": run,
            "full_run": full_run,
            "report": report,
            "extract_masks": extract_masks,
            "upload": upload_dataset,
        }
    )


if __name__ == "__main__":
    main()
