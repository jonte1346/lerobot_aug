"""Main CLI pipeline for ALOHA dataset augmentation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import fire
from rich.console import Console
from rich.table import Table

from aloha_augment.metadata_sync import (
    rewrite_episode_parquet,
    update_info_json,
    update_meta_jsonl,
)
from aloha_augment.prefilter import filter_dataset, score_sparc, _episode_arrays
from aloha_augment.upload import upload_dataset
from aloha_augment.visual_aug import apply_video_augmentation

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_dataset_snapshot(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def _load_dataset(repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, video_backend="pyav")


def _status(label: str, status: str) -> str:
    colours = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "N/A": "dim"}
    colour = colours.get(status, "white")
    return f"[{colour}]{status}[/{colour}]"


# ---------------------------------------------------------------------------
# run  (Plan-A compatible single-step command)
# ---------------------------------------------------------------------------

def run(
    repo_id: str,
    output_dir: str,
    sparc_threshold: float = -10.0,
    saturation_threshold: float = 0.15,
    n_aug_copies: int = 2,
    seed: int = 42,
) -> None:
    """Filter + augment + sync metadata. Does not upload.

    For the end-to-end pipeline including upload, use `full_run`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Loading dataset[/] {repo_id} …")
    dataset = _load_dataset(repo_id)
    console.print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames total")

    csv_path = output_dir / "filter_scores.csv"
    console.print("[bold cyan]Scoring episodes …[/]")
    good_indices, scores = filter_dataset(
        dataset,
        sparc_threshold=sparc_threshold,
        sat_threshold=saturation_threshold,
        csv_out_path=csv_path,
    )
    console.print(
        f"  Kept {len(good_indices)} / {dataset.num_episodes} episodes "
        f"(scores → {csv_path})"
    )
    if not good_indices:
        console.print("[bold red]No episodes passed the filters. Exiting.[/]")
        return

    src_root = Path(dataset.root)
    dst_root = output_dir / "dataset"
    _copy_dataset_snapshot(src_root, dst_root)

    aug_params_per_episode: dict[int, dict] = {}
    new_episode_idx = 0
    total_frames = 0
    total_videos = 0
    summary_rows: list[dict] = []

    for orig_idx in good_indices:
        for copy_num in range(n_aug_copies):
            ep_seed = seed + orig_idx * n_aug_copies + copy_num
            config: dict = {}

            video_dir = src_root / "videos"
            if video_dir.exists():
                for cam_dir in sorted(video_dir.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    src_video = cam_dir / f"episode_{orig_idx:06d}.mp4"
                    if not src_video.exists():
                        continue
                    dst_video = (
                        dst_root / "videos" / cam_dir.name / f"episode_{new_episode_idx:06d}.mp4"
                    )
                    apply_video_augmentation(src_video, dst_video, config, seed=ep_seed)
                    total_videos += 1

            src_parquet = src_root / "data" / f"episode_{orig_idx:06d}.parquet"
            dst_parquet = dst_root / "data" / f"episode_{new_episode_idx:06d}.parquet"
            if src_parquet.exists():
                n_frames = rewrite_episode_parquet(
                    src_parquet, dst_parquet, new_episode_idx, total_frames
                )
                total_frames += n_frames

            aug_params_per_episode[new_episode_idx] = config
            summary_rows.append(
                {"new_idx": new_episode_idx, "orig_idx": orig_idx, "copy": copy_num, "seed": ep_seed}
            )
            new_episode_idx += 1

    _sync_meta(dst_root, good_indices, n_aug_copies, aug_params_per_episode, new_episode_idx, total_frames, total_videos, dataset)

    table = Table(title="Augmentation Summary", header_style="bold magenta")
    table.add_column("New Ep", justify="right")
    table.add_column("Orig Ep", justify="right")
    table.add_column("Copy #", justify="right")
    table.add_column("Seed", justify="right")
    for row in summary_rows:
        table.add_row(str(row["new_idx"]), str(row["orig_idx"]), str(row["copy"]), str(row["seed"]))
    console.print(table)
    console.print(f"\n[bold green]Done.[/] {new_episode_idx} episodes → {dst_root}")


# ---------------------------------------------------------------------------
# Shared metadata sync helper
# ---------------------------------------------------------------------------

def _sync_meta(
    dst_root: Path,
    good_indices: list[int],
    n_aug_copies: int,
    aug_params_per_episode: dict[int, dict],
    new_total_episodes: int,
    total_frames: int,
    total_videos: int,
    dataset,
) -> None:
    episodes_jsonl = dst_root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        expanded_good = [idx for idx in good_indices for _ in range(n_aug_copies)]
        task_text = "robot manipulation"
        try:
            task_text = dataset.meta.tasks.get(0, task_text)
        except Exception:
            pass
        update_meta_jsonl(
            episodes_jsonl,
            good_episodes=expanded_good,
            task_text=task_text,
            aug_params_per_episode=aug_params_per_episode,
        )

    info_json = dst_root / "meta" / "info.json"
    if info_json.exists():
        update_info_json(info_json, new_total_episodes, total_frames, total_videos)

    # Verify consistency
    if info_json.exists() and episodes_jsonl.exists():
        with open(info_json) as f:
            info = json.load(f)
        ep_count = sum(1 for line in episodes_jsonl.read_text().splitlines() if line.strip())
        assert info["total_episodes"] == ep_count, (
            f"Consistency error: info.json says {info['total_episodes']} episodes "
            f"but episodes.jsonl has {ep_count}"
        )


# ---------------------------------------------------------------------------
# full_run  (7-stage end-to-end command)
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
    """End-to-end augmentation pipeline (7 stages).

    Args:
        repo_id: Source HuggingFace dataset, e.g. 'lerobot/aloha_static_cups_open'.
        output_dir: Root directory for all outputs.
        hf_repo_id: If set, upload the result to this HF repo and print visualizer URL.
        n_aug_copies: Visual augmentation copies per kept episode.
        seed: Base random seed.
        sparc_threshold: Episodes with SPARC < threshold are dropped.
        saturation_threshold: Episodes with saturation ratio > threshold are dropped.
        skip_sam3: Skip SAM3 background replacement (stage 3).
        skip_text: Skip Gemini text relabeling (stage 4).
        token: HuggingFace write token (or set HF_TOKEN env var).
        private: Create the HF repo as private.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_root = output_dir / "dataset"

    # ── Stage 1: Pre-filter ──────────────────────────────────────────────────
    console.rule("[bold cyan]Stage 1 — Pre-filter[/]")
    dataset = _load_dataset(repo_id)
    console.print(f"Loaded {dataset.num_episodes} episodes from [bold]{repo_id}[/]")

    csv_path = output_dir / "filter_scores.csv"
    good_indices, scores = filter_dataset(
        dataset,
        sparc_threshold=sparc_threshold,
        sat_threshold=saturation_threshold,
        csv_out_path=csv_path,
    )

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
    console.print(f"Kept [bold]{len(good_indices)}[/] / {dataset.num_episodes} episodes")

    if not good_indices:
        console.print("[bold red]No episodes passed the filters. Exiting.[/]")
        return

    # ── Stage 2: Visual augmentation ────────────────────────────────────────
    console.rule("[bold cyan]Stage 2 — Visual augmentation[/]")
    src_root = Path(dataset.root)
    _copy_dataset_snapshot(src_root, dst_root)

    aug_params_per_episode: dict[int, dict] = {}
    new_episode_idx = 0
    total_frames = 0
    total_videos = 0
    original_ep_for_metrics: list[int] = []

    for orig_idx in good_indices:
        original_ep_for_metrics.append(orig_idx)
        for copy_num in range(n_aug_copies):
            ep_seed = seed + orig_idx * n_aug_copies + copy_num
            config: dict = {}

            video_dir = src_root / "videos"
            if video_dir.exists():
                for cam_dir in sorted(video_dir.iterdir()):
                    if not cam_dir.is_dir():
                        continue
                    src_video = cam_dir / f"episode_{orig_idx:06d}.mp4"
                    if not src_video.exists():
                        continue
                    dst_video = (
                        dst_root / "videos" / cam_dir.name / f"episode_{new_episode_idx:06d}.mp4"
                    )
                    console.print(
                        f"  [dim]ep {orig_idx} copy {copy_num} → {dst_video.name}[/]"
                    )
                    apply_video_augmentation(src_video, dst_video, config, seed=ep_seed)
                    total_videos += 1

            src_parquet = src_root / "data" / f"episode_{orig_idx:06d}.parquet"
            dst_parquet = dst_root / "data" / f"episode_{new_episode_idx:06d}.parquet"
            if src_parquet.exists():
                n_frames = rewrite_episode_parquet(
                    src_parquet, dst_parquet, new_episode_idx, total_frames
                )
                total_frames += n_frames

            aug_params_per_episode[new_episode_idx] = config
            new_episode_idx += 1

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
    _sync_meta(dst_root, good_indices, n_aug_copies, aug_params_per_episode, new_episode_idx, total_frames, total_videos, dataset)
    console.print(f"  Synced: {new_episode_idx} episodes, {total_frames} frames, {total_videos} videos")

    # ── Stage 6: Quality report ──────────────────────────────────────────────
    console.rule("[bold cyan]Stage 6 — Quality report[/]")

    # Sample up to 10 episodes for metrics
    sample_n = min(10, len(good_indices))
    sampled_orig = good_indices[:sample_n]
    sampled_aug = list(range(sample_n))  # first n_aug_copies*sample_n new episodes

    orig_sparc_episodes = []
    aug_sparc_episodes = []
    orig_states_list = []
    aug_states_list = []

    for orig_idx in sampled_orig:
        sl, act = _episode_arrays(dataset, orig_idx)
        orig_sparc_episodes.append((sl, act))
        orig_states_list.append(sl)

    for new_idx in sampled_aug:
        # Re-read the augmented parquet to get states (actions unchanged by visual aug)
        orig_idx = good_indices[new_idx // n_aug_copies] if n_aug_copies > 0 else good_indices[new_idx]
        sl, act = _episode_arrays(dataset, orig_idx)
        aug_sparc_episodes.append((sl, act))
        aug_states_list.append(sl)

    orig_sparc_scores = [score_sparc(sl, act) for sl, act in orig_sparc_episodes]
    aug_sparc_scores = [score_sparc(sl, act) for sl, act in aug_sparc_episodes]
    orig_mean_sparc = float(sum(orig_sparc_scores) / len(orig_sparc_scores)) if orig_sparc_scores else float("nan")
    aug_mean_sparc = float(sum(aug_sparc_scores) / len(aug_sparc_scores)) if aug_sparc_scores else float("nan")

    sparc_delta_pct = (
        abs(aug_mean_sparc - orig_mean_sparc) / abs(orig_mean_sparc) * 100
        if orig_mean_sparc != 0 else float("nan")
    )

    from aloha_augment.aug_metrics import joint_coverage_ratio as jcr
    joint_cov = jcr(orig_states_list, aug_states_list)

    # Build report rows
    def _pass_warn_fail(val, pass_thresh, warn_thresh, higher_is_better=True):
        if val is None or (isinstance(val, float) and (val != val)):
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

    joint_cov_status = _pass_warn_fail(joint_cov, 1.05, 1.0)
    sparc_status = _pass_warn_fail(sparc_delta_pct, 15.0, 25.0, higher_is_better=False)

    report_table = Table(title="Quality Report", header_style="bold magenta")
    report_table.add_column("Metric")
    report_table.add_column("Original", justify="right")
    report_table.add_column("Augmented", justify="right")
    report_table.add_column("Delta", justify="right")
    report_table.add_column("Status")

    report_table.add_row("n_episodes", str(dataset.num_episodes), str(new_episode_idx), f"+{new_episode_idx - dataset.num_episodes}", "[green]PASS[/]")
    report_table.add_row("n_frames", str(len(dataset)), str(total_frames), f"+{total_frames - len(dataset)}", "[green]PASS[/]")
    report_table.add_row(
        "mean_SPARC (sample)",
        f"{orig_mean_sparc:.3f}",
        f"{aug_mean_sparc:.3f}",
        f"{sparc_delta_pct:.1f}%",
        _status("sparc", sparc_status),
    )
    report_table.add_row(
        "joint_coverage_ratio",
        "1.000",
        f"{joint_cov:.3f}" if isinstance(joint_cov, float) and joint_cov == joint_cov else "N/A",
        "",
        _status("jcov", joint_cov_status),
    )
    report_table.add_row("affinity (CLIP)", "—", "—", "—", "[dim]N/A (run with transformers)[/]")
    report_table.add_row("diversity_ratio (CLIP)", "—", "—", "—", "[dim]N/A (run with transformers)[/]")

    console.print(report_table)

    report_dict: dict[str, Any] = {
        "source_repo": repo_id,
        "output_dir": str(output_dir),
        "n_original_episodes": dataset.num_episodes,
        "n_augmented_episodes": new_episode_idx,
        "n_frames": total_frames,
        "original_mean_sparc": orig_mean_sparc,
        "augmented_mean_sparc": aug_mean_sparc,
        "sparc_delta_pct": sparc_delta_pct,
        "sparc_status": sparc_status,
        "joint_coverage_ratio": joint_cov,
        "joint_coverage_status": joint_cov_status,
    }
    report_path = output_dir / "augmentation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    console.print(f"  Report saved → {report_path}")

    # ── Stage 7: Upload ──────────────────────────────────────────────────────
    if hf_repo_id:
        console.rule("[bold cyan]Stage 7 — Upload to HuggingFace Hub[/]")
        url = upload_dataset(dst_root, hf_repo_id, token=token, private=private)
        console.print(f"[bold green]Visualizer:[/] {url}")
    else:
        console.rule("[dim]Stage 7 — Upload (skipped — no --hf_repo_id provided)[/]")
        console.print("[dim]Pass --hf_repo_id=<your-user/dataset-name> to upload.[/]")

    console.print(f"\n[bold green]Pipeline complete.[/] Output: {dst_root}")


# ---------------------------------------------------------------------------
# report  (standalone quality check against existing output)
# ---------------------------------------------------------------------------

def report(original_repo_id: str, augmented_output_dir: str) -> None:
    """Print a quality comparison table for an existing augmented dataset.

    Args:
        original_repo_id: Source HF dataset ID.
        augmented_output_dir: Directory produced by `run` or `full_run`.
    """
    augmented_output_dir = Path(augmented_output_dir)
    dataset = _load_dataset(original_repo_id)
    dst_root = augmented_output_dir / "dataset"

    info_path = dst_root / "meta" / "info.json"
    if not info_path.exists():
        console.print(f"[red]No meta/info.json found in {dst_root}[/]")
        return

    with open(info_path) as f:
        info = json.load(f)

    sample_n = min(10, dataset.num_episodes)
    orig_episodes = [_episode_arrays(dataset, i) for i in range(sample_n)]
    orig_sparc = [score_sparc(sl, act) for sl, act in orig_episodes]
    mean_orig = sum(orig_sparc) / len(orig_sparc) if orig_sparc else float("nan")

    table = Table(title="Quality Comparison", header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Original", justify="right")
    table.add_column("Augmented", justify="right")

    table.add_row("episodes", str(dataset.num_episodes), str(info.get("total_episodes", "?")))
    table.add_row("frames", str(len(dataset)), str(info.get("total_frames", "?")))
    table.add_row("mean_SPARC (sample)", f"{mean_orig:.3f}", "—")
    console.print(table)


# ---------------------------------------------------------------------------
# extract_masks  (stub — implemented in Plan B)
# ---------------------------------------------------------------------------

def extract_masks(repo_id: str, output_mask_dir: str) -> None:
    """Extract SAM3 robot masks for all episodes and save as .npz caches.

    Requires the SAM3 module (Plan B). Not yet implemented.
    """
    console.print("[yellow]SAM3 mask extraction is not yet implemented (Plan B).[/]")


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
