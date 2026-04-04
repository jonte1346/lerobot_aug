from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    """Build and upload the production no-SAM smooth dataset recipe."""
    repo_root = Path(__file__).resolve().parents[1]
    output_repo = os.getenv("OUTPUT_REPO", "jgiegold/aloha_balanced_v7_motor5_nosam_smooth")

    run_cmd = [
        sys.executable,
        "-m",
        "aloha_augment.pipeline",
        "--source",
        "lerobot/aloha_static_cups_open",
        "--output",
        output_repo,
        "--episodes",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "--num-passes",
        "1",
        "--include-originals",
        "--include-originals-decimated",
        "--augmentations",
        "color_jitter",
        "gaussian_blur",
        "sharpness",
        "random_erasing",
        "horizontal_flip",
        "--robot-type",
        "aloha",
        "--action-shift",
        "4",
        "--keep-every-n",
        "3",
        "--tail-drop-max",
        "8",
        "--prefilter-mode",
        "fast",
        "--prefilter-sample-every-n",
        "8",
        "--min-action-delta",
        "0.015",
        "--max-action-jerk",
        "0.0145",
        "--temporal-jitter-pct",
        "0.15",
        "--action-noise-std",
        "0.003",
        "--action-smoothing",
        "savgol",
        "--savgol-window-length",
        "7",
        "--savgol-polyorder",
        "2",
        "--smooth-exclude-indices",
        "6",
        "13",
        "--sparc-threshold",
        "-10.0",
        "--saturation-threshold",
        "0.12",
        "--video-backend",
        "pyav",
        "--force",
        "--image-writer-threads",
        "1",
        "--no-push",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root / "src"))

    print("Running production build...", flush=True)
    subprocess.run(run_cmd, cwd=repo_root, env=env, check=True)

    api = HfApi(token=env.get("HF_TOKEN"))
    repo_id = output_repo
    local_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print("Uploading dataset...", flush=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload production no-SAM smoothed Tier 2 motor dataset",
    )
    print(f"Done: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{repo_id.replace('/', '%2F')}%2Fepisode_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
