from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    """Build and upload the production dataset recipe."""
    repo_root = Path(__file__).resolve().parents[1]
    tier = os.getenv("TIER", "v7")  # set TIER=v7_sam to include SAM2 background augmentation
    output_repo = os.getenv("OUTPUT_REPO", f"jgiegold/aloha_balanced_{tier}")

    run_cmd = [
        sys.executable,
        "-m",
        "aloha_augment.pipeline",
        "--source", "lerobot/aloha_static_cups_open",
        "--output", output_repo,
        "--episodes", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "--tier", tier,
        "--video-backend", "pyav",
        "--force",
        "--image-writer-threads", "1",
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
        commit_message="Upload production dataset",
    )
    print(f"Done: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{repo_id.replace('/', '%2F')}%2Fepisode_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
