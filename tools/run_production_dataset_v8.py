from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Objects to segment as foreground. The prompt is passed to SAM3; if SAM3 is
# unavailable the compositor falls back to SAM2 automatic masks, then to a
# brightness heuristic.
_TEXT_PROMPT = os.getenv("SAM3_TEXT_PROMPT", "plastic cup, lid, robot hand")


def main() -> int:
    """Build and upload the v8 production dataset (SAM3 background compositing)."""
    repo_root = Path(__file__).resolve().parents[1]
    output_repo = os.getenv("OUTPUT_REPO", "jgiegold/aloha_balanced_v8")

    run_cmd = [
        sys.executable,
        "-m",
        "aloha_augment.pipeline",
        "--source", "lerobot/aloha_static_cups_open",
        "--output", output_repo,
        "--episodes", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "--tier", "v8",
        "--sam3-text-prompt", _TEXT_PROMPT,
        "--video-backend", "pyav",
        "--force",
        "--image-writer-threads", "1",
        "--no-push",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root / "src"))

    print(f"Running v8 production build (SAM3 prompt: '{_TEXT_PROMPT}')...", flush=True)
    subprocess.run(run_cmd, cwd=repo_root, env=env, check=True)

    api = HfApi(token=env.get("HF_TOKEN"))
    local_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / output_repo
    api.create_repo(repo_id=output_repo, repo_type="dataset", exist_ok=True)
    print("Uploading dataset...", flush=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=output_repo,
        repo_type="dataset",
        commit_message="Upload v8 production dataset (SAM3 background compositing)",
    )
    print(f"Done: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{output_repo.replace('/', '%2F')}%2Fepisode_0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
