from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------
# EPISODES          — space-separated episode indices, e.g. "0 1 2"  (default: "0")
# SAM3_STRIDE       — reuse SAM3 mask every N frames             (default: 5)
# KEEP_EVERY_N      — output every Nth frame                     (default: 1 = no downsampling)
# BACKGROUND_HISTORY— rolling background pool size               (default: 200)
# WRITER_THREADS    — parallel video encoding threads            (default: 4)
# OUTPUT_REPO       — HuggingFace dataset repo                   (default below)
# UPLOAD=1          — set to upload to HF after building         (default: off)
# SAM3_TEXT_PROMPT  — text prompt passed to SAM3                 (default below)
# ---------------------------------------------------------------------------

_TEXT_PROMPT    = os.getenv("SAM3_TEXT_PROMPT", "plastic cup, lid, robot hand")
_OUTPUT_REPO    = os.getenv("OUTPUT_REPO", "jgiegold/aloha_balanced_v9_mask")
_EPISODES       = os.getenv("EPISODES", "0").split()
_SAM3_STRIDE    = int(os.getenv("SAM3_STRIDE", "5"))
_KEEP_EVERY_N   = int(os.getenv("KEEP_EVERY_N", "1"))
_BG_HISTORY     = int(os.getenv("BACKGROUND_HISTORY", "200"))
_WRITER_THREADS = int(os.getenv("WRITER_THREADS", "4"))
_UPLOAD         = os.getenv("UPLOAD", "0") == "1"


def main() -> int:
    """v9_mask build: full episode length, tight mask stride, wide background pool, with SAM3 masks."""
    repo_root = Path(__file__).resolve().parents[1]

    frames_per_ep      = 130 // max(1, _KEEP_EVERY_N)
    sam3_calls_per_ep  = frames_per_ep * 4 / _SAM3_STRIDE   # 4 cameras
    est_min            = len(_EPISODES) * sam3_calls_per_ep * 12 / 60
    print(
        f"v9_mask build | episodes={_EPISODES} | sam3-stride={_SAM3_STRIDE} | "
        f"keep-every-n={_KEEP_EVERY_N} | bg-history={_BG_HISTORY} | "
        f"threads={_WRITER_THREADS} | "
        f"estimated ~{est_min:.0f} min on CPU | upload={'yes' if _UPLOAD else 'no'}",
        flush=True,
    )

    run_cmd = [
        sys.executable,
        "-m",
        "aloha_augment.pipeline",
        "--source", "lerobot/aloha_static_cups_open",
        "--output", _OUTPUT_REPO,
        "--episodes", *_EPISODES,
        "--tier", "v9",
        "--sam3-text-prompt", _TEXT_PROMPT,
        "--sam3-frame-stride", str(_SAM3_STRIDE),
        "--sam3-background-history", str(_BG_HISTORY),
        "--keep-every-n", str(_KEEP_EVERY_N),
        "--video-backend", "pyav",
        "--force",
        "--image-writer-threads", str(_WRITER_THREADS),
        "--no-push",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root / "src"))

    subprocess.run(run_cmd, cwd=repo_root, env=env, check=True)

    if _UPLOAD:
        from huggingface_hub import HfApi

        api = HfApi(token=env.get("HF_TOKEN"))
        local_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / _OUTPUT_REPO
        api.create_repo(repo_id=_OUTPUT_REPO, repo_type="dataset", exist_ok=True)
        print("Uploading dataset...", flush=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=_OUTPUT_REPO,
            repo_type="dataset",
            commit_message="Upload v9_mask production dataset (full frames, tight SAM3 stride, with segmentation masks)",
        )
        print(
            f"Done: https://huggingface.co/spaces/lerobot/visualize_dataset"
            f"?path=%2F{_OUTPUT_REPO.replace('/', '%2F')}%2Fepisode_0"
        )
    else:
        print("Build complete. Set UPLOAD=1 to push to HuggingFace.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
