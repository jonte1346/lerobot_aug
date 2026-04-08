"""SAM3 bounding-box overlay test run with optional upload.

Processes one or more episodes from lerobot/aloha_static_cups_open using
the v9_boxes tier: runs EfficientSAM3 on every frame with the text prompt
"plastic cup, lid", draws green bounding boxes on the camera frames, and
writes a lerobot dataset where the augmented episode videos show the boxes.

The original episodes are also included (unchanged) so you can compare
side-by-side in the visualizer.

Usage:
    python tools/run_sam3_box_overlay.py
    python tools/run_sam3_box_overlay.py --episodes 0 1 2 --upload
    UPLOAD=1 EPISODES="0 1 2" python tools/run_sam3_box_overlay.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_EPISODES     = os.getenv("EPISODES", "0").split()
_TEXT_PROMPT  = os.getenv("SAM3_TEXT_PROMPT", "plastic cup, lid, robot hand")
_OUTPUT_REPO  = os.getenv("OUTPUT_REPO", "jgiegold/aloha_sam3_boxes")
_UPLOAD       = os.getenv("UPLOAD", "0") == "1"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", type=int, default=[int(e) for e in _EPISODES])
    parser.add_argument("--upload", action="store_true", default=_UPLOAD)
    parser.add_argument("--output", default=_OUTPUT_REPO)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    episodes_str = " ".join(str(e) for e in args.episodes)
    print(
        f"SAM3 box overlay | episodes={episodes_str} | "
        f"prompt='{_TEXT_PROMPT}' | output={args.output} | upload={args.upload}"
    )
    print(f"Working from: {repo_root}")

    run_cmd = [
        sys.executable,
        "-m", "aloha_augment.pipeline",
        "--source", "lerobot/aloha_static_cups_open",
        "--output", args.output,
        "--episodes", *[str(e) for e in args.episodes],
        "--tier", "v9_boxes",
        "--sam3-text-prompt", _TEXT_PROMPT,
        "--keep-every-n", "1",
        "--video-backend", "pyav",
        "--force",
        "--image-writer-threads", "4",
        "--no-push",
        "--num-passes", "1",
    ]

    env = os.environ.copy()
    # Include both repo src and SAM3 in PYTHONPATH for subprocess
    src_path = str(repo_root / "src")
    sam3_path = os.getenv("SAM3_PATH", "/kaggle/working/sam3_repo")
    pythonpath = f"{src_path}:{sam3_path}" if os.path.exists(sam3_path) else src_path
    env["PYTHONPATH"] = pythonpath  # Override any existing PYTHONPATH
    print(f"PYTHONPATH: {pythonpath}")

    result = subprocess.run(run_cmd, cwd=repo_root, env=env)
    if result.returncode != 0:
        print("Pipeline failed — check logs above")
        return result.returncode

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=env.get("HF_TOKEN"))
        local_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / args.output
        api.create_repo(repo_id=args.output, repo_type="dataset", exist_ok=True)
        print(f"\nUploading to {args.output}…")
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=args.output,
            repo_type="dataset",
            commit_message="SAM3 bounding box overlay",
        )
        encoded = args.output.replace("/", "%2F")
        print(
            f"\nDone! View in visualizer:\n"
            f"  https://huggingface.co/spaces/lerobot/visualize_dataset"
            f"?path=%2F{encoded}%2Fepisode_0"
        )
    else:
        print("\nBuild complete. Run with --upload or UPLOAD=1 to push to HuggingFace.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
