"""HuggingFace Hub upload with dataset validation and visualizer URL."""

from __future__ import annotations

from pathlib import Path


_REQUIRED_PATHS = [
    "meta/info.json",
    "meta/episodes.jsonl",
    "data",
    "videos",
]

VISUALIZER_BASE = "https://huggingface.co/spaces/lerobot/visualize_dataset"


def upload_dataset(
    output_dir: str | Path,
    hf_repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Validate and upload an augmented dataset to HuggingFace Hub.

    Args:
        output_dir: Local directory produced by the pipeline (contains meta/, data/, videos/).
        hf_repo_id: Target repo ID, e.g. 'myuser/aloha_cups_aug'.
        token: HuggingFace write token. Falls back to HF_TOKEN env var if None.
        private: Whether to create the repo as private.

    Returns:
        The visualizer URL for episode_0.

    Raises:
        FileNotFoundError: If any required path is missing from output_dir.
    """
    from huggingface_hub import HfApi

    output_dir = Path(output_dir)

    # Validate
    missing = [p for p in _REQUIRED_PATHS if not (output_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"output_dir is missing required paths: {missing}\n"
            f"Make sure the pipeline ran to completion before uploading."
        )

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=hf_repo_id,
        repo_type="dataset",
    )

    url = f"{VISUALIZER_BASE}?path=%2F{hf_repo_id}%2Fepisode_0"
    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{hf_repo_id}")
    print(f"Visualizer:          {url}\n")
    return url
