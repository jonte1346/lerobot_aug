# LeRobot v3 Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets from Hugging Face Hub, applying visual and geometric transforms to multiply training data for robot learning.

## What it does

- Downloads a source dataset, applies augmentations, creates new episodes, and pushes the result back to HF Hub
- Visual augmentations: ColorJitter, GaussianBlur, Sharpness, RandomErasing, StaticErasing, DriftingBlob
- Geometric augmentations: HorizontalFlip - flips images and mirrors action/state vectors so the training signal stays consistent
- Temporal augmentations: FrameDecimator - drops every Nth frame

## Setup

```bash
uv sync --extra pipeline
```

If you prefer conda, use the included `environment.yml`.

## Usage

```bash
uv run aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_augmented \
  --augmentations color_jitter \
  --num-passes 2 \
  --include-originals

uv run aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_flipped \
  --augmentations horizontal_flip \
  --robot-type aloha \
  --num-passes 1 \
  --no-push \
  --force
```

Use `--no-push` to skip uploading to HF Hub and `--force` to overwrite the local cache for the output dataset. Run `uv run aloha-augment --help` for all options.

## Available augmentations

| Name | Description |
|---|---|
| `color_jitter` | Random brightness, contrast, saturation, hue |
| `gaussian_blur` | Random Gaussian blur |
| `sharpness` | Sharpness adjustment |
| `random_erasing` | Random rectangular patches per frame |
| `static_erasing` | Fixed rectangular patch per episode |
| `drifting_blob` | Soft blob drifting across frames |
| `frame_decimate` | Remove every Nth frame |
| `horizontal_flip` | Flip images and mirror actions/states |

## Project structure

```
src/aloha_augment/
  pipeline.py        # Main CLI
  transforms.py      # Custom transforms
  explore_dataset.py # Dataset inspection helper
environment.yml      # Conda environment
```
