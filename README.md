# LeRobot v3 Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets from Hugging Face Hub for robot learning.

## What it does

- Downloads a source dataset, applies augmentations, creates new episodes, and optionally pushes to HF Hub
- Visual augmentations: ColorJitter, GaussianBlur, Sharpness, RandomErasing, StaticErasing, DriftingBlob
- Geometric augmentations: HorizontalFlip - flips images and mirrors action/state vectors so the training signal stays consistent
- Temporal controls: frame decimation, action shift in decimated space, tail drop, temporal jitter
- Optional action smoothing: Savitzky-Golay filtering with per-dimension exclusions

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

Tiered runs for the dataset-fault diagnosis pipeline:

```bash
uv run aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_tier1 \
  --tier tier1 \
  --episodes 0 1 2 3 4 \
  --video-backend pyav

uv run aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_tier2 \
  --tier tier2 \
  --episodes 0 1 2 3 4 \
  --video-backend pyav

uv run aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_tier3 \
  --tier tier3 \
  --episodes 0 1 2 3 4 \
  --video-backend pyav
```

Production v7 recipe:

```bash
uv run --with scipy aloha-augment \
  --source lerobot/aloha_static_cups_open \
  --output your-username/aloha_balanced_v7 \
  --episodes 0 1 2 3 4 5 6 7 8 9 \
  --num-passes 1 \
  --include-originals \
  --include-originals-decimated \
  --augmentations color_jitter gaussian_blur sharpness random_erasing horizontal_flip \
  --robot-type aloha \
  --action-shift 4 \
  --keep-every-n 3 \
  --tail-drop-max 8 \
  --prefilter-mode fast \
  --prefilter-sample-every-n 8 \
  --min-action-delta 0.015 \
  --max-action-jerk 0.0145 \
  --temporal-jitter-pct 0.15 \
  --action-noise-std 0.003 \
  --action-smoothing savgol \
  --savgol-window-length 7 \
  --savgol-polyorder 2 \
  --smooth-exclude-indices 6 13 \
  --sparc-threshold -10.0 \
  --saturation-threshold 0.12 \
  --video-backend pyav \
  --force \
  --image-writer-threads 1 \
  --no-push
```

Use `--no-push` to skip uploading to HF Hub and `--force` to overwrite the local cache for the output dataset. Run `uv run aloha-augment --help` for all options.

## Operational Runner Script

The reusable production helper lives in `tools/` because it is operational automation, not part of the package API.

- Canonical runner: `tools/run_production_dataset_v7.py`

Example:

```bash
$env:HF_TOKEN="your_token"
uv run --with scipy python tools/run_production_dataset_v7.py
```

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
| `sam3` | Background compositing with SAM3 predictor fallback |

Temporal controls for the tier runs:

- `--action-shift 4` aligns each training frame with the action four steps in the future.
- `--keep-every-n N` keeps only every Nth frame to reduce temporal redundancy.
- `--frame-stride-cycle 2 3 4` varies the stride across passes for timing diversity.
- `--min-action-delta` rejects near-static episodes before augmentation.

## Project structure

```
src/aloha_augment/
  pipeline.py        # Main CLI
  transforms.py      # Custom transforms
  explore_dataset.py # Dataset inspection helper
tools/
  run_production_dataset_v7.py # Canonical production build+upload helper
environment.yml      # Conda environment
```
