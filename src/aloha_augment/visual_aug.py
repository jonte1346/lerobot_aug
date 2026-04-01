"""Visual augmentation: color jitter, occlusion patches, and video processing."""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def apply_color_jitter(
    frame: np.ndarray,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    hue_shift_deg: tuple[float, float] = (-15.0, 15.0),
    saturation_range: tuple[float, float] = (0.8, 1.2),
    rng: random.Random | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply random color jitter to a BGR frame using HSV conversion.

    Returns:
        (augmented_frame, params_dict) — params_dict records the sampled values.
    """
    if rng is None:
        rng = random.Random()

    brightness = rng.uniform(*brightness_range)
    contrast = rng.uniform(*contrast_range)
    hue_shift = rng.uniform(*hue_shift_deg)
    saturation = rng.uniform(*saturation_range)

    params = {
        "brightness": brightness,
        "contrast": contrast,
        "hue_shift": hue_shift,
        "saturation": saturation,
    }

    # Apply brightness + contrast in BGR before HSV conversion
    out = frame.astype(np.float32)
    out = out * contrast + (brightness - 1.0) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    # HSV for hue and saturation
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2.0) % 180.0  # OpenCV H is 0-180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out, params


def apply_occlusion_patch(
    frame: np.ndarray,
    size_ratio_range: tuple[float, float] = (0.05, 0.2),
    rng: random.Random | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place a black rectangle at a random position on the frame.

    Returns:
        (augmented_frame, params_dict)
    """
    if rng is None:
        rng = random.Random()

    h, w = frame.shape[:2]
    ratio = rng.uniform(*size_ratio_range)
    patch_h = max(1, int(h * ratio))
    patch_w = max(1, int(w * ratio))
    x = rng.randint(0, max(0, w - patch_w))
    y = rng.randint(0, max(0, h - patch_h))

    params = {"patch_x": x, "patch_y": y, "patch_w": patch_w, "patch_h": patch_h}

    out = frame.copy()
    out[y : y + patch_h, x : x + patch_w] = 0
    return out, params


def _detect_codec(video_path: str | Path) -> str:
    """Return the codec name of the first video stream using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return info["streams"][0]["codec_name"]


def _transcode_to_h264(src: str | Path, dst: str | Path) -> None:
    """Re-encode src to H.264 at dst using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


def apply_video_augmentation(
    video_path: str | Path,
    output_path: str | Path,
    config_dict: dict[str, Any],
    seed: int = 42,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    hue_shift_deg: tuple[float, float] = (-15.0, 15.0),
    saturation_range: tuple[float, float] = (0.8, 1.2),
    occlusion_size_ratio_range: tuple[float, float] = (0.05, 0.2),
) -> None:
    """Read a video, apply color jitter and occlusion, write H.264 output.

    Handles AV1-encoded input by transcoding to H.264 first.
    All random parameters are logged to config_dict for reproducibility.

    Args:
        video_path: Path to the source video.
        output_path: Path where the augmented video will be written.
        config_dict: Dict that will be updated with the augmentation parameters.
        seed: Random seed for reproducibility.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Transcode AV1 → H.264 if needed
    work_path = video_path
    tmp_file = None
    try:
        codec = _detect_codec(video_path)
        if codec in ("av1", "libaom-av1"):
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_file.close()
            _transcode_to_h264(video_path, tmp_file.name)
            work_path = Path(tmp_file.name)
    except (subprocess.CalledProcessError, KeyError, IndexError):
        pass  # fall through and try to open directly

    cap = cv2.VideoCapture(str(work_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {work_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_params = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, jitter_params = apply_color_jitter(
            frame,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            hue_shift_deg=hue_shift_deg,
            saturation_range=saturation_range,
            rng=rng,
        )
        frame, occ_params = apply_occlusion_patch(
            frame,
            size_ratio_range=occlusion_size_ratio_range,
            rng=rng,
        )
        frame_params.append({"frame": frame_idx, **jitter_params, **occ_params})
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if tmp_file is not None:
        Path(tmp_file.name).unlink(missing_ok=True)

    config_dict.update(
        {
            "seed": seed,
            "source_video": str(video_path),
            "output_video": str(output_path),
            "brightness_range": brightness_range,
            "contrast_range": contrast_range,
            "hue_shift_deg": hue_shift_deg,
            "saturation_range": saturation_range,
            "occlusion_size_ratio_range": occlusion_size_ratio_range,
            "frame_params": frame_params,
        }
    )
