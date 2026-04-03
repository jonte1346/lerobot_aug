"""SAM3 mask augmentation with robot arm extraction and background compositing."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import torch

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some envs
    cv2 = None


def extract_foreground_with_feathering(
    frame: np.ndarray, mask: np.ndarray, feather_radius: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract foreground (robot arm) from frame using mask with feathered edges.
    
    Args:
        frame: RGB frame, shape (H, W, 3).
        mask: Binary mask where 1 = robot, 0 = background.
        feather_radius: Gaussian blur radius for smooth feathering.
    
    Returns:
        (foreground_rgba, feathered_alpha)
    """
    # Expand mask to alpha channel
    alpha = mask.astype(np.float32)
    
    # Feather edges with Gaussian blur when available.
    if cv2 is not None and feather_radius > 0:
        feathered_alpha = cv2.GaussianBlur(alpha, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
    else:
        feathered_alpha = alpha
    
    # Create RGBA foreground
    foreground = np.concatenate([frame, (feathered_alpha * 255).astype(np.uint8)[..., None]], axis=-1)
    
    return foreground, feathered_alpha


def composite_backgrounds(
    foreground_rgba: np.ndarray,
    feathered_alpha: np.ndarray,
    background_frames: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Composite robot arm foreground onto different background frames.
    
    Args:
        foreground_rgba: Foreground with alpha, shape (H, W, 4).
        feathered_alpha: Feathered alpha channel, shape (H, W).
        background_frames: List of background RGB frames.
    
    Returns:
        List of composited RGB frames.
    """
    composited = []
    fg_rgb = foreground_rgba[..., :3]
    
    for bg in background_frames:
        # Ensure same shape
        if bg.shape != fg_rgb.shape:
            bg = cv2.resize(bg, (fg_rgb.shape[1], fg_rgb.shape[0]))
        
        # Blend: fg_alpha * fg + (1 - fg_alpha) * bg
        alpha = feathered_alpha[..., None]
        result = (alpha * fg_rgb + (1 - alpha) * bg).astype(np.uint8)
        composited.append(result)
    
    return composited


def simple_robot_mask_heuristic(frame: np.ndarray, brightness_threshold: int = 100) -> np.ndarray:
    """
    Simple heuristic to detect robot arm as relatively bright regions.
    
    This is a placeholder for SAM3. In practice, you'd use SAM3's video predictor.
    
    Args:
        frame: RGB frame, shape (H, W, 3).
        brightness_threshold: Pixel brightness threshold.
    
    Returns:
        Binary mask where 1 = likely robot, 0 = background.
    """
    # Convert to grayscale.
    if cv2 is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.mean(axis=2).astype(np.uint8)
    
    # Simple brightness-based detection
    mask = (gray > brightness_threshold).astype(np.uint8)
    
    # Morphological clean-up when OpenCV is present.
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def augment_episode_with_sam3(
    episode_frames: list[np.ndarray],
    background_pool: Optional[list[np.ndarray]] = None,
    feather_radius: int = 10,
) -> list[np.ndarray]:
    """
    Augment an episode by extracting robot arm and compositing onto backgrounds.
    
    Args:
        episode_frames: List of RGB frames for the episode.
        background_pool: Pool of background frames to composite onto.
        feather_radius: Feathering radius for smooth compositing.
    
    Returns:
        Augmented frames with different backgrounds.
    """
    if not background_pool or len(background_pool) == 0:
        # Fallback: use frame itself as background source
        background_pool = episode_frames
    
    augmented = []
    
    for frame in episode_frames:
        # Extract robot mask (using heuristic; in production use SAM3)
        mask = simple_robot_mask_heuristic(frame)
        
        # Extract foreground with feathering
        foreground_rgba, feathered_alpha = extract_foreground_with_feathering(
            frame, mask, feather_radius=feather_radius
        )
        
        # Sample random backgrounds and composite
        n_backgrounds = min(2, len(background_pool))
        selected_backgrounds = np.random.choice(len(background_pool), n_backgrounds, replace=False)
        
        for bg_idx in selected_backgrounds:
            bg_frame = background_pool[bg_idx]
            composited = composite_backgrounds(foreground_rgba, feathered_alpha, [bg_frame])[0]
            augmented.append(composited)
    
    return augmented


def get_sam3_predictor():
    """
    Initialize SAM3 video predictor if available.
    
    Returns SAM3 predictor or None if not installed.
    """
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_checkpoint = "facebook/sam2-hiera-large"
        model_cfg = "sam2_hiera_l.yaml"
        predictor = SAM2ImagePredictor.from_pretrained(sam2_checkpoint, model_cfg, device=device)
        return predictor
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Could not load SAM3: {e}")
        return None


def _to_numpy_uint8_hwc(frame):
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return arr


def _from_numpy_like(arr: np.ndarray, like):
    if isinstance(like, torch.Tensor):
        out = torch.from_numpy(arr)
        if like.ndim == 3 and like.shape[0] in (1, 3, 4):
            out = out.permute(2, 0, 1)
        out = out.to(device=like.device)
        if like.dtype != torch.uint8:
            out = out.to(dtype=like.dtype) / 255.0
        else:
            out = out.to(dtype=like.dtype)
        return out
    return arr


class SAM3BackgroundCompositor:
    """Episode-aware background compositing with SAM3 predictor fallback."""

    def __init__(
        self,
        feather_radius: int = 10,
        brightness_threshold: int = 100,
        background_history: int = 24,
    ):
        self.feather_radius = feather_radius
        self.brightness_threshold = brightness_threshold
        self.background_history = max(1, int(background_history))
        self.predictor = get_sam3_predictor()
        self._warned_predictor = False
        self._camera_key = "default"
        self._history: dict[str, list[np.ndarray]] = {}

    def reset_episode(self):
        self._history = {}

    def set_camera_key(self, camera_key: str):
        self._camera_key = camera_key

    def _predict_mask(self, frame: np.ndarray) -> np.ndarray:
        # SAM2/SAM3 API compatibility varies by install; fallback is always available.
        if self.predictor is not None:
            try:
                self.predictor.set_image(frame)
                if hasattr(self.predictor, "predict"):
                    pred = self.predictor.predict()
                    if isinstance(pred, tuple) and len(pred) >= 1:
                        masks = pred[0]
                        if masks is not None and len(masks) > 0:
                            m = masks[0]
                            return (m > 0.5).astype(np.uint8)
            except Exception as exc:  # pragma: no cover - external predictor variance
                if not self._warned_predictor:
                    print(f"Warning: SAM3 predictor failed, using heuristic mask fallback: {exc}")
                    self._warned_predictor = True

        return simple_robot_mask_heuristic(frame, brightness_threshold=self.brightness_threshold)

    def __call__(self, frame):
        frame_np = _to_numpy_uint8_hwc(frame)

        cam_history = self._history.setdefault(self._camera_key, [])
        if cam_history:
            bg = cam_history[np.random.randint(0, len(cam_history))]
        else:
            bg = frame_np

        mask = self._predict_mask(frame_np)
        fg_rgba, alpha = extract_foreground_with_feathering(frame_np, mask, feather_radius=self.feather_radius)
        composited = composite_backgrounds(fg_rgba, alpha, [bg])[0]

        cam_history.append(frame_np)
        if len(cam_history) > self.background_history:
            cam_history.pop(0)

        return _from_numpy_like(composited, frame)
