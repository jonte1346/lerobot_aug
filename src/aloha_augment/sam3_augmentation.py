"""SAM3 mask augmentation with robot arm extraction and background compositing."""

from __future__ import annotations

import numpy as np
from collections import deque
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


def get_sam3_model():
    """
    Load SAM3 (text-prompted) from transformers if available.

    Returns (Sam3Model, Sam3Processor) or (None, None).
    """
    try:
        from transformers import Sam3Model, Sam3Processor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        return model, processor
    except ImportError:
        return None, None
    except OSError as e:
        msg = str(e)
        if "gated" in msg or "403" in msg or "restricted" in msg:
            print(f"Warning: SAM3 access denied (gated repo) — request access at https://huggingface.co/facebook/sam3. Falling back.")
        else:
            print(f"Warning: Could not load SAM3 (OSError): {e}")
        return None, None
    except Exception as e:
        print(f"Warning: Could not load SAM3: {e}")
        return None, None


def get_sam3_predictor():
    """
    Load SAM2 automatic mask generator as fallback if SAM3 is unavailable.

    Returns SAM2AutomaticMaskGenerator or None.
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large", device=device)
        return model
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Could not load SAM2: {e}")
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
    """Episode-aware background compositing.

    Mask prediction priority:
      1. SAM3 (transformers, text-prompted) — most accurate
      2. SAM2 automatic mask generator — prompt-free fallback
      3. Brightness heuristic — always available
    Only one of SAM3 / SAM2 is loaded to avoid double GPU memory usage.
    """

    def __init__(
        self,
        feather_radius: int = 10,
        brightness_threshold: int = 100,
        background_history: int = 24,
        top_masks: int = 4,
        mask_iou_threshold: float = 0.75,
        text_prompt: str = "plastic cup, lid, robot hand",
    ):
        self.feather_radius = feather_radius
        self.brightness_threshold = brightness_threshold
        self.background_history = max(1, int(background_history))
        self.top_masks = max(1, int(top_masks))
        self.mask_iou_threshold = float(mask_iou_threshold)
        self.text_prompt = text_prompt
        self._warned_predictor = False
        self._camera_key = "default"
        self._history: dict[str, deque[np.ndarray]] = {}

        # Try SAM3 first; only load SAM2 if SAM3 is unavailable.
        self._sam3_model, self._sam3_processor = get_sam3_model()
        self.predictor = None if self._sam3_model is not None else get_sam3_predictor()

        if self._sam3_model is not None:
            print("SAM3BackgroundCompositor: using SAM3 (text-prompted)")
        elif self.predictor is not None:
            print("SAM3BackgroundCompositor: using SAM2 automatic mask generator")
        else:
            print("SAM3BackgroundCompositor: using brightness heuristic (no SAM model available)")

    def reset_episode(self):
        self._history = {}

    def set_camera_key(self, camera_key: str):
        self._camera_key = camera_key

    def _predict_mask_sam3(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Text-prompted SAM3 segmentation. Returns mask or None on failure."""
        device = next(self._sam3_model.parameters()).device
        inputs = self._sam3_processor(
            images=frame,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = self._sam3_model(**inputs)
        results = self._sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        if results and results[0].get("masks") is not None:
            masks = results[0]["masks"]  # bool tensor (N, H, W)
            if len(masks) > 0:
                return masks.any(dim=0).cpu().numpy().astype(np.uint8)
        return None

    def _predict_mask_sam2(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """SAM2 automatic mask generator. Returns mask or None on failure."""
        anns = self.predictor.generate(frame)
        if not anns:
            return None
        anns_sorted = sorted(anns, key=lambda x: x["area"], reverse=True)
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for ann in anns_sorted[: self.top_masks]:
            if ann.get("predicted_iou", 1.0) >= self.mask_iou_threshold:
                combined = np.logical_or(combined, ann["segmentation"]).astype(np.uint8)
        return combined if combined.any() else None

    def _predict_mask(self, frame: np.ndarray) -> np.ndarray:
        if self._sam3_model is not None:
            try:
                mask = self._predict_mask_sam3(frame)
                if mask is not None:
                    return mask
            except Exception as exc:
                if not self._warned_predictor:
                    print(f"Warning: SAM3 failed, falling back to heuristic: {exc}")
                    self._warned_predictor = True
        elif self.predictor is not None:
            try:
                mask = self._predict_mask_sam2(frame)
                if mask is not None:
                    return mask
            except Exception as exc:
                if not self._warned_predictor:
                    print(f"Warning: SAM2 failed, falling back to heuristic: {exc}")
                    self._warned_predictor = True

        return simple_robot_mask_heuristic(frame, brightness_threshold=self.brightness_threshold)

    def __call__(self, frame):
        frame_np = _to_numpy_uint8_hwc(frame)

        cam_history = self._history.setdefault(self._camera_key, deque(maxlen=self.background_history))
        if cam_history:
            bg = cam_history[np.random.randint(0, len(cam_history))]
        else:
            bg = frame_np

        mask = self._predict_mask(frame_np)
        fg_rgba, alpha = extract_foreground_with_feathering(frame_np, mask, feather_radius=self.feather_radius)
        composited = composite_backgrounds(fg_rgba, alpha, [bg])[0]

        cam_history.append(frame_np)

        return _from_numpy_like(composited, frame)
