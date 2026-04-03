"""SAM3 mask augmentation with robot arm extraction and background compositing."""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import torch


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
    
    # Feather edges with Gaussian blur
    feathered_alpha = cv2.GaussianBlur(alpha, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
    
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
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Simple brightness-based detection
    mask = (gray > brightness_threshold).astype(np.uint8)
    
    # Morphological operations to clean up
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
