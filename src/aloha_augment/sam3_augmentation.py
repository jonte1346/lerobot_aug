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
    Load EfficientSAM3 (text-prompted, CPU-friendly) from Simon7108528/EfficientSAM3.

    Returns (model, Sam3Processor) or (None, None).
    """
    try:
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as EfficientSam3Processor
        from huggingface_hub import hf_hub_download
        import sam3
        from pathlib import Path

        # Ensure required asset files are present (vocabulary for text encoder)
        sam3_assets_dir = Path(sam3.__file__).parent / "assets"
        vocab_file = sam3_assets_dir / "bpe_simple_vocab_16e6.txt.gz"
        if not vocab_file.exists():
            sam3_assets_dir.mkdir(parents=True, exist_ok=True)
            import urllib.request
            # Vocab file is in GitHub repo, not HF hub
            vocab_url = "https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            try:
                urllib.request.urlretrieve(vocab_url, vocab_file)
                print(f"[SAM3] Downloaded vocabulary file from GitHub: {vocab_file}")
            except Exception as e:
                print(f"[SAM3] Warning: Could not download vocabulary file: {e}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = hf_hub_download(
            repo_id="Simon7108528/EfficientSAM3",
            filename="stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth",
        )
        model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit",
            model_name="11m",
            text_encoder_type="MobileCLIP-S1",
            text_encoder_context_length=77,
            text_encoder_pos_embed_table_size=77,
            interpolate_pos_embed=False,
        ).to(device)
        processor = EfficientSam3Processor(model, confidence_threshold=0.1)
        return model, processor
    except ImportError:
        return None, None
    except Exception as e:
        print(f"Warning: Could not load EfficientSAM3: {e}")
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
        sam3_frame_stride: int = 5,
        box_overlay_mode: bool = False,
    ):
        self.feather_radius = feather_radius
        self.brightness_threshold = brightness_threshold
        self.background_history = max(1, int(background_history))
        self.top_masks = max(1, int(top_masks))
        self.mask_iou_threshold = float(mask_iou_threshold)
        self.text_prompt = text_prompt
        self.sam3_frame_stride = max(1, int(sam3_frame_stride))
        self.box_overlay_mode = box_overlay_mode
        self._warned_predictor = False
        self._camera_key = "default"
        self._history: dict[str, deque[np.ndarray]] = {}
        # Per-camera mask cache for frame-stride reuse
        self._mask_cache: dict[str, tuple[int, np.ndarray]] = {}  # key → (frame_idx, mask)
        self._frame_counter: dict[str, int] = {}
        # Last computed masks (raw and feathered) per camera
        self._last_raw_mask: dict[str, np.ndarray] = {}      # key → H×W uint8 binary mask
        self._last_feathered_mask: dict[str, np.ndarray] = {}  # key → H×W float32 feathered [0, 1]
        # Last detected boxes per camera (box_overlay_mode)
        self._last_boxes: dict[str, tuple] = {}  # key → (boxes_tensor, scores_tensor)
        # Episode statistics
        self._episode_stats: dict[str, int | float] = {
            "frame_count": 0,
            "sam3_calls": 0,
            "cache_hits": 0,
            "mask_coverage_total": 0.0,
            "fallback_count": 0,
        }

        # Try SAM3 first; only load SAM2 if SAM3 is unavailable.
        self._sam3_model, self._sam3_processor = get_sam3_model()
        self.predictor = None if self._sam3_model is not None else get_sam3_predictor()

        if self._sam3_model is not None:
            print(f"SAM3BackgroundCompositor: using EfficientSAM3 (text-prompted, stride={self.sam3_frame_stride})")
        elif self.predictor is not None:
            print("SAM3BackgroundCompositor: using SAM2 automatic mask generator")
        else:
            print("SAM3BackgroundCompositor: using brightness heuristic (no SAM model available)")

    def reset_episode(self):
        self._history = {}
        self._mask_cache = {}
        self._frame_counter = {}
        self._last_raw_mask = {}
        self._last_feathered_mask = {}
        self._episode_stats = {
            "frame_count": 0,
            "sam3_calls": 0,
            "cache_hits": 0,
            "mask_coverage_total": 0.0,
            "fallback_count": 0,
        }

    def seed_background_history(self, frames_by_camera: dict):
        """Pre-populate background history with frames from a previous episode.

        Call this after reset_episode() to ensure the history pool starts with
        visually diverse frames from a different scene rather than being empty.

        Args:
            frames_by_camera: dict mapping camera_key → list[np.ndarray (H,W,3 uint8)]
        """
        for cam_key, frames in frames_by_camera.items():
            pool = self._history.setdefault(cam_key, deque(maxlen=self.background_history))
            for f in frames:
                pool.append(f)

    def set_camera_key(self, camera_key: str):
        self._camera_key = camera_key

    def _predict_mask_sam3(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Text-prompted EfficientSAM3 segmentation. Returns mask or None on failure."""
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(frame)
        with torch.no_grad():
            state = self._sam3_processor.set_image(pil_image)
            state = self._sam3_processor.set_text_prompt(
                prompt=self.text_prompt, state=state
            )
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        # Debug logging on first few frames per camera
        cam = self._camera_key
        frame_idx = self._frame_counter.get(cam, 0)
        if frame_idx < 3:  # Log first 3 frames
            print(f"[SAM3-DEBUG] cam={cam} frame={frame_idx}: masks={masks is not None}, "
                  f"boxes={boxes is not None}, scores={scores is not None}")
            if boxes is not None:
                print(f"  -> {len(boxes)} boxes detected")
            if scores is not None:
                scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
                print(f"  -> confidence scores: min={scores_np.min():.4f}, max={scores_np.max():.4f}, mean={scores_np.mean():.4f}")

        # Cache boxes for box_overlay_mode regardless of mask quality
        if boxes is not None:
            self._last_boxes[cam] = (boxes.cpu(), scores.cpu() if scores is not None else None)
        if masks is not None and len(masks) > 0:
            if isinstance(masks, torch.Tensor):
                return masks.any(dim=0).squeeze(0).cpu().numpy().astype(np.uint8)
            return np.logical_or.reduce(np.asarray(masks)).astype(np.uint8)
        return None

    def _draw_boxes_on_frame(self, frame_np: np.ndarray) -> np.ndarray:
        """Draw SAM3 bounding boxes on frame. Returns annotated copy."""
        from PIL import Image as PILImage, ImageDraw
        cam = self._camera_key
        boxes, scores = self._last_boxes.get(cam, (None, None))
        if boxes is None or len(boxes) == 0:
            return frame_np
        img = PILImage.fromarray(frame_np)
        draw = ImageDraw.Draw(img)
        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes[i].tolist()
            score = scores[i].item() if scores is not None else 0.0
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
            draw.text((x0 + 2, max(0, y0 - 14)), f"{score:.2f}", fill=(0, 255, 0))
        return np.array(img)

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
            # Reuse cached mask within frame stride to avoid running SAM3 on every frame
            cam = self._camera_key
            idx = self._frame_counter.get(cam, 0)
            self._frame_counter[cam] = idx + 1
            cached_idx, cached_mask = self._mask_cache.get(cam, (-999, None))
            if cached_mask is not None and (idx - cached_idx) < self.sam3_frame_stride:
                self._episode_stats["cache_hits"] += 1
                return cached_mask
            try:
                mask = self._predict_mask_sam3(frame)
                if mask is not None:
                    coverage = mask.mean()
                    bad = coverage < 0.01 or coverage > 0.80
                    if bad:
                        if cached_mask is not None:
                            # Bad mask — reuse last good cached mask
                            print(f"[SAM3] cam={cam} frame={idx}: bad coverage {coverage:.2%}, reusing cached mask")
                            self._episode_stats["fallback_count"] += 1
                            return cached_mask
                        else:
                            # No cache to fall back to — don't store a bad mask, drop to heuristic
                            print(f"[SAM3] cam={cam} frame={idx}: bad coverage {coverage:.2%}, no cache — falling back to heuristic")
                    else:
                        self._mask_cache[cam] = (idx, mask)
                        self._episode_stats["sam3_calls"] += 1
                        return mask
                else:
                    print(f"[SAM3] cam={cam} frame={idx}: no mask returned, falling back to heuristic")
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

        self._episode_stats["fallback_count"] += 1
        return simple_robot_mask_heuristic(frame, brightness_threshold=self.brightness_threshold)

    def __call__(self, frame):
        frame_np = _to_numpy_uint8_hwc(frame)
        self._episode_stats["frame_count"] += 1

        raw_mask = self._predict_mask(frame_np)

        if self.box_overlay_mode:
            # Draw SAM3 bounding boxes on the original frame; skip background compositing.
            annotated = self._draw_boxes_on_frame(frame_np)
            return _from_numpy_like(annotated, frame)

        cam_history = self._history.setdefault(self._camera_key, deque(maxlen=self.background_history))
        if cam_history:
            bg = cam_history[np.random.randint(0, len(cam_history))]
        else:
            bg = frame_np

        fg_rgba, feathered_alpha = extract_foreground_with_feathering(frame_np, raw_mask, feather_radius=self.feather_radius)

        # Store masks for later retrieval
        self._last_raw_mask[self._camera_key] = raw_mask
        self._last_feathered_mask[self._camera_key] = feathered_alpha

        # Track mask coverage
        coverage = raw_mask.astype(np.float32).mean()
        self._episode_stats["mask_coverage_total"] += coverage

        composited = composite_backgrounds(fg_rgba, feathered_alpha, [bg])[0]
        cam_history.append(frame_np)

        return _from_numpy_like(composited, frame)

    def get_last_raw_mask(self, camera_key: Optional[str] = None) -> Optional[np.ndarray]:
        """Get the most recently computed raw (binary) mask for a camera."""
        key = camera_key if camera_key is not None else self._camera_key
        return self._last_raw_mask.get(key)

    def get_last_feathered_mask(self, camera_key: Optional[str] = None) -> Optional[np.ndarray]:
        """Get the most recently computed feathered (alpha) mask for a camera."""
        key = camera_key if camera_key is not None else self._camera_key
        return self._last_feathered_mask.get(key)

    def log_episode_stats(self, episode_idx: int, camera_keys: list[str]):
        """Log episode statistics at end of episode."""
        if self._episode_stats["frame_count"] == 0:
            return
        
        avg_coverage = self._episode_stats["mask_coverage_total"] / self._episode_stats["frame_count"]
        model_info = "SAM3" if self._sam3_model is not None else ("SAM2" if self.predictor is not None else "heuristic")
        
        print(
            f"  Episode {episode_idx} | {model_info} | "
            f"frames={self._episode_stats['frame_count']} | "
            f"sam3_calls={self._episode_stats['sam3_calls']} | "
            f"cache_hits={self._episode_stats['cache_hits']} | "
            f"avg_coverage={avg_coverage:.1%} | "
            f"fallbacks={self._episode_stats['fallback_count']}"
        )
