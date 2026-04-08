"""
Custom transforms for LeRobot datasets.

Contains both image-level transforms (StaticErasing, DriftingBlob) and
episode-level transforms (FrameDecimator) that are used by augment_dataset.py.
"""

from __future__ import annotations

import math
import random

import torch
from torchvision.transforms import functional as F


class FrameDecimator:
    """Remove every Nth frame from an episode.

    Example: remove_every_n=5 keeps 4 out of every 5 frames,
    dropping frames at positions 4, 9, 14, ... (0-indexed).
    """

    def __init__(self, remove_every_n: int = 5):
        if remove_every_n < 2:
            raise ValueError(f"remove_every_n must be >= 2, got {remove_every_n}")
        self.remove_every_n = remove_every_n

    def should_keep(self, frame_index: int) -> bool:
        return (frame_index + 1) % self.remove_every_n != 0

    def __repr__(self):
        return f"FrameDecimator(remove_every_n={self.remove_every_n})"


class FrameStride:
    """Keep every Nth frame with an optional starting offset."""

    def __init__(self, keep_every_n: int = 4, start_offset: int = 0):
        if keep_every_n < 1:
            raise ValueError(f"keep_every_n must be >= 1, got {keep_every_n}")
        if start_offset < 0 or start_offset >= keep_every_n:
            raise ValueError(f"start_offset must be in [0, {keep_every_n - 1}], got {start_offset}")
        self.keep_every_n = keep_every_n
        self.start_offset = start_offset

    def should_keep(self, frame_index: int) -> bool:
        return (frame_index - self.start_offset) % self.keep_every_n == 0

    def __repr__(self):
        return f"FrameStride(keep_every_n={self.keep_every_n}, start_offset={self.start_offset})"


class StaticErasing:
    """Erases a fixed rectangle sampled once and applied to every frame."""

    def __init__(self, scale=(0.02, 0.15), value=0.0):
        self.scale = scale
        self.value = value
        self.i = self.j = self.h = self.w = 0

    def resample(self, img_h, img_w):
        """Pick a new random rectangle for this episode."""
        area = img_h * img_w
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect = random.uniform(0.5, 2.0)
        self.h = int(round((erase_area * aspect) ** 0.5))
        self.w = int(round((erase_area / aspect) ** 0.5))
        self.h = min(self.h, img_h)
        self.w = min(self.w, img_w)
        self.i = random.randint(0, img_h - self.h)
        self.j = random.randint(0, img_w - self.w)

    def __call__(self, img):
        img = img.clone()
        img[:, self.i : self.i + self.h, self.j : self.j + self.w] = self.value
        return img

    def __repr__(self):
        return f"StaticErasing(scale={self.scale})"


class DriftingBlob:
    """A soft blob that drifts smoothly across frames."""

    def __init__(self, radius: int = 30, speed: float = 2.0, softness: float = 0.6, opacity: float = 0.5):
        self.radius = radius
        self.speed = speed
        self.softness = softness
        self.opacity = opacity
        self.cy = 0.0
        self.cx = 0.0
        self.vy = 0.0
        self.vx = 0.0
        self.img_h = 0
        self.img_w = 0
        self._mask = self._make_mask(radius, softness)

    @staticmethod
    def _make_mask(radius, softness):
        size = 2 * radius + 1
        center = radius
        sigma = radius * max(softness, 0.05)
        y = torch.arange(size, dtype=torch.float32) - center
        x = torch.arange(size, dtype=torch.float32) - center
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist_sq = yy**2 + xx**2
        mask = torch.exp(-dist_sq / (2 * sigma**2))
        mask[dist_sq > radius**2] = 0.0
        if mask.max() > 0:
            mask = mask / mask.max()
        return mask

    def resample(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w
        self.cy = random.uniform(self.radius, img_h - self.radius)
        self.cx = random.uniform(self.radius, img_w - self.radius)
        angle = random.uniform(0, 2 * math.pi)
        self.vy = self.speed * math.sin(angle)
        self.vx = self.speed * math.cos(angle)

    def __call__(self, img):
        img = img.clone()
        c, h, w = img.shape
        r = self.radius
        mask = self._mask

        cy_int = int(round(self.cy))
        cx_int = int(round(self.cx))

        m_top = max(0, r - cy_int)
        m_left = max(0, r - cx_int)
        m_bottom = min(2 * r + 1, h - cy_int + r)
        m_right = min(2 * r + 1, w - cx_int + r)

        i_top = max(0, cy_int - r)
        i_left = max(0, cx_int - r)
        i_bottom = min(h, cy_int + r + 1)
        i_right = min(w, cx_int + r + 1)

        if i_top >= i_bottom or i_left >= i_right:
            self._advance()
            return img

        mask_patch = mask[m_top:m_bottom, m_left:m_right]
        img_patch = img[:, i_top:i_bottom, i_left:i_right]

        mask_sum = mask_patch.sum()
        if mask_sum > 0:
            avg_color = (img_patch * mask_patch.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
        else:
            self._advance()
            return img

        alpha = mask_patch.unsqueeze(0) * self.opacity
        img[:, i_top:i_bottom, i_left:i_right] = img_patch * (1 - alpha) + avg_color.view(c, 1, 1) * alpha

        self._advance()
        return img

    def _advance(self):
        self.cy += self.vy
        self.cx += self.vx

        if self.cy < self.radius or self.cy > self.img_h - self.radius:
            self.vy = -self.vy
            self.cy = max(self.radius, min(self.img_h - self.radius, self.cy))
        if self.cx < self.radius or self.cx > self.img_w - self.radius:
            self.vx = -self.vx
            self.cx = max(self.radius, min(self.img_w - self.radius, self.cx))

        self.vy += random.gauss(0, self.speed * 0.1)
        self.vx += random.gauss(0, self.speed * 0.1)

        current_speed = math.sqrt(self.vy**2 + self.vx**2)
        if current_speed > self.speed * 2:
            scale = (self.speed * 2) / current_speed
            self.vy *= scale
            self.vx *= scale

    def __repr__(self):
        return (
            f"DriftingBlob(radius={self.radius}, speed={self.speed}, "
            f"softness={self.softness}, opacity={self.opacity})"
        )


ROBOT_PRESETS = {
    "aloha": {
        "action_mirror_mask": [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1],
        "state_mirror_mask": [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1],
        "swap_action_ranges": [(slice(0, 7), slice(7, 14))],
        "swap_state_ranges": [(slice(0, 7), slice(7, 14))],
    },
}


class HorizontalFlipWithActionMirror:
    """Horizontally flip camera images and mirror action/state vectors."""

    def __init__(self, action_mirror_mask, state_mirror_mask, swap_action_ranges=None, swap_state_ranges=None):
        self.action_mirror_mask = torch.tensor(action_mirror_mask, dtype=torch.float32)
        self.state_mirror_mask = torch.tensor(state_mirror_mask, dtype=torch.float32)
        self.swap_action_ranges = swap_action_ranges or []
        self.swap_state_ranges = swap_state_ranges or []

    def flip_image(self, img):
        return F.hflip(img)

    def _mirror_vector(self, vec, mask, swap_ranges):
        out = vec.clone() * mask.to(vec.device)
        for range_a, range_b in swap_ranges:
            tmp = out[range_a].clone()
            out[range_a] = out[range_b]
            out[range_b] = tmp
        return out

    def mirror_actions(self, action):
        return self._mirror_vector(action, self.action_mirror_mask, self.swap_action_ranges)

    def mirror_state(self, state):
        return self._mirror_vector(state, self.state_mirror_mask, self.swap_state_ranges)

    def __repr__(self):
        return (
            f"HorizontalFlipWithActionMirror(action_mask={self.action_mirror_mask.tolist()}, "
            f"state_mask={self.state_mirror_mask.tolist()}, "
            f"swap_action_ranges={self.swap_action_ranges}, swap_state_ranges={self.swap_state_ranges})"
        )

class SAM3MaskCapture:
    """Wrapper around SAM3BackgroundCompositor to expose masks."""

    def __init__(self, compositor):
        self.compositor = compositor
        # Only store the last mask per camera — no accumulation to avoid ~780MB memory pressure.
        self._last_masks: dict[str, tuple] = {}

    def reset_episode(self):
        self.compositor.reset_episode()
        self._last_masks = {}

    def seed_background_history(self, frames_by_camera: dict):
        self.compositor.seed_background_history(frames_by_camera)

    def set_camera_key(self, camera_key: str):
        self.compositor.set_camera_key(camera_key)

    def __call__(self, frame):
        composited = self.compositor(frame)
        cam_key = self.compositor._camera_key
        raw_mask = self.compositor.get_last_raw_mask(cam_key)
        feathered_mask = self.compositor.get_last_feathered_mask(cam_key)
        if raw_mask is not None and feathered_mask is not None:
            self._last_masks[cam_key] = (raw_mask, feathered_mask)
        return composited

    def get_episode_masks(self, camera_key: str):
        """Return a one-element list with the last mask, matching the expected [-1] access pattern."""
        last = self._last_masks.get(camera_key)
        return [last] if last is not None else []
    
    def log_episode_stats(self, episode_idx: int, camera_keys: list[str]):
        self.compositor.log_episode_stats(episode_idx, camera_keys)
    
    def __repr__(self):
        return f"SAM3MaskCapture({self.compositor})"
