"""Episode pre-filtering based on motion smoothness (SPARC) and actuator saturation."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import json


def compute_sparc(state_sequence: np.ndarray, fps: int = 50, normalize: bool = True) -> float:
    """
    Compute Spectral Arc Length (SPARC) smoothness metric.
    
    Lower (more negative) SPARC indicates smoother motion.
    Higher SPARC (closer to 0) indicates jerkier motion.
    
    Args:
        state_sequence: Joint positions over time, shape (T, n_joints).
        fps: Frame rate for velocity computation.
        normalize: Whether to normalize spectrum peak to 1.
    
    Returns:
        SPARC score (typically -3 to -17 for robot demonstrations).
    """
    if len(state_sequence) < 2:
        return 0.0
    
    # Compute per-joint velocity via finite differences
    velocity = np.gradient(state_sequence, axis=0) * fps
    
    # Compute FFT magnitude spectrum for each joint
    fft_magnitudes = []
    for j in range(velocity.shape[1]):
        fft_result = np.fft.rfft(velocity[:, j])
        magnitude = np.abs(fft_result)
        if normalize:
            magnitude = magnitude / (magnitude.max() + 1e-8)
        fft_magnitudes.append(magnitude)
    
    # Compute arc length in frequency space
    arc_lengths = []
    for magnitude in fft_magnitudes:
        if len(magnitude) < 2:
            arc_lengths.append(0.0)
            continue
        # Compute cumulative distance
        df = 1.0 / (2 * len(magnitude))
        distances = np.sqrt((df ** 2) + (np.diff(magnitude) ** 2))
        arc_length = -np.sum(distances)
        arc_lengths.append(arc_length)
    
    # Return mean SPARC across joints
    return float(np.mean(arc_lengths))


def compute_actuator_saturation(
    action_sequence: np.ndarray,
    state_sequence: np.ndarray,
    saturation_threshold: float = 7.0,
) -> float:
    """
    Compute fraction of timesteps where actuator saturation detected.
    
    Saturation = |action[t] - state[t+1]| > threshold (per joint).
    
    Args:
        action_sequence: Joint commands, shape (T, n_joints).
        state_sequence: Joint positions, shape (T, n_joints).
        saturation_threshold: Degree threshold (default 7 degrees).
    
    Returns:
        Fraction of timesteps with saturation on any joint.
    """
    if len(action_sequence) < 2 or len(state_sequence) < 2:
        return 0.0
    
    # Compare action[t] relative to state[t+1]
    # This detects when the controller commanded action but state didn't follow
    action = action_sequence[:-1]
    next_state = state_sequence[1:]
    
    error = np.abs(action - next_state)
    saturation_per_step = (error > saturation_threshold).any(axis=1)
    
    return float(np.mean(saturation_per_step))


def filter_episodes(
    dataset,
    episode_indices: list[int] | None = None,
    sparc_threshold: float = -10.0,
    saturation_threshold_frac: float = 0.15,
    fps: int | None = None,
) -> tuple[list[int], dict]:
    """
    Filter episodes based on smoothness and saturation metrics.
    
    Args:
        dataset: LeRobot dataset instance.
        episode_indices: Specific episodes to filter. If None, filter all.
        sparc_threshold: Reject if SPARC > this value (less smooth).
        saturation_threshold_frac: Reject if saturation fraction > this.
        fps: Frame rate. If None, use dataset.fps.
    
    Returns:
        (kept_episode_indices, scores_dict)
    """
    if fps is None:
        fps = dataset.fps
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))
    
    scores = {}
    kept_indices = []
    
    for ep_idx in episode_indices:
        try:
            # Get episode range
            episodes = dataset.meta.episodes
            if hasattr(episodes, 'iloc'):
                ep_record = episodes.iloc[ep_idx]
            elif hasattr(episodes, '__getitem__'):
                ep_record = episodes[ep_idx]
            else:
                raise ValueError("Cannot index episodes")
            
            # Handle different record types
            if hasattr(ep_record, 'to_dict'):
                ep_record = ep_record.to_dict()
            elif isinstance(ep_record, dict):
                pass
            else:
                ep_record = dict(ep_record)
            
            from_idx = ep_record["dataset_from_index"]
            to_idx = ep_record["dataset_to_index"]
            
            # Load action and state sequences
            action_seq = []
            state_seq = []
            for global_idx in range(from_idx, to_idx):
                item = dataset[global_idx]
                if "action" in item:
                    action_seq.append(item["action"].numpy() if hasattr(item["action"], "numpy") else item["action"])
                if "observation.state" in item:
                    state_seq.append(item["observation.state"].numpy() if hasattr(item["observation.state"], "numpy") else item["observation.state"])
            
            if not action_seq or not state_seq:
                scores[ep_idx] = {"sparc": None, "saturation": None, "kept": False, "reason": "missing action/state"}
                continue
            
            action_seq = np.array(action_seq)
            state_seq = np.array(state_seq)
            
            # Compute metrics
            sparc_score = compute_sparc(state_seq, fps=fps)
            sat_frac = compute_actuator_saturation(action_seq, state_seq)
            
            kept = (sparc_score <= sparc_threshold) and (sat_frac <= saturation_threshold_frac)
            reason = None
            if sparc_score > sparc_threshold:
                reason = f"jerky_motion (sparc={sparc_score:.2f})"
            if sat_frac > saturation_threshold_frac:
                reason = f"saturation (frac={sat_frac:.2f})"
            
            scores[ep_idx] = {
                "sparc": float(sparc_score),
                "saturation": float(sat_frac),
                "kept": kept,
                "reason": reason,
            }
            
            if kept:
                kept_indices.append(ep_idx)
        
        except Exception as e:
            scores[ep_idx] = {"sparc": None, "saturation": None, "kept": False, "reason": str(e)}
    
    return kept_indices, scores


def save_filter_scores(scores: dict, output_path: str | Path) -> None:
    """Save filter scores as JSON for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
