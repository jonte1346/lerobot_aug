"""Pre-filter module: SPARC smoothness scoring and actuator saturation detection."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def score_sparc(state_list: list[dict[str, Any]], actions: np.ndarray) -> float:
    """Compute the SPARC (spectral arc length) score for an episode.

    A higher (less negative) score means smoother motion.
    Episodes with mean SPARC below -10 are considered fragmented.

    Args:
        state_list: List of dicts with at least a 'q' key containing joint positions
                    as a 1-D array-like.
        actions: numpy array of shape (T, n_joints) — not used directly but kept for
                 interface consistency.

    Returns:
        Mean SPARC score across all joints (scalar, ≤ 0).
    """
    if len(state_list) < 4:
        return -float("inf")

    positions = np.array([s["q"] for s in state_list], dtype=float)  # (T, n_joints)
    T, n_joints = positions.shape

    sparc_per_joint = []
    for j in range(n_joints):
        vel = np.gradient(positions[:, j])  # velocity via finite differences
        spectrum = np.fft.rfft(vel)
        magnitude = np.abs(spectrum)

        # Normalize so max magnitude == 1
        max_mag = magnitude.max()
        if max_mag == 0.0:
            sparc_per_joint.append(0.0)
            continue
        magnitude = magnitude / max_mag

        n_freqs = len(magnitude)
        fc = n_freqs - 1  # normalizing frequency count

        # Arc length: -sum( sqrt( (df/fc)^2 + da^2 ) )
        df = 1.0 / fc if fc > 0 else 1.0
        arc = 0.0
        for i in range(1, n_freqs):
            da = magnitude[i] - magnitude[i - 1]
            arc += math.sqrt(df**2 + da**2)
        sparc_per_joint.append(-arc)

    return float(np.mean(sparc_per_joint))


def score_actuator_saturation(
    state_list: list[dict[str, Any]], actions: np.ndarray, threshold_deg: float = 7.0
) -> float:
    """Compute the actuator saturation ratio for an episode.

    Saturation is defined as |action[t] - state[t+1]| > threshold_deg for any joint.

    Args:
        state_list: List of T dicts with 'q' key (joint positions in degrees).
        actions: numpy array of shape (T, n_joints).
        threshold_deg: Per-joint error threshold in degrees.

    Returns:
        Fraction of timesteps (in [0, 1]) where any joint exceeds the threshold.
    """
    T = len(state_list) - 1  # we compare action[t] vs state[t+1]
    if T <= 0:
        return 0.0

    next_states = np.array([state_list[t + 1]["q"] for t in range(T)], dtype=float)
    act = actions[:T].astype(float)

    error = np.abs(act - next_states)  # (T, n_joints)
    saturated = np.any(error > threshold_deg, axis=1)  # (T,)
    return float(saturated.mean())


def _episode_arrays(dataset, episode_idx: int):
    """Extract state_list and actions array for one episode from a LeRobotDataset."""
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    # Clamp in case dataset.__len__ is smaller than the index range
    to_idx = min(to_idx, len(dataset))

    state_list = []
    action_rows = []
    for i in range(from_idx, to_idx):
        sample = dataset[i]
        q = np.array(sample["observation.state"])
        state_list.append({"timestamp": i - from_idx, "q": q})
        action_rows.append(np.array(sample["action"]))

    actions = np.stack(action_rows) if action_rows else np.empty((0,))
    return state_list, actions


def filter_dataset(
    dataset,
    sparc_threshold: float = -10.0,
    sat_threshold: float = 0.15,
    csv_out_path: str | Path | None = None,
) -> tuple[list[int], list[dict]]:
    """Score all episodes and return the indices that pass both filters.

    Args:
        dataset: LeRobotDataset instance.
        sparc_threshold: Episodes with SPARC < threshold are rejected.
        sat_threshold: Episodes with saturation ratio > threshold are rejected.
        csv_out_path: Optional path to write a CSV report.

    Returns:
        Tuple of (good_indices, scores) where scores is a list of dicts with keys
        episode_idx, sparc, saturation, kept.
    """
    n_episodes = dataset.num_episodes
    scores = []
    good_indices = []

    for ep_idx in range(n_episodes):
        state_list, actions = _episode_arrays(dataset, ep_idx)
        sparc = score_sparc(state_list, actions)
        sat = score_actuator_saturation(state_list, actions)
        kept = sparc >= sparc_threshold and sat <= sat_threshold
        scores.append(
            {
                "episode_idx": ep_idx,
                "sparc": sparc,
                "saturation": sat,
                "kept": kept,
            }
        )
        if kept:
            good_indices.append(ep_idx)

    if csv_out_path is not None:
        csv_out_path = Path(csv_out_path)
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_idx", "sparc", "saturation", "kept"])
            writer.writeheader()
            writer.writerows(scores)

    return good_indices, scores
