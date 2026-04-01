"""Tests for prefilter SPARC scoring and actuator saturation."""

import numpy as np
import pytest

from aloha_augment.prefilter import score_actuator_saturation, score_sparc


def _make_state_list(positions: np.ndarray) -> list[dict]:
    """Create state_list from a (T, n_joints) positions array."""
    return [{"timestamp": t, "q": positions[t]} for t in range(len(positions))]


def test_sparc_smooth_vs_jerky():
    """Smooth (sinusoidal) motion should have a higher SPARC than random noise."""
    T = 200
    t = np.linspace(0, 2 * np.pi, T)
    n_joints = 6

    # Smooth: sinusoidal motion for each joint
    smooth_pos = np.column_stack([np.sin(t + j * 0.5) for j in range(n_joints)])
    # Jerky: independent random noise per joint — maximally fragmented motion
    rng = np.random.default_rng(42)
    jerky_pos = rng.normal(size=(T, n_joints))

    dummy_actions = np.zeros((T, n_joints))

    smooth_score = score_sparc(_make_state_list(smooth_pos), dummy_actions)
    jerky_score = score_sparc(_make_state_list(jerky_pos), dummy_actions)

    assert smooth_score > jerky_score, (
        f"Smooth SPARC ({smooth_score:.2f}) should be > jerky SPARC ({jerky_score:.2f})"
    )
    assert (smooth_score - jerky_score) >= 5.0, (
        f"Expected ≥5 unit gap, got {smooth_score - jerky_score:.2f}"
    )


def test_sparc_smooth_above_threshold():
    """Smooth sinusoidal motion should score above −10."""
    T = 200
    t = np.linspace(0, 2 * np.pi, T)
    pos = np.column_stack([np.sin(t)] * 4)
    state_list = _make_state_list(pos)
    score = score_sparc(state_list, np.zeros((T, 4)))
    assert score > -10.0, f"Expected smooth SPARC > -10, got {score:.2f}"


def test_sparc_jerky_below_threshold():
    """Random-noise motion should score well below −10 (filter threshold)."""
    T = 200
    rng = np.random.default_rng(7)
    pos = rng.normal(size=(T, 4))
    state_list = _make_state_list(pos)
    score = score_sparc(state_list, np.zeros((T, 4)))
    assert score < -15.0, f"Expected jerky SPARC < -15, got {score:.2f}"


def test_saturation_zero_error():
    """When actions match next states exactly, saturation should be 0."""
    T = 20
    n_joints = 6
    positions = np.random.default_rng(0).uniform(-1, 1, (T, n_joints))
    state_list = _make_state_list(positions)
    # actions[t] == state[t+1] → zero error
    actions = positions[1:]  # (T-1, n_joints)
    actions = np.vstack([actions, actions[-1]])  # pad to length T

    sat = score_actuator_saturation(state_list, actions)
    assert sat == pytest.approx(0.0), f"Expected 0.0 saturation, got {sat}"


def test_saturation_all_exceed():
    """When all actions are far from next states, saturation should be 1."""
    T = 20
    n_joints = 6
    positions = np.zeros((T, n_joints))
    state_list = _make_state_list(positions)
    # actions are 100° away from zero
    actions = np.full((T, n_joints), 100.0)

    sat = score_actuator_saturation(state_list, actions)
    assert sat == pytest.approx(1.0), f"Expected 1.0 saturation, got {sat}"
