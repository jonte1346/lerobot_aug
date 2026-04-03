"""Tests for pre-filtering helpers."""

import numpy as np

from aloha_augment.prefilter import compute_mean_action_delta


def test_mean_action_delta_distinguishes_static_and_active_sequences():
    static_actions = np.zeros((6, 4), dtype=np.float32)
    active_actions = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.05, 0.02, 0.01, 0.00],
            [0.10, 0.05, 0.03, 0.01],
            [0.16, 0.09, 0.05, 0.02],
        ],
        dtype=np.float32,
    )

    assert compute_mean_action_delta(static_actions) == 0.0
    assert compute_mean_action_delta(active_actions) > 0.02