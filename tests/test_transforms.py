"""Tests for the core transform helpers."""

import pytest

torch = pytest.importorskip("torch")

from aloha_augment.transforms import FrameDecimator, HorizontalFlipWithActionMirror, ROBOT_PRESETS


def test_frame_decimator_pattern():
    decimator = FrameDecimator(remove_every_n=5)
    keep = [decimator.should_keep(i) for i in range(10)]
    assert keep == [True, True, True, True, False, True, True, True, True, False]


def test_horizontal_flip_aloha_mirrors_actions_and_state():
    preset = ROBOT_PRESETS["aloha"]
    flip = HorizontalFlipWithActionMirror(
        action_mirror_mask=preset["action_mirror_mask"],
        state_mirror_mask=preset["state_mirror_mask"],
        swap_action_ranges=preset["swap_action_ranges"],
        swap_state_ranges=preset["swap_state_ranges"],
    )

    action = torch.arange(14, dtype=torch.float32)
    state = torch.arange(14, dtype=torch.float32) + 100.0

    mirrored_action = flip.mirror_actions(action)
    mirrored_state = flip.mirror_state(state)

    assert mirrored_action.shape == action.shape
    assert mirrored_state.shape == state.shape
    assert mirrored_action[1] == -8.0
    assert mirrored_action[8] == -1.0
    assert mirrored_state[1] == -108.0
    assert mirrored_state[8] == -101.0
