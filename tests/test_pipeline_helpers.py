"""Tests for tier presets and temporal selection."""

from types import SimpleNamespace

from aloha_augment.pipeline import (
    _parser_defaults,
    apply_tier_configuration,
    build_parser,
    compute_effective_action_shift,
    get_temporal_selector,
)
from aloha_augment.transforms import FrameStride


def test_tier1_configuration_applies_structural_fixes():
    parser = build_parser()
    defaults = _parser_defaults(parser)
    args = SimpleNamespace(
        tier="tier1",
        include_originals=False,
        num_passes=2,
        augmentations=None,
        robot_type=None,
        action_shift=0,
        keep_every_n=1,
        frame_stride_cycle=None,
        skip_prefilter=True,
        min_action_delta=0.02,
        sparc_threshold=-8.0,
        saturation_threshold=0.2,
    )

    apply_tier_configuration(args, defaults)

    assert args.num_passes == 1
    assert args.augmentations == []
    assert args.robot_type == "aloha"
    assert args.action_shift == 4
    assert args.keep_every_n == 4
    assert args.skip_prefilter is True
    assert args.prefilter_mode == "fast"
    assert args.prefilter_sample_every_n == 8
    assert args.max_action_jerk is None
    assert args.tail_drop_max == 0


def test_tier2_and_tier3_skip_full_prefilter():
    parser = build_parser()
    defaults = _parser_defaults(parser)

    tier2 = SimpleNamespace(
        tier="tier2",
        include_originals=False,
        num_passes=2,
        augmentations=None,
        robot_type=None,
        action_shift=0,
        keep_every_n=1,
        frame_stride_cycle=None,
        skip_prefilter=False,
        prefilter_mode="full",
        prefilter_sample_every_n=1,
        min_action_delta=0.02,
        sparc_threshold=-8.0,
        saturation_threshold=0.2,
    )

    apply_tier_configuration(tier2, defaults)
    assert tier2.include_originals is True
    assert tier2.num_passes == 3
    assert tier2.skip_prefilter is False
    assert tier2.prefilter_mode == "fast"
    assert tier2.min_action_delta == 0.0188
    assert tier2.max_action_jerk == 0.0138
    assert tier2.tail_drop_max == 12

    tier3 = SimpleNamespace(
        tier="tier3",
        include_originals=False,
        num_passes=2,
        augmentations=None,
        robot_type=None,
        action_shift=0,
        keep_every_n=1,
        frame_stride_cycle=None,
        skip_prefilter=False,
        prefilter_mode="full",
        prefilter_sample_every_n=1,
        min_action_delta=0.02,
        sparc_threshold=-8.0,
        saturation_threshold=0.2,
    )

    apply_tier_configuration(tier3, defaults)
    assert tier3.skip_prefilter is True
    assert tier3.prefilter_mode == "fast"
    assert tier3.max_action_jerk is None
    assert tier3.tail_drop_max == 12


def test_temporal_selector_cycles_across_passes():
    args = SimpleNamespace(frame_stride_cycle=[2, 3, 4], keep_every_n=1)

    assert get_temporal_selector(args, 0).keep_every_n == 2
    assert get_temporal_selector(args, 1).keep_every_n == 3
    assert get_temporal_selector(args, 2).keep_every_n == 4


def test_effective_action_shift_matches_decimated_space():
    selector = FrameStride(keep_every_n=3, start_offset=0)
    assert compute_effective_action_shift(4, selector) == 1
    assert compute_effective_action_shift(6, selector) == 2
    assert compute_effective_action_shift(0, selector) == 0
    assert compute_effective_action_shift(4, None) == 4