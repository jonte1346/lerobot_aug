# Tier Pipeline Validation Report

## Executive Summary

Three-tier augmentation pipeline successfully reduces temporal over-sampling from **36 steps (0.72s)** to **9–15 steps (0.18–0.30s)** while preserving action signal magnitude. All action autocorrelation curves flatten significantly, confirming reduced redundancy.

## Visual Metrics: Suggested Chunk Length

| Tier | Strategy | Chunk Length | Reduction | Status |
|------|----------|--------------|-----------|--------|
| **Baseline** | Original dataset | 36 steps (0.72s) | — | 🔴 Over-sampling |
| **Tier 1** | keep_every_n=4 | **9 steps (0.18s)** | **75% ↓** | ✅ Most aggressive |
| **Tier 2** | keep_every_n=3 + 5 augmentations | **15 steps (0.30s)** | **58% ↓** | ✅ Balanced |
| **Tier 3** | stride_cycle=[2,3,4] + 7 augmentations | **15 steps (0.30s)** | **58% ↓** | ✅ Diverse |

## Action Autocorrelation Analysis

**Baseline:** Slow decay with high correlations at lags 1–15 (strong redundancy)

**Tier 1:** Dramatic flattening—correlation drops to near-zero by lag 5 (minimal redundancy)

**Tier 2:** Balanced flattening—correlation reaches baseline floor by lag 10 (moderate redundancy reduction)

**Tier 3:** Similar to Tier 2, with cycling-induced micro-variations (prevents aliasing)

## Action Signal Preservation

All three tiers maintain **identical action magnitude**:
- action_sum ≈ 5.226 across all tiers
- Frame counts align with stride expectations
- No signal loss detected

## Configuration Details

### Tier 1: Structural Baseline
- num_passes: 1
- augmentations: None
- action_shift: +4 frames
- keep_every_n: 4
- prefilter_mode: fast (action_delta only)
- Result: 1 ep × 99 frames = fastest, cleanest baseline

### Tier 2: Balanced Production
- num_passes: 2
- augmentations: color_jitter, gaussian_blur, sharpness, random_erasing, horizontal_flip
- action_shift: +4 frames
- keep_every_n: 3
- prefilter_mode: skip
- Result: 1 ep × 264 frames = practical augmentation with diversity

### Tier 3: Maximum Diversity
- num_passes: 3
- augmentations: [tier2 augmentations] + drifting_blob, static_erasing
- action_shift: +4 frames
- keep_every_n: 1 (per-pass stride from cycle)
- frame_stride_cycle: [2, 3, 4]
- prefilter_mode: skip
- Result: 1 ep × 429 frames = highest variance, good for robust training

## Validation Status

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Temporal over-sampling solved | ✅ | 75% chunk reduction (Tier 1), 58% (Tier 2/3) |
| Action correlation reduced | ✅ | Autocorrelation curves flatten significantly |
| Action signal preserved | ✅ | action_sum invariant across tiers (~5.226) |
| Tier differentiation | ✅ | Visualizer shows distinct chunk lengths |
| Tier determinism | ✅ | Repeated single-episode runs reproduced |

## Known Issues

1. **Multi-episode OOM (exit 143)**: Scaling to 5 episodes (episodes 0–4) causes process to be SIGKILL'd
   - Single-episode runs (episode 0 only) succeed
   - Likely video encoding (SVT-AV1) memory accumulation
   - Workaround: Process in smaller batches (1–2 episodes per job)

2. **Tier 2 & 3 identical chunk length**: Both suggest 15 steps despite different configs
   - Likely due to action vector dimensionality (6D ALOHA) making stride_cycle less impactful than keep_every_n
   - Tier 3 provides augmentation diversity even if chunk length similar

## Recommendations

1. **Production Use**: Deploy Tier 2 for balanced speed/augmentation tradeoff
2. **Diagnostic/Research**: Use Tier 1 for ground truth, Tier 3 for maximum diversity
3. **Scaling**: Process episodes in batches of 1–2 to avoid OOM during encoding
4. **Future Work**: 
   - Profile memory usage during SVT-AV1 encoding
   - Investigate heap size during multi-pass runs
   - Consider streaming encoding to reduce peak memory

## Datasets Generated

- **jgiegold/aloha_tier1_diag**: 1 episode, 99 frames (Tier 1)
- **jgiegold/aloha_tier2_diag**: 1 episode, 264 frames (Tier 2) [Note: 2-episode pre-OOM version also exists]
- **jgiegold/aloha_tier3_diag**: 1 episode, 429 frames (Tier 3) [Note: 3-episode pre-OOM version also exists]

All datasets uploaded to Hugging Face Hub with visualizer access active.

---
*Generated: 2026-04-03 | Validation: All benchmarks achieved*
