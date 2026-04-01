"""Post-augmentation quality metrics: CLIP affinity/diversity, joint coverage, SPARC comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CLIP embeddings
# ---------------------------------------------------------------------------

def compute_clip_embeddings(
    video_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
    sample_every: int = 30,
) -> np.ndarray:
    """Embed sampled frames from a video using CLIP.

    Results are cached to a .npy file alongside the video to avoid recomputation.

    Args:
        video_path: Path to the video file.
        model_name: HuggingFace model ID for CLIPModel.
        sample_every: Sample one frame every this many frames.

    Returns:
        Float32 numpy array of shape (N, 512).
    """
    import cv2
    from PIL import Image

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        raise ImportError("transformers is required for CLIP embeddings: pip install transformers") from e

    video_path = Path(video_path)
    cache_path = video_path.with_suffix(".clip.npy")
    if cache_path.exists():
        return np.load(cache_path)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()

    if not frames:
        return np.empty((0, 512), dtype=np.float32)

    import torch
    embeddings = []
    with torch.no_grad():
        for img in frames:
            inputs = processor(images=img, return_tensors="pt")
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.squeeze(0).numpy().astype(np.float32))

    result = np.stack(embeddings)
    np.save(cache_path, result)
    return result


# ---------------------------------------------------------------------------
# Affinity score
# ---------------------------------------------------------------------------

def affinity_score(
    original_embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
) -> float:
    """Mean nearest-neighbor cosine similarity: augmented → original.

    For each augmented embedding, find the nearest original embedding by cosine
    similarity (embeddings assumed L2-normalised). Returns the mean of those
    nearest-neighbour similarities.

    PASS threshold: >= 0.75
    FAIL threshold: < 0.60
    """
    if original_embeddings.shape[0] == 0 or augmented_embeddings.shape[0] == 0:
        return float("nan")

    # L2-normalise
    orig = original_embeddings / (np.linalg.norm(original_embeddings, axis=1, keepdims=True) + 1e-8)
    aug = augmented_embeddings / (np.linalg.norm(augmented_embeddings, axis=1, keepdims=True) + 1e-8)

    # (n_aug, n_orig) similarity matrix
    sim = aug @ orig.T
    nearest = sim.max(axis=1)
    return float(nearest.mean())


# ---------------------------------------------------------------------------
# Diversity score
# ---------------------------------------------------------------------------

def diversity_score(
    original_embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
) -> float:
    """Ratio of mean intra-set pairwise cosine distance: augmented / original.

    A ratio > 1.0 means augmentation increased visual diversity.
    A ratio > 2.0 is suspicious — check for unrealistic frames.
    """
    def _mean_pairwise_distance(emb: np.ndarray) -> float:
        if emb.shape[0] < 2:
            return 0.0
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        # cosine distance = 1 - cosine similarity
        sim = norm @ norm.T
        n = sim.shape[0]
        # exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        return float((1.0 - sim[mask]).mean())

    orig_dist = _mean_pairwise_distance(original_embeddings)
    aug_dist = _mean_pairwise_distance(augmented_embeddings)
    if orig_dist == 0.0:
        return float("nan")
    return aug_dist / orig_dist


# ---------------------------------------------------------------------------
# Joint space coverage
# ---------------------------------------------------------------------------

def joint_coverage_ratio(
    original_states_list: list[list[dict[str, Any]]],
    augmented_states_list: list[list[dict[str, Any]]],
) -> float:
    """ConvexHull volume ratio (augmented / original) on last-3-joint angles.

    Values above 1.05 indicate meaningful new trajectory coverage.
    """
    from scipy.spatial import ConvexHull, QhullError

    def _hull_volume(states_episodes: list[list[dict]]) -> float:
        pts = []
        for ep in states_episodes:
            for s in ep:
                q = np.asarray(s["q"])
                if len(q) >= 3:
                    pts.append(q[-3:])
        if len(pts) < 5:
            return 0.0
        pts_arr = np.array(pts, dtype=float)
        try:
            return ConvexHull(pts_arr).volume
        except QhullError:
            return 0.0

    orig_vol = _hull_volume(original_states_list)
    aug_vol = _hull_volume(augmented_states_list)
    if orig_vol == 0.0:
        return float("nan")
    return aug_vol / orig_vol


# ---------------------------------------------------------------------------
# SPARC comparison
# ---------------------------------------------------------------------------

def sparc_comparison(
    original_episodes: list[tuple[list[dict], np.ndarray]],
    augmented_episodes: list[tuple[list[dict], np.ndarray]],
) -> tuple[float, float]:
    """Return (original_mean_sparc, augmented_mean_sparc).

    The augmented mean should be within 15% of the original.
    Each element of the input lists is a (state_list, actions) tuple.
    """
    from aloha_augment.prefilter import score_sparc

    def _mean(episodes: list[tuple]) -> float:
        scores = [score_sparc(sl, act) for sl, act in episodes]
        return float(np.mean(scores)) if scores else float("nan")

    return _mean(original_episodes), _mean(augmented_episodes)


# ---------------------------------------------------------------------------
# Full report dict
# ---------------------------------------------------------------------------

def build_quality_report(
    original_video_paths: list[Path],
    augmented_video_paths: list[Path],
    original_states: list[list[dict]] | None = None,
    augmented_states: list[list[dict]] | None = None,
    original_episodes_for_sparc: list[tuple] | None = None,
    augmented_episodes_for_sparc: list[tuple] | None = None,
    sample_every: int = 30,
) -> dict[str, Any]:
    """Compute all quality metrics and return a structured report dict."""
    report: dict[str, Any] = {}

    # CLIP metrics (optional — skip if transformers not available)
    try:
        orig_emb_list = [compute_clip_embeddings(p, sample_every=sample_every) for p in original_video_paths]
        aug_emb_list = [compute_clip_embeddings(p, sample_every=sample_every) for p in augmented_video_paths]

        orig_emb = np.concatenate(orig_emb_list) if orig_emb_list else np.empty((0, 512))
        aug_emb = np.concatenate(aug_emb_list) if aug_emb_list else np.empty((0, 512))

        report["affinity"] = affinity_score(orig_emb, aug_emb)
        report["diversity_ratio"] = diversity_score(orig_emb, aug_emb)
    except (ImportError, Exception) as exc:
        report["affinity"] = None
        report["diversity_ratio"] = None
        report["clip_error"] = str(exc)

    # Joint coverage
    if original_states is not None and augmented_states is not None:
        report["joint_coverage_ratio"] = joint_coverage_ratio(original_states, augmented_states)
    else:
        report["joint_coverage_ratio"] = None

    # SPARC comparison
    if original_episodes_for_sparc is not None and augmented_episodes_for_sparc is not None:
        orig_sparc, aug_sparc = sparc_comparison(original_episodes_for_sparc, augmented_episodes_for_sparc)
        report["original_mean_sparc"] = orig_sparc
        report["augmented_mean_sparc"] = aug_sparc
        if orig_sparc != 0 and not np.isnan(orig_sparc):
            report["sparc_delta_pct"] = abs(aug_sparc - orig_sparc) / abs(orig_sparc) * 100
        else:
            report["sparc_delta_pct"] = None
    else:
        report["original_mean_sparc"] = None
        report["augmented_mean_sparc"] = None
        report["sparc_delta_pct"] = None

    return report
