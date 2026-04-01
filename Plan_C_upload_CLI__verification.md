Extend the aloha-augment repository with three final modules.
No training integration is needed — laptop-only execution.

Module 1 — affinity and diversity proxies at src/aloha_augment/aug_metrics.py:

Implement compute_clip_embeddings(video_path, sample_every=30) using
transformers CLIPModel and CLIPProcessor (model="openai/clip-vit-base-patch32")
to embed sampled frames. Return a (N, 512) float32 numpy array. Cache results
to a .npy file alongside the video to avoid recomputation.

Implement affinity_score(original_embeddings, augmented_embeddings) that:
  - For each augmented embedding, finds its nearest original embedding using
    cosine similarity (use numpy dot product after L2 normalization).
  - Returns the mean of those nearest-neighbor cosine similarities.
  - A score >= 0.75 means augmented frames are visually plausible relative to
    the original. Below 0.60 means the augmentation has drifted too far.

Implement diversity_score(original_embeddings, augmented_embeddings) that:
  - Computes the mean pairwise cosine distance within the augmented set.
  - Computes the same for the original set.
  - Returns the ratio (augmented_intra_distance / original_intra_distance).
  - A ratio > 1.0 means augmentation increased visual diversity.
  - A ratio > 2.0 is suspicious — check for unrealistic frames.

Implement joint_coverage_ratio(original_states_list, augmented_states_list)
that uses scipy.spatial.ConvexHull on the last 3 joint angles (used as a
(q4, q5, q6) proxy for end-effector position) from both datasets and returns
augmented_hull_volume / original_hull_volume. Values above 1.05 indicate
meaningful new trajectory coverage.

Implement sparc_comparison(original_episodes, augmented_episodes) that runs
score_sparc() (already implemented in prefilter.py) on both sets and returns
(original_mean_sparc, augmented_mean_sparc). The augmented mean should be
within 15% of the original — if it degrades significantly, the temporal
augmentation has introduced jerkiness.

Module 2 — HuggingFace upload at src/aloha_augment/upload.py:

Implement upload_dataset(output_dir, hf_repo_id, token=None, private=False)
using huggingface_hub.HfApi().upload_folder(). Before uploading, validate that
output_dir contains meta/info.json, meta/episodes.jsonl, data/, and videos/.
After uploading, print the visualizer URL in this exact format:
  https://huggingface.co/spaces/lerobot/visualize_dataset?path=/{hf_repo_id}/episode_0

Module 3 — end-to-end CLI command in src/aloha_augment/pipeline.py:

Add a full_run(repo_id, output_dir, hf_repo_id=None, n_aug_copies=2, seed=42,
sparc_threshold=-10.0, saturation_threshold=0.15, skip_sam3=False,
skip_text=False, token=None) command that runs these stages in order:

  Stage 1 — Pre-filter: run SPARC and actuator saturation scoring on all
  episodes. Print a rich table showing each episode's scores. Remove episodes
  below thresholds. Print how many episodes were kept vs rejected.

  Stage 2 — Visual augmentation: apply color jitter + occlusion to each kept
  episode n_aug_copies times with seeds (seed, seed+1, ...). Handle AV1 codec
  via ffprobe detection and re-encode to H.264 before processing. Write all
  augmented videos and Parquet files to output_dir.

  Stage 3 — SAM3 mask augmentation (skipped if skip_sam3=True): extract robot
  masks for kept episodes, cache them, then produce one additional background-
  replaced copy per episode using a backgrounds/ subdirectory that the user
  populates with any JPEG images.

  Stage 4 — Text relabeling (skipped if skip_text=True): generate task
  description variants using Gemini and inject them into episodes.jsonl.

  Stage 5 — Metadata sync: reindex all Parquet files and update all meta/
  JSONs so the dataset is self-consistent. Verify consistency by checking that
  the total frame count in info.json matches the sum of lengths in episodes.jsonl.

  Stage 6 — Quality report: run aug_metrics.py against a sample of 10
  original and 10 augmented episodes. Print a rich table with these columns:
    metric | original | augmented | delta | status (PASS/WARN/FAIL)
  Rows: affinity (PASS >= 0.75), diversity_ratio (PASS > 1.0),
  joint_coverage_ratio (PASS > 1.05), mean_sparc delta (PASS within 15%),
  n_episodes, n_frames.
  Write the full report to output_dir/augmentation_report.json.

  Stage 7 — Upload (only if hf_repo_id is provided): call upload_dataset()
  and print the visualizer URL.

The command should be callable as:
  aloha-augment full_run lerobot/aloha_static_cups_open ./output \
    --hf_repo_id myuser/aloha_cups_aug --n_aug_copies 3 --seed 42

Add a pyproject.toml [project.scripts] entry:
  aloha-augment = "aloha_augment.pipeline:main"
where main() calls fire.Fire({'full_run': full_run, 'report': report,
'extract_masks': extract_masks, 'upload': upload_dataset}).

Update README.md with:
  - A "How to verify the augmented dataset is better" section explaining what
    each metric in the quality report means, with the PASS thresholds and
    why each threshold was chosen (affinity 0.75 from cosine similarity
    distribution of in-domain robot frames; diversity >1.0 means expansion;
    SPARC within 15% ensures temporal augmentation didn't introduce jerk).
  - A quick-start section showing the single full_run command.
  - A note on the AV1 codec detection rationale.
  - A section on which stages can be skipped with --skip_sam3 and --skip_text
    flags for faster iteration.