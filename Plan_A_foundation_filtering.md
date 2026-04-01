Create a new Python repository called `aloha-augment` for augmenting the LeRobot dataset `lerobot/aloha_static_cups_open`.

Step 1 — repo scaffold:
- pyproject.toml with dependencies: lerobot, opencv-python, numpy, scipy, rich, fire, torch, huggingface_hub
- src/aloha_augment/ package with __init__.py
- A README.md explaining the pipeline

Step 2 — pre-filter module at src/aloha_augment/prefilter.py:
- Load the dataset using LeRobotDataset
- Implement score_sparc() using the spectral arc length formula: compute velocity via np.gradient on joint positions, take the rfft, normalize the magnitude spectrum, compute the arc length in (freq, amplitude) space as −sum(sqrt((df/fc)² + da²)), and average across joints. Episodes with mean SPARC below −10 are fragmented and should be rejected.
- Implement score_actuator_saturation() that computes |action[t] - state[t+1]| and returns the fraction of timesteps where any joint exceeds 7 degrees. Episodes with saturation ratio > 0.15 should be rejected.
- Both functions should accept a dict of {timestamp, q} state entries and an actions numpy array, matching the interface in the existing score_lerobot_episodes codebase.
- Output a CSV report of per-episode SPARC and saturation scores.

Step 3 — visual augmentation at src/aloha_augment/visual_aug.py:
- Implement apply_color_jitter(frame, brightness_range=(0.8,1.2), contrast_range=(0.8,1.2), hue_shift_deg=(-15,15), saturation_range=(0.8,1.2)) using OpenCV HSV conversion
- Implement apply_occlusion_patch(frame, size_ratio_range=(0.05,0.2)) that places a black rectangle at a random position
- Implement apply_video_augmentation(video_path, output_path, config_dict) that reads the MP4 with ffmpeg subprocess (to handle AV1 codec), applies the augmentation frame by frame, and writes H.264 output. Log all random parameters to config_dict for reproducibility.
- Handle the AV1 codec issue by detecting codec with ffprobe and re-encoding to H.264 before processing if needed.

Step 4 — metadata sync at src/aloha_augment/metadata_sync.py:
- Implement rewrite_episode_parquet(old_path, new_path, new_episode_idx, start_global_index) that reads the Parquet, replaces episode_index, frame_index, and index columns, and writes back with zstd compression
- Implement update_meta_jsonl(episodes_jsonl_path, good_episodes, task_text, aug_params_per_episode) that filters to only kept episodes, reindexes, and injects aug_params into each episode dict
- Implement update_info_json(info_path, new_total_episodes, new_total_frames, new_total_videos) that updates the totals and splits field

Step 5 — main CLI at src/aloha_augment/pipeline.py using fire:
- Command: run(repo_id, output_dir, sparc_threshold=-10.0, saturation_threshold=0.15, n_aug_copies=2, seed=42)
- It should: load dataset, pre-filter, apply visual augmentation to each kept episode n_aug_copies times with different seeds, sync all metadata, and print a summary table using rich

Write tests in tests/ for the SPARC implementation (use a synthetic smooth sinusoid vs a step function and verify SPARC scores differ by at least 5 units) and for the metadata sync (verify episode_index is contiguous after filtering).