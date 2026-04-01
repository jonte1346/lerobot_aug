Extend the aloha-augment repository with two additional augmentation modules.

Module 1 — SAM3 mask-based background replacement at src/aloha_augment/sam3_aug.py:
- Implement extract_robot_masks(video_path, text_prompt="robotic gripper arm") that uses SAM3's video predictor API (build_sam3_video_predictor from sam3.model_builder, following the pattern in annotate/video_annotate.py) to track the robot arm across all frames, returning a dict of {frame_idx: binary_mask (H,W bool)}
- Implement replace_background(frame, mask, background_frame) that composites the foreground (where mask is True) onto the background using alpha blending with a 3-pixel feathered edge (cv2.dilate + Gaussian blur on mask boundary)
- Implement apply_background_augmentation(video_path, output_path, background_images_dir, mask_cache_path=None) that: checks for a cached mask file first (saves SAM3 inference time on repeated runs), extracts masks if not cached, randomly samples backgrounds from background_images_dir, and writes the composited video. Cache masks as a .npz file with frame indices as keys.
- Add a CLI command to pipeline.py: extract_masks(repo_id, output_mask_dir) that runs mask extraction across all episodes and saves caches.

Module 2 — text relabeling at src/aloha_augment/text_aug.py:
- Implement generate_task_variants(task_description, n_variants=9, model="gemini-2.0-flash") using the google-generativeai SDK. The prompt should ask for natural-language variations that preserve the exact physical meaning (no added constraints, no extra steps). Return a list of strings including the original.
- Implement apply_text_augmentation(episodes_jsonl_path, output_path, n_variants=9) that reads tasks.jsonl, calls generate_task_variants for each unique task, and writes a new episodes.jsonl where each episode's action_config has a randomly sampled variant. This matches what dataset_mod/text/text_gen.py + apply_prompt_to_episodes.py do in the SAM3 repo, but unified into one function.
- The function should be idempotent: if a prompt_augment.jsonl cache already exists, skip the API call.

Module 3 — post-augmentation distribution shift check at src/aloha_augment/quality_check.py:
- Implement compute_frame_embeddings(video_path, model="clip-vit-base-patch32", sample_every=30) using transformers CLIPModel to embed sampled frames, returning a (N, 512) numpy array
- Implement distribution_shift_score(original_embeddings, augmented_embeddings) that computes the mean cosine distance between the two embedding clouds. A score above 0.3 suggests the augmentation has drifted too far from the original appearance distribution.
- Implement joint_space_coverage(states_list) that computes the convex hull volume of end-effector positions (last 3 joints as a proxy for (x,y,z)) using scipy.spatial.ConvexHull. Higher coverage = more diverse demonstrations.
- Add a report command to pipeline.py: report(original_repo_id, augmented_output_dir) that prints a rich table comparing original vs augmented on: n_episodes, mean_SPARC, mean_saturation, distribution_shift_score, joint_coverage.

Update the README with a section explaining each module, the codec handling rationale (AV1 → H.264 via ffprobe detection), and the SPARC threshold choice.