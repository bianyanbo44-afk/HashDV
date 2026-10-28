# Contribution-led strengthening, 2026-10-27

This is a new supplementary analysis plan, written after the original results and before the new results. Original L1/L1b/L1d protocols and outputs remain immutable. It is not a retrospective preregistration.

The manuscript's central contribution is an actionable industrial benchmark audit that distinguishes repeated physical instances from repetitive textures and separates copy-induced accuracy from image difficulty through retraining.

## New analyses

1. Evaluate the official SSCD DISC-mixup ResNet50 checkpoint on exactly the existing L1b query-candidate pairs. Use both officially documented preprocessing choices (short side 288, and square 320), cosine descriptors, and gallery-mean centering followed by L2 normalization. Report every variant; do not select using test scores. The default comparison is uncentered square 320; the other variants are sensitivity checks. The weight source is the authors' official repository/CDN. Save the SHA256 and per-pair scores.
2. Evaluate conventional SIFT matching followed by partial-affine RANSAC on the same pairs, with 448-pixel square input, 1,024 keypoints, Lowe ratio 0.75, mutual accepted matches, and a 1.5/32-image-width RANSAC tolerance. Score by inlier count divided by 1,024. This is a fixed conventional geometric-verification baseline, not a fitted competitor.
3. Recompute retrieval-aware TPR using the stored top-20 candidate lists. The added oracle source is excluded whenever it was outside top 20. Use the same pair-level ROC operating point and explicitly distinguish verification, retrieval recall, and retrieval-aware detection. Report severity strata and real-query false-match maxima descriptively; do not interpret pair FPR as query FPR.
4. Use the saved detector predictions for paired-seed and acquisition-cluster sensitivity analyses where acquisition metadata identify defensible units. Do not present a new interval as the originally registered primary test.
5. Add a class-matched random-deletion training control on NEU-DET, using three seeds and the unchanged registered YOLO11n recipe (150 epochs, patience 30, batch 8, 640 input, workers 1, RAM cache, best validation checkpoint). Remove exactly 17 non-copy training images per seed, matching the class distribution of the 17 removed copies. The removal draw uses a separate fixed seed and excludes known copies R. Compare the change in the leaked-vs-other accuracy gap against the targeted-removal change. The main endpoint is the three-seed mean of (random-deletion L-U gap) minus (targeted-deletion L-U gap), in mAP50 points. All seeds and mAP50-95 are reported regardless of sign. This control tests whether generic class-balanced loss of 17 training images explains the measured effect. The original full/leak-free results remain unchanged.

## Writing and delivery

Lead with new audit capability and the distinction between memorisation and intrinsic easiness. Treat GC10-DET as a diagnostic discrimination result, not a failed replication. Use method and intervention evidence to support affirmative claims. Place calibration details, checks, and all extra comparison variants in the supplementary methods. Do not manufacture labels, authorship facts, numbers, general detector rankings, or monotonic lower bounds for mAP effects.

Deliver updated LaTeX/PDF, synchronized cover letter/highlights/supplement, reproducible scripts and result records, and a short Chinese change log. Author-owned metadata remain unfilled until supplied.
