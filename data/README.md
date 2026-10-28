# Supplementary data

Supplementary data for the manuscript "HashDV: Exposing hidden copies and measuring accuracy inflation in industrial
defect benchmarks". The package holds file names, saved predictions, scores and parameters, and no
images. NEU-DET and GC10-DET images remain with their original distributors. DsPCBSD+ is distributed under CC BY 4.0
(figshare, https://doi.org/10.6084/m9.figshare.24970329), and its use requires attribution to Lv et al. (2024).

## leak_lists/

- `neu_det_copy_pairs.csv`, `gc10_det_copy_pairs.csv`: every flagged test–training pair of the audited
  class-stratified random splits. Columns: test image, training image, fused score F, hash cue H and dense
  verification cue D.
- `dspcbsd_official_copy_pairs.csv`: the same for the official DsPCBSD+ split (validation image vs training image).
- `dspcbsd_official_closure_R1.csv`: training images added by one step of closure (copies of matched training
  images).
- `audit_parameters.json`: the fusion weight α and the audit threshold τ of each benchmark.

## splits/

- `neu_det_split.csv`, `gc10_det_split.csv`: the class-stratified random splits used in the paper (train / validation /
  test).
- `dspcbsd_official_split.csv`: the official DsPCBSD+ split (train / validation).
- `*_leakfree_train.csv`: the leak-free training splits, i.e. the training split without the removed set R.
- `dspcbsd_subset1800_for_synthetic_copies.csv`: the 1,800-image DsPCBSD+ subset from which the synthetic copies of
  DsPCBSD+ were generated.

## synthetic_copies/

One row per synthetic copy: role (`fit` copies come from validation images and seen degradation combinations; `test`
copies come from test images and held-out combinations), copy file name, source image, crop area fraction, crop
offset, rotation in degrees, degradations and severity (1–5).

## source_data/

- `figures/`: the data plotted in every figure (`figNN_*.csv`), as written by the plotting scripts.
- `tables_and_text/`: the result files from which the tables and the numbers in the text were generated.

## additional_analyses/

- The SSCD, SIFT, published SSCD postprocessing, shared-calibration, retrieval-aware and bootstrap result records.
- `*_comparison_pairs.npz`: all frozen candidate indices, query roles, source labels, cue scores and comparison
  scores. Load with NumPy `allow_pickle=False`. `*_pair_index.json` maps indices to benchmark file names and synthetic
  copy parameters; machine-specific paths have been removed.
- `random_deletion_results.json` and `random_deletion_source_data.csv`: all three training seeds and both endpoints.
- `real_pair_geometry.json`: fixed SIFT corroboration for every real flagged pair; these are geometric evidence
  counts, not human copy labels or precision estimates.

## random_deletion/

Three class-matched removal manifests and completed-checkpoint metadata. Every removal contains 17 non-copy training
images and retains the known copies. `saved_predictions/` contains one NPZ per arm and seed. The files hold image
names, per-detection confidence/classes, precomputed true-positive indicators at IoU 0.50--0.95, ground-truth class
labels and offsets delimiting each image. These support exact AP and contrast verification without GPU inference;
they do not replace the original images and bounding-box annotations for checking a different matching rule.

## analysis_protocols/ and verification/

The original and additional dated analysis plans distinguish confirmation from supplementary comparisons. Run
`python verification/verify_saved_results.py supplementary_data` to verify the comparison scores and paired contrasts.
Run `python verification/verify_control_ap.py supplementary_data` to recompute subset mAP from saved matched
predictions. Both verification scripts require NumPy and no GPU.
