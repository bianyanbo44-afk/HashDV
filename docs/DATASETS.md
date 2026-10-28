# Data access and file mapping

| Benchmark | Files supplied here | Original data |
| --- | --- | --- |
| NEU-DET | Study-specific class-stratified train/validation/test split, matched pairs, corrected training list | NEU surface-defect detection benchmark; Song and Yan (2013), He et al. (2020) |
| GC10-DET | Study-specific class-stratified split, matched pairs, corrected training list | Lv et al. (2020), GC10-DET steel-surface benchmark |
| DsPCBSD+ | Official train/validation split, matched pairs, closure list, corrected training list, 1,800-image synthetic-copy subset | [Lv et al. dataset record](https://doi.org/10.6084/m9.figshare.24970329), CC BY 4.0 |

Images and annotations must be obtained from their distributors and attributed according to their terms. This repository does not redistribute them or place them under its code license.

The manifests preserve original basenames. `scripts/prepare_split.py` accepts recursive image and YOLO-label trees with unique file names, verifies that every selected image has its label file, and copies only the selected files into a new directory. Calibration images in the source experiments are named `validation` in the published split CSVs. Official DsPCBSD+ evaluation uses its validation set with no validation-based checkpoint selection.

## Stored evidence

`data/SHA256.json` records the original anonymous supplementary data hashes. NPZ arrays contain scores or saved per-detection TP/confidence/class records and image offsets; use `allow_pickle=False`. All original numeric files retain their hashes. `data/README.md` describes the individual files. Result archives do not contain raw model checkpoints, raw images or bounding-box coordinates for redoing detector-to-label matching. Recomputing that matching requires the original images/annotations and trained checkpoints.
