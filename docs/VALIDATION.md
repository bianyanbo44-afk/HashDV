# Release validation

## Submitted execution-matched release, 12 November 2026

- 48 local tests pass, including submitted steel point estimates, the current default reproduction entry point, current and original missing-file failures, training recipe pairing, original evidence hashes and shared-calibration reconstruction.
- The default CPU command reconstructs Table VI using execution-matched steel full arms and the original official-PCB arms. All three main-table 1,000-resample image-bootstrap intervals, both AP endpoints and the NEU equal-size Gamma intervals were recomputed successfully.
- Dry-runs for NEU full/targeted/random, GC10 full/targeted and PCB full/targeted all match the submitted runtime settings. Earlier records remain accessible with the explicit original-version option.
- The class-support sensitivity reports subset classes, per-class image/box counts, missing-class frequencies, common-class comparisons and 1,000 accepted conditional resamples per dataset.
- 86 original evidence files, 58 additional evidence/protocol files and 51 code/configuration/documentation files pass their recorded SHA256 checks.

## Original release, 28 October 2026

- 16 local tests cover original evidence hashes, fusion fits and thresholds, all stored comparison scores, nine-detector AP/contrast reconstruction, corrected split sizes, class-matched deletion manifests, recipe pairing, rotation hashing, geometric consistency, and recomputed shared calibration.
- A sampled CUDA check produced bitwise-identical DINOv2 class and dense features between the package and the original implementation. Hash and RANSAC scores also matched.
- On a procedural image, the synthetic generator matched the original pixels and metadata at all five degradation severities with the complete operator chain.
- An end-to-end image-audit smoke test found the known exact copy and excluded its self-match in the one-step training closure, including with an LRU cache smaller than the candidate list.
- All 86 original numeric supplementary files retain their recorded SHA256 digests.

The original full experiments and stored prediction/score arrays supply the paper results. Release checks exercise the packaged implementation and CPU reconstruction; training scripts expose the recorded recipes for users who rerun the detectors.

GitHub Actions reruns the CPU tests and numeric reconstruction. The optional CUDA parity check was performed locally with the paper's checkpoint and environment; source datasets and model checkpoints are obtained separately.
