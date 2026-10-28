# Release validation — 28 October 2026

- 16 local tests cover original evidence hashes, fusion fits and thresholds, all stored comparison scores, nine-detector AP/contrast reconstruction, corrected split sizes, class-matched deletion manifests, recipe pairing, rotation hashing, geometric consistency, and recomputed shared calibration.
- A sampled CUDA check produced bitwise-identical DINOv2 class and dense features between the package and the original implementation. Hash and RANSAC scores also matched.
- On a procedural image, the synthetic generator matched the original pixels and metadata at all five degradation severities with the complete operator chain.
- An end-to-end image-audit smoke test found the known exact copy and excluded its self-match in the one-step training closure, including with an LRU cache smaller than the candidate list.
- All 86 original numeric supplementary files retain their recorded SHA256 digests.

The original full experiments and stored prediction/score arrays supply the paper results. Release checks exercise the packaged implementation and CPU reconstruction; training scripts expose the recorded recipes for users who rerun the detectors.

GitHub Actions reruns the CPU tests and numeric reconstruction. The optional CUDA parity check was performed locally with the paper's checkpoint and environment; source datasets and model checkpoints are obtained separately.
