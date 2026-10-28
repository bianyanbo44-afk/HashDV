# L1d protocol: causal leakage inflation on the official DsPCBSD+ split
(written before the official split is re-audited with the confirmed configuration, and before any detector is trained on it)

Date: 2026-10-26. This follows L1b.
- In L1b, the fusion was confirmed.
- Causal inflation was positive on NEU-DET (17 leaked test images) and null on GC10-DET, where seed noise swamped the effect.

**Why this split.**
- It is the split the community uses: train 8,208 and val 2,051 images, 9 classes.
- Its authors deduplicated it with dHash.
- The L1 audit flagged 172–213 of its val images, about ten times the number of leaked images NEU had in L1b.

**Prior knowledge, disclosed.**
- The L1 audit of this split used the L1 configurations:
  - F: α 0.1, threshold 0.183, 213 val images flagged;
  - F_raw: α 0.1, threshold 0.268, 172 flagged.
- A visual check of 30 F_raw flags found 29–30 same-region pairs.
- No detector has been trained or evaluated on this split.
- A speed pilot (one epoch on 25% of train, validated on 64 train images) touched no val image.

## Part A: leak sets (configuration frozen from L1b)

**Score.**
- F = α·H + (1−α)·D_B1, with α = 0.15 and threshold 0.3056640625.
- Both values were fitted on the fresh DsPCBSD+ fit queries of L1b (`l1b_out/l1b_results.json`, `pcb_det`), not on this split.
- **H.** pHash, with the minimum Hamming distance over the six rotation and flip variants of the query.
- **D_B1.** Raw DINOv2-B/14 dense geometric verification, as in L1b:
  - 448 px input, 32 × 32 patches;
  - mutual nearest neighbours, then partial-affine RANSAC at 1.5 cells;
  - score = inliers / 1024.

**Candidates.** The top 20 train images by raw DINOv2-B/14 CLS cosine.

**Sets.**
- **L.** Val images whose best F is at or above the threshold.
- **R0.** Train images in any flagged (val, train) pair.
- **Closure R1.** Train images not in R0 that are among the top-20 CLS candidates of some R0 image with F at or above the threshold.
- **R = R0 ∪ R1.** The leak-free model then sees no flagged near-copy of a leaked val image, and no flagged near-copy of such a copy.
- **U.** Val − L.

All val → train candidate scores are saved.

## Part B: detectors (one pipeline for both groups)

**Groups.**
- **a (full).** The official train split, 8,208 images.
- **b (leak-free).** Train − R.

**Shared settings.** Everything else is identical:
- YOLO11n from `yolo11n.pt` (SHA256 0ebbc80d…);
- seeds 2027 / 3407 / 8888;
- imgsz 640, batch 16, 50 epochs;
- optimizer `auto`, AMP, deterministic;
- dataloader workers 1, no cache;
- `close_mosaic` 10, no per-epoch validation (`val=False`);
- Ultralytics 8.4.115.

**No model selection on val.** Evaluation uses `last.pt` from a fixed budget, so val never influences training.

**Run order.** a2027, b2027, a3407, b3407, a8888, b8888.

**Budget.** The pilot gave about 135 s per epoch, so about 1.9 h per run and about 11 h in total.

**Failures.** A run that fails for a technical reason (a crash or the commit watchdog) is restarted from scratch with the same settings.

## Part C: evaluation

**Predictions.**
- Clean val images, with the Model.val conventions of D1: rect canvas, conf 0.001, NMS IoU 0.7.
- Per-image predictions for all 6 detectors.

**Metrics.** mAP50 (primary) and mAP50-95 on any subset, using Ultralytics' matching and `ap_per_class`.

**Estimands.** As in L1b, with mAP_x averaged over the 3 seeds:
- Δ_L = [mAP_a(L) − mAP_b(L)] − [mAP_a(U) − mAP_b(U)];
- Δ_T = [mAP_a(T) − mAP_b(T)] − [mAP_a(U) − mAP_b(U)];
- per-seed versions Δ_L(s), from the seed-s pair.

**Uncertainty.** 95% bootstrap intervals from 1,000 resamples of val images, stratified by L / U; the seeds stay fixed.

## Hypotheses

- **H1 (primary).** Both of the following hold:
  - Δ_L(mAP50) > 0, with a 95% interval excluding 0;
  - Δ_L(s) > 0 for each of the three seeds.
- **H2 (secondary).** Δ_T(mAP50) > 0, with a 95% interval excluding 0.
- **Always reported:**
  - the mAP50-95 versions;
  - the six subset mAPs;
  - the raw gaps a_L − a_U and b_L − b_U;
  - |L|, |R0| and |R|.

Nothing above changes after the re-audit, the training or the evaluation.
