# L1b protocol: confirmation of hash + raw-DINOv2 dense verification, and causal leakage inflation
(written before any fresh synthetic copy, leak set or retrained detector exists)

Date: 2026-10-26. This follows section 3 of the round-5 report.

- **Origin.** In L1, F_raw = α·H + (1−α)·D_raw was a pre-registered comparator, not the primary method. It beat both of its single channels on all three datasets: test TPR@1%FPR 0.803 against 0.698 for D_raw and 0.518 for H.
- **Part I** re-tests that claim on fresh data and on a second backbone, with everything frozen from L1 except the random draws.
- **Part II** measures causally how much leakage inflates detector accuracy.

## Part I: confirmation

- **Fresh queries.**
  - Same generator as L1 (`l1_build.make_copy`): crop keeping 70–95% of the area, ±3° rotation, DeQCal compound operators with one severity in 1–5. Fit queries use `calibration_seen` operator sets; test queries use `test_unseen` sets.
  - Every random draw is salted with the string `L1b`, so no query of L1 is reused.
  - Sources, gallery and candidate rule are the same as in L1: top 20 by raw DINOv2-B CLS cosine, plus the true source if missing.
- **Scores.** H (pHash, A's official augmentation) and the dense geometric verification of L1, with two backbones:
  - **B1 = raw DINOv2-B/14.** As in L1: 448 × 448 input, 32 × 32 patches. Score = RANSAC inliers / 1024.
  - **B2 = ConvNeXt-T (`timm convnext_tiny.in12k_ft_in1k`, supervised, from the local cache).**
    - Stage-3 feature map (stride 16) at 448 × 448, giving 28 × 28 descriptors, L2-normalised.
    - Same mutual-NN + `estimateAffinePartial2D` rule; the reprojection threshold is 1.5 cells.
    - Score = inliers / 784.
- **Fusions.** F1 = α1·H + (1−α1)·D_B1 and F2 = α2·H + (1−α2)·D_B2. α is chosen per dataset on the fresh fit queries, by the same rule as in L1 (max TPR@1%FPR, then AUROC, then the smaller α).
- **Metric.** Fresh test queries, TPR@1%FPR by ROC interpolation, as in L1 Amendment 1.
- **Gates.**
  - **C1 (primary, B1).** F1 ≥ max(H, D_B1) + 0.05 averaged over the three datasets, and F1 > max(H, D_B1) in each dataset.
  - **C2 (generality, B2).** The same condition for F2, H and D_B2.
  - **C3 (intermediate α).** α1 is strictly inside (0, 1) in at least two of the three datasets. In L1, DsPCBSD+ chose α = 0.1 because its hash is weak, so all three are not required.
- **Decision.** The fusion is **confirmed** if C1 and C3 hold. It is **backbone-general** if C2 also holds.

## Part II: causal inflation (NEU-DET and GC10-DET)

**Leak sets.** These use the Part I fusion F1 with the fresh α1, at the threshold giving 0.1% FPR on the fresh fit-query negatives of that dataset.
- L = test images whose best F1 over their top-20 train candidates is at or above the threshold.
- R = every train image that, for some test image, is among its top-20 candidates with F1 at or above the threshold.
- U = test images not in L.

DsPCBSD+ is not retrained: its split has only about a dozen leaked test images, too few for a subset mAP.

**Detectors.**
- **a.** The existing frozen DeQCal YOLO11n (seeds 2027, 3407, 8888), trained on the full train split.
- **b.** New YOLO11n with the same seeds, trained on train − R; calibration and test splits are unchanged.
- The training recipe is DeQCal's (150 epochs, imgsz 640, batch 8, patience 30, deterministic, AMP, from `yolo11n.pt`, `best.pt` by calibration mAP50-95). Two changes:
  - dataloader workers = 1 (commit limit of this laptop);
  - `cache='ram'` (resized images held in memory) to make GC10 training feasible.

  Both changes affect L and U alike and cancel in the difference-in-differences.

**Evaluation.**
- Clean test images, with the Model.val conventions of D1 (rect canvas, conf 0.001, NMS IoU 0.7), producing per-image predictions for all 12 detectors.
- mAP50 (primary) and mAP50-95 on any image subset with Ultralytics' matching and `ap_per_class`.

**Estimands.** mAP_x(S) is the mAP of detector group x ∈ {a, b} on subset S, averaged over the 3 seeds. T is the whole test split.
- **Leaked-image inflation:** Δ_L = [mAP_a(L) − mAP_b(L)] − [mAP_a(U) − mAP_b(U)].
- **Benchmark-level inflation:** Δ_T = [mAP_a(T) − mAP_b(T)] − [mAP_a(U) − mAP_b(U)].
- **Uncertainty.** 95% bootstrap intervals from 1,000 resamples of test images, stratified by L / U; the seeds stay fixed.

**Hypothesis.** Δ_L > 0 with a 95% interval excluding 0 in both datasets, i.e. leakage causally inflates accuracy on leaked test images. Δ_T and mAP50-95 are reported.

Nothing above changes after any fresh score, leak set or retrained detector is computed.
