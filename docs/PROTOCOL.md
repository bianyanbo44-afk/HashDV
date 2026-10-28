# Evaluation and intervention protocol

## Verification

Synthetic sources come from separate fit and test splits. Fit and test use different degradation combinations. Each candidate list contains top-20 DINOv2 class-token neighbours; the source is appended when it is missing so verification can be evaluated independently of retrieval. Retrieval-aware results count a source outside the original top 20 as a miss.

HashDV fuses H and D using `F = alpha*H + (1-alpha)*D`. Alpha is selected on the fit pairs from 0 to 1 in steps of 0.05, maximizing TPR at 1% pair FPR, then AUROC, then preferring smaller alpha. The audit threshold is the fit-negative 99.9th percentile using the original order-statistic rule. The ordinary real audit uses this raw fused score and frozen threshold. Shared-calibration comparisons are a separate, controlled experiment and do not change the audit removal sets.

SSCD's published normalization and whitening variants remain in the released results. Its strongest full-background configuration reaches 82.9% mean TPR; the controlled 88.0% versus 80.8% comparison uses identical retrieved-background calibration. Baseline conclusions should retain these comparison conditions.

## Real-image matching

L is the set of flagged evaluation images, R0 their matched training images, and R1 the additional training matches from one closure step on DsPCBSD+. Targeted deletion removes R=R0 union R1. U contains the other evaluation images. The frozen lists specify these operational sets exactly; no independent human copy labels are asserted. The SIFT cross-check supplies geometric corroboration counts.

## Fixed-test retraining

Let `m_a(S)` and `m_b(S)` be seed-averaged subset mAP for full and targeted-removal training. Every evaluated image is retained.

```
Delta_L = [m_a(L)-m_b(L)] - [m_a(U)-m_b(U)]
Delta_T = [m_a(T)-m_b(T)] - [m_a(U)-m_b(U)]
```

The contrast uses U to adjust for the shared training-content/size effect. A copy-specific interpretation uses the shared-effect assumption stated in the manuscript. Subset AP is computed by pooling detections and labels, rather than averaging image AP. Pooled AP is non-additive; `p*Delta_L` is an illustrative prevalence reference rather than an identity for Delta_T.

For NEU-DET, the equal-size, class-matched deletion contrast is `Gamma=G_random-G_targeted`, with `G_x=m_x(L)-m_x(U)`. The shared full-data arm cancels, so Gamma compares two deletion arms with matching size, class counts and runtime settings. Gamma is 6.5 points, with a 95% image-bootstrap interval of -0.4 to 11.2, positive in all three seeds. This interval is not a statistically significant positive result. The original primary NEU-DET Delta_L is 7.1 points, CI 0.9 to 10.3.

Intervals resample evaluation images within L and U with the three trained seed pairs held fixed; seed-wise results show training variation. DsPCBSD+'s closest-third analysis is post hoc. GC10-DET's approximately 13-point flagged-subset advantage persists after removal and is not itself a measurement of copy-induced gain.

The dated original and supplementary analysis plans are retained under `data/analysis_protocols/`. No new detector training is needed to execute the saved-prediction verification.
