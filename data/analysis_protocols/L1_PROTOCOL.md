# L1 probe protocol (written before any synthetic duplicate is generated or any feature is computed)

Date: 2026-10-26. Round 5 (ESWA), top candidate from `candidates15_r5.md`.

**Idea.** Leakage-robust near-duplicate detection for industrial defect benchmarks: perceptual hashing (A) fused with denoised foundation-model dense correspondences (B).

- **A (host).** Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets (CVPR 2026 Oral; `yeshwanth95/Hash_and_search`).
  - Method: pHash of the image and of its 6 rotations and flips; a pair is a duplicate if the minimum Hamming distance is ≤ 5.
  - Stated limitation: the hashing "is not inherently invariant to strong geometric & radiometric transformations".
- **B (mechanism).** Denoising Vision Transformers (ECCV 2024 Oral; `Jiawei-Yang/Denoising-ViT`): removes the position-dependent artefacts from ViT dense features. The ImageNet-distilled DINOv2-B/14 checkpoint is used.
- **Why A+B is expected to help on industrial images.**
  - Hashing is precise but breaks under small shifts, illumination change, blur, noise or compression.
  - Global semantic features are robust but confuse different images of the same defect texture.
  - Dense correspondences that agree on one geometric transform are robust and precise, and artefact tokens corrupt exactly these correspondences.
  - The convex fusion α·hash + (1−α)·dense should therefore beat both ends.
- **Motivation (pilot, `pilot_leakage.py`, not a gate).** Cross-split pairs that are the same physical region exist in all three benchmarks. In the official DsPCBSD+ split they include shifted or relit copies at Hamming distance 47–71, which the authors' dHash de-duplication did not remove.

## Data

- **Datasets and gallery.** NEU-DET and GC10-DET use the DeQCal splits; DsPCBSD+ uses our 1,080/360/360 splits. The gallery of a dataset is all its images (NEU 1,800; GC10 2,300; DsPCBSD+ 1,800).
- **Synthetic leaked copy** of a source image (ground truth by construction):
  - Geometric part: a random crop covering a fraction u ~ U[0.70, 0.95] of the area, at a random position, keeping the aspect ratio, resized back to the original size, then rotated by a random angle in [−3°, 3°].
  - Radiometric part: DeQCal compound operators (`apply_compound`) with one severity s ~ U{1..5} for all operators of a set.
  - Randomness comes from numpy `default_rng`, seeded by sha256(dataset | role | source file | copy index).
- **Fit queries.** One copy per calibration-split image, with operator sets from `calibration_seen`. They are used only to choose α.
- **Test queries.** Two copies per test-split image, with operator sets from `test_unseen`. These operator compositions never occur among the fit queries.

## Candidates and scores

- **Candidates.** For each query, the 20 gallery images with the highest raw DINOv2-B/14 CLS cosine, plus the true source if it is not among them. Positive pair = (query, source); negative pairs = (query, any other candidate), which are hard negatives.
- **Scores**, all in [0, 1], higher meaning more duplicate-like:
  - **H (A, official).** 1 − min over the 6 rotations and flips of the query of Hamming(pHash(query variant), pHash(candidate)) / 64, using `imagehash.phash` with its defaults.
  - **G.** Raw DINOv2-B/14 CLS cosine (`timm vit_base_patch14_dinov2.lvd142m`, 224 × 224 input).
  - **D_raw / D_dvt.** Dense geometric verification with raw or DVT-distilled DINOv2-B/14 patch features:
    - images are resized to 448 × 448, giving 32 × 32 patches with L2-normalised descriptors;
    - take the mutual nearest-neighbour patch matches;
    - fit a partial affine transform with `cv2.estimateAffinePartial2D` (RANSAC, reprojection threshold 1.5 patches);
    - score = number of inliers / 1024 (0 if fewer than 3 matches).
- **Fusion (primary A+B).** F = α·H + (1−α)·D_dvt, with α ∈ {0, 0.05, …, 1}. α is chosen per dataset on the fit queries to maximise TPR at 1% FPR. F_raw is the same fusion with D_raw and its own α.

## Metrics (test queries)

- **Primary.** TPR at 1% FPR over all positive and negative pairs of a dataset.
- **Also reported.** AUROC, and rank-1 accuracy (the source ranked first among the candidates by the score).

## Decision rule

Go only if all of the following hold:
- **G1, A+B gain.** TPR@1%FPR of F ≥ max over H, G, D_raw and D_dvt + 0.05, averaged over the three datasets. In addition, F ≥ the best single score in each dataset.
- **G2, intermediate α.** The α chosen on fit queries is strictly inside (0, 1) for all three datasets.
- **G3, B's mechanism matters.** TPR@1%FPR of F ≥ that of F_raw + 0.02, averaged over the three datasets.
- **Ceiling rule.** If any single score reaches TPR@1%FPR ≥ 0.97 in all three datasets, G1 is treated as failed.

## Application analysis (reported, not gating)

- **Threshold.** The fused detector with each dataset's fit α, at the threshold giving 0.1% FPR on that dataset's fit-query negatives.
- **Real audit.** Count the test images that have a near-duplicate in the training images:
  - NEU-DET and GC10-DET: DeQCal train vs test;
  - DsPCBSD+: official train vs val, and our train vs test.
- **Visual check.** 30 random flagged pairs per dataset are inspected, and the number judged "same physical region" is reported.
- **Inflation.** Clean mAP50 of the three frozen YOLO11n per dataset on flagged vs unflagged test images.

Nothing above is changed after any score is computed.

## Amendment 1 (2026-10-26, after the weights were verified by SHA256 and before any synthetic copy, feature or score existed)

These are computational details the draft left open.
- **Primary metric.** TPR at 1% FPR is the linear interpolation at FPR = 0.01 of `sklearn.metrics.roc_curve` on the pairs of a dataset.
- **Choosing α.** When several α tie on fit-query TPR@1%FPR, the one with the higher fit-query AUROC is taken; if they still tie, the smallest α.
- **ViT features.**
  - Inference runs under bfloat16 autocast; features are cast to float32 before L2 normalisation.
  - Patch descriptors are the final-norm patch tokens of `forward_features` (DINOv2-B/14 without registers). The CLS score uses token 0.
  - Inputs are resized to 224 × 224 (CLS) or 448 × 448 (dense) without keeping the aspect ratio. This includes the 2048 × 1000 GC10-DET images, treated identically for queries and gallery. Normalisation uses the ImageNet mean and std.
- **Geometric copy.** Rotation uses `cv2.warpAffine` with `BORDER_REFLECT_101`.
- **RANSAC.** `cv2.setRNGSeed(0)` is called before every pair, so dense scores are deterministic.
- **Weights.**
  - `timm/vit_base_patch14_dinov2.lvd142m`, `model.safetensors`, sha256 55cbb5d8…;
  - `jjiaweiyang/DVT`, `imgnet_distilled/vit_base_patch14_dinov2.lvd142m.pth`, sha256 71e07a1d….

  Both load into `timm.create_model('vit_base_patch14_dinov2.lvd142m', dynamic_img_size=True)` with no missing or unexpected keys.
