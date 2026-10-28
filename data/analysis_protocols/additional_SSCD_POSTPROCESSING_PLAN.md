# SSCD postprocessing sensitivity, 2026-10-27

This additional sensitivity analysis was specified after the four descriptor comparisons were available, before computing any score-normalized or whitened results. It is not part of the original confirmation.

The official evaluation code uses query score normalization with beta 1 and background-neighbour indices [0,2], and optionally full-dimensional PCA whitening. Evaluate these published operations for both already computed image preprocessing choices. Background descriptors come only from the original training split within each synthetic benchmark. They exclude the calibration/test source images by split membership; no copy labels or held-out test scores select parameters. Evaluate raw, training-mean-centered and 512-dimensional whitened descriptors with the published score normalization, and whitening without score normalization. Report all configurations, irrespective of ranking.

Score normalization subtracts the mean of the three largest background similarities from each query's candidate similarities. PCA whitening is fit in float64 to background descriptors, followed by L2 normalization. Use a numerical singular-value floor of 1e-12, not a test-selected regularization. This is a stronger postprocessing check of the dedicated baseline, rather than a reason to suppress its strongest results.

Authority: https://raw.githubusercontent.com/facebookresearch/sscd-copy-detection/main/sscd/disc_eval.py and docs/Evaluation.md.
