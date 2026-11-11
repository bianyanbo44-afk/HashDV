# Instance-label sensitivity and visual evidence protocol

Recorded 2026-11-11 before the new sensitivity results are computed.

The original arrays, original labels, frozen auditor parameters and real-image removal sets are never changed by this analysis. A non-source candidate means a different gallery file. It is not established physical-instance ground truth.

## Sensitivity rule

Join each negative comparison row to the original-image pair consisting of its synthetic query source and gallery candidate. Exclude that row only if this ordered pair is listed in the already frozen real-image flagged-pair CSV for that benchmark. Apply the identical deterministic rule to every scoring method and to fit and test roles. Do not inspect score changes to tune the exclusions. Report the original and excluded counts separately.

Compute fit fusion weights on the original 0.05 grid, using the existing tie-breaking and 99.9th percentile threshold. Report the original and sensitivity fit parameters. Report held-out TPR at 1% and 0.1% non-source-pair rates, and the fraction of queries with a threshold-passing non-source candidate, for both the frozen and refitted sensitivity parameters. These rates describe source-file discrimination, not independently verified physical-copy specificity. Multiplication by the real split size is only a synthetic-query reference, not a validated real-world chance expectation.

For uncertainty, sample original test source images with replacement within each benchmark. Both synthetic transformations and all retained candidate rows of one source receive the same bootstrap multiplicity. Keep fusion weights fixed during bootstrap. Compute 1000 paired draws for HashDV minus the better of hash and dense verification. Also compute the matched-calibration contrasts using the same source draws. Macro-average the three benchmark effects within each draw. This is a sensitivity analysis of already saved scores, not a new independent replication or causal test.

## Visual evidence rule

Inspect every NEU flagged pair (17) and deterministic score-stratified examples from GC10 and PCB. Display raw images with filenames and existing SIFT corroboration, preserving aspect ratio. Record AI-assisted assessment only. No human annotator, independent human review or inter-rater statistic may be claimed. Use categories same physical region / same labelled defect, acquisition sibling, different region, or uncertain. Visual judgments remain separate from the score-based exclusion rule and never relabel the original arrays. Matching texture or geometry alone is insufficient to declare defect identity.

Uncertain observations remain uncertain. Report the sampling boundary. A small example review cannot establish overall real-audit precision or physical-copy prevalence.

## Post-hoc intervention subgroup extension

At the parent editor's request, evaluate the nine pre-existing SIFT-corroborated NEU images separately from the eight unresolved images, using the original saved full, 17-image-targeted and 17-image-random models. Keep original U343 as reference for every contrast. Do not reclassify the eight unresolved images into a clean control. Report the three seed pairs and 1000 stratified image-bootstrap draws conditional on these trained models, for Delta_L and Gamma (random minus targeted difference, adjusted by original U343). The original pooled T360 estimate is unchanged. This is post-hoc subgroup sensitivity, not a new nine-pair training intervention. AI-assisted visual inspection agrees with the nine-correspondence grouping but does not supply independent human gold labels.
