# Intervention revision sensitivity protocol

Registered before computing the revision analyses on 2026-11-11.

## Scope

Read the original nine NEU-DET saved prediction arrays for full, targeted, and class-matched random deletion at seeds 2027, 3407, and 8888. Preserve every original prediction, training outcome, split, deletion manifest, and reported estimate. No new training is claimed.

The primary revision contrast is Gamma = [m_random(L) - m_random(U)] - [m_targeted(L) - m_targeted(U)]. It cancels the full arm algebraically and compares the two arms sharing workers=1 and RAM cache. It measures specificity to the selected deletion rather than identifying physical-copy causality.

## Prespecified sensitivity analyses

- Primary endpoint mAP50. AP is recomputed from pooled saved TP indicators, confidences, predicted classes, and target classes using the released 101-point integration convention.
- Reproduce the fixed-seed stratified image bootstrap with 1,000 draws and RNG 271027. Resample L and U separately, applying the same indices across all nine models.
- Acquisition-adjacency sensitivity uses filename class and floor((within-class numeric index - 1) / width), for widths 5, 10, and 20. These are surrogate blocks, not verified physical acquisition units. Jointly sample as many blocks as observed with replacement, keeping every image in each selected block and applying the same indices to all models. Recompute L/U AP on the sampled images. Run 500 draws per width, with RNG 11112026 + width. Draw again if either subset is empty.
- Repeat width-10 blocks with 500 paired-seed bootstrap draws, drawing three seed pairs with replacement independently of the block draw. This is a sensitivity analysis with only three training/deletion pairs, not a calibrated population guarantee.
- Report all three individual seed values and all three leave-one-seed-out means. Also enumerate all 27 equally weighted three-seed bootstrap resamples to show dependence on the limited observed seeds. Do not use positivity of this bootstrap as a significance claim.
- Report intervals at the 2.5th and 97.5th percentiles. Do not select widths, seeds, endpoints, or samples on the basis of favorable results. Include all registered outputs.

## Training provenance and follow-up scope

Inspect original args.yaml and results.csv for both arms and all three benchmarks. No multi-hour training is launched in this revision pass. Any future matched-full experiment must use the same architecture, initial checkpoint, seed, loader workers, RAM caching, optimizer, augmentation settings, validation split, early-stopping patience, and checkpoint choice as its paired targeted arm. PCB controls must retain 50 epochs, validation disabled, and final checkpoint, and match deletion counts by class and source when available. A completed future training is not part of this revision until all registered seeds finish and are evaluated on the unchanged original evaluation images.
