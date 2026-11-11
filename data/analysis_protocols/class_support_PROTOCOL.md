# Class-support sensitivity

Recorded on 2026-11-12 before running the added resampling analysis.

Use the submitted execution-matched steel full-arm predictions, the existing
steel removal predictions, and the official PCB full/removal predictions.
Retain the frozen L/U image membership and all three paired seeds.

1. Report each subset's observed classes, images per class and boxes per class.
2. Repeat the main bootstrap's RNG seed 0 and L-then-U draw order for 1,000 draws.
   Count draws missing any class from each original subset.
3. Recompute subset AP on the intersection of L/U class supports while retaining
   all original images. Report both AP endpoints, the full and removed gaps,
   and the adjusted responses. This checks macro-average class composition.
4. For a sensitivity of the primary mAP50 response intervals, use RNG seed
   11122026. Draw L and U with replacement at their original image counts.
   Accept the joint draw when every originally observed class remains present
   in its own subset. Continue until 1,000 accepted draws. Average the three
   paired seed responses within each draw and report percentile intervals.
   Record attempted and accepted draws. This is an image bootstrap conditional
   on the observed class-support event and on the trained models.

The main-table image bootstrap and the seed/acquisition sensitivities retain
their original definitions. This added analysis estimates sensitivity to class
support using existing predictions. It supplies no new trained model or natural
physical-instance label.
