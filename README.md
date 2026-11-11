# HashDV

Code and numeric evidence for **Beyond Hash Deduplication: Auditing Hidden Overlap in Industrial Defect Benchmarks** (ICDE 2027, Experiment, Analysis, and Benchmark category). The map from the paper's figures and tables to the files that reproduce them is [below](#figures-and-tables-of-the-icde-2027-submission).

HashDV uncovers hidden overlap through **complementary copy detection, acquisition-aware auditing and fixed-test evaluation**. Rotation-aware perceptual hashing and dense geometric verification recover transformed sources. Frozen audits locate matching training content, and retraining measures its evaluation consequences on the original images and annotations.

## Results and artifacts

- Fusion improves synthetic source-file recovery TPR at 1% non-source pair acceptance by **8.9 percentage points** over the best individual cue per benchmark, on average. The gain is reproduced with fresh synthetic copies and a second backbone.
- With identical candidate-background calibration, mean source-file recovery TPR is **88.0% for HashDV** and **80.8% for SSCD**.
- The frozen audit flags matches for **4.7%, 12.4%, and 7.7%** of evaluated images in NEU-DET, GC10-DET, and DsPCBSD+. Pair lists, geometric evidence, acquisition records and label sensitivities make the selected content traceable.
- Three-seed fixed-test retraining gives adjusted local mAP50 responses of **7.1 and 2.3 points** on flagged NEU-DET and DsPCBSD+ images. Adjusted pooled responses are **0.14–0.51 points**. The paper defines their interpretation conditions and reports equal-size removal and match-evidence sensitivities.
- GC10-DET retains an approximately **13-point flagged-subset advantage** after match removal, showing why overlap and its evaluation consequences must be measured together.

The repository contains an installable verifier, frozen audit configurations, deterministic copy generation, split preparation, detector training, and the released scores/predictions needed to check the reported comparisons and intervention contrasts on CPU. Original dataset images and pretrained weights stay with their distributors.

## Figures and tables of the ICDE 2027 submission

Paths are relative to `data/` unless they start with `experiments/` or `configs/`. Figure CSV names keep the plotting scripts' internal numbering.

| Paper item | Files |
|---|---|
| Fig. 1 | `source_data/figures/fig01_pairs.csv`, `source_data/figures/pair_examples.json` |
| Fig. 2 | schematic, no data |
| Fig. 3 | `source_data/figures/fig04_roc.csv`, `fig04_tpr_at_1pct_fpr.csv`, `fig04_fusion_gain.csv`, `fig04_calibrated_gain.csv` |
| Fig. 4 | `source_data/figures/fig05_by_severity.csv`, `fig05_alpha_curve.csv` |
| Fig. 5 | `source_data/figures/fig06_prevalence.csv`, `fig06_best_score_survival.csv`, `fig06_adjacency.csv` |
| Fig. 6 | `source_data/figures/fig08_slopes.csv`, `fig08_forest.csv` |
| Fig. 7 | `source_data/figures/fig09_dose.csv`, `fig09_control.csv`, `fig09_gap_decomposition.csv` |
| Table I | `splits/*.csv` |
| Table II | `additional_analyses/baseline_source_data.csv`, `baseline_gain_bootstrap.json`, `matched_calibration.json`, `calibrated_gain_bootstrap.json`, `sscd_results.json`, `sscd_postprocessing.json`, `sift_results.json` |
| Table III | `additional_analyses/end_to_end_results.json`, `source_data/tables_and_text/l1e_retrieval_recall.json` |
| Table IV | `leak_lists/`, `additional_analyses/real_pair_geometry.json`; natural-pair notes in `additional_analyses/instance_visual_assessment.json` |
| Table V | `configs/recipes/`, `matched_full_arm/*/training/` |
| Table VI, Figs. 6–7 | Steel rows: `additional_analyses/{neu_det,gc10_det}_matched_full_arm.json`, `matched_full_arm/` (full arms) and `random_deletion/saved_predictions/b_*`, `intervention_predictions/gc10_det/b_*` (removal arms). PCB rows: `source_data/tables_and_text/l1d_results.json`, `intervention_predictions/pcb_det/`. NEU random control: `random_deletion/saved_predictions/`. Updated figure CSVs: `matched_full_arm/source_data/` |
| Audit cost (Section V-A) | `additional_analyses/audit_timing.json` (from `experiments/time_audit.py`) |

Files added for this submission sit next to the original evidence, which is unchanged and still checked against `data/SHA256.json`; the additions are listed with their hashes in `data/SHA256_icde2027.json`.

- `data/intervention_predictions/{gc10_det,pcb_det}/{a,b}_<seed>.npz`: matched predictions of the full-split (`a`) and copy-removed (`b`) detectors for seeds 2027, 3407 and 8888, in the layout of `data/random_deletion/saved_predictions/`, with the SHA-256 of each detector's weights. The NEU-DET arms are the `a_*` and `b_*` files in `data/random_deletion/saved_predictions/`. Pooled subset mAP50 recomputed from these files with `verify_control_ap.py` reproduces the flagged-image and benchmark-level contrasts.
- `data/matched_full_arm/{neu_det,gc10_det}/a_matched_<seed>.npz` (`experiments/matched_full_arm.py`): predictions of the steel full-split detectors retrained with the removal arms' execution settings (one data-loading worker, RAM caching, plots disabled; same data, seeds, pretrained weights and hyperparameters), with each run's arguments, metrics and weight hash in `training/`. The script recomputes the steel rows of Table VI with their image and paired-seed intervals, the NEU match-evidence subgroups, the neighbour control and the subset gaps (`data/additional_analyses/*_matched_full_arm.json`), and reproduces the earlier full-arm values as a check. The earlier steel full arms (zero workers, no cache) remain as `random_deletion/saved_predictions/a_*` and `intervention_predictions/gc10_det/a_*`.
- `data/additional_analyses/pcb_seed_sensitivity.json` (`experiments/pcb_seed_sensitivity.py`): DsPCBSD+ intervals that resample the paired seeds jointly with the images.
- `data/additional_analyses/chance_flags.json` (`experiments/chance_flags.py`): non-source query exceedance rates on held-out synthetic queries at the frozen thresholds, a hypothetical rescaling to each evaluation-set size, and flagged pairs satisfying the two pHash rules. These diagnostics and their legacy expected-count field names do not estimate real-query false flags or establish natural-match precision; candidate identity is examined in the label-sensitivity analysis.
- `data/additional_analyses/audit_timing.json` (`experiments/time_audit.py`): stage-by-stage wall-clock time of the released audit. A rerun's flagged counts can differ from `data/leak_lists/` by a near-tied candidate; the frozen lists remain the sets used in the paper.

## Install

Python 3.10 or newer; Python 3.12 was used for the paper.

```bash
python -m pip install -e .
```

For image auditing and synthetic-copy generation, install PyTorch/torchvision appropriate for your machine, then:

```bash
python -m pip install -e '.[inference]'
```

For detector retraining:

```bash
python -m pip install -e '.[training]'
```

The experiment environment is recorded in `requirements-paper.txt`. Frozen-backbone inference uses CUDA bfloat16 followed by normalized float16 dense features, matching the original implementation. CPU inference uses float32 model evaluation; small numerical differences can change near-tied candidates. Saved-pair verification below reproduces the reported numeric results without loading a model.

## Check the paper results on CPU

The `data/` directory includes all pair scores, match lists, original/corrected split manifests, synthetic-copy parameters, and matched prediction arrays. These are the same numeric files supplied with the manuscript.

```bash
python -m hashdv reproduce --data data
```

This checks the released SSCD/SIFT/shared-calibration comparisons and retrieval-aware TPR. It then recomputes Table VI for **all three datasets**, using each seed's full/removed matched predictions on flagged images L, other images U and pooled images T. It rebuilds both mAP50 and mAP50-95 and six AP values per seed, checking the released means, per-seed values and both difference-in-differences contrasts against their reference records. The NEU class-matched random-deletion comparison is checked separately from its nine detector instances. Missing predictions cause failure rather than a partial success message.

This command checks the original intervention records. The steel rows of the submitted Table VI use the execution-matched full arms and are recomputed, with their intervals, by `python experiments/matched_full_arm.py` (a few minutes on CPU); `tests/test_matched_full_arm.py` checks their point estimates.

The default command checks point estimates, not confidence intervals. To rerun **all three main-table 1,000-resample stratified image-bootstrap intervals and the NEU random-control intervals**:

```bash
python -m hashdv reproduce --data data --bootstrap
```

Main-table intervals use RNG seed 0 independently for each dataset; the NEU random-control intervals use seed 271027. Seeds of the trained detectors stay fixed in these image bootstraps. This CPU check reproduces AP aggregation from saved ten-IoU TP indicators; it does not rerun GPU inference or independently re-match original boxes and annotations. In confidence-tie edge cases the main-table verifier allows at most 1e-6 in score units (0.0001 mAP point), substantially finer than the reported table precision. The historical script in `data/verification/` remains part of the unchanged original evidence; the CLI uses the maintained verifier in `hashdv/verification/`.

The added GC10-DET and DsPCBSD+ prediction files have [checkpoint and file provenance](data/intervention_predictions/provenance.json). Their recorded detector SHA-256 values were checked against the original best steel checkpoints and last official-PCB checkpoints. All original 86 evidence files remain checked by `data/SHA256.json`; subsequent evidence additions use `data/SHA256_icde2027.json`.

Post-review NEU control sensitivities can be regenerated with `python experiments/intervention_revision_sensitivity.py`. This reports paired-seed and acquisition-index block sensitivities from the existing matched predictions. It creates no new training run and supplies no physical-instance labels. Its fixed analysis protocol is in `data/analysis_protocols/intervention_revision_PROTOCOL.md`.

Post-review instance-label sensitivity is reproduced by `python experiments/instance_label_sensitivity.py`; subgroup deletion-response sensitivity is reproduced by `python experiments/instance_intervention_sensitivity.py`. These retain the original arrays and frozen audit sets, publish the identified non-source/flagged-pair intersections, and state the evidence basis of visual assessments rather than presenting them as independent human ground truth. Analysis records and their protocol are in `data/additional_analyses/instance_*`. Current maintained code, experiment scripts, tests and documentation are checked by the root `SHA256_code.json` in addition to the data checksums.

`experiments/instance_contact_sheets.py` reconstructs the 17 NEU pair contact sheets from distributor-provided images; use `python experiments/instance_contact_sheets.py --help` for image-root and output options. Original benchmark pixels are not redistributed in this artifact.

Reproduce the frozen fusion fit from the original fit pairs:

```bash
python -m hashdv calibrate data/additional_analyses/neu_det_comparison_pairs.npz
```

The fitted fusion weight and threshold match `configs/neu_det.json`; equivalent files are supplied for GC10-DET and DsPCBSD+. Each NPZ is loaded with `allow_pickle=False`.

## Audit a training/evaluation split

```bash
python -m hashdv audit --train-dir datasets/example/images/train \
  --test-dir datasets/example/images/test --config configs/neu_det.json \
  --output outputs/example_audit --checkpoint weights/dinov2_vitb14_raw.safetensors
```

The checkpoint is a timm-format `vit_base_patch14_dinov2.lvd142m` state dict. Without `--checkpoint`, timm downloads its pretrained model. The original DINOv2 model is [DINOv2](https://github.com/facebookresearch/dinov2).

Outputs are `pairs.csv`, `closure.csv`, `sets.json`, and `retained_train.csv`. The audit retrieves top-20 candidates by 224-pixel DINOv2 class-token similarity and verifies them with six-view 64-bit pHash plus partial-affine RANSAC over 448-pixel patch features. A bounded CPU feature cache controls memory. DsPCBSD+'s configuration enables one additional training-to-training closure step. Input images are read without modification; use a new output directory for each audit.

For a different benchmark, fit a new fusion/threshold configuration on separate synthetic fit copies. The supplied thresholds are the paper's fixed benchmark-specific settings. Pair FPR is a verification metric; the query decision takes the maximum over candidates. See `docs/PROTOCOL.md` for the relation between validation and real auditing.

## Generate the confirmation copies

Obtain the original benchmark images and keep the released file names. A recursive image tree with unique basenames is accepted.

```bash
python scripts/generate_copies.py --dataset neu_det \
  --image-root datasets/neu_det/images --output outputs/neu_det_copies
```

The generator replays the original `L1b` seed rule, operator pools and RNG consumption order, checking each crop, rotation, operator list and severity against the released manifest. Add `--limit 2` for a small check. The geometric transform and corruption operators are in `hashdv/synthetic.py`.

## Reproduce detector interventions

Obtain YOLO-format annotations from the dataset distributors and prepare an independent dataset copy:

```bash
python scripts/prepare_split.py --benchmark neu_det --arm targeted \
  --image-root datasets/neu_det/images --label-root datasets/neu_det/labels \
  --output outputs/neu_targeted

python scripts/train_detector.py --benchmark neu_det --arm targeted --seed 2027 \
  --data-yaml outputs/neu_targeted/data.yaml --weights yolo11n.pt --dry-run
```

Remove `--dry-run` to train. Repeat for `full` and seeds `2027`, `3407`, `8888`. The NEU-DET `random` arm reads each seed's exact class-matched deletion list; prepare a separate dataset copy for each seed. GC10-DET uses `full` and `targeted`; DsPCBSD+ uses `--benchmark pcb_det` and its official split.

NEU-DET/GC10-DET use up to 150 epochs and the best validation checkpoint. DsPCBSD+ uses 50 epochs, no validation during training, and `last.pt`. Original steel full-data runs used zero loading workers/no cache and `plots=True`; deletion arms use one worker/RAM caching and `plots=False`. DsPCBSD+ arms share runtime settings. The equal-size NEU-DET targeted/random comparison also shares runtime settings. These are the recorded execution settings, rather than an assertion that full/removed steel runs were identical. Recipes and checkpoint hashes are recorded with each completed run.

## Repository map

| Path | Contents |
| --- | --- |
| `hashdv/` | Verifier, fusion calibration, synthetic transforms, interventions, and CPU verification |
| `configs/` | Fixed benchmark parameters and fit/test degradation pools |
| `data/` | Original numeric evidence and file-level split/match manifests |
| `scripts/` | Copy generation, dataset preparation, and paired training |
| `experiments/` | Confirmation rescoring, SSCD/SIFT inference, postprocessing and shared calibration |
| `docs/` | Metric interpretation, dataset access, and reproduction protocol |
| `tests/` | Calibration, score/AP reproduction, split/control and geometric consistency checks |

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

GitHub Actions checks the CPU reproduction on each push. Geometry inference tests run when the optional PyTorch/OpenCV packages are installed.

Raw feature rescoring and full SSCD/SIFT inference instructions are in [experiments/README.md](experiments/README.md).

## Citation and licenses

This repository accompanies a manuscript submitted to ICDE 2027; no acceptance status is implied. Citation metadata are in `CITATION.cff`.

Original code is MIT-licensed. External models, software, images and annotations retain their upstream licenses; see `THIRD_PARTY.md` and `docs/DATASETS.md`. The stored result arrays and manifests contain no benchmark images. Historical filenames containing `leakfree` denote removal of the specified matched training set. Real-audit counts refer to the frozen verifier's flagged sets, which can include shared acquisition layouts; fixed-test retraining measures their consequence.
