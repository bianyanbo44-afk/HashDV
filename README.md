# HashDV

Code and numeric evidence for **HashDV: Exposing hidden copies and measuring accuracy inflation in industrial defect benchmarks**.

HashDV connects two questions: **which evaluation images have matching training content, and how much does that content contribute to detector accuracy?** It combines rotation-aware perceptual hashing with dense geometric verification, then measures the consequence of removing matched training images while keeping the evaluation images fixed.

## Results and artifacts

- Fusion improves verification TPR at 1% pair FPR by **8.9 percentage points** over the best individual cue per benchmark, on average. The gain is reproduced with fresh synthetic copies and a second backbone.
- With identical candidate-background calibration, mean verification TPR is **88.0% for HashDV** and **80.8% for SSCD**.
- The frozen audit flags matches for **4.7%, 12.4%, and 7.7%** of evaluated images in NEU-DET, GC10-DET, and DsPCBSD+.
- Three-seed fixed-test retraining gives difference-in-differences estimates of **7.1 and 2.3 mAP50 points** on flagged NEU-DET and DsPCBSD+ images. Pooled benchmark effects are **0.14–0.51 points**.

The repository contains an installable verifier, frozen audit configurations, deterministic copy generation, split preparation, detector training, and the released scores/predictions needed to check the reported comparisons and intervention contrasts on CPU. Original dataset images and pretrained weights stay with their distributors.

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

This recomputes every released SSCD/SIFT/shared-calibration comparison, retrieval-aware TPR, nine-detector subset AP, and paired deletion contrasts. To also rerun the 1,000-resample class-matched control confidence intervals:

```bash
python -m hashdv reproduce --data data --bootstrap
```

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

NEU-DET/GC10-DET use up to 150 epochs and the best validation checkpoint. DsPCBSD+ uses 50 epochs, no validation during training, and `last.pt`. Original steel full-data runs used zero loading workers/no cache; deletion arms use one worker/RAM caching. DsPCBSD+ arms share runtime settings. The equal-size NEU-DET targeted/random comparison also shares runtime settings. Recipes and checkpoint hashes are recorded with each completed run.

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

This repository accompanies a manuscript; no acceptance status or journal DOI is implied. Citation metadata are in `CITATION.cff`.

Original code is MIT-licensed. External models, software, images and annotations retain their upstream licenses; see `THIRD_PARTY.md` and `docs/DATASETS.md`. The stored result arrays and manifests contain no benchmark images. Historical filenames containing `leakfree` denote removal of the specified matched training set. Real-audit counts refer to the frozen verifier's flagged sets, which can include shared acquisition layouts; fixed-test retraining measures their consequence.
