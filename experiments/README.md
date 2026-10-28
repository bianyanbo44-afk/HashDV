# Baseline inference and score calibration

`run_baselines.py`, `sscd_postprocessing.py`, and `matched_calibration.py` retain the original scoring, SIFT and normalization computations. Their local-machine paths have been replaced by the published candidate arrays, split manifests, and `configs/inputs.example.json`.

1. Obtain the benchmark images and generate the confirmation copies with `scripts/generate_copies.py`.
2. Copy `configs/inputs.example.json` to a local JSON file and set the `HASHDV_INPUTS` environment variable to its path, or use its default relative paths. Each image tree must have unique basenames.
3. Put the official `sscd_disc_mixup.torchscript.pt` in `weights/`. The model URL and expected SHA256 are in `data/additional_analyses/sscd_results.json`.
4. Run with the inference dependencies and a CUDA-capable PyTorch installation:

```bash
python experiments/run_baselines.py sscd
python experiments/run_baselines.py sift
python experiments/sscd_postprocessing.py
python experiments/matched_calibration.py
```

These commands write to `outputs/baselines/`, leaving the paper's saved results in `data/` intact. Full-background score normalization uses only the original synthetic benchmark training split. Shared calibration uses the same eligible retrieved background candidates for all compared scores.

The following commands also work directly from the saved arrays, without model inference or original images:

```bash
python experiments/run_baselines.py end_to_end
python experiments/matched_calibration.py
```

If no newly inferred SSCD pair-score file is present, shared calibration reads the published SSCD scores. Published primary fusion weights remain fixed. The feature and pair caches under `outputs/` should be kept together for a single input configuration.

## HashDV confirmation scores

After regenerating copies and setting the input paths:

```bash
python experiments/score_confirmation.py --dataset neu_det \
  --checkpoint weights/dinov2_vitb14_raw.safetensors --output outputs/neu_scores.npz
python -m hashdv calibrate outputs/neu_scores.npz
```

Add `--convnext-checkpoint weights/convnext_tiny.safetensors` for the second backbone, the timm `convnext_tiny.in12k_ft_in1k` stage-3 patch grid. Gallery dense features stay in host memory and selected candidates move to the inference device; a full benchmark uses several GB of RAM. Candidate indices are the original frozen lists. Fresh retrieval is exercised by `hashdv audit`.
