"""Seed-aware uncertainty for the DsPCBSD+ full-versus-removed contrast from saved predictions.

Reuses the AP50 definition of experiments/intervention_revision_sensitivity.py. Reports the point estimates, each
paired seed, intervals that resample evaluation images with the trained models fixed, and intervals that resample the
three paired seeds jointly with the images.

Usage: python experiments/pcb_seed_sensitivity.py [--data data] [--output ...]
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

from intervention_revision_sensitivity import ap50

SEEDS = (2027, 3407, 8888)


def main(data, output, n=1000):
    preds, targets, names_ref = {}, None, None
    for arm in ("a", "b"):
        for seed in SEEDS:
            with np.load(data / "intervention_predictions/pcb_det" / f"{arm}_{seed}.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            names = arrays["image_names"].tolist()
            po, go = arrays["prediction_offsets"], arrays["target_offsets"]
            preds[(arm, seed)] = [(arrays["tp"][po[i]:po[i + 1], 0].astype(float), arrays["confidence"][po[i]:po[i + 1]],
                                   arrays["prediction_class"][po[i]:po[i + 1]]) for i in range(len(names))]
            t = [arrays["target_class"][go[i]:go[i + 1]] for i in range(len(names))]
            if targets is None:
                targets, names_ref = t, names
            else:
                assert names == names_ref
    with (data / "leak_lists/dspcbsd_official_copy_pairs.csv").open(encoding="utf-8") as stream:
        flagged = {r["validation_image"] for r in csv.DictReader(stream)}
    member = np.array([n in flagged for n in names_ref])
    L, U = np.flatnonzero(member), np.flatnonzero(~member)
    T = np.arange(len(names_ref))

    def estimate(li, ui, seeds=SEEDS):
        ti = np.concatenate([li, ui])
        vals = []
        for s in seeds:
            a = [ap50(preds[("a", s)], targets, idx) for idx in (li, ui, ti)]
            b = [ap50(preds[("b", s)], targets, idx) for idx in (li, ui, ti)]
            vals.append([(a[0] - b[0]) - (a[1] - b[1]), (a[2] - b[2]) - (a[1] - b[1])])
        return np.mean(vals, axis=0) * 100

    def summary(values):
        v = np.asarray(values)
        return {"resamples": len(v), "ci95": dict(zip(["delta_L", "delta_T"], np.quantile(v, [.025, .975], axis=0).T.tolist()))}

    result = {"endpoint": "mAP50 points", "n_L": int(len(L)), "n_U": int(len(U)),
              "point": dict(zip(["delta_L", "delta_T"], estimate(L, U).tolist())),
              "per_seed": {str(s): dict(zip(["delta_L", "delta_T"], estimate(L, U, [s]).tolist())) for s in SEEDS},
              "analysis_type": "Post-review seed-aware sensitivity from saved predictions; no retraining."}
    rng = np.random.default_rng(11112040)
    result["stratified_image_fixed_seeds"] = summary(
        [estimate(rng.choice(L, len(L)), rng.choice(U, len(U))) for _ in range(n)])
    rng = np.random.default_rng(11112041)
    result["paired_seed_and_image"] = summary(
        [estimate(rng.choice(L, len(L)), rng.choice(U, len(U)), rng.choice(SEEDS, 3, replace=True).tolist())
         for _ in range(n)])
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--data", type=Path, default=root / "data")
    p.add_argument("--output", type=Path, default=root / "data/additional_analyses/pcb_seed_sensitivity.json")
    p.add_argument("--resamples", type=int, default=1000)
    a = p.parse_args()
    main(a.data, a.output, a.resamples)
