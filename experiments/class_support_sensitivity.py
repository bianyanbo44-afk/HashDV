"""Audit macro-AP class support and condition image resampling on observed classes.

Uses the current execution-matched full arms and the saved removal predictions.
The sensitivity keeps every original class represented within each resampled
subset. Its intervals are conditional on that class-support event and on the
trained seed pairs. The main-table image bootstrap remains separately available.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hashdv.verification.verify_control_ap import (
    SEEDS, estimate, load_dataset,
)


def analyse(data, dataset, resamples):
    predictions, targets, local, other = load_dataset(data, dataset, full_version="current")
    masks = np.array([
        sum(1 << int(cls) for cls in np.unique(target)) for target in targets
    ], dtype=np.uint32)
    required_local = np.bitwise_or.reduce(masks[local])
    required_other = np.bitwise_or.reduce(masks[other])
    local_classes = np.unique(np.concatenate([targets[index] for index in local]))
    other_classes = np.unique(np.concatenate([targets[index] for index in other]))
    result = {
        "dataset": dataset,
        "n_L": len(local),
        "n_U": len(other),
        "class_support": {},
    }
    for name, indices, classes in (("L", local, local_classes), ("U", other, other_classes)):
        truth = np.concatenate([targets[index] for index in indices])
        result["class_support"][name] = {
            "classes": classes.tolist(),
            "images_by_class": {
                str(int(cls)): sum(int(cls in targets[index]) for index in indices)
                for cls in classes
            },
            "boxes_by_class": {str(int(cls)): int(np.count_nonzero(truth == cls)) for cls in classes},
        }

    original_rng = np.random.default_rng(0)
    missing_local = 0
    missing_other = 0
    for _ in range(resamples):
        drawn_local = original_rng.choice(local, len(local))
        drawn_other = original_rng.choice(other, len(other))
        missing_local += int(np.bitwise_or.reduce(masks[drawn_local]) != required_local)
        missing_other += int(np.bitwise_or.reduce(masks[drawn_other]) != required_other)
    result["main_bootstrap_support_diagnostic"] = {
        "rng_seed": 0,
        "draw_order": "L then U",
        "draws": resamples,
        "draws_missing_an_L_class": missing_local,
        "draws_missing_a_U_class": missing_other,
    }

    common_classes = np.intersect1d(local_classes, other_classes)
    common_targets = [target[np.isin(target, common_classes)] for target in targets]
    common, _ = estimate(predictions, common_targets, local, other)
    result["common_class_comparison"] = {
        "classes": common_classes.tolist(),
        "image_selection": "All original L and U images retained; macro-AP restricted to shared classes",
        "map50_and_map50_95_points": {key: (100 * value).tolist() for key, value in common.items()},
        "gap_full_points": (100 * (common["a_L"] - common["a_U"])).tolist(),
        "gap_removed_points": (100 * (common["b_L"] - common["b_U"])).tolist(),
    }

    primary_predictions = {
        key: [(hit[:, :1], confidence, cls) for hit, confidence, cls in values]
        for key, values in predictions.items()
    }
    rng = np.random.default_rng(11122026)
    samples = []
    attempts = 0
    while len(samples) < resamples:
        drawn_local = rng.choice(local, len(local))
        drawn_other = rng.choice(other, len(other))
        attempts += 1
        if (np.bitwise_or.reduce(masks[drawn_local]) != required_local
                or np.bitwise_or.reduce(masks[drawn_other]) != required_other):
            continue
        value, _ = estimate(primary_predictions, targets, drawn_local, drawn_other)
        samples.append([value["delta_L"][0], value["delta_T"][0]])
        if len(samples) % 200 == 0:
            print(dataset, "class-support-conditioned bootstrap", len(samples), flush=True)
    interval = 100 * np.quantile(samples, [0.025, 0.975], axis=0)
    result["class_support_conditioned_image_bootstrap"] = {
        "endpoint": "mAP50 points",
        "rng_seed": 11122026,
        "accepted_resamples": resamples,
        "attempted_resamples": attempts,
        "conditioning": "Every class observed in each original subset is present in that subset's resample",
        "trained_seeds": list(SEEDS),
        "delta_L_ci95": interval[:, 0].tolist(),
        "delta_T_ci95": interval[:, 1].tolist(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    output = args.output or args.data / "additional_analyses/class_support_sensitivity.json"
    result = {
        "analysis_date": "2026-11-12",
        "analysis_type": "Post hoc class-support sensitivity from saved execution-matched predictions",
        "ap_code": "hashdv/verification/verify_control_ap.py",
        "datasets": {
            dataset: analyse(args.data, dataset, args.resamples)
            for dataset in ("neu_det", "gc10_det", "pcb_det")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
