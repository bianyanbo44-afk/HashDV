"""The steel rows of Table VI use the execution-matched full arms in data/matched_full_arm/."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hashdv.verification.verify_control_ap import estimate, load_dataset, load_predictions  # noqa: E402

DATA = ROOT / "data"


@pytest.mark.parametrize("dataset", ["neu_det", "gc10_det"])
def test_matched_full_arm_point_estimates(dataset):
    pred, target, L, U = load_dataset(DATA, dataset)
    for seed in (2027, 3407, 8888):
        p, t, _ = load_predictions(DATA / "matched_full_arm" / dataset / f"a_matched_{seed}.npz")
        assert all(np.array_equal(a, b) for a, b in zip(t, target))
        pred[("a", seed)] = p
        done = json.loads((DATA / "matched_full_arm" / dataset / "training" / f"s{seed}_done.json").read_text())
        with np.load(DATA / "matched_full_arm" / dataset / f"a_matched_{seed}.npz", allow_pickle=False) as z:
            assert str(z["weight_sha256"].item()) == done["weight_sha256"]
        assert (done["workers"], done["cache"], done["plots"]) == (1, "ram", False)
    mean, _ = estimate(pred, target, L, U)
    ref = json.loads((DATA / "additional_analyses" / f"{dataset}_matched_full_arm.json").read_text())["matched_full"]
    for key in ("delta_L", "delta_T"):
        assert np.allclose(mean[key], [ref[key]["map50"], ref[key]["map50_95"]], rtol=0, atol=1e-9)
