"""Checks against the published arrays and interventions, independent of GPU inference."""
from collections import Counter
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from hashdv.calibration import fit, fit_pairs
from hashdv.intervention import manifest, training_recipe
from hashdv.verification.verify_saved_results import main as verify_scores
from hashdv.verification.verify_control_ap import main as verify_ap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_original_evidence_file_hashes():
    hashes = json.loads((DATA / "SHA256.json").read_text())
    assert len(hashes) == 86
    assert all(hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())


@pytest.mark.parametrize("dataset", ["neu_det", "gc10_det", "pcb_det"])
def test_fit_reproduces_paper_calibration(dataset):
    result = fit_pairs(DATA / "additional_analyses" / f"{dataset}_comparison_pairs.npz")
    expected = json.loads((ROOT / "configs" / (dataset + ".json")).read_text())
    assert result["alpha"] == expected["alpha"]
    assert result["threshold_tau"] == expected["threshold_tau"]


def test_all_saved_comparison_scores_and_control_ap():
    verify_scores(DATA)
    verify_ap(DATA)


@pytest.mark.parametrize("dataset,count", [("neu_det", 1063), ("gc10_det", 1315), ("pcb_det", 7931)])
def test_corrected_split_counts(dataset, count):
    rows, _, _ = manifest(DATA, dataset)
    assert sum(r["split"] == "train" for r in rows) == count


@pytest.mark.parametrize("seed", [2027, 3407, 8888])
def test_random_deletion_matches_classes_and_retains_targeted_images(seed):
    _, targeted, _ = manifest(DATA, "neu_det", "targeted")
    rows, removed, _ = manifest(DATA, "neu_det", "random", seed)
    category = lambda names: Counter(Path(n).stem.rsplit("_", 1)[0] for n in names)
    assert category(removed) == category(targeted)
    assert not set(removed) & set(targeted)
    assert sum(r["split"] == "train" for r in rows) == 1063


def test_uninformative_cues_choose_smallest_weight_on_tie():
    result = fit(np.zeros(4), np.zeros(4), [0, 1, 0, 1])
    assert result["alpha"] == 0


def test_checkpoint_and_runtime_pairing_rules():
    assert training_recipe("neu_det", "random", 2027) == training_recipe("neu_det", "targeted", 2027)
    assert training_recipe("neu_det", "full", 2027)["workers"] == 0
    assert training_recipe("pcb_det", "full", 2027) == training_recipe("pcb_det", "targeted", 2027)
    assert training_recipe("pcb_det", "full", 2027)["val"] is False


def test_shared_calibration_recomputes_original_scores():
    subprocess.run([sys.executable, str(ROOT / "experiments/matched_calibration.py")], cwd=ROOT,
                   check=True, capture_output=True)
    expected = json.loads((DATA / "additional_analyses/matched_calibration.json").read_text())
    got = json.loads((ROOT / "outputs/baselines/matched_calibration.json").read_text())
    for dataset in expected:
        for method in expected[dataset]:
            for metric in expected[dataset][method]:
                assert abs(got[dataset][method][metric] - expected[dataset][method][metric]) < 1e-12
