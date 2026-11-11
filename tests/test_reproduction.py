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
from hashdv.verification.verify_control_ap import main as verify_ap, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_original_evidence_file_hashes():
    hashes = json.loads((DATA / "SHA256.json").read_text())
    assert len(hashes) == 86
    assert all(hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())


def test_icde2027_addition_hashes():
    hashes = json.loads((DATA / "SHA256_icde2027.json").read_text())
    assert hashes
    assert all(hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    original = json.loads((DATA / "SHA256.json").read_text())
    observed = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    assert observed == set(original) | set(hashes) | {"SHA256.json", "SHA256_icde2027.json"}


def test_revision_code_hashes():
    hashes = json.loads((ROOT / "SHA256_code.json").read_text())
    assert hashes
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())


@pytest.mark.parametrize("dataset,arm", [(d, a) for d in ["gc10_det", "pcb_det"] for a in "ab"])
def test_intervention_predictions_cover_all_seeds(dataset, arm):
    for seed in (2027, 3407, 8888):
        z = np.load(DATA / "intervention_predictions" / dataset / f"{arm}_{seed}.npz", allow_pickle=False)
        assert len(z["prediction_offsets"]) == len(z["image_names"]) + 1
        assert z["tp"].shape == (len(z["confidence"]), 10)


@pytest.mark.parametrize("dataset", ["neu_det", "gc10_det", "pcb_det"])
def test_fit_reproduces_paper_calibration(dataset):
    result = fit_pairs(DATA / "additional_analyses" / f"{dataset}_comparison_pairs.npz")
    expected = json.loads((ROOT / "configs" / (dataset + ".json")).read_text())
    assert result["alpha"] == expected["alpha"]
    assert result["threshold_tau"] == expected["threshold_tau"]


def test_all_saved_comparison_scores_and_control_ap():
    verify_scores(DATA)
    coverage = verify_ap(DATA)
    assert set(coverage) == {"neu_det", "gc10_det", "pcb_det"}
    assert [coverage[d]["images"] for d in ("neu_det", "gc10_det", "pcb_det")] == [
        {"L": 17, "U": 343, "T": 360},
        {"L": 57, "U": 403, "T": 460},
        {"L": 157, "U": 1894, "T": 2051},
    ]
    assert np.allclose(coverage["neu_det"]["delta_L_points"], [4.30819403, 2.53514665])
    assert np.allclose(coverage["gc10_det"]["delta_L_points"], [-5.81407919, 1.33304005])


@pytest.mark.parametrize("version", ["current", "original"])
def test_missing_intervention_files_fail_instead_of_reporting_success(tmp_path, version):
    with pytest.raises(FileNotFoundError, match="gc10_det"):
        load_dataset(tmp_path, "gc10_det", full_version=version)


def test_new_prediction_provenance_matches_released_arrays():
    records = json.loads((DATA / "intervention_predictions/provenance.json").read_text())["records"]
    assert len(records) == 12
    for record in records:
        path = DATA / record["prediction_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["prediction_sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert str(z["weight_sha256"].item()) == record["weight_sha256"]
            assert len(z["image_names"]) == record["evaluation_images"]


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
    assert training_recipe("neu_det", "full", 2027) == training_recipe("neu_det", "targeted", 2027)
    assert training_recipe("neu_det", "full", 2027, version="original")["workers"] == 0
    for dataset in ("neu_det", "gc10_det"):
        full = training_recipe(dataset, "full", 2027)
        assert full == training_recipe(dataset, "targeted", 2027)
        assert (full["workers"], full["cache"], full["plots"]) == (1, "ram", False)
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
