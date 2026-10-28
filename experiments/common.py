"""Portable adapters for the original baseline scripts and frozen candidate lists."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from hashdv.calibration import metrics as pair_metrics, roc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/baselines"


def inputs(dataset):
    config = Path(os.environ.get("HASHDV_INPUTS", ROOT / "configs/inputs.example.json"))
    record = json.loads(config.read_text(encoding="utf-8"))[dataset]
    return {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in record.items()}


def context(dataset):
    folder = ROOT / "data/additional_analyses"
    meta = json.loads((folder / f"{dataset}_pair_index.json").read_text(encoding="utf-8"))
    paths = inputs(dataset)
    from hashdv.intervention import index_files
    image_index = index_files(paths["image_root"], {".jpg", ".jpeg", ".png", ".bmp"})
    files = [image_index.get(n, paths["image_root"] / n) for n in meta["gallery"]]
    for query in meta["queries"]:
        query["path"] = str(paths["copy_root"] / query["copy_file"])
    with np.load(folder / f"{dataset}_comparison_pairs.npz", allow_pickle=False) as saved:
        pairs = {k: saved[k].copy() for k in saved.files}
    return meta, pairs, files, pairs["role"]


def background_indices(meta, dataset):
    filename = ("dspcbsd_subset1800_for_synthetic_copies.csv" if dataset == "pcb_det" else dataset + "_split.csv")
    with (ROOT / "data/splits" / filename).open(encoding="utf-8", newline="") as stream:
        names = {r["image"] for r in csv.DictReader(stream) if r["split"] == "train"}
    return np.array([i for i, name in enumerate(meta["gallery"]) if name in names])


def tpr_at(scores, labels, fpr=.01):
    x, y = roc(scores, labels)
    return float(np.interp(fpr, x, y))


def metrics(scores, labels, queries):
    result = pair_metrics(scores, labels)
    matches = []
    for query in np.unique(queries):
        selected = queries == query
        positive, negative = scores[selected & (labels == 1)], scores[selected & (labels == 0)]
        matches.append(bool(len(positive)) and (not len(negative) or positive.max() > negative.max()))
    result["rank1"] = float(np.mean(matches))
    return result
