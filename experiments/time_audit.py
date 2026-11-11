"""Stage-timed rerun of the released audit (same logic and defaults as hashdv/audit.py: batch 8, feature cache 128).

Records wall-clock time per stage, the number of verified pairs and the flagged counts for NEU-DET, GC10-DET and the
official DsPCBSD+ split, run one after another on one GPU (Section V-A of the ICDE 2027 submission).

Usage:
  python experiments/time_audit.py --inputs configs/inputs.json \
      --checkpoint weights/dinov2_vitb14_raw.safetensors --output outputs/audit_timing.json

`--inputs` maps each benchmark to an `image_root`; images are found by file name directly under it or under a
sub-folder named after their split (`val` for the DsPCBSD+ validation split). See configs/inputs.example.json.
"""
import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch

from hashdv.features import Encoder, read_rgb
from hashdv.geometry import verify_dense
from hashdv.hashing import variants, reference_hash

REPO = Path(__file__).resolve().parents[1]
SPLITS = {"neu_det": ("neu_det_split.csv", "train", "test"),
          "gc10_det": ("gc10_det_split.csv", "train", "test"),
          "pcb_det": ("dspcbsd_official_split.csv", "train", "validation")}
FOLDER = {"validation": "val"}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def files(name, root):
    split_csv, tr, te = SPLITS[name]
    rows = list(csv.DictReader(open(REPO / "data" / "splits" / split_csv, encoding="utf-8")))

    def path(image, split):
        direct = root / image
        return direct if direct.exists() else root / FOLDER.get(split, split) / image
    train = [path(r["image"], tr) for r in rows if r["split"] == tr]
    test = [path(r["image"], te) for r in rows if r["split"] == te]
    missing = [p for p in train + test if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} images missing, e.g. {missing[0]}")
    return train, test


def run(name, root, encoder, batch=8, cache_images=128):
    cfg = json.loads((REPO / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    alpha, tau, top_k = float(cfg["alpha"]), float(cfg["threshold_tau"]), int(cfg.get("top_k", 20))
    train, test = files(name, root)
    t = dict.fromkeys(["embed_train", "hash_train", "query_read", "retrieve", "dense_encode", "verify",
                       "hash_compare"], 0.0)
    sync(); s = time.perf_counter()
    gallery = torch.cat([encoder.encode(train[i:i + batch]) for i in range(0, len(train), batch)]).to(encoder.device)
    sync(); t["embed_train"] = time.perf_counter() - s
    s = time.perf_counter()
    hashes = [reference_hash(read_rgb(p)) for p in train]
    t["hash_train"] = time.perf_counter() - s
    cache, counts = OrderedDict(), {"verified_pairs": 0, "dense_encoded_train": 0}

    def dense_candidates(indices):
        current = {i: cache[i] for i in indices if i in cache}
        missing = [i for i in indices if i not in current]
        for start in range(0, len(missing), batch):
            group = missing[start:start + batch]
            current.update(zip(group, encoder.encode([train[i] for i in group], dense=True)))
        counts["dense_encoded_train"] += len(missing)
        for i in indices:
            cache[i] = current[i]
            cache.move_to_end(i)
        while len(cache) > cache_images:
            cache.popitem(last=False)
        return torch.stack([current[i] for i in indices]).to(encoder.device)

    def match(path, excluded=None):
        s0 = time.perf_counter()
        image = read_rgb(path)
        t["query_read"] += time.perf_counter() - s0
        sync(); s1 = time.perf_counter()
        similarity = encoder.encode([image]).to(encoder.device) @ gallery.T
        if excluded is not None:
            similarity[0, excluded] = -torch.inf
        count = min(top_k, len(train) - int(excluded is not None))
        indices = similarity[0].topk(count).indices.cpu().tolist()
        sync(); s2 = time.perf_counter(); t["retrieve"] += s2 - s1
        query = encoder.encode([image], dense=True)[0].to(encoder.device)
        candidates = dense_candidates(indices)
        sync(); s3 = time.perf_counter(); t["dense_encode"] += s3 - s2
        dense = verify_dense(query, candidates)
        sync(); s4 = time.perf_counter(); t["verify"] += s4 - s3
        views = variants(image)
        hits = []
        for index, d in zip(indices, dense):
            h = 1 - min(v - hashes[index] for v in views) / 64
            if alpha * h + (1 - alpha) * float(d) >= tau:
                hits.append(index)
        t["hash_compare"] += time.perf_counter() - s4
        counts["verified_pairs"] += len(indices)
        return hits

    sync(); s = time.perf_counter()
    flagged, removed = 0, set()
    for path in test:
        hits = match(path)
        flagged += bool(hits)
        removed.update(hits)
    sync(); loop = time.perf_counter() - s
    closure, first = 0.0, set(removed)
    if cfg.get("closure_one_step", False):
        sync(); s = time.perf_counter()
        for i in sorted(first):
            removed.update(j for j in match(train[i], excluded=i) if j not in first)
        sync(); closure = time.perf_counter() - s
    return {"benchmark": name, "n_train": len(train), "n_test": len(test), "n_flagged": flagged, "n_R0": len(first),
            "n_R": len(removed), "stage_seconds": t, "query_loop_seconds": loop, "closure_seconds": closure,
            "total_seconds": t["embed_train"] + t["hash_train"] + loop + closure, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    sync(); s = time.perf_counter()
    encoder = Encoder(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    sync()
    result = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "model_load_seconds": time.perf_counter() - s, "batch": 8,
              "cache_images": 128, "runs": []}
    for name in SPLITS:
        result["runs"].append(run(name, Path(inputs[name]["image_root"]), encoder))
        args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(result["runs"][-1]), flush=True)


if __name__ == "__main__":
    main()
