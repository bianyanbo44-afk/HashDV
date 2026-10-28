"""Memory-bounded top-K audit with an optional one-step training closure."""
from collections import OrderedDict
import csv
import json
from pathlib import Path
import numpy as np
from .features import Encoder, read_rgb
from .geometry import verify_dense
from .hashing import variants, reference_hash

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_list(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def audit(train_dir, test_dir, output, config, checkpoint=None, device=None, batch=8, cache_images=128):
    import torch
    if batch < 1 or cache_images < 1:
        raise ValueError("Batch and cache size must be positive")
    train_dir, test_dir, output = Path(train_dir), Path(test_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    train, test = image_list(train_dir), image_list(test_dir)
    if not train or not test:
        raise ValueError("Both training and evaluation directories need images")
    alpha, threshold, top_k = float(config["alpha"]), float(config["threshold_tau"]), int(config.get("top_k", 20))
    if not 0 <= alpha <= 1 or not np.isfinite(threshold) or top_k < 1:
        raise ValueError("Invalid alpha, threshold or top_k")
    encoder = Encoder(checkpoint, device)
    gallery = torch.cat([encoder.encode(train[i:i + batch]) for i in range(0, len(train), batch)]).to(encoder.device)
    hashes = [reference_hash(read_rgb(p)) for p in train]
    cache = OrderedDict()

    def dense_candidates(indices):
        # Hold this request's features separately so LRU eviction cannot remove an in-flight candidate.
        current = {i: cache[i] for i in indices if i in cache}
        missing = [i for i in indices if i not in current]
        for start in range(0, len(missing), batch):
            group = missing[start:start + batch]
            features = encoder.encode([train[i] for i in group], dense=True)
            current.update(zip(group, features))
        for i in indices:
            cache[i] = current[i]
            cache.move_to_end(i)
        while len(cache) > cache_images:
            cache.popitem(last=False)
        return torch.stack([current[i] for i in indices]).to(encoder.device)

    def match(path, excluded=None):
        image = read_rgb(path)
        similarity = encoder.encode([image]).to(encoder.device) @ gallery.T
        if excluded is not None:
            similarity[0, excluded] = -torch.inf
        count = min(top_k, len(train) - int(excluded is not None))
        if count < 1:
            return []
        indices = similarity[0].topk(count).indices.cpu().tolist()
        query = encoder.encode([image], dense=True)[0].to(encoder.device)
        dense = verify_dense(query, dense_candidates(indices))
        views = variants(image)
        rows = []
        for index, d in zip(indices, dense):
            h = 1 - min(v - hashes[index] for v in views) / 64
            score = alpha * h + (1 - alpha) * float(d)
            if score >= threshold:
                rows.append((index, score, h, float(d)))
        return rows

    leaked, removed, pairs = set(), set(), []
    for n, path in enumerate(test):
        hits = match(path)
        for i, score, h, d in hits:
            leaked.add(path.relative_to(test_dir).as_posix())
            removed.add(i)
            pairs.append((path.relative_to(test_dir).as_posix(), train[i].relative_to(train_dir).as_posix(), score, h, d))
        if (n + 1) % 25 == 0:
            print(f"Audited {n + 1}/{len(test)} images", flush=True)
    first_order, closure = set(removed), []
    if config.get("closure_one_step", False):
        for i in sorted(first_order):
            for j, score, h, d in match(train[i], excluded=i):
                if j not in first_order:
                    removed.add(j)
                    closure.append((train[i].relative_to(train_dir).as_posix(), train[j].relative_to(train_dir).as_posix(), score, h, d))
    output.mkdir(parents=True)
    for name, rows in [("pairs.csv", pairs), ("closure.csv", closure)]:
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_image", "training_image", "fused_score_F", "hash_H", "dense_D"])
            writer.writerows(rows)
    retained = [p.relative_to(train_dir).as_posix() for i, p in enumerate(train) if i not in removed]
    with (output / "retained_train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image"])
        writer.writerows((p,) for p in retained)
    report = {"config": config, "n_test": len(test), "n_train": len(train), "n_flagged": len(leaked),
              "n_removed": len(removed), "n_R0": len(first_order), "n_R1": len(removed - first_order),
              "L": sorted(leaked), "U": sorted(p.relative_to(test_dir).as_posix() for p in test
                                               if p.relative_to(test_dir).as_posix() not in leaked),
              "R": [train[i].relative_to(train_dir).as_posix() for i in sorted(removed)]}
    (output / "sets.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
