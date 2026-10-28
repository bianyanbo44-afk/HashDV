"""Recompute HashDV scores on the published, frozen confirmation candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import context
from hashdv.features import Encoder, read_rgb
from hashdv.geometry import verify_dense
from hashdv.hashing import variants, reference_hash


def convnext_encoder(checkpoint, device):
    import torch
    import timm
    from safetensors.torch import load_file
    model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model = model.to(device).eval()

    def encode(paths):
        import cv2
        x = np.stack([cv2.resize(read_rgb(p), (448, 448), interpolation=cv2.INTER_LINEAR) for p in paths])
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255
        mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            x = model.stages[2](model.stages[1](model.stages[0](model.stem((x - mean) / std))))
        return torch.nn.functional.normalize(x.float().flatten(2).transpose(1, 2), dim=2).half().cpu()
    return encode


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["neu_det", "gc10_det", "pcb_det"], required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--convnext-checkpoint", type=Path)
    p.add_argument("--device")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new score file")
    if a.batch < 1:
        raise ValueError("Batch must be positive")
    meta, saved, files, roles = context(a.dataset)
    encoder = Encoder(a.checkpoint, a.device)
    query_paths = [Path(q["path"]) for q in meta["queries"]]
    result = {k: saved[k].copy() for k in ["query", "cand", "label", "role"]}
    gallery_global = torch.cat([encoder.encode(files[i:i+a.batch]) for i in range(0, len(files), a.batch)]).to(encoder.device)
    query_global = torch.cat([encoder.encode(query_paths[i:i+a.batch]) for i in range(0, len(query_paths), a.batch)]).to(encoder.device)
    result["G"] = np.zeros(len(roles), np.float32)
    for start in range(0, len(query_paths), a.batch):
        similarity = query_global[start:start+a.batch] @ gallery_global.T
        selected = (result["query"] >= start) & (result["query"] < start+a.batch)
        result["G"][selected] = similarity[result["query"][selected]-start, result["cand"][selected]].cpu().numpy()
    hashes = [reference_hash(read_rgb(f)) for f in files]
    hash_scores = np.zeros(len(roles), np.float32)
    for i, path in enumerate(query_paths):
        views = variants(read_rgb(path))
        for row in np.flatnonzero(result["query"] == i):
            hash_scores[row] = 1 - min(v - hashes[int(result["cand"][row])] for v in views) / 64
    result["H"] = hash_scores
    encoders = [("D_B1", lambda paths: encoder.encode(paths, dense=True))]
    if a.convnext_checkpoint:
        encoders.append(("D_B2", convnext_encoder(a.convnext_checkpoint, encoder.device)))
    for name, encode in encoders:
        gallery_dense = torch.cat([encode(files[i:i+a.batch]) for i in range(0, len(files), a.batch)])
        scores = np.zeros(len(roles), np.float32)
        for start in range(0, len(query_paths), a.batch):
            query_dense = encode(query_paths[start:start+a.batch])
            for offset, query in enumerate(query_dense):
                rows = np.flatnonzero(result["query"] == start + offset)
                scores[rows] = verify_dense(query.to(encoder.device), gallery_dense[result["cand"][rows]].to(encoder.device))
        result[name] = scores
        del gallery_dense
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **result)
    print(json.dumps({"dataset": a.dataset, "queries": len(query_paths), "pairs": len(roles),
                      "frozen_candidates": True, "backbones": [k for k, _ in encoders]}, indent=2))


if __name__ == "__main__":
    main()
