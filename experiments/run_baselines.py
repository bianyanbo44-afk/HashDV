"""Fixed supplementary SSCD/SIFT baselines and retrieval-aware detection.

Run with the inference dependencies and CUDA. Published data remain read-only.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from hashdv import features as lb
import common as la
from common import context, OUT
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ('neu_det', 'gc10_det', 'pcb_det')


@torch.inference_mode()
def embeddings(model, paths, transform, bs=32):
    arr = []
    for i in range(0, len(paths), bs):
        x = torch.stack([transform(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).cuda()
        arr.append(model(x).float().cpu().numpy())
    return np.concatenate(arr)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def summarize(score, p, role):
    test = role == 'test'
    return la.metrics(score[test], p['label'][test].astype(int), p['query'][test])


def sscd():
    torch.set_num_threads(4)
    weights = ROOT / 'weights/sscd_disc_mixup.torchscript.pt'
    model = torch.jit.load(str(weights), map_location='cuda').eval()
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    variants = {'square320': transforms.Resize([320, 320]), 'short288': transforms.Resize(288)}
    res = {'model_url': 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt',
           'model_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'per_dataset': {}}
    for ds in DATASETS:
        meta, p, files, role = context(ds)
        d = {}
        saved = {}
        for name, resize in variants.items():
            cache = OUT / f'{ds}_sscd_{name}_features.npz'
            if cache.exists():
                a = np.load(cache); g, q = a['gallery'], a['queries']
            else:
                tr = transforms.Compose([resize, transforms.ToTensor(), norm])
                # Aspect-preserving resizing can give different shapes, so group by input dimensions.
                if name == 'short288':
                    def grouped(paths):
                        groups = {}
                        for j, path in enumerate(paths):
                            with Image.open(path) as im: key = im.size
                            groups.setdefault(key, []).append((j, path))
                        out = np.empty((len(paths), 512), np.float32)
                        for vals in groups.values():
                            out[[j for j, _ in vals]] = embeddings(model, [p for _, p in vals], tr)
                        return out
                    g, q = grouped(files), grouped([x['path'] for x in meta['queries']])
                else:
                    g = embeddings(model, files, tr)
                    q = embeddings(model, [x['path'] for x in meta['queries']], tr)
                np.savez_compressed(cache, gallery=g, queries=q)
            for centered in [False, True]:
                mu = g.mean(0, keepdims=True) if centered else 0
                gn, qn = unit(g - mu), unit(q - mu)
                s = np.einsum('ij,ij->i', qn[p['query']], gn[p['cand']])
                key = name + ('_centered' if centered else '')
                saved[key] = s
                d[key] = summarize(s, p, role)
                print(ds, key, d[key], flush=True)
        np.savez_compressed(OUT / f'{ds}_sscd_pair_scores.npz', **saved)
        res['per_dataset'][ds] = d
        (OUT / 'sscd_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')


def sift_features(path):
    im = lb.read_rgb(path)
    gray = cv2.cvtColor(cv2.resize(im, (448, 448)), cv2.COLOR_RGB2GRAY)
    k, d = cv2.SIFT_create(nfeatures=1024).detectAndCompute(gray, None)
    xy = np.array([x.pt for x in k], np.float32).reshape(-1, 2)
    return xy, d


def sift_pair(a, b):
    xy, d = a; uv, e = b
    if d is None or e is None or len(d) < 3 or len(e) < 3: return 0.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    def good(x, y):
        return {m.queryIdx: m.trainIdx for pair in bf.knnMatch(x, y, k=2) if len(pair) == 2
                for m, n in [pair] if m.distance < .75 * n.distance}
    f, rev = good(d, e), good(e, d)
    pairs = [(i, j) for i, j in f.items() if rev.get(j) == i]
    if len(pairs) < 3: return 0.
    u = np.array([xy[i] for i, _ in pairs]); v = np.array([uv[j] for _, j in pairs])
    cv2.setRNGSeed(0)
    _, mask = cv2.estimateAffinePartial2D(u, v, method=cv2.RANSAC,
        ransacReprojThreshold=448 * 1.5 / 32, maxIters=2000, confidence=.99)
    return float(mask.sum()) / 1024 if mask is not None else 0.


def sift():
    cv2.setNumThreads(1)
    res = {}
    for ds in DATASETS:
        dest = OUT / f'{ds}_sift_pair_scores.npy'
        meta, p, files, role = context(ds)
        if dest.exists(): s = np.load(dest)
        else:
            with ThreadPoolExecutor(max_workers=6) as ex:
                gal = list(ex.map(sift_features, files))
                qs = list(ex.map(sift_features, [q['path'] for q in meta['queries']]))
                print(ds, 'SIFT features ready', flush=True)
                jobs = [(qs[int(q)], gal[int(c)]) for q, c in zip(p['query'], p['cand'])]
                s = np.fromiter(ex.map(lambda x: sift_pair(*x), jobs), dtype=np.float64, count=len(jobs))
            np.save(dest, s)
        res[ds] = summarize(s, p, role)
        print(ds, 'SIFT', res[ds], flush=True)
        (OUT / 'sift_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')


def end_to_end():
    fitted = json.loads((ROOT / 'data/source_data/tables_and_text/l1b_results.json').read_text(encoding='utf-8'))['per_dataset']
    res = {}
    for ds in DATASETS:
        meta, p, files, role = context(ds)
        y, q = p['label'].astype(int), p['query']
        alpha = fitted[ds]['alpha_B1']
        s = alpha * p['H'] + (1 - alpha) * p['D_B1']
        # Stored groups consist of 20 retrieved rows followed by an oracle source only if absent.
        counts = np.bincount(q, minlength=len(meta['queries']))
        assert np.all(np.isin(counts, [20, 21]))
        recovered = counts[q] == 20
        endpoint = s.copy().astype(float)
        endpoint[(y == 1) & ~recovered] = -1.
        test = role == 'test'
        pos = test & (y == 1)
        tqs = np.array([x['role'] == 'test' for x in meta['queries']])
        d = {'n_test_queries': int(tqs.sum()), 'retrieval_recall20': float((counts[tqs] == 20).mean()),
             'verification_tpr1': la.tpr_at(s[test], y[test]),
             'retrieval_aware_tpr1': la.tpr_at(endpoint[test], y[test]),
             'by_severity': {}}
        neg_scores = s[test & (y == 0)]
        for sev in range(1, 6):
            m = pos & np.array([meta['queries'][int(i)]['sev'] == sev for i in q])
            vals = np.r_[endpoint[m], neg_scores]
            labels = np.r_[np.ones(m.sum()), np.zeros(len(neg_scores))]
            d['by_severity'][sev] = la.tpr_at(vals, labels)
        tau = fitted[ds]['audit_threshold_B1']
        test_ids = np.flatnonzero(tqs)
        d['audit_threshold'] = tau
        d['test_query_false_match_fraction'] = float(np.mean([np.any(s[(q == i) & (y == 0)] >= tau) for i in test_ids]))
        d['source_detected_at_fitted_audit_threshold'] = float(np.mean(endpoint[pos] >= tau))
        res[ds] = d
        print(ds, 'retrieval-aware', d, flush=True)
    (OUT / 'end_to_end_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(); ap.add_argument('task', choices=['sscd', 'sift', 'end_to_end']); args = ap.parse_args()
    start = time.time(); globals()[args.task](); print('seconds', round(time.time() - start, 1), flush=True)
