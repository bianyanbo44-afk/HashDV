"""Mutual patch correspondences followed by partial-affine RANSAC."""
import math
import numpy as np


def verify_dense(query, candidates):
    import cv2
    import torch
    query, candidates = query.float(), candidates.float()
    if query.ndim != 2 or candidates.ndim != 3 or candidates.shape[1:] != query.shape:
        raise ValueError("Expected query [patches, dimensions], candidates [N, patches, dimensions]")
    grid = math.isqrt(len(query))
    if grid * grid != len(query):
        raise ValueError("Patch layout must be a square grid")
    if not len(candidates):
        return np.empty(0, dtype=np.float32)
    with torch.inference_mode():
        similarity = torch.einsum("id,kjd->kij", query, candidates)
        q2c, c2q = similarity.argmax(2), similarity.argmax(1)
        mutual = torch.gather(c2q, 1, q2c) == torch.arange(len(query), device=query.device)[None]
        q2c, mutual = q2c.cpu().numpy(), mutual.cpu().numpy()
    xy = np.stack(np.meshgrid(np.arange(grid) + 0.5, np.arange(grid) + 0.5), -1).reshape(-1, 2).astype(np.float32)
    scores = np.zeros(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        selected = np.flatnonzero(mutual[i])
        if len(selected) < 3:
            continue
        cv2.setRNGSeed(0)
        _, inliers = cv2.estimateAffinePartial2D(xy[selected], xy[q2c[i, selected]], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5, maxIters=2000, confidence=0.99)
        scores[i] = 0 if inliers is None else float(inliers.sum()) / len(query)
    return scores
