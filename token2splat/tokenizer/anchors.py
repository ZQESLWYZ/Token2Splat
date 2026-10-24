import numpy as np

import torch

from ..features import mat_to_quat


def lex_sort_indices(points):
    return np.lexsort((points[:, 2], points[:, 1], points[:, 0]))


def farthest_point_sampling(points, num_points):
    count = points.shape[0]
    start = int(lex_sort_indices(points)[0])
    selected = [start]
    best = np.sum((points - points[start]) ** 2, axis=1)
    for _ in range(num_points - 1):
        farthest = best.max()
        candidates = np.flatnonzero(best >= farthest - 1e-12)
        if candidates.size == 1:
            pick = int(candidates[0])
        else:
            ordered = lex_sort_indices(points[candidates])
            pick = int(candidates[ordered[0]])
        selected.append(pick)
        best = np.minimum(best, np.sum((points - points[pick]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


def lattice_pad(num_points, step):
    out = []
    level = 0
    while len(out) < num_points:
        side = 2 * level + 1
        for ix in range(side):
            for iy in range(side):
                for iz in range(side):
                    if level == 0 and ix == iy == iz == level:
                        continue
                    if len(out) >= num_points:
                        break
                    out.append([(ix - level) * step, (iy - level) * step, (iz - level) * step])
        level += 1
    return np.asarray(out[:num_points], dtype=np.float64)


def build_anchor_bank(centers, num_anchors, pool_size, pad_step):
    pts = np.asarray(centers, dtype=np.float64)
    if pool_size > 0 and pts.shape[0] > pool_size:
        index = np.linspace(0, pts.shape[0] - 1, pool_size).round().astype(np.int64)
        pts = pts[index]
    if pts.shape[0] < num_anchors:
        pad = lattice_pad(num_anchors - pts.shape[0], pad_step)
        pts = np.concatenate([pts, pad], axis=0)
    order = farthest_point_sampling(pts, num_anchors)
    return pts[order].astype(np.float32)


def patch_frame(points, eta_min):
    mu = points.mean(axis=0)
    local = points - mu
    cov = (local.T @ local) / max(points.shape[0], 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.from_numpy(cov).unsqueeze(0).double())
    frame = eigenvectors.flip(-1).squeeze(0).numpy()
    for col in (0, 1):
        lead = int(np.argmax(np.abs(frame[:, col])))
        if frame[lead, col] < 0:
            frame[:, col] = -frame[:, col]
    frame[:, 2] = np.cross(frame[:, 0], frame[:, 1])
    extent = float(np.sqrt(max(np.mean(np.sum(local ** 2, axis=1)), 0.0)))
    extent = max(extent, eta_min)
    quat = mat_to_quat(torch.from_numpy(frame).unsqueeze(0).double()).squeeze(0).float().numpy()
    return mu.astype(np.float32), frame.astype(np.float32), np.float32(extent), quat.astype(np.float32)
