import os

import numpy as np

from scipy import linalg

import torch
import torch.nn.functional as F

from ..features import normalize_object
from ..render.splat import render_object


def _gaussian_kernel(size, sigma, channels, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    weights = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    kernel = weights.view(-1, 1) * weights.view(1, -1)
    return kernel.view(1, 1, size, size).expand(channels, 1, size, size).contiguous()


def ssim(image_a, image_b, data_range=1.0, size=11, sigma=1.5):
    a, b = image_a, image_b
    if a.dim() == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    channels = a.shape[1]
    kernel = _gaussian_kernel(size, sigma, channels, a.device, a.dtype)
    pad = size // 2
    mu1 = F.conv2d(a, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(b, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1 = F.conv2d(a * a, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2 = F.conv2d(b * b, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(a * b, kernel, padding=pad, groups=channels) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return score.mean()


def sample_points(obj, num_points):
    centers = torch.as_tensor(obj["centers"], dtype=torch.float32)
    if centers.dim() != 2:
        centers = centers.reshape(-1, 3)
    if centers.shape[0] <= num_points:
        return centers
    index = torch.linspace(0, centers.shape[0] - 1, num_points).round().long()
    return centers[index]


def chamfer_squared(set_a, set_b):
    dist = torch.cdist(set_a, set_b) ** 2
    forward = dist.min(dim=1).values.mean()
    backward = dist.min(dim=0).values.mean()
    return forward + backward


def pairwise_chamfer(sets_a, sets_b):
    return torch.stack([torch.stack([chamfer_squared(a, b) for b in sets_b]) for a in sets_a])


def f_score(set_a, set_b, tau=0.01):
    dist = torch.cdist(set_a, set_b)
    precision = (dist.min(dim=1).values <= tau).to(torch.float32).mean()
    recall = (dist.min(dim=0).values <= tau).to(torch.float32).mean()
    if precision + recall <= 0:
        return torch.tensor(0.0)
    return 2.0 * precision * recall / (precision + recall)


def mmd_coverage(dist_gen_to_test, tau=None):
    nearest = dist_gen_to_test.min(dim=0).values
    mmd = nearest.mean()
    if tau is None:
        matched = int(dist_gen_to_test.argmin(dim=1).unique().numel())
        coverage = torch.tensor(matched / float(dist_gen_to_test.shape[1]))
    else:
        coverage = (nearest <= float(tau)).to(torch.float32).mean()
    return mmd, coverage


def leave_one_out_knn_accuracy(sets, labels):
    matrix = pairwise_chamfer(sets, sets)
    matrix.fill_diagonal_(float("inf"))
    predicted = matrix.argmin(dim=1)
    correct = sum(int(labels[int(predicted[index])] == labels[index]) for index in range(len(sets)))
    return correct / max(len(sets), 1)


def mean_pairwise_distance(sets):
    if len(sets) < 2:
        return torch.tensor(0.0)
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            values.append(chamfer_squared(sets[i], sets[j]))
    return torch.stack(values).mean()


def uniqueness_ratio(sets, threshold):
    if len(sets) < 2:
        return torch.tensor(0.0)
    matrix = pairwise_chamfer(sets, sets)
    matrix.fill_diagonal_(float("inf"))
    nearest = matrix.min(dim=1).values
    return (nearest > threshold).to(torch.float32).mean()


def nearest_neighbor_threshold(sets, percentile=5.0):
    if len(sets) < 2:
        return torch.tensor(0.0)
    matrix = pairwise_chamfer(sets, sets)
    matrix.fill_diagonal_(float("inf"))
    return torch.quantile(matrix.min(dim=1).values, percentile / 100.0)


def novelty_stats(sets, reference_sets):
    if len(sets) == 0 or len(reference_sets) == 0:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0}
    matrix = pairwise_chamfer(sets, reference_sets)
    nearest = matrix.min(dim=1).values
    return {
        "mean": float(nearest.mean()),
        "median": float(nearest.median()),
        "p05": float(torch.quantile(nearest, 0.05)),
    }


def _backproject(u, v, depth, camera):
    x = (u - camera["cx"]) * depth / camera["fx"]
    y = (camera["cy"] - v) * depth / camera["fy"]
    cam = torch.stack([x, y, depth], dim=-1)
    return camera["eye"].unsqueeze(0) + cam @ camera["rot"]


def _project(world, camera):
    cam = (world - camera["eye"].unsqueeze(0)) @ camera["rot"].transpose(-1, -2)
    z = cam[:, 2]
    u = camera["fx"] * cam[:, 0] / z.clamp_min(1e-4) + camera["cx"]
    v = camera["cy"] - camera["fy"] * cam[:, 1] / z.clamp_min(1e-4)
    return u, v, z


ALPHA_GATE = 0.5


def multi_view_consistency(renderings, cameras, cfg, object_extent=1.0):
    stride = max(int(cfg.mvc_stride), 1)
    max_pairs = int(cfg.mvc_max_pairs)
    depth_gate = 0.02 * float(object_extent)
    prepared = []
    for view in renderings:
        alpha_map = view["alpha"][::stride, ::stride]
        depth_map = view["depth"][::stride, ::stride]
        rgb_map = view["rgb"][:, ::stride, ::stride]
        height, width = depth_map.shape
        grid_v, grid_u = torch.meshgrid(
            torch.arange(0, height * stride, stride, dtype=torch.float32, device=view["alpha"].device)[:height],
            torch.arange(0, width * stride, stride, dtype=torch.float32, device=view["alpha"].device)[:width],
            indexing="ij",
        )
        valid = (alpha_map > ALPHA_GATE) & (depth_map > 1e-4)
        prepared.append(
            {
                "u": grid_u[valid],
                "v": grid_v[valid],
                "depth": depth_map[valid],
                "rgb": rgb_map[:, valid],
                "alpha_map": alpha_map,
                "depth_map": depth_map,
                "rgb_map": rgb_map,
                "shape": (height, width),
            }
        )
    mvc_total, dcs_total, pairs = 0.0, 0.0, 0
    for index_k, view_k in enumerate(prepared):
        for index_l, view_l in enumerate(prepared):
            if index_k == index_l or pairs >= max_pairs:
                continue
            pairs += 1
            if view_k["u"].numel() == 0:
                continue
            camera_k, camera_l = cameras[index_k], cameras[index_l]
            world = _backproject(view_k["u"], view_k["v"], view_k["depth"], camera_k)
            u_l, v_l, z_kl = _project(world, camera_l)
            height, width = view_l["shape"]
            ui = (u_l / stride).round().long()
            vi = (v_l / stride).round().long()
            inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & (z_kl > 1e-4)
            ui, vi = ui.clamp(0, width - 1), vi.clamp(0, height - 1)
            rendered_depth = view_l["depth_map"][vi, ui]
            forward_ok = inside & ((rendered_depth - z_kl).abs() <= depth_gate)
            world_l = _backproject(u_l, v_l, rendered_depth, camera_l)
            u_k, v_k, z_lk = _project(world_l, camera_k)
            back_ok = ((u_k - view_k["u"]).abs() <= stride) & ((v_k - view_k["v"]).abs() <= stride)
            back_ok = back_ok & ((view_k["depth"] - z_lk).abs() <= depth_gate)
            visible = forward_ok & back_ok
            if int(visible.sum()) == 0:
                continue
            rgb_l = view_l["rgb_map"][:, vi[visible], ui[visible]]
            mvc_total += float(torch.exp(-(rgb_l - view_k["rgb"][:, visible]).abs().sum(dim=0) / 0.1).mean())
            dcs_total += float(torch.exp(-(rendered_depth[visible] - z_kl[visible]).abs() / depth_gate).mean())
    return mvc_total / max(pairs, 1), dcs_total / max(pairs, 1)


def object_extent(obj):
    centers = torch.as_tensor(obj["centers"], dtype=torch.float32).reshape(-1, 3)
    return float((centers.max(dim=0).values - centers.min(dim=0).values).norm())


def render_feature_vectors(obj, cameras, cfg):
    views = [render_object(obj, camera, cfg.image_size, cfg.background, cfg.blur, cfg.max_primitives) for camera in cameras]
    per_view = torch.stack([view["rgb"].mean(dim=(1, 2)) for view in views])
    return per_view, views


def mean_pairwise_euclidean(vectors):
    if len(vectors) < 2:
        return torch.tensor(0.0)
    stacked = torch.stack(vectors)
    dist = torch.cdist(stacked, stacked)
    rows = torch.triu_indices(len(vectors), len(vectors), offset=1)
    return dist[rows[0], rows[1]].mean()


def frechet_distance(vectors_a, vectors_b):
    a = np.asarray(torch.stack(vectors_a).detach().cpu().numpy(), dtype=np.float64)
    b = np.asarray(torch.stack(vectors_b).detach().cpu().numpy(), dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False) if a.shape[0] > 1 else np.zeros((a.shape[1], a.shape[1]))
    cov_b = np.cov(b, rowvar=False) if b.shape[0] > 1 else np.zeros((b.shape[1], b.shape[1]))
    cross = linalg.sqrtm(cov_a @ cov_b)
    coupling = np.trace(cov_a) + np.trace(cov_b) - 2.0 * float(np.real(np.trace(cross)))
    return float(np.sum((mu_a - mu_b) ** 2) + max(coupling, 0.0))


def evaluate_class(generated, reference, training, cameras, eval_cfg, render_cfg):
    gen_points = [sample_points(normalize_object(obj), eval_cfg.point_count) for obj in generated]
    ref_points = [sample_points(normalize_object(obj), eval_cfg.point_count) for obj in reference]
    train_points = [sample_points(normalize_object(obj), eval_cfg.point_count) for obj in training]

    mmd, coverage = mmd_coverage(pairwise_chamfer(gen_points, ref_points))
    union = gen_points + ref_points
    labels = [1] * len(gen_points) + [0] * len(ref_points)
    accuracy = leave_one_out_knn_accuracy(union, labels)
    tau = nearest_neighbor_threshold(ref_points, eval_cfg.uniqueness_percentile)
    novelty = novelty_stats(gen_points, train_points)
    pooled_gen = torch.cat(gen_points, dim=0) if gen_points else torch.zeros(1, 3)
    pooled_ref = torch.cat(ref_points, dim=0) if ref_points else torch.zeros(1, 3)

    mvc_values, dcs_values, gen_features, ref_features = [], [], [], []
    for obj in generated:
        normalized = normalize_object(obj)
        per_view, views = render_feature_vectors(normalized, cameras, render_cfg)
        mvc, dcs = multi_view_consistency(views, cameras, eval_cfg, max(object_extent(normalized), 1e-3))
        mvc_values.append(mvc)
        dcs_values.append(dcs)
        gen_features.append(per_view.mean(dim=0))
    for obj in reference:
        normalized = normalize_object(obj)
        per_view, _ = render_feature_vectors(normalized, cameras, render_cfg)
        ref_features.append(per_view.mean(dim=0))

    return {
        "mmd_cd": float(mmd),
        "coverage_cd": float(coverage),
        "knn_accuracy": float(accuracy),
        "diversity_cd": float(mean_pairwise_distance(gen_points)),
        "unique_ratio": float(uniqueness_ratio(gen_points, tau)),
        "novelty_mean": novelty["mean"],
        "novelty_median": novelty["median"],
        "novelty_p05": novelty["p05"],
        "f_score_tau": float(f_score(pooled_gen, pooled_ref, eval_cfg.f_tau)),
        "mvc": float(np.mean(mvc_values)) if mvc_values else 0.0,
        "dcs": float(np.mean(dcs_values)) if dcs_values else 0.0,
        "feature_diversity": float(mean_pairwise_euclidean(gen_features)),
        "feature_frechet": frechet_distance(gen_features, ref_features) if gen_features and ref_features else 0.0,
    }


def write_csv(rows, path, columns=None):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    keys = sorted({key for row in rows for key in row if key != "class"})
    if not keys and columns is not None:
        keys = list(columns)
    header = ["class"] + keys
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join([str(row.get(key, "")) for key in header]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
