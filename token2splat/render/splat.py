import math

import torch

from ..features import gaussian_covariance

PRIMITIVE_KEYS = ("centers", "scales", "rotations", "opacities", "appearances")


def subsample_primitives(obj, max_primitives):
    total = obj["centers"].shape[0]
    if max_primitives <= 0 or total <= max_primitives:
        return obj
    stride = int(math.ceil(total / float(max_primitives)))
    return {key: obj[key][::stride] for key in PRIMITIVE_KEYS}


def project_primitives(obj, camera):
    rot_t = camera["rot"].transpose(-1, -2)
    cam = (obj["centers"] - camera["eye"]) @ rot_t
    z = cam[:, 2]
    z_safe = z.clamp_min(1e-3)
    fx, fy = camera["fx"], camera["fy"]
    u = fx * cam[:, 0] / z_safe + camera["cx"]
    v = camera["cy"] - fy * cam[:, 1] / z_safe
    cov = gaussian_covariance(obj["scales"], obj["rotations"])
    jac = torch.zeros(cam.shape[0], 2, 3, device=cam.device, dtype=cam.dtype)
    jac[:, 0, 0] = fx / z_safe
    jac[:, 0, 2] = -fx * cam[:, 0] / (z_safe * z_safe)
    jac[:, 1, 1] = -fy / z_safe
    jac[:, 1, 2] = fy * cam[:, 1] / (z_safe * z_safe)
    cov2d = jac @ cov @ jac.transpose(-1, -2)
    cov2d = 0.5 * (cov2d + cov2d.transpose(-1, -2))
    det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]).clamp_min(1e-8)
    inv = torch.stack(
        [
            torch.stack([cov2d[:, 1, 1] / det, -cov2d[:, 0, 1] / det], dim=-1),
            torch.stack([-cov2d[:, 1, 0] / det, cov2d[:, 0, 0] / det], dim=-1),
        ],
        dim=-2,
    )
    return u, v, z, cov2d, inv


def render_object(obj, camera, image_size, background=1.0, blur=0.5, max_primitives=0):
    obj = subsample_primitives(obj, max_primitives)
    centers = obj["centers"]
    device, dtype = centers.device, centers.dtype
    height = width = int(image_size)
    if centers.shape[0] == 0:
        rgb = torch.full((3, height, width), float(background), device=device, dtype=dtype)
        return {
            "rgb": rgb,
            "alpha": torch.zeros(height, width, device=device, dtype=dtype),
            "depth": torch.zeros(height, width, device=device, dtype=dtype),
        }

    u, v, z, cov2d, cov_inv = project_primitives(obj, camera)
    eigen = torch.linalg.eigvalsh(cov2d.detach())
    radius = (3.0 * eigen[:, -1].clamp_min(1e-8).sqrt() + 1.0).ceil()
    u_flat, v_flat, radius_flat = u.detach().tolist(), v.detach().tolist(), radius.detach().tolist()
    keep = ((z.detach() > 1e-3) & torch.isfinite(u.detach()) & torch.isfinite(v.detach())).tolist()

    colors = obj["appearances"]
    opacity = obj["opacities"].squeeze(-1)
    accum_rgb = torch.zeros(3, height, width, device=device, dtype=dtype)
    accum_alpha = torch.zeros(1, height, width, device=device, dtype=dtype)
    accum_depth = torch.zeros(1, height, width, device=device, dtype=dtype)
    accum_weight = torch.zeros(1, height, width, device=device, dtype=dtype)

    for index in z.detach().argsort().tolist():
        if not keep[index]:
            continue
        reach = int(radius_flat[index])
        u0 = max(0, int(math.floor(u_flat[index] - reach)))
        u1 = min(width, int(math.ceil(u_flat[index] + reach)) + 1)
        v0 = max(0, int(math.floor(v_flat[index] - reach)))
        v1 = min(height, int(math.ceil(v_flat[index] + reach)) + 1)
        if u1 <= u0 or v1 <= v0:
            continue
        du = torch.arange(u0, u1, device=device, dtype=dtype).unsqueeze(0) - u[index]
        dv = torch.arange(v0, v1, device=device, dtype=dtype).unsqueeze(1) - v[index]
        quad = cov_inv[index, 0, 0] * du * du + 2.0 * cov_inv[index, 0, 1] * du * dv + cov_inv[index, 1, 1] * dv * dv
        weight = torch.exp(-0.5 * quad).clamp(0.0, 1.0)
        alpha = (opacity[index] * weight).clamp(0.0, 0.99).unsqueeze(0)
        trans = 1.0 - accum_alpha[:, v0:v1, u0:u1]
        accum_rgb[:, v0:v1, u0:u1] = accum_rgb[:, v0:v1, u0:u1] + trans * alpha * colors[index].view(3, 1, 1)
        accum_depth[:, v0:v1, u0:u1] = accum_depth[:, v0:v1, u0:u1] + trans * alpha * z[index]
        accum_weight[:, v0:v1, u0:u1] = accum_weight[:, v0:v1, u0:u1] + trans * alpha
        accum_alpha[:, v0:v1, u0:u1] = accum_alpha[:, v0:v1, u0:u1] + trans * alpha

    rgb = accum_rgb + (1.0 - accum_alpha) * float(background)
    depth = (accum_depth / accum_weight.clamp_min(1e-6)).squeeze(0)
    return {"rgb": rgb, "alpha": accum_alpha.squeeze(0), "depth": depth}


def render_views(obj, cameras, cfg):
    return [
        render_object(
            obj,
            camera,
            cfg.image_size,
            background=cfg.background,
            blur=cfg.blur,
            max_primitives=cfg.max_primitives,
        )
        for camera in cameras
    ]


def render_views_detached(obj, cameras, cfg):
    with torch.no_grad():
        return render_views(obj, cameras, cfg)


def render_views_grad(obj, cameras, cfg):
    return render_views(obj, cameras, cfg)
