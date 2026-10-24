import math

import numpy as np

import torch


def look_at(eye, target, fov_deg, image_size):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    up_axis = np.cross(right, forward)
    rot = np.stack([right, up_axis, forward], axis=0)
    focal = (image_size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return {
        "eye": torch.as_tensor(eye, dtype=torch.float32),
        "rot": torch.as_tensor(rot, dtype=torch.float32),
        "fx": float(focal),
        "fy": float(focal),
        "cx": float(image_size) / 2.0,
        "cy": float(image_size) / 2.0,
        "position": torch.as_tensor(eye, dtype=torch.float32),
    }


def camera_grid(azimuths, elevations, radius):
    views = []
    for elevation in elevations:
        for index in range(azimuths):
            angle = 2.0 * math.pi * index / float(azimuths)
            el = math.radians(elevation)
            eye = radius * np.array(
                [math.cos(el) * math.cos(angle), math.sin(el), math.cos(el) * math.sin(angle)],
                dtype=np.float64,
            )
            views.append((eye, math.degrees(el), angle))
    return views


def build_cameras(cfg, num_views=None, device="cpu"):
    views = camera_grid(cfg.azimuths, cfg.elevations, cfg.radius)
    if num_views is not None:
        views = views[:num_views]
    cameras = [look_at(eye, np.zeros(3), cfg.fov_deg, cfg.image_size) for eye, _, _ in views]
    for camera in cameras:
        camera["eye"] = camera["eye"].to(device)
        camera["rot"] = camera["rot"].to(device)
    return cameras


def anchor_visibility(anchors, cameras, threshold=0.0):
    anchors = torch.as_tensor(np.asarray(anchors, dtype=np.float32))
    norm = anchors.norm(dim=-1, keepdim=True)
    dirs = anchors / norm.clamp_min(1e-6)
    dirs = torch.where(norm > 1e-3, dirs, torch.zeros_like(dirs))
    scores = torch.zeros(anchors.shape[0], dtype=torch.float32)
    for camera in cameras:
        cam_dir = camera["position"].to(torch.float32)
        cam_dir = cam_dir / cam_dir.norm().clamp_min(1e-6)
        facing = dirs @ cam_dir
        scores += torch.where(norm.squeeze(-1) <= 1e-3, torch.ones_like(facing), (facing > threshold).to(torch.float32))
    return scores / float(max(len(cameras), 1))
