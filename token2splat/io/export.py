import json
import math
import os

import numpy as np

import torch

SH_C0 = 0.28209479177387814
PLY_HEADER = [
    "property float x",
    "property float y",
    "property float z",
    "property float f_dc_0",
    "property float f_dc_1",
    "property float f_dc_2",
    "property float opacity",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
]


def _to_numpy(obj):
    return {key: np.asarray(obj[key].detach().cpu().numpy() if torch.is_tensor(obj[key]) else obj[key], dtype=np.float32) for key in ("centers", "scales", "rotations", "opacities", "appearances")}


def _ensure_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_gaussians_npz(path, obj, meta=None):
    _ensure_dir(path)
    arrays = _to_numpy(obj)
    payload = dict(arrays)
    if meta:
        for key, value in meta.items():
            payload[key] = np.asarray(value)
    np.savez(path, **payload)
    return path


def save_gaussians_ply(path, obj):
    _ensure_dir(path)
    arrays = _to_numpy(obj)
    count = arrays["centers"].shape[0]
    opacity = np.clip(arrays["opacities"].reshape(-1), 1e-4, 1.0 - 1e-4)
    opacity_logit = np.log(opacity / (1.0 - opacity))
    color_sh = (arrays["appearances"].reshape(count, 3) - 0.5) / SH_C0
    rows = np.concatenate(
        [arrays["centers"].reshape(count, 3), color_sh, opacity_logit[:, None], arrays["scales"].reshape(count, 3), arrays["rotations"].reshape(count, 4)],
        axis=1,
    ).astype("<f4")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"] + PLY_HEADER + ["end_header"]
    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(rows.tobytes())
    return path


def save_image(path, rgb):
    _ensure_dir(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = rgb.permute(1, 2, 0).detach().cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb)
    fig, axis = plt.subplots(figsize=(3, 3), dpi=120)
    axis.imshow(np.clip(image, 0.0, 1.0))
    axis.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return path


def save_render_grid(path, images, columns=4, cell=2.4, dpi=110):
    _ensure_dir(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = list(images)
    rows = int(math.ceil(len(images) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * cell, rows * cell), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue
        image = images[index]["rgb"] if isinstance(images[index], dict) else images[index]
        array = image.permute(1, 2, 0).detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        axis.imshow(np.clip(array, 0.0, 1.0))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def save_json(path, payload):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
