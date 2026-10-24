import os

import torch

from ..data.dataset import load_dataset_dir
from ..features import normalize_object
from ..losses import patch_targets
from ..render.splat import render_views_detached, render_views_grad


def resolve_device(name):
    if isinstance(name, str) and name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name if name else "cpu"

RENDER_KEYS = ("centers", "scales", "rotations", "opacities", "appearances")


def batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def load_splits(cfg, data_dir=None):
    return load_dataset_dir(data_dir if data_dir is not None else cfg.data.root)


def normalize_record(record):
    normalized = normalize_object(record["object"])
    out = {key: torch.as_tensor(normalized[key], dtype=torch.float32) for key in RENDER_KEYS}
    out["label"] = int(record["label"])
    return out


def gt_objects_from_batch(batch):
    centers, scales, rotations, opacities, appearances, mask = patch_targets(batch)
    batch_size = centers.shape[0]
    flat = {
        "centers": centers.reshape(batch_size, -1, 3),
        "scales": scales.reshape(batch_size, -1, 3),
        "rotations": rotations.reshape(batch_size, -1, 4),
        "opacities": opacities.reshape(batch_size, -1, 1),
        "appearances": appearances.reshape(batch_size, -1, 3),
    }
    flags = mask.reshape(batch_size, -1)
    objects = []
    for index in range(batch_size):
        rows = flags[index].nonzero(as_tuple=True)[0]
        objects.append({key: value[index, rows].detach() for key, value in flat.items()})
    return objects


class GroundTruthRenderCache:
    def __init__(self, cameras, cfg):
        self.cameras = cameras
        self.cfg = cfg
        self.entries = {}

    def get(self, key, obj):
        if key not in self.entries:
            self.entries[key] = [
                {"rgb": view["rgb"].detach(), "alpha": view["alpha"].detach()}
                for view in render_views_detached(obj, self.cameras, self.cfg)
            ]
        return self.entries[key]


def history_path(work_dir):
    return os.path.join(work_dir, "history.json")
