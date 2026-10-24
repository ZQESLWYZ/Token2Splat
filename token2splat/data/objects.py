import numpy as np

import torch

from ..features import FEATURE_KEYS, FEATURE_SPLITS, object_center_and_scale


def make_object(centers, scales, rotations, opacities, appearances, label, object_id="", scene_id=""):
    obj = {
        "centers": np.asarray(centers, dtype=np.float32),
        "scales": np.asarray(scales, dtype=np.float32),
        "rotations": np.asarray(rotations, dtype=np.float32),
        "opacities": np.asarray(opacities, dtype=np.float32).reshape(-1, 1),
        "appearances": np.asarray(appearances, dtype=np.float32).reshape(-1, 3),
        "label": int(label),
        "object_id": str(object_id),
        "scene_id": str(scene_id),
    }
    return obj


def make_record(obj, object_id, scene_id):
    return {"object": obj, "object_id": str(object_id), "scene_id": str(scene_id), "label": int(obj["label"])}


def object_tensors(obj, device="cpu"):
    out = {key: torch.as_tensor(obj[key], dtype=torch.float32, device=device) for key in FEATURE_KEYS}
    out["label"] = int(obj["label"])
    return out


def save_instance_npz(path, obj):
    payload = {key: np.asarray(obj[key]) for key in FEATURE_KEYS}
    payload["label"] = np.asarray(int(obj["label"]))
    payload["object_id"] = np.asarray(str(obj["object_id"]))
    payload["scene_id"] = np.asarray(str(obj["scene_id"]))
    np.savez(path, **payload)
    return path


def load_instance_npz(path):
    data = np.load(path, allow_pickle=False)
    obj = {key: data[key].astype(np.float32) for key in FEATURE_KEYS}
    obj["label"] = int(data["label"])
    obj["object_id"] = str(data["object_id"])
    obj["scene_id"] = str(data["scene_id"])
    return obj


def is_valid_object(obj, min_primitives, min_extent):
    centers = np.asarray(obj["centers"], dtype=np.float32)
    scales = np.asarray(obj["scales"], dtype=np.float32)
    if centers.shape[0] < min_primitives:
        return False
    if centers.shape[1] != 3:
        return False
    if not np.isfinite(centers).all() or not np.isfinite(scales).all():
        return False
    if scales.min() <= 0.0:
        return False
    _, extent = object_center_and_scale(torch.from_numpy(centers))
    return bool(extent.item() >= min_extent)


def feature_split_widths():
    return dict(zip(FEATURE_KEYS, FEATURE_SPLITS))
