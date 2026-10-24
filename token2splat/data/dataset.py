import json
import os

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from .objects import load_instance_npz, make_record


def scene_level_split(records, train_ratio, val_ratio, seed):
    scenes = sorted({record["scene_id"] for record in records})
    order = np.random.default_rng(seed).permutation(len(scenes))
    num_train = max(int(round(len(scenes) * train_ratio)), 1)
    num_val = max(int(round(len(scenes) * val_ratio)), 1) if val_ratio > 0 else 0
    if num_train + num_val > len(scenes):
        num_val = max(len(scenes) - num_train, 0)
    train = set(np.asarray(scenes)[order[:num_train]].tolist())
    val = set(np.asarray(scenes)[order[num_train:num_train + num_val]].tolist())
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        if record["scene_id"] in train:
            splits["train"].append(record)
        elif record["scene_id"] in val:
            splits["val"].append(record)
        else:
            splits["test"].append(record)
    return splits


def write_dataset(root, records, class_names, train_ratio=0.7, val_ratio=0.15, seed=0):
    inst_dir = os.path.join(root, "instances")
    os.makedirs(inst_dir, exist_ok=True)
    entries, splits = [], {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    scenes = sorted({record["scene_id"] for record in records})
    order = rng.permutation(len(scenes))
    num_train = max(int(round(len(scenes) * train_ratio)), 1)
    num_val = max(int(round(len(scenes) * val_ratio)), 1) if val_ratio > 0 else 0
    train = set(np.asarray(scenes)[order[:num_train]].tolist())
    val = set(np.asarray(scenes)[order[num_train:num_train + num_val]].tolist())
    for index, record in enumerate(records):
        obj = record["object"]
        name = f"{index:05d}_{record['object_id']}.npz"
        np.savez(
            os.path.join(inst_dir, name),
            centers=np.asarray(obj["centers"], dtype=np.float32),
            scales=np.asarray(obj["scales"], dtype=np.float32),
            rotations=np.asarray(obj["rotations"], dtype=np.float32),
            opacities=np.asarray(obj["opacities"], dtype=np.float32).reshape(-1, 1),
            appearances=np.asarray(obj["appearances"], dtype=np.float32).reshape(-1, 3),
            label=np.asarray(int(record["label"])),
            object_id=np.asarray(str(record["object_id"])),
            scene_id=np.asarray(str(record["scene_id"])),
        )
        bucket = "train" if record["scene_id"] in train else ("val" if record["scene_id"] in val else "test")
        splits[bucket].append(record["object_id"])
        entries.append(
            {
                "file": os.path.join("instances", name),
                "object_id": record["object_id"],
                "scene_id": record["scene_id"],
                "label": int(record["label"]),
                "class_name": class_names[int(record["label"])],
                "num_primitives": int(np.asarray(obj["centers"]).shape[0]),
            }
        )
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"instances": entries, "class_names": list(class_names)}, handle, indent=2)
    with open(os.path.join(root, "splits.json"), "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    return root


def load_dataset_dir(root):
    with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(os.path.join(root, "splits.json"), "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    by_id = {}
    for entry in manifest["instances"]:
        obj = load_instance_npz(os.path.join(root, entry["file"]))
        by_id[entry["object_id"]] = make_record(obj, entry["object_id"], entry["scene_id"])
    return {name: [by_id[oid] for oid in ids if oid in by_id] for name, ids in splits.items()}, manifest["class_names"]


class TokenDataset(Dataset):
    def __init__(self, records, tokenizer, cache=False):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.cache and index in self._cache:
            token = self._cache[index]
        else:
            token = self.tokenizer.tokenize(self.records[index]["object"])
            if self.cache:
                self._cache[index] = token
        sample = {
            "desc": torch.from_numpy(token["desc"]),
            "valid": torch.from_numpy(token["valid"]),
            "count": torch.from_numpy(token["count"]),
            "occ": torch.from_numpy(token["occ"]),
            "mu": torch.from_numpy(token["mu"]),
            "quat": torch.from_numpy(token["quat"]),
            "frame": torch.from_numpy(token["frame"]),
            "eta": torch.from_numpy(token["eta"]),
            "label": torch.tensor(self.records[index]["label"], dtype=torch.long),
            "object_id": self.records[index]["object_id"],
        }
        return sample


def token_collate(items):
    batch = {}
    for key in ("desc", "valid", "count", "occ", "mu", "quat", "frame", "eta", "label"):
        batch[key] = torch.stack([item[key] for item in items])
    batch["object_id"] = [item["object_id"] for item in items]
    return batch


def make_dataloader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=token_collate, drop_last=False)
