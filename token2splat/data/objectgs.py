import json
import os

import numpy as np

from .dataset import write_dataset
from .objects import is_valid_object, make_object, make_record

SCENE_KEYS = ("centers", "scales", "rotations", "opacities", "appearances", "instance_ids", "instance_boxes", "instance_classes")


def load_scene_npz(path):
    data = np.load(path, allow_pickle=False)
    missing = [key for key in SCENE_KEYS if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing scene keys: {missing}")
    return {key: data[key] for key in SCENE_KEYS}


def crop_instance(scene, instance_index):
    box = np.asarray(scene["instance_boxes"][instance_index], dtype=np.float64)
    low, high = box[:3], box[3:6]
    instance_ids = np.asarray(scene["instance_ids"])
    centers = np.asarray(scene["centers"], dtype=np.float64)
    inside = np.all((centers >= low) & (centers <= high), axis=1)
    if instance_ids.size == centers.shape[0]:
        inside = inside & (instance_ids == instance_index)
    keep = np.flatnonzero(inside)
    if keep.size == 0:
        return None
    return make_object(
        centers[keep],
        np.asarray(scene["scales"], dtype=np.float64)[keep],
        np.asarray(scene["rotations"], dtype=np.float64)[keep],
        np.asarray(scene["opacities"], dtype=np.float64)[keep],
        np.asarray(scene["appearances"], dtype=np.float64)[keep],
        0,
        object_id=f"{scene_id_of(scene)}_inst{instance_index:03d}",
        scene_id=scene_id_of(scene),
    )


def scene_id_of(scene):
    return str(scene.get("scene_id", "scene")) if isinstance(scene, dict) else "scene"


def build_objectgs_dataset(scenes_dir, out_dir, min_primitives=64, min_instances_per_class=8, min_extent=0.05):
    scene_paths = sorted(
        os.path.join(scenes_dir, name) for name in os.listdir(scenes_dir) if name.endswith(".npz")
    )
    class_names, records = [], []
    for path in scene_paths:
        scene = load_scene_npz(path)
        scene_id = os.path.splitext(os.path.basename(path))[0]
        scene["scene_id"] = scene_id
        classes = np.asarray(scene["instance_classes"])
        for instance_index in range(len(scene["instance_boxes"])):
            obj = crop_instance(scene, instance_index)
            if obj is None or not is_valid_object(obj, min_primitives, min_extent):
                continue
            name = str(classes[instance_index])
            if name not in class_names:
                class_names.append(name)
            obj["label"] = class_names.index(name)
            obj["object_id"] = f"{scene_id}_inst{instance_index:03d}"
            obj["scene_id"] = scene_id
            records.append(make_record(obj, obj["object_id"], scene_id))

    counts = {name: 0 for name in class_names}
    for record in records:
        counts[class_names[record["label"]]] += 1
    dropped = [name for name, count in counts.items() if count < min_instances_per_class]
    records = [record for record in records if class_names[record["label"]] not in dropped]
    kept_names = sorted({class_names[record["label"]] for record in records})
    remap = {name: index for index, name in enumerate(kept_names)}
    for record in records:
        record["label"] = remap[class_names[record["label"]]]
        record["object"]["label"] = record["label"]
    write_dataset(out_dir, records, kept_names)
    with open(os.path.join(out_dir, "filter_report.json"), "w", encoding="utf-8") as handle:
        json.dump({"kept_classes": kept_names, "dropped_classes": dropped, "num_instances": len(records)}, handle, indent=2)
    return out_dir, kept_names, len(records)


def make_random_scene_dump(path, num_instances=6, num_primitives=400, seed=0):
    from .synthetic import make_class_object

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    rng = np.random.default_rng(seed)
    centers, scales, rotations, opacities, appearances, instance_ids, boxes, classes = [], [], [], [], [], [], [], []
    offset = 0
    for index in range(num_instances):
        shift = rng.uniform(-1.2, 1.2, size=3)
        obj = make_class_object(int(rng.integers(0, 6)), seed * 977 + index, num_primitives)
        moved = np.asarray(obj["centers"]) + shift
        centers.append(moved)
        scales.append(obj["scales"])
        rotations.append(obj["rotations"])
        opacities.append(obj["opacities"])
        appearances.append(obj["appearances"])
        count = moved.shape[0]
        instance_ids.extend([index] * count)
        low, high = moved.min(axis=0) - 0.05, moved.max(axis=0) + 0.05
        boxes.append(np.concatenate([low, high]))
        classes.append(str(obj["label"]))
        offset += count
    np.savez(
        path,
        centers=np.concatenate(centers, axis=0).astype(np.float32),
        scales=np.concatenate(scales, axis=0).astype(np.float32),
        rotations=np.concatenate(rotations, axis=0).astype(np.float32),
        opacities=np.concatenate(opacities, axis=0).astype(np.float32),
        appearances=np.concatenate(appearances, axis=0).astype(np.float32),
        instance_ids=np.asarray(instance_ids, dtype=np.int64),
        instance_boxes=np.stack(boxes, axis=0).astype(np.float32),
        instance_classes=np.asarray(classes),
    )
    return path
