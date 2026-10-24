import numpy as np

from .objects import make_object

CLASS_NAMES = ("armchair", "dining_table", "bookshelf", "floor_lamp", "console", "car")

PALETTES = {
    0: ((0.55, 0.30, 0.24), (0.32, 0.30, 0.28)),
    1: ((0.62, 0.47, 0.30), (0.35, 0.32, 0.28)),
    2: ((0.70, 0.62, 0.48), (0.40, 0.36, 0.32)),
    3: ((0.85, 0.80, 0.62), (0.30, 0.30, 0.32)),
    4: ((0.45, 0.42, 0.40), (0.60, 0.55, 0.45)),
    5: ((0.25, 0.35, 0.60), (0.20, 0.20, 0.22)),
}


def _random_quats(rng, count):
    q = rng.normal(size=(count, 4))
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
    return q


def _box(rng, low, high, count, scale_lo, scale_hi, color, jitter):
    pts = rng.uniform(low, high, size=(count, 3))
    scales = rng.uniform(scale_lo, scale_hi, size=(count, 3))
    cols = np.clip(np.asarray(color, dtype=np.float64) + rng.normal(scale=jitter, size=(count, 3)), 0.0, 1.0)
    return pts, scales, cols


def _ring(rng, center, radius, count, scale, color, jitter):
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radius_j = radius * rng.uniform(0.85, 1.15, size=count)
    pts = np.stack(
        [center[0] + radius_j * np.cos(angle), center[1] + rng.normal(scale=0.01, size=count), center[2] + radius_j * np.sin(angle)],
        axis=1,
    )
    scales = np.tile(np.asarray(scale, dtype=np.float64), (count, 1))
    cols = np.clip(np.asarray(color, dtype=np.float64) + rng.normal(scale=jitter, size=(count, 3)), 0.0, 1.0)
    return pts, scales, cols


def _armchair(rng, budget, color, dark, jitter):
    parts = [
        _box(rng, (-0.45, -0.05, -0.45), (0.45, 0.12, 0.45), int(budget * 0.42), 0.012, 0.03, color, jitter),
        _box(rng, (-0.45, 0.10, 0.32), (0.45, 0.70, 0.48), int(budget * 0.30), 0.012, 0.03, color, jitter),
        _box(rng, (-0.52, 0.02, -0.45), (-0.42, 0.35, 0.45), int(budget * 0.08), 0.012, 0.03, dark, jitter),
        _box(rng, (0.42, 0.02, -0.45), (0.52, 0.35, 0.45), int(budget * 0.08), 0.012, 0.03, dark, jitter),
    ]
    for sx in (-0.38, 0.38):
        for sz in (-0.36, 0.36):
            parts.append(_box(rng, (sx - 0.04, -0.55, sz - 0.04), (sx + 0.04, -0.02, sz + 0.04), max(int(budget * 0.03), 4), 0.01, 0.02, dark, jitter))
    return parts


def _dining_table(rng, budget, color, dark, jitter):
    parts = [_box(rng, (-0.75, 0.32, -0.45), (0.75, 0.40, 0.45), int(budget * 0.55), 0.012, 0.03, color, jitter)]
    for sx in (-0.68, 0.68):
        for sz in (-0.38, 0.38):
            parts.append(_box(rng, (sx - 0.05, -0.55, sz - 0.05), (sx + 0.05, 0.33, sz + 0.05), max(int(budget * 0.08), 6), 0.01, 0.025, dark, jitter))
    return parts


def _bookshelf(rng, budget, color, dark, jitter):
    parts = [
        _box(rng, (-0.55, -0.75, -0.20), (-0.48, 0.75, 0.20), int(budget * 0.12), 0.01, 0.025, dark, jitter),
        _box(rng, (0.48, -0.75, -0.20), (0.55, 0.75, 0.20), int(budget * 0.12), 0.01, 0.025, dark, jitter),
        _box(rng, (-0.55, 0.70, -0.20), (0.55, 0.75, 0.20), int(budget * 0.08), 0.01, 0.02, dark, jitter),
        _box(rng, (-0.55, -0.75, -0.20), (0.55, -0.70, 0.20), int(budget * 0.08), 0.01, 0.02, dark, jitter),
    ]
    for level in range(3):
        y = -0.45 + level * 0.40
        parts.append(_box(rng, (-0.48, y, -0.18), (0.48, y + 0.04, 0.18), max(int(budget * 0.05), 6), 0.01, 0.02, color, jitter))
        parts.append(_box(rng, (-0.44, y + 0.04, -0.14), (0.44, y + 0.30, 0.12), max(int(budget * 0.05), 6), 0.02, 0.05, PALETTES[level % 3][0], jitter * 2.0))
    return parts


def _floor_lamp(rng, budget, color, dark, jitter):
    parts = [
        _ring(rng, (0.0, -0.72, 0.0), 0.20, max(int(budget * 0.10), 8), (0.02, 0.02, 0.02), dark, jitter),
        _box(rng, (-0.03, -0.72, -0.03), (0.03, 0.35, 0.03), int(budget * 0.45), 0.008, 0.018, dark, jitter),
        _box(rng, (-0.22, 0.35, -0.22), (0.22, 0.62, 0.22), int(budget * 0.35), 0.012, 0.03, color, jitter),
        _ring(rng, (0.0, 0.62, 0.0), 0.22, max(int(budget * 0.10), 8), (0.02, 0.02, 0.02), color, jitter),
    ]
    return parts


def _console(rng, budget, color, dark, jitter):
    parts = [
        _box(rng, (-0.60, -0.15, -0.25), (0.60, 0.45, 0.25), int(budget * 0.62), 0.012, 0.03, color, jitter),
        _box(rng, (-0.58, -0.10, 0.24), (-0.04, 0.42, 0.30), max(int(budget * 0.10), 8), 0.01, 0.02, dark, jitter),
        _box(rng, (0.04, -0.10, 0.24), (0.58, 0.42, 0.30), max(int(budget * 0.10), 8), 0.01, 0.02, dark, jitter),
    ]
    for sx in (-0.50, 0.50):
        parts.append(_box(rng, (sx - 0.04, -0.55, -0.04), (sx + 0.04, -0.14, 0.04), max(int(budget * 0.04), 4), 0.01, 0.02, dark, jitter))
    return parts


def _car(rng, budget, color, dark, jitter):
    parts = [
        _box(rng, (-0.75, -0.20, -0.38), (0.75, 0.10, 0.38), int(budget * 0.45), 0.012, 0.03, color, jitter),
        _box(rng, (-0.35, 0.10, -0.32), (0.30, 0.38, 0.32), int(budget * 0.20), 0.012, 0.03, color, jitter),
        _box(rng, (-0.30, 0.12, -0.34), (0.26, 0.34, -0.30), max(int(budget * 0.05), 6), 0.01, 0.02, dark, jitter),
    ]
    for sx in (-0.52, 0.52):
        for sz in (-0.34, 0.34):
            parts.append(_ring(rng, (sx, -0.22, sz), 0.16, max(int(budget * 0.07), 6), (0.02, 0.05, 0.02), dark, jitter))
    return parts


BUILDERS = {0: _armchair, 1: _dining_table, 2: _bookshelf, 3: _floor_lamp, 4: _console, 5: _car}


def make_class_object(class_index, seed, num_primitives, scale_jitter=0.015):
    rng = np.random.default_rng(seed)
    name = CLASS_NAMES[class_index]
    color, dark = PALETTES[class_index]
    parts = BUILDERS[class_index](rng, num_primitives, color, dark, scale_jitter)
    centers = np.concatenate([part[0] for part in parts], axis=0)
    scales = np.concatenate([part[1] for part in parts], axis=0)
    colors = np.concatenate([part[2] for part in parts], axis=0)
    count = centers.shape[0]
    rotations = _random_quats(rng, count)
    opacities = rng.uniform(0.65, 1.0, size=(count, 1))
    shift = rng.uniform(-0.15, 0.15, size=3)
    return make_object(
        centers + shift,
        scales,
        rotations,
        opacities,
        colors,
        class_index,
        object_id=f"{name}_{seed}",
        scene_id="",
    )


def build_synthetic_records(num_scenes, objects_per_scene, num_primitives, seed, class_names=CLASS_NAMES):
    records = []
    for scene in range(num_scenes):
        scene_id = f"scene_{scene:03d}"
        for index in range(objects_per_scene):
            class_index = (scene * objects_per_scene + index) % len(class_names)
            seed_key = seed * 100000 + scene * 1000 + index
            obj = make_class_object(class_index, seed_key, num_primitives)
            obj["scene_id"] = scene_id
            obj["object_id"] = f"{scene_id}_{CLASS_NAMES[class_index]}_{index}"
            records.append({"object": obj, "object_id": obj["object_id"], "scene_id": scene_id, "label": class_index})
    return records
