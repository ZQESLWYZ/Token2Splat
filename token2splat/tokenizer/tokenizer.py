import numpy as np

import torch

from ..features import FEATURE_DIM, mat_to_quat, normalize_object, quat_mul
from .anchors import build_anchor_bank, patch_frame
from .morton import morton_codes, morton_order

TOKEN_KEYS = ("desc", "valid", "count", "occ", "mu", "quat", "eta")


def assign_anchors(centers, anchors, chunk=4096):
    out = np.empty(centers.shape[0], dtype=np.int64)
    for start in range(0, centers.shape[0], chunk):
        block = centers[start:start + chunk]
        dist = np.sum((block[:, None, :] - anchors[None, :, :]) ** 2, axis=-1)
        out[start:start + chunk] = np.argmin(dist, axis=1)
    return out


class CanonicalPatchTokenizer:
    def __init__(self, anchors, capacity_k, morton_bits=10, eta_min=1e-3):
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.num_slots = self.anchors.shape[0]
        self.capacity_k = int(capacity_k)
        self.morton_bits = int(morton_bits)
        self.eta_min = float(eta_min)

    @classmethod
    def from_records(cls, records, cfg):
        centers = []
        for record in records:
            normalized = normalize_object(record["object"])
            centers.append(np.asarray(normalized["centers"].numpy(), dtype=np.float64))
        pooled = np.concatenate(centers, axis=0)
        anchors = build_anchor_bank(pooled, cfg.num_anchors, cfg.anchor_pool, cfg.anchor_pad_step)
        return cls(anchors, cfg.capacity_k, cfg.morton_bits, cfg.eta_min)

    def token_shape(self):
        return (self.num_slots, self.capacity_k, FEATURE_DIM)

    def tokenize(self, obj):
        normalized = normalize_object(obj)
        centers = np.asarray(normalized["centers"].numpy(), dtype=np.float64)
        scales = np.asarray(normalized["scales"].numpy(), dtype=np.float64)
        rotations = np.asarray(normalized["rotations"].numpy(), dtype=np.float64)
        opacities = np.asarray(normalized["opacities"].numpy(), dtype=np.float64)
        appearances = np.asarray(normalized["appearances"].numpy(), dtype=np.float64)

        slots = assign_anchors(centers, self.anchors.astype(np.float64))
        desc = np.zeros((self.num_slots, self.capacity_k, FEATURE_DIM), dtype=np.float32)
        valid = np.zeros((self.num_slots, self.capacity_k), dtype=np.float32)
        count = np.zeros(self.num_slots, dtype=np.int64)
        mu = self.anchors.copy()
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (self.num_slots, 1))
        frame = np.tile(np.eye(3, dtype=np.float32), (self.num_slots, 1, 1))
        eta = np.full(self.num_slots, self.eta_min, dtype=np.float32)
        morton = np.zeros((self.num_slots, self.capacity_k), dtype=np.int64)

        for slot in range(self.num_slots):
            members = np.flatnonzero(slots == slot)
            if members.size == 0:
                continue
            count[slot] = members.size
            center, rot, extent, slot_quat = patch_frame(centers[members], self.eta_min)
            mu[slot] = center
            frame[slot] = rot
            quat[slot] = slot_quat
            eta[slot] = extent

            local = (centers[members] - center) @ rot
            order = morton_order(local / extent, self.morton_bits)
            members = members[order]
            local = local[order]
            keep = min(members.size, self.capacity_k)
            inv_rot = torch.from_numpy(rot.T).unsqueeze(0).double()
            inv_quat = mat_to_quat(inv_rot).squeeze(0)
            world_quat = torch.from_numpy(rotations[members]).unsqueeze(0).double()
            local_quat = quat_mul(inv_quat.expand(world_quat.shape), world_quat).squeeze(0).numpy()
            block = np.concatenate(
                [
                    local.astype(np.float32),
                    (scales[members] / extent).astype(np.float32),
                    local_quat.astype(np.float32),
                    opacities[members].astype(np.float32),
                    appearances[members].astype(np.float32),
                ],
                axis=1,
            )
            desc[slot, :keep] = block[:keep]
            valid[slot, :keep] = 1.0
            morton[slot, :keep] = morton_codes((local / extent).astype(np.float64), self.morton_bits)[:keep]

        return {
            "desc": desc,
            "valid": valid,
            "count": count,
            "occ": (count > 0).astype(np.float32),
            "mu": mu,
            "quat": quat,
            "eta": eta,
            "frame": frame,
            "morton": morton,
            "num_primitives": np.int64(centers.shape[0]),
        }

    def tokens_to_object(self, token):
        desc = token["desc"]
        valid = token["valid"] > 0.5
        counts = token["count"]
        centers, scales, rotations, opacities, appearances = [], [], [], [], []
        for slot in range(self.num_slots):
            taken = min(int(counts[slot]), self.capacity_k)
            if taken == 0:
                continue
            rows = desc[slot, :taken]
            rot = torch.from_numpy(token["frame"][slot]).double()
            slot_quat = mat_to_quat(rot.unsqueeze(0)).squeeze(0)
            extent = float(token["eta"][slot])
            local = torch.from_numpy(rows[:, 0:3]).double()
            local_scale = torch.from_numpy(rows[:, 3:6]).double()
            local_quat = torch.from_numpy(rows[:, 6:10]).double()
            centers.append((local @ rot.transpose(-1, -2)).numpy() + token["mu"][slot])
            scales.append(local_scale.numpy() * extent)
            rotations.append(quat_mul(slot_quat.expand_as(local_quat), local_quat).numpy())
            opacities.append(rows[:, 10:11].astype(np.float64))
            appearances.append(rows[:, 11:14].astype(np.float64))
        if not centers:
            empty = np.zeros((0, 3), dtype=np.float64)
            return {
                "centers": empty,
                "scales": np.zeros((0, 3), dtype=np.float64),
                "rotations": np.zeros((0, 4), dtype=np.float64),
                "opacities": np.zeros((0, 1), dtype=np.float64),
                "appearances": np.zeros((0, 3), dtype=np.float64),
            }
        return {
            "centers": np.concatenate(centers, axis=0).astype(np.float32),
            "scales": np.concatenate(scales, axis=0).astype(np.float32),
            "rotations": np.concatenate(rotations, axis=0).astype(np.float32),
            "opacities": np.concatenate(opacities, axis=0).astype(np.float32),
            "appearances": np.concatenate(appearances, axis=0).astype(np.float32),
        }

    def state_dict(self):
        return {
            "anchors": self.anchors,
            "capacity_k": self.capacity_k,
            "morton_bits": self.morton_bits,
            "eta_min": self.eta_min,
        }

    @classmethod
    def from_state(cls, state):
        return cls(state["anchors"], state["capacity_k"], state["morton_bits"], state["eta_min"])
