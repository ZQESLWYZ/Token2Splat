from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import FEATURE_DIM, normalize_quat, quat_mul, quat_to_matrix
from .blocks import AnchorEncoding, AttentionPool, CrossAttention, MLP, ResidualSelfAttention, zero_last_layer

LOGVAR_MIN = -8.0
LOGVAR_MAX = 6.0


@dataclass
class Posterior:
    global_z: torch.Tensor
    patch_z: torch.Tensor
    global_mu: torch.Tensor
    global_logvar: torch.Tensor
    patch_mu: torch.Tensor
    patch_logvar: torch.Tensor


@dataclass
class Decoded:
    centers: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    appearances: torch.Tensor
    frame_centers: torch.Tensor
    frame_quat: torch.Tensor
    frame_extent: torch.Tensor
    valid_logits: torch.Tensor
    count_logits: torch.Tensor


def _sample(mu, logvar, generator=None):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + std * eps


def _clamp_logvar(logvar):
    return logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


class PatchEncoder(nn.Module):
    def __init__(self, dim_hidden, meta_dim=6):
        super().__init__()
        self.point = MLP([FEATURE_DIM + 1, dim_hidden, dim_hidden])
        self.slot = MLP([dim_hidden * 2 + meta_dim, dim_hidden, dim_hidden])
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, desc, valid, meta):
        features = self.point(torch.cat([desc, valid.unsqueeze(-1)], dim=-1))
        weighted = features * valid.unsqueeze(-1)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = weighted.sum(dim=-2) / denom
        filled = weighted.masked_fill(valid.unsqueeze(-1) == 0, torch.finfo(features.dtype).min)
        peak = filled.max(dim=-2).values
        peak = torch.where((valid.sum(dim=-1) > 0).unsqueeze(-1), peak, torch.zeros_like(peak))
        return self.norm(self.slot(torch.cat([mean, peak, meta], dim=-1)))


class GlobalDecoder(nn.Module):
    def __init__(self, cfg, num_slots):
        super().__init__()
        self.context_tokens = cfg.context_tokens
        self.upscale = MLP([cfg.dim_global, cfg.dim_hidden * 2, cfg.dim_hidden * cfg.context_tokens])
        self.context_norm = nn.LayerNorm(cfg.dim_hidden)
        self.slot_pe = AnchorEncoding(cfg.dim_hidden)
        self.slot_bias = nn.Linear(cfg.dim_global, cfg.dim_hidden)
        self.blocks = nn.ModuleList([ResidualSelfAttention(cfg.dim_hidden) for _ in range(cfg.decoder_blocks)])
        self.head_center = zero_last_layer(nn.Linear(cfg.dim_hidden, 3))
        self.head_quat = zero_last_layer(nn.Linear(cfg.dim_hidden, 4), (1.0, 0.0, 0.0, 0.0))
        self.head_extent = zero_last_layer(nn.Linear(cfg.dim_hidden, 1), -2.0)

    def forward(self, z_global, anchors):
        batch = z_global.shape[0]
        context = self.context_norm(self.upscale(z_global).view(batch, self.context_tokens, -1))
        slots = self.slot_pe(anchors).unsqueeze(0).expand(batch, -1, -1) + self.slot_bias(z_global).unsqueeze(1)
        for block in self.blocks:
            slots = block(slots)
        return context, slots


class LocalDecoder(nn.Module):
    def __init__(self, cfg, capacity_k):
        super().__init__()
        self.capacity_k = capacity_k
        self.pe = AnchorEncoding(cfg.dim_pe)
        self.pe_proj = nn.Linear(cfg.dim_pe, cfg.dim_local)
        self.cross = CrossAttention(cfg.dim_local, cfg.dim_hidden, heads=4)
        self.film = nn.Linear(cfg.dim_hidden, cfg.dim_local * 2)
        self.blocks = nn.ModuleList([ResidualSelfAttention(cfg.dim_local) for _ in range(cfg.decoder_blocks)])
        self.head_attrs = nn.Linear(cfg.dim_local, capacity_k * FEATURE_DIM)
        self.head_mask = nn.Linear(cfg.dim_local, capacity_k)
        self.head_count = nn.Linear(cfg.dim_local, capacity_k + 1)
        zero_last_layer(self.head_attrs)
        with torch.no_grad():
            bias = self.head_attrs.bias.view(self.capacity_k, FEATURE_DIM)
            bias[:, 6] = 1.0
            bias[:, 10] = 1.0
    def forward(self, z_patch, context, anchors):
        encoded = z_patch + self.pe_proj(self.pe(anchors)).unsqueeze(0)
        attended = self.cross(encoded, context)
        gamma, beta = self.film(context.mean(dim=1)).chunk(2, dim=-1)
        attended = (1.0 + gamma.unsqueeze(1)) * attended + beta.unsqueeze(1)
        for block in self.blocks:
            attended = block(attended)
        return attended


class HierarchicalVAE(nn.Module):
    def __init__(self, cfg, capacity_k, anchors):
        super().__init__()
        self.cfg = cfg
        self.capacity_k = int(capacity_k)
        anchors = torch.as_tensor(np.asarray(anchors, dtype=np.float32))
        self.register_buffer("anchors", anchors)
        self.num_slots = int(anchors.shape[0])
        self.patch_encoder = PatchEncoder(cfg.dim_hidden)
        self.slot_pe = AnchorEncoding(cfg.dim_hidden)
        self.global_blocks = nn.ModuleList([ResidualSelfAttention(cfg.dim_hidden) for _ in range(cfg.encoder_blocks)])
        self.global_pool = AttentionPool(cfg.dim_hidden)
        self.local_refine = MLP([cfg.dim_hidden * 2, cfg.dim_hidden])
        self.head_global = nn.Linear(cfg.dim_hidden, cfg.dim_global * 2)
        self.head_patch = nn.Linear(cfg.dim_hidden + cfg.dim_global, cfg.dim_local * 2)
        self.global_decoder = GlobalDecoder(cfg, self.num_slots)
        self.local_decoder = LocalDecoder(cfg, self.capacity_k)

    def _meta(self, batch):
        return torch.cat(
            [
                torch.log(batch["eta"] + 1e-6).unsqueeze(-1),
                (batch["count"].to(self.anchors.dtype) / float(self.capacity_k)).unsqueeze(-1),
                batch["occ"].unsqueeze(-1),
                batch["mu"] - self.anchors.unsqueeze(0),
            ],
            dim=-1,
        )

    def features(self, batch):
        hidden = self.patch_encoder(batch["desc"], batch["valid"], self._meta(batch))
        hidden = hidden + self.slot_pe(self.anchors).unsqueeze(0)
        for block in self.global_blocks:
            hidden = block(hidden)
        pooled = self.global_pool(hidden)
        refined = self.local_refine(torch.cat([hidden, pooled.unsqueeze(1).expand(-1, hidden.shape[1], -1)], dim=-1))
        return hidden, pooled, refined

    def encode(self, batch, deterministic=False, generator=None):
        _, pooled, refined = self.features(batch)
        params = self.head_global(pooled)
        global_mu, global_logvar = params.chunk(2, dim=-1)
        global_logvar = _clamp_logvar(global_logvar)
        global_z = global_mu if deterministic else _sample(global_mu, global_logvar, generator)

        conditioned = torch.cat([refined, global_z.unsqueeze(1).expand(-1, refined.shape[1], -1)], dim=-1)
        patch_params = self.head_patch(conditioned)
        patch_mu, patch_logvar = patch_params.chunk(2, dim=-1)
        patch_logvar = _clamp_logvar(patch_logvar)
        patch_z = patch_mu if deterministic else _sample(patch_mu, patch_logvar, generator)
        return Posterior(global_z, patch_z, global_mu, global_logvar, patch_mu, patch_logvar)

    def decode(self, global_z, patch_z):
        context, slot_features = self.global_decoder(global_z, self.anchors)
        frame_centers = self.anchors.unsqueeze(0) + self.global_decoder.head_center(slot_features)
        frame_quat = normalize_quat(self.global_decoder.head_quat(slot_features))
        frame_extent = F.softplus(self.global_decoder.head_extent(slot_features)).clamp_min(1e-3)

        attended = self.local_decoder(patch_z, context, self.anchors)
        batch, slots = attended.shape[0], attended.shape[1]
        attrs = self.local_decoder.head_attrs(attended).view(batch, slots, self.capacity_k, FEATURE_DIM)
        valid_logits = self.local_decoder.head_mask(attended)
        count_logits = self.local_decoder.head_count(attended)

        local_centers = attrs[..., 0:3]
        local_scales = F.softplus(attrs[..., 3:6])
        local_rotations = normalize_quat(attrs[..., 6:10])
        opacities = torch.sigmoid(attrs[..., 10:11])
        appearances = torch.sigmoid(attrs[..., 11:14])

        frame_matrix = quat_to_matrix(frame_quat)
        centers = local_centers @ frame_matrix.transpose(-1, -2) + frame_centers.unsqueeze(2)
        scales = local_scales * frame_extent.unsqueeze(-1)
        rotations = quat_mul(frame_quat.unsqueeze(2).expand(-1, -1, self.capacity_k, -1), local_rotations)
        return Decoded(
            centers=centers,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            appearances=appearances,
            frame_centers=frame_centers,
            frame_quat=frame_quat,
            frame_extent=frame_extent,
            valid_logits=valid_logits,
            count_logits=count_logits,
        )

    def forward(self, batch, deterministic=False, generator=None):
        posterior = self.encode(batch, deterministic=deterministic, generator=generator)
        decoded = self.decode(posterior.global_z, posterior.patch_z)
        return decoded, posterior

    def predicted_keep_mask(self, decoded):
        counts = decoded.count_logits.argmax(dim=-1)
        order = decoded.valid_logits.argsort(dim=-1, descending=True)
        ranks = torch.arange(self.capacity_k, device=order.device).view(1, 1, -1)
        selected = ranks < counts.unsqueeze(-1)
        return torch.zeros_like(decoded.valid_logits, dtype=torch.bool).scatter_(2, order, selected)

    def ground_truth_keep_mask(self, batch):
        return batch["valid"] > 0.5


def decoded_to_objects(decoded, keep_mask):
    batch = decoded.centers.shape[0]
    flat = {
        "centers": decoded.centers.reshape(batch, -1, 3),
        "scales": decoded.scales.reshape(batch, -1, 3),
        "rotations": decoded.rotations.reshape(batch, -1, 4),
        "opacities": decoded.opacities.reshape(batch, -1, 1),
        "appearances": decoded.appearances.reshape(batch, -1, 3),
    }
    mask = keep_mask.reshape(batch, -1)
    objects = []
    for index in range(batch):
        rows = mask[index].nonzero(as_tuple=True)[0]
        objects.append({key: value[index, rows] for key, value in flat.items()})
    return objects
