import numpy as np

import torch
import torch.nn as nn

from .blocks import AnchorEncoding, AttentionPool, MLP, ResidualSelfAttention, scalar_embedding


class GlobalLocalRectifiedFlow(nn.Module):
    def __init__(self, cfg, anchors):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("anchors", torch.as_tensor(np.asarray(anchors, dtype=np.float32)))
        self.pe = AnchorEncoding(cfg.dim_pe)
        self.pe_proj = nn.Linear(cfg.dim_pe, cfg.dim_local)
        self.local_in = nn.Linear(cfg.dim_local, cfg.flow_dim)
        self.local_blocks = nn.ModuleList([ResidualSelfAttention(cfg.flow_dim) for _ in range(cfg.flow_blocks)])
        self.structural_pool = AttentionPool(cfg.flow_dim)
        self.time_mlp = MLP([cfg.time_dim, cfg.time_dim * 2, cfg.time_dim])
        self.class_embedding = nn.Embedding(cfg.num_classes, cfg.class_dim)
        self.global_in = nn.Linear(cfg.dim_global + cfg.flow_dim + cfg.time_dim + cfg.class_dim, cfg.flow_dim)
        self.global_ffn = MLP([cfg.flow_dim, cfg.flow_dim * 2, cfg.flow_dim])
        self.head_global_velocity = nn.Linear(cfg.flow_dim, cfg.dim_global)
        self.film = nn.Linear(cfg.flow_dim, cfg.flow_dim * 2)
        self.head_local_velocity = nn.Linear(cfg.flow_dim, cfg.dim_local)

    def forward(self, z_global, z_patch, time, label):
        time_features = self.time_mlp(scalar_embedding(time, self.cfg.time_dim))
        class_features = self.class_embedding(label)
        tokens = self.local_in(z_patch + self.pe_proj(self.pe(self.anchors)).unsqueeze(0))
        for block in self.local_blocks:
            tokens = block(tokens)
        structural = self.structural_pool(tokens)
        hidden = self.global_in(torch.cat([z_global, structural, time_features, class_features], dim=-1))
        hidden = hidden + self.global_ffn(hidden)
        global_velocity = self.head_global_velocity(hidden)
        gamma, beta = self.film(hidden).chunk(2, dim=-1)
        tokens = (1.0 + gamma.unsqueeze(1)) * tokens + beta.unsqueeze(1)
        patch_velocity = self.head_local_velocity(tokens)
        return global_velocity, patch_velocity


def rectified_flow_bridge(z_global_0, z_patch_0, z_global_1, z_patch_1, time):
    t = time.view(-1, 1)
    z_global_t = (1.0 - t) * z_global_0 + t * z_global_1
    tp = time.view(-1, 1, 1)
    z_patch_t = (1.0 - tp) * z_patch_0 + tp * z_patch_1
    target_global = z_global_1 - z_global_0
    target_patch = z_patch_1 - z_patch_0
    return z_global_t, z_patch_t, target_global, target_patch


def endpoint_estimate(z_global_t, z_patch_t, velocity_global, velocity_patch, time):
    scale = (1.0 - time).view(-1, 1)
    scale3 = (1.0 - time).view(-1, 1, 1)
    return z_global_t + scale * velocity_global, z_patch_t + scale3 * velocity_patch
