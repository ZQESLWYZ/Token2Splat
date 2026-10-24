import math

import torch
import torch.nn as nn


def sinusoidal_encoding(coords, bands=8):
    freqs = (2.0 ** torch.arange(bands, dtype=torch.float32, device=coords.device)) * math.pi
    angles = coords.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)


def scalar_embedding(value, dim):
    half = max(dim // 2, 1)
    freqs = (2.0 ** torch.arange(half, dtype=torch.float32, device=value.device)) * math.pi
    angles = value.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def zero_last_layer(linear, bias_value=None):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        if bias_value is None:
            nn.init.zeros_(linear.bias)
        else:
            with torch.no_grad():
                linear.bias.copy_(torch.as_tensor(bias_value, dtype=linear.bias.dtype, device=linear.bias.device))
    return linear


class AnchorEncoding(nn.Module):
    def __init__(self, out_dim, bands=8):
        super().__init__()
        self.bands = bands
        self.proj = nn.Linear(3 * 2 * bands, out_dim)

    def forward(self, coords):
        return self.proj(sinusoidal_encoding(coords.float(), self.bands))


class MLP(nn.Module):
    def __init__(self, widths, norm=True, act="gelu", dropout=0.0):
        super().__init__()
        layers = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                if norm:
                    layers.append(nn.LayerNorm(widths[index + 1]))
                layers.append(nn.GELU() if act == "gelu" else nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        normalized = self.norm_attn(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, heads=4):
        super().__init__()
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_key = nn.LayerNorm(key_dim)
        self.attn = nn.MultiheadAttention(query_dim, heads, batch_first=True, kdim=key_dim, vdim=key_dim)
        self.out_norm = nn.LayerNorm(query_dim)

    def forward(self, query, context):
        out = self.attn(self.norm_query(query), self.norm_key(context), self.norm_key(context), need_weights=False)[0]
        return self.out_norm(out)


class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        queries = self.query.expand(x.shape[0], -1, -1)
        out = self.attn(self.norm(queries), self.norm(x), self.norm(x), need_weights=False)[0]
        return out[:, 0]
