from dataclasses import dataclass, field, asdict, fields, is_dataclass
import json

import yaml


@dataclass
class TokenCfg:
    num_anchors: int = 256
    capacity_k: int = 32
    morton_bits: int = 10
    eta_min: float = 1e-3
    anchor_pool: int = 60000
    anchor_pad_step: float = 0.02


@dataclass
class ModelCfg:
    num_classes: int = 34
    dim_hidden: int = 128
    dim_global: int = 256
    dim_local: int = 128
    dim_pe: int = 128
    encoder_blocks: int = 2
    decoder_blocks: int = 2
    context_tokens: int = 64
    flow_dim: int = 192
    flow_blocks: int = 4
    class_dim: int = 128
    time_dim: int = 64


@dataclass
class VAETrainCfg:
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    lambda_x: float = 1.0
    lambda_cov: float = 1.0
    lambda_opacity: float = 0.5
    lambda_appearance: float = 1.0
    lambda_frame: float = 0.5
    lambda_rotation: float = 0.1
    lambda_extent: float = 0.1
    lambda_mask: float = 0.5
    lambda_count: float = 0.1
    lambda_kl: float = 1e-4


@dataclass
class FlowTrainCfg:
    epochs: int = 300
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    lambda_global: float = 1.0
    lambda_local: float = 2.0
    lambda_flow: float = 1.0
    lambda_render: float = 0.5
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.2
    render_every: int = 2
    render_samples: int = 2
    density_alpha: float = 1.0
    density_beta: float = 0.5
    density_gamma: float = 0.5


@dataclass
class RenderCfg:
    image_size: int = 256
    fov_deg: float = 42.0
    radius: float = 2.2
    azimuths: int = 8
    elevations: tuple = (15.0, 30.0, 45.0)
    supervision_views: int = 4
    background: float = 1.0
    blur: float = 0.5
    max_primitives: int = 0


@dataclass
class EvalCfg:
    samples_per_class: int = 32
    point_count: int = 2048
    f_tau: float = 0.01
    uniqueness_percentile: float = 5.0
    mvc_max_pairs: int = 12
    mvc_stride: int = 4
    eval_views: int = 24
    seed: int = 1234


@dataclass
class DataCfg:
    root: str = "data/synthetic"
    num_scenes: int = 12
    objects_per_scene: int = 4
    primitives_per_object: int = 900
    min_primitives: int = 64
    min_extent: float = 0.05
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class Config:
    seed: int = 20250726
    device: str = "cpu"
    output_dir: str = "runs"
    token: TokenCfg = field(default_factory=TokenCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    vae: VAETrainCfg = field(default_factory=VAETrainCfg)
    flow: FlowTrainCfg = field(default_factory=FlowTrainCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    evaluation: EvalCfg = field(default_factory=EvalCfg)
    data: DataCfg = field(default_factory=DataCfg)


def _build(cls, payload):
    kwargs = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        if is_dataclass(f.type):
            kwargs[f.name] = _build(f.type, value)
        elif f.name == "elevations" and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def config_from_dict(payload):
    return _build(Config, payload)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return config_from_dict(yaml.safe_load(handle))


def config_to_dict(cfg):
    return json.loads(json.dumps(asdict(cfg), default=str))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, sort_keys=False)
    return path


def apply_cli_overrides(cfg, overrides):
    for item in overrides:
        name, raw = item.split("=", 1)
        node = cfg
        parts = name.split(".")
        for part in parts[:-1]:
            node = getattr(node, part)
        current = getattr(node, parts[-1])
        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(node, parts[-1], value)
    return cfg
