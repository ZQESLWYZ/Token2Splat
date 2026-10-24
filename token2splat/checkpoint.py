import os

import torch

from .config import config_from_dict, config_to_dict


def save_checkpoint(path, cfg, tokenizer_state, vae_state=None, flow_state=None, class_names=None, extra=None):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "config": config_to_dict(cfg),
        "tokenizer": tokenizer_state,
        "vae": vae_state,
        "flow": flow_state,
        "class_names": list(class_names) if class_names is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_config(checkpoint):
    return config_from_dict(checkpoint["config"])
