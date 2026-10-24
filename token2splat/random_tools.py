import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_rng(seed):
    return np.random.default_rng(seed)


def torch_generator(seed, device=None):
    gen = torch.Generator(device=device if device is not None else "cpu")
    gen.manual_seed(seed)
    return gen
