import numpy as np


def morton_codes(coords, bits):
    span = (2 ** int(bits)) - 1
    scaled = np.clip((coords + 2.0) / 4.0, 0.0, 1.0)
    q = np.clip(np.floor(scaled * span), 0, span).astype(np.int64)
    codes = np.zeros(q.shape[0], dtype=np.int64)
    for bit in range(int(bits)):
        codes |= ((q[:, 0] >> bit) & 1) << (3 * bit)
        codes |= ((q[:, 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((q[:, 2] >> bit) & 1) << (3 * bit + 2)
    return codes


def morton_order(coords, bits):
    codes = morton_codes(coords, bits)
    return np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], codes))
