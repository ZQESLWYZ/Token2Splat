import torch

FEATURE_KEYS = ("centers", "scales", "rotations", "opacities", "appearances")
FEATURE_SPLITS = (3, 3, 4, 1, 3)
FEATURE_DIM = 14


def pack_features(obj):
    return torch.cat([obj[key] for key in FEATURE_KEYS], dim=-1)


def unpack_features(feat):
    out = {}
    start = 0
    for key, width in zip(FEATURE_KEYS, FEATURE_SPLITS):
        out[key] = feat[..., start:start + width]
        start += width
    return out


def normalize_quat(quat, eps=1e-8):
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_matrix(quat):
    quat = normalize_quat(quat)
    w, x, y, z = quat.unbind(-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def mat_to_quat(mat):
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    trace = m00 + m11 + m22

    s0 = torch.sqrt((trace + 1.0).clamp_min(1e-8)) * 2.0
    q0 = torch.stack([s0 * 0.25, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], dim=-1)
    s1 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-8)) * 2.0
    q1 = torch.stack([(m21 - m12) / s1, s1 * 0.25, (m01 + m10) / s1, (m02 + m20) / s1], dim=-1)
    s2 = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-8)) * 2.0
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, s2 * 0.25, (m12 + m21) / s2], dim=-1)
    s3 = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-8)) * 2.0
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, s3 * 0.25], dim=-1)

    c0 = (trace >= m00) & (trace >= m11) & (trace >= m22)
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)
    quat = torch.where(c3.unsqueeze(-1), q3, torch.where(c2.unsqueeze(-1), q2, torch.where(c1.unsqueeze(-1), q1, q0)))
    return normalize_quat(quat)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(quat):
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_distance(a, b):
    dot = normalize_quat(a) * normalize_quat(b)
    return 1.0 - dot.sum(-1).abs()


def gaussian_covariance(scales, rotations):
    rot = quat_to_matrix(rotations)
    var = scales.pow(2)
    return (rot * var.unsqueeze(-2)) @ rot.transpose(-1, -2)


def object_center_and_scale(centers, eps=1e-4):
    tensor = torch.as_tensor(centers, dtype=torch.float32)
    low = tensor.min(dim=-2).values
    high = tensor.max(dim=-2).values
    return 0.5 * (low + high), (high - low).norm(dim=-1).clamp_min(eps)


def normalize_object(obj, eps=1e-4):
    centers = torch.as_tensor(obj["centers"], dtype=torch.float32)
    center, scale = object_center_and_scale(centers, eps)
    if centers.dim() == 2:
        shift, divisor = center.view(1, 3), scale.view(1, 1)
    else:
        shift, divisor = center, scale.unsqueeze(-1)
    out = {key: torch.as_tensor(obj[key], dtype=torch.float32).clone() for key in FEATURE_KEYS}
    out["centers"] = (centers - shift) / divisor
    out["scales"] = out["scales"] / divisor
    if "label" in obj:
        out["label"] = obj["label"]
    return out
