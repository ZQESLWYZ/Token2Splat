import torch
import torch.nn.functional as F

from .eval.metrics import ssim
from .features import gaussian_covariance, quat_distance, quat_mul, quat_to_matrix


def patch_targets(batch):
    desc = batch["desc"]
    frame = quat_to_matrix(batch["quat"])
    capacity = desc.shape[2]
    centers = desc[..., 0:3] @ frame.transpose(-1, -2) + batch["mu"].unsqueeze(2)
    scales = desc[..., 3:6] * batch["eta"].unsqueeze(-1).unsqueeze(-1)
    rotations = quat_mul(batch["quat"].unsqueeze(2).expand(-1, -1, capacity, -1), desc[..., 6:10])
    opacities = desc[..., 10:11]
    appearances = desc[..., 11:14]
    return centers, scales, rotations, opacities, appearances, batch["valid"] > 0.5


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def kl_divergence(posterior):
    def term(mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)

    global_kl = term(posterior.global_mu, posterior.global_logvar).sum(dim=-1).mean()
    patch_kl = term(posterior.patch_mu, posterior.patch_logvar).sum(dim=(-1, -2)).mean()
    return global_kl + patch_kl


def vae_reconstruction_loss(decoded, batch, cfg):
    gt_centers, gt_scales, gt_rotations, gt_opacities, gt_appearances, mask = patch_targets(batch)
    occupied = batch["occ"] > 0.5

    center_err = masked_mean((decoded.centers - gt_centers).abs().sum(dim=-1), mask)
    cov_pred = gaussian_covariance(decoded.scales, decoded.rotations)
    cov_gt = gaussian_covariance(gt_scales, gt_rotations)
    cov_err = masked_mean((cov_pred - cov_gt).abs().sum(dim=(-1, -2)), mask)
    opacity_err = masked_mean((decoded.opacities - gt_opacities).abs().sum(dim=-1), mask)
    appearance_err = masked_mean((decoded.appearances - gt_appearances).abs().sum(dim=-1), mask)

    frame_center_err = masked_mean((decoded.frame_centers - batch["mu"]).abs().sum(dim=-1), occupied)
    frame_rotation_err = masked_mean(quat_distance(decoded.frame_quat, batch["quat"]), occupied)
    frame_extent_err = masked_mean((decoded.frame_extent.squeeze(-1) - batch["eta"]).abs(), occupied)

    mask_err = F.binary_cross_entropy_with_logits(decoded.valid_logits, mask.to(decoded.valid_logits.dtype))
    count_classes = decoded.count_logits.shape[-1]
    count_err = F.cross_entropy(
        decoded.count_logits.reshape(-1, count_classes),
        batch["count"].clamp(max=count_classes - 1).reshape(-1),
    )

    parts = {
        "center": float(center_err),
        "covariance": float(cov_err),
        "opacity": float(opacity_err),
        "appearance": float(appearance_err),
        "frame_center": float(frame_center_err),
        "frame_rotation": float(frame_rotation_err),
        "frame_extent": float(frame_extent_err),
        "mask": float(mask_err),
        "count": float(count_err),
    }
    total = (
        cfg.lambda_x * center_err
        + cfg.lambda_cov * cov_err
        + cfg.lambda_opacity * opacity_err
        + cfg.lambda_appearance * appearance_err
        + cfg.lambda_frame * frame_center_err
        + cfg.lambda_rotation * frame_rotation_err
        + cfg.lambda_extent * frame_extent_err
        + cfg.lambda_mask * mask_err
        + cfg.lambda_count * count_err
    )
    return total, parts


def vae_loss(decoded, posterior, batch, cfg):
    total, parts = vae_reconstruction_loss(decoded, batch, cfg)
    kl = kl_divergence(posterior)
    parts["kl"] = float(kl)
    return total + cfg.lambda_kl * kl, parts


def density_weights(batch, cfg, visibility):
    count = batch["count"].to(torch.float32)
    eta = batch["eta"].clamp_min(1e-3)
    opacity = (batch["desc"][..., 10] * batch["valid"]).sum(dim=-1) / count.clamp_min(1.0)
    density = count / eta.pow(3)
    density = density / density.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    score = cfg.density_alpha * opacity + cfg.density_beta * density + cfg.density_gamma * visibility.unsqueeze(0)
    return torch.softmax(score, dim=-1)


def flow_loss(predicted_global, predicted_patch, target_global, target_patch, weights, cfg):
    global_term = (predicted_global - target_global).pow(2).sum(dim=-1).mean()
    patch_term = (((predicted_patch - target_patch).pow(2).sum(dim=-1)) * weights).sum(dim=-1).mean()
    total = cfg.lambda_global * global_term + cfg.lambda_local * patch_term
    return total, {"flow_global": float(global_term), "flow_patch": float(patch_term)}


def render_supervision_loss(predicted_views, target_views, cfg):
    l1_terms, ssim_terms = [], []
    for predicted, target in zip(predicted_views, target_views):
        l1_terms.append((predicted["rgb"] - target["rgb"]).abs().mean())
        ssim_terms.append(1.0 - ssim(predicted["rgb"], target["rgb"]))
    l1 = torch.stack(l1_terms).mean()
    ssim_term = torch.stack(ssim_terms).mean()
    total = cfg.lambda_l1 * l1 + cfg.lambda_ssim * ssim_term
    return total, {"render_l1": float(l1), "render_ssim": float(ssim_term)}


def reconstruction_metrics(decoded, batch):
    mask = batch["valid"] > 0.5
    gt_centers = patch_targets(batch)[0]
    from .eval.metrics import chamfer_squared, sample_points

    errors = []
    for index in range(decoded.centers.shape[0]):
        rows = mask[index].reshape(-1).nonzero(as_tuple=True)[0]
        predicted = decoded.centers[index].reshape(-1, 3)[rows]
        target = gt_centers[index].reshape(-1, 3)[rows]
        errors.append(chamfer_squared(sample_points({"centers": predicted}, 2048), sample_points({"centers": target}, 2048)))
    return float(torch.stack(errors).mean())
