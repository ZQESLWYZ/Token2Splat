import json
import os

import torch

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import save_config
from ..data.dataset import TokenDataset, make_dataloader
from ..losses import density_weights, flow_loss, render_supervision_loss
from ..models.flow import GlobalLocalRectifiedFlow, endpoint_estimate, rectified_flow_bridge
from ..models.vae import HierarchicalVAE, decoded_to_objects
from ..random_tools import set_seed, torch_generator
from ..render.cameras import anchor_visibility, build_cameras
from ..render.splat import render_views_grad
from .common import GroundTruthRenderCache, batch_to_device, gt_objects_from_batch, history_path


def validation_flow_loss(flow, vae, records, tokenizer, visibility, cfg, device, generator, max_batches=2):
    if not records:
        return 0.0
    loader = make_dataloader(TokenDataset(records, tokenizer), cfg.flow.batch_size, shuffle=False)
    flow.eval()
    values = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch_to_device(batch, device)
            posterior = vae.encode(batch, deterministic=True)
            z0_global = torch.randn(posterior.global_mu.shape, generator=generator, device=device)
            z0_patch = torch.randn(posterior.patch_mu.shape, generator=generator, device=device)
            time = torch.rand(posterior.global_mu.shape[0], generator=generator, device=device)
            z_global, z_patch, target_global, target_patch = rectified_flow_bridge(
                z0_global, z0_patch, posterior.global_mu, posterior.patch_mu, time
            )
            velocity_global, velocity_patch = flow(z_global, z_patch, time, batch["label"])
            weights = density_weights(batch, cfg.flow, visibility)
            loss, _ = flow_loss(velocity_global, velocity_patch, target_global, target_patch, weights, cfg.flow)
            values.append(float(loss))
    flow.train()
    return float(sum(values) / max(len(values), 1))


def train_flow(cfg, splits, tokenizer, vae_checkpoint, device="cpu", work_dir=None, class_names=None, log=print, cache_tokens=False):
    set_seed(cfg.seed)
    work_dir = work_dir if work_dir is not None else os.path.join(cfg.output_dir, "stage2")
    os.makedirs(work_dir, exist_ok=True)
    save_config(cfg, os.path.join(work_dir, "config.yaml"))

    checkpoint = load_checkpoint(vae_checkpoint)
    vae = HierarchicalVAE(cfg.model, tokenizer.capacity_k, tokenizer.anchors).to(device)
    vae.load_state_dict(checkpoint["vae"])
    vae.requires_grad_(False)
    vae.eval()

    flow = GlobalLocalRectifiedFlow(cfg.model, tokenizer.anchors).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.flow.lr, weight_decay=cfg.flow.weight_decay)
    cameras = build_cameras(cfg.render, cfg.render.supervision_views, device)
    visibility = anchor_visibility(tokenizer.anchors, cameras).to(device)
    gt_cache = GroundTruthRenderCache(cameras, cfg.render)
    generator = torch_generator(cfg.seed, device)
    loader = make_dataloader(TokenDataset(splits["train"], tokenizer, cache=cache_tokens), cfg.flow.batch_size, shuffle=True)

    history = []
    best = float("inf")
    best_path = os.path.join(work_dir, "flow_best.pt")
    last_path = os.path.join(work_dir, "flow_last.pt")
    render_every = max(int(cfg.flow.render_every), 1)

    for epoch in range(cfg.flow.epochs):
        flow.train()
        running, parts, steps = 0.0, {}, 0
        for iteration, batch in enumerate(loader):
            batch = batch_to_device(batch, device)
            with torch.no_grad():
                posterior = vae.encode(batch, deterministic=True)
            target_global_latent, target_patch_latent = posterior.global_mu, posterior.patch_mu
            prior_global = torch.randn(target_global_latent.shape, generator=generator, device=device)
            prior_patch = torch.randn(target_patch_latent.shape, generator=generator, device=device)
            time = torch.rand(target_global_latent.shape[0], generator=generator, device=device)
            z_global, z_patch, target_global, target_patch = rectified_flow_bridge(
                prior_global, prior_patch, target_global_latent, target_patch_latent, time
            )
            velocity_global, velocity_patch = flow(z_global, z_patch, time, batch["label"])
            weights = density_weights(batch, cfg.flow, visibility)
            loss, part = flow_loss(velocity_global, velocity_patch, target_global, target_patch, weights, cfg.flow)
            total = cfg.flow.lambda_flow * loss

            if cfg.flow.lambda_render > 0 and iteration % render_every == 0:
                take = min(int(cfg.flow.render_samples), target_global_latent.shape[0])
                endpoint_global, endpoint_patch = endpoint_estimate(
                    z_global[:take], z_patch[:take], velocity_global[:take], velocity_patch[:take], time[:take]
                )
                decoded = vae.decode(endpoint_global, endpoint_patch)
                predicted = decoded_to_objects(decoded, vae.predicted_keep_mask(decoded))
                targets = gt_objects_from_batch(batch)[:take]
                predicted_views = [view for index in range(take) for view in render_views_grad(predicted[index], cameras, cfg.render)]
                target_views = [view for index in range(take) for view in gt_cache.get(batch["object_id"][index], targets[index])]
                render_loss, render_parts = render_supervision_loss(predicted_views, target_views, cfg.flow)
                total = total + cfg.flow.lambda_render * render_loss
                part.update(render_parts)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.flow.clip_norm)
            optimizer.step()
            running += float(total)
            steps += 1
            for key, value in part.items():
                parts[key] = parts.get(key, 0.0) + float(value)

        train_loss = running / max(steps, 1)
        val_loss = validation_flow_loss(flow, vae, splits.get("val", []), tokenizer, visibility, cfg, device, generator)
        summary = {"epoch": epoch + 1, "train_loss": train_loss, "val_flow_loss": val_loss}
        summary.update({f"{key}_mean": parts.get(key, 0.0) / max(steps, 1) for key in parts})
        history.append(summary)
        log(
            f"[stage2] epoch {epoch + 1:03d}/{cfg.flow.epochs} loss {train_loss:.6f} val_flow {val_loss:.6f} "
            f"render_l1 {parts.get('render_l1', 0.0) / max(steps, 1):.6f}"
        )
        if val_loss < best:
            best = val_loss
            save_checkpoint(
                best_path, cfg, tokenizer.state_dict(), vae.state_dict(), flow.state_dict(),
                class_names=class_names, extra={"epoch": epoch + 1, "vae_checkpoint": vae_checkpoint},
            )
    save_checkpoint(
        last_path, cfg, tokenizer.state_dict(), vae.state_dict(), flow.state_dict(),
        class_names=class_names, extra={"epoch": cfg.flow.epochs, "vae_checkpoint": vae_checkpoint},
    )
    with open(history_path(work_dir), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return last_path, best_path, history
