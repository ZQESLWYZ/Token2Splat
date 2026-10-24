import json
import os

import torch

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import save_config
from ..data.dataset import TokenDataset, make_dataloader
from ..eval.metrics import chamfer_squared, sample_points, ssim
from ..losses import vae_loss
from ..models.vae import HierarchicalVAE, decoded_to_objects
from ..random_tools import set_seed, torch_generator
from ..render.cameras import build_cameras
from .common import batch_to_device, gt_objects_from_batch, history_path, render_views_detached


def evaluate_reconstruction(model, tokenizer, records, device, cfg, limit=4):
    records = list(records)
    if not records:
        return 0.0, {"recon_cd": 0.0, "recon_ssim": 0.0}
    if limit is not None:
        records = records[: max(int(limit), 1)]
    cameras = build_cameras(cfg.render, min(2, max(cfg.render.supervision_views, 1)), device)
    loader = make_dataloader(TokenDataset(records, tokenizer, cache=True), 1, shuffle=False)
    model.eval()
    losses, chamfers, similarities = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            decoded, posterior = model(batch, deterministic=True)
            total, _ = vae_loss(decoded, posterior, batch, cfg.vae)
            losses.append(float(total))
            predicted = decoded_to_objects(decoded, model.ground_truth_keep_mask(batch))[0]
            target = gt_objects_from_batch(batch)[0]
            chamfers.append(float(chamfer_squared(sample_points(predicted, 2048), sample_points(target, 2048))))
            views_pred = render_views_detached(predicted, cameras, cfg.render)
            views_gt = render_views_detached(target, cameras, cfg.render)
            similarities.append(float(torch.stack([ssim(a["rgb"], b["rgb"]) for a, b in zip(views_pred, views_gt)]).mean()))
    return (
        float(sum(losses) / max(len(losses), 1)),
        {
            "recon_cd": float(sum(chamfers) / max(len(chamfers), 1)),
            "recon_ssim": float(sum(similarities) / max(len(similarities), 1)),
        },
    )


def train_vae(cfg, splits, tokenizer, device="cpu", work_dir=None, class_names=None, log=print, cache_tokens=False, resume_path=None):
    set_seed(cfg.seed)
    device = device if device is not None else cfg.device
    work_dir = work_dir if work_dir is not None else os.path.join(cfg.output_dir, "stage1")
    os.makedirs(work_dir, exist_ok=True)
    save_config(cfg, os.path.join(work_dir, "config.yaml"))

    model = HierarchicalVAE(cfg.model, tokenizer.capacity_k, tokenizer.anchors).to(device)
    if resume_path is not None:
        model.load_state_dict(load_checkpoint(resume_path)["vae"])

    loader = make_dataloader(TokenDataset(splits["train"], tokenizer, cache=cache_tokens), cfg.vae.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.vae.lr, weight_decay=cfg.vae.weight_decay)
    generator = torch_generator(cfg.seed, device)

    history = []
    best = float("inf")
    best_path = os.path.join(work_dir, "vae_best.pt")
    last_path = os.path.join(work_dir, "vae_last.pt")
    tokenizer_state = tokenizer.state_dict()

    for epoch in range(cfg.vae.epochs):
        model.train()
        running, parts, steps = 0.0, {}, 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            decoded, posterior = model(batch, generator=generator)
            total, part = vae_loss(decoded, posterior, batch, cfg.vae)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.vae.clip_norm)
            optimizer.step()
            running += float(total)
            steps += 1
            for key, value in part.items():
                parts[key] = parts.get(key, 0.0) + float(value)
        train_loss = running / max(steps, 1)

        val_loss, extras = evaluate_reconstruction(model, tokenizer, splits.get("val", []), device, cfg, limit=4)
        summary = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        summary.update(extras)
        summary.update({f"{key}_mean": parts.get(key, 0.0) / max(steps, 1) for key in parts})
        history.append(summary)
        log(
            f"[stage1] epoch {epoch + 1:03d}/{cfg.vae.epochs} loss {train_loss:.6f} "
            f"val {val_loss:.6f} CD {extras['recon_cd']:.6f} SSIM {extras['recon_ssim']:.4f}"
        )
        if val_loss < best:
            best = val_loss
            save_checkpoint(best_path, cfg, tokenizer_state, model.state_dict(), class_names=class_names, extra={"epoch": epoch + 1})

    save_checkpoint(last_path, cfg, tokenizer_state, model.state_dict(), class_names=class_names, extra={"epoch": cfg.vae.epochs})
    with open(history_path(work_dir), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return last_path, best_path, history
