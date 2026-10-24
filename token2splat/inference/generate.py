import torch

from ..checkpoint import checkpoint_config, load_checkpoint
from ..config import apply_cli_overrides
from ..data.dataset import TokenDataset, make_dataloader
from ..models.flow import GlobalLocalRectifiedFlow
from ..models.vae import HierarchicalVAE, decoded_to_objects
from ..random_tools import torch_generator
from ..tokenizer.tokenizer import CanonicalPatchTokenizer
from ..train.common import batch_to_device


def euler_integrate(flow, z_global, z_patch, label, steps, device="cpu"):
    dt = 1.0 / max(int(steps), 1)
    for step in range(int(steps)):
        time = torch.full((z_global.shape[0],), step / float(steps), device=device, dtype=z_global.dtype)
        velocity_global, velocity_patch = flow(z_global, z_patch, time, label)
        z_global = z_global + dt * velocity_global
        z_patch = z_patch + dt * velocity_patch
    return z_global, z_patch


def sample_prior_latents(num_samples, dim_global, dim_local, num_slots, generator, device="cpu"):
    z_global = torch.randn(num_samples, dim_global, generator=generator, device=device)
    z_patch = torch.randn(num_samples, num_slots, dim_local, generator=generator, device=device)
    return z_global, z_patch


def generate_objects(vae, flow, cfg, class_ids, samples_per_class, steps=32, seed=0, device="cpu"):
    flow.eval()
    vae.eval()
    generator = torch_generator(seed, device)
    results = []
    with torch.no_grad():
        for class_id in class_ids:
            labels = torch.full((samples_per_class,), int(class_id), dtype=torch.long, device=device)
            z_global, z_patch = sample_prior_latents(
                samples_per_class, cfg.model.dim_global, cfg.model.dim_local, vae.num_slots, generator, device
            )
            z_global, z_patch = euler_integrate(flow, z_global, z_patch, labels, steps, device)
            decoded = vae.decode(z_global, z_patch)
            objects = decoded_to_objects(decoded, vae.predicted_keep_mask(decoded))
            for index, obj in enumerate(objects):
                results.append(
                    {
                        "class_id": int(class_id),
                        "seed": int(seed),
                        "sample_index": index,
                        "object": {key: value.detach() for key, value in obj.items()},
                    }
                )
    return results


def reconstruct_objects(vae, tokenizer, records, device="cpu", batch_size=1):
    vae.eval()
    loader = make_dataloader(TokenDataset(records, tokenizer), batch_size, shuffle=False)
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            decoded, _ = vae(batch, deterministic=True)
            objects = decoded_to_objects(decoded, vae.ground_truth_keep_mask(batch))
            for index, obj in enumerate(objects):
                results.append({"object_id": batch["object_id"][index], "object": obj})
    return results


def load_pipeline(checkpoint_path, device="cpu", config=None, overrides=None):
    ckpt = load_checkpoint(checkpoint_path)
    cfg = config if config is not None else checkpoint_config(ckpt)
    if overrides:
        apply_cli_overrides(cfg, overrides)
    tokenizer = CanonicalPatchTokenizer.from_state(ckpt["tokenizer"])
    vae = HierarchicalVAE(cfg.model, tokenizer.capacity_k, tokenizer.anchors).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    flow = None
    if ckpt.get("flow") is not None:
        flow = GlobalLocalRectifiedFlow(cfg.model, tokenizer.anchors).to(device)
        flow.load_state_dict(ckpt["flow"])
        flow.eval()
    return cfg, ckpt, tokenizer, vae, flow
