import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import load_config
from token2splat.inference import generate_objects, load_pipeline
from token2splat.io import save_gaussians_npz, save_gaussians_ply, save_json, save_render_grid
from token2splat.render.cameras import build_cameras
from token2splat.render.splat import render_views_detached
from token2splat.train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Inference: sample class-conditional 3D Gaussian objects with the rectified flow")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grid-views", type=int, default=4)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    device = resolve_device(args.device or (config.device if config else "cpu"))
    cfg, ckpt, tokenizer, vae, flow = load_pipeline(args.checkpoint, device=device, config=config, overrides=args.set)
    if flow is None:
        raise SystemExit("The checkpoint has no flow weights; run scripts/train_stage2.py before generating.")

    class_names = ckpt.get("class_names") or [f"class_{index}" for index in range(cfg.model.num_classes)]
    class_ids = args.classes if args.classes else list(range(len(class_names)))
    samples = args.samples if args.samples is not None else cfg.evaluation.samples_per_class
    seed = args.seed if args.seed is not None else cfg.evaluation.seed
    out = args.out if args.out is not None else os.path.join(cfg.output_dir, "generated")
    cameras = build_cameras(cfg.render, args.grid_views, device)

    manifest = {"checkpoint": args.checkpoint, "steps": args.steps, "seed": seed, "samples_per_class": samples, "classes": {}}
    for class_id in class_ids:
        started = time.perf_counter()
        items = generate_objects(vae, flow, cfg, [class_id], samples, args.steps, seed, device)
        elapsed = time.perf_counter() - started
        class_dir = os.path.join(out, f"class{class_id:02d}_{class_names[class_id]}")
        os.makedirs(class_dir, exist_ok=True)
        render_images = []
        for item in items:
            name = f"seed{seed}_sample{item['sample_index']:03d}"
            save_gaussians_npz(os.path.join(class_dir, name + ".npz"), item["object"], {"label": item["class_id"], "seed": seed})
            if not args.no_ply:
                save_gaussians_ply(os.path.join(class_dir, name + ".ply"), item["object"])
            if len(render_images) < samples:
                render_images.extend(render_views_detached(item["object"], cameras, cfg.render))
        if render_images:
            save_render_grid(os.path.join(class_dir, "grid.png"), render_images, columns=args.grid_views)
        manifest["classes"][class_names[class_id]] = {
            "class_id": class_id,
            "count": len(items),
            "primitives": [int(item["object"]["centers"].shape[0]) for item in items],
            "seconds_per_object": elapsed / max(len(items), 1),
        }
        print(f"class {class_names[class_id]:>12}: {len(items)} objects, {elapsed / max(len(items), 1):.2f}s per object")
    save_json(os.path.join(out, "generation_manifest.json"), manifest)
    print(f"generated artifacts written to {out}")


if __name__ == "__main__":
    main()
