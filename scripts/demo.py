import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import apply_cli_overrides, load_config, save_config
from token2splat.data.dataset import write_dataset
from token2splat.data.objects import is_valid_object
from token2splat.data.synthetic import CLASS_NAMES, build_synthetic_records
from token2splat.eval import evaluate_class, write_csv
from token2splat.inference import generate_objects, load_pipeline, reconstruct_objects
from token2splat.io import save_gaussians_npz, save_gaussians_ply, save_json, save_render_grid
from token2splat.render.cameras import build_cameras
from token2splat.render.splat import render_views_detached
from token2splat.tokenizer.tokenizer import CanonicalPatchTokenizer
from token2splat.train import load_splits, normalize_record, resolve_device, train_flow, train_vae


def main():
    parser = argparse.ArgumentParser(description="End-to-end demo: tokenizer, stage 1, stage 2, reconstruction, generation, evaluation")
    parser.add_argument("--config", default="configs/synthetic_smoke.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.set)
    cfg.device = resolve_device(args.device or cfg.device)
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.data.root = os.path.join(args.output_dir, "data")

    data_dir = cfg.data.root
    records = [
        record
        for record in build_synthetic_records(cfg.data.num_scenes, cfg.data.objects_per_scene, cfg.data.primitives_per_object, cfg.seed)
        if is_valid_object(record["object"], cfg.data.min_primitives, cfg.data.min_extent)
    ]
    write_dataset(data_dir, records, CLASS_NAMES, cfg.data.train_ratio, cfg.data.val_ratio, cfg.seed)
    print(f"[demo] dataset written to {data_dir} with {len(records)} instances")

    splits, class_names = load_splits(cfg, data_dir)
    tokenizer = CanonicalPatchTokenizer.from_records(splits["train"], cfg.token)
    stage1_last, stage1_best, _ = train_vae(cfg, splits, tokenizer, cfg.device, os.path.join(cfg.output_dir, "stage1"), class_names)
    stage2_last, stage2_best, _ = train_flow(cfg, splits, tokenizer, stage1_best, cfg.device, os.path.join(cfg.output_dir, "stage2"), class_names)

    cfg, ckpt, tokenizer, vae, flow = load_pipeline(stage2_best, cfg.device, config=cfg)
    splits, class_names = load_splits(cfg, data_dir)
    cameras = build_cameras(cfg.render, cfg.render.supervision_views, cfg.device)

    recon_dir = os.path.join(cfg.output_dir, "reconstruction")
    targets = splits["test"][:4]
    for record, result in zip(targets, reconstruct_objects(vae, tokenizer, targets, cfg.device)):
        obj = result["object"]
        save_gaussians_npz(os.path.join(recon_dir, "objects", result["object_id"] + ".npz"), obj)
        save_gaussians_ply(os.path.join(recon_dir, "objects", result["object_id"] + ".ply"), obj)
    grid = [view for record in targets for view in render_views_detached(normalize_record(record), cameras, cfg.render)]
    if grid:
        save_render_grid(os.path.join(recon_dir, "ground_truth_grid.png"), grid, columns=cfg.render.supervision_views)
    print(f"[demo] reconstruction artifacts written to {recon_dir}")

    gen_dir = os.path.join(cfg.output_dir, "generated")
    items = generate_objects(vae, flow, cfg, sorted({int(r["label"]) for r in splits["test"]}), args.samples, args.steps, cfg.evaluation.seed, cfg.device)
    grid = []
    for item in items:
        name = f"class{item['class_id']:02d}_sample{item['sample_index']:02d}"
        save_gaussians_npz(os.path.join(gen_dir, "objects", name + ".npz"), item["object"], {"label": item["class_id"]})
        save_gaussians_ply(os.path.join(gen_dir, "objects", name + ".ply"), item["object"])
        grid.extend(render_views_detached(item["object"], cameras, cfg.render))
    if grid:
        save_render_grid(os.path.join(gen_dir, "generated_grid.png"), grid, columns=cfg.render.supervision_views)
    print(f"[demo] generation artifacts written to {gen_dir}")

    eval_dir = os.path.join(cfg.output_dir, "evaluation")
    rows = []
    for class_id in sorted({int(record["label"]) for record in splits["test"]}):
        reference = [normalize_record(r) for r in splits["test"] if int(r["label"]) == class_id]
        training = [normalize_record(r) for r in splits["train"] if int(r["label"]) == class_id]
        generated = [item["object"] for item in items if item["class_id"] == class_id]
        row = evaluate_class(generated, reference, training, cameras, cfg.evaluation, cfg.render)
        row["class"] = class_names[class_id]
        rows.append(row)
        print(f"[demo] {class_names[class_id]:>12} MMD {row['mmd_cd']:.5f} Cov {row['coverage_cd']:.3f} MVC {row['mvc']:.3f}")
    write_csv(rows, os.path.join(eval_dir, "class_metrics.csv"))
    save_json(os.path.join(eval_dir, "evaluation.json"), {"per_class": rows})
    save_config(cfg, os.path.join(cfg.output_dir, "resolved_config.yaml"))
    print(f"[demo] pipeline complete, artifacts under {cfg.output_dir}")


if __name__ == "__main__":
    main()
