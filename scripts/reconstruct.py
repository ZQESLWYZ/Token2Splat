import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from token2splat.config import load_config
from token2splat.eval import chamfer_squared, sample_points, ssim, write_csv
from token2splat.inference import load_pipeline, reconstruct_objects
from token2splat.io import save_gaussians_npz, save_gaussians_ply, save_json, save_render_grid
from token2splat.render.cameras import build_cameras
from token2splat.render.splat import render_views_detached
from token2splat.train import load_splits, normalize_record, resolve_device


def main():
    parser = argparse.ArgumentParser(description="Paired reconstruction: encode a held-out 3DGS object and decode it back")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    device = resolve_device(args.device or (config.device if config else "cpu"))
    cfg, ckpt, tokenizer, vae, _ = load_pipeline(args.checkpoint, device=device, config=config, overrides=args.set)

    splits, class_names = load_splits(cfg, args.data)
    records = splits[args.split]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"split '{args.split}' is empty")

    out = args.out if args.out is not None else os.path.join(cfg.output_dir, "reconstruction")
    cameras = build_cameras(cfg.render, min(4, cfg.render.supervision_views), device)
    results = reconstruct_objects(vae, tokenizer, records, device)

    rows, comparison = [], []
    for index, (record, result) in enumerate(zip(records, results)):
        predicted, target = result["object"], normalize_record(record)
        chamfer = chamfer_squared(
            sample_points(predicted, cfg.evaluation.point_count),
            sample_points(target, cfg.evaluation.point_count),
        )
        views_pred = render_views_detached(predicted, cameras, cfg.render)
        views_gt = render_views_detached(target, cameras, cfg.render)
        similarity = float(torch.stack([ssim(a["rgb"], b["rgb"]) for a, b in zip(views_pred, views_gt)]).mean())
        rows.append(
            {
                "class": class_names[int(record["label"])],
                "object_id": result["object_id"],
                "chamfer": float(chamfer),
                "ssim": similarity,
                "primitives_gt": int(target["centers"].shape[0]),
                "primitives_pred": int(predicted["centers"].shape[0]),
            }
        )
        name = f"{index:02d}_{result['object_id']}"
        save_gaussians_npz(os.path.join(out, "objects", name + ".npz"), predicted, {"object_id": result["object_id"]})
        if not args.no_ply:
            save_gaussians_ply(os.path.join(out, "objects", name + ".ply"), predicted)
        if index < 4:
            comparison.extend(views_gt)
            comparison.extend(views_pred)
        print(f"{result['object_id']:>28} CD {float(chamfer):.6f} SSIM {similarity:.4f} primitives {rows[-1]['primitives_pred']}/{rows[-1]['primitives_gt']}")

    write_csv(rows, os.path.join(out, "reconstruction_metrics.csv"))
    if comparison:
        save_render_grid(os.path.join(out, "comparison_grid.png"), comparison, columns=len(cameras))
    save_json(
        os.path.join(out, "reconstruction_summary.json"),
        {"mean_chamfer": float(sum(r["chamfer"] for r in rows) / len(rows)), "mean_ssim": float(sum(r["ssim"] for r in rows) / len(rows)), "count": len(rows)},
    )
    print(f"reconstruction artifacts written to {out}")


if __name__ == "__main__":
    main()
