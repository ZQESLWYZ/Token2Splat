import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import load_config
from token2splat.eval import evaluate_class, write_csv
from token2splat.inference import generate_objects, load_pipeline
from token2splat.io import save_json
from token2splat.render.cameras import build_cameras
from token2splat.train import load_splits, normalize_record, resolve_device

METRIC_KEYS = (
    "mmd_cd",
    "coverage_cd",
    "knn_accuracy",
    "diversity_cd",
    "unique_ratio",
    "novelty_mean",
    "novelty_p05",
    "f_score_tau",
    "mvc",
    "dcs",
    "feature_diversity",
    "feature_frechet",
)


def main():
    parser = argparse.ArgumentParser(description="Class-conditional distributional evaluation of generated 3DGS sets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    device = resolve_device(args.device or (config.device if config else "cpu"))
    cfg, ckpt, tokenizer, vae, flow = load_pipeline(args.checkpoint, device=device, config=config, overrides=args.set)
    if flow is None:
        raise SystemExit("The checkpoint has no flow weights; run scripts/train_stage2.py first.")

    splits, class_names = load_splits(cfg, args.data)
    samples = args.samples if args.samples is not None else cfg.evaluation.samples_per_class
    seed = cfg.evaluation.seed
    out = args.out if args.out is not None else os.path.join(cfg.output_dir, "evaluation")
    cameras = build_cameras(cfg.render, cfg.evaluation.eval_views, device)

    test_records, train_records = splits["test"], splits["train"]
    all_records = splits["test"] + splits["val"] + splits["train"]
    class_ids = args.classes if args.classes else sorted({int(record["label"]) for record in test_records})

    rows = []
    for class_id in class_ids:
        pool = [record for record in test_records if int(record["label"]) == class_id]
        reference_split = "test"
        if not pool:
            pool = [record for record in all_records if int(record["label"]) == class_id]
            reference_split = "train+val+test"
        if not pool:
            print(f"{class_names[class_id]}: no instances available, skipped")
            continue
        reference = [normalize_record(record) for record in pool]
        training = [normalize_record(record) for record in train_records if int(record["label"]) == class_id]
        training = training if training else reference
        items = generate_objects(vae, flow, cfg, [class_id], samples, args.steps, seed + class_id, device)
        generated = [item["object"] for item in items]
        row = evaluate_class(generated, reference, training, cameras, cfg.evaluation, cfg.render)
        row["class"] = class_names[class_id]
        row["reference_split"] = reference_split
        row["num_generated"] = len(generated)
        row["num_test"] = len(reference)
        rows.append(row)
        print(f"{class_names[class_id]:>12} MMD {row['mmd_cd']:.5f} Cov {row['coverage_cd']:.3f} kNN {row['knn_accuracy']:.3f} Div {row['diversity_cd']:.5f} MVC {row['mvc']:.3f} DCS {row['dcs']:.3f}")

    averages = {key: float(sum(row[key] for row in rows) / max(len(rows), 1)) for key in METRIC_KEYS}
    write_csv(rows, os.path.join(out, "class_metrics.csv"), METRIC_KEYS)
    save_json(os.path.join(out, "evaluation.json"), {"per_class": rows, "average": averages, "samples_per_class": samples, "steps": args.steps})
    print("average: " + " ".join(f"{key}={averages[key]:.5f}" for key in METRIC_KEYS))
    print(f"evaluation artifacts written to {out}")


if __name__ == "__main__":
    main()
