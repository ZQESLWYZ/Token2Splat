import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import apply_cli_overrides, load_config, save_config
from token2splat.data.dataset import write_dataset
from token2splat.data.objects import is_valid_object
from token2splat.data.synthetic import CLASS_NAMES, build_synthetic_records


def main():
    parser = argparse.ArgumentParser(description="Build the class-conditional synthetic 3DGS benchmark used by the tests")
    parser.add_argument("--config", default="configs/synthetic_smoke.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--scenes", type=int, default=None)
    parser.add_argument("--objects-per-scene", type=int, default=None)
    parser.add_argument("--primitives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.set)
    seed = args.seed if args.seed is not None else cfg.seed
    out = args.out if args.out is not None else cfg.data.root

    records = build_synthetic_records(
        args.scenes or cfg.data.num_scenes,
        args.objects_per_scene or cfg.data.objects_per_scene,
        args.primitives or cfg.data.primitives_per_object,
        seed,
    )
    kept = [record for record in records if is_valid_object(record["object"], cfg.data.min_primitives, cfg.data.min_extent)]
    write_dataset(out, kept, CLASS_NAMES, cfg.data.train_ratio, cfg.data.val_ratio, seed)
    save_config(cfg, os.path.join(out, "preprocess_config.yaml"))

    with open(os.path.join(out, "splits.json"), "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    per_class = {}
    for record in kept:
        per_class[CLASS_NAMES[record["label"]]] = per_class.get(CLASS_NAMES[record["label"]], 0) + 1
    print(json.dumps({"dataset": out, "instances": len(kept), "splits": {k: len(v) for k, v in splits.items()}, "classes": per_class}, indent=2))


if __name__ == "__main__":
    main()
