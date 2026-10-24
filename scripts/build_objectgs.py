import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import load_config
from token2splat.data.objectgs import build_objectgs_dataset, make_random_scene_dump


def main():
    parser = argparse.ArgumentParser(description="Reproduce the ObjectGS-34 style extraction protocol from scene-level 3DGS dumps")
    parser.add_argument("--config", default="configs/objectgs34.yaml")
    parser.add_argument("--scenes-dir", default="data/scenes")
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-primitives", type=int, default=None)
    parser.add_argument("--min-instances-per-class", type=int, default=8)
    parser.add_argument("--make-demo-scenes", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out if args.out is not None else cfg.data.root
    min_primitives = args.min_primitives if args.min_primitives is not None else cfg.data.min_primitives

    if args.make_demo_scenes:
        os.makedirs(args.scenes_dir, exist_ok=True)
        for index in range(3):
            make_random_scene_dump(os.path.join(args.scenes_dir, f"demo_scene_{index:03d}.npz"), num_instances=8, num_primitives=350, seed=index + 1)
        print(f"demo scene dumps written to {args.scenes_dir}")

    out, class_names, count = build_objectgs_dataset(
        args.scenes_dir,
        out,
        min_primitives=min_primitives,
        min_instances_per_class=args.min_instances_per_class,
        min_extent=cfg.data.min_extent,
    )
    print(json.dumps({"dataset": out, "instances": count, "classes": class_names}, indent=2))


if __name__ == "__main__":
    main()
