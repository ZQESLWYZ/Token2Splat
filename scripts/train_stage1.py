import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import apply_cli_overrides, load_config
from token2splat.tokenizer.tokenizer import CanonicalPatchTokenizer
from token2splat.train import load_splits, resolve_device, train_vae


def main():
    parser = argparse.ArgumentParser(description="Stage 1: train the hierarchical Gaussian VAE over canonical patch tokens")
    parser.add_argument("--config", default="configs/synthetic_smoke.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--cache-tokens", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides such as vae.epochs=40 or token.num_anchors=128")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.set)
    cfg.device = resolve_device(args.device or cfg.device)

    splits, class_names = load_splits(cfg, args.data)
    tokenizer = CanonicalPatchTokenizer.from_records(splits["train"], cfg.token)
    print(
        f"data splits train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}; "
        f"anchor bank M={tokenizer.num_slots}, patch capacity K={tokenizer.capacity_k}"
    )
    last, best, history = train_vae(
        cfg,
        splits,
        tokenizer,
        device=cfg.device,
        work_dir=args.work_dir,
        class_names=class_names,
        cache_tokens=args.cache_tokens,
        resume_path=args.resume,
    )
    print(f"stage1 done: last={last} best={best} epochs={len(history)}")


if __name__ == "__main__":
    main()
