import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token2splat.config import apply_cli_overrides, load_config
from token2splat.tokenizer.tokenizer import CanonicalPatchTokenizer
from token2splat.train import load_splits, resolve_device, train_flow


def main():
    parser = argparse.ArgumentParser(description="Stage 2: train the class-conditional global-local rectified flow over frozen VAE latents")
    parser.add_argument("--config", default="configs/synthetic_smoke.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--vae-checkpoint", default=None)
    parser.add_argument("--cache-tokens", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.set)
    cfg.device = resolve_device(args.device or cfg.device)

    vae_checkpoint = args.vae_checkpoint if args.vae_checkpoint is not None else os.path.join(cfg.output_dir, "stage1", "vae_best.pt")
    if not os.path.exists(vae_checkpoint):
        raise SystemExit(f"Stage 1 checkpoint not found: {vae_checkpoint}. Run scripts/train_stage1.py first.")

    splits, class_names = load_splits(cfg, args.data)
    tokenizer = CanonicalPatchTokenizer.from_records(splits["train"], cfg.token)
    last, best, history = train_flow(
        cfg,
        splits,
        tokenizer,
        vae_checkpoint,
        device=cfg.device,
        work_dir=args.work_dir,
        class_names=class_names,
        cache_tokens=args.cache_tokens,
    )
    print(f"stage2 done: last={last} best={best} epochs={len(history)}")


if __name__ == "__main__":
    main()
