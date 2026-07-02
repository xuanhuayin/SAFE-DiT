"""Command-line entry point for PixArt-Sigma SAFE-DiT generation."""

from __future__ import annotations

import argparse
import json

import torch

from .adapters import PixArtSAFEGenerator
from .scheduler import SAFEConfig
from .timing import environment_record, timed_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/pixart_safe.png")
    parser.add_argument("--mode", default="safe", choices=["dense", "safe"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--mask-step", type=int, default=5)
    parser.add_argument("--skip-interval", type=int, default=2)
    parser.add_argument("--anchor-interval", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--cfg-context", type=float, default=1.0)
    parser.add_argument("--cfg-sensitive", type=float, default=7.0)
    parser.add_argument("--no-sw-cfg", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    cfg = SAFEConfig(
        keep_ratio=args.keep_ratio,
        anchor_interval=args.anchor_interval,
        mask_step=args.mask_step,
        skip_interval=args.skip_interval,
        cfg_scale=args.cfg_scale,
        cfg_context=args.cfg_context,
        cfg_sensitive=args.cfg_sensitive,
        sw_cfg=not args.no_sw_cfg,
    )
    generator = PixArtSAFEGenerator(device=args.device, dtype=dtype, cache_dir=args.cache_dir)
    seconds, peak_alloc, peak_reserved, output_path = timed_call(
        lambda: generator.save(
            prompt=args.prompt,
            output=args.output,
            mode=args.mode,
            seed=args.seed,
            steps=args.steps,
            height=args.height,
            width=args.width,
            cfg=cfg,
        )
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "mode": args.mode,
                "seconds": seconds,
                "peak_alloc_gb": peak_alloc,
                "peak_reserved_gb": peak_reserved,
                "environment": environment_record(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
