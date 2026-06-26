"""Smoke test for SAFE-DiT."""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch

from .mask_semantics import (
    explicit_attention_reference,
    is_removable_attention_mask,
    safe_scaled_dot_product_attention,
)
from .scheduler import (
    SAFEConfig,
    context_anchor_refresh,
    prompt_conditioned_sensitivity,
    select_sensitive_tokens,
    sensitivity_weighted_cfg,
    srsu_update,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_demo(device: torch.device, seed: int, tokens: int, dim: int, text_tokens: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    batch, heads = 2, 4

    query = torch.randn(batch, heads, tokens, dim, device=device, dtype=torch.float32)
    key = torch.randn(batch, heads, tokens, dim, device=device, dtype=torch.float32)
    value = torch.randn(batch, heads, tokens, dim, device=device, dtype=torch.float32)

    all_valid = torch.ones(batch, heads, tokens, tokens, device=device, dtype=torch.bool)
    reference = explicit_attention_reference(query, key, value, all_valid)
    fast = safe_scaled_dot_product_attention(query, key, value, all_valid)
    max_error = (reference - fast).abs().max().item()

    padding_mask = all_valid.clone()
    padding_mask[..., -1] = False

    cross_attn = torch.rand(batch, heads, tokens, text_tokens, device=device)
    text_importance = torch.linspace(1.0, 2.0, text_tokens, device=device)
    cfg = SAFEConfig(keep_ratio=0.5, anchor_interval=3)

    sensitivity = prompt_conditioned_sensitivity(cross_attn, text_importance, eps=cfg.eps)
    sensitive = select_sensitive_tokens(sensitivity, keep_ratio=cfg.keep_ratio)

    previous = torch.randn(batch, tokens, dim, device=device)
    candidate = torch.randn(batch, tokens, dim, device=device)
    updated = srsu_update(previous, candidate, sensitive, anchor_step=False)
    context_unchanged = torch.equal(updated[~sensitive], previous[~sensitive])
    sensitive_updated = torch.equal(updated[sensitive], candidate[sensitive])

    unconditional = torch.randn(batch, tokens, dim, device=device)
    conditional = torch.randn(batch, tokens, dim, device=device)
    guided, scale = sensitivity_weighted_cfg(
        unconditional,
        conditional,
        sensitivity,
        cfg_context=cfg.cfg_context,
        cfg_sensitive=cfg.cfg_sensitive,
    )

    return {
        "device": str(device),
        "all_valid_mask_removable": is_removable_attention_mask(all_valid),
        "padding_mask_removable": is_removable_attention_mask(padding_mask),
        "max_exactness_error": max_error,
        "selected_token_fraction": float(sensitive.float().mean().item()),
        "context_tokens_unchanged": bool(context_unchanged),
        "sensitive_tokens_updated": bool(sensitive_updated),
        "anchor_steps_0_to_5": [context_anchor_refresh(i, cfg.anchor_interval) for i in range(6)],
        "guided_shape": list(guided.shape),
        "guidance_scale_range": [float(scale.min().item()), float(scale.max().item())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--text-tokens", type=int, default=16)
    args = parser.parse_args()

    result = run_demo(
        device=resolve_device(args.device),
        seed=args.seed,
        tokens=args.tokens,
        dim=args.dim,
        text_tokens=args.text_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
