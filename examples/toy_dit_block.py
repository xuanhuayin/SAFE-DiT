"""Minimal DiT-style example using SAFE-DiT primitives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safe_dit import (
    SAFEConfig,
    prompt_conditioned_sensitivity,
    safe_scaled_dot_product_attention,
    select_sensitive_tokens,
    sensitivity_weighted_cfg,
    srsu_update,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class ToySelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = safe_scaled_dot_product_attention(query, key, value, attn_mask)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, dim)
        return self.proj(out)


def run_example(device: torch.device, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    batch, tokens, text_tokens, dim, heads = 2, 64, 16, 128, 4
    cfg = SAFEConfig(keep_ratio=0.5, anchor_interval=2)

    hidden = torch.randn(batch, tokens, dim, device=device)
    previous = torch.randn_like(hidden)
    attn_mask = torch.ones(batch, heads, tokens, tokens, dtype=torch.bool, device=device)
    image_to_text = torch.rand(batch, heads, tokens, text_tokens, device=device)
    text_importance = torch.linspace(1.0, 2.0, text_tokens, device=device)

    block = ToySelfAttention(dim=dim, heads=heads).to(device)
    dense_candidate = block(hidden, attn_mask=attn_mask)

    sensitivity = prompt_conditioned_sensitivity(image_to_text, text_importance)
    sensitive_mask = select_sensitive_tokens(sensitivity, cfg.keep_ratio)
    sparse_state = srsu_update(
        previous_state=previous,
        dense_candidate=dense_candidate,
        sensitive_mask=sensitive_mask,
        anchor_step=False,
    )

    unconditional = torch.randn_like(hidden)
    conditional = torch.randn_like(hidden)
    guided, scale = sensitivity_weighted_cfg(
        unconditional,
        conditional,
        sensitivity,
        cfg_context=cfg.cfg_context,
        cfg_sensitive=cfg.cfg_sensitive,
    )

    return {
        "device": str(device),
        "dense_candidate_shape": list(dense_candidate.shape),
        "sparse_state_shape": list(sparse_state.shape),
        "guided_shape": list(guided.shape),
        "selected_token_fraction": round(float(sensitive_mask.float().mean().item()), 4),
        "guidance_scale_min": round(float(scale.min().item()), 4),
        "guidance_scale_max": round(float(scale.max().item()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_example(resolve_device(args.device), args.seed), indent=2))


if __name__ == "__main__":
    main()
