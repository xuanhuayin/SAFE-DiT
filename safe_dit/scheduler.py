"""Tensor-level SAFE-DiT scheduling primitives.

The functions in this file are model-agnostic. Backbone adapters can call them
inside a DiT block without project-specific dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SAFEConfig:
    """Minimal public configuration for SAFE-DiT modules."""

    keep_ratio: float = 0.5
    anchor_interval: int = 2
    cfg_scale: float = 4.0
    cfg_sensitive: float = 7.0
    cfg_context: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.anchor_interval < 1:
            raise ValueError("anchor_interval must be >= 1")


def prompt_conditioned_sensitivity(
    image_to_text_attention: torch.Tensor,
    text_importance: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a per-image-token sensitivity score from image-to-text attention.

    Parameters
    ----------
    image_to_text_attention:
        Tensor with shape [batch, heads, image_tokens, text_tokens] or
        [batch, image_tokens, text_tokens].
    text_importance:
        Optional non-negative tensor with shape [batch, text_tokens] or
        [text_tokens]. It can emphasize prompt words such as objects, colors, or
        spatial attributes.
    """

    attn = image_to_text_attention
    if attn.ndim == 4:
        attn = attn.mean(dim=1)
    if attn.ndim != 3:
        raise ValueError("image_to_text_attention must have shape [B,H,N,T] or [B,N,T]")

    if text_importance is None:
        score = attn.mean(dim=-1)
    else:
        weights = text_importance.to(device=attn.device, dtype=attn.dtype)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(attn.shape[0], -1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        score = (attn * weights.unsqueeze(1)).sum(dim=-1)

    score_min = score.amin(dim=-1, keepdim=True)
    score_max = score.amax(dim=-1, keepdim=True)
    return (score - score_min) / (score_max - score_min).clamp_min(eps)


def select_sensitive_tokens(sensitivity: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Select the top sensitivity tokens in each batch item."""

    if sensitivity.ndim != 2:
        raise ValueError("sensitivity must have shape [batch, image_tokens]")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    batch, tokens = sensitivity.shape
    keep = max(1, int(round(tokens * keep_ratio)))
    topk = torch.topk(sensitivity, k=keep, dim=-1).indices
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=sensitivity.device)
    return mask.scatter_(dim=-1, index=topk, value=True)


def context_anchor_refresh(step: int, anchor_interval: int) -> bool:
    """Return whether this denoising step should refresh all token states."""

    if anchor_interval < 1:
        raise ValueError("anchor_interval must be >= 1")
    return step % anchor_interval == 0


def srsu_update(
    previous_state: torch.Tensor,
    dense_candidate: torch.Tensor,
    sensitive_mask: torch.Tensor,
    anchor_step: bool,
) -> torch.Tensor:
    """Sensitive-Region Selective Update.

    At anchor steps, all tokens are refreshed. Otherwise only sensitive tokens
    take the newly computed candidate state, and context tokens reuse their
    previous state.
    """

    if previous_state.shape != dense_candidate.shape:
        raise ValueError("previous_state and dense_candidate must have the same shape")
    if sensitive_mask.shape != previous_state.shape[:2]:
        raise ValueError("sensitive_mask must have shape [batch, image_tokens]")
    if anchor_step:
        return dense_candidate
    mask = sensitive_mask.unsqueeze(-1)
    return torch.where(mask, dense_candidate, previous_state)


def sensitivity_weighted_cfg(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    sensitivity: torch.Tensor,
    cfg_context: float,
    cfg_sensitive: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply spatially varying classifier-free guidance.

    Returns the guided prediction and the per-token guidance scale.
    """

    if unconditional.shape != conditional.shape:
        raise ValueError("unconditional and conditional predictions must have the same shape")
    if sensitivity.shape != unconditional.shape[:2]:
        raise ValueError("sensitivity must have shape [batch, image_tokens]")

    scale = cfg_context + sensitivity * (cfg_sensitive - cfg_context)
    guided = unconditional + scale.unsqueeze(-1) * (conditional - unconditional)
    return guided, scale
