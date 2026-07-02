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
    mask_step: int = 5
    skip_interval: int = 2
    cfg_scale: float = 4.0
    cfg_sensitive: float = 7.0
    cfg_context: float = 1.0
    sw_cfg: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.anchor_interval < 1:
            raise ValueError("anchor_interval must be >= 1")
        if self.mask_step < 0:
            raise ValueError("mask_step must be >= 0")
        if self.skip_interval < 0:
            raise ValueError("skip_interval must be >= 0")
        if self.cfg_sensitive < 0.0 or self.cfg_context < 0.0:
            raise ValueError("guidance scales must be non-negative")


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


def should_compute_dense_step(
    step: int,
    total_steps: int,
    mask_step: int,
    skip_interval: int,
    anchor_interval: int,
    tail_dense_steps: int = 2,
) -> bool:
    """Return whether a denoising step should perform a dense refresh.

    The public diffusers adapters use this schedule when a backbone does not
    expose row-level query execution. Before the sensitivity map is committed,
    all steps are dense. Afterward, anchor steps and the final denoising steps
    stay dense to reduce drift.
    """

    if step < mask_step:
        return True
    if step >= max(0, total_steps - tail_dense_steps):
        return True
    if skip_interval <= 0:
        return True
    if context_anchor_refresh(step - mask_step, anchor_interval):
        return True
    return (step - mask_step) % (skip_interval + 1) == 0


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


def first_order_extrapolate(history: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Predict the next state from the two most recent computed states."""

    if len(history) == 0:
        raise ValueError("history must contain at least one tensor")
    if len(history) == 1:
        return history[-1]
    return history[-1] + (history[-1] - history[-2])


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


def sensitivity_weighted_cfg_image(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    sensitivity_map: torch.Tensor,
    cfg_context: float,
    cfg_sensitive: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spatially varying CFG for image-like tensors.

    Parameters use `[batch, channels, height, width]` predictions and a
    `[batch, 1, height, width]` sensitivity map.
    """

    if unconditional.shape != conditional.shape:
        raise ValueError("unconditional and conditional predictions must have the same shape")
    if sensitivity_map.ndim != 4 or sensitivity_map.shape[1] != 1:
        raise ValueError("sensitivity_map must have shape [batch, 1, height, width]")
    if sensitivity_map.shape[0] != unconditional.shape[0] or sensitivity_map.shape[-2:] != unconditional.shape[-2:]:
        raise ValueError("sensitivity_map spatial dimensions must match the predictions")
    scale = cfg_context + sensitivity_map * (cfg_sensitive - cfg_context)
    guided = unconditional + scale * (conditional - unconditional)
    return guided, scale
