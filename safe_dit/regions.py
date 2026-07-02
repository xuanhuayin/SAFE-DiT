"""Prompt-sensitive region utilities for SAFE-DiT.

The paper uses image-to-text attention when it is available. Public diffusers
pipelines do not always expose those maps, so this module also provides a
backbone-agnostic fallback based on the local classifier-free guidance signal.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each batch item to the range [0, 1]."""

    flat = x.flatten(start_dim=1)
    lo = flat.amin(dim=1).view(-1, *([1] * (x.ndim - 1)))
    hi = flat.amax(dim=1).view(-1, *([1] * (x.ndim - 1)))
    return (x - lo) / (hi - lo).clamp_min(eps)


def cfg_signal_sensitivity(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate prompt sensitivity from the local CFG signal.

    Both inputs must be image-like predictions with shape
    `[batch, channels, height, width]`. The returned tensor has shape
    `[batch, 1, height, width]`.
    """

    if unconditional.shape != conditional.shape:
        raise ValueError("unconditional and conditional predictions must have the same shape")
    if unconditional.ndim != 4:
        raise ValueError("predictions must have shape [batch, channels, height, width]")
    saliency = (conditional - unconditional).abs().mean(dim=1, keepdim=True)
    return minmax_normalize(saliency, eps=eps)


def sensitivity_to_mask(sensitivity: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Convert a sensitivity map to a boolean sensitive-region mask."""

    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if sensitivity.ndim < 2:
        raise ValueError("sensitivity must have a batch dimension and at least one value dimension")

    flat = sensitivity.flatten(start_dim=1)
    if keep_ratio >= 1.0:
        mask = torch.ones_like(flat, dtype=torch.bool)
    else:
        threshold = torch.quantile(flat.float(), q=1.0 - keep_ratio, dim=1, keepdim=True)
        mask = flat >= threshold.to(flat.dtype)
    return mask.view_as(sensitivity)


def smooth_region_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Convert a hard region mask to a lightly smoothed float map."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [batch, 1, height, width]")
    if kernel_size <= 1:
        return mask.float()
    padding = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=torch.float32)
    weight = weight / weight.sum()
    return F.conv2d(mask.float(), weight, padding=padding).clamp(0.0, 1.0).to(mask.device)


def flatten_image_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return `[batch, height * width]` token mask from `[batch, 1, height, width]`."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [batch, 1, height, width]")
    return mask.flatten(start_dim=2).squeeze(1).bool()


def build_sensitivity_partition(
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    keep_ratio: float,
    eps: float = 1e-6,
    smooth: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a sensitivity map and smoothed sensitive-region map from CFG signal."""

    sensitivity = cfg_signal_sensitivity(unconditional, conditional, eps=eps)
    hard = sensitivity_to_mask(sensitivity, keep_ratio=keep_ratio)
    region = smooth_region_mask(hard) if smooth else hard.float()
    return sensitivity, region
