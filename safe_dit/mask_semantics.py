"""Exact mask semantics used by the SAFE-DiT fast-path rewrite.

For additive attention logits

    softmax(QK^T / sqrt(d) + M) V,

a mask can be removed exactly when each query row of M is a finite constant.
Row-wise softmax cancels such constants. Boolean all-valid masks are the common
special case in image self-attention. Padding, causal, block, and non-uniform
bias masks are not removed.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _bool_mask_is_all_valid(mask: torch.Tensor) -> bool:
    """Return True only when every key is valid for every query row."""

    if mask.dtype is not torch.bool:
        raise TypeError("_bool_mask_is_all_valid expects a boolean tensor")
    return bool(mask.all().item())


def _additive_mask_is_row_constant(mask: torch.Tensor, atol: float) -> bool:
    """Check the exact row-constant criterion for additive masks."""

    if mask.dtype is torch.bool:
        raise TypeError("_additive_mask_is_row_constant expects a numeric tensor")
    finite = torch.isfinite(mask)
    if not bool(finite.all().item()):
        return False
    row_min = mask.amin(dim=-1)
    row_max = mask.amax(dim=-1)
    return bool(torch.allclose(row_min, row_max, atol=atol, rtol=0.0))


def is_removable_attention_mask(mask: Optional[torch.Tensor], atol: float = 0.0) -> bool:
    """Return whether `mask` can be removed without changing attention output.

    Parameters
    ----------
    mask:
        A PyTorch SDPA-compatible attention mask. `None` is already mask-free.
        Boolean masks are removable only when all entries are valid. Numeric
        masks are removable when each query row is a finite constant.
    atol:
        Absolute tolerance for floating point additive masks.
    """

    if mask is None:
        return True
    if mask.dtype is torch.bool:
        return _bool_mask_is_all_valid(mask)
    return _additive_mask_is_row_constant(mask, atol=atol)


def canonical_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean or additive mask to additive logits form."""

    if mask.dtype is torch.bool:
        additive = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
        return additive.masked_fill(~mask, float("-inf"))
    return mask.to(dtype=dtype)


def explicit_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Small explicit reference implementation for exactness tests."""

    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        logits = logits + canonical_additive_mask(attn_mask, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, value)


def safe_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    atol: float = 0.0,
) -> torch.Tensor:
    """Call PyTorch SDPA after eliding only provably redundant masks."""

    if is_causal:
        effective_mask = attn_mask
    else:
        effective_mask = None if is_removable_attention_mask(attn_mask, atol=atol) else attn_mask
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=effective_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )

