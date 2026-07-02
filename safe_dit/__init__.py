"""SAFE-DiT utilities."""

from .mask_semantics import (
    explicit_attention_reference,
    is_removable_attention_mask,
    safe_scaled_dot_product_attention,
)
from .scheduler import (
    SAFEConfig,
    context_anchor_refresh,
    first_order_extrapolate,
    prompt_conditioned_sensitivity,
    select_sensitive_tokens,
    sensitivity_weighted_cfg_image,
    should_compute_dense_step,
    sensitivity_weighted_cfg,
    srsu_update,
)
from .regions import (
    build_sensitivity_partition,
    cfg_signal_sensitivity,
    flatten_image_mask,
    minmax_normalize,
    sensitivity_to_mask,
    smooth_region_mask,
)

__all__ = [
    "SAFEConfig",
    "context_anchor_refresh",
    "explicit_attention_reference",
    "first_order_extrapolate",
    "is_removable_attention_mask",
    "build_sensitivity_partition",
    "cfg_signal_sensitivity",
    "flatten_image_mask",
    "minmax_normalize",
    "prompt_conditioned_sensitivity",
    "safe_scaled_dot_product_attention",
    "select_sensitive_tokens",
    "sensitivity_to_mask",
    "sensitivity_weighted_cfg",
    "sensitivity_weighted_cfg_image",
    "should_compute_dense_step",
    "smooth_region_mask",
    "srsu_update",
]
