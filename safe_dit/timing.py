"""Timing and environment helpers used by SAFE-DiT scripts."""

from __future__ import annotations

import gc
import platform
import time
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def synchronize(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timed_call(fn: Callable[[], T], reset_peak_memory: bool = True) -> Tuple[float, float, float, T]:
    """Run `fn` once and return seconds, peak allocated GB, peak reserved GB, result."""

    clear_cuda()
    if torch.cuda.is_available() and reset_peak_memory:
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    start = time.perf_counter()
    result = fn()
    synchronize()
    seconds = time.perf_counter() - start
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    else:
        peak_alloc = 0.0
        peak_reserved = 0.0
    return seconds, peak_alloc, peak_reserved, result


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def environment_record() -> Dict[str, object]:
    """Record the runtime environment for benchmark logs."""

    record: Dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        record.update(
            {
                "device_count": torch.cuda.device_count(),
                "gpu": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            }
        )
    return record
