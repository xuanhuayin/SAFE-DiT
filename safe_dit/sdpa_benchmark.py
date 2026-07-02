"""SDPA benchmark for Mask-Induced Dispatch Tax (MIDT)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Callable[[], torch.Tensor], device: torch.device, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / reps


def profiler_kernels(fn: Callable[[], torch.Tensor], device: torch.device) -> Optional[List[Dict[str, object]]]:
    """Return attention-related profiler rows for a single call."""

    if device.type != "cuda":
        return None
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None

    fn()
    synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
        synchronize(device)
    tokens = ("flash", "efficient", "sdpa", "attention", "cutlass", "triton", "scaled_dot", "bmm", "softmax")
    rows: List[Dict[str, object]] = []
    for event in prof.key_averages():
        key = event.key
        if any(token in key.lower() for token in tokens):
            rows.append(
                {
                    "name": key,
                    "calls": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
                }
            )
    rows.sort(key=lambda item: float(item["cuda_time_total_us"]), reverse=True)
    return rows[:30]


def benchmark_case(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    reps: int,
    profile_kernels: bool = False,
) -> Dict[str, object]:
    shape = (batch, heads, seq_len, head_dim)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    all_valid = torch.ones(batch, heads, seq_len, seq_len, device=device, dtype=torch.bool)

    def no_mask() -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=None)

    def with_mask() -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=all_valid)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    no_mask_ms = time_ms(no_mask, device=device, warmup=warmup, reps=reps)
    if device.type == "cuda":
        no_mask_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        no_mask_peak_gb = 0.0

    mask_ms = time_ms(with_mask, device=device, warmup=warmup, reps=reps)
    if device.type == "cuda":
        mask_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        mask_peak_gb = 0.0

    record: Dict[str, object] = {
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "no_mask_ms": no_mask_ms,
        "all_valid_mask_ms": mask_ms,
        "latency_tax": mask_ms / no_mask_ms if no_mask_ms > 0 else float("nan"),
        "no_mask_peak_gb": no_mask_peak_gb,
        "all_valid_mask_peak_gb": mask_peak_gb,
    }
    if profile_kernels:
        record["no_mask_kernels"] = profiler_kernels(no_mask, device)
        record["all_valid_mask_kernels"] = profiler_kernels(with_mask, device)
    return record


def environment_record(device: torch.device, dtype: torch.dtype) -> Dict[str, object]:
    record: Dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        record["gpu"] = torch.cuda.get_device_name(device)
        record["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        record["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        record["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
    return record


def parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-lens", nargs="+", default=["512", "1024"])
    parser.add_argument("--head-dims", nargs="+", default=["64"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--profile-kernels", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    results = {
        "environment": environment_record(device, dtype),
        "cases": [],
    }
    for seq_len in parse_int_list(args.seq_lens):
        for head_dim in parse_int_list(args.head_dims):
            case = benchmark_case(
                batch=args.batch,
                heads=args.heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                warmup=args.warmup,
                reps=args.reps,
                profile_kernels=args.profile_kernels,
            )
            results["cases"].append(case)
            print(json.dumps(case, sort_keys=True))

    if args.out_json is not None:
        output_path = Path(args.out_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
