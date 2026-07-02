<div align="center">

# SAFE-DiT: Semantics-Aware Fast-path Execution for High-Resolution Diffusion Transformers

**Xuanhua Yin, Yuxuan Jia, Chuanzhi Xu, Weidong Cai**

**A runnable implementation of SAFE-DiT, including exact mask elision, prompt-sensitive spatial scheduling, spatial guidance, MIDT benchmarking, and public PixArt-Sigma generation.**

[arXiv](https://arxiv.org/abs/2606.29360) | [PDF](https://arxiv.org/pdf/2606.29360) | [DOI](https://doi.org/10.48550/arXiv.2606.29360) | [Installation](#installation) | [Quick Start](#quick-start) | [Citation](#citation)

</div>

<p align="center">
  <img src="assets/teaser.png" width="92%" alt="SAFE-DiT teaser">
</p>

SAFE-DiT accelerates high-resolution diffusion transformers by separating
semantics-preserving fast-path execution from approximation-based spatial
scheduling. This repository includes the certified attention-mask rewrite,
prompt-conditioned token selection, selective token update, spatially weighted
guidance, public diffusers generation code, and PyTorch SDPA benchmarks for
measuring Mask-Induced Dispatch Tax (MIDT).

## Method Overview

<p align="center">
  <img src="assets/pipeline.png" width="92%" alt="SAFE-DiT method overview">
</p>

SAFE-DiT contains four implementation pieces:

- `Mask elision`: removes only provably redundant attention masks. An all-valid
  image self-attention mask is mathematically equivalent to no mask, while
  padding, causal, block, and non-uniform bias masks are kept.
- `PCSP`: partitions image tokens using prompt-conditioned image-to-text
  attention sensitivity.
- `SRSU`: refreshes sensitive tokens while reusing context-token states between
  anchor steps.
- `SW-CFG`: applies spatially weighted classifier-free guidance so sensitive
  regions receive stronger guidance than context regions.

<p align="center">
  <img src="assets/highres_frontier.png" width="48%" alt="High-resolution frontier">
  <img src="assets/midt_tax.png" width="48%" alt="Mask-induced dispatch tax">
</p>

## Installation

CUDA is recommended for image generation and MIDT benchmarking.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you need a specific CUDA build, install the matching PyTorch wheel first,
then run `pip install -r requirements.txt`. The PixArt-Sigma demo downloads
public Hugging Face weights on first use.

## Quick Start

Run public PixArt-Sigma generation with SAFE-DiT scheduling:

```bash
python -m safe_dit.generate_pixart \
  --mode safe \
  --prompt "a cinematic photo of a glass greenhouse on a snowy mountain at sunrise" \
  --output outputs/pixart_safe.png \
  --height 1024 \
  --width 1024 \
  --steps 20 \
  --seed 0
```

The output image is written to `outputs/pixart_safe.png`. The command prints a
JSON timing and memory summary when generation finishes.

Run the dense PixArt-Sigma reference with the same prompt and seed:

```bash
python -m safe_dit.generate_pixart \
  --mode dense \
  --prompt "a cinematic photo of a glass greenhouse on a snowy mountain at sunrise" \
  --output outputs/pixart_dense.png \
  --height 1024 \
  --width 1024 \
  --steps 20 \
  --seed 0
```

## MIDT Benchmark

Measure the latency and memory effect of passing an all-valid boolean mask into
PyTorch SDPA:

```bash
python -m safe_dit.sdpa_benchmark --device auto --seq-lens 512 1024 --head-dims 64 72
```

For larger GPUs:

```bash
python -m safe_dit.sdpa_benchmark --device cuda --seq-lens 2048 4096 8192 --head-dims 64 72 128
```

To also record profiler kernel names for dispatch auditing:

```bash
python -m safe_dit.sdpa_benchmark \
  --device cuda \
  --seq-lens 2048 4096 \
  --head-dims 72 \
  --profile-kernels \
  --out-json outputs/midt_dispatch.json
```

The benchmark compares SDPA with no mask against SDPA with an all-valid mask.
The two calls are mathematically equivalent for self-attention, but they can
dispatch to different kernels and therefore have different latency and memory
behavior.

## Qualitative Results

<p align="center">
  <img src="assets/qualitative_generation.png" width="92%" alt="Qualitative generation examples">
</p>

## Using The Components

```python
import torch
from safe_dit import (
    safe_scaled_dot_product_attention,
    build_sensitivity_partition,
    prompt_conditioned_sensitivity,
    select_sensitive_tokens,
    srsu_update,
    sensitivity_weighted_cfg,
)

q = torch.randn(1, 8, 1024, 64, device="cuda")
k = torch.randn(1, 8, 1024, 64, device="cuda")
v = torch.randn(1, 8, 1024, 64, device="cuda")
mask = torch.ones(1, 8, 1024, 1024, dtype=torch.bool, device="cuda")

out = safe_scaled_dot_product_attention(q, k, v, mask)
```

## Repository Layout

```text
SAFE-DiT/
  assets/                  # Paper figures used by the README
  examples/
    toy_dit_block.py       # Minimal DiT-style component example
  safe_dit/
    adapters/
      pixart_sigma.py      # Public PixArt-Sigma dense / SAFE generation
    mask_semantics.py      # Exact attention-mask removal criterion
    regions.py             # CFG-signal and attention-based sensitivity maps
    scheduler.py           # PCSP, SRSU, CAR, and SW-CFG primitives
    demo.py                # Lightweight component check
    generate_pixart.py     # PixArt-Sigma generation CLI
    sdpa_benchmark.py      # Masked vs. mask-free SDPA benchmark
    timing.py              # Timing and environment helpers
  scripts/
    run_demo.sh
    run_midt_benchmark.sh
    run_pixart_demo.sh
  CITATION.cff
  citation.bib
  requirements.txt
```

## Citation

```bibtex
@article{yin2026safedit,
  title={SAFE-DiT: Semantics-Aware Fast-path Execution for High-Resolution Diffusion Transformers},
  author={Yin, Xuanhua and Jia, Yuxuan and Xu, Chuanzhi and Cai, Weidong},
  journal={arXiv preprint arXiv:2606.29360},
  year={2026},
  doi={10.48550/arXiv.2606.29360},
  url={https://arxiv.org/abs/2606.29360}
}
```
