#!/usr/bin/env bash
set -euo pipefail

python -m safe_dit.sdpa_benchmark \
  --device auto \
  --seq-lens 512 1024 \
  --head-dims 64 72
