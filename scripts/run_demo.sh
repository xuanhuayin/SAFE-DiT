#!/usr/bin/env bash
set -euo pipefail

python -m safe_dit.demo --device auto
python examples/toy_dit_block.py --device auto
