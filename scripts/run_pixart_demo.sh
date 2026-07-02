#!/usr/bin/env bash
set -euo pipefail

python -m safe_dit.generate_pixart \
  --mode safe \
  --prompt "a cinematic photo of a glass greenhouse on a snowy mountain at sunrise" \
  --output outputs/pixart_safe.png \
  --height 1024 \
  --width 1024 \
  --steps 20 \
  --seed 0
