#!/usr/bin/env bash

# usage:
# mamba run -n env_MPM_robot_soil script/calibrate_perzyna_flat_omega10.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/script/calibrate_perzyna_flat.py" \
  --omega 10.0 \
  --etas 0.1 1.0 10.0 \
  --stress-refs 10000 100000 \
  --target-fz-ratio 0.5 \
  --target-fx-ratio 0.5 \
  --out-root "$ROOT_DIR/data/synthetic_data/perzyna_calibration_flat_omega10"
