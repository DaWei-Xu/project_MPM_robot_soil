#!/usr/bin/env bash

# usage:
# mamba run -n env_MPM_robot_soil script/smoke_flat_leg_omega10.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="$ROOT_DIR/MPM-robot-soil-prototype"
OUT_DIR="$ROOT_DIR/data/synthetic_data/smoke_flat_results_omega10"
SUMMARY_PATH="$OUT_DIR/summary.json"
CSV_PATH="$OUT_DIR/terradyn_flat.csv"

echo "Running flat-leg MPM smoke test at omega=10..."
echo "Output: $OUT_DIR"

python3 - <<'PY'
import importlib.util
import sys

missing = [
    name for name in ("taichi", "numpy", "matplotlib")
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install them in this environment before running the smoke test.", file=sys.stderr)
    sys.exit(1)
PY

rm -rf "$OUT_DIR"

cd "$SIM_DIR"
python3 mpm_terradynamics.py \
  --leg flat \
  --omega 10.0 \
  --settle 50 \
  --threads 2 \
  --out "$OUT_DIR" \
  --summary_out "$SUMMARY_PATH" \
  --paraview

test -s "$CSV_PATH"
test -s "$SUMMARY_PATH"

echo "Smoke test passed:"
echo "  $CSV_PATH"
echo "  $SUMMARY_PATH"
