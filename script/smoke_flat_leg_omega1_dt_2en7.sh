#!/usr/bin/env bash

# usage:
# mamba run -n env_MPM_robot_soil script/smoke_flat_leg_omega1_dt_2en7.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="$ROOT_DIR/MPM-robot-soil-prototype"
OUT_DIR="$ROOT_DIR/data/synthetic_data/smoke_flat_results_omega1_dt_2en7"
SUMMARY_PATH="$OUT_DIR/summary.json"
CSV_PATH="$OUT_DIR/terradyn_flat.csv"
TMP_SIM="$(mktemp "$SIM_DIR/mpm_terradynamics_dt_2en7.XXXXXX.py")"

cleanup() {
  rm -f "$TMP_SIM"
}
trap cleanup EXIT

echo "Running flat-leg MPM smoke test at omega=1 with DT=2e-7..."
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

python3 - "$SIM_DIR/mpm_terradynamics.py" "$TMP_SIM" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()
old = "DT          = 8e-6"
new = "DT          = 2e-7"
if old not in text:
    raise SystemExit(f"Could not find expected DT assignment: {old}")
dst.write_text(text.replace(old, new, 1))
PY

cd "$SIM_DIR"
python3 "$TMP_SIM" \
  --leg flat \
  --omega 1.0 \
  --settle 50 \
  --threads 2 \
  --out "$OUT_DIR" \
  --summary_out "$SUMMARY_PATH" \

test -s "$CSV_PATH"
test -s "$SUMMARY_PATH"

echo "Smoke test passed:"
echo "  $CSV_PATH"
echo "  $SUMMARY_PATH"
