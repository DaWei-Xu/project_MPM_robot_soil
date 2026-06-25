#!/usr/bin/env bash

# Run four midpoint soil cases for flat-leg force outputs:
#   phi = 17.5, 27.5 deg
#   c   = 7.5, 17.5 kPa
# with omega = 10, 15 rad/s.
#
# usage:
# mamba run -n env_MPM_robot_soil script/sweep_midpoint_cohesion_phi_omega_forces.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/script/sweep_cohesion_phi_omega_forces.py" \
  --phis 17.5 27.5 \
  --cohesions 7.5 17.5 \
  --omegas 10 15 \
  --out-root "$ROOT_DIR/data/synthetic_data/cohesion_phi_omega_force_sweep_midpoints"
