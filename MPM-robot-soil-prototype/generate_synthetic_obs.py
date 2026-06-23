"""
generate_synthetic_obs.py
=========================
Run one MPM simulation at a known "true" (phi, c) to produce a synthetic
observation for inverse analysis.  The output JSON has the same format as
real field data would, so inverse_gp.py --obs can consume it directly.

Choose phi_true / c_true values that are NOT in the LHS design to avoid
data leakage into the GP surrogate.

Usage:
    python3 generate_synthetic_obs.py                        # defaults: phi=32, c=3000 Pa
    python3 generate_synthetic_obs.py --phi 28 --c 5000
    python3 generate_synthetic_obs.py --phi 38 --c 500 --omega 5.0
    python3 generate_synthetic_obs.py --phi 32 --c 3000 --noise 0.05
                                                             # add 5% Gaussian noise to obs

The --noise flag simulates measurement uncertainty so the inverse problem
is more realistic (default: 0, i.e. perfect synthetic observation).
"""

import argparse, json, os, subprocess, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--phi",   type=float, default=32.0,
                    help="True friction angle (deg)")
parser.add_argument("--c",     type=float, default=3000.0,
                    help="True cohesion (Pa)")
parser.add_argument("--omega", type=float, default=0.2,
                    help="Angular velocity rad/s (use 5.0 for fast test)")
parser.add_argument("--settle",type=int,   default=20000)
parser.add_argument("--noise", type=float, default=0.0,
                    help="Relative Gaussian noise added to each observable (e.g. 0.05 = 5%%)")
parser.add_argument("--out",   default="../data/synthetic_data/synthetic_obs",
                    help="Output directory")
parser.add_argument("--leg",   default="all", choices=["flat","cleg","rcleg","all"])
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
SIM_PY  = os.path.join(HERE, "mpm_terradynamics.py")
OUT_DIR = os.path.join(HERE, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

summary_path = os.path.join(OUT_DIR, "summary.json")
log_path     = os.path.join(OUT_DIR, "stdout.log")

print(f"Generating synthetic observation:")
print(f"  phi_true = {args.phi:.2f} deg")
print(f"  c_true   = {args.c:.1f} Pa  ({args.c/1000:.2f} kPa)")
print(f"  omega    = {args.omega} rad/s")
print(f"  noise    = {args.noise*100:.1f}%")
print(f"  output   → {summary_path}\n")

cmd = [
    sys.executable, SIM_PY,
    "--phi",         f"{args.phi:.6f}",
    "--c",           f"{args.c:.6f}",
    "--leg",         args.leg,
    "--omega",       f"{args.omega}",
    "--settle",      f"{args.settle}",
    "--out",         OUT_DIR,
    "--summary_out", summary_path,
]

with open(log_path, "w") as logf:
    result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

if result.returncode != 0:
    print(f"Simulation FAILED — see {log_path}")
    sys.exit(1)

if not os.path.exists(summary_path):
    print(f"Summary file not written — see {log_path}")
    sys.exit(1)

with open(summary_path) as f:
    obs = json.load(f)

# Optionally add Gaussian noise to each observable
if args.noise > 0.0:
    rng = np.random.default_rng(seed=7)
    obs_keys = [k for k in obs if k not in ("phi_deg", "c_Pa", "E_Pa")]
    for k in obs_keys:
        v = obs[k]
        obs[k] = float(v * (1.0 + rng.normal(0.0, args.noise)))
    print(f"Added {args.noise*100:.1f}% Gaussian noise to {len(obs_keys)} observables.")

# Save ground truth alongside so we know what to expect
obs["_ground_truth_phi_deg"] = args.phi
obs["_ground_truth_c_Pa"]    = args.c

with open(summary_path, "w") as f:
    json.dump(obs, f, indent=2)

print(f"\nSynthetic observation written to: {summary_path}")
print(f"\nNext step:")
print(f"  python3 inverse_gp.py --data lhs_results/lhs_samples.csv --obs {summary_path}")
