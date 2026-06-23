"""
run_lhs_samples.py
==================
Latin Hypercube Sampling over (phi, c) for inverse analysis.
Runs mpm_terradynamics.py for each sample and collects summary statistics
into a single CSV ready for GP surrogate fitting.

Usage:
    python3 run_lhs_samples.py                      # 30 samples, all 3 legs
    python3 run_lhs_samples.py --n 50               # 50 samples
    python3 run_lhs_samples.py --workers 4          # parallel (4 cores)
    python3 run_lhs_samples.py --omega 5.0          # fast runs (25x fewer steps)
    python3 run_lhs_samples.py --resume             # skip already-completed samples

Parameter ranges (edit to match your soil):
    phi : 20 – 42 deg   (friction angle)
    c   : 0  – 15 kPa   (cohesion)

Output:
    lhs_results/lhs_samples.csv   — all summary stats, one row per sample
    lhs_results/sample_NNN/       — per-run output directories
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale

# ── Parameter bounds ──────────────────────────────────────────────────────────
PHI_MIN, PHI_MAX = 20.0, 42.0   # degrees
C_MIN,   C_MAX   = 0.0,  15000. # Pa  (0 – 15 kPa)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",       type=int,   default=30,   help="Number of LHS samples")
parser.add_argument("--workers", type=int,   default=1,    help="Parallel workers")
parser.add_argument("--omega",   type=float, default=0.2,  help="Angular velocity (rad/s)")
parser.add_argument("--settle",  type=int,   default=20000,help="Settlement steps")
parser.add_argument("--out",     default="lhs_results",    help="Output directory")
parser.add_argument("--leg",     default="all", choices=["flat","cleg","rcleg","all"])
parser.add_argument("--resume",  action="store_true",      help="Skip completed samples")
parser.add_argument("--seed",    type=int,   default=42,   help="RNG seed for LHS")
parser.add_argument("--threads", type=int,   default=None,
                    help="Taichi CPU threads per simulation (default: all). "
                         "Set to 2 when using --workers > 1.")
parser.add_argument("--timeout", type=int,   default=7200,
                    help="Per-simulation timeout in seconds (default: 7200 = 2h).")
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
SIM_PY  = os.path.join(HERE, "mpm_terradynamics.py")
OUT_DIR = os.path.join(HERE, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generate LHS design ───────────────────────────────────────────────────────
sampler  = LatinHypercube(d=2, seed=args.seed)
unit_lhs = sampler.random(n=args.n)
samples  = scale(unit_lhs, [PHI_MIN, C_MIN], [PHI_MAX, C_MAX])

design_path = os.path.join(OUT_DIR, "lhs_design.csv")
np.savetxt(design_path,
           np.column_stack([np.arange(args.n), samples]),
           delimiter=",", header="sample_id,phi_deg,c_Pa", comments="", fmt="%.6g")
print(f"LHS design ({args.n} samples) written to {design_path}")
print(f"phi range : {PHI_MIN}–{PHI_MAX} deg")
print(f"c range   : {C_MIN/1000:.1f}–{C_MAX/1000:.1f} kPa")
print(f"Workers   : {args.workers}   omega = {args.omega} rad/s\n")


def run_sample(idx, phi_deg, c_pa):
    """Run one simulation and return its summary dict (or None on failure)."""
    sample_dir  = os.path.join(OUT_DIR, f"sample_{idx:03d}")
    summary_path = os.path.join(sample_dir, "summary.json")

    if args.resume and os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        print(f"  [{idx:03d}] resumed  phi={phi_deg:.2f}° c={c_pa:.0f} Pa")
        return idx, s

    os.makedirs(sample_dir, exist_ok=True)

    cmd = [
        sys.executable, SIM_PY,
        "--phi",         f"{phi_deg:.6f}",
        "--c",           f"{c_pa:.6f}",
        "--leg",         args.leg,
        "--omega",       f"{args.omega}",
        "--settle",      f"{args.settle}",
        "--out",         sample_dir,
        "--summary_out", summary_path,
    ]
    if args.threads is not None:
        cmd += ["--threads", str(args.threads)]

    log_path = os.path.join(sample_dir, "stdout.log")
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                    timeout=args.timeout)
        if result.returncode != 0:
            print(f"  [{idx:03d}] FAILED  phi={phi_deg:.2f}° c={c_pa:.0f} Pa  "
                  f"(see {log_path})")
            return idx, None
    except subprocess.TimeoutExpired:
        print(f"  [{idx:03d}] TIMEOUT phi={phi_deg:.2f}° c={c_pa:.0f} Pa")
        return idx, None

    if not os.path.exists(summary_path):
        print(f"  [{idx:03d}] NO OUTPUT  phi={phi_deg:.2f}° c={c_pa:.0f} Pa")
        return idx, None

    with open(summary_path) as f:
        s = json.load(f)
    print(f"  [{idx:03d}] done     phi={phi_deg:.2f}° c={c_pa/1000:.2f} kPa  "
          f"  flat_Fz_peak={s.get('flat_Fz_peak_N', float('nan')):.4f} N")
    return idx, s


# ── Run all samples ───────────────────────────────────────────────────────────
print(f"Running {args.n} simulations ...\n")
rows = [None] * args.n

if args.workers == 1:
    for i, (phi, c) in enumerate(samples):
        idx, s = run_sample(i, phi, c)
        rows[idx] = s
else:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_sample, i, phi, c): i
                   for i, (phi, c) in enumerate(samples)}
        for fut in as_completed(futures):
            idx, s = fut.result()
            rows[idx] = s

# ── Assemble results CSV ──────────────────────────────────────────────────────
records = []
for i, s in enumerate(rows):
    if s is None:
        continue
    row = {"sample_id": i, "phi_deg": samples[i, 0], "c_Pa": samples[i, 1]}
    row.update(s)
    records.append(row)

if not records:
    print("\nNo successful runs — check sample_*/stdout.log for errors.")
    sys.exit(1)

df = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, "lhs_samples.csv")
df.to_csv(csv_path, index=False)

n_ok  = len(records)
n_bad = args.n - n_ok
print(f"\n{'─'*60}")
print(f"Completed: {n_ok}/{args.n} samples  ({n_bad} failed/skipped)")
print(f"Results  : {csv_path}")
print(f"\nColumns  : {list(df.columns)}")
print(f"\nNext step: python3 inverse_gp.py --data {csv_path}")
