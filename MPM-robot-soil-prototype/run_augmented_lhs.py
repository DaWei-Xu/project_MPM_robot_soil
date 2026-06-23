"""
run_augmented_lhs.py
====================
Augments the existing 10-sample LHS design to 30 samples using a greedy
maximin space-filling strategy.  Reuses existing simulation results (via
symlinks) and runs only the 20 new simulations.

Maximin criterion: each new point is placed at the location (among 3000
random candidates in the normalised [0,1]^2 space) that maximises its
minimum Euclidean distance to all already-selected points.  This ensures
the 20 additions fill the gaps left by the original 10-point LHS.

Usage:
    python3 run_augmented_lhs.py                        # default settings
    python3 run_augmented_lhs.py --resume               # skip done samples
    python3 run_augmented_lhs.py --n_new 20 --seed 77
"""

import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--old_dir",  default="lhs_results",    help="Existing LHS directory")
parser.add_argument("--new_dir",  default="lhs_results_30", help="Output directory")
parser.add_argument("--n_new",    type=int,   default=20,   help="New samples to add")
parser.add_argument("--omega",    type=float, default=5.0,  help="Angular velocity (rad/s)")
parser.add_argument("--settle",   type=int,   default=2000, help="Settlement steps")
parser.add_argument("--resume",   action="store_true",       help="Skip completed samples")
parser.add_argument("--seed",     type=int,   default=77,   help="RNG seed for candidate generation")
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
OLD_DIR = os.path.join(HERE, args.old_dir)
NEW_DIR = os.path.join(HERE, args.new_dir)
SIM_PY  = os.path.join(HERE, "mpm_terradynamics.py")

PHI_MIN, PHI_MAX = 20.0, 42.0
C_MIN,   C_MAX   = 0.0,  15000.0

os.makedirs(NEW_DIR, exist_ok=True)

# ── Load existing design ──────────────────────────────────────────────────────
old_design = pd.read_csv(os.path.join(OLD_DIR, "lhs_design.csv"))
n_old = len(old_design)
print(f"Existing design: {n_old} samples from {OLD_DIR}")
print(f"  phi range: {old_design['phi_deg'].min():.1f}–{old_design['phi_deg'].max():.1f}°")
print(f"  c range  : {old_design['c_Pa'].min()/1000:.2f}–{old_design['c_Pa'].max()/1000:.2f} kPa")

# Normalise to [0,1]^2 for distance computation
old_pts_norm = np.column_stack([
    (old_design["phi_deg"].values - PHI_MIN) / (PHI_MAX - PHI_MIN),
    old_design["c_Pa"].values / C_MAX
])

# ── Maximin greedy augmentation ───────────────────────────────────────────────
N_CAND = 3000
rng = np.random.default_rng(args.seed)
candidates = rng.uniform(0, 1, size=(N_CAND, 2))

selected   = old_pts_norm.copy()
new_norm   = []

print(f"\nGreedy maximin: selecting {args.n_new} new points from {N_CAND} candidates ...")
for k in range(args.n_new):
    diffs = candidates[:, None, :] - selected[None, :, :]    # (C, S, 2)
    dists = np.min(np.linalg.norm(diffs, axis=2), axis=1)    # (C,)
    best  = int(np.argmax(dists))
    new_norm.append(candidates[best])
    selected   = np.vstack([selected, candidates[best]])
    candidates = np.delete(candidates, best, axis=0)

new_norm = np.array(new_norm)
new_phi  = new_norm[:, 0] * (PHI_MAX - PHI_MIN) + PHI_MIN
new_c    = new_norm[:, 1] * C_MAX

print("\nNew sample parameters:")
for i, (phi, c) in enumerate(zip(new_phi, new_c)):
    print(f"  [{n_old + i:03d}]  phi={phi:.2f}°  c={c/1000:.3f} kPa")

# ── Symlink existing sample directories ───────────────────────────────────────
print(f"\nLinking {n_old} existing sample dirs ...")
for i in range(n_old):
    src = os.path.abspath(os.path.join(OLD_DIR, f"sample_{i:03d}"))
    dst = os.path.join(NEW_DIR, f"sample_{i:03d}")
    if not os.path.exists(dst):
        os.symlink(src, dst)

# ── Save merged design CSV ────────────────────────────────────────────────────
new_rows_df = pd.DataFrame({
    "sample_id": range(n_old, n_old + args.n_new),
    "phi_deg":   new_phi,
    "c_Pa":      new_c,
})
full_design = pd.concat(
    [old_design[["sample_id", "phi_deg", "c_Pa"]], new_rows_df],
    ignore_index=True
)
full_design.to_csv(os.path.join(NEW_DIR, "lhs_design.csv"), index=False)
print(f"Full design ({n_old + args.n_new} pts) saved: {NEW_DIR}/lhs_design.csv\n")

# ── Run new simulations ───────────────────────────────────────────────────────
print(f"Running {args.n_new} simulations (omega={args.omega}, settle={args.settle}) ...")
new_summaries = {}

for i, (phi, c) in enumerate(zip(new_phi, new_c)):
    sid        = n_old + i
    sample_dir = os.path.join(NEW_DIR, f"sample_{sid:03d}")
    summary_p  = os.path.join(sample_dir, "summary.json")
    os.makedirs(sample_dir, exist_ok=True)

    if args.resume and os.path.exists(summary_p):
        print(f"  [sample_{sid:03d}] resumed")
        with open(summary_p) as f:
            new_summaries[sid] = json.load(f)
        continue

    print(f"  [sample_{sid:03d}]  phi={phi:.2f}°  c={c/1000:.3f} kPa ...", flush=True)
    cmd = [sys.executable, SIM_PY,
           "--phi", f"{phi:.6f}", "--c", f"{c:.6f}",
           "--leg", "all",
           "--omega", f"{args.omega}", "--settle", f"{args.settle}",
           "--out", sample_dir, "--summary_out", summary_p]
    log = os.path.join(sample_dir, "stdout.log")
    with open(log, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)

    if r.returncode != 0 or not os.path.exists(summary_p):
        print(f"    FAILED — see {log}")
    else:
        with open(summary_p) as f:
            new_summaries[sid] = json.load(f)
        fz = new_summaries[sid].get("flat_Fz_peak_N", float("nan"))
        print(f"    done  flat_Fz_peak={fz:.3f} N")

# ── Merge into lhs_samples.csv ────────────────────────────────────────────────
old_samples = pd.read_csv(os.path.join(OLD_DIR, "lhs_samples.csv"))

new_records = []
for i, (phi, c) in enumerate(zip(new_phi, new_c)):
    sid = n_old + i
    if sid not in new_summaries:
        print(f"  WARNING: sample_{sid:03d} missing summary — excluded")
        continue
    sm  = new_summaries[sid]
    row = {"sample_id": sid, "phi_deg": phi, "c_Pa": c}
    for k, v in sm.items():
        if k not in {"phi_deg", "c_Pa"}:
            row[k] = v
    new_records.append(row)

merged = pd.concat([old_samples, pd.DataFrame(new_records)], ignore_index=True)
csv_path = os.path.join(NEW_DIR, "lhs_samples.csv")
merged.to_csv(csv_path, index=False)

print(f"\n{'─'*60}")
print(f"Merged CSV: {csv_path}  ({len(merged)} rows)")
print(f"\nNext steps:")
print(f"  python3 compute_cleg_indicators.py \\")
print(f"      --lhs_dir {args.new_dir} --out cleg_results_30 \\")
print(f"      --syn_obs_dir synthetic_obs --syn_obs_out cleg_results_30/syn_cleg_obs.json")
