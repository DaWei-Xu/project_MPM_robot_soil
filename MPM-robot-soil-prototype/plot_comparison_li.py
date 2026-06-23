"""
plot_comparison_li.py
=====================
Runs MPM at the inferred (phi, c) and at the Li et al. literature defaults,
then plots simulated Fz and Fx vs theta alongside the digitized experiment.

Two simulation runs:
  1. Inferred parameters from inverse_gp.py  (read from li_obs/posterior_summary.json)
  2. Literature baseline: phi=35 deg, c=10 Pa  (dry Yuma Sand)

Both are run at omega=5.0 for speed, then scaled to omega=0.2 equivalent
using the empirical per-leg scale factors stored in li_obs/li_obs.json.

Usage:
    python3 plot_comparison_li.py
    python3 plot_comparison_li.py --omega 0.2   # slow but no scaling needed
"""

import argparse, json, os, subprocess, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--omega",    type=float, default=5.0)
parser.add_argument("--settle",   type=int,   default=2000)
parser.add_argument("--post",     default="li_obs/posterior_summary.json")
parser.add_argument("--liobs",    default="li_obs/li_obs.json")
parser.add_argument("--out",      default="li_obs")
parser.add_argument("--dir_inf",  default="comparison_inferred",
                    help="Cache dir for inferred-params simulation (default: comparison_inferred)")
parser.add_argument("--dir_base", default="comparison_baseline",
                    help="Cache dir for baseline simulation (default: comparison_baseline)")
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
DIG_DIR = os.path.join(HERE, "..", "data", "Liter Data", "Li Science", "Data_digitalized")
SIM_PY  = os.path.join(HERE, "mpm_terradynamics.py")
OUT_DIR = os.path.join(HERE, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

LEG_CSV_EXP = {
    "flat":  "Fig_S12_B.csv",
    "cleg":  "Fig_S12_A.csv",
    "rcleg": "Fig_S12_C.csv",
}
LEG_LABELS = {
    "flat":  "Flat leg",
    "cleg":  "C-leg (κ=+1/R)",
    "rcleg": "Rev. C-leg (κ=−1/R)",
}
COLORS = {"flat": "#1B3A5C", "cleg": "#C0392B", "rcleg": "#27AE60"}

# ── Load posterior and scale factors ─────────────────────────────────────────
with open(os.path.join(HERE, args.post)) as f:
    post = json.load(f)
phi_inf = post["phi_median_deg"]
c_inf   = post["c_median_Pa"]
print(f"Inferred params : phi={phi_inf:.2f} deg, c={c_inf:.1f} Pa")

with open(os.path.join(HERE, args.liobs)) as f:
    liobs = json.load(f)
scale_factors = liobs.get("_scale_factors", {"flat": 11.53, "cleg": 11.25, "rcleg": 18.13})

# If running at omega != 0.2, we need to scale; if running at 0.2, scale=1
if abs(args.omega - 0.2) < 0.01:
    scale_factors = {k: 1.0 for k in scale_factors}
    print("Running at experimental omega=0.2 — no force scaling applied.")
else:
    print(f"Running at omega={args.omega} — will scale by 1/scale_factor per leg.")

# ── Helper: run one simulation and return output directory ────────────────────
def run_sim(phi_deg, c_pa, label, cache_dir=None):
    out_d = cache_dir if cache_dir else os.path.join(HERE, f"comparison_{label}")
    os.makedirs(out_d, exist_ok=True)
    done_flag = os.path.join(out_d, "terradyn_flat.csv")
    if os.path.exists(done_flag):
        print(f"  [{label}] already exists — skipping simulation")
        return out_d
    print(f"  [{label}] running phi={phi_deg:.2f} deg, c={c_pa:.0f} Pa ...")
    cmd = [sys.executable, SIM_PY,
           "--phi", f"{phi_deg:.6f}", "--c", f"{c_pa:.6f}",
           "--leg", "all", "--omega", f"{args.omega}",
           "--settle", f"{args.settle}", "--out", out_d]
    log = os.path.join(out_d, "stdout.log")
    with open(log, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    FAILED — see {log}")
        return None
    print(f"    Done.")
    return out_d

# ── Helper: load experimental data for one leg ────────────────────────────────
def load_exp(leg_type):
    path = os.path.join(DIG_DIR, LEG_CSV_EXP[leg_type])
    raw  = np.genfromtxt(path, delimiter=",", skip_header=2, filling_values=np.nan)
    def pull(cx, cy):
        mask = ~(np.isnan(raw[:, cx]) | np.isnan(raw[:, cy]))
        xd = np.degrees(raw[mask, cx])
        y  = raw[mask, cy]
        s  = np.argsort(xd)
        return xd[s], y[s]
    return pull(0, 1), pull(4, 5)   # (fx_th, fx_N), (fz_th, fz_N)

# ── Helper: load simulated curves from CSV ────────────────────────────────────
def load_sim(sim_dir, leg_type, scale):
    path = os.path.join(sim_dir, f"terradyn_{leg_type}.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    th   = -data[:, 1]          # negate: CSV stores raw sim theta (+135→-135); Li convention is -135→+135
    Fz   = data[:, 2] / 1000.  # mN → N
    Fx   = data[:, 3] / 1000.
    # Scale to omega=0.2 equivalent
    Fz /= scale;  Fx /= scale
    # Smooth
    win = max(1, len(Fz) // 80)
    Fz_sm = np.convolve(Fz, np.ones(win)/win, mode="same")
    Fx_sm = np.convolve(Fx, np.ones(win)/win, mode="same")
    return th, Fz_sm, Fx_sm

# ── Run simulations ───────────────────────────────────────────────────────────
print("\nRunning simulations ...")
dir_inf  = run_sim(phi_inf, c_inf, "inferred",
                   cache_dir=os.path.join(HERE, args.dir_inf))
dir_base = run_sim(35.0,    10.0,  "baseline",
                   cache_dir=os.path.join(HERE, args.dir_base))

if dir_inf is None or dir_base is None:
    print("One or both simulations failed. Exiting.")
    sys.exit(1)

# ── Plot ─────────────────────────────────────────────────────────────────────
LEGS = ["flat", "cleg", "rcleg"]
fig = plt.figure(figsize=(14, 8))
fig.suptitle(
    "MPM vs. Li et al. (2013) Experiment — Yuma Sand\n"
    f"Inferred: φ={phi_inf:.1f}°, c={c_inf/1000:.1f} kPa   "
    f"Baseline: φ=35°, c=0.01 kPa  (literature)",
    fontsize=11)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

for col, leg in enumerate(LEGS):
    sc = scale_factors.get(leg, 11.5)
    (fx_th_e, fx_N_e), (fz_th_e, fz_N_e) = load_exp(leg)
    th_i, Fz_i, Fx_i = load_sim(dir_inf,  leg, sc)
    th_b, Fz_b, Fx_b = load_sim(dir_base, leg, sc)
    clr = COLORS[leg]

    # ── Fz row ──────────────────────────────────────────────────────────────
    ax_z = fig.add_subplot(gs[0, col])
    ax_z.plot(fz_th_e, fz_N_e, "k--", lw=1.6, alpha=0.85, label="Experiment")
    ax_z.plot(th_b, Fz_b, color=clr, lw=1.4, alpha=0.45, ls=":",
              label=f"Baseline (φ=35°, c≈0)")
    ax_z.plot(th_i, Fz_i, color=clr, lw=2.0,
              label=f"Inferred (φ={phi_inf:.1f}°, c={c_inf/1000:.1f} kPa)")
    ax_z.axhline(0, color="gray", lw=0.5, ls=":")
    ax_z.axvline(0, color="gray", lw=0.5, ls=":")
    ax_z.set_title(LEG_LABELS[leg], fontsize=10)
    ax_z.set_ylabel("Lift  $F_z$  (N)", fontsize=9)
    ax_z.set_xlim(-135, 135);  ax_z.set_xticks([-90, 0, 90])
    ax_z.legend(fontsize=7.5, framealpha=0.85)
    ax_z.grid(True, alpha=0.2, lw=0.5)

    # ── Fx row ──────────────────────────────────────────────────────────────
    ax_x = fig.add_subplot(gs[1, col])
    ax_x.plot(fx_th_e, fx_N_e, "k--", lw=1.6, alpha=0.85, label="Experiment")
    ax_x.plot(th_b, Fx_b, color=clr, lw=1.4, alpha=0.45, ls=":")
    ax_x.plot(th_i, Fx_i, color=clr, lw=2.0)
    ax_x.axhline(0, color="gray", lw=0.5, ls=":")
    ax_x.axvline(0, color="gray", lw=0.5, ls=":")
    ax_x.set_ylabel("Thrust  $F_x$  (N)", fontsize=9)
    ax_x.set_xlabel("Leg angle  θ  (deg)", fontsize=9)
    ax_x.set_xlim(-135, 135);  ax_x.set_xticks([-90, 0, 90])
    ax_x.grid(True, alpha=0.2, lw=0.5)

out_path = os.path.join(OUT_DIR, "comparison_li_inferred.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nComparison plot: {out_path}")
