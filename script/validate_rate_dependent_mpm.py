"""
validate_rate_dependent_mpm.py
==============================
Validates that the Perzyna rate-dependent Drucker-Prager produces physically
correct, measurably different force curves across a range of relaxation times.

Perzyna physics (rate STIFFENING — not softening):
  Large η (slow relaxation) → viscoplastic flow is slow → stress accumulates
    above the yield surface → material appears stiffer → HIGHER forces
  Small η (fast relaxation) → stress returns to yield surface quickly →
    approaches rate-independent limit → forces ≈ baseline

Correct ordering: baseline ≤ fast ≤ slow

Three runs (flat leg, ω=10 rad/s for speed):
  BASELINE  VP_ETA = 0      rate-independent D-P  (limit η → 0)
  FAST      VP_ETA = 0.05 s Perzyna fast relaxation: forces close to baseline
  SLOW      VP_ETA = 100 s  Perzyna slow relaxation: stress stays above yield
                             surface → rate stiffening → higher forces

Note on the ti.min fix:
  The fix (validate_dp_fixes.py) changes behaviour when dg_vp > dg_ri.  At
  the shallow soil stresses in this experiment (~100–500 Pa), the difference
  per step is small — its effect on macroscopic force curves is < 0.1%.  The
  unit tests in validate_dp_fixes.py are the right place to verify the
  fix at the return-mapping level; this script validates the model physics.

Usage:
  python3 script/validate_rate_dependent_mpm.py
  python3 script/validate_rate_dependent_mpm.py --keep   # keep run dirs
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_PY  = os.path.join(ROOT, "MPM-robot-soil-prototype", "mpm_terradynamics.py")
OUT_DIR = os.path.join(ROOT, "data", "synthetic_data", "validate_rate_dependent")

RUNS = {
    "baseline": dict(vp_eta=0.0,   label="Rate-independent (η=0)",       color="#1B3A5C"),
    "fast":     dict(vp_eta=0.05,  label="Perzyna fast  (η=0.05 s)",     color="#27AE60"),
    "slow":     dict(vp_eta=100.0, label="Perzyna slow  (η=100 s)",      color="#C0392B"),
}

COMMON_ARGS = [
    "--leg", "flat",
    "--omega", "10.0",
    "--settle", "50",
    "--threads", "2",
    "--vp_n", "1",
    "--vp_stress_ref", "1000",
]

# ── Pass/fail harness ──────────────────────────────────────────────────────────
_pass = 0
_fail = 0

def check(name, cond, detail=""):
    global _pass, _fail
    tag = "  PASS" if cond else "  FAIL"
    print(f"{tag}  {name}")
    if not cond and detail:
        print(f"        {detail}")
    (_pass if cond else _fail).__class__  # silence lint
    if cond:
        _pass += 1
    else:
        _fail += 1


# ── Simulation runner ──────────────────────────────────────────────────────────

def run_sim(name, cfg, out_dir):
    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    summary = os.path.join(run_dir, "summary.json")
    cmd = [
        sys.executable, SIM_PY,
        "--out", run_dir,
        "--summary_out", summary,
        "--vp_eta", str(cfg["vp_eta"]),
        *COMMON_ARGS,
    ]
    print(f"\n  Running {name}  (VP_ETA={cfg['vp_eta']} s) ...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(SIM_PY))
    if result.returncode != 0:
        print(result.stderr[-2000:])
        sys.exit(f"Simulation failed for run '{name}'")
    csv_path = os.path.join(run_dir, "terradyn_flat.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"Expected CSV not found: {csv_path}")
    return csv_path


def load_forces(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    theta_deg = -data[:, 1]   # negate → Li et al. sign convention
    Fz_mN     =  data[:, 2]
    Fx_mN     =  data[:, 3]
    return theta_deg, Fz_mN, Fx_mN


def smooth(arr):
    win = max(1, len(arr) // 80)
    return np.convolve(arr, np.ones(win) / win, mode="same")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true",
                    help="Keep per-run output directories after validation")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Run simulations ───────────────────────────────────────────────────────
    csvs = {}
    for name, cfg in RUNS.items():
        csvs[name] = run_sim(name, cfg, OUT_DIR)

    # ── Load and smooth ───────────────────────────────────────────────────────
    forces = {}
    for name, path in csvs.items():
        th, Fz, Fx = load_forces(path)
        forces[name] = dict(th=th, Fz=Fz, Fx=Fx,
                            Fz_sm=smooth(Fz), Fx_sm=smooth(Fx))

    base = forces["baseline"]
    fast = forces["fast"]
    slow = forces["slow"]

    # ── Statistics ────────────────────────────────────────────────────────────
    def stats(d):
        return (float(np.max(d["Fz_sm"])),
                float(np.max(np.abs(d["Fx_sm"]))),
                float(np.trapz(d["Fz_sm"], d["th"])))

    bp = stats(base)
    fp = stats(fast)
    sp = stats(slow)

    def pct(a, b):
        return 100.0 * (b - a) / (abs(a) + 1e-12)   # signed: + means b > a

    print("\n── Summary statistics ────────────────────────────────────────────────")
    print(f"{'Run':<10}  {'Fz_peak (mN)':>14}  {'Fx_peak (mN)':>14}  {'∫Fz (mN·deg)':>14}")
    for label, s in [("baseline", bp), ("fast", fp), ("slow", sp)]:
        print(f"  {label:<8}  {s[0]:>14.1f}  {s[1]:>14.1f}  {s[2]:>14.1f}")
    print()
    print(f"  fast vs baseline:  Fz {pct(bp[0],fp[0]):+.2f}%  |  ∫Fz {pct(bp[2],fp[2]):+.2f}%")
    print(f"  slow vs baseline:  Fz {pct(bp[0],sp[0]):+.2f}%  |  ∫Fz {pct(bp[2],sp[2]):+.2f}%")
    print(f"  slow vs fast:      Fz {pct(fp[0],sp[0]):+.2f}%  |  ∫Fz {pct(fp[2],sp[2]):+.2f}%")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n── Validation checks ─────────────────────────────────────────────────")

    # 1. Slow VP_ETA must produce noticeably higher forces than baseline.
    #    Perzyna rate stiffening: large η → stress stays above yield surface.
    #    If this fails, the Perzyna code has no effect (model is broken).
    SLOW_MIN = 2.0   # % — conservative floor; observed ~6-7%
    check(f"slow VP_ETA: Fz peak > baseline by at least {SLOW_MIN}%  [rate stiffening active]",
          pct(bp[0], sp[0]) > SLOW_MIN,
          f"slow={sp[0]:.1f}  baseline={bp[0]:.1f}  diff={pct(bp[0],sp[0]):+.2f}%")

    check(f"slow VP_ETA: ∫Fz > baseline  [stress consistently above yield surface]",
          sp[2] > bp[2],
          f"slow={sp[2]:.1f}  baseline={bp[2]:.1f}")

    # 2. Fast VP_ETA must be close to baseline.
    #    Small η → fast relaxation → approaches rate-independent limit.
    #    This is also what the OLD (pre-fix) code gave for any VP_ETA:
    #    if we still see near-zero diff here, it means the fast Perzyna
    #    limit is working correctly (not that the fix is absent).
    FAST_MAX = 2.0   # % — fast relaxation should be close to rate-independent
    check(f"fast VP_ETA: Fz peak within {FAST_MAX}% of baseline  [rapid relaxation → RI limit]",
          abs(pct(bp[0], fp[0])) < FAST_MAX,
          f"fast={fp[0]:.1f}  baseline={bp[0]:.1f}  diff={pct(bp[0],fp[0]):+.2f}%")

    # 3. Correct monotone ordering: baseline ≤ fast ≤ slow
    #    (rate stiffening increases with η)
    check("force ordering: baseline ≤ fast ≤ slow  [monotone rate stiffening]",
          bp[0] <= fp[0] * 1.02 and fp[0] <= sp[0] * 1.02,
          f"baseline={bp[0]:.1f}  fast={fp[0]:.1f}  slow={sp[0]:.1f}")

    # 4. Slow VP_ETA must produce higher forces than fast VP_ETA.
    #    Distinguishes the two Perzyna regimes from each other.
    check("slow VP_ETA forces > fast VP_ETA forces  [η sensitivity working]",
          sp[0] > fp[0],
          f"slow={sp[0]:.1f}  fast={fp[0]:.1f}")

    # 5. Sanity: all runs produced non-trivial lift forces
    for label, s in [("baseline", bp), ("fast", fp), ("slow", sp)]:
        check(f"{label}: Fz peak > 0  [leg produces lift]",
              s[0] > 0.0,
              f"Fz_peak = {s[0]:.1f} mN")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (az, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Perzyna rate-dependent D-P — flat leg, ω = 10 rad/s\n"
        "Rate stiffening: slow η → stress above yield surface → higher forces",
        fontsize=10)

    for name, d in forces.items():
        cfg = RUNS[name]
        az.plot(d["th"], d["Fz_sm"],  color=cfg["color"], lw=2.0, label=cfg["label"])
        az.plot(d["th"], d["Fz"],     color=cfg["color"], lw=0.4, alpha=0.2)
        ax2.plot(d["th"], d["Fx_sm"], color=cfg["color"], lw=2.0, label=cfg["label"])
        ax2.plot(d["th"], d["Fx"],    color=cfg["color"], lw=0.4, alpha=0.2)

    for ax in (az, ax2):
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Leg angle θ (deg)")
        ax.set_xlim(-135, 135)
        ax.set_xticks([-90, 0, 90])
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, framealpha=0.9)

    az.set_ylabel("Lift  $F_z$  (mN)")
    ax2.set_ylabel("Thrust  $F_x$  (mN)")
    az.set_title("Vertical force  (Lift $F_z$)")
    ax2.set_title("Horizontal force  (Thrust $F_x$)")

    az.text(0.02, 0.04,
            f"slow vs baseline  Fz: {pct(bp[0],sp[0]):+.1f}%\n"
            f"fast vs baseline  Fz: {pct(bp[0],fp[0]):+.1f}%",
            transform=az.transAxes, fontsize=7.5,
            bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.3"))

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "rate_dependent_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot → {plot_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep:
        for name in RUNS:
            shutil.rmtree(os.path.join(OUT_DIR, name), ignore_errors=True)

    # ── Result ────────────────────────────────────────────────────────────────
    total = _pass + _fail
    print()
    print("─" * 56)
    print(f"  {_pass}/{total} passed" + ("  ✓" if _fail == 0 else f"  — {_fail} FAILED"))
    sys.exit(0 if _fail == 0 else 1)


if __name__ == "__main__":
    main()
