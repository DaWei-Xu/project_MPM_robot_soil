"""
extract_li_obs.py
=================
Extract summary statistics from Li et al. (2013) digitized experimental
data (Fig. S12 A/B/C) and convert them to the same format expected by
inverse_gp.py.

Omega-scaling note
------------------
The LHS training data was generated at omega=5.0 rad/s for speed, while
the Li et al. experiment used omega=0.2 rad/s.  Forces do NOT scale
linearly with omega in MPM/granular media — empirically we observe a
~11x ratio (vs 25x for linear scaling).

This script applies an empirical force scale factor:
    FORCE_SCALE = mean(LHS flat_Fz_peak) / exp_flat_Fz_peak
                = ~11  (computed from the actual LHS CSV)

Angle observables (Fz_peak_ang_deg) are left unscaled because the
peak angle is largely omega-independent in the quasi-static regime —
confirmed by comparison: experiment -37.6 deg vs LHS range -33 to -50 deg.

Usage:
    python3 extract_li_obs.py
    python3 extract_li_obs.py --lhs lhs_results/lhs_samples.csv

Output:
    li_obs/li_obs.json   — ready for inverse_gp.py --obs
"""

import argparse, json, os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--lhs", default="lhs_results/lhs_samples.csv",
                    help="LHS samples CSV (used to compute force scale factor)")
parser.add_argument("--out", default="li_obs", help="Output directory")
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
DIG_DIR = os.path.join(HERE, "..", "Liter Data", "Li Science", "Data_digitalized")
OUT_DIR = os.path.join(HERE, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

LEG_CSV = {
    "flat":  "Fig_S12_B.csv",   # cols 0,1=Fx-exp  4,5=Fz-exp
    "cleg":  "Fig_S12_A.csv",
    "rcleg": "Fig_S12_C.csv",
}

# ── Load LHS data to compute empirical force scale factor ─────────────────────
lhs_path = os.path.join(HERE, args.lhs)
df = pd.read_csv(lhs_path)
lhs_flat_fz_mean = float(df["flat_Fz_peak_N"].mean())
print(f"LHS flat_Fz_peak mean (omega=5): {lhs_flat_fz_mean:.2f} N")


def load_exp_curves(leg_type):
    """Return (theta_deg, Fz_N), (theta_deg, Fx_N) from digitized CSV."""
    path = os.path.join(DIG_DIR, LEG_CSV[leg_type])
    raw  = np.genfromtxt(path, delimiter=",", skip_header=2,
                          filling_values=np.nan)

    def pull(cx, cy):
        mask = ~(np.isnan(raw[:, cx]) | np.isnan(raw[:, cy]))
        xd   = np.degrees(raw[mask, cx])
        y    = raw[mask, cy]
        s    = np.argsort(xd)
        return xd[s], y[s]

    fx_th, fx_N = pull(0, 1)
    fz_th, fz_N = pull(4, 5)
    return (fz_th, fz_N), (fx_th, fx_N)


# ── Extract summary stats from each leg ───────────────────────────────────────
obs = {}
scale_factors = {}

for leg_type in ["flat", "cleg", "rcleg"]:
    path = os.path.join(DIG_DIR, LEG_CSV[leg_type])
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {leg_type}")
        continue

    (fz_th, fz_N), (fx_th, fx_N) = load_exp_curves(leg_type)

    # Smooth experimental curves (moving average, window=5 pts)
    win   = min(5, len(fz_N))
    fz_sm = np.convolve(fz_N, np.ones(win)/win, mode="same")
    fx_sm = np.convolve(fx_N, np.ones(win)/win, mode="same")

    # Raw experimental summary stats (omega=0.2)
    fz_peak_raw  = float(np.max(fz_sm))
    fz_peak_ang  = float(fz_th[np.argmax(fz_sm)])
    fx_peak_raw  = float(np.max(np.abs(fx_sm)))
    fz_integral_raw = float(np.trapezoid(fz_sm, fz_th))

    # Empirical force scale factor for this leg:
    # ratio of LHS mean force (omega=5) to experimental force (omega=0.2)
    lhs_col = f"{leg_type}_Fz_peak_N"
    if lhs_col in df.columns:
        scale = float(df[lhs_col].mean()) / fz_peak_raw
    else:
        # Fallback: use flat-leg ratio
        scale = lhs_flat_fz_mean / obs.get("flat_Fz_peak_N", fz_peak_raw)

    scale_factors[leg_type] = scale

    # Scale force magnitudes to omega=5 equivalent; angles unchanged
    obs[f"{leg_type}_Fz_peak_N"]       = fz_peak_raw  * scale
    obs[f"{leg_type}_Fz_peak_ang_deg"] = fz_peak_ang          # no scaling
    obs[f"{leg_type}_Fx_peak_N"]       = fx_peak_raw  * scale
    obs[f"{leg_type}_Fz_integral_Ndeg"] = fz_integral_raw * scale

    print(f"\n  [{leg_type}]  scale factor = {scale:.2f}x")
    print(f"    Fz_peak : {fz_peak_raw:.3f} N (exp)  → {obs[f'{leg_type}_Fz_peak_N']:.2f} N (scaled)")
    print(f"    Fz_peak_ang: {fz_peak_ang:.1f} deg  (no scaling)")
    print(f"    Fx_peak : {fx_peak_raw:.3f} N (exp)  → {obs[f'{leg_type}_Fx_peak_N']:.2f} N (scaled)")
    print(f"    Fz_integral: {fz_integral_raw:.1f} N·deg → {obs[f'{leg_type}_Fz_integral_Ndeg']:.1f} N·deg (scaled)")

# ── Save ──────────────────────────────────────────────────────────────────────
obs["_source"]    = "Li et al. (2013) Fig. S12 digitized data"
obs["_omega_exp"] = 0.2
obs["_omega_sim"] = 5.0
obs["_scale_factors"] = scale_factors
obs["_note"] = (
    "Force magnitudes scaled by empirical ratio LHS_mean(omega=5)/exp(omega=0.2). "
    "Angles left unscaled. Use results qualitatively — re-run LHS at omega=0.2 "
    "for quantitative accuracy."
)

out_path = os.path.join(OUT_DIR, "li_obs.json")
with open(out_path, "w") as f:
    json.dump(obs, f, indent=2)

print(f"\nLi et al. observations saved to: {out_path}")
print(f"\nNext step:")
print(f"  python3 inverse_gp.py --data {args.lhs} --obs {out_path}")
