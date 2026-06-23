"""Rebuild lhs_samples.csv from individual sample CSV files for omega=0.2 LHS."""
import numpy as np
import pandas as pd
import os

BASE  = os.path.join(os.path.dirname(__file__), "lhs_results_omega02")
E_MOD = 20e6
LEG_W = 0.0254   # m (2D→3D conversion, matches mpm_terradynamics.py)

design = pd.read_csv(os.path.join(BASE, "lhs_design.csv"))

def peak(sdir, leg):
    p = os.path.join(sdir, f"terradyn_{leg}.csv")
    if not os.path.exists(p):
        return None
    d = np.genfromtxt(p, delimiter=",", skip_header=1)
    if d.ndim < 2 or len(d) == 0:
        return None
    # CSV stores Fz in mN (already multiplied by LEG_W inside mpm_terradynamics.py).
    # Divide by 1000 to get N (matches compute_cleg_indicators.py convention).
    # Apply same smoothing window as the simulation summary (N_ROT // 80).
    Fz_mN = d[:, 2]
    Fx_mN = d[:, 3]
    th    = -d[:, 1]                  # negate to match Li et al. convention
    win   = max(1, len(Fz_mN) // 80)
    kern  = np.ones(win) / win
    Fz_sm = np.convolve(Fz_mN, kern, mode="same")
    Fx_sm = np.convolve(Fx_mN, kern, mode="same")
    Fz_N  = Fz_sm / 1000.
    Fx_N  = Fx_sm / 1000.
    return {
        "Fz_peak_N":        float(Fz_N.max()),
        "Fz_peak_ang_deg":  float(th[np.argmax(Fz_N)]),
        "Fx_peak_N":        float(Fx_N.max()),
        "Fz_integral_Ndeg": float(np.trapezoid(Fz_N, th)),
    }

rows = []
for _, row in design.iterrows():
    sid  = int(row["sample_id"])
    sdir = os.path.join(BASE, f"sample_{sid:03d}")
    flat  = peak(sdir, "flat")
    cleg  = peak(sdir, "cleg")
    rcleg = peak(sdir, "rcleg")
    if flat is None:
        continue
    r = {"sample_id": sid, "phi_deg": row["phi_deg"], "c_Pa": row["c_Pa"], "E_Pa": E_MOD}
    for name, res in [("flat", flat), ("cleg", cleg), ("rcleg", rcleg)]:
        for k, v in (res or {}).items():
            r[f"{name}_{k}"] = v
    rows.append(r)

df  = pd.DataFrame(rows)
out = os.path.join(BASE, "lhs_samples_rebuilt.csv")
df.to_csv(out, index=False)

print(f"Rebuilt {len(df)} samples → {out}")
for leg in ["flat", "cleg", "rcleg"]:
    col = f"{leg}_Fz_peak_N"
    if col in df.columns:
        valid = df[col].dropna()
        print(f"  {leg:6s} Fz_peak_N: [{valid.min():.3f}, {valid.max():.3f}]  n={len(valid)}")
