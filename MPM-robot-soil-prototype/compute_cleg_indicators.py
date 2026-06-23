"""
compute_cleg_indicators.py
==========================
Recomputes GP training data and Li et al. observations using principled,
omega-independent indicators derived from the full cleg Fz(theta) curve.

Rationale for each indicator
-----------------------------
  cleg_Fz_peak_N       : Raw peak lift — retains amplitude information;
                          paired with the ratio below to allow GP to separate
                          magnitude from shape.
  cleg_Fx_over_Fz      : Thrust/lift ratio at respective peaks — dimensionless,
                          omega-independent, directly reflects contact friction
                          geometry (sensitive to phi, less so to c).
  cleg_Fz_zero_left_deg: Theta where Fz first crosses zero (leg entering soil) —
                          purely shape-based, omega-independent, sensitive to phi.
  cleg_Fz_zero_right_deg: Theta where Fz last crosses zero (leg exiting soil) —
                          same rationale; together with zero_left gives width.
  cleg_Fz_width_deg    : Width of positive-Fz window = zero_right - zero_left —
                          compact shape descriptor, redundant with the two
                          crossings but more robust to noise at the endpoints.
  cleg_pca1, cleg_pca2 : Top-2 PCA scores of the NORMALISED Fz curve (divided
                          by its peak so amplitude is removed) — captures
                          residual shape variation not explained by the above
                          scalars. PCA basis is fit on LHS curves only, then
                          applied identically to the Li et al. curve.

Outputs
-------
  cleg_results/lhs_cleg_indicators.csv  — new LHS training table
  cleg_results/li_cleg_obs.json         — Li et al. observations in same format
  cleg_results/pca_basis.npz            — saved PCA basis for reproducibility

Usage
-----
  python3 compute_cleg_indicators.py
  python3 inverse_gp.py \\
      --data cleg_results/lhs_cleg_indicators.csv \\
      --obs  cleg_results/li_cleg_obs.json \\
      --out  cleg_results
"""

import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument("--lhs_dir",    default="lhs_results",
                    help="LHS results directory (contains lhs_design.csv and sample_NNN/ dirs).")
parser.add_argument("--out",        default="cleg_results",
                    help="Output directory for indicators CSV, PCA basis, and obs JSONs.")
parser.add_argument("--syn_obs_dir", default="",
                    help="Extract cleg indicators from this simulation directory "
                         "and save as a synthetic obs JSON using the saved PCA basis.")
parser.add_argument("--syn_obs_out", default="",
                    help="Output path for synthetic obs JSON (default: <out>/syn_cleg_obs.json).")
args = parser.parse_args()

HERE    = os.path.dirname(os.path.abspath(__file__))
LHS_DIR = os.path.join(HERE, args.lhs_dir)
DIG_DIR = os.path.join(HERE, "..", "Liter Data", "Li Science", "Data_digitalized")
OUT_DIR = os.path.join(HERE, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

# Default syn_obs_out lives inside OUT_DIR
if not args.syn_obs_out:
    args.syn_obs_out = os.path.join(args.out, "syn_cleg_obs.json")

# Common theta grid for interpolation / PCA  (-130 to +130 avoids edge noise)
THETA_GRID = np.linspace(-130., 130., 200)
N_PCA      = 2       # number of PCA components to keep

# ── Helper: load full cleg simulation curve ───────────────────────────────────
def load_sim_cleg(sample_dir):
    """Return (theta_deg [Li conv], Fz_N, Fx_N) arrays from terradyn_cleg.csv."""
    path = os.path.join(sample_dir, "terradyn_cleg.csv")
    if not os.path.exists(path):
        return None, None, None
    d    = np.genfromtxt(path, delimiter=",", skip_header=1)
    th   = -d[:, 1]          # negate: raw CSV theta is +135→-135; Li conv -135→+135
    Fz   = d[:, 2] / 1000.  # mN → N
    Fx   = d[:, 3] / 1000.
    # Sort by ascending theta
    s    = np.argsort(th)
    return th[s], Fz[s], Fx[s]


def load_exp_cleg():
    """Return (theta_deg, Fz_N, Fx_N) from Li et al. Fig_S12_A.csv."""
    path = os.path.join(DIG_DIR, "Fig_S12_A.csv")
    raw  = np.genfromtxt(path, delimiter=",", skip_header=2, filling_values=np.nan)
    def pull(cx, cy):
        mask = ~(np.isnan(raw[:, cx]) | np.isnan(raw[:, cy]))
        xd   = np.degrees(raw[mask, cx])
        y    = raw[mask, cy]
        s    = np.argsort(xd)
        return xd[s], y[s]
    fx_th, fx_N = pull(0, 1)
    fz_th, fz_N = pull(4, 5)
    # Interpolate Fx onto Fz theta grid for ratio computation
    if len(fx_th) > 1:
        fx_interp = interp1d(fx_th, fx_N, bounds_error=False, fill_value=0.0)
        fx_on_fz  = fx_interp(fz_th)
    else:
        fx_on_fz = np.zeros_like(fz_N)
    return fz_th, fz_N, fx_on_fz


# ── Helper: smooth a curve ────────────────────────────────────────────────────
def smooth(y, win=7):
    if len(y) < win:
        return y
    return np.convolve(y, np.ones(win)/win, mode="same")


# ── Helper: interpolate curve onto common grid ────────────────────────────────
def to_grid(th, y, grid=THETA_GRID):
    """Interpolate (th, y) onto THETA_GRID. Clamp to 0 outside data range."""
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return np.zeros_like(grid)
    f = interp1d(th[mask], y[mask], kind="linear",
                 bounds_error=False, fill_value=0.0)
    return f(grid)


# ── Helper: find zero crossings ───────────────────────────────────────────────
def zero_crossings(th, y):
    """Return (left, right) theta where y crosses zero, bounding the main
    positive lobe. Returns (nan, nan) if no positive region found."""
    sm   = smooth(y, win=9)
    sign = np.sign(sm)
    pos  = np.where(sign > 0)[0]
    if len(pos) == 0:
        return np.nan, np.nan
    left_idx  = pos[0]
    right_idx = pos[-1]
    # Refine by linear interpolation
    def interp_crossing(i):
        if i == 0 or i >= len(th)-1:
            return th[i]
        y0, y1 = sm[i-1], sm[i]
        if abs(y1 - y0) < 1e-12:
            return th[i]
        return th[i-1] - y0 * (th[i] - th[i-1]) / (y1 - y0)
    left  = interp_crossing(left_idx)
    right = interp_crossing(right_idx + 1) if right_idx + 1 < len(th) else th[right_idx]
    return float(left), float(right)


# ── Helper: compute all indicators from one cleg run ─────────────────────────
def compute_indicators(th, Fz, Fx, scale=1.0):
    """
    scale : divide forces by this to convert to omega=0.2 equivalent.
            Pass 1.0 for simulation data (keeps omega=5 units for GP training);
            pass the empirical factor for Li et al. data.
    """
    Fz_s  = smooth(Fz) / scale
    Fx_s  = smooth(Fx) / scale

    # Interpolate Fx onto Fz theta grid if grids differ
    if len(th) != len(Fz_s):
        Fz_s = to_grid(th, Fz_s)

    fz_peak     = float(np.max(Fz_s))
    fz_peak_ang = float(th[np.argmax(Fz_s)])
    fx_peak     = float(np.max(np.abs(Fx_s)))

    # Ratio: thrust/lift at respective peaks (omega-independent)
    fx_over_fz  = fx_peak / fz_peak if fz_peak > 1e-9 else np.nan

    # Zero crossings bounding positive Fz lobe
    zl, zr      = zero_crossings(th, Fz_s)
    fz_width    = (zr - zl) if (not np.isnan(zl) and not np.isnan(zr)) else np.nan

    # Normalised curve for PCA (remove amplitude; shape only)
    fz_norm     = to_grid(th, Fz_s / fz_peak if fz_peak > 1e-9 else Fz_s)

    return {
        "cleg_Fz_peak_N":        fz_peak,          # amplitude (kept for GP)
        "cleg_Fx_over_Fz":       fx_over_fz,        # ratio — omega-independent
        "cleg_Fz_zero_left_deg": zl,                # shape — omega-independent
        "cleg_Fz_zero_right_deg":zr,                # shape — omega-independent
        "cleg_Fz_width_deg":     fz_width,          # shape — omega-independent
    }, fz_norm   # return normalised curve separately for PCA


# ── Load LHS design ───────────────────────────────────────────────────────────
design = pd.read_csv(os.path.join(LHS_DIR, "lhs_design.csv"))
n_samples = len(design)
print(f"Loading {n_samples} LHS samples ...")

rows      = []
fz_norms  = []    # normalised curves for PCA fitting

for _, row in design.iterrows():
    idx  = int(row["sample_id"])
    sd   = os.path.join(LHS_DIR, f"sample_{idx:03d}")
    th, Fz, Fx = load_sim_cleg(sd)
    if th is None:
        print(f"  [{idx:03d}] missing — skipped")
        continue
    inds, fz_norm = compute_indicators(th, Fz, Fx, scale=1.0)
    rec = {"sample_id": idx,
           "phi_deg":   float(row["phi_deg"]),
           "c_Pa":      float(row["c_Pa"])}
    rec.update(inds)
    rows.append(rec)
    fz_norms.append(fz_norm)
    print(f"  [{idx:03d}] phi={row['phi_deg']:.1f}° c={row['c_Pa']:.0f} Pa  "
          f"Fz_peak={inds['cleg_Fz_peak_N']:.2f} N  "
          f"Fx/Fz={inds['cleg_Fx_over_Fz']:.3f}  "
          f"width={inds['cleg_Fz_width_deg']:.1f}°")

# ── Fit PCA on normalised simulation curves ───────────────────────────────────
print(f"\nFitting PCA (n_components={N_PCA}) on {len(fz_norms)} normalised cleg curves ...")
X_norm = np.array(fz_norms)
pca    = PCA(n_components=N_PCA)
scores = pca.fit_transform(X_norm)
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Cumulative: {pca.explained_variance_ratio_.sum():.3f}")

for i, rec in enumerate(rows):
    rec["cleg_pca1"] = float(scores[i, 0])
    rec["cleg_pca2"] = float(scores[i, 1])

np.savez(os.path.join(OUT_DIR, "pca_basis.npz"),
         components=pca.components_,
         mean=pca.mean_,
         theta_grid=THETA_GRID,
         explained_variance_ratio=pca.explained_variance_ratio_)

# ── Save LHS indicators CSV ───────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "lhs_cleg_indicators.csv")
df.to_csv(csv_path, index=False)
print(f"\nLHS indicators saved: {csv_path}")
print(f"Columns: {list(df.columns)}\n")

# ── Compute Li et al. indicators ──────────────────────────────────────────────
print("Extracting Li et al. cleg indicators ...")
fz_th_e, fz_N_e, fx_N_e = load_exp_cleg()

# Empirical scale factor for cleg (from li_obs/li_obs.json if available)
liobs_path = os.path.join(HERE, "li_obs", "li_obs.json")
if os.path.exists(liobs_path):
    with open(liobs_path) as f:
        liobs = json.load(f)
    cleg_scale = liobs.get("_scale_factors", {}).get("cleg", 11.25)
else:
    cleg_scale = df["cleg_Fz_peak_N"].mean() / float(np.max(smooth(fz_N_e)))
print(f"  cleg scale factor (sim/exp): {cleg_scale:.2f}x")

# Compute indicators — scale experimental forces UP to omega=5 equivalent
# so they live in the same space as the GP training data
li_inds, li_fz_norm = compute_indicators(fz_th_e, fz_N_e, fx_N_e, scale=1.0/cleg_scale)

# Project Li et al. normalised curve onto PCA basis
li_score = pca.transform(li_fz_norm.reshape(1, -1))
li_inds["cleg_pca1"] = float(li_score[0, 0])
li_inds["cleg_pca2"] = float(li_score[0, 1])
li_inds["_source"]   = "Li et al. (2013) Fig. S12A — cleg"
li_inds["_cleg_scale_factor"] = cleg_scale
li_inds["_note"] = (
    "Force magnitude (cleg_Fz_peak_N) scaled by empirical ratio to match omega=5 "
    "simulation units. All other indicators are omega-independent."
)

print(f"  cleg_Fz_peak_N (scaled) : {li_inds['cleg_Fz_peak_N']:.2f} N")
print(f"  cleg_Fx_over_Fz         : {li_inds['cleg_Fx_over_Fz']:.3f}")
print(f"  cleg_Fz_zero_left_deg   : {li_inds['cleg_Fz_zero_left_deg']:.1f}°")
print(f"  cleg_Fz_zero_right_deg  : {li_inds['cleg_Fz_zero_right_deg']:.1f}°")
print(f"  cleg_Fz_width_deg       : {li_inds['cleg_Fz_width_deg']:.1f}°")
print(f"  cleg_pca1               : {li_inds['cleg_pca1']:.4f}")
print(f"  cleg_pca2               : {li_inds['cleg_pca2']:.4f}")

obs_path = os.path.join(OUT_DIR, "li_cleg_obs.json")
with open(obs_path, "w") as f:
    json.dump(li_inds, f, indent=2)
print(f"\nLi et al. observations: {obs_path}")

print(f"\nNext step:")
print(f"  python3 inverse_gp.py \\")
print(f"      --data {csv_path} \\")
print(f"      --obs  {obs_path} \\")
print(f"      --out  {OUT_DIR}")

# ── Optional: extract indicators from a synthetic obs simulation dir ──────────
if args.syn_obs_dir:
    syn_dir = os.path.join(HERE, args.syn_obs_dir)
    print(f"\n── Synthetic obs extraction from: {syn_dir}")

    # Load saved PCA basis (fit on LHS above or from npz if re-running)
    pca_npz = os.path.join(OUT_DIR, "pca_basis.npz")
    if not os.path.exists(pca_npz):
        print("  ERROR: pca_basis.npz not found — run without --syn_obs_dir first.")
    else:
        th_s, Fz_s, Fx_s = load_sim_cleg(syn_dir)
        if th_s is None:
            print(f"  ERROR: terradyn_cleg.csv not found in {syn_dir}")
        else:
            syn_inds, syn_norm = compute_indicators(th_s, Fz_s, Fx_s, scale=1.0)
            syn_score = pca.transform(syn_norm.reshape(1, -1))   # use in-memory PCA
            syn_inds["cleg_pca1"] = float(syn_score[0, 0])
            syn_inds["cleg_pca2"] = float(syn_score[0, 1])

            # Read ground truth from summary.json if present
            summary_path = os.path.join(syn_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    sm = json.load(f)
                syn_inds["_ground_truth_phi_deg"] = sm.get("phi_deg",
                    sm.get("_ground_truth_phi_deg", None))
                syn_inds["_ground_truth_c_Pa"]    = sm.get("c_Pa",
                    sm.get("_ground_truth_c_Pa",   None))

            syn_inds["_source"] = f"Synthetic simulation from {args.syn_obs_dir}"

            out_path = os.path.join(HERE, args.syn_obs_out)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(syn_inds, f, indent=2)

            print(f"  cleg_Fz_peak_N  : {syn_inds['cleg_Fz_peak_N']:.2f} N")
            print(f"  cleg_Fx_over_Fz : {syn_inds['cleg_Fx_over_Fz']:.3f}")
            print(f"  cleg_pca1       : {syn_inds['cleg_pca1']:.4f}")
            print(f"  Ground truth    : phi={syn_inds.get('_ground_truth_phi_deg')}°  "
                  f"c={syn_inds.get('_ground_truth_c_Pa')} Pa")
            print(f"  Saved to: {out_path}")
