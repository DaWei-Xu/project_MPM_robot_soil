"""
inverse_gp.py
=============
GP surrogate + MCMC Bayesian inference for soil parameters (phi, c)
from MPM terradynamics summary statistics.

Workflow:
  1. Load LHS simulation data (lhs_samples.csv from run_lhs_samples.py)
  2. Fit one GP per observable (standardised inputs/outputs)
  3. Run MCMC (emcee) on the GP likelihood to get P(phi, c | obs_data)
  4. Plot posterior corner plot + GP surrogate surfaces

Usage:
    python3 inverse_gp.py --data lhs_results/lhs_samples.csv
    python3 inverse_gp.py --data lhs_results/lhs_samples.csv --obs obs_data.json
    python3 inverse_gp.py --data lhs_results/lhs_samples.csv --walkers 64 --steps 3000

Observed data format (obs_data.json) — same keys as summary_stats.json:
    {
      "flat_Fz_peak_N": 0.18,
      "flat_Fx_peak_N": 0.09,
      "cleg_Fz_peak_N": 0.22,
      ...
    }
If --obs is not provided, a synthetic "true" observation is generated from
the nearest sample to phi=35°, c=500 Pa for testing purposes.

Dependencies:
    pip install scikit-learn emcee corner pandas matplotlib scipy
"""

import argparse
import json
import os
import warnings

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",    required=True, help="Path to lhs_samples.csv")
parser.add_argument("--obs",     default="",    help="Path to observed data JSON")
parser.add_argument("--walkers", type=int, default=32,   help="MCMC walkers")
parser.add_argument("--steps",   type=int, default=2000, help="MCMC steps")
parser.add_argument("--obs_cols", default="", help="Comma-separated list of observable columns to use (default: all)")
parser.add_argument("--burnin",  type=int, default=500,  help="MCMC burn-in steps")
parser.add_argument("--out",     default="",    help="Output directory (default: same as --data)")
parser.add_argument("--phi_prior_mu",    type=float, default=None,
                    help="Mean of Gaussian prior on phi (deg). Default: flat uniform prior.")
parser.add_argument("--phi_prior_sigma", type=float, default=4.0,
                    help="Std-dev of Gaussian prior on phi (deg). Used only if --phi_prior_mu is set.")
parser.add_argument("--c_prior_mu",      type=float, default=None,
                    help="Mean of Gaussian prior on c (Pa). Default: flat uniform prior.")
parser.add_argument("--c_prior_sigma",   type=float, default=2000.0,
                    help="Std-dev of Gaussian prior on c (Pa). Used only if --c_prior_mu is set.")
args = parser.parse_args()

DATA_PATH = os.path.abspath(args.data)
OUT_DIR   = args.out if args.out else os.path.dirname(DATA_PATH)
os.makedirs(OUT_DIR, exist_ok=True)

# Parameter bounds (must match run_lhs_samples.py)
PHI_MIN, PHI_MAX = 20.0, 42.0
C_MIN,   C_MAX   = 0.0,  15000.0

# ── Load simulation data ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples from {DATA_PATH}")
print(f"Columns: {list(df.columns)}\n")

# Input parameters
X_raw = df[["phi_deg", "c_Pa"]].values.astype(float)

# Observable columns — all summary stat columns (exclude metadata)
META_COLS = {"sample_id", "phi_deg", "c_Pa", "phi_deg_x", "c_Pa_x",
             "E_Pa", "phi_deg_y", "c_Pa_y"}
OBS_COLS  = [c for c in df.columns
             if c not in META_COLS and not c.startswith("Unnamed")]
if args.obs_cols:
    keep = [c.strip() for c in args.obs_cols.split(",")]
    missing = [c for c in keep if c not in OBS_COLS]
    if missing:
        raise ValueError(f"--obs_cols requested columns not in data: {missing}")
    OBS_COLS = keep
    print(f"Using user-specified observables ({len(OBS_COLS)}): {OBS_COLS}\n")
else:
    print(f"Observables ({len(OBS_COLS)}): {OBS_COLS}\n")

Y_raw = df[OBS_COLS].values.astype(float)

# Drop rows with NaN in any observable
mask = ~np.any(np.isnan(Y_raw), axis=1)
X_raw = X_raw[mask];  Y_raw = Y_raw[mask]
print(f"Clean samples after NaN removal: {np.sum(mask)}/{len(mask)}\n")

if len(X_raw) < 5:
    raise ValueError("Too few clean samples — check that simulations completed successfully.")

# ── Standardise inputs and outputs ───────────────────────────────────────────
x_scaler = StandardScaler().fit(X_raw)
X = x_scaler.transform(X_raw)

y_scalers = []
Y = np.zeros_like(Y_raw)
for k in range(Y_raw.shape[1]):
    sc = StandardScaler().fit(Y_raw[:, k:k+1])
    Y[:, k] = sc.transform(Y_raw[:, k:k+1]).ravel()
    y_scalers.append(sc)

# ── Fit one GP per observable ─────────────────────────────────────────────────
print("Fitting GP surrogates ...")
gps = []
for k, col in enumerate(OBS_COLS):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   normalize_y=False, alpha=1e-6)
    gp.fit(X, Y[:, k])
    gps.append(gp)
    print(f"  [{col}]  log-marginal-likelihood = {gp.log_marginal_likelihood_value_:.2f}")

print()

# ── Observed data ─────────────────────────────────────────────────────────────
true_phi = None;  true_c = None   # set if ground truth is known

if args.obs:
    with open(args.obs) as f:
        obs_dict = json.load(f)
    # Ground truth embedded by generate_synthetic_obs.py (keys prefixed with _)
    true_phi = obs_dict.get("_ground_truth_phi_deg", None)
    true_c   = obs_dict.get("_ground_truth_c_Pa",   None)
    # Only keep observable keys (no metadata)
    obs_raw = np.array([obs_dict[c] for c in OBS_COLS])
    if true_phi is not None:
        print(f"Synthetic observation loaded from {args.obs}")
        print(f"  Ground truth: phi={true_phi:.2f}°, c={true_c:.1f} Pa\n")
    else:
        print(f"Observed data loaded from {args.obs}\n")
else:
    raise ValueError(
        "No --obs file provided.\n"
        "Generate a synthetic observation first:\n"
        "  python3 generate_synthetic_obs.py --phi 32 --c 3000\n"
        "Then re-run with:\n"
        "  python3 inverse_gp.py --data lhs_results/lhs_samples.csv "
        "--obs synthetic_obs/summary.json"
    )

# Standardise obs with same scalers
obs_std = np.array([y_scalers[k].transform([[obs_raw[k]]])[0, 0]
                    for k in range(len(OBS_COLS))])

# ── MCMC log-posterior ────────────────────────────────────────────────────────
# Measurement noise: assume 5% of observed value (relative), min 1e-4 in standardised units
OBS_NOISE_REL = 0.05

PHI_PRIOR_MU  = args.phi_prior_mu
PHI_PRIOR_SIG = args.phi_prior_sigma
C_PRIOR_MU    = args.c_prior_mu
C_PRIOR_SIG   = args.c_prior_sigma
if PHI_PRIOR_MU is not None:
    print(f"Informed prior on phi: N(mu={PHI_PRIOR_MU:.1f}°, sigma={PHI_PRIOR_SIG:.1f}°) "
          f"truncated to [{PHI_MIN}, {PHI_MAX}]")
if C_PRIOR_MU is not None:
    print(f"Informed prior on c  : N(mu={C_PRIOR_MU:.0f} Pa, sigma={C_PRIOR_SIG:.0f} Pa) "
          f"truncated to [{C_MIN}, {C_MAX}]")
if PHI_PRIOR_MU is not None or C_PRIOR_MU is not None:
    print()

def log_prior(theta):
    phi, c = theta
    if not (PHI_MIN < phi < PHI_MAX and C_MIN < c < C_MAX):
        return -np.inf
    lp = 0.0
    if PHI_PRIOR_MU is not None:
        lp += -0.5 * ((phi - PHI_PRIOR_MU) / PHI_PRIOR_SIG) ** 2
    if C_PRIOR_MU is not None:
        lp += -0.5 * ((c - C_PRIOR_MU) / C_PRIOR_SIG) ** 2
    return lp


def log_likelihood(theta):
    phi, c = theta
    x_in = x_scaler.transform([[phi, c]])
    ll   = 0.0
    for k, gp in enumerate(gps):
        mu, sigma = gp.predict(x_in, return_std=True)
        mu    = float(mu[0]);  sigma = float(sigma[0])
        noise = max(OBS_NOISE_REL * abs(obs_std[k]), 1e-4)
        total_var = sigma**2 + noise**2
        ll += -0.5 * (obs_std[k] - mu)**2 / total_var - 0.5 * np.log(2*np.pi*total_var)
    return ll


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# ── Run MCMC ──────────────────────────────────────────────────────────────────
print(f"Running MCMC: {args.walkers} walkers × {args.steps} steps "
      f"(burn-in {args.burnin}) ...")

ndim = 2
# Initialise walkers near the MAP-ish centre of the prior
rng0 = np.random.default_rng(0)
p0   = rng0.uniform([PHI_MIN, C_MIN], [PHI_MAX, C_MAX],
                     size=(args.walkers, ndim))

sampler = emcee.EnsembleSampler(args.walkers, ndim, log_posterior)
sampler.run_mcmc(p0, args.steps, progress=True)

# Discard burn-in, flatten chains
flat_chain = sampler.get_chain(discard=args.burnin, thin=5, flat=True)
print(f"\nPosterior samples: {len(flat_chain)}")

phi_post = flat_chain[:, 0]
c_post   = flat_chain[:, 1]

phi_map = float(np.median(phi_post));  phi_std = float(np.std(phi_post))
c_map   = float(np.median(c_post));    c_std   = float(np.std(c_post))

print(f"\n{'─'*50}")
print(f"Posterior median ± std:")
print(f"  phi = {phi_map:.2f} ± {phi_std:.2f} deg")
print(f"  c   = {c_map:.1f} ± {c_std:.1f} Pa  ({c_map/1000:.2f} ± {c_std/1000:.2f} kPa)")
print(f"{'─'*50}\n")

# ── Corner plot ───────────────────────────────────────────────────────────────
fig_corner = corner.corner(
    flat_chain,
    labels=[r"$\phi$ (deg)", r"$c$ (Pa)"],
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 11},
    truths=[true_phi, true_c] if true_phi is not None else None,
    truth_color="#E74C3C",
)
fig_corner.suptitle("Posterior: friction angle φ and cohesion c\n"
                     "(GP surrogate + MCMC)", y=1.02, fontsize=12)
corner_path = os.path.join(OUT_DIR, "posterior_corner.png")
fig_corner.savefig(corner_path, dpi=160, bbox_inches="tight")
plt.close(fig_corner)
print(f"Corner plot: {corner_path}")

# ── GP surrogate surface plot ─────────────────────────────────────────────────
# Show the first two most-informative observables on a 2D grid
phi_grid = np.linspace(PHI_MIN, PHI_MAX, 60)
c_grid   = np.linspace(C_MIN,   C_MAX,   60)
PP, CC   = np.meshgrid(phi_grid, c_grid)
X_grid   = x_scaler.transform(
    np.column_stack([PP.ravel(), CC.ravel()])
)

n_show = min(4, len(OBS_COLS))
fig_surf, axes = plt.subplots(1, n_show, figsize=(4.5*n_show, 4))
if n_show == 1:
    axes = [axes]

for ax, k in zip(axes, range(n_show)):
    mu_std = gps[k].predict(X_grid).reshape(60, 60)
    # Back-transform to physical units
    mu_phys = y_scalers[k].inverse_transform(mu_std.ravel().reshape(-1,1)).reshape(60,60)
    cf = ax.contourf(PP, CC/1000, mu_phys, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="N")
    ax.scatter(X_raw[:, 0], X_raw[:, 1]/1000,
               c="white", s=12, edgecolors="k", lw=0.4, zorder=5, label="LHS samples")
    # Posterior 90% region
    ax.scatter(np.percentile(phi_post, [5, 95]),
               [np.percentile(c_post, 50)/1000]*2,
               marker="|", color="red", s=80, zorder=6)
    ax.scatter([np.percentile(phi_post, 50)],
               [np.percentile(c_post, 5)/1000],
               marker="_", color="red", s=80, zorder=6)
    ax.scatter([phi_map], [c_map/1000], marker="*", color="red",
               s=200, zorder=7, label="Posterior median")
    if true_phi is not None:
        ax.scatter([true_phi], [true_c/1000], marker="x", color="cyan",
                   s=120, zorder=8, lw=2, label="True value")
    ax.set_xlabel("φ (deg)"); ax.set_ylabel("c (kPa)")
    ax.set_title(OBS_COLS[k], fontsize=9)
    ax.legend(fontsize=7)

plt.suptitle("GP Surrogate Surfaces with Posterior Overlay", fontsize=11)
plt.tight_layout()
surf_path = os.path.join(OUT_DIR, "gp_surrogate_surfaces.png")
plt.savefig(surf_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"Surrogate surfaces: {surf_path}")

# ── Save posterior summary ────────────────────────────────────────────────────
posterior_summary = {
    "phi_median_deg": phi_map,
    "phi_std_deg":    phi_std,
    "phi_p5_deg":     float(np.percentile(phi_post, 5)),
    "phi_p95_deg":    float(np.percentile(phi_post, 95)),
    "c_median_Pa":    c_map,
    "c_std_Pa":       c_std,
    "c_p5_Pa":        float(np.percentile(c_post, 5)),
    "c_p95_Pa":       float(np.percentile(c_post, 95)),
    "n_posterior_samples": len(flat_chain),
    "observables_used": OBS_COLS,
    "phi_prior": (f"N(mu={PHI_PRIOR_MU:.1f}, sigma={PHI_PRIOR_SIG:.1f}) truncated [{PHI_MIN},{PHI_MAX}]"
                  if PHI_PRIOR_MU is not None else f"Uniform({PHI_MIN},{PHI_MAX})"),
    "c_prior":   (f"N(mu={C_PRIOR_MU:.0f}, sigma={C_PRIOR_SIG:.0f}) truncated [{C_MIN},{C_MAX}]"
                  if C_PRIOR_MU is not None else f"Uniform({C_MIN},{C_MAX})"),
}
if true_phi is not None:
    posterior_summary["synthetic_true_phi_deg"] = float(true_phi)
    posterior_summary["synthetic_true_c_Pa"]    = float(true_c)

ps_path = os.path.join(OUT_DIR, "posterior_summary.json")
with open(ps_path, "w") as f:
    json.dump(posterior_summary, f, indent=2)
print(f"Posterior summary: {ps_path}")
print("\n✓ Inverse analysis complete.")
