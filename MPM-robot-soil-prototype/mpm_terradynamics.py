"""
mpm_terradynamics.py
====================
MPM simulation of Li et al. (2013) terradynamics experiment:
  rotating rigid leg through Yuma Sand (Fig. S12 data).
  Three leg types: flat, c-leg (kappa = +1/R), reversed c-leg (kappa = -1/R).
  Records lift Fz and thrust Fx vs. leg angle theta.
  Compares against digitized Li et al. experimental data (Fig. S12 A/B/C).

  Rotation convention: CCW (counter-clockwise), matching Li et al. 2013.
    theta sweeps from +3pi/4 → -3pi/4 (leg enters from right, exits left).

Physical parameters (Yuma Sand):
  Leg: max length 2R = 7.62 cm, R = 3.81 cm; width w = 2.54 cm; half-thickness t = 0.32 cm
  Hip height: h = 2 cm above granular surface
  Angular velocity: omega = 0.2 rad/s (experiment); default 5.0 for CPU speed
  Granular medium (Yuma Sand, dry):
    bulk density 1650 kg/m3, friction angle 35 deg
    E = 20 MPa (low-confining-pressure estimate for loose Yuma sand)

Domain: 30 cm x 30 cm square (24 cm container + 3 cm margins)
Grid  : 128 x 128, DX = 2.344 mm
NP    : ~26,100 particles

CFL check: c_s = sqrt(E/rho) = sqrt(20e6/1650) = 110 m/s
           DT_cfl = DX/(2*c_s) = 2.344e-3/220 = 1.07e-5 s
           DT = 8e-6 s < DT_cfl  OK

Usage:
    python3 mpm_terradynamics.py                     # all three legs, ω=0.2 rad/s (experiment)
    python3 mpm_terradynamics.py --leg flat          # single leg (~2.9M steps, ~hours on CPU)
    python3 mpm_terradynamics.py --leg flat --omega 5.0  # fast test run (25x fewer steps)
    python3 mpm_terradynamics.py --paraview          # export ParaView VTP time series

References:
  Li, Zhang & Goldman (2013) Science 339:1408
  Li et al. supplementary material — Eq. S8, Table S2 (RFT Fourier fit)
"""

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse, os, json
from datetime import datetime

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--leg",    default="all", choices=["flat", "cleg", "rcleg", "all"])
parser.add_argument("--omega",  type=float, default=0.2,
                    help="Angular velocity rad/s (Li et al. experiment = 0.2)")
parser.add_argument("--damp",   type=float, default=0.997,
                    help="Grid velocity damping per step (0.997 → τ≈2.7ms; 0.999 → τ≈8ms)")
parser.add_argument("--settle",      type=int,   default=20000)
parser.add_argument("--out",         default="results")
parser.add_argument("--frames",      action="store_true",
                    help="Save matplotlib PNG frames for ffmpeg animation")
parser.add_argument("--paraview",    action="store_true",
                    help="Export particle CSV files for ParaView TableToPoints")
parser.add_argument("--frame_every", type=int, default=0,
                    help="Frame/export interval in steps (0 = auto ~100 frames)")
parser.add_argument("--phi",    type=float, default=35.0,
                    help="Friction angle in degrees (default: 35 for Yuma Sand)")
parser.add_argument("--c",      type=float, default=10.0,
                    help="Cohesion in Pa (default: 10 Pa for near-cohesionless Yuma Sand)")
parser.add_argument("--summary_out", default="",
                    help="If set, write summary statistics JSON to this path")
parser.add_argument("--threads", type=int, default=None,
                    help="Max CPU threads for Taichi (default: all cores). "
                         "Set to 2 when running many parallel workers.")
args = parser.parse_args()

_ti_kwargs = dict(arch=ti.cpu, default_fp=ti.f64)
if args.threads is not None:
    _ti_kwargs["cpu_max_num_threads"] = args.threads
ti.init(**_ti_kwargs)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN  = 0.30          # m — 30 cm square
GRID_N  = 128
DX      = DOMAIN / GRID_N       # 2.344 mm
INV_DX  = GRID_N / DOMAIN

# Yuma Sand (Li et al. 2013 Fig. S12)
RHO     = 1650.0        # kg/m³  (dry Yuma sand, medium-loose)
E_MOD   = 20e6          # Pa  (20 MPa — fixed; identifiable from penetration, not rotation)
NU      = 0.30
PHI_RAD = np.radians(args.phi)   # friction angle — CLI-settable for inverse analysis
C_COH   = args.c                  # cohesion (Pa) — CLI-settable for inverse analysis
K0      = 1.0 - np.sin(PHI_RAD)

MU_E    = E_MOD / (2.0 * (1.0 + NU))
LAM_E   = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
ALPHA_DP = 2.0 * np.sin(PHI_RAD) / (np.sqrt(3.0) * (3.0 - np.sin(PHI_RAD)))
K_C_DP   = 6.0 * C_COH * np.cos(PHI_RAD) / (np.sqrt(3.0) * (3.0 - np.sin(PHI_RAD)))

MU_LEG  = 0.35          # Coulomb friction coefficient: leg–sand interface

G_GRAV      = 9.81
DT          = 8e-6      # s   (c_s=110 m/s → DT_cfl=1.07e-5 s)
DAMP        = args.damp  # grid velocity damping per step (set via --damp; default 0.997 → τ≈2.7ms)
                         # D-P plasticity dominates energy dissipation; damping is numerical only

# Granular bed layout
# Margins = 2 cells × DX = 2 × 2.344 mm ≈ 5 mm — numerical minimum for quadratic B-spline kernel
# Sand fills nearly the full domain width and sits against the bottom wall,
# matching the confined container geometry of Li et al. (2013).
BED_X0  = 0.005;  BED_X1 = 0.295   # 29 cm wide, 5 mm margins
BED_Y0  = 0.005;  BED_Y1 = 0.17    # 16.5 cm deep, 5 mm bottom margin
SURF_Y  = BED_Y1

# Leg geometry (Li et al. supplement)
LEG_R   = 0.0381        # m   arc radius; arc length = 2*LEG_R per leg
LEG_T   = 0.0032        # m   half-thickness (0.64 cm / 2)
LEG_W   = 0.0254        # m   leg width — 2D→3D force conversion
HIP_X   = DOMAIN / 2.0
HIP_H   = 0.020         # m   hip height above granular surface

# Arc geometry constants — curved legs are semicircles (pi radians = 180 deg arc)
# Chord = 2R = 7.62 cm for all three leg types (same as flat leg length).
# C-leg arc centre at (u=0, v=+R), spans alpha in [-pi/2, +pi/2]:
#   tip body: (R*cos(pi/2), R+R*sin(pi/2)) = (0, 2R)
# Rcleg arc centre at (u=0, v=-R), spans alpha in [-pi/2, +pi/2]:
#   tip body: (R*cos(-pi/2), -R+R*sin(-pi/2)) = (0, -2R)
LEG_U_TIP = 0.0                # tip u-coordinate in body frame (semicircle: u=0)
LEG_V_TIP = float(2.0 * LEG_R) # tip v-magnitude: 2R = 7.62 cm  (c-leg +, rcleg -)

# Rotation sweep
OMEGA   = args.omega
TH_ST   = -3.0 * np.pi / 4.0
TH_EN   =  3.0 * np.pi / 4.0
N_ROT   = int((TH_EN - TH_ST) / (abs(OMEGA) * DT)) + 1
SETTLE  = args.settle
SAVE_EVERY  = max(1, N_ROT // 300)
FRAME_EVERY = args.frame_every if args.frame_every > 0 else max(1, N_ROT // 100)

# Particle grid
PPG     = 4
nx_bed  = int((BED_X1 - BED_X0) * INV_DX + 0.5)
ny_bed  = int((BED_Y1 - BED_Y0) * INV_DX + 0.5)
NP      = nx_bed * ny_bed * PPG

print(f"Domain : {DOMAIN*100:.0f} cm  |  DX = {DX*1000:.2f} mm  |  Grid {GRID_N}^2")
print(f"Soil   : rho={RHO:.0f} kg/m3, E={E_MOD:.0e} Pa, phi={np.degrees(PHI_RAD):.0f} deg")
print(f"NP = {NP}   Settle = {SETTLE}   N_ROT = {N_ROT}   omega = {OMEGA:.2f} rad/s")

# ═══════════════════════════════════════════════════════════════════════════════
# TAICHI FIELDS
# ═══════════════════════════════════════════════════════════════════════════════

xp    = ti.Vector.field(2, ti.f64, NP)
vp    = ti.Vector.field(2, ti.f64, NP)
Cp    = ti.Matrix.field(2, 2, ti.f64, NP)
sig   = ti.Matrix.field(2, 2, ti.f64, NP)
mp    = ti.field(ti.f64, NP)
vol0  = ti.field(ti.f64, NP)

gv    = ti.Vector.field(2, ti.f64, (GRID_N, GRID_N))
gm    = ti.field(ti.f64, (GRID_N, GRID_N))
grf_x = ti.field(ti.f64, ())
grf_y = ti.field(ti.f64, ())
damp  = ti.field(ti.f64, ())   # uniform grid damping coefficient
damp[None] = DAMP

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_particles():
    sub   = int(np.sqrt(PPG))
    cvol  = (DX / sub) ** 2
    pmass = RHO * cvol
    # Jittered sampling: place each particle at a uniformly random location
    # within its sub-cell rather than at the sub-cell center.  This breaks
    # the regular grid pattern and reduces MPM quadrature errors from
    # particle-grid alignment.  Each sub-cell still contains exactly one
    # particle so bulk density is preserved.
    rng   = np.random.default_rng(seed=42)
    sdx   = (BED_X1 - BED_X0) / (nx_bed * sub)
    sdy   = (BED_Y1 - BED_Y0) / (ny_bed * sub)
    xs = np.zeros((NP, 2))
    idx = 0
    for i in range(nx_bed * sub):
        for j in range(ny_bed * sub):
            if idx >= NP:
                break
            xs[idx] = [
                BED_X0 + (i + rng.uniform(0.05, 0.95)) * sdx,
                BED_Y0 + (j + rng.uniform(0.05, 0.95)) * sdy,
            ]
            idx += 1
    xp.from_numpy(xs)
    vp.from_numpy(np.zeros((NP, 2)))
    Cp.from_numpy(np.zeros((NP, 2, 2)))
    mp.from_numpy(np.full(NP, pmass))
    vol0.from_numpy(np.full(NP, cvol))
    sv = -RHO * G_GRAV * np.maximum(BED_Y1 - xs[:, 1], 0.0)
    sig_np = np.zeros((NP, 2, 2))
    sig_np[:, 0, 0] = K0 * sv
    sig_np[:, 1, 1] = sv
    sig.from_numpy(sig_np)


def save_state():
    return (xp.to_numpy().copy(), vp.to_numpy().copy(),
            Cp.to_numpy().copy(), sig.to_numpy().copy())


def restore_state(s):
    xp.from_numpy(s[0]); vp.from_numpy(s[1])
    Cp.from_numpy(s[2]); sig.from_numpy(s[3])


# ═══════════════════════════════════════════════════════════════════════════════
# MPM KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def clear_grid():
    for i, j in gm:
        gv[i, j] = ti.Vector([0.0, 0.0])
        gm[i, j] = 0.0


@ti.kernel
def p2g():
    """APIC P2G with incremental hypoelastic Drucker-Prager stress update."""
    for p in range(NP):
        Xp   = xp[p] * INV_DX
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - ti.cast(base, ti.f64)
        w    = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]

        deps = DT * 0.5 * (Cp[p] + Cp[p].transpose())
        I2   = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        tr_d = deps[0, 0] + deps[1, 1]
        sig_tr = sig[p] + LAM_E * tr_d * I2 + 2.0 * MU_E * deps

        # Drucker-Prager return mapping (tensile-positive convention)
        p_s   = (sig_tr[0, 0] + sig_tr[1, 1]) * 0.5
        s_dev = sig_tr - p_s * I2
        q_s   = ti.sqrt(0.5 * (s_dev[0,0]**2 + s_dev[1,1]**2 + 2.0*s_dev[0,1]**2))
        f_yld = q_s + ALPHA_DP * p_s - K_C_DP
        s_ret = sig_tr
        if f_yld > 0.0:
            H_dp  = MU_E + ALPHA_DP**2 * (LAM_E + MU_E)
            dg    = f_yld / ti.max(H_dp, 1e-14)
            s_new = s_dev
            if q_s > 1e-12:
                s_new = s_dev * (1.0 - dg * MU_E / q_s)
            p_new = p_s - ALPHA_DP * (LAM_E + MU_E) * dg
            s_ret = s_new + p_new * I2
            p_chk = (s_ret[0, 0] + s_ret[1, 1]) * 0.5
            if p_chk > 0.0:
                s_ret = s_ret - p_chk * I2
        sig[p] = s_ret

        sc = (-DT * vol0[p] * 4.0 * INV_DX * INV_DX) * sig[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            off  = ti.Vector([i, j])
            dpos = (ti.cast(off, ti.f64) - fx) * DX
            wij  = w[i][0] * w[j][1]
            node = base + off
            if 0 <= node[0] < GRID_N and 0 <= node[1] < GRID_N:
                gv[node] += wij * (mp[p]*vp[p] + sc @ dpos)
                gm[node] += wij * mp[p]


@ti.kernel
def grid_settle():
    for i, j in gm:
        if gm[i, j] > 1e-14:
            gv[i, j] /= gm[i, j]
            gv[i, j][1] -= DT * G_GRAV
            gv[i, j] *= damp[None]
            if i < 3:          gv[i, j][0] = ti.max(gv[i, j][0], 0.0)
            if i > GRID_N - 4: gv[i, j][0] = ti.min(gv[i, j][0], 0.0)
            if j < 3:          gv[i, j][1] = ti.max(gv[i, j][1], 0.0)
            if j > GRID_N - 4: gv[i, j][1] = ti.min(gv[i, j][1], 0.0)


@ti.kernel
def g2p():
    for p in range(NP):
        Xp   = xp[p] * INV_DX
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - ti.cast(base, ti.f64)
        w    = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        nv   = ti.Vector([0.0, 0.0])
        nC   = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i, j in ti.static(ti.ndrange(3, 3)):
            off  = ti.Vector([i, j])
            dpos = (ti.cast(off, ti.f64) - fx) * DX
            wij  = w[i][0] * w[j][1]
            node = base + off
            if 0 <= node[0] < GRID_N and 0 <= node[1] < GRID_N:
                nv += wij * gv[node]
                nC += 4.0 * INV_DX * wij * gv[node].outer_product(dpos)
        vp[p] = nv
        Cp[p] = nC
        xp[p] += DT * vp[p]
        xp[p] = ti.Vector([
            ti.max(0.005, ti.min(DOMAIN - 0.005, xp[p][0])),
            ti.max(0.005, ti.min(DOMAIN - 0.005, xp[p][1])),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# SDF HELPER FUNCTIONS  (@ti.func — inlined into contact kernels)
#
# Body-frame coordinates for world point (nx_, ny_) with hip at (hx, hy) and
# leg at angle theta:
#   u = along-leg  = (nx_-hx)*sin(theta) - (ny_-hy)*cos(theta)   [0..2R hip-to-tip]
#   v = perp-leg   = (nx_-hx)*cos(theta) + (ny_-hy)*sin(theta)   [+ = forward]
#
# C-leg arc:  circle of radius LEG_R centred at (u=0, v=+LEG_R) in body frame.
#   alpha range: [-pi/2, +pi/2] rad (CCW, pi rad = semicircle)
#   Tip (body): (0, 2R)  — chord = 2R, same as flat leg
#   theta convention: chord angle from vertical; body frame offset = -pi/2
#
# Rev-c-leg:  circle centred at (u=0, v=-LEG_R).
#   alpha range: [-pi/2, +pi/2] rad (pi rad = semicircle)
#   Tip (body): (0, -2R)  — body frame offset = +pi/2
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def _flat_sd(nx_: ti.f64, ny_: ti.f64, theta: ti.f64, hx: ti.f64, hy: ti.f64) -> ti.f64:
    """SDF to flat rectangular leg. Box [0, 2R] x [-T, T] in body (u, v)."""
    ddx = nx_ - hx;  ddy = ny_ - hy
    u = ddx * ti.sin(theta) - ddy * ti.cos(theta)
    v = ddx * ti.cos(theta) + ddy * ti.sin(theta)
    return ti.max(ti.max(-u, u - 2.0*LEG_R), ti.abs(v) - LEG_T)


@ti.func
def _cleg_sd(nx_: ti.f64, ny_: ti.f64, theta: ti.f64, hx: ti.f64, hy: ti.f64) -> ti.f64:
    """SDF to c-leg arc tube (kappa = +1/R), semicircle (pi rad).
    Arc centre at (u=0, v=+LEG_R) in body frame; spans alpha in [-pi/2, +pi/2].
    theta is the chord angle; body frame u-axis is pi/2 behind the chord."""
    th = theta - 1.5707963   # chord-to-u-axis offset = -pi/2
    ddx = nx_ - hx;  ddy = ny_ - hy
    uu = ddx * ti.sin(th) - ddy * ti.cos(th)
    vv = ddx * ti.cos(th) + ddy * ti.sin(th)
    wu = uu;  wv = vv - LEG_R
    dc  = ti.sqrt(wu*wu + wv*wv)
    al  = ti.atan2(wv, wu)
    dcl = ti.abs(dc - LEG_R)
    # Past hip end (alpha < -pi/2): distance to hip point (0, 0)
    if al < -1.5707963:
        dcl = ti.sqrt(uu*uu + vv*vv)
    # Past tip end (alpha > +pi/2): distance to tip point (0, 2R)
    if al > 1.5707963:
        dcl = ti.sqrt((uu - LEG_U_TIP)**2 + (vv - LEG_V_TIP)**2)
    return dcl - LEG_T


@ti.func
def _rcleg_sd(nx_: ti.f64, ny_: ti.f64, theta: ti.f64, hx: ti.f64, hy: ti.f64) -> ti.f64:
    """SDF to reversed c-leg arc tube (kappa = -1/R), semicircle (pi rad).
    Arc centre at (u=0, v=-LEG_R) in body frame; spans alpha in [-pi/2, +pi/2].
    theta is the chord angle; body frame u-axis is pi/2 ahead of the chord."""
    th = theta + 1.5707963   # chord-to-u-axis offset = +pi/2
    ddx = nx_ - hx;  ddy = ny_ - hy
    uu = ddx * ti.sin(th) - ddy * ti.cos(th)
    vv = ddx * ti.cos(th) + ddy * ti.sin(th)
    wu = uu;  wv = vv + LEG_R
    dc  = ti.sqrt(wu*wu + wv*wv)
    al  = ti.atan2(wv, wu)
    dcl = ti.abs(dc - LEG_R)
    # Past hip end (alpha > pi/2): distance to hip point (0, 0)
    if al > 1.5707963:
        dcl = ti.sqrt(uu*uu + vv*vv)
    # Past tip end (alpha < -pi/2): distance to rev-c tip (0, -2R)
    if al < -1.5707963:
        dcl = ti.sqrt((uu - LEG_U_TIP)**2 + (vv + LEG_V_TIP)**2)
    return dcl - LEG_T


# ═══════════════════════════════════════════════════════════════════════════════
# CONTACT KERNELS — one per leg type
#
# Pattern (identical for all three; only the SDF function changes):
#   1. Normalise grid velocity, apply gravity + damping
#   2. Evaluate SDF.  If sd < 2*DX, node is in the contact band.
#   3. Finite-difference SDF gradient → outward unit normal nrm
#   4. Leg surface velocity at node: rotating rigid body about hip
#        vleg_x = -omega*(ny_ - hy),  vleg_y = +omega*(nx_ - hx)
#   5. Relative normal velocity vrel_n = (v_node - v_leg) . nrm
#   6. If vrel_n < 0 (node penetrating): correct velocity, accumulate GRF
#        GRF on leg = gm * (-vrel_n) * (-nrm) / DT  (Newton 3rd law)
#   7. Enforce wall boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def grid_contact_flat_leg(theta: ti.f64, omega: ti.f64, hip_y: ti.f64):
    hx = HIP_X;  hy = hip_y;  e = DX * 0.5
    for i, j in gm:
        if gm[i, j] > 1e-14:
            gv[i, j] /= gm[i, j]
            gv[i, j][1] -= DT * G_GRAV
            gv[i, j] *= damp[None]
            nx_ = i * DX;  ny_ = j * DX
            sd = _flat_sd(nx_, ny_, theta, hx, hy)
            if sd < 2.0 * DX:
                gsdx = (_flat_sd(nx_+e, ny_, theta, hx, hy) - _flat_sd(nx_-e, ny_, theta, hx, hy)) / (2.0*e)
                gsdy = (_flat_sd(nx_, ny_+e, theta, hx, hy) - _flat_sd(nx_, ny_-e, theta, hx, hy)) / (2.0*e)
                ln = ti.sqrt(gsdx*gsdx + gsdy*gsdy)
                nrm_x = 0.0;  nrm_y = -1.0
                if ln > 1e-12:
                    nrm_x = gsdx / ln;  nrm_y = gsdy / ln
                vleg_x = -omega * (ny_ - hy)
                vleg_y =  omega * (nx_ - hx)
                vrel_n = (gv[i,j][0] - vleg_x)*nrm_x + (gv[i,j][1] - vleg_y)*nrm_y
                if vrel_n < 0.0:
                    gv[i,j][0] -= vrel_n * nrm_x
                    gv[i,j][1] -= vrel_n * nrm_y
                    ti.atomic_add(grf_x[None], gm[i,j] * (-vrel_n) * (-nrm_x) / DT)
                    ti.atomic_add(grf_y[None], gm[i,j] * (-vrel_n) * (-nrm_y) / DT)
                    vt_x = (gv[i,j][0] - vleg_x) - vrel_n * nrm_x
                    vt_y = (gv[i,j][1] - vleg_y) - vrel_n * nrm_y
                    vt_mag = ti.sqrt(vt_x*vt_x + vt_y*vt_y)
                    fric_cap = MU_LEG * ti.abs(vrel_n)
                    if vt_mag > fric_cap:
                        scale = fric_cap / vt_mag
                        gv[i,j][0] -= vt_x * (1.0 - scale)
                        gv[i,j][1] -= vt_y * (1.0 - scale)
            if i < 3:          gv[i,j][0] = ti.max(gv[i,j][0], 0.0)
            if i > GRID_N - 4: gv[i,j][0] = ti.min(gv[i,j][0], 0.0)
            if j < 3:          gv[i,j][1] = ti.max(gv[i,j][1], 0.0)
            if j > GRID_N - 4: gv[i,j][1] = ti.min(gv[i,j][1], 0.0)


@ti.kernel
def grid_contact_cleg(theta: ti.f64, omega: ti.f64, hip_y: ti.f64):
    hx = HIP_X;  hy = hip_y;  e = DX * 0.5
    for i, j in gm:
        if gm[i, j] > 1e-14:
            gv[i, j] /= gm[i, j]
            gv[i, j][1] -= DT * G_GRAV
            gv[i, j] *= damp[None]
            nx_ = i * DX;  ny_ = j * DX
            sd = _cleg_sd(nx_, ny_, theta, hx, hy)
            if sd < 2.0 * DX:
                gsdx = (_cleg_sd(nx_+e, ny_, theta, hx, hy) - _cleg_sd(nx_-e, ny_, theta, hx, hy)) / (2.0*e)
                gsdy = (_cleg_sd(nx_, ny_+e, theta, hx, hy) - _cleg_sd(nx_, ny_-e, theta, hx, hy)) / (2.0*e)
                ln = ti.sqrt(gsdx*gsdx + gsdy*gsdy)
                nrm_x = 0.0;  nrm_y = -1.0
                if ln > 1e-12:
                    nrm_x = gsdx / ln;  nrm_y = gsdy / ln
                vleg_x = -omega * (ny_ - hy)
                vleg_y =  omega * (nx_ - hx)
                vrel_n = (gv[i,j][0] - vleg_x)*nrm_x + (gv[i,j][1] - vleg_y)*nrm_y
                if vrel_n < 0.0:
                    gv[i,j][0] -= vrel_n * nrm_x
                    gv[i,j][1] -= vrel_n * nrm_y
                    ti.atomic_add(grf_x[None], gm[i,j] * (-vrel_n) * (-nrm_x) / DT)
                    ti.atomic_add(grf_y[None], gm[i,j] * (-vrel_n) * (-nrm_y) / DT)
                    vt_x = (gv[i,j][0] - vleg_x) - vrel_n * nrm_x
                    vt_y = (gv[i,j][1] - vleg_y) - vrel_n * nrm_y
                    vt_mag = ti.sqrt(vt_x*vt_x + vt_y*vt_y)
                    fric_cap = MU_LEG * ti.abs(vrel_n)
                    if vt_mag > fric_cap:
                        scale = fric_cap / vt_mag
                        gv[i,j][0] -= vt_x * (1.0 - scale)
                        gv[i,j][1] -= vt_y * (1.0 - scale)
            if i < 3:          gv[i,j][0] = ti.max(gv[i,j][0], 0.0)
            if i > GRID_N - 4: gv[i,j][0] = ti.min(gv[i,j][0], 0.0)
            if j < 3:          gv[i,j][1] = ti.max(gv[i,j][1], 0.0)
            if j > GRID_N - 4: gv[i,j][1] = ti.min(gv[i,j][1], 0.0)


@ti.kernel
def grid_contact_rcleg(theta: ti.f64, omega: ti.f64, hip_y: ti.f64):
    hx = HIP_X;  hy = hip_y;  e = DX * 0.5
    for i, j in gm:
        if gm[i, j] > 1e-14:
            gv[i, j] /= gm[i, j]
            gv[i, j][1] -= DT * G_GRAV
            gv[i, j] *= damp[None]
            nx_ = i * DX;  ny_ = j * DX
            sd = _rcleg_sd(nx_, ny_, theta, hx, hy)
            if sd < 2.0 * DX:
                gsdx = (_rcleg_sd(nx_+e, ny_, theta, hx, hy) - _rcleg_sd(nx_-e, ny_, theta, hx, hy)) / (2.0*e)
                gsdy = (_rcleg_sd(nx_, ny_+e, theta, hx, hy) - _rcleg_sd(nx_, ny_-e, theta, hx, hy)) / (2.0*e)
                ln = ti.sqrt(gsdx*gsdx + gsdy*gsdy)
                nrm_x = 0.0;  nrm_y = -1.0
                if ln > 1e-12:
                    nrm_x = gsdx / ln;  nrm_y = gsdy / ln
                vleg_x = -omega * (ny_ - hy)
                vleg_y =  omega * (nx_ - hx)
                vrel_n = (gv[i,j][0] - vleg_x)*nrm_x + (gv[i,j][1] - vleg_y)*nrm_y
                if vrel_n < 0.0:
                    gv[i,j][0] -= vrel_n * nrm_x
                    gv[i,j][1] -= vrel_n * nrm_y
                    ti.atomic_add(grf_x[None], gm[i,j] * (-vrel_n) * (-nrm_x) / DT)
                    ti.atomic_add(grf_y[None], gm[i,j] * (-vrel_n) * (-nrm_y) / DT)
                    vt_x = (gv[i,j][0] - vleg_x) - vrel_n * nrm_x
                    vt_y = (gv[i,j][1] - vleg_y) - vrel_n * nrm_y
                    vt_mag = ti.sqrt(vt_x*vt_x + vt_y*vt_y)
                    fric_cap = MU_LEG * ti.abs(vrel_n)
                    if vt_mag > fric_cap:
                        scale = fric_cap / vt_mag
                        gv[i,j][0] -= vt_x * (1.0 - scale)
                        gv[i,j][1] -= vt_y * (1.0 - scale)
            if i < 3:          gv[i,j][0] = ti.max(gv[i,j][0], 0.0)
            if i > GRID_N - 4: gv[i,j][0] = ti.min(gv[i,j][0], 0.0)
            if j < 3:          gv[i,j][1] = ti.max(gv[i,j][1], 0.0)
            if j > GRID_N - 4: gv[i,j][1] = ti.min(gv[i,j][1], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RFT ANALYTICAL PREDICTION  (Li et al. 2013, Table S2)
#
# sigma_z(|z|, beta, gamma) = alpha_z(beta, gamma) * |z|
# sigma_x(|z|, beta, gamma) = alpha_x(beta, gamma) * |z|
#
# Fourier fit (Eq. S8, loosely packed poppy seeds):
#   alpha_z = A00 + A10*cos(2b) + B11*sin(2b+g) + B01*sin(g) + Bm11*sin(-2b+g)
#   alpha_x = C11*cos(2b+g) + C01*cos(g) + Cm11*cos(-2b+g) + D10*sin(2b)
#   coefficients in N/cm^3
#
# beta  = angle of element surface normal from HORIZONTAL
# gamma = angle of element velocity from HORIZONTAL
#
# Force on 3D leg (width LEG_W):
#   F_z = int_submerged alpha_z(b,g) * |z| * LEG_W * ds   [N]
#   F_x = int_submerged alpha_x(b,g) * |z| * LEG_W * ds   [N]
# where z [cm] is depth below surface, ds [cm] is arc-length element.
#
# For a FLAT leg at angle theta (all elements share same tangent):
#   beta = gamma = theta  (velocity and normal both perpendicular to leg axis)
#
# For C-LEG and REVERSED C-LEG: beta and gamma vary along the arc (computed below).
# ═══════════════════════════════════════════════════════════════════════════════

_A00  =  0.094;  _A10  =  0.051
_B11  =  0.092;  _B01  =  0.047;  _Bm11 =  0.020
_C11  = -0.026;  _C01  =  0.086;  _Cm11 =  0.018;  _D10  =  0.046


def _alpha_z(b, g):
    return (_A00
            + _A10  * np.cos(2*b)
            + _B11  * np.sin(2*b + g)
            + _B01  * np.sin(g)
            + _Bm11 * np.sin(-2*b + g))


def _alpha_x(b, g):
    return (_C11  * np.cos(2*b + g)
            + _C01  * np.cos(g)
            + _Cm11 * np.cos(-2*b + g)
            + _D10  * np.sin(2*b))


def rft_forces(leg_type, theta_arr, hip_h_m, n_seg=300):
    """
    RFT lift Fz and thrust Fx vs. leg angle theta.

    Parameters
    ----------
    leg_type : str   "flat" | "cleg" | "rcleg"
    theta_arr: array leg angles [rad]
    hip_h_m  : float actual hip height above settled surface [m]

    Returns
    -------
    Fz_mN, Fx_mN : arrays (milli-Newtons, full 3D leg)
    """
    R_cm   = LEG_R * 100.0
    h_cm   = hip_h_m * 100.0
    w_cm   = LEG_W  * 100.0
    s_arr  = np.linspace(0.0, 2.0*R_cm, n_seg + 1)
    ds     = s_arr[1] - s_arr[0]

    Fz_out = np.zeros(len(theta_arr))
    Fx_out = np.zeros(len(theta_arr))

    for k, theta in enumerate(theta_arr):
        Fz = 0.0;  Fx = 0.0
        cos_t = np.cos(theta);  sin_t = np.sin(theta)

        for s in s_arr:
            if leg_type == "flat":
                # depth z [cm]: element at distance s along leg below hip
                z     = s * cos_t - h_cm
                beta  = theta     # normal direction from horizontal
                gamma = theta     # velocity direction from horizontal

            elif leg_type == "cleg":
                # Body-frame position of element: u = R*sin(s/R), v = R*(1-cos(s/R))
                u_s = R_cm * np.sin(s / R_cm)
                v_s = R_cm * (1.0 - np.cos(s / R_cm))
                # World y relative to hip: -u*cos(theta) + v*sin(theta)
                wy_rel = -u_s * cos_t + v_s * sin_t
                z = -wy_rel - h_cm           # depth below surface (positive into soil)
                beta = theta + s / R_cm      # element normal tilted by cumulative arc angle
                # Velocity of element (rotation omega about hip in world frame)
                wx_rel =  u_s * sin_t + v_s * cos_t
                vel_x  = -OMEGA * wy_rel
                vel_y  =  OMEGA * wx_rel
                speed  = abs(vel_x) + abs(vel_y)
                gamma  = float(np.arctan2(vel_y, vel_x)) if speed > 1e-12 else 0.0

            else:  # rcleg
                u_s =  R_cm * np.sin(s / R_cm)
                v_s = -R_cm * (1.0 - np.cos(s / R_cm))
                wy_rel = -u_s * cos_t + v_s * sin_t
                z = -wy_rel - h_cm
                beta = theta - s / R_cm
                wx_rel =  u_s * sin_t + v_s * cos_t
                vel_x  = -OMEGA * wy_rel
                vel_y  =  OMEGA * wx_rel
                speed  = abs(vel_x) + abs(vel_y)
                gamma  = float(np.arctan2(vel_y, vel_x)) if speed > 1e-12 else 0.0

            if z > 0.0:
                Fz += _alpha_z(beta, gamma) * z * w_cm * ds   # N/cm^3 * cm * cm * cm = N
                Fx += _alpha_x(beta, gamma) * z * w_cm * ds

        Fz_out[k] = Fz * 1000.0   # → mN
        Fx_out[k] = Fx * 1000.0

    return Fz_out, Fx_out


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (--frames PNG animation  |  --paraview CSV export)
#
# Body-frame → world-frame transform  (inverse of SDF body transform):
#   wx = u·sin(θ) + v·cos(θ) + hx
#   wy = −u·cos(θ) + v·sin(θ) + hy
#
# C-leg outline  : arc circle of radius LEG_R centred at (u=0, v=+LEG_R),
#                  outer/inner at (LEG_R ± LEG_T), α ∈ [−π/2, 0.4293]
# Rev-c-leg      : centre at (u=0, v=−LEG_R), α ∈ [−0.4293, π/2]
# ═══════════════════════════════════════════════════════════════════════════════

_VM_MAX      = None   # calibrated on first draw call per leg run
_XP0         = None   # particle positions at start of each leg rotation (reference for displacement)
hip_y_global = None   # hip height set in main(), needed by export_paraview_csv


def _leg_strip_world(leg_type, theta, hx, hy, n=80):
    """Return (2*n, 3) triangle-strip points for the leg cross-section.
    Points alternate outer[i] / inner[i] along the arc hip→tip, so
    ParaView fills only the material between the two arcs (no concave fill).
    """
    if leg_type == "flat":
        cos_t = np.cos(theta);  sin_t = np.sin(theta)
        us = np.linspace(0.0, 2.0*LEG_R, n)
        outer_u = us;          outer_v = np.full(n,  LEG_T)
        inner_u = us;          inner_v = np.full(n, -LEG_T)
    elif leg_type == "cleg":
        cos_t = np.cos(theta - np.pi/2);  sin_t = np.sin(theta - np.pi/2)
        al = np.linspace(-np.pi/2, np.pi/2, n)
        outer_u = (LEG_R + LEG_T) * np.cos(al)
        outer_v = LEG_R + (LEG_R + LEG_T) * np.sin(al)
        inner_u = (LEG_R - LEG_T) * np.cos(al)
        inner_v = LEG_R + (LEG_R - LEG_T) * np.sin(al)
    else:  # rcleg
        cos_t = np.cos(theta + np.pi/2);  sin_t = np.sin(theta + np.pi/2)
        al = np.linspace(np.pi/2, -np.pi/2, n)   # hip(pi/2) → tip(-pi/2)
        outer_u = (LEG_R + LEG_T) * np.cos(al)
        outer_v = -LEG_R + (LEG_R + LEG_T) * np.sin(al)
        inner_u = (LEG_R - LEG_T) * np.cos(al)
        inner_v = -LEG_R + (LEG_R - LEG_T) * np.sin(al)

    def bw(uu, vv):
        return uu * sin_t + vv * cos_t + hx, -uu * cos_t + vv * sin_t + hy

    owx, owy = bw(outer_u, outer_v)
    iwx, iwy = bw(inner_u, inner_v)

    # Interleave: outer[0], inner[0], outer[1], inner[1], ...
    pts = np.empty((2 * n, 3), dtype=np.float32)
    pts[0::2, 0] = owx;  pts[0::2, 1] = owy;  pts[0::2, 2] = 0.0
    pts[1::2, 0] = iwx;  pts[1::2, 1] = iwy;  pts[1::2, 2] = 0.0
    return pts


def _leg_outline_world(leg_type, theta, hx, hy, n=160):
    """Return (N,2) world-frame boundary polygon for the leg cross-section."""
    cos_t = np.cos(theta);  sin_t = np.sin(theta)

    def _bw(uu, vv):
        return uu * sin_t + vv * cos_t + hx, -uu * cos_t + vv * sin_t + hy

    if leg_type == "flat":
        us = np.linspace(0.0, 2.0*LEG_R, n // 3)
        vr = np.linspace( LEG_T, -LEG_T, n // 6)
        vl = np.linspace(-LEG_T,  LEG_T, n // 6)
        uu = np.concatenate([us, np.full(len(vr), 2*LEG_R), us[::-1], np.zeros(len(vl))])
        vv = np.concatenate([np.full(len(us), LEG_T), vr, np.full(len(us), -LEG_T), vl])

    elif leg_type == "cleg":
        cos_t = np.cos(theta - np.pi/2);  sin_t = np.sin(theta - np.pi/2)
        al = np.linspace(-np.pi/2, np.pi/2, n // 2)   # semicircle: hip→tip
        uu = np.concatenate([(LEG_R+LEG_T)*np.cos(al),
                             (LEG_R-LEG_T)*np.cos(al[::-1]),
                             [(LEG_R+LEG_T)*np.cos(al[0])]])
        vv = np.concatenate([LEG_R + (LEG_R+LEG_T)*np.sin(al),
                             LEG_R + (LEG_R-LEG_T)*np.sin(al[::-1]),
                             [LEG_R + (LEG_R+LEG_T)*np.sin(al[0])]])

    else:  # rcleg — arc centre at (0, -LEG_R)
        cos_t = np.cos(theta + np.pi/2);  sin_t = np.sin(theta + np.pi/2)
        al = np.linspace(-np.pi/2, np.pi/2, n // 2)   # semicircle: tip→hip (reversed)
        uu = np.concatenate([(LEG_R+LEG_T)*np.cos(al[::-1]),
                             (LEG_R-LEG_T)*np.cos(al),
                             [(LEG_R+LEG_T)*np.cos(al[-1])]])
        vv = np.concatenate([-LEG_R + (LEG_R+LEG_T)*np.sin(al[::-1]),
                             -LEG_R + (LEG_R-LEG_T)*np.sin(al),
                             [-LEG_R + (LEG_R+LEG_T)*np.sin(al[-1])]])

    wx, wy = _bw(uu, vv)
    return np.column_stack([wx, wy])


def draw_frame(leg_type, theta, Fz_N, Fx_N, step, frame_dir, hip_y, surf_y):
    """Render MPM snapshot: particles colored by von Mises stress, displacement quiver,
    and scaled resultant resistive force arrow on the leg."""
    global _VM_MAX
    pos    = xp.to_numpy()
    sig_np = sig.to_numpy()

    # Von Mises stress
    p_s = (sig_np[:, 0, 0] + sig_np[:, 1, 1]) * 0.5
    vm  = np.sqrt(0.5 * ((sig_np[:, 0, 0] - p_s)**2
                       + (sig_np[:, 1, 1] - p_s)**2
                       + 2.0 * sig_np[:, 0, 1]**2))
    if _VM_MAX is None:
        _VM_MAX = max(float(np.percentile(vm, 99)), 1e-3)

    # Displacement from reference positions (start of this leg's rotation)
    disp     = pos - _XP0           # (NP, 2) in metres
    disp_mag = np.linalg.norm(disp, axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#F0EBE0")
    fig.patch.set_facecolor("#F0EBE0")

    # Particles colored by von Mises stress
    sc = ax.scatter(pos[:, 0]*100, pos[:, 1]*100,
                    c=vm, cmap="YlOrRd", vmin=0, vmax=_VM_MAX,
                    s=2.0, linewidths=0, rasterized=True, zorder=2)

    # Displacement quiver — subsample, amplify 8× for visibility, threshold 0.5 mm
    QSTEP = max(1, NP // 600)
    idx_q  = np.arange(0, NP, QSTEP)
    pos_q  = pos[idx_q]
    disp_q = disp[idx_q]
    mag_q  = disp_mag[idx_q]
    show   = mag_q > 5e-4           # > 0.5 mm moved
    if show.any():
        ax.quiver(pos_q[show, 0]*100, pos_q[show, 1]*100,
                  disp_q[show, 0]*800, disp_q[show, 1]*800,   # ×8 amplify, m→cm
                  color="#2980B9", alpha=0.65, angles="xy",
                  scale_units="xy", scale=1.0,
                  width=0.002, headwidth=3, headlength=4, zorder=3)

    # Soil surface reference line
    ax.axhline(surf_y*100, color="#8B7355", lw=0.8, ls="--", alpha=0.55, zorder=1)

    # Leg polygon
    outline = _leg_outline_world(leg_type, theta, HIP_X, hip_y)
    ax.fill(outline[:, 0]*100, outline[:, 1]*100,
            color="#2C3E50", alpha=0.90, zorder=5)
    ax.plot(HIP_X*100, hip_y*100, "o", color="white", ms=5, zorder=6,
            markeredgecolor="#2C3E50", markeredgewidth=0.8)

    # Resultant resistive force arrow (from hip, 1 N = 1.5 cm)
    F_mag = float(np.sqrt(Fx_N**2 + Fz_N**2))
    ARROW_SCALE = 1.5   # cm / N
    if F_mag > 0.01:
        ax.annotate("",
                    xy=(HIP_X*100 + Fx_N * ARROW_SCALE,
                        hip_y*100 + Fz_N * ARROW_SCALE),
                    xytext=(HIP_X*100, hip_y*100),
                    arrowprops=dict(arrowstyle="-|>", color="#E74C3C",
                                   lw=2.2, mutation_scale=18),
                    zorder=7)
        ax.text(HIP_X*100 + Fx_N*ARROW_SCALE + 0.4,
                hip_y*100 + Fz_N*ARROW_SCALE + 0.4,
                f"|F| = {F_mag:.2f} N", fontsize=8.5,
                color="#E74C3C", fontweight="bold", zorder=7)

    # Labels
    leg_names = {"flat": "Flat", "cleg": "C-leg (κ=+1/R)", "rcleg": "Rev. C-leg (κ=−1/R)"}
    ax.set_title(f"{leg_names[leg_type]}   θ = {-np.degrees(theta):+.1f}°", fontsize=11)
    ax.text(0.02, 0.97,
            f"$F_z$ = {Fz_N:+.2f} N  (lift)\n$F_x$ = {Fx_N:+.2f} N  (thrust)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.80, boxstyle="round,pad=0.3"))
    ax.text(0.98, 0.97, "— displacement (×8)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            color="#2980B9")

    plt.colorbar(sc, ax=ax, label="von Mises stress (Pa)", fraction=0.03, pad=0.04)
    ax.set_xlim(0, DOMAIN*100)
    ax.set_ylim(0, DOMAIN*100)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")

    plt.savefig(os.path.join(frame_dir, f"frame_{step:06d}.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)


def _write_vtp(filepath, pts3, point_arrays, poly_pts3=None, strip_pts3=None):
    """Write an ASCII VTK PolyData XML (.vtp) file — no VTK library needed.
    ASCII format is used (not binary) for maximum ParaView compatibility.
    Open simulation.pvd in ParaView to load all frames as a time series.
    strip_pts3: interleaved outer/inner arc points written as a triangle strip
                (avoids concave-fill artifacts for curved legs).
    poly_pts3:  simple polygon (used only for debugging; prefer strip_pts3).
    """
    def fmt_block(arr):
        flat = np.asarray(arr, dtype=np.float32).ravel()
        return '\n'.join(
            '          ' + ' '.join(f'{v:.6g}' for v in flat[i:i+9])
            for i in range(0, len(flat), 9)
        )

    N = len(pts3)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
        '  <PolyData>',
        f'    <Piece NumberOfPoints="{N}" NumberOfVerts="{N}" '
        f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">',
        '      <Points>',
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        fmt_block(pts3),
        '        </DataArray>',
        '      </Points>',
        '      <Verts>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
        '          ' + ' '.join(map(str, range(N))),
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
        '          ' + ' '.join(map(str, range(1, N + 1))),
        '        </DataArray>',
        '      </Verts>',
        '      <PointData>',
    ]
    for name, arr in point_arrays.items():
        arr = np.asarray(arr, dtype=np.float32)
        nc  = arr.shape[1] if arr.ndim == 2 else 1
        lines += [
            f'        <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="{nc}" format="ascii">',
            fmt_block(arr),
            '        </DataArray>',
        ]
    lines += ['      </PointData>', '    </Piece>']

    if strip_pts3 is not None:
        # Triangle strip: outer[0],inner[0],outer[1],inner[1],...
        # Fills only the material between arcs — no concave-fill artifacts.
        M = len(strip_pts3)
        lines += [
            f'    <Piece NumberOfPoints="{M}" NumberOfVerts="0" '
            f'NumberOfLines="0" NumberOfStrips="1" NumberOfPolys="0">',
            '      <Points>',
            '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
            fmt_block(strip_pts3),
            '        </DataArray>',
            '      </Points>',
            '      <Strips>',
            '        <DataArray type="Int32" Name="connectivity" format="ascii">',
            '          ' + ' '.join(map(str, range(M))),
            '        </DataArray>',
            '        <DataArray type="Int32" Name="offsets" format="ascii">',
            f'          {M}',
            '        </DataArray>',
            '      </Strips>',
            '    </Piece>',
        ]
    elif poly_pts3 is not None:
        M = len(poly_pts3)
        lines += [
            f'    <Piece NumberOfPoints="{M}" NumberOfVerts="0" '
            f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">',
            '      <Points>',
            '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
            fmt_block(poly_pts3),
            '        </DataArray>',
            '      </Points>',
            '      <Polys>',
            '        <DataArray type="Int32" Name="connectivity" format="ascii">',
            '          ' + ' '.join(map(str, range(M))),
            '        </DataArray>',
            '        <DataArray type="Int32" Name="offsets" format="ascii">',
            f'          {M}',
            '        </DataArray>',
            '      </Polys>',
            '    </Piece>',
        ]

    lines += ['  </PolyData>', '</VTKFile>']
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def export_paraview_csv(frame_idx, pv_dir, theta, Fz_N, Fx_N, leg_type):
    """Write per-frame VTP files (particles + leg outline) for ParaView."""
    pos    = xp.to_numpy()
    vel    = vp.to_numpy()
    sig_np = sig.to_numpy()

    # Stress scalars
    p_s = (sig_np[:, 0, 0] + sig_np[:, 1, 1]) * 0.5
    vm  = np.sqrt(0.5 * ((sig_np[:, 0, 0] - p_s)**2
                       + (sig_np[:, 1, 1] - p_s)**2
                       + 2.0 * sig_np[:, 0, 1]**2))

    # Displacement from reference positions
    disp     = pos - _XP0
    disp_mag = np.linalg.norm(disp, axis=1)

    # Particle positions (3D with z=0)
    pts3  = np.column_stack([pos, np.zeros(NP)])
    disp3 = np.column_stack([disp, np.zeros(NP)])

    # Particles — rendered as Point Gaussian in ParaView
    _write_vtp(
        os.path.join(pv_dir, f"particles_{frame_idx:04d}.vtp"),
        pts3,
        {
            "vm_stress_Pa": vm,
            "pressure_Pa":  -p_s,
            "disp_mag":     disp_mag,
            "displacement": disp3,
            "velocity":     np.column_stack([vel, np.zeros(NP)]),
        },
    )

    # Leg surface — triangle strip between outer and inner arcs; no concave fill.
    strip3d = _leg_strip_world(leg_type, theta, HIP_X, hip_y_global)
    _write_vtp(
        os.path.join(pv_dir, f"leg_{frame_idx:04d}.vtp"),
        strip3d,   # pts3 (dummy — strip_pts3 drives rendering)
        {},
        strip_pts3=strip3d,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_leg(leg_type, hip_y, out_dir, settled_surf=None):
    """Rotate one leg type; return (theta_rad, Fz_N, Fx_N) arrays."""
    global _VM_MAX, _XP0, hip_y_global
    _VM_MAX      = None                  # recalibrate stress scale per leg
    _XP0         = xp.to_numpy().copy()  # reference positions for displacement tracking
    hip_y_global = hip_y
    label = {"flat": "Flat", "cleg": "C-leg (κ=+1/R)", "rcleg": "Rev. C-leg (κ=−1/R)"}
    print(f"\n  [{label[leg_type]}]  {N_ROT} rotation steps ...")

    surf_y = settled_surf if settled_surf is not None else hip_y - HIP_H

    frame_dir = os.path.join(out_dir, f"frames_{leg_type}")
    pv_dir    = os.path.join(out_dir, f"paraview_{leg_type}")
    if args.frames:
        os.makedirs(frame_dir, exist_ok=True)
        print(f"    PNG frames  → {frame_dir}/  (every {FRAME_EVERY} steps, ~{N_ROT//FRAME_EVERY} frames)")
    if args.paraview:
        os.makedirs(pv_dir, exist_ok=True)
        print(f"    ParaView CSV → {pv_dir}/  (every {FRAME_EVERY} steps)")

    contact_fn = {"flat":  grid_contact_flat_leg,
                  "cleg":  grid_contact_cleg,
                  "rcleg": grid_contact_rcleg}[leg_type]

    th_hist = [];  Fz_hist = [];  Fx_hist = []
    pv_frame = 0   # sequential frame counter for ParaView file naming

    for step in range(N_ROT):
        theta = TH_EN - OMEGA * step * DT   # CCW: +3π/4 → −3π/4
        grf_x[None] = 0.0
        grf_y[None] = 0.0
        clear_grid()
        p2g()
        contact_fn(theta, -OMEGA, hip_y)    # negative omega → CCW surface velocity
        Fx_now = grf_x[None]
        Fz_now = grf_y[None]
        g2p()
        th_hist.append(float(theta))
        Fz_hist.append(float(Fz_now))
        Fx_hist.append(float(Fx_now))
        if step % SAVE_EVERY == 0:
            print(f"    step {step:6d}/{N_ROT}  theta={np.degrees(theta):+6.1f}°  "
                  f"Fz={Fz_now*1e3:+7.2f} mN/m  Fx={Fx_now*1e3:+7.2f} mN/m")
        if step % FRAME_EVERY == 0:
            if args.frames:
                draw_frame(leg_type, theta,
                           float(Fz_now)*LEG_W, float(Fx_now)*LEG_W,
                           step, frame_dir, hip_y, surf_y)
            if args.paraview:
                export_paraview_csv(pv_frame, pv_dir,
                                    theta, float(Fz_now)*LEG_W, float(Fx_now)*LEG_W,
                                    leg_type)
                pv_frame += 1

    th  = np.array(th_hist)
    Fz  = np.array(Fz_hist) * LEG_W   # N/m → N (multiply by 3D width)
    Fx  = np.array(Fx_hist) * LEG_W

    # Time-series metadata for ParaView annotations
    if args.paraview:
        frame_steps = np.arange(0, N_ROT, FRAME_EVERY)
        ts_path = os.path.join(pv_dir, "time_series.csv")
        np.savetxt(ts_path,
                   np.column_stack([
                       frame_steps,
                       -np.degrees(th[frame_steps]),   # Li et al. convention
                       Fz[frame_steps],
                       Fx[frame_steps],
                       np.sqrt(Fz[frame_steps]**2 + Fx[frame_steps]**2),
                   ]),
                   delimiter=",", fmt="%.6g",
                   header="step,theta_deg,Fz_N,Fx_N,F_resultant_N",
                   comments="")
        print(f"    Time-series metadata: {ts_path}")

        # Two PVD collection files — open both in ParaView
        for prefix, label_str in [("particles", "particles (Point Gaussian)"),
                                   ("leg",       "leg polygon (Surface)")]:
            pvd_path = os.path.join(pv_dir, f"{prefix}.pvd")
            pvd_lines = [
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="0.1">',
                '  <Collection>',
            ] + [f'    <DataSet timestep="{i}" file="{prefix}_{i:04d}.vtp"/>'
                 for i in range(pv_frame)] + [
                '  </Collection>',
                '</VTKFile>',
            ]
            with open(pvd_path, 'w') as f:
                f.write('\n'.join(pvd_lines) + '\n')
            print(f"    ParaView {label_str}: {pvd_path}")

    csv_path = os.path.join(out_dir, f"terradyn_{leg_type}.csv")
    np.savetxt(csv_path,
               np.column_stack([th, np.degrees(th), Fz*1e3, Fx*1e3]),
               delimiter=",",
               header="theta_rad,theta_deg,Fz_mN,Fx_mN",
               comments="")
    print(f"    CSV: {csv_path}")
    return th, Fz, Fx


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(out_dir, exist_ok=True)

    init_particles()

    # Phase 1: gravity settlement
    print(f"\nPhase 1: settlement ({SETTLE} steps) ...")
    for step in range(SETTLE):
        clear_grid()
        p2g()
        grid_settle()
        g2p()

    pos_np = xp.to_numpy()
    settled_surf = float(np.percentile(pos_np[:, 1], 97))
    hip_y = settled_surf + HIP_H
    hip_h_actual = hip_y - settled_surf   # should equal HIP_H, small numerical difference
    print(f"  Settled surface: y = {settled_surf:.4f} m  |  Hip y = {hip_y:.4f} m")

    settled = save_state()

    # Phase 2: rotation for each requested leg type
    legs = ["flat", "cleg", "rcleg"] if args.leg == "all" else [args.leg]
    results = {}
    for leg_type in legs:
        restore_state(settled)
        th, Fz3, Fx3 = run_leg(leg_type, hip_y, out_dir, settled_surf=settled_surf)
        results[leg_type] = (th, Fz3, Fx3)

    # Phase 3: plots
    print("\nGenerating plots ...")

    colors = {"flat": "#1B3A5C", "cleg": "#C0392B", "rcleg": "#27AE60"}
    labels = {"flat": "Flat leg", "cleg": "C-leg (κ=+1/R)", "rcleg": "Rev. C-leg (κ=−1/R)"}

    # Load digitized Li et al. experimental data (Fig. S12, Yuma Sand)
    _dig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "Liter Data", "Li Science", "Data_digitalized")
    _leg_csv = {"flat": "Fig_S12_B.csv", "cleg": "Fig_S12_A.csv", "rcleg": "Fig_S12_C.csv"}

    def _load_exp(leg_type):
        """Return (fx_deg, fx_N), (fz_deg, fz_N) from digitized Fig S12 CSV."""
        path = os.path.join(_dig_dir, _leg_csv[leg_type])
        if not os.path.exists(path):
            return None, None
        raw = np.genfromtxt(path, delimiter=",", skip_header=2, filling_values=np.nan)
        def pull(cx, cy):
            mask = ~(np.isnan(raw[:, cx]) | np.isnan(raw[:, cy]))
            xd = np.degrees(raw[mask, cx]);  y = raw[mask, cy]
            s  = np.argsort(xd)
            return xd[s], y[s]
        return pull(0, 1), pull(4, 5)   # Fx-exp, Fz-exp

    # MPM force scale factor: our ω vs. experiment ω=0.2 (quasi-static linear scaling)
    OMEGA_EXP  = 0.2
    SCALE_FACTOR = OMEGA_EXP / OMEGA    # =1 when running at experimental ω

    omega_note = (f"ω = {OMEGA:.2g} rad/s  (experimental)"
                  if abs(SCALE_FACTOR - 1.0) < 0.01
                  else f"ω = {OMEGA:.1f} rad/s  (rescaled ×{SCALE_FACTOR:.3f} → ω_exp = {OMEGA_EXP} rad/s)")

    # Per-leg panel figure (2 rows: Fz top, Fx bottom)
    fig, axes = plt.subplots(2, len(legs), figsize=(5.5*len(legs), 8), squeeze=False)
    fig.suptitle(
        "MPM Simulation vs. Li et al. (2013) Experiment — Yuma Sand\n"
        f"E = {E_MOD/1e6:.0f} MPa  |  φ = {np.degrees(PHI_RAD):.0f}°  |  {omega_note}",
        fontsize=10.5, y=1.01)

    for col, leg_type in enumerate(legs):
        th, Fz3, Fx3 = results[leg_type]
        clr = colors[leg_type]
        win = max(1, N_ROT // 80)
        # Rescale MPM forces to experiment omega, convert N → N (already N after LEG_W)
        Fz_sm = np.convolve(Fz3 * SCALE_FACTOR, np.ones(win)/win, mode="same")
        Fx_sm = np.convolve(Fx3 * SCALE_FACTOR, np.ones(win)/win, mode="same")

        fx_exp, fz_exp = _load_exp(leg_type)

        ax_z = axes[0][col];  ax_x = axes[1][col]

        th_plot = -np.degrees(th)   # negate to match Li et al. sign convention
        ax_z.plot(th_plot, Fz_sm, color=clr, lw=2.0, label="MPM (rescaled)")
        ax_z.plot(th_plot, Fz3*SCALE_FACTOR, color=clr, lw=0.5, alpha=0.15)
        if fz_exp is not None:
            ax_z.plot(*fz_exp, color="k", ls="--", lw=1.5, alpha=0.80,
                      label="Li et al. exp (Fig. S12)")
        ax_z.axhline(0, color="gray", lw=0.6, ls=":"); ax_z.axvline(0, color="gray", lw=0.6, ls=":")
        ax_z.set_ylabel("Lift  $F_z$  (N)"); ax_z.set_title(labels[leg_type])
        ax_z.legend(fontsize=8, framealpha=0.85); ax_z.grid(True, alpha=0.2, lw=0.5)

        ax_x.plot(th_plot, Fx_sm, color=clr, lw=2.0, label="MPM (rescaled)")
        ax_x.plot(th_plot, Fx3*SCALE_FACTOR, color=clr, lw=0.5, alpha=0.15)
        if fx_exp is not None:
            ax_x.plot(*fx_exp, color="k", ls="--", lw=1.5, alpha=0.80,
                      label="Li et al. exp (Fig. S12)")
        ax_x.axhline(0, color="gray", lw=0.6, ls=":"); ax_x.axvline(0, color="gray", lw=0.6, ls=":")
        ax_x.set_ylabel("Thrust  $F_x$  (N)"); ax_x.set_xlabel("Leg angle  θ  (deg)")
        ax_x.legend(fontsize=8, framealpha=0.85); ax_x.grid(True, alpha=0.2, lw=0.5)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(np.degrees(TH_ST), np.degrees(TH_EN))   # -135 → +135
            ax.set_xticks([-90.0, 90.0])
            ax.set_xticklabels([r"$-\pi/2$", r"$\pi/2$"])

    plt.tight_layout()
    p1 = os.path.join(out_dir, "terradynamics_exp_comparison.png")
    plt.savefig(p1, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Panel figure: {p1}")

    # Combined proposal figure (only when all three legs run)
    if len(legs) == 3:
        fig2, (az, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        fig2.suptitle(
            "MPM vs. Li et al. (2013) Experiment — Yuma Sand\n"
            "Lift ($F_z$) and Thrust ($F_x$) for Three Leg Geometries",
            fontsize=11)
        for leg_type in legs:
            th, Fz3, Fx3 = results[leg_type]
            clr = colors[leg_type];  lbl = labels[leg_type]
            win = max(1, N_ROT // 80)
            Fz_sm = np.convolve(Fz3*SCALE_FACTOR, np.ones(win)/win, mode="same")
            Fx_sm = np.convolve(Fx3*SCALE_FACTOR, np.ones(win)/win, mode="same")
            fx_exp, fz_exp = _load_exp(leg_type)
            th_plot = -np.degrees(th)   # negate to match Li et al. sign convention
            az.plot(th_plot,  Fz_sm, color=clr, lw=2.0,  label=f"{lbl} — MPM")
            ax2.plot(th_plot, Fx_sm, color=clr, lw=2.0,  label=f"{lbl} — MPM")
            if fz_exp is not None:
                az.plot(*fz_exp,  color=clr, ls="--", lw=1.4, alpha=0.70,
                        label=f"{lbl} — exp")
            if fx_exp is not None:
                ax2.plot(*fx_exp, color=clr, ls="--", lw=1.4, alpha=0.70,
                         label=f"{lbl} — exp")
        for ax in (az, ax2):
            ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(0, color="k", lw=0.6, ls=":")
            ax.set_xlim(np.degrees(TH_ST), np.degrees(TH_EN))   # -135 → +135
            ax.set_xticks([-90.0, 90.0])
            ax.set_xticklabels([r"$-\pi/2$", r"$\pi/2$"])
            ax.set_xlabel("Leg angle  θ"); ax.grid(True, alpha=0.2, lw=0.5)
            ax.legend(fontsize=7.5, framealpha=0.88, ncol=2)
        az.set_ylabel("Lift  $F_z$  (N)");   az.set_title("Vertical force  (Lift $F_z$)")
        ax2.set_ylabel("Thrust  $F_x$  (N)"); ax2.set_title("Horizontal force  (Thrust $F_x$)")
        plt.tight_layout()
        p2 = os.path.join(out_dir, "terradynamics_proposal.png")
        plt.savefig(p2, dpi=180, bbox_inches="tight"); plt.close()
        print(f"  Proposal figure: {p2}")

    # ── Summary statistics for inverse analysis ───────────────────────────────
    # 4 stats × up to 3 legs = up to 12 scalar observables per simulation run.
    summary = {
        "phi_deg": float(np.degrees(PHI_RAD)),
        "c_Pa":    float(C_COH),
        "E_Pa":    float(E_MOD),
    }
    for leg_type, (th, Fz3, Fx3) in results.items():
        win   = max(1, N_ROT // 80)
        Fz_sm = np.convolve(Fz3, np.ones(win)/win, mode="same")
        Fx_sm = np.convolve(Fx3, np.ones(win)/win, mode="same")
        th_plot = -np.degrees(th)   # Li et al. sign convention
        summary[f"{leg_type}_Fz_peak_N"]      = float(np.max(Fz_sm))
        summary[f"{leg_type}_Fz_peak_ang_deg"] = float(th_plot[np.argmax(Fz_sm)])
        summary[f"{leg_type}_Fx_peak_N"]      = float(np.max(np.abs(Fx_sm)))
        summary[f"{leg_type}_Fz_integral_Ndeg"] = float(np.trapz(Fz_sm, th_plot))

    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary stats: {args.summary_out}")
    else:
        sp = os.path.join(out_dir, "summary_stats.json")
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary stats: {sp}")

    # Run log
    log = {
        "timestamp": datetime.now().isoformat(),
        "GRID_N": GRID_N, "NP": NP, "DT": DT,
        "SETTLE": SETTLE, "N_ROT": N_ROT, "OMEGA": OMEGA,
        "legs_run": legs,
        "settled_surf_m": float(settled_surf),
        "hip_y_m": float(hip_y),
        "hip_h_actual_m": float(hip_h_actual),
        "soil": {"rho": RHO, "E": E_MOD, "nu": NU,
                 "phi_deg": float(np.degrees(PHI_RAD)), "c_Pa": C_COH},
        "leg_geometry": {"R_m": LEG_R, "T_m": LEG_T, "W_m": LEG_W, "HIP_H_m": HIP_H},
    }
    lp = os.path.join(out_dir, "terradynamics_log.json")
    with open(lp, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log: {lp}")
    if args.frames:
        print("\nTo stitch frames into MP4 (requires ffmpeg):")
        for leg_type in legs:
            fdir = os.path.join(out_dir, f"frames_{leg_type}")
            mp4  = os.path.join(out_dir, f"animation_{leg_type}.mp4")
            print(f"  ffmpeg -r 24 -i '{fdir}/frame_%06d.png' "
                  f"-vcodec libx264 -pix_fmt yuv420p -crf 20 '{mp4}'")

    if args.paraview:
        print("\nParaView workflow:")
        for leg_type in legs:
            pdir = os.path.join(out_dir, f"paraview_{leg_type}")
            print(f"  [{leg_type}]  data → {pdir}/")
            print(f"    File > Open > frame_*.vtp  (select all, ParaView auto-detects series)")
            print(f"    Click Apply — particles and leg outline render immediately")
            print(f"    Colour by: vm_stress_Pa | pressure_Pa | disp_mag")
            print(f"    Representation: Point Gaussian (particles) | Surface (leg polygon)")
            print(f"    Displacement: Filters > Warp By Vector > select 'displacement' array")
            print(f"    Metadata: time_series.csv  (step, theta_deg, Fz_N, Fx_N, F_resultant_N)")

    print("\n✓ Terradynamics simulation complete.")


if __name__ == "__main__":
    main()
