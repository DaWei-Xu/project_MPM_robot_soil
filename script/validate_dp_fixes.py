"""
validate_dp_fixes.py
====================
Unit tests for two fixes applied to the Drucker-Prager return mapping in
mpm_terradynamics.py.  The return-mapping logic is re-implemented in NumPy so
tests run instantly without Taichi or the full MPM simulation.

Fix 1 — Perzyna clamp removed:
  OLD:  dg = min(dg_vp, dg_ri)   → silently degenerates to rate-independent
                                    when VP_ETA is small
  NEW:  dg = dg_vp               → pure Perzyna; stress may remain outside or
                                    land inside yield surface depending on step size

Fix 2 — Tensile cutoff zeros full stress tensor:
  OLD:  s_ret -= p_chk * I       → removes only hydrostatic; deviatoric residual
                                    can leave stress OUTSIDE the yield surface
  NEW:  s_ret = 0                → cohesionless soil: zero all stress components
"""

import sys
import numpy as np

# ── Parameters (must match mpm_terradynamics.py defaults) ─────────────────────
E_MOD   = 20e6
NU      = 0.30
C_COH   = 10.0       # Pa
DT      = 8e-6       # s  (simulation time step)

PHI_RAD  = np.radians(35.0)
MU_E     = E_MOD / (2.0 * (1.0 + NU))
LAM_E    = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
ALPHA_DP = 2.0 * np.sin(PHI_RAD) / (np.sqrt(3.0) * (3.0 - np.sin(PHI_RAD)))
K_C_DP   = 6.0 * C_COH * np.cos(PHI_RAD) / (np.sqrt(3.0) * (3.0 - np.sin(PHI_RAD)))
H_DP     = MU_E + ALPHA_DP**2 * (LAM_E + MU_E)


# ── Return-mapping implementations ────────────────────────────────────────────

def _dp_state(sig):
    """Yield-function components for a 2×2 stress matrix."""
    I2  = np.eye(2)
    p_s = (sig[0, 0] + sig[1, 1]) * 0.5
    s_d = sig - p_s * I2
    q_s = np.sqrt(0.5 * (s_d[0, 0]**2 + s_d[1, 1]**2 + 2.0 * s_d[0, 1]**2))
    f   = q_s + ALPHA_DP * p_s - K_C_DP
    return p_s, s_d, q_s, f


def f_of(sig):
    return _dp_state(sig)[3]


def return_map_old(sig_tr, VP_ETA=0.0, VP_N=1.0, VP_STRESS_REF=1000.0):
    """OLD: min-clamp Perzyna + hydrostatic-only tensile cutoff."""
    I2 = np.eye(2)
    p_s, s_dev, q_s, f_yld = _dp_state(sig_tr)
    s_ret = sig_tr.copy()
    if f_yld > 0.0:
        dg_ri = f_yld / max(H_DP, 1e-14)
        dg    = dg_ri
        if VP_ETA > 0.0:
            overstress = max(f_yld / VP_STRESS_REF, 0.0)
            dg_vp = (DT / VP_ETA) * overstress**VP_N
            dg = min(dg_vp, dg_ri)                  # OLD clamp
        s_new = s_dev * (1.0 - dg * MU_E / q_s) if q_s > 1e-12 else s_dev.copy()
        p_new = p_s - ALPHA_DP * (LAM_E + MU_E) * dg
        s_ret = s_new + p_new * I2
        p_chk = (s_ret[0, 0] + s_ret[1, 1]) * 0.5
        if p_chk > 0.0:
            s_ret = s_ret - p_chk * I2               # OLD: only remove hydrostatic
    return s_ret


def return_map_new(sig_tr, VP_ETA=0.0, VP_N=1.0, VP_STRESS_REF=1000.0):
    """NEW: pure Perzyna (no clamp) + full-tensor tensile cutoff."""
    I2 = np.eye(2)
    p_s, s_dev, q_s, f_yld = _dp_state(sig_tr)
    s_ret = sig_tr.copy()
    if f_yld > 0.0:
        dg_ri = f_yld / max(H_DP, 1e-14)
        dg    = dg_ri
        if VP_ETA > 0.0:
            overstress = max(f_yld / VP_STRESS_REF, 0.0)
            dg_vp = (DT / VP_ETA) * overstress**VP_N
            dg = dg_vp                               # NEW: pure Perzyna
        s_new = s_dev * (1.0 - dg * MU_E / q_s) if q_s > 1e-12 else s_dev.copy()
        p_new = p_s - ALPHA_DP * (LAM_E + MU_E) * dg
        s_ret = s_new + p_new * I2
        p_chk = (s_ret[0, 0] + s_ret[1, 1]) * 0.5
        if p_chk > 0.0:
            s_ret = np.zeros((2, 2))                 # NEW: zero full stress tensor
    return s_ret


# ── Test harness ───────────────────────────────────────────────────────────────

_pass = 0
_fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    tag = "  PASS" if cond else "  FAIL"
    print(f"{tag}  {name}")
    if not cond and detail:
        print(f"        {detail}")
    if cond:
        _pass += 1
    else:
        _fail += 1


# ── Stress states ──────────────────────────────────────────────────────────────
#
# SIG_SHEAR:  deep compression + shear → outside yield, stays compressive after return
#   σ = [[-5000, 5000],[5000,-5000]] Pa
#   p = −5000 Pa, q = 5000 Pa, f ≈ +3624 Pa
#
# SIG_INSIDE: well inside yield surface → elastic, unchanged
#   σ = [[-2000, 100],[100,-1000]] Pa
#
# SIG_TENS:   small tensile + deviatoric → outside yield, p_new > 0 after DP return
#   σ = [[3500, 0],[0, 500]] Pa
#   p = +2000 Pa, q = 1500 Pa, f ≈ +2034 Pa → p_new ≈ +830 Pa after return
#
# VP timing:
#   VP_SLOW = 10 s    → dg_vp ≪ dg_ri (both implementations agree)
#   VP_FAST = 0.036 s → dg_vp ≈ 2×dg_ri (implementations diverge; fix matters)

SIG_SHEAR  = np.array([[-5000.0, 5000.0], [5000.0, -5000.0]])
# Deep compression + tiny shear → f = 100 - 1365 - 12 < 0 (elastic, inside yield)
SIG_INSIDE = np.array([[-5000.0,  100.0], [ 100.0, -5000.0]])
SIG_TENS   = np.array([[ 3500.0,    0.0], [   0.0,   500.0]])
SIG_COMP   = np.array([[-1000.0, 3000.0], [3000.0, -2000.0]])  # compressive; no tensile cutoff

VP_SLOW = 10.0    # s — dg_vp ≪ dg_ri
VP_FAST = 0.036   # s — dg_vp ≈ 2×dg_ri (diverges; fix is visible here)
# Note: VP_ETA < 2*q_s*DT/(MU_E*overstress) causes deviatoric amplification in
# explicit integration — the stability floor for SIG_SHEAR is ~0.022 s.


# ══════════════════════════════════════════════════════════════════════════════
print("\n── Baseline (rate-independent, VP_ETA=0) ─────────────────────────────")

check("elastic state unchanged",
      np.allclose(return_map_new(SIG_INSIDE), SIG_INSIDE),
      f"f_inside = {f_of(SIG_INSIDE):.1f} Pa (should be < 0)")

f_ri_ret = f_of(return_map_new(SIG_SHEAR))
check("rate-independent return: f ≈ 0 after projection",
      abs(f_ri_ret) < 1.0,
      f"f_returned = {f_ri_ret:.4f} Pa (tol 1 Pa)")

check("old == new for VP_ETA=0",
      np.allclose(return_map_old(SIG_SHEAR), return_map_new(SIG_SHEAR), atol=1e-8))


# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fix 1: Perzyna clamp ──────────────────────────────────────────────")

f_trial  = f_of(SIG_SHEAR)
dg_ri_v  = f_trial / H_DP
overstress = f_trial / 1000.0  # VP_STRESS_REF default

dg_vp_slow = (DT / VP_SLOW) * overstress
dg_vp_fast = (DT / VP_FAST) * overstress

check("slow VP_ETA: dg_vp < dg_ri (pre-condition for agreement)",
      dg_vp_slow < dg_ri_v,
      f"dg_vp={dg_vp_slow:.3e}  dg_ri={dg_ri_v:.3e}")

sig_old_slow = return_map_old(SIG_SHEAR, VP_ETA=VP_SLOW)
sig_new_slow = return_map_new(SIG_SHEAR, VP_ETA=VP_SLOW)
check("slow VP_ETA: old == new (min clamp inactive, dg = dg_vp in both)",
      np.allclose(sig_old_slow, sig_new_slow, atol=1e-8))

check("slow VP_ETA: stress remains outside yield (f > 0) — viscoplastic partial return",
      f_of(sig_new_slow) > 1.0,
      f"f = {f_of(sig_new_slow):.2f} Pa")

check("fast VP_ETA: dg_vp > dg_ri (pre-condition for divergence)",
      dg_vp_fast > dg_ri_v,
      f"dg_vp={dg_vp_fast:.3e}  dg_ri={dg_ri_v:.3e}")

f_old_fast = f_of(return_map_old(SIG_SHEAR, VP_ETA=VP_FAST))
f_new_fast = f_of(return_map_new(SIG_SHEAR, VP_ETA=VP_FAST))

check("fast VP_ETA OLD: degenerates to rate-independent (f ≈ 0) — the bug",
      abs(f_old_fast) < 1.0,
      f"f = {f_old_fast:.4f} Pa")

check("fast VP_ETA NEW: stress driven inside yield surface (f < 0) — fix active",
      f_new_fast < -1.0,
      f"f = {f_new_fast:.4f} Pa")

check("fast VP_ETA: old != new (fix changes the result)",
      not np.allclose(return_map_old(SIG_SHEAR, VP_ETA=VP_FAST),
                      return_map_new(SIG_SHEAR, VP_ETA=VP_FAST), atol=1e-6))

# Stability note: explicit Perzyna requires VP_ETA > 2*q_s*DT/(MU_E*overstress).
# For SIG_SHEAR that floor is ~0.022 s.  Below it the deviatoric correction
# overshoots and amplifies — the OLD min-clamp accidentally prevented this, but the
# physically correct fix is to enforce VP_ETA above the stability floor, not to cap dg.
dg_stability_limit = 2.0 * 5000.0 / MU_E  # 2*q_s/MU_E
check("stability floor documented: dg_ri < dg_fast < dg_stability_limit",
      dg_ri_v < dg_vp_fast < dg_stability_limit,
      f"dg_ri={dg_ri_v:.3e}  dg_fast={dg_vp_fast:.3e}  limit={dg_stability_limit:.3e}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n── Fix 2: tensile cutoff ─────────────────────────────────────────────")

p_s_tens   = _dp_state(SIG_TENS)[0]
f_tens     = _dp_state(SIG_TENS)[3]
p_new_tens = p_s_tens - ALPHA_DP * (LAM_E + MU_E) * (f_tens / H_DP)
check("tensile pre-condition: p_new > 0 after DP return (tensile cutoff will fire)",
      p_new_tens > 0.0,
      f"p_new = {p_new_tens:.2f} Pa")

sig_old_tens = return_map_old(SIG_TENS)
sig_new_tens = return_map_new(SIG_TENS)

f_old_tens = f_of(sig_old_tens)
f_new_tens = f_of(sig_new_tens)

check("OLD tensile cutoff: stress still OUTSIDE yield (f > 0) — deviatoric residual bug",
      f_old_tens > 1.0,
      f"f = {f_old_tens:.2f} Pa  (nonzero deviatoric left: {sig_old_tens.ravel()})")

p_old = (sig_old_tens[0, 0] + sig_old_tens[1, 1]) * 0.5
check("OLD tensile cutoff: mean stress is zero but deviatoric survives",
      abs(p_old) < 1e-6 and not np.allclose(sig_old_tens, 0.0, atol=1e-3),
      f"p = {p_old:.2e} Pa  deviatoric = {(sig_old_tens - p_old*np.eye(2)).ravel()}")

check("NEW tensile cutoff: full stress tensor is zero",
      np.allclose(sig_new_tens, 0.0, atol=1e-12))

check("NEW tensile cutoff: f < 0 (trivially inside yield)",
      f_new_tens < 0.0,
      f"f = {f_new_tens:.4f} Pa")

# Compressive state: tensile cutoff must NOT fire in either implementation
sig_old_comp = return_map_old(SIG_COMP)
sig_new_comp = return_map_new(SIG_COMP)
p_comp = (sig_new_comp[0, 0] + sig_new_comp[1, 1]) * 0.5
check("compressive state: old == new (tensile cutoff not triggered)",
      np.allclose(sig_old_comp, sig_new_comp, atol=1e-8))
check("compressive state: mean stress stays compressive after return",
      p_comp <= 0.0,
      f"p = {p_comp:.2f} Pa")


# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 56)
total = _pass + _fail
print(f"  {_pass}/{total} passed" + ("  ✓" if _fail == 0 else f"  — {_fail} FAILED"))
sys.exit(0 if _fail == 0 else 1)
