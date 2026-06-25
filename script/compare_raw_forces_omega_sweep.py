#!/usr/bin/env python3
"""
Run flat-leg MPM cases at several angular velocities and compare raw forces.

This script explicitly enables Perzyna rate-dependent Drucker-Prager soil.

Default sweep:
    omega = 2, 4, 6, 8, 10 rad/s

Usage:
    mamba run -n env_MPM_robot_soil python script/compare_raw_forces_omega_sweep.py

The script writes one output directory per omega plus:
    raw_force_comparison.png
    raw_force_peak_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_omegas(values: list[str]) -> list[float]:
    omegas: list[float] = []
    for value in values:
        for part in value.replace(",", " ").split():
            omegas.append(float(part))
    if not omegas:
        raise argparse.ArgumentTypeError("expected at least one omega")
    return omegas


def omega_token(omega: float) -> str:
    return f"{omega:.6g}".replace(".", "p").replace("-", "m")


def smooth(arr: np.ndarray, target_windows: int = 80) -> np.ndarray:
    win = max(1, len(arr) // target_windows)
    return np.convolve(arr, np.ones(win) / win, mode="same")


def run_case(
    *,
    python_exe: str,
    sim_py: Path,
    sim_dir: Path,
    out_dir: Path,
    omega: float,
    settle: int,
    threads: int,
    vp_eta: float,
    vp_n: float,
    vp_stress_ref: float,
    force: bool,
    dry_run: bool,
) -> None:
    csv_path = out_dir / "terradyn_flat.csv"
    summary_path = out_dir / "summary.json"
    if csv_path.exists() and summary_path.exists() and not force:
        print(f"Skipping omega={omega:g}; existing output found: {out_dir}")
        return

    cmd = [
        python_exe,
        str(sim_py),
        "--leg",
        "flat",
        "--omega",
        str(omega),
        "--settle",
        str(settle),
        "--threads",
        str(threads),
        "--out",
        str(out_dir),
        "--summary_out",
        str(summary_path),
        "--vp_eta",
        str(vp_eta),
        "--vp_n",
        str(vp_n),
        "--vp_stress_ref",
        str(vp_stress_ref),
    ]

    print(
        f"\nRunning rate-dependent case, omega={omega:g}, "
        f"eta={vp_eta:g}, n={vp_n:g}, stress_ref={vp_stress_ref:g}"
    )
    print("  " + " ".join(cmd))
    if dry_run:
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.log").open("w") as log:
        subprocess.run(
            cmd,
            cwd=sim_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def load_case(out_dir: Path) -> dict[str, np.ndarray | float]:
    csv_path = out_dir / "terradyn_flat.csv"
    summary_path = out_dir / "summary.json"
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    with summary_path.open() as f:
        summary = json.load(f)

    # CSV theta_deg is the simulator convention. Negate it to match the plots
    # used elsewhere in this repo and Li et al.'s displayed sign convention.
    theta_deg = -data[:, 1]
    fz_n = data[:, 2] / 1000.0
    fx_n = data[:, 3] / 1000.0

    return {
        "theta_deg": theta_deg,
        "Fz_N": fz_n,
        "Fx_N": fx_n,
        "Fz_sm_N": smooth(fz_n),
        "Fx_sm_N": smooth(fx_n),
        "Fz_peak_N": float(summary["flat_Fz_peak_N"]),
        "Fx_peak_N": float(summary["flat_Fx_peak_N"]),
        "Fz_peak_angle_deg": float(summary["flat_Fz_peak_ang_deg"]),
        "Fz_integral_Ndeg": float(summary["flat_Fz_integral_Ndeg"]),
    }


def write_peak_summary(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "omega_rad_s",
                "Fz_peak_N",
                "Fx_peak_N",
                "Fz_peak_angle_deg",
                "Fz_integral_Ndeg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(out_root: Path, cases: dict[float, dict[str, np.ndarray | float]]) -> Path:
    cmap = plt.get_cmap("viridis")
    omegas = sorted(cases)
    colors = {omega: cmap(i / max(1, len(omegas) - 1)) for i, omega in enumerate(omegas)}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Raw Flat-Leg MPM Forces vs. Angular Velocity, Rate-Dependent Soil", fontsize=12)

    for omega in omegas:
        case = cases[omega]
        theta = case["theta_deg"]
        color = colors[omega]
        label = f"omega={omega:g}"

        axes[0].plot(theta, case["Fz_N"], color=color, lw=0.35, alpha=0.18)
        axes[0].plot(theta, case["Fz_sm_N"], color=color, lw=1.8, label=label)
        axes[1].plot(theta, case["Fx_N"], color=color, lw=0.35, alpha=0.18)
        axes[1].plot(theta, case["Fx_sm_N"], color=color, lw=1.8, label=label)

    axes[0].set_ylabel("Raw lift Fz (N)")
    axes[1].set_ylabel("Raw thrust Fx (N)")
    axes[1].set_xlabel("Leg angle theta (deg)")

    for ax in axes:
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.set_xlim(-135.0, 135.0)
        ax.set_xticks([-90.0, 0.0, 90.0])
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.legend(fontsize=8, framealpha=0.9, ncol=3)

    plt.tight_layout()
    plot_path = out_root / "raw_force_comparison.png"
    plt.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sim_dir = repo_root / "MPM-robot-soil-prototype"
    sim_py = sim_dir / "mpm_terradynamics.py"

    parser = argparse.ArgumentParser()
    parser.add_argument("--omegas", nargs="+", default=["2", "4", "6", "8", "10"])
    parser.add_argument("--settle", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--vp-eta",
        type=float,
        default=100.0,
        help="Perzyna relaxation time in seconds. Default gives visible rate stiffening.",
    )
    parser.add_argument("--vp-n", type=float, default=1.0)
    parser.add_argument("--vp-stress-ref", type=float, default=1000.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root / "data" / "synthetic_data" / "omega_sweep_raw_forces",
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing cases.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    omegas = parse_omegas(args.omegas)
    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for omega in omegas:
        run_case(
            python_exe=sys.executable,
            sim_py=sim_py,
            sim_dir=sim_dir,
            out_dir=out_root / f"omega_{omega_token(omega)}",
            omega=omega,
            settle=args.settle,
            threads=args.threads,
            vp_eta=args.vp_eta,
            vp_n=args.vp_n,
            vp_stress_ref=args.vp_stress_ref,
            force=args.force,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        return 0

    cases = {
        omega: load_case(out_root / f"omega_{omega_token(omega)}")
        for omega in omegas
    }
    rows = [
        {
            "omega_rad_s": omega,
            "Fz_peak_N": float(cases[omega]["Fz_peak_N"]),
            "Fx_peak_N": float(cases[omega]["Fx_peak_N"]),
            "Fz_peak_angle_deg": float(cases[omega]["Fz_peak_angle_deg"]),
            "Fz_integral_Ndeg": float(cases[omega]["Fz_integral_Ndeg"]),
        }
        for omega in omegas
    ]

    summary_path = out_root / "raw_force_peak_summary.csv"
    plot_path = plot_comparison(out_root, cases)
    write_peak_summary(summary_path, rows)

    print("\nRate-dependent omega sweep complete.")
    print(
        f"Perzyna parameters: eta={args.vp_eta:g} s, "
        f"n={args.vp_n:g}, stress_ref={args.vp_stress_ref:g} Pa"
    )
    print(f"Peak summary: {summary_path}")
    print(f"Raw-force plot: {plot_path}")
    for row in rows:
        print(
            "  "
            f"omega={row['omega_rad_s']:g}: "
            f"Fz_peak={row['Fz_peak_N']:.3g} N, "
            f"Fx_peak={row['Fx_peak_N']:.3g} N, "
            f"Fz_angle={row['Fz_peak_angle_deg']:.2f} deg"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
