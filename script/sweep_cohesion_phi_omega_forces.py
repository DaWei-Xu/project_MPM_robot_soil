#!/usr/bin/env python3
"""
Sweep cohesion, friction angle, and omega for flat-leg force outputs.

Default design:
    36 soil cases = 6 cohesion values x 6 friction-angle values
    cohesion c    = 0, 5, 10, 15, 20, 25 kPa
    friction phi  = 10, 15, 20, 25, 30, 35 deg
    omega         = 10, 15 rad/s for each soil case

That gives 72 total simulations.

If you want to include the upper endpoints from the verbal range, run:
    --cohesions 0 5 10 15 20 25 30 --phis 10 15 20 25 30 35 40

Usage:
    mamba run -n env_MPM_robot_soil python script/sweep_cohesion_phi_omega_forces.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_float_list(values: list[str]) -> list[float]:
    parsed = []
    for value in values:
        for part in value.replace(",", " ").split():
            parsed.append(float(part))
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return parsed


def token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p").replace("e", "en")


def run_case(
    *,
    python_exe: str,
    sim_py: Path,
    sim_dir: Path,
    out_dir: Path,
    phi: float,
    cohesion_kpa: float,
    omega: float,
    settle: int,
    threads: int,
    force: bool,
    dry_run: bool,
) -> dict[str, float | str] | None:
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "terradyn_flat.csv"

    if summary_path.exists() and csv_path.exists() and not force:
        print(f"Skipping phi={phi:g}, c={cohesion_kpa:g} kPa, omega={omega:g}; output exists.")
    else:
        cohesion_pa = cohesion_kpa * 1000.0
        cmd = [
            python_exe,
            str(sim_py),
            "--leg",
            "flat",
            "--phi",
            str(phi),
            "--c",
            str(cohesion_pa),
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
            "0.0",
        ]

        print(
            f"\nRunning phi={phi:g} deg, c={cohesion_kpa:g} kPa "
            f"({cohesion_pa:g} Pa), omega={omega:g} rad/s"
        )
        print("  " + " ".join(cmd))
        if dry_run:
            return None

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

    if dry_run:
        return None

    with summary_path.open() as f:
        summary = json.load(f)

    return {
        "phi_deg": phi,
        "cohesion_kPa": cohesion_kpa,
        "cohesion_Pa": cohesion_kpa * 1000.0,
        "omega_rad_s": omega,
        "Fz_peak_N": float(summary["flat_Fz_peak_N"]),
        "Fx_peak_N": float(summary["flat_Fx_peak_N"]),
        "Fz_peak_angle_deg": float(summary["flat_Fz_peak_ang_deg"]),
        "Fz_integral_Ndeg": float(summary["flat_Fz_integral_Ndeg"]),
        "case_dir": str(out_dir),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
    }


def write_results(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sim_dir = repo_root / "MPM-robot-soil-prototype"
    sim_py = sim_dir / "mpm_terradynamics.py"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohesions",
        nargs="+",
        default=["0", "5", "10", "15", "20", "25"],
        help="Cohesion values in kPa. Default gives 6 values for 36 soil cases.",
    )
    parser.add_argument(
        "--phis",
        nargs="+",
        default=["10", "15", "20", "25", "30", "35"],
        help="Friction angles in degrees. Default gives 6 values for 36 soil cases.",
    )
    parser.add_argument("--omegas", nargs="+", default=["10", "15"])
    parser.add_argument("--settle", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root / "data" / "synthetic_data" / "cohesion_phi_omega_force_sweep",
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing cases.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cohesions_kpa = parse_float_list(args.cohesions)
    phis = parse_float_list(args.phis)
    omegas = parse_float_list(args.omegas)

    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(cohesions_kpa) * len(phis) * len(omegas)
    print(
        f"Preparing {len(cohesions_kpa) * len(phis)} soil cases x "
        f"{len(omegas)} omegas = {total} simulations."
    )
    print("Cohesion inputs are interpreted as kPa and converted to Pa for mpm_terradynamics.py.")

    rows: list[dict[str, float | str]] = []
    for idx, (cohesion_kpa, phi, omega) in enumerate(
        itertools.product(cohesions_kpa, phis, omegas), start=1
    ):
        case_dir = (
            out_root
            / f"c_{token(cohesion_kpa)}kPa"
            / f"phi_{token(phi)}deg"
            / f"omega_{token(omega)}"
        )
        print(f"\n[{idx}/{total}]")
        row = run_case(
            python_exe=sys.executable,
            sim_py=sim_py,
            sim_dir=sim_dir,
            out_dir=case_dir,
            phi=phi,
            cohesion_kpa=cohesion_kpa,
            omega=omega,
            settle=args.settle,
            threads=args.threads,
            force=args.force,
            dry_run=args.dry_run,
        )
        if row is not None:
            rows.append(row)

    if args.dry_run:
        return 0

    result_csv = out_root / "force_sweep_summary.csv"
    write_results(result_csv, rows)

    print("\nForce sweep complete.")
    print(f"Summary CSV: {result_csv}")
    print(f"Rows written: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
