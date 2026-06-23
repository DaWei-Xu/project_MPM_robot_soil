#!/usr/bin/env python3
"""
Sweep Perzyna Drucker-Prager parameters for the flat-leg smoke problem.

Usage:
    mamba run -n env_MPM_robot_soil python script/calibrate_perzyna_flat.py

The default objective is deliberately simple: compare each Perzyna run against
the rate-independent flat-leg baseline and rank settings whose raw force peaks
move toward a requested peak-force ratio. This is a calibration scaffold, not a
final physical fit to experiment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    vp_eta: float
    vp_n: float
    vp_stress_ref: float


def parse_float_list(values: list[str] | None) -> list[float]:
    parsed = []
    for value in values or []:
        for part in value.replace(",", " ").split():
            parsed.append(float(part))
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one number")
    return parsed


def fmt_token(value: float) -> str:
    text = f"{value:.6g}"
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace("e", "en")
    )


def patch_dt(src: Path, dst: Path, dt: float) -> None:
    text = src.read_text()
    old = "DT          = 8e-6"
    new = f"DT          = {dt:.12g}"
    if old not in text:
        raise RuntimeError(f"Could not find expected DT assignment: {old}")
    dst.write_text(text.replace(old, new, 1))


def load_summary(path: Path) -> dict[str, float]:
    with path.open() as f:
        return json.load(f)


def run_case(
    *,
    python_exe: str,
    sim_path: Path,
    sim_dir: Path,
    out_dir: Path,
    omega: float,
    settle: int,
    threads: int,
    cfg: RunConfig,
    dry_run: bool,
    verbose: bool,
) -> dict[str, float] | None:
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "run.log"
    cmd = [
        python_exe,
        str(sim_path),
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
        str(cfg.vp_eta),
        "--vp_n",
        str(cfg.vp_n),
        "--vp_stress_ref",
        str(cfg.vp_stress_ref),
    ]

    print(
        "\nRunning "
        f"eta={cfg.vp_eta:g}, n={cfg.vp_n:g}, "
        f"stress_ref={cfg.vp_stress_ref:g} Pa"
    )
    print("  " + " ".join(cmd))
    if dry_run:
        return None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        subprocess.run(cmd, cwd=sim_dir, check=True)
    else:
        with log_path.open("w") as log:
            subprocess.run(
                cmd,
                cwd=sim_dir,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    if not summary_path.exists():
        raise RuntimeError(f"Simulation did not write summary: {summary_path}")
    return load_summary(summary_path)


def row_from_summary(
    cfg: RunConfig,
    summary: dict[str, float],
    baseline: dict[str, float],
    target_fz_ratio: float,
    target_fx_ratio: float,
) -> dict[str, float | str]:
    fz_peak = float(summary["flat_Fz_peak_N"])
    fx_peak = float(summary["flat_Fx_peak_N"])
    fz_base = float(baseline["flat_Fz_peak_N"])
    fx_base = float(baseline["flat_Fx_peak_N"])
    fz_ratio = fz_peak / fz_base if fz_base else float("nan")
    fx_ratio = fx_peak / fx_base if fx_base else float("nan")
    score = abs(fz_ratio - target_fz_ratio) + abs(fx_ratio - target_fx_ratio)
    return {
        "score": score,
        "vp_eta_s": cfg.vp_eta,
        "vp_n": cfg.vp_n,
        "vp_stress_ref_Pa": cfg.vp_stress_ref,
        "Fz_peak_N": fz_peak,
        "Fx_peak_N": fx_peak,
        "Fz_peak_ratio_vs_baseline": fz_ratio,
        "Fx_peak_ratio_vs_baseline": fx_ratio,
        "Fz_peak_angle_deg": float(summary["flat_Fz_peak_ang_deg"]),
        "Fz_integral_Ndeg": float(summary["flat_Fz_integral_Ndeg"]),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sim_dir = repo_root / "MPM-robot-soil-prototype"
    sim_src = sim_dir / "mpm_terradynamics.py"

    parser = argparse.ArgumentParser(
        description="Calibrate/sweep Perzyna Drucker-Prager parameters for the flat leg."
    )
    parser.add_argument("--omega", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=8e-6)
    parser.add_argument("--settle", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--etas",
        nargs="+",
        default=["0.001", "0.01", "0.1", "1.0", "10.0"],
        help="Space- or comma-separated vp_eta values.",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        default=["1.0"],
        help="Space- or comma-separated Perzyna exponents.",
    )
    parser.add_argument(
        "--stress-refs",
        nargs="+",
        default=["1000", "10000", "100000"],
        help="Space- or comma-separated reference stresses in Pa.",
    )
    parser.add_argument(
        "--target-fz-ratio",
        type=float,
        default=0.5,
        help="Desired Fz peak ratio relative to rate-independent baseline.",
    )
    parser.add_argument(
        "--target-fx-ratio",
        type=float,
        default=0.5,
        help="Desired Fx peak ratio relative to rate-independent baseline.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root / "data" / "synthetic_data" / "perzyna_calibration_flat",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream full simulator output instead of writing each run to run.log.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.etas = parse_float_list(args.etas)
    args.ns = parse_float_list(args.ns)
    args.stress_refs = parse_float_list(args.stress_refs)

    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_sim = sim_dir / f"mpm_terradynamics_calib_dt_{fmt_token(args.dt)}.py"
    patch_dt(sim_src, tmp_sim, args.dt)

    try:
        baseline_cfg = RunConfig(vp_eta=0.0, vp_n=1.0, vp_stress_ref=1000.0)
        baseline_out = out_root / "baseline_rate_independent"
        baseline = run_case(
            python_exe=sys.executable,
            sim_path=tmp_sim,
            sim_dir=sim_dir,
            out_dir=baseline_out,
            omega=args.omega,
            settle=args.settle,
            threads=args.threads,
            cfg=baseline_cfg,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if args.dry_run:
            return 0
        assert baseline is not None

        rows: list[dict[str, float | str]] = []
        for eta, n_exp, stress_ref in itertools.product(
            args.etas, args.ns, args.stress_refs
        ):
            cfg = RunConfig(vp_eta=eta, vp_n=n_exp, vp_stress_ref=stress_ref)
            case_name = (
                f"eta_{fmt_token(eta)}"
                f"_n_{fmt_token(n_exp)}"
                f"_sref_{fmt_token(stress_ref)}"
            )
            summary = run_case(
                python_exe=sys.executable,
                sim_path=tmp_sim,
                sim_dir=sim_dir,
                out_dir=out_root / case_name,
                omega=args.omega,
                settle=args.settle,
                threads=args.threads,
                cfg=cfg,
                dry_run=False,
                verbose=args.verbose,
            )
            assert summary is not None
            rows.append(
                row_from_summary(
                    cfg,
                    summary,
                    baseline,
                    args.target_fz_ratio,
                    args.target_fx_ratio,
                )
            )

        rows.sort(key=lambda row: float(row["score"]))
        csv_path = out_root / "calibration_results.csv"
        json_path = out_root / "calibration_results.json"
        write_csv(csv_path, rows)
        with json_path.open("w") as f:
            json.dump(
                {
                    "objective": {
                        "target_fz_ratio": args.target_fz_ratio,
                        "target_fx_ratio": args.target_fx_ratio,
                        "baseline_summary": baseline,
                    },
                    "results": rows,
                },
                f,
                indent=2,
            )

        print("\nCalibration sweep complete.")
        print(f"Results CSV : {csv_path}")
        print(f"Results JSON: {json_path}")
        print("\nBest candidates:")
        for row in rows[:5]:
            print(
                "  "
                f"score={row['score']:.4g}, "
                f"eta={row['vp_eta_s']}, "
                f"n={row['vp_n']}, "
                f"sref={row['vp_stress_ref_Pa']}, "
                f"Fz_ratio={row['Fz_peak_ratio_vs_baseline']:.3g}, "
                f"Fx_ratio={row['Fx_peak_ratio_vs_baseline']:.3g}"
            )
    finally:
        try:
            tmp_sim.unlink()
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
