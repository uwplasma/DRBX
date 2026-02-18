#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_metrics(output: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in output.strip().splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        if key not in ("curv_x", "curv_y", "dpar_factor", "B"):
            continue
        try:
            stats = eval(parts[1].strip(), {"__builtins__": {}})
        except Exception:
            continue
        if isinstance(stats, dict):
            metrics[key] = {str(k): float(v) for k, v in stats.items() if isinstance(v, (int, float))}
    return metrics


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alignment comparisons for Hermes/GBS/JAX-DRB.")
    parser.add_argument("--output-dir", default="benchmarks/alignment_outputs", help="Output directory")
    parser.add_argument("--compare-only", action="store_true", help="Skip running external codes")
    parser.add_argument("--run-hermes", action="store_true", help="Run Hermes-3 before comparing")
    parser.add_argument("--run-gbs", action="store_true", help="Run GBS before comparing")
    parser.add_argument("--hermes-exe", default="external/hermes-3/build/hermes-3")
    parser.add_argument("--hermes-case", default="external/hermes-3/examples_min/salpha_grid")
    parser.add_argument("--gbs-exe", default="", help="Path to GBS executable (optional)")
    parser.add_argument("--gbs-case", default="external/gbs/bin")
    parser.add_argument("--mpi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mpi-n", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hermes_case = root / args.hermes_case
    gbs_case = root / args.gbs_case

    if not args.compare_only:
        if args.run_hermes:
            hermes_exe = Path(args.hermes_exe)
            if not hermes_exe.is_absolute():
                hermes_exe = root / hermes_exe
            if hermes_exe.exists():
                cmd = [str(hermes_exe), "-d", str(hermes_case)]
                if args.mpi and shutil.which("mpirun"):
                    cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
                _run(cmd, cwd=root)
            else:
                print("Hermes-3 executable not found, skipping run.")

        if args.run_gbs:
            if args.gbs_exe:
                gbs_exe = Path(args.gbs_exe)
                if not gbs_exe.is_absolute():
                    gbs_exe = root / gbs_exe
                if gbs_exe.exists():
                    cmd = [str(gbs_exe), str(gbs_case / "in_min")]
                    _run(cmd, cwd=gbs_case)
                else:
                    print("GBS executable not found, skipping run.")
            else:
                print("No GBS executable provided, skipping run.")

    # Copy useful outputs into alignment folder
    _copy_if_exists(hermes_case / "BOUT.inp", output_dir / "hermes" / "BOUT.inp")
    _copy_if_exists(hermes_case / "BOUT.dmp.0.nc", output_dir / "hermes" / "BOUT.dmp.0.nc")
    _copy_if_exists(hermes_case / "BOUT.restart.0.nc", output_dir / "hermes" / "BOUT.restart.0.nc")
    _copy_if_exists(hermes_case / "BOUT.log.0", output_dir / "hermes" / "BOUT.log.0")
    _copy_if_exists(hermes_case / "salpha.nc", output_dir / "hermes" / "salpha.nc")

    _copy_if_exists(gbs_case / "in_min", output_dir / "gbs" / "in_min")
    _copy_if_exists(gbs_case / "results_min_00.h5", output_dir / "gbs" / "results_min_00.h5")
    _copy_if_exists(gbs_case / "restart_min_00.h5", output_dir / "gbs" / "restart_min_00.h5")

    # Compare geometry metrics
    hermes_config = root / "configs/benchmarks/salpha_hermes_gridmatch_small.toml"
    hermes_grid = hermes_case / "salpha.nc"
    gbs_config = root / "configs/benchmarks/salpha_gbs_match.toml"
    gbs_results = gbs_case / "results_min_00.h5"

    metrics = {}
    if hermes_config.exists() and hermes_grid.exists():
        out = _run(
            [
                sys.executable,
                str(root / "tools/compare_geometry_metrics.py"),
                "--config",
                str(hermes_config),
                "--bout-grid",
                str(hermes_grid),
                "--mapping",
                "canonical",
                "--x-index",
                "0",
            ]
        )
        metrics["hermes"] = _parse_metrics(out)
    else:
        metrics["hermes"] = {}

    if gbs_config.exists() and gbs_results.exists():
        out = _run(
            [
                sys.executable,
                str(root / "tools/compare_geometry_gbs.py"),
                "--config",
                str(gbs_config),
                "--gbs-file",
                str(gbs_results),
                "--mapping",
                "canonical",
            ]
        )
        metrics["gbs"] = _parse_metrics(out)
    else:
        metrics["gbs"] = {}

    metrics_file = output_dir / "alignment_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote metrics to {metrics_file}")


if __name__ == "__main__":
    main()
