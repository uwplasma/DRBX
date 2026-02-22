#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, stdin_path: Path | None = None) -> None:
    if stdin_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    else:
        with open(stdin_path, "rb") as f:
            proc = subprocess.run(
                cmd, cwd=str(cwd) if cwd else None, stdin=f, capture_output=True, text=True
            )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")


def _ensure_salpha(case_dir: Path, template_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    src = template_dir / "salpha.nc"
    dst = case_dir / "salpha.nc"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Run Hermes/GBS/jax_drb benchmarks and generate plots.")
    p.add_argument("--output-dir", default="benchmarks/full_benchmark")
    p.add_argument("--hermes-exe", default="external/hermes-3/build/hermes-3")
    p.add_argument("--gbs-exe", default="external/gbs/bin/skel")
    p.add_argument("--mpi", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mpi-n", type=int, default=1)
    p.add_argument("--gbs-dims", default="1,1")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hermes_exe = Path(args.hermes_exe)
    if not hermes_exe.is_absolute():
        hermes_exe = root / hermes_exe

    gbs_exe = Path(args.gbs_exe)
    if not gbs_exe.is_absolute():
        gbs_exe = root / gbs_exe

    hermes_template = root / "external/hermes-3/examples_min/salpha_grid"
    hermes_linear = root / "benchmarks/cases/hermes_salpha_linear"
    hermes_nonlinear = root / "benchmarks/cases/hermes_salpha_nonlinear"
    _ensure_salpha(hermes_linear, hermes_template)
    _ensure_salpha(hermes_nonlinear, hermes_template)

    if hermes_exe.exists():
        cmd = [str(hermes_exe), "-d", "."]
        if args.mpi and shutil.which("mpirun"):
            cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
        _run(cmd, cwd=hermes_linear)
        hermes_linear_out = hermes_linear / "BOUT.dmp.0.nc"
        if hermes_linear_out.exists():
            shutil.copy2(hermes_linear_out, output_dir / "hermes_linear.nc")
        _run(cmd, cwd=hermes_nonlinear)
        hermes_non_out = hermes_nonlinear / "BOUT.dmp.0.nc"
        if hermes_non_out.exists():
            shutil.copy2(hermes_non_out, output_dir / "hermes_nonlinear.nc")

    gbs_case = root / "benchmarks/cases/gbs_salpha"
    dims = [d.strip() for d in args.gbs_dims.split(",") if d.strip()]
    if len(dims) != 2:
        dims = ["1", "1"]
    if gbs_exe.exists():
        cmd = [str(gbs_exe)] + dims
        if args.mpi and shutil.which("mpirun"):
            cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
        _run(cmd, cwd=gbs_case, stdin_path=gbs_case / "in_linear")
        gbs_linear = gbs_case / "results_min_00.h5"
        if gbs_linear.exists():
            shutil.copy2(gbs_linear, output_dir / "gbs_linear.h5")
        _run(cmd, cwd=gbs_case, stdin_path=gbs_case / "in_nonlinear")
        gbs_non = gbs_case / "results_min_00.h5"
        if gbs_non.exists():
            shutil.copy2(gbs_non, output_dir / "gbs_nonlinear.h5")

    # jax_drb runs
    jax_run = root / "benchmarks/run_jaxdrb_sim.py"
    _run(
        [
            "python",
            str(jax_run),
            "--config",
            str(root / "benchmarks/cases/jaxdrb/salpha_linear.toml"),
            "--dt",
            "1e-4",
            "--nsteps",
            "1000",
            "--save-every",
            "10",
            "--output",
            str(output_dir / "jaxdrb_linear.npz"),
        ]
    )
    _run(
        [
            "python",
            str(jax_run),
            "--config",
            str(root / "benchmarks/cases/jaxdrb/salpha_nonlinear.toml"),
            "--dt",
            "1e-4",
            "--nsteps",
            "1000",
            "--save-every",
            "10",
            "--output",
            str(output_dir / "jaxdrb_nonlinear.npz"),
        ]
    )

    # Analysis
    analysis = root / "benchmarks/analysis/benchmark_report.py"
    _run(
        [
            "python",
            str(analysis),
            "--hermes",
            str(output_dir / "hermes_linear.nc"),
            "--gbs",
            str(output_dir / "gbs_linear.h5"),
            "--jaxdrb",
            str(output_dir / "jaxdrb_linear.npz"),
            "--output",
            str(output_dir / "linear"),
            "--gbs-input",
            str(root / "benchmarks/cases/gbs_salpha/in_linear"),
            "--jaxdrb-config",
            str(root / "benchmarks/cases/jaxdrb/salpha_linear.toml"),
        ]
    )
    _run(
        [
            "python",
            str(analysis),
            "--hermes",
            str(output_dir / "hermes_nonlinear.nc"),
            "--gbs",
            str(output_dir / "gbs_nonlinear.h5"),
            "--jaxdrb",
            str(output_dir / "jaxdrb_nonlinear.npz"),
            "--output",
            str(output_dir / "nonlinear"),
            "--gbs-input",
            str(root / "benchmarks/cases/gbs_salpha/in_nonlinear"),
            "--jaxdrb-config",
            str(root / "benchmarks/cases/jaxdrb/salpha_nonlinear.toml"),
        ]
    )


if __name__ == "__main__":
    main()
