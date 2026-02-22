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
    p = argparse.ArgumentParser(
        description="Run SOL benchmark (Hermes/GBS/jax_drb) and generate plots."
    )
    p.add_argument("--output-dir", default="benchmarks/sol_benchmark")
    p.add_argument("--hermes-exe", default="external/hermes-3/build/hermes-3")
    p.add_argument("--gbs-exe", default="external/gbs/bin/skel")
    p.add_argument("--mpi", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mpi-n", type=int, default=1)
    p.add_argument("--gbs-dims", default="1,1")
    p.add_argument("--hermes-axes", default="xzy")
    p.add_argument("--gbs-axes", default="zxy")
    p.add_argument("--jaxdrb-axes", default="zxy")
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
    hermes_sol = root / "benchmarks/cases/hermes_salpha_sol"
    _ensure_salpha(hermes_sol, hermes_template)

    if hermes_exe.exists():
        cmd = [str(hermes_exe), "-d", "."]
        if args.mpi and shutil.which("mpirun"):
            cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
        _run(cmd, cwd=hermes_sol)
        hermes_out = hermes_sol / "BOUT.dmp.0.nc"
        if hermes_out.exists():
            shutil.copy2(hermes_out, output_dir / "hermes_sol.nc")

    gbs_case = root / "benchmarks/cases/gbs_salpha"
    dims = [d.strip() for d in args.gbs_dims.split(",") if d.strip()]
    if len(dims) != 2:
        dims = ["1", "1"]
    if gbs_exe.exists():
        cmd = [str(gbs_exe)] + dims
        if args.mpi and shutil.which("mpirun"):
            cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
        _run(cmd, cwd=gbs_case, stdin_path=gbs_case / "in_sol")
        gbs_out = gbs_case / "results_min_00.h5"
        if gbs_out.exists():
            shutil.copy2(gbs_out, output_dir / "gbs_sol.h5")

    # jax_drb run
    jax_run = root / "benchmarks/run_jaxdrb_sim.py"
    _run(
        [
            "python",
            str(jax_run),
            "--config",
            str(root / "benchmarks/cases/jaxdrb/salpha_sol.toml"),
            "--dt",
            "5e-5",
            "--nsteps",
            "4000",
            "--save-every",
            "20",
            "--output",
            str(output_dir / "jaxdrb_sol.npz"),
        ]
    )

    # Analysis
    analysis = root / "benchmarks/analysis/benchmark_report.py"
    _run(
        [
            "python",
            str(analysis),
            "--hermes",
            str(output_dir / "hermes_sol.nc"),
            "--gbs",
            str(output_dir / "gbs_sol.h5"),
            "--jaxdrb",
            str(output_dir / "jaxdrb_sol.npz"),
            "--output",
            str(output_dir / "sol"),
            "--gbs-input",
            str(root / "benchmarks/cases/gbs_salpha/in_sol"),
            "--jaxdrb-config",
            str(root / "benchmarks/cases/jaxdrb/salpha_sol.toml"),
            "--hermes-axes",
            args.hermes_axes,
            "--gbs-axes",
            args.gbs_axes,
            "--jaxdrb-axes",
            args.jaxdrb_axes,
        ]
    )


if __name__ == "__main__":
    main()
