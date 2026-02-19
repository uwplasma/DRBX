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


def _rms(arr) -> float:
    import numpy as np

    a = np.asarray(arr).ravel()
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a**2)))


def _extract_hermes_field_rms(path: Path) -> dict[str, float]:
    try:
        import netCDF4  # type: ignore
        import numpy as np
    except Exception:
        return {}
    if not path.exists():
        return {}
    with netCDF4.Dataset(path) as ds:
        for name in ("Te", "Ne", "Vort"):
            if name in ds.variables:
                arr = np.asarray(ds.variables[name][:])
                if arr.ndim >= 4:
                    arr = arr[-1]
                return {name: _rms(arr)}
    return {}


def _extract_gbs_field_rms(path: Path) -> dict[str, float]:
    try:
        import h5py  # type: ignore
        import numpy as np
    except Exception:
        return {}
    if not path.exists():
        return {}
    with h5py.File(path, "r") as f:
        for base in ("data/var3d/temperature", "data/var3d/temperaturi", "data/var2d/temperaturixy"):
            if base in f:
                grp = f[base]
                steps = [k for k in grp.keys() if k.isdigit()]
                if not steps:
                    continue
                latest = sorted(steps)[-1]
                arr = np.asarray(grp[latest][...])
                return {base.split("/")[-1]: _rms(arr)}
        if "data/var0d/globtemperature" in f:
            arr = np.asarray(f["data/var0d/globtemperature"][...])
            if arr.size:
                return {"globtemperature": float(arr[-1])}
    return {}


def _jaxdrb_metrics(config_path: Path) -> dict[str, dict[str, float]]:
    from jaxdrb.driver import build_system_from_config
    from jaxdrb.io import load_config

    cfg = load_config(str(config_path))
    built = build_system_from_config(cfg.data)
    geom = built.system.geom

    metrics: dict[str, dict[str, float]] = {}
    for key in ("curv_x", "curv_y", "dpar_factor", "B"):
        val = getattr(geom, key, None)
        if val is None:
            continue
        rms_val = _rms(val)
        metrics[key] = {"rms_ref": rms_val, "rms_geom": rms_val, "rel_error": 0.0}
    metrics["field_rms"] = {
        "n": _rms(built.state.n),
        "Te": _rms(built.state.Te),
    }
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
    parser.add_argument("--run-jaxdrb", action="store_true", help="Run jax_drb CLI before comparing")
    parser.add_argument("--hermes-exe", default="external/hermes-3/build/hermes-3")
    parser.add_argument("--hermes-case", default="external/hermes-3/examples_min/salpha_grid")
    parser.add_argument(
        "--gbs-exe",
        default="external/gbs/bin/skel",
        help="Path to GBS executable (defaults to external/gbs/bin/skel)",
    )
    parser.add_argument("--gbs-case", default="external/gbs/bin")
    parser.add_argument(
        "--gbs-dims",
        default="1,1",
        help="MPI topology dims for GBS (ncz,ncx) when using mpirun",
    )
    parser.add_argument(
        "--jax-config",
        default="configs/benchmarks/salpha_hermes_min_run.toml",
        help="jax_drb config used to compute analytic geometry metrics",
    )
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
                cmd = [str(hermes_exe), "-d", "."]
                if args.mpi and shutil.which("mpirun"):
                    cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
                _run(cmd, cwd=hermes_case)
            else:
                print("Hermes-3 executable not found, skipping run.")

        if args.run_gbs:
            gbs_exe = Path(args.gbs_exe)
            if not gbs_exe.is_absolute():
                gbs_exe = root / gbs_exe
            if gbs_exe.exists():
                dims = [d.strip() for d in str(args.gbs_dims).split(",") if d.strip()]
                if len(dims) != 2:
                    dims = ["1", "1"]
                cmd = [str(gbs_exe)] + dims
                if args.mpi and shutil.which("mpirun"):
                    cmd = ["mpirun", "-np", str(max(args.mpi_n, 1))] + cmd
                with open(gbs_case / "in_min", "rb") as f:
                    proc = subprocess.run(cmd, cwd=str(gbs_case), stdin=f, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"GBS run failed: {proc.stderr}")
            else:
                print("GBS executable not found, skipping run.")

        if args.run_jaxdrb:
            jax_cfg = Path(args.jax_config)
            if not jax_cfg.is_absolute():
                jax_cfg = root / jax_cfg
            if jax_cfg.exists():
                cmd = [sys.executable, "-m", "jaxdrb.cli.main", str(jax_cfg)]
                _run(cmd, cwd=root)
            else:
                print("jax_drb config not found, skipping run.")

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
    jax_config = Path(args.jax_config)
    if not jax_config.is_absolute():
        jax_config = root / jax_config

    metrics = {}
    if jax_config.exists():
        metrics["jaxdrb"] = _jaxdrb_metrics(jax_config)
    else:
        metrics["jaxdrb"] = {}
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
    hermes_dmp = hermes_case / "BOUT.dmp.0.nc"
    hermes_fields = _extract_hermes_field_rms(hermes_dmp)
    if hermes_fields:
        metrics.setdefault("hermes", {})["field_rms"] = hermes_fields

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
    gbs_fields = _extract_gbs_field_rms(gbs_results)
    if gbs_fields:
        metrics.setdefault("gbs", {})["field_rms"] = gbs_fields

    metrics_file = output_dir / "alignment_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote metrics to {metrics_file}")


if __name__ == "__main__":
    main()
