from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.driver import run_simulation


def _save_run_npz(path: Path, result) -> None:
    payload = dict(result.diagnostics)
    if "times" not in payload:
        payload["times"] = np.asarray(result.times)
    if "t" not in payload:
        payload["t"] = np.asarray(result.times)
    np.savez(path, **payload)


def _run_jax(cfg: dict[str, Any], t_end: float, out_npz: Path) -> None:
    time_cfg = dict(cfg.get("time", {}))
    dt = float(time_cfg.get("dt", 1e-3))
    if dt <= 0.0:
        raise ValueError("time.dt must be > 0.")
    nsteps = int(max(1, math.ceil(float(t_end) / dt)))
    time_cfg["nsteps"] = nsteps
    time_cfg.setdefault("save_every", max(1, nsteps // 50))
    time_cfg["save_fields"] = True
    time_cfg.setdefault("snapshot_fields", ["n", "Te", "omega", "phi"])
    time_cfg["return_numpy"] = True
    cfg["time"] = time_cfg
    print(f"[jax] running t_end={t_end} with dt={dt} nsteps={nsteps}")
    result = run_simulation(cfg, as_numpy=True)
    _save_run_npz(out_npz, result)
    print(f"[jax] saved: {out_npz}")


def _call(repo_root: Path, *cmd: str) -> None:
    subprocess.run([sys.executable, *cmd], cwd=repo_root, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run a tokamak ES/cold/Bouss benchmark workflow:\n"
            "1) short Hermes-vs-jax alignment panel, 2) jax poloidal/3D turbulence media."
        )
    )
    p.add_argument(
        "--jax-config",
        default="examples/open_field_line/input_tokamak_bxcv_benchmark_es_cold.toml",
        help="jax_drb benchmark TOML config.",
    )
    p.add_argument(
        "--hermes-data",
        default="runs/hermes_open_field_short/data",
        help="Hermes data directory containing BOUT.dmp.*.nc files.",
    )
    p.add_argument(
        "--out-dir",
        default="runs/tokamak_benchmark_latest",
        help="Output directory for run/bundle artifacts.",
    )
    p.add_argument(
        "--fig-dir",
        default="docs/figures",
        help="Figure/movie output directory.",
    )
    p.add_argument(
        "--t-end-short",
        type=float,
        default=0.1,
        help="Short-window end time for Hermes alignment panel.",
    )
    p.add_argument(
        "--t-end-visual",
        type=float,
        default=0.5,
        help="Longer window used for turbulence snapshots/movies.",
    )
    p.add_argument(
        "--field",
        default="n",
        choices=("n", "Te", "omega", "phi"),
        help="Field used in panel and media generation.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / args.jax_config).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    fig_dir = (repo_root / args.fig_dir).resolve()
    hermes_dir = (repo_root / args.hermes_data).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with cfg_path.open("rb") as f:
        base_cfg = tomllib.load(f)

    coeff_path = str(base_cfg.get("geometry", {}).get("coeff_path", ""))

    # 1) Short run for direct Hermes-vs-jax panel.
    jax_short = out_dir / "jax_short.npz"
    _run_jax(dict(base_cfg), float(args.t_end_short), jax_short)

    bundle_jax = out_dir / "bundle_jax_short.npz"
    _call(
        repo_root,
        "tools/build_benchmark_bundle.py",
        "--code",
        "jax",
        "--input",
        str(jax_short),
        "--output",
        str(bundle_jax),
        "--config",
        str(cfg_path),
        "--geometry",
        "tokamak_open_field",
    )

    bundle_hermes = out_dir / "bundle_hermes_short.npz"
    _call(
        repo_root,
        "tools/build_benchmark_bundle.py",
        "--code",
        "hermes",
        "--input",
        str(hermes_dir),
        "--output",
        str(bundle_hermes),
        "--config",
        str(cfg_path),
        "--geometry",
        "tokamak_open_field",
    )

    panel_png = fig_dir / "tokamak_sol_benchmark_panel.png"
    panel_csv = fig_dir / "tokamak_sol_benchmark_panel.csv"
    panel_cmd = [
        "tools/plot_benchmark_panel.py",
        "--hermes",
        str(bundle_hermes),
        "--jax",
        str(bundle_jax),
        "--out",
        str(panel_png),
        "--summary-csv",
        str(panel_csv),
        "--field",
        args.field,
        "--snapshot-mode",
        "fluct",
        "--plane",
        "xz",
    ]
    if coeff_path:
        panel_cmd.extend(["--coeff-path", str((repo_root / coeff_path).resolve())])
    _call(repo_root, *panel_cmd)

    # 2) Longer run for turbulence media in tokamak geometry.
    jax_visual = out_dir / "jax_visual.npz"
    _run_jax(dict(base_cfg), float(args.t_end_visual), jax_visual)

    poloidal_png = fig_dir / "tokamak_sol_poloidal_fluct.png"
    _call(
        repo_root,
        "tools/plot_poloidal_plane.py",
        str(jax_visual),
        "--config",
        str(cfg_path),
        "--field",
        args.field,
        "--out",
        str(poloidal_png),
        "--equilibrium",
        "none",
        "--overlay-mask",
        "--fluct",
        "zonal",
        "--lowpass",
        "0.25",
        "--field-scale",
        "6.0",
        "--cmap",
        "coolwarm",
        "--symmetric",
        "--interp-grid",
        "300",
    )

    poloidal_gif = fig_dir / "tokamak_sol_movie.gif"
    _call(
        repo_root,
        "tools/make_poloidal_movie.py",
        str(jax_visual),
        "--config",
        str(cfg_path),
        "--field",
        f"snapshots_{args.field}",
        "--out",
        str(poloidal_gif),
        "--stride",
        "2",
        "--fluct",
        "zonal",
        "--lowpass",
        "0.25",
        "--skip-fraction",
        "0.2",
        "--symmetric",
        "--range-tail",
        "--field-scale",
        "6.0",
        "--range-scale",
        "0.7",
        "--interp-grid",
        "320",
    )

    tok3d_gif = fig_dir / "tokamak_sol_3d_movie.gif"
    _call(
        repo_root,
        "tools/make_tokamak_3d_movie.py",
        str(jax_visual),
        "--config",
        str(cfg_path),
        "--field",
        f"snapshots_{args.field}",
        "--out",
        str(tok3d_gif),
        "--time-stride",
        "3",
        "--skip-fraction",
        "0.2",
        "--fluct",
        "zonal",
        "--symmetric",
        "--range-tail",
        "--tail-fraction",
        "0.3",
        "--phi-cut-1",
        "-0.52",
        "--phi-cut-2",
        "0.52",
        "--theta-cut",
        "3.14159",
        "--fps",
        "8",
        "--dpi",
        "110",
    )

    print("Benchmark artifacts:")
    print(f"  panel_png: {panel_png}")
    print(f"  panel_csv: {panel_csv}")
    print(f"  poloidal_png: {poloidal_png}")
    print(f"  poloidal_gif: {poloidal_gif}")
    print(f"  tokamak_3d_gif: {tok3d_gif}")


if __name__ == "__main__":
    main()
