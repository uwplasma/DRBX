from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config, run_simulation
from jaxdrb.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-field-line example run")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("input.toml")),
        help="Path to input TOML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("output.npz")),
        help="Output .npz path",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default=str(Path("docs/figures")),
        help="Figure output directory",
    )
    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate poloidal + 3D slice figures after the run.",
    )
    parser.add_argument(
        "--make-movies",
        action="store_true",
        help="Generate a midplane movie GIF after the run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.make_figures or args.make_movies:
        time_cfg = cfg.data.get("time", {})
        if not isinstance(time_cfg, dict):
            time_cfg = {}
        time_cfg = dict(time_cfg)
        time_cfg["save_fields"] = True
        time_cfg["snapshot_fields"] = ["n"]
        time_cfg["diag_phi_use_guess"] = False
        cfg.data["time"] = time_cfg
    result = run_simulation(cfg.data, as_numpy=True)

    repo_root = Path(__file__).resolve().parents[2]
    out_path = (
        (repo_root / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result.diagnostics)
    if "times" not in payload:
        payload["times"] = result.times
    if "t" not in payload:
        payload["t"] = result.times

    built = build_system_from_config(cfg.data)
    state = result.final_state
    n_phys = built.system._phys_n(state.n)
    phi = built.system._phi_from_omega(state.omega, n=n_phys)
    snapshots_n = payload.get("snapshots_n", None)
    snapshots_omega = payload.get("snapshots_omega", None)
    snapshots_Te = payload.get("snapshots_Te", None)
    snapshots_phi = payload.get("snapshots_phi", None)

    def _last_finite(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim < 1:
            return arr
        finite = np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)
        if not np.any(finite):
            return arr[-1]
        idx = int(np.where(finite)[0][-1])
        return arr[idx]

    payload.update(
        {
            "snapshot_n": np.asarray(_last_finite(snapshots_n))
            if snapshots_n is not None
            else np.asarray(state.n),
            "snapshot_omega": np.asarray(_last_finite(snapshots_omega))
            if snapshots_omega is not None
            else np.asarray(state.omega),
            "snapshot_Te": np.asarray(_last_finite(snapshots_Te))
            if snapshots_Te is not None
            else np.asarray(state.Te),
            "snapshot_vpar_e": np.asarray(state.vpar_e),
            "snapshot_vpar_i": np.asarray(state.vpar_i),
            "snapshot_phi": np.asarray(_last_finite(snapshots_phi))
            if snapshots_phi is not None
            else np.asarray(phi),
            "snapshot_Ti": np.asarray(state.Ti) if state.Ti is not None else None,
            "snapshot_psi": np.asarray(state.psi) if state.psi is not None else None,
            "snapshot_N": np.asarray(state.N) if state.N is not None else None,
        }
    )
    np.savez(out_path, **payload)

    figdir = (
        (repo_root / args.figdir).resolve()
        if not Path(args.figdir).is_absolute()
        else Path(args.figdir)
    )
    figdir.mkdir(parents=True, exist_ok=True)

    if args.make_figures:
        poloidal_eq = figdir / "open_field_poloidal_eq.png"
        poloidal_fluct = figdir / "open_field_poloidal_fluct.png"
        sol_cfg = cfg.data.get("physics", {})
        sep = float(sol_cfg.get("sol_xs", 0.7))
        subprocess.run(
            [
                sys.executable,
                "tools/plot_poloidal_plane.py",
                str(out_path),
                "--config",
                str(Path(args.config).resolve()),
                "--field",
                "n",
                "--out",
                str(poloidal_eq),
                "--separatrix",
                str(sep),
                "--equilibrium",
                "only",
                "--overlay-mask",
                "--interp-grid",
                "260",
            ],
            check=True,
            cwd=repo_root,
        )
        subprocess.run(
            [
                sys.executable,
                "tools/plot_poloidal_plane.py",
                str(out_path),
                "--config",
                str(Path(args.config).resolve()),
                "--field",
                "n",
                "--out",
                str(poloidal_fluct),
                "--separatrix",
                str(sep),
                "--equilibrium",
                "none",
                "--overlay-mask",
                "--fluct",
                "zonal",
                "--lowpass",
                "0.25",
                "--field-scale",
                "8.0",
                "--cmap",
                "coolwarm",
                "--symmetric",
                "--interp-grid",
                "260",
            ],
            check=True,
            cwd=repo_root,
        )
    if args.make_movies:
        movie_path = figdir / "open_field_movie.gif"
        subprocess.run(
            [
                sys.executable,
                "tools/make_poloidal_movie.py",
                str(out_path),
                "--config",
                str(Path(args.config).resolve()),
                "--field",
                "snapshots_n",
                "--out",
                str(movie_path),
                "--stride",
                "3",
                "--fluct",
                "zonal",
                "--lowpass",
                "0.25",
                "--skip-fraction",
                "0.35",
                "--symmetric",
                "--range-tail",
                "--field-scale",
                "6.0",
                "--range-scale",
                "0.6",
                "--interp-grid",
                "320",
                "--separatrix",
                str(sep),
            ],
            check=True,
            cwd=repo_root,
        )


if __name__ == "__main__":
    main()
