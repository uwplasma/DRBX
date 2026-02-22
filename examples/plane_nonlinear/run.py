from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config, run_simulation
from jaxdrb.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Nonlinear plane example run")
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
        help="Generate panel + RMS figures after the run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    payload.update(
        {
            "snapshot_n": np.asarray(state.n),
            "snapshot_omega": np.asarray(state.omega),
            "snapshot_Te": np.asarray(state.Te),
            "snapshot_vpar_e": np.asarray(state.vpar_e),
            "snapshot_vpar_i": np.asarray(state.vpar_i),
            "snapshot_phi": np.asarray(phi),
            "snapshot_Ti": np.asarray(state.Ti) if state.Ti is not None else None,
            "snapshot_psi": np.asarray(state.psi) if state.psi is not None else None,
            "snapshot_N": np.asarray(state.N) if state.N is not None else None,
        }
    )
    np.savez(out_path, **payload)

    if args.make_figures:
        figdir = (
            (repo_root / args.figdir).resolve()
            if not Path(args.figdir).is_absolute()
            else Path(args.figdir)
        )
        figdir.mkdir(parents=True, exist_ok=True)
        panel = figdir / "nonlinear_panel.png"
        rms = figdir / "nonlinear_rms_timeseries.png"

        subprocess.run(
            [sys.executable, "tools/plot_nonlinear_panel.py", str(out_path), "--out", str(panel)],
            check=True,
            cwd=repo_root,
        )
        subprocess.run(
            [sys.executable, "tools/plot_rms_timeseries.py", str(out_path), "--out", str(rms)],
            check=True,
            cwd=repo_root,
        )


if __name__ == "__main__":
    main()
