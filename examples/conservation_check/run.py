from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config, run_simulation
from jaxdrb.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservation check example")
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    time_cfg = cfg.data.get("time", {})
    if not isinstance(time_cfg, dict):
        time_cfg = {}
    time_cfg = dict(time_cfg)
    time_cfg["save_fields"] = True
    time_cfg["snapshot_fields"] = ["n", "omega", "Te"]
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
    np.savez(out_path, **payload)

    built = build_system_from_config(cfg.data)
    system = built.system
    snaps_n = jnp.asarray(payload["snapshots_n"])
    snaps_omega = jnp.asarray(payload["snapshots_omega"])
    snaps_Te = jnp.asarray(payload["snapshots_Te"])
    zeros = jnp.zeros_like(snaps_n)

    def _energy(n, omega, Te):
        state = DRBSystemState(
            n=n,
            omega=omega,
            vpar_e=zeros[0],
            vpar_i=zeros[0],
            Te=Te,
            Ti=None,
            psi=None,
            N=None,
        )
        return system.energy(state)

    energy = jax.vmap(_energy)(snaps_n, snaps_omega, snaps_Te)
    energy = np.asarray(jax.device_get(energy))
    rel_err = (energy - energy[0]) / energy[0]
    times = np.asarray(payload.get("times", np.arange(len(energy))))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(times, rel_err, color="#1f77b4", lw=2.0)
    ax.set_xlabel("t")
    ax.set_ylabel("relative energy error")
    ax.set_title("Energy Conservation (advection only)")
    ax.grid(True, alpha=0.3)
    figdir = (
        (repo_root / args.figdir).resolve()
        if not Path(args.figdir).is_absolute()
        else Path(args.figdir)
    )
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "energy_error.png", dpi=200)


if __name__ == "__main__":
    main()
