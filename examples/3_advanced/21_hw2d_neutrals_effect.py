"""Nonlinear HW2D testbed with a minimal neutral particle model.

This example demonstrates a simple, togglable plasma–neutral interaction:
  - neutrals N are advected by E×B and diffuse
  - ionization transfers particles from neutrals to plasma density (n)

While this periodic 2D setup is not a full SOL geometry, it is a clean and fast
way to validate coupling terms and test algorithmic performance in JAX.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.neutrals import NeutralParams
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nsteps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--save-stride", type=int, default=20)
    parser.add_argument("--out", type=str, default="out_hw2d_neutrals")
    parser.add_argument("--nu-ion", type=float, default=2.0)
    parser.add_argument("--nu-rec", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    neutrals = NeutralParams(
        enabled=True,
        Dn0=1e-3,
        nu_ion=float(args.nu_ion),
        nu_rec=float(args.nu_rec),
        S0=0.0,
        nu_sink=0.0,
    )
    params = HW2DParams(
        kappa=1.0,
        alpha=0.5,
        Dn=2e-4,
        DOmega=2e-4,
        bracket="spectral",
        dealias_on=True,
        neutrals=neutrals,
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    y0 = hw2d_random_ic(key, grid, amp=1e-3, include_neutrals=True)

    dt = float(args.dt)
    nsteps = int(args.nsteps)
    save_stride = int(args.save_stride)
    print(f"[hw2d-neutrals] dt={dt} nsteps={nsteps} save_stride={save_stride}")

    def rhs(t, y):
        return model.rhs(t, y)

    t = 0.0
    y = y0
    ts = []
    nbar = []
    Nbar = []

    nchunks = nsteps // save_stride
    for k in range(nchunks):
        _, y = rk4_scan(y, t0=t, dt=dt, nsteps=save_stride, rhs=rhs)
        t = t + dt * save_stride
        ts.append(t)
        nbar.append(float(jnp.mean(y.n)))
        Nbar.append(float(jnp.mean(y.N)))
        print(
            f"[hw2d-neutrals] chunk {k+1}/{nchunks} t={t:.3f} <n>={nbar[-1]:.3e} <N>={Nbar[-1]:.3e}"
        )

    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "grid": {"nx": grid.nx, "ny": grid.ny, "Lx": grid.Lx, "Ly": grid.Ly},
                "dt": dt,
                "nsteps": nsteps,
                "neutrals": {
                    "Dn0": neutrals.Dn0,
                    "nu_ion": neutrals.nu_ion,
                    "nu_rec": neutrals.nu_rec,
                    "S0": neutrals.S0,
                    "nu_sink": neutrals.nu_sink,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    jnp.savez(out_dir / "means.npz", t=jnp.array(ts), nbar=jnp.array(nbar), Nbar=jnp.array(Nbar))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ts, nbar, label="<n>")
    ax.plot(ts, Nbar, label="<N>")
    ax.set_xlabel("t")
    ax.set_title("Mean densities (plasma + neutrals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "means.png", dpi=200)
    plt.close(fig)

    # Final snapshots.
    phi = model.phi_from_omega(y.omega)
    fields = {"n": y.n, "N": y.N, "phi": phi}
    for name, arr in fields.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    print(f"[hw2d-neutrals] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
