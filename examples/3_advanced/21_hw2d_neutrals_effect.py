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

from jaxdrb.analysis.plotting import set_mpl_style
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
    parser.add_argument("--nu-ion", type=float, default=0.2)
    parser.add_argument("--nu-rec", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    neutrals = NeutralParams(
        enabled=True,
        Dn0=1e-3,
        nu_ion=float(args.nu_ion),
        nu_rec=float(args.nu_rec),
        n_background=1.0,
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
    fields = {"n": (y.n, "RdBu_r"), "N": (y.N, "viridis"), "phi": (phi, "RdBu_r")}
    for name, (arr, cmap) in fields.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    # Summary panel.
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, nbar, label=r"$\langle n \rangle$", lw=2)
    ax0.plot(ts, Nbar, label=r"$\langle N \rangle$", lw=2)
    ax0.set_xlabel("t")
    ax0.set_title("Mean particle content")
    ax0.legend()

    for ax, (name, arr, cmap) in zip(
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])],
        [("n", y.n, "RdBu_r"), ("N", y.N, "viridis"), ("phi", phi, "RdBu_r")],
    ):
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(gs[1, 2])
    ax1.plot(ts, (jnp.array(nbar) + jnp.array(Nbar)), lw=2)
    ax1.set_xlabel("t")
    ax1.set_title(r"$\langle n + N \rangle$")

    fig.suptitle("Nonlinear HW2D with minimal neutrals (periodic)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[hw2d-neutrals] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
