"""Make a short movie of a nonlinear HW2D run (periodic).

This script is designed to be:
  - fast enough to run end-to-end locally (and in a CI-like environment),
  - deterministic (fixed time step and seed),
  - pedagogic (clear diagnostics + a movie + static summary figures).

It saves:
  - `movie.gif`: a GIF showing n(x,y,t) snapshots
  - `panel.png`: final snapshots + diagnostics
  - `timeseries.npz`: E(t), Z(t)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    jax.config.update("jax_enable_x64", False)
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--tmax", type=float, default=30.0)
    parser.add_argument("--save-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_hw2d_movie")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=1.0,
        alpha=0.5,
        Dn=2e-4,
        DOmega=2e-4,
        bracket="spectral",
        poisson="spectral",
        dealias_on=True,
    )
    model = HW2DModel(params=params, grid=grid)

    y = hw2d_random_ic(jax.random.key(args.seed), grid, amp=1e-3, include_neutrals=False)
    dt = float(args.dt)
    nsteps = int(jnp.ceil(float(args.tmax) / dt))
    save_stride = int(args.save_stride)
    nframes = max(1, nsteps // save_stride)

    print(
        f"[hw2d-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"nsteps={nsteps} save_stride={save_stride} frames={nframes}"
    )

    def rhs(t, y_):
        return model.rhs(t, y_)

    frames_n = []
    ts = []
    Es = []
    Zs = []

    t = 0.0
    for k in range(nframes):
        _, y = rk4_scan(y, t0=t, dt=dt, nsteps=save_stride, rhs=rhs)
        t = t + dt * save_stride
        diag = model.diagnostics(y)
        ts.append(float(t))
        Es.append(float(diag["E"]))
        Zs.append(float(diag["Z"]))
        frames_n.append(jax.device_get(y.n))
        print(f"[hw2d-movie] frame {k+1}/{nframes} t={t:.3f} E={Es[-1]:.3e} Z={Zs[-1]:.3e}")

    jnp.savez(out_dir / "timeseries.npz", t=jnp.array(ts), E=jnp.array(Es), Z=jnp.array(Zs))

    # Build movie (GIF) using fixed color limits for readability.
    arr0 = frames_n[0]
    vmax = float(jnp.quantile(jnp.abs(arr0), 0.995) + 1e-12)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(arr0.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, animated=True)
    ax.set_title("n(x,y,t)")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("n")

    def update(i):
        im.set_array(frames_n[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_n), interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=20))
    plt.close(fig)

    # Final summary panel.
    phi = model.phi_from_omega(y.omega)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, Es, label="E", lw=2)
    ax0.plot(ts, Zs, label="Z", lw=2)
    ax0.set_yscale("log")
    ax0.set_xlabel("t")
    ax0.set_title("Diagnostics")
    ax0.legend()

    for ax, (name, arr) in zip(
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])],
        [("n", y.n), ("phi", phi), ("omega", y.omega)],
    ):
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(gs[1, 2])
    ax1.plot(ts, jnp.gradient(jnp.log(jnp.maximum(jnp.array(Es), 1e-30)), jnp.array(ts)), lw=2)
    ax1.set_xlabel("t")
    ax1.set_title(r"Instantaneous $\gamma(t)=d\ln E/dt$")

    fig.suptitle("HW2D nonlinear run (movie saved)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[hw2d-movie] wrote {gif_path} and summary figures to {out_dir}")


if __name__ == "__main__":
    main()
