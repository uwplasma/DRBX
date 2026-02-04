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
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults chosen to produce a longer-time movie without being slow/heavy.
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    # dt is the main stability knob for long-time runs.
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=80.0)
    parser.add_argument("--save-stride", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_hw2d_movie")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=1.0,
        alpha=1.0,
        Dn=1e-3,
        DOmega=1e-3,
        # Stabilize the enstrophy cascade at high k (Camargo et al. 1995 use hyperdiffusion).
        nu4_n=1e-6,
        nu4_omega=1e-6,
        bracket="arakawa",
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
        if not jnp.isfinite(diag["E"]) or not jnp.isfinite(diag["Z"]):
            raise FloatingPointError(
                f"Non-finite diagnostics at frame {k + 1}/{nframes}, t={t:.3f}: {diag}"
            )
        ts.append(float(t))
        Es.append(float(diag["E"]))
        Zs.append(float(diag["Z"]))
        frames_n.append(jax.device_get(y.n))
        print(f"[hw2d-movie] frame {k + 1}/{nframes} t={t:.3f} E={Es[-1]:.3e} Z={Zs[-1]:.3e}")

    jnp.savez(out_dir / "timeseries.npz", t=jnp.array(ts), E=jnp.array(Es), Z=jnp.array(Zs))

    # Build movie (GIF) using fixed color limits for readability.
    frames_arr = np.stack([np.asarray(a) for a in frames_n], axis=0)
    vmax = robust_symmetric_vlim(frames_arr, q=0.995)
    arr0 = frames_arr[0]

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
    fig.set_dpi(95)
    im = ax.imshow(
        arr0.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
        interpolation="nearest",
    )
    ax.set_title("HW2D: density fluctuation n(x,y,t)")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames_arr[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_n), interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
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
        arr_np = np.asarray(arr)
        vmax = robust_symmetric_vlim(arr_np, q=0.995)
        im = ax.imshow(
            arr_np.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
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
