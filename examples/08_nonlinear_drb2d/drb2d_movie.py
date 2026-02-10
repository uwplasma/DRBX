"""Make a short movie of a nonlinear DRB2D run (periodic).

Designed to run in ~5-10 seconds on a laptop. If the wall time exceeds the
limit, the script stops early and still writes a movie from collected frames.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=8.0)
    parser.add_argument("--save-stride", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall", type=float, default=10.0)
    parser.add_argument("--out", type=str, default="out_drb2d_movie")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DParams(
        omega_n=0.2,
        omega_Te=0.1,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        Dn=1e-4,
        DOmega=1e-4,
        DTe=1e-4,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(args.seed)
    shape = (grid.nx, grid.ny)
    n = 1e-3 * jax.random.normal(key, shape)
    omega = 1e-3 * jax.random.normal(jax.random.key(args.seed + 1), shape)
    vpar_e = 1e-3 * jax.random.normal(jax.random.key(args.seed + 2), shape)
    vpar_i = 1e-3 * jax.random.normal(jax.random.key(args.seed + 3), shape)
    Te = 1e-3 * jax.random.normal(jax.random.key(args.seed + 4), shape)
    y = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    dt = float(args.dt)
    nsteps = int(jnp.ceil(float(args.tmax) / dt))
    save_stride = int(args.save_stride)
    nframes = max(1, nsteps // save_stride)

    print(
        f"[drb2d-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"nsteps={nsteps} save_stride={save_stride} frames={nframes}"
    )

    frames_n = []
    ts = []
    t = 0.0
    t_start = time.time()

    def rhs(t_, y_):
        return model.rhs(t_, y_)

    for k in range(nframes):
        _, y = rk4_scan(y, t0=t, dt=dt, nsteps=save_stride, rhs=rhs)
        t = t + dt * save_stride
        frames_n.append(jax.device_get(y.n))
        ts.append(float(t))
        print(f"[drb2d-movie] frame {k + 1}/{nframes} t={t:.2f}")
        if time.time() - t_start > float(args.max_wall):
            print("[drb2d-movie] wall-time limit reached, stopping early")
            break

    frames_arr = np.stack([np.asarray(a) for a in frames_n], axis=0)
    vmax = robust_symmetric_vlim(frames_arr, q=0.995)

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
    fig.set_dpi(95)
    im = ax.imshow(
        frames_arr[0].T,
        origin="lower",
        cmap="cividis",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
        interpolation="nearest",
    )
    ax.set_title("DRB2D: density fluctuation n(x,y,t)")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames_arr[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_arr), interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    assets_dir = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images"
    if assets_dir.exists():
        (assets_dir / "drb2d_turbulence.gif").write_bytes(gif_path.read_bytes())

    print(f"[drb2d-movie] wrote {gif_path}")


if __name__ == "__main__":
    main()
