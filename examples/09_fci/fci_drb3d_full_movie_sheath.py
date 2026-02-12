"""Movie: FCI DRB3D full model with target/sheath coupling (slab).

This script runs a short open-field-line slab case (Dirichlet plates in z via target-aware
FCI operators) with a Loizu-like linearized sheath model. It saves a GIF of mid-plane Ω.

Compared to the periodic movie, this case includes explicit target/sheath loss channels,
so the dynamics tend to be more strongly damped. The defaults are chosen to still show
clear nonlinear structure while staying stable and fast.
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
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _random_state(key: jax.Array, shape: tuple[int, int, int], amp: float) -> FCIDRB3DFullState:
    k = jax.random.split(key, 5)
    return FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=amp * jax.random.normal(k[4], shape),
    )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="out_fci_drb3d_movie_sheath")
    parser.add_argument("--nx", type=int, default=22)
    parser.add_argument("--ny", type=int, default=22)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--tmax", type=float, default=1.2)
    parser.add_argument("--save-stride", type=int, default=10)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall", type=float, default=45.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = FCISlabGrid.make(
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.15,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    params = FCIDRB3DFullParams(
        omega_n=1.4,
        omega_Te=0.9,
        kappa=0.7,
        alpha=0.35,
        eta_par=0.04,
        Dn=4e-4,
        DOmega=4e-4,
        Dvpar=4e-4,
        DTe=4e-4,
        chi_par=6e-4,
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_particle=0.18,
        sheath_nu_energy=0.10,
        sheath_nu_mom=0.45,
        sheath_gamma_e=3.2,
        sheath_gamma_i=3.0,
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)
    y0 = _random_state(jax.random.key(int(args.seed)), (grid.nz, grid.nx, grid.ny), amp=4e-4)

    dt = float(args.dt)
    nsteps = int(round(float(args.tmax) / dt))
    save_stride = int(args.save_stride)
    save_ts = dt * np.arange(save_stride, nsteps + 1, save_stride)
    print(
        f"[fci-drb3d-sheath-movie] grid=({grid.nx},{grid.ny},{grid.nz}) dt={dt} "
        f"tmax={nsteps * dt:.3f} frames={len(save_ts)} solver={args.solver}"
    )

    t0 = time.time()
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        save_every=save_stride,
        solver=str(args.solver),
        progress=False,
    )
    wall = time.time() - t0
    if wall > float(args.max_wall):
        print(
            f"[fci-drb3d-sheath-movie] warning: wall time {wall:.1f}s exceeded max-wall={args.max_wall}s"
        )

    kz = grid.nz // 2
    omega = np.asarray(jax.device_get(ys.omega))
    frames = [omega[i, kz] for i in range(omega.shape[0])]
    frames_arr = np.stack(frames, axis=0)
    frames_fluct = frames_arr - frames_arr.mean(axis=(1, 2), keepdims=True)
    frame_rms = np.sqrt(np.mean(frames_fluct**2, axis=(1, 2), keepdims=True))
    frames_plot = frames_fluct / (frame_rms + 1e-30)
    vmax = robust_symmetric_vlim(frames_plot, q=0.995)

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
    fig.set_dpi(95)
    im = ax.imshow(
        frames_plot[0].T,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
        interpolation="nearest",
    )
    ax.set_title(r"FCI DRB3D sheath: normalized $\Omega$ at mid-plane")
    ax.set_xticks([])
    ax.set_yticks([])

    ts = [float(t) for t in save_ts]

    def update(i: int):
        im.set_array(frames_plot[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_plot), interval=45, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    print(f"[fci-drb3d-sheath-movie] wrote {gif_path}")


if __name__ == "__main__":
    main()
