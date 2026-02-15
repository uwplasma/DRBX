"""Movie: FCI DRB3D full model with open/closed (SOL/core) sheath mask.

This slab proxy uses an open-field-line FCI grid but gates the sheath loss terms
with a smooth radial mask to mimic an LCFS (closed core, open SOL). The goal is to
seed interchange activity and look for intermittent blob-like transport in 3D.
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

from jaxdrb.analysis.plotting import robust_symmetric_vlim, save_animation_gif, set_mpl_style
from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _smooth_lcfs_mask(x: jnp.ndarray, *, x_lcfs: float, width: float) -> jnp.ndarray:
    """Smooth 0->1 transition for SOL mask centered at x_lcfs."""
    return 0.5 * (1.0 + jnp.tanh((x - x_lcfs) / max(width, 1e-6)))


def _blob_profile(
    x: jnp.ndarray, y: jnp.ndarray, *, x0: float, y0: float, sigma: float
) -> jnp.ndarray:
    r2 = ((x - x0) ** 2 + (y - y0) ** 2) / max(sigma, 1e-8) ** 2
    return jnp.exp(-r2)


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
    parser.add_argument("--out", type=str, default="out_fci_drb3d_movie_open_closed")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=14)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--tmax", type=float, default=1.2)
    parser.add_argument("--save-stride", type=int, default=10)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall", type=float, default=60.0)
    parser.add_argument("--lcfs", type=float, default=3.0)
    parser.add_argument("--lcfs-width", type=float, default=0.35)
    parser.add_argument("--blob-amp", type=float, default=0.25)
    parser.add_argument("--blob-sigma", type=float, default=0.45)
    parser.add_argument("--blob-x", type=float, default=3.1)
    parser.add_argument("--blob-y", type=float, default=3.1)
    parser.add_argument("--noise-amp", type=float, default=2e-4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Lx = 2.0 * jnp.pi
    Ly = 2.0 * jnp.pi
    Lz = 6.0

    grid0 = FCISlabGrid.make(
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        Lx=float(Lx),
        Ly=float(Ly),
        Lz=float(Lz),
        Bx=0.0,
        By=0.15,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )

    x = grid0.x0 + grid0.dx * (jnp.arange(grid0.nx) + 0.5)
    y = grid0.y0 + grid0.dy * (jnp.arange(grid0.ny) + 0.5)
    xg, yg = jnp.meshgrid(x, y, indexing="ij")

    sol_mask = _smooth_lcfs_mask(x, x_lcfs=float(args.lcfs), width=float(args.lcfs_width))
    sol_mask = jnp.broadcast_to(sol_mask[None, :, None], (grid0.nz, grid0.nx, grid0.ny))

    sheath_mask = jnp.asarray(grid0.sheath_mask) * sol_mask
    sheath_sign = jnp.asarray(grid0.sheath_sign) * sol_mask

    grid = FCISlabGrid.from_maps(
        x0=grid0.x0,
        y0=grid0.y0,
        dx=grid0.dx,
        dy=grid0.dy,
        nx=grid0.nx,
        ny=grid0.ny,
        l=grid0.l,
        map_fwd=grid0.map_fwd,
        map_bwd=grid0.map_bwd,
        open_field_line=grid0.open_field_line,
        cell_centered=grid0.cell_centered,
        Bx=grid0.Bx,
        By=grid0.By,
        Bz=grid0.Bz,
        sheath_mask=sheath_mask,
        sheath_sign=sheath_sign,
    )

    params = FCIDRB3DFullParams(
        omega_n=1.6,
        omega_Te=1.1,
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

    blob = _blob_profile(
        xg, yg, x0=float(args.blob_x), y0=float(args.blob_y), sigma=float(args.blob_sigma)
    )
    n0 = 1.0 + float(args.blob_amp) * blob
    Te0 = 1.0 + 0.8 * (n0 - 1.0)
    n0 = jnp.broadcast_to(n0[None, ...], (grid.nz, grid.nx, grid.ny))
    Te0 = jnp.broadcast_to(Te0[None, ...], (grid.nz, grid.nx, grid.ny))

    noise = _random_state(
        jax.random.key(int(args.seed)), (grid.nz, grid.nx, grid.ny), amp=float(args.noise_amp)
    )
    y0 = FCIDRB3DFullState(
        n=n0 + noise.n,
        omega=noise.omega,
        vpar_e=noise.vpar_e,
        vpar_i=noise.vpar_i,
        Te=Te0 + noise.Te,
    )

    dt = float(args.dt)
    nsteps = int(round(float(args.tmax) / dt))
    save_stride = int(args.save_stride)
    save_ts = dt * np.arange(save_stride, nsteps + 1, save_stride)
    print(
        f"[fci-drb3d-open-closed] grid=({grid.nx},{grid.ny},{grid.nz}) dt={dt} "
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
        progress=bool(args.progress),
    )
    wall = time.time() - t0
    if wall > float(args.max_wall):
        print(
            f"[fci-drb3d-open-closed] warning: wall time {wall:.1f}s exceeded max-wall={args.max_wall}s"
        )

    kz = grid.nz // 2
    n_series = np.asarray(jax.device_get(ys.n))
    frames = [n_series[i, kz] - 1.0 for i in range(n_series.shape[0])]
    frames_arr = np.stack(frames, axis=0)
    vmax = robust_symmetric_vlim(frames_arr, q=0.995)

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
    fig.set_dpi(95)
    im = ax.imshow(
        frames_arr[0].T,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
        interpolation="nearest",
    )
    ax.set_title(r"FCI DRB3D open/closed: $n'$ at mid-plane")
    ax.set_xticks([])
    ax.set_yticks([])

    ts = [float(t) for t in save_ts]

    def update(i: int):
        im.set_array(frames_arr[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_arr), interval=45, blit=True)
    gif_path = out_dir / "movie.gif"
    if args.gif:
        save_animation_gif(ani, gif_path, fps=12, dpi=95)
    plt.close(fig)

    if args.gif:
        print(f"[fci-drb3d-open-closed] wrote {gif_path}")


if __name__ == "__main__":
    main()
