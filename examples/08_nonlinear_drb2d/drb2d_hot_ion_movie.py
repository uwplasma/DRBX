"""Make a short movie of a nonlinear hot-ion DRB2D run (periodic).

This example is intended as a **second** DRB2D movie case in addition to
`drb2d_movie.py`. It exercises:

- hot-ion state (`Ti`) and the corresponding energy functional,
- the same conservative advection kernel used throughout DRB2D,
- Diffrax fixed-step integration (reproducible + fast).

The default parameters are tuned to show a brief linear onset and a nonlinear
phase on a coarse grid without producing NaNs on typical CPU-only laptops.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise FloatingPointError(f"{name} contains non-finite values; first bad index={bad[0][0]}")


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        type=str,
        default="omega",
        choices=["n", "omega", "Te", "Ti"],
        help="Field to animate. Default is omega (usually most visually turbulent).",
    )
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--tmax", type=float, default=22.0)
    parser.add_argument("--save-stride", type=int, default=16)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--fixed-step", action="store_true", default=True)
    parser.add_argument(
        "--adaptive",
        dest="fixed_step",
        action="store_false",
        help="Use adaptive time stepping (overrides fixed-step default).",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall", type=float, default=35.0)
    parser.add_argument("--out", type=str, default="out_drb2d_hot_ion_movie")

    parser.add_argument(
        "--omega-n", type=float, default=0.9, help="Background density-gradient drive."
    )
    parser.add_argument(
        "--omega-Te", type=float, default=0.25, help="Background Te-gradient drive."
    )
    parser.add_argument("--omega-Ti", type=float, default=0.2, help="Background Ti-gradient drive.")
    parser.add_argument(
        "--tau-i",
        type=float,
        default=1.0,
        help="Ion-to-electron temperature ratio in normalization.",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.6,
        help="Curvature coefficient (0 disables curvature drive).",
    )
    parser.add_argument("--Dn", type=float, default=1.2e-3, help="Laplacian diffusion on n.")
    parser.add_argument(
        "--DOmega", type=float, default=1.2e-3, help="Laplacian diffusion on omega."
    )
    parser.add_argument("--DTe", type=float, default=1.2e-3, help="Laplacian diffusion on Te.")
    parser.add_argument("--DTi", type=float, default=1.2e-3, help="Laplacian diffusion on Ti.")
    parser.add_argument("--Dn4", type=float, default=4e-5, help="Hyperdiffusion (-Dn4*∇^4) on n.")
    parser.add_argument(
        "--DOmega4", type=float, default=4e-5, help="Hyperdiffusion (-DOmega4*∇^4) on omega."
    )
    parser.add_argument(
        "--DTe4", type=float, default=4e-5, help="Hyperdiffusion (-DTe4*∇^4) on Te."
    )
    parser.add_argument(
        "--DTi4", type=float, default=4e-5, help="Hyperdiffusion (-DTi4*∇^4) on Ti."
    )
    parser.add_argument(
        "--mu-zonal-omega",
        type=float,
        default=0.1,
        help="Drag coefficient on zonal (ky=0) omega component.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DHotIonParams(
        omega_n=float(args.omega_n),
        omega_Te=float(args.omega_Te),
        omega_Ti=float(args.omega_Ti),
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=float(args.tau_i),
        curvature_on=(float(args.curvature) != 0.0),
        curvature_coeff=float(args.curvature),
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        DTe=float(args.DTe),
        DTi=float(args.DTi),
        Dn4=float(args.Dn4),
        DOmega4=float(args.DOmega4),
        DTe4=float(args.DTe4),
        DTi4=float(args.DTi4),
        mu_zonal_omega=float(args.mu_zonal_omega),
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DHotIonModel(params=params, grid=grid)

    key = jax.random.key(args.seed)
    shape = (grid.nx, grid.ny)
    amp = 6e-3
    n = amp * jax.random.normal(key, shape)
    omega = amp * jax.random.normal(jax.random.key(args.seed + 1), shape)
    vpar_e = amp * jax.random.normal(jax.random.key(args.seed + 2), shape)
    vpar_i = amp * jax.random.normal(jax.random.key(args.seed + 3), shape)
    Te = amp * jax.random.normal(jax.random.key(args.seed + 4), shape)
    Ti = amp * jax.random.normal(jax.random.key(args.seed + 5), shape)
    y = DRB2DHotIonState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te, Ti=Ti)

    dt = float(args.dt)
    save_stride = int(args.save_stride)
    frame_dt = dt * save_stride
    save_ts = jnp.arange(frame_dt, float(args.tmax) + 1e-12, frame_dt)
    nframes = int(save_ts.size)

    print(
        f"[drb2d-hot-ion-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"save_stride={save_stride} frames={nframes} solver={args.solver} "
        f"adaptive={not args.fixed_step} field={args.field}"
    )

    t_start = time.time()
    sol = model.diffeqsolve(
        y0=y,
        t0=0.0,
        t1=float(args.tmax),
        dt0=dt,
        save_ts=save_ts,
        solver=str(args.solver),
        adaptive=not args.fixed_step,
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=300_000,
        progress=bool(args.progress),
    )
    wall = time.time() - t_start
    if wall > float(args.max_wall):
        print(
            f"[drb2d-hot-ion-movie] warning: wall time {wall:.1f}s exceeded max-wall={args.max_wall}s"
        )

    field = args.field
    if field == "n":
        frames = [jax.device_get(sol.ys.n[i]) for i in range(nframes)]
    elif field == "omega":
        frames = [jax.device_get(sol.ys.omega[i]) for i in range(nframes)]
    elif field == "Te":
        frames = [jax.device_get(sol.ys.Te[i]) for i in range(nframes)]
    else:
        frames = [jax.device_get(sol.ys.Ti[i]) for i in range(nframes)]

    ts = [float(t) for t in np.asarray(save_ts)]
    frames_arr = np.stack([np.asarray(a) for a in frames], axis=0)
    _assert_finite("frames", frames_arr)

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
    ax.set_title(f"DRB2D (hot-ion): normalized {field} fluctuation ({field}-<{field}>)/rms")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames_plot[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_arr), interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    # Static summary panel (final snapshots + time series).
    E = np.asarray(
        jax.device_get(
            jax.jit(lambda ys: jax.vmap(model.energy)(ys))(sol.ys),
        )
    )
    _assert_finite("E(t)", E)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, E, label="E", lw=2)
    ax0.plot(ts, np.asarray(frame_rms[:, 0, 0]) ** 2, label=r"$\mathrm{rms}^2$", lw=2)
    ax0.set_yscale("log")
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax0.set_xlabel("t")
    ax0.set_title("Diagnostics")
    ax0.legend()

    phi_f = np.asarray(jax.device_get(model.phi_from_omega(sol.ys.omega[-1], n=sol.ys.n[-1])))
    for ax, (name, arr) in zip(
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])],
        [("n", sol.ys.n[-1]), ("phi", phi_f), ("omega", sol.ys.omega[-1])],
    ):
        arr_np = np.asarray(jax.device_get(arr))
        _assert_finite(name, arr_np)
        vmax = robust_symmetric_vlim(arr_np, q=0.995)
        im = ax.imshow(
            arr_np.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    ax1 = fig.add_subplot(gs[1, 2])
    ax1.plot(ts, np.gradient(np.log(np.maximum(E, 1e-30)), np.asarray(ts)), lw=2)
    ax1.set_xlabel("t")
    ax1.set_title(r"Instantaneous $\gamma(t)=d\ln E/dt$")

    fig.suptitle("DRB2D hot-ion nonlinear run (movie saved)")
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[drb2d-hot-ion-movie] wrote {gif_path}")
    print(
        "[drb2d-hot-ion-movie] to update README assets, run: "
        "python docs/assets/scripts/make_drb2d_readme_assets.py"
    )


if __name__ == "__main__":
    main()
