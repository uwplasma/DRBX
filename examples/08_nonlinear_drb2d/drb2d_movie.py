"""Make a short movie of a nonlinear DRB2D run (periodic).

Designed to run in ~20-30 seconds on a laptop. If the wall time exceeds the
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
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        type=str,
        default="omega",
        choices=["n", "omega", "Te"],
        help="Field to animate. Default is omega (usually most visually turbulent).",
    )
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tmax", type=float, default=60.0)
    parser.add_argument("--save-stride", type=int, default=15)
    parser.add_argument("--solver", type=str, default="dopri8")
    parser.add_argument("--fixed-step", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wall", type=float, default=30.0)
    parser.add_argument("--out", type=str, default="out_drb2d_movie")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DParams(
        # A curvature-driven case that rapidly develops nonlinear dynamics.
        omega_n=0.8,
        omega_Te=0.2,
        # Keep kpar=0 for a real-valued nonlinear turbulence movie (no Fourier-parallel coupling).
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.35,
        # Dissipation to control the cascade on coarse grids.
        Dn=8e-4,
        DOmega=8e-4,
        DTe=8e-4,
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
    amp = 1e-2
    n = amp * jax.random.normal(key, shape)
    omega = amp * jax.random.normal(jax.random.key(args.seed + 1), shape)
    vpar_e = amp * jax.random.normal(jax.random.key(args.seed + 2), shape)
    vpar_i = amp * jax.random.normal(jax.random.key(args.seed + 3), shape)
    Te = amp * jax.random.normal(jax.random.key(args.seed + 4), shape)
    y = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    dt = float(args.dt)
    save_stride = int(args.save_stride)
    frame_dt = dt * save_stride
    save_ts = jnp.arange(frame_dt, float(args.tmax) + 1e-12, frame_dt)
    nframes = int(save_ts.size)

    print(
        f"[drb2d-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
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
        solver=args.solver,
        adaptive=not args.fixed_step,
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=300_000,
        progress=bool(args.progress),
    )
    wall = time.time() - t_start
    if wall > float(args.max_wall):
        print(f"[drb2d-movie] warning: wall time {wall:.1f}s exceeded max-wall={args.max_wall}s")

    if args.field == "n":
        frames = [jax.device_get(sol.ys.n[i]) for i in range(nframes)]
    elif args.field == "omega":
        frames = [jax.device_get(sol.ys.omega[i]) for i in range(nframes)]
    else:
        frames = [jax.device_get(sol.ys.Te[i]) for i in range(nframes)]
    ts = [float(t) for t in np.asarray(save_ts)]
    rms = [float(np.sqrt(np.mean(np.asarray(fr) ** 2))) for fr in frames]
    for k in range(nframes):
        if k == 0 or (k + 1) % max(1, nframes // 10) == 0 or k == nframes - 1:
            print(
                f"[drb2d-movie] frame {k + 1}/{nframes} t={ts[k]:.2f} rms({args.field})={rms[k]:.3e}"
            )

    frames_arr = np.stack([np.asarray(a) for a in frames], axis=0)
    # Plot normalized fluctuations so the early-time linear phase is visible even
    # when the late-time nonlinear amplitude is much larger.
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
    ax.set_title(f"DRB2D: normalized {args.field} fluctuation ({args.field}-<{args.field}>)/rms")
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

    # Save a static summary panel (final-time snapshots + a simple timeseries).
    E = np.asarray(
        jax.device_get(
            jax.jit(lambda ys: jax.vmap(model.energy)(ys))(sol.ys),
        )
    )
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, E, label="E", lw=2)
    ax0.plot(ts, np.asarray(rms) ** 2, label=r"$\mathrm{rms}(n)^2$", lw=2)
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

    fig.suptitle("DRB2D nonlinear run (movie saved)")
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[drb2d-movie] wrote {gif_path}")
    print(
        "[drb2d-movie] to update README assets, run: "
        "python docs/assets/scripts/make_drb2d_readme_assets.py"
    )


if __name__ == "__main__":
    main()
