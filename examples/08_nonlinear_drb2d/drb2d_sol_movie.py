"""SOL-like DRB2D movie with a closed→open radial setup (LCFS at x = x_s).

This example implements a pragmatic SOL proxy in a periodic 2D box:

- closed side (x < x_s): relax n, Te toward prescribed core profiles,
- open side (x > x_s): stronger relaxation toward SOL profiles + optional sinks.

The goal is to produce blob-like structures that propagate radially outward
and to provide simple, reproducible blob-statistic diagnostics.
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
from matplotlib.ticker import MaxNLocator
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, save_json, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _blob_center_x(n: np.ndarray, x: np.ndarray, mask_open: np.ndarray) -> float:
    n_fluct = n - np.mean(n)
    n_pos = np.maximum(n_fluct, 0.0) * mask_open
    denom = np.sum(n_pos) + 1e-30
    return float(np.sum(n_pos * x) / denom)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmax", type=float, default=160.0)
    p.add_argument("--save-stride", type=int, default=200)
    p.add_argument("--solver", type=str, default="tsit5")
    p.add_argument("--fixed-step", action="store_true", default=False)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=400_000)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-wall", type=float, default=45.0)
    p.add_argument("--out", type=str, default="out_drb2d_sol_movie")

    # LCFS + SOL proxy parameters.
    p.add_argument("--xs-frac", type=float, default=0.6, help="LCFS location x_s / Lx.")
    p.add_argument("--sol-width", type=float, default=0.08, help="LCFS transition width (in Lx units).")
    p.add_argument("--n-core", type=float, default=1.0)
    p.add_argument("--n-sol", type=float, default=0.2)
    p.add_argument("--Te-core", type=float, default=1.0)
    p.add_argument("--Te-sol", type=float, default=0.25)
    p.add_argument("--relax-core", type=float, default=0.08)
    p.add_argument("--relax-open", type=float, default=0.25)
    p.add_argument("--sink-open-n", type=float, default=0.08)
    p.add_argument("--sink-open-Te", type=float, default=0.05)
    p.add_argument("--sink-open-omega", type=float, default=0.02)

    # Turbulence knobs.
    p.add_argument("--omega-n", type=float, default=0.8)
    p.add_argument("--omega-Te", type=float, default=0.25)
    p.add_argument("--curvature", type=float, default=0.7)
    p.add_argument("--Dn", type=float, default=3e-3)
    p.add_argument("--DOmega", type=float, default=3e-3)
    p.add_argument("--DTe", type=float, default=3e-3)
    p.add_argument("--Dn4", type=float, default=6e-5)
    p.add_argument("--DOmega4", type=float, default=6e-5)
    p.add_argument("--DTe4", type=float, default=6e-5)
    p.add_argument("--mu-zonal-omega", type=float, default=0.08)
    p.add_argument("--mu-lin-omega", type=float, default=0.25)
    p.add_argument("--mu-lin-n", type=float, default=0.1)
    p.add_argument("--mu-lin-Te", type=float, default=0.1)

    args = p.parse_args()
    jax.config.update("jax_enable_x64", True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    xs = float(args.xs_frac) * float(grid.Lx)
    params = DRB2DParams(
        omega_n=float(args.omega_n),
        omega_Te=float(args.omega_Te),
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=(float(args.curvature) != 0.0),
        curvature_coeff=float(args.curvature),
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        DTe=float(args.DTe),
        Dn4=float(args.Dn4),
        DOmega4=float(args.DOmega4),
        DTe4=float(args.DTe4),
        mu_zonal_omega=float(args.mu_zonal_omega),
        mu_lin_n=float(args.mu_lin_n),
        mu_lin_omega=float(args.mu_lin_omega),
        mu_lin_Te=float(args.mu_lin_Te),
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
        sol_on=True,
        sol_xs=xs,
        sol_width=float(args.sol_width) * float(grid.Lx),
        sol_n_core=float(args.n_core),
        sol_n_sol=float(args.n_sol),
        sol_Te_core=float(args.Te_core),
        sol_Te_sol=float(args.Te_sol),
        sol_relax_core=float(args.relax_core),
        sol_relax_open=float(args.relax_open),
        sol_sink_open_n=float(args.sink_open_n),
        sol_sink_open_Te=float(args.sink_open_Te),
        sol_sink_open_omega=float(args.sink_open_omega),
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 5e-3
    shape = (grid.nx, grid.ny)
    y0 = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(int(args.seed) + 1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(int(args.seed) + 2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(int(args.seed) + 3), shape),
        Te=amp * jax.random.normal(jax.random.key(int(args.seed) + 4), shape),
    )

    dt = float(args.dt)
    save_stride = int(args.save_stride)
    frame_dt = dt * save_stride
    save_ts = jnp.arange(frame_dt, float(args.tmax) + 1e-12, frame_dt)
    nframes = int(save_ts.size)

    print(
        f"[drb2d-sol-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"frames={nframes} xs={xs:.3f} solver={args.solver}"
    )

    t_start = time.time()
    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(args.tmax),
        dt0=dt,
        save_ts=save_ts,
        solver=str(args.solver),
        adaptive=not bool(args.fixed_step),
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        progress=bool(args.progress),
    )
    wall = time.time() - t_start
    if wall > float(args.max_wall):
        print(f"[drb2d-sol-movie] warning: wall time {wall:.1f}s exceeded max-wall={args.max_wall}s")

    frames = [np.asarray(jax.device_get(sol.ys.n[i])) for i in range(nframes)]
    frames = [np.real(f) if np.iscomplexobj(f) else f for f in frames]
    ts = [float(t) for t in np.asarray(save_ts)]

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
    ax.axvline(xs / float(grid.Lx) * (grid.nx - 1), color="k", lw=1.0, alpha=0.6)
    ax.set_title("DRB2D SOL: normalized n fluctuation")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames_plot[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_plot), interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    # Diagnostics: radial flux + blob center-of-mass velocity.
    x = np.asarray(grid.x)[:, None]
    mask_open = (x > xs).astype(float)
    flux = []
    x_cm = []
    for i in range(nframes):
        yi = sol.ys
        n_i = np.asarray(jax.device_get(yi.n[i]))
        phi_i = np.asarray(jax.device_get(model.phi_from_omega(yi.omega[i], n=yi.n[i])))
        vEx = -np.gradient(phi_i, float(grid.dy), axis=1)
        flux.append(float(np.mean(n_i * vEx * mask_open)))
        x_cm.append(_blob_center_x(n_i, x, mask_open))
    flux = np.asarray(flux)
    x_cm = np.asarray(x_cm)

    # Fit blob radial velocity over the final third.
    tail = slice(int(2 * len(ts) / 3), None)
    coeffs = np.polyfit(np.asarray(ts)[tail], x_cm[tail], deg=1)
    v_blob = float(coeffs[0])

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, x_cm, lw=2, label="blob x_cm")
    ax0.axvline(ts[tail.start], color="k", ls="--", lw=1)
    ax0.set_xlabel("t")
    ax0.set_ylabel("x_cm (open-side, n>0)")
    ax0.set_title("Blob center-of-mass (open side)")
    ax0.grid(alpha=0.3)
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(ts, flux, lw=2)
    ax1.set_xlabel("t")
    ax1.set_title(r"Radial particle flux $\langle n v_{E,x}\rangle$")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(ts, np.gradient(x_cm, np.asarray(ts)), lw=2)
    ax2.set_xlabel("t")
    ax2.set_title("Instantaneous blob velocity")
    ax2.grid(alpha=0.3)

    fig.suptitle("DRB2D SOL diagnostics")
    fig.savefig(out_dir / "sol_diagnostics.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "sol_metrics.json",
        {
            "xs": xs,
            "mean_flux_tail": float(np.mean(flux[tail])),
            "blob_velocity_tail": v_blob,
        },
    )
    print(f"[drb2d-sol-movie] wrote {gif_path}")


if __name__ == "__main__":
    main()

