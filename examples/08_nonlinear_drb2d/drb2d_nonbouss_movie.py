"""Make a short movie of a nonlinear DRB2D run with non-Boussinesq polarization.

This example exists specifically to stress the variable-coefficient SPD solve

    -∇·(n ∇φ) = Ω,

which is one of the key numerical bottlenecks/instability sources for long-time
nonlinear runs.

The parameter defaults are conservative: they are tuned to run quickly and avoid NaNs
on a laptop, while still showing a nonlinear phase.
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

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--x64",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable float64 in JAX (slower but more robust).",
    )
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmax", type=float, default=60.0)
    p.add_argument("--save-stride", type=int, default=200)
    p.add_argument("--solver", type=str, default="tsit5")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-wall", type=float, default=45.0)
    p.add_argument("--out", type=str, default="out_drb2d_nonbouss_movie")

    p.add_argument("--omega-n", type=float, default=0.6)
    p.add_argument("--omega-Te", type=float, default=0.2)
    p.add_argument("--curvature", type=float, default=0.6)

    p.add_argument("--Dn", type=float, default=4e-3)
    p.add_argument("--DOmega", type=float, default=4e-3)
    p.add_argument("--DTe", type=float, default=4e-3)
    p.add_argument("--Dn4", type=float, default=8e-5)
    p.add_argument("--DOmega4", type=float, default=8e-5)
    p.add_argument("--DTe4", type=float, default=8e-5)

    p.add_argument("--mu-zonal-omega", type=float, default=0.12)
    p.add_argument("--mu-lin-omega", type=float, default=0.45)
    p.add_argument("--mu-lin-n", type=float, default=0.18)
    p.add_argument("--mu-lin-Te", type=float, default=0.18)

    p.add_argument(
        "--n0",
        type=float,
        default=1.0,
        help="Background density level used in non-Boussinesq polarization coefficient.",
    )
    p.add_argument(
        "--n0-min",
        type=float,
        default=0.2,
        help="Hard floor on the effective polarization coefficient n_eff = max(n0 + n, n0_min).",
    )

    args = p.parse_args()
    jax.config.update("jax_enable_x64", bool(args.x64))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
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
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=float(args.n0),
        n0_min=float(args.n0_min),
        bracket="arakawa",
        poisson="cg_fd",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 4e-3
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
        f"[drb2d-nonbouss-movie] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"save_stride={save_stride} frames={nframes} solver={args.solver} x64={bool(args.x64)}"
    )

    t_start = time.time()
    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(args.tmax),
        dt0=dt,
        save_ts=save_ts,
        solver=str(args.solver),
        adaptive=True,
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        progress=bool(args.progress),
    )
    wall = time.time() - t_start
    if wall > float(args.max_wall):
        print(
            f"[drb2d-nonbouss-movie] warning: wall time {wall:.1f}s exceeded "
            f"max-wall={args.max_wall}s"
        )

    frames = [np.asarray(jax.device_get(sol.ys.omega[i])) for i in range(nframes)]
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
    ax.set_title("DRB2D non-Boussinesq: normalized $\\Omega$ fluctuation")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames_plot[i].T)
        ax.set_xlabel(f"t = {ts[i]:.2f}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=nframes, interval=40, blit=True)
    gif_path = out_dir / "movie.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)

    E = np.asarray(jax.device_get(jax.vmap(model.energy)(sol.ys)))
    E = np.real(E)
    dlogE_dt = np.gradient(np.log(np.maximum(E, 1e-30)), np.asarray(ts))
    tail = slice(int(2 * len(dlogE_dt) / 3), None)
    print(
        f"[drb2d-nonbouss-movie] mean dlnE/dt over final third: {float(np.mean(dlogE_dt[tail])):.3e}"
    )

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(ts, E, label="E", lw=2)
    ax0.set_yscale("log")
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax0.set_xlabel("t")
    ax0.set_title("Diagnostics")
    ax0.legend()

    phi_f = np.asarray(jax.device_get(model.phi_from_omega(sol.ys.omega[-1], n=sol.ys.n[-1])))
    phi_f = np.real(phi_f) if np.iscomplexobj(phi_f) else phi_f
    for ax, (name, arr) in zip(
        [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1])],
        [("n", sol.ys.n[-1]), ("phi", phi_f), ("omega", sol.ys.omega[-1])],
    ):
        arr_np = np.asarray(jax.device_get(arr))
        if np.iscomplexobj(arr_np):
            arr_np = np.real(arr_np)
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

    fig.suptitle("DRB2D non-Boussinesq nonlinear run (movie saved)")
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[drb2d-nonbouss-movie] wrote {gif_path}")


if __name__ == "__main__":
    main()
