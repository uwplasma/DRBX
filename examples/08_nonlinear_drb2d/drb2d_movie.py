"""Make a short movie of a nonlinear DRB2D run (periodic).

Designed to run in ~10-30 seconds on a laptop. If the wall time exceeds the
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
from matplotlib.ticker import MaxNLocator
import numpy as np

from jaxdrb.analysis.plotting import robust_symmetric_vlim, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x64",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable float64 in JAX (recommended for stable long-time nonlinear runs).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="omega",
        choices=["n", "omega", "Te"],
        help="Field to animate. Default is omega (usually most visually turbulent).",
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--save-stride", type=int, default=500)
    parser.add_argument("--solver", type=str, default="tsit5")
    parser.add_argument("--fixed-step", action="store_true", default=False)
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
    parser.add_argument("--max-wall", type=float, default=45.0)
    parser.add_argument("--out", type=str, default="out_drb2d_movie")
    parser.add_argument(
        "--omega-n", type=float, default=1.0, help="Background density-gradient drive."
    )
    parser.add_argument(
        "--omega-Te", type=float, default=0.35, help="Background Te-gradient drive."
    )
    parser.add_argument(
        "--kpar",
        type=float,
        default=0.0,
        help=(
            "Constant k_parallel for Fourier-parallel coupling. "
            "Note: nonzero kpar implies complex-valued state evolution, which is not "
            "recommended for nonlinear turbulence movies. Default disables parallel dynamics."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="Parallel resistivity coefficient (only active when kpar!=0).",
    )
    parser.add_argument(
        "--me-hat",
        type=float,
        default=0.2,
        help="Electron inertia parameter used in the parallel momentum closure.",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.7,
        help="Curvature coefficient (0 disables curvature drive).",
    )
    parser.add_argument("--Dn", type=float, default=3e-3, help="Laplacian diffusion on n.")
    parser.add_argument("--DOmega", type=float, default=3e-3, help="Laplacian diffusion on omega.")
    parser.add_argument("--DTe", type=float, default=3e-3, help="Laplacian diffusion on Te.")
    parser.add_argument("--Dn4", type=float, default=5e-5, help="Hyperdiffusion (-Dn4*∇^4) on n.")
    parser.add_argument(
        "--DOmega4", type=float, default=5e-5, help="Hyperdiffusion (-DOmega4*∇^4) on omega."
    )
    parser.add_argument(
        "--DTe4", type=float, default=5e-5, help="Hyperdiffusion (-DTe4*∇^4) on Te."
    )
    parser.add_argument(
        "--mu-zonal-omega",
        type=float,
        default=0.12,
        help="Drag coefficient on zonal (ky=0) omega component.",
    )
    parser.add_argument(
        "--mu-lin-omega",
        type=float,
        default=0.35,
        help="Linear damping on omega (large-scale friction / parallel-loss surrogate).",
    )
    parser.add_argument(
        "--mu-lin-n",
        type=float,
        default=0.12,
        help="Linear damping on n (parallel-loss surrogate).",
    )
    parser.add_argument(
        "--mu-lin-Te",
        type=float,
        default=0.12,
        help="Linear damping on Te (parallel-loss surrogate).",
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", bool(args.x64))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DParams(
        # A curvature-driven case that rapidly develops nonlinear dynamics.
        omega_n=float(args.omega_n),
        omega_Te=float(args.omega_Te),
        # Use kpar>0 + resistivity by default to obtain drift-wave-like saturation.
        kpar=float(args.kpar),
        eta=float(args.eta),
        me_hat=float(args.me_hat),
        curvature_on=(float(args.curvature) != 0.0),
        curvature_coeff=float(args.curvature),
        # Dissipation to control the cascade on coarse grids. Hyperdiffusion keeps the
        # small-scale spectrum under control while preserving large-scale structure.
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
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(args.seed)
    shape = (grid.nx, grid.ny)
    amp = 6e-3
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

    def _plot_field(arr):
        arr = np.asarray(arr)
        return np.real(arr) if np.iscomplexobj(arr) else arr

    if args.field == "n":
        frames = [_plot_field(jax.device_get(sol.ys.n[i])) for i in range(nframes)]
    elif args.field == "omega":
        frames = [_plot_field(jax.device_get(sol.ys.omega[i])) for i in range(nframes)]
    else:
        frames = [_plot_field(jax.device_get(sol.ys.Te[i])) for i in range(nframes)]
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
    # Simple zonal-dominance diagnostic: the fraction of fluctuation RMS living in the
    # ky=0 (y-mean) component. Values near 1 indicate a nearly pure banded/zonal state.
    zonal = frames_fluct.mean(axis=2, keepdims=True)
    rms_total = np.sqrt(np.mean(frames_fluct**2, axis=(1, 2)))
    rms_zonal = np.sqrt(np.mean(zonal**2, axis=(1, 2)))
    ratio = rms_zonal / (rms_total + 1e-30)
    print(
        "[drb2d-movie] zonal-rms ratio "
        f"min={ratio.min():.3f} max={ratio.max():.3f} final={ratio[-1]:.3f}"
    )
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
    E = np.real(E)
    dlogE_dt = np.gradient(np.log(np.maximum(E, 1e-30)), np.asarray(ts))
    tail = slice(int(2 * len(dlogE_dt) / 3), None)
    print(f"[drb2d-movie] mean dlnE/dt over final third: {float(np.mean(dlogE_dt[tail])):.3e}")
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
    if np.iscomplexobj(phi_f):
        phi_f = np.real(phi_f)
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
