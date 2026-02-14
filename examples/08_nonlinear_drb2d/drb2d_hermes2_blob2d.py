"""Hermes-2 blob2d proxy in DRB2D (open-field-line 2D SOL).

This example mirrors the Hermes-2 `blob2d/BOUT.inp` setup:
  - 2D open-field-line slab (ixseps=-1 in Hermes), approximated here with periodic x
    and linear damping to mimic parallel losses on a reduced grid
  - Gaussian density/pressure perturbation (Ne:function, Pe/Pi ~ 1.2*(n-1))
  - Curvature-driven interchange dynamics (bxcvz = 1/R^2 with R=1.5 m)
  - Domain sizes matched to Lrad=Lpol=0.3 m (coordinates normalized so Lx=Ly=1 corresponds to 0.3 m)

The run is intentionally small and fast; the goal is a reproducible 2D SOL
benchmark anchored to Hermes-2 parameters rather than long-time intermittency.
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

from jaxdrb.analysis.plotting import (
    robust_symmetric_vlim,
    save_animation_gif,
    save_animation_mp4,
    set_mpl_style,
)
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _hermes_blob_profile(x: np.ndarray, y: np.ndarray, *, Lx: float, Ly: float) -> np.ndarray:
    # Hermes-2 blob2d BOUT.inp:
    # Ne:function = 1 + 0.27*gauss(x-0.33, 0.21/4.)*gauss((z/(2*pi))-0.5,0.21/4.)
    sigma = 0.21 / 4.0
    x0 = 0.33
    y0 = 0.5
    xn = x / Lx
    yn = y / Ly
    blob = np.exp(-(((xn - x0) / sigma) ** 2)) * np.exp(-(((yn - y0) / sigma) ** 2))
    return 1.0 + 0.27 * blob


def _blob_center(x: np.ndarray, n: np.ndarray, *, n0: float) -> float:
    n_fluct = n - n0
    pos = np.maximum(n_fluct, 0.0)
    denom = np.sum(pos) + 1e-12
    return float(np.sum(x * pos) / denom)


def _radial_flux(n: np.ndarray, phi: np.ndarray, *, dy: float, n0: float) -> float:
    # vEx = -dphi/dy, mean of n' vEx
    dphi_dy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dy)
    v_ex = -dphi_dy
    n_fluct = n - n0
    return float(np.mean(n_fluct * v_ex))


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--Lx", type=float, default=1.0)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--tmax", type=float, default=16.0)
    parser.add_argument("--save-stride", type=int, default=12)
    parser.add_argument("--curvature", type=float, default=-(1.0 / (1.5**2)))
    parser.add_argument("--Dn", type=float, default=3e-3)
    parser.add_argument("--DOmega", type=float, default=3e-3)
    parser.add_argument("--DTe", type=float, default=3e-3)
    parser.add_argument("--mu-lin-n", type=float, default=0.05)
    parser.add_argument("--mu-lin-omega", type=float, default=0.05)
    parser.add_argument("--mu-lin-Te", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_drb2d_hermes2_blob")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-stride", type=int, default=2)
    parser.add_argument("--max-wall", type=float, default=45.0)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", bool(args.x64))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(
        nx=int(args.nx),
        ny=int(args.ny),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )

    params = DRB2DParams(
        log_n=False,
        log_Te=False,
        kpar=0.0,
        eta=0.0,
        me_hat=1.0,
        curvature_on=True,
        curvature_coeff=float(args.curvature),
        omega_n=0.0,
        omega_Te=0.0,
        sol_on=False,
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        DTe=float(args.DTe),
        mu_lin_n=float(args.mu_lin_n),
        mu_lin_omega=float(args.mu_lin_omega),
        mu_lin_Te=float(args.mu_lin_Te),
        bracket="arakawa",
        bracket_zero_mean=False,
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    x = np.asarray(grid.x)[:, None]
    y = np.asarray(grid.y)[None, :]
    n0 = _hermes_blob_profile(x, y, Lx=float(args.Lx), Ly=float(args.Ly))
    Te0 = 1.0 + 1.2 * (n0 - 1.0)
    omega0 = np.zeros_like(n0)
    v0 = np.zeros_like(n0)

    y0 = DRB2DState(
        n=jnp.asarray(n0),
        omega=jnp.asarray(omega0),
        vpar_e=jnp.asarray(v0),
        vpar_i=jnp.asarray(v0),
        Te=jnp.asarray(Te0),
    )

    dt = float(args.dt)
    nsteps = int(np.ceil(float(args.tmax) / dt))
    save_every = max(int(args.save_stride), 1)
    frame_dt = dt * save_every
    print(
        f"[hermes2-blob2d] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"nsteps={nsteps} save_every={save_every} frame_dt={frame_dt:.4g}"
    )

    start = time.time()
    ys, _ = model.diffeqsolve_fixed_steps(
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=save_every,
        progress=bool(args.progress),
    )
    wall = time.time() - start
    print(f"[hermes2-blob2d] runtime {wall:.2f}s")
    if wall > float(args.max_wall):
        print("[hermes2-blob2d] warning: runtime exceeded max-wall target.")

    n_series = np.asarray(ys.n)
    omega_series = np.asarray(ys.omega)
    t_series = np.asarray(jnp.arange(save_every, nsteps + 1, save_every) * dt)
    n_series = np.concatenate([n0[None, ...], n_series], axis=0)
    omega_series = np.concatenate([omega0[None, ...], omega_series], axis=0)
    t_series = np.concatenate([np.array([0.0]), t_series], axis=0)

    # Movie of density fluctuations.
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    n_fluct0 = n_series[0] - 1.0
    vlim = robust_symmetric_vlim(n_fluct0)
    img = ax.imshow(
        n_fluct0.T,
        origin="lower",
        cmap="coolwarm",
        extent=[0, float(args.Lx), 0, float(args.Ly)],
        vmin=-vlim,
        vmax=vlim,
        aspect="auto",
    )
    ax.set_title("Hermes-2 blob2d proxy: n'")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label(r"$n'$")
    t_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )

    def update(frame: int):
        data = n_series[frame] - 1.0
        vlim = robust_symmetric_vlim(data)
        img.set_data(data.T)
        img.set_clim(-vlim, vlim)
        t_text.set_text(f"t = {t_series[frame]:.2f}")
        return (img, t_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(n_series),
        interval=1000 / max(int(args.fps), 1),
        blit=False,
    )

    mp4_path = out_dir / "hermes2_blob2d.mp4"
    save_animation_mp4(ani, mp4_path, fps=int(args.fps), dpi=100)
    print(f"[hermes2-blob2d] wrote {mp4_path}")
    if args.gif:
        gif_path = out_dir / "hermes2_blob2d.gif"
        save_animation_gif(ani, gif_path, fps=int(args.fps))
        print(f"[hermes2-blob2d] wrote {gif_path}")

    if args.analysis:
        stride = max(int(args.analysis_stride), 1)
        n_a = n_series[::stride]
        omega_a = omega_series[::stride]
        t_a = t_series[::stride]
        x_cm = []
        flux = []
        for n_i, w_i in zip(n_a, omega_a, strict=False):
            phi_i = np.asarray(model.phi_from_omega(jnp.asarray(w_i)))
            x_cm.append(_blob_center(x, n_i, n0=1.0))
            flux.append(_radial_flux(n_i, phi_i, dy=float(grid.dy), n0=1.0))
        x_cm = np.asarray(x_cm)
        flux = np.asarray(flux)

        n_mean = np.mean(n_a, axis=2)
        n_rms = np.sqrt(np.mean((n_a - n_mean[:, :, None]) ** 2, axis=2))
        y_probe = int(0.5 * (grid.ny - 1))
        probe = n_a[:, :, y_probe] - 1.0
        hist, edges = np.histogram(probe.reshape(-1), bins=60, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig2, axs = plt.subplots(2, 3, figsize=(11.4, 6.6), constrained_layout=True)
        ax_ts = axs[0, 0]
        ax_ts.plot(t_a, x_cm, label="x_cm")
        ax_ts.plot(t_a, flux, label="mean flux")
        ax_ts.set_title("Blob center + flux")
        ax_ts.set_xlabel("t")
        ax_ts.legend()

        ax_pdf = axs[0, 1]
        ax_pdf.semilogy(centers, hist + 1e-30)
        ax_pdf.set_title("Probe n' PDF")
        ax_pdf.set_xlabel("n'")
        ax_pdf.set_ylabel("pdf")

        ax_prof = axs[0, 2]
        ax_prof.plot(np.asarray(grid.x), np.mean(n_mean, axis=0), label="mean n")
        ax_prof.plot(np.asarray(grid.x), np.mean(n_rms, axis=0), label="rms n")
        ax_prof.set_title("Radial profiles (tail)")
        ax_prof.set_xlabel("x")
        ax_prof.legend()

        snap_mid = int(np.argmin(np.abs(t_series - 0.6 * float(args.tmax))))
        n_snap = n_series[snap_mid] - 1.0
        w_snap = omega_series[snap_mid]
        phi_snap = np.asarray(model.phi_from_omega(jnp.asarray(w_snap)))

        ax_n = axs[1, 0]
        vlim = robust_symmetric_vlim(n_snap)
        im_n = ax_n.imshow(
            n_snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax_n.set_title(f"n' t={t_series[snap_mid]:.1f}")
        fig2.colorbar(im_n, ax=ax_n, pad=0.02, label="n'")

        ax_w = axs[1, 1]
        vlim = robust_symmetric_vlim(w_snap)
        im_w = ax_w.imshow(
            w_snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax_w.set_title(f"omega t={t_series[snap_mid]:.1f}")
        fig2.colorbar(im_w, ax=ax_w, pad=0.02, label=r"$\omega$")

        ax_phi = axs[1, 2]
        vlim = robust_symmetric_vlim(phi_snap)
        im_phi = ax_phi.imshow(
            phi_snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax_phi.set_title(f"phi t={t_series[snap_mid]:.1f}")
        fig2.colorbar(im_phi, ax=ax_phi, pad=0.02, label=r"$\phi$")

        panel_path = out_dir / "hermes2_blob2d_panel.png"
        fig2.savefig(panel_path, dpi=150)
        plt.close(fig2)
        print(f"[hermes2-blob2d] wrote {panel_path}")


if __name__ == "__main__":
    main()
