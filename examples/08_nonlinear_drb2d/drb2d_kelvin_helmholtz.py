"""Kelvin-Helmholtz (KH) 2D turbulence in the DRB2D vorticity system.

This config uses the DRB2D model as an incompressible 2D vorticity solver:
  - curvature/parallel/drive terms are disabled
  - n, Te are uniform (non-evolving backgrounds)
  - omega evolves via ExB advection + viscosity

The initial condition is a double-shear layer with a small sinusoidal
perturbation, which should roll up into KH vortices.
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
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _omega_from_shear(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    Lx: float,
    Ly: float,
    u0: float,
    shear_width: float,
    pert_amp: float,
    pert_mode: int,
) -> jnp.ndarray:
    """Double shear-layer vorticity with a sinusoidal perturbation."""

    y0 = 0.25 * Ly
    y1 = 0.75 * Ly
    a = float(shear_width)
    sech0 = 1.0 / jnp.cosh((y - y0) / a)
    sech1 = 1.0 / jnp.cosh((y - y1) / a)
    omega0 = (u0 / a) * (sech1**2 - sech0**2)
    if pert_amp != 0.0 and pert_mode > 0:
        envelope = jnp.exp(-(((y - y0) / a) ** 2)) + jnp.exp(-(((y - y1) / a) ** 2))
        omega0 = omega0 + pert_amp * u0 * envelope * jnp.sin(2.0 * jnp.pi * pert_mode * x / Lx)
    return omega0


def _spectral_phi_from_omega(omega: np.ndarray, *, k2: np.ndarray, k2_min: float) -> np.ndarray:
    omega_hat = np.fft.fft2(omega)
    denom = np.where(k2 > 0.0, k2, 1.0)
    phi_hat = -omega_hat / np.maximum(denom, k2_min)
    phi_hat[0, 0] = 0.0 + 0.0j
    return np.fft.ifft2(phi_hat).real


def _energy_enstrophy(
    omega: np.ndarray, *, kx: np.ndarray, ky: np.ndarray, k2: np.ndarray
) -> tuple[float, float]:
    phi = _spectral_phi_from_omega(omega, k2=k2, k2_min=1e-6)
    phi_hat = np.fft.fft2(phi)
    dphi_dx = np.fft.ifft2(1j * kx * phi_hat).real
    dphi_dy = np.fft.ifft2(1j * ky * phi_hat).real
    energy = 0.5 * float(np.mean(dphi_dx**2 + dphi_dy**2))
    enstrophy = 0.5 * float(np.mean(omega**2))
    return energy, enstrophy


def _spectrum_slope(k: np.ndarray, spec: np.ndarray, kmin: float, kmax: float) -> float:
    mask = (k >= kmin) & (k <= kmax) & np.isfinite(spec) & (spec > 0.0)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    x = np.log10(k[mask])
    y = np.log10(spec[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _make_forcing_sequence(
    key: jax.Array,
    *,
    nsteps: int,
    nx: int,
    ny: int,
    kx: np.ndarray,
    ky: np.ndarray,
    kmax: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    k_mag = np.sqrt(kx**2 + ky**2)
    mask = (k_mag <= float(kmax)) & (k_mag > 0.0)
    forcing = np.zeros((nsteps, nx, ny), dtype=np.float64)
    alpha = float(np.exp(-dt / max(tau, 1e-12)))
    prev = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nsteps):
        key, sub = jax.random.split(key)
        noise = jax.random.normal(sub, (nx, ny))
        noise_hat = np.fft.fft2(np.asarray(noise))
        noise_hat = noise_hat * mask
        f = np.fft.ifft2(noise_hat).real
        f = f - np.mean(f)
        f = f / max(float(np.sqrt(np.mean(f**2))), 1e-12)
        prev = alpha * prev + np.sqrt(1.0 - alpha**2) * f
        forcing[i] = prev
    return forcing


def _isotropic_spectrum(
    field_hat: np.ndarray,
    *,
    k_mag: np.ndarray,
    weight: np.ndarray,
    nbins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    kmax = float(np.max(k_mag))
    bins = np.linspace(0.0, kmax, nbins + 1)
    spec = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=float)
    for i in range(nbins):
        mask = (k_mag >= bins[i]) & (k_mag < bins[i + 1])
        if not np.any(mask):
            continue
        if np.ndim(weight) == 0:
            spec[i] = float(weight) * np.sum(np.abs(field_hat[mask]) ** 2)
        else:
            spec[i] = np.sum(weight[mask] * np.abs(field_hat[mask]) ** 2)
        counts[i] = np.sum(mask)
    spec = np.where(counts > 0.0, spec / np.maximum(counts, 1.0), 0.0)
    k_center = 0.5 * (bins[:-1] + bins[1:])
    return k_center, spec


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=192)
    parser.add_argument("--Lx", type=float, default=2.0 * np.pi)
    parser.add_argument("--Ly", type=float, default=2.0 * np.pi)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--save-stride", type=int, default=10)
    parser.add_argument("--u0", type=float, default=3.0)
    parser.add_argument("--shear-width", type=float, default=0.1)
    parser.add_argument("--pert-amp", type=float, default=0.12)
    parser.add_argument("--pert-mode", type=int, default=8)
    parser.add_argument("--nu", type=float, default=1.0e-4)
    parser.add_argument("--nu4", type=float, default=5.0e-7)
    parser.add_argument("--forcing-amp", type=float, default=2.0e-3)
    parser.add_argument("--forcing-kmax", type=float, default=6.0)
    parser.add_argument("--forcing-tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out_drb2d_kh")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-wall", type=float, default=45.0)
    parser.add_argument("--analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-stride", type=int, default=1)
    parser.add_argument(
        "--snapshots",
        type=str,
        default="0,10,20,30,40",
        help="Comma-separated list of snapshot times to save (nearest saved frame).",
    )
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
        curvature_on=False,
        curvature_coeff=0.0,
        omega_n=0.0,
        omega_Te=0.0,
        sol_on=False,
        Dn=0.0,
        DTe=0.0,
        DOmega=float(args.nu),
        Dn4=0.0,
        DTe4=0.0,
        DOmega4=float(args.nu4),
        mu_lin_n=0.0,
        mu_lin_Te=0.0,
        mu_lin_omega=0.0,
        mu_zonal_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        k2_min=1e-6,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    x = grid.x[:, None]
    y = grid.y[None, :]
    omega0 = _omega_from_shear(
        x,
        y,
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        u0=float(args.u0),
        shear_width=float(args.shear_width),
        pert_amp=float(args.pert_amp),
        pert_mode=int(args.pert_mode),
    )
    omega0 = omega0 - jnp.mean(omega0)

    n0 = jnp.ones_like(omega0)
    Te0 = jnp.ones_like(omega0)
    vpar_e0 = jnp.zeros_like(omega0)
    vpar_i0 = jnp.zeros_like(omega0)
    y0 = DRB2DState(n=n0, omega=omega0, vpar_e=vpar_e0, vpar_i=vpar_i0, Te=Te0)

    dt = float(args.dt)
    nsteps = int(np.ceil(float(args.tmax) / dt))
    save_every = max(int(args.save_stride), 1)
    frame_dt = dt * save_every
    print(
        f"[drb2d-kh] grid=({grid.nx},{grid.ny}) dt={dt} tmax={args.tmax} "
        f"nsteps={nsteps} save_every={save_every} frame_dt={frame_dt:.4g}"
    )

    forcing_seq = None
    if float(args.forcing_amp) > 0.0:
        key = jax.random.key(int(args.seed))
        forcing_seq = _make_forcing_sequence(
            key,
            nsteps=nsteps,
            nx=grid.nx,
            ny=grid.ny,
            kx=np.asarray(grid.kx),
            ky=np.asarray(grid.ky),
            kmax=float(args.forcing_kmax),
            dt=dt,
            tau=float(args.forcing_tau),
        )
        forcing_seq = float(args.forcing_amp) * forcing_seq

    def rhs_with_forcing(t, y):
        dy = model.rhs(t, y)
        if forcing_seq is None:
            return dy
        idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, forcing_seq.shape[0] - 1)
        f = jnp.asarray(forcing_seq)[idx]
        return DRB2DState(
            n=dy.n,
            omega=dy.omega + f,
            vpar_e=dy.vpar_e,
            vpar_i=dy.vpar_i,
            Te=dy.Te,
            N=dy.N,
        )

    start = time.time()
    ys, y_end = diffeqsolve_fixed_steps(
        rhs_with_forcing,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=save_every,
        progress=bool(args.progress),
    )
    wall = time.time() - start
    print(f"[drb2d-kh] runtime {wall:.2f}s")
    if wall > float(args.max_wall):
        print("[drb2d-kh] warning: runtime exceeded max-wall target.")

    omega_series = np.asarray(ys.omega)
    t_series = np.asarray(jnp.arange(save_every, nsteps + 1, save_every) * dt)
    omega_series = np.concatenate([np.asarray(omega0)[None, ...], omega_series], axis=0)
    t_series = np.concatenate([np.array([0.0]), t_series], axis=0)

    # Use a size that yields even pixel dimensions at dpi=100 for ffmpeg.
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    vlim = robust_symmetric_vlim(omega_series[0])
    img = ax.imshow(
        omega_series[0].T,
        origin="lower",
        cmap="coolwarm",
        extent=[0, float(args.Lx), 0, float(args.Ly)],
        vmin=-vlim,
        vmax=vlim,
        aspect="auto",
    )
    ax.set_title("KH vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label(r"$\omega$")
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
        data = omega_series[frame]
        vlim = robust_symmetric_vlim(data)
        img.set_data(data.T)
        img.set_clim(-vlim, vlim)
        t_text.set_text(f"t = {t_series[frame]:.2f}")
        return (img, t_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(omega_series),
        interval=1000 / max(int(args.fps), 1),
        blit=False,
    )

    # Save representative stills.
    snap_list = []
    for token in str(args.snapshots).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            snap_list.append(float(token))
        except ValueError:
            continue
    if not snap_list:
        snap_list = [0.0, float(args.tmax)]
    for t_snap in snap_list:
        idx = int(np.argmin(np.abs(t_series - t_snap)))
        update(idx)
        fig.savefig(out_dir / f"kh_vorticity_t{t_series[idx]:.2f}.png", dpi=150)

    mp4_path = out_dir / "kh_vorticity.mp4"
    save_animation_mp4(ani, mp4_path, fps=int(args.fps), dpi=100)
    print(f"[drb2d-kh] wrote {mp4_path}")
    if args.gif:
        gif_path = out_dir / "kh_vorticity.gif"
        save_animation_gif(ani, gif_path, fps=int(args.fps))
        print(f"[drb2d-kh] wrote {gif_path}")

    if args.analysis:
        stride = max(int(args.analysis_stride), 1)
        omega_a = omega_series[::stride]
        t_a = t_series[::stride]
        kx = np.asarray(grid.kx)
        ky = np.asarray(grid.ky)
        k2 = np.asarray(grid.k2)
        k_mag = np.sqrt(k2)
        nscale = float(grid.nx * grid.ny)

        energy = []
        enstrophy = []
        for om in omega_a:
            e, z = _energy_enstrophy(om, kx=kx, ky=ky, k2=k2)
            energy.append(e)
            enstrophy.append(z)
        energy = np.asarray(energy)
        enstrophy = np.asarray(enstrophy)

        # Spectra from the last few frames.
        spec_frames = omega_series[-min(len(omega_series), 5) :]
        spec_energy = []
        spec_enstrophy = []
        for om in spec_frames:
            phi = _spectral_phi_from_omega(om, k2=k2, k2_min=1e-6)
            phi_hat = np.fft.fft2(phi)
            omega_hat = np.fft.fft2(om)
            e_weight = 0.5 * k2 / (nscale**2)
            z_weight = 0.5 / (nscale**2)
            k_bins, e_spec = _isotropic_spectrum(phi_hat, k_mag=k_mag, weight=e_weight, nbins=36)
            _, z_spec = _isotropic_spectrum(omega_hat, k_mag=k_mag, weight=z_weight, nbins=36)
            spec_energy.append(e_spec)
            spec_enstrophy.append(z_spec)
        spec_energy = np.mean(np.asarray(spec_energy), axis=0)
        spec_enstrophy = np.mean(np.asarray(spec_enstrophy), axis=0)

        # Vorticity PDF (tail window).
        tail = omega_series[int(0.6 * len(omega_series)) :]
        omega_flat = tail.reshape(-1)
        omega_flat = omega_flat[np.isfinite(omega_flat)]
        omega_flat = omega_flat - np.mean(omega_flat)
        hist, edges = np.histogram(omega_flat, bins=80, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig2, axs = plt.subplots(2, 3, figsize=(11.4, 6.6), constrained_layout=True)
        ax_ts = axs[0, 0]
        ax_ts.plot(t_a, energy, label="energy")
        ax_ts.plot(t_a, enstrophy, label="enstrophy")
        ax_ts.set_title("Energy & enstrophy")
        ax_ts.set_xlabel("t")
        ax_ts.set_ylabel("value")
        ax_ts.legend()

        ax_spec = axs[0, 1]
        ax_spec.loglog(k_bins[1:], spec_energy[1:], label="E(k)")
        ax_spec.loglog(k_bins[1:], spec_enstrophy[1:], label="Z(k)")
        slope_e = _spectrum_slope(k_bins, spec_energy, kmin=5.0, kmax=25.0)
        slope_z = _spectrum_slope(k_bins, spec_enstrophy, kmin=5.0, kmax=25.0)
        ax_spec.set_title("Spectra (tail-avg)")
        ax_spec.set_xlabel("k")
        ax_spec.set_ylabel("spectrum")
        ax_spec.legend()
        ax_spec.text(
            0.02,
            0.05,
            f"slope E={slope_e:.2f}\\nslope Z={slope_z:.2f}",
            transform=ax_spec.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        ax_pdf = axs[0, 2]
        ax_pdf.semilogy(centers, hist + 1e-30)
        omega_mean = float(np.mean(omega_flat))
        omega_std = float(np.std(omega_flat))
        skew = float(np.mean(((omega_flat - omega_mean) / (omega_std + 1e-12)) ** 3))
        kurt = float(np.mean(((omega_flat - omega_mean) / (omega_std + 1e-12)) ** 4) - 3.0)
        ax_pdf.set_title("Vorticity PDF (tail)")
        ax_pdf.set_xlabel(r"$\omega'$")
        ax_pdf.set_ylabel("pdf")
        ax_pdf.text(
            0.02,
            0.05,
            f"skew={skew:.2f}\\nkurt={kurt:.2f}",
            transform=ax_pdf.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        snap_idx = int(np.argmin(np.abs(t_series - float(args.tmax))))
        snap_mid = int(np.argmin(np.abs(t_series - (0.6 * float(args.tmax)))))

        ax_phi = axs[1, 0]
        phi_snap = _spectral_phi_from_omega(omega_series[snap_mid], k2=k2, k2_min=1e-6)
        vlim_phi = robust_symmetric_vlim(phi_snap)
        im_phi = ax_phi.imshow(
            phi_snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim_phi,
            vmax=vlim_phi,
            aspect="auto",
        )
        ax_phi.set_title(f"Streamfunction t={t_series[snap_mid]:.1f}")
        ax_phi.set_xlabel("x")
        ax_phi.set_ylabel("y")
        fig2.colorbar(im_phi, ax=ax_phi, pad=0.02, label=r"$\phi$")

        ax_omega_mid = axs[1, 1]
        snap = omega_series[snap_mid]
        vlim = robust_symmetric_vlim(snap)
        im_mid = ax_omega_mid.imshow(
            snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax_omega_mid.set_title(f"Vorticity t={t_series[snap_mid]:.1f}")
        ax_omega_mid.set_xlabel("x")
        ax_omega_mid.set_ylabel("y")
        fig2.colorbar(im_mid, ax=ax_omega_mid, pad=0.02, label=r"$\omega$")

        ax_omega = axs[1, 2]
        snap = omega_series[snap_idx]
        vlim = robust_symmetric_vlim(snap)
        im = ax_omega.imshow(
            snap.T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, float(args.Lx), 0, float(args.Ly)],
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax_omega.set_title(f"Vorticity t={t_series[snap_idx]:.1f}")
        ax_omega.set_xlabel("x")
        ax_omega.set_ylabel("y")
        fig2.colorbar(im, ax=ax_omega, pad=0.02, label=r"$\omega$")

        panel_path = out_dir / "kh_analysis_panel.png"
        fig2.savefig(panel_path, dpi=150)
        plt.close(fig2)
        print(f"[drb2d-kh] wrote {panel_path}")


if __name__ == "__main__":
    main()
