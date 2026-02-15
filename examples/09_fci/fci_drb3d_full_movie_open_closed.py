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


def _make_forcing_sequence(
    key: jax.Array, *, nsteps: int, shape: tuple[int, int, int], dt: float, tau: float
) -> np.ndarray:
    """Ornstein-Uhlenbeck forcing sequence (white-in-time -> smoothed)."""
    alpha = float(np.exp(-dt / max(tau, 1e-12)))
    seq = np.zeros((nsteps,) + shape, dtype=np.float64)
    prev = np.zeros(shape, dtype=np.float64)
    for i in range(nsteps):
        key, sub = jax.random.split(key)
        noise = np.asarray(jax.random.normal(sub, shape))
        prev = alpha * prev + np.sqrt(1.0 - alpha**2) * noise
        seq[i] = prev
    return seq


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
    parser.add_argument("--omega-n", type=float, default=1.6)
    parser.add_argument("--omega-Te", type=float, default=1.1)
    parser.add_argument("--kappa", type=float, default=0.7)
    parser.add_argument("--Dn", type=float, default=4e-4)
    parser.add_argument("--DOmega", type=float, default=4e-4)
    parser.add_argument("--Dvpar", type=float, default=4e-4)
    parser.add_argument("--DTe", type=float, default=4e-4)
    parser.add_argument("--chi-par", type=float, default=6e-4)
    parser.add_argument("--sheath-nu-particle", type=float, default=0.18)
    parser.add_argument("--sheath-nu-energy", type=float, default=0.10)
    parser.add_argument("--sheath-nu-mom", type=float, default=0.45)
    parser.add_argument("--forcing-amp", type=float, default=0.0)
    parser.add_argument("--forcing-tau", type=float, default=0.15)
    parser.add_argument("--source-tau", type=float, default=0.0)
    parser.add_argument("--source-amp", type=float, default=0.0)
    parser.add_argument("--source-Te-amp", type=float, default=0.0)
    parser.add_argument("--analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--analysis-tail", type=float, default=0.5)
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
    core_mask = 1.0 - sol_mask

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
        omega_n=float(args.omega_n),
        omega_Te=float(args.omega_Te),
        kappa=float(args.kappa),
        alpha=0.35,
        eta_par=0.04,
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        Dvpar=float(args.Dvpar),
        DTe=float(args.DTe),
        chi_par=float(args.chi_par),
        sheath_on=True,
        sheath_bc_model="loizu_linear",
        sheath_nu_particle=float(args.sheath_nu_particle),
        sheath_nu_energy=float(args.sheath_nu_energy),
        sheath_nu_mom=float(args.sheath_nu_mom),
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

    if float(args.source_tau) > 0.0:
        profile = 0.5 * (1.0 - jnp.tanh((x - float(args.lcfs)) / max(float(args.lcfs_width), 1e-6)))
        n_bg = 1.0 + float(args.source_amp) * profile
        Te_bg = 1.0 + float(args.source_Te_amp) * profile
        n_bg = jnp.broadcast_to(n_bg[None, :, None], (grid.nz, grid.nx, grid.ny))
        Te_bg = jnp.broadcast_to(Te_bg[None, :, None], (grid.nz, grid.nx, grid.ny))
    else:
        n_bg = None
        Te_bg = None

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
    forcing_seq = None
    if float(args.forcing_amp) > 0.0:
        key = jax.random.key(int(args.seed) + 17)
        forcing_seq = float(args.forcing_amp) * _make_forcing_sequence(
            key,
            nsteps=nsteps,
            shape=(grid.nz, grid.nx, grid.ny),
            dt=dt,
            tau=float(args.forcing_tau),
        )

    def rhs_with_forcing(t, y):
        dy = model.rhs(t, y)
        dn_source = 0.0
        dTe_source = 0.0
        if n_bg is not None and float(args.source_tau) > 0.0:
            tau = float(args.source_tau)
            dn_source = core_mask * (n_bg - y.n) / tau
            dTe_source = core_mask * (Te_bg - y.Te) / tau
        if forcing_seq is None and n_bg is None:
            return dy
        if forcing_seq is None:
            f = 0.0
        else:
            idx = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, forcing_seq.shape[0] - 1)
            f = jnp.asarray(forcing_seq)[idx]
        return FCIDRB3DFullState(
            n=dy.n + dn_source,
            omega=dy.omega + f,
            vpar_e=dy.vpar_e,
            vpar_i=dy.vpar_i,
            Te=dy.Te + dTe_source,
            Ti=dy.Ti,
            psi=dy.psi,
            N=dy.N,
        )

    ys, _ = diffeqsolve_fixed_steps(
        rhs_with_forcing,
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
    omega_series = np.asarray(jax.device_get(ys.omega))
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

    if args.analysis:
        import matplotlib.gridspec as gridspec

        t_series = np.asarray(save_ts)
        n_mid = n_series[:, kz]
        omega_mid = omega_series[:, kz]
        n_fluct = n_mid - 1.0
        omega_fluct = omega_mid - omega_mid.mean(axis=(1, 2), keepdims=True)

        rms_n = np.sqrt(np.mean(n_fluct**2, axis=(1, 2)))
        rms_omega = np.sqrt(np.mean(omega_fluct**2, axis=(1, 2)))

        tail = int(max(1, np.floor(float(args.analysis_tail) * len(t_series))))
        n_tail = n_fluct[-tail:]
        omega_tail = omega_fluct[-tail:]

        tail_rms_n = float(np.mean(rms_n[-tail:]))
        tail_rms_omega = float(np.mean(rms_omega[-tail:]))
        peak_rms_n = float(np.max(rms_n))
        peak_rms_omega = float(np.max(rms_omega))
        print(
            "[fci-drb3d-open-closed] rms_n tail/peak="
            f"{tail_rms_n:.3e}/{peak_rms_n:.3e} (ratio={tail_rms_n / max(peak_rms_n,1e-30):.2f}), "
            "rms_omega tail/peak="
            f"{tail_rms_omega:.3e}/{peak_rms_omega:.3e} (ratio={tail_rms_omega / max(peak_rms_omega,1e-30):.2f})"
        )

        mean_n = n_tail.mean(axis=(0, 2))
        rms_n_rad = np.sqrt(np.mean(n_tail**2, axis=(0, 2)))

        x = grid.x0 + grid.dx * (np.arange(grid.nx) + 0.5)
        y = grid.y0 + grid.dy * (np.arange(grid.ny) + 0.5)
        x_probe = min(float(args.lcfs) + 0.6, float(x[-1]))
        y_probe = float(y[len(y) // 2])
        ix = int(np.argmin(np.abs(x - x_probe)))
        iy = int(np.argmin(np.abs(y - y_probe)))
        probe = n_fluct[:, ix, iy]
        probe_tail = probe[-tail:]
        probe_mean = probe_tail.mean()
        probe_std = probe_tail.std()
        thresh = 2.5 * probe_std
        event_idx = np.where(np.abs(probe_tail - probe_mean) > thresh)[0]
        max_tail = np.max(n_tail, axis=(1, 2))
        max_mean = max_tail.mean()
        max_std = max_tail.std()
        max_thresh = max_mean + 2.0 * max_std
        event_idx_max = np.where(max_tail > max_thresh)[0]
        print(
            "[fci-drb3d-open-closed] probe events="
            f"{event_idx.size} |n'-mean|> {thresh:.3e} "
            f"probe mean/std={probe_mean:.3e}/{probe_std:.3e}"
        )
        print(
            "[fci-drb3d-open-closed] max-n' events="
            f"{event_idx_max.size} max_thresh={max_thresh:.3e} "
            f"max mean/std={max_mean:.3e}/{max_std:.3e}"
        )
        if event_idx_max.size > 0:
            cond_avg = n_tail[event_idx_max].mean(axis=0)
        elif event_idx.size > 0:
            cond_avg = n_tail[event_idx].mean(axis=0)
        else:
            cond_avg = n_tail.mean(axis=0)

        fig = plt.figure(figsize=(11.0, 6.6))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(t_series, rms_n, label="rms(n')")
        ax0.plot(t_series, rms_omega, label="rms(Ω')")
        ax0.set_title("Mid-plane RMS")
        ax0.set_xlabel("t")
        ax0.legend(frameon=False)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(probe_tail, bins=50, density=True)
        ax1.axvline(thresh, color="r", lw=1.0)
        ax1.set_title("Probe n' PDF (tail)")
        ax1.set_xlabel("n'")

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(x, mean_n, label="mean(n')")
        ax2.plot(x, rms_n_rad, label="rms(n')")
        ax2.axvline(float(args.lcfs), color="k", lw=1.0)
        ax2.set_title("Radial profiles (tail)")
        ax2.set_xlabel("x")
        ax2.legend(frameon=False)

        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(cond_avg.T, origin="lower", cmap="coolwarm", aspect="auto")
        fig.colorbar(im3, ax=ax3, pad=0.02)
        ax3.set_title("Conditional avg n'")
        ax3.set_xticks([])
        ax3.set_yticks([])

        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(omega_tail[-1].T, origin="lower", cmap="coolwarm", aspect="auto")
        fig.colorbar(im4, ax=ax4, pad=0.02)
        ax4.set_title("Ω' snapshot (tail)")
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1, 2])
        im5 = ax5.imshow(n_tail[-1].T, origin="lower", cmap="coolwarm", aspect="auto")
        fig.colorbar(im5, ax=ax5, pad=0.02)
        ax5.set_title("n' snapshot (tail)")
        ax5.set_xticks([])
        ax5.set_yticks([])

        fig.tight_layout()
        panel_path = out_dir / "diagnostics_panel.png"
        fig.savefig(panel_path, dpi=140)
        plt.close(fig)
        print(f"[fci-drb3d-open-closed] wrote {panel_path}")


if __name__ == "__main__":
    main()
