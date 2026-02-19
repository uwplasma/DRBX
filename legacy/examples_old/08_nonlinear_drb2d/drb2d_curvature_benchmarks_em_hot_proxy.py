"""DRB2D EM/hot-ion curvature-drive comparison vs Tokam1D proxy.

This companion to `drb2d_curvature_benchmarks.py` isolates the EM and hot-ion
branches and compares their drive-threshold scans against the Tokam1D dispersion
proxy (JPP 2025). It also highlights curvature-driven trends at fixed drive.

Outputs (in --out):
  - drb2d_curvature_benchmarks_em_hot.png
  - curvature_em_hot_proxy.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


def _gamma_proxy_tokam1d(*, omega_star: float, g: float, ky: float, kperp2: float) -> float:
    b = -g * ky
    c = -(g * ky / kperp2) * (g * ky - omega_star)
    disc = b * b - 4.0 * c
    root = np.sqrt(disc + 0.0j)
    w1 = 0.5 * (g * ky + root)
    w2 = 0.5 * (g * ky - root)
    return float(max(w1.imag, w2.imag))


def _linear_gamma_hot(model: DRB2DHotIonModel, v0: DRB2DHotIonState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DHotIonState(n=zero, omega=zero, vpar_e=zero, vpar_i=zero, Te=zero, Ti=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=15.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def _linear_gamma_em(model: DRB2DEMModel, v0: DRB2DEMState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DEMState(n=zero, omega=zero, psi=zero, vpar_i=zero, Te=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=15.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--ky", type=float, default=1.0)
    p.add_argument("--out", type=str, default="out_drb2d_curvature_em_hot")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (args.kx * X + args.ky * Y))
    amp = 1e-6
    v0_hot = DRB2DHotIonState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
        Ti=jnp.asarray(amp * mode),
    )
    v0_em = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    g_ref = 0.3
    omega_scan = np.linspace(0.0, 0.9, 7)
    gamma_hot = []
    gamma_em = []
    gamma_proxy = []
    for omega_n in omega_scan:
        params_hot = DRB2DHotIonParams(
            omega_n=float(omega_n),
            omega_Te=0.0,
            omega_Ti=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            tau_i=1.0,
            curvature_on=True,
            curvature_coeff=float(g_ref),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            DTi=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        params_em = DRB2DEMParams(
            omega_n=float(omega_n),
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            beta=0.2,
            Dpsi=0.0,
            curvature_on=True,
            curvature_coeff=float(g_ref),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        gamma_hot.append(_linear_gamma_hot(DRB2DHotIonModel(params=params_hot, grid=grid), v0_hot))
        gamma_em.append(_linear_gamma_em(DRB2DEMModel(params=params_em, grid=grid), v0_em))
        gamma_proxy.append(
            _gamma_proxy_tokam1d(
                omega_star=float(omega_n),
                g=float(g_ref),
                ky=float(args.ky),
                kperp2=float(args.kx**2 + args.ky**2),
            )
        )

    omega_scan = np.asarray(omega_scan)
    gamma_hot = np.asarray(gamma_hot)
    gamma_em = np.asarray(gamma_em)
    gamma_proxy = np.asarray(gamma_proxy)
    omega_crit = g_ref * float(args.ky) * (1.0 + (args.kx**2 + args.ky**2) / 4.0)

    curv_vals = np.linspace(0.0, 0.4, 5)
    omega_drive = 0.6
    gamma_hot_curv = []
    gamma_em_curv = []
    for curv in curv_vals:
        params_hot = DRB2DHotIonParams(
            omega_n=omega_drive,
            omega_Te=0.0,
            omega_Ti=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            tau_i=1.0,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            DTi=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        params_em = DRB2DEMParams(
            omega_n=omega_drive,
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            beta=0.2,
            Dpsi=0.0,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        gamma_hot_curv.append(
            _linear_gamma_hot(DRB2DHotIonModel(params=params_hot, grid=grid), v0_hot)
        )
        gamma_em_curv.append(_linear_gamma_em(DRB2DEMModel(params=params_em, grid=grid), v0_em))

    gamma_hot_curv = np.asarray(gamma_hot_curv)
    gamma_em_curv = np.asarray(gamma_em_curv)

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.4))
    ax = axs[0]
    ax.plot(omega_scan, gamma_hot, "o-", lw=2, label="hot-ion")
    ax.plot(omega_scan, gamma_em, "s-", lw=2, label="EM")
    ax.plot(omega_scan, gamma_proxy, "--", lw=2, label="Tokam1D proxy")
    ax.axvline(omega_crit, color="k", ls=":", lw=1.4, label=r"$\omega_*^{\rm crit}$")
    ax.set_xlabel(r"$\omega_n$ (drive)")
    ax.set_ylabel("growth γ")
    ax.set_title("Drive threshold vs Tokam1D proxy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1]
    ax.plot(curv_vals, gamma_hot_curv, "o-", lw=2, label="hot-ion")
    ax.plot(curv_vals, gamma_em_curv, "s-", lw=2, label="EM")
    ax.set_xlabel("curvature coefficient")
    ax.set_ylabel("growth γ")
    ax.set_title("Curvature-driven growth (ω_n=0.6)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_curvature_benchmarks_em_hot.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "curvature_em_hot_proxy.npz",
        omega_scan=omega_scan,
        gamma_hot=gamma_hot,
        gamma_em=gamma_em,
        gamma_proxy=gamma_proxy,
        omega_crit=omega_crit,
        curvature=curv_vals,
        gamma_hot_curv=gamma_hot_curv,
        gamma_em_curv=gamma_em_curv,
        omega_drive=omega_drive,
    )
    print(f"[drb2d-curv-emhot] wrote {out_dir}")


if __name__ == "__main__":
    main()
