"""DRB2D curvature-drive benchmarks (interchange vs resistive-like coupling).

We estimate linear growth rates from a linearized DRB2D operator and compare
curvature trends for:
  - interchange-like (kpar=0, eta=0)
  - resistive-like (kpar>0, eta>0)

We also compare a drive-threshold scan against a published dispersion proxy
for interchange / resistive drift-wave coupling (see Tokam1D, J. Plasma Phys.).

Outputs (in --out):
  - drb2d_curvature_benchmarks.png
  - curvature_benchmarks.npz
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
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _gamma_proxy_tokam1d(*, omega_star: float, g: float, ky: float, kperp2: float) -> float:
    """Tokam1D dispersion proxy (JPP Eq. 3.9) for interchange/CDW coupling."""

    b = -g * ky
    c = -(g * ky / kperp2) * (g * ky - omega_star)
    disc = b * b - 4.0 * c
    root = np.sqrt(disc + 0.0j)
    w1 = 0.5 * (g * ky + root)
    w2 = 0.5 * (g * ky - root)
    return float(max(w1.imag, w2.imag))


def _linear_gamma(model: DRB2DModel, v0: DRB2DState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DState(n=zero, omega=zero, vpar_e=zero, vpar_i=zero, Te=zero)
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
    p.add_argument("--out", type=str, default="out_drb2d_curvature_benchmarks")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (args.kx * X + args.ky * Y))
    amp = 1e-6
    v0 = DRB2DState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    curv_vals = np.linspace(0.0, 0.64, 5)
    gamma_interchange = []
    gamma_resistive = []

    for curv in curv_vals:
        params_interchange = DRB2DParams(
            omega_n=0.0,
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        params_resistive = DRB2DParams(
            omega_n=0.0,
            omega_Te=0.0,
            kpar=0.3,
            eta=0.2,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        gamma_interchange.append(
            _linear_gamma(DRB2DModel(params=params_interchange, grid=grid), v0)
        )
        gamma_resistive.append(_linear_gamma(DRB2DModel(params=params_resistive, grid=grid), v0))
        print(
            f"[drb2d-curv] curv={curv:.2f} "
            f"gamma_int={gamma_interchange[-1]:.3e} "
            f"gamma_res={gamma_resistive[-1]:.3e}"
        )

    gamma_interchange = np.asarray(gamma_interchange)
    gamma_resistive = np.asarray(gamma_resistive)

    g_ref = 0.3
    omega_scan = np.linspace(0.0, 0.9, 7)
    gamma_scan = []
    gamma_proxy = []
    for omega_n in omega_scan:
        params_scan = DRB2DParams(
            omega_n=float(omega_n),
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=float(g_ref),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        gamma_scan.append(_linear_gamma(DRB2DModel(params=params_scan, grid=grid), v0))
        gamma_proxy.append(
            _gamma_proxy_tokam1d(
                omega_star=float(omega_n),
                g=float(g_ref),
                ky=float(args.ky),
                kperp2=float(args.kx**2 + args.ky**2),
            )
        )

    omega_scan = np.asarray(omega_scan)
    gamma_scan = np.asarray(gamma_scan)
    gamma_proxy = np.asarray(gamma_proxy)
    omega_crit = g_ref * float(args.ky) * (1.0 + (args.kx**2 + args.ky**2) / 4.0)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axs[0]
    ax.plot(curv_vals, gamma_interchange, "o-", lw=2, label="interchange (kpar=0, eta=0)")
    ax.plot(curv_vals, gamma_resistive, "s-", lw=2, label="resistive-like (kpar=0.3, eta=0.2)")
    ax.set_xlabel("curvature coefficient")
    ax.set_ylabel("linear growth γ")
    ax.set_title("Curvature-driven growth")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1]
    ax.plot(omega_scan, gamma_scan, "o-", lw=2, label="DRB2D (linearized)")
    ax.plot(omega_scan, gamma_proxy, "--", lw=2, label="Tokam1D proxy (JPP Eq. 3.9)")
    ax.axvline(omega_crit, color="k", ls=":", lw=1.5, label=r"$\omega_*^{\rm crit}$ (Eq. 3.10)")
    ax.set_xlabel(r"$\omega_n$ (drive)")
    ax.set_ylabel("growth γ")
    ax.set_title("Drive threshold proxy")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_curvature_benchmarks.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "curvature_benchmarks.npz",
        curvature=curv_vals,
        gamma_interchange=gamma_interchange,
        gamma_resistive=gamma_resistive,
        omega_scan=omega_scan,
        gamma_scan=gamma_scan,
        gamma_proxy=gamma_proxy,
        omega_crit=omega_crit,
    )
    print(f"[drb2d-curv] wrote {out_dir}")


if __name__ == "__main__":
    main()
