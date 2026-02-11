"""Linear-phase benchmark for DRB2D hot-ion and EM branches.

This compares growth rates using:
  1) a constant-geometry linear flux-tube solver (matrix-free),
  2) a linearized DRB2D operator about the zero state.

Outputs (in --out):
  - drb2d_linear_phase_hot_ion.png
  - drb2d_linear_phase_em.png
  - metrics_hot_ion.json
  - metrics_em.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.linear.growthrate import estimate_growth_rate_jax
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.cold_ion_drb import Equilibrium
from jaxdrb.models.em_drb import State as EMState
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.hot_ion_drb import State as HotIonState
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


@dataclass
class ConstantGeometry:
    kpar: float
    kperp2_value: float
    curvature_coeff: float = 0.0

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:  # type: ignore[override]
        return jnp.asarray([self.kperp2_value])

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return 1j * float(self.kpar) * f

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        if self.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        return -1j * float(self.curvature_coeff) * float(ky) * f


def _linear_growth_hot_ion(
    *,
    kx: float,
    ky: float,
    kpar: float,
    curvature_coeff: float,
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    geom = ConstantGeometry(kpar=kpar, kperp2_value=kx**2 + ky**2, curvature_coeff=curvature_coeff)
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.3,
        omega_Ti=0.2,
        eta=0.5,
        me_hat=0.2,
        tau_i=1.0,
        curvature_on=True,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    eq = Equilibrium.constant(1, n0=1.0, Te0=1.0)
    y0 = HotIonState.zeros(1)
    matvec = linear_matvec_from_rhs(hot_rhs, y0, params, geom, kx=kx, ky=ky, rhs_kwargs={"eq": eq})
    v0 = HotIonState(
        n=jnp.asarray([1e-6 + 0j]),
        omega=jnp.asarray([1e-6 + 0j]),
        vpar_e=jnp.asarray([0.0 + 0j]),
        vpar_i=jnp.asarray([0.0 + 0j]),
        Te=jnp.asarray([1e-6 + 0j]),
        Ti=jnp.asarray([1e-6 + 0j]),
    )
    res = estimate_growth_rate_jax(matvec, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma), res.t, res.log_norm


def _linear_growth_em(
    *,
    kx: float,
    ky: float,
    kpar: float,
    curvature_coeff: float,
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    geom = ConstantGeometry(kpar=kpar, kperp2_value=kx**2 + ky**2, curvature_coeff=curvature_coeff)
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.3,
        eta=0.5,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    eq = Equilibrium.constant(1, n0=1.0, Te0=1.0)
    y0 = EMState.zeros(1)
    matvec = linear_matvec_from_rhs(em_rhs, y0, params, geom, kx=kx, ky=ky, rhs_kwargs={"eq": eq})
    v0 = EMState(
        n=jnp.asarray([1e-6 + 0j]),
        omega=jnp.asarray([1e-6 + 0j]),
        psi=jnp.asarray([1e-6 + 0j]),
        vpar_i=jnp.asarray([0.0 + 0j]),
        Te=jnp.asarray([1e-6 + 0j]),
    )
    res = estimate_growth_rate_jax(matvec, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma), res.t, res.log_norm


def _drb2d_growth_hot_ion(
    *, kx: float, ky: float, kpar: float, curvature_coeff: float
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DHotIonParams(
        omega_n=0.8,
        omega_Te=0.3,
        omega_Ti=0.2,
        kpar=kpar,
        eta=0.5,
        me_hat=0.2,
        tau_i=1.0,
        alpha_Te_ohm=1.71,
        alpha_Ti=1.0,
        curvature_on=True,
        curvature_coeff=curvature_coeff,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DHotIonModel(params=params, grid=grid)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DHotIonState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
        Ti=jnp.asarray(amp * mode),
    )
    zero = jnp.zeros((grid.nx, grid.ny), dtype=jnp.complex128)
    y_zero = DRB2DHotIonState(n=zero, omega=zero, vpar_e=zero, vpar_i=zero, Te=zero, Ti=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate_jax(jvp_fn, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma), res.t, res.log_norm


def _drb2d_growth_em(
    *, kx: float, ky: float, kpar: float, curvature_coeff: float
) -> tuple[float, jnp.ndarray, jnp.ndarray]:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DEMParams(
        omega_n=0.8,
        omega_Te=0.3,
        kpar=kpar,
        eta=0.5,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        curvature_coeff=curvature_coeff,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DEMModel(params=params, grid=grid)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )
    zero = jnp.zeros((grid.nx, grid.ny), dtype=jnp.complex128)
    y_zero = DRB2DEMState(n=zero, omega=zero, psi=zero, vpar_i=zero, Te=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate_jax(jvp_fn, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma), res.t, res.log_norm


def main() -> None:
    out_dir = Path("out/examples/08_nonlinear_drb2d/drb2d_linear_phase_benchmark_ext")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    kx = 1.0
    ky = 1.0
    kpar = 0.4
    curvature_coeff = 0.6

    gamma_lin_hot, t_hot, logn_hot = _linear_growth_hot_ion(
        kx=kx, ky=ky, kpar=kpar, curvature_coeff=curvature_coeff
    )
    gamma_drb_hot, t_hot_d, logn_hot_d = _drb2d_growth_hot_ion(
        kx=kx, ky=ky, kpar=kpar, curvature_coeff=curvature_coeff
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(np.asarray(t_hot), np.asarray(logn_hot), lw=2.0, label="linear solver")
    ax.plot(np.asarray(t_hot_d), np.asarray(logn_hot_d), lw=2.0, label="DRB2D (linearized)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\ln |\hat Y_{k_x,k_y}|$")
    ax.set_title("DRB2D hot-ion linear-phase growth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_linear_phase_hot_ion.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "metrics_hot_ion.json",
        {
            "gamma_linear": gamma_lin_hot,
            "gamma_drb2d": gamma_drb_hot,
            "kx": kx,
            "ky": ky,
            "kpar": kpar,
            "curvature_coeff": curvature_coeff,
        },
    )

    gamma_lin_em, t_em, logn_em = _linear_growth_em(
        kx=kx, ky=ky, kpar=kpar, curvature_coeff=curvature_coeff
    )
    gamma_drb_em, t_em_d, logn_em_d = _drb2d_growth_em(
        kx=kx, ky=ky, kpar=kpar, curvature_coeff=curvature_coeff
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(np.asarray(t_em), np.asarray(logn_em), lw=2.0, label="linear solver")
    ax.plot(np.asarray(t_em_d), np.asarray(logn_em_d), lw=2.0, label="DRB2D (linearized)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\ln |\hat Y_{k_x,k_y}|$")
    ax.set_title("DRB2D EM linear-phase growth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_linear_phase_em.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "metrics_em.json",
        {
            "gamma_linear": gamma_lin_em,
            "gamma_drb2d": gamma_drb_em,
            "kx": kx,
            "ky": ky,
            "kpar": kpar,
            "curvature_coeff": curvature_coeff,
        },
    )

    print(f"gamma_hot: linear={gamma_lin_hot:.4e} drb2d={gamma_drb_hot:.4e}")
    print(f"gamma_em:  linear={gamma_lin_em:.4e} drb2d={gamma_drb_em:.4e}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
