from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


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


def test_curvature_drive_increases_hot_ion_growth() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 0.0
    ky = 1.0
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

    params_flat = DRB2DHotIonParams(
        omega_n=0.6,
        omega_Te=0.0,
        omega_Ti=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=1.0,
        curvature_on=True,
        curvature_coeff=0.0,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    params_curv = DRB2DHotIonParams(
        omega_n=0.6,
        omega_Te=0.0,
        omega_Ti=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=1.0,
        curvature_on=True,
        curvature_coeff=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        DTi=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    gamma0 = _linear_gamma_hot(DRB2DHotIonModel(params=params_flat, grid=grid), v0)
    gamma1 = _linear_gamma_hot(DRB2DHotIonModel(params=params_curv, grid=grid), v0)
    assert gamma1 > gamma0


def test_curvature_drive_increases_em_growth() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 0.0
    ky = 1.0
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    params_flat = DRB2DEMParams(
        omega_n=0.6,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        curvature_coeff=0.0,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    params_curv = DRB2DEMParams(
        omega_n=0.6,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        curvature_coeff=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    gamma0 = _linear_gamma_em(DRB2DEMModel(params=params_flat, grid=grid), v0)
    gamma1 = _linear_gamma_em(DRB2DEMModel(params=params_curv, grid=grid), v0)
    assert gamma1 > gamma0
