from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _linear_gamma(model: DRB2DModel, v0: DRB2DState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DState(n=zero, omega=zero, vpar_e=zero, vpar_i=zero, Te=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=15.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def test_curvature_benchmarks_match_proxy_threshold() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 0.0
    ky = 1.0
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    g_ref = 0.3
    omega_crit = g_ref * ky * (1.0 + (kx**2 + ky**2) / 4.0)
    omega_low = 0.5 * omega_crit
    omega_high = 1.5 * omega_crit

    def gamma_for_drive(omega_n: float) -> float:
        params = DRB2DParams(
            omega_n=omega_n,
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=g_ref,
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
            operator_split_on=False,
        )
        return _linear_gamma(DRB2DModel(params=params, grid=grid), v0)

    gamma_low = gamma_for_drive(omega_low)
    gamma_high = gamma_for_drive(omega_high)
    assert gamma_high > gamma_low

    params_flat = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.0,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    params_curv = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.4,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    gamma0 = _linear_gamma(DRB2DModel(params=params_flat, grid=grid), v0)
    gamma1 = _linear_gamma(DRB2DModel(params=params_curv, grid=grid), v0)
    assert gamma1 > gamma0
