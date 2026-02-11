from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def test_drb2d_zonal_collapse_gate() -> None:
    """Regression guard: prevent silent collapse into purely zonal states."""

    grid = Grid2D.make(nx=36, ny=36, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = DRB2DParams(
        omega_n=1.0,
        omega_Te=0.5,
        kpar=0.0,
        eta=0.2,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.6,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        Dn4=2e-4,
        DOmega4=2e-4,
        DTe4=2e-4,
        mu_zonal_omega=0.12,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(123)
    shape = (grid.nx, grid.ny)
    amp = 5e-3
    z = jnp.zeros(shape)
    y0 = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=z,
        vpar_i=z,
        Te=amp * jax.random.normal(jax.random.key(2), shape),
    )

    _, y_end = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.02,
        nsteps=300,
        solver="dopri5",
    )

    omega = y_end.omega
    omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)
    frac = jnp.sqrt(jnp.mean(omega_zonal**2)) / jnp.maximum(jnp.sqrt(jnp.mean(omega**2)), 1e-30)

    assert float(frac) < 0.85
