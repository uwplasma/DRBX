from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _rand(key, shape, amp: float):
    return amp * jax.random.normal(key, shape)


def test_drb2d_smoke_dirichlet_centered_cg_no_nans():
    grid = Grid2D.make(
        nx=24,
        ny=24,
        Lx=2 * jnp.pi,
        Ly=2 * jnp.pi,
        dealias=False,
        bc_x="dirichlet",
        bc_y="dirichlet",
        bc_value_x=0.0,
        bc_value_y=0.0,
    )
    model = DRB2DModel(
        params=DRB2DParams(
            # Drive terms currently assume periodic y; keep this non-periodic
            # smoke test focused on BC/bracket/Poisson stability.
            omega_n=0.0,
            curvature_on=True,
            curvature_coeff=0.3,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            bracket="centered",
            poisson="cg_fd",
            dealias_on=False,
            bc_enforce_nu=5.0,
            operator_split_on=True,
        ),
        grid=grid,
    )

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    amp = 1e-3
    z = jnp.zeros(shape)
    y0 = DRB2DState(
        n=_rand(key, shape, amp),
        omega=_rand(jax.random.key(1), shape, amp),
        vpar_e=z,
        vpar_i=z,
        Te=z,
    )
    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.05,
        nsteps=5,
        solver="dopri5",
    )
    assert jnp.all(jnp.isfinite(y1.n))
    assert jnp.all(jnp.isfinite(y1.omega))


def test_drb2d_hot_ion_smoke_periodic_arakawa_spectral_no_nans():
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    model = DRB2DHotIonModel(
        params=DRB2DHotIonParams(
            omega_n=0.5,
            omega_Te=0.1,
            omega_Ti=0.1,
            tau_i=1.0,
            curvature_on=True,
            curvature_coeff=0.2,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            DTi=1e-3,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
            operator_split_on=True,
        ),
        grid=grid,
    )

    key = jax.random.key(3)
    shape = (grid.nx, grid.ny)
    amp = 1e-3
    z = jnp.zeros(shape)
    y0 = DRB2DHotIonState(
        n=_rand(key, shape, amp),
        omega=_rand(jax.random.key(4), shape, amp),
        vpar_e=z,
        vpar_i=z,
        Te=_rand(jax.random.key(5), shape, amp),
        Ti=_rand(jax.random.key(6), shape, amp),
    )
    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.05,
        nsteps=3,
        solver="dopri5",
    )
    assert jnp.all(jnp.isfinite(y1.n))
    assert jnp.all(jnp.isfinite(y1.omega))
    assert jnp.all(jnp.isfinite(y1.Te))
    assert jnp.all(jnp.isfinite(y1.Ti))


def test_drb2d_em_smoke_periodic_arakawa_spectral_no_nans():
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    model = DRB2DEMModel(
        params=DRB2DEMParams(
            omega_n=0.5,
            omega_Te=0.1,
            beta=1e-3,
            eta=1e-2,
            Dpsi=1e-3,
            curvature_on=True,
            curvature_coeff=0.2,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
            operator_split_on=True,
        ),
        grid=grid,
    )

    key = jax.random.key(7)
    shape = (grid.nx, grid.ny)
    amp = 1e-3
    z = jnp.zeros(shape)
    y0 = DRB2DEMState(
        n=_rand(key, shape, amp),
        omega=_rand(jax.random.key(8), shape, amp),
        psi=_rand(jax.random.key(9), shape, amp),
        vpar_i=z,
        Te=_rand(jax.random.key(10), shape, amp),
    )
    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.05,
        nsteps=3,
        solver="dopri5",
    )
    assert jnp.all(jnp.isfinite(y1.n))
    assert jnp.all(jnp.isfinite(y1.omega))
    assert jnp.all(jnp.isfinite(y1.psi))
