from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d_hot_ion import DRB2DHotIonModel, DRB2DHotIonParams, DRB2DHotIonState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def test_drb2d_hot_ion_short_turbulence_regression_no_nan():
    """Nonlinear-phase regression smoke test (hot-ion DRB2D).

    This is intentionally a light gate:
    - ensures the hot-ion branch remains numerically stable for a short run,
    - ensures the fluctuation level grows from the seeded perturbation (i.e. not trivially damped),
    - avoids brittle spectral-shape comparisons that depend on resolution and solver details.
    """

    grid = Grid2D.make(nx=32, ny=32, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DHotIonParams(
        omega_n=0.9,
        omega_Te=0.25,
        omega_Ti=0.2,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        tau_i=1.0,
        curvature_on=True,
        curvature_coeff=0.6,
        Dn=1.2e-3,
        DOmega=1.2e-3,
        DTe=1.2e-3,
        DTi=1.2e-3,
        Dn4=4e-5,
        DOmega4=4e-5,
        DTe4=4e-5,
        DTi4=4e-5,
        mu_zonal_omega=0.1,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DHotIonModel(params=params, grid=grid)

    amp = 6e-3
    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    y0 = DRB2DHotIonState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(3), shape),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
        Ti=amp * jax.random.normal(jax.random.key(5), shape),
    )

    dt = 0.02
    nsteps = 500  # t=10.0: includes a brief linear onset and a nonlinear phase on a coarse grid.
    ys, y_end = diffeqsolve_fixed_steps(
        model.rhs, y0=y0, t0=0.0, dt=dt, nsteps=nsteps, save_every=50
    )
    _ = y_end

    # Basic numerical stability checks.
    n = ys.n
    omega = ys.omega
    Te = ys.Te
    Ti = ys.Ti
    assert jnp.isfinite(n).all()
    assert jnp.isfinite(omega).all()
    assert jnp.isfinite(Te).all()
    assert jnp.isfinite(Ti).all()

    # Growth check: final omega rms should exceed the initial rms by a modest factor.
    omega_rms = jnp.sqrt(
        jnp.mean((omega - omega.mean(axis=(1, 2), keepdims=True)) ** 2, axis=(1, 2))
    )
    assert float(omega_rms[-1]) > 1.1 * float(omega_rms[0])
