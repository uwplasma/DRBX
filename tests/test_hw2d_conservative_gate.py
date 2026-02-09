from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.spectral import ddx, ddy
from jaxdrb.nonlinear.stepper import rk4_step


def _integrate_fixed_steps(model: HW2DModel, y0: HW2DState, *, dt: float, nsteps: int) -> HW2DState:
    @jax.jit
    def advance(y: HW2DState) -> HW2DState:
        def body(i, carry):
            t, y_ = carry
            y_next = rk4_step(y_, t, dt, model.rhs)
            return (t + dt, y_next)

        _, y_end = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0), y))
        return y_end

    return advance(y0)


def _mean_exb_velocity(model: HW2DModel, y: HW2DState) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi = model.phi_from_omega(y.omega)
    ux = -ddy(phi, model.grid.ky)
    uy = ddx(phi, model.grid.kx)
    return jnp.mean(ux), jnp.mean(uy)


def test_hw2d_conservative_gate_energy_mass_charge_enstrophy_momentum() -> None:
    """Validation-gate test for the ideal HW2D conservative subset.

    In periodic geometry with Arakawa bracket and no sources/sinks, this checks conservation of:
      - HW energy proxy E,
      - mean density <n> (mass proxy),
      - mean vorticity <omega> (charge/current-balance proxy),
      - enstrophy proxy Z,
      - mean E×B velocity (net momentum proxy).
    """

    nx = 64
    ny = 64
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=True)
    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(11)
    n0 = 1e-2 * jax.random.normal(key, (nx, ny))
    omega0 = 1e-2 * jax.random.normal(jax.random.split(key, 2)[1], (nx, ny))
    y0 = HW2DState(n=n0, omega=omega0)

    diag0 = model.diagnostics(y0)
    nbar0 = jnp.mean(y0.n)
    wbar0 = jnp.mean(y0.omega)
    uxbar0, uybar0 = _mean_exb_velocity(model, y0)

    y1 = _integrate_fixed_steps(model, y0, dt=0.01, nsteps=700)
    diag1 = model.diagnostics(y1)
    nbar1 = jnp.mean(y1.n)
    wbar1 = jnp.mean(y1.omega)
    uxbar1, uybar1 = _mean_exb_velocity(model, y1)

    relE = float(jnp.abs(diag1["E"] - diag0["E"]) / jnp.maximum(diag0["E"], 1e-30))
    relZ = float(jnp.abs(diag1["Z"] - diag0["Z"]) / jnp.maximum(diag0["Z"], 1e-30))
    abs_nbar = float(jnp.abs(nbar1 - nbar0))
    abs_wbar = float(jnp.abs(wbar1 - wbar0))
    abs_uxbar = float(jnp.abs(uxbar1 - uxbar0))
    abs_uybar = float(jnp.abs(uybar1 - uybar0))

    assert relE < 7e-5
    assert relZ < 7e-5
    assert abs_nbar < 3e-10
    assert abs_wbar < 3e-10
    assert abs_uxbar < 3e-10
    assert abs_uybar < 3e-10
