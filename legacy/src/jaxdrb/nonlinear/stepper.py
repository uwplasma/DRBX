from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def rk4_step(y, t: float, dt: float, rhs: Callable[[float, object], object]):
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, jax.tree.map(lambda yi, ki: yi + 0.5 * dt * ki, y, k1))
    k3 = rhs(t + 0.5 * dt, jax.tree.map(lambda yi, ki: yi + 0.5 * dt * ki, y, k2))
    k4 = rhs(t + dt, jax.tree.map(lambda yi, ki: yi + dt * ki, y, k3))
    return jax.tree.map(
        lambda yi, a, b, c, d: yi + (dt / 6.0) * (a + 2 * b + 2 * c + d), y, k1, k2, k3, k4
    )


@eqx.filter_jit
def implicit_midpoint_step(
    y, t: float, dt: float, rhs: Callable[[float, object], object], *, n_iter: int = 6
):
    """Fixed-point implicit midpoint step.

    This is time-reversible and preserves quadratic invariants for skew-symmetric
    operators up to solver tolerance, so it is useful for strict conservation checks.
    """

    k0 = rhs(t, y)
    y1 = jax.tree.map(lambda yi, ki: yi + dt * ki, y, k0)

    def body(_, y_curr):
        y_mid = jax.tree.map(lambda yi, yi1: 0.5 * (yi + yi1), y, y_curr)
        k_mid = rhs(t + 0.5 * dt, y_mid)
        return jax.tree.map(lambda yi, ki: yi + dt * ki, y, k_mid)

    return jax.lax.fori_loop(0, int(n_iter), body, y1)


def rk4_scan(y0, *, t0: float, dt: float, nsteps: int, rhs: Callable[[float, object], object]):
    """Fixed-step RK4 time stepping using `lax.scan` (fast under `jit`)."""

    def step(carry, _):
        t, y = carry
        y_next = rk4_step(y, t, dt, rhs)
        return (t + dt, y_next), y_next

    (t_end, y_end), ys = jax.lax.scan(step, (jnp.asarray(t0), y0), xs=None, length=int(nsteps))
    _ = t_end
    return ys, y_end
