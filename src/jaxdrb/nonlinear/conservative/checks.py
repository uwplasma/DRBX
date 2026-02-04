from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.stepper import rk4_step


@eqx.filter_jit
def energy_time_series(
    *,
    y0,
    rhs: Callable[[float, object], object],
    energy: Callable[[object], jnp.ndarray],
    t0: float,
    dt: float,
    nsteps: int,
) -> jnp.ndarray:
    """Compute E(t) along a fixed-step nonlinear evolution.

    This is intended for *quick, reviewer-proof* conservation checks in tests/examples.
    """

    def step(carry, _):
        t, y = carry
        y_next = rk4_step(y, t, dt, rhs)
        E_next = energy(y_next)
        return (t + dt, y_next), E_next

    (_, _), Es = jax.lax.scan(step, (jnp.asarray(t0), y0), xs=None, length=int(nsteps))
    return Es


def energy_drift(E: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Return simple scalar measures of energy drift for a time series E(t)."""

    E0 = E[0]
    Emin = jnp.min(E)
    Emax = jnp.max(E)
    rel_span = (Emax - Emin) / jnp.maximum(jnp.abs(E0), 1e-30)
    rel_end = (E[-1] - E0) / jnp.maximum(jnp.abs(E0), 1e-30)
    return {"rel_span": rel_span, "rel_end": rel_end, "E0": E0, "Emin": Emin, "Emax": Emax}
