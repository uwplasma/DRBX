from __future__ import annotations

from collections.abc import Callable
import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.stepper import implicit_midpoint_step


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

    ys, _ = diffeqsolve_fixed_steps(
        rhs,
        y0=y0,
        t0=t0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=1,
    )
    return jax.vmap(energy)(ys)


def energy_time_series_midpoint(
    *,
    y0,
    rhs: Callable[[float, object], object],
    energy: Callable[[object], jnp.ndarray],
    t0: float,
    dt: float,
    nsteps: int,
    n_iter: int = 6,
) -> jnp.ndarray:
    """Energy time series using implicit midpoint stepping."""

    def step(carry, _):
        t, y = carry
        y_next = implicit_midpoint_step(y, t, dt, rhs, n_iter=n_iter)
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
