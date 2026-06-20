from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend


_SQRT_DERIVATIVE_FLOOR = 1.0e-30


@jax.custom_jvp
def _sqrt_nonnegative_jax(value: jax.Array) -> jax.Array:
    """Return ``sqrt(max(value, 0))`` with a finite JVP at the clipping point."""

    return jnp.sqrt(jnp.maximum(value, 0.0))


@_sqrt_nonnegative_jax.defjvp
def _sqrt_nonnegative_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    primal = _sqrt_nonnegative_jax(value)
    derivative = jnp.where(
        value > 0.0,
        0.5 / jnp.sqrt(jnp.maximum(value, _SQRT_DERIVATIVE_FLOOR)),
        0.0,
    )
    return primal, derivative * tangent


def sqrt_nonnegative(value: object) -> object:
    """Backend-preserving ``sqrt(max(value, 0))`` with finite JAX tangents.

    The primal matches the standard nonnegative square-root limiter used in the
    open-field characteristic speeds. The custom JVP selects the inactive-branch
    subgradient at clipped states, avoiding ``0 * inf`` NaNs when an implicit
    Newton linearization evaluates a zero tangent at a floored temperature.
    """

    if use_jax_backend(value):
        return _sqrt_nonnegative_jax(jnp.asarray(value, dtype=jnp.float64))
    return np.sqrt(np.maximum(np.asarray(value, dtype=np.float64), 0.0))
