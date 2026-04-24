from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def is_jax_array(value: Any) -> bool:
    """Return whether ``value`` is a JAX array or tracer.

    Several hot-path kernels mix static NumPy metric arrays with dynamic JAX
    state arrays during ``jax.jvp`` or ``jax.linearize``. Backend selection must
    prefer JAX whenever any dynamic argument is a JAX value; otherwise a single
    NumPy metric would force ``np.asarray`` on a tracer.
    """

    if value is None:
        return False
    module = type(value).__module__
    return hasattr(value, "aval") or module.startswith("jax") or module.startswith("jaxlib")


def use_jax_backend(*values: Any) -> bool:
    """Select JAX when any argument is a JAX array/tracer."""

    return any(is_jax_array(value) for value in values if value is not None)


def asarray(value: Any, *, use_jax: bool):
    """Cast to the selected floating array backend."""

    if use_jax:
        return jnp.asarray(value, dtype=jnp.float64)
    return np.asarray(value, dtype=np.float64)


def zeros_like(value: Any, *, use_jax: bool):
    """Create a float64 zeros array on the selected backend."""

    if use_jax:
        return jnp.zeros_like(jnp.asarray(value, dtype=jnp.float64), dtype=jnp.float64)
    return np.zeros_like(np.asarray(value, dtype=np.float64), dtype=np.float64)


def maximum_reduce(values: tuple[Any, ...], *, use_jax: bool):
    """Backend-preserving equivalent of ``np.maximum.reduce``."""

    if not values:
        raise ValueError("maximum_reduce requires at least one value.")
    if use_jax:
        result = jnp.asarray(values[0], dtype=jnp.float64)
        for value in values[1:]:
            result = jnp.maximum(result, jnp.asarray(value, dtype=jnp.float64))
        return result
    return np.maximum.reduce(tuple(np.asarray(value, dtype=np.float64) for value in values))
