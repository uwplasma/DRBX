from __future__ import annotations

from numbers import Number
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
    if isinstance(value, (np.ndarray, np.generic, Number, bool)):
        return False
    module = type(value).__module__
    if module == "numpy" or module.startswith("numpy."):
        return False
    return hasattr(value, "aval") or module.startswith("jax") or module.startswith("jaxlib")


def use_jax_backend(*values: Any) -> bool:
    """Select JAX when any argument is a JAX array/tracer."""

    for value in values:
        if value is not None and is_jax_array(value):
            return True
    return False


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


