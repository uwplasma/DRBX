"""Tracer-safe host conversions for optional eager-only validation.

JAX builds tracer-conversion error messages by walking the whole jaxpr under
construction, so catching them inside a large trace makes tracing quadratic.
These helpers raise the same exception types with a constant message instead.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

__all__ = ["host_bool", "host_asarray"]


class _ConstantMessage:
    def __init__(self) -> None:
        # Skip the JAX base-class __init__, which builds a jaxpr-walking message.
        Exception.__init__(self, "traced value in host-only validation")


class _TracedBoolError(_ConstantMessage, jax.errors.TracerBoolConversionError):
    pass


class _TracedArrayError(_ConstantMessage, jax.errors.TracerArrayConversionError):
    pass


def host_bool(value: Any) -> bool:
    """``bool(value)``; raises a cheap ``TracerBoolConversionError`` for tracers."""
    if isinstance(value, jax.core.Tracer):
        raise _TracedBoolError()
    return bool(value)


def host_asarray(value: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """``np.asarray``; raises a cheap ``TracerArrayConversionError`` for tracers."""
    if isinstance(value, jax.core.Tracer):
        raise _TracedArrayError()
    return np.asarray(value, *args, **kwargs)
