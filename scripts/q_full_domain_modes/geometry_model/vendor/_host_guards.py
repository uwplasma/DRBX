"""Cheap host-concreteness guards for optional eager-only validation.

Several FCI builders validate geometry/BC payloads on the host when their
inputs are concrete and skip that validation while JAX is tracing.  They did
so by calling ``bool(tracer)``/``np.asarray(tracer)`` inside ``try`` blocks and
catching JAX's tracer-conversion errors.  Constructing those JAX errors is
expensive: the message builder walks the *entire* jaxpr under construction to
find the tracer's progenitors, so each swallowed error costs O(graph size) and
tracing a large advance becomes quadratic.

These helpers raise the *same* exception types (so existing ``except``
clauses are unchanged) without building JAX's diagnostic message.  They trace
exactly the same operations as before; only the swallowed error is cheaper.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

__all__ = ["host_bool", "host_float", "host_asarray"]


class _TracedBoolError(jax.errors.TracerBoolConversionError):
    def __init__(self) -> None:  # noqa: D401 - skip JAX's O(graph) message
        Exception.__init__(self, "traced value in host-only validation")


class _TracedArrayError(jax.errors.TracerArrayConversionError):
    def __init__(self) -> None:
        Exception.__init__(self, "traced value in host-only validation")


class _TracedConcretizationError(jax.errors.ConcretizationTypeError):
    def __init__(self) -> None:
        Exception.__init__(self, "traced value in host-only validation")


def _is_tracer(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def host_bool(value: Any) -> bool:
    """``bool(value)``; raises a cheap TracerBoolConversionError for tracers."""
    if _is_tracer(value):
        raise _TracedBoolError()
    return bool(value)


def host_float(value: Any) -> float:
    """``float(value)``; raises a cheap ConcretizationTypeError for tracers."""
    if _is_tracer(value):
        raise _TracedConcretizationError()
    return float(value)


def host_asarray(value: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """``np.asarray``; raises a cheap TracerArrayConversionError for tracers."""
    if _is_tracer(value):
        raise _TracedArrayError()
    return np.asarray(value, *args, **kwargs)
