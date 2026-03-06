"""Hermes derivative mirror helpers.

Source of truth:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/index_derivs_interface.hxx`
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`

These helpers operate on local Hermes/BOUT arrays that still include guard
cells. They are intended for dump-backed parity work before the mirrored
operators are wired into the runtime path.
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import GuardLayout


def _broadcast_metric(arr: jnp.ndarray, shape: tuple[int, int, int], name: str) -> jnp.ndarray:
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim == 0:
        return jnp.full(shape, out, dtype=jnp.float64)
    if out.ndim == 2 and out.shape == shape[1:]:
        return jnp.broadcast_to(out[None, :, :], shape)
    if out.ndim == 3 and out.shape == shape:
        return out
    raise ValueError(f"{name} has unsupported shape {out.shape}; expected {shape[1:]} or {shape}.")


def ddx_centered_guarded(
    field: jnp.ndarray,
    dx: jnp.ndarray,
    *,
    layout: GuardLayout | None = None,
) -> jnp.ndarray:
    """Mirror of centred `DDX(f)` on guard-inclusive local arrays.

    Hermes `Coordinates::DDX` applies the index-space centred derivative and
    then divides by `dx`. For a local guard-inclusive field this means boundary
    interior cells can still use a centred stencil because the required x-guard
    values are present in the dump.
    """

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    if field_arr.ndim != 3:
        raise ValueError(f"field must have shape `(nz, nx, ny)`, got {field_arr.shape}.")
    if layout is not None:
        layout.validate(tuple(int(v) for v in field_arr.shape))

    dx_arr = _broadcast_metric(dx, tuple(int(v) for v in field_arr.shape), "dx")
    out = jnp.zeros_like(field_arr)
    if field_arr.shape[1] <= 2:
        return out
    return out.at[:, 1:-1, :].set(
        (field_arr[:, 2:, :] - field_arr[:, :-2, :]) / jnp.maximum(2.0 * dx_arr[:, 1:-1, :], 1e-30)
    )
