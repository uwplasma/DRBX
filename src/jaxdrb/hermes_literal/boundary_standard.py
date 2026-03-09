"""Literal boundary helpers from BOUT/Hermes standard boundary behavior."""

from __future__ import annotations

import jax.numpy as jnp

from .types import Field3DLayout


def apply_neumann_boundary_average_z(
    field: jnp.ndarray,
    *,
    layout: Field3DLayout,
    lower_x: bool = True,
    upper_x: bool = True,
) -> jnp.ndarray:
    """Mirror Hermes `neumann_boundary_average_z` on a guard-inclusive field.

    Source:
    - `/Users/rogerio/local/hermes-3/src/evolve_density.cxx`
    - `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx`
    """

    arr = jnp.asarray(field, dtype=jnp.float64)
    layout.validate(tuple(int(v) for v in arr.shape))
    out = arr
    if lower_x:
        x = int(layout.xstart)
        avg = jnp.mean(out[:, x, :], axis=-1, keepdims=True)
        first = 2.0 * avg - out[:, x, :]
        out = out.at[:, x - 1, :].set(first)
        for offset in range(2, layout.guard_width + 1):
            out = out.at[:, x - offset, :].set(first)
    if upper_x:
        x = int(layout.xend)
        avg = jnp.mean(out[:, x, :], axis=-1, keepdims=True)
        first = 2.0 * avg - out[:, x, :]
        out = out.at[:, x + 1, :].set(first)
        for offset in range(2, layout.guard_width + 1):
            out = out.at[:, x + offset, :].set(first)
    return out


def apply_neumann_field3d(
    field: jnp.ndarray,
    *,
    axis: int,
    interior_start: int,
    interior_end: int,
    spacing: jnp.ndarray | float,
    lower_gradient: float = 0.0,
    upper_gradient: float = 0.0,
    guard_width: int = 2,
    apply_lower: bool = True,
    apply_upper: bool = True,
) -> jnp.ndarray:
    """Literal centered Neumann guard update on raw arrays."""

    arr = jnp.asarray(field, dtype=jnp.float64)
    out = arr
    axis_i = int(axis)
    spacing_arr = jnp.asarray(spacing, dtype=jnp.float64)

    def _take(a: jnp.ndarray, idx: int) -> jnp.ndarray:
        return jnp.take(a, idx, axis=axis_i)

    def _set(a: jnp.ndarray, idx: int, value: jnp.ndarray) -> jnp.ndarray:
        sl = [slice(None)] * a.ndim
        sl[axis_i] = idx
        return a.at[tuple(sl)].set(value)

    if apply_lower:
        for offset in range(1, guard_width + 1):
            guard = interior_start - offset
            inside = guard + 1
            dx = _take(spacing_arr, inside) if spacing_arr.ndim == arr.ndim else spacing_arr
            out = _set(
                out,
                guard,
                _take(out, inside) - float(lower_gradient) * dx,
            )
    if apply_upper:
        for offset in range(1, guard_width + 1):
            guard = interior_end + offset
            inside = guard - 1
            dx = _take(spacing_arr, inside) if spacing_arr.ndim == arr.ndim else spacing_arr
            out = _set(
                out,
                guard,
                _take(out, inside) + float(upper_gradient) * dx,
            )
    return out


def set_boundary_to(
    field: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    layout: Field3DLayout,
    apply_x: bool = True,
    apply_parallel: bool = False,
) -> jnp.ndarray:
    """Mirror `Field3D::setBoundaryTo(const Field3D&)` midpoint preservation."""

    out = jnp.asarray(field, dtype=jnp.float64)
    ref = jnp.asarray(reference, dtype=jnp.float64)
    if out.shape != ref.shape:
        raise ValueError(f"field shape {out.shape} != reference shape {ref.shape}")
    layout.validate(tuple(int(v) for v in out.shape))

    if apply_x:
        for x in range(layout.xstart - 1, layout.xstart - layout.guard_width - 1, -1):
            inside = x + 1
            midpoint = 0.5 * (ref[:, x, :] + ref[:, inside, :])
            out = out.at[:, x, :].set(2.0 * midpoint - out[:, inside, :])
        for x in range(layout.xend + 1, layout.xend + layout.guard_width + 1):
            inside = x - 1
            midpoint = 0.5 * (ref[:, x, :] + ref[:, inside, :])
            out = out.at[:, x, :].set(2.0 * midpoint - out[:, inside, :])

    if apply_parallel:
        for z in range(layout.pstart - 1, layout.pstart - layout.guard_width - 1, -1):
            inside = z + 1
            midpoint = 0.5 * (ref[z, :, :] + ref[inside, :, :])
            out = out.at[z, :, :].set(2.0 * midpoint - out[inside, :, :])
        for z in range(layout.pend + 1, layout.pend + layout.guard_width + 1):
            inside = z - 1
            midpoint = 0.5 * (ref[z, :, :] + ref[inside, :, :])
            out = out.at[z, :, :].set(2.0 * midpoint - out[inside, :, :])

    return out
