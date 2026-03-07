from __future__ import annotations

import jax.numpy as jnp

from .types import GuardLayout


def _as_field3d(field: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"Mirror boundary helpers expect `(nz, nx, ny)` arrays, got {arr.shape}.")
    return arr


def apply_neumann_boundary_average_z(
    field: jnp.ndarray,
    *,
    layout: GuardLayout,
    lower_x: bool = True,
    upper_x: bool = True,
) -> jnp.ndarray:
    """Mirror Hermes `neumann_boundary_average_z` on `(nz, nx, ny)` arrays.

    Source of truth:
    - `/Users/rogerio/local/hermes-3/src/evolve_density.cxx`
    - `/Users/rogerio/local/hermes-3/src/evolve_pressure.cxx`

    Hermes averages over the toroidal `z` axis at the first/last interior
    radial cell, then applies that average as a Neumann-style radial boundary
    condition for the first two x guard cells. In JAX layout that average is
    over axis 0, while the radial boundary lives on axis 1.
    """

    out = _as_field3d(field)
    layout.validate(out.shape)

    if lower_x:
        x = layout.xstart
        avg = jnp.mean(out[:, x, :], axis=0)
        guard = 2.0 * avg[None, :] - out[:, x, :]
        out = out.at[:, x - 1, :].set(guard)
        for offset in range(2, layout.x_guards + 1):
            out = out.at[:, x - offset, :].set(guard)

    if upper_x:
        x = layout.xend
        avg = jnp.mean(out[:, x, :], axis=0)
        guard = 2.0 * avg[None, :] - out[:, x, :]
        out = out.at[:, x + 1, :].set(guard)
        for offset in range(2, layout.x_guards + 1):
            out = out.at[:, x + offset, :].set(guard)

    return out


def _slice_axis(arr: jnp.ndarray, axis: int, index: int) -> jnp.ndarray:
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = index
    return arr[tuple(slicer)]


def _set_axis(arr: jnp.ndarray, axis: int, index: int, values: jnp.ndarray) -> jnp.ndarray:
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = index
    return arr.at[tuple(slicer)].set(values)


def _broadcast_boundary_value(
    value: jnp.ndarray | float, target_shape: tuple[int, ...]
) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), target_shape)


def apply_neumann_field3d(
    field: jnp.ndarray,
    *,
    axis: int,
    interior_start: int,
    interior_end: int,
    spacing: jnp.ndarray | float = 1.0,
    lower_gradient: jnp.ndarray | float = 0.0,
    upper_gradient: jnp.ndarray | float = 0.0,
    guard_width: int = 2,
    apply_lower: bool = True,
    apply_upper: bool = True,
) -> jnp.ndarray:
    """Mirror the centred-field `BoundaryNeumann::apply(Field3D&)` branch.

    Source of truth:
    `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx`

    This translates the non-staggered `CELL_CENTRE` branch:

    `f(boundary) = f(interior) + delta * grad`

    with the width-2 outer guard update:

    `f(outer_guard) = f(second_interior) + 3 * delta * grad`

    The boundary direction is passed explicitly as `axis` because the active
    JAX field layout is reordered relative to Hermes/BOUT storage.
    """

    if guard_width not in (1, 2):
        raise ValueError(f"guard_width must be 1 or 2, got {guard_width}.")

    out = _as_field3d(field)
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += out.ndim
    if axis_i < 0 or axis_i >= out.ndim:
        raise ValueError(f"axis={axis} is out of bounds for shape {out.shape}.")
    if interior_start < guard_width:
        raise ValueError(
            f"interior_start={interior_start} leaves fewer than {guard_width} lower guards."
        )
    if interior_end >= out.shape[axis_i] - guard_width:
        raise ValueError(
            f"interior_end={interior_end} leaves fewer than {guard_width} upper guards."
        )

    spacing_arr = jnp.broadcast_to(jnp.asarray(spacing, dtype=jnp.float64), out.shape)
    target_shape = tuple(dim for i, dim in enumerate(out.shape) if i != axis_i)
    lower_grad = _broadcast_boundary_value(lower_gradient, target_shape)
    upper_grad = _broadcast_boundary_value(upper_gradient, target_shape)

    lower_guard = interior_start - 1
    upper_guard = interior_end + 1

    lower_delta = -_slice_axis(spacing_arr, axis_i, lower_guard)
    upper_delta = _slice_axis(spacing_arr, axis_i, upper_guard)

    if apply_lower:
        out = _set_axis(
            out,
            axis_i,
            lower_guard,
            _slice_axis(out, axis_i, interior_start) + (lower_delta * lower_grad),
        )
    if apply_upper:
        out = _set_axis(
            out,
            axis_i,
            upper_guard,
            _slice_axis(out, axis_i, interior_end) + (upper_delta * upper_grad),
        )

    if guard_width == 2:
        if apply_lower:
            out = _set_axis(
                out,
                axis_i,
                interior_start - 2,
                _slice_axis(out, axis_i, interior_start + 1) + (3.0 * lower_delta * lower_grad),
            )
        if apply_upper:
            out = _set_axis(
                out,
                axis_i,
                interior_end + 2,
                _slice_axis(out, axis_i, interior_end - 1) + (3.0 * upper_delta * upper_grad),
            )

    return out


def apply_free_o2_field3d(
    field: jnp.ndarray,
    *,
    axis: int,
    interior_start: int,
    interior_end: int,
    guard_width: int = 2,
    apply_lower: bool = True,
    apply_upper: bool = True,
) -> jnp.ndarray:
    """Mirror `BoundaryFree_O2::apply(Field3D&)` for centred fields.

    Source of truth:
    `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/boundary_standard.cxx`

    The non-staggered branch recursively extrapolates guard values from the
    last two evolved cells:

    `f_g = 2 * f_1 - f_2`

    where the updated inner guard becomes the new `f_1` for the outer guard.
    """

    if guard_width not in (1, 2):
        raise ValueError(f"guard_width must be 1 or 2, got {guard_width}.")

    out = _as_field3d(field)
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += out.ndim
    if axis_i < 0 or axis_i >= out.ndim:
        raise ValueError(f"axis={axis} is out of bounds for shape {out.shape}.")
    if interior_start < guard_width:
        raise ValueError(
            f"interior_start={interior_start} leaves fewer than {guard_width} lower guards."
        )
    if interior_end >= out.shape[axis_i] - guard_width:
        raise ValueError(
            f"interior_end={interior_end} leaves fewer than {guard_width} upper guards."
        )

    if apply_lower:
        for offset in range(1, guard_width + 1):
            guard = interior_start - offset
            in1 = guard + 1
            in2 = guard + 2
            out = _set_axis(
                out,
                axis_i,
                guard,
                (2.0 * _slice_axis(out, axis_i, in1)) - _slice_axis(out, axis_i, in2),
            )

    if apply_upper:
        for offset in range(1, guard_width + 1):
            guard = interior_end + offset
            in1 = guard - 1
            in2 = guard - 2
            out = _set_axis(
                out,
                axis_i,
                guard,
                (2.0 * _slice_axis(out, axis_i, in1)) - _slice_axis(out, axis_i, in2),
            )

    return out


def set_boundary_to_midpoint(
    field: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    layout: GuardLayout,
    apply_x: bool = True,
    apply_y: bool = True,
) -> jnp.ndarray:
    """Mirror `Field3D::setBoundaryTo(const Field3D&)`.

    Source of truth:
    `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/field/field3d.cxx`

    For each guard cell, Hermes preserves the boundary midpoint of the
    reference field:

    `0.5 * (u_g + u_i) = 0.5 * (v_g + v_i)`

    which gives `u_g = v_g + v_i - u_i`. The update is recursive for outer
    guard cells because `u_i` becomes the previously updated guard value.
    """

    out = _as_field3d(field)
    ref = _as_field3d(reference)
    if out.shape != ref.shape:
        raise ValueError(f"field shape {out.shape} != reference shape {ref.shape}")
    layout.validate(out.shape)

    if apply_x:
        for x in range(layout.xstart - 1, layout.xstart - layout.x_guards - 1, -1):
            interior = x + 1
            midpoint = 0.5 * (ref[:, x, :] + ref[:, interior, :])
            out = out.at[:, x, :].set(2.0 * midpoint - out[:, interior, :])
        for x in range(layout.xend + 1, layout.xend + layout.x_guards + 1):
            interior = x - 1
            midpoint = 0.5 * (ref[:, x, :] + ref[:, interior, :])
            out = out.at[:, x, :].set(2.0 * midpoint - out[:, interior, :])

    if apply_y:
        for y in range(layout.ystart - 1, layout.ystart - layout.y_guards - 1, -1):
            interior = y + 1
            midpoint = 0.5 * (ref[:, :, y] + ref[:, :, interior])
            out = out.at[:, :, y].set(2.0 * midpoint - out[:, :, interior])
        for y in range(layout.yend + 1, layout.yend + layout.y_guards + 1):
            interior = y - 1
            midpoint = 0.5 * (ref[:, :, y] + ref[:, :, interior])
            out = out.at[:, :, y].set(2.0 * midpoint - out[:, :, interior])

    return out
