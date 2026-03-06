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
