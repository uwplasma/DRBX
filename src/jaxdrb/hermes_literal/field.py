from __future__ import annotations

import jax.numpy as jnp

from .types import Field3DLayout


def empty_guarded_field(
    *,
    interior: jnp.ndarray,
    guard_width: int = 2,
) -> tuple[jnp.ndarray, Field3DLayout]:
    """Allocate a guard-inclusive field around a physical `(z, x, y)` array."""

    arr = jnp.asarray(interior, dtype=jnp.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected `(z, x, y)` field, got shape {arr.shape}.")
    nz, nx, ny = (int(v) for v in arr.shape)
    gw = int(guard_width)
    out = jnp.zeros((nz + 2 * gw, nx + 2 * gw, ny), dtype=jnp.float64)
    out = out.at[gw : gw + nz, gw : gw + nx, :].set(arr)
    layout = Field3DLayout(
        pstart=gw,
        pend=gw + nz - 1,
        xstart=gw,
        xend=gw + nx - 1,
        guard_width=gw,
    )
    return out, layout


def interior_view(field: jnp.ndarray, layout: Field3DLayout) -> jnp.ndarray:
    """Return the physical cells from a guard-inclusive field."""

    arr = jnp.asarray(field, dtype=jnp.float64)
    layout.validate(tuple(int(v) for v in arr.shape))
    return arr[layout.pstart : layout.pend + 1, layout.xstart : layout.xend + 1, :]


def pad_field3d(
    interior: jnp.ndarray,
    *,
    x_periodic: bool,
    parallel_periodic: bool,
    guard_width: int = 2,
) -> tuple[jnp.ndarray, Field3DLayout]:
    """Create a guard-inclusive field using simple edge/periodic fill.

    This is a storage/bootstrap helper only. Literal boundary operators still
    need to overwrite guard values in Hermes order before parity checks.
    """

    out, layout = empty_guarded_field(interior=interior, guard_width=guard_width)
    arr = jnp.asarray(out, dtype=jnp.float64)
    if x_periodic:
        arr = arr.at[:, : layout.guard_width, :].set(
            arr[
                :,
                layout.xend - layout.guard_width + 1 : layout.xend + 1,
                :,
            ]
        )
        arr = arr.at[:, layout.xend + 1 :, :].set(
            arr[:, layout.xstart : layout.xstart + layout.guard_width, :]
        )
    else:
        arr = arr.at[:, : layout.xstart, :].set(arr[:, layout.xstart : layout.xstart + 1, :])
        arr = arr.at[:, layout.xend + 1 :, :].set(arr[:, layout.xend : layout.xend + 1, :])
    if parallel_periodic:
        arr = arr.at[: layout.guard_width, :, :].set(
            arr[
                layout.pend - layout.guard_width + 1 : layout.pend + 1,
                :,
                :,
            ]
        )
        arr = arr.at[layout.pend + 1 :, :, :].set(
            arr[layout.pstart : layout.pstart + layout.guard_width, :, :]
        )
    else:
        arr = arr.at[: layout.pstart, :, :].set(arr[layout.pstart : layout.pstart + 1, :, :])
        arr = arr.at[layout.pend + 1 :, :, :].set(arr[layout.pend : layout.pend + 1, :, :])
    return arr, layout
