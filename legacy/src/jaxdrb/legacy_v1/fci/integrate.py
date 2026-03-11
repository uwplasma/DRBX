from __future__ import annotations

import jax
import jax.numpy as jnp

from .map import FCIBilinearMap


def map_plane_to_reference(
    f_plane: jnp.ndarray, *, map_fwd: FCIBilinearMap, steps: int
) -> jnp.ndarray:
    """Map a plane-k field back to the reference plane (k=0) coordinates.

    For a constant-B slab map, applying the forward map `steps` times corresponds to
    a shift by `steps * dz` along the field line.
    """

    def body(_, val):
        return map_fwd.apply(val)

    steps_i = jnp.asarray(steps, dtype=jnp.int32)
    return jax.lax.fori_loop(0, steps_i, body, f_plane)


def map_stack_to_reference(f_planes: jnp.ndarray, *, map_fwd: FCIBilinearMap) -> jnp.ndarray:
    """Map a stack of planes (nz, nx, ny) to the reference plane coordinates."""

    nz = f_planes.shape[0]
    steps = jnp.arange(nz)

    def map_one(k, f_plane):
        return map_plane_to_reference(f_plane, map_fwd=map_fwd, steps=k)

    return jax.vmap(map_one, in_axes=(0, 0))(steps, f_planes)


def line_integral_trapezoid(
    f_planes_ref: jnp.ndarray, *, dl: jnp.ndarray | float, periodic: bool
) -> jnp.ndarray:
    """Trapezoidal line integral along the field line in reference-plane coordinates."""

    nz = f_planes_ref.shape[0]
    weights = jnp.ones((nz,), dtype=f_planes_ref.dtype)
    if not periodic and nz >= 2:
        weights = weights.at[0].set(0.5)
        weights = weights.at[-1].set(0.5)

    accum = jnp.tensordot(weights, f_planes_ref, axes=1)
    return accum * dl


def line_integral_mapped(
    f_planes: jnp.ndarray,
    *,
    map_fwd: FCIBilinearMap,
    dl: jnp.ndarray | float,
    periodic: bool,
) -> jnp.ndarray:
    """Convenience wrapper: map all planes to the reference plane then integrate."""

    mapped = map_stack_to_reference(f_planes, map_fwd=map_fwd)
    return line_integral_trapezoid(mapped, dl=dl, periodic=periodic)
