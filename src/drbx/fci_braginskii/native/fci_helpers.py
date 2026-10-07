from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.fci_geometry import HaloLayout3D, LocalDomain3D

# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ...native.fci_helpers import (  # noqa: F401
    _as_float64_array,
    _as_face_flux_array,
    _validate_axis,
    local_side_plane_shape,
    _as_local_wall_array,
    _as_local_wall_int_array,
    _as_local_wall_bool_array,
    _as_local_wall_stencil_index_array,
    _as_local_wall_stencil_weight_array,
)


def local_physical_side_active(
    domain: LocalDomain3D,
    axis: int,
    side: Literal["lower", "upper"],
) -> bool | jnp.ndarray:
    """Return runtime physical-side ownership for a local boundary payload.

    The result is a Python boolean for an undecomposed axis and a traced JAX
    boolean inside an SPMD context with a configured mesh axis. It is true
    only on the runtime global side whose side kind is ``SIDE_PHYSICAL``.
    """

    axis = _validate_axis(axis)
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
    if side == "lower":
        return domain.runtime_has_physical_lower(axis)
    return domain.runtime_has_physical_upper(axis)


def _local_side_mask(
    domain: LocalDomain3D,
    layout: HaloLayout3D,
    axis: int,
    side: Literal["lower", "upper"],
) -> jnp.ndarray:
    """Build a boolean mask for the physical side-plane payload."""

    active = local_physical_side_active(domain, axis, side)
    shape = local_side_plane_shape(layout, axis)
    return jnp.broadcast_to(jnp.asarray(active, dtype=bool), shape)


def _host_eig(matrix):
    values, vectors = np.linalg.eig(np.asarray(matrix))
    return values.astype(np.complex128), vectors.astype(np.complex128)


def small_batched_eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``jnp.linalg.eig`` for batches of small real matrices on any backend.

    CPU lowers to the LAPACK ``geev`` custom call exactly as before.  Other
    backends call the same LAPACK routine on the host: XLA's GPU ``eig``
    solves one matrix at a time (about 20 ms per 4x4 matrix on an RTX A4000),
    which made the explicit operator two orders of magnitude slower than on
    CPU.  The input must already be stopped-gradient.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    cplx = jnp.complex128
    shapes = (
        jax.ShapeDtypeStruct(matrix.shape[:-1], cplx),
        jax.ShapeDtypeStruct(matrix.shape, cplx),
    )

    def host(m):
        return jax.pure_callback(_host_eig, shapes, m, vmap_method="broadcast_all")

    return jax.lax.platform_dependent(
        matrix, cpu=lambda m: tuple(jnp.linalg.eig(m)), default=host
    )
