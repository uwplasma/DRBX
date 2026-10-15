from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
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


_HOST_EIG_CHUNK = 4096
_HOST_EIG_POOL: ThreadPoolExecutor | None = None
# jaxlib >= 0.11 runs the batched CPU ``geev`` custom call on one thread; up to
# 0.10 it was spread over the intra-op pool.  At HSX 32^3 that made the
# explicit operator phase about 5x slower.
_CPU_EIG_ON_HOST = tuple(int(p) for p in jax.__version__.split(".")[:2]) >= (0, 11)


def _host_eig_threads() -> int:
    try:
        threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        threads = 0
    return max(1, threads or (os.cpu_count() or 1))


def _host_eig(matrix):
    """Batched LAPACK ``geev`` on the host, chunked over a thread pool.

    NumPy releases the GIL in ``linalg.eig`` and calls the same ``geev`` as
    jaxlib's CPU kernel, so each matrix's result is bitwise the one
    ``jnp.linalg.eig`` returns; only the batch is split across threads.
    """

    global _HOST_EIG_POOL
    matrix = np.asarray(matrix)
    batch = matrix.reshape((-1,) + matrix.shape[-2:])
    chunks = [batch[i : i + _HOST_EIG_CHUNK] for i in range(0, max(batch.shape[0], 1), _HOST_EIG_CHUNK)]
    if len(chunks) > 1:
        if _HOST_EIG_POOL is None:
            _HOST_EIG_POOL = ThreadPoolExecutor(_host_eig_threads())
        results = list(_HOST_EIG_POOL.map(np.linalg.eig, chunks))
    else:
        results = [np.linalg.eig(chunks[0])]
    values = np.concatenate([r[0] for r in results])
    vectors = np.concatenate([r[1] for r in results])
    return (
        values.astype(np.complex128).reshape(matrix.shape[:-1]),
        vectors.astype(np.complex128).reshape(matrix.shape),
    )


def small_batched_eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``jnp.linalg.eig`` for batches of small real matrices on any backend.

    The input must already be stopped-gradient.  Results are bitwise those of
    ``jnp.linalg.eig`` on CPU.  The host path (threaded LAPACK ``geev`` via
    ``pure_callback``) is used on non-CPU backends, where XLA's GPU ``eig``
    solves one matrix at a time (about 20 ms per 4x4 matrix on an RTX A4000),
    and on CPU with jaxlib >= 0.11, whose batched ``geev`` custom call is
    single-threaded.  CPU with older jaxlib keeps the in-graph custom call.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    cplx = jnp.complex128
    shapes = (
        jax.ShapeDtypeStruct(matrix.shape[:-1], cplx),
        jax.ShapeDtypeStruct(matrix.shape, cplx),
    )

    def host(m):
        return jax.pure_callback(_host_eig, shapes, m, vmap_method="broadcast_all")

    if _CPU_EIG_ON_HOST:
        return host(matrix)
    return jax.lax.platform_dependent(
        matrix, cpu=lambda m: tuple(jnp.linalg.eig(m)), default=host
    )
