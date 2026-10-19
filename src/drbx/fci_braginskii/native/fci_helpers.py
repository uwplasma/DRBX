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


_HOST_EIG_MIN_CHUNK = 1024
_HOST_EIG_POOL: ThreadPoolExecutor | None = None
# jaxlib >= 0.11 runs the batched CPU ``geev`` custom call on one thread; up to
# 0.10 it was spread over XLA's intra-op pool (one thread per core).  At HSX
# 32^3 that made the explicit operator phase about 5x slower.
_CPU_EIG_ON_HOST = tuple(int(p) for p in jax.__version__.split(".")[:2]) >= (0, 11)


# pip NumPy's OpenBLAS is built for at most 64 threads; more Python threads
# calling into it overflow its per-thread buffers ("precompiled NUM_THREADS
# exceeded", then heap corruption), as seen on 128/256-core nodes.
_HOST_EIG_MAX_THREADS = 64


def _host_eig_threads() -> int:
    """CPUs available to this process (affinity-aware), capped at 64."""

    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(available, _HOST_EIG_MAX_THREADS))


def _host_eig(matrix):
    """Batched LAPACK ``geev`` on the host, split over one thread per core.

    NumPy releases the GIL in ``linalg.eig`` and calls LAPACK ``geev`` like
    jaxlib's CPU kernel, so each matrix's result agrees with
    ``jnp.linalg.eig`` to round-off (bitwise when NumPy and jaxlib use the
    same LAPACK build; pip wheels ship different ones and can differ in the
    last bits).  Only the batch is split across threads, as XLA's intra-op
    pool did before jaxlib 0.11; the result does not depend on the split.
    The matrices are at most 5x5, where ``geev`` does not use BLAS threading,
    so the worker threads do not oversubscribe cores.
    """

    global _HOST_EIG_POOL
    matrix = np.asarray(matrix)
    batch = matrix.reshape((-1,) + matrix.shape[-2:])
    threads = _host_eig_threads()
    pieces = min(threads, batch.shape[0] // _HOST_EIG_MIN_CHUNK)
    if pieces > 1:
        if _HOST_EIG_POOL is None:
            _HOST_EIG_POOL = ThreadPoolExecutor(threads)
        results = list(_HOST_EIG_POOL.map(np.linalg.eig, np.array_split(batch, pieces)))
        values = np.concatenate([r[0] for r in results])
        vectors = np.concatenate([r[1] for r in results])
    else:
        values, vectors = np.linalg.eig(batch)
    return (
        values.astype(np.complex128).reshape(matrix.shape[:-1]),
        vectors.astype(np.complex128).reshape(matrix.shape),
    )


def small_batched_eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``jnp.linalg.eig`` for batches of small real matrices on any backend.

    The input must already be stopped-gradient.  Results agree with
    ``jnp.linalg.eig`` on CPU to round-off.  The host path (threaded LAPACK ``geev`` via
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
