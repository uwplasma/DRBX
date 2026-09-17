"""Pairwise owner-overlap parallel diffusion for angular RLPs.

This module is deliberately independent of the production FCI stencil stack.
The host builder supplies one canonical row per geometric overlap link.  At
runtime links are applied as two-owner finite-volume fluxes, which gives an
M-matrix generator after division by the two aggregate volumes.  Raw cells
which alias an aggregate owner are representation only and never receive an
independent update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


def _host_value(host: Any, name: str) -> Any:
    """Read a required field from an object or mapping."""

    if isinstance(host, dict):
        if name not in host:
            raise ValueError(f"overlap host geometry must provide {name}")
        return host[name]
    try:
        return getattr(host, name)
    except AttributeError as exc:
        raise ValueError(f"overlap host geometry must provide {name}") from exc


def _compact_owner_volume(host: Any, owner_flat_ids: np.ndarray) -> np.ndarray:
    """Normalize compact or cell-shaped owner volumes."""

    raw_shape = tuple(int(v) for v in _host_value(host, "raw_shape"))
    volume = np.asarray(_host_value(host, "owner_volume"), dtype=np.float64)
    n_owner = owner_flat_ids.size
    if volume.ndim == 1 and volume.size == n_owner:
        compact = volume
    elif volume.shape == raw_shape:
        compact = volume.reshape(-1)[owner_flat_ids]
    else:
        raise ValueError(
            "owner_volume must be compact (n_owner,) or cell-shaped raw_shape; "
            f"got {volume.shape}, expected {(n_owner,)} or {raw_shape}"
        )
    if not np.all(np.isfinite(compact)) or np.any(compact <= 0.0):
        raise ValueError("owner_volume must contain finite positive values")
    return compact


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalRlpParallelOverlapGeometry:
    """JAX payload for pairwise owner-overlap diffusion.

    ``owner_flat_ids`` are sorted raw storage indices of the unique owners.
    ``link_owner_a`` and ``link_owner_b`` index this compact owner list, and
    ``transmissibility`` contains geometry-only nonnegative link weights.
    All arrays are dynamic PyTree leaves so the payload can be passed through
    ``jax.jit``.  ``raw_shape`` is static metadata.
    """

    owner_flat_ids: Array
    owner_volume: Array
    link_owner_a: Array
    link_owner_b: Array
    transmissibility: Array
    raw_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.raw_shape)
        if len(shape) != 3 or any(v <= 0 for v in shape):
            raise ValueError("raw_shape must contain three positive dimensions")
        object.__setattr__(self, "raw_shape", shape)

        owners = np.asarray(self.owner_flat_ids, dtype=np.int32).reshape(-1)
        volume = np.asarray(self.owner_volume, dtype=np.float64).reshape(-1)
        a = np.asarray(self.link_owner_a, dtype=np.int32).reshape(-1)
        b = np.asarray(self.link_owner_b, dtype=np.int32).reshape(-1)
        tau = np.asarray(self.transmissibility, dtype=np.float64).reshape(-1)
        n_raw = int(np.prod(shape))
        if owners.size == 0 or np.any(owners < 0) or np.any(owners >= n_raw):
            raise ValueError("owner_flat_ids must be nonempty and in raw storage")
        if np.any(np.diff(owners) <= 0):
            raise ValueError("owner_flat_ids must be strictly increasing")
        if volume.size != owners.size:
            raise ValueError("owner_volume must have one entry per owner")
        if a.size != b.size or a.size != tau.size:
            raise ValueError("link arrays must have equal lengths")
        if np.any(a < 0) or np.any(a >= owners.size) or np.any(b < 0) or np.any(b >= owners.size):
            raise ValueError("link owner indices must index the compact owner list")
        if np.any(a == b):
            raise ValueError("overlap links may not connect an owner to itself")
        if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
            raise ValueError("owner_volume must contain finite positive values")
        if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
            raise ValueError("transmissibility must contain finite nonnegative values")
        object.__setattr__(self, "owner_flat_ids", owners)
        object.__setattr__(self, "owner_volume", volume)
        object.__setattr__(self, "link_owner_a", a)
        object.__setattr__(self, "link_owner_b", b)
        object.__setattr__(self, "transmissibility", tau)

    @property
    def n_owner(self) -> int:
        return int(self.owner_flat_ids.shape[0])

    @property
    def n_link(self) -> int:
        return int(self.transmissibility.shape[0])

    def tree_flatten(self):
        return (
            (
                jnp.asarray(self.owner_flat_ids, dtype=jnp.int32),
                jnp.asarray(self.owner_volume, dtype=jnp.float64),
                jnp.asarray(self.link_owner_a, dtype=jnp.int32),
                jnp.asarray(self.link_owner_b, dtype=jnp.int32),
                jnp.asarray(self.transmissibility, dtype=jnp.float64),
            ),
            (self.raw_shape,),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (raw_shape,) = aux_data
        obj = object.__new__(cls)
        object.__setattr__(obj, "raw_shape", raw_shape)
        for name, value in zip(
            ("owner_flat_ids", "owner_volume", "link_owner_a", "link_owner_b", "transmissibility"),
            children,
            strict=True,
        ):
            object.__setattr__(obj, name, value)
        return obj


def lower_rlp_parallel_overlap_geometry(host: Any) -> LocalRlpParallelOverlapGeometry:
    """Lower a host overlap graph to a JAX-native geometry payload.

    The host contract uses compact owner indices for links.  For convenience,
    links expressed as raw ``owner_flat_ids`` are also accepted when an index
    falls outside the compact range; this keeps the lowering useful with
    simple sparse host records without changing the canonical representation.
    """

    raw_shape = tuple(int(v) for v in _host_value(host, "raw_shape"))
    owners = np.asarray(_host_value(host, "owner_flat_ids"), dtype=np.int64).reshape(-1)
    if owners.size == 0:
        raise ValueError("owner_flat_ids must be nonempty")
    if np.any(owners > np.iinfo(np.int32).max):
        raise ValueError("owner_flat_ids exceed int32 index range")
    volumes = _compact_owner_volume(host, owners)
    a = np.asarray(_host_value(host, "link_owner_a"), dtype=np.int64).reshape(-1)
    b = np.asarray(_host_value(host, "link_owner_b"), dtype=np.int64).reshape(-1)
    tau = np.asarray(_host_value(host, "transmissibility"), dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size != tau.size:
        raise ValueError("link arrays must have equal lengths")
    # Canonical builders use compact owner slots.  Permit raw flat owner IDs
    # as an unambiguous fallback, but reject mixed/out-of-range references.
    if a.size and (np.any(a < 0) or np.any(b < 0)):
        raise ValueError("link owner IDs must be nonnegative")
    if a.size and (np.any(a >= owners.size) or np.any(b >= owners.size)):
        lookup = {int(flat): slot for slot, flat in enumerate(owners)}
        try:
            a = np.asarray([lookup[int(v)] for v in a], dtype=np.int64)
            b = np.asarray([lookup[int(v)] for v in b], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("link owner IDs must be compact slots or owner_flat_ids") from exc
    return LocalRlpParallelOverlapGeometry(
        owner_flat_ids=np.asarray(owners, dtype=np.int32),
        owner_volume=volumes,
        link_owner_a=np.asarray(a, dtype=np.int32),
        link_owner_b=np.asarray(b, dtype=np.int32),
        transmissibility=tau,
        raw_shape=raw_shape,
    )


# Naming aliases used by geometry/lowering code in different research lanes.
build_local_rlp_parallel_overlap_geometry = lower_rlp_parallel_overlap_geometry
compile_local_rlp_parallel_overlap_geometry = lower_rlp_parallel_overlap_geometry


def _compact_owner_values(owner_values: Array, geometry: LocalRlpParallelOverlapGeometry) -> tuple[Array, tuple[int, ...]]:
    values = jnp.asarray(owner_values)
    raw_shape = geometry.raw_shape
    if values.ndim >= 3 and tuple(values.shape[:3]) == raw_shape:
        flat = values.reshape((int(np.prod(raw_shape)),) + values.shape[3:])
        return flat[geometry.owner_flat_ids], tuple(values.shape[3:])
    if values.ndim >= 1 and values.shape[0] == geometry.n_owner:
        return values, tuple(values.shape[1:])
    raise ValueError(
        "owner_values must be compact with leading shape (n_owner,) or "
        f"cell-shaped with leading shape {raw_shape}; got {values.shape}"
    )


def local_parallel_diffusion_fci_rlp_overlap_op(
    owner_values: Array,
    geometry: LocalRlpParallelOverlapGeometry,
    chi_parallel: float | Array,
) -> Array:
    """Apply pairwise owner-overlap diffusion and return owner-sparse cells.

    Every canonical link is evaluated once.  The update is
    ``dT[a] += chi*tau*(T[b]-T[a])/V[a]`` and the equal/opposite integrated
    flux is applied to ``b``.  Thus aliases in a cell-shaped input are ignored
    and all alias slots in the result are exactly zero.
    """

    if not isinstance(geometry, LocalRlpParallelOverlapGeometry):
        raise TypeError(
            "local_parallel_diffusion_fci_rlp_overlap_op requires "
            f"LocalRlpParallelOverlapGeometry, got {type(geometry).__name__}"
        )
    compact, trailing = _compact_owner_values(owner_values, geometry)
    conductance = jnp.asarray(chi_parallel, dtype=compact.dtype) * geometry.transmissibility
    delta = compact[geometry.link_owner_b] - compact[geometry.link_owner_a]
    link_shape = (geometry.n_link,) + (1,) * len(trailing)
    scale_a = (
        conductance / geometry.owner_volume[geometry.link_owner_a]
    ).reshape(link_shape)
    scale_b = (
        conductance / geometry.owner_volume[geometry.link_owner_b]
    ).reshape(link_shape)
    result = jnp.zeros_like(compact)
    result = result.at[geometry.link_owner_a].add(delta * scale_a)
    result = result.at[geometry.link_owner_b].add(-delta * scale_b)

    output = jnp.zeros((int(np.prod(geometry.raw_shape)),) + trailing, dtype=result.dtype)
    output = output.at[geometry.owner_flat_ids].set(result)
    return output.reshape(geometry.raw_shape + trailing)


def assemble_rlp_parallel_overlap_generator(
    geometry: LocalRlpParallelOverlapGeometry,
    chi_parallel: float = 1.0,
):
    """Assemble the compact owner-space generator as a SciPy CSR matrix."""

    from scipy.sparse import coo_matrix

    a = np.asarray(geometry.link_owner_a, dtype=np.int64)
    b = np.asarray(geometry.link_owner_b, dtype=np.int64)
    g = float(chi_parallel) * np.asarray(geometry.transmissibility, dtype=np.float64)
    volume = np.asarray(geometry.owner_volume, dtype=np.float64)
    rows = np.concatenate((a, a, b, b))
    cols = np.concatenate((b, a, a, b))
    data = np.concatenate((g / volume[a], -g / volume[a], g / volume[b], -g / volume[b]))
    return coo_matrix((data, (rows, cols)), shape=(geometry.n_owner, geometry.n_owner)).tocsr()


def audit_rlp_parallel_overlap_generator(geometry: LocalRlpParallelOverlapGeometry, *, chi_parallel: float = 1.0) -> dict[str, float]:
    """Return compact matrix invariant residuals for a lowered graph."""

    matrix = assemble_rlp_parallel_overlap_generator(geometry, chi_parallel)
    volume = np.asarray(geometry.owner_volume)
    constant_residual = np.asarray(matrix @ np.ones(geometry.n_owner))
    conservation_residual = np.asarray(volume @ matrix)
    coo = matrix.tocoo()
    offdiagonal = coo.data[coo.row != coo.col]
    weighted = matrix.multiply(volume[:, None])
    symmetry_defect = weighted - weighted.T
    return {
        "constant_residual": float(np.max(np.abs(constant_residual))) if geometry.n_owner else 0.0,
        "weighted_conservation_residual": float(np.max(np.abs(conservation_residual))) if geometry.n_owner else 0.0,
        "weighted_symmetry_residual": (
            float(np.max(np.abs(symmetry_defect.data))) if symmetry_defect.nnz else 0.0
        ),
        "minimum_offdiagonal": float(np.min(offdiagonal)) if offdiagonal.size else 0.0,
        "maximum_row_sum_residual": float(np.max(np.abs(constant_residual))) if geometry.n_owner else 0.0,
    }


__all__ = [
    "LocalRlpParallelOverlapGeometry",
    "lower_rlp_parallel_overlap_geometry",
    "build_local_rlp_parallel_overlap_geometry",
    "compile_local_rlp_parallel_overlap_geometry",
    "local_parallel_diffusion_fci_rlp_overlap_op",
    "assemble_rlp_parallel_overlap_generator",
    "audit_rlp_parallel_overlap_generator",
]
