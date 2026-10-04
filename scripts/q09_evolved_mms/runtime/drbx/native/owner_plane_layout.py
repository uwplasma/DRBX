"""Shared owner-plane ordering and periodic exchange, independent of P/Q numerics.

The validated owner layout retains the existing single-ring/single-plane owner
contract. Plane-major permutation and halo exchange preserve the P implementation
and are compatibility-exported there; Q selects its own actual donor reach.
"""
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
from jax import lax

__all__ = ['owner_layout','PlaneLayout','plane_major_permutation',
           'to_plane_major','from_plane_major','exchange_plane_halo']

def owner_layout(raw_to_owner: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(owner_ring, owner_plane, owner_theta)`` of every owner from ``raw_to_owner`` (raw cell ``(i n + j) n + k``).

    ``ring = i`` and ``plane = k`` (both must be shared by all raw cells of an owner, else ``ValueError``); ``theta`` is
    the smallest raw theta index ``j`` of the owner, an angular key whose order within one ring and plane is the angular
    order.  ``raw_to_owner`` has shape ``(n^3,)`` (``-1`` marks raw cells without an owner); the owners are
    ``0 .. max(raw_to_owner)`` and each must own at least one raw cell.
    """
    n = int(n)
    raw = np.asarray(raw_to_owner).reshape(-1)
    if raw.shape != (n ** 3,):
        raise ValueError(f"raw_to_owner must have shape ({n ** 3},), got {raw.shape}")
    raw = raw.astype(np.int64, copy=False)
    valid = raw >= 0
    if not valid.any():
        raise ValueError("raw_to_owner has no valid entry")
    n_owners = int(raw.max()) + 1
    cell = np.arange(n ** 3, dtype=np.int64)
    ii, jj, kk = cell // n ** 2, (cell // n) % n, cell % n
    owners = raw[valid]
    ring = np.full(n_owners, -1, dtype=np.int64)
    plane = np.full(n_owners, -1, dtype=np.int64)
    ring[owners] = ii[valid]
    plane[owners] = kk[valid]
    if np.any(ring < 0):
        raise ValueError("some owners have no raw cell in raw_to_owner")
    if not np.array_equal(ring[owners], ii[valid]):
        raise ValueError("raw cells of one owner do not share the same ring")
    if not np.array_equal(plane[owners], kk[valid]):
        raise ValueError("raw cells of one owner do not share the same eta plane")
    theta = np.full(n_owners, n, dtype=np.int64)
    np.minimum.at(theta, owners, jj[valid])
    return ring, plane, theta


class PlaneLayout(NamedTuple):
    """``perm[old] = new``, ``inverse[new] = old`` and the owners per plane ``m``."""

    perm: np.ndarray
    inverse: np.ndarray
    m: int


def plane_major_permutation(raw_to_owner, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """``(perm, inverse, m)``: ``new = plane * m + rank of old among the owners of its plane`` (ordered by old id).

    Raises ``ValueError`` unless every owner is single-ring and single-plane (``owner_layout``) and all planes hold
    the same number of owners. ``perm[old] = new`` and ``inverse[new] = old`` are int64 arrays."""
    _ring, plane, _theta = owner_layout(raw_to_owner, n)
    counts = np.bincount(plane, minlength=int(n))
    if len(counts) != int(n) or counts.min() != counts.max() or counts[0] == 0:
        raise ValueError(f"the eta planes must hold equally many owners, got counts {sorted(set(counts.tolist()))}")
    m = int(counts[0])
    inverse = np.lexsort((np.arange(len(plane)), plane)).astype(np.int64)
    perm = np.empty_like(inverse)
    perm[inverse] = np.arange(len(inverse), dtype=np.int64)
    return perm, inverse, m


def to_plane_major(values, inverse):
    """Owner array (leading axis old ids) -> plane-major order: ``out[new] = values[inverse[new]]``."""
    return values[inverse]


def from_plane_major(values, perm):
    """Plane-major owner array -> old owner order: ``out[old] = values[perm[old]]``."""
    return values[perm]


# --------------------------------------------------------------------------
# Halo exchange and mesh
# --------------------------------------------------------------------------

def exchange_plane_halo(owned, halo: int, axis_name: str, n_shards: int):
    """``(p, m, F...)`` owned planes -> ``(p + 2 halo, m, F...)``: ``[lower halo | owned | upper halo]`` on the
    periodic ring of ``n_shards`` shards (call inside ``shard_map`` over ``axis_name``).

    The upper ``halo`` planes go to shard ``s + 1`` (its lower halo) and the lower ones to shard ``s - 1`` (its upper
    halo), by two ``lax.ppermute`` calls (needs ``p >= halo``). With one shard the halos are a local periodic wrap
    without collectives (any ``p``)."""
    halo = int(halo)
    p = owned.shape[0]
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if halo == 0:
        return owned
    if int(n_shards) == 1:
        return owned[np.arange(-halo, p + halo) % p]
    if p < halo:
        raise ValueError(f"the halo ({halo}) is wider than the owned block ({p} planes)")
    lower = lax.ppermute(owned[p - halo:], axis_name, [(s, (s + 1) % n_shards) for s in range(n_shards)])
    upper = lax.ppermute(owned[:halo], axis_name, [(s, (s - 1) % n_shards) for s in range(n_shards)])
    return jnp.concatenate([lower, owned, upper], axis=0)
