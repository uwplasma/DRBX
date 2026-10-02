"""eta-sharded execution of the combined perpendicular RHS (P08 step 6, stage A).

Design: ``work/p08_step6_sharding_20261002/design.md`` (sections Layout, Sharded plan, Verification 1-2).
The RHS (:func:`drbx.native.fci_perpendicular_rhs.perpendicular_rhs`) runs unchanged on a *local* plan per shard
inside ``jax.shard_map`` over a one-axis mesh ``"z"``; the operators are not touched.

Layout
------
Every owner lies in one eta plane (``owner_layout``) and every plane holds the same number ``m`` of owners, so the
owners are relabelled *plane-major*: ``new = plane * m + rank`` (:func:`plane_major_permutation`). Global owner arrays
then have shape ``(n * m, F)`` and shard ``s`` of ``Sz`` owns planes ``[s p, (s + 1) p)``, ``p = n / Sz >= halo``.
The local owner array has ``(p + 2 h) * m + 1`` rows: ``[h lower halo planes | p owned planes | h upper halo planes
| trash row]``; the global plane of local plane ``e`` is ``(s p - h + e) mod n`` (eta is periodic) and the trailing
trash row is zero and absorbs padding. :func:`exchange_plane_halo` fills the halos with two ``ppermute`` calls on
the periodic ring (a local wrap for one shard).

Sharded plan
------------
:func:`shard_perpendicular_plan` lowers a global :class:`PerpendicularPlan` (host, NumPy) into one local plan per
shard, stacked along a leading shard axis:

* raw cells (P05 centred, P06 q1) are kept when their owner is owned (cells are single-plane);
* faces (P05 jump, P06 characteristic, P07 flux) are kept when either side's owner is owned. A face on a block
  boundary is therefore computed on *both* shards (the design's halo of 2 planes is not enough for this: source rows read
  eta planes within +-2 of the plane of the owner they are anchored at, and the side row anchored at the halo owner of a
  boundary face reaches 3 planes beyond the block; :data:`DEFAULT_HALO` is 3); the scatter into the side the shard does not own goes to the trash
  row, so each owned owner receives exactly its single-device contributions (only the summation order of the
  scatter-adds can differ). Halo rows are never scatter targets, they are only gathered from;
* source rows / tensor targets / integrated rows / Neumann rows are kept when an entity kept above reads their output
  slot; slots, boundary tables and all owner / donor ids are renumbered locally. Every donor with a non-zero
  weight must lie in the extended window ``[s p - h, (s + 1) p + h)``; otherwise the lowering raises, reporting the
  offending eta offsets (zero-weight padding donors are redirected to the trash row);
* all shards are padded to identical shapes and batch structure. Padded cells and faces copy a real entity's geometry
  but carry zero volume / weight and the trash owner; padded faces are flagged ``collapsed`` (so the P06 counters skip
  them) and ``p07_valid = False`` (so P05 does not jump over them); padded source rows write unused dummy slots;
  padded Neumann / integrated-row targets are out of range (``.at[].set`` drops them), as are the padded entries of
  ``FacePlan.wall_faces`` (local wall-face indices, padded with the out-of-range face index; the P06 wall solve gathers
  clipped and scatters / counts only the in-range entries).

Tensor-encoded sources keep their grid-global eta tables: their per-apply theta tables are still built for all ``n``
planes (planes outside the window gather the trash row), so a sharded apply does not yet reduce that cost.

Diagnostics of :func:`sharded_perpendicular_rhs` are returned *per shard* (arrays of shape ``(Sz,)``, not reduced): the
P06 counters count a block-boundary face on both shards, ``antisymmetry`` is a per-shard maximum.

Limits (stage A): no ``raw_pairs``, ``jump_mask`` or ``face_multiplier`` overrides (the plan's own defaults apply);
every shard must keep at least one cell and face of each part the plan holds (a bounded owner closure must cover every
shard).
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from drbx.native.fci_perpendicular_integrated_rows import IntegratedFaceBatch
from drbx.native.fci_perpendicular_plane_preconditioner import owner_layout
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
from drbx.native.fci_perpendicular_rhs import (
    FIELDS, PerpendicularParams, PerpendicularTerms, _TERM_ORDER, _kinds, _validate, perpendicular_columns,
    perpendicular_rhs)
from drbx.native.fci_perpendicular_source_rows import SourceRowBatch, SourceRowPayload
from drbx.native.fci_perpendicular_tensor_rows import TensorRowBatch
from drbx.stencils.operator_plan import (
    CellPlan, FacePlan, IntegratedRows, NeumannRows, P07Plan, PerpendicularPlan)

__all__ = [
    "AXIS", "DEFAULT_HALO", "PlaneLayout", "ShardedPerpendicularPlan", "exchange_plane_halo", "from_plane_major",
    "local_plan", "make_plane_mesh", "place_sharded", "plane_major_permutation", "shard_boundary_data",
    "shard_perpendicular_plan", "sharded_perpendicular_rhs", "to_plane_major"]

AXIS = "z"
#: halo planes of the duplicated-face scheme. Every source row of an owner reads eta planes within +-2 of the owner's
#: plane; a face on a block boundary is kept on both shards, and its side row anchored at the *halo* owner (needed for
#: the jump / characteristic correction of the owned side) therefore reaches 2 planes beyond the first halo plane.
DEFAULT_HALO = 3


# --------------------------------------------------------------------------
# Plane-major relabelling
# --------------------------------------------------------------------------

from .owner_plane_layout import (
    PlaneLayout, plane_major_permutation, to_plane_major, from_plane_major, exchange_plane_halo)


def make_plane_mesh(n_shards: int) -> Mesh:
    """One-axis mesh ``("z",)`` over the first ``n_shards`` devices."""
    devices = np.asarray(jax.devices()[:int(n_shards)], dtype=object)
    if devices.size < int(n_shards):
        raise RuntimeError(f"{n_shards} shards need {n_shards} devices, only {devices.size} are available")
    return Mesh(devices, (AXIS,))


# --------------------------------------------------------------------------
# The sharded plan
# --------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class ShardedPerpendicularPlan:
    """Stacked local plans (leading shard axis ``Sz`` on every leaf) plus static layout metadata.

    ``dirichlet_ids`` / ``neumann_ids`` ``(Sz, Qd)`` / ``(Sz, Qn)`` are the global rows of ``plan.dirichlet_points`` /
    ``plan.neumann_points`` that each shard's local boundary tables hold (``shard_boundary_data`` gathers by them).
    ``layout`` is the host-side plane-major permutation (not part of the pytree)."""

    def __init__(self, plan: PerpendicularPlan, dirichlet_ids, neumann_ids, n: int, m: int, n_shards: int,
                 halo: int, layout: Optional[PlaneLayout] = None):
        self.plan, self.dirichlet_ids, self.neumann_ids = plan, dirichlet_ids, neumann_ids
        self.n, self.m, self.n_shards, self.halo = int(n), int(m), int(n_shards), int(halo)
        self.layout = layout

    @property
    def planes_per_shard(self) -> int:
        return self.n // self.n_shards

    @property
    def local_rows(self) -> int:
        """Rows of the local extended owner arrays, trash row included."""
        return (self.planes_per_shard + 2 * self.halo) * self.m + 1

    def tree_flatten(self):
        return (self.plan, self.dirichlet_ids, self.neumann_ids), (self.n, self.m, self.n_shards, self.halo)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def place_sharded(tree, mesh: Mesh):
    """``device_put`` a sharded plan / boundary data (leading axis = shards) with ``P("z")``."""
    return jax.device_put(tree, NamedSharding(mesh, P(AXIS)))


def local_plan(sharded: ShardedPerpendicularPlan, shard: int) -> PerpendicularPlan:
    """The local plan of one shard (for debugging; it runs with ``perpendicular_rhs`` on its extended owner array)."""
    return jax.tree_util.tree_map(lambda a: np.asarray(a)[shard], sharded.plan)


class _Ctx:
    """Plane layout shared by all shards."""

    def __init__(self, layout: PlaneLayout, n: int, n_shards: int, halo: int):
        n, n_shards, halo = int(n), int(n_shards), int(halo)
        if n_shards < 1 or n % n_shards:
            raise ValueError(f"the shard count {n_shards} must divide the plane count {n}")
        if halo < 0 or (n_shards > 1 and n // n_shards < halo):
            raise ValueError(f"the owned block ({n // n_shards} planes) must be at least the halo ({halo})")
        self.layout, self.n, self.n_shards, self.halo = layout, n, n_shards, halo
        self.m, self.p = layout.m, n // n_shards
        self.plane_of_old = layout.perm // layout.m
        self.rank_of_old = layout.perm % layout.m
        self.n_rows = (self.p + 2 * halo) * layout.m + 1
        self.trash = self.n_rows - 1


class _Shard:
    """Owner maps of one shard: ``donor_map`` (old id -> local row of its canonical copy, trash outside the window)
    and ``owner_map`` (old id -> owned local row, trash for non-owned owners)."""

    def __init__(self, ctx: _Ctx, s: int):
        n, m, p, h = ctx.n, ctx.m, ctx.p, ctx.halo
        self.ctx, self.s = ctx, int(s)
        self.lo, self.hi = s * p, (s + 1) * p
        plane, rank = ctx.plane_of_old, ctx.rank_of_old
        self.owned = (plane >= self.lo) & (plane < self.hi)
        rel = (plane - (self.lo - h)) % n
        in_window = self.owned | (rel < p + 2 * h)
        row = np.where(self.owned, h + plane - self.lo, rel) * m + rank
        self.trash = ctx.trash
        self.donor_map = np.where(in_window, row, ctx.trash).astype(np.int64)
        self.owner_map = np.where(self.owned, row, ctx.trash).astype(np.int64)
        planes = (self.lo - h + np.arange(p + 2 * h)) % n
        self.window_plane = np.zeros(n, dtype=bool)
        self.window_plane[planes] = True
        self.ext_old = ctx.layout.inverse[(planes[:, None] * m + np.arange(m)[None, :]).ravel()]

    def excess(self, planes) -> np.ndarray:
        """Signed number of planes by which ``planes`` lie beyond the owned block (0 inside)."""
        n = self.ctx.n
        planes = np.asarray(planes)
        up, down = (planes - (self.hi - 1)) % n, (self.lo - planes) % n
        out = np.where(up <= down, up, -down)
        return np.where((planes >= self.lo) & (planes < self.hi), 0, out)

    def side_owned(self, owner) -> np.ndarray:
        owner = np.asarray(owner)
        return (owner >= 0) & self.owned[np.maximum(owner, 0)]

    def side_map(self, owner) -> np.ndarray:
        """Face owner ids -> owned local row; the side this shard does not own goes to the trash row, -1 stays."""
        owner = np.asarray(owner)
        return np.where(owner >= 0, self.owner_map[np.maximum(owner, 0)], -1).astype(owner.dtype)

    def vector(self, values, fill=1.0) -> np.ndarray:
        """Per-owner vector (old order) -> local extended order plus the trash row."""
        values = np.asarray(values)
        return np.concatenate([values[self.ext_old], np.full(1, fill, dtype=values.dtype)])

    def donors(self, ids, nonzero, what: str) -> np.ndarray:
        """Donor ids -> local rows; zero-weight donors go to the trash row, a weighted donor outside the window raises."""
        ids = np.asarray(ids)
        local = self.donor_map[ids]
        bad = np.asarray(nonzero) & (local == self.trash)
        if bad.any():
            raise ValueError(
                f"{what}: shard {self.s} (planes [{self.lo}, {self.hi})) has {int(bad.sum())} weighted donors outside "
                f"the extended window of halo {self.ctx.halo}; eta offsets beyond the owned block: "
                f"{sorted(set(self.excess(self.ctx.plane_of_old[ids[bad]]).tolist()))}")
        return np.where(nonzero, local, self.trash).astype(ids.dtype)


# --------------------------------------------------------------------------
# Localization (unpadded local plan of one shard, global boundary ids)
# --------------------------------------------------------------------------

def _select_sources(payload: SourceRowPayload, need_v, need_g, sh: _Shard, what: str):
    """Keep the sources / tensor targets that write a needed slot; ``(payload, value map, gradient map)``.

    The local payload has dense slots in kept order, donors renumbered, and still *global* boundary query ids."""
    vmap = np.full(payload.n_targets, -1, dtype=np.int64)
    gmap = np.full(payload.n_gradient_targets, -1, dtype=np.int64)
    counts = [0, 0]

    def assign(slots, table, k):
        new = np.arange(counts[k], counts[k] + slots.size, dtype=np.int64).reshape(slots.shape)
        table[slots.ravel()] = new.ravel()
        counts[k] += slots.size
        return new.astype(np.int32)

    batches = []
    for b in payload.batches:
        keep = need_v[b.value_slots].any(axis=1)
        if b.gradient_slots is not None:
            keep |= need_g[b.gradient_slots].any(axis=1)
        k = np.flatnonzero(keep)
        nonzero = (b.value[k] != 0).any(axis=1)
        if b.gradient is not None:
            nonzero |= (b.gradient[k] != 0).any(axis=(1, 2))
        batches.append(SourceRowBatch(
            assign(b.value_slots[k], vmap, 0), sh.donors(b.donor_ids[k], nonzero, what + " source rows"),
            b.value[k], None if b.gradient_slots is None else assign(b.gradient_slots[k], gmap, 1),
            None if b.gradient is None else b.gradient[k],
            None if b.donor_query is None else b.donor_query[k],
            None if b.target_query is None else b.target_query[k]))
    tables = payload.tensor_tables
    tensor = []
    if payload.tensor_batches:
        ring_nz = (tables.ring_value != 0) | (tables.ring_derivative != 0)
    for b in payload.tensor_batches:
        keep = need_v[b.value_slots]
        if b.gradient_slots is not None:
            keep |= need_g[b.gradient_slots]
        k = np.flatnonzero(keep)
        if b.t_ring is None:
            bad = ~sh.window_plane[tables.eta_index[b.t_eta[k]]]
        else:
            entry = b.t_ring[k]
            bad = ring_nz[entry] & (sh.donor_map[tables.ring_owner[entry]] == sh.trash)
        if bad.any():
            raise ValueError(f"{what} tensor rows: shard {sh.s} has targets reading eta planes outside the extended "
                             f"window of halo {sh.ctx.halo}")
        take = lambda a: None if a is None else a[k]
        tensor.append(TensorRowBatch(
            b.family, b.dr_divisor, assign(b.value_slots[k], vmap, 0),
            None if b.gradient_slots is None else assign(b.gradient_slots[k], gmap, 1), take(b.t_eta),
            take(b.t_radial), take(b.layer_id), take(b.t_theta), take(b.t_ring)))
    if tables is not None:
        raw = np.asarray(tables.raw_to_owner).astype(np.int64)
        tables = replace(
            tables, raw_to_owner=np.where(raw >= 0, sh.donor_map[np.maximum(raw, 0)], sh.trash).astype(np.int32),
            ring_owner=sh.donor_map[np.asarray(tables.ring_owner)].astype(np.int32))
    local = SourceRowPayload(tuple(batches), counts[0], counts[1], payload.n_boundary_queries, tuple(tensor), tables)
    return local, vmap, gmap


def _slots(table, slots, what: str) -> np.ndarray:
    out = table[slots]
    if (out < 0).any():
        raise ValueError(f"{what}: a needed row slot has no source row")
    return out.astype(np.int32)


def _local_neumann(rows: Optional[NeumannRows], keep, sh: _Shard, what: str) -> Optional[NeumannRows]:
    """Rows ``keep`` (indices) of a Neumann row set, donors localized, boundary ids still global."""
    if rows is None:
        return None
    k = np.asarray(keep, dtype=np.int64)
    nonzero = (rows.value_weights[k] != 0) | (rows.gradient_weights[k] != 0).any(axis=1)
    return NeumannRows(sh.donors(rows.donor_ids[k], nonzero, what), rows.value_weights[k], rows.gradient_weights[k],
                       rows.boundary_ids[k], rows.boundary_value_weights[k], rows.boundary_gradient_weights[k])


def _local_cells(c: CellPlan, sh: _Shard) -> CellPlan:
    kc = np.flatnonzero(sh.owned[c.raw_owner])
    if not len(kc):
        raise ValueError(f"shard {sh.s} owns no raw cell of the plan: a bounded plan must cover every shard")
    need_v = np.zeros(c.rows.n_targets, dtype=bool)
    need_g = np.zeros(c.rows.n_gradient_targets, dtype=bool)
    need_v[c.value_slot[kc]] = True
    need_g[c.gradient_slot[kc]] = True
    rows, vmap, gmap = _select_sources(c.rows, need_v, need_g, sh, "cells")
    loc = np.full(len(c.raw_ids), -1, dtype=np.int64)
    loc[kc] = np.arange(len(kc))
    neumann = neumann_cell = None
    if c.neumann is not None:
        kr = np.flatnonzero(loc[c.neumann_cell] >= 0)
        neumann = _local_neumann(c.neumann, kr, sh, "cells Neumann rows")
        neumann_cell = loc[c.neumann_cell[kr]].astype(np.int32)
    else:
        neumann_cell = c.neumann_cell[:0]
    take = lambda a: np.asarray(a)[kc]
    return CellPlan(
        rows=rows, value_slot=_slots(vmap, c.value_slot[kc], "cells"), gradient_slot=_slots(gmap, c.gradient_slot[kc], "cells"),
        conditioned=take(c.conditioned), neumann=neumann, neumann_cell=neumann_cell, raw_ids=take(c.raw_ids),
        raw_owner=sh.owner_map[c.raw_owner[kc]].astype(c.raw_owner.dtype), raw_volume=take(c.raw_volume),
        owner_volume=sh.vector(c.owner_volume), evolution_volume=sh.vector(c.evolution_volume), h=take(c.h),
        jac=take(c.jac), B=take(c.B), K=take(c.K), J=take(c.J), weight=take(c.weight),
        evolution_weight=take(c.evolution_weight))


def _local_faces(f: FacePlan, sh: _Shard) -> tuple[FacePlan, np.ndarray]:
    kf = np.flatnonzero(sh.side_owned(f.lower_owner) | sh.side_owned(f.upper_owner))
    if not len(kf):
        raise ValueError(f"shard {sh.s} owns no face of the plan: a bounded plan must cover every shard")
    Qf = f.nodes
    need_v = np.zeros(f.rows.n_targets, dtype=bool)
    need_g = np.zeros(f.rows.n_gradient_targets, dtype=bool)
    lp, up = f.lower_present[kf], f.upper_present[kf]
    need_v[f.common_value_slot[kf].ravel()] = True
    need_v[f.lower_slot[kf][lp].ravel()] = True
    need_v[f.upper_slot[kf][up].ravel()] = True
    need_g[f.common_gradient_slot[kf].ravel()] = True
    rows, vmap, gmap = _select_sources(f.rows, need_v, need_g, sh, "faces")
    loc = np.full(len(f.census_row), -1, dtype=np.int64)
    loc[kf] = np.arange(len(kf))

    def side(slots, present):
        return _checked(np.where(present[:, None], vmap[slots], 0), "faces")

    def neumann(rows_, target):
        if rows_ is None:
            return None, target[:0]
        kr = np.flatnonzero(loc[target // Qf] >= 0)
        return (_local_neumann(rows_, kr, sh, "faces Neumann rows"),
                (loc[target[kr] // Qf] * Qf + target[kr] % Qf).astype(np.int32))

    common_neumann, common_target = neumann(f.common_neumann, f.common_neumann_target)
    side_neumann, side_target = neumann(f.side_neumann, f.side_neumann_target)
    take = lambda a: np.asarray(a)[kf]
    return FacePlan(
        rows=rows, common_value_slot=_slots(vmap, f.common_value_slot[kf], "faces"),
        common_gradient_slot=_slots(gmap, f.common_gradient_slot[kf], "faces"),
        lower_slot=side(f.lower_slot[kf], lp), upper_slot=side(f.upper_slot[kf], up), lower_present=take(f.lower_present),
        upper_present=take(f.upper_present), common_conditioned=take(f.common_conditioned),
        lower_conditioned=take(f.lower_conditioned), upper_conditioned=take(f.upper_conditioned),
        fallback_query=take(f.fallback_query), common_neumann=common_neumann, common_neumann_target=common_target,
        side_neumann=side_neumann, side_neumann_target=side_target, census_row=take(f.census_row),
        p07_id=take(f.p07_id), axis=take(f.axis), lower_owner=sh.side_map(take(f.lower_owner)),
        upper_owner=sh.side_map(take(f.upper_owner)), wall=take(f.wall),
        wall_faces=np.flatnonzero(take(f.wall)).astype(np.int32), collapsed=take(f.collapsed),
        p07_valid=take(f.p07_valid), face_multiplier=take(f.face_multiplier), h=take(f.h), jac=take(f.jac),
        weight=take(f.weight), J=take(f.J), B=take(f.B), K=take(f.K), owner_volume=sh.vector(f.owner_volume),
        has_missing_side=f.has_missing_side), loc


def _checked(out, what: str) -> np.ndarray:
    out = np.asarray(out)
    if (out < 0).any():
        raise ValueError(f"{what}: a needed row slot has no source row")
    return out.astype(np.int32)


def _local_p07(p: P07Plan, sh: _Shard, face_loc: Optional[np.ndarray]) -> P07Plan:
    kp = np.flatnonzero(sh.side_owned(p.lower_owner) | sh.side_owned(p.upper_owner))
    if not len(kp):
        raise ValueError(f"shard {sh.s} owns no P07 face of the plan: a bounded plan must cover every shard")
    loc = np.full(len(p.p07_id), -1, dtype=np.int64)
    loc[kp] = np.arange(len(kp))
    batches = []
    for b in p.rows.batches:
        k = np.flatnonzero(loc[b.face_ids] >= 0)
        batches.append(IntegratedFaceBatch(
            loc[b.face_ids[k]].astype(b.face_ids.dtype), sh.donors(b.donor_ids[k], b.weights[k] != 0, "P07 rows"),
            b.weights[k], b.boundary_donor_ids[k], b.tangential_ids[k], b.tangential_weights[k], b.conditioned[k]))
    rows = IntegratedRows(tuple(batches), sh.side_map(np.asarray(p.rows.lower_owner)[kp]),
                          sh.side_map(np.asarray(p.rows.upper_owner)[kp]), p.rows.boundary_query_count, len(kp))
    neumann, neumann_face, integrand = None, p.neumann_face[:0], p.integrand[:0]
    if p.neumann is not None:
        kn = np.flatnonzero(loc[p.neumann_face] >= 0)
        neumann = _local_neumann(p.neumann, (kn[:, None] * 9 + np.arange(9)[None, :]).ravel(), sh, "P07 Neumann rows")
        neumann_face = loc[p.neumann_face[kn]].astype(np.int32)
        integrand = p.integrand[kn]
    face_index = np.asarray(p.face_index)[kp]
    face_index = (np.where(face_index >= 0, face_loc[np.maximum(face_index, 0)], -1) if face_loc is not None
                  else np.full(len(kp), -1)).astype(np.asarray(p.face_index).dtype)
    take = lambda a: np.asarray(a)[kp]
    return P07Plan(
        rows=rows, conditioned=take(p.conditioned), neumann=neumann, neumann_face=neumann_face, integrand=integrand,
        p07_id=take(p.p07_id), census_row=take(p.census_row), face_index=face_index, family=take(p.family),
        lower_owner=rows.lower_owner, upper_owner=rows.upper_owner, owner_volume=sh.vector(p.owner_volume))


def _visit_dirichlet(plan: PerpendicularPlan, fn) -> PerpendicularPlan:
    """``plan`` with ``fn`` applied to every array of ids into ``plan.dirichlet_points``."""
    def payload(rows):
        batches = tuple(b if b.donor_query is None else b._replace(donor_query=fn(b.donor_query),
                                                                    target_query=fn(b.target_query))
                        for b in rows.batches)
        return replace(rows, batches=batches)

    cells, faces, p07 = plan.cells, plan.faces, plan.p07
    if cells is not None:
        cells = replace(cells, rows=payload(cells.rows))
    if faces is not None:
        faces = replace(faces, rows=payload(faces.rows), fallback_query=fn(faces.fallback_query))
    if p07 is not None:
        batches = tuple(b._replace(boundary_donor_ids=fn(b.boundary_donor_ids), tangential_ids=fn(b.tangential_ids))
                        for b in p07.rows.batches)
        p07 = replace(p07, rows=replace(p07.rows, batches=batches))
    return replace(plan, cells=cells, faces=faces, p07=p07)


def _visit_neumann(plan: PerpendicularPlan, fn) -> PerpendicularPlan:
    """``plan`` with ``fn`` applied to every array of ids into ``plan.neumann_points``."""
    def rows(r):
        return None if r is None else replace(r, boundary_ids=fn(r.boundary_ids))

    cells, faces, p07 = plan.cells, plan.faces, plan.p07
    if cells is not None:
        cells = replace(cells, neumann=rows(cells.neumann))
    if faces is not None:
        faces = replace(faces, common_neumann=rows(faces.common_neumann), side_neumann=rows(faces.side_neumann))
    if p07 is not None:
        p07 = replace(p07, neumann=rows(p07.neumann))
    return replace(plan, cells=cells, faces=faces, p07=p07)


def _localize_tables(plan: PerpendicularPlan):
    """Renumber the boundary tables to the rows the plan references; ``(plan, dirichlet gids, neumann gids)``."""
    out = []
    for visit, points in ((_visit_dirichlet, plan.dirichlet_points), (_visit_neumann, plan.neumann_points)):
        used = []
        visit(plan, lambda a: (used.append(np.asarray(a).ravel()), a)[1])
        gid = np.unique(np.concatenate(used)) if used else np.zeros(0, dtype=np.int64)
        table = np.full(len(points), -1, dtype=np.int64)
        table[gid] = np.arange(len(gid))
        plan = visit(plan, lambda a, table=table: table[a].astype(np.asarray(a).dtype))
        out.append(gid.astype(np.int64))
    plan = replace(plan, dirichlet_points=np.asarray(plan.dirichlet_points)[out[0]],
                   neumann_points=np.asarray(plan.neumann_points)[out[1]])
    return plan, out[0], out[1]


def _localize(plan: PerpendicularPlan, sh: _Shard):
    cells = None if plan.cells is None else _local_cells(plan.cells, sh)
    faces = face_loc = None
    if plan.faces is not None:
        faces, face_loc = _local_faces(plan.faces, sh)
    p07 = None if plan.p07 is None else _local_p07(plan.p07, sh, face_loc)
    local = PerpendicularPlan(cells, faces, p07, plan.dirichlet_points, plan.neumann_points, plan.n)
    return _localize_tables(local)


# --------------------------------------------------------------------------
# Padding to identical shapes
# --------------------------------------------------------------------------

def _pad(a, count: int, fill=0):
    """Pad the leading axis of ``a`` to ``count`` rows with ``fill``."""
    a = np.asarray(a)
    out = np.full((count,) + a.shape[1:], fill, dtype=a.dtype)
    out[:len(a)] = a
    return out


def _pad_like_first(a, count: int):
    """Pad the leading axis to ``count`` rows by copies of row 0 (a valid row, so padded entities stay finite)."""
    a = np.asarray(a)
    return np.concatenate([a, np.repeat(a[:1], count - len(a), axis=0)]) if count > len(a) else a


def _pad_neumann(rows: Sequence[Optional[NeumannRows]], count: int, trash: int) -> list:
    if rows[0] is None:
        return [None] * len(rows)
    return [NeumannRows(_pad(r.donor_ids, count, trash), _pad(r.value_weights, count), _pad(r.gradient_weights, count),
                        _pad(r.boundary_ids, count), _pad(r.boundary_value_weights, count),
                        _pad(r.boundary_gradient_weights, count)) for r in rows]


def _neumann_count(rows) -> int:
    return max((0 if r is None else len(r.donor_ids)) for r in rows)


def _pad_payloads(payloads: Sequence[SourceRowPayload], trash: int, n_queries: int) -> list:
    nb, nt = len(payloads[0].batches), len(payloads[0].tensor_batches)
    smax = [max(len(p.batches[i].value_slots) for p in payloads) for i in range(nb)]
    tmax = [max(len(p.tensor_batches[i].value_slots) for p in payloads) for i in range(nt)]

    def dummies(p, grad):
        total = 0
        for i, b in enumerate(p.batches):
            slots = b.gradient_slots if grad else b.value_slots
            total += 0 if slots is None else (smax[i] - len(b.value_slots)) * slots.shape[1]
        for i, b in enumerate(p.tensor_batches):
            if not grad or b.gradient_slots is not None:
                total += tmax[i] - len(b.value_slots)
        return total

    n_targets = max(p.n_targets + dummies(p, False) for p in payloads)
    n_gradient = max(p.n_gradient_targets + dummies(p, True) for p in payloads)
    out = []
    for p in payloads:
        nxt = [p.n_targets, p.n_gradient_targets]

        def dummy(count, shape, k):
            slots = np.arange(nxt[k], nxt[k] + count, dtype=np.int32).reshape(shape)
            nxt[k] += count
            return slots

        batches = []
        for i, b in enumerate(p.batches):
            if smax[i] == 0:
                continue
            k, q = smax[i] - len(b.value_slots), b.value_slots.shape[1]
            batches.append(SourceRowBatch(
                np.concatenate([b.value_slots, dummy(k * q, (k, q), 0)]), _pad(b.donor_ids, smax[i], trash),
                _pad(b.value, smax[i]),
                None if b.gradient_slots is None else np.concatenate([b.gradient_slots, dummy(k * q, (k, q), 1)]),
                None if b.gradient is None else _pad(b.gradient, smax[i]),
                None if b.donor_query is None else _pad(b.donor_query, smax[i]),
                None if b.target_query is None else _pad(b.target_query, smax[i])))
        tensor = []
        for i, b in enumerate(p.tensor_batches):
            if tmax[i] == 0:
                continue
            k = tmax[i] - len(b.value_slots)
            pad = lambda a: None if a is None else _pad(a, tmax[i])
            tensor.append(TensorRowBatch(
                b.family, b.dr_divisor, np.concatenate([b.value_slots, dummy(k, (k,), 0)]),
                None if b.gradient_slots is None else np.concatenate([b.gradient_slots, dummy(k, (k,), 1)]),
                pad(b.t_eta), pad(b.t_radial), pad(b.layer_id), pad(b.t_theta), pad(b.t_ring)))
        out.append(SourceRowPayload(tuple(batches), n_targets, n_gradient, n_queries, tuple(tensor), p.tensor_tables))
    return out


def _pad_cells(cells: Sequence[CellPlan], trash: int, n_queries: int) -> list:
    rc = max(len(c.raw_ids) for c in cells)
    rn = _neumann_count([c.neumann for c in cells])
    payloads = _pad_payloads([c.rows for c in cells], trash, n_queries)
    neumann = _pad_neumann([c.neumann for c in cells], rn, trash)
    out = []
    for c, rows, nr in zip(cells, payloads, neumann):
        first = lambda a: _pad_like_first(a, rc)
        out.append(replace(
            c, rows=rows, value_slot=first(c.value_slot), gradient_slot=first(c.gradient_slot),
            conditioned=_pad(c.conditioned, rc, False), neumann=nr, neumann_cell=_pad(c.neumann_cell, rn, rc),
            raw_ids=_pad(c.raw_ids, rc, -1), raw_owner=_pad(c.raw_owner, rc, trash), raw_volume=_pad(c.raw_volume, rc),
            h=first(c.h), jac=first(c.jac), B=first(c.B), K=first(c.K), J=first(c.J), weight=_pad(c.weight, rc),
            evolution_weight=_pad(c.evolution_weight, rc)))
    return out


def _pad_faces(faces: Sequence[FacePlan], trash: int, n_queries: int) -> list:
    fc = max(len(f.census_row) for f in faces)
    Qf = faces[0].nodes
    payloads = _pad_payloads([f.rows for f in faces], trash, n_queries)
    rc = _neumann_count([f.common_neumann for f in faces])
    rs = _neumann_count([f.side_neumann for f in faces])
    common = _pad_neumann([f.common_neumann for f in faces], rc, trash)
    side = _pad_neumann([f.side_neumann for f in faces], rs, trash)
    nw = max(len(f.wall_faces) for f in faces)
    out = []
    for f, rows, cn, sn in zip(faces, payloads, common, side):
        first = lambda a: _pad_like_first(a, fc)
        out.append(replace(
            f, rows=rows, common_value_slot=first(f.common_value_slot), common_gradient_slot=first(f.common_gradient_slot),
            lower_slot=first(f.lower_slot), upper_slot=first(f.upper_slot), lower_present=first(f.lower_present),
            upper_present=first(f.upper_present), common_conditioned=_pad(f.common_conditioned, fc, False),
            lower_conditioned=_pad(f.lower_conditioned, fc, False), upper_conditioned=_pad(f.upper_conditioned, fc, False),
            fallback_query=first(f.fallback_query), common_neumann=cn,
            common_neumann_target=_pad(f.common_neumann_target, rc, fc * Qf), side_neumann=sn,
            side_neumann_target=_pad(f.side_neumann_target, rs, fc * Qf), census_row=_pad(f.census_row, fc, -1),
            p07_id=_pad(f.p07_id, fc, -1), axis=first(f.axis), lower_owner=_pad(f.lower_owner, fc, trash),
            upper_owner=_pad(f.upper_owner, fc, trash), wall=_pad(f.wall, fc, False),
            wall_faces=_pad(f.wall_faces, nw, fc), collapsed=_pad(f.collapsed, fc, True),
            p07_valid=_pad(f.p07_valid, fc, False), face_multiplier=_pad(f.face_multiplier, fc, 1.0), h=first(f.h),
            jac=first(f.jac), weight=first(f.weight), J=first(f.J), B=first(f.B), K=first(f.K)))
    return out


def _pad_p07(p07s: Sequence[P07Plan], trash: int, n_queries: int) -> list:
    fp = max(len(p.p07_id) for p in p07s)
    nb = len(p07s[0].rows.batches)
    smax = [max(len(p.rows.batches[i].face_ids) for p in p07s) for i in range(nb)]
    fn = max(len(p.neumann_face) for p in p07s)
    neumann = _pad_neumann([p.neumann for p in p07s], 9 * fn, trash)
    out = []
    for p, nr in zip(p07s, neumann):
        batches = []
        for i, b in enumerate(p.rows.batches):
            if smax[i] == 0:
                continue
            s = smax[i]
            batches.append(IntegratedFaceBatch(
                _pad(b.face_ids, s, fp), _pad(b.donor_ids, s, trash), _pad(b.weights, s), _pad(b.boundary_donor_ids, s),
                _pad(b.tangential_ids, s), _pad(b.tangential_weights, s), _pad(b.conditioned, s, False)))
        lower, upper = _pad(p.rows.lower_owner, fp, trash), _pad(p.rows.upper_owner, fp, trash)
        out.append(replace(
            p, rows=IntegratedRows(tuple(batches), lower, upper, n_queries, fp), conditioned=_pad(p.conditioned, fp, False),
            neumann=nr, neumann_face=_pad(p.neumann_face, fn, fp), integrand=_pad(p.integrand, fn),
            p07_id=_pad(p.p07_id, fp, -1), census_row=_pad(p.census_row, fp, -1), face_index=_pad(p.face_index, fp, -1),
            family=_pad(p.family, fp, 0), lower_owner=lower, upper_owner=upper))
    return out


# --------------------------------------------------------------------------
# Public lowering
# --------------------------------------------------------------------------

def shard_perpendicular_plan(plan: PerpendicularPlan, raw_to_owner, n: int, n_shards: int,
                             halo: int = DEFAULT_HALO) -> ShardedPerpendicularPlan:
    """Lower ``plan`` into per-shard local plans over the extended owner sets (module docstring).

    ``raw_to_owner`` ``(n**3,)`` is the grid's raw-cell -> owner map (it defines the plane-major relabelling;
    state / phi / outputs of the sharded RHS are in that order, see :func:`plane_major_permutation`). ``n_shards``
    must divide ``n`` and the owned block must be at least ``halo`` planes (:data:`DEFAULT_HALO` = 3, see the module
    docstring: a block-boundary face is computed on both shards and needs the side rows anchored at the halo owner).
    Raises ``ValueError`` if a weighted donor
    falls outside the extended window (with the offending eta offsets) or a shard holds no cell / face."""
    if int(plan.n) != int(n):
        raise ValueError(f"the plan was lowered for n = {plan.n}, not {n}")
    perm, inverse, m = plane_major_permutation(raw_to_owner, n)
    layout = PlaneLayout(perm, inverse, m)
    ctx = _Ctx(layout, n, n_shards, halo)
    for part in (plan.cells, plan.faces, plan.p07):
        if part is not None and len(part.owner_volume) != len(perm):
            raise ValueError("the plan's owner arrays do not match raw_to_owner")
    locals_ = [_localize(plan, _Shard(ctx, s)) for s in range(ctx.n_shards)]
    plans = [loc[0] for loc in locals_]
    trash = ctx.trash
    qd = max((len(loc[1]) for loc in locals_), default=0)
    qn = max((len(loc[2]) for loc in locals_), default=0)
    qd, qn = (max(qd, 1) if len(plan.dirichlet_points) else 0), (max(qn, 1) if len(plan.neumann_points) else 0)
    cells = faces = p07 = [None] * len(plans)
    if plan.cells is not None:
        cells = _pad_cells([p.cells for p in plans], trash, qd)
    if plan.faces is not None:
        faces = _pad_faces([p.faces for p in plans], trash, qd)
    if plan.p07 is not None:
        p07 = _pad_p07([p.p07 for p in plans], trash, qd)
    padded = [PerpendicularPlan(c, f, q, _pad(p.dirichlet_points, qd).reshape(-1, 3), _pad(p.neumann_points, qn).reshape(-1, 3),
                                p.n)
              for p, c, f, q in zip(plans, cells, faces, p07)]
    stacked = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *padded)
    return ShardedPerpendicularPlan(
        stacked, np.stack([_pad(loc[1], qd) for loc in locals_]).astype(np.int32),
        np.stack([_pad(loc[2], qn) for loc in locals_]).astype(np.int32), n, m, n_shards, halo, layout)


def shard_boundary_data(bc: BoundaryData, sharded: ShardedPerpendicularPlan) -> BoundaryData:
    """The per-shard boundary tables ``(Sz, Qd, F)`` / ``(Sz, Qd, 2, F)`` / ``(Sz, Qn, F)`` of global ``bc`` (gathered
    by the plan's ``dirichlet_ids`` / ``neumann_ids``; ``None`` parts stay ``None``)."""
    def gather(part, ids):
        return None if part is None else jnp.asarray(part)[jnp.asarray(ids)]

    return BoundaryData(gather(bc.dirichlet_value, sharded.dirichlet_ids),
                        gather(bc.dirichlet_tangential, sharded.dirichlet_ids),
                        gather(bc.neumann_normal, sharded.neumann_ids))


# --------------------------------------------------------------------------
# Sharded RHS
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("mesh", "columns", "fields", "terms", "kinds", "n_shards", "halo", "p", "m"))
def _sharded_rhs(plan, state, phi, bc, params, *, mesh, columns, fields, terms, kinds, n_shards, halo, p, m):
    def body(plan_s, state_s, phi_s, bc_s, params_s):
        plan_s = jax.tree_util.tree_map(lambda a: a[0], plan_s)
        bc_s = jax.tree_util.tree_map(lambda a: a[0], bc_s)
        owned = jnp.stack([state_s[c] for c in columns[:-1]] + [phi_s], axis=1).reshape(p, m, len(columns))
        ext = exchange_plane_halo(owned, halo, AXIS, n_shards).reshape(((p + 2 * halo) * m, len(columns)))
        ext = jnp.concatenate([ext, jnp.zeros((1, ext.shape[1]), ext.dtype)], axis=0)       # the trash row
        res = perpendicular_rhs(plan_s, {c: ext[:, i] for i, c in enumerate(columns[:-1])}, ext[:, -1], bc_s,
                                kinds, params_s, fields=fields, terms=terms)
        own = lambda a: a[halo * m:(halo + p) * m]
        per_shard = lambda a: jnp.reshape(a, (1,))
        return (jax.tree_util.tree_map(own, res.terms), jax.tree_util.tree_map(own, res.total),
                jax.tree_util.tree_map(own, res.detail), jax.tree_util.tree_map(per_shard, res.diagnostics))

    spec = P(AXIS)
    return jax.shard_map(body, mesh=mesh, in_specs=(spec, spec, spec, spec, P()), out_specs=spec,
                         check_vma=False)(plan, state, phi, bc, params)


def sharded_perpendicular_rhs(sharded: ShardedPerpendicularPlan, state: Mapping[str, object], phi,
                              bc: BoundaryData, field_kinds, params: PerpendicularParams, mesh: Mesh, *,
                              fields: Sequence[str] = FIELDS, terms: Sequence[str] = _TERM_ORDER) -> PerpendicularTerms:
    """:func:`perpendicular_rhs` on ``sharded`` under ``jax.shard_map`` over ``mesh``'s ``"z"`` axis.

    ``state`` (column name -> ``(n * m,)``) and ``phi`` are in plane-major owner order; ``bc`` is the per-shard
    :func:`shard_boundary_data` (leading axis ``Sz``), ``field_kinds`` / ``params`` / ``fields`` / ``terms`` as in
    :func:`perpendicular_rhs`. Returns the same :class:`PerpendicularTerms` with plane-major ``(n * m,)`` owner arrays
    (``raw_pairs`` is ``None``) and ``diagnostics`` as per-shard ``(Sz,)`` arrays (module docstring). Each shard
    exchanges the halos of the stacked columns once, runs the unchanged RHS on its local plan and returns its owned
    slice."""
    fields, terms, _ = _validate(fields, terms, None)
    columns = perpendicular_columns(fields, terms, None)
    missing = [c for c in columns[:-1] if c not in state]
    if missing:
        raise KeyError(f"state is missing the columns {missing}; the layout is {columns}")
    if int(mesh.shape[AXIS]) != sharded.n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the plan {sharded.n_shards}")
    kinds = _kinds(field_kinds, columns)
    used = {c: jnp.asarray(state[c]) for c in columns[:-1]}
    t, total, detail, diagnostics = _sharded_rhs(
        sharded.plan, used, jnp.asarray(phi), bc, params, mesh=mesh, columns=columns, fields=fields, terms=terms,
        kinds=kinds, n_shards=sharded.n_shards, halo=sharded.halo, p=sharded.planes_per_shard, m=sharded.m)
    return PerpendicularTerms(t, total, detail, None, diagnostics)
