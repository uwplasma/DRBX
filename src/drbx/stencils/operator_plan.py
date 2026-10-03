"""Operator plan: row-artifact payloads + geometry + census -> one JAX-ready pytree (P08 step 2b, E3).

The perpendicular operators (P05 bracket, P06 curvature, P07 diffusion) are functions of owner
fields and boundary data applied from the step-2a loader payloads. ``PerpendicularPlan`` is the
field-independent object they are applied *from*: it is lowered once per grid, or per bounded owner
set, and passed to ``jax.jit`` as an argument (arrays are pytree data, counts are static).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` sections 2.1 and 5.

What the plan holds
-------------------
``cells``  one R1 source per raw cell (value + gradient rows), the R1 Neumann rows of the
           conditioned cells, and the raw geometry (``h, |J|, B, K, J`` and the q1 evolution weight
           ``w = p06_raw_weight * J / max(B, 1e-30)``; ``p06_raw_weight`` is the plain q1 quadrature
           weight, J/B is *not* included in it).
``faces``  R2 common rows (value + gradient at the ``Qf`` face nodes: 9 q3, or 4 q2), R3 lower/upper side rows (values),
           their Neumann rows, per-face geometry and masks, in ``face_row_selection`` order.
``p07``    the R4 integrated rows (one per P07 face, in ``p07_rows`` order), the R4 Neumann rows
           of the conditioned faces (families 1, 2, 4) and their q3 integrand
           ``contract_face_tensor(weight_q3, p07_face_tensor, axis)`` from ``geometry``.

Boundary data. Every conditioned row refers to prescribed data through ids. The plan re-numbers
all of them into two global, bit-pattern-deduplicated point tables:

``dirichlet_points`` (Qd, 3)  every conditioned donor query and target point (cells, faces, p07)
                              plus, for faces with a missing side row, the common row's 9 target
                              points (``faces.fallback_query``: the side fallback);
``neumann_points``  (Qn, 3)   every Neumann row's 28 wall-lattice points.

Every payload of the plan indexes these tables directly (``n_boundary_queries == Qd`` in all source
payloads), so one ``BoundaryData`` (see ``drbx.native.fci_perpendicular_reconstruction_state``) serves
all of them.

Inputs of the lowering (:func:`lower_perpendicular_plan`) are chunk *sources*: any of a list/tuple of
chunks or a zero-argument callable returning a fresh iterable (needed for lazily read artifact files,
which are consumed more than once: faces = R2 + R3, Neumann = R1 + R2 + R3 + R4). Point chunks may be
``PointRowChunk`` or ``FactoredChunk`` (tensor sources are never expanded). Entry points:

* :func:`lower_perpendicular_plan`            from chunk sources,
* :func:`lower_perpendicular_plan_from_artifact`  from an artifact directory (streaming, hashes checked),
* :func:`lower_perpendicular_plan_from_rows`  from in-memory row dictionaries of a bounded owner set
  (the ``p_shared.owner_closure.build_owner_rows`` output), packed with the artifact packers.

Lowering is vectorized (loops only over chunks and buckets inside the step-2a loader); only the
in-memory row packer loops over row objects, as the artifact packers do.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields as dc_fields, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import jax
import numpy as np

from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor
from drbx.native.fci_perpendicular_integrated_rows import IntegratedFaceBatch, pad_integrated_batch
from drbx.native.fci_perpendicular_neumann_rows import NeumannPayload
from drbx.native.fci_perpendicular_source_rows import SourceRowPayload
from drbx.stencils import artifact as art
from drbx.stencils.census import NO_ID, FaceCensus
from drbx.stencils.geometry_arrays import GeometryArrays
from drbx.stencils.loader import (
    FactoredChunk, LoaderGrid, NeumannRowPlan, RowSelection, _unique_rows, lower_integrated_chunks,
    lower_neumann_chunks, lower_point_chunks)

__all__ = [
    "CellPlan", "FacePlan", "P07Plan", "NeumannRows", "IntegratedRows", "PerpendicularPlan",
    "face_row_selection", "p07_row_selection", "lower_perpendicular_plan",
    "lower_perpendicular_plan_from_artifact", "lower_perpendicular_plan_from_rows",
    "pack_owner_rows", "plan_nbytes", "EVOLUTION_FLOOR", "Q3_NODES", "FACE_NODE_COUNTS", "NEUMANN_WALL_POINTS"]

#: floor of the q1 evolution weight's ``B`` (host: ``np.maximum(B, 1.0e-30)``)
EVOLUTION_FLOOR = 1.0e-30
Q3_NODES = 9
#: admissible P05/P06 face-node counts per face: q3 (3x3) and q2 (2x2)
FACE_NODE_COUNTS = (9, 4)
NEUMANN_WALL_POINTS = 28

_R1, _R2, _R3, _R4 = (art.REQUEST_KINDS.index(name) for name in ("R1", "R2", "R3", "R4"))
_P07_NEUMANN_DEGREE = {1: 4, 2: 4, 4: 3}


# --------------------------------------------------------------------------
# Plan pytrees
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NeumannRows:
    """Padded Neumann rows (``NeumannPayload`` without its count): ``R`` rows, donor width ``W``.

    ``boundary_ids`` index the plan's global ``neumann_points``; rows are in the order documented
    by the owning sub-plan (``cells.neumann_cell`` etc.).

    The lowering stores only what the operators read: the P07 rows (``p07.neumann``) contract gradients only,
    so their value weights are ``None``; the R3 side rows (``faces.side_neumann``) are read for values only, so
    their gradient weights are ``None``. ``apply_neumann_point_rows`` returns ``None`` for the missing part.
    """

    donor_ids: np.ndarray                   # (R, W) int32
    value_weights: np.ndarray | None        # (R, W)
    gradient_weights: np.ndarray | None     # (R, 3, W)
    boundary_ids: np.ndarray                # (R, 28) int32
    boundary_value_weights: np.ndarray | None       # (R, 28)
    boundary_gradient_weights: np.ndarray | None    # (R, 3, 28)

    def payload(self, boundary_query_count: int) -> NeumannPayload:
        return NeumannPayload(self.donor_ids, self.value_weights, self.gradient_weights, self.boundary_ids,
                              self.boundary_value_weights, self.boundary_gradient_weights,
                              int(boundary_query_count))


@dataclass(frozen=True)
class IntegratedRows:
    """``IntegratedFacePayload`` with static counts as meta fields (a NamedTuple with int fields would
    turn its counts into traced leaves under ``jax.jit``).

    Plan lowering (:func:`_remap_integrated`) drops the boundary arrays of buckets without conditioned faces
    (``None`` fields: a static flag, the lift is skipped) and pads buckets larger than
    ``FACE_CHUNK`` faces with no-op faces of id ``face_count`` (the chunked contraction); consumers of ``batches``
    must ignore faces ``>= face_count``."""

    batches: tuple[IntegratedFaceBatch, ...]
    lower_owner: np.ndarray                 # (Fp,) int32, -1: none
    upper_owner: np.ndarray                 # (Fp,) int32
    boundary_query_count: int               # Qd
    face_count: int                         # Fp


@dataclass(frozen=True)
class CellPlan:
    """R1 rows and raw-cell geometry. Cell ``c`` is raw cell ``raw_ids[c]`` (``raw_ids`` ascending).

    ``rows`` is a ``SourceRowPayload`` (one source, one target per cell); its value output row of cell
    ``c`` is ``value_slot[c]`` and its gradient row ``gradient_slot[c]``. ``neumann`` holds one row
    per *conditioned* cell, row ``r`` belonging to cell ``neumann_cell[r]``.
    """

    rows: SourceRowPayload
    value_slot: np.ndarray          # (R,) int32
    gradient_slot: np.ndarray       # (R,) int32
    conditioned: np.ndarray         # (R,) bool, R1 row is boundary-conditioned
    neumann: NeumannRows | None
    neumann_cell: np.ndarray        # (Rn,) int32
    raw_ids: np.ndarray             # (R,) int64
    raw_owner: np.ndarray           # (R,) int32
    raw_volume: np.ndarray          # (R,)
    owner_volume: np.ndarray        # (n_owners,)
    evolution_volume: np.ndarray    # (n_owners,) per-owner sum of ``evolution_weight`` over the plan's cells
    h: np.ndarray                   # (R, 3)   p05_raw_h
    jac: np.ndarray                 # (R,)     |p05_raw_jacobian|
    B: np.ndarray                   # (R,)     p06_raw_B
    K: np.ndarray                   # (R, 3)   p06_raw_K
    J: np.ndarray                   # (R,)     p06_raw_J
    weight: np.ndarray              # (R,)     p06_raw_weight (plain q1 quadrature weight)
    evolution_weight: np.ndarray    # (R,)     weight * J / max(B, 1e-30)


@dataclass(frozen=True)
class FacePlan:
    """R2/R3 rows, Neumann rows, geometry and masks of the ``Fc`` faces ``census_row`` (face_row_selection order).

    ``Qf`` = :attr:`nodes` is the per-plan face-node count, 9 (q3, default) or 4 (q2): the P05/P06 face
    rule. All ``(Fc, Qf)`` slot arrays index the value/gradient outputs of ``rows``. Missing sides carry slot 0
    and ``*_present`` False (their values are replaced by the fallback). ``*_conditioned`` flags a
    boundary-conditioned side row. The Neumann row ``r`` of ``common_neumann`` (R2) / ``side_neumann``
    (R3, one set per face, serving both sides) belongs to the flat (face, node) index
    ``face * Qf + node = *_neumann_target[r]``.
    """

    rows: SourceRowPayload
    common_value_slot: np.ndarray       # (Fc, Qf) int32
    common_gradient_slot: np.ndarray    # (Fc, Qf) int32
    lower_slot: np.ndarray              # (Fc, Qf) int32
    upper_slot: np.ndarray              # (Fc, Qf) int32
    lower_present: np.ndarray           # (Fc,) bool  side exists (census) and its R3 row is stored
    upper_present: np.ndarray           # (Fc,) bool
    common_conditioned: np.ndarray      # (Fc,) bool
    lower_conditioned: np.ndarray       # (Fc,) bool  (False for a missing side)
    upper_conditioned: np.ndarray       # (Fc,) bool
    fallback_query: np.ndarray          # (Fc, Qf) int32, ids into dirichlet_points (0 where no side is missing)
    common_neumann: NeumannRows | None
    common_neumann_target: np.ndarray   # (Rn2,) int32
    side_neumann: NeumannRows | None
    side_neumann_target: np.ndarray     # (Rn3,) int32
    census_row: np.ndarray              # (Fc,) int64
    p07_id: np.ndarray                  # (Fc,) int64, -1: no P07 id
    axis: np.ndarray                    # (Fc,) int32
    lower_owner: np.ndarray             # (Fc,) int32, -1: none
    upper_owner: np.ndarray             # (Fc,) int32
    wall: np.ndarray                    # (Fc,) bool, axis 0 and i == n
    collapsed: np.ndarray               # (Fc,) bool (all False in face_row_selection)
    p07_valid: np.ndarray               # (Fc,) bool, valid p07 id and not collapsed (the P05 jump domain)
    face_multiplier: np.ndarray         # (Fc,) float64, default 1 (harness: 2 on legacy seam faces)
    h: np.ndarray                       # (Fc, Qf, 3)  p05_face_h
    jac: np.ndarray                     # (Fc, Qf)     |p05_face_jacobian|
    weight: np.ndarray                  # (Fc, Qf)     p06_face_weight (q3, or q2)
    J: np.ndarray                       # (Fc, Qf)
    B: np.ndarray                       # (Fc, Qf)
    K: np.ndarray                       # (Fc, Qf, 3)
    owner_volume: np.ndarray            # (n_owners,)
    has_missing_side: bool              # any face lacks a side row (static: fallback gather is skipped otherwise)
    wall_faces: np.ndarray | None = None  # (Wn,) int32, ascending face indices of ``wall`` (static size under jit);
    #                                       sharded plans pad it with the out-of-range index ``Fc`` (a no-op entry)

    @property
    def nodes(self) -> int:
        """``Qf``: the number of face nodes per face (9: q3, 4: q2)."""
        return int(self.common_value_slot.shape[1])


@dataclass(frozen=True)
class P07Plan:
    """R4 integrated rows of the ``Fp`` P07 faces.

    Face order is the order of the R4 chunks (``IntegratedRowPlan.face_entity_id``); ``p07_id`` is
    the P07 face id and ``census_row`` its census row. ``neumann`` holds ``9 * Fn`` rows, face-major
    and node-minor, for the ``Fn`` conditioned faces ``neumann_face`` (indices into the ``Fp`` faces,
    ascending P07 id); ``integrand[f]`` is the ``(9, 3)`` q3 integrand of conditioned face
    ``neumann_face[f]``. ``face_index`` is the face's position in ``face_row_selection`` order
    (``-1`` for the collapsed r=0 face, family 0, which has no R2/R3 rows and carries zero flux).
    """

    rows: IntegratedRows
    conditioned: np.ndarray             # (Fp,) bool, integrated row is boundary-conditioned
    neumann: NeumannRows | None
    neumann_face: np.ndarray            # (Fn,) int32
    integrand: np.ndarray               # (Fn, 9, 3)
    p07_id: np.ndarray                  # (Fp,) int64
    census_row: np.ndarray              # (Fp,) int64
    face_index: np.ndarray              # (Fp,) int64
    family: np.ndarray                  # (Fp,) int16
    lower_owner: np.ndarray             # (Fp,) int32
    upper_owner: np.ndarray             # (Fp,) int32
    owner_volume: np.ndarray            # (n_owners,)


@dataclass(frozen=True)
class PerpendicularPlan:
    cells: CellPlan | None
    faces: FacePlan | None
    p07: P07Plan | None
    dirichlet_points: np.ndarray        # (Qd, 3)
    neumann_points: np.ndarray          # (Qn, 3)
    n: int                              # grid size (static)


for _cls, _meta in ((NeumannRows, ()), (IntegratedRows, ("boundary_query_count", "face_count")),
                    (CellPlan, ()), (FacePlan, ("has_missing_side",)), (P07Plan, ()),
                    (PerpendicularPlan, ("n",))):
    jax.tree_util.register_dataclass(
        _cls, data_fields=[f.name for f in dc_fields(_cls) if f.name not in _meta], meta_fields=list(_meta))


def plan_nbytes(plan: PerpendicularPlan) -> int:
    """Total bytes of the array leaves of a plan."""
    return sum(int(np.asarray(a).nbytes) for a in jax.tree_util.tree_leaves(plan))


# --------------------------------------------------------------------------
# Selections
# --------------------------------------------------------------------------

def face_row_selection(census: FaceCensus) -> np.ndarray:
    """Census rows of the R2/R3 faces: every slot except the collapsed r=0 face and the periodic alias
    slots (the order of the ``geometry.npz`` face arrays)."""
    return np.flatnonzero(~(census.collapsed_r0 | census.legacy_alias_slots))


def p07_row_selection(census: FaceCensus) -> np.ndarray:
    """Census rows of the P07 faces: valid P07 id, alias slots excluded (includes the collapsed r=0 face)."""
    return np.flatnonzero((census.p07_id != NO_ID) & ~census.legacy_alias_slots)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _source(chunks) -> Callable[[], Iterable]:
    """A zero-argument callable giving a fresh iterable of chunks."""
    if callable(chunks):
        return chunks
    items = tuple(chunks)
    return lambda: iter(items)


def _positions(sorted_ids: np.ndarray, ids: np.ndarray, what: str) -> np.ndarray:
    """Position of each of ``ids`` in the ascending ``sorted_ids`` (raises if one is absent)."""
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return np.zeros(0, dtype=np.int64)
    if not len(sorted_ids):
        raise ValueError(f"{what}: no ids to index into")
    pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    if np.any(sorted_ids[pos] != ids):
        raise ValueError(f"{what}: ids outside the plan's set")
    return pos


def _check_ascending_unique(values: np.ndarray, what: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or (len(values) > 1 and np.any(np.diff(values) <= 0)):
        raise ValueError(f"{what} must be a strictly ascending 1-D id array")
    return values


def _neumann_rows(nplan: NeumannRowPlan, order: np.ndarray | None, remap: np.ndarray, *,
                  values: bool = True, gradients: bool = True) -> NeumannRows:
    """Plan rows of a loader Neumann plan; ``values`` / ``gradients`` False store ``None`` for the weights
    the operators never read (see :class:`NeumannRows`)."""
    p = nplan.payload
    take = (lambda a: a) if order is None else (lambda a: a[order])
    ids = take(p.boundary_ids)
    return NeumannRows(take(p.donor_ids), take(p.value_weights) if values else None,
                       take(p.gradient_weights) if gradients else None,
                       remap[ids].astype(np.int32), take(p.boundary_value_weights) if values else None,
                       take(p.boundary_gradient_weights) if gradients else None)


def _global_table(parts: Sequence[np.ndarray]):
    """Deduplicate concatenated ``(m_i, 3)`` point tables by bit pattern.

    Returns ``(points (Q, 3), remaps)``, ``remaps[i]`` (int32) mapping row ``r`` of part ``i`` to its row
    in the global table."""
    if not parts or sum(len(p) for p in parts) == 0:
        return np.zeros((0, 3)), [np.zeros(len(p), dtype=np.int32) for p in parts]
    points, inverse = _unique_rows(np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in parts]))
    bounds = np.cumsum([0] + [len(p) for p in parts])
    return points, [inverse[bounds[i]:bounds[i + 1]].astype(np.int32) for i in range(len(parts))]


def _remap_source_payload(payload: SourceRowPayload, remap: np.ndarray, count: int) -> SourceRowPayload:
    batches = tuple(b if b.donor_query is None else
                    b._replace(donor_query=remap[b.donor_query], target_query=remap[b.target_query])
                    for b in payload.batches)
    return replace(payload, batches=batches, n_boundary_queries=int(count))


def _remap_integrated(rows_plan, remap: np.ndarray, count: int) -> IntegratedRows:
    """Integrated rows with the global boundary ids; buckets without conditioned faces lose their (never read)
    boundary arrays, and buckets larger than ``FACE_CHUNK`` are padded to a multiple of it."""
    payload = rows_plan.payload
    face_count = int(payload.face_count)
    batches = []
    for b in payload.batches:
        if np.any(b.conditioned):
            b = b._replace(boundary_donor_ids=remap[b.boundary_donor_ids], tangential_ids=remap[b.tangential_ids])
        else:
            b = b._replace(boundary_donor_ids=None, tangential_ids=None, tangential_weights=None)
        batches.append(pad_integrated_batch(b, face_count))
    return IntegratedRows(tuple(batches), payload.lower_owner.astype(np.int32), payload.upper_owner.astype(np.int32),
                          int(count), face_count)


def _row_counts_ok(rows_of_entity: np.ndarray, expected: np.ndarray, size: int, what: str) -> None:
    got = np.bincount(rows_of_entity, minlength=size)
    if not np.array_equal(got, expected):
        raise ValueError(f"{what}: stored Neumann rows do not match the conditioned rows")


# --------------------------------------------------------------------------
# Lowering
# --------------------------------------------------------------------------

def lower_perpendicular_plan(*, grid: LoaderGrid, census: FaceCensus, geometry: GeometryArrays,
                             raw_volume, owner_volume,
                             cell_chunks=(), face_chunks=(), p07_chunks=(), neumann_chunks=(),
                             raw_ids=None, face_rows=None, p07_rows=None,
                             include: Sequence[str] = ("cells", "faces", "p07")) -> PerpendicularPlan:
    """Lower chunk sources into a :class:`PerpendicularPlan`.

    ``geometry`` raw arrays are aligned with ``raw_ids`` (ascending; default: all ``n**3`` raw cells)
    and its face arrays with ``face_rows`` (ascending census rows; default :func:`face_row_selection`).
    ``p07_rows`` (ascending census rows; default :func:`p07_row_selection`) is the P07 face set; the
    R4 chunks must hold exactly those faces. ``raw_volume`` is ``(n**3,)``, ``owner_volume``
    ``(n_owners,)``. ``cell_chunks``/``face_chunks``/``neumann_chunks`` need only cover the parts in
    ``include``. Raises ``ValueError`` if a chunk set does not match the row sets, or a conditioned row has
    no Neumann row (or vice versa).
    """
    include = tuple(include)
    bad = set(include) - {"cells", "faces", "p07"}
    if bad:
        raise ValueError(f"unknown plan parts: {sorted(bad)}")
    n = grid.n
    owner_volume = np.asarray(owner_volume, dtype=np.float64)
    if owner_volume.shape != (grid.n_owners,):
        raise ValueError("owner_volume must have one entry per owner")
    raw_volume = np.asarray(raw_volume, dtype=np.float64)
    raw_ids = np.arange(n ** 3, dtype=np.int64) if raw_ids is None else _check_ascending_unique(raw_ids, "raw_ids")
    face_rows = (face_row_selection(census) if face_rows is None
                 else _check_ascending_unique(face_rows, "face_rows"))
    p07_rows = (p07_row_selection(census) if p07_rows is None
                else _check_ascending_unique(p07_rows, "p07_rows"))
    neumann_src = _source(neumann_chunks)

    def lower_neumann(request):
        plan = lower_neumann_chunks(neumann_src(), select=RowSelection(requests=(request,)))
        return plan if len(plan.entity_id) else None

    # --- R1 --------------------------------------------------------------
    cells = None
    if "cells" in include:
        if geometry.raw_points.shape[0] != len(raw_ids):
            raise ValueError("geometry raw arrays must be aligned with raw_ids")
        cplan = lower_point_chunks(_source(cell_chunks)(), grid=grid, select=RowSelection(requests=("R1",)))
        t = cplan.targets
        R = len(raw_ids)
        if len(t.entity_id) != R or np.any(t.quad_node != 0):
            raise ValueError("cells: exactly one R1 target per raw id is required")
        cell = _positions(raw_ids, t.entity_id, "cells: R1 entity ids")
        if len(np.unique(cell)) != R:
            raise ValueError("cells: duplicate R1 rows")
        value_slot = np.empty(R, dtype=np.int32); value_slot[cell] = t.value_slot
        gradient_slot = np.empty(R, dtype=np.int32); gradient_slot[cell] = t.gradient_slot
        if np.any(gradient_slot < 0):
            raise ValueError("cells: R1 rows must store gradients")
        conditioned = np.empty(R, dtype=bool); conditioned[cell] = t.conditioned
        cells_neumann = lower_neumann("R1")
        expected = conditioned.astype(np.int64)
        if cells_neumann is None:
            if conditioned.any():
                raise ValueError("cells: conditioned R1 rows without Neumann rows")
            neumann_cell = np.zeros(0, dtype=np.int32)
        else:
            if np.any(cells_neumann.quad_node != 0):
                raise ValueError("cells: R1 Neumann rows have one node")
            neumann_cell = _positions(raw_ids, cells_neumann.entity_id, "cells: R1 Neumann entity ids")
            _row_counts_ok(neumann_cell, expected, R, "cells")
            neumann_cell = neumann_cell.astype(np.int32)
        raw_owner = grid.raw_to_owner[raw_ids]
        weight = np.asarray(geometry.p06_raw_weight)
        J = np.asarray(geometry.p06_raw_J)
        B = np.asarray(geometry.p06_raw_B)
        evolution_weight = weight * J / np.maximum(B, EVOLUTION_FLOOR)
        evolution_volume = np.zeros(grid.n_owners)
        np.add.at(evolution_volume, raw_owner, evolution_weight)
        cells = (cplan, cells_neumann, dict(
            value_slot=value_slot, gradient_slot=gradient_slot, conditioned=conditioned, neumann_cell=neumann_cell,
            raw_ids=raw_ids, raw_owner=raw_owner.astype(np.int32), raw_volume=raw_volume[raw_ids],
            owner_volume=owner_volume, evolution_volume=evolution_volume,
            h=np.asarray(geometry.p05_raw_h), jac=np.abs(np.asarray(geometry.p05_raw_jacobian)),
            B=B, K=np.asarray(geometry.p06_raw_K), J=J, weight=weight, evolution_weight=evolution_weight))

    # --- R2 / R3 -----------------------------------------------------------
    faces = None
    if "faces" in include:
        Fc = len(face_rows)
        if geometry.face_points.shape[0] != Fc:
            raise ValueError("geometry face arrays must be aligned with face_rows")
        fplan = lower_point_chunks(_source(face_chunks)(), grid=grid, select=RowSelection(requests=("R2", "R3")))
        t = fplan.targets
        if np.any((t.quad_node < 0) | (t.quad_node >= Q3_NODES)):
            raise ValueError("faces: q3 node index outside 0..8")
        # the per-plan face-node count, from the R2 targets: q3 (9) or q2 (4); every face must have exactly Qf
        r2_nodes = t.quad_node[t.request == _R2]
        Qf = int(r2_nodes.max()) + 1 if len(r2_nodes) else int(geometry.face_points.shape[1])
        if Qf not in FACE_NODE_COUNTS:
            raise ValueError(f"faces: R2 targets span {Qf} nodes per face; expected {FACE_NODE_COUNTS}")
        if np.any(t.quad_node >= Qf):
            raise ValueError(f"faces: R3 node index beyond the {Qf} R2 nodes per face")
        if geometry.face_points.shape[1] != Qf:
            raise ValueError(f"faces: geometry has {geometry.face_points.shape[1]} face nodes but the rows have {Qf}")

        def slot_table(mask, face_of, what):
            table = np.full((Fc, Qf), -1, dtype=np.int64)
            gtable = np.full((Fc, Qf), -1, dtype=np.int64)
            table[face_of, t.quad_node[mask]] = t.value_slot[mask]
            gtable[face_of, t.quad_node[mask]] = t.gradient_slot[mask]
            if len(np.unique(face_of * Qf + t.quad_node[mask])) != int(mask.sum()):
                raise ValueError(f"faces: duplicate {what} targets")
            return table, gtable

        r2 = t.request == _R2
        f2 = _positions(face_rows, t.entity_id[r2], "faces: R2 entity ids")
        common, common_g = slot_table(r2, f2, "R2")
        if np.any(common < 0) or np.any(common_g < 0):
            raise ValueError(f"faces: every face needs {Qf} R2 targets with gradients")
        r3 = t.request == _R3
        face3 = _positions(face_rows, t.entity_id[r3] // 2, "faces: R3 entity ids")
        side = t.entity_id[r3] % 2
        sides = []
        for s in (0, 1):
            sel = side == s
            table = np.full((Fc, Qf), -1, dtype=np.int64)
            table[face3[sel], t.quad_node[r3][sel]] = t.value_slot[r3][sel]
            if len(np.unique(face3[sel] * Qf + t.quad_node[r3][sel])) != int(sel.sum()):
                raise ValueError("faces: duplicate R3 targets")
            cond = np.zeros(Fc, dtype=bool)
            cond[face3[sel]] = t.conditioned[r3][sel]
            stored = (table >= 0)
            if np.any(stored.any(axis=1) != stored.all(axis=1)):
                raise ValueError("faces: partial R3 side rows")
            present = stored.all(axis=1)
            sides.append((table, present, cond))
        exists_lo, exists_hi = census.raw_lo[face_rows] >= 0, census.raw_hi[face_rows] >= 0
        if np.any(sides[0][1] != exists_lo) or np.any(sides[1][1] != exists_hi):
            raise ValueError("faces: stored R3 side rows disagree with the census side incidence")
        (lower, lower_present, lower_cond), (upper, upper_present, upper_cond) = sides
        common_cond = np.zeros(Fc, dtype=bool)
        common_cond[f2] = t.conditioned[r2]
        # Neumann rows: R2 per (face, node) of the conditioned common rows; R3 per (face, node) of the
        # faces with a conditioned side row (one set serving both sides).
        n2 = lower_neumann("R2")
        n3 = lower_neumann("R3")

        def neumann_targets(nplan, needed, what):
            if nplan is None:
                if needed.any():
                    raise ValueError(f"faces: conditioned {what} rows without Neumann rows")
                return np.zeros(0, dtype=np.int32)
            f = _positions(face_rows, nplan.entity_id, f"faces: {what} Neumann entity ids")
            if np.any(nplan.quad_node >= Qf):
                raise ValueError(f"faces: {what} Neumann node index beyond the {Qf} nodes per face")
            target = f * Qf + nplan.quad_node
            if len(np.unique(target)) != len(target):
                raise ValueError(f"faces: duplicate {what} Neumann rows")
            _row_counts_ok(f, needed.astype(np.int64) * Qf, Fc, f"faces {what}")
            return target.astype(np.int32)

        common_target = neumann_targets(n2, common_cond, "R2")
        side_target = neumann_targets(n3, lower_cond | upper_cond, "R3")
        missing = ~(lower_present & upper_present)
        fallback_points = np.zeros((0, 3))
        if missing.any():
            fallback_points = t.target_points[common[missing].reshape(-1)]
        axis = census.axis[face_rows]
        p07_id = census.p07_id[face_rows]
        collapsed = census.collapsed_r0[face_rows]
        wall = (axis == 0) & (census.i[face_rows] == n)
        faces = (fplan, n2, n3, fallback_points, missing, dict(
            common_value_slot=common.astype(np.int32), common_gradient_slot=common_g.astype(np.int32),
            lower_slot=np.maximum(lower, 0).astype(np.int32), upper_slot=np.maximum(upper, 0).astype(np.int32),
            lower_present=lower_present, upper_present=upper_present, common_conditioned=common_cond,
            lower_conditioned=lower_cond & lower_present, upper_conditioned=upper_cond & upper_present,
            common_neumann_target=common_target, side_neumann_target=side_target,
            census_row=face_rows, p07_id=p07_id.astype(np.int64), axis=axis.astype(np.int32),
            lower_owner=census.owner_lo[face_rows].astype(np.int32),
            upper_owner=census.owner_hi[face_rows].astype(np.int32),
            wall=wall, wall_faces=np.flatnonzero(wall).astype(np.int32), collapsed=collapsed,
            p07_valid=(p07_id != NO_ID) & ~collapsed, face_multiplier=np.ones(Fc),
            h=np.asarray(geometry.p05_face_h), jac=np.abs(np.asarray(geometry.p05_face_jacobian)),
            weight=np.asarray(geometry.p06_face_weight), J=np.asarray(geometry.p06_face_J),
            B=np.asarray(geometry.p06_face_B), K=np.asarray(geometry.p06_face_K), owner_volume=owner_volume))

    # --- R4 ------------------------------------------------------------------
    p07 = None
    if "p07" in include:
        ids = census.p07_id[p07_rows].astype(np.int64)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        if len(sorted_ids) > 1 and np.any(np.diff(sorted_ids) == 0):
            raise ValueError("p07_rows: duplicate P07 ids (alias slots must be excluded)")
        endpoints = np.full((int(sorted_ids[-1]) + 1 if len(sorted_ids) else 0, 2), -1, dtype=np.int64)
        endpoints[ids, 0] = census.owner_lo[p07_rows]
        endpoints[ids, 1] = census.owner_hi[p07_rows]
        iplan = lower_integrated_chunks(_source(p07_chunks)(), grid=grid, endpoints=endpoints,
                                        owner_volume=owner_volume, select=RowSelection(requests=("R4",)))
        face_id = np.asarray(iplan.face_entity_id, dtype=np.int64)
        if len(face_id) != len(ids) or not np.array_equal(np.sort(face_id), sorted_ids):
            raise ValueError("p07: the R4 chunks must hold exactly the P07 faces of p07_rows")
        census_row = p07_rows[order][np.searchsorted(sorted_ids, face_id)] if len(face_id) else p07_rows[:0]
        conditioned = np.zeros(len(face_id), dtype=bool)
        for b in iplan.payload.batches:
            conditioned[b.face_ids[b.conditioned]] = True
        family = census.family[census_row].astype(np.int16)
        if np.any(conditioned != np.isin(family, tuple(_P07_NEUMANN_DEGREE))):
            raise ValueError("p07: conditioned integrated rows disagree with families 1, 2, 4")
        p07_neumann = lower_neumann("R4")
        if p07_neumann is None:
            if conditioned.any():
                raise ValueError("p07: conditioned faces without R4 Neumann rows")
            neumann_face = np.zeros(0, dtype=np.int32)
            perm = None
            integrand = np.zeros((0, Q3_NODES, 3))
        else:
            perm = np.lexsort((p07_neumann.quad_node, p07_neumann.entity_id))
            ent, node = p07_neumann.entity_id[perm], p07_neumann.quad_node[perm]
            if len(ent) % Q3_NODES or np.any(node != np.tile(np.arange(Q3_NODES), len(ent) // Q3_NODES)) \
                    or np.any(ent.reshape(-1, Q3_NODES) != ent.reshape(-1, Q3_NODES)[:, :1]):
                raise ValueError("p07: R4 Neumann rows must be 9 nodes per face")
            cond_id = ent[::Q3_NODES]
            if np.any(np.diff(cond_id) <= 0):
                raise ValueError("p07: duplicate R4 Neumann faces")
            by_id = np.argsort(face_id, kind="stable")
            neumann_face = by_id[_positions(face_id[by_id], cond_id, "p07: R4 Neumann entity ids")]
            if not np.array_equal(np.sort(neumann_face), np.flatnonzero(conditioned)):
                raise ValueError("p07: R4 Neumann rows do not cover exactly the conditioned faces")
            fpos = _positions(face_rows, census_row[neumann_face], "p07: conditioned faces in face_rows")
            if geometry.face_points.shape[0] != len(face_rows):
                raise ValueError("geometry face arrays must be aligned with face_rows")
            integrand = np.asarray(contract_face_tensor(
                np.asarray(geometry.p07_weight)[fpos], np.asarray(geometry.p07_face_tensor)[fpos],
                census.axis[census_row[neumann_face]]))
            neumann_face = neumann_face.astype(np.int32)
        in_faces = np.isin(census_row, face_rows)
        face_index = np.full(len(census_row), -1, dtype=np.int64)
        face_index[in_faces] = np.searchsorted(face_rows, census_row[in_faces])
        p07 = (iplan, p07_neumann, perm, dict(
            conditioned=conditioned, neumann_face=neumann_face, integrand=integrand, p07_id=face_id,
            census_row=census_row.astype(np.int64), face_index=face_index, family=family,
            lower_owner=iplan.payload.lower_owner.astype(np.int32),
            upper_owner=iplan.payload.upper_owner.astype(np.int32), owner_volume=owner_volume))

    # --- global boundary tables --------------------------------------------
    d_parts, d_names = [], []
    if cells is not None:
        d_parts.append(cells[0].boundary_points); d_names.append("cells")
    if faces is not None:
        d_parts.append(faces[0].boundary_points); d_names.append("faces")
        d_parts.append(faces[3]); d_names.append("fallback")
    if p07 is not None:
        d_parts.append(p07[0].boundary_points); d_names.append("p07")
    dirichlet_points, d_remaps = _global_table(d_parts)
    d_remap = dict(zip(d_names, d_remaps))
    Qd = len(dirichlet_points)

    n_parts, n_names = [], []
    for name, nplan in (("cells", None if cells is None else cells[1]),
                        ("faces_common", None if faces is None else faces[1]),
                        ("faces_side", None if faces is None else faces[2]),
                        ("p07", None if p07 is None else p07[1])):
        if nplan is not None:
            n_parts.append(nplan.wall_points); n_names.append(name)
    neumann_points, n_remaps = _global_table(n_parts)
    n_remap = dict(zip(n_names, n_remaps))

    cell_plan = face_plan = p07_plan = None
    if cells is not None:
        cplan, cn, fields = cells
        cell_plan = CellPlan(
            rows=_remap_source_payload(cplan.payload, d_remap["cells"], Qd),
            neumann=None if cn is None else _neumann_rows(cn, None, n_remap["cells"]), **fields)
    if faces is not None:
        fplan, n2, n3, _fallback, missing, fields = faces
        Qf = fields["common_value_slot"].shape[1]
        fallback_query = np.zeros((len(face_rows), Qf), dtype=np.int32)
        if missing.any():
            fallback_query[missing] = d_remap["fallback"].reshape(-1, Qf)
        face_plan = FacePlan(
            rows=_remap_source_payload(fplan.payload, d_remap["faces"], Qd), fallback_query=fallback_query,
            common_neumann=None if n2 is None else _neumann_rows(n2, None, n_remap["faces_common"]),
            side_neumann=None if n3 is None else _neumann_rows(n3, None, n_remap["faces_side"], gradients=False),
            has_missing_side=bool(missing.any()), **fields)
    if p07 is not None:
        iplan, pn, perm, fields = p07
        p07_plan = P07Plan(
            rows=_remap_integrated(iplan, d_remap["p07"], Qd),
            neumann=None if pn is None else _neumann_rows(pn, perm, n_remap["p07"], values=False), **fields)
    return PerpendicularPlan(cell_plan, face_plan, p07_plan, dirichlet_points, neumann_points, int(n))


# --------------------------------------------------------------------------
# Artifact entry point
# --------------------------------------------------------------------------

def lower_perpendicular_plan_from_artifact(root, n: int, *, grid: LoaderGrid, census: FaceCensus,
                                           geometry: GeometryArrays, raw_volume, owner_volume,
                                           identity: dict | None = None,
                                           include: Sequence[str] = ("cells", "faces", "p07"),
                                           raw_ids=None, face_rows=None, p07_rows=None) -> PerpendicularPlan:
    """Lower ``<root>/N<n>/`` (a v2/v3 row artifact) chunk file by chunk file (streaming).

    Schema and per-chunk sha256 are checked (``identity`` too, when given; ``None`` skips it). Point
    chunks are decoded factored (tensor sources are not expanded). Only the groups needed by ``include``
    are read; the Neumann chunk files are pre-filtered by their stage tag when the file names carry one
    (``neumann_<stage>_<index>.npz``)."""
    grid_dir = Path(root) / f"N{int(n)}"
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no row artifact manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in art.SUPPORTED_SCHEMAS:
        raise ValueError(f"row artifact schema mismatch: {manifest.get('schema')!r}")
    if identity is not None and manifest.get("identity") != art._json_safe(identity):
        raise ValueError("row artifact identity mismatch")
    identity = manifest.get("identity", {})
    include = tuple(include)

    def entries(group, stage=None):
        for entry in manifest.get("chunks", {}).get(group, ()):
            if stage is None or f"_{stage}_" in Path(entry["file"]).name or \
                    not any(f"_{s}_" in Path(entry["file"]).name for s in ("cells", "faces", "p07")):
                yield entry

    def read(entry):
        data = (grid_dir / entry["file"]).read_bytes()
        if art.hash_bytes(data) != entry["sha256"]:
            raise ValueError(f"row artifact chunk corrupted: {entry['file']}")
        return data

    def point_chunks(group):
        return lambda: (FactoredChunk(*art.decode_chunk_factored(group, read(e))) for e in entries(group))

    def p07_chunks():
        return (art.decode_chunk("p07", read(e)) for e in entries("p07"))

    def neumann_chunks():
        stages = [s for s, part in (("cells", "cells"), ("faces", "faces"), ("p07", "p07")) if part in include]
        seen = set()
        for stage in stages:
            for e in entries("neumann", stage):
                if e["file"] not in seen:
                    seen.add(e["file"])
                    yield art.decode_chunk("neumann", read(e))

    return lower_perpendicular_plan(
        grid=grid, census=census, geometry=geometry, raw_volume=raw_volume, owner_volume=owner_volume,
        cell_chunks=point_chunks("cells") if "cells" in include else (),
        face_chunks=point_chunks("faces") if "faces" in include else (),
        p07_chunks=p07_chunks if "p07" in include else (), neumann_chunks=neumann_chunks,
        raw_ids=raw_ids, face_rows=face_rows, p07_rows=p07_rows, include=include)


# --------------------------------------------------------------------------
# In-memory rows of a bounded owner set
# --------------------------------------------------------------------------

class OwnerRowChunks(NamedTuple):
    cells: tuple
    faces: tuple
    p07: tuple
    neumann: tuple


def pack_owner_rows(row_index: Mapping, neumann_index: Mapping, *, raw_ids, face_rows, p07_ids,
                    include: Sequence[str] = ("cells", "faces", "p07"), face_nodes: int = Q3_NODES) -> OwnerRowChunks:
    """Pack the row dictionaries of ``p_shared.owner_closure.build_owner_rows`` into one chunk per group.

    ``row_index`` maps ``(request, entity_id) -> PointRows`` (R1: raw id; R2: census row; R3: ``2 * census
    row + side``) and ``entity_id -> IntegratedFaceRow`` (R4: P07 id, an ``int`` key);
    ``neumann_index`` maps ``(request, entity_id, quad_node) -> NeumannPointRows``. Only the R3 sides
    that have a row are packed; a conditioned row needs its Neumann rows (packed for the whole set
    of nodes 0..8 (R4) or 0..``face_nodes``-1 (R2/R3; 9 q3, 4 q2), or node 0 for R1). Row objects are packed with the artifact packers (a Python loop
    over the bounded row set; ``bc_variant`` and the informational point-row ``radial_degree`` are not
    kept by the closure and are packed as ``"D"``/``""`` and 0)."""
    include = tuple(include)
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    face_rows = np.asarray(face_rows, dtype=np.int64)
    p07_ids = np.asarray(p07_ids, dtype=np.int64)

    def point_chunk(items):
        rows = [row for _, _, row in items]
        return art.pack_point_rows(
            rows, request=[r for r, _, _ in items], entity_id=[e for _, e, _ in items],
            bc_variant=["D" if row.boundary_conditioned else "" for row in rows], radial_degree=0,
            store_gradient=[not str(r).startswith("R3") for r, _, _ in items])

    neumann_items = []                              # (request, entity, node, degree, row)

    def add_neumann(request, entity, nodes, degree=0):
        for q in nodes:
            key = (request, int(entity), int(q))
            if key not in neumann_index:
                raise ValueError(f"no Neumann row for {key}")
            neumann_items.append((request, int(entity), int(q), degree, neumann_index[key]))

    cells = faces = p07 = ()
    if "cells" in include:
        items = []
        for raw in raw_ids:
            row = row_index[("R1", int(raw))]
            items.append(("R1", int(raw), row))
            if row.boundary_conditioned:
                add_neumann("R1", raw, (0,))
        cells = (point_chunk(items),)
    if "faces" in include:
        items = []
        for ridx in face_rows:
            row = row_index[("R2", int(ridx))]
            items.append(("R2", int(ridx), row))
            if row.boundary_conditioned:
                add_neumann("R2", ridx, range(face_nodes))
            side_conditioned = False
            for side in (0, 1):
                side_row = row_index.get(("R3", int(ridx) * 2 + side))
                if side_row is not None:
                    items.append(("R3", int(ridx) * 2 + side, side_row))
                    side_conditioned |= bool(side_row.boundary_conditioned)
            if side_conditioned:
                add_neumann("R3", ridx, range(face_nodes))
        faces = (point_chunk(items),)
    if "p07" in include:
        rows = [row_index[int(pid)] for pid in p07_ids]
        p07 = (art.pack_integrated_rows(rows, entity_id=p07_ids),)
        for pid, row in zip(p07_ids, rows):
            if row.boundary_conditioned:
                add_neumann("R4", pid, range(Q3_NODES), _P07_NEUMANN_DEGREE.get(int(row.family), 0))
    neumann = ()
    if neumann_items:
        neumann = (art.pack_neumann_rows(
            [it[4] for it in neumann_items], entity_id=[it[1] for it in neumann_items],
            quad_node=[it[2] for it in neumann_items], request=[it[0] for it in neumann_items],
            radial_degree=[it[3] for it in neumann_items]),)
    return OwnerRowChunks(cells, faces, p07, neumann)


def lower_perpendicular_plan_from_rows(row_index: Mapping, neumann_index: Mapping, *, grid: LoaderGrid,
                                       census: FaceCensus, geometry: GeometryArrays, raw_volume, owner_volume,
                                       raw_ids, face_rows, p07_rows,
                                       include: Sequence[str] = ("cells", "faces", "p07")) -> PerpendicularPlan:
    """Lower the in-memory rows of a bounded owner set (``build_owner_rows``: ``row_index``,
    ``neumann_index``, ``raw_ids``, ``face_row_indices``, ``p07_row_indices`` and its ``geometry``).

    The plan is the same object the artifact path builds, restricted to the given raw ids / face rows /
    P07 rows: ``geometry`` is aligned with ``raw_ids`` and ``face_rows`` as ``build_owner_rows`` builds
    it. ``owner_volume`` stays the full per-owner vector, so an owner's action is exact only if all
    its incident faces and raw cells are in the set (the owner closure)."""
    p07_ids = census.p07_id[np.asarray(p07_rows, dtype=np.int64)]
    packed = pack_owner_rows(row_index, neumann_index, raw_ids=raw_ids, face_rows=face_rows, p07_ids=p07_ids,
                             include=include, face_nodes=int(geometry.face_points.shape[1]))
    return lower_perpendicular_plan(
        grid=grid, census=census, geometry=geometry, raw_volume=raw_volume, owner_volume=owner_volume,
        cell_chunks=packed.cells, face_chunks=packed.faces, p07_chunks=packed.p07,
        neumann_chunks=packed.neumann, raw_ids=raw_ids, face_rows=face_rows, p07_rows=p07_rows,
        include=include)
