"""Vectorized NumPy loader: in-memory row-artifact chunks -> JAX runtime payloads.

Scope (P08 step 2a, task B; see
``work/p08_step2_layout_loader_design_20260929/design.md`` section 3): consume
the in-memory chunk dataclasses of ``drbx.stencils.artifact``
(``PointRowChunk``, ``IntegratedRowChunk``, ``NeumannRowChunk``) -- whatever
the on-disk schema decoded them from -- and produce

- ``lower_point_chunks``   -> ``SourceRowPlan`` (a source-major
  ``drbx.native.fci_perpendicular_source_rows.SourceRowPayload``),
- ``lower_integrated_chunks`` -> the existing ``IntegratedFacePayload``
  (applied by the qualified ``apply_integrated_face_rows`` /
  ``scatter_integrated_face_flux``),
- ``lower_neumann_chunks`` -> the existing ``NeumannPayload``.

The per-row Python lowerings in ``drbx.native.fci_perpendicular_*_rows`` are the
equivalence references and are not touched. Nothing here loops per row, per
target or per donor: loops run over chunks, buckets and bounded blocks of
sources, and everything inside is array algebra. Chunks may be a lazy iterable;
each is consumed once and only its bucket pieces (never the chunk) are kept, so
peak memory is the output payload plus one chunk's temporaries (Neumann rows
additionally pad to one global width, see ``lower_neumann_chunks``).

Conventions kept from the reference lowerings:

- donor widths are bucketed up to a multiple of 16; padded donors carry id 0,
  weight 0 and (conditioned rows) query id 0;
- conditioned rows keep the Dirichlet-lift layout (donor and target boundary
  query ids into one deduplicated query table);
- the eta offset of a donor owner is taken over the owner's member planes
  ``members(owner) % n`` (raw id modulo n), wrapped
  ``(k - target_plane + n//2) % n - n//2``, with target plane
  ``argmin |centers[2] - z|``. Point rows evaluate this per target, integrated
  rows at q3 node 4; the min/max over targets and donors is returned per source
  (per face);
- boundary query points are deduplicated by their exact float64 bit pattern
  (so ``-0.0`` and ``0.0`` are distinct points, unlike a float-tuple key).

Tensor-encoded sources (task D2; design section 6): ``lower_point_chunks`` also takes the
factored decode of a chunk (``drbx.stencils.artifact.decode_chunk_factored`` gives
``(chunk, tensor_rows, is_tensor)``; :class:`FactoredChunk`), or files
(:func:`lower_point_files`, :func:`iter_factored_point_chunks`). CSR sources go to the
buckets unchanged; the tensor sources are never expanded: their per-chunk factor tables
are merged into grid-global tables (deduplicated by exact bit pattern) and their targets
become ``TensorRowBatch`` es (a few table indices per target), numbered in the same
output slots as the CSR sources of the chunk. A paired source (a ``cell_stencil="symmetric"`` cell row ``1/2 (A + B)``:
two consecutive ``TensorRows`` sources, ``TensorRows.pair_part``) becomes a target of a *paired* batch, which carries
the index arrays of B as a ``TensorMirror`` beside those of A; the kernel contracts both from the same tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import json

import numpy as np

from drbx.native.fci_perpendicular_integrated_rows import (
    IntegratedFaceBatch, IntegratedFacePayload)
from drbx.native.fci_perpendicular_neumann_rows import NeumannPayload
from drbx.native.fci_perpendicular_source_rows import (
    SourceRowBatch, SourceRowPayload)
from drbx.native.fci_perpendicular_tensor_rows import (
    DONOR_BLOCK, TensorMirror, TensorRowBatch, TensorTables, tensor_nbytes)
from drbx.stencils.artifact import (
    BC_VARIANTS, REQUEST_KINDS, SUPPORTED_SCHEMAS, IntegratedRowChunk, NeumannRowChunk,
    PointRowChunk, _json_safe, decode_chunk_factored, hash_bytes)
from drbx.stencils.tensor_rows import TensorRows, _dedupe, _raw_ids

#: Donor widths are padded to a multiple of this (matches the reference lowerings).
WIDTH_MULTIPLE = 16
#: Bound on padded elements / donor entries touched by one vectorized block.
_BLOCK_ELEMENTS = 1 << 22

_INT32_MAX = np.iinfo(np.int32).max


# --------------------------------------------------------------------------
# Grid topology and selection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoaderGrid:
    """The small explicit topology the loader needs (no field, no geometry).

    ``raw_to_owner`` maps a raw cell id (C order over ``(n, n, n)``, eta
    fastest) to its compact owner id; ``eta_centers`` are ``centers[2]``.
    ``plane_ptr``/``plane_index`` are derived: the sorted unique eta planes
    ``members(owner) % n`` of each owner, as CSR.
    """

    n: int
    n_owners: int
    raw_to_owner: np.ndarray
    eta_centers: np.ndarray
    plane_ptr: np.ndarray
    plane_index: np.ndarray

    @classmethod
    def from_arrays(cls, *, n: int, raw_to_owner, eta_centers, n_owners: int | None = None):
        n = int(n)
        ro = np.asarray(raw_to_owner, dtype=np.int64).reshape(-1)
        eta = np.asarray(eta_centers, dtype=np.float64).reshape(-1)
        if ro.shape != (n ** 3,) or eta.shape != (n,):
            raise ValueError("raw_to_owner must have n**3 entries and eta_centers n entries")
        if n_owners is None:
            n_owners = int(ro.max()) + 1
        n_owners = int(n_owners)
        if np.any(ro < 0) or np.any(ro >= n_owners):
            raise ValueError("raw_to_owner has entries outside [0, n_owners)")
        pair = np.unique(ro * n + np.arange(n ** 3, dtype=np.int64) % n)
        owner = pair // n
        counts = np.bincount(owner, minlength=n_owners)
        if np.any(counts == 0):
            raise ValueError("an owner has no raw members")
        ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        return cls(n, n_owners, ro, eta, ptr, (pair % n).astype(np.int32))

    @classmethod
    def from_context(cls, context):
        """From a ``PointRowContext`` (``raw_to_owner`` is ``context.ro``)."""
        return cls.from_arrays(n=context.n, raw_to_owner=context.ro,
                               eta_centers=context.centers[2], n_owners=len(context.vol))


@dataclass(frozen=True)
class RowSelection:
    """Restrict a lowering to a subset of rows; ``None`` means no restriction.

    ``requests`` are ``REQUEST_KINDS`` (for Neumann rows the originating
    request, for integrated rows only ``"R4"`` exists); ``bc_variants`` are
    ``BC_VARIANTS`` (point rows only, ignored for the other kinds);
    ``entity_ids`` restricts the face-or-raw entity id.
    """

    requests: tuple[str, ...] | None = None
    bc_variants: tuple[str, ...] | None = None
    entity_ids: np.ndarray | None = None

    def __post_init__(self):
        if self.requests is not None:
            bad = set(self.requests) - set(REQUEST_KINDS)
            if bad:
                raise ValueError(f"unsupported request kinds in selection: {sorted(bad)}")
        if self.bc_variants is not None:
            bad = set(self.bc_variants) - set(BC_VARIANTS)
            if bad:
                raise ValueError(f"unsupported BC variants in selection: {sorted(bad)}")


def _selected(select, request, bc_variant, entity_id, n):
    mask = np.ones(n, dtype=bool)
    if select is None:
        return mask
    if select.requests is not None and request is not None:
        mask &= np.isin(request, list(select.requests))
    if select.bc_variants is not None and bc_variant is not None:
        mask &= np.isin(bc_variant, list(select.bc_variants))
    if select.entity_ids is not None:
        mask &= np.isin(entity_id, np.asarray(select.entity_ids))
    return mask


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------

def _blocks(counts, limit=_BLOCK_ELEMENTS):
    """Yield ``(i0, i1)`` runs of items whose ``counts`` sum to about ``limit``."""
    cum = np.cumsum(np.asarray(counts, dtype=np.int64))
    n = len(cum)
    i0 = 0
    while i0 < n:
        base = int(cum[i0 - 1]) if i0 else 0
        i1 = min(max(int(np.searchsorted(cum, base + limit, side="right")), i0 + 1), n)
        yield i0, i1
        i0 = i1


def _ragged(starts, counts):
    """Flat indices of the ragged runs ``[starts[i], starts[i] + counts[i])`` and run owner."""
    counts = np.asarray(counts, dtype=np.int64)
    owner = np.repeat(np.arange(len(counts)), counts)
    first = np.cumsum(counts) - counts
    flat = (np.arange(int(counts.sum()), dtype=np.int64) - first[owner]
            + np.asarray(starts, dtype=np.int64)[owner])
    return flat, owner


def _pad_gather(source, starts, counts, width, dtype=None):
    """``out[..., j] = source[starts + j]`` for ``j < counts`` else 0 (width columns)."""
    source = np.asarray(source)
    lane = np.arange(width)
    mask = lane < np.asarray(counts)[..., None]
    dtype = source.dtype if dtype is None else dtype
    if source.shape[-1] == 0 or width == 0:
        return np.zeros(np.shape(starts) + (width,), dtype=dtype)
    index = np.where(mask, np.asarray(starts)[..., None] + lane, 0)
    return np.where(mask, source[index], 0).astype(dtype, copy=False)


def _pad_gather_gradient(gradient, starts, counts, width):
    """``gradient`` is ``(3, nnz)``; returns ``(..., 3, width)`` float64, zero padded."""
    lane = np.arange(width)
    mask = lane < np.asarray(counts)[..., None]
    if gradient.shape[-1] == 0 or width == 0:
        return np.zeros(np.shape(starts) + (3, width))
    index = np.where(mask, np.asarray(starts)[..., None] + lane, 0)
    out = np.where(mask, gradient[:, index], 0).astype(np.float64, copy=False)   # (3, ..., w)
    return np.ascontiguousarray(np.moveaxis(out, 0, -2))


def _unique_rows(points):
    """Deduplicate ``(M, 3)`` float64 points by exact bit pattern.

    Returns ``(unique, inverse)`` with the unique points ordered by first
    appearance.
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if not len(pts):
        return pts, np.zeros(0, dtype=np.int64)
    void = pts.view(np.dtype((np.void, 24))).ravel()
    _, first, inverse = np.unique(void, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return pts[first[order]], rank[inverse.reshape(-1)]


class _QueryPieces:
    """Accumulates per-block query points; ids are offsets into their concatenation."""

    def __init__(self):
        self.tables: list[np.ndarray] = []
        self.count = 0

    def add(self, points):
        unique, inverse = _unique_rows(points)
        offset = self.count
        self.tables.append(unique)
        self.count += len(unique)
        if self.count > _INT32_MAX:
            raise ValueError("boundary query table exceeds int32 ids")
        return offset + inverse

    def finalize(self):
        """Global dedup: ``(points (Q, 3), remap (count,) int32)``."""
        if not self.tables:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
        points, inverse = _unique_rows(np.concatenate(self.tables))
        self.tables = []
        return points, inverse.astype(np.int32)


def _nearest_plane(centers, z):
    """``argmin |centers - z|`` per entry (first minimum, as ``np.argmin``)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(len(z), dtype=np.int64)
    step = max(1, _BLOCK_ELEMENTS // max(len(centers), 1))
    for i in range(0, len(z), step):
        out[i:i + step] = np.argmin(np.abs(centers[None, :] - z[i:i + step, None]), axis=1)
    return out


def _eta_offset_range(grid: LoaderGrid, donor_flat, donor_ptr, target_plane, target_ptr):
    """Min/max wrapped eta offset per unit (source or face).

    ``donor_flat`` holds each unit's donor owners, unit ``u`` at
    ``donor_ptr[u]:donor_ptr[u+1]``; ``target_plane`` holds each unit's target
    eta planes, unit ``u`` at ``target_ptr[u]:target_ptr[u+1]``. The offset of
    member plane ``k`` from target plane ``p`` is ``(k - p + n//2) % n - n//2``,
    monotone in ``v = (k - p + n//2) % n``, so min/max come from the first and
    last member plane in that rotated order. Units without donors or targets
    return 0.
    """
    n = grid.n
    units = len(donor_ptr) - 1
    out_min = np.zeros(units, dtype=np.int32)
    out_max = np.zeros(units, dtype=np.int32)
    donor_ptr = np.asarray(donor_ptr, dtype=np.int64)
    target_ptr = np.asarray(target_ptr, dtype=np.int64)
    weight = np.diff(donor_ptr) + np.diff(target_ptr) * max(n, 1)
    lane = np.arange(n)
    for u0, u1 in _blocks(weight, _BLOCK_ELEMENTS):
        ub = u1 - u0
        d0, d1 = int(donor_ptr[u0]), int(donor_ptr[u1])
        t0, t1 = int(target_ptr[u0]), int(target_ptr[u1])
        owners = np.asarray(donor_flat[d0:d1], dtype=np.int64)
        unit_of_donor = np.repeat(np.arange(ub), np.diff(donor_ptr[u0:u1 + 1]))
        nplanes = grid.plane_ptr[owners + 1] - grid.plane_ptr[owners]
        flat, owner_of = _ragged(grid.plane_ptr[owners], nplanes)
        present = np.zeros((ub, n), dtype=bool)
        present[unit_of_donor[owner_of], grid.plane_index[flat]] = True
        unit_of_target = np.repeat(np.arange(ub), np.diff(target_ptr[u0:u1 + 1]))
        plane = np.asarray(target_plane[t0:t1], dtype=np.int64)
        rotated = present[unit_of_target[:, None], (lane[None, :] + (plane - n // 2)[:, None]) % n]
        has = rotated.any(axis=1)
        vmin = np.where(has, np.argmax(rotated, axis=1), n)
        vmax = np.where(has, n - 1 - np.argmax(rotated[:, ::-1], axis=1), -1)
        counts = np.diff(target_ptr[u0:u1 + 1])
        nonempty = counts > 0
        if not nonempty.any():
            continue
        starts = (target_ptr[u0:u1] - t0)[nonempty]
        umin = np.minimum.reduceat(vmin, starts)
        umax = np.maximum.reduceat(vmax, starts)
        found = umax >= 0
        rows = np.flatnonzero(nonempty) + u0
        out_min[rows] = np.where(found, umin - n // 2, 0)
        out_max[rows] = np.where(found, umax - n // 2, 0)
    return out_min, out_max


def _check_donor_ids(donors, grid, what):
    if len(donors) and (donors.min() < 0 or donors.max() >= grid.n_owners):
        raise ValueError(f"{what} has invalid compact donor ids")


def _widths16(counts):
    counts = np.asarray(counts, dtype=np.int64)
    return WIDTH_MULTIPLE * ((counts + WIDTH_MULTIPLE - 1) // WIDTH_MULTIPLE)


def _string_codes(values, table, *, extend=False):
    """Codes of a string array in ``table`` (a list). Unseen values are appended when
    ``extend`` is true and raise otherwise. Loops only over the distinct strings."""
    unique, inverse = np.unique(values, return_inverse=True)
    codes = np.empty(len(unique), dtype=np.int64)
    for i, name in enumerate(unique):
        name = str(name)
        if name not in table:
            if not extend:
                raise ValueError(f"unsupported tag value in chunk: {name!r}")
            table.append(name)
        codes[i] = table.index(name)
    return codes[inverse.reshape(-1)]


def _tuple_nbytes(arrays):
    return sum(int(a.nbytes) for a in arrays if a is not None)


# --------------------------------------------------------------------------
# Point rows (R1/R2/R3) -> source-major plan
# --------------------------------------------------------------------------

class SourceRowTargets(NamedTuple):
    """Flat per-target index arrays, in output-slot order (``value_slot == arange``)."""

    request: np.ndarray         # (T,) int8, index into REQUEST_KINDS
    entity_id: np.ndarray       # (T,) int64
    quad_node: np.ndarray       # (T,) int16
    bc_variant: np.ndarray      # (T,) int8, index into BC_VARIANTS
    radial_degree: np.ndarray   # (T,) int16
    family: np.ndarray          # (T,) int16, index into SourceRowPlan.family_names
    conditioned: np.ndarray     # (T,) bool
    source: np.ndarray          # (T,) int32, index of the source (into the per-source arrays)
    value_slot: np.ndarray      # (T,) int32, row of the value output
    gradient_slot: np.ndarray   # (T,) int32, row of the gradient output, -1 without gradient
    target_points: np.ndarray   # (T, 3) float64


class BucketSummary(NamedTuple):
    family: str
    conditioned: bool
    has_gradient: bool
    nodes: int
    width: int
    sources: int
    targets: int
    donor_entries: int      # sum of true (unpadded) donor counts over the sources
    payload_bytes: int
    encoding: str = "csr"   # "tensor" rows: nodes 0 (varies), width 112 = the donor block per target


@dataclass(frozen=True)
class SourceRowPlan:
    payload: SourceRowPayload
    boundary_points: np.ndarray             # (Q, 3) float64, deduplicated by bit pattern
    targets: SourceRowTargets
    source_target_ptr: np.ndarray           # (S+1,) int64, targets of source s
    source_eta_offset_min: np.ndarray       # (S,) int32
    source_eta_offset_max: np.ndarray       # (S,) int32
    family_names: tuple[str, ...]
    bucket_summary: tuple[BucketSummary, ...]
    boundary_kind: str = "dirichlet"
    tensor_table_bytes: int = 0             # grid-global tensor tables (not part of any bucket's bytes)


class _PointChunkSources(NamedTuple):
    t0: np.ndarray          # (S,) first target of each non-empty source
    q: np.ndarray           # (S,) targets per source
    donor_count: np.ndarray  # (S,)
    target_source: np.ndarray  # (T,) source of each target


def _adjacent_change_ok(values, first_mask, what):
    """Raise if ``values`` changes anywhere inside a source (between adjacent targets)."""
    if len(values) > 1:
        changed = values[1:] != values[:-1]
        if np.any(changed & ~first_mask[1:]):
            raise ValueError(f"point-row chunk: targets of a source disagree on {what}")


def _check_point_chunk(chunk: PointRowChunk) -> _PointChunkSources:
    """Vectorized structural checks; every target of a source must share its tags,
    donor list, ``donor_query`` and gradient storage."""
    sp = np.asarray(chunk.source_ptr, dtype=np.int64)
    n_targets = len(chunk.entity_id)
    if len(sp) < 1 or sp[0] != 0 or sp[-1] != n_targets or np.any(np.diff(sp) < 0):
        raise ValueError("point-row chunk: source_ptr is not a partition of the targets")
    dptr = np.asarray(chunk.donor_ptr, dtype=np.int64)
    gptr = np.asarray(chunk.gradient_ptr, dtype=np.int64)
    if (len(dptr) != n_targets + 1 or len(gptr) != n_targets + 1 or dptr[0] != 0
            or np.any(np.diff(dptr) < 0)):
        raise ValueError("point-row chunk: donor_ptr/gradient_ptr do not match the targets")
    if not (dptr[-1] == len(chunk.donor) == len(chunk.value) == len(chunk.donor_query)):
        raise ValueError("point-row chunk: donor/value/donor_query lengths disagree with donor_ptr")
    if chunk.gradient.shape != (3, int(gptr[-1])):
        raise ValueError("point-row chunk: gradient shape disagrees with gradient_ptr")
    for name in ("request", "family", "conditioned", "bc_variant", "radial_degree",
                 "has_gradient", "quad_node"):
        if len(getattr(chunk, name)) != n_targets:
            raise ValueError(f"point-row chunk: {name} does not have one entry per target")
    q_all = np.diff(sp)
    nonempty = q_all > 0
    t0 = sp[:-1][nonempty]
    q = q_all[nonempty]
    first_mask = np.zeros(max(n_targets, 1), dtype=bool)
    first_mask[t0] = True
    for name in ("request", "entity_id", "family", "conditioned", "bc_variant",
                 "radial_degree", "has_gradient"):
        _adjacent_change_ok(np.asarray(getattr(chunk, name)), first_mask, name)
    widths = np.diff(dptr)
    _adjacent_change_ok(widths, first_mask, "donor width")
    gwidths = np.diff(gptr)
    has_gradient = np.asarray(chunk.has_gradient, dtype=bool)
    if np.any(gwidths != np.where(has_gradient, widths, 0)):
        raise ValueError("point-row chunk: gradient_ptr disagrees with has_gradient/donor width")
    target_source = np.repeat(np.arange(len(t0)), q)
    # Donor lists (and donor_query of conditioned sources) must be identical per source.
    rest = np.flatnonzero(~first_mask[:n_targets])
    if len(rest):
        ref = t0[target_source[rest]]
        conditioned = np.asarray(chunk.conditioned, dtype=bool)
        for b0, b1 in _blocks(widths[rest]):
            t = rest[b0:b1]
            here, _ = _ragged(dptr[t], widths[t])
            there, owner = _ragged(dptr[ref[b0:b1]], widths[t])
            if np.any(chunk.donor[here] != chunk.donor[there]):
                raise ValueError("point-row chunk: donor ids differ within one source")
            entry_conditioned = conditioned[t][owner]
            if np.any(chunk.donor_query[here][entry_conditioned]
                      != chunk.donor_query[there][entry_conditioned]):
                raise ValueError("point-row chunk: donor_query differs within one source")
    return _PointChunkSources(t0, q, widths[t0], target_source)


class FactoredChunk(NamedTuple):
    """A point chunk as ``decode_chunk_factored`` returns it: tensor sources are not expanded.

    ``chunk`` stores zero donors and no value/gradient for the sources flagged in ``is_tensor``
    (per chunk source); ``tensor_rows`` holds their factors in chunk order: one row per tensor source, two (A then B,
    ``tensor_rows.pair_part``) for a paired one, so ``tensor_rows.logical_sources`` are the rows of the flagged sources.
    """

    chunk: PointRowChunk
    tensor_rows: TensorRows | None
    is_tensor: np.ndarray | None


def _unpack_point_item(item):
    """``(chunk, tensor_rows, is_tensor)`` of a ``PointRowChunk`` or a ``(chunk, tensor_rows, is_tensor)``."""
    if isinstance(item, PointRowChunk):
        return item, None, None
    chunk, tensor_rows, is_tensor = item
    if tensor_rows is None:
        return chunk, None, None
    is_tensor = np.asarray(is_tensor, dtype=bool)
    if len(is_tensor) != len(chunk.source_ptr) - 1:
        raise ValueError("point-row chunk: is_tensor must have one entry per source")
    return chunk, tensor_rows, is_tensor if is_tensor.any() else None


def lower_point_chunks(chunks: Iterable, *, grid: LoaderGrid,
                       select: RowSelection | None = None) -> SourceRowPlan:
    """Lower point-row chunks (R1/R2/R3) into a source-major ``SourceRowPlan``.

    Each item is a ``PointRowChunk`` (all sources CSR) or the factored decode
    ``(chunk, tensor_rows, is_tensor)`` of ``drbx.stencils.artifact.decode_chunk_factored``
    (:class:`FactoredChunk`); both kinds may be mixed, and ``chunks`` may be a lazy iterable
    (each item is consumed once). CSR sources are bucketed by
    ``(family, conditioned, has_gradient, nodes, width)`` with the donor width rounded up to a
    multiple of 16. Tensor sources (unconditioned singleton, ringwise, centered_radial) are
    never expanded: their targets go to ``SourceRowPayload.tensor_batches`` (one batch per
    (family, has_gradient, paired)) over grid-global factor tables, see
    ``drbx.native.fci_perpendicular_tensor_rows``; a paired source's targets go to a paired batch, which stores the
    index arrays of both parts, and its eta offsets are those of the union of their donors. Targets are numbered in output order:
    selected sources in chunk order (CSR and tensor alike), targets in stored order.
    ``select`` restricts requests, BC variants and entity ids before slots are assigned.
    Raises if a chunk violates the source invariants (shared donor list, donor query, tags and
    gradient storage within a source; tensor factors consistent with the grid and the tags).
    """
    families: list[str] = []
    buckets: dict[tuple, list[SourceRowBatch]] = {}
    bucket_counts: dict[tuple, list[int]] = {}
    pieces = _QueryPieces()
    tensor = _TensorAccumulator(grid)
    per_target: list[tuple] = []
    eta_min: list[np.ndarray] = []
    eta_max: list[np.ndarray] = []
    q_per_source: list[np.ndarray] = []
    n_targets = n_gradient = n_sources = 0
    for item in chunks:
        chunk, tensor_rows, is_tensor = _unpack_point_item(item)
        info = _check_point_chunk(chunk)
        if not len(info.t0):
            continue
        t0_all, q_all = info.t0, info.q
        request_s = np.asarray(chunk.request)[t0_all]
        keep = _selected(select, request_s, np.asarray(chunk.bc_variant)[t0_all],
                         np.asarray(chunk.entity_id)[t0_all], len(t0_all))
        if not keep.any():
            continue
        t0, q, d = t0_all[keep], q_all[keep], info.donor_count[keep]
        s_count = len(t0)
        conditioned = np.asarray(chunk.conditioned, dtype=bool)[t0]
        has_gradient = np.asarray(chunk.has_gradient, dtype=bool)[t0]
        family = _string_codes(np.asarray(chunk.family)[t0], families, extend=True)
        dptr = np.asarray(chunk.donor_ptr, dtype=np.int64)
        gptr = np.asarray(chunk.gradient_ptr, dtype=np.int64)
        # Tensor sources of the selection (chunk order); ``tensor_index`` are their rows in ``tensor_rows``.
        is_t = np.zeros(s_count, dtype=bool)
        if is_tensor is not None:
            nonempty = np.diff(np.asarray(chunk.source_ptr, dtype=np.int64)) > 0
            if np.any(is_tensor & ~nonempty):
                raise ValueError("point-row chunk: a tensor source has no targets")
            tensor_rows.check_pairs()
            first_row = tensor_rows.logical_sources            # rows of the tensor sources (the A part of a pair)
            if len(first_row) != int(np.count_nonzero(is_tensor)):
                raise ValueError(f"point-row chunk: the tensor rows hold {len(first_row)} sources for "
                                 f"{int(np.count_nonzero(is_tensor))} tensor sources")
            tensor_of_source = np.cumsum(is_tensor) - 1
            is_t = is_tensor[nonempty][keep]
            tensor_index = first_row[tensor_of_source[nonempty][keep][is_t]]
        # Output slots.
        node_base = np.cumsum(q) - q
        grad_q = np.where(has_gradient, q, 0)
        grad_base = np.cumsum(grad_q) - grad_q
        chunk_targets = int(q.sum())
        chunk_gradient = int(grad_q.sum())
        if n_targets + chunk_targets > _INT32_MAX or n_gradient + chunk_gradient > _INT32_MAX:
            raise ValueError("target count exceeds int32 slots")
        value_slot0 = n_targets + node_base
        gradient_slot0 = n_gradient + grad_base
        # Donors, per-source donor CSR, eta offset range (tensor sources have no stored donors).
        donor_flat, _ = _ragged(dptr[t0], d)
        donors = np.asarray(chunk.donor)[donor_flat]
        _check_donor_ids(donors, grid, "point row")
        target_index, target_owner = _ragged(t0, q)
        target_plane = _nearest_plane(grid.eta_centers, chunk.target_point[target_index, 2])
        source_donor_ptr = np.r_[0, np.cumsum(d)]
        source_target_ptr = np.r_[0, np.cumsum(q)]
        lo, hi = _eta_offset_range(grid, donors, source_donor_ptr, target_plane, source_target_ptr)
        if is_t.any():
            sel_t = np.flatnonzero(is_t)
            _check_tensor_sources(grid, tensor_rows, tensor_index, q[sel_t], d[sel_t], conditioned[sel_t],
                                  has_gradient[sel_t], np.asarray(families)[family[sel_t]])
            counts_t = q[sel_t]
            tensor_slot_ids, tensor_owner = _ragged(node_base[sel_t], counts_t)
            local = tensor_slot_ids - node_base[sel_t][tensor_owner]
            tensor_value_slot = value_slot0[sel_t][tensor_owner] + local
            tensor_gradient_slot = np.where(has_gradient[sel_t][tensor_owner],
                                            gradient_slot0[sel_t][tensor_owner] + local, -1)
            lo[sel_t], hi[sel_t] = tensor.add(
                tensor_rows, tensor_index, target_plane[tensor_slot_ids],
                tensor_value_slot, tensor_gradient_slot)
        eta_min.append(lo)
        eta_max.append(hi)
        q_per_source.append(q)
        # Per-target tables.
        node_in_source = np.arange(chunk_targets) - node_base[target_owner]
        gradient_slot = np.where(has_gradient[target_owner],
                                 n_gradient + grad_base[target_owner] + node_in_source, -1)
        request_codes = _string_codes(request_s[keep], _REQUEST_TABLE)
        bc_codes = _string_codes(np.asarray(chunk.bc_variant)[t0], _BC_TABLE)
        per_target.append((
            request_codes[target_owner].astype(np.int8),
            np.asarray(chunk.entity_id)[t0][target_owner].astype(np.int64),
            np.asarray(chunk.quad_node)[target_index].astype(np.int16),
            bc_codes[target_owner].astype(np.int8),
            np.asarray(chunk.radial_degree)[t0][target_owner].astype(np.int16),
            family[target_owner].astype(np.int16),
            conditioned[target_owner],
            (n_sources + target_owner).astype(np.int32),
            (n_targets + np.arange(chunk_targets)).astype(np.int32),
            gradient_slot.astype(np.int32),
            np.asarray(chunk.target_point, dtype=np.float64)[target_index],
        ))
        # Buckets (CSR sources only).
        csr = np.flatnonzero(~is_t)
        if len(csr):
            width = _widths16(d)
            if (np.any(q >= 64) or np.any(width // WIDTH_MULTIPLE >= 4096)
                    or len(families) >= 2 ** 14):
                raise ValueError("bucket key overflow (nodes >= 64, width >= 65536 or too many families)")
            key = ((((family * 2 + conditioned) * 2 + has_gradient) * 64 + q) * 4096
                   + width // WIDTH_MULTIPLE)
            order = csr[np.argsort(key[csr], kind="stable")]
            sorted_key = key[order]
            starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
            stops = np.r_[starts[1:], len(order)]
            for a, b in zip(starts, stops):
                g = order[a:b]
                fam, cond, hasg = int(family[g[0]]), bool(conditioned[g[0]]), bool(has_gradient[g[0]])
                nodes, w = int(q[g[0]]), int(width[g[0]])
                batch = _fill_source_batch(chunk, g, t0, d, nodes, w, cond, hasg,
                                           value_slot0, gradient_slot0, pieces)
                bkey = (fam, cond, hasg, nodes, w)
                buckets.setdefault(bkey, []).append(batch)
                bucket_counts.setdefault(bkey, [0, 0])
                bucket_counts[bkey][0] += int(d[g].sum())
        n_targets += chunk_targets
        n_gradient += chunk_gradient
        n_sources += s_count
    return _finish_source_plan(buckets, bucket_counts, pieces, per_target, eta_min, eta_max,
                               q_per_source, families, n_targets, n_gradient, tensor)


def iter_factored_point_chunks(root, n: int, identity: dict, groups=("cells", "faces")):
    """Yield ``FactoredChunk`` of every point chunk of a row artifact, one file at a time.

    Same checks as ``load_row_artifact`` (schema, identity, per-chunk sha256), but each chunk is
    decoded with ``decode_chunk_factored`` -- tensor sources are not expanded -- and only one
    chunk is alive at a time, so the iterator can feed ``lower_point_chunks`` directly.
    """
    grid_dir = Path(root) / f"N{int(n)}"
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no row artifact manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"row artifact schema mismatch: {manifest.get('schema')!r} not in {SUPPORTED_SCHEMAS!r}")
    if manifest.get("identity") != _json_safe(identity):
        raise ValueError("row artifact identity mismatch: geometry/topology/policy/platform inputs differ")
    for group in groups:
        if group not in ("cells", "faces"):
            raise ValueError(f"not a point-row group: {group!r}")
        for entry in manifest.get("chunks", {}).get(group, ()):
            data = (grid_dir / entry["file"]).read_bytes()
            if hash_bytes(data) != entry["sha256"]:
                raise ValueError(f"row artifact chunk corrupted: {entry['file']}")
            yield FactoredChunk(*decode_chunk_factored(group, data))


def lower_point_files(paths: Iterable, *, grid: LoaderGrid,
                      select: RowSelection | None = None) -> SourceRowPlan:
    """``lower_point_chunks`` straight from chunk files, without expanding tensor sources.

    Each item is a path ``.../cells_<i>.npz`` / ``.../faces_<i>.npz`` (the group is read from the
    file name) or a ``(group, path)`` pair; files are read and decoded one at a time
    (``decode_chunk_factored``). No manifest hash is checked here; use
    :func:`iter_factored_point_chunks` for a verified artifact directory.
    """
    def chunks():
        for item in paths:
            group, path = item if isinstance(item, tuple) else (Path(item).name.split("_")[0], item)
            if group not in ("cells", "faces"):
                raise ValueError(f"cannot tell the point-row group of {path!s}")
            yield FactoredChunk(*decode_chunk_factored(group, Path(path).read_bytes()))
    return lower_point_chunks(chunks(), grid=grid, select=select)


_REQUEST_TABLE = list(REQUEST_KINDS)
_BC_TABLE = list(BC_VARIANTS)


def _fill_source_batch(chunk, g, t0, d, nodes, w, conditioned, has_gradient,
                       value_slot0, gradient_slot0, pieces) -> SourceRowBatch:
    """Dense arrays of the sources ``g`` (one bucket), filled in bounded blocks."""
    count = len(g)
    lane = np.arange(nodes)
    donor_ids = np.zeros((count, w), dtype=np.int32)
    value = np.zeros((count, nodes, w))
    gradient = np.zeros((count, nodes, 3, w)) if has_gradient else None
    donor_query = np.zeros((count, w), dtype=np.int32) if conditioned else None
    target_query = np.zeros((count, nodes), dtype=np.int32) if conditioned else None
    dptr = np.asarray(chunk.donor_ptr, dtype=np.int64)
    gptr = np.asarray(chunk.gradient_ptr, dtype=np.int64)
    step = max(1, _BLOCK_ELEMENTS // max(nodes * max(w, 1) * (3 if has_gradient else 1), 1))
    for i in range(0, count, step):
        s = g[i:i + step]
        first = t0[s]
        targets = first[:, None] + lane                      # (b, nodes)
        ds = d[s]
        donor_ids[i:i + step] = _pad_gather(chunk.donor, dptr[first], ds, w, np.int32)
        value[i:i + step] = _pad_gather(chunk.value, dptr[targets], ds[:, None], w, np.float64)
        if has_gradient:
            gradient[i:i + step] = _pad_gather_gradient(chunk.gradient, gptr[targets],
                                                        ds[:, None], w)
        if conditioned:
            local = _pad_gather(chunk.donor_query, dptr[first], ds, w)
            valid = np.arange(w) < ds[:, None]
            used = local[valid]
            if len(used) and (used.min() < 0 or used.max() >= len(chunk.query_table)):
                raise ValueError("conditioned point row has an invalid donor query id")
            points = np.concatenate([chunk.query_table[used],
                                     np.asarray(chunk.target_point)[targets.reshape(-1)]])
            ids = pieces.add(points)
            block_dq = np.zeros(local.shape, dtype=np.int32)
            block_dq[valid] = ids[:len(used)]
            donor_query[i:i + step] = block_dq
            target_query[i:i + step] = ids[len(used):].reshape(len(s), nodes)
    value_slots = (value_slot0[g][:, None] + lane).astype(np.int32)
    gradient_slots = (gradient_slot0[g][:, None] + lane).astype(np.int32) if has_gradient else None
    return SourceRowBatch(value_slots, donor_ids, value, gradient_slots, gradient,
                          donor_query, target_query)


# --------------------------------------------------------------------------
# Tensor-encoded point sources (task D2)
# --------------------------------------------------------------------------

_FAMILY_TAGS = {"singleton": 0, "ringwise": 1, "centered_radial": 2}
_FAMILY_NAMES = tuple(_FAMILY_TAGS)
_RINGWISE = 1


def _check_tensor_sources(grid, tr: TensorRows, tensor_index, q, d, conditioned, has_gradient, tags):
    """Structural checks of the selected tensor sources against the chunk tags and the grid.

    ``tags`` are the family strings of the selected tensor sources; ``q``/``d`` their target and
    stored donor counts. Every table index the runtime arithmetic will derive an owner from is
    range checked here (theta/eta plane ids and radial layers < n) and the artifact's own
    ``raw_to_owner`` entries are compared with ``grid.raw_to_owner``; ring owners are checked
    when the sources' owners are derived (:func:`_tensor_eta_range`).
    """
    n = grid.n
    if tr.n != n:
        raise ValueError(f"tensor rows are for n={tr.n}, the grid has n={n}")
    if np.any(d != 0) or np.any(conditioned):
        raise ValueError("point-row chunk: a tensor source must be unconditioned and store no donors")
    if np.any(np.diff(tr.target_ptr)[tensor_index] != q):
        raise ValueError("point-row chunk: tensor rows and source_ptr disagree on the targets of a source")
    if np.any(tr.has_gradient[tensor_index] != has_gradient):
        raise ValueError("point-row chunk: tensor rows and the has_gradient tag disagree")
    names, inverse = np.unique(tags, return_inverse=True)
    codes = np.array([_FAMILY_TAGS.get(str(name), -1) for name in names], dtype=np.int64)
    if np.any(codes[inverse.reshape(-1)] != tr.family[tensor_index]):
        raise ValueError("point-row chunk: tensor rows and the family tag disagree")
    if tr.theta_index.size and (tr.theta_index.min() < 0 or tr.theta_index.max() >= n):
        raise ValueError("tensor rows: theta plane id outside the grid")
    if tr.eta_index.size and (tr.eta_index.min() < 0 or tr.eta_index.max() >= n):
        raise ValueError("tensor rows: eta plane id outside the grid")
    rows = np.concatenate([tensor_index, tensor_index[tr.pair_part[tensor_index] == 1] + 1])      # + the B parts
    layers = tr.layers[rows]
    rid = np.where(layers < 0, -layers.astype(np.int64) - 1, layers)
    if rid.min() < 0 or rid.max() >= n:
        raise ValueError("tensor rows: radial layer outside the grid")
    if len(tr.raw_ids):
        if tr.raw_ids.min() < 0 or tr.raw_ids.max() >= n ** 3 or np.any(
                grid.raw_to_owner[tr.raw_ids] != tr.raw_owner):
            raise ValueError("tensor rows: stored raw_to_owner entries disagree with the grid")


def _target_owners(grid, tr: TensorRows, tt, layout) -> np.ndarray:
    """Owner ids ``(m, 4, 4, 7)`` of the donor block of the tensor targets ``tt`` (indices into ``tr``).

    Singleton / centered_radial: raw-cell arithmetic and ``grid.raw_to_owner``; ringwise: the
    ring entry's owners (design section 6.4).
    """
    tt = np.asarray(tt, dtype=np.int64)
    owners = np.empty((len(tt), 4, 4, 7), dtype=np.int64)
    ring = layout["family"][tt] == _RINGWISE
    angular = ~ring
    if angular.any():
        a = tt[angular]
        raw = _raw_ids(tr.n, tr.layers[layout["source"][a]],
                       tr.theta_index[tr.t_theta[layout["theta_row"][a]]], tr.eta_index[tr.t_eta[a]])
        owners[angular] = grid.raw_to_owner[raw]
    if ring.any():
        owners[ring] = tr.ring_owner[tr.t_ring[layout["ring_row"][tt[ring]]]]
    return owners


def _tensor_eta_range(grid, tr: TensorRows, ts, plane, layout):
    """Min/max wrapped eta offset of the tensor sources ``ts`` (rows of ``tr``), like the CSR sources'.

    ``plane`` are the eta planes of their targets (the concatenated targets of ``ts``). The donors of
    a source are the owners of all 112 entries of all its targets, taken in bounded blocks; those of a pair
    (the A part's row in ``ts``) are the owners of both its parts, so its range is that of the merged row.
    """
    ts = np.asarray(ts, dtype=np.int64)
    counts = np.diff(tr.target_ptr)[ts]
    target_ptr = np.r_[0, np.cumsum(counts)]
    lo = np.zeros(len(ts), dtype=np.int32)
    hi = np.zeros(len(ts), dtype=np.int32)
    for a, b in _blocks(counts * DONOR_BLOCK, _BLOCK_ELEMENTS):
        tt, _ = _ragged(tr.target_ptr[ts[a:b]], counts[a:b])
        owners = _target_owners(grid, tr, tt, layout).reshape(-1)
        _check_donor_ids(owners, grid, "tensor row")
        donor_ptr = np.r_[0, np.cumsum(counts[a:b] * DONOR_BLOCK)]
        lo[a:b], hi[a:b] = _eta_offset_range(grid, owners, donor_ptr, plane[target_ptr[a]:target_ptr[b]],
                                             target_ptr[a:b + 1] - target_ptr[a])
    pair = np.flatnonzero(tr.pair_part[ts] == 1)
    if len(pair):                       # the B parts read the same target planes through their own donors
        at, _ = _ragged(target_ptr[pair], counts[pair])
        lo_b, hi_b = _tensor_eta_range(grid, tr, ts[pair] + 1, plane[at], layout)
        lo[pair], hi[pair] = np.minimum(lo[pair], lo_b), np.maximum(hi[pair], hi_b)
    return lo, hi


def _narrow_index(array, size):
    """Table-row indices as uint16 when the table is small enough, else int32."""
    return np.ascontiguousarray(array).astype(np.uint16 if size <= 1 << 16 else np.int32)


def _theta_combos(n, layers, n_theta, batches):
    """``(combo_key, combo_lookup)`` of the (theta row, half-turn shift, radial layer) combos the targets use.

    ``batches`` are ``(layer_row, theta_row)`` arrays (rows of the global ``layers`` and theta tables) of
    the singleton / centered_radial batches. A target with theta row ``a`` and signed layer ``l``
    reads the ring ``rid = l or -l-1`` at shift ``s = (l < 0)`` (design section 6.4); its combo key is
    ``(a*2 + s)*n + rid``. ``combo_key`` is the sorted unique keys, ``combo_lookup`` (``n_theta*2*n``)
    the row of a key in it (-1: unused).
    """
    lookup = np.full(max(n_theta, 0) * 2 * n, -1, dtype=np.int32)
    if not batches:
        return np.zeros(0, np.int32), lookup
    layers = np.asarray(layers, dtype=np.int64)
    pair = np.unique(np.concatenate([np.asarray(l, dtype=np.int64) * n_theta + np.asarray(a, dtype=np.int64)
                                     for l, a in batches]))
    layer_row, theta_row = pair // n_theta, pair % n_theta
    signed = layers[layer_row]                                                    # (P, 4)
    key = np.unique(((theta_row[:, None] * 2 + (signed < 0)) * n
                     + np.where(signed < 0, -signed - 1, signed)).reshape(-1))
    if len(key) > _INT32_MAX or (len(key) and key[-1] >= len(lookup)):
        raise ValueError("tensor rows: theta/radial combination table out of range")
    lookup[key] = np.arange(len(key), dtype=np.int32)
    return key.astype(np.int32), lookup


class _TensorAccumulator:
    """Merges the per-chunk factor tables of tensor sources into grid-global tables and collects the
    per-target arrays of ``TensorRowBatch`` es (one group per (family, has_gradient)).

    Tables are deduplicated by exact bit pattern, in order of first use, incrementally per chunk (the
    per-chunk ring tables in particular repeat across chunks). Only the rows a chunk's *selected*
    sources use enter the global tables.
    """

    def __init__(self, grid: LoaderGrid):
        self.grid = grid
        self.tables: dict[str, tuple] = {}
        self.pieces: dict[tuple, list[tuple]] = {}
        self.sources: dict[tuple, int] = {}

    def _merge(self, name, *arrays):
        """Global row of every input row of ``arrays`` (parallel tables, rows deduplicated jointly)."""
        current = self.tables.get(name)
        if current is None:
            self.tables[name], index = _dedupe(*arrays)
            return index
        start = len(current[0])
        merged, index = _dedupe(*(np.concatenate([c, a]) for c, a in zip(current, arrays)))
        if len(merged[0]) > _INT32_MAX or not np.array_equal(index[:start], np.arange(start)):
            raise ValueError("tensor table merge is inconsistent")
        self.tables[name] = merged
        return index[start:]

    def _rows(self, name, ids, *tables):
        """Global row of ``tables[k][ids]``: only the used rows are merged."""
        ids = np.asarray(ids)
        used, inverse = np.unique(ids, return_inverse=True)
        return self._merge(name, *(t[used] for t in tables))[inverse.reshape(ids.shape)]

    def add(self, tr: TensorRows, ts, plane, value_slot, gradient_slot):
        """Add the tensor sources ``ts`` (rows of ``tr``, chunk order) of one chunk; a pair is its A part's row.

        ``plane``, ``value_slot``, ``gradient_slot`` (-1: none) are per target of those sources (in
        order). Returns their min/max eta offsets (of the union of both parts for a pair).
        """
        ts = np.asarray(ts, dtype=np.int64)
        layout = tr.target_layout()
        lo, hi = _tensor_eta_range(self.grid, tr, ts, plane, layout)
        counts = np.diff(tr.target_ptr)[ts]
        tt, src = _ragged(tr.target_ptr[ts], counts)
        family_t = tr.family[ts][src]
        ring = family_t == _RINGWISE
        eta = self._rows("eta", tr.t_eta[tt], tr.eta_index, tr.eta_value, tr.eta_derivative)
        radial = self._rows("radial", tr.t_radial[tt], tr.radial)
        layer = self._merge("layers", tr.layers[ts].astype(np.int16))[src]
        theta = np.full(len(tt), -1, dtype=np.int64)
        if (~ring).any():
            theta[~ring] = self._rows("theta", tr.t_theta[layout["theta_row"][tt[~ring]]],
                                      tr.theta_index, tr.theta_value, tr.theta_derivative)
        ring_row = np.cumsum(ring) - 1
        ring_ids = None
        if ring.any():
            ring_ids = self._rows("ring", tr.t_ring[layout["ring_row"][tt[ring]]],
                                  tr.ring_owner, tr.ring_value, tr.ring_derivative)
        has_gradient = tr.has_gradient[ts][src]
        pair = tr.pair_part[ts] == 1
        paired = pair[src]                                  # targets of a pair: part B's targets follow part A's
        mirror_layer = (self._merge("layers", tr.layers[ts[pair] + 1].astype(np.int16))[(np.cumsum(pair) - 1)[src]]
                        if pair.any() else None)
        for family in np.unique(family_t):
            for gradient in (False, True):
                group = np.flatnonzero((family_t == family) & (has_gradient == gradient))
                for is_pair in (False, True):
                    sel = group[paired[group] == is_pair]
                    if not len(sel):
                        continue
                    key = (int(family), gradient, is_pair)
                    ringwise = family == _RINGWISE
                    mirror = (None,) * 5
                    if is_pair:
                        mt = tt[sel] + counts[src[sel]]               # the B targets
                        mirror = (
                            self._rows("eta", tr.t_eta[mt], tr.eta_index, tr.eta_value, tr.eta_derivative),
                            self._rows("radial", tr.t_radial[mt], tr.radial),
                            None if ringwise else mirror_layer[sel],
                            None if ringwise else self._rows("theta", tr.t_theta[layout["theta_row"][mt]], tr.theta_index,
                                                             tr.theta_value, tr.theta_derivative),
                            self._rows("ring", tr.t_ring[layout["ring_row"][mt]], tr.ring_owner, tr.ring_value,
                                       tr.ring_derivative) if ringwise else None)
                    self.pieces.setdefault(key, []).append((
                        value_slot[sel].astype(np.int32),
                        gradient_slot[sel].astype(np.int32) if gradient else None,
                        eta[sel], radial[sel],
                        None if ringwise else layer[sel],
                        None if ringwise else theta[sel],
                        ring_ids[ring_row[sel]] if ringwise else None, *mirror))
                    self.sources[key] = self.sources.get(key, 0) + int(len(np.unique(src[sel])))
        return lo, hi

    def finalize(self):
        """``(TensorTables or None, batches, summary rows)``; index arrays are narrowed to the final table sizes."""
        if not self.pieces:
            return None, (), ()
        t = self.tables
        (theta_index, theta_value, theta_derivative) = t.get("theta", (np.zeros((0, 7), np.int32),
                                                                      np.zeros((0, 7)), np.zeros((0, 7))))
        (eta_index, eta_value, eta_derivative) = t["eta"]
        (radial,) = t["radial"]
        (ring_owner, ring_value, ring_derivative) = t.get("ring", (np.zeros((0, 7), np.int32),
                                                                   np.zeros((0, 7)), np.zeros((0, 7))))
        (layers,) = t["layers"]
        angular = any(family != _RINGWISE for family, _, _ in self.pieces)
        grid = self.grid
        columns = {key: [None if parts[0] is None else np.concatenate(parts) for parts in zip(*self.pieces.pop(key))]
                   for key in sorted(self.pieces, key=lambda k: (_FAMILY_NAMES[k[0]], k[1], k[2]))}
        # columns: value slots, gradient slots, eta, radial, layer, theta, ring, then the same four/five of part B
        used = [(cols[4], cols[5]) for (family, _, _), cols in columns.items() if family != _RINGWISE]
        used += [(cols[9], cols[10]) for (family, _, paired), cols in columns.items() if family != _RINGWISE and paired]
        combo_key, combo_lookup = _theta_combos(grid.n, layers, len(theta_index), used)
        tables = TensorTables(
            grid.n, theta_index.astype(np.int32), theta_value, theta_derivative, eta_index.astype(np.int32),
            eta_value, eta_derivative, radial, ring_owner.astype(np.int32), ring_value, ring_derivative,
            layers.astype(np.int16),
            np.ascontiguousarray(grid.raw_to_owner, dtype=np.int32) if angular else np.zeros(0, np.int32),
            combo_key, combo_lookup)
        sizes = dict(eta=len(eta_index), radial=len(radial), layers=len(layers), theta=len(theta_index),
                     ring=len(ring_owner))
        batches, summary = [], []
        for key, cols in columns.items():
            family, gradient, paired = key
            ring = family == _RINGWISE
            mirror = None
            if paired:
                mirror = TensorMirror(
                    _narrow_index(cols[7], sizes["eta"]), _narrow_index(cols[8], sizes["radial"]),
                    None if ring else _narrow_index(cols[9], sizes["layers"]),
                    None if ring else _narrow_index(cols[10], sizes["theta"]),
                    _narrow_index(cols[11], sizes["ring"]) if ring else None)
            batch = TensorRowBatch(
                _FAMILY_NAMES[family], float(1 / grid.n) if family == 2 else 1.0, cols[0], cols[1],
                _narrow_index(cols[2], sizes["eta"]), _narrow_index(cols[3], sizes["radial"]),
                None if ring else _narrow_index(cols[4], sizes["layers"]),
                None if ring else _narrow_index(cols[5], sizes["theta"]),
                _narrow_index(cols[6], sizes["ring"]) if ring else None, mirror)
            batches.append(batch)
            targets = len(batch.value_slots)
            parts = 2 if paired else 1                        # factorizations contracted per target
            summary.append(BucketSummary(_FAMILY_NAMES[family], False, gradient, 0, parts * DONOR_BLOCK,
                                         self.sources[key], targets, parts * DONOR_BLOCK * targets,
                                         tensor_nbytes((), batch), "tensor_paired" if paired else "tensor"))
        return tables, tuple(batches), tuple(summary)


def _concat_batches(batches: list[SourceRowBatch]) -> SourceRowBatch:
    if len(batches) == 1:
        return batches.pop()
    fields = []
    for values in zip(*batches):
        fields.append(None if values[0] is None else np.concatenate(values))
    batches.clear()
    return SourceRowBatch(*fields)


def _finish_source_plan(buckets, bucket_counts, pieces, per_target, eta_min, eta_max,
                        q_per_source, families, n_targets, n_gradient, tensor) -> SourceRowPlan:
    points, remap = pieces.finalize()
    names = tuple(families)
    order = sorted(buckets, key=lambda k: (names[k[0]], k[1], k[2], k[3], k[4]))
    batches, summary = [], []
    for bkey in order:
        fam, cond, hasg, nodes, w = bkey
        batch = _concat_batches(buckets.pop(bkey))
        if cond:
            batch = batch._replace(donor_query=remap[batch.donor_query],
                                   target_query=remap[batch.target_query])
        batches.append(batch)
        sources = len(batch.donor_ids)
        summary.append(BucketSummary(names[fam], cond, hasg, nodes, w, sources, sources * nodes,
                                     bucket_counts[bkey][0], _tuple_nbytes(batch)))
    if per_target:
        targets = SourceRowTargets(*(np.concatenate(cols) for cols in zip(*per_target)))
        eta_lo, eta_hi = np.concatenate(eta_min), np.concatenate(eta_max)
        q = np.concatenate(q_per_source)
    else:
        targets = SourceRowTargets(
            np.zeros(0, np.int8), np.zeros(0, np.int64), np.zeros(0, np.int16), np.zeros(0, np.int8),
            np.zeros(0, np.int16), np.zeros(0, np.int16), np.zeros(0, bool), np.zeros(0, np.int32),
            np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 3)))
        eta_lo = eta_hi = np.zeros(0, np.int32)
        q = np.zeros(0, np.int64)
    tables, tensor_batches, tensor_summary = tensor.finalize()
    payload = SourceRowPayload(tuple(batches), int(n_targets), int(n_gradient), len(points),
                               tensor_batches, tables)
    table_bytes = 0 if tables is None else tensor_nbytes(tables, ())
    return SourceRowPlan(payload, points, targets, np.r_[0, np.cumsum(q)].astype(np.int64),
                         eta_lo, eta_hi, names, tuple(summary) + tensor_summary,
                         tensor_table_bytes=table_bytes)


def old_point_layout_bytes(chunks: Iterable[PointRowChunk], *, select: RowSelection | None = None) -> int:
    """Analytic bytes of the reference ``PointRowPayload`` layout for the same rows.

    Per target the reference ``PointRowBatch`` stores (bucket width ``w``): target
    id (int64), donor ids (int64 x w), value (float64 x w), gradient
    (float64 x 3w, zero filled for value-only rows), donor query ids (int64 x w),
    target query id (int64) and a conditioned flag (1 byte); plus the
    ``output_template`` (8 bytes per target). The ``boundary_query_template``
    (8 bytes per query) is not counted. Total per target: ``25 + 48 w`` bytes.
    """
    total = 0
    for item in chunks:
        chunk, _, is_tensor = _unpack_point_item(item)
        if is_tensor is not None:
            raise ValueError("old_point_layout_bytes needs expanded (CSR) chunks: tensor sources store no donors")
        info = _check_point_chunk(chunk)
        if not len(info.t0):
            continue
        keep = _selected(select, np.asarray(chunk.request)[info.t0],
                         np.asarray(chunk.bc_variant)[info.t0],
                         np.asarray(chunk.entity_id)[info.t0], len(info.t0))
        w = _widths16(info.donor_count[keep])
        total += int((info.q[keep] * (25 + 48 * w)).sum())
    return total


# --------------------------------------------------------------------------
# Integrated (P07) rows -> existing IntegratedFacePayload
# --------------------------------------------------------------------------

class IntegratedBucketSummary(NamedTuple):
    family: int
    conditioned: bool
    width: int
    faces: int
    donor_entries: int
    payload_bytes: int


@dataclass(frozen=True)
class IntegratedRowPlan:
    payload: IntegratedFacePayload
    boundary_points: np.ndarray             # (Q, 3) float64
    face_entity_id: np.ndarray              # (F,) int64, P07 face id of output face f
    eta_offset_min: np.ndarray              # (F,) int32
    eta_offset_max: np.ndarray              # (F,) int32
    bucket_summary: tuple[IntegratedBucketSummary, ...]


def _check_integrated_chunk(chunk: IntegratedRowChunk):
    n = len(chunk.entity_id)
    dptr = np.asarray(chunk.donor_ptr, dtype=np.int64)
    if len(dptr) != n + 1 or dptr[0] != 0 or np.any(np.diff(dptr) < 0):
        raise ValueError("integrated chunk: donor_ptr does not match the faces")
    nnz = int(dptr[-1])
    if not (nnz == len(chunk.donor) == len(chunk.weight) == len(chunk.donor_query)
            == len(chunk.value_loading)):
        raise ValueError("integrated chunk: donor arrays disagree with donor_ptr")
    if (chunk.target_points.shape != (n, 9, 3) or chunk.tangential_loading.shape != (n, 9, 2)
            or len(chunk.conditioned) != n or len(chunk.family) != n):
        raise ValueError("integrated chunk: per-face array shapes are inconsistent")
    return dptr


def lower_integrated_chunks(chunks: Iterable[IntegratedRowChunk], *, grid: LoaderGrid,
                            endpoints, owner_volume,
                            select: RowSelection | None = None) -> IntegratedRowPlan:
    """Lower P07 integrated-face chunks into the existing ``IntegratedFacePayload``.

    Buckets are ``(family, conditioned, width)`` as in ``lower_integrated_face_rows``
    (ids are int32 here; unconditioned buckets keep zero-filled query arrays because
    ``apply_integrated_face_rows`` indexes them unconditionally). Output face ``f``
    is the ``f``-th selected face in chunk order; ``endpoints`` is ``(faces, 2)``
    lower/upper owner ids (-1 for none) indexed by the chunk ``entity_id`` (the
    P07 face id). ``select.requests`` must contain ``"R4"`` if given; BC variants
    do not apply. Also returns per-face eta offset min/max (0 for a face without
    donors), evaluated at q3 node 4 as the reference does.
    """
    endpoints = np.asarray(endpoints, dtype=np.int64)
    owner_volume = np.asarray(owner_volume, dtype=np.float64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("endpoints must have shape (faces, 2)")
    if np.any((endpoints < -1) | (endpoints >= grid.n_owners)):
        raise ValueError("invalid face owner endpoint")
    if owner_volume.shape != (grid.n_owners,):
        raise ValueError("owner_volume must have one entry per owner")
    buckets: dict[tuple, list[tuple]] = {}
    donor_entries: dict[tuple, int] = {}
    pieces = _QueryPieces()
    entities, lows, highs = [], [], []
    faces = 0
    requests_ok = select is None or select.requests is None or "R4" in select.requests
    for chunk in chunks:
        dptr = _check_integrated_chunk(chunk)
        if not requests_ok or not len(chunk.entity_id):
            continue
        entity = np.asarray(chunk.entity_id, dtype=np.int64)
        keep = _selected(select, None, None, entity, len(entity))
        idx = np.flatnonzero(keep)
        m = len(idx)
        if not m:
            continue
        if entity[idx].min() < 0 or entity[idx].max() >= len(endpoints):
            raise ValueError("integrated face id outside the endpoints table")
        widths = np.diff(dptr)[idx]
        starts = dptr[idx]
        family = np.asarray(chunk.family)[idx].astype(np.int64)
        conditioned = np.asarray(chunk.conditioned, dtype=bool)[idx]
        flat, owner = _ragged(starts, widths)
        donors = np.asarray(chunk.donor)[flat]
        _check_donor_ids(donors, grid, "integrated row")
        entry_conditioned = conditioned[owner]
        if np.any(np.asarray(chunk.value_loading)[flat][entry_conditioned]
                  != -np.asarray(chunk.weight)[flat][entry_conditioned]):
            raise ValueError("unsupported integrated boundary value loading")
        plane = _nearest_plane(grid.eta_centers, chunk.target_points[idx, 4, 2])
        lo, hi = _eta_offset_range(grid, donors, np.r_[0, np.cumsum(widths)], plane,
                                   np.arange(m + 1))
        entities.append(entity[idx]); lows.append(lo); highs.append(hi)
        width = _widths16(widths)
        if len(np.unique(family)) and (family.min() < 0 or family.max() >= 2 ** 15):
            raise ValueError("integrated family code out of range")
        if width.max() >= 65536:
            raise ValueError("integrated donor width out of range")
        key = (family * 2 + conditioned) * 65536 + width
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        cuts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
        stops = np.r_[cuts[1:], m]
        for a, b in zip(cuts, stops):
            g = order[a:b]
            fam, cond, w = int(family[g[0]]), bool(conditioned[g[0]]), int(width[g[0]])
            face_ids = (faces + g).astype(np.int32)
            donor_ids = np.zeros((len(g), w), dtype=np.int32)
            weights = np.zeros((len(g), w))
            boundary_ids = np.zeros((len(g), w), dtype=np.int32)
            tangent_ids = np.zeros((len(g), 9), dtype=np.int32)
            tangent_weights = np.zeros((len(g), 9, 2))
            step = max(1, _BLOCK_ELEMENTS // max(w, 1))
            for i in range(0, len(g), step):
                s = g[i:i + step]
                donor_ids[i:i + step] = _pad_gather(chunk.donor, starts[s], widths[s], w, np.int32)
                weights[i:i + step] = _pad_gather(chunk.weight, starts[s], widths[s], w, np.float64)
                if cond:
                    local = _pad_gather(chunk.donor_query, starts[s], widths[s], w)
                    valid = np.arange(w) < widths[s][:, None]
                    used = local[valid]
                    if len(used) and (used.min() < 0 or used.max() >= len(chunk.query_table)):
                        raise ValueError("conditioned integrated row has an invalid donor query id")
                    target_pts = np.asarray(chunk.target_points)[idx[s]].reshape(-1, 3)
                    ids = pieces.add(np.concatenate([chunk.query_table[used], target_pts]))
                    block = np.zeros(local.shape, dtype=np.int32)
                    block[valid] = ids[:len(used)]
                    boundary_ids[i:i + step] = block
                    tangent_ids[i:i + step] = ids[len(used):].reshape(len(s), 9)
                    tangent_weights[i:i + step] = np.asarray(chunk.tangential_loading)[idx[s]]
            bkey = (fam, cond, w)
            buckets.setdefault(bkey, []).append(
                (face_ids, donor_ids, weights, boundary_ids, tangent_ids, tangent_weights))
            donor_entries[bkey] = donor_entries.get(bkey, 0) + int(widths[g].sum())
        faces += m
    points, remap = pieces.finalize()
    batches, summary = [], []
    for bkey in sorted(buckets):
        fam, cond, w = bkey
        parts_ = buckets.pop(bkey)
        cols = list(parts_[0]) if len(parts_) == 1 else [np.concatenate(c) for c in zip(*parts_)]
        if cond:
            cols[3] = remap[cols[3]]
            cols[4] = remap[cols[4]]
        batch = IntegratedFaceBatch(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5],
                                    np.full(len(cols[0]), cond, dtype=bool))
        batches.append(batch)
        summary.append(IntegratedBucketSummary(fam, cond, w, len(cols[0]), donor_entries[bkey],
                                               _tuple_nbytes(batch)))
    entity = np.concatenate(entities) if entities else np.zeros(0, np.int64)
    payload = IntegratedFacePayload(tuple(batches), endpoints[entity, 0].copy(),
                                    endpoints[entity, 1].copy(), owner_volume,
                                    len(points), int(faces))
    return IntegratedRowPlan(payload, points, entity,
                             np.concatenate(lows) if lows else np.zeros(0, np.int32),
                             np.concatenate(highs) if highs else np.zeros(0, np.int32),
                             tuple(summary))


# --------------------------------------------------------------------------
# Neumann wall-lattice rows -> existing NeumannPayload
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NeumannRowPlan:
    payload: NeumannPayload
    wall_points: np.ndarray     # (Q, 3) float64, deduplicated by bit pattern
    entity_id: np.ndarray       # (R,) int64
    quad_node: np.ndarray       # (R,) int16
    request: np.ndarray         # (R,) int8, originating request, index into REQUEST_KINDS
    radial_degree: np.ndarray   # (R,) int8


def lower_neumann_chunks(chunks: Iterable[NeumannRowChunk], *,
                         select: RowSelection | None = None) -> NeumannRowPlan:
    """Lower Neumann chunks into the existing ``NeumannPayload`` (one padded bucket).

    Rows keep chunk order (selected rows only). Donors pad to the largest
    donor count with id 0 and weight 0, as ``lower_neumann_point_rows`` does.
    ``select.requests`` filters the *originating* request tag; BC variants do not
    apply. The 28 wall-lattice points per row are deduplicated globally by exact
    float64 bit pattern.

    The chunk boundary weights (``boundary_value``/``boundary_gradient``) have the
    builder's dtype, longdouble (80-bit) on x86. They are cast to float64 here,
    explicitly: that is the precision ``apply_neumann_point_rows`` already uses
    implicitly, and it keeps the runtime payload float64 on every platform.
    Chunk pieces are padded to one global width at the end, one field at a time.
    """
    pieces = _QueryPieces()
    parts: list[dict] = []
    request_table = list(REQUEST_KINDS)
    for chunk in chunks:
        n = len(chunk.entity_id)
        dptr = np.asarray(chunk.donor_ptr, dtype=np.int64)
        if len(dptr) != n + 1 or dptr[0] != 0 or np.any(np.diff(dptr) < 0) \
                or not (dptr[-1] == len(chunk.donor) == len(chunk.value)) \
                or chunk.gradient.shape != (3, int(dptr[-1])):
            raise ValueError("Neumann chunk: donor arrays disagree with donor_ptr")
        if (chunk.wall_query.shape != (n, 28) or chunk.boundary_value.shape != (n, 28)
                or chunk.boundary_gradient.shape != (n, 3, 28)):
            raise ValueError("Neumann chunk: wall-lattice arrays must cover 28 points per row")
        idx = np.flatnonzero(_selected(select, np.asarray(chunk.request), None,
                                       np.asarray(chunk.entity_id), n))
        if not len(idx):
            continue
        widths, starts = np.diff(dptr)[idx], dptr[idx]
        flat, _ = _ragged(starts, widths)
        _check_neumann_donors(np.asarray(chunk.donor)[flat])
        wall = np.asarray(chunk.wall_query)[idx]
        if wall.min() < 0 or wall.max() >= len(chunk.query_table):
            raise ValueError("Neumann chunk: invalid wall query id")
        ids = pieces.add(np.asarray(chunk.query_table)[wall.reshape(-1)]).reshape(len(idx), 28)
        parts.append({
            "widths": widths, "starts": starts, "chunk": chunk, "idx": idx, "wall": ids.astype(np.int32),
            "entity": np.asarray(chunk.entity_id)[idx].astype(np.int64),
            "node": np.asarray(chunk.quad_node)[idx].astype(np.int16),
            "request": _string_codes(np.asarray(chunk.request)[idx], request_table).astype(np.int8),
            "degree": np.asarray(chunk.radial_degree)[idx].astype(np.int8),
        })
        # Pad within the chunk now so the chunk itself is not retained.
        parts[-1].update(_neumann_piece(parts[-1]))
        del parts[-1]["chunk"]
    points, remap = pieces.finalize()
    rows = sum(len(p["idx"]) for p in parts)
    width = max((int(p["widths"].max()) for p in parts), default=0)
    payload_fields = {}
    for name, shape, dtype in (("donor", (rows, width), np.int32), ("value", (rows, width), np.float64),
                               ("gradient", (rows, 3, width), np.float64),
                               ("wall", (rows, 28), np.int32),
                               ("bvalue", (rows, 28), np.float64),
                               ("bgradient", (rows, 3, 28), np.float64)):
        out = np.zeros(shape, dtype=dtype)
        r0 = 0
        for p in parts:
            block = p[name]
            if name == "wall":
                block = remap[block]
            r1 = r0 + len(block)
            out[r0:r1, ..., :block.shape[-1]] = block
            r0 = r1
            if name not in ("wall",):
                p[name] = None      # release the piece as soon as it is copied
        payload_fields[name] = out
    payload = NeumannPayload(payload_fields["donor"], payload_fields["value"],
                             payload_fields["gradient"], payload_fields["wall"],
                             payload_fields["bvalue"], payload_fields["bgradient"], len(points))
    cat = lambda key, dtype: (np.concatenate([p[key] for p in parts]) if parts
                              else np.zeros(0, dtype))
    return NeumannRowPlan(payload, points, cat("entity", np.int64), cat("node", np.int16),
                          cat("request", np.int8), cat("degree", np.int8))


def _check_neumann_donors(donors):
    if len(donors) and donors.min() < 0:
        raise ValueError("Neumann row has invalid compact donor ids")


def _neumann_piece(part) -> dict:
    """Chunk-width padded arrays of the selected rows (boundary weights cast to float64)."""
    chunk, idx, widths, starts = part["chunk"], part["idx"], part["widths"], part["starts"]
    w = int(widths.max()) if len(widths) else 0
    return {
        "donor": _pad_gather(chunk.donor, starts, widths, w, np.int32),
        "value": _pad_gather(chunk.value, starts, widths, w, np.float64),
        "gradient": _pad_gather_gradient(np.asarray(chunk.gradient), starts, widths, w),
        "bvalue": np.asarray(chunk.boundary_value)[idx].astype(np.float64),
        "bgradient": np.asarray(chunk.boundary_gradient)[idx].astype(np.float64),
    }
