"""Exact tensor-factored storage of unconditioned point rows (P08 step 2, task D1).

The unconditioned ``singleton``, ``ringwise`` and ``centered_radial`` sources of the
row artifact are, target by target, sums of products of shared 1-D factors, and
``StructuredReconstruction.rows_with_factors`` records the factors it actually used
(``drbx.geometry.fci_perpendicular_reconstruction.PointFactors``). :class:`TensorRows`
holds them with bit-pattern-deduplicated tables and a few small integers per target;
:func:`expand_tensor_rows` reproduces the dense per-source rows **bit for bit**
(same donor union, same association and accumulation order, ``0.0 + x`` for signed
zeros), and :func:`apply_tensor_rows_numpy` contracts the factors directly. The
format is specified in ``work/p08_step2_layout_loader_design_20260929/design.md``
section 6; the on-disk members (``tr_*``) are written by ``drbx.stencils.artifact``.

Nothing is fitted back from weights, and nothing is recomputed at load: every table
entry is the float the construction produced.

A ``cell_stencil="symmetric"`` cell row is not one factorization but ``1/2 (A + B)`` of two (A the biased row, B
its mirror; ``PairedFactors``). It is stored as two consecutive sources of the same :class:`TensorRows`, sharing its
deduplicated tables; :func:`average_expanded_rows` merges their expansions exactly as
``StructuredReconstruction._average_rows`` does (union of the donors, A then B, ``0.5 *``) and
:func:`merge_paired_expansion` applies it to a whole expansion. :func:`verified_tensor_rows` accepts a paired
candidate only if that merge reproduces its dense row bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from drbx.geometry.fci_perpendicular_reconstruction import PairedFactors, PointFactors

__all__ = ["FAMILY_CODES", "TensorRows", "TensorExpansion", "build_tensor_rows", "expand_tensor_rows",
           "apply_tensor_rows_numpy", "verified_tensor_rows", "average_expanded_rows", "merge_paired_expansion"]

#: family code of ``TensorRows.family`` (per source).
FAMILY_CODES = {"singleton": 0, "ringwise": 1, "centered_radial": 2}
_SINGLETON, _RINGWISE, _CENTERED = 0, 1, 2
_K = 112                                        # donor entries per target: 4 layers x 4 planes x 7 slots

_TABLE_ARRAYS = ("theta_index", "theta_value", "theta_derivative", "eta_index", "eta_value", "eta_derivative",
                 "radial", "ring_owner", "ring_value", "ring_derivative", "raw_ids", "raw_owner")
_SOURCE_ARRAYS = ("family", "layers")
_TARGET_ARRAYS = ("t_eta", "t_radial", "t_theta", "t_ring")
_INDEX_TABLE = {"t_eta": "eta_index", "t_radial": "radial", "t_theta": "theta_index", "t_ring": "ring_owner"}


@dataclass(frozen=True)
class TensorRows:
    """The factors of ``Ts`` sources (``Tt`` targets in all) of one chunk; see design section 6."""

    n: int
    theta_index: np.ndarray       # (NT, 7) int32   theta plane ids of a target's 7 slots
    theta_value: np.ndarray       # (NT, 7) float64 tv
    theta_derivative: np.ndarray  # (NT, 7) float64 td
    eta_index: np.ndarray         # (NE, 4) int32   eta plane ids
    eta_value: np.ndarray         # (NE, 4) float64 ev
    eta_derivative: np.ndarray    # (NE, 4) float64 ed
    radial: np.ndarray            # (NR, 2, 4) float64  [L, D] over the source's 4 layers (D undivided by h)
    ring_owner: np.ndarray        # (NG, 7) int32   owner ids of a ringwise ring entry
    ring_value: np.ndarray        # (NG, 7) float64 vv
    ring_derivative: np.ndarray   # (NG, 7) float64 dd
    raw_ids: np.ndarray           # (NW,) int32 sorted raw cell ids  } the raw_to_owner entries the
    raw_owner: np.ndarray         # (NW,) int32 their owner ids      } singleton/centered donors use
    family: np.ndarray            # (Ts,) int8   FAMILY_CODES
    layers: np.ndarray            # (Ts, 4) int16 signed radial layers (negative: wrapped across the axis)
    has_gradient: np.ndarray      # (Ts,) bool
    target_ptr: np.ndarray        # (Ts+1,) int64 targets of source s: [target_ptr[s], target_ptr[s+1])
    t_eta: np.ndarray             # (Tt,) int32   row of the eta table
    t_radial: np.ndarray          # (Tt,) int32   row of the radial table
    t_theta: np.ndarray           # (Ta,) int32   row of the theta table, for the Ta non-ringwise targets in order
    t_ring: np.ndarray            # (Tr, 4, 4) int32 ring-entry row per (layer, plane), Tr ringwise targets in order

    @property
    def n_sources(self) -> int:
        return len(self.family)

    @property
    def n_targets(self) -> int:
        return int(self.target_ptr[-1])

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in _TABLE_ARRAYS + _SOURCE_ARRAYS + _TARGET_ARRAYS)

    def target_layout(self) -> dict:
        """Per-target helper arrays: source, position within its source, family, and the theta / ring
        table position of each non-ringwise / ringwise target (-1 elsewhere)."""
        counts = np.diff(self.target_ptr)
        source = np.repeat(np.arange(len(counts)), counts)
        local = np.arange(len(source)) - self.target_ptr[:-1][source]
        family = self.family[source]
        ring = family == _RINGWISE
        theta_row = np.full(len(source), -1, dtype=np.int64)
        theta_row[~ring] = np.arange(int((~ring).sum()))
        ring_row = np.full(len(source), -1, dtype=np.int64)
        ring_row[ring] = np.arange(int(ring.sum()))
        return dict(source=source, local=local, family=family, theta_row=theta_row, ring_row=ring_row)

    def take_sources(self, sources) -> "TensorRows":
        """The rows of the sources ``sources`` (indices, in the order given): their per-source and per-target
        arrays. Every factor table is kept whole (a row no remaining target uses is harmless)."""
        sources = np.asarray(sources, dtype=np.int64)
        counts = np.diff(self.target_ptr)[sources]
        layout = self.target_layout()
        target = _flat_index(self.target_ptr[sources], counts)
        ring = layout["family"][target] == _RINGWISE
        rows = replace(
            self, family=self.family[sources], layers=self.layers[sources], has_gradient=self.has_gradient[sources],
            target_ptr=_ptr(counts), t_eta=self.t_eta[target], t_radial=self.t_radial[target],
            t_theta=self.t_theta[layout["theta_row"][target[~ring]]], t_ring=self.t_ring[layout["ring_row"][target[ring]]])
        rows.validate()
        return rows

    def validate(self) -> None:
        ts, tt = self.n_sources, self.n_targets
        layout = self.target_layout()
        angular = int((layout["family"] != _RINGWISE).sum())
        expect = {"layers": (ts, 4), "has_gradient": (ts,), "t_eta": (tt,), "t_radial": (tt,), "t_theta": (angular,),
                  "t_ring": (tt - angular, 4, 4)}
        for name, shape in expect.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"corrupted tensor rows: {name} has shape {getattr(self, name).shape}, expected {shape}")
        if len(self.target_ptr) != ts + 1 or self.target_ptr[0] != 0 or np.any(np.diff(self.target_ptr) <= 0):
            raise ValueError("corrupted tensor rows: target_ptr")
        for name, table in _INDEX_TABLE.items():
            ids, size = getattr(self, name), len(getattr(self, table))
            if ids.size and (ids.min() < 0 or ids.max() >= size):
                raise ValueError(f"corrupted tensor rows: {name} outside its table")
        if np.any((self.family < 0) | (self.family > _CENTERED)):
            raise ValueError("corrupted tensor rows: unknown family code")

    # -- npz members (prefix ``tr_``), written/read by drbx.stencils.artifact -----------------

    def to_arrays(self) -> dict:
        arrays = {"tr_n": np.asarray(self.n, dtype=np.int64)}
        small = {"theta_index": np.int16, "eta_index": np.int16, "ring_owner": np.int32, "raw_ids": np.int32,
                 "raw_owner": np.int32, "family": np.int8, "layers": np.int16}
        for name in _TABLE_ARRAYS + _SOURCE_ARRAYS:
            array = getattr(self, name)
            if name in small:
                narrow = array.astype(small[name])
                if not np.array_equal(narrow, array):
                    raise ValueError(f"tensor rows: {name} does not fit {np.dtype(small[name])}")
                array = narrow
            arrays["tr_" + name] = array
        for name, table in _INDEX_TABLE.items():
            arrays["tr_" + name] = _narrow_index(getattr(self, name), len(getattr(self, table)))
        return arrays

    @classmethod
    def from_arrays(cls, arrays: dict, *, has_gradient, target_counts) -> "TensorRows":
        """Rebuild from the ``tr_*`` members; ``has_gradient`` and ``target_counts`` are those of the tensor
        sources (they live in the chunk's ``src_has_gradient`` / ``source_ptr``)."""
        wide = {"theta_index": np.int32, "eta_index": np.int32, "ring_owner": np.int32, "raw_ids": np.int32,
                "raw_owner": np.int32, "family": np.int8, "layers": np.int16}
        kwargs = {}
        for name in _TABLE_ARRAYS + _SOURCE_ARRAYS:
            array = np.asarray(arrays["tr_" + name])
            kwargs[name] = array.astype(wide[name]) if name in wide else array
        for name in _TARGET_ARRAYS:
            kwargs[name] = np.asarray(arrays["tr_" + name]).astype(np.int32)
        counts = np.asarray(target_counts, dtype=np.int64)
        rows = cls(n=int(np.asarray(arrays["tr_n"])), has_gradient=np.asarray(has_gradient, dtype=bool),
                   target_ptr=_ptr(counts), **kwargs)
        rows.validate()
        return rows


@dataclass(frozen=True)
class TensorExpansion:
    """Dense rows of the tensor sources, in the chunk's own flat layout (design section 6).

    Source ``s`` has ``donor[donor_ptr[s]:donor_ptr[s+1]]`` (its ``d_s`` donors, ascending), a
    value block ``value[value_ptr[s]:value_ptr[s+1]]`` of ``q_s`` rows of ``d_s`` (row-major), and, if
    ``has_gradient``, a block ``gradient[:, gradient_ptr[s]:gradient_ptr[s+1]]`` of ``q_s`` rows of ``d_s`` per
    component (empty otherwise).
    """

    donor_ptr: np.ndarray
    donor: np.ndarray
    value_ptr: np.ndarray
    value: np.ndarray
    gradient_ptr: np.ndarray
    gradient: np.ndarray

    def source_rows(self, s: int, target_count: int, has_gradient: bool):
        """``(donor (d,), value (q, d), gradient (q, 3, d) or None)`` of source ``s``."""
        d0, d1 = int(self.donor_ptr[s]), int(self.donor_ptr[s + 1])
        d, q = d1 - d0, target_count
        value = self.value[self.value_ptr[s]:self.value_ptr[s + 1]].reshape(q, d)
        gradient = None
        if has_gradient:
            gradient = self.gradient[:, self.gradient_ptr[s]:self.gradient_ptr[s + 1]].reshape(3, q, d).transpose(1, 0, 2)
        return self.donor[d0:d1], value, gradient


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _ptr(counts) -> np.ndarray:
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=ptr[1:])
    return ptr


def _flat_index(base, counts) -> np.ndarray:
    """``[base[0] + 0..counts[0]), base[1] + 0..counts[1]), ...]`` (int64)."""
    counts = np.asarray(counts, dtype=np.int64)
    ptr = _ptr(counts)
    return np.repeat(np.asarray(base, dtype=np.int64) - ptr[:-1], counts) + np.arange(ptr[-1], dtype=np.int64)


def _narrow_index(array: np.ndarray, size: int) -> np.ndarray:
    return array.astype(np.uint16 if size < 2 ** 16 else np.int32)


def _bits(array: np.ndarray) -> np.ndarray:
    """(M, W) uint64 exact bit patterns of a float64 or integer array."""
    array = np.asarray(array)
    array = array.astype(np.float64 if array.dtype.kind == "f" else np.int64).reshape(len(array), -1)
    return np.ascontiguousarray(array).view(np.uint64)


def _dedupe(*arrays):
    """Deduplicate the rows of parallel ``arrays`` by exact bit pattern, in order of first use.

    Returns ``(tables, index)``: the tables (first occurrences) and, per input row, its table row.
    """
    m = len(arrays[0])
    if m == 0:
        return tuple(a[:0] for a in arrays), np.zeros(0, dtype=np.int64)
    bits = np.concatenate([_bits(a) for a in arrays], axis=1)
    void = np.ascontiguousarray(bits).view(np.dtype((np.void, bits.shape[1] * 8))).ravel()
    _, first, inverse = np.unique(void, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return tuple(a[first[order]] for a in arrays), rank[inverse.ravel()]


def _raw_ids(n: int, layers, theta_index, eta_index) -> np.ndarray:
    """Raw cell ids (m, 4, 4, 7) of the (layer, plane, slot) donors of ``m`` singleton/centered targets.

    A negative layer wraps across the axis: radial index ``-l-1`` and theta shifted by ``n//2``.
    """
    layers = np.asarray(layers, dtype=np.int64)
    rid = np.where(layers < 0, -layers - 1, layers)
    theta = (np.asarray(theta_index, dtype=np.int64)[:, None, :] + np.where(layers < 0, n // 2, 0)[:, :, None]) % n
    return (rid[:, :, None, None] * n + theta[:, :, None, :]) * n + np.asarray(eta_index, dtype=np.int64)[:, None, :, None]


def _lookup_owner(raw_ids, raw_owner, raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.int64)
    pos = np.searchsorted(raw_ids, raw)
    if raw_ids.size == 0 or np.any(raw_ids[np.minimum(pos, len(raw_ids) - 1)] != raw):
        raise ValueError("tensor rows: a donor raw cell is missing from the stored raw_to_owner entries")
    return raw_owner[pos].astype(np.int64)


# --------------------------------------------------------------------------------------
# construction from captured factors
# --------------------------------------------------------------------------------------

def build_tensor_rows(factors: Sequence[PointFactors], has_gradient: Sequence[bool]) -> TensorRows:
    """Tables and indices for the sources whose captured ``factors`` are given (in order)."""
    factors = list(factors)
    if not factors:
        raise ValueError("no factors")
    n = int(factors[0].n)
    counts = np.array([len(f.radial) for f in factors], dtype=np.int64)
    family = np.array([FAMILY_CODES[f.family] for f in factors], dtype=np.int8)
    if any(int(f.n) != n for f in factors):
        raise ValueError("tensor rows: sources of different grids")
    layers = np.array([np.asarray(f.layers) for f in factors], dtype=np.int64)
    fam_t = np.repeat(family, counts)
    ring_t = fam_t == _RINGWISE

    def cat(name, of=None):
        parts = [getattr(f, name) for f in factors if of is None or FAMILY_CODES[f.family] in of]
        return np.concatenate(parts) if parts else None

    (eta_index, eta_value, eta_derivative), t_eta = _dedupe(cat("eta_index"), cat("eta_value"), cat("eta_derivative"))
    (radial,), t_radial = _dedupe(cat("radial"))
    angular = cat("theta_index", (_SINGLETON, _CENTERED))
    if angular is not None:
        (theta_index, theta_value, theta_derivative), t_theta = _dedupe(
            angular, cat("theta_value", (_SINGLETON, _CENTERED)), cat("theta_derivative", (_SINGLETON, _CENTERED)))
    else:
        theta_index, theta_value, theta_derivative = np.zeros((0, 7), np.int64), np.zeros((0, 7)), np.zeros((0, 7))
        t_theta = np.zeros(0, dtype=np.int64)
    ring = cat("ring_owner", (_RINGWISE,))
    if ring is not None:
        flat = lambda a: a.reshape(-1, 7)
        (ring_owner, ring_value, ring_derivative), t_ring = _dedupe(
            flat(ring), flat(cat("ring_value", (_RINGWISE,))), flat(cat("ring_derivative", (_RINGWISE,))))
        t_ring = t_ring.reshape(-1, 4, 4)
    else:
        ring_owner, ring_value, ring_derivative = np.zeros((0, 7), np.int64), np.zeros((0, 7)), np.zeros((0, 7))
        t_ring = np.zeros((0, 4, 4), dtype=np.int64)

    # the raw_to_owner entries the singleton/centered donors use
    ang_src = np.flatnonzero(family != _RINGWISE)
    if len(ang_src):
        owners = np.concatenate([factors[s].owner.reshape(-1) for s in ang_src])
        raw = _raw_ids(n, np.repeat(layers[ang_src], counts[ang_src], axis=0), angular,
                       np.concatenate([factors[s].eta_index for s in ang_src])).reshape(-1)
        table = np.full(n ** 3, -1, dtype=np.int64)          # raw cell -> owner, dense (raw ids are < n**3)
        table[raw] = owners
        if not np.array_equal(table[raw], owners):
            raise ValueError("tensor rows: one raw cell maps to two owners")
        raw_ids = np.flatnonzero(table >= 0)
        raw_owner = table[raw_ids]
    else:
        raw_ids = raw_owner = np.zeros(0, dtype=np.int64)
    if ring is not None and ring_owner.size and ring_owner.max() >= 2 ** 31:
        raise ValueError("tensor rows: owner id does not fit int32")

    i32 = lambda a: np.asarray(a).astype(np.int32)
    rows = TensorRows(
        n=n, theta_index=i32(theta_index), theta_value=np.asarray(theta_value, dtype=np.float64),
        theta_derivative=np.asarray(theta_derivative, dtype=np.float64), eta_index=i32(eta_index),
        eta_value=np.asarray(eta_value, dtype=np.float64), eta_derivative=np.asarray(eta_derivative, dtype=np.float64),
        radial=np.asarray(radial, dtype=np.float64), ring_owner=i32(ring_owner),
        ring_value=np.asarray(ring_value, dtype=np.float64), ring_derivative=np.asarray(ring_derivative, dtype=np.float64),
        raw_ids=i32(raw_ids), raw_owner=i32(raw_owner), family=family, layers=layers.astype(np.int16),
        has_gradient=np.asarray(has_gradient, dtype=bool), target_ptr=_ptr(counts),
        t_eta=i32(t_eta), t_radial=i32(t_radial), t_theta=i32(t_theta), t_ring=i32(t_ring))
    rows.validate()
    return rows


# --------------------------------------------------------------------------------------
# expansion to dense rows (bitwise)
# --------------------------------------------------------------------------------------

def _accumulate(out: np.ndarray, flat: np.ndarray, block: np.ndarray) -> None:
    """``np.add.at(out, flat[t], block[t])`` per target row t, in (layer, plane, slot) order, into zeros:
    rows without repeated positions are assigned as ``0.0 + x`` (normalizes ``-0.0`` exactly like the
    first ``add.at`` addition into zeros); rows with an aggregate owner hit several times accumulate."""
    ordered = np.sort(flat, axis=1)
    dup = (ordered[:, 1:] == ordered[:, :-1]).any(axis=1)
    single = ~dup
    out[flat[single]] = 0.0 + block[single]
    if dup.any():
        np.add.at(out, flat[dup].ravel(), block[dup].ravel())


def _expand_products(tr: TensorRows, layout: dict):
    """Singleton and ringwise sources, vectorized over targets: donors, values and gradients."""
    src_ids = np.flatnonzero(tr.family != _CENTERED)
    if not len(src_ids):
        return src_ids, None
    counts = np.diff(tr.target_ptr)
    sel = np.flatnonzero(tr.family[layout["source"]] != _CENTERED)          # global targets
    local_of_src = np.full(len(tr.family), -1, dtype=np.int64)
    local_of_src[src_ids] = np.arange(len(src_ids))
    src_of = local_of_src[layout["source"][sel]]
    ql = layout["local"][sel]
    fam = layout["family"][sel]
    m, n = len(sel), tr.n
    L = tr.radial[tr.t_radial[sel]]
    D, L = L[:, 1], L[:, 0]
    ev, ed = tr.eta_value[tr.t_eta[sel]], tr.eta_derivative[tr.t_eta[sel]]
    owners = np.empty((m, 4, 4, 7), dtype=np.int64)
    vv = np.empty((m, 4, 4, 7))
    dd = np.empty((m, 4, 4, 7))
    ring = fam == _RINGWISE
    if (~ring).any():
        a = ~ring
        theta = tr.t_theta[layout["theta_row"][sel[a]]]
        raw = _raw_ids(n, tr.layers[layout["source"][sel[a]]], tr.theta_index[theta], tr.eta_index[tr.t_eta[sel[a]]])
        owners[a] = _lookup_owner(tr.raw_ids, tr.raw_owner, raw)
        vv[a] = tr.theta_value[theta][:, None, None, :]
        dd[a] = tr.theta_derivative[theta][:, None, None, :]
    if ring.any():
        entry = tr.t_ring[layout["ring_row"][sel[ring]]]
        owners[ring], vv[ring], dd[ring] = tr.ring_owner[entry], tr.ring_value[entry], tr.ring_derivative[entry]
    Lc, Dc = L[:, :, None, None], D[:, :, None, None]
    ev4, ed4 = ev[:, None, :, None], ed[:, None, :, None]
    lev = Lc * ev4                       # (L*ev)*tv, (D*ev)*tv, (L*ev)*td, (L*ed)*tv
    blocks = (lev * vv, (Dc * ev4) * vv, lev * dd, (Lc * ed4) * vv)

    owners = owners.reshape(m, _K)
    key = (src_of.astype(np.int64) << 32)[:, None] | owners
    unique, inverse = np.unique(key.ravel(), return_inverse=True)
    width = np.bincount(unique >> 32, minlength=len(src_ids)).astype(np.int64)
    donor_ptr = _ptr(width)
    column = inverse.reshape(m, _K) - donor_ptr[src_of][:, None]
    q = counts[src_ids]
    has_gradient = tr.has_gradient[src_ids]
    value_ptr = _ptr(q * width)
    value = np.zeros(value_ptr[-1])
    _accumulate(value, (value_ptr[src_of] + ql * width[src_of])[:, None] + column, blocks[0].reshape(m, _K))
    gradient_ptr = _ptr(np.where(has_gradient, q * width, 0))
    gradient = np.zeros((3, gradient_ptr[-1]))
    g = has_gradient[src_of]
    if g.any():
        flat = (gradient_ptr[src_of[g]] + ql[g] * width[src_of[g]])[:, None] + column[g]
        for a in range(3):
            _accumulate(gradient[a], flat, blocks[a + 1][g].reshape(-1, _K))
    return src_ids, dict(width=width, donor=(unique & 0xFFFFFFFF), value=value, gradient=gradient)


def _expand_centered(tr: TensorRows, s: int, layout: dict):
    """One centered_radial source: ``boundary_map``'s arithmetic replayed literally (add.at into zeros over
    the union of owners, then the same einsum over layers)."""
    t0, t1 = int(tr.target_ptr[s]), int(tr.target_ptr[s + 1])
    q, n = t1 - t0, tr.n
    layout_theta = tr.t_theta[layout["theta_row"][t0:t1]]
    ti, eta = tr.theta_index[layout_theta], tr.t_eta[t0:t1]
    tv, td = tr.theta_value[layout_theta], tr.theta_derivative[layout_theta]
    ev, ed = tr.eta_value[eta], tr.eta_derivative[eta]
    radial = tr.radial[tr.t_radial[t0:t1]]
    L, D = np.ascontiguousarray(radial[:, 0]), np.ascontiguousarray(radial[:, 1])
    layers = np.repeat(tr.layers[s][None].astype(np.int64), q, axis=0)
    owners = _lookup_owner(tr.raw_ids, tr.raw_owner, _raw_ids(n, layers, ti, tr.eta_index[eta]))
    ids = np.unique(owners)
    idx = np.searchsorted(ids, owners)
    angular = np.stack((ev[:, :, None] * tv[:, None, :], ev[:, :, None] * td[:, None, :], ed[:, :, None] * tv[:, None, :]))
    layer = np.zeros((3, q, 4, len(ids)))
    qi, li = np.arange(q)[:, None, None, None], np.arange(4)[None, :, None, None]
    np.add.at(layer, (slice(None), qi, li, idx), angular[:, :, None, :, :])
    h = 1 / n
    value = np.einsum('ql,qld->qd', L, layer[0])
    gradient = None
    if tr.has_gradient[s]:
        gradient = np.empty((q, 3, len(ids)))
        gradient[:, 0] = np.einsum('ql,qld->qd', D / h, layer[0])
        gradient[:, 1] = np.einsum('ql,qld->qd', L, layer[1])
        gradient[:, 2] = np.einsum('ql,qld->qd', L, layer[2])
    return ids, value, gradient


def expand_tensor_rows(tr: TensorRows) -> TensorExpansion:
    """The exact dense rows of every source of ``tr`` (same arrays the construction produced: donors are the
    ascending union of the owners of all targets; ``np.unique`` for singleton/centered_radial, ``sorted`` of
    the accumulated owners for ringwise)."""
    ts = tr.n_sources
    counts = np.diff(tr.target_ptr)
    layout = tr.target_layout()
    src_ids, product = _expand_products(tr, layout)
    centered = np.flatnonzero(tr.family == _CENTERED)
    pieces = [_expand_centered(tr, int(s), layout) for s in centered]
    width = np.zeros(ts, dtype=np.int64)
    if product is not None:
        width[src_ids] = product["width"]
    for s, (ids, _, _) in zip(centered, pieces):
        width[s] = len(ids)
    donor_ptr = _ptr(width)
    value_ptr = _ptr(counts * width)
    gradient_ptr = _ptr(np.where(tr.has_gradient, counts * width, 0))
    donor = np.empty(donor_ptr[-1], dtype=np.int64)
    value = np.empty(value_ptr[-1])
    gradient = np.empty((3, gradient_ptr[-1]))
    if product is not None:
        donor[_flat_index(donor_ptr[src_ids], width[src_ids])] = product["donor"]
        value[_flat_index(value_ptr[src_ids], (counts * width)[src_ids])] = product["value"]
        g = tr.has_gradient[src_ids]
        if g.any():
            gradient[:, _flat_index(gradient_ptr[src_ids[g]], (counts * width)[src_ids[g]])] = product["gradient"]
    for s, (ids, v, grad) in zip(centered, pieces):
        donor[donor_ptr[s]:donor_ptr[s + 1]] = ids
        value[value_ptr[s]:value_ptr[s + 1]] = v.reshape(-1)
        if grad is not None:
            gradient[:, gradient_ptr[s]:gradient_ptr[s + 1]] = grad.transpose(1, 0, 2).reshape(3, -1)
    return TensorExpansion(donor_ptr, donor, value_ptr, value, gradient_ptr, gradient)


# --------------------------------------------------------------------------------------
# paired sources: 1/2 (A + B) of two consecutive expanded sources
# --------------------------------------------------------------------------------------

def average_expanded_rows(donor_a, value_a, gradient_a, donor_b, value_b, gradient_b):
    """``(donor, value, gradient)`` of ``1/2 (A + B)`` for two dense rows, exactly as
    ``StructuredReconstruction._average_rows`` merges them: the ascending union of the donors, zeros, ``+= A``,
    ``+= B`` (a donor of one row only gets ``0.0 + w``, so ``-0.0`` becomes ``+0.0``), then ``0.5 *`` the whole.

    A row is ``donor (d,)``, ``value (q, d)`` and ``gradient (q, 3, d)`` (``None`` for a source that stores no
    gradient; then both are ``None`` and so is the result's). The one place the artifact encoder's verification and
    its decode merge a pair: they cannot disagree with each other, and ``tests/test_perpendicular_paired_tensor_rows.py``
    pins this to ``_average_rows`` itself.
    """
    if (gradient_a is None) != (gradient_b is None):
        raise ValueError("paired rows: only one of the two stores a gradient")
    ids = np.union1d(donor_a, donor_b)
    value = np.zeros((len(value_a), len(ids)))
    gradient = None if gradient_a is None else np.zeros((len(value_a), 3, len(ids)))
    for donor, v, g in ((donor_a, value_a, gradient_a), (donor_b, value_b, gradient_b)):
        pos = np.searchsorted(ids, donor)
        value[:, pos] += v
        if gradient is not None:
            gradient[:, :, pos] += g
    return ids, 0.5 * value, None if gradient is None else 0.5 * gradient


def merge_paired_expansion(exp: TensorExpansion, counts, has_gradient, multiplicity) -> TensorExpansion:
    """The expansion of the *logical* sources of an expansion of ``TensorRows`` sources.

    ``multiplicity[k]`` (1 or 2) consecutive expanded sources form logical source ``k``: a single is itself, a
    pair ``(A, B)`` is :func:`average_expanded_rows` of the two (equal target counts and gradient storage).
    ``counts`` and ``has_gradient`` are those of the expanded sources. All singles: ``exp`` itself.
    """
    multiplicity = np.asarray(multiplicity, dtype=np.int64)
    counts, has_gradient = np.asarray(counts, dtype=np.int64), np.asarray(has_gradient, dtype=bool)
    if (np.any((multiplicity < 1) | (multiplicity > 2)) or int(multiplicity.sum()) != len(exp.donor_ptr) - 1
            or len(counts) != len(has_gradient) or len(counts) != len(exp.donor_ptr) - 1):
        raise ValueError("paired expansion: multiplicity does not match the expanded sources")
    if (multiplicity == 1).all():
        return exp
    first = np.cumsum(multiplicity) - multiplicity
    donors, values, gradients = [], [], []
    for s, m in zip(first.tolist(), multiplicity.tolist()):
        parts = [exp.source_rows(s + k, int(counts[s + k]), bool(has_gradient[s + k])) for k in range(m)]
        if m == 2 and (counts[s] != counts[s + 1] or has_gradient[s] != has_gradient[s + 1]):
            raise ValueError("paired expansion: the two parts of a pair differ in targets or gradient storage")
        donor, value, gradient = parts[0] if m == 1 else average_expanded_rows(*parts[0], *parts[1])
        donors.append(donor)
        values.append(value)
        gradients.append(gradient)
    width = np.array([len(d) for d in donors], dtype=np.int64)
    q, with_gradient = counts[first], has_gradient[first]
    return TensorExpansion(
        _ptr(width), np.concatenate(donors).astype(np.int64),
        _ptr(q * width), np.concatenate([v.reshape(-1) for v in values]),
        _ptr(np.where(with_gradient, q * width, 0)),
        np.concatenate([g.transpose(1, 0, 2).reshape(3, -1) for g in gradients if g is not None] or [np.zeros((3, 0))],
                       axis=1))


# --------------------------------------------------------------------------------------
# verified construction (the encoder's use)
# --------------------------------------------------------------------------------------

def _bit_differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(np.uint64) != np.ascontiguousarray(b).view(np.uint64)


def _source_of_entries(ptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))


def _failed_sources(tr: TensorRows, expected: dict, multiplicity=None) -> np.ndarray:
    """Boolean per candidate: does the expansion differ (in donors, value or gradient bits) from ``expected``
    (``donor_ptr, donor, value, gradient`` in the expansion's own flat layout)?

    ``multiplicity`` (per candidate, default all 1) is the number of consecutive sources of ``tr`` that form the
    candidate: a candidate of 2 is a pair and its expected rows are those of their merge
    (:func:`merge_paired_expansion`)."""
    counts, has_gradient = np.diff(tr.target_ptr), tr.has_gradient
    multiplicity = np.ones(tr.n_sources, dtype=np.int64) if multiplicity is None else np.asarray(multiplicity)
    ts = len(multiplicity)
    bad = np.zeros(ts, dtype=bool)
    try:
        exp = expand_tensor_rows(tr)
        if (multiplicity == 2).any():
            exp = merge_paired_expansion(exp, counts, has_gradient, multiplicity)
            first = np.cumsum(multiplicity) - multiplicity
            counts, has_gradient = counts[first], has_gradient[first]
    except (ValueError, IndexError, KeyError):
        return np.ones(ts, dtype=bool)
    width = np.diff(exp.donor_ptr)
    bad |= width != np.diff(expected["donor_ptr"])
    if bad.any():
        return bad
    bad |= np.bincount(_source_of_entries(exp.donor_ptr), weights=(exp.donor != expected["donor"]), minlength=ts) > 0
    src_v = np.repeat(np.arange(ts), counts * width)
    bad |= np.bincount(src_v, weights=_bit_differs(exp.value, expected["value"]), minlength=ts) > 0
    g_counts = np.where(has_gradient, counts * width, 0)
    if g_counts.sum():
        src_g = np.repeat(np.arange(ts), g_counts)
        differs = _bit_differs(exp.gradient, expected["gradient"]).any(axis=0)
        bad |= np.bincount(src_g, weights=differs, minlength=ts) > 0
    return bad


def _parts(candidate) -> tuple:
    """The ``PointFactors`` a verified candidate is stored as: itself, or ``(a, b)`` of a ``PairedFactors``."""
    return (candidate.a, candidate.b) if isinstance(candidate, PairedFactors) else (candidate,)


def _raw_owner_conflicts(factors) -> np.ndarray:
    """Sources touching a raw cell that two of the given factors map to different owners (their donor
    tables cannot be stored consistently): never happens for one geometry; such sources stay CSR."""
    bad = np.zeros(len(factors), dtype=bool)
    ang = [s for s, f in enumerate(factors) if FAMILY_CODES.get(f.family) in (_SINGLETON, _CENTERED)]
    if not ang:
        return bad
    try:
        raw = np.concatenate([_raw_ids(int(factors[s].n), np.repeat(np.asarray(factors[s].layers, dtype=np.int64)[None],
                                                                    len(factors[s].radial), axis=0),
                                       factors[s].theta_index, factors[s].eta_index).reshape(-1) for s in ang])
        owner = np.concatenate([np.asarray(factors[s].owner).reshape(-1) for s in ang])
    except (ValueError, IndexError, TypeError, AttributeError):
        return np.array([f.family not in FAMILY_CODES for f in factors])
    source = np.repeat(ang, [factors[s].owner.size for s in ang])
    unique, first, inverse = np.unique(raw, return_index=True, return_inverse=True)
    conflicting = np.unique(inverse.ravel()[owner != owner[first][inverse.ravel()]])
    bad[np.unique(source[np.isin(inverse.ravel(), conflicting)])] = True
    return bad


def _candidate_conflicts(candidates) -> np.ndarray:
    """``_raw_owner_conflicts`` per candidate (a pair conflicts if either of its parts does)."""
    parts = [part for candidate in candidates for part in _parts(candidate)]
    owner = np.repeat(np.arange(len(candidates)), [len(_parts(candidate)) for candidate in candidates])
    bad = np.zeros(len(candidates), dtype=bool)
    bad[owner[_raw_owner_conflicts(parts)]] = True
    return bad


def verified_tensor_rows(factors: Sequence[PointFactors | PairedFactors], has_gradient: Sequence[bool], expected_for):
    """Build :class:`TensorRows` for ``factors`` keeping only the candidates whose expansion reproduces the dense
    rows bit for bit.

    A candidate is a ``PointFactors`` (one source of the result) or a ``PairedFactors`` (two consecutive sources,
    A then B, whose :func:`average_expanded_rows` merge is its dense row). ``expected_for(indices)`` returns the dense
    rows of the given input candidates as ``dict(donor_ptr, donor, value, gradient)`` in the expansion's flat layout
    (one entry per candidate, a pair's being its merged row). Returns ``(tensor_rows or None, accepted)`` with
    ``accepted`` a boolean per input candidate; candidates that fail are dropped (the caller stores them as CSR)
    and the survivors are re-verified with their own tables.
    """
    factors, has_gradient = list(factors), np.asarray(has_gradient, dtype=bool)
    multiplicity = np.array([len(_parts(candidate)) for candidate in factors], dtype=np.int64)
    live = np.arange(len(factors))
    conflicts_checked = False

    def tabulate(indices):
        return build_tensor_rows([part for i in indices for part in _parts(factors[i])],
                                 np.repeat(has_gradient[indices], multiplicity[indices]))

    while len(live):
        try:
            tr = tabulate(live)
        except (ValueError, KeyError, IndexError, TypeError):
            if not conflicts_checked:            # one raw cell with two owners across sources: drop those sources
                conflicts_checked = True
                keep = ~_candidate_conflicts([factors[i] for i in live])
                if not keep.all():
                    live = live[keep]
                    continue
            # a candidate whose factors cannot even be tabulated: isolate it
            keep = []
            for i in live:
                try:
                    tabulate(np.array([i]))
                    keep.append(i)
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
            if len(keep) == len(live):
                return None, np.zeros(len(factors), dtype=bool)
            live = np.array(keep, dtype=np.int64)
            continue
        bad = _failed_sources(tr, expected_for(live), multiplicity[live])
        if not bad.any():
            accepted = np.zeros(len(factors), dtype=bool)
            accepted[live] = True
            return tr, accepted
        live = live[~bad]
    return None, np.zeros(len(factors), dtype=bool)


# --------------------------------------------------------------------------------------
# NumPy reference apply (no dense expansion)
# --------------------------------------------------------------------------------------

def apply_tensor_rows_numpy(tr: TensorRows, owner_fields, *, values: bool = True, gradients: bool = True,
                            block: int = 1024):
    """Contract the factors with ``owner_fields`` (owners, F) without expanding the rows.

    Returns ``(values (Tt, F), gradients (Tg, 3, F))``: one value row per target in order, one gradient
    row per target of a source with ``has_gradient``. Per target and field, with ``G`` the gathered donor
    block (layer l, plane e, slot j):
    value = sum L[l] ev[e] w[j] G, d/dr = sum Dr[l] ev[e] w[j] G, d/dtheta = sum L[l] ev[e] w'[j] G,
    d/deta = sum L[l] ed[e] w[j] G, where (w, w') = (tv, td) for singleton/centered_radial (the same for all
    (l, e)), the ring entry's (vv, dd) per (l, e) for ringwise, and Dr = D (singleton, ringwise) or D/h
    (centered_radial, h = 1/n). Summation order differs from the expansion: results agree to rounding.
    """
    fields = np.asarray(owner_fields, dtype=np.float64)
    if fields.ndim == 1:
        fields = fields[:, None]
    layout = tr.target_layout()
    tt = tr.n_targets
    grad_target = tr.has_gradient[layout["source"]]
    grad_row = np.cumsum(grad_target) - 1
    out_v = np.zeros((tt, fields.shape[1])) if values else None
    out_g = np.zeros((int(grad_target.sum()), 3, fields.shape[1])) if gradients else None
    h = 1 / tr.n
    for a in range(0, tt, block):
        t = np.arange(a, min(tt, a + block))
        fam = layout["family"][t]
        radial = tr.radial[tr.t_radial[t]]
        L, D = radial[:, 0], np.where((fam == _CENTERED)[:, None], radial[:, 1] / h, radial[:, 1])
        ev, ed = tr.eta_value[tr.t_eta[t]], tr.eta_derivative[tr.t_eta[t]]
        owners = np.empty((len(t), 4, 4, 7), dtype=np.int64)
        vv = np.empty((len(t), 4, 4, 7))
        dd = np.empty((len(t), 4, 4, 7))
        ring = fam == _RINGWISE
        if (~ring).any():
            m = ~ring
            theta = tr.t_theta[layout["theta_row"][t[m]]]
            raw = _raw_ids(tr.n, tr.layers[layout["source"][t[m]]], tr.theta_index[theta], tr.eta_index[tr.t_eta[t[m]]])
            owners[m] = _lookup_owner(tr.raw_ids, tr.raw_owner, raw)
            vv[m] = tr.theta_value[theta][:, None, None, :]
            dd[m] = tr.theta_derivative[theta][:, None, None, :]
        if ring.any():
            entry = tr.t_ring[layout["ring_row"][t[ring]]]
            owners[ring], vv[ring], dd[ring] = tr.ring_owner[entry], tr.ring_value[entry], tr.ring_derivative[entry]
        gathered = fields[owners]                                     # (m, 4, 4, 7, F)
        contract = lambda lw, ew, w: np.einsum('ml,me,mlej,mlejf->mf', lw, ew, w, gathered)
        if values:
            out_v[t] = contract(L, ev, vv)
        if gradients:
            g = grad_target[t]
            if g.any():
                rows = grad_row[t[g]]
                out_g[rows, 0] = contract(D, ev, vv)[g]
                out_g[rows, 1] = contract(L, ev, dd)[g]
                out_g[rows, 2] = contract(L, ed, vv)[g]
    return out_v, out_g
