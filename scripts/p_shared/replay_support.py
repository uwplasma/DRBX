"""Shared replay support: the generic Tier-B/pointwise-cap comparison core,
the campaign environment (geometry/reference/census, shared by every unit),
the bounded Neumann-row rebuild (preflight-only -- see the module docstring
below), and small per-campaign helpers every unit in
:mod:`p_shared.replay_units` needs.

This module replaces ``scripts/p_shared/replay.py`` ("the older monolithic
reference"), which is retired (deleted, not just unused) as of the P08
step-1 replay-units restructure. ``replay.py`` did the right *numerics* but
(a) its ``RowIndex.from_artifact_files_streaming`` expanded the *whole*
artifact into memory before replaying a single campaign -- the failure mode
that motivated ``replay_units.py`` in the first place -- and (b) every one
of its per-campaign ``replay_p05n``/``replay_p06n``/... functions is now
dead code: nothing imports them any more, since ``replay_units.py``'s own
per-unit ``compute_cells_unit``/``compute_faces_unit``/``compute_p07_unit``
reimplement the same arithmetic directly. Only the pieces those *live*
functions (and ``p08_step1_global/campaign.py``) still need are kept here;
``replay.py``'s monolithic ``replay_p05``/``replay_p05n``/``replay_p06n``/
``replay_p06_legacy``/``replay_p07``/``replay_p07n``/``run_replay``/``main``
and the always-empty-in-practice ``RowIndex`` full-grid indexer are not
carried forward (nothing referenced ``env.row_index`` -- ``replay_units.py``
always passed a placeholder empty one).

Neumann rows: read from the artifact, never rebuilt on the fly
----------------------------------------------------------------
Every accepted campaign's row artifact already stores every Neumann
companion row it built, tagged ``(request, entity_id, quad_node)`` in its
``neumann_<stage>_<chunk>.npz`` files (see ``drbx.stencils.artifact``'s
``NeumannRowChunk`` and ``drbx.stencils.builder``'s ``NeumannRowRequest``).
``replay_units.py`` now looks those rows up directly (see
``load_neumann_chunk_for_unit``/``row_index_from_neumann_chunk`` there) --
it never calls :func:`drbx.geometry.fci_perpendicular_neumann_trace
.prepare_neumann_point_rows` in its main compute path any more. This both
eliminates the wrong-query-point bug class the P08 step-1 task report found
(rebuilding at ``PointRows.trace_target_points``, the wall-projected
Dirichlet-lift anchor, instead of the row's own true query point) and is
cheaper (no per-row SVD/solve at replay time).

:class:`NeumannSource` (kept here) and :func:`prepare_neumann_point_rows`
are used in exactly one place now: ``replay_units.neumann_rebuild_compare_check``,
a bounded preflight check that rebuilds a *few* Neumann rows at their true
query points and compares them bitwise against the artifact's own stored
rows -- never as a substitute for reading the artifact.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

import sys

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
_REPO = _SCRIPTS.parent
_WORKSPACE = _REPO.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drbx.stencils import artifact as artifact_mod                    # noqa: E402
from drbx.stencils.census import FaceCensus                            # noqa: E402
from drbx.geometry.fci_perpendicular_reconstruction import (            # noqa: E402
    PointRowContext, StructuredReconstruction)
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows  # noqa: E402

from p_shared import provider as pshared_provider                       # noqa: E402

N = 32  # the row artifact/replay run to completion locally; see the task report.

# ---------------------------------------------------------------------------
# Test-only local-workspace defaults (design section 5's table; each is a
# frozen campaign's own output directory, read-only). Nothing on this
# package's runtime path (``p08_step1_global.campaign``, this module's own
# ``build_environment``, ``p_shared.replay_units``, ``p_shared.owner_closure``)
# uses these as a fallback default any more -- every runtime caller takes its
# ``sidecar_path``/``paths`` explicitly (``campaign.py``'s ``--input-root``/
# ``--oracle-root``), so a remote run whose checkout does not sit inside this
# local workspace is never affected by ``_WORKSPACE`` being wrong there. The
# only remaining use is a handful of gated tests (``tests/test_p_shared_
# owner_closure.py``, guarded by its own ``needs_geometry``/``needs_oracle``
# skip marks) that want a real local oracle tree to compare against when one
# happens to be present.
# ---------------------------------------------------------------------------
DEFAULT_SIDECAR = _WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"

DEFAULT_PATHS = {
    "p05": _WORKSPACE / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3",
    "p05_upwind_chunks": _WORKSPACE / "work/p05_failed_58880191/p05/chunks",
    "p05n_frozen": _WORKSPACE / "work/p05n_field_derived_05be9063_20260927T230447Z_6140a1",
    "p05n_p06n_upwind": _WORKSPACE / "work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd",
    "p06_legacy": _WORKSPACE / "work/p06_completed_58880303/p06",
    "p07": _WORKSPACE / "work/p07_validated_58882298/p07",
    "p07n": _WORKSPACE / "work/p07n_field_derived_274e93e9_20260927T054625Z_72cfa1",
}

# Every oracle campaign's catalogue file, pinned explicitly and independent of
# whatever a package's own (mutable) ``configuration.json`` currently points
# at -- see the task report's "catalogue pinning" finding. Each entry is
# ``(module_attr, table_key)``: ``table_key`` indexes that module's own
# ``_CATALOGUE_TABLES``/frozen-catalogue-file dict, never the module's
# import-time-bound globals (``NAMES``/``ROLES``/``PAIRINGS``/...), which
# reflect only whichever catalogue is *currently* configured. P06N has no
# such switch (``p06n_field_derived_global.core`` always loads the one
# ``p06n_catalogue.json`` by its own hardcoded default -- see
# ``load_catalogue()`` there); P07/P07N's fields are hardcoded in
# ``p07n_field_derived_global/fields.py`` with no config-driven catalogue
# selection at all. Both were checked and found not to have this risk.
CAMPAIGN_CATALOGUE_FILES = {
    "p05n_frozen": "scripts/p05n_field_derived_global/p05n_catalogue.json",
    "p05n_upwind": "scripts/p05n_field_derived_global/p05n_upwind_catalogue.json",
    "p06n": "scripts/p06n_field_derived_global/p06n_catalogue.json",
}

CAMPAIGN_FUNCS = ("p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n")


# ===========================================================================
# Generic comparison / tolerance core (design section 5 "Tolerances";
# fully unit-testable without any workspace data).
# ===========================================================================
def owner_weighted_l2(diff: np.ndarray, owner_volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Owner-volume-weighted L2 of ``diff`` (owners, ...), optionally restricted
    to ``mask``. ``None`` if the (masked) selection is empty or has zero
    total volume -- callers must treat that as "not evaluable", not zero."""
    diff = np.asarray(diff, dtype=np.float64)
    volume = np.asarray(owner_volume, dtype=np.float64)
    if diff.shape[0] != volume.shape[0]:
        raise ValueError(f"owner axis mismatch: diff {diff.shape} vs volume {volume.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != volume.shape:
            raise ValueError("mask must have one entry per owner")
        diff = diff[mask]
        volume = volume[mask]
    total_volume = float(np.sum(volume))
    if diff.shape[0] == 0 or total_volume <= 0.0:
        return None
    per_owner_sq = np.sum(diff.reshape(diff.shape[0], -1) ** 2, axis=1)
    return float(np.sqrt(np.sum(volume * per_owner_sq) / total_volume))


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None:
        return None
    if numerator == 0.0:
        return 0.0
    if denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _column_reference(saved: np.ndarray) -> np.ndarray:
    """The default cap reference: per-column max of ``|saved|`` over owners
    (axis 0, ``keepdims``); the whole-array max for 1-D input."""
    mag = np.abs(saved)
    if mag.size == 0:
        return np.zeros((1,) * max(mag.ndim, 1))
    if mag.ndim <= 1:
        return np.max(mag)
    return np.max(mag, axis=0, keepdims=True)


def pointwise_cap(saved: np.ndarray, *, mode: str = "scaled", scale_factor: float = 1e-11,
                  flat_abs: float = 1e-9, zero_atol: float = 1e-11, zero_threshold: float = 1e-11,
                  cap_reference: Optional[np.ndarray] = None) -> np.ndarray:
    """The per-entry absolute cap (design section 5 "Pointwise caps").

    ``mode="scaled"``: ``scale_factor * scale`` with ``scale =
    max(|saved|, reference)`` elementwise (the "1e-11 x scale" rule). The
    reference is the term's column magnitude: ``cap_reference`` (broadcastable
    to ``saved``, e.g. shape ``(1, ncols)``) when given, otherwise the
    per-column max of ``|saved|`` over owners (the array max for 1-D input).
    An entrywise cap ``1e-11 * |saved|`` fails at cancellation points, where an
    entry is far smaller than the numbers that produced it and roundoff is
    set by the latter (a difference of O(1..100) terms landing near 1e-7).
    An entry whose ``scale`` is (numerically) zero gets ``zero_atol`` instead
    ("1e-11 absolute for zero targets"). ``mode="flat"``: a flat ``flat_abs``
    absolute cap (the "1e-9 absolute for P06 and P06N raw" rule, for the
    ~2e-10 cross-platform finite-difference curvature gap), with the same
    zero-target rule keyed on ``|saved|``; ``cap_reference`` is ignored.
    """
    saved = np.asarray(saved, dtype=np.float64)
    if mode == "scaled":
        reference = _column_reference(saved) if cap_reference is None else np.abs(
            np.asarray(cap_reference, dtype=np.float64))
        scale = np.maximum(np.abs(saved), reference)
        general = scale_factor * scale
        is_zero = scale <= zero_threshold
    elif mode == "flat":
        general = np.full(saved.shape, flat_abs, dtype=np.float64)
        is_zero = np.abs(saved) <= zero_threshold
    else:
        raise ValueError(f"unknown pointwise cap mode {mode!r}")
    return np.where(is_zero, zero_atol, general)


def compare_owner_term(name: str, replay: np.ndarray, saved: np.ndarray, *, owner_volume: np.ndarray,
                       archived_error: np.ndarray, region_masks: Optional[dict] = None,
                       rtol: float = 1e-3, cap_mode: str = "scaled", cap_scale_factor: float = 1e-11,
                       cap_flat_abs: float = 1e-9, cap_zero_atol: float = 1e-11,
                       cap_reference: Optional[np.ndarray] = None) -> dict:
    """Design section 5, Tier B + pointwise caps, for one term/array.

    Returns a JSON-safe dict: ``ratios`` (global + per region), ``worst_ratio``,
    the pointwise-cap violation count/location, and an overall ``pass``.
    A ``None`` ratio means "not evaluable" (e.g. an empty/zero-volume
    region) and never counts as a pass *or* a fail on its own. The pointwise
    cap scales with the term's column magnitude (``cap_reference``, default the
    per-column max of ``|saved|``; see :func:`pointwise_cap`).
    """
    replay = np.asarray(replay, dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    if replay.shape != saved.shape:
        raise ValueError(f"{name}: replay shape {replay.shape} != saved shape {saved.shape}")
    owner_volume = np.asarray(owner_volume, dtype=np.float64)
    archived_error = np.asarray(archived_error, dtype=np.float64)
    if archived_error.shape != saved.shape:
        raise ValueError(f"{name}: archived_error shape {archived_error.shape} != saved shape {saved.shape}")

    diff = replay - saved
    ratios: dict = {"global": _safe_ratio(owner_weighted_l2(diff, owner_volume),
                                         owner_weighted_l2(archived_error, owner_volume))}
    region_detail: dict = {}
    for region, mask in (region_masks or {}).items():
        d_l2 = owner_weighted_l2(diff, owner_volume, mask)
        a_l2 = owner_weighted_l2(archived_error, owner_volume, mask)
        ratios[region] = _safe_ratio(d_l2, a_l2)
        region_detail[region] = {"diff_l2": d_l2, "archived_l2": a_l2, "owner_count": int(np.count_nonzero(mask))}

    finite_ratios = [v for v in ratios.values() if v is not None]
    worst_ratio = max(finite_ratios) if finite_ratios else None
    worst_region = None
    if worst_ratio is not None:
        worst_region = next(k for k, v in ratios.items() if v == worst_ratio)

    abs_diff = np.abs(diff)
    cap = pointwise_cap(saved, mode=cap_mode, scale_factor=cap_scale_factor,
                        flat_abs=cap_flat_abs, zero_atol=cap_zero_atol, cap_reference=cap_reference)
    violation = abs_diff > cap
    n_violations = int(np.count_nonzero(violation))
    if abs_diff.size:
        worst_flat = int(np.argmax(abs_diff))
        worst_index = [int(x) for x in np.unravel_index(worst_flat, abs_diff.shape)]
        max_abs_diff = float(abs_diff.reshape(-1)[worst_flat])
        max_abs_diff_cap = float(cap.reshape(-1)[worst_flat])
    else:
        worst_index, max_abs_diff, max_abs_diff_cap = None, 0.0, 0.0

    passed = (worst_ratio is None or worst_ratio <= rtol) and n_violations == 0
    return {
        "name": name,
        "shape": list(replay.shape),
        "ratio_tolerance": rtol,
        "ratios": ratios,
        "worst_ratio": worst_ratio,
        "worst_region": worst_region,
        "region_detail": region_detail,
        "pointwise": {"mode": cap_mode, "violations": n_violations, "max_abs_diff": max_abs_diff,
                     "max_abs_diff_cap": max_abs_diff_cap, "worst_index": worst_index},
        "pass": bool(passed),
    }


def compare_pointwise_only(name: str, replay: np.ndarray, saved: np.ndarray, *, cap_mode: str = "scaled",
                          cap_scale_factor: float = 1e-11, cap_flat_abs: float = 1e-9,
                          cap_zero_atol: float = 1e-11, cap_reference: Optional[np.ndarray] = None) -> dict:
    """A pointwise-cap-only check (no owner-volume/region concept), for
    per-face targets such as P05's saved ``upwind`` array. The cap scales
    with the column magnitude (see :func:`pointwise_cap`)."""
    replay = np.asarray(replay, dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    if replay.shape != saved.shape:
        raise ValueError(f"{name}: replay shape {replay.shape} != saved shape {saved.shape}")
    diff = replay - saved
    abs_diff = np.abs(diff)
    cap = pointwise_cap(saved, mode=cap_mode, scale_factor=cap_scale_factor,
                        flat_abs=cap_flat_abs, zero_atol=cap_zero_atol, cap_reference=cap_reference)
    violation = abs_diff > cap
    n_violations = int(np.count_nonzero(violation))
    if abs_diff.size:
        worst_flat = int(np.argmax(abs_diff))
        worst_index = [int(x) for x in np.unravel_index(worst_flat, abs_diff.shape)]
        max_abs_diff = float(abs_diff.reshape(-1)[worst_flat])
    else:
        worst_index, max_abs_diff = None, 0.0
    return {
        "name": name, "shape": list(replay.shape),
        "pointwise": {"mode": cap_mode, "violations": n_violations, "max_abs_diff": max_abs_diff,
                     "worst_index": worst_index},
        "pass": n_violations == 0,
    }


# ===========================================================================
# Row lookup: expand a RowArtifact (or a small on-the-fly builder batch) into
# an O(1)-lookup index. Kept as a generic, artifact-scale-independent utility
# (its ``from_point_requests`` classmethod is what
# ``tests/test_stencils_builder.py``-style bounded on-the-fly checks use);
# ``replay_units.py``'s live per-unit compute path builds its own
# chunk-scoped equivalents (``row_index_from_point_chunk``/
# ``row_index_from_integrated_chunk``/``row_index_from_neumann_chunk``)
# instead of this class's whole-grid ``from_artifact``/
# ``from_artifact_files_streaming`` (the full-grid, all-chunks-in-memory-at-
# once construction the P08 step-1 task report's restructure replaced).
# ===========================================================================
@dataclass
class RowIndex:
    point: dict     # (request, entity_id) -> PointRows
    integrated: dict  # entity_id -> IntegratedFaceRow

    def get_point(self, request: str, entity_id: int):
        return self.point.get((request, int(entity_id)))

    def get_integrated(self, entity_id: int):
        return self.integrated.get(int(entity_id))

    @classmethod
    def from_artifact(cls, art) -> "RowIndex":
        point: dict = {}
        for chunk in tuple(art.cells) + tuple(art.faces):
            rows = artifact_mod.expand_point_rows(chunk)
            starts = chunk.source_ptr[:-1]
            requests = chunk.request[starts]
            entity_ids = chunk.entity_id[starts]
            for row, request, entity_id in zip(rows, requests, entity_ids):
                point[(str(request), int(entity_id))] = row
        integrated: dict = {}
        for chunk in tuple(art.p07):
            rows = artifact_mod.expand_integrated_rows(chunk)
            for row, entity_id in zip(rows, chunk.entity_id.tolist()):
                integrated[int(entity_id)] = row
        return cls(point, integrated)

    @classmethod
    def from_point_requests(cls, point_requests: Sequence = (), integrated_requests: Sequence = ()) -> "RowIndex":
        """Build the same index directly from ``drbx.stencils.builder``'s own
        ``PointRowRequest``/``IntegratedRowRequest`` lists -- used to develop
        and unit-test against small on-the-fly-built owner/face subsets
        without needing the (multi-GB) row artifact at all."""
        point = {(r.request, int(r.entity_id)): r.row for r in point_requests}
        integrated = {int(r.entity_id): r.row for r in integrated_requests}
        return cls(point, integrated)


class WallDataCache:
    """Per-(worker process, grid) cache of any pointwise field-evaluation
    callback's output on the fixed n x n Neumann wall lattice (``u == 1``,
    every ``(t.centers[1][i], t.centers[2][k])`` pair) -- generalizes
    ``p05n_field_derived_global.core.WallLattice`` (whose own docstring
    reports this amortization taking a boundary-face kernel from
    ~0.66s/face to sub-0.1s/face) to any campaign's own trace/normal-data
    closure, keyed by an explicit, stable string (never function identity:
    ``replay_units.py`` builds a fresh closure per unit/campaign call, so
    ``id(fn)`` would never repeat across units and the cache would never
    hit).

    Root cause this targets (see the task report): every trace/normal-data
    callback ultimately calls the frozen continuum reference's
    ``MetricEvaluator.evaluate_prepared`` finite-difference machinery, whose
    *dominant* cost is a largely size-independent fixed cost per distinct
    call (confirmed by direct profiling: a single call costs over a second
    regardless of whether it is handed 1 or ~1000 points) -- so the real win
    is not making each call faster, it is making far fewer of them: pay that
    fixed cost once per (key, grid) over the whole lattice, not once per
    unit.

    ``lookup_or_compute(key, fn, q)`` is a drop-in replacement for calling
    ``fn(q)`` directly: every query point that is bitwise exactly a lattice
    node is served from the cached table (built by calling ``fn`` once over
    the *whole* lattice, the first time ``key`` is seen); every other point
    falls back to calling ``fn`` live, on exactly the off-lattice subset
    (never silently dropped or approximated) -- so a batch that mixes
    lattice and non-lattice points (e.g. a face's own Dirichlet target
    points, only some of which sit exactly at the physical wall) is always
    handled correctly, just with a smaller live call. ``fn`` may return a
    single ``(Q, ...)`` array (a ``normal_data_fn``) or a tuple of such
    arrays (a ``dirichlet_trace_fn``'s ``(value, gradient)``); either way,
    every value returned here is either read verbatim out of the cached
    table (bitwise identical to calling ``fn`` on that exact point directly,
    since it *is* the same call, same point) or computed by ``fn`` itself on
    a genuine subset of ``q`` -- never re-derived differently.
    """

    def __init__(self, t):
        theta = np.asarray(t.centers[1], dtype=np.float64)
        eta = np.asarray(t.centers[2], dtype=np.float64)
        self._theta_index = {float(v): i for i, v in enumerate(theta)}
        self._eta_index = {float(v): k for k, v in enumerate(eta)}
        self._eta_count = len(eta)
        self.lattice_points = np.column_stack(
            (np.ones(len(theta) * len(eta), dtype=np.float64),
             np.repeat(theta, len(eta)), np.tile(eta, len(theta))))
        self._table_cache: dict = {}

    def _lattice_index(self, q: np.ndarray) -> np.ndarray:
        """``(len(q),)`` int64, ``-1`` at any row that is not bitwise a
        lattice node (``u != 1`` or ``(theta, eta)`` not an exact
        ``t.centers`` value)."""
        out = np.full(len(q), -1, dtype=np.int64)
        for i in range(len(q)):
            if float(q[i, 0]) != 1.0:
                continue
            ti = self._theta_index.get(float(q[i, 1]))
            ei = self._eta_index.get(float(q[i, 2]))
            if ti is not None and ei is not None:
                out[i] = ti * self._eta_count + ei
        return out

    def _table_for(self, key: str, fn: Callable):
        table = self._table_cache.get(key)
        if table is None:
            table = fn(self.lattice_points)
            self._table_cache[key] = table
        return table

    @staticmethod
    def _index_result(result, idx: np.ndarray):
        if isinstance(result, tuple):
            return tuple(part[idx] for part in result)
        return result[idx]

    def lookup_or_compute(self, key: str, fn: Callable, q: np.ndarray):
        """Drop-in replacement for ``fn(q)`` -- see the class docstring."""
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0:
            return fn(q)
        idx = self._lattice_index(q)
        on_mask = idx >= 0
        if not on_mask.any():
            return fn(q)
        table = self._table_for(key, fn)
        if on_mask.all():
            return self._index_result(table, idx[on_mask])

        on_result = self._index_result(table, idx[on_mask])
        off_result = fn(q[~on_mask])
        is_tuple = isinstance(table, tuple)
        on_parts = on_result if is_tuple else (on_result,)
        off_parts = off_result if is_tuple else (off_result,)
        merged = []
        for on_piece, off_piece in zip(on_parts, off_parts):
            out = np.empty((len(q),) + on_piece.shape[1:], dtype=np.result_type(on_piece, off_piece))
            out[on_mask] = on_piece
            out[~on_mask] = off_piece
            merged.append(out)
        return tuple(merged) if is_tuple else merged[0]


class NeumannSource:
    """Rebuilds a Neumann companion row batch on demand.

    Kept only for the bounded preflight check
    (``replay_units.neumann_rebuild_compare_check``), which rebuilds a *few*
    rows at their true query points and compares them bitwise against the
    artifact's own stored rows -- see this module's docstring. The live
    replay compute path (``compute_cells_unit``/``compute_faces_unit``/
    ``compute_p07_unit`` in ``replay_units.py``) reads stored rows instead
    and never constructs one of these for its main arithmetic.
    """

    def __init__(self, ctx: PointRowContext, normal_coefficients: Callable, patch_cache: Optional[dict] = None):
        self.ctx = ctx
        self.normal_coefficients = normal_coefficients
        self.patch_cache = {} if patch_cache is None else patch_cache

    def rows_for(self, points: np.ndarray, radial_degree: int):
        points = np.asarray(points, dtype=np.float64)
        return prepare_neumann_point_rows(self.ctx, points, normal_coefficients=self.normal_coefficients,
                                          radial_degree=radial_degree, patch_cache=self.patch_cache)


# ---------------------------------------------------------------------------
# Owner-weighted scatter helpers not already provided generically by
# scripts/p_shared/apply.py (design section 5 "P06 q1 raw material/remainder/
# total" -- an evolution-weight-scattered owner mean, not a face-jump/flux
# scatter).
# ---------------------------------------------------------------------------
def scatter_owner_weighted_mean(term: np.ndarray, weight: np.ndarray, owner_ids: np.ndarray, owner_count: int):
    """Owner-level ``sum(weight*term)/sum(weight)`` -- the q1 "evolution
    -weighted" raw-midpoint aggregation every P06(N) raw term uses (design
    section 2 "raw midpoints with evolution weight"; mirrors
    ``p05n_field_derived_global.campaign.project_raw``'s pattern, generalized
    to an arbitrary per-raw weight instead of the geometric raw volume)."""
    term = np.asarray(term, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    owner_ids = np.asarray(owner_ids, dtype=np.int64)
    if term.shape[0] != len(weight) or term.shape[0] != len(owner_ids):
        raise ValueError("term/weight/owner_ids must share their leading (raw) axis")
    numerator = np.zeros((owner_count,) + term.shape[1:], dtype=np.float64)
    denominator = np.zeros(owner_count, dtype=np.float64)
    np.add.at(numerator, owner_ids, weight.reshape(-1, *([1] * (term.ndim - 1))) * term)
    np.add.at(denominator, owner_ids, weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = numerator / denominator.reshape(-1, *([1] * (term.ndim - 1)))
    return mean, denominator


# ===========================================================================
# Shared campaign environment: the one canonical geometry/reference every
# campaign's row artifact was built against (design task 2's provider) --
# never each campaign's own possibly-differently-localized copy.
# ===========================================================================
@dataclass
class Environment:
    n: int
    t: object
    ctx: PointRowContext
    S: StructuredReconstruction
    census: FaceCensus
    ref: object
    normal_coefficients: Callable
    neumann: NeumannSource
    wall_cache: WallDataCache


def build_environment(*, n: int, input_root: Path, sidecar_path: Path) -> Environment:
    from perpendicular_structured.reconstruction import load_context
    from p07n_field_derived_global.fields import normal as p07n_normal

    t = load_context(n, str(input_root))
    ctx = PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro,
                                      raw_volume=t.rv, owner_volume=t.vol,
                                      owner_centroid_xy=t.g.owner_centroid_xy, eta_period=t.g.eta_period,
                                      dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)
    S = StructuredReconstruction(ctx)
    census = FaceCensus.build(n, t.ro)
    ref = pshared_provider.ScriptsGeometryProvider.from_sidecar(str(sidecar_path), verify_hashes=False).reference

    def normal_coefficients(q):
        return p07n_normal(ref, np.asarray(q, dtype=np.float64))

    neumann = NeumannSource(ctx, normal_coefficients)
    wall_cache = WallDataCache(t)
    return Environment(n=n, t=t, ctx=ctx, S=S, census=census, ref=ref,
                       normal_coefficients=normal_coefficients, neumann=neumann, wall_cache=wall_cache)


# ---------------------------------------------------------------------------
# Small per-campaign helpers ``replay_units.py``'s live compute functions use.
# ---------------------------------------------------------------------------
def _face_weight_for_key(env: Environment, key, points):
    """The q3 weight for one face key's already-known target points (the
    same shared quadrature primitive R2/R3 used to build ``points``)."""
    face_keys = np.asarray([key], dtype=np.int64)
    _points, weight = pshared_provider._quadrature(env.t.faces, face_keys, 3, face=True)
    return weight[0],


def _load_p05_upwind(chunk_dir: Path, n: int) -> np.ndarray:
    """Concatenate the accepted (pre-fix) P05 campaign's per-face ``upwind``
    chunks (``work/p05_failed_58880191/p05/chunks/N{n}_faces_*.npz``) into a
    dense ``(id_count, pairs)`` array indexed by the p07-topology face id."""
    import glob
    files = sorted(glob.glob(str(chunk_dir / f"N{n}_faces_*.npz")))
    if not files:
        raise FileNotFoundError(f"no P05 upwind face chunks found under {chunk_dir}")
    ids_all = []
    upwind_all = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            ids_all.append(z["ids"])
            upwind_all.append(z["upwind"])
    ids_all = np.concatenate(ids_all)
    upwind_all = np.concatenate(upwind_all)
    dense = np.full((int(ids_all.max()) + 1, upwind_all.shape[1]), np.nan, dtype=np.float64)
    dense[ids_all] = upwind_all
    return dense


def _p05n_evaluate(name, ref, points, period):
    """Evaluate one P05N/P06N catalogue field by name, using the broadest
    (P06N) field module -- a strict superset of P05N's own (design's own
    ``p06n_field_derived_global.fields`` docstring: "a strict superset of
    p05n's fields plus the shifted variants and the two rich P06N
    additions") -- so both P05N catalogues' field names resolve correctly
    regardless of which one is the currently *active* configuration."""
    import p06n_field_derived_global.fields as p06n_fields
    return p06n_fields.evaluate(ref, points, name, period)


def _tables_trace_all(tables, ref, q, period):
    q = np.asarray(q, dtype=np.float64)
    v = np.empty((len(q), len(tables.names)))
    g = np.empty((len(q), 3, len(tables.names)))
    for j, name in enumerate(tables.names):
        vv, gg, _ = tables.evaluate(ref, q, name, period)
        v[:, j] = vv
        g[:, :, j] = gg
    return v, g


def _summarize_variants(results: list) -> dict:
    """Fold a per-variant list of :func:`compare_owner_term` results into one
    entry: every variant's own detail, plus the worst ratio/violation count
    across all variants (a single term/array in design section 5's table
    corresponds to many catalogue variants here -- see design's "raw M/R/T
    [14,owners,4]")."""
    worst_ratio = max((r["worst_ratio"] for r in results if r["worst_ratio"] is not None), default=None)
    total_violations = sum(r["pointwise"]["violations"] for r in results)
    return {"variants": results, "worst_ratio": worst_ratio, "total_pointwise_violations": total_violations,
           "pass": all(r["pass"] for r in results)}


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3e}"


def write_report(replay: dict, output_dir: Path) -> Path:
    lines = [f"# P08 step 1 replay gate -- N{replay['n']}", "",
            f"Generated {replay['generated_at']} against artifact `{replay['artifact_root']}` "
            f"in {replay['wall_seconds']:.1f} s.", "",
            f"Reduction ran on {platform.system()} ({platform.node()}). Oracle arrays come from their "
            "original campaigns, so small cross-platform differences are possible "
            "(worst ratios reported below).", "",
            "| Campaign | Status | Terms | Worst ratio | Pointwise violations | Pass |",
            "|---|---|---|---|---|---|"]
    for name, result in replay["campaigns"].items():
        status = result.get("status")
        if status != "ok":
            lines.append(f"| {name} | {status} | - | - | - | FAIL ({result.get('error', '')[:80]}) |")
            continue
        terms = result.get("terms", {})
        worst_ratios = []
        violations = 0
        all_pass = True
        for term in terms.values():
            wr = term.get("worst_ratio")
            if wr is None and "variants" in term:
                wr = term.get("worst_ratio")
            if wr is not None:
                worst_ratios.append(wr)
            if "pointwise" in term:
                violations += term["pointwise"]["violations"]
            elif "total_pointwise_violations" in term:
                violations += term["total_pointwise_violations"]
            all_pass = all_pass and term.get("pass", False)
        worst = max(worst_ratios) if worst_ratios else None
        lines.append(f"| {name} | ok | {len(terms)} | "
                    f"{'n/a' if worst is None else f'{worst:.3e}'} | {violations} | "
                    f"{'PASS' if all_pass else 'FAIL'} |")
    lines.append("")
    for name, result in replay["campaigns"].items():
        lines.append(f"## {name}")
        lines.append("")
        if result.get("status") != "ok":
            lines.append(f"Status: **{result.get('status')}** -- {result.get('error', '')}")
            lines.append("")
            continue
        lines.append("| Term | Worst ratio | Worst region | Pointwise violations | Pass |")
        lines.append("|---|---|---|---|---|")
        for term_name, term in result.get("terms", {}).items():
            if "variants" in term:
                lines.append(f"| {term_name} | {_fmt(term['worst_ratio'])} | (per-variant) | "
                            f"{term['total_pointwise_violations']} | {'PASS' if term['pass'] else 'FAIL'} |")
            else:
                lines.append(f"| {term_name} | {_fmt(term.get('worst_ratio'))} | {term.get('worst_region')} | "
                            f"{term['pointwise']['violations']} | {'PASS' if term['pass'] else 'FAIL'} |")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
