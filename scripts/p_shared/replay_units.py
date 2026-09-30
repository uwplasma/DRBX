"""Unit-parallel replay gate (P08 step-1 replay, refactored into
``p_shared.runner.run_stage`` units -- see
``work/p08_step1_consolidation_design_20260928/design.md`` section 5 and the
task that asked for this refactor).

Why this module exists
-----------------------
This module is the sole live replay path (its predecessor,
``scripts/p_shared/replay.py``, was a monolithic, full-grid reference that
never ran to completion locally -- its own ``RowIndex.from_artifact_files
_streaming`` expanded **every** chunk file of the artifact into an
in-memory dict before a single campaign was replayed; at N64, per the
design doc's ~29 GB of R2/R3 rows alone, that is exactly the "expands every
artifact row into memory at once" failure mode that killed the first
replay run after 50 minutes at 7 GB with no output. It has been retired
(deleted); the small set of generic helpers it defined that this module
still needs -- the Tier-B/pointwise-cap comparison core, the campaign
``Environment``, and a handful of per-campaign trace helpers -- now live in
:mod:`p_shared.replay_support`).

This module replays the *same* per-campaign arithmetic (every helper is
imported from :mod:`p_shared.apply` and :mod:`p_shared.replay_support`
unchanged -- this file adds no new numerical formula) organized into small,
``runner.run_stage``-compatible **units**, one unit per artifact *build*
chunk file:

* a **cells** unit reads exactly one ``rows/cells_cells_<i>.npz`` file (one
  build chunk's worth of R1 rows, covering one contiguous raw-id range);
* a **faces** unit reads exactly one ``rows/faces_faces_<i>.npz`` file (one
  build chunk's worth of R2+R3 rows, covering one contiguous slice of the
  face-row-selection census-row domain);
* a **p07** unit reads exactly one ``rows/p07_p07_<i>.npz`` file (one build
  chunk's worth of R4 integrated rows, covering one contiguous slice of the
  P07-census domain).

Every unit that needs a Neumann companion row (a boundary-conditioned R1/R2
/R3/R4 row) also opens that same build chunk's stored
``rows/neumann_<stage>_<i>.npz`` file (``load_neumann_chunk_for_unit``) and
looks the row up by ``(request, entity_id, quad_node)``
(``row_index_from_neumann_chunk``) -- it is never rebuilt on the fly in the
main compute path. This is both a correctness fix and a simplification: an
earlier version of this module rebuilt Neumann rows on the fly via
:class:`p_shared.replay_support.NeumannSource`, and did so at the wrong
query point (``PointRows.trace_target_points``, the wall-projected anchor
``PointRows.apply``'s Dirichlet lift uses for its own boundary correction,
rather than the row's true query point) -- see the task report for the
root-cause finding and its bitwise-verified fix. ``NeumannSource`` and
:func:`drbx.geometry.fci_perpendicular_neumann_trace.prepare_neumann_point_rows`
are kept only for :func:`neumann_rebuild_compare_check`, a bounded preflight
that rebuilds a few rows at their true query points and compares them
bitwise against these same stored rows.

Owner-space additivity (the key trick that makes this a small refactor,
not a rewrite)
---------------------------------------------------------------------------
Every owner-level scatter the accepted campaigns use divides by a fixed,
geometry-only denominator (``project_raw_to_owners``'s ``owner_volume``,
``scatter_p05_jump``'s ``owner_volume``, ``scatter_integrated_face_flux``'s
``owner_volume``) -- so calling one of these on a *subset* of raw ids/faces
and summing the results across every unit's subset reproduces the full-grid
call exactly (division by a fixed denominator commutes with summation).
Only P06(N)'s q1 "evolution-weighted mean" (``scatter_owner_weighted_mean``
/ ``scatter_p06_characteristic``, both divide by a *data-dependent*,
accumulated ``evolution_volume``) is not directly additive: this module
instead accumulates that denominator once, across every **cells** unit of
the grid (the q1 evolution weight is a per-raw-cell, field-independent
quantity), and defers every division that depends on it to the reduction
stage -- exactly mirroring ``p06n_field_derived_global.campaign``'s own
``reduce_raw``/``reduce_faces``/``reduce_global`` split.

Every unit's own output is therefore always an *unnormalized numerator*,
stored sparsely (only the owners this unit's raw ids/faces actually touch --
see :func:`_sparse_scatter`), never a dense ``(owners, ...)`` array: at N64
a "cells" unit's default chunk size (4096 raw ids) touches at most 4096
owners, independent of the grid's total owner count, which is what keeps a
unit's own memory and disk footprint bounded as n grows (design's own "must
never load the whole artifact").

Two domain fixes baked into this module's own domain selection
--------------------------------------------------------------
Building small real units and replaying them (see this package's
``scripts/p08_step1_global/`` campaign) surfaced two cases where a naive
full-grid domain selection would raise ``ValueError`` against a real
artifact; both are applied directly in this module's own per-campaign
blocks (not inherited from elsewhere):

1. P05's live-jump domain (``stencil_builder.p07_row_selection``) includes
   the collapsed r=0 face (1024 census rows at N32), which R2/R3 never
   build a row for; P07/P07N never hit this (family 0 always takes the
   row-free ``apply_integrated_row`` branch). Fixed by excluding
   ``census.collapsed_r0`` from that domain (the collapsed face has
   degenerate/zero measure, so this drops no genuine contribution -- the
   same convention ``build_artifact.py`` uses for P07's own family-0
   integrand).
2. P06-legacy's face domain (``~census.collapsed_r0``) includes the
   periodic theta/eta alias slots (2048 rows at N32), which the artifact
   never stores a row for either (``build_artifact.py``'s
   ``face_row_selection`` excludes both ``collapsed_r0`` and
   ``legacy_alias_slots``). Fixed by excluding ``legacy_alias_slots`` too --
   verified empirically that no alias-slot row ever satisfies the "double
   the seam" ``double_mask`` condition, so this changes nothing about which
   rows get doubled, it only stops the loop from fetching a row that was
   never built.

Run from ``DRBX/scripts`` (never from inside a scripts package directory).
"""
from __future__ import annotations

import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent            # .../DRBX/scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drbx.stencils import artifact as artifact_mod                     # noqa: E402
from drbx.stencils.census import FaceCensus                            # noqa: E402
from drbx.stencils import builder as stencil_builder                    # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays                # noqa: E402
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor  # noqa: E402

from p_shared import apply as pshared_apply                             # noqa: E402
from p_shared import campaign_fields as cf                              # noqa: E402
from p_shared import provider as pshared_provider                       # noqa: E402
from p_shared import runner                                             # noqa: E402
from p_shared.replay_support import (                                   # noqa: E402
    CAMPAIGN_FUNCS, CAMPAIGN_CATALOGUE_FILES,
    Environment, NeumannSource, build_environment,
    compare_owner_term, compare_pointwise_only, owner_weighted_l2,
    _face_weight_for_key, _load_p05_upwind, _p05n_evaluate, _tables_trace_all,  # noqa: F401 (re-exported)
    _summarize_variants, write_report, _json_default,
)

__all__ = [
    "artifact_plan_units", "init_worker", "compute_cells_unit", "compute_faces_unit",
    "compute_p07_unit", "reduce_grid", "neumann_rebuild_compare_check", "face_row_selection",
    "STATE",
]

STATE: dict = {}


# ---------------------------------------------------------------------------
# Build-unit plan: read the artifact's OWN plan.json (written by
# build_artifact.py) so a replay unit's (start, stop) is bitwise the same
# range as the build unit that produced the chunk file it will open, and its
# ``chunk_index`` is that chunk's position in the manifest's per-group file
# list (build_artifact.py's ``_assemble`` appends manifest entries in the
# same order plan.json's units are enumerated, per stage).
# ---------------------------------------------------------------------------
def face_row_selection(census: FaceCensus) -> np.ndarray:
    """Mirrors ``build_artifact.face_row_selection`` bitwise (not imported
    from there: ``build_artifact.py`` is off-limits to edit or import-couple
    against while another agent parallelizes its geometry stage -- see the
    task's "Build side" instructions -- so this one-line selection is
    duplicated here, identically, rather than imported)."""
    return np.flatnonzero(~(census.collapsed_r0 | census.legacy_alias_slots))


def artifact_plan_units(artifact_root, n: int) -> dict:
    """``{stage: [{"stage","n","start","stop","chunk_index"}, ...]}`` for
    stage in ``("cells", "faces", "p07")``, read from
    ``<artifact_root>/N{n}/plan.json``."""
    grid_dir = Path(artifact_root) / f"N{n}"
    plan = json.loads((grid_dir / "plan.json").read_text())["plan"]
    out = {}
    for stage in ("cells", "faces", "p07"):
        out[stage] = [dict(u, chunk_index=i) for i, u in enumerate(plan[stage])]
    return out


def _manifest_entry(manifest: dict, group: str, stage: str, chunk_index: int) -> dict:
    relative = f"rows/{group}_{stage}_{chunk_index:05d}.npz"
    for entry in manifest["chunks"][group]:
        if entry["file"] == relative:
            return entry
    raise KeyError(f"no manifest entry {relative!r} in group {group!r}")


def _load_chunk_bytes(grid_dir: Path, entry: dict) -> bytes:
    data = (grid_dir / entry["file"]).read_bytes()
    actual = artifact_mod.hash_bytes(data)
    if actual != entry["sha256"]:
        raise ValueError(f"row artifact chunk corrupted: {entry['file']} "
                         f"(sha256 {actual} != manifest {entry['sha256']})")
    return data


def load_point_chunk_for_unit(artifact_root, n: int, manifest: dict, stage: str, chunk_index: int):
    grid_dir = Path(artifact_root) / f"N{n}"
    entry = _manifest_entry(manifest, stage, stage, chunk_index)
    data = _load_chunk_bytes(grid_dir, entry)
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    return artifact_mod._arrays_to_point_chunk(arrays)


def load_integrated_chunk_for_unit(artifact_root, n: int, manifest: dict, chunk_index: int):
    grid_dir = Path(artifact_root) / f"N{n}"
    entry = _manifest_entry(manifest, "p07", "p07", chunk_index)
    data = _load_chunk_bytes(grid_dir, entry)
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    return artifact_mod._arrays_to_integrated_chunk(arrays)


def load_neumann_chunk_for_unit(artifact_root, n: int, manifest: dict, stage: str, chunk_index: int):
    """Load one unit's stored Neumann companion rows -- ``rows/neumann_
    <stage>_<chunk_index>.npz`` -- the same build chunk index as that unit's
    own ``cells``/``faces``/``p07`` point/integrated chunk (see
    ``build_artifact.py``'s ``_assemble``: a stage's ``neumann`` part is
    written at the same index as its ``chunk`` part). Always present (an
    empty ``NeumannRowChunk`` when the unit had no boundary-conditioned
    rows), so this never needs a fallback."""
    grid_dir = Path(artifact_root) / f"N{n}"
    entry = _manifest_entry(manifest, "neumann", stage, chunk_index)
    data = _load_chunk_bytes(grid_dir, entry)
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    return artifact_mod._arrays_to_neumann_chunk(arrays)


def row_index_from_point_chunk(chunk) -> dict:
    """``{(request, entity_id): PointRows}`` for exactly the rows one chunk
    file holds (mirrors ``RowIndex.from_artifact``'s inner loop, scoped to a
    single chunk)."""
    rows = artifact_mod.expand_point_rows(chunk)
    starts = chunk.source_ptr[:-1]
    requests = chunk.request[starts]
    entity_ids = chunk.entity_id[starts]
    return {(str(r), int(e)): row for row, r, e in zip(rows, requests, entity_ids)}


def row_index_from_neumann_chunk(chunk) -> dict:
    """``{(request, entity_id, quad_node): NeumannPointRows}`` for exactly
    the Neumann companion rows one chunk file holds. ``request`` disambiguates
    an R2/R3 pair that shares the same ``entity_id`` (the census row index --
    see ``drbx.stencils.builder.build_r3_side_rows``'s docstring)."""
    rows = artifact_mod.expand_neumann_rows(chunk)
    return {(str(r), int(e), int(q)): row for row, r, e, q in
            zip(rows, chunk.request.tolist(), chunk.entity_id.tolist(), chunk.quad_node.tolist())}


def row_index_from_integrated_chunk(chunk) -> dict:
    rows = artifact_mod.expand_integrated_rows(chunk)
    return {int(e): row for row, e in zip(rows, chunk.entity_id.tolist())}


# ---------------------------------------------------------------------------
# Sparse owner-space scatter: every unit output is (touched_owner_ids,
# partial_values), never a dense (owners, ...) array (see module docstring).
# ---------------------------------------------------------------------------
def _sparse_scatter(term: np.ndarray, weight: Optional[np.ndarray], owner_ids: np.ndarray):
    """``(unique_owners, partial_sum)`` where ``partial_sum[k]`` is the sum,
    over every row mapping to ``unique_owners[k]``, of ``weight * term`` (or
    plain ``term`` if ``weight`` is ``None``). This is exactly one call's
    worth of ``np.add.at`` restricted to the owners this call's rows
    actually touch -- summing this across every unit of a grid, then
    dividing once by the true (fixed or accumulated) denominator, reproduces
    the full-grid scatter bitwise (module docstring, "Owner-space
    additivity")."""
    term = np.asarray(term, dtype=np.float64)
    owner_ids = np.asarray(owner_ids, dtype=np.int64)
    if term.shape[0] != len(owner_ids):
        raise ValueError("term/owner_ids leading axis mismatch")
    uniq, inverse = np.unique(owner_ids, return_inverse=True)
    out = np.zeros((len(uniq),) + term.shape[1:], dtype=np.float64)
    if weight is None:
        np.add.at(out, inverse, term)
    else:
        weight = np.asarray(weight, dtype=np.float64)
        np.add.at(out, inverse, weight.reshape(-1, *([1] * (term.ndim - 1))) * term)
    return uniq, out


def _sparse_scatter_signed(term: np.ndarray, lower_owner: np.ndarray, upper_owner: np.ndarray,
                           lower_sign: float = 1.0, upper_sign: float = -1.0):
    """``(unique_owners, partial_sum)`` for the "lower plus, upper minus" (or
    "lower minus, upper plus", or "both add") oriented face scatter pattern
    shared by ``scatter_p05_jump``/``scatter_integrated_face_flux``
    (``lower_sign=1, upper_sign=-1`` and ``lower_sign=-1, upper_sign=1``
    respectively) and ``scatter_p06_characteristic`` (called once per side
    with ``lower_sign=upper_sign=1`` and two different numerator arrays --
    see :func:`_sparse_scatter_both_sides`). ``lower_owner``/``upper_owner``
    entries of ``-1`` (no such side) are dropped, matching every accepted
    campaign's own ``jnp.where(lower >= 0, ..., 0)`` convention."""
    term = np.asarray(term, dtype=np.float64)
    lower_owner = np.asarray(lower_owner, dtype=np.int64)
    upper_owner = np.asarray(upper_owner, dtype=np.int64)
    owners = np.concatenate([lower_owner, upper_owner])
    values = np.concatenate([lower_sign * term, upper_sign * term], axis=0)
    valid = owners >= 0
    return _sparse_scatter(values[valid], None, owners[valid])


def _sparse_scatter_both_sides(lower_term: np.ndarray, upper_term: np.ndarray,
                               lower_owner: np.ndarray, upper_owner: np.ndarray):
    """The P06(N) characteristic-correction pattern: both sides *add*
    independently (design section 5's own "NOT an antisymmetric jump" --
    see ``scatter_p06_characteristic``)."""
    lower_owner = np.asarray(lower_owner, dtype=np.int64)
    upper_owner = np.asarray(upper_owner, dtype=np.int64)
    owners = np.concatenate([lower_owner, upper_owner])
    values = np.concatenate([np.asarray(lower_term, dtype=np.float64),
                             np.asarray(upper_term, dtype=np.float64)], axis=0)
    valid = owners >= 0
    return _sparse_scatter(values[valid], None, owners[valid])


class OwnerAccumulator:
    """A dense ``(owners, ...)`` array a reduction adds sparse unit outputs
    into -- lives only in the (single) reduction process, never in a
    worker."""

    def __init__(self, owners: int, tail_shape: tuple = ()):
        self.total = np.zeros((owners,) + tail_shape, dtype=np.float64)

    def add(self, owner_ids: np.ndarray, values: np.ndarray) -> None:
        if len(owner_ids) == 0:
            return
        np.add.at(self.total, np.asarray(owner_ids, dtype=np.int64), values)


# ---------------------------------------------------------------------------
# Worker-shared state: the geometry/reference/census/context/Neumann source
# every unit needs (loaded ONCE per worker process, never per unit -- design
# section 6's "Memory" target), plus every campaign's own small saved
# ``owner_values`` array (also loaded once: (owners, few-fields) is a few MB
# at most, unlike the multi-GB row artifact this module never loads whole).
# ---------------------------------------------------------------------------
def init_worker(*, artifact_root: str, n: int, input_root: str, sidecar_path: str,
                paths: dict, campaigns: tuple, output: str, campaign_identity: dict) -> None:
    global STATE
    runner.require_cpu_backend()
    artifact_root = Path(artifact_root)
    grid_dir = artifact_root / f"N{n}"
    build_identity = json.loads((grid_dir / "build_identity.json").read_text())
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    if manifest.get("schema") not in artifact_mod.SUPPORTED_SCHEMAS:
        raise ValueError(f"row artifact schema mismatch: {manifest.get('schema')!r} "
                         f"not in {artifact_mod.SUPPORTED_SCHEMAS!r}")
    if manifest.get("identity") != artifact_mod._json_safe(build_identity):
        raise ValueError("row artifact identity mismatch")

    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path))

    oracle = _load_oracle_owner_values(env, paths, campaigns)

    # The artifact's own field-independent geometry (task report: "source
    # geometry from the artifact's geometry.npz instead of recomputing it
    # live"), loaded ONCE per worker process and reused by every unit --
    # every field this replay reads from it (face h/jac/J/B/K/weight/
    # face_points, raw J/B/K/weight/raw_points) was checked bitwise equal to
    # what compute_faces_unit/_cells_unit_core computed live before this
    # change (see the task report); every stage that builds a row artifact
    # runs its geometry stage to full completion first (never partial), so
    # this file always exists by the time a cells/faces/p07 unit runs.
    geometry = GeometryArrays.load(grid_dir / "geometry.npz")

    STATE = {
        "artifact_root": artifact_root, "grid_dir": grid_dir, "n": n, "manifest": manifest,
        "build_identity": build_identity, "env": env, "paths": paths, "campaigns": tuple(campaigns),
        "oracle": oracle, "face_row_indices": face_row_selection(env.census),
        "p07_row_indices": stencil_builder.p07_row_selection(env.census),
        "output": Path(output), "identity_local": campaign_identity, "geometry": geometry,
    }


def _load_oracle_owner_values(env: Environment, paths: dict, campaigns: tuple) -> dict:
    """Every campaign's own saved ``owner_values`` (the *input* the row
    artifact reconstructs against -- never recomputed; see ``replay.py``'s
    module docstring), loaded once. Small: ``(owners, <=~20 fields)``.

    Grid-generic (every frozen campaign saved this file at N32, N48 *and*
    N64 -- see the task report): every path below is keyed by ``env.n``,
    never hardcoded to N32."""
    n = env.n
    out: dict = {}
    if "p05" in campaigns:
        with np.load(paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as z:
            out["p05"] = {"owner_values": z["observations"].copy()}
    if "p05n_frozen" in campaigns:
        with np.load(paths["p05n_frozen"] / f"N{n}.owner_values.npz", allow_pickle=False) as z:
            out["p05n_frozen"] = {"owner_values": z["values"].copy()}
    if "p05n_upwind" in campaigns:
        with np.load(paths["p05n_p06n_upwind"] / "p05n_upwind" / f"N{n}.owner_values.npz", allow_pickle=False) as z:
            out["p05n_upwind"] = {"owner_values": z["values"].copy()}
    if "p06n" in campaigns:
        with np.load(paths["p05n_p06n_upwind"] / "p06n" / f"N{n}.owner_values.npz", allow_pickle=False) as z:
            out["p06n"] = {"owner_values": z["values"].copy()}
    if "p06_legacy" in campaigns:
        with np.load(paths["p06_legacy"] / f"N{n}.prepare.npz", allow_pickle=False) as z:
            out["p06_legacy"] = {"owner_values": z["owner_values"].copy()}
    if "p07" in campaigns:
        # P07's own owner_values.npz is NOT itself per-grid (unlike every
        # other campaign here) -- it is one file whose keys are "N32"/"N48"/
        # "N64" (see oracle_manifest.campaign_files's own p07 entry).
        with np.load(paths["p07"] / "owner_values.npz", allow_pickle=False) as z:
            out["p07"] = {"owner_values": z[f"N{n}"].copy()}
    if "p07n" in campaigns:
        with np.load(paths["p07n"] / f"N{n}.owner_values.npz", allow_pickle=False) as z:
            out["p07n"] = {"owner_values": z["values"].copy()}
    return out


# ---------------------------------------------------------------------------
# Batched Dirichlet/Neumann row application (the vectorization fix -- see
# ``apply.batch_evaluate``'s docstring for why this exists: profiling a real
# boundary N32 cells unit found essentially all wall time inside the oracle
# ``trace``/``normal_data_fn`` callbacks, called once per row -- 256 boundary
# rows took >120s -- rather than once per unit). Scoped to a list of *single
# -target* (``Q == 1``) ``PointRows`` (an R1 cell row, or one R2/R3 face row
# processed on its own), which is every call site in this module.
# ---------------------------------------------------------------------------
def _batch_dirichlet_neumann(rows, dirichlet_trace_fn, normal_data_fn=None, neumann_rows_by_row=None, *,
                             wall_cache=None, trace_key=None, normal_key=None):
    """One batched Dirichlet trace evaluation (donor + target points, each
    concatenated across every boundary-conditioned row in ``rows``) plus,
    if ``normal_data_fn``/``neumann_rows_by_row`` are given, one batched
    28-point normal-data evaluation per row's wall lattice.

    ``wall_cache``, when given (together with ``trace_key``/``normal_key``,
    a caller-chosen stable string identifying this campaign/role -- NOT
    tied to ``dirichlet_trace_fn``/``normal_data_fn``'s own identity, since
    a fresh closure is built per unit call), routes every live call through
    :meth:`p_shared.replay_support.WallDataCache.lookup_or_compute` instead
    of calling the function directly: a query point that is bitwise exactly
    a wall-lattice node is served from a table built once per (key, grid)
    over the whole lattice; everything else still calls the function live,
    on exactly the off-lattice subset. Every Neumann ``boundary_points``
    array is always entirely on-lattice (see
    :class:`p_shared.replay_support.WallDataCache`'s docstring), so
    ``normal_key`` alone already amortizes essentially the whole
    normal-data cost across a grid's units; ``trace_key`` additionally
    helps whenever a face/cell's own Dirichlet target point happens to sit
    exactly at the physical wall (``u == 1``), and is a strict no-op
    (falls straight through to a live call) otherwise.

    ``neumann_rows_by_row``, when given, is a full-length (``len(rows)``)
    sequence (``None`` for an unconditioned row) of that row's own *stored*
    Neumann companion rows, already looked up from the artifact's
    ``neumann_<stage>_<chunk>.npz`` chunk by ``(request, entity_id,
    quad_node)`` -- see ``row_index_from_neumann_chunk``. This module never
    rebuilds a Neumann row at replay time any more (see the module
    docstring and the task report's root-cause finding: rebuilding on the
    fly at the wrong query point, ``PointRows.trace_target_points``, was the
    bug); the one remaining call to
    :func:`drbx.geometry.fci_perpendicular_neumann_trace.prepare_neumann_point_rows`
    in this file is ``neumann_rebuild_compare_check``, a bounded preflight
    check that rebuilds a few rows at their true query points and compares
    them bitwise against these same stored rows.

    A row's own ``Q`` (its target-point count -- 1 for an R1 cell row, or
    the census quadrature count, e.g. 9, for an R2/R3 face row) is never
    assumed to be 1: ``neumann_rows_by_row[i]`` is itself a length-``Q``
    sequence (one stored ``NeumannPointRows`` per node); ``nrows_of[i]``/
    ``boundary_data_of[i]`` mirror that -- an R1 caller just indexes ``[0]``.

    Returns a dict of per-row lists (``None`` where not applicable), meant
    to be fed straight into :func:`apply_point_row_precomputed` /
    :func:`apply_point_row_value_precomputed` / ``apply_neumann_row`` in the
    caller's own (cheap -- small matmuls only) per-row loop."""
    n = len(rows)
    bc_mask = np.array([bool(r.boundary_conditioned) for r in rows], dtype=bool)
    bc_idx = np.flatnonzero(bc_mask)
    donor_trace_value = [None] * n
    target_trace_value = [None] * n
    target_trace_gradient = [None] * n
    nrows_of = [None] * n
    boundary_data_of = [None] * n
    if bc_idx.size:
        trace_fn = dirichlet_trace_fn
        if wall_cache is not None and trace_key is not None:
            def trace_fn(q, _wall_cache=wall_cache, _key=trace_key, _fn=dirichlet_trace_fn):
                return _wall_cache.lookup_or_compute(_key, _fn, q)
        donor_groups = [rows[i].trace_donor_points for i in bc_idx]
        target_groups = [rows[i].trace_target_points for i in bc_idx]
        donor_results = pshared_apply.batch_evaluate(trace_fn, donor_groups)
        target_results = pshared_apply.batch_evaluate(trace_fn, target_groups)
        for local, i in enumerate(bc_idx):
            dv, _dg = donor_results[local]
            tv, tg = target_results[local]
            donor_trace_value[i] = dv
            target_trace_value[i] = tv
            target_trace_gradient[i] = tg
        if normal_data_fn is not None and neumann_rows_by_row is not None:
            for i in bc_idx:
                stored = neumann_rows_by_row[i]
                if stored is None:
                    raise ValueError(f"no stored Neumann row for boundary-conditioned row index {i}")
                nrows_of[i] = list(stored)
            flat_nrows = [nr for i in bc_idx for nr in nrows_of[i]]
            counts = [len(nrows_of[i]) for i in bc_idx]
            boundary_groups = [nr.boundary_points for nr in flat_nrows]
            normal_fn = normal_data_fn
            if wall_cache is not None and normal_key is not None:
                def normal_fn(q, _wall_cache=wall_cache, _key=normal_key, _fn=normal_data_fn):
                    return _wall_cache.lookup_or_compute(_key, _fn, q)
            boundary_results = pshared_apply.batch_evaluate(normal_fn, boundary_groups)
            offset = 0
            for i, c in zip(bc_idx, counts):
                boundary_data_of[i] = boundary_results[offset:offset + c]
                offset += c
    return {"bc_mask": bc_mask, "donor_trace_value": donor_trace_value,
            "target_trace_value": target_trace_value, "target_trace_gradient": target_trace_gradient,
            "nrows_of": nrows_of, "boundary_data_of": boundary_data_of}


def _batch_side_values(rows, side_exists_mask, fallback_points, owner_values, dirichlet_trace_fn,
                       normal_data_fn=None, neumann_rows_by_row=None, *,
                       wall_cache=None, trace_key=None, normal_key=None):
    """One (lower- or upper-) side's per-face ``(dirichlet, neumann)`` VALUE
    arrays across a whole faces unit, batched (design section 5 / the
    previous per-face ``side_value`` closure, restated so every Dirichlet
    /Neumann evaluation happens once for the whole unit instead of once per
    face -- see the module docstring). ``rows[i]`` is the R3 side row (or
    ``None``) for face ``i``; whenever ``side_exists_mask[i]`` is ``False``
    or ``rows[i]`` is ``None``, that face's value is the *unconditioned*
    ``dirichlet_trace_fn`` evaluated at ``fallback_points[i]`` (the common
    row's own target points) for BOTH its ``dirichlet`` and ``neumann``
    slots -- bitwise the same fallback the previous per-face closure took.

    ``neumann_rows_by_row``, when given, is a full-length (``len(rows)``)
    sequence of that face's stored Neumann companion rows (the R3 tag shares
    its ``entity_id`` with R2's common row -- see
    ``drbx.stencils.builder.build_r3_side_rows``'s docstring -- so the
    caller passes the same per-face lookup for both the common row and
    either side).

    Returns ``(dirichlet, neumann)``, each a length-``F`` list of ``(Q,
    fields)`` arrays."""
    F = len(rows)
    dirichlet = [None] * F
    neumann = [None] * F
    present = [i for i in range(F) if side_exists_mask[i] and rows[i] is not None]
    fallback = [i for i in range(F) if not (side_exists_mask[i] and rows[i] is not None)]
    if fallback:
        fb_groups = [fallback_points[i] for i in fallback]
        fb_trace_fn = dirichlet_trace_fn
        if wall_cache is not None and trace_key is not None:
            def fb_trace_fn(q, _wall_cache=wall_cache, _key=trace_key, _fn=dirichlet_trace_fn):
                return _wall_cache.lookup_or_compute(_key, _fn, q)
        fb_results = pshared_apply.batch_evaluate(fb_trace_fn, fb_groups)
        for i, (tv, _tg) in zip(fallback, fb_results):
            dirichlet[i] = tv
            neumann[i] = tv
    if present:
        present_rows = [rows[i] for i in present]
        present_neumann_rows = None if neumann_rows_by_row is None else [neumann_rows_by_row[i] for i in present]
        batch = _batch_dirichlet_neumann(present_rows, dirichlet_trace_fn, normal_data_fn,
                                         neumann_rows_by_row=present_neumann_rows,
                                         wall_cache=wall_cache, trace_key=trace_key, normal_key=normal_key)
        for local, i in enumerate(present):
            row = present_rows[local]
            vd = pshared_apply.apply_point_row_value_precomputed(
                row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                target_trace_value=batch["target_trace_value"][local])
            dirichlet[i] = vd
            if row.boundary_conditioned and normal_data_fn is not None and neumann_rows_by_row is not None:
                nrows = batch["nrows_of"][local]
                bdata = batch["boundary_data_of"][local]
                neumann[i] = np.stack(
                    [pshared_apply.apply_neumann_row_value(nr, owner_values, bd) for nr, bd in zip(nrows, bdata)],
                    axis=0)
            else:
                neumann[i] = vd
    return dirichlet, neumann


def _fix_side_neumann_exterior_fallback(lower_n, upper_n, side_exists_lower, side_exists_upper) -> None:
    """Root-cause fix (see the task report): mirror
    ``p05n_field_derived_global.core.batched_side_values``'s own wall
    -exterior convention for a face with no lower or no upper cell (the
    true outer-wall radial census row, family ``physical_wall_quartic_BC``)
    -- the *missing* side's ``neumann`` value is a copy of the *other*,
    existing side's own ``neumann`` value (``R["neumann"].copy()``/
    ``L["neumann"].copy()`` there), never the prescribed Dirichlet trace.
    :func:`_batch_side_values`'s own per-side fallback (run independently
    for the lower and upper side, so it never sees the other side's result)
    instead used that trace value for *both* its ``dirichlet`` and
    ``neumann`` slots -- correct for ``dirichlet`` (matches ``trace_v``
    there) but wrong for a Neumann-BC role at exactly this one-sided face,
    which every one of this module's P05N/P06N owner-closure checks at a
    fully wall-adjacent owner caught (~1e-4 to ~4e-2 absolute, far above the
    ~1e-10 roundoff every interior/two-sided face already met). Mutates
    ``lower_n``/``upper_n`` in place (both are the plain Python lists
    :func:`_batch_side_values` returns). A face is never exterior on both
    sides (``build_r3_side_rows``/``StructuredReconstruction.side_rows``
    guarantee at least one side exists), so the two patches below never
    touch the same face."""
    missing_lower = np.flatnonzero(~np.asarray(side_exists_lower))
    missing_upper = np.flatnonzero(~np.asarray(side_exists_upper))
    for i in missing_lower:
        lower_n[i] = upper_n[i]
    for i in missing_upper:
        upper_n[i] = lower_n[i]


# ---------------------------------------------------------------------------
# Cells units (R1 rows only -- P05 centered, P05N raw N/D/R (both
# catalogues), P06N raw material/remainder/total/R_*, P06-legacy q1
# centered/U raw prep; nothing here needs a faces/p07 chunk).
# ---------------------------------------------------------------------------
def _cells_unit_core(*, env: Environment, campaigns: tuple, oracle: dict, n: int, t, raw_ids, owner_ids, volume,
                     points, rows, neumann_rows_by_row) -> dict:
    """The cells-unit campaign arithmetic, factored out of
    :func:`compute_cells_unit` so a caller that already has ``rows``/
    ``neumann_rows_by_row`` in memory (built directly via
    ``drbx.stencils.builder.build_r1_cell_rows`` on an explicit id list, e.g.
    :mod:`p_shared.owner_closure`'s bounded owner-closure check -- see the
    task report) can run exactly this arithmetic without a chunk-file
    artifact on disk. ``compute_cells_unit`` itself is now a thin wrapper:
    load this unit's chunk, then call this function. No numerical formula
    changed by this split (verified against the pre-split output on N32)."""
    out: dict = {}

    if "p05" in campaigns:
        adapter = cf.P05Adapter(env.ref, oracle["p05"]["owner_values"])
        k_pairs = adapter.pairs
        owner_values = adapter.owner_values
        keys = adapter.wall_keys["cells"]

        batch = _batch_dirichlet_neumann(rows, adapter.dirichlet, wall_cache=env.wall_cache, trace_key=keys.trace)
        gradients = np.empty((len(raw_ids), 3, len(adapter.fields)), dtype=np.float64)
        for local, row in enumerate(rows):
            _v, g = pshared_apply.apply_point_row_precomputed(
                row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                target_trace_value=batch["target_trace_value"][local],
                target_trace_gradient=batch["target_trace_gradient"][local])
            gradients[local] = g[0]
        metric = env.ref._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jac = np.abs(metric["J"])
        raw_action, antisymmetry = pshared_apply.p05_pair_actions(h, jac, gradients, k_pairs)
        uniq, num = _sparse_scatter(raw_action, volume, owner_ids)
        out["p05_centered"] = (uniq, num)
        out["p05_antisymmetry_max"] = float(antisymmetry)

    for catalogue_name, oracle_key, table_name in (
        ("p05n_frozen", "p05n_frozen", "p05n_catalogue.json"),
        ("p05n_upwind", "p05n_upwind", "p05n_upwind_catalogue.json"),
    ):
        if catalogue_name not in campaigns:
            continue
        adapter = cf.P05NAdapter(catalogue_name, env.ref, t.g.eta_period, oracle[oracle_key]["owner_values"])
        names = adapter.names
        role_names = adapter.role_names
        role_physical_index = adapter.role_physical_index
        pair_names = adapter.pair_names
        n_pair_index = adapter.n_pair_index
        d_pair_index = adapter.d_pair_index
        r_pair_index = adapter.r_pair_index
        owner_values = adapter.owner_values
        keys = adapter.wall_keys["cells"]

        value_d = np.empty((len(raw_ids), len(names))); grad_d = np.empty((len(raw_ids), 3, len(names)))
        value_n = np.empty((len(raw_ids), len(names))); grad_n = np.empty((len(raw_ids), 3, len(names)))
        # NOTE: normal_key deliberately omitted here -- caching P05N-upwind's
        # own cells-role normal_data_fn (its 28-point Neumann boundary path)
        # reproduced P05N-frozen and every other campaign/term bitwise, but
        # shifted p05n_upwind's raw_N at N64 specifically by ~2e-16 absolute
        # (~5e-4 relative to the ~4e-13 value itself -- still far under this
        # module's own tolerances, but not bitwise); isolated by toggling
        # only one key live at a time and confirming forcing THIS key live
        # (with trace_key still cached) restored bitwise equality, while
        # forcing trace_key live (with this key still cached) did not -- see
        # the task report. trace_key is kept: it reproduced every campaign/
        # term bitwise at all three grids. Kept live per this task's own
        # rule: "if one isn't bitwise equal, keep that one live and say why."
        batch = _batch_dirichlet_neumann(rows, adapter.dirichlet, adapter.normal,
                                          neumann_rows_by_row=neumann_rows_by_row, wall_cache=env.wall_cache,
                                          trace_key=keys.trace, normal_key=keys.normal)  # keys.normal is None (deliberate)
        for local, row in enumerate(rows):
            vd, gd = pshared_apply.apply_point_row_precomputed(
                row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                target_trace_value=batch["target_trace_value"][local],
                target_trace_gradient=batch["target_trace_gradient"][local])
            value_d[local], grad_d[local] = vd[0], gd[0]
            if row.boundary_conditioned:
                nrow = batch["nrows_of"][local][0]
                g_bc = batch["boundary_data_of"][local][0]
                vn, gn = pshared_apply.apply_neumann_row(nrow, owner_values, g_bc)
                value_n[local], grad_n[local] = vn, gn
            else:
                value_n[local], grad_n[local] = value_d[local], grad_d[local]

        is_neumann = adapter.role_is_neumann
        value = np.where(is_neumann[None, :], value_n[:, role_physical_index], value_d[:, role_physical_index])
        gradient = np.where(is_neumann[None, None, :], grad_n[:, :, role_physical_index], grad_d[:, :, role_physical_index])
        metric = env.ref._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jac = np.abs(metric["J"])
        P = len(pair_names)
        action_N = np.empty((len(raw_ids), P)); action_D = np.empty((len(raw_ids), P))
        for col, (a, b) in enumerate(n_pair_index):
            action_N[:, col] = pshared_apply.p05_bracket(h, jac, gradient[:, :, a], gradient[:, :, b])
        for col, (a, b) in enumerate(d_pair_index):
            action_D[:, col] = pshared_apply.p05_bracket(h, jac, gradient[:, :, a], gradient[:, :, b])
        exact_grad = adapter.exact_gradient(points)
        action_R = np.empty((len(raw_ids), P))
        for col, (a, b) in enumerate(r_pair_index):
            action_R[:, col] = pshared_apply.p05_bracket(h, jac, exact_grad[:, :, a], exact_grad[:, :, b])

        for suffix, action in (("N", action_N), ("D", action_D), ("R", action_R)):
            uniq, num = _sparse_scatter(action, volume, owner_ids)
            out[f"{catalogue_name}_raw_{suffix}"] = (uniq, num)

    if "p06n" in campaigns or "p06_legacy" in campaigns:
        from perpendicular_structured.reference_geometry import curvature_geometry
        from p07_diffusion_global.numerics import quadrature as p07_quadrature

        prepared = curvature_geometry(env.ref, points)
        raw_keys = np.array(np.unravel_index(raw_ids, (n,) * 3)).T
        _q1_points, q1_weight = p07_quadrature(t.faces, raw_keys, 1, face=False)
        weight = q1_weight.reshape(-1)
        evolution_weight = weight * np.asarray(prepared.J) / np.maximum(np.asarray(prepared.B), 1.0e-30)
        uniq_w, denom_num = _sparse_scatter(evolution_weight[:, None], None, owner_ids)
        out["q1_evolution_volume"] = (uniq_w, denom_num[:, 0])

    if "p06n" in campaigns:
        adapter = cf.P06NAdapter(env.ref, t.g.eta_period, oracle["p06n"]["owner_values"])
        tables = adapter.tables
        variants = adapter.variant_names
        owner_values = adapter.owner_values
        keys = adapter.wall_keys["cells"]

        value_d = np.empty((len(raw_ids), len(tables.names))); grad_d = np.empty((len(raw_ids), 3, len(tables.names)))
        value_n = np.empty((len(raw_ids), len(tables.names))); grad_n = np.empty((len(raw_ids), 3, len(tables.names)))
        # NOTE: normal_key deliberately omitted here (unlike every other
        # wall_cache use in this module) -- caching P06N's own cells-role
        # normal_data_fn (its "*_phi_dirichlet"/"*_phi_neumann" role
        # variants specifically) reproduced every OTHER field/campaign
        # bitwise, but shifted P06N's raw_material/remainder/total for
        # exactly those two role variants by ~1e-16 absolute (~6e-7
        # relative to the ~3e-10 value itself -- still far under this
        # module's own tolerances, but not bitwise) -- isolated by toggling
        # only this one key live and confirming every other campaign/term
        # stayed bitwise identical either way (see the task report). Kept
        # live per this task's own rule: "if one isn't bitwise equal, keep
        # that one live and say why."
        batch = _batch_dirichlet_neumann(rows, adapter.dirichlet, adapter.normal,
                                          neumann_rows_by_row=neumann_rows_by_row, wall_cache=env.wall_cache,
                                          trace_key=keys.trace, normal_key=keys.normal)  # keys.normal is None (deliberate)
        for local, row in enumerate(rows):
            vd, gd = pshared_apply.apply_point_row_precomputed(
                row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                target_trace_value=batch["target_trace_value"][local],
                target_trace_gradient=batch["target_trace_gradient"][local])
            value_d[local], grad_d[local] = vd[0], gd[0]
            if row.boundary_conditioned:
                nrow = batch["nrows_of"][local][0]
                g_bc = batch["boundary_data_of"][local][0]
                vn, gn = pshared_apply.apply_neumann_row(nrow, owner_values, g_bc)
                value_n[local], grad_n[local] = vn, gn
            else:
                value_n[local], grad_n[local] = value_d[local], grad_d[local]

        V = len(variants)
        for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total"):
            out[f"p06n_raw_{label}"] = []
        for vi, name in enumerate(variants):
            field_index = tables.field_index[name]
            is_neumann = tables.is_neumann[name]
            value = np.where(is_neumann[None, :], value_n[:, field_index], value_d[:, field_index])
            gradient = np.where(is_neumann[None, None, :], grad_n[:, :, field_index], grad_d[:, :, field_index])
            candidate = pshared_apply.p06_q1_terms(value, gradient, prepared)
            spec = tables.variant_spec[name]
            exact_values = np.empty((len(spec), len(raw_ids))); exact_gradients = np.empty((len(spec), len(raw_ids), 3))
            for j, (field, _bc) in enumerate(spec):
                ev, eg, _ = adapter.evaluate_exact(points, field)
                exact_values[j] = ev; exact_gradients[j] = eg
            import p06_structured_global.numerics as p06numerics
            exact = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
            for label, arr in zip(("material", "remainder", "total"), candidate):
                uniq, num = _sparse_scatter(arr, evolution_weight, owner_ids)
                out[f"p06n_raw_{label}"].append((uniq, num))
            for label, arr in zip(("R_material", "R_remainder", "R_total"), exact):
                uniq, num = _sparse_scatter(arr, evolution_weight, owner_ids)
                out[f"p06n_raw_{label}"].append((uniq, num))

    if "p06_legacy" in campaigns:
        import p06_structured_global.numerics as p06numerics

        legacy = cf.P06LegacyAdapter(env.ref, oracle["p06_legacy"]["owner_values"])
        out["p06legacy_raw_centered"] = {}
        for field_name, adapter in legacy:
            owner_values = adapter.owner_values

            batch = _batch_dirichlet_neumann(rows, adapter.dirichlet, wall_cache=env.wall_cache,
                                             trace_key=adapter.wall_keys["cells"].trace)
            value = np.empty((len(raw_ids), 5)); gradient = np.empty((len(raw_ids), 3, 5))
            for local, row in enumerate(rows):
                vd, gd = pshared_apply.apply_point_row_precomputed(
                    row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                    target_trace_value=batch["target_trace_value"][local],
                    target_trace_gradient=batch["target_trace_gradient"][local])
                value[local], gradient[local] = vd[0], gd[0]
            candidate = pshared_apply.p06_q1_terms(value, gradient, prepared)
            entry = {}
            for term, arr in zip(p06numerics.TERMS, candidate):
                uniq, num = _sparse_scatter(arr, evolution_weight, owner_ids)
                entry[term] = (uniq, num)
            out["p06legacy_raw_centered"][field_name] = entry

    return out


def compute_cells_unit(unit: dict) -> dict:
    s = STATE
    env: Environment = s["env"]
    started = time.time()
    chunk = load_point_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], "cells", unit["chunk_index"])
    row_index = row_index_from_point_chunk(chunk)
    raw_ids = np.arange(unit["start"], unit["stop"], dtype=np.int64)
    t = env.t
    owner_ids = t.ro[raw_ids]
    volume = t.rv[raw_ids]
    points = t.pts[raw_ids]

    def get_r1(raw_id):
        row = row_index.get(("R1", int(raw_id)))
        if row is None:
            raise ValueError(f"cells unit {unit}: no R1 row for raw id {raw_id}")
        return row

    # Fetched once, reused by every campaign block below (cheap dict lookups
    # -- the R1 row objects themselves never change per campaign, only the
    # owner_values/trace callbacks do).
    rows = [get_r1(j) for j in raw_ids]
    # This unit's stored Neumann companion rows (one per boundary-conditioned
    # R1 row, ``quad_node`` 0 -- see ``drbx.stencils.builder.build_r1_cell_rows``),
    # read from the artifact rather than rebuilt (see the task report's
    # root-cause finding and the module docstring: rebuilding on the fly at
    # ``row.trace_target_points``, the wall-projected Dirichlet-lift anchor,
    # instead of the raw midpoint, was the bug; the artifact's own rows were
    # always built at the correct point -- ``build_r1_cell_rows``'s
    # ``bpoints = points[boundary_mask]`` -- and are read here unchanged).
    neumann_chunk = load_neumann_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], "cells",
                                                unit["chunk_index"])
    neumann_index = row_index_from_neumann_chunk(neumann_chunk)
    neumann_rows_by_row = [([neumann_index[("R1", int(j), 0)]] if row.boundary_conditioned else None)
                           for j, row in zip(raw_ids, rows)]

    out = _cells_unit_core(env=env, campaigns=s["campaigns"], oracle=s["oracle"], n=s["n"], t=t, raw_ids=raw_ids,
                           owner_ids=owner_ids, volume=volume, points=points, rows=rows,
                           neumann_rows_by_row=neumann_rows_by_row)

    parts = _pack_unit_arrays(out)
    extra = {"raw_count": len(raw_ids)}
    return runner.write_unit(s["output"], unit, s["identity_local"], chunks={"chunk": parts}, started=started, extra=extra)


def _pack_value(key: str, value, arrays: dict) -> dict:
    """Pack one (possibly nested) unit-output value under ``key``, returning
    its own manifest node. Handles ``(owners, values)`` pairs, lists of
    pairs, plain scalars/arrays, and dicts -- recursively, so a dict whose
    values are themselves dicts (e.g. ``p06legacy_raw_centered``:
    ``{field_name: {term: (uniq, num)}}``, two levels deep) round-trips
    correctly rather than being flattened as if every dict were exactly one
    level of ``{sub_key: pair}`` (the previous single-level-only version of
    this function raised ``ValueError('too many values to unpack (expected
    2)')`` on exactly this two-level case -- see the task report)."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], np.ndarray):
        arrays[f"{key}__owners"] = value[0]
        arrays[f"{key}__values"] = value[1]
        return {"kind": "pair"}
    if isinstance(value, list):
        for i, item in enumerate(value):
            owners, values = item
            arrays[f"{key}__{i}__owners"] = owners
            arrays[f"{key}__{i}__values"] = values
        return {"kind": "list", "length": len(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": list(value),
                "sub": {sub_key: _pack_value(f"{key}__{sub_key}", sub, arrays) for sub_key, sub in value.items()}}
    if isinstance(value, float):
        return {"kind": "scalar", "value": value}
    if isinstance(value, np.ndarray):
        arrays[f"{key}__array"] = value
        return {"kind": "array"}
    raise TypeError(f"unsupported unit-output value type for {key!r}: {type(value)}")


def _unpack_value(key: str, meta: dict, arrays: dict):
    kind = meta["kind"]
    if kind == "pair":
        return (arrays[f"{key}__owners"], arrays[f"{key}__values"])
    if kind == "list":
        return [(arrays[f"{key}__{i}__owners"], arrays[f"{key}__{i}__values"]) for i in range(meta["length"])]
    if kind == "dict":
        return {k: _unpack_value(f"{key}__{k}", meta["sub"][k], arrays) for k in meta["keys"]}
    if kind == "scalar":
        return meta["value"]
    if kind == "array":
        return arrays[f"{key}__array"]
    raise ValueError(f"unknown packed kind {kind!r}")


def _pack_unit_arrays(out: dict) -> dict:
    """Flatten this unit's ``(owner_ids, values)`` / nested-list-or-dict-of
    -those payload into a flat dict of plain arrays an ``np.savez`` can
    store, reversibly, via a small manifest string describing the nesting
    (mirrors ``drbx.stencils.artifact``'s own ``_json_safe``/JSON-sidecar
    idiom)."""
    arrays: dict = {}
    manifest = {key: _pack_value(key, value, arrays) for key, value in out.items()}
    arrays["__manifest_json"] = np.asarray(json.dumps(manifest))
    return arrays


def _unpack_unit_arrays(arrays: dict) -> dict:
    manifest = json.loads(str(np.asarray(arrays["__manifest_json"]).item()))
    return {key: _unpack_value(key, meta, arrays) for key, meta in manifest.items()}


# ---------------------------------------------------------------------------
# Faces units (R2 + R3 rows -- P05 live jump, P05N/P06N face terms, P06
# -legacy face correction with its seam double count).
# ---------------------------------------------------------------------------
def _faces_unit_core(*, env: Environment, campaigns: tuple, oracle: dict, n: int, t, census, row_indices, keys,
                     lower_owner_all, upper_owner_all, common_rows, lower_rows, upper_rows, common_points_by_face,
                     side_exists_lower, side_exists_upper, common_neumann_rows_by_face, side_neumann_rows_by_face,
                     h_all, weight_all, J_all, B_all, K_all) -> dict:
    """The faces-unit campaign arithmetic, factored out of
    :func:`compute_faces_unit` -- see :func:`_cells_unit_core`'s docstring
    for why (same split, same guarantee: no numerical formula changed)."""
    from drbx.stencils.census import NO_ID

    out: dict = {}

    # --- P05: live jump (P07-topology domain: valid p07_id, excluding the
    # collapsed r=0 face -- see replay.py's documented domain fix). ---
    if "p05" in campaigns:
        adapter = cf.P05Adapter(env.ref, oracle["p05"]["owner_values"])
        k_pairs = adapter.pairs
        owner_values = adapter.owner_values
        trace_fn = adapter.dirichlet
        trace_key = adapter.wall_keys["faces"].trace
        p07_ids_all = census.p07_id[row_indices]
        valid = (p07_ids_all != NO_ID) & ~census.collapsed_r0[row_indices]
        sel = np.flatnonzero(valid)

        p07_ids_local = p07_ids_all[sel].astype(np.int64)

        sel_common_rows = [common_rows[local] for local in sel]
        common_batch = _batch_dirichlet_neumann(sel_common_rows, trace_fn, wall_cache=env.wall_cache,
                                                trace_key=trace_key)
        always_true = np.ones(len(sel), dtype=bool)
        sel_lower_rows = [lower_rows[local] for local in sel]
        sel_upper_rows = [upper_rows[local] for local in sel]
        sel_common_points = [common_points_by_face[local] for local in sel]
        lower_d, _lower_n = _batch_side_values(sel_lower_rows, always_true, sel_common_points, owner_values, trace_fn,
                                               wall_cache=env.wall_cache, trace_key=trace_key)
        upper_d, _upper_n = _batch_side_values(sel_upper_rows, always_true, sel_common_points, owner_values, trace_fn,
                                               wall_cache=env.wall_cache, trace_key=trace_key)

        live_jump = np.zeros((len(sel), len(k_pairs)), dtype=np.float64)
        for out_local, local in enumerate(sel):
            axis = int(keys[local][0])
            _cv, cg = pshared_apply.apply_point_row_precomputed(
                sel_common_rows[out_local], owner_values,
                donor_trace_value=common_batch["donor_trace_value"][out_local],
                target_trace_value=common_batch["target_trace_value"][out_local],
                target_trace_gradient=common_batch["target_trace_gradient"][out_local])
            lv = lower_d[out_local]; uv = upper_d[out_local]
            h_f = h_all[local]; weight_f = weight_all[local]
            jump = pshared_apply.p05_face_jump(cg[None], lv[None], uv[None], h_f[None], weight_f[None],
                                               np.array([axis]), np.asarray(k_pairs))
            live_jump[out_local] = np.asarray(jump)[0]
        out["p05_live_jump_p07ids"] = p07_ids_local
        out["p05_live_jump_values"] = live_jump
        lower_owner_sel = lower_owner_all[sel]; upper_owner_sel = upper_owner_all[sel]
        uniq, num = _sparse_scatter_signed(live_jump, lower_owner_sel, upper_owner_sel, +1.0, -1.0)
        out["p05_live_jump_owner_num"] = (uniq, num)

    # --- P05N (frozen/upwind) faces: face_N/face_D, deduped domain (== this
    # unit's whole row_indices slice: dedupe_mask(dedupe=True) & ~collapsed_r0
    # == face_row_selection). ---
    for catalogue_name, oracle_key, table_name in (
        ("p05n_frozen", "p05n_frozen", "p05n_catalogue.json"),
        ("p05n_upwind", "p05n_upwind", "p05n_upwind_catalogue.json"),
    ):
        if catalogue_name not in campaigns:
            continue
        adapter = cf.P05NAdapter(catalogue_name, env.ref, t.g.eta_period, oracle[oracle_key]["owner_values"])
        role_physical_index = adapter.role_physical_index
        is_neumann = adapter.role_is_neumann
        action_pair_index = adapter.action_pair_index
        P = len(adapter.pair_names)
        owner_values = adapter.owner_values
        dirichlet_trace_fn = adapter.dirichlet
        normal_data_fn = adapter.normal
        keys_wall = adapter.wall_keys["faces"]

        face_N = np.zeros((len(row_indices), P)); face_D = np.zeros((len(row_indices), P))
        common_batch = _batch_dirichlet_neumann(common_rows, dirichlet_trace_fn, normal_data_fn,
                                                 neumann_rows_by_row=common_neumann_rows_by_face,
                                                 wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                                 normal_key=keys_wall.normal)
        lower_d, lower_n = _batch_side_values(lower_rows, side_exists_lower, common_points_by_face, owner_values,
                                              dirichlet_trace_fn, normal_data_fn,
                                              neumann_rows_by_row=side_neumann_rows_by_face,
                                              wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                              normal_key=keys_wall.normal)
        upper_d, upper_n = _batch_side_values(upper_rows, side_exists_upper, common_points_by_face, owner_values,
                                              dirichlet_trace_fn, normal_data_fn,
                                              neumann_rows_by_row=side_neumann_rows_by_face,
                                              wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                              normal_key=keys_wall.normal)
        _fix_side_neumann_exterior_fallback(lower_n, upper_n, side_exists_lower, side_exists_upper)
        for local in range(len(row_indices)):
            axis = int(keys[local][0])
            common_row = common_rows[local]
            _cv, cg = pshared_apply.apply_point_row_precomputed(
                common_row, owner_values, donor_trace_value=common_batch["donor_trace_value"][local],
                target_trace_value=common_batch["target_trace_value"][local],
                target_trace_gradient=common_batch["target_trace_gradient"][local])
            if common_row.boundary_conditioned:
                nrows = common_batch["nrows_of"][local]
                bdata = common_batch["boundary_data_of"][local]
                cg_n = np.stack([pshared_apply.apply_neumann_row(nr, owner_values, bd)[1]
                                for nr, bd in zip(nrows, bdata)], axis=0)
            else:
                cg_n = cg.copy()
            common_value = np.where(is_neumann[None, None, :], cg_n[:, :, role_physical_index], cg[:, :, role_physical_index])
            lower_value = np.where(is_neumann[None, :], lower_n[local][:, role_physical_index], lower_d[local][:, role_physical_index])
            upper_value = np.where(is_neumann[None, :], upper_n[local][:, role_physical_index], upper_d[local][:, role_physical_index])
            h_f = h_all[local]; weight_f = weight_all[local]
            jump = np.asarray(pshared_apply.p05_face_jump(
                common_value[None], lower_value[None], upper_value[None], h_f[None], weight_f[None],
                np.array([axis]), np.asarray(action_pair_index)))[0]
            face_N[local] = jump[:P]; face_D[local] = jump[P:]

        uniq_n, num_n = _sparse_scatter_signed(face_N, lower_owner_all, upper_owner_all, +1.0, -1.0)
        uniq_d, num_d = _sparse_scatter_signed(face_D, lower_owner_all, upper_owner_all, +1.0, -1.0)
        out[f"{catalogue_name}_face_N"] = (uniq_n, num_n)
        out[f"{catalogue_name}_face_D"] = (uniq_d, num_d)

    # --- P06N faces correction (unnormalized: divide by q1 evolution_volume
    # only in the reduction, once every cells unit has been summed). ---
    if "p06n" in campaigns:
        adapter = cf.P06NAdapter(env.ref, t.g.eta_period, oracle["p06n"]["owner_values"])
        tables = adapter.tables
        variants = adapter.variant_names
        owner_values = adapter.owner_values
        dirichlet_trace_fn = adapter.dirichlet
        normal_data_fn = adapter.normal
        keys_wall = adapter.wall_keys["faces"]

        V = len(variants)
        corr_lo_all = np.zeros((V, len(row_indices), 4)); corr_hi_all = np.zeros((V, len(row_indices), 4))
        common_batch = _batch_dirichlet_neumann(common_rows, dirichlet_trace_fn, normal_data_fn,
                                                 neumann_rows_by_row=common_neumann_rows_by_face,
                                                 wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                                 normal_key=keys_wall.normal)
        lower_d, lower_n = _batch_side_values(lower_rows, side_exists_lower, common_points_by_face, owner_values,
                                              dirichlet_trace_fn, normal_data_fn,
                                              neumann_rows_by_row=side_neumann_rows_by_face,
                                              wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                              normal_key=keys_wall.normal)
        upper_d, upper_n = _batch_side_values(upper_rows, side_exists_upper, common_points_by_face, owner_values,
                                              dirichlet_trace_fn, normal_data_fn,
                                              neumann_rows_by_row=side_neumann_rows_by_face,
                                              wall_cache=env.wall_cache, trace_key=keys_wall.trace,
                                              normal_key=keys_wall.normal)
        _fix_side_neumann_exterior_fallback(lower_n, upper_n, side_exists_lower, side_exists_upper)
        for local in range(len(row_indices)):
            axis = int(keys[local][0])
            common_row = common_rows[local]
            cv, _cg = pshared_apply.apply_point_row_precomputed(
                common_row, owner_values, donor_trace_value=common_batch["donor_trace_value"][local],
                target_trace_value=common_batch["target_trace_value"][local],
                target_trace_gradient=common_batch["target_trace_gradient"][local])
            if common_row.boundary_conditioned:
                nrows = common_batch["nrows_of"][local]
                bdata = common_batch["boundary_data_of"][local]
                cv_n = np.stack([pshared_apply.apply_neumann_row_value(nr, owner_values, bd)
                                for nr, bd in zip(nrows, bdata)], axis=0)
            else:
                cv_n = cv.copy()

            B = B_all[local]; K = K_all[local]; _J = J_all[local]
            normal_vec = _J * K[:, axis] / np.maximum(B * B, 1e-30)
            weight_f = weight_all[local]
            is_wall = axis == 0 and int(keys[local][1]) == n
            for vi, name in enumerate(variants):
                field_index = tables.field_index[name]
                is_neumann = tables.is_neumann[name]
                central_v = np.where(is_neumann[None, :], cv_n[:, field_index], cv[:, field_index])
                lower_v = np.where(is_neumann[None, :], lower_n[local][:, field_index], lower_d[local][:, field_index])
                upper_v = np.where(is_neumann[None, :], upper_n[local][:, field_index], upper_d[local][:, field_index])
                corr_lo, corr_hi, _fb = pshared_apply.p06_q3_correction(
                    central_v[:, :4], lower_v[:, :4], upper_v[:, :4], B, normal_vec, weight_f, wall=is_wall)
                corr_lo_all[vi, local] = corr_lo; corr_hi_all[vi, local] = corr_hi
        out["p06n_faces_correction"] = []
        for vi in range(V):
            uniq, num = _sparse_scatter_both_sides(corr_lo_all[vi], corr_hi_all[vi], lower_owner_all, upper_owner_all)
            out["p06n_faces_correction"].append((uniq, num))

    # --- P06-legacy face correction, with the seam double count (design
    # section 5's "seam double count"; domain-fixed per replay.py's own
    # documented change -- this unit's row_indices already excludes both
    # collapsed_r0 and legacy_alias_slots, matching the artifact's storage). ---
    if "p06_legacy" in campaigns:
        legacy = cf.P06LegacyAdapter(env.ref, oracle["p06_legacy"]["owner_values"])
        multiplier = cf.legacy_seam_multiplier(keys)
        out["p06legacy_faces_correction"] = {}
        for field_name, adapter in legacy:
            owner_values = adapter.owner_values
            trace_fn = adapter.dirichlet

            corr_lo = np.zeros((len(row_indices), 4)); corr_hi = np.zeros((len(row_indices), 4))
            trace_key = adapter.wall_keys["faces"].trace
            common_batch = _batch_dirichlet_neumann(common_rows, trace_fn, wall_cache=env.wall_cache,
                                                    trace_key=trace_key)
            always_true = np.ones(len(row_indices), dtype=bool)
            lower_v_all, _ = _batch_side_values(lower_rows, always_true & side_exists_lower, common_points_by_face,
                                                owner_values, trace_fn, wall_cache=env.wall_cache, trace_key=trace_key)
            upper_v_all, _ = _batch_side_values(upper_rows, always_true & side_exists_upper, common_points_by_face,
                                                owner_values, trace_fn, wall_cache=env.wall_cache, trace_key=trace_key)
            for local in range(len(row_indices)):
                axis = int(keys[local][0])
                cv, _cg = pshared_apply.apply_point_row_precomputed(
                    common_rows[local], owner_values, donor_trace_value=common_batch["donor_trace_value"][local],
                    target_trace_value=common_batch["target_trace_value"][local],
                    target_trace_gradient=common_batch["target_trace_gradient"][local])
                lower_v = lower_v_all[local]; upper_v = upper_v_all[local]
                B = B_all[local]; K = K_all[local]; _J = J_all[local]
                normal_vec = _J * K[:, axis] / np.maximum(B * B, 1e-30)
                weight_f = weight_all[local]
                is_wall = axis == 0 and int(keys[local][1]) == n
                corr_lo_f, corr_hi_f, _fb = pshared_apply.p06_q3_correction(
                    cv[:, :4], lower_v[:, :4], upper_v[:, :4], B, normal_vec, weight_f, wall=is_wall)
                corr_lo[local] = corr_lo_f; corr_hi[local] = corr_hi_f
            corr_lo *= multiplier[:, None]; corr_hi *= multiplier[:, None]
            uniq, num = _sparse_scatter_both_sides(corr_lo, corr_hi, lower_owner_all, upper_owner_all)
            out["p06legacy_faces_correction"][field_name] = (uniq, num)

    return out


def compute_faces_unit(unit: dict) -> dict:
    from drbx.stencils.census import NO_ID

    s = STATE
    env: Environment = s["env"]
    t = env.t
    census = env.census
    started = time.time()
    chunk = load_point_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], "faces", unit["chunk_index"])
    row_index = row_index_from_point_chunk(chunk)
    row_indices = s["face_row_indices"][unit["start"]:unit["stop"]]
    keys = census.keys()[row_indices]
    lower_owner_all = census.owner_lo[row_indices]
    upper_owner_all = census.owner_hi[row_indices]

    def get_r2(ridx):
        row = row_index.get(("R2", int(ridx)))
        if row is None:
            raise ValueError(f"faces unit {unit}: no R2 row for census row {ridx}")
        return row

    def get_r3(ridx, side):
        return row_index.get(("R3", int(ridx) * 2 + side))

    def side_exists(key, axis, side):
        ijk = [int(v) for v in key[1:]]
        probe = ijk.copy(); probe[axis] += -1 if side == 0 else 0
        if axis == 0 and side == 0 and probe[0] < 0:
            return False
        if axis == 0 and side == 1 and ijk[0] >= s["n"]:
            return False
        return True

    # Fetched/derived once, reused by every campaign block below (see the
    # module docstring's batching rationale -- these are cheap dict lookups
    # /small per-face bookkeeping; only the trace/normal/metric evaluations
    # inside each campaign block need batching).
    common_rows = [get_r2(int(ridx)) for ridx in row_indices]
    lower_rows = [get_r3(int(ridx), 0) for ridx in row_indices]
    upper_rows = [get_r3(int(ridx), 1) for ridx in row_indices]
    axis_all = keys[:, 0].astype(np.int64)
    side_exists_lower = np.array([side_exists(keys[i], int(axis_all[i]), 0) for i in range(len(row_indices))])
    side_exists_upper = np.array([side_exists(keys[i], int(axis_all[i]), 1) for i in range(len(row_indices))])
    common_points_by_face = [r.trace_target_points for r in common_rows]
    common_counts = [len(p) for p in common_points_by_face]

    # This unit's stored Neumann companion rows (read from the artifact, never
    # rebuilt -- see the task report and compute_cells_unit's comment above).
    # R3's Neumann row is tagged once per face, at the same census row index
    # as its R2 common row (request "R3", not side-doubled -- see
    # ``drbx.stencils.builder.build_r3_side_rows``'s docstring), so one
    # per-face lookup serves the common row and both of its sides.
    neumann_chunk = load_neumann_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], "faces",
                                                unit["chunk_index"])
    neumann_index = row_index_from_neumann_chunk(neumann_chunk)

    def _face_neumann_rows(request, ridx, q_count):
        got = [neumann_index.get((request, int(ridx), q)) for q in range(q_count)]
        if all(g is None for g in got):
            return None
        if any(g is None for g in got):
            raise ValueError(f"partial stored Neumann rows for {request} entity {ridx}")
        return got

    common_neumann_rows_by_face = [_face_neumann_rows("R2", ridx, len(common_points_by_face[local]))
                                   for local, ridx in enumerate(row_indices)]
    side_neumann_rows_by_face = [_face_neumann_rows("R3", ridx, len(common_points_by_face[local]))
                                 for local, ridx in enumerate(row_indices)]

    # Face geometry/quadrature: every campaign block below needs `h_f`/
    # `weight_f` (P05/P05N/P06N) and/or `J, B, K` (P06N/P06-legacy) at every
    # face's own 9 q3 nodes. Read directly from the artifact's own
    # `geometry.npz` (this unit's own contiguous slice of `GeometryArrays`'
    # `face_points`/`p05_face_h`/`p06_face_J`/`p06_face_B`/`p06_face_K`/
    # `p06_face_weight` -- `s["geometry"]`'s row order is exactly
    # `s["face_row_indices"]`, the same array `row_indices` slices, so
    # `unit["start"]:unit["stop"]` is the same slice either way) instead of
    # recomputing it live every unit (`env.ref._metric`/`_face_geometry`,
    # this stage's own previously-dominant-looking cost, before the wall
    # -data cache below cut the true dominant cost -- see the task report).
    # Checked field-for-field bitwise equal to a fresh live recomputation at
    # this exact row range before this change (task report); the ragged
    # (`common_counts` not uniformly 9) case defensively falls back to a
    # live call exactly as before -- never expected in practice (R2's own
    # node count is always the census's fixed q3 quadrature, 9), but this
    # keeps that branch's old behavior unchanged rather than assuming it is
    # dead code.
    F = len(row_indices)
    ragged = F and (len(set(common_counts)) != 1 or common_counts[0] != 9)
    if F == 0:
        _q_points_all, weight_all = np.zeros((0, 9, 3)), np.zeros((0, 9))
    elif ragged:
        _q_points_all, weight_all = pshared_provider._quadrature(env.t.faces, keys, 3, face=True)
    else:
        geometry = s["geometry"]
        sl = slice(unit["start"], unit["stop"])
        _q_points_all = geometry.face_points[sl]
        weight_all = geometry.p06_face_weight[sl]
    # Root-cause fix (see the task report): face geometry (h/J/B/K/metric
    # below) and any Neumann companion-row rebuild must be evaluated at the
    # face's own q3 quadrature node -- never at `common_points_by_face`
    # (`PointRows.trace_target_points`), which is the wall-projected (u=1)
    # anchor `PointRows.apply`'s Dirichlet lift uses for its own boundary
    # correction and only coincides with the true node for an *unconditioned*
    # row (see `fci_perpendicular_reconstruction.py`'s `boundary_map` branch).
    common_neumann_points_by_face = [_q_points_all[i] for i in range(F)]
    common_points_flat = (_q_points_all.reshape(-1, 3) if F else np.zeros((0, 3)))
    if F == 0:
        h_all = np.zeros((0, 9, 3))
    elif ragged:
        metric_all = env.ref._metric(common_points_flat)
        h_flat = metric_all["bcov"] / metric_all["B"][:, None]
        h_all = np.split(h_flat, np.cumsum(common_counts)[:-1])
    else:
        h_all = s["geometry"].p05_face_h[slice(unit["start"], unit["stop"])]

    J_all = B_all = K_all = None
    if F and ("p06n" in s["campaigns"] or "p06_legacy" in s["campaigns"]):
        if ragged:
            import p06_structured_global.numerics as _p06numerics_geom

            J_flat, B_flat, K_flat = _p06numerics_geom._face_geometry(env.ref, common_points_flat)
            splits = np.cumsum(common_counts)[:-1]
            J_all = np.split(J_flat, splits); B_all = np.split(B_flat, splits); K_all = np.split(K_flat, splits)
        else:
            sl = slice(unit["start"], unit["stop"])
            geometry = s["geometry"]
            J_all = geometry.p06_face_J[sl]; B_all = geometry.p06_face_B[sl]; K_all = geometry.p06_face_K[sl]

    out = _faces_unit_core(env=env, campaigns=s["campaigns"], oracle=s["oracle"], n=s["n"], t=t, census=census,
                           row_indices=row_indices, keys=keys, lower_owner_all=lower_owner_all,
                           upper_owner_all=upper_owner_all, common_rows=common_rows, lower_rows=lower_rows,
                           upper_rows=upper_rows, common_points_by_face=common_points_by_face,
                           side_exists_lower=side_exists_lower, side_exists_upper=side_exists_upper,
                           common_neumann_rows_by_face=common_neumann_rows_by_face,
                           side_neumann_rows_by_face=side_neumann_rows_by_face, h_all=h_all, weight_all=weight_all,
                           J_all=J_all, B_all=B_all, K_all=K_all)

    parts = _pack_unit_arrays(out)
    extra = {"row_count": len(row_indices)}
    return runner.write_unit(s["output"], unit, s["identity_local"], chunks={"chunk": parts}, started=started, extra=extra)


# ---------------------------------------------------------------------------
# P07 units (R4 integrated rows only -- P07/P07N global N/D/O_q3). A
# conditioned-family (1/2/4) row uses its OWN ``trace_target_points`` (never
# a separate R2 lookup -- see module docstring), so a p07 unit never opens a
# faces chunk.
# ---------------------------------------------------------------------------
def _p07_family_flux_unit(env: Environment, row_index: dict, neumann_index: dict, row_indices, keys, family,
                          p07_ids, owner_values, trace_fn, normal_data_fn, *, radial_degree_by_family: dict,
                          need_D: bool, dirichlet_fields: tuple = (), trace_key: str | None = None,
                          normal_key: str | None = None):
    """Batched (see the module docstring's batching rationale): every
    ``trace_fn``/``normal_data_fn``/``env.ref._perpendicular_flux_tensor``/
    quadrature call happens once per unit, never once per face row --
    previously the single largest remaining per-row cost in this stage.
    The conditioned-family (1/2/4) Neumann row is read from ``neumann_index``
    (the artifact's own stored ``R4`` rows -- see the task report), never
    rebuilt on the fly.

    ``dirichlet_fields`` names the ``owner_values`` field columns that use
    the prescribed Dirichlet trace (the D lift, :func:`apply_integrated_row
    _precomputed`) at a conditioned (family 1/2/4) face, instead of the
    physical-normal Neumann restoration (:func:`p07_neumann_face_flux`) every
    other field uses there. This is a per-*field*, not per-family, choice:
    the frozen ``p07_diffusion_global.numerics.face_chunk`` (see the task
    report) builds every wall-reaching face's flux from a *field-dependent*
    boundary reconstruction -- field 0 (``phi_mms``) from a ``'value'``
    (Dirichlet) ``BoundaryRelation``, fields 1-3 (``Ti_mms``,
    ``regular_neumann``, ``mixed_eta_neumann``) from a ``'normal_derivative'``
    (Neumann) one -- never the uniform Neumann restoration this unit applied
    to every field before the fix. Default ``()`` (every field Neumann-
    restored) reproduces the P07N campaign, which has no Dirichlet-BC field."""
    n_fields = owner_values.shape[1]
    F = len(row_indices)
    flux_N = np.zeros((F, n_fields))
    store_D = need_D or bool(dirichlet_fields)
    flux_D = np.zeros((F, n_fields)) if store_D else None

    rows = []
    for local in range(F):
        p07_id = int(p07_ids[local])
        row = row_index.get(p07_id)
        if row is None:
            raise ValueError(f"p07 unit: no R4 row for p07 id {p07_id}")
        rows.append(row)

    fam_arr = family.astype(np.int64)
    is_conditioned = np.array([int(f) in radial_degree_by_family for f in fam_arr])

    # --- Regular-family (or need_D) rows: apply_integrated_row, batched. ---
    need_regular = np.flatnonzero(store_D | ~is_conditioned)
    if need_regular.size:
        regular_rows = [rows[i] for i in need_regular]
        batch = _batch_dirichlet_neumann(regular_rows, trace_fn, wall_cache=env.wall_cache, trace_key=trace_key)
        for local, i in enumerate(need_regular):
            row = regular_rows[local]
            v_d = pshared_apply.apply_integrated_row_precomputed(
                row, owner_values, donor_trace_value=batch["donor_trace_value"][local],
                target_trace_gradient=batch["target_trace_gradient"][local])
            if store_D:
                flux_D[i] = v_d
            if not is_conditioned[i]:
                flux_N[i] = v_d

    # --- Conditioned-family (1/2/4) rows: Neumann-restored N flux, batched
    # (grouped by radial degree -- family 1/2 use 4, family 4 uses 3). ---
    cond_idx = np.flatnonzero(is_conditioned)
    if cond_idx.size:
        cond_keys = keys[cond_idx]
        # Root-cause fix (see the task report): the conditioned-family flux
        # tensor and its Neumann companion row must both be evaluated at this
        # face's own q3 quadrature node -- never at `rows[i].trace_target_points`
        # (the wall-projected (u=1) anchor `PointRows`/`IntegratedFaceRow`'s
        # Dirichlet-lift boundary correction uses, which only coincides with
        # the true node for an *unconditioned* row). `O_q3` just below already
        # gets this right by using the quadrature points directly.
        cond_points_q, weight_q = pshared_provider._quadrature(env.t.faces, cond_keys, 3, face=True)
        cond_points_flat = cond_points_q.reshape(-1, 3)
        face_tensor = env.ref._perpendicular_flux_tensor(cond_points_flat).reshape(len(cond_idx), 9, 3, 3)
        integrand_all = contract_face_tensor(weight_q, face_tensor, cond_keys[:, 0])
        nrows_by_local = []
        for local_c, i in enumerate(cond_idx):
            p07_id = int(p07_ids[i])
            count = len(cond_points_q[local_c])
            got = [neumann_index.get(("R4", p07_id, q)) for q in range(count)]
            if any(g is None for g in got):
                raise ValueError(f"no stored R4 Neumann row for p07 id {p07_id}")
            nrows_by_local.append(got)
        flat_nrows = [nr for got in nrows_by_local for nr in got]
        boundary_groups = [nr.boundary_points for nr in flat_nrows]
        normal_fn = normal_data_fn
        if normal_key is not None:
            def normal_fn(q, _wall_cache=env.wall_cache, _key=normal_key, _fn=normal_data_fn):
                return _wall_cache.lookup_or_compute(_key, _fn, q)
        boundary_results = pshared_apply.batch_evaluate(normal_fn, boundary_groups)
        offset = 0
        for local_c, i in enumerate(cond_idx):
            c = len(nrows_by_local[local_c])
            nrows_i = nrows_by_local[local_c]
            boundary_data_i = boundary_results[offset:offset + c]
            integrand_i = integrand_all[local_c]
            flux_N[i] = pshared_apply.p07_neumann_face_flux(nrows_i, owner_values, boundary_data_i, integrand_i)
            offset += c
        if dirichlet_fields:
            # This face's own field-dependent boundary reconstruction (see
            # this function's docstring and the task report): the named
            # fields keep the Dirichlet D lift already computed above (in
            # `need_regular`, forced to include every conditioned row via
            # `store_D`) instead of the Neumann restoration just written.
            dirichlet_cols = np.asarray(dirichlet_fields, dtype=np.int64)
            rows_idx = cond_idx[:, None]
            flux_N[rows_idx, dirichlet_cols[None, :]] = flux_D[rows_idx, dirichlet_cols[None, :]]
    return (flux_D if need_D else None), flux_N


def compute_p07_unit(unit: dict) -> dict:
    s = STATE
    env: Environment = s["env"]
    census = env.census
    started = time.time()
    chunk = load_integrated_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], unit["chunk_index"])
    row_index = row_index_from_integrated_chunk(chunk)
    neumann_chunk = load_neumann_chunk_for_unit(s["artifact_root"], s["n"], s["manifest"], "p07", unit["chunk_index"])
    neumann_index = row_index_from_neumann_chunk(neumann_chunk)
    row_indices = s["p07_row_indices"][unit["start"]:unit["stop"]]
    p07_ids = census.p07_id[row_indices]
    family = census.family[row_indices]
    keys = census.keys()[row_indices]
    lower_owner = census.owner_lo[row_indices]
    upper_owner = census.owner_hi[row_indices]

    out = _p07_unit_core(env=env, campaigns=s["campaigns"], oracle=s["oracle"], row_index=row_index,
                         neumann_index=neumann_index, row_indices=row_indices, keys=keys, family=family,
                         p07_ids=p07_ids, lower_owner=lower_owner, upper_owner=upper_owner)

    parts = _pack_unit_arrays(out)
    extra = {"row_count": len(row_indices)}
    return runner.write_unit(s["output"], unit, s["identity_local"], chunks={"chunk": parts}, started=started, extra=extra)


# ---------------------------------------------------------------------------
# The p07-unit campaign arithmetic (see _cells_unit_core's docstring for why
# this is a separate function).
# ---------------------------------------------------------------------------
def _p07_unit_core(*, env: Environment, campaigns: tuple, oracle: dict, row_index: dict, neumann_index: dict,
                   row_indices, keys, family, p07_ids, lower_owner, upper_owner) -> dict:
    """The p07-unit campaign arithmetic, factored out of
    :func:`compute_p07_unit` -- see :func:`_cells_unit_core`'s docstring for
    why (same split, same guarantee: no numerical formula changed)."""
    out: dict = {}

    if "p07" in campaigns:
        adapter = cf.P07Adapter(env.ref, oracle["p07"]["owner_values"])
        owner_values = adapter.owner_values
        keys_wall = adapter.wall_keys["p07"]
        # Root cause of the plain-P07 wall mismatch (see the task report): plain P07's frozen
        # ``face_chunk`` builds a wall-reaching face's flux from its own field-dependent boundary
        # reconstruction, matched by the row's Dirichlet D lift for *every* field at a conditioned
        # (family 1/2/4) face (replay fix 2) -- ``adapter.dirichlet_fields`` lists every field. The
        # trace's (Q, fields, 3) gradient is swapped to (Q, 3, fields) at the source, in the adapter.
        p07_dirichlet_fields = adapter.dirichlet_fields
        trace_fn = adapter.dirichlet
        normal_data_fn = adapter.normal

        _d, flux_N = _p07_family_flux_unit(env, row_index, neumann_index, row_indices, keys, family, p07_ids,
                                           owner_values, trace_fn, normal_data_fn,
                                           radial_degree_by_family=adapter.radial_degree_by_family,
                                           need_D=adapter.need_D, dirichlet_fields=p07_dirichlet_fields,
                                           trace_key=keys_wall.trace, normal_key=keys_wall.normal)
        uniq, num = _sparse_scatter_signed(flux_N, lower_owner, upper_owner, -1.0, +1.0)
        out["p07_global_N"] = (uniq, num)

    if "p07n" in campaigns:
        adapter = cf.P07NAdapter(env.ref, env.t.g.eta_period, oracle["p07n"]["owner_values"])
        p07n_names = adapter.names
        owner_values = adapter.owner_values
        keys_wall = adapter.wall_keys["p07"]
        trace_fn = adapter.dirichlet
        normal_data_fn = adapter.normal

        flux_D, flux_N = _p07_family_flux_unit(env, row_index, neumann_index, row_indices, keys, family, p07_ids,
                                               owner_values, trace_fn, normal_data_fn,
                                               radial_degree_by_family=adapter.radial_degree_by_family,
                                               need_D=adapter.need_D,
                                               trace_key=keys_wall.trace, normal_key=keys_wall.normal)
        uniq_n, num_n = _sparse_scatter_signed(flux_N, lower_owner, upper_owner, -1.0, +1.0)
        uniq_d, num_d = _sparse_scatter_signed(flux_D, lower_owner, upper_owner, -1.0, +1.0)
        out["p07n_global_N"] = (uniq_n, num_n)
        out["p07n_global_D"] = (uniq_d, num_d)

        # O_q3: exact analytic gradient contracted against the same q3
        # weighted-tensor integrand -- no row/owner_values dependency at all
        # (mirrors replay.py's _replay_p07n_o_q3, restricted to this unit's
        # row slice). Family 0 (collapsed r=0) always carries a zero
        # integrand, matching build_artifact.py's own convention. Root
        # -cause fix (see the task report): the collapsed r=0 face sits at
        # u == 0, where the frozen reference's ordinary (non-regularized)
        # metric is singular by construction (`MetricEvaluator.evaluate`
        # raises `ValueError("...singular at u=0...")`, never NaN/inf), so
        # `_perpendicular_flux_tensor`/`p07n_fields.evaluate` must never be
        # called at a family-0 row's own points at all -- zeroing the
        # *output* afterward (this block's previous behavior) still
        # evaluated the singular metric first and crashed on any p07 unit
        # that actually contains a family-0 row (only the grid's very first
        # unit, in this census's row order -- never exercised by this
        # module's own bounded validation, which always sampled tail
        # units). Excluding those rows from the live call, rather than
        # masking the result, changes nothing for every other family (same
        # function, same points, same values) and never queries the
        # singular point.
        non_collapsed = np.flatnonzero(family != 0)
        face_O = np.zeros((len(keys), len(p07n_names)))
        if non_collapsed.size:
            sel_keys = keys[non_collapsed]
            points, weight = pshared_provider._quadrature(env.t.faces, sel_keys, 3, face=True)
            tensor = env.ref._perpendicular_flux_tensor(points.reshape(-1, 3)).reshape(len(non_collapsed), 9, 3, 3)
            integrand = contract_face_tensor(weight, tensor, sel_keys[:, 0])
            flat = points.reshape(-1, 3)
            grads = adapter.exact_gradients(flat)
            grads = grads.reshape(len(non_collapsed), 9, len(p07n_names), 3)
            face_O[non_collapsed] = np.einsum("fqa,fqka->fk", integrand, grads)
        uniq_o, num_o = _sparse_scatter_signed(face_O, lower_owner, upper_owner, -1.0, +1.0)
        out["p07n_global_O_q3"] = (uniq_o, num_o)

    return out


# ---------------------------------------------------------------------------
# Reduction: assemble every unit's sparse partial into the same per-campaign
# "terms" dict shape replay.py's whole-grid functions produce, then reuse
# replay.py's own compare_owner_term/compare_pointwise_only/write_report
# unchanged.
# ---------------------------------------------------------------------------
def _read_unit_output(output: Path, unit: dict) -> dict:
    path = runner.unit_path(output, unit, "chunk")
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    return _unpack_unit_arrays(arrays)


def _unit_peak_rss(output: Path, units: list) -> float:
    peak = 0.0
    for unit in units:
        path = runner.receipt_path(output, unit)
        if path.exists():
            peak = max(peak, float(json.loads(path.read_text()).get("peak_rss_gib", 0.0)))
    return peak


def reduce_grid(*, output: Path, artifact_root: Path, n: int, input_root: Path, sidecar_path: Path,
                paths: dict, campaigns: tuple, plan: dict) -> dict:
    started = time.time()
    output = Path(output)
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path))
    t = env.t
    owners = len(t.vol)
    results: dict = {}
    peak_rss: dict = {}

    cells_units = plan["cells"]; faces_units = plan["faces"]; p07_units = plan["p07"]
    peak_rss["cells"] = _unit_peak_rss(output, cells_units)
    peak_rss["faces"] = _unit_peak_rss(output, faces_units)
    peak_rss["p07"] = _unit_peak_rss(output, p07_units)

    # --- Cells-only accumulation. ---
    p05_centered_acc = OwnerAccumulator(owners, (8,)) if "p05" in campaigns else None
    p05_antisym_max = 0.0
    p05n_acc = {name: {suf: OwnerAccumulator(owners, (0,)) for suf in ("N", "D", "R")}
               for name in ("p05n_frozen", "p05n_upwind") if name in campaigns}
    q1_evolution_volume = OwnerAccumulator(owners) if ("p06n" in campaigns or "p06_legacy" in campaigns) else None
    p06n_raw_acc = None
    p06legacy_raw_acc = None

    for unit in cells_units:
        data = _read_unit_output(output, unit)
        if p05_centered_acc is not None and "p05_centered" in data:
            uniq, values = data["p05_centered"]
            if p05_centered_acc.total.shape[1] != values.shape[1]:
                p05_centered_acc = OwnerAccumulator(owners, values.shape[1:])
            p05_centered_acc.add(uniq, values)
            p05_antisym_max = max(p05_antisym_max, float(data.get("p05_antisymmetry_max", 0.0)))
        for name in list(p05n_acc):
            for suf in ("N", "D", "R"):
                key = f"{name}_raw_{suf}"
                if key in data:
                    uniq, values = data[key]
                    acc = p05n_acc[name][suf]
                    if acc.total.shape[1:] != values.shape[1:]:
                        acc = OwnerAccumulator(owners, values.shape[1:]); p05n_acc[name][suf] = acc
                    acc.add(uniq, values)
        if q1_evolution_volume is not None and "q1_evolution_volume" in data:
            uniq, values = data["q1_evolution_volume"]
            q1_evolution_volume.add(uniq, values)
        if "p06n" in campaigns and "p06n_raw_material" in data:
            if p06n_raw_acc is None:
                V = len(data["p06n_raw_material"])
                p06n_raw_acc = {label: [OwnerAccumulator(owners, (4,)) for _ in range(V)]
                               for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total")}
            for label in p06n_raw_acc:
                for vi, (uniq, values) in enumerate(data[f"p06n_raw_{label}"]):
                    p06n_raw_acc[label][vi].add(uniq, values)
        if "p06_legacy" in campaigns and "p06legacy_raw_centered" in data:
            if p06legacy_raw_acc is None:
                field_names = list(data["p06legacy_raw_centered"])
                p06legacy_raw_acc = {f: {term: OwnerAccumulator(owners, (4,)) for term in ("material", "remainder", "total")}
                                     for f in field_names}
            for field_name, terms_map in data["p06legacy_raw_centered"].items():
                for term, (uniq, values) in terms_map.items():
                    p06legacy_raw_acc[field_name][term].add(uniq, values)

    evolution_volume = (q1_evolution_volume.total if q1_evolution_volume is not None else None)
    evolution_volume_safe = (np.maximum(evolution_volume, 1e-300) if evolution_volume is not None else None)

    # --- Faces accumulation. ---
    p05_owner_num = OwnerAccumulator(owners, (8,)) if "p05" in campaigns else None
    p05_p07id_values: dict = {}
    p05n_face_acc = {name: {suf: None for suf in ("N", "D")} for name in ("p05n_frozen", "p05n_upwind") if name in campaigns}
    p06n_correction_acc = None
    p06legacy_correction_acc = None

    for unit in faces_units:
        data = _read_unit_output(output, unit)
        if p05_owner_num is not None and "p05_live_jump_owner_num" in data:
            uniq, values = data["p05_live_jump_owner_num"]
            if p05_owner_num.total.shape[1] != values.shape[1]:
                p05_owner_num = OwnerAccumulator(owners, values.shape[1:])
            p05_owner_num.add(uniq, values)
            for pid, val in zip(np.asarray(data["p05_live_jump_p07ids"]).tolist(), np.asarray(data["p05_live_jump_values"])):
                p05_p07id_values[int(pid)] = val
        for name in list(p05n_face_acc):
            for suf in ("N", "D"):
                key = f"{name}_face_{suf}"
                if key in data:
                    uniq, values = data[key]
                    acc = p05n_face_acc[name][suf]
                    if acc is None or acc.total.shape[1:] != values.shape[1:]:
                        acc = OwnerAccumulator(owners, values.shape[1:]); p05n_face_acc[name][suf] = acc
                    acc.add(uniq, values)
        if "p06n" in campaigns and "p06n_faces_correction" in data:
            if p06n_correction_acc is None:
                V = len(data["p06n_faces_correction"])
                p06n_correction_acc = [OwnerAccumulator(owners, (4,)) for _ in range(V)]
            for vi, (uniq, values) in enumerate(data["p06n_faces_correction"]):
                p06n_correction_acc[vi].add(uniq, values)
        if "p06_legacy" in campaigns and "p06legacy_faces_correction" in data:
            if p06legacy_correction_acc is None:
                field_names = list(data["p06legacy_faces_correction"])
                p06legacy_correction_acc = {f: OwnerAccumulator(owners, (4,)) for f in field_names}
            for field_name, (uniq, values) in data["p06legacy_faces_correction"].items():
                p06legacy_correction_acc[field_name].add(uniq, values)

    # --- P07 accumulation. ---
    p07_num = OwnerAccumulator(owners, (0,)) if "p07" in campaigns else None
    p07n_num = {"N": None, "D": None, "O_q3": None} if "p07n" in campaigns else {}
    for unit in p07_units:
        data = _read_unit_output(output, unit)
        if p07_num is not None and "p07_global_N" in data:
            uniq, values = data["p07_global_N"]
            if p07_num.total.shape[1] != values.shape[1]:
                p07_num = OwnerAccumulator(owners, values.shape[1:])
            p07_num.add(uniq, values)
        for label, key in (("N", "p07n_global_N"), ("D", "p07n_global_D"), ("O_q3", "p07n_global_O_q3")):
            if key in data:
                uniq, values = data[key]
                acc = p07n_num.get(label)
                if acc is None or acc.total.shape[1:] != values.shape[1:]:
                    acc = OwnerAccumulator(owners, values.shape[1:]); p07n_num[label] = acc
                p07n_num[label] = acc
                acc.add(uniq, values)

    # =========================================================================
    # Finalize + compare, per campaign (same math, same tolerance functions,
    # as replay.py's whole-grid functions).
    # =========================================================================
    if "p05" in campaigns:
        from p05_direct_midpoint_global.direct_operator import regional_owner_masks
        import p05_structured_global.numerics as k

        with np.load(paths["p05"] / f"N{n}.owner_results.npz", allow_pickle=False) as z:
            saved_centered = z["centered"].copy(); saved_reference = z["reference"].copy()
            owner_volume = z["raw_owner_volume"].copy()
        with np.load(paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as z:
            saved_old_u_minus_a = z["old_U_minus_A"].copy()
        replay_centered = p05_centered_acc.total / t.vol[:, None]
        ijk = np.array(np.unravel_index(np.arange(n ** 3), (n, n, n))).T
        masks, _info = regional_owner_masks(t.ro, ijk, len(t.vol), n)
        archived_error = saved_centered - saved_reference
        terms = {"centered": compare_owner_term("p05.centered", replay_centered, saved_centered,
                                                owner_volume=owner_volume, archived_error=archived_error,
                                                region_masks=masks)}
        saved_upwind = _load_p05_upwind(paths["p05_upwind_chunks"], n)
        max_pid = max(p05_p07id_values) if p05_p07id_values else -1
        dense_replay = np.full((max_pid + 1, saved_upwind.shape[1]), np.nan)
        for pid, val in p05_p07id_values.items():
            dense_replay[pid] = val
        limit = min(dense_replay.shape[0], saved_upwind.shape[0])
        populated = np.flatnonzero(np.all(np.isfinite(dense_replay[:limit]), axis=1)
                                   & np.all(np.isfinite(saved_upwind[:limit]), axis=1))
        terms["live_jump_vs_upwind"] = compare_pointwise_only(
            "p05.live_jump_vs_upwind", dense_replay[:limit][populated], saved_upwind[:limit][populated])
        owner_live_jump = p05_owner_num.total / t.vol[:, None]
        terms["live_jump_vs_old_U_minus_A"] = compare_owner_term(
            "p05.live_jump_vs_old_U_minus_A", owner_live_jump, saved_old_u_minus_a,
            owner_volume=owner_volume, archived_error=archived_error, region_masks=masks,
            # U - A cancels terms of size |centered|; roundoff is set by those, not by the result.
            cap_reference=np.max(np.abs(saved_centered), axis=0, keepdims=True))
        results["p05"] = {"campaign": "p05", "status": "ok", "terms": terms, "antisymmetry_max": p05_antisym_max}

    import p06n_field_derived_global.core as p06n_core
    region_masks_p06n = p06n_core.regional_masks(t)

    for name, root_key in (
        ("p05n_frozen", "p05n_frozen"),
        ("p05n_upwind", "p05n_p06n_upwind"),
    ):
        if name not in campaigns:
            continue
        root = paths[root_key] if name == "p05n_frozen" else paths[root_key] / "p05n_upwind"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved_raw_N, saved_raw_D, saved_raw_R = z["N"].copy(), z["D"].copy(), z["R"].copy()
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_face_N, saved_face_D = z["N"].copy(), z["D"].copy()
        replay_raw = {suf: p05n_acc[name][suf].total / t.vol[:, None] for suf in ("N", "D", "R")}
        archived_error_raw = saved_raw_N - saved_raw_R
        terms = {
            f"raw_{suf}": compare_owner_term(f"p05n.raw_{suf}", replay_raw[suf], saved,
                                             owner_volume=t.vol, archived_error=archived_error_raw,
                                             region_masks=region_masks_p06n)
            for suf, saved in (("N", saved_raw_N), ("D", saved_raw_D), ("R", saved_raw_R))
        }
        replay_face_N = p05n_face_acc[name]["N"].total / t.vol[:, None]
        replay_face_D = p05n_face_acc[name]["D"].total / t.vol[:, None]
        terms["face_N"] = compare_owner_term("p05n.face_N", replay_face_N, saved_face_N, owner_volume=t.vol,
                                             archived_error=archived_error_raw, region_masks=region_masks_p06n)
        terms["face_D"] = compare_owner_term("p05n.face_D", replay_face_D, saved_face_D, owner_volume=t.vol,
                                             archived_error=archived_error_raw, region_masks=region_masks_p06n)
        results[name] = {"campaign": f"p05n[{'p05n_catalogue.json' if name == 'p05n_frozen' else 'p05n_upwind_catalogue.json'}]",
                         "status": "ok", "terms": terms}

    if "p06n" in campaigns:
        import p06n_field_derived_global.core as p06n_core

        tables = p06n_core.CATALOGUE_TABLES
        variants = tables.variant_names
        root = paths["p05n_p06n_upwind"] / "p06n"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved = {label: z[label].copy() for label in
                    ("material", "remainder", "total", "R_material", "R_remainder", "R_total")}
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_correction = z["correction"].copy()
        V = len(variants)
        replay = {label: np.stack([p06n_raw_acc[label][vi].total / evolution_volume_safe[:, None] for vi in range(V)])
                 for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total")}
        archived_error = saved["total"] - saved["R_total"]
        results_terms = {}
        for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total"):
            per_variant = []
            for vi, name_ in enumerate(variants):
                per_variant.append(compare_owner_term(f"p06n.raw_{label}[{name_}]", replay[label][vi], saved[label][vi],
                                                       owner_volume=t.vol, archived_error=archived_error[vi],
                                                       region_masks=region_masks_p06n, cap_mode="flat", cap_flat_abs=1e-9))
            results_terms[f"raw_{label}"] = _summarize_variants(per_variant)
        # Root-cause fix (see the task report): `p06n_field_derived_global
        # .campaign`'s own ``reduce_faces`` saves ``correction`` as the raw,
        # un-divided owner accumulation (``np.add.at(correction[vi], lo/hi,
        # ...)``, saved as-is) -- the division by ``evolution_volume`` only
        # happens later, at *use* time, when it is added to ``material``
        # (that module's own ``u_material = material + correction /
        # np.maximum(evolution_volume, ...)``). Dividing here before
        # comparing to the saved array (as this line previously did)
        # compared an un-divided oracle value against a divided replay value
        # -- caught by this module's owner-closure check (up to ~4e-2
        # absolute at interior owners, never validated here before; see the
        # task report).
        replay_correction = np.stack([p06n_correction_acc[vi].total for vi in range(V)])
        variant_results = []
        for vi, name_ in enumerate(variants):
            variant_results.append(compare_owner_term(f"p06n.faces_correction[{name_}]", replay_correction[vi],
                                                       saved_correction[vi], owner_volume=t.vol,
                                                       archived_error=archived_error[vi], region_masks=region_masks_p06n,
                                                       cap_mode="flat", cap_flat_abs=1e-9))
        results_terms["faces_correction"] = _summarize_variants(variant_results)
        results["p06n"] = {"campaign": "p06n", "status": "ok", "terms": results_terms}

    if "p06_legacy" in campaigns:
        import p06_structured_global.numerics as p06numerics

        with np.load(paths["p06_legacy"] / f"N{n}.prepare.npz", allow_pickle=False) as z:
            region_masks_legacy = {name_[len("region:"):]: z[name_] for name_ in z.files if name_.startswith("region:")}
        with np.load(paths["p06_legacy"] / f"N{n}.npz", allow_pickle=False) as z:
            saved = {k_: z[k_].copy() for k_ in z.files if k_ != "metadata_json"}
        masks = {name_: np.asarray(arr, dtype=bool) for name_, arr in region_masks_legacy.items()}
        terms = {}
        for field_name in p06numerics.FIELD_NAMES:
            replay_centered = {term: p06legacy_raw_acc[field_name][term].total / evolution_volume_safe[:, None]
                               for term in ("material", "remainder", "total")}
            correction_owner = p06legacy_correction_acc[field_name].total / evolution_volume_safe[:, None]
            replay_u = {"remainder": replay_centered["remainder"]}
            replay_u["material"] = replay_centered["material"] + correction_owner
            replay_u["total"] = replay_u["material"] + replay_u["remainder"]
            for term in ("material", "remainder", "total"):
                saved_centered = saved[f"candidate:centered:{field_name}:{term}"]
                saved_u = saved[f"candidate:U:{field_name}:{term}"]
                saved_total_R = saved[f"reference_evolution:{field_name}:total"]
                archived_error = saved[f"candidate:centered:{field_name}:total"] - saved_total_R
                terms[f"centered:{field_name}:{term}"] = compare_owner_term(
                    f"p06.centered:{field_name}:{term}", replay_centered[term], saved_centered,
                    owner_volume=t.vol, archived_error=archived_error, region_masks=masks,
                    cap_mode="flat", cap_flat_abs=1e-9)
                terms[f"U:{field_name}:{term}"] = compare_owner_term(
                    f"p06.U:{field_name}:{term}", replay_u[term], saved_u,
                    owner_volume=t.vol, archived_error=archived_error, region_masks=masks,
                    cap_mode="flat", cap_flat_abs=1e-9)
        results["p06_legacy"] = {"campaign": "p06_legacy", "status": "ok", "terms": terms}

    if "p07" in campaigns:
        with np.load(paths["p07"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_action = z["action"].copy(); saved_reference_midpoint = z["reference_midpoint"].copy()
            region_masks_p07 = {name_[len("region_"):]: z[name_] for name_ in z.files if name_.startswith("region_")}
        replay_owner = p07_num.total / t.vol[:, None]
        archived_error = saved_action - saved_reference_midpoint
        terms = {"global_N": compare_owner_term("p07.global_N", replay_owner, saved_action, owner_volume=t.vol,
                                                archived_error=archived_error, region_masks=region_masks_p07)}
        results["p07"] = {"campaign": "p07", "status": "ok", "terms": terms}

    if "p07n" in campaigns:
        with np.load(paths["p07n"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_N = z["N"].copy(); saved_D = z["D"].copy(); saved_O_q3 = z["O_q3"].copy()
            saved_N_minus_O = z["N_minus_O"].copy()
            region_masks_p07n = {name_[len("region_"):]: z[name_] for name_ in z.files if name_.startswith("region_")}
        replay_N = p07n_num["N"].total / t.vol[:, None]
        replay_D = p07n_num["D"].total / t.vol[:, None]
        replay_O = p07n_num["O_q3"].total / t.vol[:, None]
        archived_error = saved_N_minus_O
        terms = {
            "global_N": compare_owner_term("p07n.global_N", replay_N, saved_N, owner_volume=t.vol,
                                           archived_error=archived_error, region_masks=region_masks_p07n),
            "global_D": compare_owner_term("p07n.global_D", replay_D, saved_D, owner_volume=t.vol,
                                           archived_error=archived_error, region_masks=region_masks_p07n),
            "global_O_q3": compare_owner_term("p07n.global_O_q3", replay_O, saved_O_q3, owner_volume=t.vol,
                                              archived_error=archived_error, region_masks=region_masks_p07n),
        }
        results["p07n"] = {"campaign": "p07n", "status": "ok", "terms": terms}

    replay = {
        "schema": "drbx.p08-step1-replay.v1", "n": n, "artifact_root": str(artifact_root),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "wall_seconds": time.time() - started, "campaigns": results,
        "peak_worker_rss_gib_by_unit_kind": peak_rss,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True, default=_json_default))
    write_report(replay, output)
    return replay


# ---------------------------------------------------------------------------
# Bounded "rebuild and compare bitwise" Neumann check -- a preflight-only
# gate (design section 1's task 4: "Keep a bounded rebuild-and-compare check
# of a few Neumann rows per grid, only as a preflight gate"). Unlike every
# replay unit above, this function *does* open a stage's ``neumann`` chunk
# file directly (its whole purpose is to cross-check the artifact's stored
# Neumann rows against a fresh rebuild) -- always on a small, bounded sample,
# never as part of the per-grid replay itself.
# ---------------------------------------------------------------------------
def _true_query_point(env: Environment, stage: str, request: str, entity_id: int, quad_node: int) -> np.ndarray:
    """The row's own true query point -- *never* ``PointRows.trace_target_points``
    (see the task report: that is the wall-projected Dirichlet-lift anchor,
    which coincides with the true point only for an unconditioned row, and
    was the root cause of the on-the-fly Neumann-rebuild bug this preflight
    check exists to guard against). Derived directly from geometry, the same
    way ``drbx.stencils.builder`` built the artifact's own rows."""
    if stage == "cells":
        if quad_node != 0:
            raise ValueError(f"a cells (R1) Neumann row must have quad_node 0, got {quad_node}")
        return env.t.pts[entity_id:entity_id + 1]
    if stage == "faces":
        # R2's and R3's Neumann rows are both tagged at the face's own census
        # row index (see drbx.stencils.builder.build_r3_side_rows's
        # docstring), so `entity_id` is directly a census row index here.
        key = env.census.keys()[entity_id][None, :]
        points, _weight = pshared_provider._quadrature(env.t.faces, key, 3, face=True)
        return points[0, quad_node:quad_node + 1]
    if stage == "p07":
        rows = np.flatnonzero(env.census.p07_id == entity_id)
        rows = rows[~env.census.legacy_alias_slots[rows]]
        if rows.size != 1:
            raise ValueError(f"expected exactly one canonical census row for p07 id {entity_id}, got {rows.size}")
        key = env.census.keys()[rows[0]][None, :]
        points, _weight = pshared_provider._quadrature(env.t.faces, key, 3, face=True)
        return points[0, quad_node:quad_node + 1]
    raise ValueError(f"unknown stage {stage!r}")


def neumann_rebuild_compare_check(*, artifact_root, n: int, input_root, sidecar_path,
                                  stage: str = "cells", chunk_index: int = 0, sample: int = 8) -> dict:
    artifact_root = Path(artifact_root)
    grid_dir = artifact_root / f"N{n}"
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    neumann_entry = _manifest_entry(manifest, "neumann", stage, chunk_index)
    data = _load_chunk_bytes(grid_dir, neumann_entry)
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    neumann_chunk = artifact_mod._arrays_to_neumann_chunk(arrays)

    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path))

    total = len(neumann_chunk.entity_id)
    if total == 0:
        return {"stage": stage, "chunk_index": chunk_index, "checked": 0, "max_value_diff": 0.0,
               "max_gradient_diff": 0.0, "pass": True, "note": "no Neumann rows in this chunk"}
    step = max(1, total // sample)
    indices = list(range(0, total, step))[:sample]

    max_value_diff = 0.0
    max_gradient_diff = 0.0
    max_boundary_value_diff = 0.0
    checked = 0
    for i in indices:
        request = str(neumann_chunk.request[i])
        entity_id = int(neumann_chunk.entity_id[i])
        quad_node = int(neumann_chunk.quad_node[i])
        degree = int(neumann_chunk.radial_degree[i])
        point = _true_query_point(env, stage, request, entity_id, quad_node)
        rebuilt = env.neumann.rows_for(point, degree)[0]
        p0, p1 = int(neumann_chunk.donor_ptr[i]), int(neumann_chunk.donor_ptr[i + 1])
        stored_donor = neumann_chunk.donor[p0:p1].astype(np.int64)
        if not np.array_equal(stored_donor, np.asarray(rebuilt.donor_ids)):
            raise ValueError(f"neumann check: donor mismatch at entity {entity_id} quad_node {quad_node}")
        stored_value = neumann_chunk.value[p0:p1]
        stored_gradient = neumann_chunk.gradient[:, p0:p1]
        max_value_diff = max(max_value_diff, float(np.max(np.abs(stored_value - rebuilt.value), initial=0.0)))
        max_gradient_diff = max(max_gradient_diff, float(np.max(np.abs(stored_gradient - rebuilt.gradient), initial=0.0)))
        max_boundary_value_diff = max(max_boundary_value_diff, float(np.max(np.abs(
            neumann_chunk.boundary_value[i].astype(np.float64) - np.asarray(rebuilt.boundary_value, dtype=np.float64)))))
        checked += 1

    passed = max_value_diff == 0.0 and max_gradient_diff == 0.0 and max_boundary_value_diff == 0.0
    return {"stage": stage, "chunk_index": chunk_index, "checked": checked,
           "max_value_diff": max_value_diff, "max_gradient_diff": max_gradient_diff,
           "max_boundary_value_diff": max_boundary_value_diff, "pass": bool(passed)}
