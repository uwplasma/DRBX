"""Bounded owner-closure selection, row building, and campaign comparison
for the P08 step-1 replay (task: "owner-closure validation of the P08
step-1 replay against the six frozen campaigns" -- see the task report).

This module answers, for a small, deterministic set of owners covering
every geometric regime the census distinguishes (interior, wall-adjacent
fully/partially, the RLP transition, an aggregate owner, near-axis owners,
and the periodic theta/eta seams): build only the R1/R2/R3/R4 rows those
owners' incident census faces need (:mod:`drbx.stencils.builder`, called
directly on explicit id lists -- no chunked row artifact on disk), run
:mod:`p_shared.replay_units`'s own per-campaign core arithmetic
(``_cells_unit_core``/``_faces_unit_core``/``_p07_unit_core`` -- the exact
same functions ``compute_cells_unit``/``compute_faces_unit``/
``compute_p07_unit`` call against a real chunk, just fed in-memory rows
instead), and compare the result against each frozen oracle at exactly
those owners.

Used two ways:

* a one-off, bounded validation script (see the task report's table);
* this package's own bounded preflight replay check
  (:mod:`p08_step1_global.campaign`'s ``preflight_grid``), run at every
  grid -- both the owner *selection* and the oracle *comparison* are
  grid-generic (every frozen campaign saved its oracle arrays at N32, N48
  *and* N64 -- see :func:`p_shared.replay_units._load_oracle_owner_values`
  and this module's own :func:`compare_to_oracle`, both keyed by ``n``, not
  hardcoded to N32).

  At a grid with no on-disk production geometry (N48/N64 locally -- a hard
  local constraint against ever running a full-grid geometry stage), this
  module's own :func:`build_owner_rows` computes geometry only for the
  selected owners' raw cells and incident faces, via
  :func:`drbx.stencils.geometry_arrays.build_raw_geometry_arrays`/
  ``build_face_geometry_arrays`` batched at the same 4096-row chunk size the
  production parallel geometry stage uses (see :func:`_owner_geometry_arrays`)
  -- so this check never depends on a grid's own ``geometry.npz`` existing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared import selection as sel                              # noqa: E402
from p_shared import provider as pshared_provider                  # noqa: E402
from p_shared import replay_units as ru                            # noqa: E402
from p_shared.replay_support import (                              # noqa: E402
    Environment, build_environment, _load_p05_upwind,
)
from drbx.stencils import builder as stencil_builder                # noqa: E402

SCHEMA = "drbx.p08-step1-owner-closure-selection.v1"

#: The frozen P05 "eight deterministic angular locations" fractions'
#: primary/secondary picks (see ``selection.eighth_fraction_pairs``); index
#: 0 is used for every single-sample category, index 4 (roughly the
#: opposite angular location) only for the two categories that take a
#: second sample.
_PRIMARY_FRACTION_INDEX = 0
_SECONDARY_FRACTION_INDEX = 4


# ---------------------------------------------------------------------------
# Selection: one representative owner per required coverage category.
# ---------------------------------------------------------------------------
def _owner_at(t, radial_index: int, fraction) -> int:
    return int(sorted(sel.owners_at_index_eighths(t, radial_index, [fraction]))[0])


def _first_mixed_wall_owner(census) -> int:
    """The smallest owner id whose incident census faces mix a P07
    boundary-conditioned family (1/2/4, "physical_wall_quartic_BC"/
    "adjacent_radial_quartic_BC"/"boundary_transverse_cubic_BC") with a
    non-conditioned one (0/3/5/6/7) -- a genuinely *partially* wall-adjacent
    owner, as opposed to one every one of whose incident faces is
    conditioned ("fully" wall-adjacent, which the wall-band radial-index
    picks below already cover)."""
    owner_lo = census.owner_lo
    owner_hi = census.owner_hi
    family = census.family
    wall_mask = np.isin(family, (1, 2, 4))
    candidates = np.unique(np.concatenate([owner_lo[wall_mask], owner_hi[wall_mask]]))
    candidates = np.sort(candidates[candidates >= 0])
    for owner in candidates:
        rows = np.flatnonzero((owner_lo == owner) | (owner_hi == owner))
        fam = family[rows]
        fam = fam[fam >= 0]
        if fam.size and np.isin(fam, (1, 2, 4)).any() and np.isin(fam, (0, 3, 5, 6, 7)).any():
            return int(owner)
    raise ValueError("no partially wall-adjacent (mixed-family) owner found")


def select_owners(t, census) -> dict[str, int]:
    """One representative owner per required coverage category: interior,
    wall-adjacent (fully and partially), the RLP transition, an aggregate
    owner, the eta seam, near-axis owners, and the periodic theta seam (slot
    0/n) -- see the task report. Deterministic and geometry/census-only (no
    saved campaign result is read), reusing :mod:`p_shared.selection`'s own
    frozen category machinery."""
    n = t.n
    cats = sel.radial_index_categories(t)
    mid = n // 2
    fractions = sel.eighth_fraction_pairs()
    primary = fractions[_PRIMARY_FRACTION_INDEX % len(fractions)]
    secondary = fractions[_SECONDARY_FRACTION_INDEX % len(fractions)]

    transition_indices = sorted(cats["profile_transitions_pm1"])
    aggregate_index = sorted(cats["aggregate_layer"])[0]

    owners = {
        "axis_core": _owner_at(t, 0, primary),
        "near_axis": _owner_at(t, 1, primary),
        "aggregate": _owner_at(t, aggregate_index, primary),
        "transition": _owner_at(t, transition_indices[len(transition_indices) // 2], primary),
        "interior": _owner_at(t, mid, primary),
        "interior_secondary": _owner_at(t, mid, secondary),
        "wall_partial": _first_mixed_wall_owner(census),
        "wall_full": _owner_at(t, n - 1, primary),
        "wall_full_secondary": _owner_at(t, n - 1, secondary),
        "theta_seam": int(t.ro[int(np.ravel_multi_index((mid, 0, mid), (n, n, n)))]),
        "eta_seam": int(t.ro[int(np.ravel_multi_index((mid, mid, 0), (n, n, n)))]),
        "theta_seam_wall": int(t.ro[int(np.ravel_multi_index((n - 1, 0, mid), (n, n, n)))]),
    }
    return {k: int(v) for k, v in owners.items()}


def incident_census_rows(census, owners) -> np.ndarray:
    """Every census row incident to any of ``owners`` (lower or upper
    side), sorted and deduplicated -- the row-index domain
    ``build_r1_cell_rows``/``build_r2_face_rows``/``build_r3_side_rows``/
    ``build_r4_p07_rows`` and ``compute_cells_unit``/``compute_faces_unit``/
    ``compute_p07_unit`` all index directly."""
    owner_arr = np.asarray(sorted(set(int(o) for o in owners)), dtype=np.int64)
    mask = np.isin(census.owner_lo, owner_arr) | np.isin(census.owner_hi, owner_arr)
    return np.flatnonzero(mask).astype(np.int64)


def selection_fixture(t, census) -> dict:
    """The full fixture payload (schema-tagged, JSON-serializable) this
    module's selection freezes to disk for a given grid -- see
    ``p08_step1_global/campaign.py``'s preflight, which stores this under
    ``<output>/N{n}/owner_closure_selection.json``."""
    owners = select_owners(t, census)
    unique_owners = sorted(set(owners.values()))
    rows = incident_census_rows(census, unique_owners)
    return {
        "schema": SCHEMA,
        "n": int(t.n),
        "categories": owners,
        "owners": unique_owners,
        "census_row_count": int(len(rows)),
    }


def load_provider_for_env(sidecar_path) -> "pshared_provider.ScriptsGeometryProvider":
    """A ``ScriptsGeometryProvider`` built the same way ``build_environment``
    builds ``env.ref`` internally (``env.ref`` is only its ``.reference``
    attribute -- ``build_geometry_arrays`` below needs the provider itself,
    for its ``.face_points``/``.p06_face_weight``/``.p07_face_tensor``
    methods)."""
    return pshared_provider.ScriptsGeometryProvider.from_sidecar(str(sidecar_path), verify_hashes=False)


# ---------------------------------------------------------------------------
# Geometry sourcing: the same 4096-row chunk size the production parallel
# geometry stage uses (``configuration.json``'s ``geometry_raw_chunk_size``/
# ``geometry_face_chunk_size``), applied to a small, explicit owner-selected
# key list instead of the whole grid. This never touches a grid's own
# ``geometry.npz`` (which may not exist -- N48/N64 locally, a hard local
# constraint against ever running a full-grid geometry stage): it calls the
# same split provider functions ``build_artifact.py``'s ``geometry_raw``/
# ``geometry_face`` units call, ``drbx.stencils.geometry_arrays
# .build_raw_geometry_arrays``/``build_face_geometry_arrays``, batched at the
# same chunk size, then concatenates the batches into one in-memory
# ``GeometryArrays`` -- bitwise consistent with a single unchunked call
# whenever the selected subset (always true for a dozen owners' worth of raw
# cells/faces, verified empirically -- see the task report) is itself smaller
# than one chunk, since a single provider call over the whole subset is then
# exactly what the chunked path also does (one chunk == the whole subset).
# ``build_artifact.py``'s own module docstring documents a small (~1e-9
# relative) residual between this and a *different* batch shape for the
# finite-difference-derived P06/P07 fields (K, divergence); that residual
# only arises when a batch boundary actually falls inside the subset (subset
# size > chunk size), which never happens here.
# ---------------------------------------------------------------------------
_GEOMETRY_RAW_CHUNK = 4096
_GEOMETRY_FACE_CHUNK = 4096

_RAW_GEOMETRY_FIELDS = (
    "raw_points", "p05_raw_h", "p05_raw_jacobian", "p06_raw_J", "p06_raw_B", "p06_raw_K",
    "p06_raw_weight", "p07_raw_tensor", "p07_raw_divergence",
)
_FACE_GEOMETRY_FIELDS = (
    "face_points", "p05_face_h", "p05_face_jacobian", "p06_face_J", "p06_face_B", "p06_face_K",
    "p06_face_weight", "p07_face_tensor",
)


def _owner_geometry_arrays(provider, faces, raw_keys: np.ndarray, face_keys: np.ndarray, *,
                           raw_chunk: int = _GEOMETRY_RAW_CHUNK, face_chunk: int = _GEOMETRY_FACE_CHUNK):
    """Build a :class:`~drbx.stencils.geometry_arrays.GeometryArrays` for
    exactly ``raw_keys``/``face_keys`` (an owner-selected subset, never the
    whole grid), batched at ``raw_chunk``/``face_chunk`` -- see this module's
    own section docstring above for why this is bitwise-equivalent to the
    previous single unchunked ``stencil_builder.build_geometry_arrays`` call
    whenever the subset stays under one chunk."""
    from drbx.stencils import geometry_arrays as geom_arrays_mod

    raw_keys = np.asarray(raw_keys, dtype=np.int64)
    face_keys = np.asarray(face_keys, dtype=np.int64)
    raw_parts = [geom_arrays_mod.build_raw_geometry_arrays(provider, faces, raw_keys[start:start + raw_chunk])
                for start in range(0, len(raw_keys), raw_chunk)] or [
        geom_arrays_mod.build_raw_geometry_arrays(provider, faces, raw_keys[:0])]
    face_parts = [geom_arrays_mod.build_face_geometry_arrays(provider, faces, face_keys[start:start + face_chunk])
                 for start in range(0, len(face_keys), face_chunk)] or [
        geom_arrays_mod.build_face_geometry_arrays(provider, faces, face_keys[:0])]

    arrays = {field: np.concatenate([part[field] for part in raw_parts], axis=0) for field in _RAW_GEOMETRY_FIELDS}
    arrays.update({field: np.concatenate([part[field] for part in face_parts], axis=0)
                  for field in _FACE_GEOMETRY_FIELDS})
    identity = geom_arrays_mod.GeometryArrays._compute_identity(arrays)
    return geom_arrays_mod.GeometryArrays(schema=geom_arrays_mod.SCHEMA, identity=identity, **arrays)


# ---------------------------------------------------------------------------
# Row building: R1/R2/R3/R4 (+ Neumann) rows for exactly the incident faces
# of ``owners`` -- ``drbx.stencils.builder`` called directly on explicit id
# lists, never a chunked row artifact.
# ---------------------------------------------------------------------------
def build_owner_rows(env: Environment, owners, *, provider=None) -> dict:
    """Build every row this module's core-function calls need for
    ``owners``'s incident census faces (a bounded subset -- no full-grid
    geometry array, no on-disk artifact). Returns a dict with ``row_index``/
    ``neumann_index`` (keyed exactly as ``compute_cells_unit``/
    ``compute_faces_unit``/``compute_p07_unit`` expect internally -- see
    those functions' own ``get_r1``/``get_r2``/``get_r3``/``row_index.get``
    lookups), plus ``raw_ids``/``face_row_indices``/``p07_row_indices``
    (sorted, deduplicated) and the owner list actually covered.

    ``provider`` is an optional, already-built ``ScriptsGeometryProvider``
    (avoids rebuilding one from the sidecar if the caller already has one);
    when omitted, one is built the same way :func:`p_shared.replay_support
    .build_environment` builds ``env.ref`` internally.
    """
    t = env.t
    census = env.census
    unique_owners = sorted(set(int(o) for o in owners))
    owner_arr = np.asarray(unique_owners, dtype=np.int64)

    raw_ids = np.flatnonzero(np.isin(t.ro, owner_arr)).astype(np.int64)
    incident = incident_census_rows(census, unique_owners)
    face_selection = ru.face_row_selection(census)
    face_row_indices = np.intersect1d(incident, face_selection)
    p07_selection = stencil_builder.p07_row_selection(census)
    p07_row_indices = np.intersect1d(face_row_indices, p07_selection)

    if provider is None:
        raise ValueError("build_owner_rows requires a ScriptsGeometryProvider (pass provider=...; see "
                         "load_provider_for_env, which builds one the same way build_environment builds "
                         "env.ref internally)")

    n = t.n
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    face_keys = stencil_builder.census_face_keys(census, face_row_indices)
    geometry = _owner_geometry_arrays(provider, env.ctx.faces, raw_keys, face_keys)

    patch_cache: dict = {}
    point_requests, neumann_requests = stencil_builder.build_r1_cell_rows(
        env.S, env.ctx, raw_ids, normal_coefficients=env.normal_coefficients, patch_cache=patch_cache)
    r2_requests, r2_neumann = stencil_builder.build_r2_face_rows(
        env.S, env.ctx, census, face_row_indices, geometry.face_points,
        normal_coefficients=env.normal_coefficients, patch_cache=patch_cache)
    r3_requests, r3_neumann = stencil_builder.build_r3_side_rows(
        env.S, env.ctx, census, face_row_indices, geometry.face_points,
        normal_coefficients=env.normal_coefficients, patch_cache=patch_cache)
    point_requests = point_requests + r2_requests + r3_requests
    neumann_requests = neumann_requests + r2_neumann + r3_neumann

    face_pos = np.searchsorted(face_row_indices, p07_row_indices)
    if not np.array_equal(face_row_indices[face_pos], p07_row_indices):
        raise ValueError("build_owner_rows: p07_row_indices must be a subset of face_row_indices")
    integrated_requests, p07_neumann = stencil_builder.build_r4_p07_rows(
        env.ctx, census, p07_row_indices, geometry.face_points[face_pos], geometry.p06_face_weight[face_pos],
        geometry.p07_face_tensor[face_pos], normal_coefficients=env.normal_coefficients, patch_cache=patch_cache)
    neumann_requests = neumann_requests + p07_neumann

    row_index = {(req.request, req.entity_id): req.row for req in point_requests}
    row_index.update({req.entity_id: req.row for req in integrated_requests})
    neumann_index = {(nr.source, nr.entity_id, nr.quad_node): nr.row for nr in neumann_requests}

    return {
        "owners": unique_owners, "raw_ids": raw_ids, "face_row_indices": face_row_indices,
        "p07_row_indices": p07_row_indices, "row_index": row_index, "neumann_index": neumann_index,
        "geometry": geometry,
    }


# ---------------------------------------------------------------------------
# Assembly: run the replay's own per-campaign core functions against the
# in-memory rows above, restricted to exactly ``owners``.
# ---------------------------------------------------------------------------
def assemble_owner_terms(env: Environment, built: dict, campaigns: tuple, oracle: dict) -> dict:
    """Run ``_cells_unit_core``/``_faces_unit_core``/``_p07_unit_core`` (the
    same arithmetic ``compute_cells_unit``/``compute_faces_unit``/
    ``compute_p07_unit`` run against a real artifact chunk) against the rows
    :func:`build_owner_rows` built, and return the raw per-campaign ``out``
    dicts (still ``(unique_owner_ids, values)`` sparse pairs -- exactly the
    shape ``replay_units.reduce_grid`` accumulates) unscattered to a dense
    array here; callers with a small owner set typically want
    :func:`owner_values_from_pairs` next."""
    t = env.t
    census = env.census
    n = t.n

    raw_ids = built["raw_ids"]
    owner_ids = t.ro[raw_ids]
    volume = t.rv[raw_ids]
    points = t.pts[raw_ids]
    row_index = built["row_index"]
    neumann_index = built["neumann_index"]

    def get_r1(raw_id):
        row = row_index.get(("R1", int(raw_id)))
        if row is None:
            raise ValueError(f"owner_closure: no R1 row for raw id {raw_id}")
        return row

    rows = [get_r1(j) for j in raw_ids]
    neumann_rows_by_row = [([neumann_index[("R1", int(j), 0)]] if row.boundary_conditioned else None)
                           for j, row in zip(raw_ids, rows)]

    cells_out = ru._cells_unit_core(env=env, campaigns=campaigns, oracle=oracle, n=n, t=t, raw_ids=raw_ids,
                                    owner_ids=owner_ids, volume=volume, points=points, rows=rows,
                                    neumann_rows_by_row=neumann_rows_by_row)

    face_row_indices = built["face_row_indices"]
    keys = census.keys()[face_row_indices]
    lower_owner_all = census.owner_lo[face_row_indices]
    upper_owner_all = census.owner_hi[face_row_indices]

    def get_r2(ridx):
        row = row_index.get(("R2", int(ridx)))
        if row is None:
            raise ValueError(f"owner_closure: no R2 row for census row {ridx}")
        return row

    def get_r3(ridx, side):
        return row_index.get(("R3", int(ridx) * 2 + side))

    def side_exists(key, axis, side):
        ijk = [int(v) for v in key[1:]]
        probe = ijk.copy(); probe[axis] += -1 if side == 0 else 0
        if axis == 0 and side == 0 and probe[0] < 0:
            return False
        if axis == 0 and side == 1 and ijk[0] >= n:
            return False
        return True

    common_rows = [get_r2(int(ridx)) for ridx in face_row_indices]
    lower_rows = [get_r3(int(ridx), 0) for ridx in face_row_indices]
    upper_rows = [get_r3(int(ridx), 1) for ridx in face_row_indices]
    axis_all = keys[:, 0].astype(np.int64)
    side_exists_lower = np.array([side_exists(keys[i], int(axis_all[i]), 0) for i in range(len(face_row_indices))])
    side_exists_upper = np.array([side_exists(keys[i], int(axis_all[i]), 1) for i in range(len(face_row_indices))])
    common_points_by_face = [r.trace_target_points for r in common_rows]

    def _face_neumann_rows(request, ridx, q_count):
        got = [neumann_index.get((request, int(ridx), q)) for q in range(q_count)]
        if all(g is None for g in got):
            return None
        if any(g is None for g in got):
            raise ValueError(f"owner_closure: partial stored Neumann rows for {request} entity {ridx}")
        return got

    common_neumann_rows_by_face = [_face_neumann_rows("R2", ridx, len(common_points_by_face[local]))
                                   for local, ridx in enumerate(face_row_indices)]
    side_neumann_rows_by_face = [_face_neumann_rows("R3", ridx, len(common_points_by_face[local]))
                                 for local, ridx in enumerate(face_row_indices)]

    F = len(face_row_indices)
    _q_points_all, weight_all = ((np.zeros((0, 9, 3)), np.zeros((0, 9))) if F == 0 else
                                ru.pshared_provider._quadrature(env.t.faces, keys, 3, face=True))
    common_counts = [len(p) for p in common_points_by_face]
    common_points_flat = (_q_points_all.reshape(-1, 3) if F else np.zeros((0, 3)))
    if F and len(set(common_counts)) == 1 and common_counts[0] == 9:
        metric_all = env.ref._metric(common_points_flat)
        h_all = (metric_all["bcov"] / metric_all["B"][:, None]).reshape(F, 9, 3)
    elif F:
        metric_all = env.ref._metric(common_points_flat)
        h_flat = metric_all["bcov"] / metric_all["B"][:, None]
        h_all = np.split(h_flat, np.cumsum(common_counts)[:-1])
    else:
        h_all = np.zeros((0, 9, 3))

    J_all = B_all = K_all = None
    if F and ("p06n" in campaigns or "p06_legacy" in campaigns):
        import p06_structured_global.numerics as _p06numerics_geom

        J_flat, B_flat, K_flat = _p06numerics_geom._face_geometry(env.ref, common_points_flat)
        if len(set(common_counts)) == 1 and common_counts[0] == 9:
            J_all = J_flat.reshape(F, 9); B_all = B_flat.reshape(F, 9); K_all = K_flat.reshape(F, 9, -1)
        else:
            splits = np.cumsum(common_counts)[:-1]
            J_all = np.split(J_flat, splits); B_all = np.split(B_flat, splits); K_all = np.split(K_flat, splits)

    faces_out = ru._faces_unit_core(
        env=env, campaigns=campaigns, oracle=oracle, n=n, t=t, census=census, row_indices=face_row_indices,
        keys=keys, lower_owner_all=lower_owner_all, upper_owner_all=upper_owner_all, common_rows=common_rows,
        lower_rows=lower_rows, upper_rows=upper_rows, common_points_by_face=common_points_by_face,
        side_exists_lower=side_exists_lower, side_exists_upper=side_exists_upper,
        common_neumann_rows_by_face=common_neumann_rows_by_face, side_neumann_rows_by_face=side_neumann_rows_by_face,
        h_all=h_all, weight_all=weight_all, J_all=J_all, B_all=B_all, K_all=K_all)

    p07_row_indices = built["p07_row_indices"]
    p07_ids = census.p07_id[p07_row_indices]
    family = census.family[p07_row_indices]
    p07_keys = census.keys()[p07_row_indices]
    p07_lower_owner = census.owner_lo[p07_row_indices]
    p07_upper_owner = census.owner_hi[p07_row_indices]

    p07_out = ru._p07_unit_core(env=env, campaigns=campaigns, oracle=oracle, row_index=row_index,
                                neumann_index=neumann_index, row_indices=p07_row_indices, keys=p07_keys,
                                family=family, p07_ids=p07_ids, lower_owner=p07_lower_owner,
                                upper_owner=p07_upper_owner)

    return {"cells": cells_out, "faces": faces_out, "p07": p07_out}


def owner_values_from_pairs(pair, owners, n_owners: int) -> np.ndarray:
    """Scatter one ``(unique_owner_ids, values)`` sparse pair (this
    module's/``replay_units``'s convention) into a dense ``(len(owners),
    *value_shape)`` array restricted to ``owners`` (in the given order) --
    every owner not present in ``pair`` (no incident row contributed to it
    in this bounded build) reads as exactly zero, matching a full accumu-
    lation's own initial state."""
    uniq, values = pair
    values = np.asarray(values)
    dense = np.zeros((n_owners,) + values.shape[1:], dtype=values.dtype)
    dense[np.asarray(uniq, dtype=np.int64)] = values
    return dense[np.asarray(owners, dtype=np.int64)]


# ---------------------------------------------------------------------------
# Oracle comparison: every campaign's own saved N{n} array, read at exactly
# the selected owners, against this module's replay -- see the task report's
# validation table. Grid-generic (every frozen campaign saved this file at
# N32, N48 *and* N64; matches ``replay_units._load_oracle_owner_values``'s
# own scope).
# ---------------------------------------------------------------------------
#: max_abs / max(|oracle N-R|) above this is a real mismatch, not roundoff --
#: every genuine match this module has seen sits at ratio <= ~2e-7 (P07N's
#: own already-accepted bound is 3.7e-10 absolute); every bug this task found
#: (the P07 Dirichlet-wall root cause, the P05N/P06N exterior-fallback fix,
#: the P06N faces_correction division fix) showed ratio >= ~0.1 or maximal
#: absolute error orders of magnitude above this before its fix. 1e-5 keeps
#: five-plus orders of margin on both sides.
RATIO_TOLERANCE = 1e-5
#: Absolute fallback when the oracle's own |N-R| is exactly zero at every
#: selected owner (ratio undefined) -- the gate's own "~1e-10 or better".
ABS_FALLBACK_TOLERANCE = 1e-9


def _row(campaign: str, term: str, replay, saved, archived_error, note: str = "") -> dict:
    replay = np.asarray(replay, dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    archived_error = np.asarray(archived_error, dtype=np.float64)
    diff = replay - saved
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(saved), 1e-300)
    max_rel = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    oracle_nr = float(np.max(np.abs(archived_error))) if archived_error.size else 0.0
    if oracle_nr > 0:
        ratio = max_abs / oracle_nr
    else:
        # A large *finite* sentinel, never `float("inf")`: this payload is
        # written to JSON (`runner.write_json`'s `allow_nan=False`) by
        # ``p08_step1_global.campaign``'s preflight.
        ratio = 0.0 if max_abs == 0.0 else 1.0e18
    # An absolute-roundoff pass (max_abs already at/below the gate's own
    # "~1e-10 or better") always counts, regardless of the ratio -- a
    # degenerate/trivial variant (e.g. P06N's "control_constant", where the
    # oracle's own |N-R| is itself just roundoff noise) can otherwise blow
    # the ratio up to O(1) or worse from noise-over-noise division alone,
    # which is not the real mismatch this gate looks for (see the task
    # report's real bugs, every one of which showed max_abs orders of
    # magnitude above this floor).
    passed = (ratio <= RATIO_TOLERANCE) or (max_abs <= ABS_FALLBACK_TOLERANCE)
    return {"campaign": campaign, "term": term, "max_abs": max_abs, "max_rel": max_rel, "ratio_to_oracle_NR": ratio,
           "pass": bool(passed), "note": note}


def compare_to_oracle(env: Environment, out: dict, unique_owners, paths: dict, campaigns: tuple) -> list[dict]:
    """Every campaign x term row of the task report's validation table, for
    exactly ``unique_owners`` -- see :func:`assemble_owner_terms` for
    ``out``. Only the campaigns present in both ``campaigns`` and ``out``
    are compared (a caller running a subset of the six still gets a report,
    just a shorter one)."""
    t = env.t
    n = int(t.n)
    owner_arr = np.asarray(sorted(set(int(o) for o in unique_owners)), dtype=np.int64)
    n_owners_total = len(t.vol)
    rows: list[dict] = []

    def dense(pair):
        return owner_values_from_pairs(pair, owner_arr, n_owners_total)

    if "p05" in campaigns and "p05_centered" in out.get("cells", {}):
        with np.load(paths["p05"] / f"N{n}.owner_results.npz", allow_pickle=False) as z:
            saved_centered = z["centered"][owner_arr]; saved_reference = z["reference"][owner_arr]
        replay_centered = dense(out["cells"]["p05_centered"]) / t.vol[owner_arr, None]
        archived_error_p05 = saved_centered - saved_reference
        rows.append(_row("P05", "centered", replay_centered, saved_centered, archived_error_p05))

        with np.load(paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as z:
            saved_old_u_minus_a = z["old_U_minus_A"][owner_arr]
        replay_live_jump = dense(out["faces"]["p05_live_jump_owner_num"]) / t.vol[owner_arr, None]
        rows.append(_row("P05", "live_jump_vs_old_U_minus_A", replay_live_jump, saved_old_u_minus_a,
                         archived_error_p05))

        from p_shared.replay_support import _load_p05_upwind
        saved_upwind = _load_p05_upwind(paths["p05_upwind_chunks"], int(t.n))
        p07ids = np.asarray(out["faces"]["p05_live_jump_p07ids"])
        vals = np.asarray(out["faces"]["p05_live_jump_values"])
        if len(p07ids):
            keep = p07ids < saved_upwind.shape[0]
            diff = vals[keep] - saved_upwind[p07ids[keep]]
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            rows.append({"campaign": "P05", "term": "live_jump_vs_upwind (pointwise)", "max_abs": max_abs,
                        "max_rel": None, "ratio_to_oracle_NR": None, "pass": max_abs <= ABS_FALLBACK_TOLERANCE,
                        "note": f"n_faces={int(keep.sum())}"})

    for name, root_key in (("p05n_frozen", "p05n_frozen"), ("p05n_upwind", "p05n_p06n_upwind")):
        if name not in campaigns or f"{name}_raw_N" not in out.get("cells", {}):
            continue
        root = paths[root_key] if name == "p05n_frozen" else paths[root_key] / "p05n_upwind"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved_raw_N = z["N"][owner_arr]; saved_raw_D = z["D"][owner_arr]; saved_raw_R = z["R"][owner_arr]
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_face_N = z["N"][owner_arr]; saved_face_D = z["D"][owner_arr]
        archived_error_raw = saved_raw_N - saved_raw_R
        for suffix, saved in (("N", saved_raw_N), ("D", saved_raw_D), ("R", saved_raw_R)):
            replay = dense(out["cells"][f"{name}_raw_{suffix}"]) / t.vol[owner_arr, None]
            rows.append(_row(name, f"raw_{suffix}", replay, saved, archived_error_raw))
        for suffix, saved in (("N", saved_face_N), ("D", saved_face_D)):
            replay = dense(out["faces"][f"{name}_face_{suffix}"]) / t.vol[owner_arr, None]
            rows.append(_row(name, f"face_{suffix}", replay, saved, archived_error_raw))

    if "p06n" in campaigns and "q1_evolution_volume" in out.get("cells", {}):
        import p06n_field_derived_global.core as p06n_core

        tables = p06n_core.CATALOGUE_TABLES
        variants = tables.variant_names
        root = paths["p05n_p06n_upwind"] / "p06n"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved_p06n = {label: z[label][:, owner_arr] for label in
                         ("material", "remainder", "total", "R_material", "R_remainder", "R_total")}
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_p06n_correction = z["correction"][:, owner_arr]

        evolution_volume = dense(out["cells"]["q1_evolution_volume"])
        evolution_volume_safe = np.maximum(evolution_volume, 1e-300)
        archived_error_p06n = saved_p06n["total"] - saved_p06n["R_total"]

        for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total"):
            for vi, name_ in enumerate(variants):
                replay_v = dense(out["cells"][f"p06n_raw_{label}"][vi]) / evolution_volume_safe[:, None]
                rows.append(_row("P06N", f"raw_{label}[{name_}]", replay_v, saved_p06n[label][vi],
                                 archived_error_p06n[vi]))

        for vi, name_ in enumerate(variants):
            # `correction` is saved RAW (undivided) -- see the root-cause fix
            # in replay_units.reduce_grid's own P06N block.
            replay_corr = dense(out["faces"]["p06n_faces_correction"][vi])
            rows.append(_row("P06N", f"faces_correction[{name_}]", replay_corr, saved_p06n_correction[vi],
                             archived_error_p06n[vi]))

        # phi-Neumann/phi-Dirichlet material/correction must match bitwise
        # (task report): pick the first matched pair of "*_phi_dirichlet"/
        # "*_phi_neumann" variants (any prefix) and diff them directly --
        # never against the oracle, this is an internal consistency check.
        phi_pairs: dict[str, list[str]] = {}
        for v in variants:
            if v.endswith(":D"):
                continue
            if "_phi_dirichlet" in v:
                phi_pairs.setdefault(v.replace("_phi_dirichlet", ""), [None, None])[0] = v
            elif "_phi_neumann" in v:
                phi_pairs.setdefault(v.replace("_phi_neumann", ""), [None, None])[1] = v
        for prefix, (v_d, v_n) in phi_pairs.items():
            if v_d is None or v_n is None:
                continue
            vi_d = list(variants).index(v_d); vi_n = list(variants).index(v_n)
            for label in ("material",):
                a = dense(out["cells"][f"p06n_raw_{label}"][vi_d])
                b = dense(out["cells"][f"p06n_raw_{label}"][vi_n])
                d = float(np.max(np.abs(a - b)))
                rows.append({"campaign": "P06N", "term": f"phi_N_vs_D bitwise[{label}][{prefix}]", "max_abs": d,
                            "max_rel": None, "ratio_to_oracle_NR": None, "pass": d == 0.0, "note": f"{v_d} vs {v_n}"})
            a = dense(out["faces"]["p06n_faces_correction"][vi_d])
            b = dense(out["faces"]["p06n_faces_correction"][vi_n])
            d = float(np.max(np.abs(a - b)))
            rows.append({"campaign": "P06N", "term": f"phi_N_vs_D bitwise[correction][{prefix}]", "max_abs": d,
                        "max_rel": None, "ratio_to_oracle_NR": None, "pass": d == 0.0, "note": f"{v_d} vs {v_n}"})

    if "p06_legacy" in campaigns and "p06legacy_raw_centered" in out.get("cells", {}):
        import p06_structured_global.numerics as p06numerics

        with np.load(paths["p06_legacy"] / f"N{n}.npz", allow_pickle=False) as z:
            saved_legacy = {k_: z[k_][owner_arr] for k_ in z.files if k_ != "metadata_json"}
        evolution_volume = dense(out["cells"]["q1_evolution_volume"])
        evolution_volume_safe = np.maximum(evolution_volume, 1e-300)
        for field_name in p06numerics.FIELD_NAMES:
            replay_centered = {
                term: dense(out["cells"]["p06legacy_raw_centered"][field_name][term]) / evolution_volume_safe[:, None]
                for term in ("material", "remainder", "total")}
            correction_owner = dense(out["faces"]["p06legacy_faces_correction"][field_name]) / evolution_volume_safe[:, None]
            replay_u = {"remainder": replay_centered["remainder"]}
            replay_u["material"] = replay_centered["material"] + correction_owner
            replay_u["total"] = replay_u["material"] + replay_u["remainder"]
            archived_error = (saved_legacy[f"candidate:centered:{field_name}:total"]
                              - saved_legacy[f"reference_evolution:{field_name}:total"])
            for term in ("material", "remainder", "total"):
                rows.append(_row("P06_legacy", f"centered:{field_name}:{term}", replay_centered[term],
                                 saved_legacy[f"candidate:centered:{field_name}:{term}"], archived_error))
                rows.append(_row("P06_legacy", f"U:{field_name}:{term}", replay_u[term],
                                 saved_legacy[f"candidate:U:{field_name}:{term}"], archived_error))

    if "p07" in campaigns and "p07_global_N" in out.get("p07", {}):
        with np.load(paths["p07"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_action = z["action"][owner_arr]; saved_reference_midpoint = z["reference_midpoint"][owner_arr]
        replay_p07 = dense(out["p07"]["p07_global_N"]) / t.vol[owner_arr, None]
        rows.append(_row("P07", "global_N", replay_p07, saved_action, saved_action - saved_reference_midpoint))

    if "p07n" in campaigns and "p07n_global_N" in out.get("p07", {}):
        with np.load(paths["p07n"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_N = z["N"][owner_arr]; saved_D = z["D"][owner_arr]; saved_O_q3 = z["O_q3"][owner_arr]
            saved_N_minus_O = z["N_minus_O"][owner_arr]
        replay_N = dense(out["p07"]["p07n_global_N"]) / t.vol[owner_arr, None]
        replay_D = dense(out["p07"]["p07n_global_D"]) / t.vol[owner_arr, None]
        replay_O = dense(out["p07"]["p07n_global_O_q3"]) / t.vol[owner_arr, None]
        rows.append(_row("P07N", "global_N", replay_N, saved_N, saved_N_minus_O))
        rows.append(_row("P07N", "global_D", replay_D, saved_D, saved_N_minus_O))
        rows.append(_row("P07N", "global_O_q3", replay_O, saved_O_q3, saved_N_minus_O))

    return rows


#: Every oracle path this module's :func:`compare_to_oracle` reads, used only
#: to decide (cheaply, via ``Path.exists``) whether a grid/oracle-root
#: combination actually has the saved arrays this check needs -- never
#: opened here. Templated by ``{n}`` -- grid-generic (every frozen campaign
#: saved this file at N32, N48 *and* N64).
_ORACLE_PROBE_FILES = {
    "p05": ("p05", "N{n}.owner_results.npz"),
    "p05n_frozen": ("p05n_frozen", "N{n}.raw.npz"),
    "p07": ("p07", "N{n}.global.npz"),
}


def oracle_available(paths: dict, campaigns: tuple, n: int = 32) -> bool:
    """Whether ``paths`` actually has the grid-``n`` oracle arrays
    :func:`compare_to_oracle` needs. ``n`` defaults to 32 only for backward
    -compatible callers that never pass it explicitly (every current caller
    in this repo passes ``n`` explicitly -- see
    :mod:`p08_step1_global.campaign`'s ``preflight_grid``)."""
    for campaign, (key, filename_template) in _ORACLE_PROBE_FILES.items():
        if campaign in campaigns and not (paths[key] / filename_template.format(n=n)).is_file():
            return False
    return True


def run_owner_closure_check(*, n: int, input_root: Path, sidecar_path: Path, paths: dict, campaigns: tuple,
                            compare: bool) -> dict:
    """The full bounded owner-closure check (task report): build ``env``,
    select owners, build only their incident rows, run every campaign's own
    replay-unit arithmetic, and -- when ``compare`` -- diff against each
    frozen oracle at exactly those owners. Returns a JSON-safe payload
    (``selection``, ``row_counts``, and, when ``compare``, ``table``/
    ``all_pass``) -- this package's ``preflight_grid`` writes ``selection``
    to ``<output>/N{n}/owner_closure_selection.json`` as the frozen fixture
    and folds ``all_pass`` into its own."""
    import time as _time

    started = _time.time()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path))
    t = env.t; census = env.census

    fixture = selection_fixture(t, census)
    unique_owners = fixture["owners"]

    provider = load_provider_for_env(sidecar_path)
    built = build_owner_rows(env, unique_owners, provider=provider)
    payload = {
        "n": n, "selection": fixture,
        "row_counts": {"raw_ids": int(len(built["raw_ids"])), "face_row_indices": int(len(built["face_row_indices"])),
                      "p07_row_indices": int(len(built["p07_row_indices"]))},
    }
    if not compare:
        payload["seconds"] = _time.time() - started
        return payload

    oracle = ru._load_oracle_owner_values(env, paths, campaigns)
    out = assemble_owner_terms(env, built, campaigns, oracle)
    table = compare_to_oracle(env, out, unique_owners, paths, campaigns)
    payload["table"] = table
    payload["all_pass"] = all(r["pass"] for r in table)
    payload["seconds"] = _time.time() - started
    return payload
