"""Tests for ``drbx.stencils.builder`` (P08 step 1, task 4 -- see
``work/p08_step1_consolidation_design_20260928/design.md`` sections 2, 3, 5
and 6).

Two independent groups:

* **Tier A** (design section 5): on real N32 targets sampled to span every
  row family -- ``collapsed_r0``, ``singleton``, ``ringwise``,
  ``coupled_quartic``, ``boundary_transverse`` (D and N, degree 3),
  ``quartic_wall`` (D and N, degree 4), ``centered_radial``, and P07
  integrated-row families 0-7 -- this asserts that the artifact rows
  ``drbx.stencils.builder`` produces, packed and expanded through
  ``drbx.stencils.artifact``, are *bitwise* equal (donor ids and
  coefficients) to calling the same frozen ``drbx.geometry`` builders
  directly with the parameters the accepted campaigns use (as read from
  ``scripts/p05n_field_derived_global/core.py``/``rows.py``,
  ``scripts/p06n_field_derived_global/core.py``/``rows.py`` and
  ``scripts/p07n_field_derived_global/core.py`` -- see those modules'
  docstrings and this file's own comments for the exact call sites each
  assertion mirrors). Skipped cleanly (module-wide) if the local HSX
  workspace inputs are not present, the same way
  ``tests/test_stencils_geometry_provider.py`` locates them.
* Synthetic runner tests: resume (already-valid units are skipped, not
  recomputed), identity rejection, and chunk-corruption rejection, against
  a tiny fake build that needs no real geometry at all.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from drbx.geometry.fci_perpendicular_reconstruction import (  # noqa: E402
    PointRowContext, StructuredReconstruction)
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows  # noqa: E402
from drbx.geometry.fci_perpendicular_integrated_rows import (  # noqa: E402
    contract_face_tensor, prepare_integrated_face_rows)
from drbx.stencils.census import FaceCensus  # noqa: E402
from drbx.stencils import builder  # noqa: E402
from drbx.stencils import artifact as artifact_mod  # noqa: E402
from p_shared import runner  # noqa: E402
from p_shared import build_artifact  # noqa: E402

N = 32
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"


def _workspace_inputs_available() -> bool:
    if not (GEOMETRY / f"{N}x{N}x{N}" / "rlp_topology.npz").is_file():
        return False
    if not SIDECAR.is_file():
        return False
    try:
        side = json.loads(SIDECAR.read_text())
        for key in ("metric_cache", "makegrid"):
            if not Path(side[key]["path"]).is_file():
                return False
    except Exception:
        return False
    return True


needs_workspace = pytest.mark.skipif(
    not _workspace_inputs_available(),
    reason="HSX workspace geometry_artifacts / localized sidecar / metric cache are unavailable",
)


# ---------------------------------------------------------------------------
# Real-N32 fixtures (module-scoped: loaded once for every Tier A test).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_n32():
    if not _workspace_inputs_available():
        pytest.skip("workspace inputs unavailable")
    from perpendicular_structured.reconstruction import load_context
    from p_shared import provider as p_shared_provider

    t = load_context(N, str(WORKSPACE))
    context = build_artifact._make_context(t)
    S = StructuredReconstruction(context)
    census = FaceCensus.build(N, t.ro)
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="fd")
    normal_coefficients = build_artifact._normal_coefficients_fn(provider.reference)
    return {"t": t, "context": context, "S": S, "census": census, "provider": provider,
            "normal_coefficients": normal_coefficients}


@pytest.fixture(scope="module")
def radial_regions(real_n32):
    """Radial layers whose 4-wide reconstruction window (design's R1/R2
    windows: axis0 ``i-2..i+1``, transverse ``i-1..i+2``) sits entirely
    within one ``StructuredReconstruction.rows`` regime, found from the real
    owner profile -- never hand-picked, so this works regardless of the
    specific HSX aggregation numbers."""
    t = real_n32["t"]
    n = t.n
    profile = np.array([len(np.unique(t.ro.reshape(n, n, n)[i, :, 0])) for i in range(n)])

    def window(i, axis):
        layers = np.arange(i - 2, i + 2) if axis == 0 else np.arange(i - 1, i + 3)
        return np.where(layers < 0, -layers - 1, layers)

    def find(predicate, axis, hi):
        for i in range(2, hi):
            if predicate(profile[window(i, axis)]):
                return i
        raise AssertionError(f"no radial index satisfies the requested profile region below {hi}")

    boundary_hi = n - 6  # StructuredReconstruction.rows only takes this branch below n-6 (axis0) / n-2 (else)
    singleton_i = find(lambda p: np.all(p == n), axis=0, hi=boundary_hi)
    coupled_i = find(lambda p: np.min(p) < 7, axis=0, hi=boundary_hi)
    ringwise_i = None
    for i in range(2, boundary_hi):
        p = profile[window(i, 0)]
        if np.min(p) >= 7 and not np.all(p == n):
            ringwise_i = i
            break
    if ringwise_i is None:
        pytest.skip("no radial index in this grid falls in the ringwise (non-singleton, non-coupled) regime")
    return {"singleton": singleton_i, "coupled": coupled_i, "ringwise": ringwise_i, "n": n}


# ---------------------------------------------------------------------------
# R1: raw-midpoint cell rows.
# ---------------------------------------------------------------------------
def _assert_point_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.gradient, expected.gradient)
    assert bool(actual.boundary_conditioned) == bool(expected.boundary_conditioned)
    # drbx.stencils.artifact's pack/expand round trip only preserves
    # trace_donor_points for boundary-conditioned rows (documented there: an
    # unconditioned row -- including an unconditioned 'centered_radial' row,
    # whose own r.boundary_map call still populates a trace_donor array even
    # though it is never used at that BC -- expands back with an empty
    # (0, 3) placeholder instead). The task's own bitwise criterion is donor
    # ids and coefficients (value/gradient), which this still checks
    # unconditionally.
    if actual.boundary_conditioned:
        np.testing.assert_array_equal(actual.trace_donor_points, expected.trace_donor_points)
    np.testing.assert_array_equal(actual.trace_target_points, expected.trace_target_points)


def _assert_neumann_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.gradient, expected.gradient)
    np.testing.assert_array_equal(actual.boundary_points, expected.boundary_points)
    np.testing.assert_array_equal(actual.boundary_value, expected.boundary_value)
    np.testing.assert_array_equal(actual.boundary_gradient, expected.boundary_gradient)
    assert actual.condition == expected.condition
    assert actual.constraint_residual == expected.constraint_residual


def _assert_integrated_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.weights, expected.weights)
    assert bool(actual.boundary_conditioned) == bool(expected.boundary_conditioned)
    # Same artifact.py round-trip contract as _assert_point_rows_equal:
    # trace_donor_points is only preserved for boundary-conditioned rows
    # (an unconditioned P07 family, e.g. 3 = centered_radial_cubic, still
    # gets a populated trace_donor array from r.boundary_map, but expand()
    # deliberately drops it for an unconditioned row).
    if actual.boundary_conditioned:
        np.testing.assert_array_equal(actual.trace_donor_points, expected.trace_donor_points)
        np.testing.assert_array_equal(actual.value_loading, expected.value_loading)
        np.testing.assert_array_equal(actual.tangential_loading, expected.tangential_loading)
    np.testing.assert_array_equal(actual.trace_target_points, expected.trace_target_points)
    assert actual.family == expected.family


@needs_workspace
@pytest.mark.slow
def test_r1_cell_rows_bitwise_equal_across_families(real_n32, radial_regions):
    """Mirrors ``p05n_field_derived_global.core.batched_cell_values``: one
    ``S.rows(key, points, location='cell')`` call per raw cell; interior
    cells get one row, boundary cells (radial >= n-2) get a D row plus a
    shared Neumann companion (radial degree 3, batched over every boundary
    point in the call -- see ``batched_cell_values``'s own ``bpoints``)."""
    t, context, S = real_n32["t"], real_n32["context"], real_n32["S"]
    n = t.n
    nc, patch_cache = real_n32["normal_coefficients"], {}

    keys = [
        (radial_regions["singleton"], 3, 5),
        (radial_regions["ringwise"], 3, 5),
        (radial_regions["coupled"], 3, 5),
        (n - 1, 3, 5),   # boundary_transverse (conditioned)
    ]
    raw_ids = np.array([np.ravel_multi_index(k, (n, n, n)) for k in keys], dtype=np.int64)

    point_rows, neumann_rows = builder.build_r1_cell_rows(
        S, context, raw_ids, normal_coefficients=nc, patch_cache=patch_cache)
    assert len(point_rows) == 4
    families = {r.row.diagnostics.get("family") for r in point_rows}
    assert families == {"singleton", "ringwise", "coupled_quartic", "boundary_transverse"}
    assert sum(r.bc_variant == "D" for r in point_rows) == 1
    assert len(neumann_rows) == 1  # one boundary cell -> one shared Neumann companion

    point_chunk = artifact_mod.pack_point_rows(
        [r.row for r in point_rows], request=[r.request for r in point_rows],
        entity_id=[r.entity_id for r in point_rows], bc_variant=[r.bc_variant for r in point_rows],
        radial_degree=[r.radial_degree for r in point_rows])
    expanded = artifact_mod.expand_point_rows(point_chunk)
    for actual, req in zip(expanded, point_rows, strict=True):
        _assert_point_rows_equal(actual, req.row)

    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    for actual, req in zip(artifact_mod.expand_neumann_rows(neumann_chunk), neumann_rows, strict=True):
        _assert_neumann_rows_equal(actual, req.row)

    # Reference: direct calls with the exact parameters batched_cell_values uses.
    for j, key in enumerate(keys):
        expected = S.rows(key, context.pts[raw_ids[j]:raw_ids[j] + 1], location="cell")
        _assert_point_rows_equal(point_rows[j].row, expected)
    boundary_point = context.pts[raw_ids[3]:raw_ids[3] + 1]
    expected_n = prepare_neumann_point_rows(context, boundary_point, normal_coefficients=nc,
                                           radial_degree=3, patch_cache={})[0]
    _assert_neumann_rows_equal(neumann_rows[0].row, expected_n)


# ---------------------------------------------------------------------------
# R2 / R3: face-common and face-side rows.
# ---------------------------------------------------------------------------
def _face_points(real_n32, key):
    """The 9 q3 nodes of one face, via the same shared quadrature
    ``ScriptsGeometryProvider.face_node_weight`` the campaigns/geometry
    arrays use."""
    provider = real_n32["provider"]
    t = real_n32["t"]
    points, _weight = provider.face_node_weight(t.faces, np.asarray([key], dtype=np.int64))
    return points[0]


@needs_workspace
@pytest.mark.slow
def test_r2_face_rows_bitwise_equal_across_families(real_n32, radial_regions):
    """Mirrors ``p05n_field_derived_global.core.batched_face_common_gradient``
    / ``p06n_field_derived_global.rows.batched_face_common_value``: one
    ``S.rows(key, points, location='face')`` per census face; boundary faces
    (``quartic_wall`` degree 4, ``boundary_transverse`` degree 3) get a
    Neumann companion."""
    t, context, S, census = real_n32["t"], real_n32["context"], real_n32["S"], real_n32["census"]
    n = t.n
    nc, patch_cache = real_n32["normal_coefficients"], {}

    face_keys = [
        (0, n, 3, 5),                     # quartic_wall (physical wall)
        (0, n - 4, 3, 5),                 # centered_radial
        (1, radial_regions["singleton"], n - 1, 5),   # boundary_transverse (transverse, i = radial = singleton region here reused for a valid n-2.. check below)
        (0, radial_regions["singleton"], 3, 5),       # interior singleton face
        (0, radial_regions["ringwise"], 3, 5),        # interior ringwise face
        (0, radial_regions["coupled"], 3, 5),         # interior coupled face
    ]
    # The boundary_transverse target must itself have radial index >= n-2.
    face_keys[2] = (1, n - 1, 3, 5)

    keys_arr = np.column_stack([np.asarray(k) for k in face_keys]).T if False else np.asarray(face_keys)
    row_index_of = {}
    census_keys = census.keys()
    for fk in face_keys:
        matches = np.flatnonzero(np.all(census_keys == np.asarray(fk), axis=1))
        assert len(matches) == 1, f"census key {fk} not found uniquely"
        row_index_of[fk] = int(matches[0])
    row_indices = np.array([row_index_of[fk] for fk in face_keys], dtype=np.int64)
    face_points = np.stack([_face_points(real_n32, fk) for fk in face_keys])

    point_rows, neumann_rows = builder.build_r2_face_rows(
        S, context, census, row_indices, face_points, normal_coefficients=nc, patch_cache=patch_cache)
    kinds = {builder.face_kind(n, fk) for fk in face_keys[:3]}
    assert kinds == {"quartic_wall", "centered_radial", "boundary_transverse"}
    families = {r.row.diagnostics.get("family") for r in point_rows[3:]}
    assert families == {"singleton", "ringwise", "coupled_quartic"}

    point_chunk = artifact_mod.pack_point_rows(
        [r.row for r in point_rows], request=[r.request for r in point_rows],
        entity_id=[r.entity_id for r in point_rows], bc_variant=[r.bc_variant for r in point_rows],
        radial_degree=[r.radial_degree for r in point_rows])
    expanded = artifact_mod.expand_point_rows(point_chunk)
    for actual, req, fk in zip(expanded, point_rows, face_keys, strict=True):
        expected = S.rows(fk, _face_points(real_n32, fk), location="face")
        _assert_point_rows_equal(actual, expected)
        _assert_point_rows_equal(req.row, expected)

    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    expanded_n = artifact_mod.expand_neumann_rows(neumann_chunk)
    by_entity = {}
    for req, actual in zip(neumann_rows, expanded_n, strict=True):
        by_entity.setdefault(req.entity_id, []).append((req, actual))
    # quartic_wall (degree 4) and boundary_transverse (degree 3) each got a companion.
    wall_ridx = row_index_of[(0, n, 3, 5)]
    trans_ridx = row_index_of[(1, n - 1, 3, 5)]
    assert wall_ridx in by_entity and trans_ridx in by_entity
    wall_expected = prepare_neumann_point_rows(context, _face_points(real_n32, (0, n, 3, 5)),
                                               normal_coefficients=nc, radial_degree=4, patch_cache={})
    trans_expected = prepare_neumann_point_rows(context, _face_points(real_n32, (1, n - 1, 3, 5)),
                                                normal_coefficients=nc, radial_degree=3, patch_cache={})
    for (req, actual), expected in zip(sorted(by_entity[wall_ridx], key=lambda x: x[0].quad_node), wall_expected, strict=True):
        assert req.radial_degree == 4
        _assert_neumann_rows_equal(actual, expected)
    for (req, actual), expected in zip(sorted(by_entity[trans_ridx], key=lambda x: x[0].quad_node), trans_expected, strict=True):
        assert req.radial_degree == 3
        _assert_neumann_rows_equal(actual, expected)


@needs_workspace
@pytest.mark.slow
def test_collapsed_r0_is_a_zero_donor_row_and_is_excluded_from_r2_selection(real_n32):
    """``S.rows`` at the collapsed r=0 face returns a documented zero-donor,
    unconditioned row (design section 2's R2 entry); the census-row
    selection R2/R3 iterate over must exclude it (design: "collapsed r=0 and
    alias slots excluded")."""
    t, context, S, census = real_n32["t"], real_n32["context"], real_n32["S"], real_n32["census"]
    n = t.n
    key = (0, 0, 4, 6)
    points = _face_points(real_n32, key)
    row = S.rows(key, points, location="face")
    assert row.diagnostics.get("family") == "collapsed_r0"
    assert len(row.donor_ids) == 0
    assert row.boundary_conditioned is False

    selection = build_artifact.face_row_selection(census)
    census_keys = census.keys()[selection]
    assert not np.any((census_keys[:, 0] == 0) & (census_keys[:, 1] == 0))
    assert not np.any(census.legacy_alias_slots[selection])


@needs_workspace
@pytest.mark.slow
def test_r3_side_rows_share_one_neumann_companion_between_lower_and_upper(real_n32):
    """Design section 2, R3: "N: degree 3, target-anchored, the same row for
    both sides." Verified against direct ``S.side_rows``/
    ``prepare_neumann_point_rows`` calls, mirroring
    ``p05n_field_derived_global.core.batched_side_values``."""
    t, context, S, census = real_n32["t"], real_n32["context"], real_n32["S"], real_n32["census"]
    n = t.n
    key = (0, n - 1, 3, 5)  # physical wall face: lower cell (n-2) and upper cell (n-1) both conditioned
    points = _face_points(real_n32, key)
    nc, patch_cache = real_n32["normal_coefficients"], {}

    census_keys = census.keys()
    ridx = int(np.flatnonzero(np.all(census_keys == np.asarray(key), axis=1))[0])

    point_rows, neumann_rows = builder.build_r3_side_rows(
        S, context, census, np.array([ridx]), points[None], normal_coefficients=nc, patch_cache=patch_cache)
    assert len(point_rows) == 2  # lower + upper, both boundary-conditioned here
    assert all(r.bc_variant == "D" for r in point_rows)
    assert len(neumann_rows) == 9  # one shared row per q3 node, not one per side

    lower, upper = S.side_rows(key, points)
    _assert_point_rows_equal(point_rows[0].row, lower)
    _assert_point_rows_equal(point_rows[1].row, upper)

    expected_n = prepare_neumann_point_rows(context, points, normal_coefficients=nc, radial_degree=3, patch_cache={})
    for req, expected in zip(sorted(neumann_rows, key=lambda r: r.quad_node), expected_n, strict=True):
        _assert_neumann_rows_equal(req.row, expected)
    # The same NeumannPointRows objects are what both sides would read at apply time
    # (batched_side_values passes the identical `points` to both its left/right calls).
    for req, expected in zip(sorted(neumann_rows, key=lambda r: r.quad_node), expected_n, strict=True):
        assert req.entity_id == ridx


# ---------------------------------------------------------------------------
# R4: P07 integrated face rows, families 0-7.
# ---------------------------------------------------------------------------
@needs_workspace
@pytest.mark.slow
def test_r4_integrated_rows_bitwise_equal_across_all_p07_families(real_n32):
    """Mirrors ``scripts/p07n_field_derived_global/core.py``'s ``face_chunk``:
    ``prepare_integrated_face_rows`` at every P07-census face, plus a
    Neumann companion for boundary families 1/2/4 (``degree = 3 if fam == 4
    else 4``)."""
    t, context, S, census = real_n32["t"], real_n32["context"], real_n32["S"], real_n32["census"]
    provider = real_n32["provider"]
    nc, patch_cache = real_n32["normal_coefficients"], {}

    p07_row_indices = builder.p07_row_selection(census)
    families_present = np.unique(census.family[p07_row_indices])
    assert set(families_present.tolist()) == set(range(8)), "sample grid does not cover every P07 family 0-7"

    chosen = []
    for code in range(8):
        candidates = p07_row_indices[census.family[p07_row_indices] == code]
        chosen.append(int(candidates[len(candidates) // 2]))
    chosen = np.array(sorted(set(chosen)), dtype=np.int64)

    census_keys = census.keys()
    face_keys = census_keys[chosen]
    face_points = np.stack([_face_points(real_n32, tuple(int(v) for v in fk)) for fk in face_keys])
    # Family 0 (collapsed_r0) sits at u == 0, where the ordinary metric is
    # singular; prepare_integrated_face_rows never reads its integrand (see
    # its own `if code == 0` branch), and the frozen
    # p07n_field_derived_global.core.face_chunk mirrors this by only
    # querying the tensor at `active = flatnonzero(fam != 0)`. This test
    # does the same: zero-filled placeholder tensor/weight at family 0.
    family_here = census.family[chosen].astype(np.int64)
    face_weight = np.zeros((len(face_keys), 9))
    face_tensor = np.zeros((len(face_keys), 9, 3, 3))
    active = np.flatnonzero(family_here != 0)
    if len(active):
        _pts, w = provider.face_node_weight(t.faces, face_keys[active])
        face_weight[active] = w
        face_tensor[active] = provider.p07_perpendicular_tensor(_pts.reshape(-1, 3)).reshape(len(active), 9, 3, 3)

    integrated_rows, neumann_rows = builder.build_r4_p07_rows(
        context, census, chosen, face_points, face_weight, face_tensor,
        normal_coefficients=nc, patch_cache=patch_cache)
    assert {r.row.family for r in integrated_rows} == set(range(8))

    chunk = artifact_mod.pack_integrated_rows(
        [r.row for r in integrated_rows], entity_id=[r.entity_id for r in integrated_rows])
    expanded = artifact_mod.expand_integrated_rows(chunk)

    family_arr = census.family[chosen].astype(np.int64)
    integ = contract_face_tensor(face_weight, face_tensor, face_keys[:, 0])
    expected_rows = prepare_integrated_face_rows(context, face_keys, family_arr, face_points, integ)
    for actual, expected in zip(expanded, expected_rows, strict=True):
        _assert_integrated_rows_equal(actual, expected)

    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    expanded_n = artifact_mod.expand_neumann_rows(neumann_chunk)
    degree_by_family = {1: 4, 2: 4, 4: 3}
    p07_ids = census.p07_id[chosen]
    for req, actual in zip(neumann_rows, expanded_n, strict=True):
        j = int(np.flatnonzero(p07_ids == req.entity_id)[0])
        fam = int(family_arr[j])
        assert fam in degree_by_family
        expected = prepare_neumann_point_rows(context, face_points[j], normal_coefficients=nc,
                                              radial_degree=degree_by_family[fam], patch_cache={})[req.quad_node]
        _assert_neumann_rows_equal(actual, expected)


# ---------------------------------------------------------------------------
# Synthetic runner: resume, identity rejection, corruption rejection.
# ---------------------------------------------------------------------------
_TOY_STATE: dict = {}


def _toy_initializer(output, identity):
    global _TOY_STATE
    _TOY_STATE = {"output": output, "identity": identity}


def _toy_compute(unit):
    started = time.time()
    chunks = {"chunk": {"value": np.array([unit["start"], unit["stop"]], dtype=np.int64)},
              "aux": {"value": np.array([unit["stop"] - unit["start"]], dtype=np.int64)}}
    return runner.write_unit(_TOY_STATE["output"], unit, _TOY_STATE["identity"], chunks=chunks, started=started)


def test_runner_plan_resume_and_rejection(tmp_path):
    units = runner.chunk_units("toy", 8, count=10, chunk_size=3)
    assert len(units) == 4

    summary = runner.run_stage(tmp_path, "toy", units, "identity-1", compute=_toy_compute,
                               initializer=_toy_initializer, initargs=(str(tmp_path), "identity-1"),
                               workers=2, parts=("chunk", "aux"))
    assert summary["executed_units"] == 4
    assert summary["resumed_units"] == 0
    for u in units:
        assert runner.valid_unit(tmp_path, u, "identity-1", parts=("chunk", "aux"))

    # Resume: every unit is already valid, nothing is recomputed.
    summary2 = runner.run_stage(tmp_path, "toy", units, "identity-1", compute=_toy_compute,
                                initializer=_toy_initializer, initargs=(str(tmp_path), "identity-1"),
                                workers=2, parts=("chunk", "aux"))
    assert summary2["executed_units"] == 0
    assert summary2["resumed_units"] == 4

    # A changed identity is rejected, not silently accepted.
    with pytest.raises(ValueError, match="stale checkpoint"):
        runner.valid_unit(tmp_path, units[0], "identity-2", parts=("chunk", "aux"))

    # A corrupted chunk file is rejected.
    path = runner.unit_path(tmp_path, units[0], "chunk")
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        runner.valid_unit(tmp_path, units[0], "identity-1", parts=("chunk", "aux"))


def test_runner_missing_part_is_invalid(tmp_path):
    units = runner.chunk_units("toy2", 8, count=3, chunk_size=3)
    runner.run_stage(tmp_path, "toy2", units, "identity-1", compute=_toy_compute,
                     initializer=_toy_initializer, initargs=(str(tmp_path), "identity-1"),
                     workers=1, parts=("chunk", "aux"))
    # Asking for a part that was never requested/written is not silently "valid".
    assert runner.valid_unit(tmp_path, units[0], "identity-1", parts=("chunk", "aux")) is True
    with pytest.raises(ValueError, match="part set mismatch"):
        runner.valid_unit(tmp_path, units[0], "identity-1", parts=("chunk", "aux", "extra"))


def test_build_identity_helper_is_json_safe_and_deterministic():
    identity = artifact_mod.build_identity(
        component_hashes={"geometry:base_geometry.npz": "a" * 64},
        source_hashes={"src/drbx/stencils/builder.py": "b" * 64},
        policy=build_artifact.POLICY)
    assert json.loads(json.dumps(identity, sort_keys=True)) == identity
    again = artifact_mod.build_identity(
        component_hashes={"geometry:base_geometry.npz": "a" * 64},
        source_hashes={"src/drbx/stencils/builder.py": "b" * 64},
        policy=build_artifact.POLICY)
    assert identity == again
