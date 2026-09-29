"""Tests for ``scripts/p_shared/owner_closure.py`` -- the bounded owner
-closure selection/row-building/comparison this task added (see the task
report). Two layers, matching this repo's own convention
(``tests/test_p_shared_selection.py``/``tests/test_p_shared_replay_support
.py``): fully synthetic tests for the small, geometry-independent helpers,
and a skip-gated real-N32-geometry/oracle test for the end-to-end selection
+ row-build + campaign comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import owner_closure as oc  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic: the small, geometry-independent helpers.
# ---------------------------------------------------------------------------
def _synthetic_census(owner_lo, owner_hi, family):
    """A minimal stand-in for ``drbx.stencils.census.FaceCensus`` carrying
    only the three arrays :func:`incident_census_rows`/
    :func:`_first_mixed_wall_owner` read."""
    return SimpleNamespace(owner_lo=np.asarray(owner_lo, dtype=np.int64),
                           owner_hi=np.asarray(owner_hi, dtype=np.int64),
                           family=np.asarray(family, dtype=np.int64))


def test_incident_census_rows_matches_either_side():
    census = _synthetic_census(owner_lo=[0, 1, 2, -1, 3], owner_hi=[1, 2, 3, 0, 4], family=[5, 5, 5, 1, 2])
    rows = oc.incident_census_rows(census, [2])
    # rows 1 (hi=2) and 2 (lo=2) are incident to owner 2; nothing else is.
    np.testing.assert_array_equal(rows, [1, 2])


def test_incident_census_rows_dedupes_and_sorts_multi_owner_query():
    census = _synthetic_census(owner_lo=[0, 1, 2, 3], owner_hi=[1, 2, 3, 4], family=[5, 5, 5, 5])
    rows = oc.incident_census_rows(census, [3, 1, 1])  # duplicate owner id in the query
    np.testing.assert_array_equal(rows, [0, 1, 2, 3])


def test_first_mixed_wall_owner_picks_smallest_owner_with_both_kinds():
    # Owner 5: only conditioned (family 1) faces -- "fully" wall-adjacent, not what we want.
    # Owner 6: one conditioned (family 2) face and one unconditioned (family 5) face -- "partial".
    # Owner 9: same mixed pattern as owner 6 but a larger id -- must not be picked.
    census = _synthetic_census(
        owner_lo=[5, 5, 6, 6, 9, 9],
        owner_hi=[-1, -1, 7, -1, 7, -1],
        family=[1, 1, 2, 5, 2, 5],
    )
    assert oc._first_mixed_wall_owner(census) == 6


def test_first_mixed_wall_owner_raises_when_none_exists():
    census = _synthetic_census(owner_lo=[0, 1], owner_hi=[1, 2], family=[5, 5])  # no conditioned face at all
    with pytest.raises(ValueError):
        oc._first_mixed_wall_owner(census)


def test_owner_values_from_pairs_scatters_and_selects_in_order():
    uniq = np.array([3, 1, 4], dtype=np.int64)
    values = np.array([[30.0, 31.0], [10.0, 11.0], [40.0, 41.0]])
    dense = oc.owner_values_from_pairs((uniq, values), owners=[4, 3, 2], n_owners=6)
    np.testing.assert_allclose(dense, [[40.0, 41.0], [30.0, 31.0], [0.0, 0.0]])


def test_row_jsontable_pass_uses_absolute_floor_when_oracle_error_is_zero():
    """A degenerate/trivial oracle (its own |N-R| is exactly zero) still
    passes when the replay-vs-saved difference is at or below the absolute
    roundoff floor -- the false-positive this task's own preflight ran into
    for P06N's "control_constant" variant (see the task report) before this
    fallback was added."""
    row = oc._row("C", "t", replay=np.array([1e-12]), saved=np.array([0.0]), archived_error=np.array([0.0]))
    assert row["pass"] is True
    assert row["ratio_to_oracle_NR"] == 0.0 or np.isfinite(row["ratio_to_oracle_NR"])


def test_oracle_available_is_keyed_by_n_not_hardcoded_to_32(tmp_path):
    """``oracle_available``'s probe files are templated by ``{n}`` (task:
    grid-generic oracle paths) -- a tree with only N48 files must report
    available at n=48 and unavailable at n=32, and vice versa."""
    campaigns = ("p05", "p05n_frozen", "p07")
    root = tmp_path
    (root / "p05").mkdir()
    (root / "p05n_frozen").mkdir()
    (root / "p07").mkdir()
    (root / "p05" / "N48.owner_results.npz").write_bytes(b"")
    (root / "p05n_frozen" / "N48.raw.npz").write_bytes(b"")
    (root / "p07" / "N48.global.npz").write_bytes(b"")
    paths = {"p05": root / "p05", "p05n_frozen": root / "p05n_frozen", "p07": root / "p07"}

    assert oc.oracle_available(paths, campaigns, n=48) is True
    assert oc.oracle_available(paths, campaigns, n=32) is False
    assert oc.oracle_available(paths, campaigns, n=64) is False


# ---------------------------------------------------------------------------
# Geometry-array sourcing: ``_owner_geometry_arrays`` batches the same split
# ``build_raw_geometry_arrays``/``build_face_geometry_arrays`` functions the
# production parallel geometry stage uses, at the same 4096-row chunk size --
# fully synthetic (a deterministic fake provider, never real geometry), so
# this exercises the batching/concatenation logic itself, independent of any
# real metric evaluator.
# ---------------------------------------------------------------------------
class _FakeProvider:
    """A deterministic stand-in ``GeometryProvider``: every field is some
    fixed, elementwise function of the query point, so concatenating batches
    of any size reproduces exactly one call over the whole array (no
    batch-size-dependent numerics -- this test is about the *plumbing*, not
    about the real metric evaluator's own batch-size roundoff, which
    ``p_shared.owner_closure``'s module docstring documents separately)."""

    def raw_cell_weight(self, faces, keys):
        keys = np.asarray(keys, dtype=np.float64)
        points = (keys.sum(axis=1, keepdims=True) * 0.01)[:, None, :] * np.ones((1, 1, 3))
        weight = np.full((len(keys), 1), 0.5)
        return points, weight

    def face_node_weight(self, faces, keys):
        keys = np.asarray(keys, dtype=np.float64)
        base = (keys.sum(axis=1) * 0.01)
        points = base[:, None, None] * np.ones((1, 9, 3)) + np.arange(9)[None, :, None] * 0.001
        weight = np.full((len(keys), 9), 0.25)
        return points, weight

    def p05_metric(self, points):
        points = np.asarray(points)
        h = points * 2.0
        jac = points.sum(axis=-1)
        return h, jac

    def p06_curvature(self, points):
        points = np.asarray(points)
        J = points.sum(axis=-1)
        B = points.sum(axis=-1) + 1.0
        K = points * 3.0
        return J, B, K

    def p06_face_curvature(self, points):
        return self.p06_curvature(points)

    def p07_perpendicular_tensor(self, points):
        points = np.asarray(points)
        return points[:, :, None] * points[:, None, :]

    def p07_perpendicular_tensor_and_divergence(self, points):
        tensor = self.p07_perpendicular_tensor(points)
        divergence = np.asarray(points) * 5.0
        return tensor, divergence


def test_owner_geometry_arrays_batched_matches_single_unchunked_call():
    provider = _FakeProvider()
    rng = np.random.default_rng(0)
    raw_keys = rng.integers(0, 10, size=(9, 3)).astype(np.int64)
    face_keys = rng.integers(0, 10, size=(7, 4)).astype(np.int64)
    faces = None  # never dereferenced by _FakeProvider

    unchunked = oc._owner_geometry_arrays(provider, faces, raw_keys, face_keys,
                                         raw_chunk=1000, face_chunk=1000)
    chunked = oc._owner_geometry_arrays(provider, faces, raw_keys, face_keys,
                                       raw_chunk=2, face_chunk=3)

    for field in oc._RAW_GEOMETRY_FIELDS + oc._FACE_GEOMETRY_FIELDS:
        np.testing.assert_array_equal(getattr(unchunked, field), getattr(chunked, field),
                                      err_msg=f"field {field!r} differs between batchings")
    assert chunked.raw_points.shape[0] == 9
    assert chunked.face_points.shape[0] == 7
    unchunked.verify()  # identity round-trips for a real (non-frozen-dataclass-bypassing) instance
    chunked.verify()


def test_owner_geometry_arrays_handles_a_single_full_size_batch():
    """When the selected subset is smaller than the chunk size (always true
    for a dozen owners at N32/N48/N64 -- see the task report), the batched
    path degenerates to exactly one call, so this is the case the real
    owner-closure check exercises at every grid."""
    provider = _FakeProvider()
    raw_keys = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    face_keys = np.array([[0, 1, 2, 3]], dtype=np.int64)
    geometry = oc._owner_geometry_arrays(provider, None, raw_keys, face_keys)
    assert geometry.raw_points.shape[0] == 2
    assert geometry.face_points.shape[0] == 1
    geometry.verify()


def test_row_is_json_safe_even_for_a_genuine_failure():
    """A genuine mismatch against a zero-oracle-error term must still
    serialize (this payload is written to JSON with ``allow_nan=False`` by
    ``p08_step1_global.campaign``'s preflight) -- no ``inf``/``nan``."""
    import json

    row = oc._row("C", "t", replay=np.array([1.0, 2.0]), saved=np.array([0.0, 0.0]), archived_error=np.array([0.0, 0.0]))
    assert row["pass"] is False
    assert np.isfinite(row["ratio_to_oracle_NR"])
    assert np.isfinite(row["max_rel"])
    json.dumps(row, allow_nan=False)  # must not raise


# ---------------------------------------------------------------------------
# Skip-gated real N32/N48/N64 geometry/oracle: selection + row build +
# assembly + comparison, end to end -- grid-generic (task: "run the
# owner-closure check at N48 and N64 locally").
# ---------------------------------------------------------------------------
N = 32
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"


def _geometry_available(n: int) -> bool:
    directory = GEOMETRY / f"{n}x{n}x{n}"
    return (directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()


def _oracle_available(n: int) -> bool:
    from p_shared.replay_support import DEFAULT_PATHS, CAMPAIGN_FUNCS
    return oc.oracle_available(dict(DEFAULT_PATHS), CAMPAIGN_FUNCS, n=n) if _geometry_available(n) else False


needs_geometry = pytest.mark.skipif(not (_geometry_available(N) and SIDECAR.is_file()),
                                    reason="HSX N32 geometry/sidecar inputs are unavailable")
needs_oracle = pytest.mark.skipif(not _oracle_available(N),
                                  reason="the six frozen campaigns' N32 oracle arrays are unavailable")

REQUIRED_CATEGORIES = {
    "axis_core", "near_axis", "aggregate", "transition", "interior", "interior_secondary",
    "wall_partial", "wall_full", "wall_full_secondary", "theta_seam", "eta_seam", "theta_seam_wall",
}


@needs_geometry
@pytest.mark.slow
def test_select_owners_is_deterministic_and_covers_every_required_category():
    from p_shared.replay_support import build_environment

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR)
    owners_a = oc.select_owners(env.t, env.census)
    owners_b = oc.select_owners(env.t, env.census)
    assert owners_a == owners_b  # deterministic, no RNG/dict-order dependence
    assert set(owners_a) == REQUIRED_CATEGORIES

    fixture = oc.selection_fixture(env.t, env.census)
    assert fixture["schema"] == oc.SCHEMA
    assert fixture["n"] == N
    assert fixture["owners"] == sorted(set(owners_a.values()))
    assert len(fixture["owners"]) >= 10  # a handful of categories legitimately coincide
    assert fixture["census_row_count"] > 0


@needs_geometry
@pytest.mark.slow
def test_build_owner_rows_covers_every_selected_owners_incident_faces():
    from p_shared.replay_support import build_environment

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR)
    owners = sorted(set(oc.select_owners(env.t, env.census).values()))
    provider = oc.load_provider_for_env(SIDECAR)
    built = oc.build_owner_rows(env, owners, provider=provider)

    assert built["owners"] == owners
    assert len(built["raw_ids"]) > 0
    assert len(built["face_row_indices"]) > 0
    assert len(built["p07_row_indices"]) > 0
    # Every raw member of every selected owner is present.
    for owner in owners:
        assert np.any(env.t.ro[built["raw_ids"]] == owner)
    # Row lookups a real compute_cells_unit/compute_faces_unit/compute_p07_unit
    # would need are all present (no missing R1 row for any built raw id).
    for raw_id in built["raw_ids"]:
        assert ("R1", int(raw_id)) in built["row_index"]


@needs_oracle
@pytest.mark.slow
def test_run_owner_closure_check_passes_against_every_frozen_oracle():
    """The task report's own gate: build only the selected owners' rows, run
    every campaign's replay-unit arithmetic, and diff against each frozen
    oracle at exactly those owners -- must pass at roundoff level (see
    ``p_shared.owner_closure.RATIO_TOLERANCE``/``ABS_FALLBACK_TOLERANCE``)."""
    from p_shared.replay_support import DEFAULT_PATHS, CAMPAIGN_FUNCS

    payload = oc.run_owner_closure_check(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR,
                                         paths=dict(DEFAULT_PATHS), campaigns=CAMPAIGN_FUNCS, compare=True)
    failed = [r for r in payload["table"] if not r["pass"]]
    assert failed == []
    assert payload["all_pass"] is True
    assert len(payload["table"]) > 20  # every campaign x term/variant row actually ran


@pytest.mark.slow
@pytest.mark.parametrize("n", [48, 64])
def test_run_owner_closure_check_passes_at_n48_and_n64(n):
    """Grid-generic oracle comparison (task): the same gate as the N32 test
    above, but at N48/N64, where there is no on-disk production geometry --
    ``build_owner_rows`` must compute its own geometry for just the selected
    owners (:func:`p_shared.owner_closure._owner_geometry_arrays`), never
    read a grid's ``geometry.npz``."""
    from p_shared.replay_support import DEFAULT_PATHS, CAMPAIGN_FUNCS

    if not (_geometry_available(n) and SIDECAR.is_file()):
        pytest.skip(f"HSX N{n} geometry/sidecar inputs are unavailable")
    if not _oracle_available(n):
        pytest.skip(f"the six frozen campaigns' N{n} oracle arrays are unavailable")

    payload = oc.run_owner_closure_check(n=n, input_root=WORKSPACE, sidecar_path=SIDECAR,
                                         paths=dict(DEFAULT_PATHS), campaigns=CAMPAIGN_FUNCS, compare=True)
    failed = [r for r in payload["table"] if not r["pass"]]
    assert failed == []
    assert payload["all_pass"] is True
    assert len(payload["table"]) > 20
