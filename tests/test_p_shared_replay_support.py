"""Tests for ``scripts/p_shared/replay_support.py`` (P08 step-1 design task 7 --
see ``work/p08_step1_consolidation_design_20260928/design.md`` section 5
"Replay gate" and section 6, task 7). ``replay_support`` is what remains of
the retired ``scripts/p_shared/replay.py`` (deleted; see the task report):
the generic comparison core, ``RowIndex``, and the campaign environment --
everything ``scripts/p_shared/replay_units.py`` (the live replay path) still
imports.

Two independent groups, per the task spec ("tests for the comparison logic
on small synthetic/bounded inputs; skip cleanly if workspace data is
absent"):

* **Synthetic, always run** -- the Tier B ratio/pointwise-cap comparison
  core (``owner_weighted_l2``, ``pointwise_cap``, ``compare_owner_term``,
  ``compare_pointwise_only``), the owner-weighted scatter helper
  (``scatter_owner_weighted_mean``), and :class:`RowIndex`'s two
  constructors (``from_point_requests`` directly, and ``from_artifact``
  through a real ``drbx.stencils.artifact`` pack/save/load roundtrip) --
  none of this touches the HSX workspace or the row artifact.
* **Real-geometry, skipped cleanly if unavailable** -- builds a small
  owner/face subset *on the fly* via ``drbx.stencils.builder`` (never the
  multi-GB row artifact) and checks that :class:`RowIndex` built from that
  subset reproduces the same row objects the builder itself returned, and
  that applying an R1 row through it via ``p_shared.apply`` matches a
  direct application -- the same convention ``tests/test_p_shared_apply.py``
  and ``tests/test_stencils_builder.py`` already use to locate the
  workspace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from p_shared import replay_support as replay  # noqa: E402


# ---------------------------------------------------------------------------
# Tier B ratio / pointwise cap core -- fully synthetic.
# ---------------------------------------------------------------------------
def test_owner_weighted_l2_matches_hand_computation():
    diff = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    volume = np.array([1.0, 2.0, 3.0])
    expected = np.sqrt((1.0 * 1.0 + 2.0 * 4.0 + 3.0 * 25.0) / 6.0)
    assert replay.owner_weighted_l2(diff, volume) == pytest.approx(expected)


def test_owner_weighted_l2_masks_and_handles_1d():
    diff = np.array([1.0, 2.0, 3.0])
    volume = np.array([1.0, 1.0, 1.0])
    mask = np.array([True, False, True])
    expected = np.sqrt((1.0 + 9.0) / 2.0)
    assert replay.owner_weighted_l2(diff, volume, mask) == pytest.approx(expected)


def test_owner_weighted_l2_none_when_empty_or_zero_volume():
    diff = np.zeros((3, 2))
    volume = np.zeros(3)
    assert replay.owner_weighted_l2(diff, volume) is None
    assert replay.owner_weighted_l2(diff, np.array([1.0, 1.0, 1.0]), np.zeros(3, dtype=bool)) is None


def test_owner_weighted_l2_rejects_owner_axis_mismatch():
    with pytest.raises(ValueError):
        replay.owner_weighted_l2(np.zeros((3, 2)), np.zeros(4))


def test_pointwise_cap_scaled_flat_and_zero_target():
    saved = np.array([0.0, 1.0, -1000.0])
    # Scaled: the default reference is the array max (1-D), so every entry gets 1e-11 * 1000.
    cap = replay.pointwise_cap(saved, mode="scaled", scale_factor=1e-11, zero_atol=1e-11, zero_threshold=1e-11)
    np.testing.assert_allclose(cap, [1e-8, 1e-8, 1e-8])

    cap_flat = replay.pointwise_cap(saved, mode="flat", flat_abs=1e-9, zero_atol=1e-11, zero_threshold=1e-11)
    np.testing.assert_allclose(cap_flat, [1e-11, 1e-9, 1e-9])

    with pytest.raises(ValueError):
        replay.pointwise_cap(saved, mode="bogus")


def test_pointwise_cap_default_reference_is_per_column_max_over_owners():
    saved = np.array([[1e-7, 2.0, 0.0],
                      [-4.0, 1e-3, 0.0],
                      [0.5, -8.0, 0.0]])
    cap = replay.pointwise_cap(saved, mode="scaled", scale_factor=1e-11)
    column_max = np.array([4.0, 8.0, 0.0])
    expected = np.where(column_max[None, :] <= 1e-11, 1e-11, 1e-11 * np.broadcast_to(column_max, saved.shape))
    np.testing.assert_allclose(cap, expected)
    # An all-zero column falls back to the zero-target atol.
    np.testing.assert_allclose(cap[:, 2], 1e-11)


def test_pointwise_cap_explicit_reference_and_elementwise_max_with_saved():
    saved = np.array([[1.0, 100.0], [1e-9, 1e-9]])
    reference = np.array([[10.0, 10.0]])  # (1, ncols), broadcast over owners
    cap = replay.pointwise_cap(saved, mode="scaled", scale_factor=1e-2, cap_reference=reference)
    # scale = max(|saved|, reference) elementwise.
    np.testing.assert_allclose(cap, 1e-2 * np.array([[10.0, 100.0], [10.0, 10.0]]))
    # A 1-D (ncols,) reference broadcasts the same way.
    cap_1d = replay.pointwise_cap(saved, mode="scaled", scale_factor=1e-2, cap_reference=np.array([10.0, 10.0]))
    np.testing.assert_allclose(cap_1d, cap)
    # A zero reference and a zero-ish saved entry -> zero_atol; a large reference lifts a tiny entry out of it.
    zero = replay.pointwise_cap(np.array([[0.0], [0.0]]), mode="scaled", cap_reference=np.array([[0.0]]),
                                zero_atol=3e-11)
    np.testing.assert_allclose(zero, 3e-11)
    lifted = replay.pointwise_cap(np.array([[0.0]]), mode="scaled", scale_factor=1e-11, cap_reference=np.array([[5.0]]))
    np.testing.assert_allclose(lifted, 5e-11)


def test_pointwise_cap_flat_mode_ignores_reference_and_keys_zero_rule_on_saved():
    saved = np.array([[0.0, 1.0], [-1000.0, 1e-3]])
    cap = replay.pointwise_cap(saved, mode="flat", flat_abs=1e-9, zero_atol=1e-11, cap_reference=np.array([[1e6, 1e6]]))
    np.testing.assert_allclose(cap, [[1e-11, 1e-9], [1e-9, 1e-9]])


def test_pointwise_cap_scales_with_column_magnitude_at_cancellation_points():
    owner_volume = np.ones(4)
    # Column 0: entries near 1e-7 in a column whose terms are O(100); column 1: O(1).
    saved = np.array([[100.0, 1.0], [1e-7, 1.0], [-50.0, 1.0], [2e-2, 1.0]])
    archived_error = np.full_like(saved, 1e3)  # huge archived error: ratio never the limiting check
    roundoff = saved.copy()
    roundoff[1, 0] += 1e-14  # roundoff of the O(100) terms, 7 orders above the entry's own 1e-11 cap
    result = replay.compare_owner_term("demo", roundoff, saved, owner_volume=owner_volume,
                                       archived_error=archived_error)
    assert result["pointwise"]["violations"] == 0
    assert result["pass"]
    # Under the old entrywise rule this entry would have failed.
    assert 1e-14 > 1e-11 * abs(saved[1, 0])

    # A genuine 1e-9-relative error at a large entry still fails (cap there is 1e-11 * 100 = 1e-9).
    genuine = saved.copy()
    genuine[0, 0] *= 1.0 + 1e-9
    bad = replay.compare_owner_term("demo", genuine, saved, owner_volume=owner_volume, archived_error=archived_error)
    assert bad["pointwise"]["violations"] == 1
    assert bad["pointwise"]["worst_index"] == [0, 0]
    assert not bad["pass"]

    # An explicit constituent-scale reference (p05 live_jump_vs_old_U_minus_A) lifts the cap further.
    result_ref = replay.compare_owner_term("demo", genuine, saved, owner_volume=owner_volume,
                                           archived_error=archived_error,
                                           cap_reference=np.array([[1e5, 1.0]]))
    assert result_ref["pointwise"]["violations"] == 0


def test_compare_owner_term_passes_when_ratio_and_cap_are_satisfied():
    owner_volume = np.array([1.0, 1.0, 1.0, 1.0])
    saved = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    # Archived error much larger than the replay/saved difference -> tiny ratio.
    archived_error = saved * 10.0
    replay_arr = saved + 1e-6  # well within the default scaled pointwise cap given |saved|>=1
    region_masks = {"lower_half": np.array([True, True, False, False])}
    result = replay.compare_owner_term("demo", replay_arr, saved, owner_volume=owner_volume,
                                       archived_error=archived_error, region_masks=region_masks,
                                       cap_mode="scaled", cap_scale_factor=1.0)
    assert result["pass"]
    assert result["ratios"]["global"] is not None
    assert result["ratios"]["global"] < result["ratio_tolerance"]
    assert "lower_half" in result["region_detail"]
    assert result["pointwise"]["violations"] == 0


def test_compare_owner_term_fails_on_ratio_when_replay_diverges_from_archived_error():
    owner_volume = np.ones(4)
    saved = np.ones((4, 1))
    archived_error = np.full((4, 1), 1e-6)  # a tiny archived spatial error
    replay_arr = saved + 1.0  # a huge mismatch relative to that tiny archived error
    result = replay.compare_owner_term("demo", replay_arr, saved, owner_volume=owner_volume,
                                       archived_error=archived_error, region_masks=None)
    assert not result["pass"]
    assert result["worst_ratio"] > result["ratio_tolerance"]


def test_compare_owner_term_fails_on_pointwise_cap_even_with_a_fine_ratio():
    owner_volume = np.ones(4)
    saved = np.zeros((4, 1))  # every target is a "zero target"
    archived_error = np.full((4, 1), 10.0)  # huge archived error -> ratio is fine
    replay_arr = np.full((4, 1), 1e-6)  # tiny in absolute terms, but >> the 1e-11 zero_atol cap
    result = replay.compare_owner_term("demo", replay_arr, saved, owner_volume=owner_volume,
                                       archived_error=archived_error, region_masks=None)
    assert result["ratios"]["global"] < result["ratio_tolerance"]
    assert result["pointwise"]["violations"] == 4
    assert not result["pass"]


def test_compare_owner_term_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        replay.compare_owner_term("demo", np.zeros((3, 2)), np.zeros((3, 1)), owner_volume=np.ones(3),
                                  archived_error=np.zeros((3, 1)))
    with pytest.raises(ValueError):
        replay.compare_owner_term("demo", np.zeros((3, 2)), np.zeros((3, 2)), owner_volume=np.ones(3),
                                  archived_error=np.zeros((3, 1)))


def test_compare_pointwise_only_has_no_ratio_concept():
    saved = np.array([[1.0, 2.0], [3.0, 4.0]])
    replay_arr = saved.copy()
    result = replay.compare_pointwise_only("demo", replay_arr, saved)
    assert result["pass"]
    assert "ratios" not in result
    replay_arr2 = saved.copy()
    replay_arr2[0, 0] += 1.0
    result2 = replay.compare_pointwise_only("demo", replay_arr2, saved)
    assert not result2["pass"]
    assert result2["pointwise"]["violations"] == 1
    assert result2["pointwise"]["worst_index"] == [0, 0]

    # The column-magnitude reference applies here too.
    small_at_large_column = np.array([[100.0], [1e-7]])
    ok = replay.compare_pointwise_only("demo", small_at_large_column + np.array([[0.0], [1e-14]]),
                                       small_at_large_column)
    assert ok["pass"]
    lifted = replay.compare_pointwise_only("demo", small_at_large_column + 1e-8, small_at_large_column,
                                           cap_reference=np.array([[1e4]]))
    assert lifted["pass"]


def test_write_report_names_the_actual_reduction_host(tmp_path):
    import platform
    payload = {"n": 8, "generated_at": "2026-01-01T00:00:00+00:00", "artifact_root": "/x", "wall_seconds": 1.0,
               "campaigns": {}}
    text = replay.write_report(payload, tmp_path).read_text()
    assert f"Reduction ran on {platform.system()} ({platform.node()})" in text
    assert "Oracle arrays come from their original campaigns" in text
    assert "macOS" not in text and "Perlmutter" not in text


# ---------------------------------------------------------------------------
# scatter_owner_weighted_mean -- fully synthetic.
# ---------------------------------------------------------------------------
def test_scatter_owner_weighted_mean_matches_hand_computation():
    term = np.array([[1.0], [2.0], [3.0], [4.0]])
    weight = np.array([1.0, 1.0, 2.0, 1.0])
    owner_ids = np.array([0, 0, 1, 1])
    mean, denom = replay.scatter_owner_weighted_mean(term, weight, owner_ids, owner_count=2)
    expected_owner0 = (1.0 * 1.0 + 1.0 * 2.0) / (1.0 + 1.0)
    expected_owner1 = (2.0 * 3.0 + 1.0 * 4.0) / (2.0 + 1.0)
    np.testing.assert_allclose(mean[:, 0], [expected_owner0, expected_owner1])
    np.testing.assert_allclose(denom, [2.0, 3.0])


def test_scatter_owner_weighted_mean_rejects_length_mismatch():
    with pytest.raises(ValueError):
        replay.scatter_owner_weighted_mean(np.zeros((3, 1)), np.zeros(2), np.zeros(3, dtype=np.int64), 1)


# ---------------------------------------------------------------------------
# RowIndex -- synthetic PointRows/IntegratedFaceRow, no real geometry needed
# for from_point_requests; a real drbx.stencils.artifact pack/save/load
# roundtrip for from_artifact.
# ---------------------------------------------------------------------------
def _make_point_rows(rng, *, count, donors, fields, conditioned=False):
    from drbx.geometry.fci_perpendicular_reconstruction import PointRows
    rows = []
    for _ in range(count):
        donor_ids = np.arange(donors, dtype=np.int64)
        value = rng.normal(size=(1, donors))
        gradient = rng.normal(size=(1, 3, donors))
        if conditioned:
            trace_donor_points = rng.normal(size=(donors, 3))
        else:
            trace_donor_points = np.empty((0, 3))
        trace_target_points = rng.normal(size=(1, 3))
        rows.append(PointRows(donor_ids, value, gradient, conditioned, trace_donor_points,
                              trace_target_points, {"family": "synthetic"}))
    return rows


def test_row_index_from_point_requests_looks_up_by_request_and_entity_id():
    from drbx.stencils.builder import PointRowRequest, IntegratedRowRequest
    from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow

    rng = np.random.default_rng(0)
    rows = _make_point_rows(rng, count=3, donors=2, fields=2)
    requests = [PointRowRequest("R1", i, "", 0, row) for i, row in enumerate(rows)]
    integrated_row = IntegratedFaceRow(np.array([0, 1], dtype=np.int64), rng.normal(size=2), False,
                                       np.empty((0, 3)), rng.normal(size=(9, 3)), np.empty(0), np.empty((9, 2)), 3)
    integrated_requests = [IntegratedRowRequest(7, integrated_row)]

    index = replay.RowIndex.from_point_requests(requests, integrated_requests)
    for i, row in enumerate(rows):
        found = index.get_point("R1", i)
        assert found is row
    assert index.get_point("R1", 999) is None
    assert index.get_point("R2", 0) is None
    assert index.get_integrated(7) is integrated_row
    assert index.get_integrated(0) is None


def test_row_index_from_artifact_roundtrips_request_and_entity_id_tags():
    from drbx.stencils import artifact as artifact_mod

    rng = np.random.default_rng(1)
    r1_rows = _make_point_rows(rng, count=2, donors=2, fields=2)
    r2_rows = _make_point_rows(rng, count=2, donors=3, fields=2)
    cells_chunk = artifact_mod.pack_point_rows(r1_rows, request="R1", entity_id=[10, 11])
    faces_chunk = artifact_mod.pack_point_rows(r2_rows, request="R2", entity_id=[20, 21])

    from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
    p07_row = IntegratedFaceRow(np.array([0, 1], dtype=np.int64), rng.normal(size=2), False,
                               np.empty((0, 3)), rng.normal(size=(9, 3)), np.empty(0), np.empty((9, 2)), 3)
    p07_chunk = artifact_mod.pack_integrated_rows([p07_row], entity_id=[42])

    identity = {"schema_test": True}
    from drbx.stencils.artifact import RowArtifact
    art = RowArtifact(n=4, identity=identity, cells=(cells_chunk,), faces=(faces_chunk,),
                      neumann=(), p07=(p07_chunk,))
    root = Path(pytest_tmp_root())
    art.save(root)
    loaded = artifact_mod.load_row_artifact(root, 4, identity)

    index = replay.RowIndex.from_artifact(loaded)
    for local, entity_id in enumerate((10, 11)):
        row = index.get_point("R1", entity_id)
        assert row is not None
        np.testing.assert_array_equal(row.value, r1_rows[local].value)
        np.testing.assert_array_equal(row.donor_ids, r1_rows[local].donor_ids)
    for local, entity_id in enumerate((20, 21)):
        row = index.get_point("R2", entity_id)
        assert row is not None
        np.testing.assert_array_equal(row.value, r2_rows[local].value)
    assert index.get_point("R1", 20) is None  # request tag must disambiguate, not just entity_id
    got = index.get_integrated(42)
    assert got is not None
    np.testing.assert_array_equal(got.weights, p07_row.weights)


def pytest_tmp_root():
    import tempfile
    return tempfile.mkdtemp(prefix="p_shared_replay_test_")


# ---------------------------------------------------------------------------
# Real-geometry, on-the-fly small subset -- skipped cleanly if the HSX
# workspace inputs are unavailable (same convention as
# tests/test_p_shared_apply.py / tests/test_stencils_builder.py).
# ---------------------------------------------------------------------------
RUNTIME_INPUTS = WORKSPACE / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/runtime_inputs"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32


def _workspace_inputs_available() -> bool:
    if not RUNTIME_INPUTS.is_dir() or not SIDECAR.is_file():
        return False
    geometry_root = RUNTIME_INPUTS / "geometry_artifacts/rlp_convergence_32_48_64_20260917" / f"{N}x{N}x{N}"
    return (geometry_root / "base_geometry.npz").is_file()


needs_workspace = pytest.mark.skipif(not _workspace_inputs_available(),
                                     reason="HSX workspace geometry inputs are unavailable")


@needs_workspace
@pytest.mark.slow
def test_row_index_from_on_the_fly_builder_subset_matches_direct_application():
    """Develop/verify replay's row plumbing against a *small* owner subset
    built directly via drbx.stencils.builder (never the multi-GB row
    artifact) -- exactly the workflow the task asked for."""
    from perpendicular_structured.reconstruction import load_context
    from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, StructuredReconstruction
    from drbx.stencils import builder as stencil_builder
    from p_shared import apply as pshared_apply

    t = load_context(N, str(RUNTIME_INPUTS))
    ctx = PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro,
                                      raw_volume=t.rv, owner_volume=t.vol,
                                      owner_centroid_xy=t.g.owner_centroid_xy, eta_period=t.g.eta_period,
                                      dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)
    S = StructuredReconstruction(ctx)

    def normal_coefficients(q):
        from p07n_field_derived_global.fields import normal as p07n_normal
        import json
        ref_holder = normal_coefficients.ref
        return p07n_normal(ref_holder, np.asarray(q, dtype=np.float64))

    from p_shared import provider as pshared_provider
    ref = pshared_provider.ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False).reference
    normal_coefficients.ref = ref

    raw_ids = np.array([0, 1, N ** 3 - 1], dtype=np.int64)  # an interior cell and two boundary (near-axis / wall) cells
    point_requests, neumann_requests = stencil_builder.build_r1_cell_rows(
        S, ctx, raw_ids, normal_coefficients=normal_coefficients, patch_cache={})

    index = replay.RowIndex.from_point_requests(point_requests, ())
    rng = np.random.default_rng(2)
    owner_values = rng.normal(size=(len(t.vol), 3))

    def trace(q):
        v = np.zeros((len(q), 3))
        g = np.zeros((len(q), 3, 3))
        return v, g

    for req in point_requests:
        row_from_index = index.get_point("R1", req.entity_id)
        assert row_from_index is req.row  # identity: no copy, no reordering
        expected = pshared_apply.apply_point_row(req.row, owner_values, trace if req.row.boundary_conditioned else None)
        actual = pshared_apply.apply_point_row(row_from_index, owner_values,
                                               trace if row_from_index.boundary_conditioned else None)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
