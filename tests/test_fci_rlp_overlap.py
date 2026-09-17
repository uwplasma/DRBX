import numpy as np

from drbx.geometry.fci_rlp_overlap import (
    _decompose_mapped_boundary,
    build_rlp_parallel_overlap_geometry,
    compare_rlp_parallel_overlap_refinement,
    load_rlp_parallel_overlap_geometry,
    merge_rlp_parallel_overlap_geometries,
    write_rlp_parallel_overlap_geometry,
)


def _faces(nz=2):
    faces = []
    for k in range(nz):
        for i, (lower, upper) in enumerate(((0.5, 1.0), (1.0, 1.5))):
            faces.append(
                {
                    "owner": i + 2 * k,
                    "plane": k,
                    "vertices": [[lower, 0.0], [upper, 0.0], [upper, 1.0], [lower, 1.0]],
                }
            )
    return faces


def test_identity_faces_make_unique_pairwise_links_and_positive_tau():
    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=_faces(), subdivision_tolerance=1.0e-6
    )
    assert np.array_equal(graph.owner_flat_ids, [0, 1, 2, 3])
    assert graph.n_link == 4
    assert np.all(graph.overlap_measure > 0.0)
    assert np.all(graph.transmissibility > 0.0)
    assert np.unique(np.column_stack((graph.link_owner_a, graph.link_owner_b, graph.link_interface)), axis=0).shape[0] == graph.n_link


def test_callback_trace_preserves_unique_owners_and_lengths():
    faces = _faces()

    def shift(points, direction, eta):
        values = np.asarray(points, dtype=float).copy()
        values[:, 1] += 0.0
        return values, np.full(values.shape[0], 2.0), np.zeros(values.shape[0], dtype=bool)

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=faces, trace_callback=shift
    )
    assert graph.n_owner == 4
    assert graph.n_link == 4
    assert np.all(graph.overlap_measure >= 0.0)
    reference = build_rlp_parallel_overlap_geometry(raw_shape=(2, 1, 2), raw_faces=faces)
    assert np.allclose(graph.transmissibility, 0.5 * reference.transmissibility)


def test_wall_terminated_faces_are_homogeneous_neumann_no_links():
    faces = _faces()

    def wall(points, direction, eta):
        return np.asarray(points), np.ones(len(points)), np.ones(len(points), dtype=bool)

    graph = build_rlp_parallel_overlap_geometry(raw_shape=(2, 1, 2), raw_faces=faces, trace_callback=wall)
    assert graph.n_link == 0
    assert np.all(np.asarray(graph.diagnostics["source_coverage"]) == 0.0)
    assert np.all(np.asarray(graph.diagnostics["destination_coverage"]) == 0.0)


def test_mixed_wall_faces_retain_only_interior_overlap_and_report_coverage():
    """A bounded wall cut removes only the traced wall portion.

    The transition is intentionally inside one radial face.  The callback is
    an identity map, so the retained overlap is exactly the annular area below
    the wall threshold and the discarded part is homogeneous-Neumann loss.
    """
    faces = _faces()

    def partial_wall(points, direction, eta):
        values = np.asarray(points, dtype=float)
        return values, np.ones(len(values)), values[:, 0] > 1.25

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2),
        raw_faces=faces,
        trace_callback=partial_wall,
        interface_indices=(0,),
        max_subdivision=6,
    )
    reference = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=faces, interface_indices=(0,)
    )
    # Face 0 is unchanged, while face 1 retains a positive interior portion
    # and loses the wall-terminated remainder.
    np.testing.assert_allclose(graph.overlap_measure[0], reference.overlap_measure[0])
    assert 0.0 < graph.overlap_measure[1] < reference.overlap_measure[1]
    mixed = graph.diagnostics["mixed_wall"]
    assert mixed
    assert any("wall_area" in item and item["wall_area"] >= 0.0 for item in mixed)
    np.testing.assert_allclose(
        graph.diagnostics["source_coverage"][0],
        np.sum(graph.overlap_measure),
    )
    np.testing.assert_allclose(
        graph.diagnostics["destination_coverage"][0],
        np.sum(graph.overlap_measure),
    )


def test_mixed_wall_terminated_loss_is_reported():
    """An interior fragment facing a wall is allowed and recorded as loss."""
    faces = _faces()

    def asymmetric_wall(points, *args, **kwargs):
        values = np.asarray(points, dtype=float)
        direction = kwargs.get("direction", args[0] if args else "")
        wall = np.ones(len(values), dtype=bool)
        if direction == "forward":
            wall = values[:, 0] > 1.25
        return values, np.ones(len(values)), wall

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=faces,
        trace_callback=asymmetric_wall, interface_indices=(0,), max_subdivision=6,
    )
    assert np.all(np.asarray(graph.diagnostics["wall_terminated_source"]) >= 0.0)
    assert np.all(np.asarray(graph.diagnostics["wall_terminated_destination"]) >= 0.0)
    assert graph.diagnostics["wall_terminated_source"][0] > 0.0
    assert graph.diagnostics["source_undercoverage_at_destination_wall"][0] > 0.0
    assert (
        graph.diagnostics["source_undercoverage_at_destination_wall"][0]
        <= graph.diagnostics["destination_wall_capacity_for_source_undercoverage"][0]
    )


def test_two_interior_nodes_form_positive_wall_cut_polygon():
    """Entry/exit wall-hit nodes preserve a small positive component."""
    faces = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [1.5, 0.0], [1.5, 1.0], [1.0, 1.0]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [1.5, 0.0], [1.5, 1.0], [1.0, 1.0]]},
    ]

    def half_face_wall(points, *args, **kwargs):
        values = np.asarray(points, dtype=float)
        return values, np.ones(len(values)), values[:, 1] < 0.5

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2), raw_faces=faces,
        trace_callback=half_face_wall, interface_indices=(0,), max_subdivision=0,
    )
    assert graph.n_link == 1
    assert graph.overlap_measure[0] > 0.0
    assert any(item.get("component_count") == 1 for item in graph.diagnostics["mixed_wall"])


def test_wall_hit_endpoint_coordinates_are_not_common_plane_polygon_vertices():
    """Boundary-hit coordinates may be off-plane and must be ignored.

    The tracer reports a wall hit before reaching the requested eta plane.
    Make those coordinates deliberately remote; the resolved interior run
    must still produce the same bounded cut as the ordinary identity trace.
    """
    faces = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
    ]

    def remote_wall_hits(points, *args, **kwargs):
        values = np.asarray(points, dtype=float)
        wall = values[:, 0] > 1.75
        mapped = values.copy()
        mapped[wall, 0] += 100.0
        mapped[wall, 1] -= 70.0
        return mapped, np.ones(len(values)), wall

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2),
        raw_faces=faces,
        trace_callback=remote_wall_hits,
        interface_indices=(0,),
        max_subdivision=0,
    )
    reference = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2),
        raw_faces=faces,
        interface_indices=(0,),
    )
    assert graph.n_link == 1
    assert 0.0 < graph.overlap_measure[0] < reference.overlap_measure[0]
    assert all(
        item.get("mode") != "boundary_endpoint_vertices"
        for item in graph.diagnostics["mixed_wall"]
    )


def test_folded_mapped_boundary_is_split_into_simple_positive_lobes():
    """A folded common-plane boundary is retained as separate simple pieces."""
    boundary = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=float,
    )
    lobes = _decompose_mapped_boundary(boundary)
    assert len(lobes) == 2
    assert all(abs(0.5 * np.sum(
        lobe[:, 0] * np.roll(lobe[:, 1], -1)
        - lobe[:, 1] * np.roll(lobe[:, 0], -1)
    )) > 0.0 for lobe in lobes)


def test_nonwall_closure_mismatch_fails_closed():
    """Unexplained ordinary-domain loss is not hidden by wall accounting."""
    import pytest

    def shifted(points, *args, **kwargs):
        values = np.asarray(points, dtype=float).copy()
        direction = kwargs.get("direction", args[0] if args else "")
        if direction == "forward":
            values[:, 1] += 0.4
        return values, np.ones(len(values)), np.zeros(len(values), dtype=bool)

    with pytest.raises(ValueError, match="closure failed"):
        build_rlp_parallel_overlap_geometry(
            raw_shape=(2, 1, 2), raw_faces=_faces(),
            trace_callback=shifted, interface_indices=(0,),
        )


def test_nontrivial_mixed_wall_cut_uses_ordered_boundary_chord():
    """A diagonal wall cut stays bounded and cannot self-intersect."""
    faces = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
    ]

    call_counter = {"calls": 0, "points": 0}

    def diagonal_wall(points, *args, **kwargs):
        call_counter["calls"] += 1
        call_counter["points"] += len(points)
        values = np.asarray(points, dtype=float)
        x = values[:, 0] * np.cos(values[:, 1])
        y = values[:, 0] * np.sin(values[:, 1])
        return values, np.ones(len(values)), (x + y) > 1.6

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2), raw_faces=faces,
        trace_callback=diagonal_wall, interface_indices=(0,), max_subdivision=5,
    )
    reference = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2), raw_faces=faces, interface_indices=(0,)
    )
    assert graph.n_link == 1
    assert 0.0 < graph.overlap_measure[0] < reference.overlap_measure[0]
    counters = graph.diagnostics["resource_counters"]
    assert any(item.get("mode") == "ordered_boundary_chord" for item in graph.diagnostics["mixed_wall"])
    # Wall cuts reuse the edge trace; no additional triangle evaluations are
    # permitted after the boundary records have been accepted.
    assert call_counter["calls"] > 0
    assert counters["traced_points"] > 0
    assert call_counter["points"] == counters["trace_evaluated_points"]
    assert counters["trace_padding_points"] > 0
    assert counters["traced_points"] < counters["trace_evaluated_points"]


def test_multiple_mixed_wall_components_fail_closed():
    """Multiple positive interior runs remain separate mapped fragments."""

    faces = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]},
    ]

    def disconnected_wall(points, *args, **kwargs):
        values = np.asarray(points, dtype=float)
        return values, np.ones(len(values)), (values[:, 0] > 1.25) != (values[:, 1] > 0.5)

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2), raw_faces=faces,
        trace_callback=disconnected_wall, interface_indices=(0,), max_subdivision=4,
    )
    assert graph.n_link == 1
    assert graph.overlap_measure[0] > 0.0
    assert any(
        item.get("component_count") == 2
        for item in graph.diagnostics["mixed_wall"]
    )


def test_noncontiguous_flat_owner_ids_still_use_compact_link_slots():
    faces = _faces()
    ids = [5, 9, 15, 19]
    for item, owner in zip(faces, ids * 1):
        item["owner"] = owner
    graph = build_rlp_parallel_overlap_geometry(raw_shape=(10, 1, 2), raw_faces=faces)
    assert np.array_equal(graph.owner_flat_ids, ids)
    assert np.all(graph.link_owner_a < graph.n_owner)
    assert np.all(graph.link_owner_b < graph.n_owner)
    assert np.array_equal(graph.owner_slot.reshape(-1)[ids], [0, 1, 2, 3])


def test_versioned_cache_round_trip_and_identity_rejection(tmp_path):
    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=_faces()
    )
    path = tmp_path / "owner_overlap.npz"
    identity = {"case": "identity", "trace_substeps": 4}
    write_rlp_parallel_overlap_geometry(path, graph, identity=identity)
    loaded = load_rlp_parallel_overlap_geometry(
        path, expected_identity=identity
    )
    np.testing.assert_array_equal(loaded.owner_flat_ids, graph.owner_flat_ids)
    np.testing.assert_array_equal(loaded.link_owner_slots, graph.link_owner_slots)
    np.testing.assert_allclose(loaded.transmissibility, graph.transmissibility)

    import pytest

    with pytest.raises(ValueError, match="identity mismatch"):
        load_rlp_parallel_overlap_geometry(
            path, expected_identity={"case": "different"}
        )


def test_partial_interface_graphs_merge_to_full_graph():
    faces = _faces(nz=4)
    reference = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces
    )
    first = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces, interface_indices=(0, 1)
    )
    second = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces, interface_indices=(2, 3)
    )
    merged = merge_rlp_parallel_overlap_geometries((first, second))
    np.testing.assert_array_equal(merged.owner_flat_ids, reference.owner_flat_ids)
    np.testing.assert_array_equal(merged.link_owner_slots, reference.link_owner_slots)
    np.testing.assert_array_equal(merged.link_interface, reference.link_interface)
    np.testing.assert_allclose(merged.overlap_measure, reference.overlap_measure)
    np.testing.assert_allclose(merged.transmissibility, reference.transmissibility)
    assert merged.metadata["merged_partial_interfaces"] is True
    assert merged.diagnostics["interfaces_built"] == [0, 1, 2, 3]
    assert merged.diagnostics["interface_work"] == sorted(
        merged.diagnostics["interface_work"], key=lambda item: item["interface"]
    )
    assert merged.diagnostics["resource_counters"]["interfaces_completed"] == 4
    assert merged.diagnostics["resource_counters"]["peak_pending_edge_segments"] == reference.diagnostics["resource_counters"]["peak_pending_edge_segments"]


def test_partial_interface_merge_rejects_duplicate_and_missing_batches():
    import pytest

    faces = _faces(nz=4)
    first = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces, interface_indices=(0, 1)
    )
    second = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces, interface_indices=(2, 3)
    )
    with pytest.raises(ValueError, match="duplicate"):
        merge_rlp_parallel_overlap_geometries((first, first, second))
    with pytest.raises(ValueError, match="missing"):
        merge_rlp_parallel_overlap_geometries((first,))
    altered = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=faces, interface_indices=(2, 3),
        second_order_tolerance_factor=0.1,
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_rlp_parallel_overlap_geometries((first, altered))


def test_edge_refinement_has_a_hard_memory_budget():
    import pytest

    with pytest.raises(MemoryError, match="bounded segment budget"):
        build_rlp_parallel_overlap_geometry(
            raw_shape=(2, 1, 2),
            raw_faces=_faces(),
            max_edge_segments=1,
        )


def test_second_order_floor_and_interface_scoping_report_bounded_work():
    """Default tracing stops at an absolute O(h**2) edge error floor.

    Building one interface is also the safe unit for the HSX campaign: the
    graph retains only compact links, while scalar resource counters make the
    peak host-side work auditable before attempting a larger case.
    """

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2),
        raw_faces=_faces(),
        interface_indices=(0,),
        max_edge_segments=64,
        max_mapped_pieces=64,
        max_overlap_candidates=128,
    )
    assert graph.metadata["subdivision_tolerance"] is None
    assert graph.metadata["effective_tolerance_order"] == "O(h^2)"
    assert graph.metadata["second_order_tolerance_factor"] == 5.0e-2
    assert graph.metadata["interface_indices"] == [0]

    counters = graph.diagnostics["resource_counters"]
    assert counters["interfaces_completed"] == 1
    assert counters["mapped_faces"] == 2 * 2  # source and destination faces
    assert counters["mapped_pieces"] <= 64
    assert counters["overlap_candidates"] <= 128
    assert counters["peak_pending_edge_segments"] <= 64


def test_default_refinement_is_second_order_and_interfaces_can_be_selected():
    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 4), raw_faces=_faces(nz=4), interface_indices=(2,)
    )
    assert graph.metadata["effective_tolerance_order"] == "O(h^2)"
    assert graph.metadata["interface_indices"] == [2]
    assert graph.diagnostics["interfaces_built"] == [2]
    assert graph.diagnostics["resource_counters"]["interfaces_completed"] == 1
    assert np.all(graph.link_interface == 2)


def test_trace_samples_use_canonical_padding_for_odd_batches():
    """Odd trace batches are padded to one reusable evaluator shape."""
    calls = []

    def trace(points, *args, **kwargs):
        calls.append(len(points))
        values = np.asarray(points, dtype=float)
        return values, np.ones(len(values)), np.zeros(len(values), dtype=bool)

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=_faces(),
        trace_callback=trace, interface_indices=(0,), trace_chunk_size=8,
    )
    assert graph.metadata["trace_chunk_size"] == 8
    assert calls and all(size == 8 for size in calls)
    counters = graph.diagnostics["resource_counters"]
    assert counters["trace_evaluated_points"] == sum(calls)
    assert counters["trace_padding_points"] > 0
    assert counters["traced_points"] + counters["trace_padding_points"] == counters["trace_evaluated_points"]


def test_polygon_and_candidate_budgets_fail_closed():
    import pytest

    with pytest.raises(MemoryError, match="mapped-piece refinement"):
        build_rlp_parallel_overlap_geometry(
            raw_shape=(2, 1, 2), raw_faces=_faces(), max_mapped_pieces=1
        )
    with pytest.raises(MemoryError, match="candidate budget"):
        build_rlp_parallel_overlap_geometry(
            raw_shape=(2, 1, 2), raw_faces=_faces(), max_overlap_candidates=1
        )


def test_refinement_comparison_reduces_to_canonical_links():
    result = compare_rlp_parallel_overlap_refinement(
        tolerances=(1.0e-2, 1.0e-3),
        builder_kwargs={"raw_shape": (2, 1, 2), "raw_faces": _faces()},
    )
    assert result["reference_tolerance"] == 1.0e-3
    assert result["summaries"][-1]["link_count"] == 4
    comparison = result["comparisons_to_reference"][0]
    assert comparison["same_canonical_links"]
    assert comparison["relative_transmissibility_error"] == 0.0


def test_owner_face_dissolution_does_not_trace_internal_raw_edges():
    """Adjacent raw subfaces of one owner are traced as one boundary ring."""

    faces = [
        {
            "owner": 0,
            "plane": 0,
            "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]],
        },
        {
            "owner": 0,
            "plane": 0,
            "vertices": [[1.0, 0.5], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0]],
        },
        {
            "owner": 1,
            "plane": 1,
            "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]],
        },
        {
            "owner": 1,
            "plane": 1,
            "vertices": [[1.0, 0.5], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0]],
        },
    ]
    traced = []

    def trace(points, direction, eta):
        traced.extend((direction, point) for point in np.asarray(points, dtype=float))
        return np.asarray(points), np.ones(len(points)), np.zeros(len(points), dtype=bool)

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2),
        raw_faces=faces,
        trace_callback=trace,
        interface_indices=(0,),
    )

    # theta=.5, 1<u<2 is the common radial raw-face edge.  It is absent from
    # the dissolved annular-sector boundary (including adaptive edge points).
    assert not any(
        direction == "forward"
        and np.isclose(point[1], 0.5)
        and 1.1 < point[0] < 1.8
        for direction, point in traced
    )
    assert graph.metadata["raw_face_count"] == 4
    assert graph.metadata["dissolved_owner_face_count"] == 2


def test_owner_face_dissolution_preserves_identity_graph():
    split = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]]},
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.5], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.5], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0]]},
    ]
    merged = [
        {"owner": 0, "plane": 0, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0], [1.0, 0.5]]},
        {"owner": 1, "plane": 1, "vertices": [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0], [1.0, 0.5]]},
    ]
    split_graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=split, interface_indices=(0,)
    )
    merged_graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2), raw_faces=merged, interface_indices=(0,)
    )
    np.testing.assert_array_equal(split_graph.owner_flat_ids, merged_graph.owner_flat_ids)
    np.testing.assert_array_equal(split_graph.link_owner_slots, merged_graph.link_owner_slots)
    np.testing.assert_array_equal(split_graph.link_interface, merged_graph.link_interface)
    np.testing.assert_allclose(split_graph.overlap_measure, merged_graph.overlap_measure)
    np.testing.assert_allclose(split_graph.transmissibility, merged_graph.transmissibility)


def _twisted_annulus_faces(n_theta):
    faces = []
    for plane in range(2):
        for j in range(n_theta):
            theta0 = 2.0 * np.pi * j / n_theta
            theta1 = 2.0 * np.pi * (j + 1) / n_theta
            faces.append(
                {
                    "owner": j + n_theta * plane,
                    "plane": plane,
                    "vertices": [[1.0, theta0], [2.0, theta0], [2.0, theta1], [1.0, theta1]],
                }
            )
    return faces


def test_smooth_area_preserving_twist_has_second_order_tau_convergence():
    """A smooth field-line twist converges under bounded overlap refinement.

    The twist is ``theta -> theta +/- alpha*sin(pi*(u-1))`` for the two
    half-traces.  It preserves ``du*dtheta`` and fixes both radial boundaries,
    so the exact logical-area conductance is known.  The host representation
    uses straight mapped polygon edges; the test therefore checks its
    measured asymptotic convergence, rather than claiming exact curved-face
    integration.
    """

    def twist(points, direction, eta):
        values = np.asarray(points, dtype=float).copy()
        sign = 1.0 if direction == "forward" else -1.0
        values[:, 1] += sign * 0.45 * np.sin(np.pi * (values[:, 0] - 1.0))
        return values, np.ones(values.shape[0]), np.zeros(values.shape[0], dtype=bool)

    totals = []
    closure_errors = []
    for n_theta in (8, 16, 32):
        graph = build_rlp_parallel_overlap_geometry(
            raw_shape=(1, n_theta, 2),
            raw_faces=_twisted_annulus_faces(n_theta),
            trace_callback=twist,
            interface_indices=(0,),
            # Curved mapped polygons are intentionally allowed their
            # O(h**2) straight-edge closure discrepancy in this host-only
            # convergence probe; the pairwise operator remains exact-
            # conservative for every resulting graph.
            coverage_tolerance=0.1,
            max_subdivision=6,
        )
        totals.append(float(np.sum(graph.transmissibility)))
        closure_errors.append(float(graph.diagnostics["source_closure_error"][0]))
        assert np.all(graph.transmissibility >= 0.0)

    exact_total = np.pi  # (u1-u0)*2*pi / (ell_a+ell_b), with ell=1.
    errors = np.abs(np.asarray(totals) - exact_total)
    # Refining the angular owner partition reduces the straight-edge error by
    # roughly a factor of four, as expected for the second-order geometry.
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0
    assert closure_errors[1] < closure_errors[0] / 3.0


def test_metric_quadrature_batching_matches_pointwise_reference_and_reduces_calls():
    """Batched metric evaluation preserves per-fragment quadrature exactly."""

    def metric(points, interface=None):
        values = np.asarray(points, dtype=float)
        u = values[:, 0]
        theta = values[:, 1]
        return {
            "J": 1.0 + 0.15 * u,
            "b_eta": 0.8 + 0.03 * np.cos(theta),
            "ell_a": 1.2 + 0.1 * u,
            "ell_b": 1.7 + 0.05 * np.sin(theta) ** 2,
        }

    reference_calls = []

    def metric_reference(points, interface=None):
        reference_calls.append(len(points))
        return metric(points, interface=interface)

    batched_calls = []

    def metric_batched(points, interface=None):
        batched_calls.append(len(points))
        return metric(points, interface=interface)

    reference = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 8, 2),
        raw_faces=_twisted_annulus_faces(8),
        trace_callback=lambda points, direction, eta: (
            np.asarray(points),
            np.ones(len(points)),
            np.zeros(len(points), dtype=bool),
        ),
        metric_callback=metric_reference,
        interface_indices=(0,),
        metric_batch_size=1,
        coverage_tolerance=1.0e-10,
    )
    batched = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 8, 2),
        raw_faces=_twisted_annulus_faces(8),
        trace_callback=lambda points, direction, eta: (
            np.asarray(points),
            np.ones(len(points)),
            np.zeros(len(points), dtype=bool),
        ),
        metric_callback=metric_batched,
        interface_indices=(0,),
        metric_batch_size=32768,
        coverage_tolerance=1.0e-10,
    )
    np.testing.assert_array_equal(reference.link_owner_slots, batched.link_owner_slots)
    np.testing.assert_allclose(reference.overlap_measure, batched.overlap_measure)
    np.testing.assert_allclose(reference.transmissibility, batched.transmissibility)
    assert len(batched_calls) < len(reference_calls)
    assert max(batched_calls) <= 32768
    counters = batched.diagnostics["resource_counters"]
    assert counters["quadrature_real_points"] == counters["quadrature_evaluated_points"]
    assert counters["quadrature_batches"] == len(batched_calls)
    assert counters["peak_quadrature_batch_points"] == max(batched_calls)


def test_owner_overlap_reports_bounded_per_interface_work_and_releases_payloads():
    events = []
    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(2, 1, 2),
        raw_faces=_faces(),
        interface_indices=(0, 1),
        max_quadrature_points=32,
        metric_batch_size=8,
        progress_callback=events.append,
    )
    assert len(events) == 2
    assert [event["interface"] for event in events] == [0, 1]
    assert all(event["temporary_payload_released"] for event in events)
    assert all(event["source_piece_count"] > 0 for event in events)
    assert all(event["destination_piece_count"] > 0 for event in events)
    assert all(event["quadrature_buffer_peak_points"] <= 32 for event in events)
    assert graph.diagnostics["interface_work"] == events
    counters = graph.diagnostics["resource_counters"]
    assert counters["interfaces_completed"] == 2
    assert counters["peak_quadrature_buffer_points"] <= 32


def test_spatially_varying_traced_lengths_are_linearly_interpolated():
    """Mapped-piece length models retain the resolved spatial denominator."""

    vertices = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]], dtype=float
    )
    faces = [
        {"owner": 0, "plane": 0, "vertices": vertices.tolist()},
        {"owner": 1, "plane": 1, "vertices": vertices.tolist()},
    ]

    def trace(points, direction, eta):
        values = np.asarray(points, dtype=float)
        xy = np.column_stack(
            (values[:, 0] * np.cos(values[:, 1]), values[:, 0] * np.sin(values[:, 1]))
        )
        lengths = 2.0 + 0.2 * xy[:, 0] + 0.1 * xy[:, 1]
        return values, lengths, np.zeros(len(values), dtype=bool)

    graph = build_rlp_parallel_overlap_geometry(
        raw_shape=(1, 1, 2),
        raw_faces=faces,
        trace_callback=trace,
        interface_indices=(0,),
    )

    # Reproduce the prescribed degree-2 triangle rule on the unchanged
    # identity overlap.  Since the traced length is affine in mapped x/y, the
    # per-piece affine model should agree with this reference to roundoff.
    polygon = vertices.copy()
    xy_polygon = np.column_stack(
        (polygon[:, 0] * np.cos(polygon[:, 1]), polygon[:, 0] * np.sin(polygon[:, 1]))
    )
    expected = 0.0
    for index in range(1, xy_polygon.shape[0] - 1):
        triangle = np.asarray(
            (xy_polygon[0], xy_polygon[index], xy_polygon[index + 1])
        )
        area = abs(
            0.5
            * np.sum(
                triangle[:, 0] * np.roll(triangle[:, 1], -1)
                - triangle[:, 1] * np.roll(triangle[:, 0], -1)
            )
        )
        for barycentric in (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        ):
            point = np.einsum("i,ij->j", barycentric, triangle)
            radius = np.hypot(point[0], point[1])
            length = 2.0 + 0.2 * point[0] + 0.1 * point[1]
            expected += area / 3.0 / (radius * (2.0 * length))
    np.testing.assert_allclose(graph.transmissibility, [expected], rtol=1.0e-12, atol=1.0e-14)
