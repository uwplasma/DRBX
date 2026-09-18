"""Focused tests for the trace-free lean owner-boundary producer."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from drbx.geometry.fci_owner_boundary_overlap import build_owner_boundary_overlap_geometry
from drbx.geometry.fci_simulation_geometry import FciVertexTraceAtlas
from drbx.geometry.fci_rlp_overlap import GlobalRlpParallelOverlapGeometry


def _fixture():
    shape = (2, 4, 1)
    xfaces = np.array([0.0, 0.5, 1.0])
    yfaces = np.linspace(0.0, 2.0 * np.pi, shape[1] + 1)
    grid = SimpleNamespace(
        x=SimpleNamespace(faces=xfaces, centers=0.5 * (xfaces[:-1] + xfaces[1:])),
        y=SimpleNamespace(faces=yfaces, centers=0.5 * (yfaces[:-1] + yfaces[1:])),
        z=SimpleNamespace(faces=np.array([0.0, 1.0]), centers=np.array([0.0])),
        eta_period=1.0,
    )
    endpoint_x = np.full(shape, 0.75)
    endpoint_y = np.zeros(shape)
    endpoint_z = np.zeros(shape)
    maps = SimpleNamespace(
        forward_endpoint_x=endpoint_x,
        forward_endpoint_y=endpoint_y,
        forward_endpoint_z=endpoint_z,
        backward_endpoint_x=endpoint_x,
        backward_endpoint_y=endpoint_y,
        backward_endpoint_z=endpoint_z,
        forward_length=np.ones(shape),
        backward_length=np.ones(shape),
        forward_boundary=np.zeros(shape, dtype=bool),
        backward_boundary=np.zeros(shape, dtype=bool),
    )
    geometry = SimpleNamespace(shape=shape, grid=grid, maps=maps)
    aggregate_id = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    owner = SimpleNamespace(
        topology=SimpleNamespace(shape=shape, aggregate_id=aggregate_id),
        aggregate_chart_volume=np.ones(shape),
    )
    vertices = np.empty((shape[0] + 1, shape[1], shape[2], 3))
    for i, u in enumerate(xfaces):
        for j in range(shape[1]):
            vertices[i, j, 0] = (u * np.cos(yfaces[j]), u * np.sin(yfaces[j]), 0.0)
    atlas = FciVertexTraceAtlas(
        vertex_positions=vertices,
        forward_endpoint=np.stack((np.broadcast_to(xfaces[:, None, None], vertices.shape[:-1]), np.broadcast_to(yfaces[:-1][None, :, None], vertices.shape[:-1]), np.zeros(vertices.shape[:-1])), axis=-1),
        backward_endpoint=np.stack((np.broadcast_to(xfaces[:, None, None], vertices.shape[:-1]), np.broadcast_to(yfaces[:-1][None, :, None], vertices.shape[:-1]), np.zeros(vertices.shape[:-1])), axis=-1),
        forward_length=np.ones(vertices.shape[:-1]),
        backward_length=np.ones(vertices.shape[:-1]),
        forward_boundary=np.zeros(vertices.shape[:-1], dtype=bool),
        backward_boundary=np.zeros(vertices.shape[:-1], dtype=bool),
        metadata={"endpoint_coordinates": "logical", "trace_substeps": 64},
    )
    return geometry, owner, atlas


def test_builder_consumes_pretraced_atlas_without_trace_callback():
    geometry, owner, atlas = _fixture()
    result = build_owner_boundary_overlap_geometry(
        geometry,
        owner,
        atlas,
        cell_center_wall_masks=(
            geometry.maps.forward_boundary,
            geometry.maps.backward_boundary,
        ),
    )
    assert result.raw_shape == (2, 4, 1)
    assert np.array_equal(result.owner_flat_ids, np.arange(8))
    assert np.all(result.owner_volumes > 0.0)
    assert np.all(result.overlap_measure >= 0.0)
    assert np.all(result.transmissibility >= 0.0)
    assert result.metadata["trace_free"] is True


def test_builder_has_no_tracer_parameter():
    import inspect

    names = inspect.signature(build_owner_boundary_overlap_geometry).parameters
    assert "trace_callback" not in names
    assert "trace_substeps" not in names


def test_owner_graph_invariants_and_minimum_principle():
    """The canonical pair graph has the expected M-matrix properties."""
    graph = GlobalRlpParallelOverlapGeometry(
        raw_shape=(2, 1, 1),
        owner_flat_ids=np.array([0, 1]),
        owner_volumes=np.array([2.0, 1.0]),
        link_owner_a=np.array([0]),
        link_owner_b=np.array([1]),
        link_interface=np.array([0]),
        overlap_measure=np.array([1.0]),
        transmissibility=np.array([3.0]),
    )
    volume = graph.owner_volumes
    matrix = np.zeros((2, 2))
    for a, b, tau in zip(graph.link_owner_a, graph.link_owner_b, graph.transmissibility):
        matrix[a, b] += tau / volume[a]
        matrix[b, a] += tau / volume[b]
        matrix[a, a] -= tau / volume[a]
        matrix[b, b] -= tau / volume[b]
    assert np.allclose(matrix @ np.ones(2), 0.0)
    assert np.allclose(volume[:, None] * matrix, (volume[:, None] * matrix).T)
    assert matrix[0, 1] >= 0.0 and matrix[1, 0] >= 0.0
    assert matrix[0, 0] <= 0.0 and matrix[1, 1] <= 0.0
    assert np.dot(np.array([1.0, 4.0]), volume * (matrix @ np.array([1.0, 4.0]))) <= 0.0
    # At a discrete minimum, the diffusion derivative is nonnegative.
    state = np.array([0.0, 1.0])
    assert (matrix @ state)[0] >= 0.0


def test_atlas_wall_mask_terminates_source_cell():
    geometry, owner, atlas = _fixture()
    blocked = np.array(atlas.forward_boundary, copy=True)
    blocked[1, 0, 0] = True
    atlas = FciVertexTraceAtlas(
        vertex_positions=atlas.vertex_positions,
        forward_endpoint=atlas.forward_endpoint,
        backward_endpoint=atlas.backward_endpoint,
        forward_length=atlas.forward_length,
        backward_length=atlas.backward_length,
        forward_boundary=blocked,
        backward_boundary=atlas.backward_boundary,
        metadata=atlas.metadata,
    )
    result = build_owner_boundary_overlap_geometry(
        geometry,
        owner,
        atlas,
        cell_center_wall_masks=(
            geometry.maps.forward_boundary,
            geometry.maps.backward_boundary,
        ),
    )
    records = [r for r in result.diagnostics["interfaces"] if r["direction"] == "forward"]
    assert records and records[0]["active_cells"] < 8


def test_overlap_closure_excursion_is_diagnostic_not_a_gate():
    geometry, owner, atlas = _fixture()
    forward = np.array(atlas.forward_endpoint, copy=True)
    backward = np.array(atlas.backward_endpoint, copy=True)
    forward[-1, :, :, 0] = 1.1
    backward[-1, :, :, 0] = 1.1
    atlas = replace(
        atlas,
        forward_endpoint=forward,
        backward_endpoint=backward,
    )

    result = build_owner_boundary_overlap_geometry(
        geometry,
        owner,
        atlas,
        cell_center_wall_masks=(
            geometry.maps.forward_boundary,
            geometry.maps.backward_boundary,
        ),
        coverage_tolerance=1.0e-10,
    )

    assert result.diagnostics["max_closure_error"] > 0.0
    assert result.diagnostics["max_relative_closure_error"] > 0.0
    assert result.diagnostics["closure_reference_exceedance_count"] > 0
    assert result.diagnostics["closure_within_reference_tolerance"] is False
