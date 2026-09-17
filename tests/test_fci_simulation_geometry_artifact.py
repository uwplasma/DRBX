from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from drbx.geometry.fci_control_volumes import build_global_control_volume_topology_from_owner_map
from drbx.geometry.fci_control_volumes import PolarAngularAgglomerationGeometry3D
from drbx.geometry.fci_geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    MetricGeometry,
    Spacing3D,
)
from drbx.geometry.fci_simulation_geometry import (
    FciSimulationGeometry3D,
    FciVertexTraceAtlas,
    audit_fci_simulation_geometry_checksums,
    load_fci_simulation_geometry,
    validate_fci_simulation_geometry_producer,
    write_fci_simulation_geometry,
)
from drbx.geometry.fci_rlp_overlap import GlobalRlpParallelOverlapGeometry
from drbx.native.fci_rlp_overlap import (
    local_parallel_diffusion_fci_rlp_overlap_op,
    lower_rlp_parallel_overlap_geometry,
)
from drbx.native.fci_sharding import build_local_fci_geometries


def _metric(shape: tuple[int, int, int], value: float = 1.0) -> MetricGeometry:
    arrays = {name: np.full(shape, value, dtype=np.float64) for name in (
        "J", "g11", "g22", "g33", "g12", "g13", "g23",
        "g_11", "g_22", "g_33", "g_12", "g_13", "g_23",
    )}
    return MetricGeometry(**arrays)


def _geometry() -> FciGeometry3D:
    shape = (2, 2, 2)
    grid = CellCenteredGrid3D(
        Grid1D(np.array([0.25, 0.75]), np.array([0.0, 0.5, 1.0])),
        Grid1D(np.array([0.25, 0.75]), np.array([0.0, 0.5, 1.0])),
        Grid1D(np.array([0.25, 0.75]), np.array([0.0, 0.5, 1.0])),
    )
    zeros = np.zeros(shape)
    maps = FciMaps3D(
        forward_x=zeros, forward_y=zeros, backward_x=zeros, backward_y=zeros,
        forward_endpoint_x=zeros, forward_endpoint_y=zeros, forward_endpoint_z=zeros,
        backward_endpoint_x=zeros, backward_endpoint_y=zeros, backward_endpoint_z=zeros,
        forward_length=np.ones(shape), backward_length=np.ones(shape),
        forward_boundary=np.zeros(shape, dtype=bool), backward_boundary=np.zeros(shape, dtype=bool),
    )
    spacing = Spacing3D(dx=np.ones(shape), dy=np.ones(shape), dz=np.ones(shape))
    face_metric = FaceMetricGeometry(
        _metric((3, 2, 2), 1.1), _metric((2, 3, 2), 1.2), _metric((2, 2, 3), 1.3)
    )
    cell_b = BFieldGeometry(np.broadcast_to(np.array([1.0, 0.0, 0.0]), shape + (3,)), np.ones(shape))
    face_b = FaceBFieldGeometry(
        BFieldGeometry(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (3, 2, 2, 3)), np.ones((3, 2, 2))),
        BFieldGeometry(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (2, 3, 2, 3)), np.ones((2, 3, 2))),
        BFieldGeometry(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (2, 2, 3, 3)), np.ones((2, 2, 3))),
    )
    return FciGeometry3D(grid, maps, spacing, _metric(shape), face_metric, cell_b, face_b)


def _atlas() -> FciVertexTraceAtlas:
    # Canonical raw transverse vertex lattice: x faces, theta vertices except
    # the duplicated periodic endpoint, and eta cell centers.
    shape = (3, 2, 2)
    positions = np.arange(np.prod(shape) * 3, dtype=np.float64).reshape(shape + (3,))
    return FciVertexTraceAtlas(
        vertex_positions=positions,
        forward_endpoint=positions + 0.1,
        backward_endpoint=positions - 0.1,
        forward_length=np.ones(shape), backward_length=np.ones(shape) * 2,
        forward_boundary=np.zeros(shape, dtype=bool), backward_boundary=np.ones(shape, dtype=bool),
        forward_endpoint_b_contra=np.ones(shape + (3,)), forward_endpoint_bmag=np.ones(shape),
        metadata={"producer": "analytic-fixture", "trace_substeps": 64},
    )


def _artifact() -> FciSimulationGeometry3D:
    geometry = _geometry()
    shape = geometry.shape
    owner = GlobalRlpParallelOverlapGeometry(
        raw_shape=shape,
        owner_flat_ids=np.arange(np.prod(shape), dtype=np.int64),
        owner_volumes=np.ones(np.prod(shape)),
        link_owner_a=np.array([0], dtype=np.int32), link_owner_b=np.array([1], dtype=np.int32),
        link_interface=np.array([0], dtype=np.int32), overlap_measure=np.array([0.25]),
        transmissibility=np.array([0.5]), metadata={"fixture": True},
        diagnostics={"interfaces": [{"max_closure_error": 0.0, "volume_weighted_closure_error": 0.0}]},
    )
    owner_index = np.stack(np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1)
    raw_volume = np.ones(shape)
    raw_centroid = np.zeros(shape + (3,))
    raw_second = np.zeros(shape + (3, 3))
    raw_third = np.zeros(shape + (3, 3, 3))
    topology = build_global_control_volume_topology_from_owner_map(
        owner_index=owner_index,
        positive_mask=np.ones(shape, dtype=bool),
        raw_volume=raw_volume,
        raw_centroid=raw_centroid,
        raw_second_moment=raw_second,
        raw_third_moment=raw_third,
        face_open_measure=(
            np.ones((shape[0] + 1, shape[1], shape[2])),
            np.ones((shape[0], shape[1] + 1, shape[2])),
            np.ones((shape[0], shape[1], shape[2] + 1)),
        ),
    )
    polar = PolarAngularAgglomerationGeometry3D(
        topology=topology,
        angular_group_size=np.ones(shape[0], dtype=np.int32),
        radial_centers=np.array([0.25, 0.75]), radial_widths=np.full(shape[0], 0.5),
        raw_volume=raw_volume, raw_chart_centroid=raw_centroid,
        raw_chart_second_moment=raw_second, raw_chart_third_moment=raw_third,
        raw_radial_centroid=np.zeros(shape), raw_radial_second_moment=np.zeros(shape),
        raw_radial_third_moment=np.zeros(shape),
        aggregate_chart_volume=topology.aggregate_volume,
        aggregate_chart_centroid=topology.aggregate_centroid,
        aggregate_chart_second_moment=topology.aggregate_second_moment,
        aggregate_chart_third_moment=topology.aggregate_third_moment,
        theta_period=2.0 * np.pi, eta_period=2.0 * np.pi, quadrature_order=3,
    )
    return FciSimulationGeometry3D(
        geometry=geometry,
        cell_positions=np.arange(np.prod(shape) * 3, dtype=np.float64).reshape(shape + (3,)),
        topology_name="periodic_closed_field_lines",
        nfp=4,
        vertex_traces=_atlas(), polar_angular_geometry=polar, owner_overlap=owner,
        curvature_edge_one_form=np.zeros(shape + (3,)),
        metadata={"campaign": "compact", "flags": [1, 2]},
        producer_validation_report={"valid": True},
    )


def _assert_exact_tree(actual, expected) -> None:
    """Compare every persisted dataclass field, including every array."""

    if is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_exact_tree(getattr(actual, field.name), getattr(expected, field.name))
        return
    if isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_exact_tree(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_exact_tree(actual_item, expected_item)
        return
    if hasattr(expected, "shape") and hasattr(expected, "dtype"):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        return
    assert actual == expected


def test_directory_artifact_round_trips_all_array_components(tmp_path: Path) -> None:
    original = _artifact()
    target = write_fci_simulation_geometry(original, tmp_path / "geometry")
    loaded = load_fci_simulation_geometry(target)

    _assert_exact_tree(loaded, original)


def test_round_trip_preserves_short_owner_conduction_to_roundoff(tmp_path: Path) -> None:
    """Serialization cannot change the finalized owner-link heat operator."""

    original = _artifact()
    loaded = load_fci_simulation_geometry(
        write_fci_simulation_geometry(original, tmp_path / "geometry")
    )
    before = lower_rlp_parallel_overlap_geometry(original.owner_overlap)
    after = lower_rlp_parallel_overlap_geometry(loaded.owner_overlap)
    initial = np.linspace(0.7, 1.4, np.prod(original.geometry.shape)).reshape(
        original.geometry.shape
    )

    def rk4(values, graph, dt=1.0e-3, chi=0.2):
        def rhs(state):
            return local_parallel_diffusion_fci_rlp_overlap_op(state, graph, chi)

        k1 = rhs(values)
        k2 = rhs(values + 0.5 * dt * k1)
        k3 = rhs(values + 0.5 * dt * k2)
        k4 = rhs(values + dt * k3)
        return values + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    np.testing.assert_allclose(
        np.asarray(rk4(initial, after)),
        np.asarray(rk4(initial, before)),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )


def test_one_artifact_has_equivalent_single_and_eta_sharded_runtime_arrays() -> None:
    geometry = _artifact().global_geometry
    single = build_local_fci_geometries(
        geometry,
        (1, 1, 1),
        periodic_axes=(False, True, True),
    )
    eta_sharded = build_local_fci_geometries(
        geometry,
        (1, 1, 2),
        periodic_axes=(False, True, True),
    )
    np.testing.assert_array_equal(single.cell_fields, eta_sharded.cell_fields)
    np.testing.assert_array_equal(single.map_fields, eta_sharded.map_fields)
    assert single.global_shape == eta_sharded.global_shape == geometry.shape


def test_manifest_has_separate_components_and_checksums_are_informational(tmp_path: Path) -> None:
    target = write_fci_simulation_geometry(_artifact(), tmp_path / "geometry")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["schema"] == "drbx.fci_simulation_geometry"
    assert set(manifest["components"]) == {
        "base_geometry", "center_maps", "cell_positions", "vertex_traces", "rlp_topology",
        "owner_overlap", "curvature_edge_one_form", "producer_validation_report"
    }
    for item in manifest["checksums"].values():
        assert len(item["sha256"]) == 64
    assert audit_fci_simulation_geometry_checksums(target)["valid"] is True
    # A stale informational digest does not make an otherwise readable payload unusable.
    manifest["checksums"]["cell_positions"]["sha256"] = "0" * 64
    (target / "manifest.json").write_text(json.dumps(manifest))
    audit = audit_fci_simulation_geometry_checksums(target)
    assert audit["valid"] is False
    assert audit["components"]["cell_positions"]["matches"] is False
    loaded = load_fci_simulation_geometry(target)
    np.testing.assert_array_equal(loaded.cell_positions, _artifact().cell_positions)


def test_validation_is_explicit_and_loader_does_not_qualify(tmp_path: Path) -> None:
    artifact = _artifact()
    report = validate_fci_simulation_geometry_producer(artifact, require_vertex_traces=True, require_owner_overlap=True)
    assert report["valid"] is True
    bad = FciSimulationGeometry3D(
        geometry=artifact.geometry, cell_positions=artifact.cell_positions,
        topology_name=artifact.topology_name, nfp=artifact.nfp,
    )
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad, require_vertex_traces=True)
    loaded = load_fci_simulation_geometry(write_fci_simulation_geometry(bad, tmp_path / "bad"))
    assert loaded.vertex_traces is None


def test_producer_validation_rejects_incomplete_vertex_atlas() -> None:
    artifact = _artifact()
    atlas = _atlas()
    # The extra periodic theta endpoint is not part of the canonical atlas.
    extra = np.concatenate((atlas.vertex_positions, atlas.vertex_positions[:, :1]), axis=1)
    malformed = replace(
        atlas,
        vertex_positions=extra,
        forward_endpoint=np.concatenate((atlas.forward_endpoint, atlas.forward_endpoint[:, :1]), axis=1),
        backward_endpoint=np.concatenate((atlas.backward_endpoint, atlas.backward_endpoint[:, :1]), axis=1),
        forward_length=np.concatenate((atlas.forward_length, atlas.forward_length[:, :1]), axis=1),
        backward_length=np.concatenate((atlas.backward_length, atlas.backward_length[:, :1]), axis=1),
        forward_boundary=np.concatenate((atlas.forward_boundary, atlas.forward_boundary[:, :1]), axis=1),
        backward_boundary=np.concatenate((atlas.backward_boundary, atlas.backward_boundary[:, :1]), axis=1),
        forward_endpoint_b_contra=np.concatenate((atlas.forward_endpoint_b_contra, atlas.forward_endpoint_b_contra[:, :1]), axis=1),
        forward_endpoint_bmag=np.concatenate((atlas.forward_endpoint_bmag, atlas.forward_endpoint_bmag[:, :1]), axis=1),
    )
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(replace(artifact, vertex_traces=malformed))


def test_producer_validation_rejects_wrong_trace_substeps() -> None:
    artifact = _artifact()
    bad_atlas = replace(artifact.vertex_traces, metadata={"trace_substeps": 32})
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(replace(artifact, vertex_traces=bad_atlas))


def test_producer_validation_rejects_owner_and_link_data_errors() -> None:
    bad_volume = _artifact()
    bad_volume.owner_overlap.owner_volumes[0] = 0.0
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad_volume)

    bad_overlap = _artifact()
    bad_overlap.owner_overlap.overlap_measure[0] = -1.0
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad_overlap)

    bad_transmissibility = _artifact()
    bad_transmissibility.owner_overlap.transmissibility[0] = np.nan
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad_transmissibility)

    bad_link = _artifact()
    bad_link.owner_overlap.link_owner_b[0] = 99
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad_link)


def test_producer_validation_rejects_closure_and_trace_qualification_failures() -> None:
    artifact = _artifact()
    diagnostics = {"interfaces": [{"max_closure_error": 1.0e-3, "volume_weighted_closure_error": 0.0}]}
    bad_closure = replace(
        artifact,
        owner_overlap=replace(artifact.owner_overlap, diagnostics=diagnostics),
    )
    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(bad_closure)

    with np.testing.assert_raises(ValueError):
        validate_fci_simulation_geometry_producer(
            artifact,
            require_trace_qualification=True,
            trace_qualification_report={"comparison": "64-vs-128", "passed": False},
        )

    report = validate_fci_simulation_geometry_producer(
        artifact,
        require_trace_qualification=True,
        trace_qualification={"comparison": "64-vs-128", "passed": True},
    )
    assert report["checks"]["trace_qualification_64_vs_128_passed"] is True
