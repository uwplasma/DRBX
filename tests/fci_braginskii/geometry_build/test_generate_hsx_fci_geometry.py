from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drbx.fci_braginskii.geometry_build import hsx_simulation_geometry as producer
from drbx.fci_braginskii.geometry_build import fci_simulation_geometry as artifact_io


def test_config_requires_toroidal_resolution() -> None:
    try:
        producer.HsxSimulationGeometryConfig(
            makegrid_path=Path("mgrid.nc"),
            vessel_path=Path("vessel.txt"),
            resolution=(4, 5, 8),
        )
    except ValueError as error:
        assert "even" in str(error)
    else:  # pragma: no cover
        raise AssertionError("odd toroidal NTHETA was accepted")


def test_producer_orchestration_and_resumable_status(monkeypatch, tmp_path: Path) -> None:
    shape = (3, 4, 4)
    grid = SimpleNamespace(
        x=SimpleNamespace(faces=np.linspace(0, 1, 4), centers=np.array([1 / 6, .5, 5 / 6])),
        y=SimpleNamespace(faces=np.linspace(0, 2 * np.pi, 5), centers=np.linspace(np.pi / 4, 7 * np.pi / 4, 4)),
        z=SimpleNamespace(faces=np.linspace(0, 4, 5), centers=np.array([.5, 1.5, 2.5, 3.5])),
    )
    geometry = SimpleNamespace(
        shape=shape,
        grid=grid,
        maps=SimpleNamespace(
            forward_boundary=np.zeros(shape, dtype=bool),
            backward_boundary=np.zeros(shape, dtype=bool),
        ),
    )
    evaluator = object()
    monkeypatch.setattr(producer, "_artifact_api", lambda: (object, object, lambda *_a, **_k: None, lambda *_a, **_k: None))
    monkeypatch.setattr(producer, "_call_builder", lambda _config: (geometry, np.zeros(shape + (3,)), 4, None, evaluator, None))
    monkeypatch.setattr(producer, "_continuous_field_callback", lambda *_a: object())
    sentinel_atlas = object()
    sentinel_host = object()
    sentinel_boundary = object()
    monkeypatch.setattr(producer, "_trace_atlas", lambda *_a, **_k: sentinel_atlas)
    monkeypatch.setattr(producer, "_read_stage_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(producer, "_write_stage_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(artifact_io, "_vertex_payload", lambda _value: {})
    monkeypatch.setattr(artifact_io, "_polar_payload", lambda _value: {})
    monkeypatch.setattr(artifact_io, "_overlap_payload", lambda _value: {})
    monkeypatch.setattr(producer, "build_metric_aware_polar_angular_agglomeration_geometry", lambda *_a, **_k: (sentinel_host, 1.0))
    monkeypatch.setattr(producer, "build_owner_boundary_overlap_geometry", lambda *_a, **_k: sentinel_boundary)
    events = []
    class FakeArtifact:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    def validator(artifact, **_kwargs):
        events.append(("validate", artifact, _kwargs))
        return {"valid": True}
    def writer(artifact, path):
        events.append(("write", artifact))
    monkeypatch.setattr(producer, "_artifact_api", lambda: (FakeArtifact, object, validator, writer))
    output = tmp_path / "geometry"
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"mgrid")
    vessel.write_text("vessel")
    config = producer.HsxSimulationGeometryConfig(
        makegrid_path=makegrid,
        vessel_path=vessel,
        resolution=(3, 4, 4),
        fit_sample_shape=(12, 13, 14),
        radial_degree=5,
        vertical_degree=6,
        toroidal_modes=7,
        metric_spline_degree=2,
        mmpde_iterations=11,
        metric_mesh_shape=(16, 18, 20),
        metric_radial_degree=9,
        metric_poloidal_modes=8,
        metric_toroidal_modes=6,
        eta_projection_iterations=3,
        include_curvature_edge_one_form=True,
        output=output,
    )
    monkeypatch.setattr(producer, "_call_builder", lambda _config: (geometry, np.zeros(shape + (3,)), 4, None, evaluator, None))
    # Use the real canonical construction shape while replacing expensive work.
    result = producer.build_hsx_simulation_geometry(config)
    assert isinstance(result, FakeArtifact)
    assert events[0][0] == "validate" and events[1][0] == "write"
    assert events[0][2]["require_trace_qualification"] is False
    assert result.vertex_traces is sentinel_atlas
    assert result.owner_overlap is sentinel_boundary
    assert result.metadata["fit_sample_shape"] == [12, 13, 14]
    assert result.metadata["radial_degree"] == 5
    assert result.metadata["vertical_degree"] == 6
    assert result.metadata["toroidal_modes"] == 7
    assert result.metadata["metric_spline_degree"] == 2
    assert result.metadata["mmpde_iterations"] == 11
    assert result.metadata["metric_mesh_shape"] == [16, 18, 20]
    assert result.metadata["metric_radial_degree"] == 9
    assert result.metadata["metric_poloidal_modes"] == 8
    assert result.metadata["metric_toroidal_modes"] == 6
    assert result.metadata["eta_projection_iterations"] == 3
    assert result.metadata["include_curvature_edge_one_form"] is True
    status = json.loads(
        (tmp_path / ".geometry.producer-checkpoints" / "process_status.json").read_text()
    )
    assert status["state"] == "completed"


def test_stage_checkpoint_reuse_requires_exact_producer_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vertex_traces.npz"
    payload = {"values": np.arange(6).reshape(2, 3)}
    producer._write_stage_checkpoint(checkpoint, "identity-a", payload)

    loaded = producer._read_stage_checkpoint(
        checkpoint,
        "identity-a",
        lambda data: data["values"],
    )
    np.testing.assert_array_equal(loaded, payload["values"])
    assert (
        producer._read_stage_checkpoint(
            checkpoint,
            "identity-b",
            lambda data: data["values"],
        )
        is None
    )
    assert not (checkpoint.parent / "manifest.json").exists()


def test_trace_qualification_includes_paths_entering_physical_core(monkeypatch) -> None:
    shape = (4, 8, 4)
    x_faces = np.linspace(0.0, 1.0, shape[0] + 1)
    y_faces = np.linspace(0.0, 2.0 * np.pi, shape[1] + 1)
    z_faces = np.linspace(0.0, 2.0 * np.pi, shape[2] + 1)
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    geometry = SimpleNamespace(
        shape=shape,
        grid=SimpleNamespace(
            x=SimpleNamespace(faces=x_faces, widths=np.diff(x_faces)),
            y=SimpleNamespace(faces=y_faces),
            z=SimpleNamespace(faces=z_faces, centers=z_centers),
        ),
    )
    source = np.stack(
        np.meshgrid(x_faces, y_faces[:-1], z_centers, indexing="ij"), axis=-1
    )
    endpoint = np.array(source, copy=True)
    entering = (
        np.isclose(source[..., 0], x_faces[2])
        & np.isclose(source[..., 1], y_faces[3])
        & np.isclose(source[..., 2], z_centers[1])
    )
    endpoint[entering, 0] = 0.01
    atlas = artifact_io.FciVertexTraceAtlas(
        vertex_positions=source,
        forward_endpoint=endpoint,
        backward_endpoint=endpoint,
        forward_length=np.ones(source.shape[:-1]),
        backward_length=np.ones(source.shape[:-1]),
        forward_boundary=np.zeros(source.shape[:-1], dtype=bool),
        backward_boundary=np.zeros(source.shape[:-1], dtype=bool),
        metadata={"trace_substeps": 64},
    )

    def reference_trace(_grid, _field, seeds, _step, **_kwargs):
        traced = np.array(seeds, copy=True)
        selected = (
            np.isclose(traced[:, 0], x_faces[2])
            & np.isclose(traced[:, 1], y_faces[3])
            & np.isclose(traced[:, 2], z_centers[1])
        )
        traced[selected, 0] = 0.01
        return {"endpoint": traced}

    monkeypatch.setattr(
        producer, "trace_fci_points_to_plane_from_callbacks", reference_trace
    )
    qualified = producer._trace_qualification(
        geometry, object(), atlas, tolerance_cells=1.0e-12
    )
    report = qualified.metadata["qualification"]
    assert report["passed"] is True
    assert report["core_entry_paths"]["forward"] >= 1
    assert report["core_entry_paths"]["backward"] >= 1


def test_trace_atlas_one_field_period_matches_full_torus() -> None:
    nx, ny, nz, nfp = 3, 6, 8, 4
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 2 * np.pi, ny + 1)
    z_faces = np.linspace(0.0, 2 * np.pi, nz + 1)

    def axis(faces):
        return SimpleNamespace(
            faces=faces, centers=0.5 * (faces[1:] + faces[:-1]), widths=np.diff(faces)
        )

    geometry = SimpleNamespace(
        shape=(nx, ny, nz),
        grid=SimpleNamespace(
            shape=(nx, ny, nz), x=axis(x_faces), y=axis(y_faces), z=axis(z_faces)
        ),
    )

    calls = []

    def field(points):
        calls.append(len(points))
        x, y, z = points.T
        b = np.stack(
            [
                0.2 * x * np.sin(nfp * z),
                0.3 + 0.1 * np.cos(nfp * z) * np.cos(y),
                1.0 + 0.2 * np.cos(nfp * z + y),
            ],
            axis=-1,
        )
        return b, np.linalg.norm(b, axis=-1)

    full = producer._trace_atlas(geometry, field, substeps=8)
    full_work, calls[:] = sum(calls), []
    period = producer._trace_atlas(geometry, field, substeps=8, field_periods=nfp)
    assert sum(calls) * nfp == full_work
    for name in ("forward_endpoint", "backward_endpoint", "forward_length", "backward_length"):
        np.testing.assert_allclose(
            getattr(period, name), getattr(full, name), rtol=0.0, atol=1.0e-12, err_msg=name
        )
    for name in ("forward_boundary", "backward_boundary"):
        np.testing.assert_array_equal(getattr(period, name), getattr(full, name))


def test_center_maps_one_field_period_match_full_torus() -> None:
    from drbx.fci_braginskii.geometry_build.fci_geometry import (
        CellCenteredGrid3D,
        Grid1D,
        build_fci_maps_from_callbacks,
    )

    nfp = 4

    def axis(lower, upper, n):
        faces = np.linspace(lower, upper, n + 1)
        return Grid1D(centers=0.5 * (faces[1:] + faces[:-1]), faces=faces)

    grid = CellCenteredGrid3D(
        x=axis(0.0, 1.0, 3), y=axis(0.0, 2 * np.pi, 6), z=axis(0.0, 2 * np.pi, 8)
    )
    calls = []

    def field(points):
        calls.append(len(points))
        x, y, z = points.T
        b = np.stack(
            [
                0.2 * x * np.sin(nfp * z),
                0.3 + 0.1 * np.cos(nfp * z) * np.cos(y),
                1.0 + 0.2 * np.cos(nfp * z + y),
            ],
            axis=-1,
        )
        return b, np.linalg.norm(b, axis=-1)

    kwargs = dict(substeps=8, axis_regular_axes=(True, False, False))
    full = build_fci_maps_from_callbacks(grid, field, **kwargs)
    full_rows, calls[:] = sum(calls), []
    period = build_fci_maps_from_callbacks(grid, field, field_periods=nfp, **kwargs)
    assert sum(calls) * nfp == full_rows
    assert set(period) == set(full)
    for name in full:
        a, b = np.asarray(period[name]), np.asarray(full[name])
        if a.dtype == bool:
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            np.testing.assert_allclose(a, b, rtol=0.0, atol=1.0e-12, err_msg=name)
