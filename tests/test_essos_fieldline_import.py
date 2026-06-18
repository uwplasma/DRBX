from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_drb.geometry import (
    build_essos_imported_fci_geometry,
    essos_runtime_available,
    load_essos_field_line_bundle_npz,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
)
from jax_drb.validation import (
    build_essos_imported_connection_length_refinement_diagnostics,
    build_live_essos_imported_connection_length_levels,
    create_essos_fieldline_import_package,
    create_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_connection_length_refinement_package,
    create_essos_imported_drb_movie_package,
    create_essos_imported_fci_campaign_package,
    create_essos_imported_pytree_campaign_package,
    create_essos_vmec_fieldline_surface_package,
)
import jax_drb.validation.essos_imported_fci_campaign as imported_fci_campaign


def _logical_coordinates(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    nx, ny, nz = shape
    rho = np.linspace(0.12, 0.34, nx)
    phi = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    minor_radius, toroidal_angle, poloidal_angle = np.meshgrid(
        rho,
        phi,
        theta,
        indexing="ij",
    )
    return {
        "minor_radius": minor_radius,
        "toroidal_angle": toroidal_angle,
        "poloidal_angle": poloidal_angle,
    }


def _has_essos_landreman_runtime() -> bool:
    if os.environ.get("JAX_DRB_RUN_ESSOS_TESTS") != "1":
        return False
    try:
        resolve_essos_landreman_qa_json()
        resolve_essos_landreman_qa_wout()
    except FileNotFoundError:
        return False
    return essos_runtime_available()


def test_imported_connection_length_refinement_campaign_is_self_contained(tmp_path: Path) -> None:
    artifacts = create_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "connection_length_refinement",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    arrays = np.load(artifacts.arrays_npz_path)

    assert report["passed"] is True
    assert report["manufactured"] is True
    assert report["diagnostics"]["level_count"] == 3
    assert report["finest_normalized_rms_error"] < 0.02
    assert report["finest_normalized_linf_error"] < 0.05
    assert report["minimum_observed_order_actual"] > 1.5
    assert report["observed_order_required"] is False
    assert report["diagnostics"]["observed_order_available"] is True
    assert arrays["level_0"].shape == (4, 6, 8)
    assert arrays["level_2"].shape == (16, 24, 32)
    assert arrays["pair_normalized_rms_error"][1] < arrays["pair_normalized_rms_error"][0]
    assert artifacts.plot_png_path.exists()


def test_connection_length_refinement_can_use_coordinate_interpolation() -> None:
    coarse_coordinates = _logical_coordinates((3, 4, 8))
    fine_coordinates = _logical_coordinates((6, 8, 16))
    coarse = 10.0 + 2.0 * coarse_coordinates["minor_radius"]
    fine = 10.0 + 2.0 * fine_coordinates["minor_radius"]

    block_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    coordinate_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )

    assert block_report["restriction_method"] == "block_average"
    assert coordinate_report["restriction_method"] == "coordinate_interpolation"
    assert block_report["passed"] is False
    assert coordinate_report["passed"] is True
    assert coordinate_report["pair_reports"][0]["normalized_linf_error"] < 1.0e-14


def test_connection_length_refinement_can_require_observed_order_for_promotion(tmp_path: Path) -> None:
    coarse_coordinates = _logical_coordinates((3, 4, 8))
    fine_coordinates = _logical_coordinates((6, 8, 16))
    coarse = 10.0 + 2.0 * coarse_coordinates["minor_radius"]
    fine = 10.0 + 2.0 * fine_coordinates["minor_radius"]

    advisory_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    promotion_report = build_essos_imported_connection_length_refinement_diagnostics(
        (coarse, fine),
        coordinate_levels=(coarse_coordinates, fine_coordinates),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
        require_observed_order=True,
    )
    packaged = create_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "test_connection_length_refinement_requires_order",
        level_shapes=((4, 6, 8), (8, 12, 16), (16, 24, 32)),
        require_observed_order=True,
    )
    packaged_report = json.loads(packaged.report_json_path.read_text(encoding="utf-8"))

    assert advisory_report["passed"] is True
    assert advisory_report["observed_order_available"] is False
    assert promotion_report["passed"] is False
    assert promotion_report["observed_order_required"] is True
    assert promotion_report["observed_order_available"] is False
    assert packaged_report["passed"] is True
    assert packaged_report["observed_order_required"] is True
    assert packaged_report["diagnostics"]["observed_order_available"] is True


def test_live_imported_connection_length_refinement_uses_geometry_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def fake_geometry(**kwargs):
        nx = int(kwargs["nx"])
        ny = int(kwargs["ny"])
        nz = int(kwargs["nz"])
        map_source = str(kwargs["map_source"])
        calls.append((map_source, nx, ny, nz))
        coordinates = _logical_coordinates((nx, ny, nz))
        level = 12.5 + 0.1 * coordinates["minor_radius"]
        return SimpleNamespace(
            connection_length=level,
            maps=SimpleNamespace(dphi=2.0 * np.pi / float(ny)),
            minor_radius=coordinates["minor_radius"],
            toroidal_angle=coordinates["toroidal_angle"],
            poloidal_angle=coordinates["poloidal_angle"],
            metadata={
                "map_source": map_source,
                "shape": [nx, ny, nz],
                "geometry_family": "test_imported_geometry",
            },
        )

    monkeypatch.setattr(
        imported_fci_campaign,
        "build_essos_imported_fci_geometry",
        fake_geometry,
    )

    levels = build_live_essos_imported_connection_length_levels(
        map_source="hybrid",
        level_shapes=((2, 2, 4), (4, 4, 8)),
    )

    assert calls == [("hybrid", 2, 2, 4), ("hybrid", 4, 4, 8)]
    assert levels.labels == ("hybrid_2x2x4", "hybrid_4x4x8")
    assert levels.levels[0].shape == (2, 2, 4)
    assert levels.coordinates[0]["minor_radius"].shape == (2, 2, 4)
    assert levels.metadata[0]["geometry_family"] == "test_imported_geometry"

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "live_connection_length_refinement",
        map_source="hybrid",
        level_shapes=((2, 2, 4), (4, 4, 8)),
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))

    assert report["live_imported"] is True
    assert report["manufactured"] is False
    assert report["map_source"] == "hybrid"
    assert report["source"] == "live imported connection-length refinement gate"
    assert report["diagnostics"]["level_labels"] == [
        "hybrid_2x2x4",
        "hybrid_4x4x8",
    ]
    assert report["diagnostics"]["restriction_method"] == "coordinate_interpolation"
    assert report["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


def test_live_imported_connection_length_refinement_can_use_step_length_per_radian(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_geometry(**kwargs):
        nx = int(kwargs["nx"])
        ny = int(kwargs["ny"])
        nz = int(kwargs["nz"])
        coordinates = _logical_coordinates((nx, ny, nz))
        dphi = 2.0 * np.pi / float(ny)
        level_per_radian = 3.0 + 0.2 * coordinates["minor_radius"]
        return SimpleNamespace(
            connection_length=99.0 + dphi * level_per_radian,
            adjacent_step_length=dphi * level_per_radian,
            target_exit_length=50.0 + coordinates["minor_radius"],
            maps=SimpleNamespace(dphi=dphi),
            minor_radius=coordinates["minor_radius"],
            toroidal_angle=coordinates["toroidal_angle"],
            poloidal_angle=coordinates["poloidal_angle"],
            metadata={
                "map_source": kwargs["map_source"],
                "shape": [nx, ny, nz],
                "geometry_family": "test_imported_geometry",
            },
        )

    monkeypatch.setattr(
        imported_fci_campaign,
        "build_essos_imported_fci_geometry",
        fake_geometry,
    )

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=tmp_path / "live_connection_length_refinement_per_radian",
        map_source="vmec",
        connection_quantity="parallel_step_per_toroidal_radian",
        level_shapes=((2, 2, 4), (4, 4, 8)),
        convergence_threshold=1.0e-12,
        linf_threshold=1.0e-12,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    arrays = np.load(artifacts.arrays_npz_path)

    assert report["connection_quantity"] == "parallel_step_per_toroidal_radian"
    assert report["diagnostics"]["restriction_method"] == "coordinate_interpolation"
    assert report["passed"] is True
    assert np.max(arrays["level_0"]) < 4.0


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_fieldline_import_generates_portable_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_fieldline_import_package(
        output_root=tmp_path / "essos_import",
        n_field_lines=3,
        times_to_trace=768,
        maxtime=150.0,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    bundle = load_essos_field_line_bundle_npz(artifacts.arrays_npz_path, metadata=report["metadata"])

    assert report["passed"] is True
    assert report["source"] == "ESSOS external field and field-line import"
    assert report["metadata"]["field_model"] == "essos.fields.BiotSavart"
    assert report["metadata"]["tracing_model"] == "essos.dynamics.Tracing(FieldLineAdaptative)"
    assert bundle.trajectories_xyz.shape == (3, 768, 3)
    assert bundle.field_sample_xyz.shape == bundle.field_sample_b_xyz.shape
    assert bundle.poincare_point_count == report["poincare_point_count"]
    assert bundle.poincare_point_count > 0
    assert np.all(np.isfinite(bundle.trajectories_xyz))
    assert np.all(np.isfinite(bundle.field_sample_b_xyz))
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_fci_maps_feed_native_sheath_and_neutral_gates(tmp_path: Path) -> None:
    geometry = build_essos_imported_fci_geometry(
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )

    assert geometry.shape == (3, 4, 6)
    assert geometry.metadata["geometry_family"] == "essos_imported_vmec_qa_fci"
    assert geometry.metadata["coordinate_model"] == "scaled_vmec_fourier_flux_surfaces"
    assert geometry.metadata["surface_nonaxisymmetric_major_rms"] > 1.0e-2
    assert np.all(np.isfinite(np.asarray(geometry.magnetic_field_magnitude)))
    assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
    assert geometry.adjacent_step_length is not None
    assert geometry.target_exit_length is not None
    assert geometry.forward_target_exit_length is not None
    assert geometry.backward_target_exit_length is not None
    assert np.any(np.isfinite(np.asarray(geometry.adjacent_step_length)))
    assert np.any(np.isfinite(np.asarray(geometry.target_exit_length)))
    assert np.any(np.isfinite(np.asarray(geometry.forward_target_exit_length)))
    assert np.any(np.isfinite(np.asarray(geometry.backward_target_exit_length)))
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool))) < 0.95
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.backward_boundary, dtype=bool))) < 0.95

    artifacts = create_essos_imported_fci_campaign_package(
        output_root=tmp_path / "essos_imported_fci",
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported field-line maps with jax_drb FCI closures"
    assert report["endpoint_length_diagnostics"]["passed"] is True
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    arrays = np.load(artifacts.arrays_npz_path)
    assert "target_exit_toroidal" in arrays.files
    assert "adjacent_step_toroidal" in arrays.files
    assert np.any(np.isfinite(arrays["target_exit_toroidal"]))


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil/VMEC inputs are not available")
def test_essos_imported_fci_map_sources_expose_coil_vmec_and_hybrid_semantics() -> None:
    geometries = {
        source: build_essos_imported_fci_geometry(
            map_source=source,
            nx=3,
            ny=4,
            nz=6,
            rho_min=0.12,
            rho_max=0.34,
            maxtime=40.0,
            times_to_trace=160,
        )
        for source in ("coil", "vmec", "hybrid")
    }

    for source, geometry in geometries.items():
        assert geometry.shape == (3, 4, 6)
        assert geometry.metadata["map_source"] == source
        assert np.all(np.isfinite(np.asarray(geometry.maps.forward_x)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.forward_z)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.backward_x)))
        assert np.all(np.isfinite(np.asarray(geometry.maps.backward_z)))
        assert np.all(np.isfinite(np.asarray(geometry.magnetic_field_magnitude)))
        assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
        assert geometry.metadata["surface_nonaxisymmetric_major_rms"] > 1.0e-2

    coil = geometries["coil"]
    vmec = geometries["vmec"]
    hybrid = geometries["hybrid"]
    coil_forward_fraction = float(np.mean(np.asarray(coil.maps.forward_boundary, dtype=bool)))
    coil_backward_fraction = float(np.mean(np.asarray(coil.maps.backward_boundary, dtype=bool)))
    vmec_forward_fraction = float(np.mean(np.asarray(vmec.maps.forward_boundary, dtype=bool)))
    vmec_backward_fraction = float(np.mean(np.asarray(vmec.maps.backward_boundary, dtype=bool)))
    hybrid_forward_fraction = float(np.mean(np.asarray(hybrid.maps.forward_boundary, dtype=bool)))
    hybrid_backward_fraction = float(np.mean(np.asarray(hybrid.maps.backward_boundary, dtype=bool)))

    assert 0.05 < coil_forward_fraction < 0.95
    assert 0.05 < coil_backward_fraction < 0.95
    assert vmec_forward_fraction == 0.0
    assert vmec_backward_fraction == 0.0
    assert hybrid_forward_fraction == coil_forward_fraction
    assert hybrid_backward_fraction == coil_backward_fraction
    assert np.allclose(np.asarray(hybrid.maps.forward_x), np.asarray(vmec.maps.forward_x))
    assert np.allclose(np.asarray(hybrid.maps.forward_z), np.asarray(vmec.maps.forward_z))
    assert not np.allclose(np.asarray(coil.maps.forward_z), np.asarray(vmec.maps.forward_z))
    assert float(np.min(np.asarray(vmec.connection_length))) > 0.0
    assert float(np.max(np.asarray(vmec.magnetic_field_magnitude)) / np.min(np.asarray(vmec.magnetic_field_magnitude))) > 1.01


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_fieldline_poincare_quantifies_scaled_vmec_surface_registration(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_fieldline_surface",
        rho_min=0.20,
        rho_max=0.82,
        n_surfaces=3,
        ntheta_surface=96,
        maxtime=180.0,
        times_to_trace=768,
        sections=(0.0, float(np.pi / 2.0)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS coil field-line tracing compared with scaled Landreman-Paul QA VMEC surfaces"
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert np.isfinite(report["same_surface_distance_normalized_p95"])
    assert np.isfinite(report["nearest_surface_distance_normalized_p95"])
    assert isinstance(report["fieldline_surface_match_passed"], bool)
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA VMEC wout are not available")
def test_essos_vmec_fieldline_poincare_preserves_scaled_vmec_surfaces(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_equilibrium_fieldline_surface",
        field_source="vmec",
        rho_min=0.20,
        rho_max=0.82,
        n_surfaces=3,
        ntheta_surface=96,
        maxtime=180.0,
        times_to_trace=768,
        sections=(0.0, float(np.pi / 2.0)),
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["field_source"] == "vmec"
    assert report["fieldline_surface_match_passed"] is True
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert report["fieldline_s_drift_max"] < 1.0e-7
    assert report["same_surface_distance_normalized_p95"] < 5.0e-2
    assert report["nearest_surface_distance_normalized_p95"] < 5.0e-2
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_maps_feed_pytree_jvp_rhs_gate(tmp_path: Path) -> None:
    artifacts = create_essos_imported_pytree_campaign_package(
        output_root=tmp_path / "essos_imported_pytree",
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
        steps=3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported field-line maps with JAXDRB fixed-layout PyTree RHS"
    assert report["jvp_relative_error"] < 1.0e-2
    assert report["vmap_serial_linf"] < 1.0e-6
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_maps_generate_drb_movie_gate(tmp_path: Path) -> None:
    artifacts = create_essos_imported_drb_movie_package(
        output_root=tmp_path / "essos_imported_drb_movie",
        nx=3,
        ny=4,
        nz=8,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=32.0,
        times_to_trace=120,
        frames=4,
        substeps_per_frame=2,
        dt=2.0e-3,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported Landreman-Paul QA coil FCI maps with JAXDRB fixed-layout DRB transient"
    assert report["final_potential_residual_l2"] < 5.0
    assert report["final_fluctuation_rms"] > 1.0e-4
    assert report["particle_recycling_relative_error"] < 1.0e-10
    assert report["neutral_particle_relative_error"] < 1.0e-10
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.snapshot_png_path.exists()
    assert artifacts.diagnostics_png_path.exists()
    assert artifacts.poster_png_path.exists()
    assert artifacts.movie_gif_path.exists()
