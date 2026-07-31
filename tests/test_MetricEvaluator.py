import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
from drbx.geometry.MetricEvaluator import (
    MetricEvaluator,
    build_metric_evaluator,
    build_wall_fitted_initial_mesh,
)
from drbx.geometry import (
    MetricQualityJumpLocation,
    MetricQualityLocation,
    MetricQualityRegion,
    MMPDEResult,
)
from drbx.geometry.ScalarPotential_evaluator import scalar_potential_evaluator_from_bfield
from drbx.geometry.WallEvaluator import WallEvaluator


def analytic_map(u, v, eta):
    # Smooth disk-like coordinates with a positive-radius toroidal embedding.
    theta = 3 * eta
    R = 2.0 + 0.25 * u + 0.15 * v + 0.04 * u * v + 0.03 * (1 - u**2) * np.cos(2 * theta)
    Z = 0.45 * (2 * u - 1) + 0.18 * (2 * v - 1) + 0.02 * u * v * np.sin(theta)
    delta = 0.035 * u * (1 - v) * np.cos(theta) + 0.018 * (2 * u - 1) * (2 * v - 1) * np.sin(2 * theta)
    return R, Z, delta


def analytic_derivatives(u, v, eta):
    theta = 3 * eta
    R_u = 0.25 + 0.04 * v - 0.06 * u * np.cos(2 * theta)
    R_v = 0.15 + 0.04 * u
    R_e = -0.18 * (1 - u**2) * np.sin(2 * theta)
    Z_u = 0.90 + 0.02 * v * np.sin(theta)
    Z_v = 0.36 + 0.02 * u * np.sin(theta)
    Z_e = 0.06 * u * v * np.cos(theta)
    d_u = 0.035 * (1 - v) * np.cos(theta) + 0.036 * (2 * v - 1) * np.sin(2 * theta)
    d_v = -0.035 * u * np.cos(theta) + 0.036 * (2 * u - 1) * np.sin(2 * theta)
    d_e = -0.105 * u * (1 - v) * np.sin(theta) + 0.108 * (2 * u - 1) * (2 * v - 1) * np.cos(2 * theta)
    return R_u, R_v, R_e, Z_u, Z_v, Z_e, d_u, d_v, d_e


def expected_position_and_A(q):
    u, v, eta = q[..., 0], q[..., 1], q[..., 2]
    R, Z, delta = analytic_map(u, v, eta)
    derivatives = analytic_derivatives(u, v, eta)
    phi = eta + delta
    cp, sp = np.cos(phi), np.sin(phi)
    position = np.stack((R * cp, R * sp, Z), axis=-1)
    cols = []
    for rd, pd, zd in zip(derivatives[:3], (derivatives[6], derivatives[7], 1 + derivatives[8]), derivatives[3:6]):
        cols.append(np.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1))
    return position, np.stack(cols, axis=-1)


def make_evaluator(nu=9, nv=8, neta=16):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    eta = np.arange(neta) * (2 * np.pi / 3 / neta)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    R, Z, delta = analytic_map(ug, vg, eg)
    positions = np.stack((R * np.cos(eg + delta), R * np.sin(eg + delta), Z), axis=-1)
    return MetricEvaluator(u, v, eta, positions, nfp=3)


def make_cylindrical_evaluator(u=None, v=None, eta=None):
    u = np.linspace(0.0, 1.0, 6) if u is None else np.asarray(u, dtype=float)
    v = np.linspace(0.0, 1.0, 5) if v is None else np.asarray(v, dtype=float)
    eta = (
        np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
        if eta is None else np.asarray(eta, dtype=float)
    )
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    radius = 3.0 + 0.1 * ug + 0.4 * vg
    height = 0.5 * ug + 0.2 * vg
    positions = np.stack(
        (radius * np.cos(eg), radius * np.sin(eg), height), axis=-1
    )
    return MetricEvaluator(u, v, eta, positions, period=2.0 * np.pi)


class CylindricalToroidalField:
    def __init__(self, tilt=0.0):
        self.tilt = float(tilt)

    def evaluate_cartesian(self, points):
        points = np.asarray(points, dtype=float)
        radius = np.hypot(points[..., 0], points[..., 1])
        ephi = np.stack(
            (-points[..., 1] / radius, points[..., 0] / radius, np.zeros_like(radius)),
            axis=-1,
        )
        return ephi + self.tilt * np.array([0.0, 0.0, 1.0])


def test_position_and_jacobian_match_analytic_map():
    evaluator = make_evaluator()
    rng = np.random.default_rng(1234)
    q = np.column_stack((rng.random(13), rng.random(13), rng.random(13) * evaluator.period))
    expected_x, expected_A = expected_position_and_A(q)
    np.testing.assert_allclose(evaluator.position(q), expected_x, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(evaluator.jacobian_matrix(q), expected_A, rtol=2e-9, atol=2e-9)


def test_metrics_are_consistent_and_jacobian_positive():
    evaluator = make_evaluator()
    q = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.3], [1.0, 1.0, evaluator.period * 0.999]])
    result = evaluator.evaluate(q)
    assert np.all(result.signed_J > 0)
    identity = np.broadcast_to(np.eye(3), result.covariant_metric.shape)
    np.testing.assert_allclose(np.einsum("...ik,...kj->...ij", result.covariant_metric, result.contravariant_metric), identity, atol=2e-12)
    assert np.max(result.inverse_residual) < 2e-12


def test_quasiperiodicity_and_batch_shapes():
    evaluator = make_evaluator()
    q = np.array([[[0.2, 0.4, 0.1], [0.7, 0.1, 0.8]], [[0.4, 0.8, 0.2], [0.1, 0.9, 0.5]]])
    x = evaluator.position(q)
    xp = evaluator.position(q + np.array([0, 0, evaluator.period]))
    angle = evaluator.period
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(xp, x @ rot.T, atol=3e-11)
    assert x.shape == (2, 2, 3)
    assert evaluator.jacobian_matrix(q).shape == (2, 2, 3, 3)


def test_position_uses_value_only_channel_path(monkeypatch):
    evaluator = make_evaluator()
    derivative_orders = []
    for channel in evaluator._channels:
        original = channel.evaluate_prepared

        def recording_evaluate(
            prepared, du=0, dv=0, deta=0, *, _original=original
        ):
            derivative_orders.append((du, dv, deta))
            return _original(prepared, du=du, dv=dv, deta=deta)

        monkeypatch.setattr(channel, "evaluate_prepared", recording_evaluate)

    evaluator.position(np.array([[0.2, 0.4, 0.1], [0.7, 0.8, 0.3]]))
    assert derivative_orders == [(0, 0, 0)] * 3


def test_field_transformation_and_reconstruction():
    evaluator = make_evaluator()

    class BField:
        def evaluate_cartesian(self, points):
            return np.stack((points[..., 1] + 1, 2 * points[..., 2], points[..., 0] - 0.5), axis=-1)

    q = np.array([[0.15, 0.25, 0.2], [0.75, 0.65, 0.7]])
    fields = evaluator.evaluate_magnetic_field(q, BField())
    A = evaluator.jacobian_matrix(q)
    np.testing.assert_allclose(np.einsum("...ij,...j->...i", A, fields.B_contravariant), fields.B_cartesian, atol=2e-12)
    np.testing.assert_allclose(np.einsum("...ji,...j->...i", A, fields.B_cartesian), fields.B_covariant, atol=2e-12)
    np.testing.assert_allclose(fields.magnitude, np.linalg.norm(fields.B_cartesian, axis=-1))


def test_structured_sampling():
    evaluator = make_evaluator()
    result = evaluator.sample(np.linspace(0, 1, 3), np.linspace(0, 1, 4), np.arange(5) * evaluator.period / 5)
    assert result.position.shape == (3, 4, 5, 3)
    assert result.signed_J.shape == (3, 4, 5)


def test_finite_volume_sampling_helpers_exclude_square_corners():
    evaluator = make_evaluator()
    centers = evaluator.cell_center_logical_points()
    faces = evaluator.open_boundary_face_center_logical_points()
    assert centers.shape == (
        evaluator.u.size - 1,
        evaluator.v.size - 1,
        evaluator.eta.size,
        3,
    )
    assert faces.shape[-1] == 3
    assert not np.any(
        ((faces[:, 0] == 0.0) | (faces[:, 0] == 1.0))
        & ((faces[:, 1] == 0.0) | (faces[:, 1] == 1.0))
    )
    assert np.all(evaluator.sample_cell_centers().valid)
    assert np.all(evaluator.sample_open_boundary_faces().valid)


def test_quality_report_combines_cells_and_open_faces():
    evaluator = make_evaluator()
    report = evaluator.quality_report()
    cell_count = (
        (evaluator.u.size - 1)
        * (evaluator.v.size - 1)
        * evaluator.eta.size
    )
    expected_count = 8 * cell_count + evaluator.open_boundary_face_center_logical_points().shape[0]
    assert report.sample_count == expected_count
    assert report.points_per_cell == 8
    assert report.quadrature_sample_count == 8 * cell_count
    assert report.face_sample_count == evaluator.open_boundary_face_center_logical_points().shape[0]
    assert report.nonpositive_J_count == 0
    assert report.nonpositive_J_fraction == 0.0
    assert report.valid_fraction == 1.0
    assert 0.0 < report.raw_J_min <= report.raw_J_p01 <= report.raw_J_median <= report.raw_J_max
    assert 0.0 < report.raw_J_min_over_median <= 1.0
    assert 0.0 < report.scaled_J_min <= report.scaled_J_p01 <= 1.0
    assert 1.0 <= report.mapping_condition_median <= report.mapping_condition_p95 <= report.mapping_condition_max
    assert np.isfinite(report.max_neighbor_log_J_jump)
    assert report.inverse_residual_max < 1e-10
    summary = report.summary("synthetic")
    assert summary.startswith("synthetic: ")
    assert "samples=" in summary
    assert "scaled_J=" in summary
    assert "cond(H)=" in summary
    assert "max_dlogV=" in summary


def test_quality_report_can_sample_a_grid_different_from_evaluator_nodes():
    evaluator = make_evaluator(nu=7, nv=6, neta=10)
    requested_counts = (9, 8, 14)
    report = evaluator.quality_report(logical_cell_counts=requested_counts)

    assert report.cell_count == int(np.prod(requested_counts))
    assert report.quadrature_sample_count == report.points_per_cell * report.cell_count
    assert report.representation_metadata["mesh_shape"] == (7, 6, 10)
    assert report.representation_metadata["evaluator_cell_counts"] == (6, 5, 10)
    assert report.representation_metadata["quality_cell_counts"] == requested_counts


def test_quality_report_has_region_and_location_diagnostics():
    evaluator = make_evaluator()
    report = evaluator.quality_report()
    labels = [region.label for region in report.regions]
    assert labels == [
        "all",
        "core",
        "wall_adjacent",
        "u_min_adjacent",
        "u_max_adjacent",
        "v_min_adjacent",
        "v_max_adjacent",
        "corner_adjacent",
    ]
    regions = {region.label: region for region in report.regions}
    nu = evaluator.u.size - 1
    nv = evaluator.v.size - 1
    neta = evaluator.eta.size
    q = report.points_per_cell
    assert regions["all"].sample_count == nu * nv * neta * q
    assert regions["core"].sample_count == (nu - 2) * (nv - 2) * neta * q
    assert regions["wall_adjacent"].sample_count == (
        nu * nv - (nu - 2) * (nv - 2)
    ) * neta * q
    assert regions["u_min_adjacent"].sample_count == nv * neta * q
    assert regions["u_max_adjacent"].sample_count == nv * neta * q
    assert regions["v_min_adjacent"].sample_count == nu * neta * q
    assert regions["v_max_adjacent"].sample_count == nu * neta * q
    assert regions["corner_adjacent"].sample_count == 4 * neta * q
    assert all(0.0 <= region.valid_fraction <= 1.0 for region in report.regions)
    assert all(region.nonpositive_J_count == 0 for region in report.regions)
    assert all(region.scaled_J_min > 0.0 for region in report.regions)
    assert all(region.mapping_condition_max >= 1.0 for region in report.regions)
    assert all(region.scaled_J_p01 >= region.scaled_J_min for region in report.regions)
    assert all(
        region.mapping_condition_max >= region.mapping_condition_p95 >= 1.0
        for region in report.regions
    )
    assert all(region.stretch_max >= region.stretch_p95 >= 1.0 for region in report.regions)
    assert all(region.volume_p01_over_median > 0.0 for region in report.regions)
    assert all(region.volume_p99_over_p01 >= 1.0 for region in report.regions)
    for extremum in (report.worst_scaled_jacobian, report.worst_mapping_condition):
        assert extremum is not None
        assert extremum.region in labels
        assert len(extremum.logical) == 3
        assert len(extremum.cartesian) == 3
        assert np.all(np.isfinite(extremum.logical))
        assert np.all(np.isfinite(extremum.cartesian))
    assert report.max_neighbor_log_J_jump_axis in {"u", "v", "eta", "eta (periodic seam)"}
    assert len(report.max_neighbor_log_J_jump_endpoint_a) == 3
    assert len(report.max_neighbor_log_J_jump_endpoint_b) == 3
    if report.max_neighbor_log_J_jump_axis == "eta (periodic seam)":
        assert report.max_neighbor_log_J_jump_endpoint_b[2] > evaluator.period
    detail = report.detailed_summary("synthetic detail")
    assert detail.startswith("synthetic detail:\n")
    assert "all:" in detail
    assert "u_min_adjacent:" in detail
    assert "worst_scaled_J:" in detail
    assert "worst_cond(H):" in detail
    assert "q_a=" in detail and "q_b=" in detail


def test_quality_region_and_location_types_are_package_exports():
    assert MetricQualityRegion.__name__ == "MetricQualityRegion"
    assert MetricQualityLocation.__name__ == "MetricQualityLocation"
    assert MetricQualityJumpLocation.__name__ == "MetricQualityJumpLocation"


def test_quality_report_diagnostics_survive_inverted_map():
    evaluator = make_evaluator()
    u = evaluator.u
    v = evaluator.v
    eta = evaluator.eta
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    R, Z, delta = analytic_map(ug, vg, eg)
    positions = np.stack((R * np.cos(eg + delta), R * np.sin(eg + delta), -Z), axis=-1)
    inverted = MetricEvaluator(evaluator.u, evaluator.v, evaluator.eta, positions, nfp=3)
    report = inverted.quality_report()
    assert report.valid_fraction < 1.0
    assert report.nonpositive_J_count > 0
    assert report.nonpositive_J_fraction > 0.0
    assert report.detailed_summary("inverted")


def test_quality_report_gauss_points_detect_interior_defect_missed_by_centers(
    monkeypatch,
):
    evaluator = make_cylindrical_evaluator()
    original = evaluator._position_and_jacobian
    target_u = 0.5 * (
        evaluator.u[0]
        + evaluator.u[1]
        - (evaluator.u[1] - evaluator.u[0]) / np.sqrt(3.0)
    )

    def defective_jacobian(points):
        position, jacobian = original(points)
        logical = np.asarray(points, dtype=float)
        jacobian = jacobian.copy()
        defect = np.isclose(logical[..., 0], target_u, rtol=0.0, atol=1e-13)
        jacobian[..., :, 0] *= np.where(defect, -1.0, 1.0)[..., None]
        return position, jacobian

    monkeypatch.setattr(evaluator, "_position_and_jacobian", defective_jacobian)
    assert np.all(evaluator.sample_cell_centers(reject_nonpositive_J=False).valid)

    report = evaluator.quality_report(gauss_order=2)
    assert report.nonpositive_J_count > 0
    assert report.valid_fraction < 1.0
    assert report.worst_jacobian is not None
    assert report.worst_jacobian.value < 0.0


def test_quality_report_keeps_wall_face_validity_separate(monkeypatch):
    evaluator = make_cylindrical_evaluator()
    original = evaluator._position_and_jacobian

    def defective_wall_jacobian(points):
        position, jacobian = original(points)
        logical = np.asarray(points, dtype=float)
        jacobian = jacobian.copy()
        defect = np.isclose(logical[..., 0], 0.0, rtol=0.0, atol=1e-14)
        jacobian[..., :, 0] *= np.where(defect, -1.0, 1.0)[..., None]
        return position, jacobian

    monkeypatch.setattr(
        evaluator, "_position_and_jacobian", defective_wall_jacobian
    )
    report = evaluator.quality_report()
    assert report.valid_fraction == 1.0
    assert report.nonpositive_J_count == 0
    assert report.face_nonpositive_J_count > 0
    assert report.face_valid_fraction < 1.0


def test_quality_report_gauss_order_controls_interior_sampling():
    evaluator = make_evaluator(nu=5, nv=4, neta=8)
    cells = (evaluator.u.size - 1) * (evaluator.v.size - 1) * evaluator.eta.size
    faces = evaluator.open_boundary_face_center_logical_points().shape[0]

    default = evaluator.quality_report()
    high_resolution = evaluator.quality_report(gauss_order=3)

    assert default.points_per_cell == 2**3
    assert default.quadrature_sample_count == cells * 2**3
    assert default.sample_count == cells * 2**3 + faces
    assert high_resolution.points_per_cell == 3**3
    assert high_resolution.quadrature_sample_count == cells * 3**3
    assert high_resolution.sample_count == cells * 3**3 + faces
    assert high_resolution.face_sample_count == default.face_sample_count == faces
    assert default.representation_metadata["gauss_order"] == 2
    assert high_resolution.representation_metadata["gauss_order"] == 3

    with pytest.raises(ValueError, match="gauss"):
        evaluator.quality_report(gauss_order=1)


def test_quality_report_expanded_H_shape_and_volume_statistics_are_ordered():
    evaluator = make_cylindrical_evaluator(
        u=np.array([0.0, 0.08, 0.35, 1.0]),
        v=np.array([0.0, 0.2, 0.55, 1.0]),
    )
    report = evaluator.quality_report(gauss_order=3)

    assert report.raw_volume_min > 0.0
    assert 0.0 < report.volume_min_over_median <= report.volume_p01_over_median
    assert report.volume_p01_over_median <= report.volume_p05_over_median <= 1.0
    assert 1.0 <= report.volume_p95_over_median <= report.volume_p99_over_median
    assert report.volume_p99_over_p01 >= 1.0
    assert report.volume_coefficient_of_variation >= 0.0
    assert 0.0 < report.scaled_J_min <= report.scaled_J_p01
    assert report.scaled_J_p01 <= report.scaled_J_p05 <= report.scaled_J_median <= 1.0
    assert 1.0 <= report.mapping_condition_median <= report.mapping_condition_p95
    assert report.mapping_condition_p95 <= report.mapping_condition_p99 <= report.mapping_condition_max
    assert 1.0 <= report.stretch_median <= report.stretch_p95
    assert report.stretch_p95 <= report.stretch_p99 <= report.stretch_max

    assert set(report.angle_cosines) == {"uv", "ueta", "veta"}
    for p95, maximum in report.angle_cosines.values():
        assert 0.0 <= p95 <= maximum <= 1.0

    for extremum in (
        report.worst_jacobian,
        report.worst_volume,
        report.worst_scaled_jacobian,
        report.worst_mapping_condition,
        report.worst_stretch,
        report.worst_angle_cosine,
    ):
        assert extremum is not None
        assert extremum.cell_index is not None
        assert len(extremum.cell_index) == 3
        assert np.all(np.isfinite(extremum.logical))
        assert np.all(np.isfinite(extremum.cartesian))

    metadata = report.representation_metadata
    assert tuple(metadata["mesh_shape"]) == (
        evaluator.u.size,
        evaluator.v.size,
        evaluator.eta.size,
    )
    assert tuple(metadata["cell_counts"]) == (
        evaluator.u.size - 1,
        evaluator.v.size - 1,
        evaluator.eta.size,
    )


def test_quality_report_directional_smoothness_includes_periodic_seam():
    report = make_evaluator(nu=6, nv=5, neta=12).quality_report()
    expected = {"u", "v", "eta", "eta_seam"}
    assert set(report.directional_log_volume_jumps) == expected
    assert set(report.directional_K_jumps) == expected
    for diagnostics in (
        report.directional_log_volume_jumps,
        report.directional_K_jumps,
    ):
        for p95, maximum in diagnostics.values():
            assert np.isfinite(p95)
            assert np.isfinite(maximum)
            assert 0.0 <= p95 <= maximum
    assert report.worst_volume_jump is not None
    assert report.worst_K_jump is not None
    for jump in (report.worst_volume_jump, report.worst_K_jump):
        assert jump.direction in expected
        assert len(jump.cell_index_a) == len(jump.cell_index_b) == 3
        assert jump.region_a != "unknown"
        assert jump.region_b != "unknown"
        assert np.all(np.isfinite(jump.logical_a))
        assert np.all(np.isfinite(jump.logical_b))


class _CylindricalEtaEvaluator:
    period = 2.0 * np.pi
    nfp = 1

    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def evaluate_cartesian(self, points, *, wrapped=True):
        points = np.asarray(points, dtype=float)
        eta = np.arctan2(points[..., 1], points[..., 0]) + self.offset
        return np.mod(eta, self.period) if wrapped else eta


def test_quality_report_eta_constraint_and_periodic_seam_checks():
    evaluator = make_cylindrical_evaluator()
    exact = evaluator.quality_report(eta_evaluator=_CylindricalEtaEvaluator())

    assert set(exact.eta_constraint_residuals) >= {"median", "p95", "p99", "max"}
    assert exact.eta_constraint_residuals["max"] < 1e-10
    assert exact.worst_eta_constraint is not None
    assert set(exact.periodic_seam_residuals) == {
        "position",
        "jacobian_matrix",
        "metric_tensor",
        "J",
    }
    assert max(exact.periodic_seam_residuals.values()) < 1e-9

    offset = 2.5e-3
    shifted = evaluator.quality_report(
        eta_evaluator=_CylindricalEtaEvaluator(offset=offset)
    )
    assert shifted.eta_constraint_residuals["median"] == pytest.approx(offset, abs=1e-10)
    assert shifted.eta_constraint_residuals["max"] == pytest.approx(offset, abs=1e-10)
    assert shifted.worst_eta_constraint is not None
    assert shifted.worst_eta_constraint.value == pytest.approx(offset, abs=1e-10)


def test_quality_report_validates_eta_evaluator_output():
    evaluator = make_cylindrical_evaluator()

    class BadShape(_CylindricalEtaEvaluator):
        def evaluate_cartesian(self, points, *, wrapped=True):
            return 0.0

    class Nonfinite(_CylindricalEtaEvaluator):
        def evaluate_cartesian(self, points, *, wrapped=True):
            return np.full(np.asarray(points).shape[:-1], np.nan)

    with pytest.raises(ValueError, match="one value"):
        evaluator.quality_report(eta_evaluator=BadShape())
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.quality_report(eta_evaluator=Nonfinite())


def test_quality_report_includes_mmpde_optimization_metadata():
    evaluator = make_cylindrical_evaluator()
    evaluator._mmpde_result = MMPDEResult(
        positions=np.empty((0, 3)),
        converged=False,
        iterations=2,
        energy_history=np.array([10.0, 6.0, 5.0]),
        residual_history=np.array([2.0e-2, 7.0e-3]),
        minimum_jacobian_history=np.array([0.2, 0.3, 0.4]),
        component_energy_history={
            "alignment": np.array([4.0, 2.0]),
            "volume_smoothness": np.array([3.0, 1.5]),
        },
    )
    evaluator._mmpde_fit_scale = 0.75

    report = evaluator.quality_report()
    metadata = report.mmpde_metadata
    assert metadata["converged"] is False
    assert metadata["iterations"] == 2
    assert metadata["initial_energy"] == 10.0
    assert metadata["final_energy"] == 5.0
    assert metadata["final_over_initial_energy"] == 0.5
    assert metadata["final_max_nodal_update"] == 7.0e-3
    assert metadata["initial_minimum_discrete_J"] == 0.2
    assert metadata["final_minimum_discrete_J"] == 0.4
    assert metadata["alignment_final_over_initial"] == 0.5
    assert report.representation_metadata["mmpde_fit_scale"] == 0.75
    assert "Optimization:" in report.detailed_summary()


def test_large_structured_sampling_is_chunked():
    evaluator = make_evaluator()
    result = evaluator.sample(
        np.linspace(0, 1, 12),
        np.linspace(0, 1, 24),
        np.arange(16) * evaluator.period / 16,
    )
    assert result.position.shape == (12, 24, 16, 3)
    assert np.all(np.isfinite(result.signed_J))


@pytest.mark.parametrize("bad", ["axes", "shape", "eta", "radius", "orientation"])
def test_invalid_inputs_are_rejected(bad):
    u = np.linspace(0, 1, 6)
    v = np.linspace(0, 1, 6)
    eta = np.arange(8) * (2 * np.pi / 8)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    R, Z, delta = analytic_map(ug, vg, eg)
    positions = np.stack((R * np.cos(eg + delta), R * np.sin(eg + delta), Z), axis=-1)
    if bad == "axes":
        with pytest.raises(ValueError):
            MetricEvaluator(np.array([0, 0.5, 0.4, 1]), v, eta, positions, nfp=1)
    elif bad == "shape":
        with pytest.raises(ValueError):
            MetricEvaluator(u, v, eta, positions[..., :2], nfp=1)
    elif bad == "eta":
        with pytest.raises(ValueError):
            MetricEvaluator(u, v, eta + np.linspace(0, 0.1, eta.size), positions, nfp=1)
    elif bad == "radius":
        positions[0, 0, 0, :2] = 0
        with pytest.raises(ValueError):
            MetricEvaluator(u, v, eta, positions, nfp=1)
    else:
        with pytest.raises(ValueError):
            MetricEvaluator(u, v, eta, positions[:, ::-1], nfp=1).evaluate(np.array([0.5, 0.5, 0.1]))


class SyntheticEtaEvaluator:
    period = 2.0 * np.pi

    def evaluate_cartesian(self, points, *, wrapped=False):
        points = np.asarray(points, dtype=float)
        values = np.arctan2(points[..., 1], points[..., 0])
        if wrapped:
            values = np.mod(values, self.period)
        return values

    def gradient_cartesian(self, points):
        points = np.asarray(points, dtype=float)
        radius_squared = np.sum(points[..., :2] ** 2, axis=-1)
        return np.stack((-points[..., 1] / radius_squared, points[..., 0] / radius_squared, np.zeros_like(radius_squared)), axis=-1)


class FiniteOnlyEtaEvaluator(SyntheticEtaEvaluator):
    """Synthetic evaluator used to catch accidental NaN plot queries."""

    def evaluate_cartesian(self, points, *, wrapped=False):
        points = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(points)):
            raise AssertionError("plot passed nonfinite points to eta evaluator")
        return super().evaluate_cartesian(points, wrapped=wrapped)


class SyntheticWallEvaluator:
    nfp = 1
    period = 2.0 * np.pi

    def constant_eta_boundary_curve(self, eta_evaluator, eta, npoints=256):
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        phi = float(eta)
        radius = 3.0 + 0.45 * np.cos(theta) + 0.08 * np.cos(2.0 * theta)
        height = 0.55 * np.sin(theta)
        return np.stack(
            (radius * np.cos(phi), radius * np.sin(phi), height), axis=-1
        )


class PhaseStableShapeWallEvaluator(SyntheticWallEvaluator):
    """Wall whose extrema move with eta while its poloidal phase is fixed."""

    def constant_eta_boundary_curve(self, eta_evaluator, eta, npoints=256):
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        phi = float(eta)
        radius = (
            3.0
            + (0.42 + 0.06 * np.sin(phi)) * np.cos(theta)
            + (0.08 + 0.02 * np.cos(phi)) * np.cos(2.0 * theta)
        )
        height = (0.55 + 0.04 * np.cos(phi)) * np.sin(theta) + 0.025 * np.sin(phi) * np.sin(2.0 * theta)
        return np.stack((radius * np.cos(phi), radius * np.sin(phi), height), axis=-1)


def make_eta_constrained_mesh(nu=5, nv=4, neta=8, period=2.0 * np.pi):
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    eta = np.arange(neta) * (period / neta)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    R = 3.0 + 0.5 * vg + 0.03 * np.sin(eg)
    Z = 0.5 * ug - 0.25 + 0.02 * np.cos(eg)
    return np.stack((R * np.cos(eg), R * np.sin(eg), Z), axis=-1)


def test_build_metric_evaluator_runs_mmpde_and_preserves_eta_and_boundary(monkeypatch):
    metric_module = importlib.import_module("drbx.geometry.MetricEvaluator")
    original_solve = metric_module.solve_mmpde
    calls = []

    def recording_solve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(metric_module, "solve_mmpde", recording_solve)
    initial = make_eta_constrained_mesh()
    evaluator = build_metric_evaluator(
        SyntheticEtaEvaluator(),
        initial,
        options=metric_module.MMPDEOptions(max_iterations=4, initial_step=0.05),
    )

    assert len(calls) == 1
    assert isinstance(evaluator, MetricEvaluator)
    assert evaluator.mmpde_result is not None
    assert evaluator.mmpde_result.iterations <= 4
    assert 0.0 <= evaluator.mmpde_fit_scale <= 1.0
    assert evaluator.nfp is None
    u, v, eta = evaluator.u, evaluator.v, evaluator.eta
    grid = np.stack(np.meshgrid(u, v, eta, indexing="ij"), axis=-1)
    fitted = evaluator.position(grid)
    values = SyntheticEtaEvaluator().evaluate_cartesian(fitted, wrapped=True)
    residual = (values - eta[None, None, :] + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(residual)) < 5.0e-8

    boundary = np.zeros(initial.shape[:3], dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    np.testing.assert_allclose(fitted[boundary], initial[boundary], atol=2.0e-11)
    assert np.all(evaluator.evaluate(grid).signed_J > 0)


def test_build_metric_evaluator_preserves_consistent_nfp_metadata():
    class NfpEtaEvaluator(SyntheticEtaEvaluator):
        period = 2.0 * np.pi / 3.0
        nfp = 3

    eta_evaluator = NfpEtaEvaluator()
    initial = make_eta_constrained_mesh(period=eta_evaluator.period)
    evaluator = build_metric_evaluator(
        eta_evaluator,
        initial,
        options=importlib.import_module("drbx.geometry.MetricEvaluator").MMPDEOptions(max_iterations=0),
    )

    assert evaluator.nfp == 3
    np.testing.assert_allclose(evaluator.period, 2.0 * np.pi / 3.0)


def test_build_metric_evaluator_rejects_inconsistent_nfp_metadata():
    class InconsistentEtaEvaluator(SyntheticEtaEvaluator):
        period = 2.0 * np.pi / 3.0
        nfp = 2

    with pytest.raises(ValueError, match="inconsistent"):
        build_metric_evaluator(
            InconsistentEtaEvaluator(),
            make_eta_constrained_mesh(period=2.0 * np.pi / 3.0),
        )


@pytest.mark.parametrize("bad_nfp", [0, -1, 2.5, "3", True])
def test_build_metric_evaluator_rejects_invalid_nfp_metadata(bad_nfp):
    class InvalidNfpEtaEvaluator(SyntheticEtaEvaluator):
        period = 2.0 * np.pi / 3.0

    eta_evaluator = InvalidNfpEtaEvaluator()
    eta_evaluator.nfp = bad_nfp
    with pytest.raises(ValueError, match="positive integer"):
        build_metric_evaluator(
            eta_evaluator,
            make_eta_constrained_mesh(period=eta_evaluator.period),
        )


def test_build_metric_evaluator_accepts_explicit_axes_and_rejects_bad_inputs():
    eta_evaluator = SyntheticEtaEvaluator()
    initial = make_eta_constrained_mesh(4, 4, 8)
    axes = (
        np.linspace(0.0, 1.0, 4),
        np.linspace(0.0, 1.0, 4),
        np.arange(8) * eta_evaluator.period / 8,
    )
    evaluator = build_metric_evaluator(
        eta_evaluator,
        initial,
        logical_axes=axes,
        options=importlib.import_module("drbx.geometry.MetricEvaluator").MMPDEOptions(max_iterations=0),
    )
    np.testing.assert_allclose(evaluator.eta, axes[2])

    with pytest.raises(ValueError):
        build_metric_evaluator(eta_evaluator, initial[..., :2])
    with pytest.raises(ValueError):
        build_metric_evaluator(eta_evaluator, initial, logical_axes=(axes[0], axes[1], np.linspace(0.0, eta_evaluator.period, 8)))

    class MissingPeriod:
        pass

    with pytest.raises(ValueError):
        build_metric_evaluator(MissingPeriod(), initial)


def test_wall_fitted_initial_mesh_has_harmonic_interior_and_rotational_seam():
    wall = SyntheticWallEvaluator()
    eta_evaluator = SyntheticEtaEvaluator()
    shape = (7, 6, 8)
    positions = build_wall_fitted_initial_mesh(eta_evaluator, wall, shape)
    assert positions.shape == shape + (3,)
    assert np.all(np.hypot(positions[..., 0], positions[..., 1]) > 0)
    # Every interior node is the 5-point average of its four neighbors.
    interior = positions[1:-1, 1:-1]
    average = 0.25 * (
        positions[:-2, 1:-1]
        + positions[2:, 1:-1]
        + positions[1:-1, :-2]
        + positions[1:-1, 2:]
    )
    np.testing.assert_allclose(interior, average, atol=2.0e-12)
    for k, eta in enumerate(np.arange(shape[2]) * wall.period / shape[2]):
        angle = float(eta)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(positions[..., k, :], positions[..., 0, :] @ rotation.T, atol=2.0e-12)


def test_wall_fitted_initial_mesh_factorizes_harmonic_operator_once(monkeypatch):
    metric_module = importlib.import_module("drbx.geometry.MetricEvaluator")
    original_splu = metric_module.splu
    calls = []

    def recording_splu(*args, **kwargs):
        calls.append(args[0].shape)
        return original_splu(*args, **kwargs)

    monkeypatch.setattr(metric_module, "splu", recording_splu)
    build_wall_fitted_initial_mesh(
        SyntheticEtaEvaluator(), SyntheticWallEvaluator(), (9, 8, 12)
    )
    assert calls == [((9 - 2) * (8 - 2),) * 2]


def test_wall_fitted_mesh_preserves_poloidal_phase_when_shape_extrema_move():
    wall = PhaseStableShapeWallEvaluator()
    eta_evaluator = SyntheticEtaEvaluator()
    shape = (8, 7, 8)
    positions = build_wall_fitted_initial_mesh(eta_evaluator, wall, shape)
    perimeter = 2 * (shape[0] + shape[1]) - 4
    for k, eta in enumerate(np.arange(shape[2]) * wall.period / shape[2]):
        curve = wall.constant_eta_boundary_curve(eta_evaluator, float(eta), npoints=perimeter + 1)
        radius = np.hypot(curve[:-1, 0], curve[:-1, 1])
        area = 0.5 * np.sum(radius * np.roll(curve[:-1, 2], -1) - np.roll(radius, -1) * curve[:-1, 2])
        expected = curve[:-1] if area < 0.0 else np.concatenate((curve[:1], curve[-2:0:-1]), axis=0)
        boundary = np.concatenate((
            positions[:, 0, k],
            positions[-1, 1:, k],
            positions[-2::-1, -1, k],
            positions[0, -2:0:-1, k],
        ))
        # The anchor and every subsequent boundary node retain the same
        # WallEvaluator poloidal identity; only orientation may be reversed.
        np.testing.assert_allclose(boundary, expected, atol=2.0e-12)
    # Check finite-difference cell orientation, including the rotational seam.
    minimum = np.inf
    for k in range(shape[2]):
        kp = (k + 1) % shape[2]
        next_plane = positions[..., kp, :]
        if kp == 0:
            angle = wall.period
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                                 [np.sin(angle), np.cos(angle), 0.0],
                                 [0.0, 0.0, 1.0]])
            next_plane = next_plane @ rotation.T
        du = positions[1:, :-1, k] - positions[:-1, :-1, k]
        dv = positions[:-1, 1:, k] - positions[:-1, :-1, k]
        de = next_plane[:-1, :-1] - positions[:-1, :-1, k]
        minimum = min(minimum, float(np.min(np.einsum("...i,...i->...", np.cross(du, dv), de))))
    assert minimum > 0.0


def test_wall_mode_builds_metric_evaluator_and_fixes_perimeter():
    wall = SyntheticWallEvaluator()
    eta_evaluator = SyntheticEtaEvaluator()
    evaluator = build_metric_evaluator(
        eta_evaluator,
        wall_evaluator=wall,
        mesh_shape=(7, 6, 8),
        options=importlib.import_module("drbx.geometry.MetricEvaluator").MMPDEOptions(max_iterations=0),
    )
    assert isinstance(evaluator, MetricEvaluator)
    assert evaluator.eta.size == 8
    q = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.5 * evaluator.period]])
    result = evaluator.evaluate(q)
    assert np.all(result.valid)
    assert np.all(result.signed_J > 0)


def test_wall_mode_argument_conflicts_are_rejected():
    wall = SyntheticWallEvaluator()
    initial = make_eta_constrained_mesh()
    with pytest.raises(ValueError, match="either initial_positions"):
        build_metric_evaluator(
            SyntheticEtaEvaluator(), initial, wall_evaluator=wall, mesh_shape=initial.shape[:3]
        )
    with pytest.raises(ValueError, match="provided together"):
        build_metric_evaluator(SyntheticEtaEvaluator(), wall_evaluator=wall)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_metric_spline_degree_is_configurable(degree):
    base = make_evaluator()
    q = np.stack(np.meshgrid(base.u, base.v, base.eta, indexing="ij"), axis=-1)
    evaluator = MetricEvaluator(
        base.u,
        base.v,
        base.eta,
        base.position(q),
        nfp=base.nfp,
        metric_spline_degree=degree,
    )
    assert evaluator.metric_spline_degree == degree
    assert np.all(evaluator.evaluate(np.array([[0.37, 0.43, 0.21]])).valid)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fitted_metric_evaluator_cache_roundtrip_skips_refit(degree, monkeypatch):
    base = make_evaluator()
    nodes = np.stack(
        np.meshgrid(base.u, base.v, base.eta, indexing="ij"),
        axis=-1,
    )
    evaluator = MetricEvaluator(
        base.u,
        base.v,
        base.eta,
        base.position(nodes),
        nfp=base.nfp,
        metric_spline_degree=degree,
    )
    payload = evaluator.to_cache_payload(prefix="metric_")
    metric_module = importlib.import_module("drbx.geometry.MetricEvaluator")
    monkeypatch.setattr(
        metric_module,
        "RectBivariateSpline",
        lambda *args, **kwargs: pytest.fail("cache restore must not refit splines"),
    )
    restored = MetricEvaluator.from_cache_payload(payload, prefix="metric_")
    rng = np.random.default_rng(8675309)
    query = np.column_stack(
        (
            rng.random(17),
            rng.random(17),
            rng.random(17) * evaluator.period,
        )
    )
    np.testing.assert_array_equal(restored.position(query), evaluator.position(query))
    np.testing.assert_array_equal(
        restored.jacobian_matrix(query),
        evaluator.jacobian_matrix(query),
    )
    assert restored.nfp == evaluator.nfp
    assert restored.metric_spline_degree == evaluator.metric_spline_degree


@pytest.mark.parametrize("bad_degree", [0, 4, 1.5, True, "2"])
def test_metric_spline_degree_rejects_invalid_values(bad_degree):
    base = make_evaluator()
    q = np.stack(np.meshgrid(base.u, base.v, base.eta, indexing="ij"), axis=-1)
    with pytest.raises(ValueError, match="metric_spline_degree"):
        MetricEvaluator(base.u, base.v, base.eta, base.position(q), nfp=base.nfp, metric_spline_degree=bad_degree)


def test_metric_spline_degree_reduces_to_available_axis_samples():
    u = np.linspace(0.0, 1.0, 2)
    v = np.linspace(0.0, 1.0, 4)
    eta = np.arange(8) * (2.0 * np.pi / 8)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    positions = np.stack(((3.0 + 0.2 * ug) * np.cos(eg), (3.0 + 0.2 * ug) * np.sin(eg), -vg), axis=-1)
    evaluator = MetricEvaluator(u, v, eta, positions, period=2.0 * np.pi, metric_spline_degree=2)
    assert evaluator.metric_spline_degree == 2
    assert np.all(evaluator.evaluate(np.array([[0.5, 0.5, 0.3]])).valid)


def test_wall_mode_degree_one_is_positive_on_cells_and_open_boundary_faces_after_iterations():
    wall = SyntheticWallEvaluator()
    eta_evaluator = SyntheticEtaEvaluator()
    shape = (8, 7, 8)
    initial = build_wall_fitted_initial_mesh(eta_evaluator, wall, shape)
    metric_module = importlib.import_module("drbx.geometry.MetricEvaluator")
    evaluator = build_metric_evaluator(
        eta_evaluator,
        wall_evaluator=wall,
        mesh_shape=shape,
        options=metric_module.MMPDEOptions(max_iterations=2, initial_step=0.03),
    )
    assert evaluator.metric_spline_degree == 1
    assert evaluator.mmpde_result is not None
    assert np.min(evaluator.mmpde_result.minimum_jacobian_history) > 0.0

    eta_centers = (evaluator.eta + np.roll(evaluator.eta, -1)) * 0.5
    eta_centers[-1] = 0.5 * (evaluator.eta[-1] + evaluator.period)
    u_centers = 0.5 * (evaluator.u[:-1] + evaluator.u[1:])
    v_centers = 0.5 * (evaluator.v[:-1] + evaluator.v[1:])
    cell_q = np.stack(np.meshgrid(u_centers, v_centers, eta_centers, indexing="ij"), axis=-1)
    assert np.min(evaluator.evaluate(cell_q).signed_J) > 0.0

    open_faces = []
    for u_face in (0.0, 1.0):
        open_faces.append(np.stack(np.meshgrid([u_face], v_centers, eta_centers, indexing="ij"), axis=-1))
    for v_face in (0.0, 1.0):
        open_faces.append(np.stack(np.meshgrid(u_centers, [v_face], eta_centers, indexing="ij"), axis=-1))
    face_q = np.concatenate([face.reshape(-1, 3) for face in open_faces], axis=0)
    assert np.min(evaluator.evaluate(face_q).signed_J) > 0.0

    fixed = np.zeros(shape, dtype=bool)
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    np.testing.assert_allclose(evaluator.mmpde_result.positions[fixed], initial[fixed], atol=2.0e-11)


def plot_constant_eta_mesh(
    metric_evaluator: MetricEvaluator,
    filename: str | Path,
    *,
    eta_evaluator=None,
    wall_evaluator=None,
    wall_points: int = 256,
    surface_count: int = 8,
    surface_nu: int = 40,
    surface_nv: int = 40,
    show_mesh_nodes: bool = True,
    show: bool = False,
    _figure=None,
    _scene_name: str = "scene",
    _show_legend: bool = True,
    _write_html: bool = True,
):
    """Save a self-contained Plotly view of fitted constant-eta mesh planes.

    The displayed planes are selected from the evaluator's stored eta nodes,
    so every wireframe node lies on the corresponding solved MMPDE plane.
    ``eta_evaluator`` is optional; when supplied, its periodic residual is
    included in mesh hover data as ``eta residual [rad]``. When both
    ``eta_evaluator`` and ``wall_evaluator`` are supplied, the corresponding
    physical vessel contour is overlaid on every displayed eta plane.
    """

    import plotly.graph_objects as go

    if not isinstance(metric_evaluator, MetricEvaluator):
        raise TypeError("metric_evaluator must be a MetricEvaluator")
    if surface_count < 1 or surface_count > metric_evaluator.eta.size:
        raise ValueError("surface_count must be between 1 and the stored eta count")
    if surface_nu < 2 or surface_nv < 2:
        raise ValueError("surface_nu and surface_nv must be at least 2")
    if wall_evaluator is not None and eta_evaluator is None:
        raise ValueError("wall_evaluator requires eta_evaluator")
    if wall_points < 4:
        raise ValueError("wall_points must be at least 4")
    filename = Path(filename)
    if filename.suffix.lower() != ".html":
        raise ValueError("interactive Plotly output filename must end in .html")

    # Eta is periodic, so the final stored plane is not a special endpoint.
    # Select planes around the periodic ring instead of including eta[-1]
    # preferentially as a nonperiodic endpoint.
    eta_indices = np.floor(
        np.arange(surface_count) * metric_evaluator.eta.size / surface_count
    ).astype(int)

    period = float(metric_evaluator.period)

    def eta_residual(points, target):
        if eta_evaluator is None:
            return np.zeros(points.shape[:-1], dtype=float)
        values = None
        try:
            values = eta_evaluator.evaluate_cartesian(points, wrapped=True)
        except TypeError:
            values = eta_evaluator.evaluate_cartesian(points)
        values = np.asarray(values, dtype=float)
        if values.shape != points.shape[:-1]:
            raise ValueError(
                "eta_evaluator must return values with shape points.shape[:-1]"
            )
        finite_points = np.all(np.isfinite(points), axis=-1)
        if not np.all(np.isfinite(values[finite_points])):
            raise ValueError(
                "eta_evaluator must return finite values for finite points"
            )
        residual = np.full(points.shape[:-1], np.nan, dtype=float)
        finite_values = values[finite_points]
        residual[finite_points] = (
            finite_values - target + 0.5 * period
        ) % period - 0.5 * period
        return residual

    uu = np.linspace(metric_evaluator.u[0], metric_evaluator.u[-1], surface_nu)
    vv = np.linspace(metric_evaluator.v[0], metric_evaluator.v[-1], surface_nv)
    U, V = np.meshgrid(uu, vv, indexing="ij")
    figure = go.Figure() if _figure is None else _figure

    for plane_number, eta_index in enumerate(eta_indices):
        target = float(metric_evaluator.eta[eta_index])
        surface_q = np.stack(
            (U, V, np.full_like(U, target)), axis=-1
        )
        surface_xyz = metric_evaluator.position(surface_q)
        surface_residual = eta_residual(surface_xyz, target)
        surface_customdata = np.stack(
            (np.full_like(surface_residual, target), surface_residual), axis=-1
        )
        figure.add_trace(
            go.Surface(
                x=surface_xyz[..., 0],
                y=surface_xyz[..., 1],
                z=surface_xyz[..., 2],
                customdata=surface_customdata,
                opacity=0.28,
                colorscale="Turbo",
                surfacecolor=np.full_like(U, plane_number, dtype=float),
                cmin=0.0,
                cmax=max(1.0, surface_count - 1),
                showscale=_show_legend and plane_number == 0,
                colorbar=dict(title=dict(text="stored eta plane index")),
                name=f"eta={target:.5f}",
                showlegend=_show_legend,
                hovertemplate=(
                    "eta=%{customdata[0]:.6f} rad"
                    "<br>x=%{x:.6f} m<br>y=%{y:.6f} m<br>z=%{z:.6f} m"
                    "<br>eta residual=%{customdata[1]:.3e} rad<extra></extra>"
                ),
                scene=_scene_name,
            )
        )

        # Use one NaN-separated trace for each family of mesh lines. This
        # keeps the Plotly object small while preventing false line segments.
        mesh_u = np.stack(
            [
                metric_evaluator.position(
                    np.stack(
                        (
                            uu,
                            np.full_like(uu, v),
                            np.full_like(uu, target),
                        ),
                        axis=-1,
                    )
                )
                for v in vv
            ],
            axis=0,
        )
        mesh_v = np.stack(
            [
                metric_evaluator.position(
                    np.stack(
                        (
                            np.full_like(vv, u),
                            vv,
                            np.full_like(vv, target),
                        ),
                        axis=-1,
                    )
                )
                for u in uu
            ],
            axis=0,
        )

        def separated_lines(lines):
            nan_row = np.full((1, lines.shape[-1]), np.nan)
            return np.concatenate(
                [np.concatenate((line, nan_row), axis=0) for line in lines], axis=0
            )

        # Evaluate eta only on finite line vertices. NaN separators are a
        # Plotly transport detail and must never reach the eta evaluator.
        u_lines_finite = mesh_u.reshape(-1, mesh_u.shape[-1])
        v_lines_finite = mesh_v.reshape(-1, mesh_v.shape[-1])
        u_residual_finite = eta_residual(u_lines_finite, target)
        v_residual_finite = eta_residual(v_lines_finite, target)
        u_lines = separated_lines(mesh_u)
        v_lines = separated_lines(mesh_v)
        u_residual = separated_lines(u_residual_finite.reshape(mesh_u.shape[:-1] + (1,)))[..., 0]
        v_residual = separated_lines(v_residual_finite.reshape(mesh_v.shape[:-1] + (1,)))[..., 0]
        figure.add_trace(
            go.Scatter3d(
                x=u_lines[:, 0], y=u_lines[:, 1], z=u_lines[:, 2],
                mode="lines", line=dict(color="black", width=2),
                customdata=u_residual[:, None], name="u mesh lines",
                legendgroup=f"mesh-{plane_number}",
                showlegend=_show_legend and plane_number == 0,
                scene=_scene_name,
                hovertemplate="eta residual=%{customdata[0]:.3e} rad<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=v_lines[:, 0], y=v_lines[:, 1], z=v_lines[:, 2],
                mode="lines", line=dict(color="black", width=2),
                customdata=v_residual[:, None], name="v mesh lines",
                legendgroup=f"mesh-{plane_number}",
                showlegend=_show_legend and plane_number == 0,
                scene=_scene_name,
                hovertemplate="eta residual=%{customdata[0]:.3e} rad<extra></extra>",
            )
        )
        if show_mesh_nodes:
            nodes = metric_evaluator.position(
                np.stack(
                    np.meshgrid(uu, vv, [target], indexing="ij"),
                    axis=-1,
                )
            )[..., 0, :]
            node_residual = eta_residual(nodes, target)
            figure.add_trace(
                go.Scatter3d(
                    x=nodes[..., 0].ravel(), y=nodes[..., 1].ravel(), z=nodes[..., 2].ravel(),
                    mode="markers", marker=dict(size=2.5, color="black"),
                    customdata=node_residual.ravel()[:, None], name="solved mesh nodes",
                    legendgroup=f"mesh-{plane_number}",
                    showlegend=_show_legend and plane_number == 0,
                    scene=_scene_name,
                    hovertemplate="eta residual=%{customdata[0]:.3e} rad<extra></extra>",
                )
            )
        if wall_evaluator is not None:
            wall_xyz = np.asarray(
                wall_evaluator.constant_eta_boundary_curve(
                    eta_evaluator, target, npoints=wall_points
                ),
                dtype=float,
            )
            if wall_xyz.shape != (wall_points, 3):
                raise ValueError(
                    "wall_evaluator must return a closed (wall_points, 3) curve"
                )
            wall_residual = eta_residual(wall_xyz, target)
            figure.add_trace(
                go.Scatter3d(
                    x=wall_xyz[:, 0],
                    y=wall_xyz[:, 1],
                    z=wall_xyz[:, 2],
                    mode="lines",
                    line=dict(color="crimson", width=6),
                    customdata=wall_residual[:, None],
                    name="HSX vessel boundary",
                    legendgroup="wall",
                    showlegend=_show_legend and plane_number == 0,
                    scene=_scene_name,
                    hovertemplate=(
                        "vessel boundary"
                        "<br>eta residual=%{customdata[0]:.3e} rad<extra></extra>"
                    ),
                )
            )

    if _write_html:
        figure.update_layout(
            title=f"{surface_count} constant eta planes with fitted D2 mesh",
            scene=dict(
                xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                aspectmode="data", camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
            ),
            legend=dict(title="Constant eta surfaces / mesh"),
            margin=dict(l=0, r=0, b=0, t=55),
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(filename), include_plotlyjs=True, full_html=True, auto_open=False)
        if show:
            figure.show()
    return figure


def plot_constant_eta_mesh_comparison(
    pre_relaxation_evaluator: MetricEvaluator,
    final_evaluator: MetricEvaluator,
    filename: str | Path,
    *,
    eta_evaluator=None,
    wall_evaluator=None,
    wall_points: int = 256,
    surface_count: int = 8,
    surface_nu: int = 40,
    surface_nv: int = 40,
    show: bool = False,
):
    """Write side-by-side pre/post-relaxation constant-eta mesh views."""

    import plotly.graph_objects as go

    if not isinstance(pre_relaxation_evaluator, MetricEvaluator):
        raise TypeError("pre_relaxation_evaluator must be a MetricEvaluator")
    if not isinstance(final_evaluator, MetricEvaluator):
        raise TypeError("final_evaluator must be a MetricEvaluator")
    if not np.isclose(pre_relaxation_evaluator.period, final_evaluator.period):
        raise ValueError("comparison evaluators must have the same period")
    figure = go.Figure()
    plot_constant_eta_mesh(
        pre_relaxation_evaluator,
        filename,
        eta_evaluator=eta_evaluator,
        wall_evaluator=wall_evaluator,
        wall_points=wall_points,
        surface_count=surface_count,
        surface_nu=surface_nu,
        surface_nv=surface_nv,
        _figure=figure,
        _scene_name="scene",
        _write_html=False,
    )
    plot_constant_eta_mesh(
        final_evaluator,
        filename,
        eta_evaluator=eta_evaluator,
        wall_evaluator=wall_evaluator,
        wall_points=wall_points,
        surface_count=surface_count,
        surface_nu=surface_nu,
        surface_nv=surface_nv,
        _figure=figure,
        _scene_name="scene2",
        _show_legend=False,
        _write_html=False,
    )
    coordinate_values = [[], [], []]
    for trace in figure.data:
        for index, coordinate in enumerate((trace.x, trace.y, trace.z)):
            values = np.asarray(coordinate, dtype=float).ravel()
            finite = values[np.isfinite(values)]
            if finite.size:
                coordinate_values[index].append(finite)

    def padded_range(values):
        combined = np.concatenate(values)
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        span = max(upper - lower, np.finfo(float).eps)
        padding = 0.02 * span
        return [lower - padding, upper + padding]

    x_range, y_range, z_range = [padded_range(values) for values in coordinate_values]
    scene_layout = dict(
        domain=dict(x=[0.0, 0.47]),
        xaxis=dict(title="x [m]", range=x_range),
        yaxis=dict(title="y [m]", range=y_range),
        zaxis=dict(title="z [m]", range=z_range),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
    )
    scene2_layout = dict(
        domain=dict(x=[0.53, 1.0]),
        xaxis=dict(title="x [m]", range=list(x_range)),
        yaxis=dict(title="y [m]", range=list(y_range)),
        zaxis=dict(title="z [m]", range=list(z_range)),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
    )
    figure.update_layout(
        title=f"{surface_count} constant eta planes: pre- and post-MMPDE",
        scene=scene_layout,
        scene2=scene2_layout,
        annotations=[
            dict(text="Pre-relaxation mesh", x=0.225, y=1.02, xref="paper", yref="paper", showarrow=False),
            dict(text="MMPDE-relaxed metric", x=0.775, y=1.02, xref="paper", yref="paper", showarrow=False),
        ],
        legend=dict(title="Constant eta surfaces / mesh"),
        margin=dict(l=0, r=0, b=0, t=75),
        width=1400,
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(filename), include_plotlyjs=True, full_html=True, auto_open=False)
    if show:
        figure.show()
    return figure


def compute_epsilon_plane_diagnostic(metric_evaluator, bfield_evaluator):
    """Measure alignment of B with the actual fitted constant-eta planes.

    The plane normal is computed from the fitted map, ``grad(eta) = A^{-T}e3``;
    this intentionally does not use the scalar-potential gradient.
    """
    if not isinstance(metric_evaluator, MetricEvaluator):
        raise TypeError("metric_evaluator must be a MetricEvaluator")
    u_widths = np.diff(metric_evaluator.u)
    v_widths = np.diff(metric_evaluator.v)
    eta_widths = np.full(
        metric_evaluator.eta.size,
        metric_evaluator.period / metric_evaluator.eta.size,
        dtype=float,
    )
    u_centers = metric_evaluator.u[:-1] + 0.5 * u_widths
    v_centers = metric_evaluator.v[:-1] + 0.5 * v_widths
    eta_centers = metric_evaluator.eta + 0.5 * eta_widths
    logical = np.stack(
        np.meshgrid(u_centers, v_centers, eta_centers, indexing="ij"), axis=-1
    )
    metric = metric_evaluator.evaluate(logical, reject_nonpositive_J=False)
    A = np.asarray(metric.jacobian_matrix, dtype=float)
    position = np.asarray(metric.position, dtype=float)
    B = np.asarray(bfield_evaluator.evaluate_cartesian(position), dtype=float)
    if B.shape != position.shape:
        raise ValueError("bfield_evaluator must return B with position.shape")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("epsilon_plane inputs contain nonfinite values")
    e3 = np.zeros(A.shape[:-1], dtype=float)
    e3[..., 2] = 1.0
    grad_eta = np.linalg.solve(
        np.swapaxes(A, -1, -2), e3[..., None]
    )[..., 0]
    B_norm = np.linalg.norm(B, axis=-1)
    normal_norm = np.linalg.norm(grad_eta, axis=-1)
    if (
        not np.all(np.isfinite(A))
        or not np.all(np.isfinite(grad_eta))
        or np.any(B_norm <= 0.0)
        or np.any(normal_norm <= 0.0)
    ):
        raise ValueError("epsilon_plane requires finite A, B, and nonzero normals")
    epsilon = np.linalg.norm(np.cross(B, grad_eta), axis=-1) / (B_norm * normal_norm)
    epsilon = np.clip(epsilon, 0.0, 1.0)
    angles = np.degrees(np.arcsin(epsilon))
    area = np.linalg.norm(np.cross(A[..., :, 0], A[..., :, 1]), axis=-1)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("epsilon_plane requires positive finite plane areas")
    cell_weight = (
        u_widths[:, None, None]
        * v_widths[None, :, None]
        * eta_widths[None, None, :]
    )
    area_weight = area * u_widths[:, None, None] * v_widths[None, :, None]
    signed_J = np.asarray(metric.signed_J, dtype=float)
    volume_weight = signed_J * cell_weight
    if np.any(~np.isfinite(volume_weight)) or np.any(volume_weight <= 0.0):
        raise ValueError("epsilon_plane requires positive finite signed cell volumes")

    def weighted_rms(values, weights):
        return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))

    plane_rms = np.sqrt(
        np.sum(area_weight * epsilon**2, axis=(0, 1))
        / np.sum(area_weight, axis=(0, 1))
    )
    return {
        "logical_points": logical,
        "eta_centers": eta_centers,
        "epsilon": epsilon,
        "angle_degrees": angles,
        "area": area,
        "area_weight": area_weight,
        "volume_weight": volume_weight,
        "plane_rms": np.asarray(plane_rms, dtype=float),
        "domain_rms": weighted_rms(epsilon, volume_weight),
        "pointwise_rms": float(np.sqrt(np.mean(epsilon**2))),
        "pointwise_max": float(np.max(epsilon)),
        "angle_rms_degrees": weighted_rms(angles, volume_weight),
        "angle_max_degrees": float(np.max(angles)),
    }


def _print_epsilon_plane_diagnostic(label, diagnostic):
    plane_values = ", ".join(
        f"eta={eta:.6e}:{rms:.6e}"
        for eta, rms in zip(diagnostic["eta_centers"], diagnostic["plane_rms"])
    )
    print(
        f"epsilon_plane ({label}): "
        f"plane_rms=[{plane_values}], "
        f"domain_rms={diagnostic['domain_rms']:.6e}, "
        f"pointwise_rms={diagnostic['pointwise_rms']:.6e}, "
        f"max={diagnostic['pointwise_max']:.6e}, "
        f"angle_rms={diagnostic['angle_rms_degrees']:.6e} deg, "
        f"angle_max={diagnostic['angle_max_degrees']:.6e} deg"
    )


def _epsilon_plot_filename(output):
    output = Path(output)
    if output.suffix.lower() != ".html":
        raise ValueError("epsilon_plane output filename must end in .html")
    return output.with_name(f"{output.stem}_epsilon_plane{output.suffix}")


def plot_epsilon_plane(
    metric_evaluator,
    bfield_evaluator,
    filename,
    *,
    surface_count=8,
    surface_nu=24,
    surface_nv=24,
    show=False,
):
    """Write epsilon_plane surfaces for selected stored eta planes."""
    import plotly.graph_objects as go

    if surface_count < 1 or surface_count > metric_evaluator.eta.size:
        raise ValueError("surface_count must be between 1 and the stored eta count")
    if surface_nu < 2 or surface_nv < 2:
        raise ValueError("surface_nu and surface_nv must be at least 2")
    filename = Path(filename)
    if filename.suffix.lower() != ".html":
        raise ValueError("interactive Plotly output filename must end in .html")
    eta_indices = np.floor(
        np.arange(surface_count) * metric_evaluator.eta.size / surface_count
    ).astype(int)
    # Move only the chart endpoints inward by one representable float. This
    # avoids square corners while retaining essentially the full wall extent.
    uu = np.linspace(metric_evaluator.u[0], metric_evaluator.u[-1], surface_nu)
    vv = np.linspace(metric_evaluator.v[0], metric_evaluator.v[-1], surface_nv)
    uu[[0, -1]] = [
        np.nextafter(metric_evaluator.u[0], metric_evaluator.u[-1]),
        np.nextafter(metric_evaluator.u[-1], metric_evaluator.u[0]),
    ]
    vv[[0, -1]] = [
        np.nextafter(metric_evaluator.v[0], metric_evaluator.v[-1]),
        np.nextafter(metric_evaluator.v[-1], metric_evaluator.v[0]),
    ]
    U, V = np.meshgrid(uu, vv, indexing="ij")
    surfaces = []
    for eta_index in eta_indices:
        target = float(metric_evaluator.eta[eta_index])
        logical = np.stack((U, V, np.full_like(U, target)), axis=-1)
        metric = metric_evaluator.evaluate(logical, reject_nonpositive_J=False)
        A = metric.jacobian_matrix
        xyz = metric.position
        B = np.asarray(bfield_evaluator.evaluate_cartesian(xyz), dtype=float)
        if B.shape != xyz.shape:
            raise ValueError("bfield_evaluator must return B with position.shape")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
            raise ValueError("epsilon_plane plot encountered nonfinite A or B")
        e3 = np.zeros(A.shape[:-1], dtype=float)
        e3[..., 2] = 1.0
        normal = np.linalg.solve(
            np.swapaxes(A, -1, -2), e3[..., None]
        )[..., 0]
        B_norm = np.linalg.norm(B, axis=-1)
        normal_norm = np.linalg.norm(normal, axis=-1)
        if (
            B.shape != xyz.shape
            or not np.all(np.isfinite(A))
            or not np.all(np.isfinite(B))
            or not np.all(np.isfinite(normal))
            or np.any(B_norm <= 0.0)
            or np.any(normal_norm <= 0.0)
        ):
            raise ValueError("epsilon_plane plot encountered invalid A, B, or normal")
        epsilon = np.linalg.norm(np.cross(B, normal), axis=-1) / (B_norm * normal_norm)
        angle = np.degrees(np.arcsin(np.clip(epsilon, -1.0, 1.0)))
        surfaces.append((target, xyz, epsilon, angle))
    plotted_values = np.concatenate([epsilon.ravel() for _, _, epsilon, _ in surfaces])
    cmax = max(float(np.percentile(plotted_values, 95.0)), np.finfo(float).eps)
    figure = go.Figure()
    for plane_number, (target, xyz, epsilon, angle) in enumerate(surfaces):
        customdata = np.stack((epsilon, angle), axis=-1)
        figure.add_trace(
            go.Surface(
                x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2],
                surfacecolor=epsilon, cmin=0.0, cmax=cmax,
                colorscale="Viridis", showscale=plane_number == 0,
                colorbar=dict(title="epsilon_plane") if plane_number == 0 else None,
                customdata=customdata, name=f"eta={target:.5f}",
                hovertemplate=(
                    "eta=%{text:.6f} rad<br>epsilon=%{customdata[0]:.6e}"
                    "<br>angle=%{customdata[1]:.6f} deg<extra></extra>"
                ),
                text=np.full(epsilon.shape, target),
            )
        )
    figure.update_layout(
        title=f"epsilon_plane on {surface_count} constant eta planes",
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(filename), include_plotlyjs=True, full_html=True, auto_open=False)
    if show:
        figure.show()
    return figure


def test_constant_eta_mesh_plot_smoke_and_alignment(tmp_path):
    u = np.linspace(0.0, 1.0, 7)
    v = np.linspace(0.0, 1.0, 6)
    eta = np.arange(12) * (2.0 * np.pi / 12)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    R = 3.0 + 0.5 * vg + 0.03 * np.sin(eg)
    Z = 0.5 * ug - 0.25 + 0.02 * np.cos(eg)
    positions = np.stack((R * np.cos(eg), R * np.sin(eg), Z), axis=-1)
    evaluator = MetricEvaluator(u, v, eta, positions, period=2.0 * np.pi)
    output = tmp_path / "constant_eta_mesh.html"
    figure = plot_constant_eta_mesh(
        evaluator,
        output,
        eta_evaluator=FiniteOnlyEtaEvaluator(),
        surface_count=4,
        surface_nu=9,
        surface_nv=8,
    )
    assert output.is_file()
    contents = output.read_text()
    assert "Plotly.newPlot" in contents
    assert "constant eta planes with fitted D2 mesh" in contents
    assert len(figure.data) == 4 * 4
    surfaces = [trace for trace in figure.data if trace.type == "surface"]
    assert len(surfaces) == 4
    selected = np.floor(np.arange(4) * evaluator.eta.size / 4).astype(int)
    np.testing.assert_allclose(
        [float(np.asarray(trace.customdata)[0, 0, 0]) for trace in surfaces],
        evaluator.eta[selected],
    )
    u_lines = [trace for trace in figure.data if trace.name == "u mesh lines"]
    v_lines = [trace for trace in figure.data if trace.name == "v mesh lines"]
    assert len(u_lines) == 4
    assert len(v_lines) == 4
    # The wireframe uses the plotting axes, not the full solved mesh axes.
    assert len(u_lines[0].x) == 8 * (9 + 1)
    assert len(v_lines[0].x) == 9 * (8 + 1)
    nodes = [trace for trace in figure.data if trace.name == "solved mesh nodes"]
    assert len(nodes) == 4
    assert len(nodes[0].x) == 9 * 8
    for trace in figure.data:
        if trace.type == "scatter3d":
            residual = np.asarray(trace.customdata, dtype=float)
            assert residual.shape[-1] == 1
            finite = np.isfinite(residual[:, 0])
            assert np.max(np.abs(residual[finite, 0])) < 1.0e-10


def test_constant_eta_mesh_plot_does_not_pass_nan_separators_to_eta_evaluator(tmp_path):
    u = np.linspace(0.0, 1.0, 4)
    v = np.linspace(0.0, 1.0, 3)
    eta = np.arange(6) * (2.0 * np.pi / 6)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    positions = np.stack(
        ((3.0 + 0.1 * vg) * np.cos(eg), (3.0 + 0.1 * vg) * np.sin(eg), ug),
        axis=-1,
    )
    evaluator = MetricEvaluator(
        u, v, eta, positions, period=2.0 * np.pi, metric_spline_degree=1
    )
    plot_constant_eta_mesh(
        evaluator,
        tmp_path / "finite_only_eta.html",
        eta_evaluator=FiniteOnlyEtaEvaluator(),
        surface_count=3,
        surface_nu=11,
        surface_nv=10,
    )


def test_constant_eta_mesh_plot_overlays_wall(tmp_path):
    evaluator = build_metric_evaluator(
        SyntheticEtaEvaluator(),
        wall_evaluator=SyntheticWallEvaluator(),
        mesh_shape=(5, 5, 6),
        options=importlib.import_module(
            "drbx.geometry.MetricEvaluator"
        ).MMPDEOptions(max_iterations=0),
    )
    figure = plot_constant_eta_mesh(
        evaluator,
        tmp_path / "wall_overlay.html",
        eta_evaluator=SyntheticEtaEvaluator(),
        wall_evaluator=SyntheticWallEvaluator(),
        wall_points=40,
        surface_count=3,
        surface_nu=8,
        surface_nv=8,
    )
    wall_traces = [
        trace for trace in figure.data if trace.name == "HSX vessel boundary"
    ]
    assert len(wall_traces) == 3
    assert all(len(trace.x) == 40 for trace in wall_traces)


def test_constant_eta_mesh_comparison_routes_pre_and_post_samples_to_two_scenes(tmp_path):
    u = np.linspace(0.0, 1.0, 6)
    v = np.linspace(0.0, 1.0, 5)
    eta = np.arange(8) * (2.0 * np.pi / 8)
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    base_positions = np.stack(
        ((3.0 + 0.5 * vg) * np.cos(eg), (3.0 + 0.5 * vg) * np.sin(eg), ug), axis=-1
    )
    relaxed_positions = base_positions.copy()
    relaxed_positions[2:-2, 2:-2, :, 2] += 0.08
    pre = MetricEvaluator(u, v, eta, base_positions, period=2.0 * np.pi)
    post = MetricEvaluator(u, v, eta, relaxed_positions, period=2.0 * np.pi)
    figure = plot_constant_eta_mesh_comparison(
        pre,
        post,
        tmp_path / "comparison.html",
        surface_count=4,
        surface_nu=7,
        surface_nv=6,
    )
    assert figure.layout.scene is not None
    assert figure.layout.scene2 is not None
    assert figure.layout.scene.domain.x[1] < figure.layout.scene2.domain.x[0]
    assert figure.layout.scene.xaxis.range == figure.layout.scene2.xaxis.range
    assert figure.layout.scene.yaxis.range == figure.layout.scene2.yaxis.range
    assert figure.layout.scene.zaxis.range == figure.layout.scene2.zaxis.range
    assert figure.layout.scene.zaxis.range != figure.layout.scene.xaxis.range
    assert {trace.scene for trace in figure.data} == {"scene", "scene2"}
    pre_surfaces = [trace for trace in figure.data if trace.type == "surface" and trace.scene == "scene"]
    post_surfaces = [trace for trace in figure.data if trace.type == "surface" and trace.scene == "scene2"]
    assert len(pre_surfaces) == len(post_surfaces) == 4
    assert any(not np.allclose(pre_trace.z, post_trace.z) for pre_trace, post_trace in zip(pre_surfaces, post_surfaces))
    pre_nodes = [trace for trace in figure.data if trace.name == "solved mesh nodes" and trace.scene == "scene"]
    post_nodes = [trace for trace in figure.data if trace.name == "solved mesh nodes" and trace.scene == "scene2"]
    assert len(pre_nodes) == len(post_nodes) == 4
    assert not np.allclose(pre_nodes[0].z, post_nodes[0].z)


def test_epsilon_plane_exactly_aligned_synthetic_field_is_near_zero():
    evaluator = make_cylindrical_evaluator()
    diagnostic = compute_epsilon_plane_diagnostic(
        evaluator, CylindricalToroidalField()
    )
    assert diagnostic["epsilon"].shape == (
        evaluator.u.size - 1, evaluator.v.size - 1, evaluator.eta.size
    )
    np.testing.assert_allclose(diagnostic["epsilon"], 0.0, atol=2.0e-12)
    assert diagnostic["domain_rms"] < 2.0e-12
    assert diagnostic["pointwise_max"] < 2.0e-12


def test_epsilon_plane_controlled_tilt_matches_known_value():
    evaluator = make_cylindrical_evaluator()
    tilt = 0.4
    diagnostic = compute_epsilon_plane_diagnostic(
        evaluator, CylindricalToroidalField(tilt)
    )
    expected = tilt / np.sqrt(1.0 + tilt**2)
    np.testing.assert_allclose(diagnostic["epsilon"], expected, atol=2.0e-11)
    np.testing.assert_allclose(diagnostic["plane_rms"], expected, atol=2.0e-11)
    np.testing.assert_allclose(diagnostic["domain_rms"], expected, atol=2.0e-11)
    np.testing.assert_allclose(diagnostic["pointwise_rms"], expected, atol=2.0e-11)
    np.testing.assert_allclose(
        diagnostic["angle_max_degrees"], np.degrees(np.arctan(tilt)), atol=2.0e-10
    )


def test_epsilon_plane_nonuniform_cell_sampling_weights_are_finite():
    evaluator = make_cylindrical_evaluator(
        u=[0.0, 0.1, 0.6, 1.0],
        v=[0.0, 0.25, 1.0],
        eta=np.arange(8) * (2.0 * np.pi / 8.0),
    )
    diagnostic = compute_epsilon_plane_diagnostic(
        evaluator, CylindricalToroidalField(0.2)
    )
    assert diagnostic["plane_rms"].shape == (evaluator.eta.size,)
    for key in ("domain_rms", "pointwise_rms", "pointwise_max", "angle_rms_degrees", "angle_max_degrees"):
        assert np.isfinite(diagnostic[key])
    assert np.all(np.isfinite(diagnostic["area_weight"]))
    assert np.all(np.isfinite(diagnostic["volume_weight"]))
    assert np.all(diagnostic["area_weight"] > 0.0)
    assert np.all(diagnostic["volume_weight"] > 0.0)


def test_epsilon_plane_rejects_negative_signed_jacobian():
    evaluator = make_cylindrical_evaluator()
    positions = evaluator.position(
        np.stack(np.meshgrid(evaluator.u, evaluator.v, evaluator.eta, indexing="ij"), axis=-1)
    )
    inverted = MetricEvaluator(
        evaluator.u,
        evaluator.v,
        evaluator.eta,
        positions[..., [0, 1, 2]] * np.array([1.0, 1.0, -1.0]),
        period=evaluator.period,
    )
    with pytest.raises(ValueError, match="signed cell volumes"):
        compute_epsilon_plane_diagnostic(inverted, CylindricalToroidalField())


def test_epsilon_plane_plot_smoke_and_derived_filename(tmp_path):
    evaluator = make_cylindrical_evaluator()
    output = tmp_path / "hsx_QHS.html"
    epsilon_output = _epsilon_plot_filename(output)
    assert epsilon_output.name == "hsx_QHS_epsilon_plane.html"
    figure = plot_epsilon_plane(
        evaluator,
        CylindricalToroidalField(0.25),
        epsilon_output,
        surface_count=3,
        surface_nu=7,
        surface_nv=6,
    )
    assert epsilon_output.is_file()
    assert "Plotly.newPlot" in epsilon_output.read_text()
    surfaces = [trace for trace in figure.data if trace.type == "surface"]
    assert len(surfaces) == 3
    assert all(np.asarray(trace.surfacecolor).shape == (7, 6) for trace in surfaces)
    assert all(trace.cmin == 0.0 for trace in surfaces)
    assert all(trace.cmax == surfaces[0].cmax for trace in surfaces)
    assert "epsilon" in surfaces[0].hovertemplate
    assert "angle" in surfaces[0].hovertemplate


def test_epsilon_plane_plot_uses_global_p95_without_clipping_surface_values(tmp_path):
    evaluator = make_cylindrical_evaluator()

    class OutlierField(CylindricalToroidalField):
        def evaluate_cartesian(self, points):
            points = np.asarray(points, dtype=float)
            radius = np.hypot(points[..., 0], points[..., 1])
            ephi = np.stack(
                (-points[..., 1] / radius, points[..., 0] / radius, np.zeros_like(radius)),
                axis=-1,
            )
            tilt = np.where(points[..., 2] > 0.68, 10.0, 0.1)
            return ephi + tilt[..., None] * np.array([0.0, 0.0, 1.0])

    output = tmp_path / "p95_epsilon.html"
    figure = plot_epsilon_plane(
        evaluator,
        OutlierField(),
        output,
        surface_count=3,
        surface_nu=7,
        surface_nv=6,
    )
    surfaces = [trace for trace in figure.data if trace.type == "surface"]
    values = np.concatenate([np.asarray(trace.surfacecolor).ravel() for trace in surfaces])
    expected_cmax = max(float(np.percentile(values, 95.0)), np.finfo(float).eps)
    assert all(trace.cmax == expected_cmax for trace in surfaces)
    assert np.any(values > expected_cmax)
    hover_values = np.concatenate(
        [np.asarray(trace.customdata)[..., 0].ravel() for trace in surfaces]
    )
    np.testing.assert_allclose(np.sort(hover_values), np.sort(values))
    assert np.max(hover_values) > expected_cmax


def _padded_clipped_bounds(values, domain, fraction=0.02):
    """Pad raw wall extrema, clip to the B-field box, and stay interior."""

    values = np.asarray(values, dtype=float)
    domain = np.asarray(domain, dtype=float)
    padding = max(fraction * max(float(np.ptp(values)), np.finfo(float).eps), 1.0e-4)
    lower = max(float(domain[0]), float(values.min()) - padding)
    upper = min(float(domain[1]), float(values.max()) + padding)
    lower = max(lower, float(np.nextafter(domain[0], domain[1])))
    upper = min(upper, float(np.nextafter(domain[1], domain[0])))
    if not lower < upper:
        raise ValueError("wall extrema do not leave a nonempty B-field fit interval")
    return lower, upper


def _print_mmpde_objective_components(result):
    """Print raw initial/final objective components when histories exist."""

    histories = getattr(result, "component_energy_history", None) or {}
    names = (
        "dirichlet",
        "alignment",
        "equidistribution",
        "jacobian_barrier",
        "volume_smoothness",
        "total",
    )
    parts = []
    for name in names:
        values = histories.get(name)
        if values is None:
            continue
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size:
            parts.append(f"{name}={values[0]:.6e}->{values[-1]:.6e}")
    if parts:
        print("MMPDE objective components: " + ", ".join(parts))


def test_mmpde_objective_component_print_is_concise_and_backward_compatible(capsys):
    result = type("Result", (), {
        "component_energy_history": {
            "dirichlet": np.array([1.0, 0.8]),
            "alignment": np.array([2.0, 1.5]),
            "total": np.array([3.0, 2.3]),
        }
    })()
    _print_mmpde_objective_components(result)
    output = capsys.readouterr().out
    assert output == (
        "MMPDE objective components: dirichlet=1.000000e+00->8.000000e-01, "
        "alignment=2.000000e+00->1.500000e+00, total=3.000000e+00->2.300000e+00\n"
    )
    _print_mmpde_objective_components(type("LegacyResult", (), {})())
    _print_mmpde_objective_components(type("EmptyResult", (), {"component_energy_history": {}})())
    assert capsys.readouterr().out == ""


def build_hsx_metric_plot(
    makegrid_path,
    vessel_path,
    *,
    currents=None,
    radial_degree=3,
    vertical_degree=3,
    toroidal_modes=2,
    sample_shape=(8, 9, 8),
    R_bounds=None,
    Z_bounds=None,
    mesh_shape=(8, 16, 8),
    mmpde_iterations=0,
    metric_spline_degree=1,
    axis_core_radius=0.03,
    plot_surfaces=8,
    plot_nu=24,
    plot_nv=24,
    plot_wall_points=256,
    output="hsx_metric_mesh.html",
    show=False,
):
    """Build a wall-fitted HSX metric evaluator and Plotly view."""

    bfield = bfield_evaluator_from_makegrid(
        makegrid_path, currents=currents, method="cubic"
    )
    wall = WallEvaluator.from_file(vessel_path)
    if wall.nfp != bfield.nfp:
        raise ValueError("MAKEGRID and vessel field-period counts disagree")
    raw_rz = np.asarray(wall.raw["RZ"], dtype=float)
    fit_R_bounds = R_bounds or _padded_clipped_bounds(
        raw_rz[..., 0], (bfield.R[0], bfield.R[-1])
    )
    fit_Z_bounds = Z_bounds or _padded_clipped_bounds(
        raw_rz[..., 1], (bfield.Z[0], bfield.Z[-1])
    )

    def fit_mask(points):
        points = np.asarray(points, dtype=float)
        axis_R, axis_Z, _, _ = wall.reference_axis(points[:, 1])
        outside_core = (
            np.hypot(points[:, 0] - axis_R, points[:, 2] - axis_Z)
            > axis_core_radius
        )
        return wall.contains_cylindrical(points) & outside_core

    eta_evaluator = scalar_potential_evaluator_from_bfield(
        bfield,
        radial_degree=radial_degree,
        vertical_degree=vertical_degree,
        toroidal_modes=toroidal_modes,
        sample_shape=tuple(sample_shape),
        R_bounds=tuple(fit_R_bounds),
        Z_bounds=tuple(fit_Z_bounds),
        mask=fit_mask,
        reference_axis=wall.reference_axis,
    )
    metric_module = importlib.import_module("drbx.geometry.MetricEvaluator")
    initial_positions = build_wall_fitted_initial_mesh(
        eta_evaluator,
        wall,
        mesh_shape,
    )
    unrelaxed_evaluator = build_metric_evaluator(
        eta_evaluator,
        initial_positions,
        options=metric_module.MMPDEOptions(max_iterations=0),
        metric_spline_degree=metric_spline_degree,
    )
    projected_initial = unrelaxed_evaluator.mmpde_result.positions
    metric_evaluator = build_metric_evaluator(
        eta_evaluator,
        projected_initial,
        options=metric_module.MMPDEOptions(max_iterations=int(mmpde_iterations)),
        metric_spline_degree=metric_spline_degree,
    )
    unrelaxed_quality = unrelaxed_evaluator.quality_report()
    relaxed_quality = metric_evaluator.quality_report()
    print(unrelaxed_quality.summary("Metric quality (unrelaxed)"))
    print(unrelaxed_quality.detailed_summary("Metric quality details (unrelaxed)"))
    print(relaxed_quality.summary("Metric quality (relaxed)"))
    print(relaxed_quality.detailed_summary("Metric quality details (relaxed)"))
    solve = metric_evaluator.mmpde_result
    if solve is None:
        raise RuntimeError("build_metric_evaluator did not return an MMPDE result")
    _print_mmpde_objective_components(solve)
    fit_diagnostics = dict(eta_evaluator.diagnostics)
    u_centers = 0.5 * (
        metric_evaluator.u[:-1] + metric_evaluator.u[1:]
    )
    v_centers = 0.5 * (
        metric_evaluator.v[:-1] + metric_evaluator.v[1:]
    )
    eta_centers = metric_evaluator.eta + 0.5 * (
        metric_evaluator.period / metric_evaluator.eta.size
    )
    logical_grid = np.stack(
        np.meshgrid(
            u_centers,
            v_centers,
            eta_centers,
            indexing="ij",
        ),
        axis=-1,
    )
    metric = metric_evaluator.evaluate(
        logical_grid, reject_nonpositive_J=False
    )
    magnetic = metric_evaluator.evaluate_magnetic_field(
        logical_grid, bfield, reject_nonpositive_J=False
    )
    eta_values = np.asarray(
        eta_evaluator.evaluate_cartesian(metric.position, wrapped=True),
        dtype=float,
    )
    eta_targets = logical_grid[..., 2]
    eta_residual = (
        eta_values - eta_targets + 0.5 * metric_evaluator.period
    ) % metric_evaluator.period - 0.5 * metric_evaluator.period
    print(
        "Scalar-potential fit:"
        f" rms_abs={fit_diagnostics.get('rms_absolute_error', float('nan')):.6e} T,"
        f" max_abs={fit_diagnostics.get('max_absolute_error', float('nan')):.6e} T,"
        f" condition={fit_diagnostics.get('condition_number', float('nan')):.6e}"
    )
    print(
        "MMPDE solve:"
        f" converged={solve.converged}, iterations={solve.iterations},"
        f" final_update={solve.max_free_node_update:.6e},"
        f" accepted_fit_scale={metric_evaluator.mmpde_fit_scale:.6e}"
    )
    print(
        "Metric cell centers:"
        f" J=[{np.min(metric.signed_J):.6e}, {np.max(metric.signed_J):.6e}],"
        f" inverse_residual_max={np.max(metric.inverse_residual):.6e},"
        f" eta_residual_max={np.max(np.abs(eta_residual)):.6e},"
        f" |B|=[{np.min(magnetic.magnitude):.6e}, {np.max(magnetic.magnitude):.6e}] T"
    )
    unrelaxed_epsilon = compute_epsilon_plane_diagnostic(
        unrelaxed_evaluator, bfield
    )
    relaxed_epsilon = compute_epsilon_plane_diagnostic(metric_evaluator, bfield)
    _print_epsilon_plane_diagnostic("unrelaxed", unrelaxed_epsilon)
    _print_epsilon_plane_diagnostic("relaxed", relaxed_epsilon)
    plot_constant_eta_mesh_comparison(
        unrelaxed_evaluator,
        metric_evaluator,
        output,
        eta_evaluator=eta_evaluator,
        wall_evaluator=wall,
        surface_count=min(plot_surfaces, metric_evaluator.eta.size),
        surface_nu=plot_nu,
        surface_nv=plot_nv,
        wall_points=plot_wall_points,
        show=show,
    )
    print(f"Interactive mesh plot: {Path(output).resolve()}")
    epsilon_output = _epsilon_plot_filename(output)
    plot_epsilon_plane(
        metric_evaluator,
        bfield,
        epsilon_output,
        surface_count=min(plot_surfaces, metric_evaluator.eta.size),
        surface_nu=plot_nu,
        surface_nv=plot_nv,
        show=show,
    )
    print(f"Interactive epsilon_plane plot: {epsilon_output.resolve()}")
    return metric_evaluator


def test_real_hsx_metric_plot_if_input_files_exist(tmp_path):
    makegrid = Path(os.environ.get("DRBX_HSX_MGRID", "/Users/yxie/Desktop/HSX drbx/mgrid_res2p5cm_180pln.nc"))
    vessel = Path(os.environ.get("DRBX_HSX_VESSEL", "/Users/yxie/Desktop/HSX drbx/vessel_hsx_flare.txt"))
    if not makegrid.is_file() or not vessel.is_file():
        pytest.skip("real HSX MAKEGRID/vessel files are not available")
    evaluator = build_hsx_metric_plot(makegrid, vessel, output=tmp_path / "hsx_metric_mesh.html")
    result = evaluator.evaluate(np.array([[0.5, 0.5, 0.0]]))
    assert bool(result.valid[0])
    assert result.signed_J[0] > 0.0


def _hsx_cli(argv=None):
    parser = argparse.ArgumentParser(description="Build an interactive HSX metric evaluator plot")
    parser.add_argument("mgrid", type=Path)
    parser.add_argument("vessel", type=Path)
    parser.add_argument(
        "--currents",
        nargs="+",
        type=float,
        help=(
            "coil-group currents for scaled mgrid data, or dimensionless "
            "multipliers for raw mgrid data"
        ),
    )
    parser.add_argument("--radial-degree", type=int, default=3)
    parser.add_argument("--vertical-degree", type=int, default=3)
    parser.add_argument("--toroidal-modes", type=int, default=2)
    parser.add_argument("--sample-shape", nargs=3, type=int, default=(8, 9, 8), metavar=("NR", "NPHI", "NZ"))
    parser.add_argument("--R-bounds", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--Z-bounds", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--mesh-shape", nargs=3, type=int, default=(8, 16, 8), metavar=("NU", "NV", "NETA"))
    parser.add_argument("--mmpde-iterations", type=int, default=0)
    parser.add_argument(
        "--metric-spline-degree", type=int, choices=(1, 2, 3), default=1
    )
    parser.add_argument("--axis-core-radius", type=float, default=0.03)
    parser.add_argument("--plot-surfaces", type=int, default=8)
    parser.add_argument("--plot-nu", type=int, default=24)
    parser.add_argument("--plot-nv", type=int, default=24)
    parser.add_argument("--plot-wall-points", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("hsx_metric_mesh.html"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    build_hsx_metric_plot(
        args.mgrid,
        args.vessel,
        currents=args.currents,
        radial_degree=args.radial_degree,
        vertical_degree=args.vertical_degree,
        toroidal_modes=args.toroidal_modes,
        sample_shape=tuple(args.sample_shape),
        R_bounds=None if args.R_bounds is None else tuple(args.R_bounds),
        Z_bounds=None if args.Z_bounds is None else tuple(args.Z_bounds),
        mesh_shape=tuple(args.mesh_shape),
        mmpde_iterations=args.mmpde_iterations,
        metric_spline_degree=args.metric_spline_degree,
        axis_core_radius=args.axis_core_radius,
        plot_surfaces=args.plot_surfaces,
        plot_nu=args.plot_nu,
        plot_nv=args.plot_nv,
        plot_wall_points=args.plot_wall_points,
        output=args.output,
        show=args.show,
    )
    return 0


def test_hsx_cli_exposes_plot_sampling_controls(monkeypatch):
    calls = []

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(globals(), "build_hsx_metric_plot", record_call)
    assert _hsx_cli(
        [
            "mgrid.nc",
            "vessel.txt",
            "--currents",
            "2.0",
            "0.0",
            "--plot-surfaces",
            "5",
            "--plot-nu",
            "31",
            "--plot-nv",
            "29",
            "--plot-wall-points",
            "180",
        ]
    ) == 0
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["currents"] == [2.0, 0.0]
    assert kwargs["plot_surfaces"] == 5
    assert kwargs["plot_nu"] == 31
    assert kwargs["plot_nv"] == 29
    assert kwargs["plot_wall_points"] == 180


if __name__ == "__main__":
    raise SystemExit(_hsx_cli())
