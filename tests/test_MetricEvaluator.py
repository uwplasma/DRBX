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
    figure = go.Figure()

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
                showscale=plane_number == 0,
                colorbar=dict(title=dict(text="stored eta plane index")),
                name=f"eta={target:.5f}",
                showlegend=True,
                hovertemplate=(
                    "eta=%{customdata[0]:.6f} rad"
                    "<br>x=%{x:.6f} m<br>y=%{y:.6f} m<br>z=%{z:.6f} m"
                    "<br>eta residual=%{customdata[1]:.3e} rad<extra></extra>"
                ),
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
                legendgroup=f"mesh-{plane_number}", showlegend=plane_number == 0,
                hovertemplate="eta residual=%{customdata[0]:.3e} rad<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=v_lines[:, 0], y=v_lines[:, 1], z=v_lines[:, 2],
                mode="lines", line=dict(color="black", width=2),
                customdata=v_residual[:, None], name="v mesh lines",
                legendgroup=f"mesh-{plane_number}", showlegend=plane_number == 0,
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
                    legendgroup=f"mesh-{plane_number}", showlegend=plane_number == 0,
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
                    showlegend=plane_number == 0,
                    hovertemplate=(
                        "vessel boundary"
                        "<br>eta residual=%{customdata[0]:.3e} rad<extra></extra>"
                    ),
                )
            )

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


def build_hsx_metric_plot(
    makegrid_path,
    vessel_path,
    *,
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

    bfield = bfield_evaluator_from_makegrid(makegrid_path, method="cubic")
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
    metric_evaluator = build_metric_evaluator(
        eta_evaluator,
        wall_evaluator=wall,
        mesh_shape=mesh_shape,
        options=metric_module.MMPDEOptions(max_iterations=int(mmpde_iterations)),
        metric_spline_degree=metric_spline_degree,
    )
    fit_diagnostics = dict(eta_evaluator.diagnostics)
    solve = metric_evaluator.mmpde_result
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
    plot_constant_eta_mesh(
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
    assert kwargs["plot_surfaces"] == 5
    assert kwargs["plot_nu"] == 31
    assert kwargs["plot_nv"] == 29
    assert kwargs["plot_wall_points"] == 180


if __name__ == "__main__":
    raise SystemExit(_hsx_cli())
