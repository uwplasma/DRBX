"""Compact analytic checks for the axis-regular toroidal mesh builder."""

import numpy as np
import pytest

from drbx.geometry.MetricEvaluator import (
    align_wall_curves,
    axis_regular_initializer,
    build_metric_evaluator,
    evaluate_toroidal_quality,
    project_interior_eta,
    resample_periodic_curve,
    rotate_z,
)


class _EtaEvaluator:
    period = 2.0 * np.pi
    nfp = 1

    def evaluate_and_gradient_cartesian(self, xyz, *, wrapped=False):
        xyz = np.asarray(xyz, dtype=float)
        x, y = xyz[..., 0], xyz[..., 1]
        value = np.arctan2(y, x)
        denominator = x * x + y * y
        gradient = np.stack((-y / denominator, x / denominator, np.zeros_like(x)), axis=-1)
        return value, gradient


class _NormalizedEtaEvaluator(_EtaEvaluator):
    def evaluate_and_gradient_cartesian(self, xyz, *, wrapped=False):
        value, gradient = super().evaluate_and_gradient_cartesian(xyz, wrapped=wrapped)
        return value / (2.0 * np.pi), gradient / (2.0 * np.pi)


class _CircularWall:
    phi0 = 0.0

    def centerline(self, phi):
        phi = np.asarray(phi, dtype=float)
        return np.full_like(phi, 3.0), np.zeros_like(phi)

    def cartesian(self, phi, theta):
        phi, theta = np.broadcast_arrays(np.asarray(phi), np.asarray(theta))
        radius = 3.0 + 0.7 * np.cos(theta)
        return np.stack((radius * np.cos(phi), radius * np.sin(phi), -0.7 * np.sin(theta)), axis=-1)

    def constant_eta_boundary_curve(self, eta_evaluator, target_eta, *, npoints):
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        return self.cartesian(float(target_eta), theta)

    def contains_cartesian(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        radius = np.hypot(xyz[..., 0], xyz[..., 1])
        return ((radius - 3.0) ** 2 + xyz[..., 2] ** 2) <= 0.7 ** 2 + 1.0e-12


class _OffsetCenterlineWall(_CircularWall):
    def centerline(self, phi):
        phi = np.asarray(phi, dtype=float)
        return np.full_like(phi, 4.25), np.full_like(phi, 0.15)


def _torus_wall(theta, eta, *, major=3.0, minor=0.7):
    theta, eta = np.meshgrid(theta, eta, indexing="ij")
    radius = major + minor * np.cos(theta)
    return np.stack((radius * np.cos(eta), radius * np.sin(eta), -minor * np.sin(theta)), axis=-1)


def test_resample_periodic_curve_is_uniform_and_endpoint_exclusive():
    curve = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sampled = resample_periodic_curve(curve, 16)

    assert sampled.shape == (16, 3)
    assert not np.allclose(sampled[0], sampled[-1])
    edge_lengths = np.linalg.norm(np.roll(sampled, -1, axis=0) - sampled, axis=1)
    assert np.allclose(edge_lengths, edge_lengths[0], atol=1e-12)


def test_align_wall_curves_handles_cyclic_phase_and_field_period_rotation():
    ntheta, neta = 24, 6
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    base = np.stack((3.0 + 0.4 * np.cos(theta), np.zeros(ntheta), -0.4 * np.sin(theta)), axis=-1)
    curves = np.stack([rotate_z(base, k * np.pi / neta) for k in range(neta)], axis=0)
    curves = np.stack([np.roll(curves[k], (3 * k) % ntheta, axis=0) for k in range(neta)])

    aligned = align_wall_curves(curves, field_period=np.pi)
    assert aligned.shape == curves.shape
    assert np.allclose(aligned[0], base, atol=1e-12)
    assert np.allclose(rotate_z(aligned[0], np.pi), rotate_z(base, np.pi), atol=1e-12)
    assert np.allclose(np.sort(np.linalg.norm(aligned[-1], axis=1)), np.sort(np.linalg.norm(curves[-1], axis=1)))


def test_axis_regular_initializer_preserves_axis_wall_and_radial_mode_scaling():
    ntheta, neta = 32, 9
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    eta = 2.0 * np.pi * np.arange(neta) / neta
    axis = np.stack((3.0 * np.cos(eta), 3.0 * np.sin(eta), np.zeros(neta)), axis=-1)
    radial_modes = (
        0.18 * np.ones((1, neta, 3))
        + 0.35 * np.cos(theta)[:, None, None] * np.array([1.0, 0.0, 0.0])[None, None, :]
        + 0.22 * np.cos(2.0 * theta)[:, None, None] * np.array([0.0, 0.0, 1.0])[None, None, :]
    )
    wall = axis[None, :, :] + radial_modes
    u = np.array([0.0, 0.5, 1.0])
    positions = axis_regular_initializer(wall, axis, u, poloidal_modes=3)

    assert positions.shape == (3, ntheta, neta, 3)
    assert np.allclose(positions[0], axis[None, :, :])
    assert np.allclose(positions[-1], wall)
    relative = positions[1] - axis[None, :, :]
    coeff = np.fft.fft(relative, axis=0) / ntheta
    wall_coeff = np.fft.fft((wall - axis[None, :, :]), axis=0) / ntheta
    assert np.allclose(coeff[0], 0.25 * wall_coeff[0], atol=1e-12)
    assert np.allclose(coeff[1], 0.5 * wall_coeff[1], atol=1e-12)
    assert np.allclose(coeff[2], 0.25 * wall_coeff[2], atol=1e-12)


def test_project_interior_eta_uses_cartesian_gradient_and_keeps_boundaries():
    eta = 2.0 * np.pi * np.arange(12) / 12
    theta = 2.0 * np.pi * np.arange(8) / 8
    u = np.array([0.0, 0.4, 1.0])
    positions = np.empty((u.size, theta.size, eta.size, 3))
    for i, radius in enumerate(3.0 + 0.5 * u):
        phase = eta + (0.25 if i == 1 else 0.0)
        positions[i] = np.stack(
            (
                np.broadcast_to(radius * np.cos(phase), (theta.size, eta.size)),
                np.broadcast_to(radius * np.sin(phase), (theta.size, eta.size)),
                np.zeros((theta.size, eta.size)),
            ),
            axis=-1,
        )
    original_boundaries = positions[[0, -1]].copy()

    projected = project_interior_eta(positions, _EtaEvaluator(), eta, period=2.0 * np.pi, iterations=4)
    projected_eta = np.mod(np.arctan2(projected[1, :, :, 1], projected[1, :, :, 0]), 2.0 * np.pi)
    assert np.max(np.abs(np.angle(np.exp(1j * (projected_eta - eta[None, :]))))) < 1e-9
    assert np.allclose(projected[[0, -1]], original_boundaries)


def test_build_circular_torus_has_positive_regularized_quality_and_exact_wall():
    ntheta, neta = 20, 24
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    eta = 2.0 * np.pi * np.arange(neta) / neta
    u = np.linspace(0.0, 1.0, 6)
    evaluator = build_metric_evaluator(
        _EtaEvaluator(), topology="toroidal", wall_evaluator=_CircularWall(),
        mesh_shape=(u.size, theta.size, eta.size), logical_axes=(u, theta, eta),
        radial_degree=5, poloidal_modes=5,
        projection_iterations=2, resample_wall=True, wall_sample_count=ntheta + 1,
    )

    assert evaluator.topology == "toroidal"
    grid = np.stack(np.meshgrid(u, theta, eta, indexing="ij"), axis=-1)
    positions = evaluator.position(grid)
    expected_wall = _torus_wall(theta, eta)
    quality = evaluate_toroidal_quality(
        positions, u, eta, period=2.0 * np.pi,
        wall_reference=expected_wall, axis_reference=np.stack((3.0 * np.cos(eta), 3.0 * np.sin(eta), np.zeros_like(eta)), axis=-1),
        eta_evaluator=_EtaEvaluator(), axis_oversample=6,
    )
    regularized = evaluator.evaluate_regularized(np.array([[0.0, 0.31, 1.17], [0.2, 1.2, 2.1]]))
    assert quality.min_J_reg > 0.0
    assert quality.wall_error_max < 2e-8
    assert quality.eta_residual_max < 2e-8
    assert np.all(regularized.valid)
    assert np.all(regularized.J > 0.0)
    assert np.all(np.isfinite(regularized.condition))
    with pytest.raises(ValueError, match="singular|regularized"):
        evaluator.evaluate(np.array([[0.0, 0.3, 1.0]]))


def test_default_axis_uses_aligned_wall_centroid_and_projects_normalized_eta():
    ntheta, neta = 20, 24
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    eta = 2.0 * np.pi * np.arange(neta) / neta
    u = np.linspace(0.0, 1.0, 6)
    wall = _OffsetCenterlineWall()
    eta_evaluator = _NormalizedEtaEvaluator()
    evaluator = build_metric_evaluator(
        eta_evaluator, topology="toroidal", wall_evaluator=wall,
        mesh_shape=(u.size, theta.size, eta.size), logical_axes=(u, theta, eta),
        radial_degree=5, poloidal_modes=5, eta_is_normalized=True,
        resample_wall=True, wall_sample_count=ntheta + 1,
    )

    axis = evaluator.position(np.stack(np.meshgrid([0.0], theta, eta, indexing="ij"), axis=-1))[0]
    expected_axis = np.stack(
        (3.0 * np.cos(eta), 3.0 * np.sin(eta), np.zeros_like(eta)), axis=-1
    )
    assert np.allclose(axis, expected_axis[None, :, :], atol=2.0e-10)
    axis_eta = eta_evaluator.evaluate_and_gradient_cartesian(expected_axis)[0] * (2.0 * np.pi)
    residual = (axis_eta - eta + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(residual)) < 2.0e-10
    assert np.all(wall.contains_cartesian(expected_axis))

    quality = evaluate_toroidal_quality(
        evaluator.position(np.stack(np.meshgrid(u, theta, eta, indexing="ij"), axis=-1)),
        u, eta, period=2.0 * np.pi, axis_reference=expected_axis,
        eta_evaluator=eta_evaluator, eta_is_normalized=True, axis_oversample=6,
    )
    assert quality.fit_residual_max < 2.0e-10
    assert quality.eta_residual_max < 2.0e-10
    assert quality.min_J_reg > 0.0
