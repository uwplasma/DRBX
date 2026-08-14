"""Tests for host-side regular-chart geometry and J-weighted moments."""

from __future__ import annotations

import numpy as np

from drbx.geometry.fci_control_volumes import (
    integrate_polar_regular_chart_cell_moments,
    polar_regular_chart,
    polar_regular_chart_jacobian,
)


def test_chart_and_analytic_jacobian_are_regular_at_axis() -> None:
    points = np.array([[0.0, 0.37, 6.0], [0.5, 0.37, 6.0]])
    chart = polar_regular_chart(points)
    jacobian = polar_regular_chart_jacobian(points)
    np.testing.assert_allclose(chart[0, :2], 0.0)
    np.testing.assert_allclose(
        jacobian[0],
        [[np.cos(0.37), 0.0, 0.0], [np.sin(0.37), 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.all(np.isfinite(jacobian))


def test_eta_unwrap_uses_explicit_periodic_branch() -> None:
    points = np.array([[0.5, 0.0, 5.9], [0.5, 0.0, 0.1], [0.5, 0.0, 6.4]])
    chart = polar_regular_chart(
        points, eta_unwrap_origin=0.0, eta_period=2.0 * np.pi
    )
    np.testing.assert_allclose(chart[:, 2], [-0.383185307179586, 0.1, 0.116814692820414])


def test_constant_j_cell_volume_and_moments() -> None:
    volume, centroid, second, third = integrate_polar_regular_chart_cell_moments(
        np.array([0.0, 1.0]),
        np.array([0.0, 0.5 * np.pi]),
        np.array([0.0, 1.0]),
        lambda points: np.ones(points.shape[:-1]),
        quadrature_order=8,
    )
    assert volume.shape == (1, 1, 1)
    np.testing.assert_allclose(volume[0, 0, 0], 0.5 * np.pi)
    np.testing.assert_allclose(centroid[0, 0, 0], [1.0 / np.pi, 1.0 / np.pi, 0.5])
    expected_second = np.array(
        [
            [1.0 / 6.0 - 1.0 / np.pi**2, 1.0 / (3.0 * np.pi) - 1.0 / np.pi**2, 0.0],
            [1.0 / (3.0 * np.pi) - 1.0 / np.pi**2, 1.0 / 6.0 - 1.0 / np.pi**2, 0.0],
            [0.0, 0.0, 1.0 / 12.0],
        ]
    )
    np.testing.assert_allclose(second[0, 0, 0], expected_second, rtol=1.0e-12, atol=1.0e-12)
    assert np.all(np.isfinite(third))


def test_nonconstant_j_weighting_and_quadrature_order() -> None:
    def jacobian(points):
        return 2.0 + points[..., 0] + 0.3 * points[..., 2]

    faces = (np.array([0.0, 0.8]), np.array([0.0, 1.1]), np.array([-0.4, 0.6]))
    low = integrate_polar_regular_chart_cell_moments(*faces, jacobian, quadrature_order=5)
    high = integrate_polar_regular_chart_cell_moments(*faces, jacobian, quadrature_order=8)
    # J is affine, so a modest tensor rule is already very accurate;
    # the higher-order result is a reference for the trigonometric chart.
    np.testing.assert_allclose(low[0], high[0], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(low[1], high[1], rtol=2.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(low[2], high[2], rtol=2.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(low[3], high[3], rtol=1.0e-6, atol=1.0e-10)


def test_theta_rotation_covariance_of_chart_moments() -> None:
    shift = 0.41
    base = integrate_polar_regular_chart_cell_moments(
        np.array([0.1, 0.9]), np.array([0.2, 1.0]), np.array([0.0, 0.7]),
        lambda points: np.ones(points.shape[:-1]), quadrature_order=7,
    )
    rotated = integrate_polar_regular_chart_cell_moments(
        np.array([0.1, 0.9]), np.array([0.2 + shift, 1.0 + shift]), np.array([0.0, 0.7]),
        lambda points: np.ones(points.shape[:-1]), quadrature_order=7,
    )
    rotation = np.array(
        [[np.cos(shift), -np.sin(shift), 0.0], [np.sin(shift), np.cos(shift), 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(rotated[0], base[0])
    np.testing.assert_allclose(rotated[1][0, 0, 0], rotation @ base[1][0, 0, 0])
    np.testing.assert_allclose(
        rotated[2][0, 0, 0], rotation @ base[2][0, 0, 0] @ rotation.T,
        atol=1.0e-15,
    )
    expected_third = np.einsum(
        "ia,jb,kc,abc->ijk", rotation, rotation, rotation, base[3][0, 0, 0]
    )
    np.testing.assert_allclose(rotated[3][0, 0, 0], expected_third, atol=1.0e-15)

