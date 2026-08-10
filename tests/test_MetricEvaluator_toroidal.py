"""Analytic acceptance tests for the dual-topology MetricEvaluator API."""

import numpy as np
import pytest

from drbx.geometry.MetricEvaluator import MetricEvaluator, build_metric_evaluator


R0 = 3.0
A = 0.7
PERIOD = 2.0 * np.pi


def circular_torus_map(u, theta, eta):
    """Return X=((R0+a*u*cos(theta))*cos(eta), ..., a*u*sin(theta))."""
    radius = R0 + A * u * np.cos(theta)
    return np.stack(
        (
            radius * np.cos(eta),
            radius * np.sin(eta),
            -A * u * np.sin(theta),
        ),
        axis=-1,
    )


def circular_torus_jacobian(u, theta, eta):
    radius = R0 + A * u * np.cos(theta)
    cu, su = np.cos(theta), np.sin(theta)
    ce, se = np.cos(eta), np.sin(eta)
    x_u = np.stack((A * cu * ce, A * cu * se, -A * su), axis=-1)
    x_theta = np.stack((-A * u * su * ce, -A * u * su * se, -A * u * cu), axis=-1)
    x_eta = np.stack((-radius * se, radius * ce, np.zeros_like(radius)), axis=-1)
    return np.stack((x_u, x_theta, x_eta), axis=-1)


def circular_torus_axes(nu=9, ntheta=16, neta=20):
    u = np.linspace(0.0, 1.0, nu)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    eta = PERIOD * np.arange(neta) / neta
    ug, tg, eg = np.meshgrid(u, theta, eta, indexing="ij")
    return u, theta, eta, circular_torus_map(ug, tg, eg)


def make_toroidal_evaluator(**kwargs):
    u, theta, eta, positions = circular_torus_axes()
    options = dict(
        period=PERIOD,
        topology="toroidal",
        radial_degree=3,
        poloidal_modes=1,
        toroidal_modes=0,
    )
    options.update(kwargs)
    return MetricEvaluator(u, theta, eta, positions, **options)


def test_circular_torus_position_and_jacobian_match_analytic_map():
    evaluator = make_toroidal_evaluator()
    q = np.array(
        [
            [0.17, 0.23, 0.31],
            [0.52, 2.10, 1.73],
            [0.91, 5.41, 5.87],
        ]
    )
    np.testing.assert_allclose(
        evaluator.position(q), circular_torus_map(q[:, 0], q[:, 1], q[:, 2]),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        evaluator.jacobian_matrix(q),
        circular_torus_jacobian(q[:, 0], q[:, 1], q[:, 2]),
        rtol=2.0e-7,
        atol=2.0e-9,
    )


def test_axis_collapses_and_ordinary_evaluate_rejects_exact_axis():
    evaluator = make_toroidal_evaluator()
    q_axis = np.array([[0.0, angle, 0.37] for angle in np.linspace(0.0, PERIOD, 9)[:-1]])
    expected_axis = np.broadcast_to(
        np.array([R0 * np.cos(0.37), R0 * np.sin(0.37), 0.0]),
        (q_axis.shape[0], 3),
    )
    np.testing.assert_allclose(evaluator.position(q_axis), expected_axis, atol=1.0e-14)
    with pytest.raises(ValueError, match="axis|Jacobian|nonpositive|regular"):
        evaluator.evaluate(np.array([[0.0, 0.4, 0.37]]))


def test_regularized_metrics_are_finite_positive_and_exact_for_circular_torus():
    evaluator = make_toroidal_evaluator()
    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.2, 2.4],
            [0.35, 0.8, 1.1],
            [0.9, 4.7, 5.2],
        ]
    )
    result = evaluator.evaluate_regularized(q)
    expected_jreg = A * A * (R0 + A * q[:, 0] * np.cos(q[:, 1]))
    np.testing.assert_allclose(result.J_reg, expected_jreg, rtol=2.0e-7, atol=2.0e-10)
    assert np.all(np.isfinite(result.J_reg))
    assert np.all(result.J_reg > 0.0)
    assert np.all(np.isfinite(result.condition_number))
    assert np.all(result.condition_number >= 1.0)


def test_coupled_poloidal_toroidal_modes_are_representable():
    u, theta, eta, _ = circular_torus_axes(nu=9, ntheta=20, neta=24)
    ug, tg, eg = np.meshgrid(u, theta, eta, indexing="ij")
    angle = tg + 0.35 * np.sin(eg)
    radius = R0 + A * ug * (np.cos(angle) + 0.25 * np.cos(2.0 * angle))
    positions = np.stack(
        (radius * np.cos(eg), radius * np.sin(eg), -A * ug * np.sin(angle)), axis=-1
    )
    evaluator = MetricEvaluator(
        u,
        theta,
        eta,
        positions,
        period=PERIOD,
        topology="toroidal",
        radial_degree=4,
        poloidal_modes=2,
        toroidal_modes=1,
    )
    q = np.array([[0.4, 1.1, 0.7], [0.8, 4.0, 4.9]])
    assert np.all(np.isfinite(evaluator.position(q)))
    regularized = evaluator.evaluate_regularized(q)
    assert np.all(regularized.J_reg > 0.0)
    assert np.all(np.isfinite(regularized.condition_number))

    h = 2.0e-6
    finite_difference = np.empty((q.shape[0], 3, 3))
    for coordinate in range(3):
        offset = np.zeros_like(q)
        offset[:, coordinate] = h
        finite_difference[..., coordinate] = (
            evaluator.position(q + offset) - evaluator.position(q - offset)
        ) / (2.0 * h)
    np.testing.assert_allclose(
        evaluator.jacobian_matrix(q), finite_difference, rtol=3e-6, atol=3e-8
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radial_degree": 1, "poloidal_modes": 2},
        {"radial_degree": 2, "poloidal_modes": (0, 1, 3)},
        {"radial_degree": 3, "poloidal_modes": (-1, 1)},
        {"radial_degree": 3, "poloidal_modes": (0,)},
        {"radial_degree": 3, "poloidal_modes": 1, "toroidal_modes": (-1, 1)},
    ],
)
def test_forbidden_fourier_zernike_modes_are_rejected(kwargs):
    with pytest.raises(ValueError, match="mode|degree|admissible|Zernike"):
        make_toroidal_evaluator(**kwargs)


def test_invalid_axes_and_noncollapsing_axis_are_rejected():
    u, theta, eta, positions = circular_torus_axes()
    with pytest.raises(ValueError, match="theta|periodic|uniform|endpoint"):
        MetricEvaluator(
            u,
            np.linspace(0.0, PERIOD, theta.size),
            eta,
            positions,
            period=PERIOD,
            topology="toroidal",
            radial_degree=3,
            poloidal_modes=1,
            toroidal_modes=0,
        )

    noncollapsing = positions.copy()
    noncollapsing[0, 3, 2] += 0.02
    with pytest.raises(ValueError, match="axis|collapse|u=0|regular"):
        MetricEvaluator(
            u,
            theta,
            eta,
            noncollapsing,
            period=PERIOD,
            topology="toroidal",
            radial_degree=3,
            poloidal_modes=1,
            toroidal_modes=0,
        )


def test_existing_default_topology_remains_square():
    u = np.linspace(0.0, 1.0, 4)
    v = np.linspace(0.0, 1.0, 4)
    eta = PERIOD * np.arange(8) / 8.0
    ug, vg, eg = np.meshgrid(u, v, eta, indexing="ij")
    radius = R0 + 0.2 * ug + 0.1 * vg
    positions = np.stack((radius * np.cos(eg), radius * np.sin(eg), 0.3 * ug), axis=-1)
    evaluator = MetricEvaluator(u, v, eta, positions, period=PERIOD)
    assert evaluator.topology == "square"


def test_unified_factory_accepts_explicit_toroidal_positions():
    class EtaMetadata:
        period = PERIOD
        nfp = 1

    u, theta, eta, positions = circular_torus_axes()
    evaluator = build_metric_evaluator(
        EtaMetadata(),
        positions,
        topology="toroidal",
        logical_axes=(u, theta, eta),
        radial_degree=3,
        poloidal_modes=1,
        toroidal_modes=0,
    )
    assert evaluator.topology == "toroidal"
    with pytest.raises(ValueError, match="topology"):
        build_metric_evaluator(EtaMetadata(), positions, topology="annular")


def test_toroidal_cache_round_trip_preserves_position_and_regularized_frame():
    evaluator = make_toroidal_evaluator()
    restored = MetricEvaluator.from_cache_payload(
        evaluator.to_cache_payload(prefix="metric_"), prefix="metric_"
    )
    q = np.array([[0.0, 0.7, 1.1], [0.23, 2.2, 3.4], [0.91, 5.1, 5.8]])
    assert restored.topology == "toroidal"
    np.testing.assert_allclose(restored.position(q), evaluator.position(q), atol=2e-13)
    np.testing.assert_allclose(
        restored.evaluate_regularized(q).regularized_jacobian_matrix,
        evaluator.evaluate_regularized(q).regularized_jacobian_matrix,
        atol=2e-12,
    )


def test_regularized_jacobian_uses_normalized_field_period_coordinate():
    period = 0.5 * np.pi
    u, theta, _, positions = circular_torus_axes(neta=20)
    eta = period * np.arange(20) / 20
    ug, tg, eg = np.meshgrid(u, theta, eta, indexing="ij")
    positions = circular_torus_map(ug, tg, eg)
    evaluator = MetricEvaluator(
        u,
        theta,
        eta,
        positions,
        period=period,
        topology="toroidal",
        radial_degree=3,
        poloidal_modes=1,
        toroidal_modes=0,
    )
    q = np.array([[0.2, 0.7, 0.3], [0.8, 2.1, 1.2]])
    ordinary = evaluator.evaluate(q)
    regularized = evaluator.evaluate_regularized(q)
    np.testing.assert_allclose(
        regularized.J_reg,
        (period / (2.0 * np.pi)) * ordinary.J / q[:, 0],
        rtol=2e-9,
        atol=2e-12,
    )


def test_toroidal_sampling_includes_theta_seam_and_only_physical_wall_face():
    evaluator = make_toroidal_evaluator()
    centers = evaluator.cell_center_logical_points()
    assert centers.shape == (
        evaluator.u.size - 1,
        evaluator.theta.size,
        evaluator.eta.size,
        3,
    )
    wall_faces = evaluator.open_boundary_face_center_logical_points()
    assert wall_faces.shape == (
        evaluator.theta.size * evaluator.eta.size,
        3,
    )
    assert np.all(wall_faces[:, 0] == 1.0)
    with pytest.raises(NotImplementedError, match="toroidal"):
        evaluator.quality_report()
