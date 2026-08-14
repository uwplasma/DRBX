"""Focused tests for the regular-chart reconstruction primitives."""

from __future__ import annotations

import numpy as np

from drbx.native.fci_boundaries import CV_RECONSTRUCTION_EQUATION_CELL
from drbx.native.fci_control_volume_operators import (
    CUBIC_MONOMIAL_EXPONENTS,
    control_volume_average_basis,
    cubic_control_volume_average_basis,
    cubic_monomial_basis,
    cubic_projected_face_flux_target,
    monomial_basis,
    monomial_exponents,
    monomial_logical_gradient_target,
    monomial_value_target,
    precompute_local_face_functional,
    projected_face_flux_target,
)


def _box_moments(centroid: np.ndarray, widths: np.ndarray):
    second = np.diag(widths**2 / 12.0)
    third = np.zeros((3, 3, 3), dtype=np.float64)
    return centroid, second, third


def test_selected_degrees_and_explicit_exponents_have_expected_sizes() -> None:
    assert len(monomial_exponents(1)) == 4
    assert len(monomial_exponents(2)) == 10
    assert len(monomial_exponents(3)) == 20
    selected = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0))
    assert monomial_exponents(exponents=selected) == selected
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    np.testing.assert_allclose(
        monomial_basis(points, exponents=selected),
        np.array([[1.0, 1.0, 1.0, 2.0], [1.0, -1.0, 1.0, -0.5]]),
    )


def test_selected_moment_basis_reproduces_constant_linear_quadratic_and_cubic() -> None:
    centroid, second, third = _box_moments(
        np.array([0.4, -0.3, 0.2]), np.array([0.8, 1.1, 0.6])
    )
    basis = control_volume_average_basis(
        centroid, second, third, origin=np.zeros(3), scale=1.0, total_degree=3
    )
    assert basis.shape == (20,)
    # The selected moments agree with exact averages of a box.  In particular,
    # this checks the mixed quadratic and repeated-axis cubic terms.
    exponents = monomial_exponents(3)
    expected = []
    for power in exponents:
        value = 1.0
        for axis, exponent in enumerate(power):
            if exponent == 0:
                continue
            if exponent == 1:
                value *= centroid[axis]
            elif exponent == 2:
                value *= centroid[axis] ** 2 + second[axis, axis]
            elif exponent == 3:
                value *= centroid[axis] ** 3 + 3.0 * centroid[axis] * second[axis, axis]
        # The box is axis-aligned, so products of independent one-dimensional
        # moments are exact for every selected monomial.
        expected.append(value)
    np.testing.assert_allclose(basis, expected)
    np.testing.assert_allclose(
        basis,
        cubic_control_volume_average_basis(centroid, second, third),
    )


def test_generic_face_functional_dimensions_and_reproduction_for_degrees_1_to_3() -> None:
    rng = np.random.default_rng(12)
    points = rng.uniform(-1.0, 1.0, size=(28, 3))
    kinds = np.full((28,), CV_RECONSTRUCTION_EQUATION_CELL, dtype=np.int32)
    refs = np.arange(28, dtype=np.int64)
    for degree in (1, 2, 3):
        exponents = monomial_exponents(degree)
        matrix = monomial_basis(points, exponents=exponents)
        values = monomial_value_target(
            points[:1], origin=np.zeros(3), scale=1.0, exponents=exponents
        )[0]
        gradients = monomial_logical_gradient_target(
            points[:1], origin=np.zeros(3), scale=1.0, exponents=exponents
        )[0]
        functional = precompute_local_face_functional(
            matrix,
            equation_kind=kinds,
            sample_reference=refs,
            value_target=values,
            gradient_target=gradients,
            requested_order=degree,
            exponents=exponents,
            max_derivative_l1=1.0e12,
            max_projected_flux_l1=1.0e12,
            max_parallel_flux_l1=1.0e12,
            max_parallel_gradient_flux_l1=1.0e12,
        )
        assert functional.rank == len(exponents)
        assert functional.polynomial_order == degree
        assert functional.value_weights.shape == (28,)
        assert functional.gradient_weights.shape == (3, 28)
        assert functional.reproduction_residual < 1.0e-10


def test_chart_chain_rule_changes_logical_projected_flux_target() -> None:
    # points are chart coordinates chi; the area and projector are still
    # expressed in logical xi coordinates.
    points = np.array([[0.2, -0.1, 0.4]], dtype=np.float64)
    jacobian = np.ones((1,), dtype=np.float64)
    area = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    projector = np.broadcast_to(np.eye(3), (1, 3, 3)).copy()
    active = np.ones((1,), dtype=bool)
    chart_jacobian = np.array(
        [[[2.0, 0.5, 0.0], [-1.0, 3.0, 0.0], [0.25, -0.5, 1.5]]]
    )
    exponents = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    target = projected_face_flux_target(
        points,
        jacobian,
        area,
        projector,
        active,
        origin=np.zeros(3),
        scale=1.0,
        dchi_dxi=chart_jacobian,
        exponents=exponents,
    )
    expected = np.array([
        np.dot(area[0], chart_jacobian[0, 0]),
        np.dot(area[0], chart_jacobian[0, 1]),
        np.dot(area[0], chart_jacobian[0, 2]),
    ])
    np.testing.assert_allclose(target, expected)
    identity_target = cubic_projected_face_flux_target(
        points,
        jacobian,
        area,
        projector,
        active,
        origin=np.zeros(3),
        scale=1.0,
    )
    # The legacy cubic ordering starts with the constant monomial.
    np.testing.assert_allclose(identity_target[1:4], area[0])
    assert not np.allclose(target, identity_target[:3])
