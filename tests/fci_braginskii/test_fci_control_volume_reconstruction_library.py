"""Focused tests for the regular-chart reconstruction primitives."""

from __future__ import annotations

import numpy as np

from drbx.fci_braginskii.native.fci_control_volume_operators import (
    monomial_basis,
    monomial_exponents,
)


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
