"""Tests for the scalar-potential eta evaluator."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from drbx.fci_braginskii.geometry_build.Bfield_evaluator import (
    BFieldEvaluator,
)
from drbx.fci_braginskii.geometry_build.ScalarPotential_evaluator import (
    ScalarPotentialEvaluator,
    scalar_potential_evaluator_from_bfield,
)


class AnalyticPotentialField(BFieldEvaluator):
    """Independent exactly representable field with B = grad(Phi)."""

    def __init__(self) -> None:
        self._R = np.linspace(1.0, 1.6, 13)
        self._Z = np.linspace(-0.3, 0.3, 11)
        self._nfp = 2
        self._period = 2.0 * np.pi / self._nfp
        self._phi = np.arange(12, dtype=np.float64) * self._period / 12
        self._G = 1.7
        self._center = 1.3

    @property
    def R(self) -> np.ndarray:
        return self._R.copy()

    @property
    def phi(self) -> np.ndarray:
        return self._phi.copy()

    @property
    def Z(self) -> np.ndarray:
        return self._Z.copy()

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def currents(self) -> np.ndarray:
        return np.ones(1)

    @property
    def G(self) -> float:
        return self._G

    def eta_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        R, phi, Z = np.moveaxis(points, -1, 0)
        radius = R - self._center
        angle = self._nfp * phi
        periodic = (
            0.08 * radius * np.cos(angle)
            + 0.05 * Z * np.sin(angle)
            + 0.03 * radius * Z * np.cos(2.0 * angle)
            + 0.02 * radius**2 * np.sin(angle)
        )
        return phi + periodic / self._G

    def magnetic_potential_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        return self._G * self.eta_cylindrical(points_rphiz)

    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        R, phi, Z = np.moveaxis(points, -1, 0)
        radius = R - self._center
        angle = self._nfp * phi
        derivative_R = (
            0.08 * np.cos(angle)
            + 0.03 * Z * np.cos(2.0 * angle)
            + 0.04 * radius * np.sin(angle)
        )
        derivative_phi = (
            -0.08 * radius * self._nfp * np.sin(angle)
            + 0.05 * Z * self._nfp * np.cos(angle)
            - 0.06 * radius * Z * self._nfp * np.sin(2.0 * angle)
            + 0.02 * radius**2 * self._nfp * np.cos(angle)
        )
        derivative_Z = (
            0.05 * np.sin(angle)
            + 0.03 * radius * np.cos(2.0 * angle)
        )
        return np.stack(
            (derivative_R, (self._G + derivative_phi) / R, derivative_Z),
            axis=-1,
        )

    def evaluate_cartesian(self, points_xyz: Any) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        X, Y, Z = np.moveaxis(points, -1, 0)
        R = np.hypot(X, Y)
        phi = np.arctan2(Y, X)
        cylindrical = np.stack((R, phi, Z), axis=-1)
        field = self.evaluate_cylindrical(cylindrical)
        BR, Bphi, BZ = np.moveaxis(field, -1, 0)
        return np.stack(
            (
                BR * np.cos(phi) - Bphi * np.sin(phi),
                BR * np.sin(phi) + Bphi * np.cos(phi),
                BZ,
            ),
            axis=-1,
        )


class AnalyticIPotentialField(AnalyticPotentialField):
    """Exactly representable field with both poloidal I and toroidal G."""

    def __init__(self) -> None:
        super().__init__()
        self._I = 0.075

    @property
    def I(self) -> float:
        return self._I

    def reference_axis(
        self, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi = np.asarray(phi, dtype=np.float64)
        return (
            np.full_like(phi, self._center),
            np.zeros_like(phi),
            np.zeros_like(phi),
            np.zeros_like(phi),
        )

    def theta_reference(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        return np.arctan2(points[..., 2], points[..., 0] - self._center)

    def theta_gradient(self, points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        relative_R = points[..., 0] - self._center
        relative_Z = points[..., 2]
        radius_squared = relative_R**2 + relative_Z**2
        return np.stack(
            (
                -relative_Z / radius_squared,
                np.zeros_like(relative_R),
                relative_R / radius_squared,
            ),
            axis=-1,
        )

    def magnetic_potential_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        return (
            self._I * self.theta_reference(points_rphiz)
            + self._G * self.eta_cylindrical(points_rphiz)
        )

    def evaluate_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        return (
            super().evaluate_cylindrical(points_rphiz)
            + self._I * self.theta_gradient(points_rphiz)
        )


@pytest.fixture(scope="module")
def analytic_fit() -> tuple[AnalyticPotentialField, ScalarPotentialEvaluator]:
    field = AnalyticPotentialField()
    evaluator = scalar_potential_evaluator_from_bfield(
        field,
        radial_degree=2,
        vertical_degree=1,
        toroidal_modes=2,
        sample_shape=(7, 10, 6),
    )
    return field, evaluator


@pytest.fixture(scope="module")
def analytic_I_fit() -> tuple[AnalyticIPotentialField, ScalarPotentialEvaluator]:
    field = AnalyticIPotentialField()

    def annular_mask(points: np.ndarray) -> np.ndarray:
        return np.hypot(points[:, 0] - field._center, points[:, 2]) > 0.09

    evaluator = scalar_potential_evaluator_from_bfield(
        field,
        radial_degree=2,
        vertical_degree=1,
        toroidal_modes=2,
        sample_shape=(10, 12, 10),
        mask=annular_mask,
        reference_axis=field.reference_axis,
    )
    return field, evaluator


def _random_cylindrical(
    field: AnalyticPotentialField, count: int = 100
) -> np.ndarray:
    generator = np.random.default_rng(12345)
    return np.column_stack(
        (
            generator.uniform(field.R[0], field.R[-1], count),
            generator.uniform(-2.0 * field.period, 3.0 * field.period, count),
            generator.uniform(field.Z[0], field.Z[-1], count),
        )
    )


def _to_cartesian(points_rphiz: np.ndarray) -> np.ndarray:
    R, phi, Z = np.moveaxis(points_rphiz, -1, 0)
    return np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1)


def test_exact_gradient_projection_and_potential(analytic_fit):
    field, evaluator = analytic_fit
    points = _random_cylindrical(field)
    np.testing.assert_allclose(evaluator.G, field.G, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        evaluator.magnetic_field_cylindrical(points),
        field.evaluate_cylindrical(points),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.gradient_cylindrical(points),
        field.evaluate_cylindrical(points) / field.G,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_magnetic_potential_cylindrical(points),
        field.magnetic_potential_cylindrical(points),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(points),
        field.eta_cylindrical(points),
        atol=2e-11,
        rtol=2e-11,
    )


def test_joint_I_G_fit_and_normalized_eta(analytic_I_fit):
    field, evaluator = analytic_I_fit
    candidates = _random_cylindrical(field, 400)
    distance = np.hypot(
        candidates[:, 0] - field._center, candidates[:, 2]
    )
    points = candidates[distance > 0.1][:100]
    np.testing.assert_allclose(evaluator.I, field.I, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(evaluator.G, field.G, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(points),
        field.eta_cylindrical(points),
        atol=3e-11,
        rtol=3e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_magnetic_potential_cylindrical(points),
        field.magnetic_potential_cylindrical(points),
        atol=3e-11,
        rtol=3e-11,
    )
    fitted_field = evaluator.magnetic_field_cylindrical(points)
    np.testing.assert_allclose(
        fitted_field,
        field.evaluate_cylindrical(points),
        atol=3e-11,
        rtol=3e-11,
    )
    np.testing.assert_allclose(
        fitted_field,
        evaluator.I * field.theta_gradient(points)
        + evaluator.G * evaluator.gradient_cylindrical(points),
        atol=3e-11,
        rtol=3e-11,
    )
    assert evaluator.diagnostics["I"] == pytest.approx(field.I, abs=3e-12)
    assert evaluator.diagnostics["I_over_G"] == pytest.approx(
        field.I / field.G, abs=3e-12
    )


def test_scalar_potential_rejects_tiny_active_mask():
    with pytest.raises(ValueError, match="rank deficient|unconstrained"):
        scalar_potential_evaluator_from_bfield(
            AnalyticPotentialField(),
            radial_degree=1,
            vertical_degree=1,
            toroidal_modes=1,
            sample_shape=(2, 3, 2),
            mask=lambda points: np.arange(points.shape[0]) == 0,
        )


def test_quasiperiodicity_wrapping_and_batches(analytic_fit):
    field, evaluator = analytic_fit
    points = _random_cylindrical(field, 24).reshape((2, 3, 4, 3))
    shifted = points + np.array([0.0, field.period, 0.0])
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(shifted),
        evaluator.evaluate_cylindrical(points) + field.period,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_cylindrical(shifted, wrapped=True),
        evaluator.evaluate_cylindrical(points, wrapped=True),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_phase_cylindrical(shifted),
        evaluator.evaluate_phase_cylindrical(points) + field.period,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.evaluate_phase_cylindrical(shifted, wrapped=True),
        evaluator.evaluate_phase_cylindrical(points, wrapped=True),
        atol=2e-11,
        rtol=2e-11,
    )
    assert evaluator.evaluate_cylindrical(points).shape == (2, 3, 4)
    assert evaluator.gradient_cylindrical(points).shape == (2, 3, 4, 3)


def test_cartesian_value_and_gradient(analytic_fit):
    field, evaluator = analytic_fit
    cylindrical = _random_cylindrical(field, 40)
    cartesian = _to_cartesian(cylindrical)
    principal_cylindrical = cylindrical.copy()
    principal_cylindrical[:, 1] = np.arctan2(
        cartesian[:, 1], cartesian[:, 0]
    )
    np.testing.assert_allclose(
        evaluator.evaluate_cartesian(cartesian),
        field.eta_cylindrical(principal_cylindrical),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.gradient_cartesian(cartesian),
        field.evaluate_cartesian(cartesian) / field.G,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator.magnetic_field_cartesian(cartesian),
        field.evaluate_cartesian(cartesian),
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        evaluator(cartesian),
        evaluator.evaluate_cartesian(cartesian),
        atol=0.0,
        rtol=0.0,
    )


def test_value_only_path_skips_derivative_basis(monkeypatch, analytic_fit):
    _, evaluator = analytic_fit
    scalar_module = importlib.import_module(
        "drbx.fci_braginskii.geometry_build.ScalarPotential_evaluator"
    )
    original = scalar_module._chebyshev_derivative_vander

    def fail_if_called(*args, **kwargs):
        raise AssertionError("value-only evaluation constructed derivative basis")

    monkeypatch.setattr(
        scalar_module, "_chebyshev_derivative_vander", fail_if_called
    )
    points = _random_cylindrical(AnalyticPotentialField(), 20)
    evaluator.evaluate_cylindrical(points)
    monkeypatch.setattr(
        scalar_module, "_chebyshev_derivative_vander", original
    )


def test_combined_cartesian_value_and_gradient_matches_separate_calls(analytic_fit):
    field, evaluator = analytic_fit
    cylindrical = _random_cylindrical(field, 40)
    cartesian = _to_cartesian(cylindrical)
    value, gradient = evaluator.evaluate_and_gradient_cartesian(cartesian)
    np.testing.assert_allclose(
        value, evaluator.evaluate_cartesian(cartesian), atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        gradient, evaluator.gradient_cartesian(cartesian), atol=2e-12, rtol=2e-12
    )


def test_fit_diagnostics_and_bounds(analytic_fit):
    field, evaluator = analytic_fit
    diagnostics = evaluator.diagnostics
    required = {
        "sample_count",
        "unknown_count",
        "rank",
        "condition_number",
        "G",
        "I",
        "I_over_G",
        "weighted_relative_l2_error",
        "rms_absolute_error",
        "relative_l2_error",
        "max_absolute_error",
        "max_relative_error",
        "component_rms_errors",
        "min_normalized_phase_derivative",
        "max_normalized_phase_derivative",
        "folded_fraction",
    }
    assert required <= diagnostics.keys()
    assert diagnostics["weighted_relative_l2_error"] < 2e-11
    assert diagnostics["relative_l2_error"] < 2e-11
    assert diagnostics["max_absolute_error"] < 2e-10
    assert diagnostics["rank"] == diagnostics["unknown_count"]
    assert diagnostics["folded_fraction"] == 0.0
    with pytest.raises(ValueError):
        evaluator.evaluate_cylindrical(
            [field.R[0] - 0.01, field.phi[0], field.Z[0]]
        )


def test_constant_eta_planes_are_single_valued_phi_graphs(analytic_fit):
    """Each eta level is crossed once per period and bisection reaches 1e-10."""

    _, evaluator = analytic_fit
    RR, ZZ = np.meshgrid(
        np.linspace(evaluator.R[0], evaluator.R[-1], 10),
        np.linspace(evaluator.Z[0], evaluator.Z[-1], 9),
        indexing="ij",
    )
    period = evaluator.period
    for target in np.arange(5) * period / 5:
        phi = evaluator.phi[0] + target + np.linspace(-0.5, 0.5, 33) * period
        points = np.stack(np.broadcast_arrays(RR[..., None], phi, ZZ[..., None]), axis=-1)
        eta = np.asarray(evaluator.evaluate_cylindrical(points), dtype=np.float64)
        assert np.all(np.diff(eta, axis=-1) > 0.0)
        assert np.all(RR[..., None] * evaluator.gradient_cylindrical(points)[..., 1] > 0.0)
        lower = np.full_like(RR, phi[0])
        upper = np.full_like(RR, phi[-1])
        assert np.all(eta[..., 0] < target) and np.all(eta[..., -1] > target)
        for _ in range(48):
            middle = 0.5 * (lower + upper)
            above = evaluator.evaluate_cylindrical(np.stack((RR, middle, ZZ), axis=-1)) > target
            upper = np.where(above, middle, upper)
            lower = np.where(above, lower, middle)
        root = np.stack((RR, 0.5 * (lower + upper), ZZ), axis=-1)
        np.testing.assert_allclose(evaluator.evaluate_cylindrical(root), target, rtol=0.0, atol=1e-10)
