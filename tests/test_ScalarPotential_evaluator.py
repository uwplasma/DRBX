"""Tests, diagnostics CLI, and constant-eta plotting for scalar potentials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drbx.geometry.Bfield_evaluator import (
    BFieldEvaluator,
    bfield_evaluator_from_makegrid,
)
from drbx.geometry.ScalarPotential_evaluator import (
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


def plot_constant_eta_planes(
    evaluator: ScalarPotentialEvaluator,
    filename: str | Path,
    *,
    surface_count: int = 8,
    nR: int = 28,
    nZ: int = 28,
    bisection_iterations: int = 48,
    residual_tolerance: float = 1e-10,
    mask: Any = None,
    show: bool = False,
):
    """Save interactive constant-eta surfaces as a self-contained HTML file."""

    import plotly.graph_objects as go

    if surface_count < 1 or nR < 2 or nZ < 2:
        raise ValueError("surface_count must be positive and nR/nZ must be >= 2")
    if bisection_iterations < 1 or residual_tolerance <= 0.0:
        raise ValueError(
            "bisection_iterations and residual_tolerance must be positive"
        )
    if abs(evaluator.G) <= np.finfo(np.float64).eps:
        raise ValueError("constant-eta planes require a nonzero fitted G")
    filename = Path(filename)
    if filename.suffix.lower() != ".html":
        raise ValueError("interactive Plotly output filename must end in .html")

    R = np.linspace(evaluator.R[0], evaluator.R[-1], nR)
    Z = np.linspace(evaluator.Z[0], evaluator.Z[-1], nZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    period = evaluator.period
    phi0 = evaluator.phi[0]
    targets = np.arange(surface_count, dtype=np.float64) * period / surface_count

    figure = go.Figure()
    for surface_index, target in enumerate(targets):
        branch_center = phi0 + target
        lower = np.full_like(RR, branch_center - 0.5 * period)
        upper = np.full_like(RR, branch_center + 0.5 * period)

        def eta_residual(query_phi: np.ndarray) -> np.ndarray:
            points = np.stack((RR, query_phi, ZZ), axis=-1)
            return np.asarray(
                evaluator.evaluate_cylindrical(points), dtype=np.float64
            ) - target

        lower_residual = eta_residual(lower)
        upper_residual = eta_residual(upper)
        bracketed = (
            np.isfinite(lower_residual)
            & np.isfinite(upper_residual)
            & (
                (lower_residual == 0.0)
                | (upper_residual == 0.0)
                | (np.signbit(lower_residual) != np.signbit(upper_residual))
            )
        )
        for _ in range(bisection_iterations):
            midpoint = 0.5 * (lower + upper)
            midpoint_residual = eta_residual(midpoint)
            root_in_lower_half = (
                (lower_residual == 0.0)
                | (midpoint_residual == 0.0)
                | (
                    np.signbit(lower_residual)
                    != np.signbit(midpoint_residual)
                )
            )
            update_upper = bracketed & root_in_lower_half
            update_lower = bracketed & ~root_in_lower_half
            upper = np.where(update_upper, midpoint, upper)
            upper_residual = np.where(
                update_upper, midpoint_residual, upper_residual
            )
            lower = np.where(update_lower, midpoint, lower)
            lower_residual = np.where(
                update_lower, midpoint_residual, lower_residual
            )

        phi = 0.5 * (lower + upper)
        points = np.stack((RR, phi, ZZ), axis=-1)
        residual = np.abs(eta_residual(phi))
        derivative = RR * evaluator.gradient_cylindrical(points)[..., 1]
        valid = (
            bracketed
            & np.isfinite(residual)
            & (residual <= residual_tolerance)
            & np.isfinite(derivative)
            & (derivative > 0.0)
        )
        if mask is not None:
            valid &= np.asarray(mask(points), dtype=bool)

        X = RR * np.cos(phi)
        Y = RR * np.sin(phi)
        X = np.where(valid, X, np.nan)
        Y = np.where(valid, Y, np.nan)
        surface_Z = np.where(valid, ZZ, np.nan)
        surface_phi = np.where(valid, phi, np.nan)
        surface_color = np.full_like(X, target)
        figure.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=surface_Z,
                customdata=surface_phi,
                surfacecolor=surface_color,
                cmin=0.0,
                cmax=period,
                colorscale="Turbo",
                opacity=0.62,
                showscale=surface_index == 0,
                colorbar=dict(
                    title=dict(text="eta modulo period [rad]"),
                    len=0.75,
                ),
                name=f"eta={target:.4f}",
                showlegend=True,
                hovertemplate=(
                    f"eta={target:.5f} rad"
                    "<br>x=%{x:.5f} m"
                    "<br>y=%{y:.5f} m"
                    "<br>z=%{z:.5f} m"
                    "<br>phi=%{customdata:.5f} rad"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=(
            f"{surface_count} constant eta planes over one field period "
            f"(NFP={evaluator.nfp})"
        ),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
        ),
        legend=dict(title="Constant eta surfaces"),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(filename),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    if show:
        figure.show()
    return figure


def _load_flare_vessel_geometry(
    filename: str | Path, expected_nfp: int
):
    """Return a vessel mask and periodic centroid reference-axis evaluator."""

    from matplotlib.path import Path as Polygon
    from scipy.interpolate import CubicSpline

    tokens: list[str] = []
    for line in Path(filename).read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            tokens.extend(stripped.split())
    if len(tokens) < 3:
        raise ValueError("vessel file is missing its dimensions")
    nplanes, npoints, nfp = map(int, tokens[:3])
    if nfp != expected_nfp:
        raise ValueError(
            f"vessel NFP={nfp} does not match magnetic-field NFP={expected_nfp}"
        )
    offset = 3
    phi = np.empty(nplanes)
    wall_R = np.empty((nplanes, npoints))
    wall_Z = np.empty((nplanes, npoints))
    for plane in range(nplanes):
        phi[plane] = float(tokens[offset])
        offset += 1
        coordinates = np.asarray(
            tokens[offset : offset + 2 * npoints], dtype=np.float64
        ).reshape((npoints, 2))
        offset += 2 * npoints
        wall_R[plane] = coordinates[:, 0]
        wall_Z[plane] = coordinates[:, 1]
    if offset != len(tokens):
        raise ValueError("vessel file contains unexpected trailing data")
    if np.max(np.abs(phi)) > 2.0 * np.pi + 1e-9:
        phi = np.deg2rad(phi)
    if np.any(np.diff(phi) <= 0.0):
        raise ValueError("vessel toroidal planes must be strictly increasing")
    period = 2.0 * np.pi / nfp
    if not np.isclose(phi[-1] - phi[0], period, rtol=1e-6, atol=1e-9):
        raise ValueError("vessel planes must include both field-period endpoints")
    centroid_R = np.mean(wall_R, axis=1)
    centroid_Z = np.mean(wall_Z, axis=1)
    centroid_R[-1] = centroid_R[0]
    centroid_Z[-1] = centroid_Z[0]
    axis_R_spline = CubicSpline(phi, centroid_R, bc_type="periodic")
    axis_Z_spline = CubicSpline(phi, centroid_Z, bc_type="periodic")

    def reference_axis(
        query_phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wrapped = phi[0] + np.mod(np.asarray(query_phi) - phi[0], period)
        return (
            axis_R_spline(wrapped),
            axis_Z_spline(wrapped),
            axis_R_spline(wrapped, 1),
            axis_Z_spline(wrapped, 1),
        )

    def vessel_mask(points_rphiz: Any) -> np.ndarray:
        points = np.asarray(points_rphiz, dtype=np.float64)
        leading_shape = points.shape[:-1]
        flat = points.reshape((-1, 3))
        wrapped_phi = phi[0] + np.mod(flat[:, 1] - phi[0], period)
        upper = np.searchsorted(phi, wrapped_phi, side="right")
        upper = np.clip(upper, 1, len(phi) - 1)
        lower = upper - 1
        fraction = (wrapped_phi - phi[lower]) / (phi[upper] - phi[lower])
        result = np.zeros(flat.shape[0], dtype=bool)
        for lower_index, upper_index in np.unique(
            np.column_stack((lower, upper)), axis=0
        ):
            selected = (lower == lower_index) & (upper == upper_index)
            local_fraction = fraction[selected]
            # Collocation and plot meshes use one phi value per selected
            # group, but retain this fallback for arbitrary point clouds.
            for interpolation_fraction in np.unique(local_fraction):
                local = selected & np.isclose(
                    fraction, interpolation_fraction, rtol=0.0, atol=1e-14
                )
                contour_R = (
                    (1.0 - interpolation_fraction) * wall_R[lower_index]
                    + interpolation_fraction * wall_R[upper_index]
                )
                contour_Z = (
                    (1.0 - interpolation_fraction) * wall_Z[lower_index]
                    + interpolation_fraction * wall_Z[upper_index]
                )
                polygon = Polygon(
                    np.column_stack((contour_R, contour_Z)), closed=True
                )
                result[local] = polygon.contains_points(
                    flat[local][:, (0, 2)], radius=1e-12
                )
        return result.reshape(leading_shape)

    return vessel_mask, reference_axis


def test_constant_eta_plot_smoke(analytic_fit, tmp_path):
    _, evaluator = analytic_fit
    output = tmp_path / "constant_eta_planes.html"
    figure = plot_constant_eta_planes(
        evaluator, output, surface_count=5, nR=10, nZ=9
    )
    assert output.is_file()
    contents = output.read_text()
    assert output.stat().st_size > 100_000
    assert "Plotly.newPlot" in contents
    assert "constant eta planes" in contents

    targets = np.arange(5, dtype=np.float64) * evaluator.period / 5
    for trace, target in zip(figure.data, targets, strict=True):
        X = np.asarray(trace.x, dtype=np.float64)
        Y = np.asarray(trace.y, dtype=np.float64)
        Z = np.asarray(trace.z, dtype=np.float64)
        phi = np.asarray(trace.customdata, dtype=np.float64)
        valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        plot_R = np.linspace(evaluator.R[0], evaluator.R[-1], X.shape[0])
        plot_Z = np.linspace(evaluator.Z[0], evaluator.Z[-1], X.shape[1])
        RR, ZZ = np.meshgrid(plot_R, plot_Z, indexing="ij")
        points = np.stack(
            (RR[valid], phi[valid], ZZ[valid]),
            axis=-1,
        )
        np.testing.assert_allclose(
            evaluator.evaluate_cylindrical(points),
            target,
            rtol=0.0,
            atol=1e-10,
        )
        assert np.max(np.abs(phi[valid] - (evaluator.phi[0] + target))) <= (
            0.5 * evaluator.period + 1e-12
        )

    # eta=0 straddles the field-period seam. Its unwrapped phi branch must
    # remain continuous so Plotly does not connect phi=0 to phi=period with
    # long, unphysical polygons.
    first_surface = figure.data[0]
    coordinates = np.stack(
        (
            np.asarray(first_surface.x, dtype=np.float64),
            np.asarray(first_surface.y, dtype=np.float64),
            np.asarray(first_surface.z, dtype=np.float64),
        ),
        axis=-1,
    )
    radial_edges = np.linalg.norm(np.diff(coordinates, axis=0), axis=-1)
    vertical_edges = np.linalg.norm(np.diff(coordinates, axis=1), axis=-1)
    assert max(np.max(radial_edges), np.max(vertical_edges)) < 0.25


def _print_diagnostics(diagnostics: Any) -> None:
    print("scalar-potential fit diagnostics")
    for key, value in diagnostics.items():
        array = np.asarray(value)
        if array.ndim == 0:
            item = array.item()
            if isinstance(item, float):
                print(f"  {key}: {item:.8e}")
            else:
                print(f"  {key}: {item}")
        else:
            print(f"  {key}: {np.array2string(array, precision=8)}")


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Phi by minimizing the volume-weighted error between grad(Phi) "
            "and a MAKEGRID B field, then construct normalized mesh eta"
        )
    )
    parser.add_argument("makegrid", type=Path)
    parser.add_argument("--currents", nargs="+", type=float)
    parser.add_argument("--radial-degree", type=int, default=5)
    parser.add_argument("--vertical-degree", type=int, default=5)
    parser.add_argument("--toroidal-modes", type=int, default=3)
    parser.add_argument("--sample-shape", nargs=3, type=int, metavar=("NR", "NPHI", "NZ"))
    parser.add_argument("--R-bounds", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--Z-bounds", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--vessel-file",
        type=Path,
        help=(
            "optional FLARE/Kisslinger vessel used for the fit mask and "
            "theta_ref centroid axis"
        ),
    )
    parser.add_argument(
        "--axis-core-radius",
        type=float,
        default=0.03,
        help=(
            "radius [m] excluded from fit samples around the reference axis, "
            "where grad(theta_ref) is singular (default: 0.03)"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="self-contained interactive Plotly output (.html)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="open the interactive Plotly figure after saving it",
    )
    parser.add_argument("--plot-surfaces", type=int, default=8)
    parser.add_argument("--plot-nR", type=int, default=28)
    parser.add_argument("--plot-nZ", type=int, default=28)
    args = parser.parse_args(argv)

    bfield = bfield_evaluator_from_makegrid(
        args.makegrid, currents=args.currents
    )
    vessel_mask = None
    fit_mask = None
    reference_axis = None
    if args.vessel_file is not None:
        if args.axis_core_radius <= 0.0:
            parser.error("--axis-core-radius must be positive")
        vessel_mask, reference_axis = _load_flare_vessel_geometry(
            args.vessel_file, bfield.nfp
        )

        def fit_mask(points: np.ndarray) -> np.ndarray:
            axis_R, axis_Z, _, _ = reference_axis(points[:, 1])
            outside_core = (
                np.hypot(points[:, 0] - axis_R, points[:, 2] - axis_Z)
                > args.axis_core_radius
            )
            return vessel_mask(points) & outside_core

    evaluator = scalar_potential_evaluator_from_bfield(
        bfield,
        radial_degree=args.radial_degree,
        vertical_degree=args.vertical_degree,
        toroidal_modes=args.toroidal_modes,
        sample_shape=None
        if args.sample_shape is None
        else tuple(args.sample_shape),
        R_bounds=None if args.R_bounds is None else tuple(args.R_bounds),
        Z_bounds=None if args.Z_bounds is None else tuple(args.Z_bounds),
        mask=fit_mask,
        reference_axis=reference_axis,
    )
    _print_diagnostics(evaluator.diagnostics)
    if args.plot is not None:
        plot_constant_eta_planes(
            evaluator,
            args.plot,
            surface_count=args.plot_surfaces,
            nR=args.plot_nR,
            nZ=args.plot_nZ,
            mask=vessel_mask,
            show=args.show_plot,
        )
        print(f"saved constant-eta plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
