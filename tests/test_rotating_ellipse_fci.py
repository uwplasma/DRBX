"""Rotating-ellipse FCI gate.

A genuinely non-axisymmetric benchmark: the classical rotating-ellipse
(``l = 2``) torus, whose elliptical cross-section rotates with the toroidal
angle so the metric depends on all three logical coordinates. The tests pin

- that the geometry is genuinely non-axisymmetric and its autodiff metric is a
  consistent inverse pair with a positive Jacobian;
- second-order convergence of the FCI parallel gradient on this geometry, both
  the direct ``b^i d_i f`` operator and the traced-field-line operator
  ``grad_parallel_op_fci`` (the FCI-specific path); and
- that the geometry is differentiable with respect to its shape (elongation),
  which is what makes stellarator-shape optimization through the FCI stack
  possible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.geometry import (
    FciGeometry3D,
    FciMaps3D,
    build_rotating_ellipse_geometry,
    logical_grid_from_axis_vectors,
    rotating_ellipse_position,
)
from dkx.geometry.fci_geometry import build_fci_maps_from_b_contravariant
from dkx.native import LocalStencil1D, LocalStencil3D
from dkx.native.fci_operators import grad_parallel_op_direct, grad_parallel_op_fci

jax.config.update("jax_enable_x64", True)

X_MIN = 0.2
X_MAX = 1.0
ELONGATION = 0.35
N_FIELD_PERIODS = 1
IOTA = 0.9
C_PHI = 3.0
MMS_M = 2
MMS_N = 1

_GEOMETRY_KWARGS = dict(
    x_min=X_MIN,
    x_max=X_MAX,
    elongation=ELONGATION,
    n_field_periods=N_FIELD_PERIODS,
    iota=IOTA,
    c_phi=C_PHI,
)


def _mms_field(x, theta, zeta, *, derivatives=False):
    envelope = jnp.sin(jnp.pi * (x - X_MIN) / (X_MAX - X_MIN))
    poloidal = jnp.cos(MMS_M * theta)
    toroidal = jnp.sin(MMS_N * zeta)
    field = envelope * poloidal * toroidal
    if not derivatives:
        return field
    envelope_x = (jnp.pi / (X_MAX - X_MIN)) * jnp.cos(jnp.pi * (x - X_MIN) / (X_MAX - X_MIN))
    field_x = envelope_x * poloidal * toroidal
    field_theta = -MMS_M * envelope * jnp.sin(MMS_M * theta) * toroidal
    field_zeta = MMS_N * envelope * poloidal * jnp.cos(MMS_N * zeta)
    return field, field_x, field_theta, field_zeta


def _sample(x_axis, y_axis, z_axis):
    x, y, z = jnp.broadcast_arrays(
        jnp.asarray(x_axis)[:, None, None],
        jnp.asarray(y_axis)[None, :, None],
        jnp.asarray(z_axis)[None, None, :],
    )
    return _mms_field(x, y, z)


def _expected_grad_parallel(geometry: FciGeometry3D) -> jnp.ndarray:
    logical_grid = logical_grid_from_axis_vectors(
        geometry.grid.x.centers, geometry.grid.y.centers, geometry.grid.z.centers
    )
    x = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    _, field_x, field_theta, field_zeta = _mms_field(x, theta, zeta, derivatives=True)
    df = jnp.stack((field_x, field_theta, field_zeta), axis=-1)
    return jnp.einsum("...i,...i->...", geometry.cell_bfield.b_contra, df)


def _direct_coordinate_stencil(geometry: FciGeometry3D) -> LocalStencil3D:
    x_centers = geometry.grid.x.centers
    y_centers = geometry.grid.y.centers
    z_centers = geometry.grid.z.centers
    x_faces = geometry.grid.x.faces
    y_faces = geometry.grid.y.faces
    z_faces = geometry.grid.z.faces

    def _neighbors(centers, faces, periodic):
        if periodic:
            period = faces[-1] - faces[0]
            minus = jnp.concatenate((centers[-1:] - period, centers[:-1]))
            plus = jnp.concatenate((centers[1:], centers[:1] + period))
        else:
            minus = jnp.concatenate((jnp.array([2.0 * faces[0] - centers[0]]), centers[:-1]))
            plus = jnp.concatenate((centers[1:], jnp.array([2.0 * faces[-1] - centers[-1]])))
        return minus, plus

    x_minus, x_plus = _neighbors(x_centers, x_faces, False)
    y_minus, y_plus = _neighbors(y_centers, y_faces, True)
    z_minus, z_plus = _neighbors(z_centers, z_faces, True)
    shape = geometry.shape
    center = _sample(x_centers, y_centers, z_centers)

    def _axis_stencil(minus, plus, dx_min, dx_plus):
        return LocalStencil1D(
            center=center,
            minus=minus,
            plus=plus,
            dx_min=jnp.broadcast_to(dx_min, shape),
            dx_plus=jnp.broadcast_to(dx_plus, shape),
        )

    return LocalStencil3D(
        x=_axis_stencil(
            _sample(x_minus, y_centers, z_centers),
            _sample(x_plus, y_centers, z_centers),
            (x_centers - x_minus)[:, None, None],
            (x_plus - x_centers)[:, None, None],
        ),
        y=_axis_stencil(
            _sample(x_centers, y_minus, z_centers),
            _sample(x_centers, y_plus, z_centers),
            (y_centers - y_minus)[None, :, None],
            (y_plus - y_centers)[None, :, None],
        ),
        z=_axis_stencil(
            _sample(x_centers, y_centers, z_minus),
            _sample(x_centers, y_centers, z_plus),
            (z_centers - z_minus)[None, None, :],
            (z_plus - z_centers)[None, None, :],
        ),
    )


def _field_line_stencil(geometry: FciGeometry3D) -> LocalStencil1D:
    maps = geometry.maps
    center = _sample(geometry.grid.x.centers, geometry.grid.y.centers, geometry.grid.z.centers)
    forward = _mms_field(maps.forward_endpoint_x, maps.forward_endpoint_y, maps.forward_endpoint_z)
    backward = _mms_field(maps.backward_endpoint_x, maps.backward_endpoint_y, maps.backward_endpoint_z)
    return LocalStencil1D(
        center=center,
        minus=backward,
        plus=forward,
        dx_min=jnp.asarray(maps.backward_length, dtype=jnp.float64),
        dx_plus=jnp.asarray(maps.forward_length, dtype=jnp.float64),
    )


def _interior_rms(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    error = (actual - expected)[1:-1, :, :]
    return float(jnp.sqrt(jnp.mean(error**2)))


def _convergence_order(resolutions, errors) -> float:
    slope = np.polyfit(np.log(np.asarray(resolutions, dtype=np.float64)), np.log(np.asarray(errors)), 1)[0]
    return float(-slope)


def test_geometry_is_genuinely_non_axisymmetric() -> None:
    geometry = build_rotating_ellipse_geometry((16, 16, 16), **_GEOMETRY_KWARGS)

    g_cov = geometry.cell_metric.g_cov
    # The covariant metric must vary along the toroidal axis: an axisymmetric
    # geometry would be constant in zeta.
    zeta_variation = float(jnp.max(jnp.std(g_cov, axis=2)) / (jnp.mean(jnp.abs(g_cov)) + 1e-30))
    assert zeta_variation > 0.5

    # The autodiff metric is a consistent inverse pair with a positive Jacobian.
    identity = jnp.einsum("...ik,...kj->...ij", geometry.cell_metric.g_contra, g_cov)
    assert float(jnp.max(jnp.abs(identity - jnp.eye(3)))) < 1e-10
    assert float(jnp.min(geometry.cell_metric.J)) > 0.0

    # The field lies in the flux surfaces (no radial contravariant component),
    # so the tracer never leaves the domain radially.
    assert float(jnp.max(jnp.abs(geometry.cell_bfield.B_contra[..., 0]))) == 0.0


def test_grad_parallel_direct_converges_second_order() -> None:
    resolutions = [16, 24, 32]
    errors = []
    for resolution in resolutions:
        geometry = build_rotating_ellipse_geometry((resolution, resolution, resolution), **_GEOMETRY_KWARGS)
        actual = grad_parallel_op_direct(_direct_coordinate_stencil(geometry), geometry)
        errors.append(_interior_rms(actual, _expected_grad_parallel(geometry)))

    assert errors[-1] < 1e-2
    assert _convergence_order(resolutions, errors) > 1.8


def test_grad_parallel_fci_traced_converges_second_order() -> None:
    resolutions = [16, 24, 32]
    errors = []
    boundary_fractions = []
    for resolution in resolutions:
        geometry = build_rotating_ellipse_geometry(
            (resolution, resolution, resolution), construct_fci_maps=True, map_substeps=8, **_GEOMETRY_KWARGS
        )
        actual = grad_parallel_op_fci(_field_line_stencil(geometry), geometry)
        errors.append(_interior_rms(actual, _expected_grad_parallel(geometry)))
        boundary_fractions.append(float(jnp.mean(geometry.maps.forward_boundary.astype(jnp.float64))))

    # Field lines stay on flux surfaces, so no cell hits a radial boundary.
    assert max(boundary_fractions) == 0.0
    assert errors[-1] < 2e-2
    assert _convergence_order(resolutions, errors) > 1.8


def test_shape_is_differentiable_in_elongation() -> None:
    # A shape diagnostic: the mean local volume element sqrt(det g_cov) over a
    # fixed sample of logical points, as a function of the ellipse elongation.
    logical_grid = logical_grid_from_axis_vectors(
        jnp.linspace(X_MIN, X_MAX, 6),
        jnp.linspace(0.0, 2.0 * jnp.pi, 6, endpoint=False),
        jnp.linspace(0.0, 2.0 * jnp.pi, 6, endpoint=False),
    )
    points = logical_grid.reshape(-1, 3)

    def mean_volume_element(delta):
        def position(u):
            return rotating_ellipse_position(
                u[0], u[1], u[2], r0=3.0, elongation=delta, n_field_periods=N_FIELD_PERIODS
            )

        jac = jax.vmap(jax.jacfwd(position))(points)
        g_cov = jnp.einsum("pki,pkj->pij", jac, jac)
        return jnp.mean(jnp.sqrt(jnp.abs(jnp.linalg.det(g_cov))))

    delta0 = 0.35
    gradient = float(jax.grad(mean_volume_element)(delta0))
    step = 1e-5
    finite_difference = float(
        (mean_volume_element(delta0 + step) - mean_volume_element(delta0 - step)) / (2 * step)
    )
    assert gradient == pytest.approx(finite_difference, rel=1e-5, abs=1e-8)
