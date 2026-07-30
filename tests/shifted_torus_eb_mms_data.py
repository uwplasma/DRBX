"""Analytic shifted-torus EB MMS data used by sharded regressions.

This module intentionally contains no global EB RHS, boundary-condition, or
RHS-result objects.  ``FciGeometry3D`` is only the host-side staging geometry;
the sharded test builds and executes its local operators independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from drbx.geometry import (
    BFieldGeometry,
    FciGeometry3D,
    FciMaps3D,
    FaceBFieldGeometry,
    Spacing3D,
    build_curvature_coefficients,
    logical_grid_from_axis_vectors,
)
from drbx.native import FciDrbEBRhsParameters, FciDrbEBState

from shifted_torus_4field_mms_helpers import (
    alpha_value,
    build_shifted_torus_4field_geometry,
    c_phi,
    iota,
    r0,
    sigma,
    x_max,
    x_min,
)


PERIODIC_AXES = (False, True, True)
N0 = TE0 = TI0 = 1.0
EPS_N = EPS_TE = EPS_TI = 1.0e-2
EPS_PHI = EPS_VE = EPS_VI = 1.0e-3
W_N, W_TE, W_TI, W_PHI, W_VE, W_VI = 0.7, 0.9, 1.1, 0.5, 0.8, 1.2


@dataclass(frozen=True)
class ShiftedTorusEbMmsContext:
    geometry: FciGeometry3D
    parameters: FciDrbEBRhsParameters
    curvature_coefficients: jnp.ndarray


@dataclass(frozen=True)
class AnalyticMmsData:
    density: jnp.ndarray
    phi: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray
    Vi: jnp.ndarray
    Ve: jnp.ndarray
    vorticity: jnp.ndarray
    density_t: jnp.ndarray
    phi_t: jnp.ndarray
    Te_t: jnp.ndarray
    Ti_t: jnp.ndarray
    Vi_t: jnp.ndarray
    Ve_t: jnp.ndarray
    vorticity_t: jnp.ndarray
    density_grad: jnp.ndarray
    phi_grad: jnp.ndarray
    Te_grad: jnp.ndarray
    Ti_grad: jnp.ndarray
    Vi_grad: jnp.ndarray
    Ve_grad: jnp.ndarray
    vorticity_grad: jnp.ndarray


def _bmag(B: jnp.ndarray, g_cov: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.einsum("...i,...ij,...j->...", B, g_cov, B), 0.0))


def build_shifted_torus_eb_mms_geometry(shape: tuple[int, int, int]) -> FciGeometry3D:
    base = build_shifted_torus_4field_geometry(shape, construct_fci_maps=False)
    grid = base.grid

    def zero_radial(bfield: BFieldGeometry, metric) -> BFieldGeometry:
        B = jnp.asarray(bfield.B_contra, dtype=jnp.float64)
        B = jnp.stack((jnp.zeros_like(B[..., 0]), B[..., 1], B[..., 2]), axis=-1)
        return BFieldGeometry(B_contra=B, Bmag=_bmag(B, metric.g_cov))

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    maps = FciMaps3D(
        forward_x=zeros, forward_y=zeros, backward_x=zeros, backward_y=zeros,
        forward_endpoint_x=zeros, forward_endpoint_y=zeros, forward_endpoint_z=zeros,
        backward_endpoint_x=zeros, backward_endpoint_y=zeros, backward_endpoint_z=zeros,
        forward_length=jnp.ones(shape), backward_length=jnp.ones(shape),
        forward_boundary=zeros.astype(bool), backward_boundary=zeros.astype(bool),
    )
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
    )
    return FciGeometry3D(
        grid=grid, maps=maps, spacing=spacing, cell_metric=base.cell_metric,
        face_metric=base.face_metric,
        cell_bfield=zero_radial(base.cell_bfield, base.cell_metric),
        face_bfield=FaceBFieldGeometry(
            x=zero_radial(base.face_bfield.x, base.face_metric.x),
            y=zero_radial(base.face_bfield.y, base.face_metric.y),
            z=zero_radial(base.face_bfield.z, base.face_metric.z),
        ),
    )


def build_shifted_torus_eb_mms_context(shape: tuple[int, int, int]) -> ShiftedTorusEbMmsContext:
    geometry = build_shifted_torus_eb_mms_geometry(shape)
    parameters = FciDrbEBRhsParameters(
        n0=1.0, Te0=1.0, Ti0=1.0, tau=1.0, mi_over_me=100.0, rho_star=1.0,
        density_D_perp=0.0, density_D_parallel=0.0,
        electron_temperature_chi_parallel=0.0, electron_temperature_D_perp=0.0,
        ion_temperature_chi_parallel=0.0, ion_temperature_D_perp=0.0,
        Ve_nu=1.0e-3, Ve_D_perp=0.0,
        Vi_D_perp=0.0, vorticity_D_perp=0.0,
        vorticity_D_parallel=0.0,
    )
    return ShiftedTorusEbMmsContext(
        geometry=geometry, parameters=parameters,
        curvature_coefficients=build_curvature_coefficients(geometry, periodic_axes=PERIODIC_AXES),
    )


def _coordinates(geometry: FciGeometry3D):
    grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    rho, theta, zeta = [jnp.asarray(grid[..., i], dtype=jnp.float64) for i in range(3)]
    rho_min, rho_max = float(geometry.grid.x.faces[0]), float(geometry.grid.x.faces[-1])
    theta_s = theta + float(sigma) * (rho - 0.5 * (rho_min + rho_max))
    H = jnp.sin(jnp.pi * (rho - rho_min) / (rho_max - rho_min)) ** 6
    return H, theta_s, zeta


def _scalar_geometry(coord: jnp.ndarray):
    rho, theta, _ = coord
    theta_s = theta + float(sigma) * (rho - 0.5 * (float(x_min) + float(x_max)))
    # The shifted-torus test geometry uses the same fixed radial interval as
    # the host grid; recover its metric through the analytic coordinate map.
    cos_s, sin_s = jnp.cos(theta_s), jnp.sin(theta_s)
    R = float(r0) + float(alpha_value) * rho + rho * cos_s
    Q = 1.0 + float(alpha_value) * cos_s
    J = jnp.maximum(rho * R * Q, 1.0e-14)
    g_cov = jnp.array([[1 + 2 * float(alpha_value) * cos_s + float(alpha_value) ** 2,
                        -float(alpha_value) * rho * sin_s, 0.0],
                       [-float(alpha_value) * rho * sin_s, rho ** 2, 0.0],
                       [0.0, 0.0, R ** 2]])
    g_contra = jnp.linalg.inv(g_cov)
    B = jnp.array([0.0, float(iota) * float(c_phi) / J, float(c_phi) / J])
    Bmag = _bmag(B, g_cov)
    return J, g_cov, g_contra, B / jnp.maximum(Bmag, 1.0e-30), Bmag


def _field_functions(coord, time, rho_min, rho_max):
    rho, theta, zeta = coord
    theta_s = theta + float(sigma) * (rho - 0.5 * (rho_min + rho_max))
    H = jnp.sin(jnp.pi * (rho - rho_min) / (rho_max - rho_min)) ** 6
    t = jnp.asarray(time, dtype=jnp.float64)
    return (
        N0 + EPS_N * H * (0.70 * jnp.cos(2 * theta_s + 3 * zeta + W_N * t) + 0.30 * jnp.sin(3 * theta_s - 2 * zeta + W_N * t)),
        EPS_PHI * H * (jnp.cos(2 * theta_s) * jnp.sin(3 * zeta + W_PHI * t) + 0.50 * jnp.sin(3 * theta_s + 2 * zeta + W_PHI * t) + 0.25 * jnp.cos(4 * theta_s - zeta + W_PHI * t)),
        TE0 + EPS_TE * H * (0.60 * jnp.sin(theta_s + 2 * zeta + W_TE * t) + 0.25 * jnp.cos(4 * theta_s - zeta + W_TE * t)),
        TI0 + EPS_TI * H * (0.50 * jnp.cos(3 * theta_s - zeta + W_TI * t) + 0.25 * jnp.sin(2 * theta_s + 4 * zeta + W_TI * t)),
        EPS_VI * H * (jnp.cos(theta_s + 2 * zeta + W_VI * t) + 0.30 * jnp.sin(3 * theta_s - zeta + W_VI * t)),
        EPS_VE * H * (jnp.sin(theta_s - 3 * zeta + W_VE * t) + 0.30 * jnp.cos(2 * theta_s + zeta + W_VE * t)),
    )


def _vorticity(coord, time, rho_min, rho_max):
    def laplacian(field):
        def flux(point):
            J, _, g, b, _ = _scalar_geometry(point)
            grad = jax.grad(lambda q: field(q))(point)
            return J * (g - jnp.outer(b, b)) @ grad
        J, _, _, _, _ = _scalar_geometry(coord)
        return jnp.trace(jax.jacfwd(flux)(coord)) / J
    return laplacian(lambda q: _field_functions(q, time, rho_min, rho_max)[1]) + laplacian(lambda q: _field_functions(q, time, rho_min, rho_max)[3])


def _data_at(context: ShiftedTorusEbMmsContext, time: float) -> AnalyticMmsData:
    geometry = context.geometry
    coords = jnp.asarray(logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors), dtype=jnp.float64).reshape((-1, 3))
    rho_min, rho_max = float(geometry.grid.x.faces[0]), float(geometry.grid.x.faces[-1])
    t = jnp.asarray(time, dtype=jnp.float64)

    def one_field(index, point, local_time):
        if index < 6:
            return _field_functions(point, local_time, rho_min, rho_max)[index]
        return _vorticity(point, local_time, rho_min, rho_max)

    values, grads, time_grads = [], [], []
    times = jnp.full((coords.shape[0],), t)
    for index in range(7):
        value_grad = jax.vmap(
            jax.value_and_grad(lambda point, local_time: one_field(index, point, local_time), argnums=(0, 1))
        )
        value, (gradient, time_gradient) = value_grad(coords, times)
        values.append(value)
        grads.append(gradient)
        time_grads.append(time_gradient)
    values = jnp.stack(values, axis=-1)
    grads = jnp.stack(grads, axis=-2)
    time_grads = jnp.stack(time_grads, axis=-1)
    shape = geometry.shape
    v = values.reshape(shape + (7,))
    g = grads.reshape(shape + (7, 3))
    dt = time_grads.reshape(shape + (7,))
    return AnalyticMmsData(
        density=v[..., 0], phi=v[..., 1], Te=v[..., 2], Ti=v[..., 3], Vi=v[..., 4], Ve=v[..., 5], vorticity=v[..., 6],
        density_t=dt[..., 0], phi_t=dt[..., 1], Te_t=dt[..., 2], Ti_t=dt[..., 3], Vi_t=dt[..., 4], Ve_t=dt[..., 5], vorticity_t=dt[..., 6],
        density_grad=g[..., 0, :], phi_grad=g[..., 1, :], Te_grad=g[..., 2, :], Ti_grad=g[..., 3, :], Vi_grad=g[..., 4, :], Ve_grad=g[..., 5, :], vorticity_grad=g[..., 6, :],
    )


def _mms_exact_state(context, time):
    d = _data_at(context, time)
    return FciDrbEBState(d.density, d.phi, d.Te, d.Ti, d.Vi, d.Ve, d.vorticity)


def _analytic_eb_rhs_from_data(data: AnalyticMmsData, context: ShiftedTorusEbMmsContext):
    geometry, p = context.geometry, context.parameters
    bmag = jnp.maximum(geometry.cell_bfield.Bmag, 1.0e-30)
    b = geometry.cell_bfield.b_contra
    bcov = jnp.einsum("...ij,...j->...i", geometry.cell_metric.g_cov, b)
    J = jnp.maximum(geometry.cell_metric.J, 1.0e-30)
    poisson = lambda a, c: jnp.sum(bcov * jnp.cross(a, c), axis=-1) / J
    parallel = lambda a: jnp.einsum("...i,...i->...", b, a)
    curvature = lambda a: jnp.einsum("...i,...i->...", context.curvature_coefficients, a)
    n = jnp.maximum(data.density, 1.0e-30)
    tau, mi, nu, rho = p.tau, p.mi_over_me, p.Ve_nu, p.rho_star
    Pe, Pi = data.density * data.Te, data.density * data.Ti
    pressure = Pe + tau * Pi
    current = data.density * (data.Vi - data.Ve)
    Peg = data.Te[..., None] * data.density_grad + data.density[..., None] * data.Te_grad
    Pig = data.Ti[..., None] * data.density_grad + data.density[..., None] * data.Ti_grad
    pg = Peg + tau * Pig
    cg = (data.Vi - data.Ve)[..., None] * data.density_grad + data.density[..., None] * (data.Vi_grad - data.Ve_grad)
    dfg = data.Ve[..., None] * data.density_grad + data.density[..., None] * data.Ve_grad
    cp, cpress, cphi = curvature(Peg), curvature(pg), curvature(data.phi_grad)
    cte, cti = curvature(data.Te_grad), curvature(data.Ti_grad)
    density = -poisson(data.phi_grad, data.density_grad) / (rho * bmag) - parallel(dfg) + 2 * (cp - data.density * cphi) / bmag
    Te = -poisson(data.phi_grad, data.Te_grad) / (rho * bmag) - data.Ve * parallel(data.Te_grad) + 4 * data.Te / (3 * bmag) * (cp / n + 2.5 * cte - cphi) + 2 * data.Te / (3 * n) * (0.71 * parallel(cg) - data.density * parallel(data.Ve_grad))
    Ti = -poisson(data.phi_grad, data.Ti_grad) / (rho * bmag) - data.Vi * parallel(data.Ti_grad) + 4 * data.Ti / (3 * bmag) * (cp / n - 2.5 * tau * cti - cphi) + 2 * data.Ti / (3 * n) * (parallel(cg) - data.density * parallel(data.Vi_grad))
    Vi = -poisson(data.phi_grad, data.Vi_grad) / (rho * bmag) - data.Vi * parallel(data.Vi_grad) - parallel(pg) / n
    Ve = -poisson(data.phi_grad, data.Ve_grad) / (rho * bmag) - data.Ve * parallel(data.Ve_grad) + mi * (nu * current + parallel(data.phi_grad) - parallel(Peg) / n - 0.71 * parallel(data.Te_grad))
    omega = -poisson(data.phi_grad, data.vorticity_grad) / (rho * bmag) - data.Vi * parallel(data.vorticity_grad) + bmag**2 * parallel(cg) / n + 2 * bmag * cpress / n
    return FciDrbEBState(density, data.phi, Te, Ti, Vi, Ve, omega)


__all__ = ["AnalyticMmsData", "ShiftedTorusEbMmsContext", "build_shifted_torus_eb_mms_context", "build_shifted_torus_eb_mms_geometry", "_analytic_eb_rhs_from_data", "_data_at", "_mms_exact_state"]
