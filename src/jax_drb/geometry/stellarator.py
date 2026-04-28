from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .fci_maps import FciMaps
from .metric_tensor import MetricTensor3D


@dataclass(frozen=True)
class SyntheticStellaratorGeometry:
    """Analytic non-axisymmetric stellarator-like geometry for validation."""

    coordinates_x: jnp.ndarray
    coordinates_y: jnp.ndarray
    coordinates_z: jnp.ndarray
    radial: jnp.ndarray
    toroidal_angle: jnp.ndarray
    poloidal_angle: jnp.ndarray
    iota: jnp.ndarray
    curvature: jnp.ndarray
    connection_length: jnp.ndarray
    metric: MetricTensor3D
    maps: FciMaps
    metadata: dict[str, float | int | str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.radial.shape)


def build_synthetic_stellarator_geometry(
    *,
    nx: int = 36,
    ny: int = 32,
    nz: int = 64,
    major_radius: float = 3.8,
    minor_radius: float = 0.7,
    elongation: float = 1.45,
    field_periods: int = 5,
    island_mode: int = 2,
    island_amplitude: float = 0.030,
    mirror_amplitude: float = 0.16,
    iota_axis: float = 0.38,
    iota_edge: float = 0.58,
) -> SyntheticStellaratorGeometry:
    """Construct a deterministic analytic 3D stellarator/SOL validation geometry."""

    s_1d = np.linspace(0.08, 1.0, nx)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    theta_1d = np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False)
    s, phi, theta = np.meshgrid(s_1d, phi_1d, theta_1d, indexing="ij")

    iota = iota_axis + (iota_edge - iota_axis) * s**1.7
    rotating_angle = theta - 0.38 * np.sin(field_periods * phi)
    radial_size = minor_radius * s
    ellipse_modulation = (
        1.0
        + 0.10 * np.cos(field_periods * phi)
        + 0.035 * s * np.cos(island_mode * theta - field_periods * phi)
    )
    vertical_modulation = 1.0 + 0.08 * np.sin(field_periods * phi + 0.4)
    r_major = major_radius + radial_size * ellipse_modulation * np.cos(rotating_angle)
    z_vertical = elongation * radial_size * vertical_modulation * np.sin(rotating_angle)
    x_cart = r_major * np.cos(phi)
    y_cart = r_major * np.sin(phi)

    metric = _metric_from_coordinates(
        x_cart,
        y_cart,
        z_vertical,
        s_1d=s_1d,
        phi_1d=phi_1d,
        theta_1d=theta_1d,
        Bxy=_magnetic_field_strength(
            s=s,
            phi=phi,
            theta=theta,
            field_periods=field_periods,
            island_mode=island_mode,
            mirror_amplitude=mirror_amplitude,
        ),
    )
    maps = _build_maps(
        s=s,
        phi=phi,
        theta=theta,
        iota=iota,
        nx=nx,
        ny=ny,
        nz=nz,
        field_periods=field_periods,
        island_mode=island_mode,
        island_amplitude=island_amplitude,
    )
    curvature = _curvature_proxy(
        s=s,
        phi=phi,
        theta=theta,
        field_periods=field_periods,
        island_mode=island_mode,
        major_radius=major_radius,
    )
    connection_length = _estimate_connection_length(maps, metric, max_steps=max(2 * ny, 16))
    return SyntheticStellaratorGeometry(
        coordinates_x=jnp.asarray(x_cart, dtype=jnp.float64),
        coordinates_y=jnp.asarray(y_cart, dtype=jnp.float64),
        coordinates_z=jnp.asarray(z_vertical, dtype=jnp.float64),
        radial=jnp.asarray(s, dtype=jnp.float64),
        toroidal_angle=jnp.asarray(phi, dtype=jnp.float64),
        poloidal_angle=jnp.asarray(theta, dtype=jnp.float64),
        iota=jnp.asarray(iota, dtype=jnp.float64),
        curvature=jnp.asarray(curvature, dtype=jnp.float64),
        connection_length=jnp.asarray(connection_length, dtype=jnp.float64),
        metric=metric,
        maps=maps,
        metadata={
            "geometry_family": "analytic_non_axisymmetric_stellarator",
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "major_radius": float(major_radius),
            "minor_radius": float(minor_radius),
            "elongation": float(elongation),
            "field_periods": int(field_periods),
            "island_mode": int(island_mode),
            "island_amplitude": float(island_amplitude),
            "mirror_amplitude": float(mirror_amplitude),
            "iota_axis": float(iota_axis),
            "iota_edge": float(iota_edge),
        },
    )


def _magnetic_field_strength(
    *,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    field_periods: int,
    island_mode: int,
    mirror_amplitude: float,
) -> np.ndarray:
    mirror = mirror_amplitude * np.cos(field_periods * phi - island_mode * theta)
    radial = 0.10 * (s - np.mean(s))
    corrugation = 0.035 * s * np.sin((field_periods + 1) * phi + theta)
    return 1.0 + mirror + radial + corrugation


def _metric_from_coordinates(
    x_cart: np.ndarray,
    y_cart: np.ndarray,
    z_cart: np.ndarray,
    *,
    s_1d: np.ndarray,
    phi_1d: np.ndarray,
    theta_1d: np.ndarray,
    Bxy: np.ndarray,
) -> MetricTensor3D:
    ds = float(s_1d[1] - s_1d[0]) if s_1d.size > 1 else 1.0
    dphi = float(phi_1d[1] - phi_1d[0]) if phi_1d.size > 1 else 2.0 * np.pi
    dtheta = float(theta_1d[1] - theta_1d[0]) if theta_1d.size > 1 else 2.0 * np.pi

    derivs = []
    for coords in (x_cart, y_cart, z_cart):
        derivs.append(np.gradient(coords, ds, dphi, dtheta, edge_order=2))

    r_s = np.stack([derivs[0][0], derivs[1][0], derivs[2][0]], axis=-1)
    r_phi = np.stack([derivs[0][1], derivs[1][1], derivs[2][1]], axis=-1)
    r_theta = np.stack([derivs[0][2], derivs[1][2], derivs[2][2]], axis=-1)
    cov = np.empty(x_cart.shape + (3, 3), dtype=np.float64)
    basis = (r_s, r_phi, r_theta)
    for i, left in enumerate(basis):
        for j, right in enumerate(basis):
            cov[..., i, j] = np.sum(left * right, axis=-1)
    determinant = np.linalg.det(cov)
    regularization = np.maximum(1.0e-12, 1.0e-11 * np.nanmax(np.abs(determinant)))
    bad = determinant <= regularization
    if np.any(bad):
        cov[bad] = cov[bad] + np.eye(3) * regularization
    contrav = np.linalg.inv(cov)
    jacobian = np.sqrt(np.maximum(np.linalg.det(cov), regularization))
    dx = np.full_like(x_cart, ds)
    dy = np.full_like(x_cart, dphi)
    dz = np.full_like(x_cart, dtheta)
    return MetricTensor3D(
        dx=jnp.asarray(dx, dtype=jnp.float64),
        dy=jnp.asarray(dy, dtype=jnp.float64),
        dz=jnp.asarray(dz, dtype=jnp.float64),
        J=jnp.asarray(jacobian, dtype=jnp.float64),
        Bxy=jnp.asarray(Bxy, dtype=jnp.float64),
        g11=jnp.asarray(contrav[..., 0, 0], dtype=jnp.float64),
        g22=jnp.asarray(contrav[..., 1, 1], dtype=jnp.float64),
        g33=jnp.asarray(contrav[..., 2, 2], dtype=jnp.float64),
        g12=jnp.asarray(contrav[..., 0, 1], dtype=jnp.float64),
        g13=jnp.asarray(contrav[..., 0, 2], dtype=jnp.float64),
        g23=jnp.asarray(contrav[..., 1, 2], dtype=jnp.float64),
        g_11=jnp.asarray(cov[..., 0, 0], dtype=jnp.float64),
        g_22=jnp.asarray(cov[..., 1, 1], dtype=jnp.float64),
        g_33=jnp.asarray(cov[..., 2, 2], dtype=jnp.float64),
        g_12=jnp.asarray(cov[..., 0, 1], dtype=jnp.float64),
        g_13=jnp.asarray(cov[..., 0, 2], dtype=jnp.float64),
        g_23=jnp.asarray(cov[..., 1, 2], dtype=jnp.float64),
    )


def _build_maps(
    *,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    iota: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    field_periods: int,
    island_mode: int,
    island_amplitude: float,
) -> FciMaps:
    dphi = 2.0 * np.pi / float(ny)
    phase = island_mode * theta - field_periods * phi
    island_envelope = np.exp(-((s - 0.74) / 0.19) ** 2)
    dx_dphi = island_amplitude * island_envelope * np.sin(phase)
    dtheta_dphi = iota + 0.07 * island_envelope * np.cos(phase)
    x_index = np.arange(nx, dtype=np.float64)[:, None, None]
    z_index = np.arange(nz, dtype=np.float64)[None, None, :]
    forward_x = x_index + dx_dphi * dphi * (nx - 1)
    backward_x = x_index - dx_dphi * dphi * (nx - 1)
    forward_z = z_index + dtheta_dphi * dphi * nz / (2.0 * np.pi)
    backward_z = z_index - dtheta_dphi * dphi * nz / (2.0 * np.pi)
    return FciMaps(
        forward_x=jnp.asarray(forward_x, dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z, dtype=jnp.float64),
        backward_x=jnp.asarray(backward_x, dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z, dtype=jnp.float64),
        forward_boundary=jnp.asarray((forward_x < 0.0) | (forward_x > nx - 1)),
        backward_boundary=jnp.asarray((backward_x < 0.0) | (backward_x > nx - 1)),
        dphi=float(dphi),
    )


def _curvature_proxy(
    *,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    field_periods: int,
    island_mode: int,
    major_radius: float,
) -> np.ndarray:
    bad_curvature = -np.cos(theta) / major_radius
    non_axisymmetric = 0.35 * np.cos(field_periods * phi - island_mode * theta) / major_radius
    radial_weight = 0.4 + 0.6 * s
    return radial_weight * (bad_curvature + non_axisymmetric)


def _estimate_connection_length(maps: FciMaps, metric: MetricTensor3D, *, max_steps: int) -> np.ndarray:
    nx, ny, nz = maps.shape
    x0 = np.arange(nx, dtype=np.float64)[:, None, None]
    z0 = np.arange(nz, dtype=np.float64)[None, None, :]
    x = np.broadcast_to(x0, (nx, ny, nz)).copy()
    z = np.broadcast_to(z0, (nx, ny, nz)).copy()
    alive = np.ones((nx, ny, nz), dtype=bool)
    steps = np.zeros((nx, ny, nz), dtype=np.float64)
    step_length = float(np.nanmean(np.sqrt(np.asarray(metric.g_22)))) * float(maps.dphi)
    forward_x = np.asarray(maps.forward_x)
    forward_z = np.asarray(maps.forward_z)
    for _ in range(max_steps):
        i = np.clip(np.rint(x).astype(np.int64), 0, nx - 1)
        k = np.mod(np.rint(z).astype(np.int64), nz)
        j = np.broadcast_to(np.arange(ny, dtype=np.int64)[None, :, None], (nx, ny, nz))
        next_x = forward_x[i, j, k]
        next_z = forward_z[i, j, k]
        still_inside = (next_x >= 0.0) & (next_x <= nx - 1.0)
        steps = np.where(alive & still_inside, steps + 1.0, steps)
        alive = alive & still_inside
        x = np.where(alive, next_x, x)
        z = np.where(alive, next_z, z)
    radial = np.broadcast_to(x0 / float(max(nx - 1, 1)), (nx, ny, nz))
    radial_shift = np.abs(forward_x - x0)
    island_lobe = np.exp(-((radial - 0.76) / 0.16) ** 2) * (1.0 + 2.5 * radial_shift)
    wall_distance = np.maximum(1.0 - radial, 0.02)
    proxy_steps = 3.0 + 42.0 * wall_distance + 16.0 * island_lobe
    bounded_steps = np.minimum(np.maximum(steps, 1.0), proxy_steps)
    return bounded_steps * step_length
