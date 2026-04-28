from __future__ import annotations

import jax.numpy as jnp

from ..geometry import FciMaps, MetricTensor3D


def interpolate_fci_plane(
    field: jnp.ndarray,
    x_prime: jnp.ndarray,
    z_prime: jnp.ndarray,
    *,
    y_offset: int,
    boundary_value: float = 0.0,
) -> jnp.ndarray:
    """Bilinearly interpolate a 3D field on traced FCI intersections."""

    values = jnp.asarray(field, dtype=jnp.float64)
    nx, ny, nz = values.shape
    y = (jnp.arange(ny)[None, :, None] + int(y_offset)) % ny
    y = jnp.broadcast_to(y, values.shape)

    x = jnp.asarray(x_prime, dtype=jnp.float64)
    z = jnp.mod(jnp.asarray(z_prime, dtype=jnp.float64), float(nz))
    valid = (x >= 0.0) & (x <= float(nx - 1))
    x_clipped = jnp.clip(x, 0.0, float(nx - 1))
    x0 = jnp.floor(x_clipped).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, nx - 1)
    z0 = jnp.floor(z).astype(jnp.int32) % nz
    z1 = (z0 + 1) % nz
    wx = x_clipped - x0.astype(jnp.float64)
    wz = z - jnp.floor(z)

    f00 = values[x0, y, z0]
    f10 = values[x1, y, z0]
    f01 = values[x0, y, z1]
    f11 = values[x1, y, z1]
    interpolated = (
        (1.0 - wx) * (1.0 - wz) * f00
        + wx * (1.0 - wz) * f10
        + (1.0 - wx) * wz * f01
        + wx * wz * f11
    )
    return jnp.where(valid, interpolated, jnp.asarray(boundary_value, dtype=jnp.float64))


def fci_yup(field: jnp.ndarray, maps: FciMaps, *, boundary_value: float = 0.0) -> jnp.ndarray:
    return interpolate_fci_plane(
        field,
        maps.forward_x,
        maps.forward_z,
        y_offset=1,
        boundary_value=boundary_value,
    )


def fci_ydown(field: jnp.ndarray, maps: FciMaps, *, boundary_value: float = 0.0) -> jnp.ndarray:
    return interpolate_fci_plane(
        field,
        maps.backward_x,
        maps.backward_z,
        y_offset=-1,
        boundary_value=boundary_value,
    )


def grad_parallel_fci(field: jnp.ndarray, maps: FciMaps, *, boundary_value: float = 0.0) -> jnp.ndarray:
    """Centered FCI parallel derivative per toroidal angle."""

    up = fci_yup(field, maps, boundary_value=boundary_value)
    down = fci_ydown(field, maps, boundary_value=boundary_value)
    return (up - down) / (2.0 * maps.dphi)


def laplace_parallel_fci(field: jnp.ndarray, maps: FciMaps, *, boundary_value: float = 0.0) -> jnp.ndarray:
    """Second centered FCI derivative along traced field-line maps."""

    values = jnp.asarray(field, dtype=jnp.float64)
    up = fci_yup(values, maps, boundary_value=boundary_value)
    down = fci_ydown(values, maps, boundary_value=boundary_value)
    return (up - 2.0 * values + down) / (maps.dphi * maps.dphi)


def conservative_parallel_diffusion_fci(
    field: jnp.ndarray,
    coefficient: jnp.ndarray,
    maps: FciMaps,
    *,
    jacobian: jnp.ndarray | float = 1.0,
    boundary_mode: str = "zero_flux",
) -> jnp.ndarray:
    """Return ``J^-1 div_parallel(J K grad_parallel(f))`` on FCI maps.

    The stencil computes mapped forward/backward face gradients, arithmetic
    face averages for ``J`` and ``K``, and a conservative difference of the two
    mapped fluxes. ``boundary_mode='zero_flux'`` turns map exits into vanishing
    target-normal diffusive fluxes, which is the stable manufactured-solution
    and neutral-diffusion gate used before wall-distance target losses are
    promoted.
    """

    values = jnp.asarray(field, dtype=jnp.float64)
    coef = jnp.asarray(coefficient, dtype=jnp.float64)
    jac = jnp.asarray(jacobian, dtype=jnp.float64)
    if jac.ndim == 0:
        jac = jnp.ones_like(values) * jac
    forward_valid = ~jnp.asarray(maps.forward_boundary, dtype=bool)
    backward_valid = ~jnp.asarray(maps.backward_boundary, dtype=bool)
    up = fci_yup(values, maps, boundary_value=0.0)
    down = fci_ydown(values, maps, boundary_value=0.0)
    coef_up = fci_yup(coef, maps, boundary_value=0.0)
    coef_down = fci_ydown(coef, maps, boundary_value=0.0)
    jac_up = fci_yup(jac, maps, boundary_value=0.0)
    jac_down = fci_ydown(jac, maps, boundary_value=0.0)
    if boundary_mode == "zero_flux":
        up = jnp.where(forward_valid, up, values)
        down = jnp.where(backward_valid, down, values)
        coef_up = jnp.where(forward_valid, coef_up, coef)
        coef_down = jnp.where(backward_valid, coef_down, coef)
        jac_up = jnp.where(forward_valid, jac_up, jac)
        jac_down = jnp.where(backward_valid, jac_down, jac)
    elif boundary_mode != "dirichlet_zero":
        raise ValueError(f"Unsupported FCI conservative boundary_mode={boundary_mode!r}")
    forward_face = 0.25 * (jac + jac_up) * (coef + coef_up)
    backward_face = 0.25 * (jac + jac_down) * (coef + coef_down)
    forward_flux = forward_face * (up - values) / maps.dphi
    backward_flux = backward_face * (values - down) / maps.dphi
    return (forward_flux - backward_flux) / (jnp.maximum(jac, 1.0e-30) * maps.dphi)


def laplace_perp_xz(field: jnp.ndarray, *, dx: float, dz: float) -> jnp.ndarray:
    """Simple perpendicular benchmark Laplacian in `(x,z)` logical space."""

    values = jnp.asarray(field, dtype=jnp.float64)
    x_part = jnp.zeros_like(values)
    x_part = x_part.at[1:-1, :, :].set((values[2:, :, :] - 2.0 * values[1:-1, :, :] + values[:-2, :, :]) / (dx * dx))
    x_part = x_part.at[0, :, :].set((values[1, :, :] - values[0, :, :]) / (dx * dx))
    x_part = x_part.at[-1, :, :].set((values[-2, :, :] - values[-1, :, :]) / (dx * dx))
    z_part = (jnp.roll(values, -1, axis=2) - 2.0 * values + jnp.roll(values, 1, axis=2)) / (dz * dz)
    return x_part + z_part


def conservative_perp_diffusion_xz(
    field: jnp.ndarray,
    coefficient: jnp.ndarray,
    metric: MetricTensor3D,
) -> jnp.ndarray:
    """Metric-weighted conservative perpendicular diffusion in logical ``x-z``.

    This first production-facing operator uses the diagonal perpendicular
    metric terms ``g11`` and ``g33`` with zero radial flux and periodic
    poloidal/toroidal-binormal flux. Cross-metric terms are deliberately left
    for the next manufactured-solution gate so this operator remains symmetric,
    compact, and directly testable.
    """

    values = jnp.asarray(field, dtype=jnp.float64)
    coef = jnp.asarray(coefficient, dtype=jnp.float64)
    jac = jnp.asarray(metric.J, dtype=jnp.float64)
    dx = jnp.asarray(metric.dx, dtype=jnp.float64)
    dz = jnp.asarray(metric.dz, dtype=jnp.float64)
    kx = jac * coef * jnp.asarray(metric.g11, dtype=jnp.float64)
    kz = jac * coef * jnp.asarray(metric.g33, dtype=jnp.float64)

    dx_face = 0.5 * (dx[1:, :, :] + dx[:-1, :, :])
    kx_face = 0.5 * (kx[1:, :, :] + kx[:-1, :, :])
    flux_x_internal = kx_face * (values[1:, :, :] - values[:-1, :, :]) / jnp.maximum(dx_face, 1.0e-30)
    flux_x_plus = jnp.zeros_like(values).at[:-1, :, :].set(flux_x_internal)
    flux_x_minus = jnp.zeros_like(values).at[1:, :, :].set(flux_x_internal)
    div_x = (flux_x_plus - flux_x_minus) / jnp.maximum(dx, 1.0e-30)

    values_plus_z = jnp.roll(values, -1, axis=2)
    kz_plus = jnp.roll(kz, -1, axis=2)
    dz_plus = 0.5 * (dz + jnp.roll(dz, -1, axis=2))
    kz_face = 0.5 * (kz + kz_plus)
    flux_z_plus = kz_face * (values_plus_z - values) / jnp.maximum(dz_plus, 1.0e-30)
    flux_z_minus = jnp.roll(flux_z_plus, 1, axis=2)
    div_z = (flux_z_plus - flux_z_minus) / jnp.maximum(dz, 1.0e-30)
    return (div_x + div_z) / jnp.maximum(jac, 1.0e-30)


def metric_weighted_scalar_laplacian_3d(
    field: jnp.ndarray,
    metric: MetricTensor3D,
    coefficient: jnp.ndarray | float = 1.0,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> jnp.ndarray:
    """Return ``J^-1 d_i(J K g^ij d_j f)`` on a logical 3D grid.

    This is the full contravariant-metric scalar diffusion form used by the
    non-axisymmetric manufactured-solution gate. It includes all cross-metric
    terms and is written only with JAX array operations, so it can be used under
    ``jit``, ``vmap``, ``jvp``, and ``linearize``. The default boundary choice
    treats the radial direction as open/one-sided and the field-line/toroidal
    directions as periodic; analytic verification tests can request fully
    periodic axes.
    """

    values = jnp.asarray(field, dtype=jnp.float64)
    coef = jnp.asarray(coefficient, dtype=jnp.float64)
    if coef.ndim == 0:
        coef = jnp.ones_like(values) * coef
    jac = jnp.asarray(metric.J, dtype=jnp.float64)
    weighted = jac * coef
    df_dx = _first_derivative_3d(values, metric.dx, axis=0, periodic=periodic_axes[0])
    df_dy = _first_derivative_3d(values, metric.dy, axis=1, periodic=periodic_axes[1])
    df_dz = _first_derivative_3d(values, metric.dz, axis=2, periodic=periodic_axes[2])

    flux_x = weighted * (
        jnp.asarray(metric.g11, dtype=jnp.float64) * df_dx
        + jnp.asarray(metric.g12, dtype=jnp.float64) * df_dy
        + jnp.asarray(metric.g13, dtype=jnp.float64) * df_dz
    )
    flux_y = weighted * (
        jnp.asarray(metric.g12, dtype=jnp.float64) * df_dx
        + jnp.asarray(metric.g22, dtype=jnp.float64) * df_dy
        + jnp.asarray(metric.g23, dtype=jnp.float64) * df_dz
    )
    flux_z = weighted * (
        jnp.asarray(metric.g13, dtype=jnp.float64) * df_dx
        + jnp.asarray(metric.g23, dtype=jnp.float64) * df_dy
        + jnp.asarray(metric.g33, dtype=jnp.float64) * df_dz
    )

    divergence = (
        _first_derivative_3d(flux_x, metric.dx, axis=0, periodic=periodic_axes[0])
        + _first_derivative_3d(flux_y, metric.dy, axis=1, periodic=periodic_axes[1])
        + _first_derivative_3d(flux_z, metric.dz, axis=2, periodic=periodic_axes[2])
    )
    return divergence / jnp.maximum(jac, 1.0e-30)


def _first_derivative_3d(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Centered first derivative with periodic or one-sided edge treatment."""

    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    centered = (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / jnp.maximum(2.0 * h, 1.0e-30)
    if periodic:
        return centered

    first = _axis_index(axis, 0)
    second = _axis_index(axis, 1)
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    forward = (values[second] - values[first]) / jnp.maximum(h[first], 1.0e-30)
    backward = (values[last] - values[penultimate]) / jnp.maximum(h[last], 1.0e-30)
    return centered.at[first].set(forward).at[last].set(backward)


def _axis_index(axis: int, index: int) -> tuple[object, object, object]:
    slices: list[object] = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    return tuple(slices)
