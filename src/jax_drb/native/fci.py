from __future__ import annotations

import jax.numpy as jnp

from ..geometry import FciGeometry3D


def interpolate_fci_plane(
    field: jnp.ndarray,
    x_prime: jnp.ndarray,
    y_prime: jnp.ndarray,
    *,
    z_offset: int,
    boundary_value: float = 0.0,
) -> jnp.ndarray:
    """Bilinearly interpolate a 3D field on traced FCI intersections."""

    values = jnp.asarray(field, dtype=jnp.float64)
    nx, ny, nz = values.shape
    z = (jnp.arange(nz)[None, None, :] + int(z_offset)) % nz
    z = jnp.broadcast_to(z, values.shape)

    x = jnp.asarray(x_prime, dtype=jnp.float64)
    y = jnp.mod(jnp.asarray(y_prime, dtype=jnp.float64), float(ny))
    valid = (x >= 0.0) & (x <= float(nx - 1))
    x_clipped = jnp.clip(x, 0.0, float(nx - 1))
    x0 = jnp.floor(x_clipped).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, nx - 1)
    y0 = jnp.floor(y).astype(jnp.int32) % ny
    y1 = (y0 + 1) % ny
    wx = x_clipped - x0.astype(jnp.float64)
    wy = y - jnp.floor(y)

    f00 = values[x0, y0, z]
    f10 = values[x1, y0, z]
    f01 = values[x0, y1, z]
    f11 = values[x1, y1, z]
    interpolated = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )
    return jnp.where(valid, interpolated, jnp.asarray(boundary_value, dtype=jnp.float64))


def fci_zup(field: jnp.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> jnp.ndarray:
    return interpolate_fci_plane(
        field,
        geometry.forward_x,
        geometry.forward_y,
        z_offset=1,
        boundary_value=boundary_value,
    )


def fci_zdown(field: jnp.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> jnp.ndarray:
    return interpolate_fci_plane(
        field,
        geometry.backward_x,
        geometry.backward_y,
        z_offset=-1,
        boundary_value=boundary_value,
    )


def grad_parallel_fci(field: jnp.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> jnp.ndarray:
    """Centered FCI parallel derivative per toroidal angle."""

    up = fci_zup(field, geometry, boundary_value=boundary_value)
    down = fci_zdown(field, geometry, boundary_value=boundary_value)
    return (up - down) / (2.0 * geometry.dz)


def laplace_parallel_fci(field: jnp.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> jnp.ndarray:
    """Second centered FCI derivative along traced field-line maps."""

    values = jnp.asarray(field, dtype=jnp.float64)
    up = fci_zup(values, geometry, boundary_value=boundary_value)
    down = fci_zdown(values, geometry, boundary_value=boundary_value)
    return (up - 2.0 * values + down) / (geometry.dz * geometry.dz)


def conservative_parallel_diffusion_fci(
    field: jnp.ndarray,
    coefficient: jnp.ndarray,
    geometry: FciGeometry3D,
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
    jac = jnp.asarray(geometry.J, dtype=jnp.float64)
    forward_valid = ~jnp.asarray(geometry.forward_boundary, dtype=bool)
    backward_valid = ~jnp.asarray(geometry.backward_boundary, dtype=bool)
    up = fci_zup(values, geometry, boundary_value=0.0)
    down = fci_zdown(values, geometry, boundary_value=0.0)
    coef_up = fci_zup(coef, geometry, boundary_value=0.0)
    coef_down = fci_zdown(coef, geometry, boundary_value=0.0)
    jac_up = fci_zup(jac, geometry, boundary_value=0.0)
    jac_down = fci_zdown(jac, geometry, boundary_value=0.0)
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
    forward_flux = forward_face * (up - values) / geometry.dz
    backward_flux = backward_face * (values - down) / geometry.dz
    return (forward_flux - backward_flux) / (jnp.maximum(jac, 1.0e-30) * geometry.dz)


def laplace_perp_xy(field: jnp.ndarray, *, dx: float, dy: float) -> jnp.ndarray:
    """Simple perpendicular benchmark Laplacian in `(x,y)` logical space."""

    values = jnp.asarray(field, dtype=jnp.float64)
    x_part = jnp.zeros_like(values)
    x_part = x_part.at[1:-1, :, :].set((values[2:, :, :] - 2.0 * values[1:-1, :, :] + values[:-2, :, :]) / (dx * dx))
    x_part = x_part.at[0, :, :].set((values[1, :, :] - values[0, :, :]) / (dx * dx))
    x_part = x_part.at[-1, :, :].set((values[-2, :, :] - values[-1, :, :]) / (dx * dx))
    y_part = (jnp.roll(values, -1, axis=1) - 2.0 * values + jnp.roll(values, 1, axis=1)) / (dy * dy)
    return x_part + y_part


def conservative_perp_diffusion_xy(
    field: jnp.ndarray,
    coefficient: jnp.ndarray,
    geometry: FciGeometry3D,
) -> jnp.ndarray:
    """Metric-weighted conservative perpendicular diffusion in logical ``x-y``.

    This first production-facing operator uses the diagonal perpendicular
    metric terms ``g11`` and ``g22`` with zero radial flux and periodic
    poloidal flux. Cross-metric terms are deliberately left
    for the next manufactured-solution gate so this operator remains symmetric,
    compact, and directly testable.
    """

    values = jnp.asarray(field, dtype=jnp.float64)
    coef = jnp.asarray(coefficient, dtype=jnp.float64)
    jac = jnp.asarray(geometry.J, dtype=jnp.float64)
    dx = jnp.asarray(geometry.dx, dtype=jnp.float64)
    dy = jnp.asarray(geometry.dy, dtype=jnp.float64)
    kx = jac * coef * jnp.asarray(geometry.g11, dtype=jnp.float64)
    ky = jac * coef * jnp.asarray(geometry.g22, dtype=jnp.float64)

    dx_face = 0.5 * (dx[1:, :, :] + dx[:-1, :, :])
    kx_face = 0.5 * (kx[1:, :, :] + kx[:-1, :, :])
    flux_x_internal = kx_face * (values[1:, :, :] - values[:-1, :, :]) / jnp.maximum(dx_face, 1.0e-30)
    flux_x_plus = jnp.zeros_like(values).at[:-1, :, :].set(flux_x_internal)
    flux_x_minus = jnp.zeros_like(values).at[1:, :, :].set(flux_x_internal)
    div_x = (flux_x_plus - flux_x_minus) / jnp.maximum(dx, 1.0e-30)

    values_plus_y = jnp.roll(values, -1, axis=1)
    ky_plus = jnp.roll(ky, -1, axis=1)
    dy_plus = 0.5 * (dy + jnp.roll(dy, -1, axis=1))
    ky_face = 0.5 * (ky + ky_plus)
    flux_y_plus = ky_face * (values_plus_y - values) / jnp.maximum(dy_plus, 1.0e-30)
    flux_y_minus = jnp.roll(flux_y_plus, 1, axis=1)
    div_y = (flux_y_plus - flux_y_minus) / jnp.maximum(dy, 1.0e-30)
    return (div_x + div_y) / jnp.maximum(jac, 1.0e-30)


def logical_exb_bracket_xy(
    potential: jnp.ndarray,
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_x: bool = False,
    periodic_y: bool = True,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the logical perpendicular ``E x B`` bracket ``{phi, f}``.

    This is the geometry-facing nonlinear advection seam for reduced
    non-axisymmetric tests. In the logical radial/poloidal plane it evaluates

    ``(d_y phi d_x f - d_x phi d_y f) / B``

    with the grid spacing stored in ``metric``. The toroidal/FCI direction is
    not differentiated here; field-line coupling enters through the FCI
    operators and the potential/vorticity solve.
    """

    phi = jnp.asarray(potential, dtype=jnp.float64)
    values = jnp.asarray(field, dtype=jnp.float64)
    metric = geometry
    dphi_dx = _first_derivative_3d(phi, metric.dx, axis=0, periodic=periodic_x)
    dphi_dy = _first_derivative_3d(phi, metric.dy, axis=1, periodic=periodic_y)
    df_dx = _first_derivative_3d(values, metric.dx, axis=0, periodic=periodic_x)
    df_dy = _first_derivative_3d(values, metric.dy, axis=1, periodic=periodic_y)
    return (dphi_dy * df_dx - dphi_dx * df_dy) / jnp.maximum(
        jnp.asarray(geometry.Bmag, dtype=jnp.float64),
        float(b_floor),
    )


def metric_weighted_scalar_laplacian_3d(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
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
    metric = geometry
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
    """Centered first derivative with periodic or second-order edge treatment."""

    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    centered = (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / jnp.maximum(2.0 * h, 1.0e-30)
    if periodic:
        return centered

    first = _axis_index(axis, 0)
    second = _axis_index(axis, 1)
    third = _axis_index(axis, 2)
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    antepenultimate = _axis_index(axis, -3)
    forward = (-3.0 * values[first] + 4.0 * values[second] - values[third]) / jnp.maximum(2.0 * h[first], 1.0e-30)
    backward = (3.0 * values[last] - 4.0 * values[penultimate] + values[antepenultimate]) / jnp.maximum(2.0 * h[last], 1.0e-30)
    return centered.at[first].set(forward).at[last].set(backward)


def _axis_index(axis: int, index: int) -> tuple[object, object, object]:
    slices: list[object] = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    return tuple(slices)
