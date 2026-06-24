from __future__ import annotations

import jax.numpy as jnp

from jax_drb.geometry import FciGeometry3D, build_fci_maps_from_b_contravariant, logical_grid_from_axis_vectors
from jax_drb.native.fci_operators import (
    curvature_op,
    debug_only_grad_parallel_op,
    grad_parallel_op,
    grad_perp_op,
    perp_laplacian_op,
    poisson_bracket_op,
)


# Global MMS parameters.
A_phi = 0.1
A_n = 0.1
A_e = 0.1
A_i = 0.08
a1 = 0.2
a2 = 0.15
a3 = 0.1
a4 = 0.12
Omega = 2.0 * jnp.pi
M_phi = 2
N_phi = 3
M_n = 3
N_n = 2
M_e = 4
N_e = 3
M_i = 2
N_i = 4
n0 = 1.0
rho_star = 1.0
Te = 1.0
mi_over_me = 1836.0
sigma = 0.75
r0 = 3.0
alpha_value = 0.25
iota = 1.1
c_phi = 3.0
x_min = 0.15
x_max = 1.0
tf = 0.1
num_steps = 100


def build_shifted_torus_4field_geometry(
    shape: tuple[int, int, int],
    *,
    x_min: float = x_min,
    x_max: float = x_max,
    r0: float = r0,
    alpha_value: float = alpha_value,
    iota: float = iota,
    c_phi: float = c_phi,
    sigma: float = sigma,
    construct_fci_maps: bool = False,
    B_contravariant: jnp.ndarray | None = None,
) -> FciGeometry3D:
    """Build the shifted-torus geometry used by the 2-field MMS scaffold.

    This matches the geometry construction in `test_mms_shifted_torus_2_field.py`
    so the 4-field MMS test can reuse the same FCI maps, metric, and logical grid
    conventions.
    """

    nx, ny, nz = shape
    target_shape = (nx, ny, nz)

    x_1d = jnp.linspace(float(x_min), float(x_max), nx, dtype=jnp.float64)
    theta_1d = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False, dtype=jnp.float64)
    zeta_1d = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False, dtype=jnp.float64)
    logical_grid = logical_grid_from_axis_vectors(x_1d, theta_1d, zeta_1d)

    x = jnp.broadcast_to(x_1d[:, None, None], target_shape)
    theta = jnp.broadcast_to(theta_1d[None, :, None], target_shape)
    zeta = jnp.broadcast_to(zeta_1d[None, None, :], target_shape)

    x_span = float(x_max) - float(x_min)
    if x_span <= 0.0:
        raise ValueError("x_max must be larger than x_min")
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)

    cos_theta = jnp.cos(theta_shift)
    sin_theta = jnp.sin(theta_shift)
    radial_factor = x

    R = float(r0) + float(alpha_value) * radial_factor + radial_factor * cos_theta
    jacobian = R * radial_factor * (1.0 + float(alpha_value) * cos_theta)
    jacobian = jnp.where(jnp.abs(jacobian) < 1.0e-14, 1.0e-14, jacobian)

    g11 = 1.0 / (1.0 + float(alpha_value) * cos_theta) ** 2
    g12 = float(alpha_value) * sin_theta / (radial_factor * (1.0 + float(alpha_value) * cos_theta) ** 2)
    g13 = jnp.zeros(target_shape, dtype=jnp.float64)
    g22 = (1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2) / (
        radial_factor**2 * (1.0 + float(alpha_value) * cos_theta) ** 2
    )
    g23 = jnp.zeros(target_shape, dtype=jnp.float64)
    g33 = 1.0 / (R**2)

    g_11 = 1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2
    g_12 = -float(alpha_value) * radial_factor * sin_theta
    g_13 = jnp.zeros(target_shape, dtype=jnp.float64)
    g_22 = radial_factor**2
    g_23 = jnp.zeros(target_shape, dtype=jnp.float64)
    g_33 = R**2

    expected_bmag = jnp.sqrt((float(iota) ** 2) * radial_factor**2 + R**2) * float(c_phi) / jacobian

    if B_contravariant is None:
        B_contravariant = jnp.stack(
            (
                jnp.zeros(target_shape, dtype=jnp.float64),
                float(iota) * float(c_phi) / jacobian,
                float(c_phi) / jacobian,
            ),
            axis=-1,
        )
    else:
        B_contravariant = jnp.asarray(B_contravariant, dtype=jnp.float64)

    if construct_fci_maps:
        map_fields = build_fci_maps_from_b_contravariant(
            logical_grid=logical_grid,
            B_contravariant=B_contravariant,
            Bmag=expected_bmag,
        )
    else:
        ones = jnp.ones(target_shape, dtype=jnp.float64)
        zeros = jnp.zeros(target_shape, dtype=jnp.float64)
        map_fields = {
            "forward_x": zeros,
            "forward_y": zeros,
            "backward_x": zeros,
            "backward_y": zeros,
            "forward_length": ones,
            "backward_length": ones,
            "forward_boundary": zeros.astype(bool),
            "backward_boundary": zeros.astype(bool),
            "dz": ones * (2.0 * jnp.pi) / float(max(nz, 1)),
        }

    return FciGeometry3D(
        logical_grid=logical_grid,
        forward_x=map_fields["forward_x"],
        forward_y=map_fields["forward_y"],
        backward_x=map_fields["backward_x"],
        backward_y=map_fields["backward_y"],
        forward_length=map_fields["forward_length"],
        backward_length=map_fields["backward_length"],
        forward_boundary=map_fields["forward_boundary"],
        backward_boundary=map_fields["backward_boundary"],
        dx=jnp.ones(target_shape, dtype=jnp.float64) * x_span / float(max(nx - 1, 1)),
        dy=jnp.ones(target_shape, dtype=jnp.float64) * (2.0 * jnp.pi) / float(max(ny, 1)),
        dz=map_fields["dz"],
        J=jacobian,
        B_contravariant=B_contravariant,
        g11=g11,
        g22=g22,
        g33=g33,
        g12=g12,
        g13=g13,
        g23=g23,
        g_11=g_11,
        g_22=g_22,
        g_33=g_33,
        g_12=g_12,
        g_13=g_13,
        g_23=g_23,
    )


# Keep the imported operators available for the MMS implementation that will be
# filled in next: the direct debug parallel gradient is the one to use.
_OPERATOR_IMPORTS = (
    grad_parallel_op,
    debug_only_grad_parallel_op,
    grad_perp_op,
    curvature_op,
    poisson_bracket_op,
    perp_laplacian_op,
)
