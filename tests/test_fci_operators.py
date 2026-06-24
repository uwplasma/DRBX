from __future__ import annotations

from pathlib import Path
import os
import sys

try:
    import pytest
except ImportError:  # pragma: no cover - optional test runner dependency
    class _PytestStub:
        @staticmethod
        def importorskip(name: str):
            return None

    pytest = _PytestStub()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
_GEOMETRY_PATH = _SRC_PATH / "jax_drb" / "geometry"
_NATIVE_PATH = _SRC_PATH / "jax_drb" / "native"
_NUMPY_DEBUG = os.environ.get("JAX_DRB_FCI_NUMPY_DEBUG", "").lower() in {"1", "true", "yes"}

if _NUMPY_DEBUG:
    for path in (_GEOMETRY_PATH, _NATIVE_PATH):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import numpy as jnp

    from fci_geometry import FciGeometry3D, build_fci_maps_from_b_contravariant_np, logical_grid_from_axis_vectors_np
    from fci_operators import (
        FciAxisBoundaryCondition,
        FciBoundaryCondition,
        curvature_op_np,
        debug_only_grad_parallel_op_np,
        grad_parallel_op_np,
        grad_perp_op_np,
        invert_perp_laplacian_cg,
        perp_laplacian_op_np,
        poisson_bracket_op_np,
    )

    build_fci_maps_from_b_contravariant = build_fci_maps_from_b_contravariant_np
    logical_grid_from_axis_vectors = logical_grid_from_axis_vectors_np
    grad_parallel_op = grad_parallel_op_np
    debug_only_grad_parallel_op = debug_only_grad_parallel_op_np
    grad_perp_op = grad_perp_op_np
    perp_laplacian_op = perp_laplacian_op_np
    curvature_op = curvature_op_np
    poisson_bracket_op = poisson_bracket_op_np

    print("NUMPY DEBUG")
else:
    if str(_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_SRC_PATH))
    import jax

    jax.config.update("jax_disable_jit", True)
    import jax.numpy as jnp

    from jax_drb.geometry import FciGeometry3D, build_fci_maps_from_b_contravariant, logical_grid_from_axis_vectors
    from jax_drb.native.fci_operators import (
        FciAxisBoundaryCondition,
        FciBoundaryCondition,
        curvature_op,
        debug_only_grad_parallel_op,
        grad_parallel_op,
        grad_perp_op,
        invert_perp_laplacian_cg,
        perp_laplacian_op,
        poisson_bracket_op,
    )


A = 0.2
M = 1
N = 1
R0 = 3.0
ALPHA = 0.25
C_PHI = 3.0
IOTA = 1.1


def build_test_fci_geometry(
    shape: tuple[int, int, int],
    *,
    r0: float = R0,
    alpha: float = ALPHA,
    C_phi: float = C_PHI,
    iota: float = IOTA,
    rho_min: float = 0.15,
    construct_fci_maps: bool = True,
    B_contravariant: jnp.ndarray | None = None,
) -> FciGeometry3D:
    """Build the shifted circular toroidal test geometry for analytic operators.

    The coordinate map is

    ``R = r0 + alpha * rho + rho * cos(theta)``
    ``X = R * cos(phi)``
    ``Y = R * sin(phi)``
    ``Z = rho * sin(theta)``

    The geometric metric terms are inserted from closed-form expressions so the
    operator tests compare against an explicit analytic reference.
    """

    nx, ny, nz = shape
    target_shape = (nx, ny, nz)
    rho_1d = jnp.linspace(float(rho_min), 1.0, nx, dtype=jnp.float64)
    theta_1d = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False, dtype=jnp.float64)
    phi_1d = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False, dtype=jnp.float64)
    rho = jnp.broadcast_to(rho_1d[:, None, None], target_shape)
    theta = jnp.broadcast_to(theta_1d[None, :, None], target_shape)
    phi = jnp.broadcast_to(phi_1d[None, None, :], target_shape)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    R = r0 + alpha * rho + rho * cos_theta
    jacobian = R * rho * (1.0 + alpha * cos_theta)
    g11 = 1.0 / (1.0 + alpha * cos_theta) ** 2
    g12 = alpha * sin_theta / (rho * (1.0 + alpha * cos_theta) ** 2)
    g13 = jnp.zeros_like(g11)
    g22 = (1.0 + 2.0 * alpha * cos_theta + alpha**2) / (rho**2 * (1.0 + alpha * cos_theta) ** 2)
    g23 = jnp.zeros_like(g11)
    g33 = 1.0 / (R**2)
    g_11 = 1.0 + 2.0 * alpha * cos_theta + alpha**2
    g_12 = -alpha * rho * sin_theta
    g_13 = jnp.zeros_like(g11)
    g_22 = rho**2
    g_23 = jnp.zeros_like(g11)
    g_33 = R**2
    expected_bmag = jnp.sqrt((float(iota) ** 2) * rho**2 + R**2) * float(C_phi) / jacobian

    if B_contravariant is None:
        B_contravariant = jnp.stack(
            (
                jnp.zeros(target_shape, dtype=jnp.float64),
                float(iota) * float(C_phi) / jacobian,
                float(C_phi) / jacobian,
            ),
            axis=-1,
        )
    else:
        B_contravariant = jnp.asarray(B_contravariant, dtype=jnp.float64)

    logical_grid = logical_grid_from_axis_vectors(rho_1d, theta_1d, phi_1d)
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
    geometry = FciGeometry3D(
        logical_grid=logical_grid,
        forward_x=map_fields["forward_x"],
        forward_y=map_fields["forward_y"],
        backward_x=map_fields["backward_x"],
        backward_y=map_fields["backward_y"],
        forward_length=map_fields["forward_length"],
        backward_length=map_fields["backward_length"],
        forward_boundary=map_fields["forward_boundary"],
        backward_boundary=map_fields["backward_boundary"],
        dx=jnp.ones(target_shape, dtype=jnp.float64) * (1.0 - float(rho_min)) / float(max(nx - 1, 1)),
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
    bmag_error = jnp.abs(expected_bmag - geometry.Bmag)
    print("Bmag error mean:", float(jnp.mean(bmag_error)))
    print("Bmag error min:", float(jnp.min(bmag_error)))
    print("Bmag error max:", float(jnp.max(bmag_error)))
    return geometry


def mms_test_grad_parallel(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the manufactured parallel-gradient test field."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    dfdtheta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    dfdphi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    R = float(R0) + float(ALPHA) * rho + rho * jnp.cos(theta)
    expected = (float(IOTA) * dfdtheta + dfdphi) / jnp.sqrt(float(IOTA) ** 2 * rho**2 + R**2)

    actual = grad_parallel_op(field, geometry)
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("grad_parallel error mean:", float(jnp.mean(error)))
    print("grad_parallel error median:", float(jnp.median(error)))
    print("grad_parallel error min:", float(jnp.min(error)))
    print("grad_parallel error max:", float(jnp.max(error)))
    print("grad_parallel max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))))


def mms_test_debug_only_grad_parallel(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the direct `b^j \partial_j f` comparison test."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    dfdtheta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    dfdphi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    R = float(R0) + float(ALPHA) * rho + rho * jnp.cos(theta)
    expected = (float(IOTA) * dfdtheta + dfdphi) / jnp.sqrt(float(IOTA) ** 2 * rho**2 + R**2)

    actual = debug_only_grad_parallel_op(field, geometry)
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("debug_only_grad_parallel error mean:", float(jnp.mean(error)))
    print("debug_only_grad_parallel error median:", float(jnp.median(error)))
    print("debug_only_grad_parallel error min:", float(jnp.min(error)))
    print("debug_only_grad_parallel error max:", float(jnp.max(error)))
    print("debug_only_grad_parallel max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))))


def mms_test_grad_perp(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the manufactured perpendicular-gradient test field."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    dfd_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    dfd_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    dfd_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + float(ALPHA) * rho + rho * cos_theta
    Q = 1.0 + float(ALPHA) * cos_theta
    D2 = (float(IOTA) ** 2) * rho**2 + R**2

    expected = jnp.stack(
        (
            (1.0 / Q**2) * dfd_rho + (float(ALPHA) * sin_theta / (rho * Q**2)) * dfd_theta,
            (float(ALPHA) * sin_theta / (rho * Q**2)) * dfd_rho
            + ((1.0 + 2.0 * float(ALPHA) * cos_theta + float(ALPHA) ** 2) / (rho**2 * Q**2) - (float(IOTA) ** 2) / D2) * dfd_theta
            - (float(IOTA) / D2) * dfd_phi,
            -(float(IOTA) / D2) * dfd_theta + (1.0 / R**2 - 1.0 / D2) * dfd_phi,
        ),
        axis=-1,
    )

    actual = grad_perp_op(field, geometry, periodic_axes=(False, True, True))
    error = jnp.linalg.norm(actual - expected, axis=-1)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("grad_perp error mean:", float(jnp.mean(error)))
    print("grad_perp error median:", float(jnp.median(error)))
    print("grad_perp error min:", float(jnp.min(error)))
    print("grad_perp error max:", float(jnp.max(error)))
    print("grad_perp max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.sum((actual - expected) ** 2, axis=-1))))


def mms_test_poisson_bracket(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the manufactured Poisson-bracket test field."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))

    f = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    g_m = float(M) + 1.0
    g_n = float(N) + 1.0
    g = amplitude * jnp.cos(g_m * theta) * jnp.sin(g_n * phi)
    g_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(g_m * theta) * jnp.sin(g_n * phi)
    g_theta = -amplitude * g_m * jnp.sin(g_m * theta) * jnp.sin(g_n * phi)
    g_phi = amplitude * g_n * jnp.cos(g_m * theta) * jnp.cos(g_n * phi)

    R = float(R0) + float(ALPHA) * rho + rho * jnp.cos(theta)
    Q = 1.0 + float(ALPHA) * jnp.cos(theta)
    J = rho * R * Q
    D = jnp.sqrt((float(IOTA) ** 2) * rho**2 + R**2)

    expected = (
        1.0
        / (J * D)
        * (
            -float(ALPHA) * float(IOTA) * rho * jnp.sin(theta) * (f_theta * g_phi - f_phi * g_theta)
            + float(IOTA) * rho**2 * (f_phi * g_rho - f_rho * g_phi)
            + R**2 * (f_rho * g_theta - f_theta * g_rho)
        )
    )

    actual = poisson_bracket_op(f, g, geometry, periodic_axes=(False, True, True))
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("poisson_bracket error mean:", float(jnp.mean(error)))
    print("poisson_bracket error median:", float(jnp.median(error)))
    print("poisson_bracket error min:", float(jnp.min(error)))
    print("poisson_bracket error max:", float(jnp.max(error)))
    print("poisson_bracket max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))))


def mms_test_curvature_op(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the manufactured curvature test field."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    f_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + rho * (float(ALPHA) + cos_theta)
    Q = 1.0 + float(ALPHA) * cos_theta
    J = rho * R * Q
    D2 = (float(IOTA) ** 2) * rho**2 + R**2
    D = jnp.sqrt(D2)
    P = float(ALPHA) + cos_theta
    E = rho * Q + float(ALPHA) * R
    A_term = (float(IOTA) ** 2) * rho + R * P

    K_rho = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * rho * R * sin_theta / D
            + 2.0 * rho * R**3 * sin_theta / D**3
            - rho * R**2 * sin_theta * E / (D * J)
        )
    )
    K_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * P / D
            - 2.0 * R**2 * A_term / D**3
            + R**2 * Q * (R + rho * P) / (D * J)
        )
    )
    K_phi = (
        float(IOTA)
        / (2.0 * J)
        * (
            rho * (2.0 + float(ALPHA) * cos_theta) / D
            - 2.0 * rho**2 * A_term / D**3
            + 2.0 * float(ALPHA) * rho**2 * R * sin_theta**2 / D**3
            + (rho**2 * Q * (R + rho * P) - float(ALPHA) * rho**2 * sin_theta**2 * E) / (D * J)
        )
    )

    expected = K_rho * f_rho + K_theta * f_theta + K_phi * f_phi

    actual = curvature_op(field, geometry, periodic_axes=(False, True, True))
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("curvature error mean:", float(jnp.mean(error)))
    print("curvature error median:", float(jnp.median(error)))
    print("curvature error min:", float(jnp.min(error)))
    print("curvature error max:", float(jnp.max(error)))
    print("curvature max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))))


def _perp_laplacian_mms_case(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the manufactured field, exact Laplacian, and exact radial fluxes."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    Fr = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    Fr_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A))
    Fr_rhorho = -(jnp.pi**2 / float(A) ** 2) * jnp.cos(jnp.pi * rho / float(A))

    field = Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    f_rho = Fr_rho * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -float(M) * Fr * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = float(N) * Fr * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    f_rhorho = Fr_rhorho * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_rhotheta = -float(M) * Fr_rho * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_rhophi = float(N) * Fr_rho * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    f_thetatheta = -(float(M) ** 2) * Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_thetaphi = -float(M) * float(N) * Fr * jnp.sin(float(M) * theta) * jnp.cos(float(N) * phi)
    f_phiphi = -(float(N) ** 2) * Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + rho * (float(ALPHA) + cos_theta)
    Q = 1.0 + float(ALPHA) * cos_theta
    J = rho * R * Q
    S = (float(IOTA) ** 2) * rho**2 + R**2
    E = rho * Q + float(ALPHA) * R
    Mcoef = 1.0 + 2.0 * float(ALPHA) * cos_theta + float(ALPHA) ** 2

    P_rhorho = 1.0 / Q**2
    P_rhotheta = float(ALPHA) * sin_theta / (rho * Q**2)
    P_thetatheta = Mcoef / (rho**2 * Q**2) - (float(IOTA) ** 2) / S
    P_thetaphi = -float(IOTA) / S
    P_phiphi = 1.0 / R**2 - 1.0 / S

    C_rho = (1.0 / J) * (
        R + rho * cos_theta + (float(ALPHA) ** 2) * R * sin_theta**2 / Q**2
    )
    C_theta = (sin_theta / J) * (
        -1.0
        + float(ALPHA) * R * (float(ALPHA) ** 2 - 1.0) / (rho * Q**2)
        + rho * (float(IOTA) ** 2) * E / S
        - 2.0 * rho**2 * (float(IOTA) ** 2) * R**2 * Q / S**2
    )
    C_phi = (rho * float(IOTA) * sin_theta / J) * (
        E / S - 2.0 * rho * R**2 * Q / S**2
    )

    expected = (
        P_rhorho * f_rhorho
        + 2.0 * P_rhotheta * f_rhotheta
        + P_thetatheta * f_thetatheta
        + 2.0 * P_thetaphi * f_thetaphi
        + P_phiphi * f_phiphi
        + C_rho * f_rho
        + C_theta * f_theta
        + C_phi * f_phi
    )

    inner_flux = (
        -(rho * R / Q) * (jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
        - (float(ALPHA) * R * sin_theta / Q) * float(M) * (1.0 + jnp.cos(jnp.pi * rho / float(A))) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    )[0, :, :]
    outer_flux = (
        -(rho * R / Q) * (jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
        - (float(ALPHA) * R * sin_theta / Q) * float(M) * (1.0 + jnp.cos(jnp.pi * rho / float(A))) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    )[-1, :, :]

    return field, expected, inner_flux, outer_flux


def mms_test_perp_laplacian_op(geometry: FciGeometry3D) -> float:
    """Return the L2 error for the manufactured perpendicular-Laplacian test field."""

    field, expected, inner_flux, outer_flux = _perp_laplacian_mms_case(geometry)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(
            lower_flux=inner_flux,
            upper_flux=outer_flux,
        ),
    )
    actual = perp_laplacian_op(field, geometry, bc=bc)
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("perp_laplacian error mean:", float(jnp.mean(error)))
    print("perp_laplacian error median:", float(jnp.median(error)))
    print("perp_laplacian error min:", float(jnp.min(error)))
    print("perp_laplacian error max:", float(jnp.max(error)))
    print("perp_laplacian max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))))


def mms_test_invert_perp_laplacian_cg_dirichlet(geometry: FciGeometry3D) -> float:
    """Return the L2 inversion error using exact radial Dirichlet data."""

    field, expected, _, _ = _perp_laplacian_mms_case(geometry)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.dirichlet(
            lower_value=field[0, :, :],
            upper_value=field[-1, :, :],
        ),
    )
    actual = invert_perp_laplacian_cg(
        expected,
        geometry,
        bc,
        tol=1.0e-6,
        maxiter=500,
    )
    error = jnp.abs(actual - field)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("invert_perp_laplacian_dirichlet error mean:", float(jnp.mean(error)))
    print("invert_perp_laplacian_dirichlet error median:", float(jnp.median(error)))
    print("invert_perp_laplacian_dirichlet error min:", float(jnp.min(error)))
    print("invert_perp_laplacian_dirichlet error max:", float(jnp.max(error)))
    print("invert_perp_laplacian_dirichlet max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - field))))


def mms_test_invert_perp_laplacian_cg_neumann(geometry: FciGeometry3D) -> float:
    """Return the L2 inversion error using zero radial Neumann flux."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    rho_min = geometry.logical_grid[0, 0, 0, 0]
    rho_max = geometry.logical_grid[-1, 0, 0, 0]
    radial_length = rho_max - rho_min
    xi = (rho - rho_min) / radial_length

    field = jnp.cos(2.0 * jnp.pi * xi)
    field_rho = -(2.0 * jnp.pi / radial_length) * jnp.sin(2.0 * jnp.pi * xi)
    field_rhorho = -((2.0 * jnp.pi / radial_length) ** 2) * jnp.cos(2.0 * jnp.pi * xi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + rho * (float(ALPHA) + cos_theta)
    Q = 1.0 + float(ALPHA) * cos_theta
    J = rho * R * Q
    P_rhorho = 1.0 / Q**2
    C_rho = (1.0 / J) * (
        R + rho * cos_theta + (float(ALPHA) ** 2) * R * sin_theta**2 / Q**2
    )
    expected = P_rhorho * field_rhorho + C_rho * field_rho

    zero_flux = jnp.zeros(field.shape[1:], dtype=jnp.float64)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(
            lower_flux=zero_flux,
            upper_flux=zero_flux,
        ),
        target_mean_phi=jnp.sum(geometry.J * field) / jnp.sum(geometry.J),
    )
    actual = invert_perp_laplacian_cg(
        expected,
        geometry,
        bc,
        tol=1.0e-6,
        maxiter=500,
    )
    error = jnp.abs(actual - field)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("invert_perp_laplacian_neumann error mean:", float(jnp.mean(error)))
    print("invert_perp_laplacian_neumann error median:", float(jnp.median(error)))
    print("invert_perp_laplacian_neumann error min:", float(jnp.min(error)))
    print("invert_perp_laplacian_neumann error max:", float(jnp.max(error)))
    print("invert_perp_laplacian_neumann max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - field))))


def debug_point_on_mms_field(geometry: FciGeometry3D, index: tuple[int, int, int]) -> None:
    """Print local MMS field and operator data at a single logical grid index."""

    rho = geometry.logical_grid[..., 0]
    theta = geometry.logical_grid[..., 1]
    phi = geometry.logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    dfdtheta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    dfdphi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    R = float(R0) + float(ALPHA) * rho + rho * jnp.cos(theta)
    expected = (float(IOTA) * dfdtheta + dfdphi) / jnp.sqrt(float(IOTA) ** 2 * rho**2 + R**2)
    i, j, k = index
    actual = grad_parallel_op(field=field, geometry=geometry)[i, j, k]
    debug_actual = debug_only_grad_parallel_op(
        field=field,
        geometry=geometry,
    )[i, j, k]

    print("debug point index:", index)
    print("debug point logical coords rho/theta/phi:", float(rho[i, j, k]), float(theta[i, j, k]), float(phi[i, j, k]))
    print("debug point field value:", float(field[i, j, k]))
    print("debug point dfdtheta:", float(dfdtheta[i, j, k]))
    print("debug point dfdphi:", float(dfdphi[i, j, k]))
    print("debug point expected grad_parallel:", float(expected[i, j, k]))
    print("debug point grad_parallel:", float(actual))
    print("debug point debug_only_grad_parallel:", float(debug_actual))
    print("debug point forward map x/y:", float(geometry.forward_x[i, j, k]), float(geometry.forward_y[i, j, k]))
    print("debug point backward map x/y:", float(geometry.backward_x[i, j, k]), float(geometry.backward_y[i, j, k]))
    print("debug point forward/backward lengths:", float(geometry.forward_length[i, j, k]), float(geometry.backward_length[i, j, k]))
    print("debug point forward/backward boundary:", bool(geometry.forward_boundary[i, j, k]), bool(geometry.backward_boundary[i, j, k]))


def build_identity_fci_geometry(shape: tuple[int, int, int], *, dz: float = 1.0) -> FciGeometry3D:
    nx, ny, nz = shape
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    logical_grid = logical_grid_from_axis_vectors(
        jnp.arange(nx, dtype=jnp.float64),
        jnp.arange(ny, dtype=jnp.float64),
        jnp.arange(nz, dtype=jnp.float64),
    )
    forward_x = jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float64)[:, None, None], shape)
    forward_y = jnp.broadcast_to(jnp.arange(ny, dtype=jnp.float64)[None, :, None], shape)
    backward_x = forward_x
    backward_y = forward_y
    return FciGeometry3D(
        logical_grid=logical_grid,
        forward_x=forward_x,
        forward_y=forward_y,
        backward_x=backward_x,
        backward_y=backward_y,
        forward_length=ones * float(dz),
        backward_length=ones * float(dz),
        forward_boundary=zeros.astype(bool),
        backward_boundary=zeros.astype(bool),
        dx=ones,
        dy=ones,
        dz=ones * float(dz),
        J=ones,
        B_contravariant=jnp.zeros(shape + (3,), dtype=jnp.float64).at[..., 2].set(1.0),
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )


def test_grad_parallel_on_identity_maps_returns_unit_toroidal_gradient() -> None:
    geometry = build_identity_fci_geometry((3, 4, 6), dz=1.0)
    field = jnp.broadcast_to(jnp.arange(6, dtype=jnp.float64)[None, None, :], geometry.shape)

    actual = grad_parallel_op(field, geometry)

    assert jnp.allclose(actual, 1.0)


def test_grad_perp_on_identity_metric_projects_out_toroidal_component() -> None:
    geometry = build_identity_fci_geometry((4, 5, 6))
    x = jnp.arange(4, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(5, dtype=jnp.float64)[None, :, None]
    z = jnp.arange(6, dtype=jnp.float64)[None, None, :]
    field = x + 2.0 * y + 3.0 * z

    actual = grad_perp_op(field, geometry, periodic_axes=(False, True, True))

    expected = jnp.stack((jnp.ones(geometry.shape), 2.0 * jnp.ones(geometry.shape), jnp.zeros(geometry.shape)), axis=-1)
    assert jnp.allclose(actual, expected)


def test_grad_perp_uses_second_order_nonperiodic_edge_stencil() -> None:
    geometry = build_identity_fci_geometry((4, 3, 2))
    x = jnp.arange(4, dtype=jnp.float64)[:, None, None]
    field = x * x

    actual = grad_perp_op(field, geometry, periodic_axes=(False, True, True))

    expected_x = 2.0 * x
    expected = jnp.stack(
        (jnp.broadcast_to(expected_x, geometry.shape), jnp.zeros(geometry.shape), jnp.zeros(geometry.shape)),
        axis=-1,
    )
    assert jnp.allclose(actual, expected)


def test_poisson_bracket_on_identity_metric_matches_cross_product_form() -> None:
    geometry = build_identity_fci_geometry((4, 5, 6))
    x = jnp.arange(4, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(5, dtype=jnp.float64)[None, :, None]
    f = x
    g = y

    actual = poisson_bracket_op(f, g, geometry, periodic_axes=(False, True, True))

    assert jnp.allclose(actual, 1.0)


def test_curvature_op_vanishes_for_constant_field_on_identity_metric() -> None:
    geometry = build_identity_fci_geometry((4, 5, 6))

    actual = curvature_op(jnp.ones(geometry.shape, dtype=jnp.float64), geometry, periodic_axes=(False, True, True))

    assert jnp.allclose(actual, 0.0)


def test_perp_laplacian_op_on_identity_metric_matches_radial_second_derivative_interior() -> None:
    geometry = build_identity_fci_geometry((5, 4, 3))
    x = jnp.arange(5, dtype=jnp.float64)[:, None, None]
    field = x * x
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(lower_flux=0.0, upper_flux=0.0),
    )

    actual = perp_laplacian_op(field, geometry, bc=bc)

    assert jnp.allclose(actual[1:-1, :, :], 2.0)


def test_perp_laplacian_op_does_not_wrap_radial_boundary() -> None:
    geometry = build_identity_fci_geometry((5, 4, 3))
    x = jnp.arange(5, dtype=jnp.float64)[:, None, None]
    field = x * x
    perturbed = field.at[-1, :, :].set(123.0)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(lower_flux=0.0, upper_flux=0.0),
    )

    base_actual = perp_laplacian_op(field, geometry, bc=bc)
    perturbed_actual = perp_laplacian_op(perturbed, geometry, bc=bc)

    assert jnp.allclose(base_actual[0, :, :], perturbed_actual[0, :, :])


def test_perp_laplacian_op_uses_explicit_radial_boundary_fluxes() -> None:
    geometry = build_identity_fci_geometry((5, 4, 3))
    field = jnp.zeros(geometry.shape, dtype=jnp.float64)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(lower_flux=2.0, upper_flux=-3.0),
    )

    actual = perp_laplacian_op(
        field,
        geometry,
        bc=bc,
    )

    expected = jnp.zeros(geometry.shape, dtype=jnp.float64)
    expected = expected.at[0, :, :].set(-16.0 / 3.0)
    expected = expected.at[-1, :, :].set(-8.0)

    assert jnp.allclose(actual, expected)


def test_invert_perp_laplacian_cg_dirichlet_mms() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((8, 6, 6), construct_fci_maps=False)

    error = mms_test_invert_perp_laplacian_cg_dirichlet(geometry)

    assert error < 1.5e-1


def test_invert_perp_laplacian_cg_dirichlet_enforces_boundary_values() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((8, 6, 6), construct_fci_maps=False)
    field, expected, _, _ = _perp_laplacian_mms_case(geometry)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.dirichlet(
            lower_value=field[0, :, :],
            upper_value=field[-1, :, :],
        ),
    )

    actual = invert_perp_laplacian_cg(expected, geometry, bc, tol=1.0e-6, maxiter=200)

    assert jnp.allclose(actual[0, :, :], field[0, :, :])
    assert jnp.allclose(actual[-1, :, :], field[-1, :, :])


def test_invert_perp_laplacian_cg_neumann_mms() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((8, 6, 6), construct_fci_maps=False)

    error = mms_test_invert_perp_laplacian_cg_neumann(geometry)

    assert error < 1.0e-1


def test_invert_perp_laplacian_cg_neumann_target_mean() -> None:
    pytest.importorskip("lineax")
    geometry = build_identity_fci_geometry((8, 6, 6))
    zero_flux = jnp.zeros(geometry.shape[1:], dtype=jnp.float64)
    target_mean = jnp.asarray(2.5, dtype=jnp.float64)
    bc = FciBoundaryCondition(
        periodic_axes=(False, True, True),
        x=FciAxisBoundaryCondition.neumann_flux(lower_flux=zero_flux, upper_flux=zero_flux),
        target_mean_phi=target_mean,
    )

    omega = jnp.zeros(geometry.shape, dtype=jnp.float64)
    actual = invert_perp_laplacian_cg(omega, geometry, bc, tol=1.0e-6, maxiter=200)

    assert jnp.allclose(actual, target_mean)


if __name__ == "__main__":
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    
    '''
    print('Building Fci test')
    a = 250
    b = 500
    test_geometry_low = build_test_fci_geometry((a,a,a), construct_fci_maps=False)
    test_geometry_high = build_test_fci_geometry((b,b,b), construct_fci_maps=False)

    
    print('Testing grad parallel')
    #grad_parallel_low = mms_test_grad_parallel(test_geometry_low)
    #grad_parallel_high = mms_test_grad_parallel(test_geometry_high)
    print('Testing debug_only grad parallel')
    grad_parallel_low = mms_test_debug_only_grad_parallel(test_geometry_low)
    grad_parallel_high = mms_test_debug_only_grad_parallel(test_geometry_high)
    print(f'Grad parallel estimated order = {jnp.log(grad_parallel_low / grad_parallel_high) / jnp.log(2.0)}')

    
    
    
    print('Testing grad perp')
    grad_perp_low = mms_test_grad_perp(test_geometry_low)
    grad_perp_high = mms_test_grad_perp(test_geometry_high)
    print(f'Grad perp estimated order = {jnp.log(grad_perp_low / grad_perp_high) / jnp.log(2.0)}')
    
    
    
    print('Testing poisson bracket')
    poisson_bracket_low = mms_test_poisson_bracket(test_geometry_low)
    poisson_bracket_high = mms_test_poisson_bracket(test_geometry_high)
    print(f'Poisson bracket estimated order = {jnp.log(poisson_bracket_low / poisson_bracket_high) / jnp.log(2.0)}')
    

    
    print('Testing curvature')
    curvature_low = mms_test_curvature_op(test_geometry_low)
    curvature_high = mms_test_curvature_op(test_geometry_high)
    print(f'Curvature estimated order = {jnp.log(curvature_low / curvature_high) / jnp.log(2.0)}')
    

    print('Testing perp laplacian')
    perp_laplacian_low = mms_test_perp_laplacian_op(test_geometry_low)
    perp_laplacian_high = mms_test_perp_laplacian_op(test_geometry_high)
    print(f'Perp laplacian estimated order = {jnp.log(perp_laplacian_low / perp_laplacian_high) / jnp.log(2.0)}')
    '''
    resolutions = [20,30,40,50]
    '''operator_specs = [
        ("grad parallel", mms_test_debug_only_grad_parallel),
        ("grad perp", mms_test_grad_perp),
        ("poisson bracket", mms_test_poisson_bracket),
        ("curvature", mms_test_curvature_op),
        ("perp laplacian", mms_test_perp_laplacian_op),
    ]'''
    operator_specs = [
        ("invert perp laplacian dirichlet", mms_test_invert_perp_laplacian_cg_dirichlet),
        ("invert perp laplacian neumann", mms_test_invert_perp_laplacian_cg_neumann),
    ]

    error_results: dict[str, list[float]] = {label: [] for label, _ in operator_specs}
    for res in resolutions:
        print(f"Building geometry for resolution {res}")
        test_geometry = build_test_fci_geometry((res, res, res), construct_fci_maps=False)
        for label, function in operator_specs:
            print(f"Testing {label}")
            error_results[label].append(function(test_geometry))

    log_resolutions = np.log(np.asarray(resolutions, dtype=float))
    ncols = 2
    nrows = math.ceil(len(operator_specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)

    for index, (label, _) in enumerate(operator_specs):
        ax = axes[index // ncols][index % ncols]
        errors = np.asarray(error_results[label], dtype=float)
        log_errors = np.log(errors)
        slope, intercept = np.polyfit(log_resolutions, log_errors, 1)
        estimated_order = -slope
        fit_errors = np.exp(intercept + slope * log_resolutions)

        print(f"{label} estimated order (best fit) = {estimated_order}")

        ax.loglog(resolutions, errors, "o-", label="data")
        ax.loglog(resolutions, fit_errors, "--", label=f"best fit order {estimated_order:.2f}")
        ax.set_title(f"{label}\norder ≈ {estimated_order:.2f}")
        ax.set_xlabel("resolution")
        ax.set_ylabel("L2 error")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.legend()

    for index in range(len(operator_specs), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.tight_layout()
    fig.savefig("fci_operator_convergence.png", dpi=200)
