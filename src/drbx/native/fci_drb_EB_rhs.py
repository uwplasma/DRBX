from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
import math
import os
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import (
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_PHYSICAL_BOUNDARY,
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    SIDE_PHYSICAL,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_cell_gradient_from_field,
    build_local_fci_stencil_from_field,
)
from .fci_model import FciModelState
from .fci_model import inject_owned_field_to_halo, inject_owned_state_to_halo
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NONE,
    ConservativeStencil3D,
    LocalBoundaryFaceBC3D,
    LocalBoundaryFaceTrace3D,
    LocalControlVolumeBoundaryBC3D,
    LocalEmbeddedControlVolumeGeometry3D,
)
from .fci_halo import (
    build_local_boundary_face_trace_from_halo,
    HaloExchange3D,
    LocalHaloClosure3D,
    MetricAwarePhysicalGhostCellFiller3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
    RemoteFciDependencyExchange,
)
from .fci_operators import (
    LocalPerpLaplacianInverseSolver,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_conservative,
    local_parallel_div_b_op,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_parallel_q_flux_div_fci_op,
    local_parallel_div_b_fci_from_q_op,
    local_grad_parallel_op_fci_compatible_from_q,
    local_grad_parallel_op_fci_compatible_from_q_components,
    local_parallel_diffusion_fci_op,
    local_grad_parallel_op_fci,
    local_center_to_outgoing_face_average_fci_op,
    local_outgoing_face_to_center_average_fci_op,
    local_center_to_outgoing_face_grad_parallel_fci_op,
    local_outgoing_face_to_center_div_parallel_fci_op,
    LocalOutgoingFciFaceTopology3D,
    prolong_local_outgoing_fci_face_owner_field,
    restrict_local_outgoing_fci_face_field,
    local_perp_laplacian_conservative_op,
    local_curvature_conservative_op,
    local_curvature_conservative_components_op,
    local_curvature_production_path_op,
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op_from_gradients,
    build_local_control_volume_field_closure,
    linear_combination_local_control_volume_closures,
    product_local_control_volume_closures,
    expand_local_control_volume_owner_field,
    aggregate_local_control_volume_average,
    _radial_characteristic_fine_glue_owner_correction,
    _radial_characteristic_third_order_owner_correction,
    _poloidal_characteristic_owner_correction,
    _poloidal_characteristic_third_order_owner_correction,
    _mask_inactive_owned,
    _mask_state_inactive_owned,
)
from .fci_gmres import SolvaxGmresConfig, SolvaxGmresInfo
from .fci_face_galerkin import build_local_fci_face_galerkin_transfer
from .fci_support_pair import build_weighted_negative_adjoint
from .fci_parallel_production_flux import (
    parallel_characteristic_wall_data,
    parallel_short_wall_backward_euler,
    parallel_target_row_material_residual,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbEBState(FciModelState):
    """State for the electrostatic Boussinesq drift-reduced Braginskii scaffold."""

    density: jax.Array
    phi: jax.Array
    Te: jax.Array
    Ti: jax.Array
    Vi: jax.Array
    Ve: jax.Array
    vorticity: jax.Array


RHS_TERM_FIELD_NAMES = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
RHS_TERM_NAMES = (
    (
        "poisson_bracket",
        "parallel_density_flux_divergence",
        "curvature",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "characteristic_leg_upwind",
        "source",
    ),
    (
        "poisson_bracket",
        "parallel_advection",
        "curvature",
        "parallel_compression",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "characteristic_leg_upwind",
        "source",
    ),
    (
        "poisson_bracket",
        "parallel_advection",
        "curvature",
        "parallel_compression",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "characteristic_leg_upwind",
        "source",
    ),
    (
        "poisson_bracket",
        "parallel_self_advection",
        "parallel_pressure",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "characteristic_leg_upwind",
        "source",
    ),
    (
        "poisson_bracket",
        "parallel_self_advection",
        "collision",
        "electrostatic",
        "electron_pressure",
        "thermal_force",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "characteristic_leg_upwind",
        "source",
    ),
    (
        "poisson_bracket",
        "parallel_advection",
        "parallel_current",
        "curvature",
        "perpendicular_diffusion",
        "parallel_diffusion",
        "source",
    ),
)
RHS_TERM_SLOT_COUNT = max(len(names) for names in RHS_TERM_NAMES)

ELECTRON_FORCE_TERM_NAMES = (
    "parallel_self_advection",
    "collision",
    "electrostatic",
    "electron_pressure",
    "thermal_force",
    "characteristic_leg_upwind",
    "vorticity_current_flux_divergence",
)
ELECTRON_FORCE_LEG_TERM_NAMES = (
    "parallel_self_advection",
    "electrostatic",
    "electron_pressure",
    "thermal_force",
    "characteristic_leg_upwind",
)
ELECTRON_FORCE_GRADIENT_NAMES = ("Ve", "phi", "Pe", "Te")
ELECTRON_FORCE_ENDPOINT_FIELD_NAMES = (
    "density", "Te", "Ti", "Vi", "Ve", "phi", "Pe"
)
ELECTRON_FORCE_STENCIL_DIRECTION_NAMES = ("backward", "center", "forward")
ELECTRON_FORCE_ENDPOINT_DIRECTION_NAMES = ("backward", "forward")

CURVATURE_COMPONENT_DIAGNOSTIC_NAMES = {
    "directional": ("u", "theta", "eta"),
    "centered-dissipation": (
        "centered_u",
        "centered_theta",
        "centered_eta",
        "dissipation_u",
        "dissipation_theta",
        "dissipation_eta",
    ),
    "radial-provenance": (
        "radial_lower_axis_face",
        "radial_upper_physical_face",
        "radial_rlp_transition_faces",
        "radial_ordinary_interior_faces",
        "radial_within_cell_path",
        "radial_nonlocal_psi_remainder",
        "theta_total",
        "eta_total",
    ),
}


def curvature_component_diagnostic_names(
    scheme: str | None = None,
) -> tuple[str, ...]:
    """Return the self-describing lane names for curvature replay output."""

    selected = (
        os.environ.get("DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME", "directional")
        if scheme is None
        else str(scheme)
    )
    try:
        return CURVATURE_COMPONENT_DIAGNOSTIC_NAMES[selected]
    except KeyError as exc:
        raise ValueError(
            "unknown curvature component diagnostic scheme "
            f"{selected!r}"
        ) from exc


def _mask_local_eb_state_inactive(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
) -> FciDrbEBState:
    """Zero inactive owned cells for a local EB state/update payload."""

    return _mask_state_inactive_owned(state, geometry)


def background_curvature_characteristic_decomposition(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Return the four background curvature speeds and spectral projectors.

    The background is n=Te=Ti=1, omega=0.  The polynomial projectors avoid a
    facewise eigendecomposition and are therefore safe inside jit/shard_map.
    ``bmag`` is retained in the matrix because the vorticity row depends on
    the local wall-face magnetic field.  The characteristic order is
    ``(electron-fast, electron-slow, ion, stationary)`` with speeds
    ``(mu_plus, mu_minus, -10*tau/3, 0)``.  Positive ``tau`` keeps the four
    roots distinct.

    Unlike :func:`_characteristic_projectors_background`, this decomposition
    keeps the two positive electron modes separate.  Their signs are the same
    for wall inflow selection, but their magnitudes differ and must remain
    separate when constructing ``|M|`` for a characteristic interface flux.
    """
    bmag = jnp.asarray(bmag, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    shape = bmag.shape
    M = jnp.zeros(shape + (4, 4), dtype=jnp.float64)
    M = M.at[..., 0, 0].set(2.0)
    M = M.at[..., 0, 1].set(2.0)
    M = M.at[..., 1, 0].set(4.0 / 3.0)
    M = M.at[..., 1, 1].set(14.0 / 3.0)
    M = M.at[..., 2, 0].set(4.0 / 3.0)
    M = M.at[..., 2, 1].set(4.0 / 3.0)
    M = M.at[..., 2, 2].set(-10.0 * tau / 3.0)
    M = M.at[..., 3, 0].set(2.0 * bmag * bmag * (1.0 + tau))
    M = M.at[..., 3, 1].set(2.0 * bmag * bmag)
    M = M.at[..., 3, 2].set(2.0 * tau * bmag * bmag)
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float64), shape + (4, 4))
    mu_plus = (10.0 + 2.0 * jnp.sqrt(10.0)) / 3.0
    mu_minus = (10.0 - 2.0 * jnp.sqrt(10.0)) / 3.0
    mu_i = jnp.broadcast_to(-10.0 * tau / 3.0, shape)

    def product(factors):
        result = eye
        for factor in factors:
            result = jnp.einsum("...ij,...jk->...ik", result, factor)
        return result

    mu_plus_field = jnp.broadcast_to(mu_plus, shape)
    mu_minus_field = jnp.broadcast_to(mu_minus, shape)
    mu_zero = jnp.zeros(shape, dtype=jnp.float64)
    mu_plus_matrix = mu_plus_field[..., None, None]
    mu_minus_matrix = mu_minus_field[..., None, None]

    P_plus = product((M - mu_minus_matrix * eye, M - mu_i[..., None, None] * eye, M))
    P_plus = P_plus / (
        (mu_plus_field - mu_minus_field)
        * (mu_plus_field - mu_i)
        * mu_plus_field
    )[..., None, None]
    P_slow = product((M - mu_plus_matrix * eye, M - mu_i[..., None, None] * eye, M))
    P_slow = P_slow / (
        (mu_minus_field - mu_plus_field)
        * (mu_minus_field - mu_i)
        * mu_minus_field
    )[..., None, None]
    P_ion = product((M - mu_plus_matrix * eye, M - mu_minus_matrix * eye, M))
    P_ion = P_ion / (
        mu_i * (mu_i - mu_plus_field) * (mu_i - mu_minus_field)
    )[..., None, None]
    P_zero = product(
        (M - mu_plus_matrix * eye, M - mu_minus_matrix * eye, M - mu_i[..., None, None] * eye)
    )
    P_zero = P_zero / (
        (-mu_plus_field) * (-mu_minus_field) * (-mu_i)
    )[..., None, None]
    return (
        (mu_plus_field, mu_minus_field, mu_i, mu_zero),
        (P_plus, P_slow, P_ion, P_zero),
    )


def background_curvature_characteristic_absolute_matrix(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> jnp.ndarray:
    """Return ``|M|=sum_j |mu_j| P_j`` for the background curvature system.

    For the positive ``tau`` regime used by the characteristic closure, the
    two electron roots are positive, the ion root is negative, and the fourth
    root is zero.  Interpolating ``abs`` on those four roots reduces the
    absolute matrix to a cubic polynomial in ``M``.  Expanding that polynomial
    once gives the entries below, avoiding a batched eigensystem/projector
    construction and its twelve small matrix products per face.
    """

    bmag = jnp.asarray(bmag, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(bmag.shape, tau.shape)
    bmag = jnp.broadcast_to(bmag, shape)
    tau = jnp.broadcast_to(tau, shape)
    denominator = 5.0 * tau * tau + 10.0 * tau + 3.0
    bmag2 = bmag * bmag
    result = jnp.zeros(shape + (4, 4), dtype=jnp.float64)
    result = result.at[..., 0, 0].set(2.0)
    result = result.at[..., 0, 1].set(2.0)
    result = result.at[..., 1, 0].set(4.0 / 3.0)
    result = result.at[..., 1, 1].set(14.0 / 3.0)
    result = result.at[..., 2, 0].set(
        -4.0 * (5.0 * tau * tau - 3.0) / (3.0 * denominator)
    )
    result = result.at[..., 2, 1].set(
        -4.0 * (5.0 * tau * tau - 10.0 * tau - 3.0)
        / (3.0 * denominator)
    )
    result = result.at[..., 2, 2].set(10.0 * tau / 3.0)
    result = result.at[..., 3, 0].set(
        2.0
        * bmag2
        * (tau + 1.0)
        * (5.0 * tau * tau + 14.0 * tau + 3.0)
        / denominator
    )
    result = result.at[..., 3, 1].set(
        2.0 * bmag2 * (9.0 * tau * tau + 10.0 * tau + 3.0)
        / denominator
    )
    result = result.at[..., 3, 2].set(-2.0 * bmag2 * tau)
    return result


def background_curvature_characteristic_metric(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> jnp.ndarray:
    """Return ``H=sum_j P_j^T P_j``, a local symmetrizer of background ``M``."""

    _speeds, projectors = background_curvature_characteristic_decomposition(
        bmag, tau
    )
    result = jnp.zeros_like(projectors[0])
    for projector in projectors:
        result = result + jnp.einsum(
            "...ki,...kj->...ij", projector, projector
        )
    return result


def background_curvature_characteristic_penalty(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> jnp.ndarray:
    """Return the symmetric PSD face block ``H |M|``."""

    metric = background_curvature_characteristic_metric(bmag, tau)
    absolute = background_curvature_characteristic_absolute_matrix(bmag, tau)
    product = jnp.einsum("...ij,...jk->...ik", metric, absolute)
    return 0.5 * (product + jnp.swapaxes(product, -1, -2))


def _characteristic_projectors_background(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return grouped (positive electron, negative ion, zero) projectors."""

    _speeds, (p_fast, p_slow, p_ion, p_zero) = (
        background_curvature_characteristic_decomposition(bmag, tau)
    )
    return p_fast + p_slow, p_ion, p_zero


def _axis_plane_slice(axis: int, side: int) -> tuple[slice, slice, slice]:
    """Slice the lower (side=0) or upper (side=1) plane of a face array."""
    if axis not in (0, 1, 2) or side not in (0, 1):
        raise ValueError(f"invalid axis/side ({axis}, {side})")
    index = 0 if side == 0 else -1
    result = [slice(None), slice(None), slice(None)]
    result[axis] = slice(index, index + 1) if side == 0 else slice(-1, None)
    return tuple(result)


def _apply_projector(projector: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", projector, values)


def parallel_characteristic_matrix(
    density: jnp.ndarray,
    Te: jnp.ndarray,
    Ti: jnp.ndarray,
    Vi: jnp.ndarray,
    Ve: jnp.ndarray,
    tau: float | jnp.ndarray,
    mu: float | jnp.ndarray,
) -> jnp.ndarray:
    """Return the analytic five-field parallel principal matrix.

    The state order is ``(density, Te, Ti, Vi, Ve)``.  This deliberately
    excludes both vorticity and the diagnostic electrostatic potential: the
    former has a defective repeated-speed coupling in the full matrix and the
    latter is supplied by a nonlocal polarization solve.
    """

    density = jnp.asarray(density, dtype=jnp.float64)
    Te = jnp.asarray(Te, dtype=jnp.float64)
    Ti = jnp.asarray(Ti, dtype=jnp.float64)
    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    Ve = jnp.asarray(Ve, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    dV = Vi - Ve
    n_safe = jnp.maximum(density, 1.0e-30)
    shape = jnp.broadcast_shapes(
        density.shape, Te.shape, Ti.shape, Vi.shape, Ve.shape
    )
    matrix = jnp.zeros(shape + (5, 5), dtype=jnp.float64)
    matrix = matrix.at[..., 0, 0].set(Ve)
    matrix = matrix.at[..., 0, 4].set(density)
    matrix = matrix.at[..., 1, 0].set(-1.42 * Te * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 1, 1].set(Ve)
    matrix = matrix.at[..., 1, 3].set(-1.42 * Te / 3.0)
    matrix = matrix.at[..., 1, 4].set(3.42 * Te / 3.0)
    matrix = matrix.at[..., 2, 0].set(-2.0 * Ti * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 2, 2].set(Vi)
    matrix = matrix.at[..., 2, 4].set(2.0 * Ti / 3.0)
    matrix = matrix.at[..., 3, 0].set((Te + tau * Ti) / n_safe)
    matrix = matrix.at[..., 3, 1].set(1.0)
    matrix = matrix.at[..., 3, 2].set(tau)
    matrix = matrix.at[..., 3, 3].set(Vi)
    matrix = matrix.at[..., 4, 0].set(mu * Te / n_safe)
    matrix = matrix.at[..., 4, 1].set(1.71 * mu)
    matrix = matrix.at[..., 4, 4].set(Ve)
    return matrix


def parallel_characteristic_split_matrices(
    matrix: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = 1.0e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split the five-field principal matrix into right/left-going parts.

    Returns ``(A_plus, A_minus, P_plus, P_minus)``.  The nonsymmetric
    eigensystem is frozen, as it is for the characteristic wall closure, while
    multiplication by the live principal matrix remains differentiable.  The
    zero-speed subspace belongs to neither directional operator.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    frozen_matrix = jax.lax.stop_gradient(matrix)
    eigenvalues, eigenvectors = jnp.linalg.eig(frozen_matrix)
    eigenvalues = jnp.real(eigenvalues)
    inverse = jnp.linalg.inv(eigenvectors)

    def projector(select: jnp.ndarray) -> jnp.ndarray:
        value = jnp.einsum(
            "...ik,...k,...kj->...ij",
            eigenvectors,
            select.astype(jnp.float64),
            inverse,
        )
        return jax.lax.stop_gradient(jnp.real(value))

    tolerance = jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
    p_plus = projector(eigenvalues > tolerance)
    p_minus = projector(eigenvalues < -tolerance)
    a_plus = jnp.einsum("...ij,...jk->...ik", matrix, p_plus)
    a_minus = jnp.einsum("...ij,...jk->...ik", matrix, p_minus)
    return a_plus, a_minus, p_plus, p_minus


def target_local_characteristic_upwind_correction(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: jnp.ndarray,
    dx_plus: jnp.ndarray,
    centered_gradient: jnp.ndarray,
    matrix: jnp.ndarray,
    backward_wall: jnp.ndarray,
    forward_wall: jnp.ndarray,
    *,
    equilibrium: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the RHS correction for wall-terminating target-local FCI legs.

    Ordinary rows return zero.  On a row with a physical endpoint, this
    replaces the existing centered principal contribution ``-A d_parallel U``
    by ``-(A+ delta- + A- delta+)``.  At a physical endpoint, incoming
    characteristic perturbations are set to zero about ``equilibrium`` while
    outgoing and stationary perturbations retain the target-cell state.
    """

    center = jnp.asarray(center, dtype=jnp.float64)
    minus = jnp.asarray(minus, dtype=jnp.float64)
    plus = jnp.asarray(plus, dtype=jnp.float64)
    centered_gradient = jnp.asarray(centered_gradient, dtype=jnp.float64)
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    backward_wall = jnp.asarray(backward_wall, dtype=bool)
    forward_wall = jnp.asarray(forward_wall, dtype=bool)
    if equilibrium is None:
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = jnp.asarray(equilibrium, dtype=jnp.float64)

    a_plus, a_minus, p_plus, p_minus = parallel_characteristic_split_matrices(
        matrix
    )
    perturbation = center - equilibrium
    # At the backward wall, lambda>0 modes enter the domain.  At the forward
    # wall, lambda<0 modes enter it.
    backward_state = equilibrium + perturbation - _apply_projector(
        p_plus, perturbation
    )
    forward_state = equilibrium + perturbation - _apply_projector(
        p_minus, perturbation
    )
    minus = jnp.where(backward_wall[..., None], backward_state, minus)
    plus = jnp.where(forward_wall[..., None], forward_state, plus)

    delta_minus = (center - minus) / jnp.maximum(
        jnp.asarray(dx_minus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    delta_plus = (plus - center) / jnp.maximum(
        jnp.asarray(dx_plus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    centered_principal = _apply_projector(matrix, centered_gradient)
    upwind_principal = _apply_projector(a_plus, delta_minus) + _apply_projector(
        a_minus, delta_plus
    )
    wall_row = backward_wall | forward_wall
    return jnp.where(
        wall_row[..., None],
        centered_principal - upwind_principal,
        0.0,
    )


def target_local_characteristic_upwind_principal_diagnostics(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: jnp.ndarray,
    dx_plus: jnp.ndarray,
    centered_gradient: jnp.ndarray,
    matrix: jnp.ndarray,
    backward_wall: jnp.ndarray,
    forward_wall: jnp.ndarray,
    *,
    equilibrium: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the two principal pieces and endpoints used by the wall correction.

    The principal outputs are masked to wall-terminating rows and satisfy
    ``centered_principal - upwind_principal == correction``.  Endpoint values
    have trailing shape ``(backward/forward, primitive)`` and contain the
    characteristic-projected state on a physical wall leg and the production
    mapped endpoint on an ordinary leg.  This helper is diagnostic-only and
    deliberately mirrors :func:`target_local_characteristic_upwind_correction`.
    """

    center = jnp.asarray(center, dtype=jnp.float64)
    minus = jnp.asarray(minus, dtype=jnp.float64)
    plus = jnp.asarray(plus, dtype=jnp.float64)
    centered_gradient = jnp.asarray(centered_gradient, dtype=jnp.float64)
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    backward_wall = jnp.asarray(backward_wall, dtype=bool)
    forward_wall = jnp.asarray(forward_wall, dtype=bool)
    if equilibrium is None:
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = jnp.asarray(equilibrium, dtype=jnp.float64)

    a_plus, a_minus, p_plus, p_minus = parallel_characteristic_split_matrices(
        matrix
    )
    perturbation = center - equilibrium
    backward_state = equilibrium + perturbation - _apply_projector(
        p_plus, perturbation
    )
    forward_state = equilibrium + perturbation - _apply_projector(
        p_minus, perturbation
    )
    minus_used = jnp.where(backward_wall[..., None], backward_state, minus)
    plus_used = jnp.where(forward_wall[..., None], forward_state, plus)
    delta_minus = (center - minus_used) / jnp.maximum(
        jnp.asarray(dx_minus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    delta_plus = (plus_used - center) / jnp.maximum(
        jnp.asarray(dx_plus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    centered_principal = _apply_projector(matrix, centered_gradient)
    upwind_principal = _apply_projector(a_plus, delta_minus) + _apply_projector(
        a_minus, delta_plus
    )
    wall_row = backward_wall | forward_wall
    return (
        jnp.where(wall_row[..., None], centered_principal, 0.0),
        jnp.where(wall_row[..., None], upwind_principal, 0.0),
        jnp.stack((minus_used, plus_used), axis=-2),
    )


def target_local_characteristic_upwind_correction_components(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: jnp.ndarray,
    dx_plus: jnp.ndarray,
    centered_gradient_components: jnp.ndarray,
    matrix: jnp.ndarray,
    backward_wall: jnp.ndarray,
    forward_wall: jnp.ndarray,
    *,
    equilibrium: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Split the characteristic wall correction by mapped-stencil lane.

    ``centered_gradient_components`` has trailing shape ``(3, 5)`` in
    ``(backward, center, forward)`` order.  Summing the returned lane axis
    reconstructs :func:`target_local_characteristic_upwind_correction` when
    the supplied components reconstruct its centered gradient.
    """

    center = jnp.asarray(center, dtype=jnp.float64)
    minus = jnp.asarray(minus, dtype=jnp.float64)
    plus = jnp.asarray(plus, dtype=jnp.float64)
    centered_gradient_components = jnp.asarray(
        centered_gradient_components, dtype=jnp.float64
    )
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    backward_wall = jnp.asarray(backward_wall, dtype=bool)
    forward_wall = jnp.asarray(forward_wall, dtype=bool)
    expected = center.shape[:-1] + (3, center.shape[-1])
    if centered_gradient_components.shape != expected:
        raise ValueError(
            "centered_gradient_components must have shape "
            f"{expected}, got {centered_gradient_components.shape}"
        )
    if equilibrium is None:
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = jnp.asarray(equilibrium, dtype=jnp.float64)

    a_plus, a_minus, p_plus, p_minus = parallel_characteristic_split_matrices(
        matrix
    )
    perturbation = center - equilibrium
    backward_state = equilibrium + perturbation - _apply_projector(
        p_plus, perturbation
    )
    forward_state = equilibrium + perturbation - _apply_projector(
        p_minus, perturbation
    )
    minus = jnp.where(backward_wall[..., None], backward_state, minus)
    plus = jnp.where(forward_wall[..., None], forward_state, plus)
    delta_minus = (center - minus) / jnp.maximum(
        jnp.asarray(dx_minus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    delta_plus = (plus - center) / jnp.maximum(
        jnp.asarray(dx_plus, dtype=jnp.float64)[..., None], 1.0e-30
    )
    components = jnp.einsum(
        "...ij,...dj->...di", matrix, centered_gradient_components
    )
    components = components.at[..., 0, :].add(
        -_apply_projector(a_plus, delta_minus)
    )
    components = components.at[..., 2, :].add(
        -_apply_projector(a_minus, delta_plus)
    )
    return jnp.where(
        (backward_wall | forward_wall)[..., None, None], components, 0.0
    )


def parallel_incoming_projector(
    matrix: jnp.ndarray,
    b_dot_n: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = 1.0e-10,
) -> jnp.ndarray:
    """Return the frozen local projector onto incoming material modes.

    ``matrix`` is the analytic five-field matrix and ``b_dot_n`` is the
    outward, normalized field-normal coefficient.  The eigensystem is used
    only to construct the boundary trace; it is stopped from participating in
    implicit JVP differentiation because eigenvectors of a nonsymmetric
    matrix are not a stable Newton unknown.  Tangential faces return zero.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    b_dot_n = jnp.asarray(b_dot_n, dtype=jnp.float64)
    # Freeze the matrix before eigendecomposition as well as freezing the
    # resulting projector.  JAX does not provide derivatives of nonsymmetric
    # eigenvectors; owner/candidate trace derivatives remain active below.
    normal_matrix = jax.lax.stop_gradient(b_dot_n[..., None, None] * matrix)
    eigenvalues, eigenvectors = jnp.linalg.eig(normal_matrix)
    eigenvalues = jnp.real(eigenvalues)
    inverse = jnp.linalg.inv(eigenvectors)
    incoming = eigenvalues < -jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
    projector = jnp.einsum(
        "...ik,...k,...kj->...ij",
        eigenvectors,
        incoming.astype(jnp.float64),
        inverse,
    )
    projector = jnp.real(projector)
    tangent = jnp.abs(b_dot_n) <= eigenvalue_tolerance
    projector = jnp.where(tangent[..., None, None], 0.0, projector)
    return jax.lax.stop_gradient(projector)


def parallel_characteristic_wall_state(
    owner: jnp.ndarray,
    candidate: jnp.ndarray,
    matrix: jnp.ndarray,
    b_dot_n: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = 1.0e-10,
) -> jnp.ndarray:
    """Replace only incoming components of a candidate wall state."""

    projector = parallel_incoming_projector(
        matrix, b_dot_n, eigenvalue_tolerance=eigenvalue_tolerance
    )
    return owner + _apply_projector(projector, candidate - owner)


def parallel_equilibrium_characteristic_wall_state(
    owner: jnp.ndarray,
    equilibrium: jnp.ndarray,
    matrix: jnp.ndarray,
    b_dot_n: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = 1.0e-10,
) -> jnp.ndarray:
    """Zero incoming perturbations while retaining owner outgoing modes.

    The equilibrium is preserved exactly: if ``owner == equilibrium``, the
    returned wall state is the equilibrium.  For a nontangential face this
    imposes ``P_in (U_wall - U_equilibrium) = 0`` and retains
    ``(I - P_in) (U_owner - U_equilibrium)``.
    """

    projector = parallel_incoming_projector(
        matrix, b_dot_n, eigenvalue_tolerance=eigenvalue_tolerance
    )
    perturbation = owner - equilibrium
    return equilibrium + perturbation - _apply_projector(projector, perturbation)


def parallel_derived_state_traces(
    state: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Derive the composite parallel traces from one projected state."""

    density, Te, Ti, Vi, Ve = [state[..., index] for index in range(5)]
    tau = jnp.asarray(tau, dtype=jnp.float64)
    return (
        density * Ve,
        density * (Vi - Ve),
        density * Te,
        density * (Te + tau * Ti),
    )


def _upwind_equilibrium_characteristic_state(
    owner_state: jnp.ndarray,
    equilibrium_state: jnp.ndarray,
    retained_projector: jnp.ndarray,
) -> jnp.ndarray:
    """Return a wall state with equilibrium incoming characteristics.

    ``owner_state`` supplies the retained outgoing and stationary
    characteristic perturbations.  Incoming perturbations are set to zero
    relative to the supplied equilibrium state.  ``retained_projector`` is
    precomputed as ``P_out + P_stationary`` for the outward wall-normal
    principal operator.
    """
    delta_owner = owner_state - equilibrium_state
    return equilibrium_state + _apply_projector(retained_projector, delta_owner)


def _characteristic_outgoing_incoming_projectors(
    q_normal: jnp.ndarray,
    p_electron: jnp.ndarray,
    p_ion: jnp.ndarray,
    p_zero: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select (outgoing, incoming, stationary) projectors for A_n=-q_n M."""
    q_normal = jnp.asarray(q_normal, dtype=jnp.float64)
    positive_q = q_normal > 0.0
    negative_q = q_normal < 0.0
    moving = positive_q | negative_q
    zero = jnp.zeros_like(p_zero)
    outgoing = jnp.where(
        positive_q[..., None, None],
        p_ion,
        jnp.where(negative_q[..., None, None], p_electron, zero),
    )
    incoming = jnp.where(
        positive_q[..., None, None],
        p_electron,
        jnp.where(negative_q[..., None, None], p_ion, zero),
    )
    stationary = p_zero + jnp.where(
        moving[..., None, None],
        zero,
        p_electron + p_ion,
    )
    return outgoing, incoming, stationary


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UpwindEquilibriumWallProjectors:
    """Geometry-only characteristic projectors for physical wall faces.

    The nested payload is indexed as ``axes[axis][side]`` and contains the
    retained projector ``P_out + P_stationary``.  It is a pytree so the
    shard-local arrays remain compatible with ``jit`` and ``shard_map`` while
    the immutable structure is created once before time integration.
    """

    axes: tuple[
        tuple[
            jax.Array,
            jax.Array,
        ],
        tuple[
            jax.Array,
            jax.Array,
        ],
        tuple[
            jax.Array,
            jax.Array,
        ],
    ]

    def tree_flatten(self):
        return tuple(
            side
            for axis in self.axes
            for side in axis
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        values = iter(children)
        axes = tuple(
            tuple(
                next(values) for _ in range(2)
            )
            for _ in range(3)
        )
        return cls(axes=axes)  # type: ignore[arg-type]


def build_upwind_equilibrium_wall_projectors(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    curvature_face_coefficients: LocalCurvatureFaceCoefficients3D,
    tau: float | jnp.ndarray,
) -> UpwindEquilibriumWallProjectors:
    """Precompute replicated wall projectors from invariant local geometry.

    A compact wall-plane output omits the normal mesh dimension.  Therefore
    each local side is first masked by its runtime physical-boundary ownership
    and then summed over the corresponding mesh axis.  This leaves one copy
    of a physical wall plane on every normal shard and zeros on periodic axes,
    satisfying the compact output's VMA requirement.
    """

    axes = []
    for axis, name in enumerate(("x", "y", "z")):
        q = jnp.asarray(getattr(curvature_face_coefficients, name), dtype=jnp.float64)
        bmag = jnp.asarray(
            getattr(geometry.face_bfield, name).Bmag_owned,
            dtype=jnp.float64,
        )
        sides = []
        for side in (0, 1):
            sl = _axis_plane_slice(axis, side)
            p_electron, p_ion, p_zero = _characteristic_projectors_background(
                bmag[sl], tau
            )
            if side == 0:
                qn = -q[sl]
            else:
                qn = q[sl]
            p_out, _p_in, p_stationary = _characteristic_outgoing_incoming_projectors(
                qn,
                p_electron,
                p_ion,
                p_zero,
            )
            side_active = (
                domain.runtime_has_physical_lower(axis)
                if side == 0
                else domain.runtime_has_physical_upper(axis)
            )
            retained = p_out + p_stationary
            masked = jnp.where(
                side_active, retained, jnp.zeros_like(retained)
            )
            mesh_axis_name = domain.mesh_axis_names[axis]
            if mesh_axis_name is not None:
                masked = jax.lax.psum(masked, axis_name=mesh_axis_name)
            sides.append(masked)
        axes.append(tuple(sides))
    return UpwindEquilibriumWallProjectors(axes=tuple(axes))  # type: ignore[arg-type]


def _wall_candidate_values(
    stencil: ConservativeStencil3D,
    face_bc: LocalBoundaryFaceBC3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return central/Neumann-reconstructed candidate values on each face family."""
    values = []
    for name, value in zip(
        ("x", "y", "z"),
        (stencil.face_values.x, stencil.face_values.y, stencil.face_values.z),
    ):
        kind = getattr(face_bc, f"kind_{name}")
        prescribed = getattr(face_bc, f"value_{name}")
        mask = getattr(face_bc, f"mask_{name}")
        value = jnp.where(mask & (kind == BC_DIRICHLET), prescribed, value)
        values.append(value)
    return tuple(values)


def _dirichlet_face_bc_from_values(
    values: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layout,
    masks: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> LocalBoundaryFaceBC3D:
    return LocalBoundaryFaceBC3D(
        kind_x=jnp.where(masks[0], BC_DIRICHLET, 0),
        kind_y=jnp.where(masks[1], BC_DIRICHLET, 0),
        kind_z=jnp.where(masks[2], BC_DIRICHLET, 0),
        value_x=values[0], value_y=values[1], value_z=values[2],
        mask_x=masks[0], mask_y=masks[1], mask_z=masks[2],
        layout=layout,
    )

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbEBRhsParameters:
    """Physical normalization constants for the electrostatic Boussinesq DRB scaffold."""

    n0: float = 1.0
    Te0: float = 1.0
    Ti0: float = 1.0
    cs_0: float = 1.0
    rhos_s0: float = 1.0
    tau: float = 1.0
    mi_over_me: float = 1836.0
    rho_star: float = 1.0
    phi_inversion_iterations: int = 80
    phi_inversion_regularization: float = 1.0e-9
    density_D_perp: float = 0.0
    density_D_parallel: float = 0.0
    electron_temperature_chi_parallel: float = 0.0
    electron_temperature_D_perp: float = 0.0
    ion_temperature_chi_parallel: float = 0.0
    ion_temperature_D_perp: float = 0.0
    Ve_nu: float = 0.0
    Ve_D_perp: float = 0.0
    Ve_parallel_viscosity: float = 0.0
    Vi_D_perp: float = 0.0
    Vi_parallel_viscosity: float = 0.0
    vorticity_D_perp: float = 0.0
    vorticity_D_parallel: float = 0.0

    def tree_flatten(self):
        return (
            (
                self.n0,
                self.Te0,
                self.Ti0,
                self.cs_0,
                self.rhos_s0,
                self.tau,
                self.mi_over_me,
                self.rho_star,
                self.phi_inversion_iterations,
                self.phi_inversion_regularization,
                self.density_D_perp,
                self.density_D_parallel,
                self.electron_temperature_chi_parallel,
                self.electron_temperature_D_perp,
                self.ion_temperature_chi_parallel,
                self.ion_temperature_D_perp,
                self.Ve_nu,
                self.Ve_D_perp,
                self.Ve_parallel_viscosity,
                self.Vi_D_perp,
                self.Vi_parallel_viscosity,
                self.vorticity_D_perp,
                self.vorticity_D_parallel,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            n0,
            Te0,
            Ti0,
            cs_0,
            rhos_s0,
            tau,
            mi_over_me,
            rho_star,
            phi_inversion_iterations,
            phi_inversion_regularization,
            density_D_perp,
            density_D_parallel,
            electron_temperature_chi_parallel,
            electron_temperature_D_perp,
            ion_temperature_chi_parallel,
            ion_temperature_D_perp,
            Ve_nu,
            Ve_D_perp,
            Ve_parallel_viscosity,
            Vi_D_perp,
            Vi_parallel_viscosity,
            vorticity_D_perp,
            vorticity_D_parallel,
        ) = children
        return cls(
            n0=n0,
            Te0=Te0,
            Ti0=Ti0,
            cs_0=cs_0,
            rhos_s0=rhos_s0,
            tau=tau,
            mi_over_me=mi_over_me,
            rho_star=rho_star,
            phi_inversion_iterations=phi_inversion_iterations,
            phi_inversion_regularization=phi_inversion_regularization,
            density_D_perp=density_D_perp,
            density_D_parallel=density_D_parallel,
            electron_temperature_chi_parallel=electron_temperature_chi_parallel,
            electron_temperature_D_perp=electron_temperature_D_perp,
            ion_temperature_chi_parallel=ion_temperature_chi_parallel,
            ion_temperature_D_perp=ion_temperature_D_perp,
            Ve_nu=Ve_nu,
            Ve_D_perp=Ve_D_perp,
            Ve_parallel_viscosity=Ve_parallel_viscosity,
            Vi_D_perp=Vi_D_perp,
            Vi_parallel_viscosity=Vi_parallel_viscosity,
            vorticity_D_perp=vorticity_D_perp,
            vorticity_D_parallel=vorticity_D_parallel,
        )


@dataclass(frozen=True)
class LocalFciDrbEBFaceBCBundle:
    """Local/domain-decomposed face boundary bundle for the EB model."""

    density: LocalBoundaryFaceBC3D
    phi: LocalBoundaryFaceBC3D
    Te: LocalBoundaryFaceBC3D
    Ti: LocalBoundaryFaceBC3D
    Vi: LocalBoundaryFaceBC3D
    Ve: LocalBoundaryFaceBC3D
    vorticity: LocalBoundaryFaceBC3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalFciDrbEBOperatorBoundaryBundle:
    """Operator-level numerical traces for the seven-field EB model.

    ``LocalFciDrbEBFaceBCBundle`` contains the model's primitive physical
    boundary conditions.  This bundle contains the actual scalar traces used
    by first-order conservative operands.  Composite traces are constructed
    from primitive wall traces, rather than from products of ghost cells.
    """

    density: LocalBoundaryFaceTrace3D
    phi: LocalBoundaryFaceTrace3D
    Te: LocalBoundaryFaceTrace3D
    Ti: LocalBoundaryFaceTrace3D
    Vi: LocalBoundaryFaceTrace3D
    Ve: LocalBoundaryFaceTrace3D
    vorticity: LocalBoundaryFaceTrace3D
    density_flux: LocalBoundaryFaceTrace3D
    current: LocalBoundaryFaceTrace3D
    Pe: LocalBoundaryFaceTrace3D
    pressure: LocalBoundaryFaceTrace3D
    Ti_squared: LocalBoundaryFaceTrace3D

    def tree_flatten(self):
        return (
            self.density, self.phi, self.Te, self.Ti, self.Vi, self.Ve,
            self.vorticity, self.density_flux, self.current, self.Pe,
            self.pressure, self.Ti_squared,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    @property
    def layout(self):
        layout = self.density.layout
        for name in self.__dataclass_fields__:
            trace = getattr(self, name)
            if trace.layout != layout:
                raise ValueError(
                    f"operator boundary trace {name!r} has a different layout"
                )
        return layout


def _combine_operator_traces(
    *traces: LocalBoundaryFaceTrace3D,
    operation: Callable[..., jax.Array],
) -> LocalBoundaryFaceTrace3D:
    """Combine collocated traces with an intersection of all active masks."""

    if not traces:
        raise ValueError("at least one trace is required")
    if not all(isinstance(trace, LocalBoundaryFaceTrace3D) for trace in traces):
        raise TypeError("operator boundary operands must be LocalBoundaryFaceTrace3D")
    layout = traces[0].layout
    if any(trace.layout != layout for trace in traces):
        raise ValueError("operator boundary traces must share one layout")

    values = []
    masks = []
    for axis, name in enumerate(("x", "y", "z")):
        axis_values = [jnp.asarray(getattr(trace, f"value_{name}"), dtype=jnp.float64) for trace in traces]
        axis_masks = [jnp.asarray(getattr(trace, f"mask_{name}"), dtype=bool) for trace in traces]
        active = jnp.logical_and.reduce(jnp.stack(axis_masks, axis=0), axis=0)
        values.append(jnp.where(active, operation(*axis_values), 0.0))
        masks.append(active)
    return LocalBoundaryFaceTrace3D(
        value_x=values[0], value_y=values[1], value_z=values[2],
        mask_x=masks[0], mask_y=masks[1], mask_z=masks[2], layout=layout,
    )


def build_local_fci_drb_eb_operator_boundary_bundle(
    state_halo: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_bc: LocalFciDrbEBFaceBCBundle,
    *,
    tau: float | jax.Array,
) -> LocalFciDrbEBOperatorBoundaryBundle:
    """Build level-2 traces from fully closed primitive field halos."""

    if not isinstance(state_halo, FciDrbEBState):
        raise TypeError("state_halo must be an FciDrbEBState")
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share one layout")
    if not isinstance(face_bc, LocalFciDrbEBFaceBCBundle):
        raise TypeError("face_bc must be a LocalFciDrbEBFaceBCBundle")
    primitive_bcs = (
        face_bc.density, face_bc.phi, face_bc.Te, face_bc.Ti,
        face_bc.Vi, face_bc.Ve, face_bc.vorticity,
    )
    if any(bc.layout != domain.layout for bc in primitive_bcs):
        raise ValueError("all primitive face BCs must share the domain layout")
    primitive_fields = (
        state_halo.density, state_halo.phi, state_halo.Te, state_halo.Ti,
        state_halo.Vi, state_halo.Ve, state_halo.vorticity,
    )
    traces = tuple(
        build_local_boundary_face_trace_from_halo(field, geometry, domain, bc)
        for field, bc in zip(primitive_fields, primitive_bcs)
    )
    density, phi, Te, Ti, Vi, Ve, vorticity = traces
    density_flux = _combine_operator_traces(
        density, Ve, operation=lambda n, ve: n * ve
    )
    current = _combine_operator_traces(
        density, Vi, Ve, operation=lambda n, vi, ve: n * (vi - ve)
    )
    Pe = _combine_operator_traces(density, Te, operation=lambda n, te: n * te)
    pressure = _combine_operator_traces(
        density, Te, Ti,
        operation=lambda n, te, ti: n * (te + jnp.asarray(tau, dtype=jnp.float64) * ti),
    )
    Ti_squared = _combine_operator_traces(Ti, Ti, operation=lambda ti0, ti1: ti0 * ti1)
    return LocalFciDrbEBOperatorBoundaryBundle(
        density=density, phi=phi, Te=Te, Ti=Ti, Vi=Vi, Ve=Ve,
        vorticity=vorticity, density_flux=density_flux, current=current,
        Pe=Pe, pressure=pressure, Ti_squared=Ti_squared,
    )


LocalFciDrbEBFaceBCBuilder = Callable[
    [FciDrbEBState, LocalFciGeometry3D, LocalDomain3D, FciDrbEBRhsParameters],
    LocalFciDrbEBFaceBCBundle,
]


def _binary_local_dirichlet_face_bc(
    left: LocalBoundaryFaceBC3D,
    right: LocalBoundaryFaceBC3D,
    operation: Callable[[jax.Array, jax.Array], jax.Array],
) -> LocalBoundaryFaceBC3D:
    """Combine collocated regular-face Dirichlet data."""

    if left.layout != right.layout:
        raise ValueError("regular-face BC operands must share one layout")

    def combine(
        left_kind,
        right_kind,
        left_value,
        right_value,
        left_mask,
        right_mask,
    ):
        active = (
            left_mask
            & right_mask
            & (left_kind == BC_DIRICHLET)
            & (right_kind == BC_DIRICHLET)
        )
        return (
            jnp.where(active, BC_DIRICHLET, BC_NONE),
            jnp.where(active, operation(left_value, right_value), 0.0),
            active,
        )

    x = combine(
        left.kind_x,
        right.kind_x,
        left.value_x,
        right.value_x,
        left.mask_x,
        right.mask_x,
    )
    y = combine(
        left.kind_y,
        right.kind_y,
        left.value_y,
        right.value_y,
        left.mask_y,
        right.mask_y,
    )
    z = combine(
        left.kind_z,
        right.kind_z,
        left.value_z,
        right.value_z,
        left.mask_z,
        right.mask_z,
    )
    return LocalBoundaryFaceBC3D(
        kind_x=x[0],
        kind_y=y[0],
        kind_z=z[0],
        value_x=x[1],
        value_y=y[1],
        value_z=z[1],
        mask_x=x[2],
        mask_y=y[2],
        mask_z=z[2],
        layout=left.layout,
    )


def _scale_local_dirichlet_face_bc(
    boundary_bc: LocalBoundaryFaceBC3D,
    scale: float | jax.Array,
) -> LocalBoundaryFaceBC3D:
    scale_value = jnp.asarray(scale, dtype=jnp.float64)
    return LocalBoundaryFaceBC3D(
        kind_x=boundary_bc.kind_x,
        kind_y=boundary_bc.kind_y,
        kind_z=boundary_bc.kind_z,
        value_x=scale_value * boundary_bc.value_x,
        value_y=scale_value * boundary_bc.value_y,
        value_z=scale_value * boundary_bc.value_z,
        mask_x=boundary_bc.mask_x,
        mask_y=boundary_bc.mask_y,
        mask_z=boundary_bc.mask_z,
        layout=boundary_bc.layout,
    )


def prepare_local_fci_drb_eb_state(
    state_owned: FciDrbEBState,
    domain: LocalDomain3D,
    *,
    face_bc: LocalFciDrbEBFaceBCBundle,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
) -> FciDrbEBState:
    """Apply complete physical/topology/corner halo closure to all EB fields."""

    state_halo = inject_owned_state_to_halo(state_owned, domain.layout)
    closure = LocalHaloClosure3D(
        physical_ghost_filler=physical_ghost_filler,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
    )
    return FciDrbEBState(
        density=closure(state_halo.density, domain, face_bc.density),
        phi=closure(state_halo.phi, domain, face_bc.phi),
        Te=closure(state_halo.Te, domain, face_bc.Te),
        Ti=closure(state_halo.Ti, domain, face_bc.Ti),
        Vi=closure(state_halo.Vi, domain, face_bc.Vi),
        Ve=closure(state_halo.Ve, domain, face_bc.Ve),
        vorticity=closure(state_halo.vorticity, domain, face_bc.vorticity),
    )


@dataclass(frozen=True)
class LocalFciDrbEBRhs:
    """Regular-grid SPMD/local EB RHS and phi reconstruction.

    Boundary values are supplied by ``face_bc_builder`` so geometry/test-specific
    wall policy can live outside the model while the EB equation assembly remains
    in the native implementation.
    """

    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: FciDrbEBRhsParameters
    curvature_coefficients_owned: jnp.ndarray | None
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SolvaxGmresConfig
    face_bc_builder: LocalFciDrbEBFaceBCBuilder
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None
    control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D | None = None
    # Outgoing FCI source-edge ownership is independent of the cell-owner
    # topology used by angular RLP.  It is required when Vi/Ve are evolved in
    # ``fci-staggered`` storage, while the other five state leaves remain in
    # cell-owner storage.
    outgoing_face_topology: LocalOutgoingFciFaceTopology3D | None = None
    diffusion_only: bool = False
    # Static diagnostic mode: retain only the production parallel subsystem
    # in the returned RHS while still reconstructing phi normally.
    parallel_subsystem_only: bool = False
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    curvature_face_coefficients: LocalCurvatureFaceCoefficients3D | None = None
    upwind_equilibrium_wall_projectors: UpwindEquilibriumWallProjectors | None = None
    curvature_scheme: str = "direct"
    # Static selector for the complete owner-face characteristic curvature
    # update.  ``legacy`` preserves every existing prototype path exactly;
    # ``production-path`` uses the all-axis wave-propagation operator.
    curvature_split_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_SPLIT_SCHEME", "legacy"
        )
    )
    # Analysis-only layout for returned production curvature component lanes.
    # ``directional`` returns the established (u, theta, eta) full residual.
    # ``centered-dissipation`` returns six lanes: the three centered/path
    # transfers followed by the three exact |A|-jump dissipative residuals.
    # ``radial-provenance`` separates radial boundary, RLP-transition,
    # ordinary-interior, within-cell-path, and nonlocal-remainder pieces,
    # followed by the complete theta and eta contributions. Every layout's
    # lane sum is identical to the production curvature RHS.
    curvature_component_diagnostic_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME", "directional"
        )
    )
    # Production-curvature evolution ablation.  ``full`` preserves the
    # established path and, importantly, does not request diagnostic lanes.
    # The two non-default choices retain only the centered principal transfer
    # (including the compatible nonlocal psi remainder) or only the
    # characteristic jump dissipation.
    curvature_evolution_component: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_EVOLUTION_COMPONENT", "full"
        )
    )
    # Physical-wall closure for the production four-field curvature split.
    # The BC-characteristic path derives its exterior candidate from the
    # primitive operator traces instead of a fixed equilibrium state.
    curvature_wall_flux_closure: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_WALL_FLUX_CLOSURE",
            "equilibrium-exterior-canonical-face-state",
        )
    )
    # Analysis-only removal of one radial production-curvature contribution.
    # The default preserves the production operator bit for bit.
    curvature_radial_ablation: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_RADIAL_ABLATION", "none"
        )
    )
    curvature_inflow_closure: str = "central"
    parallel_inflow_closure: str = "central"
    # Diagnostic/experimental trace selector for the parallel-current term in
    # the vorticity equation.  ``operator`` preserves the historical primitive
    # boundary trace.  ``parallel-characteristic`` reuses the derived current
    # from the same projected five-field wall state used by the material
    # equations, allowing an identical-state closure-consistency ablation.
    vorticity_current_inflow_trace: str = "operator"
    # Static model configuration: these are Python strings captured by the
    # jitted RHS, rather than array-valued switches.  The shared C(f)
    # evaluations below remain unchanged; this gates their assembled
    # contribution to each evolution equation.
    curvature_equations: tuple[str, ...] = (
        "density",
        "Te",
        "Ti",
        "vorticity",
    )
    # Static experiment switch for the nonlinear ion-temperature curvature
    # term.  ``product`` preserves the original Ti*C(Ti) discretization;
    # ``flux`` uses the analytically equivalent -(5*tau/(3B))*C(Ti**2).
    ion_temperature_curvature_self_form: str = "product"
    # Static experiment control for scaling all assembled curvature terms.
    # Keeping this as a Python scalar makes it part of the jitted model
    # configuration rather than a run-time array-valued switch.
    curvature_scale: float = 1.0
    # Complete five-field parallel material flux.  ``legacy`` preserves the
    # existing mapped operators; ``production-path`` uses one canonical-face
    # characteristic fluctuation on every ordinary and wall-ending FCI row.
    parallel_material_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"
        )
    )
    # RLP curvature face treatment.  ``projected-fine`` preserves R A_f P;
    # ``moment-shared`` replaces only radial angular-resolution transitions
    # by canonical moment-fitted shared face traces.  The bounded variant
    # limits their conservative correction to the local aggregate-stencil
    # range. ``constrained-flux-shared`` instead constrains the metric flux
    # correction to zero coarse-face integral and non-positive face power.
    # ``fine-glue-sat`` uses the existing physical fine subfaces as the glue
    # grid and adds a scalar jump penalty before coupled equation assembly.
    # ``fine-glue-characteristic`` instead applies an H-compatible |M| block
    # to the coupled (n,Te,Ti,omega) jump after equation assembly.  Its
    # ``-bulk`` counterpart applies that same trace-transpose penalty on every
    # interior radial face, rather than only angular-RLP transitions.
    curvature_rlp_face_scheme: str = "projected-fine"
    curvature_rlp_fine_glue_penalty: float = 1.0
    # Analysis-only selector for the bulk radial curvature principal face
    # state. Production remains the established arithmetic centered trace;
    # donor-cell replaces only interior radial faces for a causal replay.
    curvature_radial_principal_face_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_RADIAL_PRINCIPAL_FACE_SCHEME", "centered"
        )
    )
    # Canonical radial characteristic principal flux.  ``legacy`` preserves
    # the established trace-transpose correction; ``third-order-upwind``
    # replaces it with a live, owner-face third-order characteristic flux.
    curvature_radial_characteristic_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME", "legacy"
        )
    )
    # Canonical owner-lattice third-order characteristic principal flux in the
    # periodic poloidal direction.  Legacy keeps the historical selector and
    # correction behavior bitwise unchanged.
    curvature_poloidal_characteristic_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME", "legacy"
        )
    )
    # Static diagnostic selector for the coupled characteristic curvature
    # penalty.  ``legacy`` preserves the established scheme-selected radial
    # behavior; ``radial`` makes radial-only explicit; ``radial-poloidal``
    # additionally enables ordinary poloidal faces.
    curvature_characteristic_axes: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_CURVATURE_CHARACTERISTIC_AXES", "legacy"
        )
    )
    # Optional independent multiplier for the ordinary poloidal characteristic
    # correction. ``None`` inherits the radial fine-glue penalty so existing
    # radial-poloidal runs are unchanged.
    poloidal_characteristic_penalty: float | None = field(
        default_factory=lambda: (
            None
            if os.environ.get("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY") is None
            else float(os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY"])
        )
    )
    # Audit-only static selector for one radial RLP transition face. ``None``
    # keeps the production SAT active on every transition.
    curvature_rlp_fine_glue_transition_face: int | None = None
    # Experimental selectable discretization for E x B advection.  The
    # compatible-flux path returns the already-B-divided bracket and uses
    # shared conservative face data; ``direct`` preserves the established
    # reconstructed cell-gradient implementation.
    poisson_bracket_scheme: str = "direct"
    # Select the parallel operator family.  This is intentionally a static
    # Python option so JIT compilation cannot silently mix coordinate and FCI
    # discretizations within one compiled RHS.
    parallel_operator_scheme: str = "coordinate"
    # Experimental cell-centred FCI flux divergence pairing.  ``legacy``
    # preserves the established pointwise mapped divergence exactly.  The
    # support-core variant is deliberately opt-in while its support contract
    # is exercised on real mapped fixtures.
    parallel_flux_pairing: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_FLUX_PAIRING", "legacy"
        )
    )
    # Boundary composition for the support-core phi/current pair.  ``legacy``
    # retains the former independent wall-row closures for replay ablation;
    # ``current-phi`` closes the composite current with zero Neumann data and
    # derives grad(phi) from its physical-volume weighted transpose.
    # ``characteristic-sat`` uses production first-order characteristic
    # endpoint currents; its candidates must remain central here so they are
    # not projected twice.
    parallel_boundary_pairing: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_BOUNDARY_PAIRING", "current-phi"
        )
    )
    # Experimental FCI-only storage/layout selection.  In ``fci-staggered``
    # Vi and Ve retain the cell array shape but their values live on outgoing
    # mapped FCI faces (the source-edge convention).  Perpendicular operators
    # deliberately remain source-plane/cell anchored in this first version.
    parallel_velocity_layout: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_VELOCITY_LAYOUT", "cell-centered"
        )
    )
    # Optional FCI-only closure for a target row whose mapped forward or
    # backward leg terminates at the physical vessel wall.  Interior rows keep
    # the compatible centered FCI operator.
    fci_parallel_leg_scheme: str = "centered"
    # Experimental treatment for the stiff material block on very short FCI
    # wall legs.  The default is deliberately bit-for-bit explicit.  The
    # local backward-Euler option is consumed by the time integrator through
    # ``apply_short_leg_implicit_material_step``; ``evaluate_stage`` only
    # removes the selected wall-leg contribution when a nonzero selection
    # interval is supplied by that integrator.
    parallel_short_leg_treatment: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"
        )
    )
    parallel_short_leg_cfl_limit: float = field(
        default_factory=lambda: float(
            os.environ.get("DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT", "2.5")
        )
    )

    @property
    def neumann_normal_scheme(self) -> str:
        """Keep conservative face closure consistent with ghost semantics."""

        return (
            "physical"
            if isinstance(
                self.physical_ghost_filler,
                MetricAwarePhysicalGhostCellFiller3D,
            )
            else "logical"
        )

    def __post_init__(self) -> None:
        has_cv = self.control_volume_geometry is not None
        if has_cv != (self.control_volume_boundary_bc is not None):
            raise ValueError(
                "control_volume_geometry and control_volume_boundary_bc must "
                "be supplied together; RLP mode has no fallback"
            )
        if has_cv:
            if not isinstance(self.control_volume_geometry, LocalEmbeddedControlVolumeGeometry3D):
                raise TypeError("control_volume_geometry must be LocalEmbeddedControlVolumeGeometry3D")
            if not isinstance(self.control_volume_boundary_bc, LocalControlVolumeBoundaryBC3D):
                raise TypeError("control_volume_boundary_bc must be LocalControlVolumeBoundaryBC3D")
            if self.control_volume_geometry.layout != self.geometry.layout:
                raise ValueError("control_volume_geometry must share geometry.layout")
            if self.control_volume_geometry.layout != self.domain.layout:
                raise ValueError("control_volume_geometry must share domain.layout")
            if self.control_volume_boundary_bc.max_rows != self.control_volume_geometry.irregular_faces.max_rows:
                raise ValueError("control-volume boundary rows do not match geometry")
            if self.control_volume_boundary_bc.max_patches != self.control_volume_geometry.irregular_faces.max_patches:
                raise ValueError("control-volume boundary patches do not match geometry")
            if not self.control_volume_geometry.has_projected_owner_agglomeration:
                raise ValueError(
                    "the five-field control-volume path requires a projected "
                    "owner agglomeration topology"
                )
            if (
                self.control_volume_geometry.has_angular_agglomeration
                and (
                    not self.domain.axis_regular_axes[0]
                    or not self.axis_regular_axes[0]
                )
            ):
                raise ValueError("angular RLP requires lower-radial axis regularity")
            if self.poisson_bracket_scheme != "compatible-flux":
                raise ValueError(
                    "projected-owner RLP requires "
                    "poisson_bracket_scheme='compatible-flux'"
                )
            if self.curvature_scheme == "direct":
                raise ValueError(
                    "projected-owner RLP requires conservative curvature or "
                    "disabled curvature"
                )
            shard_counts = tuple(
                int(count) for count in self.domain.shard_spec.shard_counts
            )
            if shard_counts[0] != 1 or shard_counts[1] != 1:
                raise ValueError(
                    "projected-owner RLP supports eta-only decomposition; radial and "
                    "poloidal shard counts must both be one"
                )
        if self.parallel_operator_scheme not in ("coordinate", "fci"):
            raise ValueError(
                "parallel_operator_scheme must be 'coordinate' or 'fci', got "
                f"{self.parallel_operator_scheme!r}"
            )
        if self.parallel_material_scheme not in ("legacy", "production-path"):
            raise ValueError(
                "parallel_material_scheme must be 'legacy' or 'production-path', got "
                f"{self.parallel_material_scheme!r}"
            )
        if (
            self.parallel_material_scheme == "production-path"
            and self.parallel_operator_scheme != "fci"
        ):
            raise ValueError(
                "parallel_material_scheme='production-path' requires "
                "parallel_operator_scheme='fci'"
            )
        if (
            self.parallel_material_scheme == "production-path"
            and self.parallel_velocity_layout != "cell-centered"
        ):
            raise ValueError(
                "parallel_material_scheme='production-path' currently requires "
                "parallel_velocity_layout='cell-centered'"
            )
        if (
            self.parallel_material_scheme == "production-path"
            and self.parallel_flux_pairing != "support-core"
        ):
            raise ValueError(
                "parallel_material_scheme='production-path' requires "
                "parallel_flux_pairing='support-core'"
            )
        if self.parallel_flux_pairing not in ("legacy", "support-core"):
            raise ValueError(
                "parallel_flux_pairing must be 'legacy' or 'support-core', got "
                f"{self.parallel_flux_pairing!r}"
            )
        if self.parallel_boundary_pairing not in (
            "legacy", "current-phi", "characteristic-sat"
        ):
            raise ValueError(
                "parallel_boundary_pairing must be 'legacy', 'current-phi', or "
                "'characteristic-sat', got "
                f"{self.parallel_boundary_pairing!r}"
            )
        if (
            self.parallel_boundary_pairing == "characteristic-sat"
            and (
                self.parallel_material_scheme != "production-path"
                or self.parallel_operator_scheme != "fci"
                or self.parallel_velocity_layout != "cell-centered"
                or self.parallel_inflow_closure != "central"
            )
        ):
            raise ValueError(
                "parallel_boundary_pairing='characteristic-sat' requires "
                "the production FCI path with cell-centered velocities and "
                "parallel_inflow_closure='central'"
            )
        if self.parallel_velocity_layout not in ("cell-centered", "fci-staggered"):
            raise ValueError(
                "parallel_velocity_layout must be 'cell-centered' or "
                f"'fci-staggered', got {self.parallel_velocity_layout!r}"
            )
        if (
            self.parallel_flux_pairing == "support-core"
            and self.parallel_operator_scheme != "fci"
        ):
            raise ValueError(
                "parallel_flux_pairing='support-core' requires "
                "parallel_operator_scheme='fci'"
            )
        if (
            self.parallel_flux_pairing == "support-core"
            and self.parallel_velocity_layout != "cell-centered"
        ):
            raise ValueError(
                "parallel_flux_pairing='support-core' requires "
                "parallel_velocity_layout='cell-centered'"
            )
        if (
            self.parallel_material_scheme == "production-path"
            and self.fci_parallel_leg_scheme != "centered"
        ):
            raise ValueError(
                "parallel_material_scheme='production-path' cannot be combined "
                "with fci_parallel_leg_scheme='boundary-characteristic-upwind'"
            )
        if (
            self.parallel_material_scheme == "production-path"
            and self.parallel_inflow_closure != "central"
        ):
            raise ValueError(
                "parallel_material_scheme='production-path' requires "
                "parallel_inflow_closure='central'"
            )
        if self.outgoing_face_topology is not None:
            if not isinstance(self.outgoing_face_topology, LocalOutgoingFciFaceTopology3D):
                raise TypeError("outgoing_face_topology must be LocalOutgoingFciFaceTopology3D or None")
            if self.outgoing_face_topology.layout != self.domain.layout:
                raise ValueError("outgoing_face_topology must share domain.layout")
        if (
            self.parallel_velocity_layout == "fci-staggered"
            and self.outgoing_face_topology is None
        ):
            raise ValueError(
                "parallel_velocity_layout='fci-staggered' requires "
                "outgoing_face_topology"
            )
        if (
            self.parallel_velocity_layout == "fci-staggered"
            and self.parallel_operator_scheme != "fci"
        ):
            raise ValueError(
                "parallel_velocity_layout='fci-staggered' requires "
                "parallel_operator_scheme='fci'"
            )
        if (
            self.parallel_velocity_layout == "fci-staggered"
            and self.fci_parallel_leg_scheme != "centered"
        ):
            raise ValueError(
                "parallel_velocity_layout='fci-staggered' currently requires "
                "fci_parallel_leg_scheme='centered'"
            )
        if self.fci_parallel_leg_scheme not in (
            "centered",
            "boundary-characteristic-upwind",
        ):
            raise ValueError(
                "fci_parallel_leg_scheme must be 'centered' or "
                "'boundary-characteristic-upwind', got "
                f"{self.fci_parallel_leg_scheme!r}"
            )
        if (
            self.parallel_operator_scheme != "fci"
            and self.fci_parallel_leg_scheme != "centered"
        ):
            raise ValueError(
                "fci_parallel_leg_scheme requires parallel_operator_scheme='fci'"
            )
        if self.parallel_short_leg_treatment not in (
            "explicit", "local-backward-euler"
        ):
            raise ValueError(
                "parallel_short_leg_treatment must be 'explicit' or "
                "'local-backward-euler', got "
                f"{self.parallel_short_leg_treatment!r}"
            )
        if not math.isfinite(float(self.parallel_short_leg_cfl_limit)) or float(
            self.parallel_short_leg_cfl_limit
        ) <= 0.0:
            raise ValueError(
                "parallel_short_leg_cfl_limit must be a positive finite number"
            )
        if self.parallel_short_leg_treatment == "local-backward-euler":
            if self.parallel_material_scheme != "production-path":
                raise ValueError(
                    "local-backward-euler short-leg treatment requires "
                    "parallel_material_scheme='production-path'"
                )
            if self.parallel_operator_scheme != "fci":
                raise ValueError(
                    "local-backward-euler short-leg treatment requires "
                    "parallel_operator_scheme='fci'"
                )
            if self.parallel_velocity_layout != "cell-centered":
                raise ValueError(
                    "local-backward-euler short-leg treatment requires "
                    "parallel_velocity_layout='cell-centered'"
                )
        if (
            self.fci_parallel_leg_scheme == "boundary-characteristic-upwind"
            and self.parallel_inflow_closure != "equilibrium-characteristic"
        ):
            raise ValueError(
                "boundary-characteristic-upwind FCI legs require "
                "parallel_inflow_closure='equilibrium-characteristic'"
            )
        if self.parallel_operator_scheme == "fci":
            maps = self.geometry.maps
            # This constructor may run under shard_map/jit.  Map activity is
            # an array payload in that context, so Python bool(jnp.any(...))
            # would trigger concretization.  Keep only structural checks here;
            # the host-side sharding driver validates maps_valid before it
            # enters shard_map.
            if maps.mode not in ("local_halo_only", "remote_dependencies"):
                raise ValueError(
                    "parallel_operator_scheme='fci' requires a valid FCI map mode"
                )
            for direction_name, direction in (
                ("forward", maps.forward),
                ("backward", maps.backward),
            ):
                if direction.local.max_entries < 1:
                    raise ValueError(
                        f"parallel_operator_scheme='fci' requires a nonempty "
                        f"{direction_name} local map table"
                    )
        if isinstance(self.curvature_scale, bool):
            raise ValueError("curvature_scale must be a finite nonnegative scalar")
        try:
            curvature_scale = float(self.curvature_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "curvature_scale must be a finite nonnegative scalar"
            ) from exc
        if not math.isfinite(curvature_scale) or curvature_scale < 0.0:
            raise ValueError("curvature_scale must be a finite nonnegative scalar")
        if self.curvature_scheme not in ("direct", "conservative", "disabled"):
            raise ValueError(
                "curvature_scheme must be 'direct', 'conservative', or "
                "'disabled', got "
                f"{self.curvature_scheme!r}"
            )
        if self.curvature_split_scheme not in ("legacy", "production-path"):
            raise ValueError(
                "curvature_split_scheme must be 'legacy' or 'production-path', "
                f"got {self.curvature_split_scheme!r}"
            )
        if self.curvature_component_diagnostic_scheme not in (
            "directional",
            "centered-dissipation",
            "radial-provenance",
        ):
            raise ValueError(
                "curvature_component_diagnostic_scheme must be 'directional', "
                "'centered-dissipation', or 'radial-provenance', got "
                f"{self.curvature_component_diagnostic_scheme!r}"
            )
        if self.curvature_evolution_component not in (
            "full", "centered-only", "dissipation-only"
        ):
            raise ValueError(
                "curvature_evolution_component must be 'full', 'centered-only', "
                "or 'dissipation-only', got "
                f"{self.curvature_evolution_component!r}"
            )
        if self.curvature_wall_flux_closure not in (
            "equilibrium-exterior-canonical-face-state",
            "bc-characteristic-operator-trace-canonical-face-state",
        ):
            raise ValueError(
                "curvature_wall_flux_closure has invalid value "
                f"{self.curvature_wall_flux_closure!r}"
            )
        if (
            self.curvature_wall_flux_closure
            == "bc-characteristic-operator-trace-canonical-face-state"
            and self.curvature_split_scheme != "production-path"
        ):
            raise ValueError(
                "the BC-characteristic curvature wall closure requires "
                "curvature_split_scheme='production-path'"
            )
        if self.curvature_radial_ablation not in (
            "none",
            "upper-physical-face",
            "rlp-transition-faces",
            "ordinary-interior-faces",
            "last-interior-face",
            "within-cell-path",
        ):
            raise ValueError(
                "curvature_radial_ablation has invalid value "
                f"{self.curvature_radial_ablation!r}"
            )
        if (
            self.curvature_radial_ablation != "none"
            and self.curvature_split_scheme != "production-path"
        ):
            raise ValueError(
                "curvature radial ablations require "
                "curvature_split_scheme='production-path'"
            )
        if (
            self.curvature_evolution_component != "full"
            and self.curvature_split_scheme != "production-path"
        ):
            raise ValueError(
                "curvature_evolution_component='centered-only' or "
                "'dissipation-only' requires "
                "curvature_split_scheme='production-path'"
            )
        if (
            self.curvature_component_diagnostic_scheme
            in ("centered-dissipation", "radial-provenance")
            and self.curvature_split_scheme != "production-path"
        ):
            raise ValueError(
                "production curvature diagnostics require "
                "curvature_split_scheme='production-path'"
            )
        if self.curvature_split_scheme == "production-path":
            if self.curvature_scheme != "conservative":
                raise ValueError(
                    "curvature_split_scheme='production-path' requires "
                    "curvature_scheme='conservative'"
                )
            if self.curvature_face_coefficients is None:
                raise ValueError(
                    "curvature_split_scheme='production-path' requires "
                    "curvature_face_coefficients"
                )
        if self.curvature_rlp_face_scheme not in (
            "projected-fine",
            "moment-shared",
            "bounded-moment-shared",
            "constrained-flux-shared",
            "fine-glue-sat",
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            raise ValueError(
                "curvature_rlp_face_scheme must be 'projected-fine', "
                "'moment-shared', 'bounded-moment-shared', or "
                "'constrained-flux-shared', 'fine-glue-sat', "
                "'fine-glue-characteristic', or "
                "'fine-glue-characteristic-bulk'"
            )
        if self.curvature_radial_principal_face_scheme not in (
            "centered",
            "donor-cell",
        ):
            raise ValueError(
                "curvature_radial_principal_face_scheme must be 'centered' or "
                f"'donor-cell', got "
                f"{self.curvature_radial_principal_face_scheme!r}"
            )
        if self.curvature_radial_characteristic_scheme not in (
            "legacy",
            "third-order-upwind",
        ):
            raise ValueError(
                "curvature_radial_characteristic_scheme must be 'legacy' or "
                f"'third-order-upwind', got "
                f"{self.curvature_radial_characteristic_scheme!r}"
            )
        if self.curvature_poloidal_characteristic_scheme not in (
            "legacy",
            "third-order-upwind",
        ):
            raise ValueError(
                "curvature_poloidal_characteristic_scheme must be 'legacy' or "
                f"'third-order-upwind', got "
                f"{self.curvature_poloidal_characteristic_scheme!r}"
            )
        if self.curvature_radial_characteristic_scheme == "third-order-upwind":
            if self.curvature_scheme != "conservative":
                raise ValueError(
                    "third-order radial characteristic curvature requires "
                    "curvature_scheme='conservative'"
                )
            if self.control_volume_geometry is None:
                raise ValueError(
                    "third-order radial characteristic curvature requires "
                    "RLP control-volume geometry"
                )
            if self.curvature_radial_principal_face_scheme != "centered":
                raise ValueError(
                    "third-order radial characteristic curvature requires "
                    "the centered radial principal face scheme"
                )
            if self.curvature_rlp_face_scheme in (
                "fine-glue-characteristic",
                "fine-glue-characteristic-bulk",
            ) or self.curvature_characteristic_axes != "legacy":
                raise ValueError(
                    "third-order radial characteristic curvature is mutually "
                    "exclusive with the legacy characteristic correction"
                )
            if frozenset(self.curvature_equations) != frozenset(
                ("density", "Te", "Ti", "vorticity")
            ):
                raise ValueError(
                    "third-order radial characteristic curvature requires all "
                    "four coupled curvature equations"
                )
        if self.curvature_poloidal_characteristic_scheme == "third-order-upwind":
            # The poloidal principal correction shares the exact production
            # guards of radial third-order upwinding.  It intentionally does
            # not consult the legacy characteristic-axis/penalty selectors.
            if self.curvature_radial_characteristic_scheme != "third-order-upwind":
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "curvature_radial_characteristic_scheme='third-order-upwind'"
                )
            if self.curvature_scheme != "conservative":
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "curvature_scheme='conservative'"
                )
            if self.control_volume_geometry is None:
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "RLP control-volume geometry"
                )
            if self.curvature_rlp_face_scheme != "projected-fine":
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "curvature_rlp_face_scheme='projected-fine'"
                )
            if self.curvature_radial_principal_face_scheme != "centered":
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "the centered radial principal face scheme"
                )
            if self.curvature_characteristic_axes != "legacy":
                raise ValueError(
                    "third-order poloidal characteristic curvature requires "
                    "legacy curvature_characteristic_axes"
                )
            if frozenset(self.curvature_equations) != frozenset(
                ("density", "Te", "Ti", "vorticity")
            ):
                raise ValueError(
                    "third-order poloidal characteristic curvature requires all "
                    "four coupled curvature equations"
                )
            if not self.domain.periodic_axes[1]:
                raise ValueError(
                    "third-order poloidal characteristic curvature requires periodic theta"
                )
            if int(self.domain.shard_spec.shard_counts[1]) != 1:
                raise ValueError(
                    "third-order poloidal characteristic curvature requires an unsharded theta axis"
                )
        characteristic_axes = str(self.curvature_characteristic_axes)
        if characteristic_axes not in ("legacy", "radial", "radial-poloidal"):
            raise ValueError(
                "curvature_characteristic_axes must be 'legacy', 'radial', or "
                f"'radial-poloidal', got {characteristic_axes!r}"
            )
        if characteristic_axes != "legacy" and self.curvature_rlp_face_scheme not in (
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            raise ValueError(
                "curvature_characteristic_axes requires a fine-glue characteristic "
                "curvature face scheme"
            )
        if self.curvature_rlp_face_scheme in (
            "moment-shared",
            "bounded-moment-shared",
            "constrained-flux-shared",
        ):
            if self.curvature_scheme != "conservative":
                raise ValueError(
                    "moment-shared RLP curvature requires "
                    "curvature_scheme='conservative'"
                )
            if self.control_volume_geometry is None:
                raise ValueError(
                    "moment-shared RLP curvature requires control-volume geometry"
                )
            if self.control_volume_geometry.face_functionals is None:
                raise ValueError(
                    "moment-shared RLP curvature requires compiled face functionals"
                )
        try:
            fine_glue_penalty = float(self.curvature_rlp_fine_glue_penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "curvature_rlp_fine_glue_penalty must be finite and nonnegative"
            ) from exc
        if not math.isfinite(fine_glue_penalty) or fine_glue_penalty < 0.0:
            raise ValueError(
                "curvature_rlp_fine_glue_penalty must be finite and nonnegative"
            )
        poloidal_penalty = getattr(self, "poloidal_characteristic_penalty", None)
        if poloidal_penalty is not None:
            try:
                poloidal_penalty = float(poloidal_penalty)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "poloidal_characteristic_penalty must be finite and nonnegative"
                ) from exc
            if not math.isfinite(poloidal_penalty) or poloidal_penalty < 0.0:
                raise ValueError(
                    "poloidal_characteristic_penalty must be finite and nonnegative"
                )
        if self.curvature_rlp_face_scheme in (
            "fine-glue-sat",
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            if self.curvature_scheme != "conservative":
                raise ValueError(
                    "fine-glue RLP curvature requires "
                    "curvature_scheme='conservative'"
                )
            if self.control_volume_geometry is None:
                raise ValueError(
                    "fine-glue RLP curvature requires control-volume geometry"
                )
            if (
                self.curvature_rlp_face_scheme
                in ("fine-glue-characteristic", "fine-glue-characteristic-bulk")
                and frozenset(self.curvature_equations)
                != frozenset(("density", "Te", "Ti", "vorticity"))
            ):
                raise ValueError(
                    "fine-glue characteristic curvature requires all four "
                    "coupled curvature equations"
                )
            if self.curvature_rlp_fine_glue_transition_face is not None:
                if self.curvature_rlp_face_scheme == "fine-glue-characteristic-bulk":
                    raise ValueError(
                        "bulk characteristic curvature cannot select one "
                        "fine-glue transition face"
                    )
                face = self.curvature_rlp_fine_glue_transition_face
                if isinstance(face, bool):
                    raise ValueError(
                        "fine-glue transition face must be an integer index"
                    )
                try:
                    face_index = int(face)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "fine-glue transition face must be an integer index"
                    ) from exc
                if face_index != face:
                    raise ValueError(
                        "fine-glue transition face must be an integer index"
                    )
                profile = tuple(
                    int(value)
                    for value in self.control_volume_geometry.angular_group_sizes
                )
                if not 0 < face_index < len(profile):
                    raise ValueError(
                        "fine-glue transition face must be an interior radial "
                        f"face, got {face_index}"
                    )
                if profile[face_index - 1] == profile[face_index]:
                    raise ValueError(
                        f"fine-glue transition face {face_index} is not an "
                        "angular group transition"
                    )
        elif self.curvature_rlp_fine_glue_transition_face is not None:
            raise ValueError(
                "fine-glue transition-face selection requires a fine-glue "
                "curvature face scheme"
            )
        if self.curvature_inflow_closure not in ("central", "upwind-equilibrium"):
            raise ValueError(
                "curvature_inflow_closure must be 'central' or "
                "'upwind-equilibrium', "
                f"got {self.curvature_inflow_closure!r}"
            )
        if self.parallel_inflow_closure not in (
            "central",
            "local-characteristic",
            "equilibrium-characteristic",
        ):
            raise ValueError(
                "parallel_inflow_closure must be 'central' or "
                "'local-characteristic' or 'equilibrium-characteristic', "
                f"got {self.parallel_inflow_closure!r}"
            )
        if self.vorticity_current_inflow_trace not in (
            "operator",
            "parallel-characteristic",
        ):
            raise ValueError(
                "vorticity_current_inflow_trace must be 'operator' or "
                "'parallel-characteristic', got "
                f"{self.vorticity_current_inflow_trace!r}"
            )
        if self.ion_temperature_curvature_self_form not in ("product", "flux"):
            raise ValueError(
                "ion_temperature_curvature_self_form must be 'product' or "
                "'flux', got "
                f"{self.ion_temperature_curvature_self_form!r}"
            )
        if (
            self.ion_temperature_curvature_self_form == "flux"
            and self.curvature_scheme != "conservative"
        ):
            raise ValueError(
                "ion_temperature_curvature_self_form='flux' requires "
                "curvature_scheme='conservative'"
            )
        valid_curvature_equations = {"density", "Te", "Ti", "vorticity"}
        selected_curvature_equations = tuple(self.curvature_equations)
        if len(set(selected_curvature_equations)) != len(
            selected_curvature_equations
        ):
            raise ValueError(
                "curvature_equations must not contain duplicates, got "
                f"{selected_curvature_equations!r}"
            )
        invalid_curvature_equations = set(selected_curvature_equations).difference(
            valid_curvature_equations
        )
        if invalid_curvature_equations:
            raise ValueError(
                "curvature_equations contains invalid equations: "
                f"{sorted(invalid_curvature_equations)!r}"
            )
        if (
            self.curvature_split_scheme == "production-path"
            and frozenset(selected_curvature_equations)
            != frozenset(valid_curvature_equations)
        ):
            raise ValueError(
                "curvature_split_scheme='production-path' requires all four "
                "curvature equations"
            )
        if self.curvature_scheme == "conservative":
            if self.curvature_face_coefficients is None:
                raise ValueError(
                    "curvature_face_coefficients are required for the "
                    "conservative curvature scheme"
                )
            if (
                self.curvature_inflow_closure == "upwind-equilibrium"
                and self.upwind_equilibrium_wall_projectors is None
            ):
                raise ValueError(
                    "upwind_equilibrium_wall_projectors are required for the "
                    "upwind-equilibrium curvature closure"
                )
        elif (
            self.curvature_scheme == "direct"
            and self.curvature_coefficients_owned is None
        ):
            raise ValueError(
                "curvature_coefficients_owned are required for the direct "
                "curvature scheme"
            )

    def project_galerkin_state(self, state: FciDrbEBState) -> FciDrbEBState:
        """Return the canonical owner-space state."""

        return self._owner_state(state)

    def _stencil_builder_context(self) -> StencilBuilderContext:
        return StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
        )

    def _poisson_bracket_over_B(
        self,
        f_gradient,
        g_gradient,
        f_conservative_stencil: ConservativeStencil3D,
        g_conservative_stencil: ConservativeStencil3D,
        *,
        f_boundary_trace: LocalBoundaryFaceTrace3D | None = None,
        g_boundary_trace: LocalBoundaryFaceTrace3D | None = None,
        f_field_halo: jnp.ndarray | None = None,
        g_field_halo: jnp.ndarray | None = None,
        f_field_closure=None,
        g_field_closure=None,
    ) -> jnp.ndarray:
        """Evaluate the selected Poisson bracket with the RHS ``1/B`` included.

        The compatible-flux discretization uses the operator-level physical-wall
        traces.  The direct discretization continues to use gradients built from
        the already closed field halos.
        """

        if self.poisson_bracket_scheme == "compatible-flux":
            return local_poisson_bracket_compatible_flux_op(
                f_conservative_stencil,
                g_conservative_stencil,
                self.geometry,
                domain=self.domain,
                axis_regular_axes=self.axis_regular_axes,
                f_boundary_trace=f_boundary_trace,
                g_boundary_trace=g_boundary_trace,
                control_volume_geometry=(
                    self.control_volume_geometry
                    if self._uses_compact_face_operators else None
                ),
                f_field_closure=(
                    f_field_closure
                    if f_field_closure is not None
                    else self._cv_closure(f_field_halo, None)
                    if self._uses_compact_face_operators
                    else None
                ),
                g_field_closure=(
                    g_field_closure
                    if g_field_closure is not None
                    else self._cv_closure(g_field_halo, None)
                    if self._uses_compact_face_operators
                    else None
                ),
            )
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        return local_poisson_bracket_op_from_gradients(
            f_gradient,
            g_gradient,
            self.geometry,
        ) / bmag

    def _face_bcs(self, state_owned: FciDrbEBState) -> LocalFciDrbEBFaceBCBundle:
        return self.face_bc_builder(
            state_owned,
            self.geometry,
            self.domain,
            self.parameters,
        )

    def _prepare_scalar_halo(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        values_owned = self._owner_field(values_owned)
        if self.control_volume_geometry is not None:
            values_storage = expand_local_control_volume_owner_field(
                values_owned, self.control_volume_geometry.cells
            )
        else:
            values_storage = values_owned
        field_halo = inject_owned_field_to_halo(
            values_storage,
            self.domain.layout,
        )
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(field_halo, self.domain, face_bc)

    def _project_fine_center_to_cell_rlp(self, values_fine: jnp.ndarray) -> jnp.ndarray:
        """Return ``P_c R_c`` of a fine source-cell field.

        Outgoing-face velocity storage has more endpoint-support degrees of
        freedom than the angular cell RLP space.  Any quantity that is about
        to enter a cell-centred (perpendicular or cell-gradient) operator
        must first discard that unresolved source-subface component.  This is
        deliberately a cell projection, not an ``R_e/P_e`` face filter.
        """

        values = jnp.asarray(values_fine, dtype=jnp.float64)
        if self.control_volume_geometry is None:
            return values
        owners = self._owner_field(self._restrict_fine_field(values))
        return expand_local_control_volume_owner_field(
            owners, self.control_volume_geometry.cells
        )

    def _prepare_cell_rlp_halo_from_fine(
        self,
        values_fine: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        """Project a fine centre field with ``P_cR_c`` and close its halo."""

        return self._prepare_fine_storage_halo(
            self._project_fine_center_to_cell_rlp(values_fine), face_bc
        )

    def _prepare_face_halo(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        """Prolong outgoing-edge owners, then apply the ordinary field halo closure.

        The array shape deliberately remains cell-shaped: an entry represents
        the outgoing mapped FCI edge owned by that source-cell slot.  Only the
        ownership/prolongation is different in this integration slice; the
        existing parallel and perpendicular operators keep their current
        semantics.
        """

        topology = self.outgoing_face_topology
        if topology is None:
            raise RuntimeError("missing outgoing_face_topology for face storage")
        owners = self._owner_face_field(values_owned)
        fine = prolong_local_outgoing_fci_face_owner_field(owners, topology)
        field_halo = inject_owned_field_to_halo(fine, self.domain.layout)
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(field_halo, self.domain, face_bc)

    def _prepare_fine_storage_halo(
        self,
        values_fine_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        """Close a fine-grid derived field without RLP owner projection.

        Staggered source-edge products are fine storage values even when the
        primary state uses projected owners, so routing them through
        :meth:`_prepare_scalar_halo` would incorrectly collapse aliases.
        """
        field_halo = inject_owned_field_to_halo(values_fine_owned, self.domain.layout)
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(field_halo, self.domain, face_bc)

    def _prepare_state_halo(self, state_owned, face_bc):
        state_owned = self._owner_state(state_owned)
        if (
            self.control_volume_geometry is None
            and self.parallel_velocity_layout != "fci-staggered"
        ):
            return prepare_local_fci_drb_eb_state(
                state_owned, self.domain, face_bc=face_bc,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
                physical_ghost_filler=self.physical_ghost_filler,
            )
        return state_owned.replace(
            density=self._prepare_scalar_halo(state_owned.density, face_bc.density),
            phi=self._prepare_scalar_halo(state_owned.phi, face_bc.phi),
            Te=self._prepare_scalar_halo(state_owned.Te, face_bc.Te),
            Ti=self._prepare_scalar_halo(state_owned.Ti, face_bc.Ti),
            Vi=(self._prepare_face_halo(state_owned.Vi, face_bc.Vi)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._prepare_scalar_halo(state_owned.Vi, face_bc.Vi)),
            Ve=(self._prepare_face_halo(state_owned.Ve, face_bc.Ve)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._prepare_scalar_halo(state_owned.Ve, face_bc.Ve)),
            vorticity=self._prepare_scalar_halo(state_owned.vorticity, face_bc.vorticity),
        )

    def _owner_field(self, values: jnp.ndarray) -> jnp.ndarray:
        """Return a cell-owner field; merged cell aliases are never evolved."""
        values = jnp.asarray(values, dtype=jnp.float64)
        if self.control_volume_geometry is None:
            return values
        cells = self.control_volume_geometry.cells
        return jnp.where(cells.is_active_owner, values, 0.0)

    def _owner_face_field(self, values: jnp.ndarray) -> jnp.ndarray:
        """Return an owner-only outgoing-FCI-face field."""

        values = jnp.asarray(values, dtype=jnp.float64)
        topology = self.outgoing_face_topology
        if topology is None:
            return values
        return jnp.where(topology.is_active_owner, values, 0.0)

    def _owner_state(self, state: FciDrbEBState) -> FciDrbEBState:
        if (
            self.control_volume_geometry is None
            and self.parallel_velocity_layout != "fci-staggered"
        ):
            return state
        return state.replace(
            density=self._owner_field(state.density),
            phi=self._owner_field(state.phi),
            Te=self._owner_field(state.Te),
            Ti=self._owner_field(state.Ti),
            Vi=(self._owner_face_field(state.Vi)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._owner_field(state.Vi)),
            Ve=(self._owner_face_field(state.Ve)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._owner_field(state.Ve)),
            vorticity=self._owner_field(state.vorticity),
        )

    def _owner_result(self, result: jnp.ndarray) -> jnp.ndarray:
        return self._owner_field(result)

    def _cv_closure(self, field_halo: jnp.ndarray, face_bc: LocalBoundaryFaceBC3D):
        if not self._uses_compact_face_operators:
            return None
        cvbc = self.control_volume_boundary_bc
        if cvbc is None:
            raise RuntimeError("missing compact control-volume boundary data")
        return build_local_control_volume_field_closure(
            field_halo, self.control_volume_geometry, cvbc, domain=self.domain
        )

    def _curvature_cv_closure(self, field_halo: jnp.ndarray):
        if not self._uses_curvature_compact_faces:
            return None
        cvbc = self.control_volume_boundary_bc
        if cvbc is None:
            raise RuntimeError("missing compact curvature boundary data")
        return build_local_control_volume_field_closure(
            field_halo,
            self.control_volume_geometry,
            cvbc,
            domain=self.domain,
        )

    def _primitive_curvature_cv_closures(
        self,
        state_halo: FciDrbEBState,
        names: tuple[str, ...],
    ) -> dict[str, object]:
        if not self._uses_curvature_compact_faces:
            return {}
        return {
            name: self._curvature_cv_closure(getattr(state_halo, name))
            for name in names
        }

    def _primitive_cv_closures(
        self,
        state_halo: FciDrbEBState,
        names: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """Build selected primitive compact-face traces once for one RHS stage."""
        if not self._uses_compact_face_operators:
            return {}
        if names is None:
            names = ("density", "Te", "Ti", "Vi", "Ve", "phi", "vorticity")
        allowed = frozenset(("density", "Te", "Ti", "Vi", "Ve", "phi", "vorticity"))
        if len(set(names)) != len(names) or any(name not in allowed for name in names):
            raise ValueError("primitive closure names must be unique primitive state fields")
        return {
            name: self._cv_closure(getattr(state_halo, name), None)
            for name in names
        }

    def _cv_linear_combination(self, left, right, *, a=1.0, b=1.0):
        if left is None or right is None:
            return None
        return linear_combination_local_control_volume_closures(left, right, a=a, b=b)

    def _cv_product(self, left, right):
        if left is None or right is None:
            return None
        return product_local_control_volume_closures(
            left, right, self.control_volume_geometry
        )

    def _cv_operator_args(self, field_halo: jnp.ndarray, face_bc=None, *, field_closure=None) -> dict:
        if not self._uses_compact_face_operators:
            return {}
        if field_halo is None or not hasattr(field_halo, "shape"):
            raise ValueError("compact control-volume operators require the source field halo")
        return {
            "control_volume_geometry": self.control_volume_geometry,
            "field_closure": (
                self._cv_closure(field_halo, face_bc)
                if field_closure is None else field_closure
            ),
        }

    def _curvature_cv_operator_args(
        self,
        field_halo: jnp.ndarray,
        *,
        field_closure=None,
    ) -> dict:
        if self.curvature_rlp_face_scheme == "fine-glue-sat":
            return {
                "control_volume_geometry": self.control_volume_geometry,
                "use_fine_glue_transition_flux": True,
                "fine_glue_penalty": self.curvature_rlp_fine_glue_penalty,
                "fine_glue_transition_face": (
                    self.curvature_rlp_fine_glue_transition_face
                ),
            }
        if not self._uses_curvature_compact_faces:
            return {}
        if field_halo is None or not hasattr(field_halo, "shape"):
            raise ValueError(
                "compact curvature requires the source field halo"
            )
        return {
            "control_volume_geometry": self.control_volume_geometry,
            "field_closure": (
                self._curvature_cv_closure(field_halo)
                if field_closure is None
                else field_closure
            ),
            "replace_regular_transition_traces": True,
            "bound_regular_transition_trace_correction": (
                self.curvature_rlp_face_scheme == "bounded-moment-shared"
            ),
            "constrain_regular_transition_flux_correction": (
                self.curvature_rlp_face_scheme == "constrained-flux-shared"
            ),
        }

    @property
    def _uses_compact_face_operators(self) -> bool:
        """Compact pole-face operators are not part of the production path."""

        return False

    @property
    def _uses_curvature_compact_faces(self) -> bool:
        return self.curvature_rlp_face_scheme in (
            "moment-shared",
            "bounded-moment-shared",
            "constrained-flux-shared",
        )

    @property
    def _uses_projected_fine_grid(self) -> bool:
        return self.control_volume_geometry is not None

    def _restrict_fine_field(self, value: jnp.ndarray) -> jnp.ndarray:
        """Apply R to a completed fine-grid scalar operator result."""

        if not self._uses_projected_fine_grid:
            return value
        if self.control_volume_geometry is None:
            raise RuntimeError("projected-fine-grid mode is missing control-volume geometry")
        return aggregate_local_control_volume_average(
            value,
            self.control_volume_geometry.cells,
            self.domain,
        )

    def _restrict_fine_face_field(self, value: jnp.ndarray) -> jnp.ndarray:
        """Apply R_e to a completed fine outgoing-edge result."""

        topology = self.outgoing_face_topology
        if topology is None:
            return value
        return restrict_local_outgoing_fci_face_field(value, topology)

    def _restrict_fine_state(self, state: FciDrbEBState) -> FciDrbEBState:
        """Restrict every assembled fine-grid RHS leaf to owner space."""

        if not self._uses_projected_fine_grid and self.parallel_velocity_layout != "fci-staggered":
            return state
        return state.replace(
            density=self._restrict_fine_field(state.density),
            phi=self._restrict_fine_field(state.phi),
            Te=self._restrict_fine_field(state.Te),
            Ti=self._restrict_fine_field(state.Ti),
            Vi=(self._restrict_fine_face_field(state.Vi)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._restrict_fine_field(state.Vi)),
            Ve=(self._restrict_fine_face_field(state.Ve)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._restrict_fine_field(state.Ve)),
            vorticity=self._restrict_fine_field(state.vorticity),
        )

    def _prepare_phi_halo(
        self,
        phi_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        return self._prepare_scalar_halo(phi_owned, face_bc)

    def _conservative_curvature(
        self,
        conservative_stencil: ConservativeStencil3D,
        scalar_face_bc: LocalBoundaryFaceBC3D | None = None,
        *,
        boundary_trace: LocalBoundaryFaceTrace3D | None = None,
        field_halo: jnp.ndarray | None = None,
        field_closure=None,
    ) -> jnp.ndarray:
        """Evaluate centered curvature, optionally with wall-only coupled traces."""

        if self.curvature_face_coefficients is None:
            raise ValueError(
                "curvature_face_coefficients are required for conservative "
                "curvature evaluation"
            )
        return local_curvature_conservative_op(
            conservative_stencil,
            self.geometry,
            self.curvature_face_coefficients,
            domain=self.domain,
            face_bc=scalar_face_bc,
            boundary_trace=boundary_trace,
            radial_principal_face_scheme=(
                self.curvature_radial_principal_face_scheme
            ),
            axis_regular_axes=self.axis_regular_axes,
            **self._curvature_cv_operator_args(
                field_halo, field_closure=field_closure
            ),
        )

    def _conservative_curvature_components(
        self,
        conservative_stencil: ConservativeStencil3D,
        scalar_face_bc: LocalBoundaryFaceBC3D | None = None,
        *,
        boundary_trace: LocalBoundaryFaceTrace3D | None = None,
        field_halo: jnp.ndarray | None = None,
        field_closure=None,
    ) -> jnp.ndarray:
        """Diagnostic ``(u, theta, eta)`` split of production curvature."""

        if self.curvature_face_coefficients is None:
            raise ValueError(
                "curvature_face_coefficients are required for conservative "
                "curvature evaluation"
            )
        return local_curvature_conservative_components_op(
            conservative_stencil,
            self.geometry,
            self.curvature_face_coefficients,
            domain=self.domain,
            face_bc=scalar_face_bc,
            boundary_trace=boundary_trace,
            radial_principal_face_scheme=(
                self.curvature_radial_principal_face_scheme
            ),
            axis_regular_axes=self.axis_regular_axes,
            **self._curvature_cv_operator_args(
                field_halo, field_closure=field_closure
            ),
        )

    @cached_property
    def _radial_characteristic_face_penalty(self) -> jnp.ndarray:
        """Cache the static radial coupled characteristic face block."""

        bmag_face = jnp.asarray(
            self.geometry.face_bfield.x.Bmag_owned, dtype=jnp.float64
        )[1:-1]
        return background_curvature_characteristic_absolute_matrix(
            bmag_face, self.parameters.tau
        )

    @cached_property
    def _poloidal_characteristic_face_penalty(self) -> jnp.ndarray:
        """Cache the static ordinary-theta coupled characteristic face block."""

        bmag_face = jnp.asarray(
            self.geometry.face_bfield.y.Bmag_owned, dtype=jnp.float64
        )
        return background_curvature_characteristic_absolute_matrix(
            bmag_face, self.parameters.tau
        )

    def _fine_glue_characteristic_curvature_correction(
        self,
        stencils: tuple[
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
        ],
        *,
        tau: float | jnp.ndarray,
        face_penalty: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the coupled H-compatible ``|M|`` RLP jump correction."""

        if self.curvature_rlp_face_scheme not in (
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            zero = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
            return zero, zero, zero, zero
        if self.curvature_characteristic_axes not in ("legacy", "radial", "radial-poloidal"):
            raise ValueError("invalid curvature characteristic axis selector")
        if self.control_volume_geometry is None:
            raise RuntimeError("characteristic fine glue is missing RLP geometry")
        if self.curvature_face_coefficients is None:
            raise RuntimeError("characteristic fine glue is missing curvature faces")

        q_cell = jnp.stack(
            tuple(jnp.asarray(stencil.x.center, dtype=jnp.float64) for stencil in stencils),
            axis=-1,
        )
        if face_penalty is None:
            bmag_face = jnp.asarray(
                self.geometry.face_bfield.x.Bmag_owned, dtype=jnp.float64
            )[1:-1]
            face_penalty = background_curvature_characteristic_absolute_matrix(
                bmag_face, tau
            )
        correction = _radial_characteristic_fine_glue_owner_correction(
            self.curvature_face_coefficients.x,
            q_cell,
            face_penalty,
            self.geometry,
            self.control_volume_geometry,
            penalty=(
                self.curvature_scale * self.curvature_rlp_fine_glue_penalty
            ),
            transition_face=self.curvature_rlp_fine_glue_transition_face,
            include_ordinary_faces=(
                self.curvature_rlp_face_scheme == "fine-glue-characteristic-bulk"
            ),
        )
        return tuple(jnp.moveaxis(correction, -1, 0))

    def _third_order_radial_characteristic_curvature_correction(
        self,
        stencils: tuple[
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
        ],
        *,
        tau: float | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the canonical live radial characteristic flux correction."""

        if self.curvature_radial_characteristic_scheme != "third-order-upwind":
            zero = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
            return zero, zero, zero, zero
        if self.control_volume_geometry is None:
            raise RuntimeError("third-order radial characteristic correction is missing RLP geometry")
        if self.curvature_face_coefficients is None:
            raise RuntimeError("third-order radial characteristic correction is missing curvature faces")
        correction = _radial_characteristic_third_order_owner_correction(
            self.curvature_face_coefficients.x,
            stencils,
            self.geometry,
            self.control_volume_geometry,
            tau=tau,
            curvature_scale=self.curvature_scale,
        )
        return tuple(jnp.moveaxis(correction, -1, 0))

    def _third_order_poloidal_characteristic_curvature_correction(
        self,
        stencils: tuple[
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
        ],
        *,
        tau: float | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the canonical live owner-lattice theta flux correction."""

        if getattr(self, "curvature_poloidal_characteristic_scheme", "legacy") != (
            "third-order-upwind"
        ):
            zero = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
            return zero, zero, zero, zero
        if self.control_volume_geometry is None:
            raise RuntimeError(
                "third-order poloidal characteristic correction is missing RLP geometry"
            )
        if self.curvature_face_coefficients is None:
            raise RuntimeError(
                "third-order poloidal characteristic correction is missing curvature faces"
            )
        correction = _poloidal_characteristic_third_order_owner_correction(
            self.curvature_face_coefficients.y,
            stencils,
            self.geometry,
            self.control_volume_geometry,
            self.domain,
            tau=tau,
            curvature_scale=self.curvature_scale,
        )
        return tuple(jnp.moveaxis(correction, -1, 0))

    def _poloidal_characteristic_curvature_correction(
        self,
        stencils: tuple[
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
            ConservativeStencil3D,
        ],
        *,
        tau: float | jnp.ndarray,
        face_penalty: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the opt-in ordinary theta-face coupled correction."""

        zero = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        if self.curvature_characteristic_axes != "radial-poloidal":
            return zero, zero, zero, zero
        if self.curvature_rlp_face_scheme not in (
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            return zero, zero, zero, zero
        if self.control_volume_geometry is None:
            raise RuntimeError("poloidal characteristic correction is missing RLP geometry")
        if self.curvature_face_coefficients is None:
            raise RuntimeError("poloidal characteristic correction is missing curvature faces")
        if face_penalty is None:
            bmag_face = jnp.asarray(
                self.geometry.face_bfield.y.Bmag_owned, dtype=jnp.float64
            )
            face_penalty = background_curvature_characteristic_absolute_matrix(
                bmag_face, tau
            )
        correction = _poloidal_characteristic_owner_correction(
            self.curvature_face_coefficients.y,
            stencils,
            face_penalty,
            self.geometry,
            self.control_volume_geometry,
            self.domain,
            penalty=self.curvature_scale * (
                self.curvature_rlp_fine_glue_penalty
                if getattr(self, "poloidal_characteristic_penalty", None) is None
                else self.poloidal_characteristic_penalty
            ),
        )
        return tuple(jnp.moveaxis(correction, -1, 0))

    def _production_curvature_rhs_contributions(
        self,
        *,
        state_halo,
        context,
        density,
        Te,
        Ti,
        bmag,
        tau,
        density_conservative_stencil,
        Te_conservative_stencil,
        Ti_conservative_stencil,
        vorticity_conservative_stencil,
        operator_boundary,
        return_directional_components: bool = False,
    ):
        """Assemble the all-axis production characteristic curvature split.

        The material four-field path is evaluated once on every coordinate
        owner face.  The only term outside that local wave-propagation update
        is the compatible nonlocal-potential remainder, with coefficient
        vector ``(-2n, -4Te/3, -4Ti/3, 0)/B``.
        """
        evolution_component = self.curvature_evolution_component
        request_split_diagnostics = (
            return_directional_components or evolution_component != "full"
        )
        coupled = (
            density_conservative_stencil,
            Te_conservative_stencil,
            Ti_conservative_stencil,
            vorticity_conservative_stencil,
        )
        material_result = local_curvature_production_path_op(
            coupled,
            self.geometry,
            self.curvature_face_coefficients,
            tau=tau,
            domain=self.domain,
            control_volume_geometry=self.control_volume_geometry,
            equilibrium=jnp.asarray(
                (self.parameters.n0, self.parameters.Te0, self.parameters.Ti0, 0.0),
                dtype=jnp.float64,
            ),
            boundary_traces=(
                operator_boundary.density,
                operator_boundary.Te,
                operator_boundary.Ti,
                operator_boundary.vorticity,
            ),
            wall_flux_closure=self.curvature_wall_flux_closure,
            radial_ablation=self.curvature_radial_ablation,
            # Keep the production default on the original fast path.  The
            # diagnostic lanes are only materialized for an explicit request
            # or for one of the component-evolution ablations.
            return_diagnostics=request_split_diagnostics,
        )
        if request_split_diagnostics:
            material, diagnostics = material_result
        else:
            material = material_result
            diagnostics = None
        psi_halo = state_halo.phi + tau * state_halo.Ti
        psi_stencil = build_local_conservative_stencil_from_field(
            psi_halo, self.geometry, context
        )
        psi_trace = _combine_operator_traces(
            operator_boundary.phi,
            operator_boundary.Ti,
            operation=lambda phi, ti: phi + tau * ti,
        )
        psi_curvature = self._conservative_curvature(
            psi_stencil,
            boundary_trace=psi_trace,
            field_halo=psi_halo,
        )
        remainder_coeff = jnp.stack(
            (
                -2.0 * density / jnp.maximum(bmag, 1.0e-30),
                -4.0 * Te / (3.0 * jnp.maximum(bmag, 1.0e-30)),
                -4.0 * Ti / (3.0 * jnp.maximum(bmag, 1.0e-30)),
                jnp.zeros_like(density),
            ),
            axis=-1,
        )
        remainder = remainder_coeff * psi_curvature[..., None]
        # Match the legacy selector semantics: curvature_scale multiplies the
        # complete production split, including the nonlocal psi remainder.
        result = self.curvature_scale * (material + remainder)
        if not request_split_diagnostics:
            return tuple(jnp.moveaxis(result, -1, 0))
        assert diagnostics is not None
        psi_directional = self._conservative_curvature_components(
            psi_stencil,
            boundary_trace=psi_trace,
            field_halo=psi_halo,
        )
        remainder_directional = (
            remainder_coeff[..., None, :] * jnp.moveaxis(psi_directional, 0, -1)[..., None]
        )
        # psi_directional is (3, nx, ny, nz); move the component axis to the
        # final position before adding the four-field material diagnostics.
        remainder_directional = jnp.moveaxis(remainder_directional, -2, 0)
        if evolution_component == "centered-only":
            directional = self.curvature_scale * (
                diagnostics["directional_centered_transfer"]
                + remainder_directional
            )
        elif evolution_component == "dissipation-only":
            directional = self.curvature_scale * diagnostics[
                "directional_characteristic_dissipation"
            ]
        elif (
            self.curvature_component_diagnostic_scheme
            == "centered-dissipation"
        ):
            centered_directional = (
                diagnostics["directional_centered_transfer"]
                + remainder_directional
            )
            dissipative_directional = diagnostics[
                "directional_characteristic_dissipation"
            ]
            directional = self.curvature_scale * jnp.concatenate(
                (centered_directional, dissipative_directional), axis=0
            )
        elif self.curvature_component_diagnostic_scheme == "radial-provenance":
            # The material vorticity row has no psi remainder, so these lanes
            # directly resolve the radial source identified by the modal audit.
            # For the thermodynamic rows retain the nonlocal radial remainder
            # as its own lane so the eight-lane sum still closes exactly.
            radial_material = diagnostics["radial_provenance_residual"]
            angular_total = (
                diagnostics["directional_residual"][1:]
                + remainder_directional[1:]
            )
            directional = self.curvature_scale * jnp.concatenate(
                (
                    radial_material,
                    remainder_directional[0:1],
                    angular_total,
                ),
                axis=0,
            )
        else:
            directional = self.curvature_scale * (
                diagnostics["directional_residual"] + remainder_directional
            )
        if not return_directional_components:
            return tuple(jnp.moveaxis(jnp.sum(directional, axis=0), -1, 0))
        return tuple(jnp.moveaxis(directional, -1, 0))

    def _curvature_rhs_contributions(
        self,
        *,
        state_halo,
        face_bc,
        context,
        density,
        Te,
        Ti,
        bmag,
        density_safe,
        tau,
        Pe_face_bc,
        pressure_face_bc,
        operator_boundary,
        Pe_gradient,
        pressure_gradient,
        phi_gradient,
        Te_gradient,
        Ti_gradient,
        density_conservative_stencil,
        Pe_conservative_stencil,
        pressure_conservative_stencil,
        phi_conservative_stencil,
        Te_conservative_stencil,
        Ti_conservative_stencil,
        vorticity_conservative_stencil,
        primitive_cv_closures: dict[str, object] | None = None,
        return_directional_components: bool = False,
    ):
        """Assemble the production curvature contribution for each equation."""

        if return_directional_components and self.curvature_scheme != "conservative":
            raise ValueError(
                "directional curvature diagnostics require "
                "curvature_scheme='conservative'"
            )

        if self.curvature_split_scheme == "production-path":
            return self._production_curvature_rhs_contributions(
                state_halo=state_halo,
                context=context,
                density=density,
                Te=Te,
                Ti=Ti,
                bmag=bmag,
                tau=tau,
                density_conservative_stencil=density_conservative_stencil,
                Te_conservative_stencil=Te_conservative_stencil,
                Ti_conservative_stencil=Ti_conservative_stencil,
                vorticity_conservative_stencil=vorticity_conservative_stencil,
                operator_boundary=operator_boundary,
                return_directional_components=return_directional_components,
            )
        if self.curvature_scheme == "disabled":
            curvature_Pe = jnp.zeros_like(density)
            curvature_pressure = jnp.zeros_like(density)
            curvature_phi = jnp.zeros_like(density)
            curvature_Te = jnp.zeros_like(density)
            curvature_Ti = jnp.zeros_like(density)
            curvature_Ti_self = jnp.zeros_like(density)
        elif self.curvature_scheme == "conservative":
            assert self.curvature_face_coefficients is not None
            primitive = (
                primitive_cv_closures
                if primitive_cv_closures
                else self._primitive_curvature_cv_closures(
                    state_halo, names=("density", "Te", "Ti")
                )
            )
            Pe_compact_closure = self._cv_product(
                primitive.get("density"), primitive.get("Te")
            ) if primitive else None
            pressure_temperature = self._cv_linear_combination(
                primitive.get("Te"), primitive.get("Ti"), b=tau
            ) if primitive else None
            pressure_compact_closure = self._cv_product(
                primitive.get("density"), pressure_temperature
            ) if primitive else None
            Ti_squared_compact_closure = self._cv_product(
                primitive.get("Ti"), primitive.get("Ti")
            ) if primitive else None
            characteristic_bcs = None
            if self.curvature_inflow_closure == "upwind-equilibrium":
                characteristic_bcs = self._upwind_equilibrium_boundary_face_bcs(
                    (
                        build_local_conservative_stencil_from_field(
                            state_halo.density, self.geometry, context
                        ),
                        Te_conservative_stencil,
                        Ti_conservative_stencil,
                        vorticity_conservative_stencil,
                    ),
                    (face_bc.density, face_bc.Te, face_bc.Ti, face_bc.vorticity),
                )
            if characteristic_bcs is not None:
                (
                    _,
                    Te_wall_bc,
                    Ti_wall_bc,
                    Pe_wall_bc,
                    pressure_wall_bc,
                    Ti2_wall_bc,
                ) = characteristic_bcs
                phi_wall_bc = _dirichlet_face_bc_from_values(
                    _wall_candidate_values(phi_conservative_stencil, face_bc.phi),
                    self.domain.layout,
                    (face_bc.phi.mask_x, face_bc.phi.mask_y, face_bc.phi.mask_z),
                )
            else:
                (
                    Te_wall_bc,
                    Ti_wall_bc,
                    Pe_wall_bc,
                    pressure_wall_bc,
                    Ti2_wall_bc,
                ) = (None, None, None, None, None)
                phi_wall_bc = None

            conservative_operator = (
                self._conservative_curvature_components
                if return_directional_components
                else self._conservative_curvature
            )

            def curvature(conservative_stencil, scalar_face_bc, field_halo, *, field_closure=None):
                closure_kwargs = (
                    {"field_closure": field_closure}
                    if self._uses_curvature_compact_faces
                    else {}
                )
                return conservative_operator(
                    conservative_stencil,
                    scalar_face_bc,
                    field_halo=field_halo,
                    **closure_kwargs,
                )

            def central_curvature(conservative_stencil, trace, field_halo, *, field_closure=None):
                closure_kwargs = (
                    {"field_closure": field_closure}
                    if self._uses_curvature_compact_faces
                    else {}
                )
                return conservative_operator(
                    conservative_stencil,
                    boundary_trace=trace,
                    field_halo=field_halo,
                    **closure_kwargs,
                )

            if Pe_wall_bc is None:
                curvature_Pe = central_curvature(
                    Pe_conservative_stencil, operator_boundary.Pe,
                    state_halo.density * state_halo.Te,
                    field_closure=Pe_compact_closure,
                )
            else:
                curvature_Pe = curvature(
                    Pe_conservative_stencil, Pe_wall_bc,
                    state_halo.density * state_halo.Te,
                    field_closure=Pe_compact_closure,
                )
            if pressure_wall_bc is None:
                curvature_pressure = central_curvature(
                    pressure_conservative_stencil, operator_boundary.pressure,
                    state_halo.density * state_halo.Te + self.parameters.tau * state_halo.density * state_halo.Ti,
                    field_closure=pressure_compact_closure,
                )
            else:
                curvature_pressure = curvature(
                    pressure_conservative_stencil, pressure_wall_bc,
                    state_halo.density * state_halo.Te + self.parameters.tau * state_halo.density * state_halo.Ti,
                    field_closure=pressure_compact_closure,
                )
            if phi_wall_bc is None:
                curvature_phi = central_curvature(
                    phi_conservative_stencil, operator_boundary.phi, state_halo.phi
                )
            else:
                curvature_phi = curvature(
                phi_conservative_stencil, phi_wall_bc, state_halo.phi
                )
            if Te_wall_bc is None:
                curvature_Te = central_curvature(
                    Te_conservative_stencil, operator_boundary.Te, state_halo.Te
                )
            else:
                curvature_Te = curvature(
                Te_conservative_stencil, Te_wall_bc, state_halo.Te
                )
            if Ti_wall_bc is None:
                curvature_Ti = central_curvature(
                    Ti_conservative_stencil, operator_boundary.Ti, state_halo.Ti
                )
            else:
                curvature_Ti = curvature(
                Ti_conservative_stencil, Ti_wall_bc, state_halo.Ti
                )
            if self.ion_temperature_curvature_self_form == "flux":
                Ti_squared_conservative_stencil = (
                    build_local_conservative_stencil_from_field(
                        state_halo.Ti * state_halo.Ti,
                        self.geometry,
                        context,
                    )
                )
                curvature_Ti_self = (
                    curvature(
                        Ti_squared_conservative_stencil, Ti2_wall_bc,
                        state_halo.Ti * state_halo.Ti,
                        field_closure=Ti_squared_compact_closure,
                    )
                    if Ti2_wall_bc is not None else
                    central_curvature(
                        Ti_squared_conservative_stencil,
                        operator_boundary.Ti_squared, state_halo.Ti * state_halo.Ti,
                        field_closure=Ti_squared_compact_closure,
                    )
                )
            else:
                curvature_Ti_self = curvature_Ti
        else:
            assert self.curvature_coefficients_owned is not None
            assert Pe_gradient is not None
            assert pressure_gradient is not None
            assert phi_gradient is not None
            assert Te_gradient is not None
            assert Ti_gradient is not None
            curvature_Pe = local_curvature_op_from_gradient(
                Pe_gradient, self.geometry,
                curvature_coefficients=self.curvature_coefficients_owned,
            )
            curvature_pressure = local_curvature_op_from_gradient(
                pressure_gradient, self.geometry,
                curvature_coefficients=self.curvature_coefficients_owned,
            )
            curvature_phi = local_curvature_op_from_gradient(
                phi_gradient, self.geometry,
                curvature_coefficients=self.curvature_coefficients_owned,
            )
            curvature_Te = local_curvature_op_from_gradient(
                Te_gradient, self.geometry,
                curvature_coefficients=self.curvature_coefficients_owned,
            )
            curvature_Ti = local_curvature_op_from_gradient(
                Ti_gradient, self.geometry,
                curvature_coefficients=self.curvature_coefficients_owned,
            )
            curvature_Ti_self = curvature_Ti

        curvature_density_contribution = (
            self.curvature_scale * (2.0 / bmag) *
            (curvature_Pe - density * curvature_phi)
            if "density" in self.curvature_equations
            else jnp.zeros_like(curvature_Pe)
        )
        curvature_Te_contribution = (
            self.curvature_scale * (4.0 * Te / (3.0 * bmag)) *
            (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
            if "Te" in self.curvature_equations
            else jnp.zeros_like(curvature_Pe)
        )
        if "Ti" in self.curvature_equations:
            if self.ion_temperature_curvature_self_form == "flux":
                curvature_Ti_contribution = self.curvature_scale * (
                    (4.0 * Ti / (3.0 * bmag)) *
                    (curvature_Pe / density_safe - curvature_phi)
                    - (5.0 * tau / (3.0 * bmag)) * curvature_Ti_self
                )
            else:
                curvature_Ti_contribution = self.curvature_scale * (
                    (4.0 * Ti / (3.0 * bmag)) *
                    (curvature_Pe / density_safe - 2.5 * tau * curvature_Ti - curvature_phi)
                )
        else:
            curvature_Ti_contribution = jnp.zeros_like(curvature_Pe)
        curvature_vorticity_contribution = (
            self.curvature_scale * (2.0 * bmag / density_safe) * curvature_pressure
            if "vorticity" in self.curvature_equations
            else jnp.zeros_like(curvature_Pe)
        )
        coupled_curvature_stencils = (
            density_conservative_stencil,
            Te_conservative_stencil,
            Ti_conservative_stencil,
            vorticity_conservative_stencil,
        )
        if self.curvature_radial_characteristic_scheme == "third-order-upwind":
            radial_correction = self._third_order_radial_characteristic_curvature_correction(
                coupled_curvature_stencils,
                tau=tau,
            )
            if getattr(self, "curvature_poloidal_characteristic_scheme", "legacy") == (
                "third-order-upwind"
            ):
                poloidal_correction = (
                    self._third_order_poloidal_characteristic_curvature_correction(
                        coupled_curvature_stencils,
                        tau=tau,
                    )
                )
            else:
                poloidal_correction = tuple(
                    jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
                    for _ in range(4)
                )
        elif self.curvature_rlp_face_scheme in (
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            radial_correction = self._fine_glue_characteristic_curvature_correction(
                coupled_curvature_stencils,
                tau=tau,
                face_penalty=self._radial_characteristic_face_penalty,
            )
            poloidal_correction = self._poloidal_characteristic_curvature_correction(
                coupled_curvature_stencils,
                tau=tau,
                face_penalty=self._poloidal_characteristic_face_penalty,
            )
        if (
            self.curvature_radial_characteristic_scheme == "third-order-upwind"
            or self.curvature_rlp_face_scheme
            in ("fine-glue-characteristic", "fine-glue-characteristic-bulk")
        ):
            if return_directional_components:
                (
                    curvature_density_contribution,
                    curvature_Te_contribution,
                    curvature_Ti_contribution,
                    curvature_vorticity_contribution,
                ) = tuple(
                    value.at[0].add(correction)
                    for value, correction in zip(
                        (
                            curvature_density_contribution,
                            curvature_Te_contribution,
                            curvature_Ti_contribution,
                            curvature_vorticity_contribution,
                        ),
                        radial_correction,
                    )
                )
                (
                    curvature_density_contribution,
                    curvature_Te_contribution,
                    curvature_Ti_contribution,
                    curvature_vorticity_contribution,
                ) = tuple(
                    value.at[1].add(correction)
                    for value, correction in zip(
                        (
                            curvature_density_contribution,
                            curvature_Te_contribution,
                            curvature_Ti_contribution,
                            curvature_vorticity_contribution,
                        ),
                        poloidal_correction,
                    )
                )
            else:
                (
                    curvature_density_contribution,
                    curvature_Te_contribution,
                    curvature_Ti_contribution,
                    curvature_vorticity_contribution,
                ) = tuple(
                    value + correction
                    for value, correction in zip(
                        (
                            curvature_density_contribution,
                            curvature_Te_contribution,
                            curvature_Ti_contribution,
                            curvature_vorticity_contribution,
                        ),
                        tuple(
                            radial + poloidal
                            for radial, poloidal in zip(
                                radial_correction, poloidal_correction
                            )
                        ),
                    )
                )
        return (
            curvature_density_contribution,
            curvature_Te_contribution,
            curvature_Ti_contribution,
            curvature_vorticity_contribution,
        )

    def _upwind_equilibrium_boundary_face_bcs(
        self,
        stencils: tuple[ConservativeStencil3D, ConservativeStencil3D, ConservativeStencil3D, ConservativeStencil3D],
        face_bcs: tuple[LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D],
    ) -> tuple[LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D, LocalBoundaryFaceBC3D]:
        """Build upwind equilibrium boundary flux traces around U0.

        The returned BC payloads are for (n, Te, Ti, Pe, pressure, Ti^2).
        Every payload is Dirichlet only on physical wall masks; all interior
        and shard-interface faces remain centered in the existing operator.
        Incoming characteristic perturbations are supplied by the equilibrium
        state U0=(1, 1, 1, 0); outgoing and stationary perturbations come from
        the owner cell.
        """
        masks = (face_bcs[0].mask_x, face_bcs[0].mask_y, face_bcs[0].mask_z)
        wall_states = []
        u0 = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=jnp.float64)

        assert self.upwind_equilibrium_wall_projectors is not None
        for axis, name in zip((0, 1, 2), ("x", "y", "z")):
            owner = jnp.stack(tuple(getattr(stencil, name).center for stencil in stencils), axis=-1)
            for side in (0, 1):
                if side == 0 and self.axis_regular_axes[axis]:
                    continue
                sl = _axis_plane_slice(axis, side)
                retained_projector = self.upwind_equilibrium_wall_projectors.axes[axis][side]
                owner_face = owner[_axis_plane_slice(axis, side)]
                state = _upwind_equilibrium_characteristic_state(
                    owner_face, u0, retained_projector
                )
                wall_states.append((axis, side, getattr(face_bcs[0], f"mask_{name}")[sl], state))

        bases = tuple(_wall_candidate_values(stencil, bc) for stencil, bc in zip(stencils, face_bcs))
        def patch(base_values, component):
            values = [jnp.array(v) for v in base_values]
            for axis, side, mask, state in wall_states:
                sl = _axis_plane_slice(axis, side)
                state_value = (
                    state[..., component]
                    if component < 3
                    else state[..., 0] * state[..., 1]
                    if component == 3
                    else state[..., 0] * (state[..., 1] + self.parameters.tau * state[..., 2])
                    if component == 4
                    else state[..., 2] * state[..., 2]
                )
                values[axis] = values[axis].at[sl].set(jnp.where(mask, state_value, values[axis][sl]))
            return _dirichlet_face_bc_from_values(tuple(values), self.domain.layout, masks)

        # Scalar bases are only used away from the physical wall, where the
        # centered operator ignores their BC values. Wall values are replaced
        # by the coupled state above.
        n_bc = patch(bases[0], 0)
        te_bc = patch(bases[1], 1)
        ti_bc = patch(bases[2], 2)
        pe_bc = patch(tuple(a * b for a, b in zip(bases[0], bases[1])), 3)
        pressure_bc = patch(tuple(a * (b + self.parameters.tau * c) for a, b, c in zip(bases[0], bases[1], bases[2])), 4)
        ti2_bc = patch(tuple(a * a for a in bases[2]), 5)
        return n_bc, te_bc, ti_bc, pe_bc, pressure_bc, ti2_bc

    def ion_temperature_curvature_chain_rule_diagnostics(
        self,
        state_owned: FciDrbEBState,
        *,
        primitive_cv_closures: dict[str, object] | None = None,
    ) -> jnp.ndarray:
        """Return max-abs product, flux, and chain-rule-defect Ti terms.

        The two terms are the complete nonlinear Ti self-curvature
        contributions appearing in the RHS:

        ``-(10*tau*Ti/(3B))*C(Ti)`` and ``-(5*tau/(3B))*C(Ti**2)``.

        This method intentionally uses the state field's already closed halo
        before squaring it.  The derived squared Dirichlet face values are
        constructed independently so the conservative boundary closure is
        consistent with the squared scalar field.
        """

        if self.curvature_scheme != "conservative":
            raise ValueError(
                "ion-temperature curvature chain-rule diagnostics require "
                "curvature_scheme='conservative'"
            )
        face_bc = self._face_bcs(state_owned)
        state_halo = FciDrbEBState(
            density=self._prepare_scalar_halo(state_owned.density, face_bc.density),
            phi=self._prepare_scalar_halo(state_owned.phi, face_bc.phi),
            Te=self._prepare_scalar_halo(state_owned.Te, face_bc.Te),
            Ti=self._prepare_scalar_halo(state_owned.Ti, face_bc.Ti),
            Vi=(self._prepare_face_halo(state_owned.Vi, face_bc.Vi)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._prepare_scalar_halo(state_owned.Vi, face_bc.Vi)),
            Ve=(self._prepare_face_halo(state_owned.Ve, face_bc.Ve)
                if self.parallel_velocity_layout == "fci-staggered"
                else self._prepare_scalar_halo(state_owned.Ve, face_bc.Ve)),
            vorticity=self._prepare_scalar_halo(
                state_owned.vorticity, face_bc.vorticity
            ),
        )
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, self.geometry, self.domain, face_bc, tau=self.parameters.tau
        )
        ti_halo = state_halo.Ti
        context = self._stencil_builder_context()
        ti_stencil = build_local_conservative_stencil_from_field(
            ti_halo,
            self.geometry,
            context,
        )
        ti_squared_stencil = build_local_conservative_stencil_from_field(
            ti_halo * ti_halo,
            self.geometry,
            context,
        )
        ti_face_for_curvature = None
        ti_squared_face_for_curvature = None
        primitive = (
            primitive_cv_closures
            if primitive_cv_closures is not None
            else self._primitive_cv_closures(state_halo, names=("Ti",))
        )
        ti_squared_compact_closure = self._cv_product(
            primitive.get("Ti"), primitive.get("Ti")
        ) if primitive else None
        if self.curvature_inflow_closure == "upwind-equilibrium":
            density_halo = self._prepare_scalar_halo(state_owned.density, face_bc.density)
            te_halo = self._prepare_scalar_halo(state_owned.Te, face_bc.Te)
            vorticity_halo = self._prepare_scalar_halo(state_owned.vorticity, face_bc.vorticity)
            characteristic_bcs = self._upwind_equilibrium_boundary_face_bcs(
                (
                    build_local_conservative_stencil_from_field(density_halo, self.geometry, context),
                    build_local_conservative_stencil_from_field(te_halo, self.geometry, context),
                    ti_stencil,
                    build_local_conservative_stencil_from_field(vorticity_halo, self.geometry, context),
                ),
                (face_bc.density, face_bc.Te, face_bc.Ti, face_bc.vorticity),
            )
            ti_face_for_curvature = characteristic_bcs[2]
            ti_squared_face_for_curvature = characteristic_bcs[5]
        if self.curvature_inflow_closure == "upwind-equilibrium":
            ti_curvature = self._conservative_curvature(
                ti_stencil, ti_face_for_curvature, field_halo=ti_halo,
                field_closure=primitive.get("Ti") if primitive else None,
            )
            ti_squared_curvature = self._conservative_curvature(
                ti_squared_stencil, ti_squared_face_for_curvature,
                field_halo=ti_halo * ti_halo,
                field_closure=ti_squared_compact_closure,
            )
        else:
            ti_curvature = self._conservative_curvature(
                ti_stencil, boundary_trace=operator_boundary.Ti, field_halo=ti_halo,
                field_closure=primitive.get("Ti") if primitive else None,
            )
            ti_squared_curvature = self._conservative_curvature(
                ti_squared_stencil, boundary_trace=operator_boundary.Ti_squared,
                field_halo=ti_halo * ti_halo,
                field_closure=ti_squared_compact_closure,
            )
        owned = self.domain.layout.owned_slices_cell
        ti = jnp.asarray(ti_halo[owned], dtype=jnp.float64)
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        tau = jnp.asarray(self.parameters.tau, dtype=jnp.float64)
        product_term = -(10.0 * tau * ti / (3.0 * bmag)) * ti_curvature
        flux_term = -(5.0 * tau / (3.0 * bmag)) * ti_squared_curvature
        defect = product_term - flux_term
        return jnp.stack(
            (
                jnp.max(jnp.abs(product_term)),
                jnp.max(jnp.abs(flux_term)),
                jnp.max(jnp.abs(defect)),
            )
        )

    def _field_perp_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
        *,
        field_closure=None,
    ) -> jnp.ndarray:
        if float(coefficient) == 0.0:
            return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        context = self._stencil_builder_context()
        conservative = build_local_conservative_stencil_from_field(
            field_halo,
            self.geometry,
            context,
        )
        return jnp.asarray(coefficient, dtype=jnp.float64) * local_perp_laplacian_conservative_op(
            conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            **self._cv_operator_args(
                field_halo, face_bc, field_closure=field_closure
            ),
        )

    def _field_parallel_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
        *,
        field_closure=None,
    ) -> jnp.ndarray:
        if float(coefficient) == 0.0:
            return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        context = self._stencil_builder_context()
        conservative = build_local_conservative_stencil_from_field(
            field_halo,
            self.geometry,
            context,
        )
        return jnp.asarray(coefficient, dtype=jnp.float64) * local_parallel_laplacian_conservative_op(
            conservative,
            self.geometry,
            self.domain,
            face_bc=face_bc,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            **self._cv_operator_args(
                field_halo, face_bc, field_closure=field_closure
            ),
        )

    def _fci_remote_values(
        self,
        field_halo: jnp.ndarray,
        context: StencilBuilderContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Exchange the two mapped endpoint payloads for one scalar halo."""

        exchange = RemoteFciDependencyExchange()
        return (
            exchange(
                field_halo=field_halo,
                direction=self.geometry.maps.forward,
                context=context,
                cut_wall_bc=None,
            ),
            exchange(
                field_halo=field_halo,
                direction=self.geometry.maps.backward,
                context=context,
                cut_wall_bc=None,
            ),
        )

    def _outgoing_face_to_center_halo(
        self,
        face_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        context: StencilBuilderContext,
    ) -> jnp.ndarray:
        """Reconstruct outgoing FCI-edge storage to a closed centered halo."""

        forward, backward = self._fci_remote_values(face_halo, context)
        centered = local_outgoing_face_to_center_average_fci_op(
            face_halo,
            self.geometry,
            context=context,
            forward_remote_values=forward,
            backward_remote_values=backward,
        )
        return self._prepare_cell_rlp_halo_from_fine(centered, face_bc)

    def _center_owned_to_outgoing_face(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        context: StencilBuilderContext,
    ) -> jnp.ndarray:
        """Interpolate a centered value to owned outgoing FCI source edges."""

        centered_halo = self._prepare_fine_storage_halo(values_owned, face_bc)
        forward, backward = self._fci_remote_values(centered_halo, context)
        return local_center_to_outgoing_face_average_fci_op(
            centered_halo, self.geometry, context=context,
            forward_remote_values=forward, backward_remote_values=backward,
        )

    def _cell_force_to_outgoing_face_mass_adjoint(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        context: StencilBuilderContext,
    ) -> jnp.ndarray:
        """Map a centered force to outgoing FCI source edges.

        Angular-RLP staggered forces use the geometric mass-adjoint lift of
        the homogeneous mapped f2c reconstruction.  Production uses the
        linear zero-Neumann velocity closure; affine traces must not enter
        this transpose path.
        """

        if (
            self.parallel_velocity_layout == "fci-staggered"
            and self.control_volume_geometry is not None
            and self.control_volume_geometry.has_angular_agglomeration
            and self.outgoing_face_topology is not None
        ):
            transfer = build_local_fci_face_galerkin_transfer(
                self.control_volume_geometry.cells, self.outgoing_face_topology
            )
            # ``f2c`` is used under ``linear_transpose`` below.  Preserve the
            # boundary type/masks but remove prescribed data, so it is exactly
            # the homogeneous linear reconstruction.  In particular, an
            # affine physical-wall trace can never be transposed as a force.
            homogeneous_bc = replace(
                face_bc,
                value_x=jnp.zeros_like(face_bc.value_x),
                value_y=jnp.zeros_like(face_bc.value_y),
                value_z=jnp.zeros_like(face_bc.value_z),
            )

            def homogeneous_f2c(face_values_fine: jnp.ndarray) -> jnp.ndarray:
                halo = self._prepare_fine_storage_halo(
                    face_values_fine, homogeneous_bc
                )
                forward, backward = self._fci_remote_values(halo, context)
                return local_outgoing_face_to_center_average_fci_op(
                    halo, self.geometry, context=context,
                    forward_remote_values=forward,
                    backward_remote_values=backward,
                )

            owner_force = transfer.cell_restrict(values_owned)
            owner_face = transfer.cell_to_face_mass_adjoint_lift(
                owner_force, homogeneous_f2c
            )
            return transfer.face_prolong(owner_face)

        centered_halo = self._prepare_fine_storage_halo(values_owned, face_bc)
        forward, backward = self._fci_remote_values(centered_halo, context)
        return local_center_to_outgoing_face_average_fci_op(
            centered_halo,
            self.geometry,
            context=context,
            forward_remote_values=forward,
            backward_remote_values=backward,
        )

    def _cell_force_lanes_to_outgoing_face_mass_adjoint(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        context: StencilBuilderContext,
    ) -> jnp.ndarray:
        """Lift leading center-force lanes with one common f2c transpose.

        This is used for the perpendicular Poisson/diffusion diagnostics of
        one velocity species.  The returned lanes are ``P_e L R_c`` values;
        their sum is the total perpendicular force.  The scalar lift remains
        the path for parallel viscosity, where no lane batching is needed.
        """

        values = jnp.asarray(values_owned, dtype=jnp.float64)
        if values.ndim != 4 or values.shape[1:] != self.geometry.owned_shape:
            raise ValueError("values_owned must have shape (lane,) + owned_shape")
        if (
            self.parallel_velocity_layout == "fci-staggered"
            and self.control_volume_geometry is not None
            and self.control_volume_geometry.has_angular_agglomeration
            and self.outgoing_face_topology is not None
        ):
            transfer = build_local_fci_face_galerkin_transfer(
                self.control_volume_geometry.cells, self.outgoing_face_topology
            )
            homogeneous_bc = replace(
                face_bc,
                value_x=jnp.zeros_like(face_bc.value_x),
                value_y=jnp.zeros_like(face_bc.value_y),
                value_z=jnp.zeros_like(face_bc.value_z),
            )

            def homogeneous_f2c(face_values_fine: jnp.ndarray) -> jnp.ndarray:
                halo = self._prepare_fine_storage_halo(
                    face_values_fine, homogeneous_bc
                )
                forward, backward = self._fci_remote_values(halo, context)
                return local_outgoing_face_to_center_average_fci_op(
                    halo, self.geometry, context=context,
                    forward_remote_values=forward,
                    backward_remote_values=backward,
                )

            owner_force = jax.vmap(transfer.cell_restrict)(values)
            owner_face = transfer.cell_to_face_mass_adjoint_lift_batched(
                owner_force, homogeneous_f2c
            )
            return jax.vmap(transfer.face_prolong)(owner_face)

        return jax.vmap(
            lambda value: self._center_owned_to_outgoing_face(value, face_bc, context)
        )(values)

    def _fci_prepare_q(
        self,
        q_owned: jnp.ndarray,
        q_face_trace: LocalBoundaryFaceTrace3D,
        context: StencilBuilderContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Prepare a q=F/B halo and its forward/backward remote payloads.

        ``q_face_trace`` is already the operator-level trace of q on physical
        faces.  Turning it into a Dirichlet BC makes the existing ghost/leg
        filler enforce the same trace that the mapped interpolation rows see.
        """

        q_owned = jnp.asarray(q_owned, dtype=jnp.float64)
        if q_owned.shape != self.geometry.owned_shape:
            raise ValueError(
                f"q_owned must have shape {self.geometry.owned_shape}, got {q_owned.shape}"
            )
        q_bc = _dirichlet_face_bc_from_values(
            (
                q_face_trace.value_x,
                q_face_trace.value_y,
                q_face_trace.value_z,
            ),
            self.domain.layout,
            (
                q_face_trace.mask_x,
                q_face_trace.mask_y,
                q_face_trace.mask_z,
            ),
        )
        q_halo = self._prepare_scalar_halo(q_owned, q_bc)
        forward_remote, backward_remote = self._fci_remote_values(q_halo, context)
        return q_halo, forward_remote, backward_remote

    def _fci_prepare_flux_q(
        self,
        field_owned: jnp.ndarray,
        boundary_trace: LocalBoundaryFaceTrace3D,
        context: StencilBuilderContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Prepare q=F/B using an operator-specific physical wall trace."""

        b_floor = 1.0e-30
        B_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            b_floor,
        )
        face_b = tuple(
            jnp.maximum(
                jnp.asarray(face.Bmag_owned, dtype=jnp.float64),
                b_floor,
            )
            for face in self.geometry.face_bfield.axes
        )
        q_face = (
            boundary_trace.value_x / face_b[0],
            boundary_trace.value_y / face_b[1],
            boundary_trace.value_z / face_b[2],
        )
        return self._fci_prepare_q(
            jnp.asarray(field_owned, dtype=jnp.float64) / B_owned,
            LocalBoundaryFaceTrace3D(
                value_x=q_face[0], value_y=q_face[1], value_z=q_face[2],
                mask_x=boundary_trace.mask_x, mask_y=boundary_trace.mask_y,
                mask_z=boundary_trace.mask_z, layout=self.domain.layout,
            ),
            context,
        )

    def _fci_prepare_inverse_b(
        self,
        face_bc: LocalFciDrbEBFaceBCBundle,
        context: StencilBuilderContext,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Prepare q=1/B without reading physical B ghost cells."""

        b_floor = 1.0e-30
        B_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            b_floor,
        )
        face_b = tuple(
            jnp.maximum(
                jnp.asarray(face.Bmag_owned, dtype=jnp.float64),
                b_floor,
            )
            for face in self.geometry.face_bfield.axes
        )
        masks = (
            face_bc.phi.mask_x,
            face_bc.phi.mask_y,
            face_bc.phi.mask_z,
        )
        trace = LocalBoundaryFaceTrace3D(
            value_x=1.0 / face_b[0], value_y=1.0 / face_b[1],
            value_z=1.0 / face_b[2],
            mask_x=masks[0], mask_y=masks[1], mask_z=masks[2],
            layout=self.domain.layout,
        )
        return self._fci_prepare_q(1.0 / B_owned, trace, context)

    def _fci_face_galerkin_core(
        self,
        context: StencilBuilderContext,
        homogeneous_face_bc: LocalBoundaryFaceBC3D,
    ):
        """Return the angular-RLP transfer and its homogeneous mapped ``G_f``.

        Kept narrow so it can be tested independently of nonlinear RHS terms.
        The closure is suitable for ``jax.linear_transpose``: it contains
        only owner prolongation, halo/topology exchange, forward map exchange,
        and homogeneous endpoint interpolation.
        """

        if self.control_volume_geometry is None or self.outgoing_face_topology is None:
            raise RuntimeError("angular face Galerkin core requires cell and face topology")
        transfer = build_local_fci_face_galerkin_transfer(
            self.control_volume_geometry.cells, self.outgoing_face_topology,
        )

        def fine_forward_gradient(values_fine: jnp.ndarray) -> jnp.ndarray:
            halo = inject_owned_field_to_halo(values_fine, self.domain.layout)
            # This is the linear, homogeneous boundary part of the operator.
            # In particular zero-Neumann must copy the interior value rather
            # than leave a zero ghost, so constants remain in the nullspace.
            halo = LocalHaloClosure3D(
                physical_ghost_filler=self.physical_ghost_filler,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
            )(halo, self.domain, homogeneous_face_bc)
            forward = RemoteFciDependencyExchange()(
                field_halo=halo, direction=self.geometry.maps.forward,
                context=context, cut_wall_bc=None,
            )
            backward_size = (
                0 if self.geometry.maps.backward.remote is None
                else self.geometry.maps.backward.remote.max_receive_values
            )
            stencil = build_local_fci_stencil_from_field(
                halo, self.geometry, context,
                forward_remote_values=forward,
                backward_remote_values=jnp.zeros((backward_size,), dtype=halo.dtype),
            )
            return (stencil.plus - stencil.center) / jnp.maximum(stencil.dx_plus, 1.0e-30)

        return transfer, fine_forward_gradient

    def _fci_pair_target_mask(
        self,
        *,
        include_physical_wall: bool,
    ) -> jnp.ndarray:
        """Return rows admissible for a cell-centred FCI support pair.

        Ordinary support-core pairs admit only field-interior endpoints.  A
        boundary-aware pair may additionally admit physical-wall endpoints.
        Its divergence explicitly constructs the linear Neumann radial-halo
        closure before the transpose is built, so those storage dependencies
        must remain inside that complete operator.  Excluding them and then
        restoring independent legacy rows would break the very adjoint pair
        the boundary closure is intended to provide.  The ordinary support
        core, which has no such boundary closure, continues to exclude them.
        """

        layout = self.domain.layout
        radial_lo = layout.halo_width
        radial_hi = radial_lo + self.geometry.owned_shape[0]
        n_owned = int(math.prod(self.geometry.owned_shape))

        def has_local_radial_ghost(direction) -> jnp.ndarray:
            table = direction.local
            radial_ghost = (
                (table.source_i < radial_lo) | (table.source_i >= radial_hi)
            )
            bad_entry = table.active & radial_ghost
            safe_target = jnp.clip(table.target_flat, 0, n_owned - 1)
            # A padded row may carry arbitrary target/source indices, so only
            # active entries participate in the target-row reduction.
            bad_flat = jnp.zeros((n_owned,), dtype=jnp.int32).at[
                safe_target
            ].max(bad_entry.astype(jnp.int32))
            return bad_flat.reshape(self.geometry.owned_shape).astype(bool)

        def has_remote_radial_ghost(direction) -> jnp.ndarray:
            remote = direction.remote
            if remote is None or remote.max_entries == 0:
                return jnp.zeros(self.geometry.owned_shape, dtype=bool)
            if remote.max_receive_values == 0:
                return jnp.zeros(self.geometry.owned_shape, dtype=bool)
            safe_slot = jnp.clip(
                remote.receive_slot, 0, remote.max_receive_values - 1
            )
            source_i = remote.request_source_owner_local_i[safe_slot]
            request_active = remote.request_active[safe_slot]
            radial_ghost = (source_i < radial_lo) | (source_i >= radial_hi)
            safe_target = jnp.clip(remote.target_flat, 0, n_owned - 1)
            remote_flat = jnp.zeros((n_owned,), dtype=jnp.int32).at[
                safe_target
            ].max((remote.active & request_active & radial_ghost).astype(jnp.int32))
            return remote_flat.reshape(self.geometry.owned_shape).astype(bool)

        maps = self.geometry.maps
        admitted_endpoint_kinds = (
            (maps.forward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
            & (maps.backward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
        )
        if include_physical_wall:
            admitted_endpoint_kinds = (
                (
                    (maps.forward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
                    | (maps.forward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY)
                )
                & (
                    (maps.backward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
                    | (maps.backward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY)
                )
            )
        admitted = (
            admitted_endpoint_kinds
            & maps.forward.target_valid
            & maps.backward.target_valid
        )
        admitted = admitted & self.geometry.active_cell_mask_owned
        if include_physical_wall:
            return admitted
        return (
            admitted
            & ~has_local_radial_ghost(maps.forward)
            & ~has_local_radial_ghost(maps.backward)
            & ~has_remote_radial_ghost(maps.forward)
            & ~has_remote_radial_ghost(maps.backward)
        )

    def _fci_support_core_target_mask(self) -> jnp.ndarray:
        """Return ordinary interior rows admitted by the support-core pair."""

        return self._fci_pair_target_mask(include_physical_wall=False)

    def _fci_pair_cell_mass(self) -> jnp.ndarray:
        """Return the fine-cell measure preserved by the downstream RLP map."""

        if self.control_volume_geometry is not None:
            # RLP restriction is a raw-volume weighted average.  Building the
            # fine-grid adjoint with any other mass (for example midpoint
            # J*du*dtheta*deta) loses the Green identity after restriction.
            return jnp.asarray(
                self.control_volume_geometry.cells.raw_volume,
                dtype=jnp.float64,
            )
        return (
            jnp.asarray(self.geometry.cell_volume_geometry.volume, dtype=jnp.float64)
            * jnp.asarray(
                self.geometry.cell_volume_geometry.volume_fraction,
                dtype=jnp.float64,
            )
            * jnp.asarray(self.geometry.spacing.dx_owned, dtype=jnp.float64)
            * jnp.asarray(self.geometry.spacing.dy_owned, dtype=jnp.float64)
            * jnp.asarray(self.geometry.spacing.dz_owned, dtype=jnp.float64)
        )

    def _fci_current_phi_boundary_pair(
        self,
        *,
        face_bc: LocalFciDrbEBFaceBCBundle,
        context: StencilBuilderContext,
        wall_endpoint_current_values: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        build_adjoint: bool = True,
    ) -> tuple[
        Callable[[jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray], jnp.ndarray],
        jnp.ndarray,
    ]:
        """Build the wall-closed ``current-divergence/grad(phi)`` pair.

        The scalar current receives a direct zero-Neumann physical closure.
        This is the linear composite-current consequence of the primitive
        zero-Neumann density/Vi/Ve conditions.  With
        ``wall_endpoint_current_values`` supplied, only the mapped physical
        endpoint values are replaced by ``j_w/B``; ordinary mapped endpoints
        retain the prepared FCI stencil.  Such a closure is affine and must
        be used with ``build_adjoint=False``.  The homogeneous zero-endpoint
        variant is the one paired with ``G=-M^-1 D^T M``.

        No constant-nullspace correction belongs here: the adjoint potential
        has homogeneous Dirichlet wall data, so an interior constant is not a
        constant boundary state.  Physical ``phi*j`` wall power is zero for
        the present ``phi_wall=0`` closure.
        """

        target = self._fci_pair_target_mask(include_physical_wall=True)
        active = self.geometry.active_cell_mask_owned
        b_floor = 1.0e-30
        bmag_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            b_floor,
        )
        face_b = tuple(
            jnp.maximum(
                jnp.asarray(face.Bmag_owned, dtype=jnp.float64),
                b_floor,
            )
            for face in self.geometry.face_bfield.axes
        )
        inverse_b_halo = inverse_b_forward = inverse_b_backward = None
        if wall_endpoint_current_values is not None:
            inverse_b_halo, inverse_b_forward, inverse_b_backward = (
                self._fci_prepare_inverse_b(face_bc, context)
            )
            inverse_b_stencil = build_local_fci_stencil_from_field(
                inverse_b_halo,
                self.geometry,
                context,
                forward_remote_values=inverse_b_forward,
                backward_remote_values=inverse_b_backward,
            )
            backward_wall = (
                self.geometry.maps.backward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            forward_wall = (
                self.geometry.maps.forward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            endpoint_backward, endpoint_forward = wall_endpoint_current_values

        def current_divergence(values_owned: jnp.ndarray) -> jnp.ndarray:
            values = jnp.where(
                active,
                jnp.asarray(values_owned, dtype=jnp.float64),
                0.0,
            )
            current_halo = self._prepare_fine_storage_halo(
                values,
                face_bc.density,
            )
            current_trace = build_local_boundary_face_trace_from_halo(
                current_halo,
                self.geometry,
                self.domain,
                face_bc.density,
            )
            q_trace_values = (
                current_trace.value_x / face_b[0],
                current_trace.value_y / face_b[1],
                current_trace.value_z / face_b[2],
            )
            q_bc = _dirichlet_face_bc_from_values(
                q_trace_values,
                self.domain.layout,
                (
                    current_trace.mask_x,
                    current_trace.mask_y,
                    current_trace.mask_z,
                ),
            )
            q_halo = self._prepare_fine_storage_halo(values / bmag_owned, q_bc)
            forward, backward = self._fci_remote_values(q_halo, context)
            if wall_endpoint_current_values is not None:
                q_stencil = build_local_fci_stencil_from_field(
                    q_halo,
                    self.geometry,
                    context,
                    forward_remote_values=forward,
                    backward_remote_values=backward,
                )
                q_stencil = replace(
                    q_stencil,
                    minus=jnp.where(
                        backward_wall,
                        endpoint_backward * inverse_b_stencil.minus,
                        q_stencil.minus,
                    ),
                    plus=jnp.where(
                        forward_wall,
                        endpoint_forward * inverse_b_stencil.plus,
                        q_stencil.plus,
                    ),
                )
                divergence = jnp.asarray(bmag_owned) * local_grad_parallel_op_fci(
                    q_stencil, self.geometry
                )
                return jnp.where(target, divergence, 0.0)
            divergence = local_parallel_q_flux_div_fci_op(
                q_halo,
                self.geometry,
                context=context,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )
            return jnp.where(target, divergence, 0.0)

        cell_mass = self._fci_pair_cell_mass()
        if build_adjoint:
            phi_gradient = build_weighted_negative_adjoint(
                current_divergence,
                cell_mass,
                cell_mass,
                primal_active=active,
                dual_active=target,
            )
        else:
            phi_gradient = lambda values: jnp.zeros_like(values)
        return phi_gradient, current_divergence, target

    def _fci_support_core_pair(
        self,
        *,
        face_bc: LocalFciDrbEBFaceBCBundle,
        context: StencilBuilderContext,
    ) -> tuple[
        Callable[[jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray], jnp.ndarray],
        jnp.ndarray,
    ]:
        """Close a constant-exact weighted support pair from legacy ``D0``.

        ``D0`` is the legacy mapped q-flux divergence evaluated with a
        homogeneous q halo and restricted to support-core rows.  Taking its
        weighted negative adjoint yields ``H``.  The diagonal local correction
        ``G(f)=H(f)-H(1)f`` makes constants exact, then a second weighted
        negative adjoint closes the conservative ``G/D`` pair.  Neither
        transpose output is target-masked: core rows may scatter onto excluded
        or wall-adjacent cells, whose legacy closure is added by the caller.
        """

        core_target = self._fci_support_core_target_mask()
        active = self.geometry.active_cell_mask_owned
        bmag_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        # Preserve the halo closure's linear topology/periodic semantics while
        # deleting prescribed affine values before linear_transpose sees it.
        homogeneous_q_bc = replace(
            face_bc.phi,
            value_x=jnp.zeros_like(face_bc.phi.value_x),
            value_y=jnp.zeros_like(face_bc.phi.value_y),
            value_z=jnp.zeros_like(face_bc.phi.value_z),
        )
        def legacy_homogeneous_core_divergence(
            values_owned: jnp.ndarray,
        ) -> jnp.ndarray:
            values = jnp.asarray(values_owned, dtype=jnp.float64)
            q_halo = self._prepare_fine_storage_halo(
                values / bmag_owned, homogeneous_q_bc
            )
            forward, backward = self._fci_remote_values(q_halo, context)
            divergence = local_parallel_q_flux_div_fci_op(
                q_halo,
                self.geometry,
                context=context,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )
            return jnp.where(core_target, divergence, 0.0)

        cell_mass = self._fci_pair_cell_mass()
        h_adjoint = build_weighted_negative_adjoint(
            legacy_homogeneous_core_divergence,
            cell_mass,
            cell_mass,
            primal_active=active,
            dual_active=core_target,
        )
        constant_adjoint = h_adjoint(jnp.ones(self.geometry.owned_shape))

        def support_gradient(values_owned: jnp.ndarray) -> jnp.ndarray:
            values = jnp.where(active, jnp.asarray(values_owned, dtype=jnp.float64), 0.0)
            return h_adjoint(values) - constant_adjoint * values

        support_divergence = build_weighted_negative_adjoint(
            support_gradient,
            cell_mass,
            cell_mass,
            primal_active=active,
            dual_active=active,
        )
        return support_gradient, support_divergence, core_target

    def _fci_parallel_terms(
        self,
        *,
        state_halo: FciDrbEBState,
        face_bc: LocalFciDrbEBFaceBCBundle,
        operator_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        parallel_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        context: StencilBuilderContext,
        return_electron_force_diagnostics: bool = False,
        short_leg_selection_dt: Any = 0.0,
    ) -> dict[str, jnp.ndarray]:
        """Evaluate the mapped parallel operator family for one RHS state.

        Composite first-order operands use their level-2 operator traces.  The
        primitive diffusion operands use their already closed field halos.
        The returned geometry-only div(b) is shared by all compatible gradients.
        """

        owned = self.domain.layout.owned_slices_cell
        fields = {
            "density": state_halo.density,
            "Te": state_halo.Te,
            "Ti": state_halo.Ti,
            "Vi": state_halo.Vi,
            "Ve": state_halo.Ve,
            "phi": state_halo.phi,
            "vorticity": state_halo.vorticity,
            "density_flux": state_halo.density * state_halo.Ve,
            "current": state_halo.density * (state_halo.Vi - state_halo.Ve),
            "Pe": state_halo.density * state_halo.Te,
            "pressure": state_halo.density * state_halo.Te
            + self.parameters.tau * state_halo.density * state_halo.Ti,
        }
        traces = {
            "density": parallel_boundary.density,
            "Te": parallel_boundary.Te,
            "Ti": parallel_boundary.Ti,
            "Vi": parallel_boundary.Vi,
            "Ve": parallel_boundary.Ve,
            "phi": operator_boundary.phi,
            "vorticity": operator_boundary.vorticity,
            "density_flux": parallel_boundary.density_flux,
            "current": parallel_boundary.current,
            "Pe": parallel_boundary.Pe,
            "pressure": parallel_boundary.pressure,
        }
        q_data = {
            name: self._fci_prepare_flux_q(field[owned], traces[name], context)
            for name, field in fields.items()
        }
        vorticity_current_boundary = (
            parallel_boundary.current
            if self.vorticity_current_inflow_trace == "parallel-characteristic"
            else operator_boundary.current
        )
        q_data["vorticity_current"] = self._fci_prepare_flux_q(
            fields["current"][owned], vorticity_current_boundary, context
        )
        inverse_b_halo, inverse_b_forward, inverse_b_backward = (
            self._fci_prepare_inverse_b(face_bc, context)
        )

        support_gradient = None
        support_divergence = None
        support_core_target = None
        current_phi_target = None
        characteristic_sat_homogeneous_current_divergence = None
        characteristic_sat_affine_current_divergence = None
        characteristic_sat_current_divergence = None
        support_gradient_values: dict[str, jnp.ndarray] = {}
        support_flux_values: dict[str, jnp.ndarray] = {}
        if self.parallel_flux_pairing == "support-core":
            support_gradient, support_divergence, support_core_target = (
                self._fci_support_core_pair(
                    face_bc=face_bc,
                    context=context,
                )
            )
            use_current_phi_boundary_pair = self.parallel_boundary_pairing in (
                "current-phi", "characteristic-sat"
            )
            if use_current_phi_boundary_pair:
                # For characteristic-SAT, D0 is the derivative with the
                # projected wall endpoint held fixed at zero; ordinary mapped
                # endpoints retain their existing FCI values.
                current_phi_gradient, current_phi_divergence, current_phi_target = (
                    self._fci_current_phi_boundary_pair(
                        face_bc=face_bc,
                        context=context,
                        wall_endpoint_current_values=(
                            jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                            jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                        ) if self.parallel_boundary_pairing == "characteristic-sat" else None,
                    )
                )
            support_gradient_names = (
                "density", "Te", "Ti", "Vi", "Ve", "Pe",
                "pressure", "current", "vorticity",
            )
            if not use_current_phi_boundary_pair:
                support_gradient_names = support_gradient_names + ("phi",)
            # In production material mode the final RHS directly consumes
            # only Ti/phi/vorticity from this legacy gradient family.  Keep
            # the remaining values exact, but put them in a separate batch so
            # XLA can eliminate that whole transpose application when no
            # diagnostic/output path observes it.
            if self.parallel_material_scheme == "production-path":
                primary_gradient_names = (
                    ("Ti", "vorticity")
                    if use_current_phi_boundary_pair
                    else ("Ti", "phi", "vorticity")
                )
                secondary_gradient_names = tuple(
                    name for name in support_gradient_names
                    if name not in primary_gradient_names
                )
                for names in (primary_gradient_names, secondary_gradient_names):
                    batch = support_gradient(jnp.stack(
                        tuple(fields[name][owned] for name in names), axis=0
                    ))
                    support_gradient_values.update(dict(zip(names, batch)))
            else:
                support_gradient_batch = support_gradient(jnp.stack(
                    tuple(fields[name][owned] for name in support_gradient_names),
                    axis=0,
                ))
                support_gradient_values = dict(zip(
                    support_gradient_names, support_gradient_batch
                ))
            if use_current_phi_boundary_pair:
                support_gradient_values["phi"] = current_phi_gradient(
                    fields["phi"][owned]
                )
            support_flux_names = (
                "density_flux", "current", "Vi", "Ve",
            )
            support_flux_fields = {
                "density_flux": fields["density_flux"][owned],
                "current": fields["current"][owned],
                "Vi": fields["Vi"][owned],
                "Ve": fields["Ve"][owned],
            }
            if not use_current_phi_boundary_pair:
                support_flux_names = support_flux_names + ("vorticity_current",)
                support_flux_fields["vorticity_current"] = fields["current"][owned]
            if self.parallel_material_scheme == "production-path":
                support_flux_batch = support_divergence(jnp.stack(
                    tuple(
                        support_flux_fields[name]
                        for name in support_flux_names
                    ),
                    axis=0,
                ))
                support_flux_values = dict(zip(
                    support_flux_names,
                    support_flux_batch,
                ))
            else:
                support_flux_batch = support_divergence(jnp.stack(
                    tuple(support_flux_fields[name] for name in support_flux_names),
                    axis=0,
                ))
                support_flux_values = dict(zip(support_flux_names, support_flux_batch))
            if use_current_phi_boundary_pair:
                support_flux_values["vorticity_current"] = current_phi_divergence(
                    fields["current"][owned]
                )

        diagnostic_names = ("density", "Te", "Ti", "Vi", "Ve", "phi", "Pe")
        diagnostic_gradient_components: dict[str, jnp.ndarray] = {}
        diagnostic_endpoint_values: dict[str, jnp.ndarray] = {}
        if return_electron_force_diagnostics:
            for name in diagnostic_names:
                q_halo, forward, backward = q_data[name]
                components, endpoints = (
                    local_grad_parallel_op_fci_compatible_from_q_components(
                        q_halo,
                        inverse_b_halo,
                        self.geometry,
                        context=context,
                        field_owned=fields[name][owned],
                        forward_remote_q_values=forward,
                        backward_remote_q_values=backward,
                        forward_remote_inverse_b_values=inverse_b_forward,
                        backward_remote_inverse_b_values=inverse_b_backward,
                    )
                )
                if self._uses_compact_face_operators:
                    components = jax.vmap(
                        lambda value: aggregate_local_control_volume_average(
                            value,
                            self.control_volume_geometry.cells,
                            self.domain,
                        )
                    )(components)
                if support_gradient is not None:
                    # The support pair is not a three-point local stencil:
                    # its transpose can scatter from any admitted core row.
                    # Keep the diagnostic lane schema while placing the full
                    # paired gradient in the center lane, so its lane sum is
                    # exactly the production support-gradient value.
                    gradient_target = (
                        current_phi_target
                        if name == "phi" and current_phi_target is not None
                        else support_core_target
                    )
                    paired = support_gradient_values[name] + jnp.where(
                        gradient_target,
                        0.0,
                        jnp.sum(components, axis=0),
                    )
                    components = jnp.stack(
                        (jnp.zeros_like(paired), paired, jnp.zeros_like(paired)),
                        axis=0,
                    )
                diagnostic_gradient_components[name] = components
                diagnostic_endpoint_values[name] = endpoints

        def q_div(name: str) -> jnp.ndarray:
            q_halo, forward, backward = q_data[name]
            value = local_parallel_q_flux_div_fci_op(
                q_halo,
                self.geometry,
                context=context,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )
            return (aggregate_local_control_volume_average(
                value, self.control_volume_geometry.cells, self.domain
            ) if self._uses_compact_face_operators else value)

        div_b = local_parallel_div_b_fci_from_q_op(
            inverse_b_halo,
            self.geometry,
            context=context,
            forward_remote_q_values=inverse_b_forward,
            backward_remote_q_values=inverse_b_backward,
        )
        # Keep the fine-row geometric source for the production mapped-row
        # operator.  Compact-owner aggregation is applied only after the
        # complete residual (including this source) has been formed.
        div_b_fine = div_b
        if self._uses_compact_face_operators:
            div_b = aggregate_local_control_volume_average(
                div_b, self.control_volume_geometry.cells, self.domain
            )

        def flux_div(name: str) -> jnp.ndarray:
            legacy_value = q_div(name)
            if support_divergence is None or support_core_target is None:
                return legacy_value
            # The support pair owns every admissible dual row.  Its transpose
            # is intentionally left unmasked on primal targets: a core flux
            # may contribute to an excluded/wall-adjacent target.  The legacy
            # divergence supplies only the omitted target rows as boundary
            # closure, avoiding a double contribution on the support core.
            flux_target = (
                current_phi_target
                if name == "vorticity_current" and current_phi_target is not None
                else support_core_target
            )
            return support_flux_values[name] + jnp.where(
                flux_target, 0.0, legacy_value
            )

        def grad(name: str) -> jnp.ndarray:
            q_halo, forward, backward = q_data[name]
            legacy_value = local_grad_parallel_op_fci_compatible_from_q(
                q_halo,
                self.geometry,
                context=context,
                field_owned=fields[name][owned],
                div_b=div_b,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )
            if self._uses_compact_face_operators:
                legacy_value = aggregate_local_control_volume_average(
                    legacy_value, self.control_volume_geometry.cells, self.domain
                )
            if support_gradient is None or support_core_target is None:
                return legacy_value
            gradient_target = (
                current_phi_target
                if name == "phi" and current_phi_target is not None
                else support_core_target
            )
            return support_gradient_values[name] + jnp.where(
                gradient_target, 0.0, legacy_value
            )

        gradient_values = {
            name: grad(name)
            for name in (
                "density",
                "Te",
                "Ti",
                "Vi",
                "Ve",
                "phi",
                "Pe",
                "pressure",
                "current",
                "vorticity",
            )
        }

        material_upwind_correction = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        )
        material_upwind_correction_components = jnp.zeros(
            self.geometry.owned_shape + (3, 5), dtype=jnp.float64
        )
        material_centered_principal = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        )
        material_upwind_principal = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        )
        material_characteristic_endpoint_values = jnp.zeros(
            self.geometry.owned_shape + (2, 5), dtype=jnp.float64
        )
        material_characteristic_leg_lengths = jnp.zeros(
            self.geometry.owned_shape + (2,), dtype=jnp.float64
        )
        parallel_material_residual = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        )
        parallel_material_diagnostics = {
            "backward_wall": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "forward_wall": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "wall_row": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "ordinary_row": jnp.ones(self.geometry.owned_shape, dtype=bool),
            "spectral_fallback": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "positivity_fallback": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "wall_spectral_fallback": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "fallback": jnp.zeros(self.geometry.owned_shape, dtype=bool),
            "admissible": jnp.ones(self.geometry.owned_shape, dtype=bool),
        }
        if self.fci_parallel_leg_scheme == "boundary-characteristic-upwind":
            primitive_names = ("density", "Te", "Ti", "Vi", "Ve")
            primitive_stencils = []
            for name in primitive_names:
                field_halo, forward_remote, backward_remote = self._fci_prepare_q(
                    fields[name][owned], traces[name], context
                )
                primitive_stencils.append(
                    build_local_fci_stencil_from_field(
                        field_halo,
                        self.geometry,
                        context,
                        forward_remote_values=forward_remote,
                        backward_remote_values=backward_remote,
                    )
                )
            center = jnp.stack(
                tuple(stencil.center for stencil in primitive_stencils), axis=-1
            )
            minus = jnp.stack(
                tuple(stencil.minus for stencil in primitive_stencils), axis=-1
            )
            plus = jnp.stack(
                tuple(stencil.plus for stencil in primitive_stencils), axis=-1
            )
            centered_gradient = jnp.stack(
                tuple(gradient_values[name] for name in primitive_names), axis=-1
            )
            matrix = parallel_characteristic_matrix(
                center[..., 0],
                center[..., 1],
                center[..., 2],
                center[..., 3],
                center[..., 4],
                self.parameters.tau,
                self.parameters.mi_over_me,
            )
            backward_wall = (
                self.geometry.maps.backward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            forward_wall = (
                self.geometry.maps.forward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            material_upwind_correction = (
                target_local_characteristic_upwind_correction(
                    center,
                    minus,
                    plus,
                    primitive_stencils[0].dx_min,
                    primitive_stencils[0].dx_plus,
                    centered_gradient,
                    matrix,
                    backward_wall,
                    forward_wall,
                )
            )
            if return_electron_force_diagnostics:
                (
                    material_centered_principal,
                    material_upwind_principal,
                    material_characteristic_endpoint_values,
                ) = target_local_characteristic_upwind_principal_diagnostics(
                    center,
                    minus,
                    plus,
                    primitive_stencils[0].dx_min,
                    primitive_stencils[0].dx_plus,
                    centered_gradient,
                    matrix,
                    backward_wall,
                    forward_wall,
                )
                material_characteristic_leg_lengths = jnp.stack(
                    (
                        primitive_stencils[0].dx_min,
                        primitive_stencils[0].dx_plus,
                    ),
                    axis=-1,
                )
                centered_gradient_components = jnp.stack(
                    tuple(
                        jnp.moveaxis(
                            diagnostic_gradient_components[name], 0, -1
                        )
                        for name in primitive_names
                    ),
                    axis=-1,
                )
                material_upwind_correction_components = (
                    target_local_characteristic_upwind_correction_components(
                        center,
                        minus,
                        plus,
                        primitive_stencils[0].dx_min,
                        primitive_stencils[0].dx_plus,
                        centered_gradient_components,
                        matrix,
                        backward_wall,
                        forward_wall,
                    )
                )

        if self.parallel_material_scheme == "production-path":
            primitive_names = ("density", "Te", "Ti", "Vi", "Ve")
            primitive_stencils = []
            for name in primitive_names:
                field_halo, forward_remote, backward_remote = self._fci_prepare_q(
                    fields[name][owned], traces[name], context
                )
                primitive_stencils.append(
                    build_local_fci_stencil_from_field(
                        field_halo,
                        self.geometry,
                        context,
                        forward_remote_values=forward_remote,
                        backward_remote_values=backward_remote,
                    )
                )
            center = jnp.stack(
                tuple(stencil.center for stencil in primitive_stencils), axis=-1
            )
            minus = jnp.stack(
                tuple(stencil.minus for stencil in primitive_stencils), axis=-1
            )
            plus = jnp.stack(
                tuple(stencil.plus for stencil in primitive_stencils), axis=-1
            )
            backward_wall = (
                self.geometry.maps.backward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            forward_wall = (
                self.geometry.maps.forward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            # Keep one canonical live eigensystem/endpoint projection for the
            # material residual and the characteristic current closure.  The
            # projected primitive vector is a first-order modal trace, so the
            # current exported to the vorticity/phi pair is the corresponding
            # first-order characteristic current, not a nonlinear product of
            # projected primitive components.
            wall_data = None
            if self.parallel_boundary_pairing == "characteristic-sat" or (
                return_electron_force_diagnostics
            ):
                wall_data = parallel_characteristic_wall_data(
                    center,
                    minus,
                    plus,
                    primitive_stencils[0].dx_min,
                    primitive_stencils[0].dx_plus,
                    self.parameters.tau,
                    self.parameters.mi_over_me,
                    selection_dt=short_leg_selection_dt
                    if self.parallel_short_leg_treatment == "local-backward-euler"
                    else 0.0,
                    cfl_limit=self.parallel_short_leg_cfl_limit,
                    backward_wall=backward_wall,
                    forward_wall=forward_wall,
                    backward_wall_state=minus,
                    forward_wall_state=plus,
                )
            if self.parallel_boundary_pairing == "characteristic-sat":
                # The homogeneous pair is the weighted-adjoint operator used
                # for grad(phi).  The characteristic wall trace is evaluated
                # separately and its difference is an affine lift applied
                # only to the vorticity current divergence.
                _, characteristic_current_divergence, _ = (
                    self._fci_current_phi_boundary_pair(
                        face_bc=face_bc,
                        context=context,
                        wall_endpoint_current_values=(
                            wall_data["backward_wall_characteristic_current"],
                            wall_data["forward_wall_characteristic_current"],
                        ),
                        build_adjoint=False,
                    )
                )
                actual_current = fields["current"][owned]
                characteristic_sat_current_divergence = characteristic_current_divergence(
                    actual_current
                )
                if current_phi_divergence is None:
                    raise RuntimeError(
                        "characteristic-sat requires a homogeneous current pair"
                    )
                characteristic_sat_homogeneous_current_divergence = (
                    current_phi_divergence(actual_current)
                )
                characteristic_sat_affine_current_divergence = (
                    characteristic_sat_current_divergence
                    - characteristic_sat_homogeneous_current_divergence
                )
                support_flux_values["vorticity_current"] = (
                    characteristic_sat_current_divergence
                )
            parallel_material_residual, parallel_material_diagnostics = (
                parallel_target_row_material_residual(
                    center,
                    minus,
                    plus,
                    primitive_stencils[0].dx_min,
                    primitive_stencils[0].dx_plus,
                    self.parameters.tau,
                    self.parameters.mi_over_me,
                    backward_wall=backward_wall,
                    forward_wall=forward_wall,
                    backward_wall_state=minus,
                    forward_wall_state=plus,
                    div_b=div_b_fine if self._uses_compact_face_operators else div_b,
                    selection_dt=short_leg_selection_dt
                    if self.parallel_short_leg_treatment == "local-backward-euler"
                    else 0.0,
                    cfl_limit=self.parallel_short_leg_cfl_limit,
                )
            )
            if return_electron_force_diagnostics and wall_data is not None:
                # The production path has the same directional residuals and
                # endpoint states as the wall helper.  Retain them in the
                # replay diagnostics so the electron-force report does not
                # silently show zero characteristic terms in production mode.
                backward_residual = wall_data["backward_residual"]
                forward_residual = wall_data["forward_residual"]
                material_upwind_principal = (
                    backward_residual + forward_residual
                )
                material_centered_principal = (
                    parallel_material_residual - material_upwind_principal
                )
                material_upwind_correction_components = jnp.stack(
                    (
                        backward_residual,
                        material_centered_principal,
                        forward_residual,
                    ),
                    axis=-2,
                )
                material_characteristic_endpoint_values = jnp.stack(
                    (
                        wall_data["backward_endpoint_state"],
                        wall_data["forward_endpoint_state"],
                    ),
                    axis=-2,
                )
                material_characteristic_leg_lengths = jnp.stack(
                    (
                        primitive_stencils[0].dx_min,
                        primitive_stencils[0].dx_plus,
                    ),
                    axis=-1,
                )
            if self._uses_compact_face_operators:
                parallel_material_residual = jax.vmap(
                    lambda value: aggregate_local_control_volume_average(
                        value,
                        self.control_volume_geometry.cells,
                        self.domain,
                    )
                )(jnp.moveaxis(parallel_material_residual, -1, 0))
                parallel_material_residual = jnp.moveaxis(
                    parallel_material_residual, 0, -1
                )

                def aggregate_flag(value: jnp.ndarray) -> jnp.ndarray:
                    return aggregate_local_control_volume_average(
                        value.astype(jnp.float64),
                        self.control_volume_geometry.cells,
                        self.domain,
                    ) > 0.0

                # Boolean diagnostics are interpreted conservatively on an
                # aggregate owner: a flag is set if any contributing fine row
                # activates it.  Admissibility is the complement of any
                # fallback, so it remains a useful owner-level gate.
                fallback_flags = {
                    name: aggregate_flag(value)
                    for name, value in parallel_material_diagnostics.items()
                    if name not in ("admissible", "ordinary_row")
                }
                parallel_material_diagnostics = {
                    **fallback_flags,
                    "ordinary_row": ~fallback_flags["wall_row"],
                    "admissible": ~fallback_flags["fallback"],
                }
        result = {
            "parallel_div_b": div_b,
            "density_flux_div": flux_div("density_flux"),
            "current_flux_div": flux_div("current"),
            "vorticity_current_flux_div": flux_div("vorticity_current"),
            "parallel_Vi_flux_div": flux_div("Vi"),
            "Ve_flux_div": flux_div("Ve"),
            "grad_density": gradient_values["density"],
            "grad_Te": gradient_values["Te"],
            "grad_Ti": gradient_values["Ti"],
            "grad_Ve": gradient_values["Ve"],
            "grad_Vi": gradient_values["Vi"],
            "grad_phi": gradient_values["phi"],
            "grad_Pe": gradient_values["Pe"],
            "grad_pressure": gradient_values["pressure"],
            "grad_current": gradient_values["current"],
            "grad_vorticity": gradient_values["vorticity"],
            "material_upwind_correction": material_upwind_correction,
            "parallel_material_residual": parallel_material_residual,
            "parallel_material_diagnostics": parallel_material_diagnostics,
            # Boundary characteristic-SAT decomposition.  The homogeneous
            # term is the current/phi paired operator; the affine lift is
            # applied only to the vorticity current divergence.
            "characteristic_sat_homogeneous_current_divergence": (
                jnp.zeros_like(div_b)
                if characteristic_sat_homogeneous_current_divergence is None
                else characteristic_sat_homogeneous_current_divergence
            ),
            "characteristic_sat_affine_current_divergence": (
                jnp.zeros_like(div_b)
                if characteristic_sat_affine_current_divergence is None
                else characteristic_sat_affine_current_divergence
            ),
            "characteristic_sat_current_divergence": (
                jnp.zeros_like(div_b)
                if characteristic_sat_current_divergence is None
                else characteristic_sat_current_divergence
            ),
        }
        if return_electron_force_diagnostics:
            result.update(
                {
                    "electron_force_gradient_components": jnp.stack(
                        tuple(
                            diagnostic_gradient_components[name]
                            for name in ("Ve", "phi", "Pe", "Te")
                        ),
                        axis=0,
                    ),
                    "electron_force_endpoint_values": jnp.stack(
                        tuple(
                            diagnostic_endpoint_values[name]
                            for name in diagnostic_names
                        ),
                        axis=0,
                    ),
                    "material_upwind_correction_components": (
                        material_upwind_correction_components
                    ),
                    "material_centered_principal": material_centered_principal,
                    "material_upwind_principal": material_upwind_principal,
                    "material_characteristic_endpoint_values": (
                        material_characteristic_endpoint_values
                    ),
                    "material_characteristic_leg_lengths": (
                        material_characteristic_leg_lengths
                    ),
                }
            )
        if self.parallel_velocity_layout == "fci-staggered":
            # Vi/Ve are source-edge values.  Center flux divergences retain
            # the established FCI identity div(F b)=B D(F/B). ``B_face`` is
            # the reciprocal of the mapped average of 1/B (a harmonic
            # outgoing-edge B), so a uniform B reduces exactly to bare D.
            # Direct forward-edge G(f) is already the physical derivative at
            # that edge and is intentionally left unmodified here.
            def remote(field_halo: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                return self._fci_remote_values(field_halo, context)

            def c2f(field_halo: jnp.ndarray) -> jnp.ndarray:
                forward, backward = remote(field_halo)
                return local_center_to_outgoing_face_average_fci_op(
                    field_halo, self.geometry, context=context,
                    forward_remote_values=forward, backward_remote_values=backward,
                )

            use_face_galerkin = (
                self.control_volume_geometry is not None
                and self.control_volume_geometry.has_angular_agglomeration
                and self.outgoing_face_topology is not None
            )

            face_galerkin, homogeneous_fine_forward_gradient = (
                self._fci_face_galerkin_core(context, face_bc.density)
                if use_face_galerkin else (None, None)
            )

            def g(field_halo: jnp.ndarray) -> jnp.ndarray:
                if face_galerkin is not None:
                    # Keep the field's actual prepared wall trace.  Only D
                    # uses the homogeneous core; this is R_e G_actual P_c.
                    forward, backward = remote(field_halo)
                    fine_gradient = local_center_to_outgoing_face_grad_parallel_fci_op(
                        field_halo, self.geometry, context=context,
                        forward_remote_values=forward, backward_remote_values=backward,
                    )
                    owner_gradient = face_galerkin.face_restrict(fine_gradient)
                    return face_galerkin.face_prolong(owner_gradient)
                forward, backward = remote(field_halo)
                return local_center_to_outgoing_face_grad_parallel_fci_op(
                    field_halo, self.geometry, context=context,
                    forward_remote_values=forward, backward_remote_values=backward,
                )

            def d(face_halo: jnp.ndarray) -> jnp.ndarray:
                if face_galerkin is not None:
                    owner_face = face_galerkin.face_restrict(face_halo[owned])
                    owner_div = face_galerkin.coarse_divergence(
                        owner_face, homogeneous_fine_forward_gradient,
                    )
                    # The surrounding staggered RHS still completes its
                    # ordinary projected-fine-grid restriction, so return a
                    # fine storage representation exactly once here.
                    return face_galerkin.cell_prolong(owner_div)
                forward, backward = remote(face_halo)
                return local_outgoing_face_to_center_div_parallel_fci_op(
                    face_halo, self.geometry, context=context,
                    forward_remote_values=forward, backward_remote_values=backward,
                )

            def f2c(face_halo: jnp.ndarray) -> jnp.ndarray:
                forward, backward = remote(face_halo)
                return local_outgoing_face_to_center_average_fci_op(
                    face_halo, self.geometry, context=context,
                    forward_remote_values=forward, backward_remote_values=backward,
                )

            inverse_b_face = local_center_to_outgoing_face_average_fci_op(
                inverse_b_halo, self.geometry, context=context,
                forward_remote_values=inverse_b_forward,
                backward_remote_values=inverse_b_backward,
            )
            inverse_b_face_bc = _dirichlet_face_bc_from_values(
                tuple(
                    1.0 / jnp.maximum(
                        jnp.asarray(face.Bmag_owned, dtype=jnp.float64), 1.0e-30
                    )
                    for face in self.geometry.face_bfield.axes
                ),
                self.domain.layout,
                (face_bc.phi.mask_x, face_bc.phi.mask_y, face_bc.phi.mask_z),
            )
            inverse_b_face_halo = self._prepare_fine_storage_halo(
                inverse_b_face, inverse_b_face_bc
            )
            B_center = jnp.maximum(
                jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
                1.0e-30,
            )
            def flux_div(face_flux_halo: jnp.ndarray) -> jnp.ndarray:
                return B_center * d(face_flux_halo * inverse_b_face_halo)

            n_face = c2f(state_halo.density)
            n_face_halo = self._prepare_fine_storage_halo(n_face, face_bc.density)
            Vi_face_halo, Ve_face_halo = state_halo.Vi, state_halo.Ve
            nVe_face_halo = n_face_halo * Ve_face_halo
            j_face_halo = n_face_halo * (Vi_face_halo - Ve_face_halo)
            Vi_center = f2c(Vi_face_halo)
            Ve_center = f2c(Ve_face_halo)
            # Velocity gradients and parallel diffusion are cell-centred.
            # Remove source-subface modes before either enters the compatible
            # G/D pair.
            Vi_center_halo = self._prepare_cell_rlp_halo_from_fine(
                Vi_center, face_bc.Vi
            )
            Ve_center_halo = self._prepare_cell_rlp_halo_from_fine(
                Ve_center, face_bc.Ve
            )
            current_face_halo = j_face_halo
            vorticity_product_halo = self._prepare_fine_storage_halo(
                Vi_face_halo[owned] * g(state_halo.vorticity), face_bc.vorticity
            )

            result.update({
                "density_flux_div": flux_div(nVe_face_halo),
                "current_flux_div": flux_div(j_face_halo),
                "vorticity_current_flux_div": flux_div(current_face_halo),
                "parallel_Vi_flux_div": flux_div(Vi_face_halo),
                "Ve_flux_div": flux_div(Ve_face_halo),
                "grad_Te": g(state_halo.Te), "grad_Ti": g(state_halo.Ti),
                "grad_Vi": g(Vi_center_halo), "grad_Ve": g(Ve_center_halo),
                "grad_phi": g(state_halo.phi), "grad_Pe": g(fields["Pe"]),
                "grad_pressure": g(fields["pressure"]),
                "grad_current": g(self._prepare_fine_storage_halo(f2c(j_face_halo), face_bc.Ve)),
                "grad_vorticity": g(state_halo.vorticity),
                "staggered_n_face": n_face,
                "staggered_Vi_face": Vi_face_halo[owned],
                "staggered_Ve_face": Ve_face_halo[owned],
                "staggered_j_face": j_face_halo[owned],
                "staggered_Te_advection": f2c(self._prepare_fine_storage_halo(Ve_face_halo[owned] * g(state_halo.Te), face_bc.Te)),
                "staggered_Ti_advection": f2c(self._prepare_fine_storage_halo(Vi_face_halo[owned] * g(state_halo.Ti), face_bc.Ti)),
                "staggered_vorticity_advection": f2c(vorticity_product_halo),
            })

        # In staggered mode the first-order face pair above is the compatible
        # parallel operator.  Keep every diffusion operand on that pair rather
        # than feeding the legacy cell-centred diffusion result into a face
        # RHS slot.  Vi/Ve first reconstruct to centres, apply B D(B^-1 G),
        # then return to outgoing-edge storage.
        if self.parallel_velocity_layout == "fci-staggered":
            diffusion_halos = {
                "density": state_halo.density,
                "Te": state_halo.Te,
                "Ti": state_halo.Ti,
                "Vi": Vi_center_halo,
                "Ve": Ve_center_halo,
                "vorticity": state_halo.vorticity,
            }
            diffusion_bcs = {
                "density": face_bc.density,
                "Te": face_bc.Te,
                "Ti": face_bc.Ti,
                "Vi": face_bc.Vi,
                "Ve": face_bc.Ve,
                "vorticity": face_bc.vorticity,
            }
            for name, coefficient in (
                ("density", self.parameters.density_D_parallel),
                ("Te", self.parameters.electron_temperature_chi_parallel),
                ("Ti", self.parameters.ion_temperature_chi_parallel),
                ("Vi", self.parameters.Vi_parallel_viscosity),
                ("Ve", self.parameters.Ve_parallel_viscosity),
                ("vorticity", self.parameters.vorticity_D_parallel),
            ):
                if float(coefficient) == 0.0:
                    value = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
                else:
                    gradient_halo = self._prepare_fine_storage_halo(
                        g(diffusion_halos[name]), diffusion_bcs[name]
                    )
                    value = jnp.asarray(coefficient, dtype=jnp.float64) * flux_div(
                        gradient_halo
                    )
                if name in ("Vi", "Ve"):
                    value = self._cell_force_to_outgoing_face_mass_adjoint(
                        value, diffusion_bcs[name], context
                    )
                result[f"{name}_parallel_diff"] = value
            return result

        for name, coefficient in (
            ("density", self.parameters.density_D_parallel),
            ("Te", self.parameters.electron_temperature_chi_parallel),
            ("Ti", self.parameters.ion_temperature_chi_parallel),
            ("Vi", self.parameters.Vi_parallel_viscosity),
            ("Ve", self.parameters.Ve_parallel_viscosity),
            ("vorticity", self.parameters.vorticity_D_parallel),
        ):
            if float(coefficient) == 0.0:
                result[f"{name}_parallel_diff"] = jnp.zeros(
                    self.geometry.owned_shape, dtype=jnp.float64
                )
                continue
            field_remote = self._fci_remote_values(fields[name], context)
            diffusivity_halo = jnp.ones_like(fields[name], dtype=jnp.float64)
            diffusivity_remote = self._fci_remote_values(diffusivity_halo, context)
            value = jnp.asarray(
                coefficient, dtype=jnp.float64
            ) * local_parallel_diffusion_fci_op(
                fields[name],
                self.geometry,
                context=context,
                diffusivity_halo_full=diffusivity_halo,
                inverse_b_halo_full=inverse_b_halo,
                forward_remote_values=field_remote[0],
                backward_remote_values=field_remote[1],
                forward_remote_diffusivity_values=diffusivity_remote[0],
                backward_remote_diffusivity_values=diffusivity_remote[1],
                forward_remote_inverse_b_values=inverse_b_forward,
                backward_remote_inverse_b_values=inverse_b_backward,
            )
            result[f"{name}_parallel_diff"] = (
                aggregate_local_control_volume_average(
                    value, self.control_volume_geometry.cells, self.domain
                ) if self._uses_compact_face_operators else value
            )

        return result

    def _parallel_operator_boundary(
        self,
        *,
        state_halo: FciDrbEBState,
        operator_boundary: LocalFciDrbEBOperatorBoundaryBundle,
    ) -> LocalFciDrbEBOperatorBoundaryBundle:
        """Build wall traces for the five-field local characteristic closure.

        The base bundle remains the source for potential, vorticity, curvature,
        and diffusion.  Only the first-order parallel traces are replaced, and
        only on runtime physical coordinate faces.  The adjacent owner cell is
        used for both the frozen matrix and the outgoing state; the existing
        primitive operator traces supply the candidate incoming state for the
        local characteristic closure.  The equilibrium characteristic closure
        instead sets incoming perturbations to zero around
        ``(n, Te, Ti, Vi, Ve) = (1, 1, 1, 0, 0)``.
        """

        if self.parallel_inflow_closure == "central":
            return operator_boundary

        owned = self.domain.layout.owned_slices_cell
        owner_fields = tuple(
            jnp.asarray(field[owned], dtype=jnp.float64)
            for field in (
                state_halo.density,
                state_halo.Te,
                state_halo.Ti,
                state_halo.Vi,
                state_halo.Ve,
            )
        )
        base_traces = (
            operator_boundary.density,
            operator_boundary.Te,
            operator_boundary.Ti,
            operator_boundary.Vi,
            operator_boundary.Ve,
        )
        values = [
            [jnp.array(trace.value_x), jnp.array(trace.value_y), jnp.array(trace.value_z)]
            for trace in base_traces
        ]

        for axis, name in enumerate(("x", "y", "z")):
            face_bfield = getattr(self.geometry.face_bfield, name)
            face_metric = getattr(self.geometry.face_metric, name)
            b_contra = jnp.asarray(
                face_bfield.B_contra_owned[..., axis], dtype=jnp.float64
            )
            bmag = jnp.maximum(
                jnp.asarray(face_bfield.Bmag_owned, dtype=jnp.float64), 1.0e-30
            )
            gnn = jnp.maximum(
                jnp.asarray(face_metric.g_contra_owned[..., axis, axis], dtype=jnp.float64),
                1.0e-30,
            )
            b_normal = (b_contra / bmag) / jnp.sqrt(gnn)
            for side in (0, 1):
                side_kind = (
                    self.domain.shard_spec.lower_side_kind(axis)
                    if side == 0
                    else self.domain.shard_spec.upper_side_kind(axis)
                )
                if side_kind != SIDE_PHYSICAL:
                    continue
                # A lower radial axis is a coordinate singularity, not a wall.
                if side == 0 and self.axis_regular_axes[axis]:
                    continue
                sl = _axis_plane_slice(axis, side)
                mask = jnp.asarray(getattr(base_traces[0], f"mask_{name}")[sl], dtype=bool)

                owner = jnp.stack(
                    tuple(field[sl] for field in owner_fields),
                    axis=-1,
                )
                matrix = parallel_characteristic_matrix(
                    owner[..., 0], owner[..., 1], owner[..., 2],
                    owner[..., 3], owner[..., 4],
                    self.parameters.tau, self.parameters.mi_over_me,
                )
                outward_normal = (-b_normal if side == 0 else b_normal)[sl]
                if self.parallel_inflow_closure == "equilibrium-characteristic":
                    equilibrium = jnp.asarray(
                        (1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64
                    )
                    wall = parallel_equilibrium_characteristic_wall_state(
                        owner, equilibrium, matrix, outward_normal
                    )
                else:
                    candidate = jnp.stack(
                        tuple(
                            getattr(trace, f"value_{name}")[sl]
                            for trace in base_traces
                        ),
                        axis=-1,
                    )
                    wall = parallel_characteristic_wall_state(
                        owner, candidate, matrix, outward_normal
                    )
                wall = jnp.where(mask[..., None], wall, owner)
                for component in range(5):
                    values[component][axis] = values[component][axis].at[sl].set(
                        jnp.where(mask, wall[..., component], values[component][axis][sl])
                    )

        def trace_with_values(trace, value_axes):
            return LocalBoundaryFaceTrace3D(
                value_x=value_axes[0], value_y=value_axes[1], value_z=value_axes[2],
                mask_x=trace.mask_x, mask_y=trace.mask_y, mask_z=trace.mask_z,
                layout=trace.layout,
            )

        density, Te, Ti, Vi, Ve = tuple(
            trace_with_values(trace, value_axes)
            for trace, value_axes in zip(base_traces, values)
        )
        density_flux = _combine_operator_traces(density, Ve, operation=lambda n, ve: n * ve)
        current = _combine_operator_traces(density, Vi, Ve, operation=lambda n, vi, ve: n * (vi - ve))
        Pe = _combine_operator_traces(density, Te, operation=lambda n, te: n * te)
        pressure = _combine_operator_traces(
            density, Te, Ti,
            operation=lambda n, te, ti: n * (te + jnp.asarray(self.parameters.tau, dtype=jnp.float64) * ti),
        )
        return replace(
            operator_boundary,
            density=density, Te=Te, Ti=Ti, Vi=Vi, Ve=Ve,
            density_flux=density_flux, current=current, Pe=Pe, pressure=pressure,
        )

    def _coordinate_stage_parallel_terms(
        self,
        *,
        state_halo: FciDrbEBState,
        context: StencilBuilderContext,
        operator_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        parallel_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        parallel_div_b: jnp.ndarray,
        density_flux_stencil: ConservativeStencil3D,
        current_stencil: ConservativeStencil3D,
        Ve_stencil: ConservativeStencil3D,
        Vi_stencil: ConservativeStencil3D,
        Te_stencil: ConservativeStencil3D,
        Ti_stencil: ConservativeStencil3D,
        phi_stencil: ConservativeStencil3D,
        Pe_stencil: ConservativeStencil3D,
        pressure_stencil: ConservativeStencil3D,
        vorticity_stencil: ConservativeStencil3D,
        primitive_cv_closures: dict[str, object] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Evaluate the pre-existing coordinate parallel stage operators."""

        operator_kwargs = dict(
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            control_volume_geometry=(
                self.control_volume_geometry
                if self._uses_compact_face_operators else None
            ),
        )
        primitive = (
            primitive_cv_closures
            if primitive_cv_closures is not None
            else self._primitive_cv_closures(state_halo)
        )
        density_flux_closure = self._cv_product(primitive.get("density"), primitive.get("Ve")) if primitive else None
        velocity_difference = self._cv_linear_combination(primitive.get("Vi"), primitive.get("Ve"), b=-1.0) if primitive else None
        current_closure = self._cv_product(primitive.get("density"), velocity_difference) if primitive else None
        pressure_temperature = self._cv_linear_combination(primitive.get("Te"), primitive.get("Ti"), b=self.parameters.tau) if primitive else None
        Pe_closure = self._cv_product(primitive.get("density"), primitive.get("Te")) if primitive else None
        pressure_closure = self._cv_product(primitive.get("density"), pressure_temperature) if primitive else None
        density_flux_div = local_parallel_flux_div_op(
            density_flux_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.density_flux,
            field_closure=density_flux_closure, **operator_kwargs
        )
        current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.current,
            field_closure=current_closure, **operator_kwargs
        )
        vorticity_current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=(
                parallel_boundary.current
                if self.vorticity_current_inflow_trace == "parallel-characteristic"
                else operator_boundary.current
            ),
            field_closure=current_closure, **operator_kwargs
        )
        Vi_stencil = build_local_conservative_stencil_from_field(
            state_halo.Vi, self.geometry, context
        )
        parallel_Vi_flux_div = local_parallel_flux_div_op(
            Vi_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Vi,
            field_closure=primitive.get("Vi"), **operator_kwargs
        )
        Ve_flux_div = local_parallel_flux_div_op(
            Ve_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Ve,
            field_closure=primitive.get("Ve"), **operator_kwargs
        )
        grad_Te = local_grad_parallel_op_conservative(
            Te_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Te,
            field_closure=primitive.get("Te"), **operator_kwargs
        )
        grad_Ti = local_grad_parallel_op_conservative(
            Ti_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ti,
            field_closure=primitive.get("Ti"), **operator_kwargs
        )
        grad_Ve = local_grad_parallel_op_conservative(
            Ve_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ve,
            field_closure=primitive.get("Ve"), **operator_kwargs
        )
        grad_Vi = local_grad_parallel_op_conservative(
            Vi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Vi,
            field_closure=primitive.get("Vi"), **operator_kwargs
        )
        grad_phi = local_grad_parallel_op_conservative(
            phi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.phi,
            field_closure=primitive.get("phi"), **operator_kwargs
        )
        grad_Pe = local_grad_parallel_op_conservative(
            Pe_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Pe,
            field_closure=Pe_closure, **operator_kwargs
        )
        grad_pressure = local_grad_parallel_op_conservative(
            pressure_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.pressure,
            field_closure=pressure_closure, **operator_kwargs
        )
        grad_current = local_grad_parallel_op_conservative(
            current_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.current,
            field_closure=current_closure, **operator_kwargs
        )
        grad_vorticity = local_grad_parallel_op_conservative(
            vorticity_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.vorticity,
            field_closure=primitive.get("vorticity"), **operator_kwargs
        )
        return {
            "parallel_div_b": parallel_div_b,
            "density_flux_div": density_flux_div,
            "current_flux_div": current_flux_div,
            "vorticity_current_flux_div": vorticity_current_flux_div,
            "parallel_Vi_flux_div": parallel_Vi_flux_div,
            "Ve_flux_div": Ve_flux_div,
            "grad_Te": grad_Te,
            "grad_Ti": grad_Ti,
            "grad_Ve": grad_Ve,
            "grad_Vi": grad_Vi,
            "grad_phi": grad_phi,
            "grad_Pe": grad_Pe,
            "grad_pressure": grad_pressure,
            "grad_current": grad_current,
            "grad_vorticity": grad_vorticity,
        }

    def _reconstruct_phi_from_prepared(
        self,
        state_owned: FciDrbEBState,
        state_halo: FciDrbEBState,
        face_bc: LocalFciDrbEBFaceBCBundle,
        *,
        return_diagnostics: bool = False,
        ti_field_closure=None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        context = self._stencil_builder_context()
        ti_conservative = build_local_conservative_stencil_from_field(
            state_halo.Ti,
            self.geometry,
            context,
        )
        ti_laplacian = local_perp_laplacian_conservative_op(
            ti_conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            **self._cv_operator_args(
                state_halo.Ti, face_bc.Ti, field_closure=ti_field_closure
            ),
        )
        ti_laplacian = self._restrict_fine_field(ti_laplacian)
        owned = self.domain.layout.owned_slices_cell
        phi_rhs = (
            jnp.asarray(self.parameters.tau, dtype=jnp.float64) * ti_laplacian
            - jnp.asarray(state_owned.vorticity, dtype=jnp.float64)
        )
        phi_lift = jnp.asarray(state_owned.phi, dtype=jnp.float64)
        solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            control_volume_geometry=self.control_volume_geometry,
            control_volume_boundary_bc=self.control_volume_boundary_bc,
            stencil_builder=build_local_conservative_stencil_from_field,
            stencil_builder_context=context,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            face_bc=face_bc.phi,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            config=self.gmres_config,
        )
        if self.control_volume_geometry is not None:
            # Pole-CV has a distinct owner-space unknown topology.  Keep this
            # route explicit so it cannot accidentally fall through the
            # ordinary solver's backwards-compatible __call__ entry point.
            phi_result = solver.solve_rlp_owner(
                phi_rhs,
                guess_owned=state_owned.phi,
                phi_lift_owned=phi_lift,
                return_diagnostics=return_diagnostics,
            )
        else:
            phi_result = solver.solve_full_grid(
                phi_rhs,
                guess_owned=state_owned.phi,
                phi_lift_owned=phi_lift,
                return_diagnostics=return_diagnostics,
            )
        if return_diagnostics:
            phi_owned, info = phi_result
            return self._owner_result(_mask_inactive_owned(phi_owned, self.geometry)), info
        return self._owner_result(_mask_inactive_owned(phi_result, self.geometry))

    def reconstruct_phi(
        self,
        state_owned: FciDrbEBState,
        *,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        face_bc = self._face_bcs(state_owned)
        state_halo = self._prepare_state_halo(state_owned, face_bc)
        return self._reconstruct_phi_from_prepared(
            state_owned,
            state_halo,
            face_bc,
            return_diagnostics=return_diagnostics,
        )

    def polarization_residual(
        self,
        state_owned: FciDrbEBState,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return the algebraic Boussinesq-polarization residual."""

        terms = self.polarization_balance_terms(
            state_owned,
            phi_owned=phi_owned,
        )
        return _mask_inactive_owned(
            terms[0] - terms[1] - terms[2],
            self.geometry,
        )

    def polarization_balance_terms(
        self,
        state_owned: FciDrbEBState,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return ``(-Lperp phi, tau Lperp Ti, -omega)`` in owner space.

        This diagnostic decomposition uses the exact production closures and
        RLP restriction.  Consequently ``terms[0] - terms[1] - terms[2]`` is
        bit-for-bit the algebraic residual returned by
        :meth:`polarization_residual`, apart from the final inactive mask.
        """

        face_bc = self._face_bcs(state_owned)
        state_halo = self._prepare_state_halo(state_owned, face_bc)
        if phi_owned is None:
            phi_owned = state_owned.phi
        phi_owned = _mask_inactive_owned(
            jnp.asarray(phi_owned, dtype=jnp.float64), self.geometry
        )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        context = self._stencil_builder_context()
        phi_conservative = build_local_conservative_stencil_from_field(
            phi_halo, self.geometry, context
        )
        ti_conservative = build_local_conservative_stencil_from_field(
            state_halo.Ti, self.geometry, context
        )
        phi_laplacian = local_perp_laplacian_conservative_op(
            phi_conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.phi,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            **self._cv_operator_args(phi_halo, face_bc.phi),
        )
        ti_laplacian = local_perp_laplacian_conservative_op(
            ti_conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            **self._cv_operator_args(state_halo.Ti, face_bc.Ti),
        )
        phi_laplacian = self._restrict_fine_field(phi_laplacian)
        ti_laplacian = self._restrict_fine_field(ti_laplacian)
        return jnp.stack(
            (
                -phi_laplacian,
                jnp.asarray(self.parameters.tau, dtype=jnp.float64)
                * ti_laplacian,
                -jnp.asarray(state_owned.vorticity, dtype=jnp.float64),
            ),
            axis=0,
        )

    def electron_parallel_force_diagnostics(
        self,
        state_owned: FciDrbEBState,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Return exact wall-face diagnostics for the electron parallel force.

        The outputs are, respectively, total force terms, directional force
        terms, compatible-gradient components, mapped endpoint field values,
        physical-wall masks, and mapped leg lengths.  Directional arrays use
        backward/center/forward stencil order.  This is a replay-only
        diagnostic and leaves the production RHS unchanged.
        """

        if self.parallel_operator_scheme != "fci":
            raise ValueError("electron force wall diagnostics require FCI")
        if self.parallel_velocity_layout != "cell-centered":
            raise ValueError(
                "electron force wall diagnostics currently require "
                "cell-centered parallel velocities"
            )
        face_bc = self._face_bcs(state_owned)
        state_halo_without_phi = self._prepare_state_halo(state_owned, face_bc)
        if phi_owned is None:
            phi_owned = self._reconstruct_phi_from_prepared(
                state_owned, state_halo_without_phi, face_bc
            )
        else:
            phi_owned = _mask_inactive_owned(
                jnp.asarray(phi_owned, dtype=jnp.float64), self.geometry
            )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        state_halo = state_halo_without_phi.replace(phi=phi_halo)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo,
            self.geometry,
            self.domain,
            face_bc,
            tau=self.parameters.tau,
        )
        context = self._stencil_builder_context()
        parallel_boundary = self._parallel_operator_boundary(
            state_halo=state_halo,
            operator_boundary=operator_boundary,
        )
        parallel_terms = self._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=context,
            return_electron_force_diagnostics=True,
        )

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
        density_safe = jnp.maximum(density, 1.0e-30)
        mi_over_me = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)
        collision_frequency = jnp.asarray(
            self.parameters.Ve_nu, dtype=jnp.float64
        )
        gradient_components = parallel_terms[
            "electron_force_gradient_components"
        ]
        characteristic_components = jnp.moveaxis(
            parallel_terms["material_upwind_correction_components"][..., :, 4],
            -1,
            0,
        )
        directional_force_terms = jnp.stack(
            (
                -Ve[None, ...] * gradient_components[0],
                mi_over_me * gradient_components[1],
                -mi_over_me * gradient_components[2] / density_safe[None, ...],
                -0.71 * mi_over_me * gradient_components[3],
                characteristic_components,
            ),
            axis=0,
        )
        force_terms = jnp.stack(
            (
                -Ve * parallel_terms["grad_Ve"],
                mi_over_me * collision_frequency * density * (Vi - Ve),
                mi_over_me * parallel_terms["grad_phi"],
                -mi_over_me * parallel_terms["grad_Pe"] / density_safe,
                -0.71 * mi_over_me * parallel_terms["grad_Te"],
                (
                    parallel_terms["parallel_material_residual"][..., 4]
                    if self.parallel_material_scheme == "production-path"
                    else parallel_terms["material_upwind_correction"][..., 4]
                ),
                parallel_terms["vorticity_current_flux_div"],
            ),
            axis=0,
        )
        # Match ``evaluate_stage.pack_rhs_terms`` exactly: nonlinear force
        # products are assembled on the fine representation and only then
        # restricted to canonical RLP owners.
        restrict_force = lambda value: self._owner_field(
            _mask_inactive_owned(self._restrict_fine_field(value), self.geometry)
        )
        force_terms = jax.vmap(restrict_force)(force_terms)
        directional_force_terms = jax.vmap(jax.vmap(restrict_force))(
            directional_force_terms
        )
        # Replay the compatible gradients in the same owner representation
        # as the RHS.  Selecting owner entries from the fine transpose output
        # is not equivalent to the physical-volume RLP restriction and can
        # manufacture a pairing remainder in post-processing even when the
        # live fine operator is exactly adjoint.
        gradient_components = jax.vmap(jax.vmap(restrict_force))(
            gradient_components
        )
        characteristic_principal_terms = jax.vmap(restrict_force)(
            jnp.stack(
                (
                    parallel_terms["material_centered_principal"][..., 4],
                    parallel_terms["material_upwind_principal"][..., 4],
                ),
                axis=0,
            )
        )
        characteristic_primitive_endpoint_values = jnp.moveaxis(
            parallel_terms["material_characteristic_endpoint_values"],
            (-1, -2),
            (0, 1),
        )
        wall_masks = jnp.stack(
            (
                self.geometry.maps.backward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY,
                self.geometry.maps.forward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY,
            ),
            axis=0,
        )
        leg_lengths = jnp.stack(
            (
                parallel_terms["material_characteristic_leg_lengths"][..., 0],
                parallel_terms["material_characteristic_leg_lengths"][..., 1],
            ),
            axis=0,
        )
        endpoint_kinds = jnp.stack(
            (
                self.geometry.maps.backward.endpoint_kind,
                self.geometry.maps.forward.endpoint_kind,
            ),
            axis=0,
        )
        return (
            force_terms,
            directional_force_terms,
            gradient_components,
            parallel_terms["electron_force_endpoint_values"],
            wall_masks,
            leg_lengths,
            characteristic_principal_terms,
            characteristic_primitive_endpoint_values,
            endpoint_kinds,
        )

    def apply_short_leg_implicit_material_step(
        self,
        state_owned: FciDrbEBState,
        solve_dt: Any,
        selection_dt: Any | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> FciDrbEBState:
        """Apply a frozen local backward-Euler step to selected wall legs.

        This is intentionally a narrow prototype operator.  It updates only
        the five primitive material fields on rows whose physical wall leg
        exceeds ``parallel_short_leg_cfl_limit`` when measured with
        ``selection_dt``.  All mapped/bulk rows, the geometric ``div(b)``
        source, polarization, and vorticity remain in the explicit RHS.  The
        owner update is formed from the BE increment, so projected-fine/RLP
        storage cannot accidentally average an already-updated state.
        """
        if self.parallel_short_leg_treatment != "local-backward-euler":
            raise ValueError(
                "apply_short_leg_implicit_material_step requires "
                "parallel_short_leg_treatment='local-backward-euler'"
            )
        if self.parallel_material_scheme != "production-path" or self.parallel_operator_scheme != "fci":
            raise ValueError(
                "short-leg implicit material step requires the production FCI path"
            )
        if self.parallel_velocity_layout != "cell-centered":
            raise ValueError(
                "short-leg implicit material step requires cell-centered velocities"
            )
        solve_dt = jnp.asarray(solve_dt, dtype=jnp.float64)
        if selection_dt is None:
            selection_dt = solve_dt
        selection_dt = jnp.asarray(selection_dt, dtype=jnp.float64)

        state_owned = self._owner_state(state_owned)
        face_bc = self._face_bcs(state_owned)
        state_halo_without_phi = self._prepare_state_halo(state_owned, face_bc)
        if phi_owned is None:
            # The material wall block depends only on the five primitive
            # traces.  Do not trigger a second elliptic solve when the caller
            # is already carrying the diagnostic potential; phi is included
            # below solely to build the common boundary bundle.
            phi_owned = _mask_inactive_owned(state_owned.phi, self.geometry)
        else:
            phi_owned = _mask_inactive_owned(
                jnp.asarray(phi_owned, dtype=jnp.float64), self.geometry
            )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        state_halo = state_halo_without_phi.replace(phi=phi_halo)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, self.geometry, self.domain, face_bc,
            tau=self.parameters.tau,
        )
        parallel_boundary = self._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary,
        )
        context = self._stencil_builder_context()
        owned = self.domain.layout.owned_slices_cell
        primitive_names = ("density", "Te", "Ti", "Vi", "Ve")
        primitive_stencils = []
        fields = {
            "density": state_halo.density,
            "Te": state_halo.Te,
            "Ti": state_halo.Ti,
            "Vi": state_halo.Vi,
            "Ve": state_halo.Ve,
        }
        traces = {
            "density": parallel_boundary.density,
            "Te": parallel_boundary.Te,
            "Ti": parallel_boundary.Ti,
            "Vi": parallel_boundary.Vi,
            "Ve": parallel_boundary.Ve,
        }
        for name in primitive_names:
            field_halo, forward_remote, backward_remote = self._fci_prepare_q(
                fields[name][owned], traces[name], context
            )
            primitive_stencils.append(
                build_local_fci_stencil_from_field(
                    field_halo, self.geometry, context,
                    forward_remote_values=forward_remote,
                    backward_remote_values=backward_remote,
                )
            )
        center = jnp.stack(tuple(s.center for s in primitive_stencils), axis=-1)
        minus = jnp.stack(tuple(s.minus for s in primitive_stencils), axis=-1)
        plus = jnp.stack(tuple(s.plus for s in primitive_stencils), axis=-1)
        backward_wall = self.geometry.maps.backward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY
        forward_wall = self.geometry.maps.forward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY
        (
            updated,
            increment,
            info,
        ) = parallel_short_wall_backward_euler(
            center, minus, plus,
            primitive_stencils[0].dx_min,
            primitive_stencils[0].dx_plus,
            self.parameters.tau,
            self.parameters.mi_over_me,
            selection_dt=selection_dt,
            solve_dt=solve_dt,
            cfl_limit=self.parallel_short_leg_cfl_limit,
            backward_wall=backward_wall,
            forward_wall=forward_wall,
            backward_wall_state=minus,
            forward_wall_state=plus,
        )

        if self._uses_compact_face_operators:
            increment_owner = jax.vmap(
                lambda value: aggregate_local_control_volume_average(
                    value, self.control_volume_geometry.cells, self.domain
                )
            )(jnp.moveaxis(increment, -1, 0))
            increment_owner = jnp.moveaxis(increment_owner, 0, -1)
        else:
            increment_owner = increment
        if self.control_volume_geometry is not None:
            increment_owner = jnp.where(
                self.control_volume_geometry.cells.is_active_owner[..., None],
                increment_owner,
                0.0,
            )
        return self._owner_state(state_owned.replace(
            density=state_owned.density + increment_owner[..., 0],
            Te=state_owned.Te + increment_owner[..., 1],
            Ti=state_owned.Ti + increment_owner[..., 2],
            Vi=state_owned.Vi + increment_owner[..., 3],
            Ve=state_owned.Ve + increment_owner[..., 4],
        ))

    def evaluate_stage(
        self,
        state_owned: FciDrbEBState,
        source_owned: FciDrbEBState | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
        return_term_diagnostics: bool = False,
        return_term_fields: bool = False,
        return_rhs_term_fields: bool = False,
        return_curvature_component_fields: bool = False,
        short_leg_selection_dt: Any = 0.0,
    ) -> (
        FciDrbEBState
        | tuple[FciDrbEBState, jnp.ndarray]
        | tuple[FciDrbEBState, jnp.ndarray, jnp.ndarray]
    ):
        """Evaluate one EB RHS stage.

        When ``phi_owned`` is supplied it must be the potential reconstructed
        for ``state_owned``.  This avoids repeating the elliptic solve when a
        time integrator already carries a consistent diagnostic potential.
        The default preserves the standalone behavior and reconstructs
        ``phi`` internally.

        ``return_term_fields=True`` returns the full stacked eight-term
        electron-velocity RHS array instead of its maxima.  It is intended
        for postmortem localization and cannot be combined with
        ``return_term_diagnostics``.

        With ``return_term_diagnostics=True``, also return the local maximum
        absolute value of the electron-velocity RHS terms in this fixed order:
        Poisson bracket, parallel self-advection, collisional force,
        electrostatic force, electron-pressure force, thermal force,
        perpendicular diffusion, and parallel viscosity.  This diagnostic
        mode is intended for postmortem analysis and does not alter the normal
        compiled time-integration path.

        ``return_rhs_term_fields=True`` returns a padded array with shape
        ``(6, RHS_TERM_SLOT_COUNT, *owned_shape)``.  The field and term order
        is defined by ``RHS_TERM_FIELD_NAMES`` and ``RHS_TERM_NAMES``.  This
        all-equation diagnostic is mutually exclusive with the two legacy
        electron-velocity diagnostic modes.

        ``return_curvature_component_fields=True`` returns the exact
        equation-level conservative-curvature split with shape
        ``(4, 3, *owned_shape)``.  Equations are density, Te, Ti, and
        vorticity; directions are logical ``u``, ``theta``, and ``eta``.
        It may be combined with ``return_rhs_term_fields`` (yielding a
        three-item return) for a single frozen-state replay.
        """

        legacy_diagnostic_count = sum(
            bool(value) for value in (return_term_diagnostics, return_term_fields)
        )
        if legacy_diagnostic_count > 1 or (
            legacy_diagnostic_count
            and (return_rhs_term_fields or return_curvature_component_fields)
        ):
            raise ValueError(
                "RHS term diagnostic return modes are mutually exclusive"
            )

        if source_owned is None:
            source_owned = FciDrbEBState(
                density=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                phi=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                Te=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                Ti=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                Vi=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                Ve=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
                vorticity=jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64),
            )
        source_input = _mask_local_eb_state_inactive(source_owned, self.geometry)
        source_owned = self._owner_state(source_input)
        face_bc = self._face_bcs(state_owned)
        if self.parallel_velocity_layout == "fci-staggered":
            # User/model sources are center-owned fields.  Project them to
            # fine outgoing edges first, then R_e into the velocity storage.
            source_owned = source_owned.replace(
                Vi=self._owner_face_field(self._restrict_fine_face_field(
                    self._center_owned_to_outgoing_face(
                        self._owner_field(source_input.Vi), face_bc.Vi,
                        self._stencil_builder_context(),
                    )
                )),
                Ve=self._owner_face_field(self._restrict_fine_face_field(
                    self._center_owned_to_outgoing_face(
                        self._owner_field(source_input.Ve), face_bc.Ve,
                        self._stencil_builder_context(),
                    )
                )),
            )
        state_halo_without_phi = self._prepare_state_halo(state_owned, face_bc)
        primitive_cv_closures = None
        if phi_owned is None:
            primitive_cv_closures = self._primitive_cv_closures(
                state_halo_without_phi,
                names=("density", "Te", "Ti", "Vi", "Ve", "vorticity"),
            )
            phi_owned = self._reconstruct_phi_from_prepared(
                state_owned,
                state_halo_without_phi,
                face_bc,
                ti_field_closure=primitive_cv_closures.get("Ti"),
            )
        else:
            phi_owned = jnp.asarray(phi_owned, dtype=jnp.float64)
            if phi_owned.shape != self.geometry.owned_shape:
                raise ValueError(
                    "phi_owned must have shape "
                    f"{self.geometry.owned_shape}, got {phi_owned.shape}"
                )
            phi_owned = _mask_inactive_owned(phi_owned, self.geometry)
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        state_halo = state_halo_without_phi.replace(phi=phi_halo)
        if primitive_cv_closures is None:
            primitive_cv_closures = self._primitive_cv_closures(state_halo)
        else:
            primitive_cv_closures = {
                **primitive_cv_closures,
                **self._primitive_cv_closures(state_halo, names=("phi",)),
            }
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, self.geometry, self.domain, face_bc, tau=self.parameters.tau
        )
        context = self._stencil_builder_context()
        parallel_boundary = self._parallel_operator_boundary(
            state_halo=state_halo,
            operator_boundary=operator_boundary,
        )
        fci_parallel_terms = (
            self._fci_parallel_terms(
                state_halo=state_halo,
                face_bc=face_bc,
                operator_boundary=operator_boundary,
                parallel_boundary=parallel_boundary,
                context=context,
                short_leg_selection_dt=short_leg_selection_dt,
            )
            if self.parallel_operator_scheme == "fci"
            else None
        )
        if self.parallel_velocity_layout == "fci-staggered":
            # Perpendicular physics is center-located.  The primary Vi/Ve
            # state remains expanded outgoing-edge storage for the parallel
            # subsystem, but its cell gradients/stencils/diffusion see the
            # leg-length-weighted f2c reconstruction with real remote reads.
            Vi_perp_halo = self._outgoing_face_to_center_halo(
                state_halo.Vi, face_bc.Vi, context
            )
            Ve_perp_halo = self._outgoing_face_to_center_halo(
                state_halo.Ve, face_bc.Ve, context
            )
            perpendicular_operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
                state_halo.replace(Vi=Vi_perp_halo, Ve=Ve_perp_halo),
                self.geometry, self.domain, face_bc, tau=self.parameters.tau,
            )
        else:
            Vi_perp_halo = state_halo.Vi
            Ve_perp_halo = state_halo.Ve
            perpendicular_operator_boundary = operator_boundary
        # These state halos have already been closed with each field's
        # physical face BC.  Preserve those ghost values in the cell
        # gradients.  The one-sided physical builder is for
        # intermediate fields that do not have a physical ghost closure; using
        # it here silently discards the supplied Dirichlet/Neumann BCs.
        build_gradient = build_local_cell_gradient_from_field
        density_gradient = build_gradient(state_halo.density, self.geometry, context)
        Te_gradient = build_gradient(state_halo.Te, self.geometry, context)
        Ti_gradient = build_gradient(state_halo.Ti, self.geometry, context)
        Vi_gradient = build_gradient(Vi_perp_halo, self.geometry, context)
        Ve_gradient = build_gradient(Ve_perp_halo, self.geometry, context)
        vorticity_gradient = build_gradient(state_halo.vorticity, self.geometry, context)
        phi_gradient = build_gradient(state_halo.phi, self.geometry, context)

        # Derived regular-face boundary values used by conservative curvature.
        Pe_face_bc = _binary_local_dirichlet_face_bc(
            face_bc.density,
            face_bc.Te,
            lambda left, right: left * right,
        )
        ion_pressure_face_bc = _binary_local_dirichlet_face_bc(
            face_bc.density,
            face_bc.Ti,
            lambda left, right: left * right,
        )
        pressure_face_bc = _binary_local_dirichlet_face_bc(
            Pe_face_bc,
            _scale_local_dirichlet_face_bc(
                ion_pressure_face_bc,
                self.parameters.tau,
            ),
            lambda left, right: left + right,
        )

        Ve_conservative_stencil = build_local_conservative_stencil_from_field(
            Ve_perp_halo,
            self.geometry,
            context,
        )
        Vi_conservative_stencil = build_local_conservative_stencil_from_field(
            Vi_perp_halo,
            self.geometry,
            context,
        )
        Pe_halo = state_halo.density * state_halo.Te
        pressure_halo = (
            Pe_halo
            + self.parameters.tau * state_halo.density * state_halo.Ti
        )
        current_halo = state_halo.density * (Vi_perp_halo - Ve_perp_halo)
        density_flux_halo = state_halo.density * Ve_perp_halo

        if self.curvature_scheme == "direct":
            Pe_gradient = build_gradient(Pe_halo, self.geometry, context)
            pressure_gradient = build_gradient(pressure_halo, self.geometry, context)
        else:
            Pe_gradient = None
            pressure_gradient = None
        density_flux_conservative_stencil = (
            build_local_conservative_stencil_from_field(
                density_flux_halo,
                self.geometry,
                context,
            )
        )
        current_conservative_stencil = (
            build_local_conservative_stencil_from_field(
                current_halo,
                self.geometry,
                context,
            )
        )
        density_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.density,
            self.geometry,
            context,
        )
        Te_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Te,
            self.geometry,
            context,
        )
        Ti_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ti,
            self.geometry,
            context,
        )
        phi_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.phi,
            self.geometry,
            context,
        )
        Pe_conservative_stencil = build_local_conservative_stencil_from_field(
            Pe_halo,
            self.geometry,
            context,
        )
        pressure_conservative_stencil = build_local_conservative_stencil_from_field(
            pressure_halo,
            self.geometry,
            context,
        )
        vorticity_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.vorticity,
            self.geometry,
            context,
        )
        if self.parallel_operator_scheme == "coordinate":
            unit_conservative_stencil = build_local_conservative_stencil_from_field(
                jnp.ones_like(state_halo.density, dtype=jnp.float64),
                self.geometry,
                context,
            )
            parallel_div_b = local_parallel_div_b_op(
                unit_conservative_stencil,
                self.geometry,
                self.domain,
                regular_face_geometry=self.geometry.regular_face_geometry,
                axis_regular_axes=self.axis_regular_axes,
                control_volume_geometry=(
                    self.control_volume_geometry
                    if self._uses_compact_face_operators else None
                ),
                field_closure=self._cv_closure(jnp.ones_like(state_halo.density), None),
            )
        else:
            parallel_div_b = fci_parallel_terms["parallel_div_b"]

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Ti = jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64)
        Vi = jnp.asarray(Vi_perp_halo[owned], dtype=jnp.float64)
        Ve = jnp.asarray(Ve_perp_halo[owned], dtype=jnp.float64)
        density_safe = jnp.maximum(density, 1.0e-30)
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        tau = jnp.asarray(self.parameters.tau, dtype=jnp.float64)
        mi_over_me = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)
        Ve_nu = jnp.asarray(self.parameters.Ve_nu, dtype=jnp.float64)
        current = density * (Vi - Ve)

        density_diff = self._field_perp_diffusion(
            state_halo.density,
            face_bc.density,
            self.parameters.density_D_perp,
            field_closure=primitive_cv_closures.get("density"),
        )
        Te_diff = self._field_perp_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_D_perp,
            field_closure=primitive_cv_closures.get("Te"),
        )
        Ti_diff = self._field_perp_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_D_perp,
            field_closure=primitive_cv_closures.get("Ti"),
        )
        Vi_diff = self._field_perp_diffusion(
            Vi_perp_halo,
            face_bc.Vi,
            self.parameters.Vi_D_perp,
            field_closure=primitive_cv_closures.get("Vi"),
        )
        Ve_diff = self._field_perp_diffusion(
            Ve_perp_halo,
            face_bc.Ve,
            self.parameters.Ve_D_perp,
            field_closure=primitive_cv_closures.get("Ve"),
        )
        vorticity_diff = self._field_perp_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_perp,
            field_closure=primitive_cv_closures.get("vorticity"),
        )
        if self.parallel_operator_scheme == "coordinate":
            density_parallel_diff = self._field_parallel_diffusion(
                state_halo.density,
                face_bc.density,
                self.parameters.density_D_parallel,
                field_closure=primitive_cv_closures.get("density"),
            )
            Te_parallel_diff = self._field_parallel_diffusion(
                state_halo.Te,
                face_bc.Te,
                self.parameters.electron_temperature_chi_parallel,
                field_closure=primitive_cv_closures.get("Te"),
            )
            Ti_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ti,
                face_bc.Ti,
                self.parameters.ion_temperature_chi_parallel,
                field_closure=primitive_cv_closures.get("Ti"),
            )
            Vi_parallel_diff = self._field_parallel_diffusion(
                state_halo.Vi,
                face_bc.Vi,
                self.parameters.Vi_parallel_viscosity,
                field_closure=primitive_cv_closures.get("Vi"),
            )
            Ve_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ve,
                face_bc.Ve,
                self.parameters.Ve_parallel_viscosity,
                field_closure=primitive_cv_closures.get("Ve"),
            )
            vorticity_parallel_diff = self._field_parallel_diffusion(
                state_halo.vorticity,
                face_bc.vorticity,
                self.parameters.vorticity_D_parallel,
                field_closure=primitive_cv_closures.get("vorticity"),
            )
        else:
            density_parallel_diff = fci_parallel_terms["density_parallel_diff"]
            Te_parallel_diff = fci_parallel_terms["Te_parallel_diff"]
            Ti_parallel_diff = fci_parallel_terms["Ti_parallel_diff"]
            Vi_parallel_diff = fci_parallel_terms["Vi_parallel_diff"]
            Ve_parallel_diff = fci_parallel_terms["Ve_parallel_diff"]
            vorticity_parallel_diff = fci_parallel_terms["vorticity_parallel_diff"]
        if bool(self.diffusion_only):
            if return_rhs_term_fields or return_curvature_component_fields:
                raise ValueError(
                    "RHS/curvature field diagnostics are not supported with "
                    "diffusion_only"
                )
            diffusion_result = FciDrbEBState(
                density=density_diff,
                phi=jnp.zeros_like(phi_owned),
                Te=Te_diff + Te_parallel_diff,
                Ti=Ti_diff + Ti_parallel_diff,
                Vi=((self._cell_force_to_outgoing_face_mass_adjoint(
                        Vi_diff, face_bc.Vi, context)
                     if self.parallel_velocity_layout == "fci-staggered" else Vi_diff)
                    + Vi_parallel_diff),
                Ve=((self._cell_force_to_outgoing_face_mass_adjoint(
                        Ve_diff, face_bc.Ve, context)
                     if self.parallel_velocity_layout == "fci-staggered" else Ve_diff)
                    + Ve_parallel_diff),
                vorticity=vorticity_diff + vorticity_parallel_diff,
            )
            if self.parallel_velocity_layout == "fci-staggered":
                return self._owner_state(_mask_local_eb_state_inactive(
                    self._restrict_fine_state(diffusion_result), self.geometry
                ))
            return _mask_local_eb_state_inactive(diffusion_result, self.geometry)

        poisson_density = self._poisson_bracket_over_B(
            phi_gradient,
            density_gradient,
            phi_conservative_stencil,
            density_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.density,
            f_field_halo=state_halo.phi, g_field_halo=state_halo.density,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("density"),
        )
        poisson_Te = self._poisson_bracket_over_B(
            phi_gradient,
            Te_gradient,
            phi_conservative_stencil,
            Te_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Te,
            f_field_halo=state_halo.phi, g_field_halo=state_halo.Te,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("Te"),
        )
        poisson_Ti = self._poisson_bracket_over_B(
            phi_gradient,
            Ti_gradient,
            phi_conservative_stencil,
            Ti_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ti,
            f_field_halo=state_halo.phi, g_field_halo=state_halo.Ti,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("Ti"),
        )
        poisson_Vi = self._poisson_bracket_over_B(
            phi_gradient,
            Vi_gradient,
            phi_conservative_stencil,
            Vi_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=perpendicular_operator_boundary.Vi,
            f_field_halo=state_halo.phi, g_field_halo=Vi_perp_halo,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("Vi"),
        )
        poisson_Ve = self._poisson_bracket_over_B(
            phi_gradient,
            Ve_gradient,
            phi_conservative_stencil,
            Ve_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=perpendicular_operator_boundary.Ve,
            f_field_halo=state_halo.phi, g_field_halo=Ve_perp_halo,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("Ve"),
        )
        poisson_vorticity = self._poisson_bracket_over_B(
            phi_gradient,
            vorticity_gradient,
            phi_conservative_stencil,
            vorticity_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.vorticity,
            f_field_halo=state_halo.phi, g_field_halo=state_halo.vorticity,
            f_field_closure=primitive_cv_closures.get("phi"),
            g_field_closure=primitive_cv_closures.get("vorticity"),
        )

        if self.parallel_operator_scheme == "coordinate":
            stage_parallel_terms = self._coordinate_stage_parallel_terms(
                state_halo=state_halo,
                context=context,
                operator_boundary=operator_boundary,
                parallel_boundary=parallel_boundary,
                parallel_div_b=parallel_div_b,
                density_flux_stencil=density_flux_conservative_stencil,
                current_stencil=current_conservative_stencil,
                Ve_stencil=Ve_conservative_stencil,
                Vi_stencil=Vi_conservative_stencil,
                Te_stencil=Te_conservative_stencil,
                Ti_stencil=Ti_conservative_stencil,
                phi_stencil=phi_conservative_stencil,
                Pe_stencil=Pe_conservative_stencil,
                pressure_stencil=pressure_conservative_stencil,
                vorticity_stencil=vorticity_conservative_stencil,
                primitive_cv_closures=primitive_cv_closures,
            )
        else:
            stage_parallel_terms = fci_parallel_terms
        parallel_density_flux_divergence = stage_parallel_terms["density_flux_div"]
        parallel_current_flux_divergence = stage_parallel_terms["current_flux_div"]
        vorticity_current_flux_divergence = stage_parallel_terms[
            "vorticity_current_flux_div"
        ]
        parallel_Ve_flux_divergence = stage_parallel_terms["Ve_flux_div"]
        parallel_Vi_flux_divergence = stage_parallel_terms["parallel_Vi_flux_div"]
        grad_parallel_Te = stage_parallel_terms["grad_Te"]
        grad_parallel_Ti = stage_parallel_terms["grad_Ti"]
        grad_parallel_Ve = stage_parallel_terms["grad_Ve"]
        grad_parallel_Vi = stage_parallel_terms["grad_Vi"]
        grad_parallel_phi = stage_parallel_terms["grad_phi"]
        grad_parallel_Pe = stage_parallel_terms["grad_Pe"]
        grad_parallel_pressure = stage_parallel_terms["grad_pressure"]
        grad_parallel_current = stage_parallel_terms["grad_current"]
        grad_parallel_vorticity = stage_parallel_terms["grad_vorticity"]
        material_upwind_correction = stage_parallel_terms.get(
            "material_upwind_correction",
            jnp.zeros(self.geometry.owned_shape + (5,), dtype=jnp.float64),
        )
        production_material_residual = stage_parallel_terms.get(
            "parallel_material_residual",
            jnp.zeros(self.geometry.owned_shape + (5,), dtype=jnp.float64),
        )
        production_parallel = self.parallel_material_scheme == "production-path"
        if production_parallel:
            # The coupled production residual already contains all five
            # material equations, including the geometric div(b) terms.  The
            # legacy correction is deliberately disabled in this mode so the
            # same interface action cannot be counted twice.
            material_upwind_correction = jnp.zeros_like(
                production_material_residual
            )
        if self.parallel_velocity_layout == "fci-staggered":
            n_face_safe = jnp.maximum(stage_parallel_terms["staggered_n_face"], 1.0e-30)
            Vi_parallel_value = stage_parallel_terms["staggered_Vi_face"]
            Ve_parallel_value = stage_parallel_terms["staggered_Ve_face"]
            current_parallel_value = stage_parallel_terms["staggered_j_face"]
            Te_parallel_advection = -stage_parallel_terms["staggered_Te_advection"]
            Ti_parallel_advection = -stage_parallel_terms["staggered_Ti_advection"]
            vorticity_parallel_advection = -stage_parallel_terms["staggered_vorticity_advection"]
        else:
            n_face_safe = density_safe
            Vi_parallel_value, Ve_parallel_value = Vi, Ve
            current_parallel_value = density * (Vi - Ve)
            Te_parallel_advection = -Ve * grad_parallel_Te
            Ti_parallel_advection = -Vi * grad_parallel_Ti
            vorticity_parallel_advection = -Vi * grad_parallel_vorticity
        Vi_self_advection_term = -Vi_parallel_value * grad_parallel_Vi
        Vi_pressure_term = -grad_parallel_pressure / n_face_safe
        Ve_self_advection_term = -Ve_parallel_value * grad_parallel_Ve
        Ve_collision_term = mi_over_me * Ve_nu * current_parallel_value
        Ve_electrostatic_term = (
            mi_over_me * (grad_parallel_phi + tau * grad_parallel_Ti)
            if production_parallel
            else mi_over_me * grad_parallel_phi
        )
        Ve_pressure_term = -mi_over_me * grad_parallel_Pe / n_face_safe
        Ve_thermal_force_term = -0.71 * mi_over_me * grad_parallel_Te
        curvature_outputs = self._curvature_rhs_contributions(
            state_halo=state_halo,
            face_bc=face_bc,
            context=context,
            density=density,
            Te=Te,
            Ti=Ti,
            bmag=bmag,
            density_safe=density_safe,
            tau=tau,
            Pe_face_bc=Pe_face_bc,
            pressure_face_bc=pressure_face_bc,
            operator_boundary=operator_boundary,
            Pe_gradient=Pe_gradient,
            pressure_gradient=pressure_gradient,
            phi_gradient=phi_gradient,
            Te_gradient=Te_gradient,
            Ti_gradient=Ti_gradient,
            density_conservative_stencil=density_conservative_stencil,
            Pe_conservative_stencil=Pe_conservative_stencil,
            pressure_conservative_stencil=pressure_conservative_stencil,
            phi_conservative_stencil=phi_conservative_stencil,
            Te_conservative_stencil=Te_conservative_stencil,
            Ti_conservative_stencil=Ti_conservative_stencil,
            vorticity_conservative_stencil=vorticity_conservative_stencil,
            primitive_cv_closures=primitive_cv_closures,
            return_directional_components=return_curvature_component_fields,
        )
        if return_curvature_component_fields:
            curvature_component_fields = jnp.stack(curvature_outputs, axis=0)
            (
                curvature_density_contribution,
                curvature_Te_contribution,
                curvature_Ti_contribution,
                curvature_vorticity_contribution,
            ) = tuple(jnp.sum(value, axis=0) for value in curvature_outputs)
            if self._uses_projected_fine_grid:
                curvature_component_fields = jax.vmap(
                    jax.vmap(self._restrict_fine_field)
                )(curvature_component_fields)
        else:
            (
                curvature_density_contribution,
                curvature_Te_contribution,
                curvature_Ti_contribution,
                curvature_vorticity_contribution,
            ) = curvature_outputs
            curvature_component_fields = None

        density_rhs = (
            -(poisson_density / rho_star)
            + (
                production_material_residual[..., 0]
                if production_parallel
                else -parallel_density_flux_divergence
            )
            + curvature_density_contribution
            + density_diff
            + density_parallel_diff
            + material_upwind_correction[..., 0]
        )
        Te_rhs = (
            -(poisson_Te / rho_star)
            + (
                production_material_residual[..., 1]
                if production_parallel
                else Te_parallel_advection
            )
            + curvature_Te_contribution
            + (
                0.0
                if production_parallel
                else (2.0 * Te / (3.0 * density_safe))
                * (0.71 * parallel_current_flux_divergence - density * parallel_Ve_flux_divergence)
            )
            + Te_diff
            + Te_parallel_diff
            + material_upwind_correction[..., 1]
        )
        Ti_rhs = (
            -(poisson_Ti / rho_star)
            + (
                production_material_residual[..., 2]
                if production_parallel
                else Ti_parallel_advection
            )
            + curvature_Ti_contribution
            + (
                0.0
                if production_parallel
                else (2.0 * Ti / (3.0 * density_safe))
                * (parallel_current_flux_divergence - density * parallel_Vi_flux_divergence)
            )
            + Ti_diff
            + Ti_parallel_diff
            + material_upwind_correction[..., 2]
        )
        Vi_perpendicular_rhs = (
            -(poisson_Vi / rho_star)
            + Vi_diff
        )
        Ve_poisson_term = -(poisson_Ve / rho_star)
        Ve_perpendicular_rhs = Ve_poisson_term + Ve_diff
        if self.parallel_velocity_layout == "fci-staggered":
            # Only the centered perpendicular terms cross back to face
            # storage.  Batch each species' diagnostic lanes so the shared
            # homogeneous f2c transpose is traced once; total is their lane
            # sum.  Parallel terms below already live on source edges.
            Vi_poisson_term, Vi_diff_term = self._cell_force_lanes_to_outgoing_face_mass_adjoint(
                jnp.stack((-(poisson_Vi / rho_star), Vi_diff)), face_bc.Vi, context
            )
            Ve_poisson_term, Ve_diff_term = self._cell_force_lanes_to_outgoing_face_mass_adjoint(
                jnp.stack((Ve_poisson_term, Ve_diff)), face_bc.Ve, context
            )
            Vi_perpendicular_rhs = Vi_poisson_term + Vi_diff_term
            Ve_perpendicular_rhs = Ve_poisson_term + Ve_diff_term
        else:
            Vi_poisson_term = -(poisson_Vi / rho_star)
            Vi_diff_term = Vi_diff
            Ve_diff_term = Ve_diff
        Vi_rhs = (
            Vi_perpendicular_rhs
            + (
                production_material_residual[..., 3]
                if production_parallel
                else Vi_self_advection_term + Vi_pressure_term
            )
            + Vi_parallel_diff
            + material_upwind_correction[..., 3]
        )
        Ve_characteristic_upwind_term = material_upwind_correction[..., 4]
        Ve_rhs = (
            Ve_perpendicular_rhs
            + (
                production_material_residual[..., 4]
                if production_parallel
                else Ve_self_advection_term
            )
            + Ve_collision_term
            + Ve_electrostatic_term
            + (0.0 if production_parallel else Ve_pressure_term)
            + (0.0 if production_parallel else Ve_thermal_force_term)
            + Ve_parallel_diff
            + (0.0 if production_parallel else Ve_characteristic_upwind_term)
        )
        vorticity_rhs = (
            -(poisson_vorticity / rho_star)
            + vorticity_parallel_advection
            + (bmag * bmag / density_safe) * vorticity_current_flux_divergence
            + curvature_vorticity_contribution
            + vorticity_diff
            + vorticity_parallel_diff
        )

        zero_term = jnp.zeros_like(density)
        density_parallel_material_term = (
            production_material_residual[..., 0]
            if production_parallel else -parallel_density_flux_divergence
        )
        Te_parallel_material_term = (
            production_material_residual[..., 1]
            if production_parallel else (
                Te_parallel_advection
                + (2.0 * Te / (3.0 * density_safe))
                * (0.71 * parallel_current_flux_divergence
                   - density * parallel_Ve_flux_divergence)
            )
        )
        Ti_parallel_material_term = (
            production_material_residual[..., 2]
            if production_parallel else (
                Ti_parallel_advection
                + (2.0 * Ti / (3.0 * density_safe))
                * (parallel_current_flux_divergence
                   - density * parallel_Vi_flux_divergence)
            )
        )
        Vi_parallel_material_term = (
            production_material_residual[..., 3]
            if production_parallel else Vi_self_advection_term + Vi_pressure_term
        )
        Ve_parallel_material_term = (
            production_material_residual[..., 4]
            if production_parallel else Ve_self_advection_term
        )

        def pack_rhs_terms(
            fine_terms: tuple[jnp.ndarray, ...],
            source_term: jnp.ndarray,
            *,
            face_owned: bool = False,
        ) -> jnp.ndarray:
            if face_owned:
                owner_terms = [
                    self._owner_face_field(self._restrict_fine_face_field(term))
                    for term in fine_terms
                ]
                owner_terms.append(self._owner_face_field(source_term))
            else:
                owner_terms = [
                    self._owner_field(
                        _mask_inactive_owned(
                            self._restrict_fine_field(term), self.geometry
                        )
                    )
                    for term in fine_terms
                ]
                owner_terms.append(self._owner_field(source_term))
            owner_zero = jnp.zeros_like(owner_terms[0])
            owner_terms.extend(
                owner_zero
                for _ in range(RHS_TERM_SLOT_COUNT - len(owner_terms))
            )
            return jnp.stack(tuple(owner_terms), axis=0)

        def all_rhs_term_fields(*, parallel_only: bool) -> jnp.ndarray:
            nonparallel = (lambda term: zero_term if parallel_only else term)
            density_terms = pack_rhs_terms(
                (
                    nonparallel(-(poisson_density / rho_star)),
                    density_parallel_material_term,
                    nonparallel(curvature_density_contribution),
                    nonparallel(density_diff),
                    density_parallel_diff,
                    zero_term,
                    material_upwind_correction[..., 0],
                ),
                source_owned.density,
            )
            Te_terms = pack_rhs_terms(
                (
                    nonparallel(-(poisson_Te / rho_star)),
                    Te_parallel_material_term if production_parallel else Te_parallel_advection,
                    nonparallel(curvature_Te_contribution),
                    zero_term if production_parallel else (
                        (2.0 * Te / (3.0 * density_safe))
                        * (0.71 * parallel_current_flux_divergence
                           - density * parallel_Ve_flux_divergence)
                    ),
                    nonparallel(Te_diff),
                    Te_parallel_diff,
                    zero_term,
                    material_upwind_correction[..., 1],
                ),
                source_owned.Te,
            )
            Ti_terms = pack_rhs_terms(
                (
                    nonparallel(-(poisson_Ti / rho_star)),
                    Ti_parallel_material_term if production_parallel else Ti_parallel_advection,
                    nonparallel(curvature_Ti_contribution),
                    zero_term if production_parallel else (
                        (2.0 * Ti / (3.0 * density_safe))
                        * (parallel_current_flux_divergence
                           - density * parallel_Vi_flux_divergence)
                    ),
                    nonparallel(Ti_diff),
                    Ti_parallel_diff,
                    zero_term,
                    material_upwind_correction[..., 2],
                ),
                source_owned.Ti,
            )
            Vi_terms = pack_rhs_terms(
                (
                    nonparallel(Vi_poisson_term),
                    Vi_parallel_material_term if production_parallel else Vi_self_advection_term,
                    zero_term if production_parallel else Vi_pressure_term,
                    nonparallel(Vi_diff_term),
                    Vi_parallel_diff,
                    material_upwind_correction[..., 3],
                ),
                source_owned.Vi,
                face_owned=self.parallel_velocity_layout == "fci-staggered",
            )
            Ve_terms = pack_rhs_terms(
                (
                    nonparallel(Ve_poisson_term),
                    Ve_parallel_material_term if production_parallel else Ve_self_advection_term,
                    Ve_collision_term,
                    Ve_electrostatic_term,
                    zero_term if production_parallel else Ve_pressure_term,
                    zero_term if production_parallel else Ve_thermal_force_term,
                    nonparallel(Ve_diff_term),
                    Ve_parallel_diff,
                    zero_term if production_parallel else Ve_characteristic_upwind_term,
                ),
                source_owned.Ve,
                face_owned=self.parallel_velocity_layout == "fci-staggered",
            )
            vorticity_terms = pack_rhs_terms(
                (
                    nonparallel(-(poisson_vorticity / rho_star)),
                    vorticity_parallel_advection,
                    (bmag * bmag / density_safe)
                    * vorticity_current_flux_divergence,
                    nonparallel(curvature_vorticity_contribution),
                    nonparallel(vorticity_diff),
                    vorticity_parallel_diff,
                    zero_term,
                ),
                source_owned.vorticity,
            )
            return jnp.stack(
                (
                    density_terms,
                    Te_terms,
                    Ti_terms,
                    Vi_terms,
                    Ve_terms,
                    vorticity_terms,
                ),
                axis=0,
            )

        if self.parallel_subsystem_only:
            # The local_parallel_flux_div_op and
            # local_grad_parallel_op_conservative calls above use the
            # operator_boundary.* physical-wall traces and existing axis
            # machinery.  This branch only selects their already-computed
            # results; it does not construct diagnostic operators.  The
            # production calls use boundary_trace=operator_boundary.*.
            # production set includes grad_parallel_Te, grad_parallel_Ti,
            # grad_parallel_Ve, grad_parallel_Vi, grad_parallel_phi,
            # grad_parallel_Pe, grad_parallel_pressure, and
            # grad_parallel_current, even where a row below does not need
            # every one of those quantities.
            density_rhs = (
                (
                    production_material_residual[..., 0]
                    if production_parallel
                    else -parallel_density_flux_divergence
                )
                + density_parallel_diff
                + material_upwind_correction[..., 0]
            )
            Te_rhs = (
                (
                    production_material_residual[..., 1]
                    if production_parallel
                    else Te_parallel_advection
                    + (2.0 * Te / (3.0 * density_safe))
                    * (
                        0.71 * parallel_current_flux_divergence
                        - density * parallel_Ve_flux_divergence
                    )
                )
                + Te_parallel_diff
                + material_upwind_correction[..., 1]
            )
            Ti_rhs = (
                (
                    production_material_residual[..., 2]
                    if production_parallel
                    else Ti_parallel_advection
                    + (2.0 * Ti / (3.0 * density_safe))
                    * (
                        parallel_current_flux_divergence
                        - density * parallel_Vi_flux_divergence
                    )
                )
                + Ti_parallel_diff
                + material_upwind_correction[..., 2]
            )
            Vi_rhs = (
                (
                    production_material_residual[..., 3]
                    if production_parallel
                    else Vi_self_advection_term + Vi_pressure_term
                )
                + Vi_parallel_diff
                + material_upwind_correction[..., 3]
            )
            Ve_rhs = (
                (
                    production_material_residual[..., 4]
                    if production_parallel
                    else Ve_self_advection_term
                )
                + Ve_collision_term
                + Ve_electrostatic_term
                + (0.0 if production_parallel else Ve_pressure_term)
                + (0.0 if production_parallel else Ve_thermal_force_term)
                + Ve_parallel_diff
                + (0.0 if production_parallel else Ve_characteristic_upwind_term)
            )
            vorticity_rhs = (
                vorticity_parallel_advection
                + (bmag * bmag / density_safe)
                * vorticity_current_flux_divergence
                + vorticity_parallel_diff
            )
            result = self._restrict_fine_state(FciDrbEBState(
                density=density_rhs,
                phi=jnp.zeros_like(phi_owned),
                Te=Te_rhs,
                Ti=Ti_rhs,
                Vi=Vi_rhs,
                Ve=Ve_rhs,
                vorticity=vorticity_rhs,
            ))
            result = self._owner_state(_mask_local_eb_state_inactive(
                result, self.geometry
            ))
            if return_rhs_term_fields:
                rhs_terms = all_rhs_term_fields(parallel_only=True)
                if return_curvature_component_fields:
                    return result, rhs_terms, curvature_component_fields
                return result, rhs_terms
            if return_curvature_component_fields:
                return result, curvature_component_fields
            if return_term_diagnostics or return_term_fields:
                Ve_terms = jnp.stack(
                    (
                        jnp.zeros_like(Ve),
                        (
                            production_material_residual[..., 4]
                            if production_parallel else Ve_self_advection_term
                        ),
                        Ve_collision_term,
                        Ve_electrostatic_term,
                        (
                            jnp.zeros_like(Ve)
                            if production_parallel else Ve_pressure_term
                        ),
                        (
                            jnp.zeros_like(Ve)
                            if production_parallel else Ve_thermal_force_term
                        ),
                        jnp.zeros_like(Ve),
                        jnp.zeros_like(Ve),
                        Ve_parallel_diff,
                        (
                            jnp.zeros_like(Ve)
                            if production_parallel
                            else Ve_characteristic_upwind_term
                        ),
                    ),
                    axis=0,
                )
                if self.parallel_velocity_layout == "fci-staggered":
                    Ve_terms = jnp.stack(
                        tuple(self._restrict_fine_face_field(term) for term in Ve_terms),
                        axis=0,
                    )
                elif self._uses_projected_fine_grid:
                    Ve_terms = jnp.stack(
                        tuple(self._restrict_fine_field(term) for term in Ve_terms),
                        axis=0,
                    )
                if return_term_fields:
                    return result, Ve_terms
                return result, jnp.max(
                    jnp.abs(Ve_terms),
                    axis=tuple(range(1, Ve_terms.ndim)),
                )
            return result
        assembled = self._restrict_fine_state(FciDrbEBState(
            density=density_rhs,
            phi=jnp.zeros_like(phi_owned),
            Te=Te_rhs,
            Ti=Ti_rhs,
            Vi=Vi_rhs,
            Ve=Ve_rhs,
            vorticity=vorticity_rhs,
        ))
        # Sources are owner-space data.  Add them after RLP so their
        # amplitudes are not volume-diluted by fine storage aliases.
        result = self._owner_state(_mask_local_eb_state_inactive(
            assembled.replace(
                density=assembled.density + source_owned.density,
                Te=assembled.Te + source_owned.Te,
                Ti=assembled.Ti + source_owned.Ti,
                Vi=assembled.Vi + source_owned.Vi,
                Ve=assembled.Ve + source_owned.Ve,
                vorticity=assembled.vorticity + source_owned.vorticity,
            ),
            self.geometry,
        ))
        if return_rhs_term_fields:
            rhs_terms = all_rhs_term_fields(parallel_only=False)
            if return_curvature_component_fields:
                return result, rhs_terms, curvature_component_fields
            return result, rhs_terms
        if return_curvature_component_fields:
            return result, curvature_component_fields
        if return_term_diagnostics or return_term_fields:
            Ve_terms = jnp.stack(
                (
                    Ve_poisson_term,
                    (
                        production_material_residual[..., 4]
                        if production_parallel else Ve_self_advection_term
                    ),
                    Ve_collision_term,
                    Ve_electrostatic_term,
                    (
                        jnp.zeros_like(Ve)
                        if production_parallel else Ve_pressure_term
                    ),
                    (
                        jnp.zeros_like(Ve)
                        if production_parallel else Ve_thermal_force_term
                    ),
                    Ve_diff_term,
                    Ve_parallel_diff,
                    (
                        jnp.zeros_like(Ve)
                        if production_parallel
                        else Ve_characteristic_upwind_term
                    ),
                ),
                axis=0,
            )
            if self.parallel_velocity_layout == "fci-staggered":
                Ve_terms = jnp.stack(
                    tuple(self._restrict_fine_face_field(term) for term in Ve_terms),
                    axis=0,
                )
            elif self._uses_projected_fine_grid:
                Ve_terms = jnp.stack(
                    tuple(self._restrict_fine_field(term) for term in Ve_terms),
                    axis=0,
                )
            if return_term_fields:
                return result, Ve_terms
            return result, jnp.max(
                jnp.abs(Ve_terms),
                axis=tuple(range(1, Ve_terms.ndim)),
            )
        return result
