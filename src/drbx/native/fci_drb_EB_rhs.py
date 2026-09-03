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
    BC_NEUMANN,
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
    local_perp_laplacian_conservative_op,
    local_curvature_conservative_op,
    local_curvature_conservative_components_op,
    local_curvature_production_path_op,
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op_from_gradients,
    expand_local_control_volume_owner_field,
    aggregate_local_control_volume_average,
    _mask_inactive_owned,
    _mask_state_inactive_owned,
)
from .fci_gmres import SolvaxGmresConfig, SolvaxGmresInfo
from .fci_support_pair import build_weighted_negative_adjoint
from .fci_parallel_production_flux import (
    parallel_characteristic_wall_data,
    parallel_short_wall_backward_euler,
    parallel_target_row_material_residual,
    parallel_vorticity_upwind_residual,
)
from .fci_physical_wall import resolve_fci_material_wall_endpoint_state


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

CURVATURE_COMPONENT_DIAGNOSTIC_NAMES = ("u", "theta", "eta")


def curvature_component_diagnostic_names() -> tuple[str, ...]:
    """Return the production directional curvature lane names."""

    return CURVATURE_COMPONENT_DIAGNOSTIC_NAMES


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
    # Characteristic physical-wall law for the production parallel material
    # block.  The default retains the established primitive least-residual
    # projection exactly; energy-absorbing closes incoming modes against the
    # explicit equilibrium/reference state.  Both are legacy compatibility
    # paths. physical-boundary-state passes the complete wall trace to the
    # live characteristic flux without assuming an incoming rank. The wall
    # model itself is owned by the face-BC bundle, not by this RHS.
    parallel_characteristic_wall_law: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW",
            "primitive-least-residual",
        )
    )

    def __post_init__(self):
        if self.parallel_characteristic_wall_law not in (
            "primitive-least-residual", "energy-absorbing",
            "physical-boundary-state",
        ):
            raise ValueError(
                "parallel_characteristic_wall_law must be "
                "'primitive-least-residual', 'energy-absorbing', "
                "or 'physical-boundary-state', got "
                f"{self.parallel_characteristic_wall_law!r}"
            )

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
            self.parallel_characteristic_wall_law,
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
            parallel_characteristic_wall_law=_aux_data,
        )


@dataclass(frozen=True)
class LocalFciDrbEBPhysicalWallBundle:
    """Physical wall data for every primitive EB boundary consumer.

    The fields describe the physical trace/derivative law. Operator-specific
    traces and exterior representations are derived from this single bundle.
    """

    density: LocalBoundaryFaceBC3D
    phi: LocalBoundaryFaceBC3D
    Te: LocalBoundaryFaceBC3D
    Ti: LocalBoundaryFaceBC3D
    Vi: LocalBoundaryFaceBC3D
    Ve: LocalBoundaryFaceBC3D
    vorticity: LocalBoundaryFaceBC3D


# Compatibility name for existing model builders. New wall models should use
# the physical-wall name so the data ownership is unambiguous.
LocalFciDrbEBFaceBCBundle = LocalFciDrbEBPhysicalWallBundle


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalFciDrbEBOperatorBoundaryBundle:
    """Operator-level numerical traces for the seven-field EB model.

    ``LocalFciDrbEBPhysicalWallBundle`` contains the model's primitive physical
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
    curvature_face_coefficients: LocalCurvatureFaceCoefficients3D
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SolvaxGmresConfig
    face_bc_builder: LocalFciDrbEBFaceBCBuilder
    physical_wall_model_name: str = "legacy-velocity-trace"
    conducting_sheath_wall_potential: float | None = None
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None
    control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D | None = None
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    # Complete five-field parallel material flux.  ``legacy`` preserves the
    # existing mapped operators; ``production-path`` uses one canonical-face
    # characteristic fluctuation on every ordinary and wall-ending FCI row.
    parallel_material_scheme: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"
        )
    )
    # Selectable discretization for E x B advection.  The compatible paths
    # return the already-B-divided bracket and use shared conservative face
    # data. ``compatible-third-order-upwind`` keeps the same compatible skew
    # core for every equation and replaces its physical A_phi(q) channel by
    # the complete characteristic action A_phi^upwind(q).  It retains
    # D(Uq)-qD(U), the production third-order bulk stencil, and first-order
    # wall/RLP fallbacks without a tunable penalty. ``direct`` preserves the
    # reconstructed cell-gradient path.
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
    # ``cfl`` preserves the thresholded short-leg choice;
    # ``all-physical-walls`` selects every physical wall leg for the local BE
    # path under either supported characteristic wall law.
    parallel_short_leg_selection: str = field(
        default_factory=lambda: os.environ.get(
            "DRBX_PARALLEL_SHORT_LEG_SELECTION", "cfl"
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
        if self.physical_wall_model_name not in (
            "legacy-velocity-trace",
            "no-flow",
            "simple-conducting-sheath",
        ):
            raise ValueError(
                "physical_wall_model_name must identify a supported physical "
                f"wall model, got {self.physical_wall_model_name!r}"
            )
        if self.poisson_bracket_scheme not in (
            "direct",
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
        ):
            raise ValueError(
                "poisson_bracket_scheme must be 'direct', 'compatible-flux', "
                "'compatible-third-order-upwind', or "
                "'material-scalar-third-order-upwind', got "
                f"{self.poisson_bracket_scheme!r}"
            )
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
            if self.poisson_bracket_scheme not in (
                "compatible-flux",
                "compatible-third-order-upwind",
                "material-scalar-third-order-upwind",
            ):
                raise ValueError(
                    "projected-owner RLP requires "
                    "a compatible Poisson-bracket scheme"
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
        wall_law = self.parameters.parallel_characteristic_wall_law
        if wall_law not in (
            "primitive-least-residual", "energy-absorbing",
            "physical-boundary-state",
        ):
            raise ValueError(
                "parallel_characteristic_wall_law must be "
                "'primitive-least-residual', 'energy-absorbing', "
                "or 'physical-boundary-state', got "
                f"{wall_law!r}"
            )
        if wall_law == "energy-absorbing" and self.parallel_material_scheme != "production-path":
            raise ValueError(
                "parallel_characteristic_wall_law='energy-absorbing' requires "
                "parallel_material_scheme='production-path'"
            )
        if wall_law == "energy-absorbing" and self.parallel_boundary_pairing != "characteristic-sat":
            raise ValueError(
                "parallel_characteristic_wall_law='energy-absorbing' requires "
                "parallel_boundary_pairing='characteristic-sat'"
            )
        if wall_law == "physical-boundary-state" and self.parallel_material_scheme != "production-path":
            raise ValueError(
                "parallel_characteristic_wall_law='physical-boundary-state' requires "
                "parallel_material_scheme='production-path'"
            )
        if wall_law == "physical-boundary-state" and self.parallel_boundary_pairing != "characteristic-sat":
            raise ValueError(
                "parallel_characteristic_wall_law='physical-boundary-state' requires "
                "parallel_boundary_pairing='characteristic-sat'"
            )
        if (
            self.parallel_boundary_pairing == "characteristic-sat"
            and (
                self.parallel_material_scheme != "production-path"
                or self.parallel_operator_scheme != "fci"
            )
        ):
            raise ValueError(
                "parallel_boundary_pairing='characteristic-sat' requires "
                "the production FCI path"
            )
        if (
            self.parallel_flux_pairing == "support-core"
            and self.parallel_operator_scheme != "fci"
        ):
            raise ValueError(
                "parallel_flux_pairing='support-core' requires "
                "parallel_operator_scheme='fci'"
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
        if self.parallel_short_leg_selection not in ("cfl", "all-physical-walls"):
            raise ValueError(
                "parallel_short_leg_selection must be 'cfl' or "
                "'all-physical-walls', got "
                f"{self.parallel_short_leg_selection!r}"
            )
        if self.parallel_short_leg_selection == "all-physical-walls":
            if self.parallel_short_leg_treatment != "local-backward-euler":
                raise ValueError(
                    "parallel_short_leg_selection='all-physical-walls' requires "
                    "parallel_short_leg_treatment='local-backward-euler'"
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
        if self.curvature_face_coefficients is None:
            raise ValueError(
                "curvature_face_coefficients are required for the production "
                "curvature operator"
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
        g_field_halo: jnp.ndarray | None = None,
        g_positivity_floor: float | None = None,
        equation_family: str = "material",
    ) -> jnp.ndarray:
        """Evaluate the selected Poisson bracket with the RHS ``1/B`` included.

        The compatible-flux discretization uses the operator-level physical-wall
        traces.  The direct discretization continues to use gradients built from
        the already closed field halos.
        """

        if self.poisson_bracket_scheme in (
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
        ):
            return local_poisson_bracket_compatible_flux_op(
                f_conservative_stencil,
                g_conservative_stencil,
                self.geometry,
                domain=self.domain,
                axis_regular_axes=self.axis_regular_axes,
                f_boundary_trace=f_boundary_trace,
                g_boundary_trace=g_boundary_trace,
                characteristic_scheme=(
                    (
                        "centered"
                        if equation_family == "vorticity"
                        else "scalar-third-order-upwind"
                    )
                    if self.poisson_bracket_scheme
                    == "material-scalar-third-order-upwind"
                    else "third-order-upwind"
                    if self.poisson_bracket_scheme
                    == "compatible-third-order-upwind"
                    else "centered"
                ),
                g_field_halo=g_field_halo,
                g_positivity_floor=g_positivity_floor,
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

    def _prepare_fine_storage_halo(
        self,
        values_fine_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        """Close a fine-grid derived field without RLP owner projection."""
        field_halo = inject_owned_field_to_halo(values_fine_owned, self.domain.layout)
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(field_halo, self.domain, face_bc)

    def _prepare_state_halo(self, state_owned, face_bc):
        state_owned = self._owner_state(state_owned)
        if self.control_volume_geometry is None:
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
            Vi=self._prepare_scalar_halo(state_owned.Vi, face_bc.Vi),
            Ve=self._prepare_scalar_halo(state_owned.Ve, face_bc.Ve),
            vorticity=self._prepare_scalar_halo(state_owned.vorticity, face_bc.vorticity),
        )

    def _owner_field(self, values: jnp.ndarray) -> jnp.ndarray:
        """Return a cell-owner field; merged cell aliases are never evolved."""
        values = jnp.asarray(values, dtype=jnp.float64)
        if self.control_volume_geometry is None:
            return values
        cells = self.control_volume_geometry.cells
        return jnp.where(cells.is_active_owner, values, 0.0)

    def _owner_state(self, state: FciDrbEBState) -> FciDrbEBState:
        if self.control_volume_geometry is None:
            return state
        return state.replace(
            density=self._owner_field(state.density),
            phi=self._owner_field(state.phi),
            Te=self._owner_field(state.Te),
            Ti=self._owner_field(state.Ti),
            Vi=self._owner_field(state.Vi),
            Ve=self._owner_field(state.Ve),
            vorticity=self._owner_field(state.vorticity),
        )

    def _owner_result(self, result: jnp.ndarray) -> jnp.ndarray:
        return self._owner_field(result)

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

    def _restrict_fine_state(self, state: FciDrbEBState) -> FciDrbEBState:
        """Restrict every assembled fine-grid RHS leaf to owner space."""

        if not self._uses_projected_fine_grid:
            return state
        return state.replace(
            density=self._restrict_fine_field(state.density),
            phi=self._restrict_fine_field(state.phi),
            Te=self._restrict_fine_field(state.Te),
            Ti=self._restrict_fine_field(state.Ti),
            Vi=self._restrict_fine_field(state.Vi),
            Ve=self._restrict_fine_field(state.Ve),
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
        *,
        boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    ) -> jnp.ndarray:
        """Evaluate the centered scalar remainder of production curvature."""

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
            boundary_trace=boundary_trace,
            axis_regular_axes=self.axis_regular_axes,
        )

    def _conservative_curvature_components(
        self,
        conservative_stencil: ConservativeStencil3D,
        *,
        boundary_trace: LocalBoundaryFaceTrace3D | None = None,
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
            boundary_trace=boundary_trace,
            axis_regular_axes=self.axis_regular_axes,
        )

    def _curvature_rhs_contributions(
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
        request_split_diagnostics = return_directional_components
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
            wall_flux_closure="bc-characteristic-operator-trace-canonical-face-state",
            # Directional lanes are materialized only when explicitly requested
            # by the diagnostics API.
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
        result = material + remainder
        if not request_split_diagnostics:
            return tuple(jnp.moveaxis(result, -1, 0))
        assert diagnostics is not None
        psi_directional = self._conservative_curvature_components(
            psi_stencil,
            boundary_trace=psi_trace,
        )
        remainder_directional = (
            remainder_coeff[..., None, :] * jnp.moveaxis(psi_directional, 0, -1)[..., None]
        )
        # psi_directional is (3, nx, ny, nz); move the component axis to the
        # final position before adding the four-field material diagnostics.
        remainder_directional = jnp.moveaxis(remainder_directional, -2, 0)
        # The production diagnostics expose only the directional residual;
        # centered/dissipation and radial-provenance layouts were audit-only.
        directional = diagnostics["directional_residual"] + remainder_directional
        if return_directional_components:
            return tuple(jnp.moveaxis(directional, -1, 0))
        return tuple(jnp.moveaxis(jnp.sum(directional, axis=0), -1, 0))

    def ion_temperature_curvature_chain_rule_diagnostics(
        self,
        state_owned: FciDrbEBState,
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

        face_bc = self._face_bcs(state_owned)
        state_halo = FciDrbEBState(
            density=self._prepare_scalar_halo(state_owned.density, face_bc.density),
            phi=self._prepare_scalar_halo(state_owned.phi, face_bc.phi),
            Te=self._prepare_scalar_halo(state_owned.Te, face_bc.Te),
            Ti=self._prepare_scalar_halo(state_owned.Ti, face_bc.Ti),
            Vi=self._prepare_scalar_halo(state_owned.Vi, face_bc.Vi),
            Ve=self._prepare_scalar_halo(state_owned.Ve, face_bc.Ve),
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
        ti_curvature = self._conservative_curvature(
            ti_stencil,
            boundary_trace=operator_boundary.Ti,
        )
        ti_squared_curvature = self._conservative_curvature(
            ti_squared_stencil,
            boundary_trace=operator_boundary.Ti_squared,
        )
        owned = self.domain.layout.owned_slices_cell
        ti = jnp.asarray(ti_halo[owned], dtype=jnp.float64)
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
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
        )

    def _field_parallel_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
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

    def _fci_plasma_side_stencil(
        self,
        q_owned: jnp.ndarray,
        template_bc: LocalBoundaryFaceBC3D,
        context: StencilBuilderContext,
    ):
        """Return an FCI stencil whose physical endpoint is a plasma trace.

        Nonlinear wall models must receive primitive plasma-side values at the
        traced endpoint.  They cannot recover those values from a ghost halo
        that already contains a sign-selected sheath target.  This helper
        therefore applies homogeneous Neumann closure first and leaves all
        nonlinear wall evaluation to the endpoint resolver.
        """

        neumann_bc = replace(
            template_bc,
            kind_x=jnp.where(template_bc.mask_x, BC_NEUMANN, template_bc.kind_x),
            kind_y=jnp.where(template_bc.mask_y, BC_NEUMANN, template_bc.kind_y),
            kind_z=jnp.where(template_bc.mask_z, BC_NEUMANN, template_bc.kind_z),
            value_x=jnp.zeros_like(template_bc.value_x),
            value_y=jnp.zeros_like(template_bc.value_y),
            value_z=jnp.zeros_like(template_bc.value_z),
        )
        field_halo = self._prepare_scalar_halo(q_owned, neumann_bc)
        forward_remote, backward_remote = self._fci_remote_values(
            field_halo, context
        )
        return build_local_fci_stencil_from_field(
            field_halo,
            self.geometry,
            context,
            forward_remote_values=forward_remote,
            backward_remote_values=backward_remote,
        )

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
        # The vorticity current uses the same operator trace as the potential
        # pair.  The parallel-characteristic trace was an abandoned closure
        # ablation and is not part of the production contract.
        vorticity_current_boundary = operator_boundary.current
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
        characteristic_sat_effective_nonlinear_current_divergence = None
        characteristic_sat_effective_linearized_current_divergence = None
        support_gradient_values: dict[str, jnp.ndarray] = {}
        support_flux_values: dict[str, jnp.ndarray] = {}
        if self.parallel_flux_pairing == "support-core":
            support_gradient, support_divergence, support_core_target = (
                self._fci_support_core_pair(
                    face_bc=face_bc,
                    context=context,
                )
            )
            # Support-core production always uses the homogeneous current/phi
            # weighted-adjoint pair.  Characteristic-SAT changes only the
            # endpoint current lift below; it does not replace this pair.
            use_current_phi_boundary_pair = True
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
            return value

        div_b = local_parallel_div_b_fci_from_q_op(
            inverse_b_halo,
            self.geometry,
            context=context,
            forward_remote_q_values=inverse_b_forward,
            backward_remote_q_values=inverse_b_backward,
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
        material_characteristic_effective_face_states = jnp.zeros(
            self.geometry.owned_shape + (2, 5), dtype=jnp.float64
        )
        material_characteristic_current_values = jnp.zeros(
            self.geometry.owned_shape + (2, 6), dtype=jnp.float64
        )
        material_characteristic_particle_flux_values = jnp.zeros(
            self.geometry.owned_shape + (2, 2), dtype=jnp.float64
        )
        material_characteristic_wall_metadata = jnp.zeros(
            self.geometry.owned_shape + (2, 3), dtype=jnp.float64
        )
        vorticity_parallel_advection = jnp.zeros(
            self.geometry.owned_shape, dtype=jnp.float64
        )
        parallel_material_residual = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        )
        parallel_material_explicit_components = jnp.zeros(
            self.geometry.owned_shape + (3, 5,), dtype=jnp.float64
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
            "backward_wall_current": jnp.zeros(self.geometry.owned_shape),
            "forward_wall_current": jnp.zeros(self.geometry.owned_shape),
        }
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
            # Vorticity is a scalar advected by the live ion parallel speed,
            # not a member of the five-field material eigensystem.  Build its
            # raw mapped stencil with the same operator trace used by the
            # polarization/current pair.  The scalar kernel then selects the
            # upstream leg directly from the canonical live Vi center.
            vorticity_halo, vorticity_forward, vorticity_backward = (
                self._fci_prepare_q(
                    fields["vorticity"][owned],
                    traces["vorticity"],
                    context,
                )
            )
            vorticity_stencil = build_local_fci_stencil_from_field(
                vorticity_halo,
                self.geometry,
                context,
                forward_remote_values=vorticity_forward,
                backward_remote_values=vorticity_backward,
            )
            vorticity_parallel_advection = (
                parallel_vorticity_upwind_residual(
                    vorticity_stencil.center,
                    vorticity_stencil.minus,
                    vorticity_stencil.plus,
                    center[..., 3],
                    vorticity_stencil.dx_min,
                    vorticity_stencil.dx_plus,
                )
            )
            backward_wall = (
                self.geometry.maps.backward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            forward_wall = (
                self.geometry.maps.forward.endpoint_kind
                == FCI_DEP_PHYSICAL_BOUNDARY
            )
            backward_wall_state = minus
            forward_wall_state = plus
            if self.physical_wall_model_name == "simple-conducting-sheath":
                # Interpolate smooth primitive inputs and geometry to each FCI
                # hit before evaluating sign(B.n), Bohm pass-through, or the
                # electron exponential.  The regular face bundle remains the
                # correct source for coordinate-face operators, but its
                # already-branched velocity values must not be interpolated to
                # a distinct mapped endpoint.
                plasma_stencils = {
                    name: self._fci_plasma_side_stencil(
                        fields[name][owned], getattr(face_bc, name), context
                    )
                    for name in ("Vi", "Ve", "phi")
                }

                def endpoint_plasma_state(direction: str):
                    base = minus if direction == "backward" else plus
                    values = [base[..., index] for index in range(3)]
                    values.extend(
                        getattr(plasma_stencils[name], "minus" if direction == "backward" else "plus")
                        for name in ("Vi", "Ve")
                    )
                    return jnp.stack(tuple(values), axis=-1)

                backward_plasma = endpoint_plasma_state("backward")
                forward_plasma = endpoint_plasma_state("forward")
                backward_resolved = resolve_fci_material_wall_endpoint_state(
                    self.physical_wall_model_name,
                    backward_plasma,
                    plasma_stencils["phi"].minus,
                    self.geometry.maps.backward.endpoint_b_contra_x,
                    self.geometry.maps.backward.endpoint_bmag,
                    self.parameters,
                    conducting_sheath_wall_potential=(
                        self.conducting_sheath_wall_potential
                    ),
                )
                forward_resolved = resolve_fci_material_wall_endpoint_state(
                    self.physical_wall_model_name,
                    forward_plasma,
                    plasma_stencils["phi"].plus,
                    self.geometry.maps.forward.endpoint_b_contra_x,
                    self.geometry.maps.forward.endpoint_bmag,
                    self.parameters,
                    conducting_sheath_wall_potential=(
                        self.conducting_sheath_wall_potential
                    ),
                )
                backward_wall_state = jnp.where(
                    backward_wall[..., None], backward_resolved, minus
                )
                forward_wall_state = jnp.where(
                    forward_wall[..., None], forward_resolved, plus
                )
            # Keep one canonical live wall-data evaluation for the material
            # residual and, when selected, the characteristic current closure.
            # Legacy projected laws export their first-order characteristic
            # current; resolved physical-wall laws export the exact nonlinear
            # current of their resolved face.  Both consumers receive the
            # same wall-data object, so the endpoint contract cannot drift.
            wall_data = None
            if (
                self.parallel_boundary_pairing == "characteristic-sat"
                or return_electron_force_diagnostics
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
                    parallel_short_leg_selection=self.parallel_short_leg_selection,
                    backward_wall=backward_wall,
                    forward_wall=forward_wall,
                    backward_wall_state=backward_wall_state,
                    forward_wall_state=forward_wall_state,
                    parallel_characteristic_wall_law=(
                        self.parameters.parallel_characteristic_wall_law
                    ),
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
                # The affine characteristic wall-current lift is part of the
                # selected characteristic SAT closure. The former suppressed
                # variant was diagnostic-only.
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
                    backward_wall_state=backward_wall_state,
                    forward_wall_state=forward_wall_state,
                    div_b=div_b,
                    selection_dt=short_leg_selection_dt
                    if self.parallel_short_leg_treatment == "local-backward-euler"
                    else 0.0,
                    cfl_limit=self.parallel_short_leg_cfl_limit,
                    parallel_short_leg_selection=self.parallel_short_leg_selection,
                    parallel_characteristic_wall_law=(
                        self.parameters.parallel_characteristic_wall_law
                    ),
                    resolved_wall_data=wall_data,
                )
            )
            if return_electron_force_diagnostics and wall_data is not None:
                # Exact additive split of the *live explicit* production
                # residual. Selected physical-wall legs are advanced by the
                # local implicit solve and therefore contribute zero here;
                # the middle lane retains the geometric div(b) source and
                # any algebraic remainder. This uses the already-live
                # directional actions rather than restoring the retired
                # full-grid provenance diagnostics.
                explicit_backward_residual = jnp.where(
                    wall_data["selected_backward_wall"][..., None],
                    0.0,
                    wall_data["backward_residual"],
                )
                explicit_forward_residual = jnp.where(
                    wall_data["selected_forward_wall"][..., None],
                    0.0,
                    wall_data["forward_residual"],
                )
                explicit_center_geometric_residual = (
                    parallel_material_residual
                    - explicit_backward_residual
                    - explicit_forward_residual
                )
                parallel_material_explicit_components = jnp.stack(
                    (
                        explicit_backward_residual,
                        explicit_center_geometric_residual,
                        explicit_forward_residual,
                    ),
                    axis=-2,
                )
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
                incoming_projectors = jnp.stack(
                    (
                        wall_data["backward_incoming_projector"],
                        wall_data["forward_incoming_projector"],
                    ),
                    axis=-3,
                )
                endpoint_states = jnp.stack(
                    (
                        wall_data["backward_endpoint_state"],
                        wall_data["forward_endpoint_state"],
                    ),
                    axis=-2,
                )
                incoming_deltas = jnp.einsum(
                    "...dij,...dj->...di",
                    incoming_projectors,
                    endpoint_states - center[..., None, :],
                )
                material_characteristic_effective_face_states = (
                    center[..., None, :] + incoming_deltas
                )

                def nonlinear_current(state):
                    return state[..., 0] * (state[..., 3] - state[..., 4])

                owner_current = nonlinear_current(center)
                effective_current = nonlinear_current(
                    material_characteristic_effective_face_states
                )
                effective_linearized_current = (
                    owner_current[..., None]
                    + (center[..., 3] - center[..., 4])[..., None]
                    * incoming_deltas[..., 0]
                    + center[..., 0, None]
                    * (incoming_deltas[..., 3] - incoming_deltas[..., 4])
                )
                if self.parallel_boundary_pairing == "characteristic-sat":
                    _, effective_nonlinear_divergence, _ = (
                        self._fci_current_phi_boundary_pair(
                            face_bc=face_bc,
                            context=context,
                            wall_endpoint_current_values=(
                                effective_current[..., 0],
                                effective_current[..., 1],
                            ),
                            build_adjoint=False,
                        )
                    )
                    _, effective_linearized_divergence, _ = (
                        self._fci_current_phi_boundary_pair(
                            face_bc=face_bc,
                            context=context,
                            wall_endpoint_current_values=(
                                effective_linearized_current[..., 0],
                                effective_linearized_current[..., 1],
                            ),
                            build_adjoint=False,
                        )
                    )
                    actual_current = fields["current"][owned]
                    characteristic_sat_effective_nonlinear_current_divergence = (
                        effective_nonlinear_divergence(actual_current)
                    )
                    characteristic_sat_effective_linearized_current_divergence = (
                        effective_linearized_divergence(actual_current)
                    )
                raw_wall_currents = jnp.stack(
                    (
                        wall_data["backward_candidate_current"],
                        wall_data["forward_candidate_current"],
                    ),
                    axis=-1,
                )
                exported_sat_currents = jnp.stack(
                    (
                        wall_data["backward_wall_characteristic_current"],
                        wall_data["forward_wall_characteristic_current"],
                    ),
                    axis=-1,
                )
                directional_material_residual = jnp.stack(
                    (
                        wall_data["backward_residual"],
                        wall_data["forward_residual"],
                    ),
                    axis=-2,
                )
                current_gradient = jnp.stack(
                    (
                        center[..., 3] - center[..., 4],
                        jnp.zeros_like(center[..., 0]),
                        jnp.zeros_like(center[..., 0]),
                        center[..., 0],
                        -center[..., 0],
                    ),
                    axis=-1,
                )
                material_current_rate = jnp.einsum(
                    "...i,...di->...d",
                    current_gradient,
                    directional_material_residual,
                )
                material_characteristic_current_values = jnp.stack(
                    (
                        jnp.broadcast_to(
                            owner_current[..., None], raw_wall_currents.shape
                        ),
                        raw_wall_currents,
                        effective_current,
                        effective_linearized_current,
                        exported_sat_currents,
                        material_current_rate,
                    ),
                    axis=-1,
                )
                material_characteristic_particle_flux_values = jnp.stack(
                    (
                        material_characteristic_effective_face_states[..., 0]
                        * material_characteristic_effective_face_states[..., 3],
                        material_characteristic_effective_face_states[..., 0]
                        * material_characteristic_effective_face_states[..., 4],
                    ),
                    axis=-1,
                )
                material_characteristic_wall_metadata = jnp.stack(
                    (
                        jnp.stack(
                            (wall_data["backward_wall"], wall_data["forward_wall"]),
                            axis=-1,
                        ).astype(jnp.float64),
                        jnp.stack(
                            (wall_data["backward_cfl"], wall_data["forward_cfl"]),
                            axis=-1,
                        ),
                        jnp.stack(
                            (
                                wall_data["selected_backward_wall"],
                                wall_data["selected_forward_wall"],
                            ),
                            axis=-1,
                        ).astype(jnp.float64),
                    ),
                    axis=-1,
                )
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
            "vorticity_parallel_advection": vorticity_parallel_advection,
            "material_upwind_correction": material_upwind_correction,
            "parallel_material_residual": parallel_material_residual,
            "parallel_material_diagnostics": parallel_material_diagnostics,
            # Keep the canonical face resolution alive for the optional
            # short-leg BE stage.  This is the exact object consumed above by
            # the explicit material residual and characteristic current SAT.
            "parallel_characteristic_wall_data": wall_data,
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
                    "material_characteristic_effective_face_states": (
                        material_characteristic_effective_face_states
                    ),
                    "material_characteristic_current_values": (
                        material_characteristic_current_values
                    ),
                    "material_characteristic_particle_flux_values": (
                        material_characteristic_particle_flux_values
                    ),
                    "material_characteristic_wall_metadata": (
                        material_characteristic_wall_metadata
                    ),
                    "characteristic_sat_effective_nonlinear_current_divergence": (
                        jnp.zeros_like(div_b)
                        if characteristic_sat_effective_nonlinear_current_divergence
                        is None
                        else characteristic_sat_effective_nonlinear_current_divergence
                    ),
                    "characteristic_sat_effective_linearized_current_divergence": (
                        jnp.zeros_like(div_b)
                        if characteristic_sat_effective_linearized_current_divergence
                        is None
                        else characteristic_sat_effective_linearized_current_divergence
                    ),
                    "parallel_material_explicit_components": (
                        parallel_material_explicit_components
                    ),
                }
            )
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
            result[f"{name}_parallel_diff"] = value

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

        # Production FCI uses the central operator traces.  The former
        # local/equilibrium characteristic inflow closures were diagnostic
        # experiments and are intentionally no longer routed here.
        return operator_boundary

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
    ) -> dict[str, jnp.ndarray]:
        """Evaluate the pre-existing coordinate parallel stage operators."""

        operator_kwargs = dict(
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        density_flux_div = local_parallel_flux_div_op(
            density_flux_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.density_flux,
            **operator_kwargs
        )
        current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.current,
            **operator_kwargs
        )
        vorticity_current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=operator_boundary.current,
            **operator_kwargs
        )
        Vi_stencil = build_local_conservative_stencil_from_field(
            state_halo.Vi, self.geometry, context
        )
        parallel_Vi_flux_div = local_parallel_flux_div_op(
            Vi_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Vi,
            **operator_kwargs
        )
        Ve_flux_div = local_parallel_flux_div_op(
            Ve_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Ve,
            **operator_kwargs
        )
        grad_Te = local_grad_parallel_op_conservative(
            Te_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Te,
            **operator_kwargs
        )
        grad_Ti = local_grad_parallel_op_conservative(
            Ti_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ti,
            **operator_kwargs
        )
        grad_Ve = local_grad_parallel_op_conservative(
            Ve_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ve,
            **operator_kwargs
        )
        grad_Vi = local_grad_parallel_op_conservative(
            Vi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Vi,
            **operator_kwargs
        )
        grad_phi = local_grad_parallel_op_conservative(
            phi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.phi,
            **operator_kwargs
        )
        grad_Pe = local_grad_parallel_op_conservative(
            Pe_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Pe,
            **operator_kwargs
        )
        grad_pressure = local_grad_parallel_op_conservative(
            pressure_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.pressure,
            **operator_kwargs
        )
        grad_current = local_grad_parallel_op_conservative(
            current_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.current,
            **operator_kwargs
        )
        grad_vorticity = local_grad_parallel_op_conservative(
            vorticity_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.vorticity,
            **operator_kwargs
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

    def parallel_wall_current_diagnostics(
        self,
        state_owned: FciDrbEBState,
        *,
        phi_owned: jnp.ndarray | None = None,
        selection_dt: Any = 0.0,
    ) -> tuple[jnp.ndarray, ...]:
        """Return replay-only characteristic/SAT wall-current measurements.

        The outputs compare the primitive wall endpoint against the effective
        characteristic interface state ``q0 + P_in (qw - q0)``.  They are
        diagnostics only and do not alter the production boundary action.
        Direction order is backward/forward.  Current channels are owner,
        raw-wall, effective nonlinear, effective linearized, exported SAT,
        and the directional material contribution to the owner-current rate.
        """

        if self.parallel_operator_scheme != "fci":
            raise ValueError("wall-current diagnostics require FCI")
        state_owned = self._owner_state(state_owned)
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
        parallel_boundary = self._parallel_operator_boundary(
            state_halo=state_halo,
            operator_boundary=operator_boundary,
        )
        parallel_terms = self._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=self._stencil_builder_context(),
            return_electron_force_diagnostics=True,
            short_leg_selection_dt=selection_dt,
        )

        def fields_first(value):
            return jnp.moveaxis(value, (-1, -2), (0, 1))

        raw_endpoint_states = fields_first(
            parallel_terms["material_characteristic_endpoint_values"]
        )
        effective_face_states = fields_first(
            parallel_terms["material_characteristic_effective_face_states"]
        )
        current_values = fields_first(
            parallel_terms["material_characteristic_current_values"]
        )
        particle_flux_values = fields_first(
            parallel_terms["material_characteristic_particle_flux_values"]
        )
        wall_metadata = fields_first(
            parallel_terms["material_characteristic_wall_metadata"]
        )
        current_divergences = jnp.stack(
            (
                parallel_terms[
                    "characteristic_sat_homogeneous_current_divergence"
                ],
                parallel_terms["characteristic_sat_affine_current_divergence"],
                parallel_terms["characteristic_sat_current_divergence"],
                parallel_terms[
                    "characteristic_sat_effective_nonlinear_current_divergence"
                ],
                parallel_terms[
                    "characteristic_sat_effective_linearized_current_divergence"
                ],
            ),
            axis=0,
        )
        return (
            raw_endpoint_states,
            effective_face_states,
            current_values,
            particle_flux_values,
            wall_metadata,
            current_divergences,
            jnp.moveaxis(
                parallel_terms["material_characteristic_leg_lengths"], -1, 0
            ),
        )

    def apply_short_leg_implicit_material_step(
        self,
        state_owned: FciDrbEBState,
        solve_dt: Any,
        selection_dt: Any | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
        return_increment: bool = False,
    ) -> FciDrbEBState | tuple[FciDrbEBState, FciDrbEBState, dict[str, jnp.ndarray]]:
        """Apply the complete selected-wall-leg backward-Euler stage.

        The implicit residual contains both the five-field characteristic
        material action and the matching ion-temperature force
        ``mu*tau*grad_parallel(Ti)``.  That force is assembled by the
        compatible production gradient outside the material flux, but it
        cancels the Ti column of the centered electron material equation and
        therefore must cross the explicit/implicit handoff with it.

        ``mu*grad_parallel(phi)`` deliberately remains explicit together
        with its weighted-adjoint current-divergence partner in the
        vorticity equation.  Moving only the phi force into this local
        five-field solve would create a second, global current/phi handoff
        defect.  ``phi`` is still supplied so both paths use the same stage
        boundary bundle, and the time integrator reconstructs it after the
        local solve.

        In the default ``cfl`` mode, a physical wall leg is selected when it exceeds
        ``parallel_short_leg_cfl_limit`` measured with ``selection_dt``;
        ``all-physical-walls`` selects every physical wall leg.  All
        mapped/bulk rows, the geometric ``div(b)`` source, diffusion,
        collisions, perpendicular physics, polarization, and vorticity remain
        in the explicit RHS.  The fine-row increment is passed through the
        same volume-weighted RLP restriction as an ordinary RHS contribution.
        ``return_increment`` exposes that owner-space increment for an IMEX
        stage without inferring it from the algebraic ``phi`` field.
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
        parallel_terms = self._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=context,
            short_leg_selection_dt=selection_dt,
        )
        coupled_force = (
            self.parameters.mi_over_me
            * self.parameters.tau
            * parallel_terms["grad_Ti"]
        )
        coupled_residual = jnp.zeros(
            self.geometry.owned_shape + (5,), dtype=jnp.float64
        ).at[..., 4].set(coupled_force)
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
            parallel_short_leg_selection=self.parallel_short_leg_selection,
            backward_wall=backward_wall,
            forward_wall=forward_wall,
            backward_wall_state=minus,
            forward_wall_state=plus,
            parallel_characteristic_wall_law=(
                self.parameters.parallel_characteristic_wall_law
            ),
            coupled_residual=coupled_residual,
            resolved_wall_data=parallel_terms.get(
                "parallel_characteristic_wall_data"
            ),
        )

        if self._uses_projected_fine_grid:
            increment_owner = jax.vmap(
                self._restrict_fine_field
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
        updated_state = self._owner_state(state_owned.replace(
            density=state_owned.density + increment_owner[..., 0],
            Te=state_owned.Te + increment_owner[..., 1],
            Ti=state_owned.Ti + increment_owner[..., 2],
            Vi=state_owned.Vi + increment_owner[..., 3],
            Ve=state_owned.Ve + increment_owner[..., 4],
        ))
        if not return_increment:
            return updated_state
        zero = jnp.zeros_like(state_owned.density)
        increment_state = self._owner_state(FciDrbEBState(
            density=increment_owner[..., 0],
            phi=zero,
            Te=increment_owner[..., 1],
            Ti=increment_owner[..., 2],
            Vi=increment_owner[..., 3],
            Ve=increment_owner[..., 4],
            vorticity=zero,
        ))
        info = dict(info)
        info["selected_coupled_force"] = jnp.where(
            info["selected_wall"], coupled_force, 0.0
        )
        complete_residual = jnp.moveaxis(
            info["selected_complete_residual"], -1, 0
        )
        if self._uses_projected_fine_grid:
            complete_residual = jax.vmap(self._restrict_fine_field)(
                complete_residual
            )
        complete_residual_owner = jnp.moveaxis(complete_residual, 0, -1)
        if self.control_volume_geometry is not None:
            complete_residual_owner = jnp.where(
                self.control_volume_geometry.cells.is_active_owner[..., None],
                complete_residual_owner,
                0.0,
            )
        else:
            complete_residual_owner = jnp.where(
                self.geometry.active_cell_mask_owned[..., None],
                complete_residual_owner,
                0.0,
            )
        info["selected_complete_residual_owner"] = complete_residual_owner
        return updated_state, increment_state, info

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
        return_parallel_material_component_fields: bool = False,
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

        ``return_parallel_material_component_fields=True`` adds the exact
        five-field production-material split in backward, center/geometric,
        and forward order. It is available only with
        ``return_rhs_term_fields=True`` and is intended for selected-cell
        staged audits.
        """

        legacy_diagnostic_count = sum(
            bool(value) for value in (return_term_diagnostics, return_term_fields)
        )
        if legacy_diagnostic_count > 1 or (
            legacy_diagnostic_count
            and (
                return_rhs_term_fields
                or return_curvature_component_fields
                or return_parallel_material_component_fields
            )
        ):
            raise ValueError(
                "RHS term diagnostic return modes are mutually exclusive"
            )
        if (
            return_parallel_material_component_fields
            and not return_rhs_term_fields
        ):
            raise ValueError(
                "parallel-material component fields require "
                "return_rhs_term_fields=True"
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
        state_halo_without_phi = self._prepare_state_halo(state_owned, face_bc)
        if phi_owned is None:
            phi_owned = self._reconstruct_phi_from_prepared(
                state_owned,
                state_halo_without_phi,
                face_bc,
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
                return_electron_force_diagnostics=(
                    return_parallel_material_component_fields
                ),
            )
            if self.parallel_operator_scheme == "fci"
            else None
        )
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
        )
        Te_diff = self._field_perp_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_D_perp,
        )
        Ti_diff = self._field_perp_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_D_perp,
        )
        Vi_diff = self._field_perp_diffusion(
            Vi_perp_halo,
            face_bc.Vi,
            self.parameters.Vi_D_perp,
        )
        Ve_diff = self._field_perp_diffusion(
            Ve_perp_halo,
            face_bc.Ve,
            self.parameters.Ve_D_perp,
        )
        vorticity_diff = self._field_perp_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_perp,
        )
        if self.parallel_operator_scheme == "coordinate":
            density_parallel_diff = self._field_parallel_diffusion(
                state_halo.density,
                face_bc.density,
                self.parameters.density_D_parallel,
            )
            Te_parallel_diff = self._field_parallel_diffusion(
                state_halo.Te,
                face_bc.Te,
                self.parameters.electron_temperature_chi_parallel,
            )
            Ti_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ti,
                face_bc.Ti,
                self.parameters.ion_temperature_chi_parallel,
            )
            Vi_parallel_diff = self._field_parallel_diffusion(
                state_halo.Vi,
                face_bc.Vi,
                self.parameters.Vi_parallel_viscosity,
            )
            Ve_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ve,
                face_bc.Ve,
                self.parameters.Ve_parallel_viscosity,
            )
            vorticity_parallel_diff = self._field_parallel_diffusion(
                state_halo.vorticity,
                face_bc.vorticity,
                self.parameters.vorticity_D_parallel,
            )
        else:
            density_parallel_diff = fci_parallel_terms["density_parallel_diff"]
            Te_parallel_diff = fci_parallel_terms["Te_parallel_diff"]
            Ti_parallel_diff = fci_parallel_terms["Ti_parallel_diff"]
            Vi_parallel_diff = fci_parallel_terms["Vi_parallel_diff"]
            Ve_parallel_diff = fci_parallel_terms["Ve_parallel_diff"]
            vorticity_parallel_diff = fci_parallel_terms["vorticity_parallel_diff"]
        poisson_density = self._poisson_bracket_over_B(
            phi_gradient,
            density_gradient,
            phi_conservative_stencil,
            density_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.density,
            g_field_halo=state_halo.density,
            g_positivity_floor=1.0e-12,
        )
        poisson_Te = self._poisson_bracket_over_B(
            phi_gradient,
            Te_gradient,
            phi_conservative_stencil,
            Te_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Te,
            g_field_halo=state_halo.Te,
            g_positivity_floor=1.0e-12,
        )
        poisson_Ti = self._poisson_bracket_over_B(
            phi_gradient,
            Ti_gradient,
            phi_conservative_stencil,
            Ti_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ti,
            g_field_halo=state_halo.Ti,
            g_positivity_floor=1.0e-12,
        )
        poisson_Vi = self._poisson_bracket_over_B(
            phi_gradient,
            Vi_gradient,
            phi_conservative_stencil,
            Vi_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=perpendicular_operator_boundary.Vi,
            g_field_halo=Vi_perp_halo,
        )
        poisson_Ve = self._poisson_bracket_over_B(
            phi_gradient,
            Ve_gradient,
            phi_conservative_stencil,
            Ve_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=perpendicular_operator_boundary.Ve,
            g_field_halo=Ve_perp_halo,
        )
        poisson_vorticity = self._poisson_bracket_over_B(
            phi_gradient,
            vorticity_gradient,
            phi_conservative_stencil,
            vorticity_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.vorticity,
            g_field_halo=state_halo.vorticity,
            equation_family="vorticity",
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
        n_face_safe = density_safe
        Vi_parallel_value, Ve_parallel_value = Vi, Ve
        current_parallel_value = density * (Vi - Ve)
        Te_parallel_advection = -Ve * grad_parallel_Te
        Ti_parallel_advection = -Vi * grad_parallel_Ti
        vorticity_parallel_advection = (
            stage_parallel_terms["vorticity_parallel_advection"]
            if self.parallel_operator_scheme == "fci" and production_parallel
            else -Vi * grad_parallel_vorticity
        )
        Vi_self_advection_term = -Vi_parallel_value * grad_parallel_Vi
        Vi_pressure_term = -grad_parallel_pressure / n_face_safe
        Ve_self_advection_term = -Ve_parallel_value * grad_parallel_Ve
        Ve_collision_term = mi_over_me * Ve_nu * current_parallel_value
        Ve_phi_force_term = mi_over_me * grad_parallel_phi
        Ve_Ti_force_complete_term = (
            mi_over_me * tau * grad_parallel_Ti
            if production_parallel
            else jnp.zeros_like(grad_parallel_phi)
        )
        material_diagnostics = stage_parallel_terms.get(
            "parallel_material_diagnostics", {}
        )
        selected_short_wall = material_diagnostics.get(
            "selected_wall",
            jnp.zeros(self.geometry.owned_shape, dtype=bool),
        )
        # The material Ti column and mu*tau*grad(Ti) are one principal
        # balance.  Hand both to the same short-leg stage.  The phi force is
        # not masked: it stays explicit with its weighted-adjoint
        # current-divergence partner in the vorticity equation.
        Ve_Ti_force_term = (
            jnp.where(selected_short_wall, 0.0, Ve_Ti_force_complete_term)
            if (
                production_parallel
                and self.parallel_short_leg_treatment == "local-backward-euler"
            )
            else Ve_Ti_force_complete_term
        )
        Ve_electrostatic_term = Ve_phi_force_term + Ve_Ti_force_term
        vorticity_current_term = (
            (bmag * bmag / density_safe) * vorticity_current_flux_divergence
        )
        Ve_pressure_term = -mi_over_me * grad_parallel_Pe / n_face_safe
        Ve_thermal_force_term = -0.71 * mi_over_me * grad_parallel_Te
        curvature_outputs = self._curvature_rhs_contributions(
            state_halo=state_halo,
            context=context,
            density=density,
            Te=Te,
            Ti=Ti,
            bmag=bmag,
            tau=tau,
            operator_boundary=operator_boundary,
            density_conservative_stencil=density_conservative_stencil,
            Te_conservative_stencil=Te_conservative_stencil,
            Ti_conservative_stencil=Ti_conservative_stencil,
            vorticity_conservative_stencil=vorticity_conservative_stencil,
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
            + vorticity_current_term
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
        ) -> jnp.ndarray:
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

        def all_rhs_term_fields() -> jnp.ndarray:
            density_terms = pack_rhs_terms(
                (
                    -(poisson_density / rho_star),
                    density_parallel_material_term,
                    curvature_density_contribution,
                    density_diff,
                    density_parallel_diff,
                    material_upwind_correction[..., 0],
                ),
                source_owned.density,
            )
            Te_terms = pack_rhs_terms(
                (
                    -(poisson_Te / rho_star),
                    Te_parallel_material_term if production_parallel else Te_parallel_advection,
                    curvature_Te_contribution,
                    zero_term if production_parallel else (
                        (2.0 * Te / (3.0 * density_safe))
                        * (0.71 * parallel_current_flux_divergence
                           - density * parallel_Ve_flux_divergence)
                    ),
                    Te_diff,
                    Te_parallel_diff,
                    material_upwind_correction[..., 1],
                ),
                source_owned.Te,
            )
            Ti_terms = pack_rhs_terms(
                (
                    -(poisson_Ti / rho_star),
                    Ti_parallel_material_term if production_parallel else Ti_parallel_advection,
                    curvature_Ti_contribution,
                    zero_term if production_parallel else (
                        (2.0 * Ti / (3.0 * density_safe))
                        * (parallel_current_flux_divergence
                           - density * parallel_Vi_flux_divergence)
                    ),
                    Ti_diff,
                    Ti_parallel_diff,
                    material_upwind_correction[..., 2],
                ),
                source_owned.Ti,
            )
            Vi_terms = pack_rhs_terms(
                (
                    Vi_poisson_term,
                    Vi_parallel_material_term if production_parallel else Vi_self_advection_term,
                    zero_term if production_parallel else Vi_pressure_term,
                    Vi_diff_term,
                    Vi_parallel_diff,
                    material_upwind_correction[..., 3],
                ),
                source_owned.Vi,
            )
            Ve_terms = pack_rhs_terms(
                (
                    Ve_poisson_term,
                    Ve_parallel_material_term if production_parallel else Ve_self_advection_term,
                    Ve_collision_term,
                    Ve_electrostatic_term,
                    zero_term if production_parallel else Ve_pressure_term,
                    zero_term if production_parallel else Ve_thermal_force_term,
                    Ve_diff_term,
                    Ve_parallel_diff,
                    zero_term if production_parallel else Ve_characteristic_upwind_term,
                ),
                source_owned.Ve,
            )
            vorticity_terms = pack_rhs_terms(
                (
                    -(poisson_vorticity / rho_star),
                    vorticity_parallel_advection,
                    vorticity_current_term,
                    curvature_vorticity_contribution,
                    vorticity_diff,
                    vorticity_parallel_diff,
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
            rhs_terms = all_rhs_term_fields()
            diagnostic_outputs = [result, rhs_terms]
            if return_curvature_component_fields:
                diagnostic_outputs.append(curvature_component_fields)
            if return_parallel_material_component_fields:
                material_components = jnp.moveaxis(
                    fci_parallel_terms[
                        "parallel_material_explicit_components"
                    ],
                    (-2, -1),
                    (0, 1),
                )
                restrict_component = lambda value: self._owner_field(
                    _mask_inactive_owned(
                        self._restrict_fine_field(value), self.geometry
                    )
                )
                material_components = jax.vmap(jax.vmap(restrict_component))(
                    material_components
                )
                diagnostic_outputs.append(material_components)
            return tuple(diagnostic_outputs)
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
            if self._uses_projected_fine_grid:
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
