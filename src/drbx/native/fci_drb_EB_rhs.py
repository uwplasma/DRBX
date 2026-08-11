from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import (
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    SIDE_PHYSICAL,
    StencilBuilderContext,
    build_axis_core_cell_gradient_reconstruction,
    build_axis_core_face_gradient_reconstruction,
    build_local_conservative_stencil_from_field,
    build_local_cell_gradient_from_field,
)
from .fci_model import FciModelState
from .fci_model import inject_owned_field_to_halo, inject_owned_state_to_halo
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NONE,
    ConservativeStencil3D,
    LocalBoundaryFaceBC3D,
    LocalBoundaryFaceTrace3D,
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
    AxisCoreLineUPreconditioner3D,
    AxisCoreReducedSolveInfo,
    AxisCoreReducedSpace3D,
    LocalPerpLaplacianInverseSolver,
    build_axis_core_line_u_preconditioner,
    build_axis_core_reduced_space,
    build_solvax_perp_laplacian_preconditioner,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_conservative,
    local_parallel_div_b_op,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_parallel_q_flux_div_fci_op,
    local_parallel_div_b_fci_from_q_op,
    local_grad_parallel_op_fci_compatible_from_q,
    local_parallel_diffusion_fci_op,
    local_perp_laplacian_conservative_op,
    local_curvature_conservative_op,
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op_from_gradients,
    _mask_inactive_owned,
    _mask_state_inactive_owned,
)
from .fci_gmres import SolvaxGmresConfig, SolvaxGmresInfo


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


def project_axis_core_state(
    state: FciDrbEBState,
    space: AxisCoreReducedSpace3D | None,
    state_space: str,
) -> FciDrbEBState:
    """Project the six differential leaves while preserving the phi leaf."""

    if state_space == "full-grid":
        return state
    if state_space != "galerkin":
        raise ValueError(f"unknown axis-core state space {state_space!r}")
    if space is None:
        raise ValueError("Galerkin axis-core state projection requires a reduced space")
    return state.replace(
        density=space.project(state.density),
        Te=space.project(state.Te),
        Ti=space.project(state.Ti),
        Vi=space.project(state.Vi),
        Ve=space.project(state.Ve),
        vorticity=space.project(state.vorticity),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbEBImplicitState(FciModelState):
    """The six algebraic/differential unknowns of an EB IMEX stage.

    ``Vi`` deliberately does not appear here: ion parallel velocity dynamics
    remain stage-known and explicit, while ion temperature is solved together
    with the electron/acoustic--polarization block.  The field order is stable
    and is suitable for ``jax.linearize`` and sharded matrix-free
    Newton--Krylov methods.
    """

    density: jax.Array
    phi: jax.Array
    Te: jax.Array
    Ti: jax.Array
    Ve: jax.Array
    vorticity: jax.Array


def implicit_state_from_eb_state(state: FciDrbEBState) -> FciDrbEBImplicitState:
    """Extract the implicit ARK stage variables from a full EB state."""

    return FciDrbEBImplicitState(
        density=state.density,
        phi=state.phi,
        Te=state.Te,
        Ti=state.Ti,
        Ve=state.Ve,
        vorticity=state.vorticity,
    )


def eb_state_with_implicit_state(
    known_explicit_state: FciDrbEBState,
    implicit_state: FciDrbEBImplicitState,
) -> FciDrbEBState:
    """Merge IMEX unknowns with the stage-known explicit ion fields."""

    return known_explicit_state.replace(
        density=implicit_state.density,
        phi=implicit_state.phi,
        Te=implicit_state.Te,
        Ti=implicit_state.Ti,
        Ve=implicit_state.Ve,
        vorticity=implicit_state.vorticity,
    )


def build_eb_imex_phi_line_u_preconditioner(
    model: "LocalFciDrbEBRhs",
    dt_gamma: float | jax.Array,
    *,
    reference_state: FciDrbEBState | None = None,
) -> Callable[[FciDrbEBImplicitState], FciDrbEBImplicitState]:
    """Build a cheap coupled-IMEX right preconditioner.

    The current first block approximation leaves the differential
    ``(n, Te, Ti, Ve, omega)`` leaves untouched and applies the established local
    line-u approximation to the algebraic polarization block.  The Newton
    driver scales that row as ``dt_gamma * (-L_perp(phi) + ...)``; hence the
    returned phi correction is ``P_A^-1(r_phi / dt_gamma)`` for
    ``A = -L_perp``.  No nested converged phi solve is performed.

    ``reference_state`` is only used to obtain the phi face kind/mask.  The
    production phi boundary is homogeneous Dirichlet, but accepting it makes
    this helper safe for state-dependent boundary builders as well.
    """

    if not isinstance(model, LocalFciDrbEBRhs):
        raise TypeError("model must be a LocalFciDrbEBRhs instance")
    if reference_state is None:
        zeros = jnp.zeros(model.geometry.owned_shape, dtype=jnp.float64)
        reference_state = FciDrbEBState(
            density=zeros,
            phi=zeros,
            Te=zeros,
            Ti=zeros,
            Vi=zeros,
            Ve=zeros,
            vorticity=zeros,
        )
    if not isinstance(reference_state, FciDrbEBState):
        raise TypeError("reference_state must be an FciDrbEBState or None")
    face_bc = model._face_bcs(reference_state).phi
    config = replace(model.gmres_config, preconditioner="line-u")
    scalar_preconditioner = build_solvax_perp_laplacian_preconditioner(
        model.geometry,
        model.domain,
        model.face_projectors,
        face_bc,
        config,
    )
    if scalar_preconditioner is None:  # Defensive: line-u is always nonempty.
        raise RuntimeError("failed to construct line-u perpendicular preconditioner")
    dt_gamma = jnp.asarray(dt_gamma, dtype=jnp.float64)
    active = jnp.asarray(model.geometry.active_cell_mask_owned, dtype=bool)

    def preconditioner(
        residual: FciDrbEBImplicitState,
    ) -> FciDrbEBImplicitState:
        if not isinstance(residual, FciDrbEBImplicitState):
            raise TypeError("residual must be an FciDrbEBImplicitState")
        phi = scalar_preconditioner(residual.phi / dt_gamma)
        # Preserve inactive entries exactly: they are not part of the Newton
        # owned-vector norm and should never receive a line-solver update.
        phi = jnp.where(active, phi, residual.phi)
        return residual.replace(phi=phi)

    return preconditioner


def build_eb_imex_acoustic_line_uv_preconditioner(
    model: "LocalFciDrbEBRhs",
    dt_gamma: float | jax.Array,
    *,
    reference_state: FciDrbEBState | None = None,
) -> Callable[[FciDrbEBImplicitState], FciDrbEBImplicitState]:
    """Build a fixed-cost right preconditioner for an IMEX electron block.

    This is deliberately a *single-sweep* block-triangular approximation to
    the frozen Newton matrix, not an inner iterative solve.  Its ordering is

    1. apply a local ``line-uv`` approximation to the polarization block;
    2. take one frozen-coefficient electron-acoustic response sweep for
       ``(n, Te, Ve, omega)``; and
    3. back-substitute ``omega`` into the algebraic phi row.

    The frozen response is evaluated with a JVP of the production implicit
    operator.  Consequently the approximation contains the same local
    parallel D/G operators, thermodynamic coefficients, collisions, and
    parallel diffusion as the stage Jacobian, while avoiding any nested
    Krylov or nonlinear solve.  The only solve is the fixed-cost line-uv
    factor already used as a scalar perpendicular-Laplacian preconditioner.

    The Newton driver scales the algebraic row by ``dt_gamma``.  For a right
    hand side ``r`` that row is therefore

        dt_gamma * (A dphi + domega) = r_phi,

    where ``A = -L_perp``.  Both phi applications below use
    ``P_A^-1(r_phi / dt_gamma - domega)``.  This keeps the block scaling
    consistent with :func:`build_eb_imex_phi_line_u_preconditioner`.
    """

    if not isinstance(model, LocalFciDrbEBRhs):
        raise TypeError("model must be a LocalFciDrbEBRhs instance")
    if reference_state is None:
        zeros = jnp.zeros(model.geometry.owned_shape, dtype=jnp.float64)
        reference_state = FciDrbEBState(
            density=zeros,
            phi=zeros,
            Te=zeros,
            Ti=zeros,
            Vi=zeros,
            Ve=zeros,
            vorticity=zeros,
        )
    if not isinstance(reference_state, FciDrbEBState):
        raise TypeError("reference_state must be an FciDrbEBState or None")

    face_bc = model._face_bcs(reference_state).phi
    scalar_config = replace(model.gmres_config, preconditioner="line-uv")
    scalar_preconditioner = build_solvax_perp_laplacian_preconditioner(
        model.geometry,
        model.domain,
        model.face_projectors,
        face_bc,
        scalar_config,
    )
    if scalar_preconditioner is None:  # Defensive: line-uv is nonempty.
        raise RuntimeError("failed to construct line-uv perpendicular preconditioner")

    dt_gamma = jnp.asarray(dt_gamma, dtype=jnp.float64)
    active = jnp.asarray(model.geometry.active_cell_mask_owned, dtype=bool)
    frozen_implicit = implicit_state_from_eb_state(reference_state)

    def frozen_implicit_rhs(
        implicit_state: FciDrbEBImplicitState,
    ) -> FciDrbEBImplicitState:
        stage = eb_state_with_implicit_state(reference_state, implicit_state)
        return model.evaluate_implicit_rhs(
            stage, phi_owned=implicit_state.phi, include_curvature=True
        )

    def phi_back_substitution(
        phi_rhs: jax.Array,
        vorticity_correction: jax.Array,
    ) -> jax.Array:
        phi = scalar_preconditioner(
            phi_rhs / dt_gamma - vorticity_correction
        )
        return jnp.where(active, phi, phi_rhs)

    def preconditioner(
        residual: FciDrbEBImplicitState,
    ) -> FciDrbEBImplicitState:
        if not isinstance(residual, FciDrbEBImplicitState):
            raise TypeError("residual must be an FciDrbEBImplicitState")

        # First solve the algebraic block using the identity approximation to
        # the omega differential block.  The following JVP is one frozen
        # electron-acoustic/electrostatic forward sweep.
        initial = residual.replace(
            phi=phi_back_substitution(residual.phi, residual.vorticity)
        )
        _, response = jax.jvp(
            frozen_implicit_rhs,
            (frozen_implicit,),
            (initial,),
        )
        corrected = FciDrbEBImplicitState(
            density=residual.density + dt_gamma * response.density,
            phi=residual.phi,
            Te=residual.Te + dt_gamma * response.Te,
            Ti=residual.Ti + dt_gamma * response.Ti,
            Ve=residual.Ve + dt_gamma * response.Ve,
            vorticity=residual.vorticity + dt_gamma * response.vorticity,
        )
        # Back-substitute the current-divergence response into the
        # polarization relation.  Masking also ensures inactive local cells
        # remain invisible to the global Newton norm.
        corrected = corrected.replace(
            phi=phi_back_substitution(
                residual.phi,
                corrected.vorticity,
            )
        )
        return FciDrbEBImplicitState(
            density=_mask_inactive_owned(corrected.density, model.geometry),
            phi=_mask_inactive_owned(corrected.phi, model.geometry),
            Te=_mask_inactive_owned(corrected.Te, model.geometry),
            Ti=_mask_inactive_owned(corrected.Ti, model.geometry),
            Ve=_mask_inactive_owned(corrected.Ve, model.geometry),
            vorticity=_mask_inactive_owned(corrected.vorticity, model.geometry),
        )

    return preconditioner


def _mask_local_eb_state_inactive(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
) -> FciDrbEBState:
    """Zero inactive owned cells for a local EB state/update payload."""

    return _mask_state_inactive_owned(state, geometry)


def _characteristic_projectors_background(
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (positive electron, negative ion, zero) projectors for M.

    The background is n=Te=Ti=1, omega=0.  The polynomial projectors avoid a
    facewise eigendecomposition and are therefore safe inside jit/shard_map.
    ``bmag`` is retained in the matrix because the vorticity row depends on
    the local wall-face magnetic field.
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
    mu_i = -10.0 * tau / 3.0

    def product(factors):
        result = eye
        for factor in factors:
            result = jnp.einsum("...ij,...jk->...ik", result, factor)
        return result

    P_minus = product((M - mu_plus * eye, M - mu_minus * eye, M))
    P_minus = P_minus / (mu_i * (mu_i - mu_plus) * (mu_i - mu_minus))[..., None, None]
    P_zero = product((M - mu_plus * eye, M - mu_minus * eye, M - mu_i[..., None, None] * eye))
    P_zero = P_zero / ((-mu_plus) * (-mu_minus) * (-mu_i))[..., None, None]
    P_electron = eye - P_minus - P_zero
    return P_electron, P_minus, P_zero


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
    diffusion_only: bool = False
    # Static diagnostic mode: retain only the production parallel subsystem
    # in the returned RHS while still reconstructing phi normally.
    parallel_subsystem_only: bool = False
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    curvature_face_coefficients: LocalCurvatureFaceCoefficients3D | None = None
    upwind_equilibrium_wall_projectors: UpwindEquilibriumWallProjectors | None = None
    curvature_scheme: str = "direct"
    curvature_inflow_closure: str = "central"
    parallel_inflow_closure: str = "central"
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
    # Static axis-core cell-gradient reconstruction policy.  These defaults
    # match the geometry builder's existing radial-axis defaults.
    axis_core_gradient_polynomial_degree: int = 3
    axis_core_gradient_observation_ring_count: int = 6
    axis_core_gradient_target_ring_count: int = 3
    # Experimental selectable discretization for E x B advection.  The
    # compatible-flux path returns the already-B-divided bracket and uses
    # shared conservative face data; ``direct`` preserves the established
    # reconstructed cell-gradient implementation.
    poisson_bracket_scheme: str = "direct"
    # Select the parallel operator family.  This is intentionally a static
    # Python option so JIT compilation cannot silently mix coordinate and FCI
    # discretizations within one compiled RHS.
    parallel_operator_scheme: str = "coordinate"
    axis_core_line_u_preconditioner: AxisCoreLineUPreconditioner3D | None = None
    phi_solver_space: str = "full-grid"
    axis_core_reduced_space: AxisCoreReducedSpace3D | None = None
    axis_core_state_space: str = "full-grid"

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
        if self.parallel_operator_scheme not in ("coordinate", "fci"):
            raise ValueError(
                "parallel_operator_scheme must be 'coordinate' or 'fci', got "
                f"{self.parallel_operator_scheme!r}"
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
        if self.phi_solver_space not in ("full-grid", "axis-core-reduced"):
            raise ValueError(
                "phi_solver_space must be 'full-grid' or 'axis-core-reduced', "
                f"got {self.phi_solver_space!r}"
            )
        if self.axis_core_state_space not in ("full-grid", "galerkin"):
            raise ValueError(
                "axis_core_state_space must be 'full-grid' or 'galerkin', got "
                f"{self.axis_core_state_space!r}"
            )
        if (
            self.axis_core_state_space == "galerkin"
            and self.phi_solver_space != "axis-core-reduced"
        ):
            raise ValueError(
                "axis_core_state_space='galerkin' requires "
                "phi_solver_space='axis-core-reduced'"
            )
        if self.axis_core_reduced_space is not None and not isinstance(
            self.axis_core_reduced_space, AxisCoreReducedSpace3D
        ):
            raise TypeError(
                "axis_core_reduced_space must be an AxisCoreReducedSpace3D or None"
            )
        if self.phi_solver_space == "axis-core-reduced":
            if not self.domain.axis_regular_axes[0] or not self.axis_regular_axes[0]:
                raise ValueError(
                    "phi_solver_space='axis-core-reduced' requires lower radial "
                    "axis regularity in both domain.axis_regular_axes and "
                    "axis_regular_axes"
                )
            if not isinstance(self.gmres_config, SolvaxGmresConfig):
                raise TypeError(
                    "axis-core-reduced phi solving requires a SolvaxGmresConfig"
                )
            if self.gmres_config.preconditioner not in ("none", "line-u"):
                raise ValueError(
                    "axis-core-reduced phi solving supports only GMRES "
                    "preconditioner='none' or 'line-u'; got "
                    f"{self.gmres_config.preconditioner!r}"
                )
        elif self.axis_core_reduced_space is not None:
            raise ValueError(
                "axis_core_reduced_space may only be supplied when "
                "phi_solver_space='axis-core-reduced'"
            )
        if self.axis_core_line_u_preconditioner is not None and not isinstance(
            self.axis_core_line_u_preconditioner,
            AxisCoreLineUPreconditioner3D,
        ):
            raise TypeError(
                "axis_core_line_u_preconditioner must be an "
                "AxisCoreLineUPreconditioner3D or None"
            )
        if self.poisson_bracket_scheme not in ("direct", "compatible-flux"):
            raise ValueError(
                "poisson_bracket_scheme must be 'direct' or "
                f"'compatible-flux', got {self.poisson_bracket_scheme!r}"
            )
        try:
            axis_core_gradient_polynomial_degree = operator.index(
                self.axis_core_gradient_polynomial_degree
            )
            axis_core_gradient_observation_ring_count = operator.index(
                self.axis_core_gradient_observation_ring_count
            )
            axis_core_gradient_target_ring_count = operator.index(
                self.axis_core_gradient_target_ring_count
            )
        except TypeError as exc:
            raise ValueError(
                "axis-core gradient degree and ring counts must be integers"
            ) from exc
        if axis_core_gradient_polynomial_degree < 0:
            raise ValueError(
                "axis_core_gradient_polynomial_degree must be non-negative"
            )
        if (
            axis_core_gradient_observation_ring_count < 1
            or axis_core_gradient_target_ring_count < 1
        ):
            raise ValueError(
                "axis-core gradient ring counts must be positive"
            )
        if (
            axis_core_gradient_target_ring_count
            > axis_core_gradient_observation_ring_count
        ):
            raise ValueError(
                "axis_core_gradient_target_ring_count must not exceed "
                "axis_core_gradient_observation_ring_count"
            )
        object.__setattr__(
            self,
            "axis_core_gradient_polynomial_degree",
            axis_core_gradient_polynomial_degree,
        )
        object.__setattr__(
            self,
            "axis_core_gradient_observation_ring_count",
            axis_core_gradient_observation_ring_count,
        )
        object.__setattr__(
            self,
            "axis_core_gradient_target_ring_count",
            axis_core_gradient_target_ring_count,
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

    def build_axis_core_reduced_phi_space(self) -> AxisCoreReducedSpace3D:
        """Build or validate the Cartesian-core reduced phi space."""

        if self.phi_solver_space != "axis-core-reduced":
            raise ValueError(
                "build_axis_core_reduced_phi_space requires "
                "phi_solver_space='axis-core-reduced'"
            )
        if not self.domain.axis_regular_axes[0] or not self.axis_regular_axes[0]:
            raise ValueError(
                "axis-core-reduced phi space requires lower radial axis regularity"
            )
        context = self._stencil_builder_context()
        reconstruction = context.axis_core_face_gradient_reconstruction
        if reconstruction is None:
            raise RuntimeError(
                "axis-core-reduced phi space requires face-gradient reconstruction"
            )
        payload = self.axis_core_reduced_space
        if payload is not None:
            if payload.geometry.layout != self.geometry.layout:
                raise ValueError(
                    "axis_core_reduced_space geometry layout does not match RHS"
                )
            if payload.geometry.owned_shape != self.geometry.owned_shape:
                raise ValueError(
                    "axis_core_reduced_space geometry shape does not match RHS"
                )
            if payload.domain.layout != self.domain.layout:
                raise ValueError(
                    "axis_core_reduced_space domain layout does not match RHS"
                )
            if (
                payload.domain.shard_spec != self.domain.shard_spec
                or payload.domain.mesh_axis_names != self.domain.mesh_axis_names
            ):
                raise ValueError(
                    "axis_core_reduced_space domain sharding metadata does not "
                    "match RHS"
                )
            payload_reconstruction = payload.reconstruction
            if payload_reconstruction.layout != reconstruction.layout:
                raise ValueError(
                    "axis_core_reduced_space reconstruction layout does not match RHS"
                )
            if payload_reconstruction.global_shape != reconstruction.global_shape:
                raise ValueError(
                    "axis_core_reduced_space reconstruction global shape does not "
                    "match RHS"
                )
            if (
                payload_reconstruction.polynomial_degree
                != reconstruction.polynomial_degree
                or payload_reconstruction.observation_ring_count
                != reconstruction.observation_ring_count
                or payload_reconstruction.target_ring_count
                != reconstruction.target_ring_count
            ):
                raise ValueError(
                    "axis_core_reduced_space reconstruction degree/rings do not "
                    "match RHS axis-core gradient context"
                )
            return payload
        return build_axis_core_reduced_space(
            reconstruction,
            self.geometry,
            self.domain,
        )

    def project_galerkin_state(self, state: FciDrbEBState) -> FciDrbEBState:
        """Project the six evolved scalar leaves, leaving ``phi`` untouched."""

        space = (
            None
            if self.axis_core_state_space == "full-grid"
            else self.build_axis_core_reduced_phi_space()
        )
        return project_axis_core_state(state, space, self.axis_core_state_space)

    def _stencil_builder_context(self) -> StencilBuilderContext:
        if not self.domain.axis_regular_axes[0]:
            return StencilBuilderContext(
                layout=self.domain.layout,
                domain=self.domain,
            )
        cell_reconstruction = build_axis_core_cell_gradient_reconstruction(
            self.domain.layout,
            self.domain,
            polynomial_degree=self.axis_core_gradient_polynomial_degree,
            observation_ring_count=self.axis_core_gradient_observation_ring_count,
            target_ring_count=self.axis_core_gradient_target_ring_count,
        )
        face_gradient_reconstruction = build_axis_core_face_gradient_reconstruction(
            self.domain.layout,
            self.domain,
            polynomial_degree=self.axis_core_gradient_polynomial_degree,
            observation_ring_count=self.axis_core_gradient_observation_ring_count,
            target_ring_count=self.axis_core_gradient_target_ring_count,
        )
        return StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
            # Compatible conservative gradients consume materialized face
            # values through div(f b) - f div(b).  Reuse the face-gradient
            # policy's value targets so both paths protect exactly the same
            # configured radial cells.  Letting the context install its
            # smaller legacy face-value default leaves the outer target ring
            # as a direct polar row.
            axis_core_face_reconstruction=face_gradient_reconstruction.reconstruction,
            axis_core_face_gradient_reconstruction=face_gradient_reconstruction,
            axis_core_cell_gradient_reconstruction=cell_reconstruction,
        )

    def build_axis_core_line_u_phi_preconditioner(
        self,
        phi_face_bc: LocalBoundaryFaceBC3D,
    ) -> AxisCoreLineUPreconditioner3D:
        """Precompute the invariant Cartesian coefficient solve for phi."""

        if not self.domain.axis_regular_axes[0]:
            raise ValueError(
                "axis-core-line-u requires a lower radial polar axis"
            )
        context = self._stencil_builder_context()
        reconstruction = context.axis_core_face_gradient_reconstruction
        if reconstruction is None:
            raise RuntimeError("axis-core face-gradient reconstruction is missing")
        homogeneous_face_bc = replace(
            phi_face_bc,
            value_x=jnp.zeros_like(phi_face_bc.value_x),
            value_y=jnp.zeros_like(phi_face_bc.value_y),
            value_z=jnp.zeros_like(phi_face_bc.value_z),
        )
        solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            stencil_builder_context=context,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            face_bc=homogeneous_face_bc,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            config=self.gmres_config,
        )

        def apply_A(values: jnp.ndarray) -> jnp.ndarray:
            return solver._apply_A(
                values,
                face_bc=homogeneous_face_bc,
                control_volume_boundary_bc=None,
                project_mean_zero=bool(self.gmres_config.project_mean_zero),
            )

        return build_axis_core_line_u_preconditioner(
            apply_A,
            reconstruction,
            self.domain,
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
        field_halo = inject_owned_field_to_halo(
            values_owned,
            self.domain.layout,
        )
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(field_halo, self.domain, face_bc)

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
            axis_regular_axes=self.axis_regular_axes,
        )

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
        Pe_conservative_stencil,
        pressure_conservative_stencil,
        phi_conservative_stencil,
        Te_conservative_stencil,
        Ti_conservative_stencil,
        vorticity_conservative_stencil,
    ):
        """Assemble the production curvature contribution for each equation.

        This is shared by the full RHS and the IMEX implicit RHS so the
        conservative operators, equation gating, scaling, and the
        upwind-equilibrium wall closure cannot drift between the two paths.
        ``Ti`` remains the only curvature contribution intentionally left in
        the explicit complement.
        """

        if self.curvature_scheme == "disabled":
            curvature_Pe = jnp.zeros_like(density)
            curvature_pressure = jnp.zeros_like(density)
            curvature_phi = jnp.zeros_like(density)
            curvature_Te = jnp.zeros_like(density)
            curvature_Ti = jnp.zeros_like(density)
            curvature_Ti_self = jnp.zeros_like(density)
        elif self.curvature_scheme == "conservative":
            assert self.curvature_face_coefficients is not None
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

            def curvature(conservative_stencil, scalar_face_bc):
                return self._conservative_curvature(
                    conservative_stencil,
                    scalar_face_bc,
                )

            def central_curvature(conservative_stencil, trace):
                return self._conservative_curvature(
                    conservative_stencil,
                    boundary_trace=trace,
                )

            if Pe_wall_bc is None:
                curvature_Pe = self._conservative_curvature(
                    Pe_conservative_stencil,
                    boundary_trace=operator_boundary.Pe,
                )
            else:
                curvature_Pe = self._conservative_curvature(
                    Pe_conservative_stencil, Pe_wall_bc
                )
            if pressure_wall_bc is None:
                curvature_pressure = central_curvature(
                    pressure_conservative_stencil, operator_boundary.pressure
                )
            else:
                curvature_pressure = curvature(
                    pressure_conservative_stencil, pressure_wall_bc
                )
            if phi_wall_bc is None:
                curvature_phi = central_curvature(
                    phi_conservative_stencil, operator_boundary.phi
                )
            else:
                curvature_phi = curvature(
                phi_conservative_stencil, phi_wall_bc
                )
            if Te_wall_bc is None:
                curvature_Te = central_curvature(
                    Te_conservative_stencil, operator_boundary.Te
                )
            else:
                curvature_Te = curvature(
                Te_conservative_stencil, Te_wall_bc
                )
            if Ti_wall_bc is None:
                curvature_Ti = central_curvature(
                    Ti_conservative_stencil, operator_boundary.Ti
                )
            else:
                curvature_Ti = curvature(
                Ti_conservative_stencil, Ti_wall_bc
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
                    curvature(Ti_squared_conservative_stencil, Ti2_wall_bc)
                    if Ti2_wall_bc is not None else
                    central_curvature(
                        Ti_squared_conservative_stencil,
                        operator_boundary.Ti_squared,
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
            else jnp.zeros_like(density)
        )
        curvature_Te_contribution = (
            self.curvature_scale * (4.0 * Te / (3.0 * bmag)) *
            (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
            if "Te" in self.curvature_equations
            else jnp.zeros_like(Te)
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
            curvature_Ti_contribution = jnp.zeros_like(Ti)
        curvature_vorticity_contribution = (
            self.curvature_scale * (2.0 * bmag / density_safe) * curvature_pressure
            if "vorticity" in self.curvature_equations
            else jnp.zeros_like(density)
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
        ti_face_for_curvature = None
        ti_squared_face_for_curvature = None
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
                ti_stencil, ti_face_for_curvature
            )
            ti_squared_curvature = self._conservative_curvature(
                ti_squared_stencil, ti_squared_face_for_curvature
            )
        else:
            ti_curvature = self._conservative_curvature(
                ti_stencil, boundary_trace=operator_boundary.Ti
            )
            ti_squared_curvature = self._conservative_curvature(
                ti_squared_stencil, boundary_trace=operator_boundary.Ti_squared
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

    def _fci_parallel_terms(
        self,
        *,
        state_halo: FciDrbEBState,
        face_bc: LocalFciDrbEBFaceBCBundle,
        operator_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        parallel_boundary: LocalFciDrbEBOperatorBoundaryBundle,
        context: StencilBuilderContext,
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
        q_data["vorticity_current"] = self._fci_prepare_flux_q(
            fields["current"][owned], operator_boundary.current, context
        )
        inverse_b_halo, inverse_b_forward, inverse_b_backward = (
            self._fci_prepare_inverse_b(face_bc, context)
        )

        def q_div(name: str) -> jnp.ndarray:
            q_halo, forward, backward = q_data[name]
            return local_parallel_q_flux_div_fci_op(
                q_halo,
                self.geometry,
                context=context,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )

        div_b = local_parallel_div_b_fci_from_q_op(
            inverse_b_halo,
            self.geometry,
            context=context,
            forward_remote_q_values=inverse_b_forward,
            backward_remote_q_values=inverse_b_backward,
        )

        def grad(name: str) -> jnp.ndarray:
            q_halo, forward, backward = q_data[name]
            return local_grad_parallel_op_fci_compatible_from_q(
                q_halo,
                self.geometry,
                context=context,
                field_owned=fields[name][owned],
                div_b=div_b,
                forward_remote_q_values=forward,
                backward_remote_q_values=backward,
            )

        result = {
            "parallel_div_b": div_b,
            "density_flux_div": q_div("density_flux"),
            "current_flux_div": q_div("current"),
            "vorticity_current_flux_div": q_div("vorticity_current"),
            "parallel_Vi_flux_div": q_div("Vi"),
            "Ve_flux_div": q_div("Ve"),
            "grad_Te": grad("Te"),
            "grad_Ti": grad("Ti"),
            "grad_Ve": grad("Ve"),
            "grad_Vi": grad("Vi"),
            "grad_phi": grad("phi"),
            "grad_Pe": grad("Pe"),
            "grad_pressure": grad("pressure"),
            "grad_current": grad("current"),
            "grad_vorticity": grad("vorticity"),
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
                result[f"{name}_parallel_diff"] = jnp.zeros(
                    self.geometry.owned_shape, dtype=jnp.float64
                )
                continue
            field_remote = self._fci_remote_values(fields[name], context)
            diffusivity_halo = jnp.ones_like(fields[name], dtype=jnp.float64)
            diffusivity_remote = self._fci_remote_values(diffusivity_halo, context)
            result[f"{name}_parallel_diff"] = jnp.asarray(
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
    ) -> dict[str, jnp.ndarray]:
        """Evaluate the pre-existing coordinate parallel stage operators."""

        operator_kwargs = dict(
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        density_flux_div = local_parallel_flux_div_op(
            density_flux_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.density_flux, **operator_kwargs
        )
        current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.current, **operator_kwargs
        )
        vorticity_current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain,
            boundary_trace=operator_boundary.current, **operator_kwargs
        )
        Vi_stencil = build_local_conservative_stencil_from_field(
            state_halo.Vi, self.geometry, context
        )
        parallel_Vi_flux_div = local_parallel_flux_div_op(
            Vi_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Vi, **operator_kwargs
        )
        Ve_flux_div = local_parallel_flux_div_op(
            Ve_stencil, self.geometry, self.domain,
            boundary_trace=parallel_boundary.Ve, **operator_kwargs
        )
        grad_Te = local_grad_parallel_op_conservative(
            Te_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Te, **operator_kwargs
        )
        grad_Ti = local_grad_parallel_op_conservative(
            Ti_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ti, **operator_kwargs
        )
        grad_Ve = local_grad_parallel_op_conservative(
            Ve_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Ve, **operator_kwargs
        )
        grad_Vi = local_grad_parallel_op_conservative(
            Vi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Vi, **operator_kwargs
        )
        grad_phi = local_grad_parallel_op_conservative(
            phi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.phi, **operator_kwargs
        )
        grad_Pe = local_grad_parallel_op_conservative(
            Pe_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.Pe, **operator_kwargs
        )
        grad_pressure = local_grad_parallel_op_conservative(
            pressure_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.pressure, **operator_kwargs
        )
        grad_current = local_grad_parallel_op_conservative(
            current_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=parallel_boundary.current, **operator_kwargs
        )
        grad_vorticity = local_grad_parallel_op_conservative(
            vorticity_stencil, self.geometry, self.domain, div_b=parallel_div_b,
            boundary_trace=operator_boundary.vorticity, **operator_kwargs
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
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo | AxisCoreReducedSolveInfo]:
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
        owned = self.domain.layout.owned_slices_cell
        phi_rhs = (
            jnp.asarray(self.parameters.tau, dtype=jnp.float64) * ti_laplacian
            - jnp.asarray(state_owned.vorticity, dtype=jnp.float64)
        )
        phi_lift = jnp.asarray(state_owned.phi, dtype=jnp.float64)
        solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            stencil_builder_context=context,
            axis_core_line_u_preconditioner=self.axis_core_line_u_preconditioner,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            face_bc=face_bc.phi,
            axis_regular_axes=self.axis_regular_axes,
            neumann_normal_scheme=self.neumann_normal_scheme,
            config=self.gmres_config,
        )
        if self.phi_solver_space == "axis-core-reduced":
            reduced_space = self.build_axis_core_reduced_phi_space()
            phi_result = solver.solve_axis_core_reduced(
                phi_rhs,
                guess_owned=state_owned.phi,
                phi_lift_owned=phi_lift,
                space=reduced_space,
                return_diagnostics=return_diagnostics,
            )
        else:
            phi_result = solver(
                phi_rhs,
                guess_owned=state_owned.phi,
                phi_lift_owned=phi_lift,
                return_diagnostics=return_diagnostics,
            )
        if return_diagnostics:
            phi_owned, info = phi_result
            return _mask_inactive_owned(phi_owned, self.geometry), info
        return _mask_inactive_owned(phi_result, self.geometry)

    def reconstruct_phi(
        self,
        state_owned: FciDrbEBState,
        *,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        face_bc = self._face_bcs(state_owned)
        state_halo = prepare_local_fci_drb_eb_state(
            state_owned,
            self.domain,
            face_bc=face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
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
        """Return the algebraic Boussinesq-polarization residual.

        The production phi solve uses ``-L_perp(phi) = tau*L_perp(Ti)-omega``.
        Keeping this residual in the RHS module makes the sign convention
        shared by the explicit RK4 and the future monolithic IMEX stages.
        ``Ti`` is an implicit differential variable in the six-field
        partition, so this residual is differentiated with respect to Ti as
        well as phi and vorticity.
        """

        face_bc = self._face_bcs(state_owned)
        state_halo = prepare_local_fci_drb_eb_state(
            state_owned,
            self.domain,
            face_bc=face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
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
        return _mask_inactive_owned(
            -phi_laplacian
            - jnp.asarray(self.parameters.tau, dtype=jnp.float64) * ti_laplacian
            + jnp.asarray(state_owned.vorticity, dtype=jnp.float64),
            self.geometry,
        )

    def evaluate_implicit_rhs(
        self,
        state_owned: FciDrbEBState,
        *,
        phi_owned: jnp.ndarray | None = None,
        include_curvature: bool = True,
    ) -> FciDrbEBImplicitState:
        """Evaluate the stiff electron/acoustic part of the EB RHS.

        The complement is intentionally left in :meth:`evaluate_explicit_rhs`:
        ion dynamics, ion self-advection, vorticity advection by ``Vi``, and
        the remaining explicit perpendicular transport/diffusion.  For the
        already-implicit fields, the production Poisson brackets are included
        here with their direct-stencil ghost semantics, as is electron
        parallel self-advection ``-Ve*grad_parallel(Ve)``.  The returned
        ``phi`` entry is zero because phi is constrained by
        :meth:`polarization_residual`, not evolved by an ODE.

        This is an exact term-level partition of :meth:`evaluate_stage` when
        the supplied potential is consistent with ``state_owned``.  Parallel
        diffusion/viscosity and electron collisions are included here so an
        enabled stiff coefficient is not accidentally left explicit.  The
        optional ``include_curvature=False`` mode is retained for controlled
        diagnostics and alternative approximate operators; production
        residuals and the current frozen-JVP preconditioner use the default.
        """

        face_bc = self._face_bcs(state_owned)
        state_halo_without_phi = prepare_local_fci_drb_eb_state(
            state_owned,
            self.domain,
            face_bc=face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        if phi_owned is None:
            phi_owned = state_owned.phi
        phi_owned = _mask_inactive_owned(
            jnp.asarray(phi_owned, dtype=jnp.float64), self.geometry
        )
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
            )
            if self.parallel_operator_scheme == "fci"
            else None
        )
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
            _scale_local_dirichlet_face_bc(ion_pressure_face_bc, self.parameters.tau),
            lambda left, right: left + right,
        )

        build_gradient = build_local_cell_gradient_from_field
        density_gradient = build_gradient(state_halo.density, self.geometry, context)
        Te_gradient = build_gradient(state_halo.Te, self.geometry, context)
        Ti_gradient = build_gradient(state_halo.Ti, self.geometry, context)
        Ve_gradient = build_gradient(state_halo.Ve, self.geometry, context)
        vorticity_gradient = build_gradient(
            state_halo.vorticity, self.geometry, context
        )
        phi_gradient = build_gradient(state_halo.phi, self.geometry, context)
        if self.curvature_scheme == "direct":
            Pe_gradient = build_gradient(
                state_halo.density * state_halo.Te, self.geometry, context
            )
            pressure_gradient = build_gradient(
                state_halo.density * state_halo.Te
                + self.parameters.tau * state_halo.density * state_halo.Ti,
                self.geometry,
                context,
            )
        else:
            Pe_gradient = None
            pressure_gradient = None

        density_flux = state_halo.density * state_halo.Ve
        current = state_halo.density * (state_halo.Vi - state_halo.Ve)
        electron_pressure = state_halo.density * state_halo.Te
        pressure = electron_pressure + self.parameters.tau * state_halo.density * state_halo.Ti
        density_flux_stencil = build_local_conservative_stencil_from_field(
            density_flux, self.geometry, context
        )
        current_stencil = build_local_conservative_stencil_from_field(
            current, self.geometry, context
        )
        Te_stencil = build_local_conservative_stencil_from_field(
            state_halo.Te, self.geometry, context
        )
        Ti_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ti, self.geometry, context
        )
        phi_stencil = build_local_conservative_stencil_from_field(
            state_halo.phi, self.geometry, context
        )
        Pe_stencil = build_local_conservative_stencil_from_field(
            electron_pressure, self.geometry, context
        )
        pressure_stencil = build_local_conservative_stencil_from_field(
            pressure, self.geometry, context
        )
        Ve_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ve, self.geometry, context
        )
        density_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.density, self.geometry, context
        )
        phi_conservative_stencil = phi_stencil
        Te_conservative_stencil = Te_stencil
        Ti_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ti, self.geometry, context
        )
        pressure_conservative_stencil = pressure_stencil
        vorticity_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.vorticity, self.geometry, context
        )
        if self.parallel_operator_scheme == "coordinate":
            unit_stencil = build_local_conservative_stencil_from_field(
                jnp.ones_like(state_halo.density, dtype=jnp.float64),
                self.geometry,
                context,
            )
            operator_kwargs = dict(
                regular_face_geometry=self.geometry.regular_face_geometry,
                axis_regular_axes=self.axis_regular_axes,
            )
            parallel_div_b = local_parallel_div_b_op(
                unit_stencil, self.geometry, self.domain, **operator_kwargs
            )
            density_flux_div = local_parallel_flux_div_op(
                density_flux_stencil, self.geometry, self.domain,
                boundary_trace=parallel_boundary.density_flux, **operator_kwargs
            )
            current_flux_div = local_parallel_flux_div_op(
                current_stencil, self.geometry, self.domain,
                boundary_trace=parallel_boundary.current, **operator_kwargs
            )
            vorticity_current_flux_div = local_parallel_flux_div_op(
                current_stencil, self.geometry, self.domain,
                boundary_trace=operator_boundary.current, **operator_kwargs
            )
            Vi_stencil = build_local_conservative_stencil_from_field(
                state_halo.Vi, self.geometry, context
            )
            parallel_Vi_flux_div = local_parallel_flux_div_op(
                Vi_stencil, self.geometry, self.domain,
                boundary_trace=parallel_boundary.Vi, **operator_kwargs
            )
            Ve_flux_div = local_parallel_flux_div_op(
                Ve_stencil, self.geometry, self.domain,
                boundary_trace=parallel_boundary.Ve, **operator_kwargs
            )
            grad_Te = local_grad_parallel_op_conservative(
                Te_stencil, self.geometry, self.domain, div_b=parallel_div_b,
                boundary_trace=parallel_boundary.Te, **operator_kwargs
            )
            grad_Ti = local_grad_parallel_op_conservative(
                Ti_stencil, self.geometry, self.domain, div_b=parallel_div_b,
                boundary_trace=parallel_boundary.Ti, **operator_kwargs
            )
            grad_phi = local_grad_parallel_op_conservative(
                phi_stencil, self.geometry, self.domain, div_b=parallel_div_b,
                boundary_trace=operator_boundary.phi, **operator_kwargs
            )
            grad_Pe = local_grad_parallel_op_conservative(
                Pe_stencil, self.geometry, self.domain, div_b=parallel_div_b,
                boundary_trace=parallel_boundary.Pe, **operator_kwargs
            )
            grad_Ve = local_grad_parallel_op_conservative(
                Ve_stencil, self.geometry, self.domain, div_b=parallel_div_b,
                boundary_trace=parallel_boundary.Ve, **operator_kwargs
            )
        else:
            parallel_div_b = fci_parallel_terms["parallel_div_b"]
            density_flux_div = fci_parallel_terms["density_flux_div"]
            current_flux_div = fci_parallel_terms["current_flux_div"]
            vorticity_current_flux_div = fci_parallel_terms["vorticity_current_flux_div"]
            parallel_Vi_flux_div = fci_parallel_terms["parallel_Vi_flux_div"]
            Ve_flux_div = fci_parallel_terms["Ve_flux_div"]
            grad_Te = fci_parallel_terms["grad_Te"]
            grad_Ti = fci_parallel_terms["grad_Ti"]
            grad_phi = fci_parallel_terms["grad_phi"]
            grad_Pe = fci_parallel_terms["grad_Pe"]
            grad_Ve = fci_parallel_terms["grad_Ve"]
        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ti = jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
        density_safe = jnp.maximum(density, 1.0e-30)
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        tau = jnp.asarray(self.parameters.tau, dtype=jnp.float64)
        mi_over_me = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)
        Ve_nu = jnp.asarray(self.parameters.Ve_nu, dtype=jnp.float64)
        current_owned = density * (Vi - Ve)

        poisson_density = self._poisson_bracket_over_B(
            phi_gradient,
            density_gradient,
            phi_conservative_stencil,
            density_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.density,
        )
        poisson_Te = self._poisson_bracket_over_B(
            phi_gradient,
            Te_gradient,
            phi_conservative_stencil,
            Te_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Te,
        )
        poisson_Ti = self._poisson_bracket_over_B(
            phi_gradient,
            Ti_gradient,
            phi_conservative_stencil,
            Ti_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ti,
        )
        poisson_Ve = self._poisson_bracket_over_B(
            phi_gradient,
            Ve_gradient,
            phi_conservative_stencil,
            Ve_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ve,
        )
        poisson_vorticity = self._poisson_bracket_over_B(
            phi_gradient,
            vorticity_gradient,
            phi_conservative_stencil,
            vorticity_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.vorticity,
        )

        if self.parallel_operator_scheme == "coordinate":
            density_parallel_diff = self._field_parallel_diffusion(
                state_halo.density, face_bc.density, self.parameters.density_D_parallel
            )
            Te_parallel_diff = self._field_parallel_diffusion(
                state_halo.Te, face_bc.Te, self.parameters.electron_temperature_chi_parallel
            )
            Ti_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ti, face_bc.Ti, self.parameters.ion_temperature_chi_parallel
            )
            Ve_parallel_diff = self._field_parallel_diffusion(
                state_halo.Ve, face_bc.Ve, self.parameters.Ve_parallel_viscosity
            )
            vorticity_parallel_diff = self._field_parallel_diffusion(
                state_halo.vorticity, face_bc.vorticity, self.parameters.vorticity_D_parallel
            )
        else:
            density_parallel_diff = fci_parallel_terms["density_parallel_diff"]
            Te_parallel_diff = fci_parallel_terms["Te_parallel_diff"]
            Ti_parallel_diff = fci_parallel_terms["Ti_parallel_diff"]
            Ve_parallel_diff = fci_parallel_terms["Ve_parallel_diff"]
            vorticity_parallel_diff = fci_parallel_terms["vorticity_parallel_diff"]
        if include_curvature:
            (
                curvature_density_contribution,
                curvature_Te_contribution,
                curvature_Ti_contribution,
                curvature_vorticity_contribution,
            ) = self._curvature_rhs_contributions(
                state_halo=state_halo,
                face_bc=face_bc,
                context=context,
                density=density,
                Te=Te,
                Ti=jnp.asarray(
                    state_halo.Ti[self.domain.layout.owned_slices_cell],
                    dtype=jnp.float64,
                ),
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
                Pe_conservative_stencil=Pe_stencil,
                pressure_conservative_stencil=pressure_stencil,
                phi_conservative_stencil=phi_conservative_stencil,
                Te_conservative_stencil=Te_conservative_stencil,
                Ti_conservative_stencil=Ti_conservative_stencil,
                vorticity_conservative_stencil=vorticity_conservative_stencil,
            )
        else:
            curvature_density_contribution = jnp.zeros_like(density)
            curvature_Te_contribution = jnp.zeros_like(Te)
            curvature_Ti_contribution = jnp.zeros_like(Ti)
            curvature_vorticity_contribution = jnp.zeros_like(density)
        return _mask_local_eb_state_inactive(
            FciDrbEBImplicitState(
                density=(
                    -(poisson_density / rho_star)
                    - density_flux_div
                    + curvature_density_contribution
                    + density_parallel_diff
                ),
                phi=jnp.zeros_like(phi_owned),
                Te=(
                    -(poisson_Te / rho_star)
                    -Ve * grad_Te
                    + curvature_Te_contribution
                    + (2.0 * Te / (3.0 * density_safe))
                    * (0.71 * current_flux_div - density * Ve_flux_div)
                    + Te_parallel_diff
                ),
                Ti=(
                    -(poisson_Ti / rho_star)
                    - Vi * grad_Ti
                    + curvature_Ti_contribution
                    + (2.0 * Ti / (3.0 * density_safe))
                    * (current_flux_div - density * parallel_Vi_flux_div)
                    + Ti_parallel_diff
                ),
                Ve=(
                    -(poisson_Ve / rho_star)
                    - Ve * grad_Ve
                    + mi_over_me * Ve_nu * current_owned
                    + mi_over_me * grad_phi
                    - mi_over_me * grad_Pe / density_safe
                    - 0.71 * mi_over_me * grad_Te
                    + Ve_parallel_diff
                ),
                vorticity=(
                    -(poisson_vorticity / rho_star)
                    + (bmag * bmag / density_safe) * vorticity_current_flux_div
                    + curvature_vorticity_contribution
                    + vorticity_parallel_diff
                ),
            ),
            self.geometry,
        )

    def evaluate_imex_rhs(
        self,
        state_owned: FciDrbEBState,
        source_owned: FciDrbEBState | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> tuple[FciDrbEBState, FciDrbEBImplicitState]:
        """Return the explicit and implicit RHS terms for one consistent state.

        This intentionally derives the complement from the production full
        RHS.  It is more work than a hand-maintained duplicate expression but
        makes the additive-split invariant robust while the IMEX path is being
        introduced.  A future fused stage evaluator can share the intermediate
        stencils without changing this public contract.  IMEX steppers should
        call this paired method rather than the two individual evaluators so
        the full production RHS is formed only once per explicit stage.
        """

        if phi_owned is None:
            phi_owned = state_owned.phi
        full = self.evaluate_stage(
            state_owned, source_owned, phi_owned=phi_owned
        )
        implicit = self.evaluate_implicit_rhs(state_owned, phi_owned=phi_owned)
        explicit = _mask_local_eb_state_inactive(
            FciDrbEBState(
                density=full.density - implicit.density,
                phi=jnp.zeros_like(phi_owned),
                Te=full.Te - implicit.Te,
                Ti=full.Ti - implicit.Ti,
                Vi=full.Vi,
                Ve=full.Ve - implicit.Ve,
                vorticity=full.vorticity - implicit.vorticity,
            ),
            self.geometry,
        )
        return explicit, implicit

    def evaluate_explicit_rhs(
        self,
        state_owned: FciDrbEBState,
        source_owned: FciDrbEBState | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
    ) -> FciDrbEBState:
        """Evaluate the nonstiff complement of :meth:`evaluate_implicit_rhs`."""

        explicit, _ = self.evaluate_imex_rhs(
            state_owned, source_owned, phi_owned=phi_owned
        )
        return explicit

    def implicit_stage_residual(
        self,
        implicit_stage: FciDrbEBImplicitState,
        implicit_predictor: FciDrbEBImplicitState,
        known_explicit_state: FciDrbEBState,
        *,
        dt_gamma: float | jnp.ndarray,
    ) -> FciDrbEBImplicitState:
        """Return the six-field DIRK stage residual for a monolithic IMEX solve.

        ``known_explicit_state`` supplies the stage-known explicit ion velocity
        ``Vi``.  The returned phi component is algebraic; it is deliberately
        not multiplied by ``dt_gamma``.  ``Ti`` is a differential unknown and
        its residual includes the production implicit Ti terms.
        """

        stage = eb_state_with_implicit_state(known_explicit_state, implicit_stage)
        implicit_rhs = self.evaluate_implicit_rhs(stage, phi_owned=implicit_stage.phi)
        dt_gamma = jnp.asarray(dt_gamma, dtype=jnp.float64)
        return _mask_local_eb_state_inactive(
            FciDrbEBImplicitState(
                density=implicit_stage.density - implicit_predictor.density - dt_gamma * implicit_rhs.density,
                phi=self.polarization_residual(stage, phi_owned=implicit_stage.phi),
                Te=implicit_stage.Te - implicit_predictor.Te - dt_gamma * implicit_rhs.Te,
                Ti=implicit_stage.Ti - implicit_predictor.Ti - dt_gamma * implicit_rhs.Ti,
                Ve=implicit_stage.Ve - implicit_predictor.Ve - dt_gamma * implicit_rhs.Ve,
                vorticity=implicit_stage.vorticity - implicit_predictor.vorticity - dt_gamma * implicit_rhs.vorticity,
            ),
            self.geometry,
        )

    def evaluate_stage(
        self,
        state_owned: FciDrbEBState,
        source_owned: FciDrbEBState | None = None,
        *,
        phi_owned: jnp.ndarray | None = None,
        return_term_diagnostics: bool = False,
        return_term_fields: bool = False,
    ) -> FciDrbEBState | tuple[FciDrbEBState, jnp.ndarray]:
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
        """

        if return_term_diagnostics and return_term_fields:
            raise ValueError(
                "return_term_diagnostics and return_term_fields are mutually exclusive"
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
        source_owned = _mask_local_eb_state_inactive(source_owned, self.geometry)
        face_bc = self._face_bcs(state_owned)
        state_halo_without_phi = prepare_local_fci_drb_eb_state(
            state_owned,
            self.domain,
            face_bc=face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
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
            )
            if self.parallel_operator_scheme == "fci"
            else None
        )
        # These state halos have already been closed with each field's
        # physical face BC.  Preserve those ghost values in the cell
        # gradients.  The one-sided physical builder is for
        # intermediate fields that do not have a physical ghost closure; using
        # it here silently discards the supplied Dirichlet/Neumann BCs.
        build_gradient = build_local_cell_gradient_from_field
        density_gradient = build_gradient(state_halo.density, self.geometry, context)
        Te_gradient = build_gradient(state_halo.Te, self.geometry, context)
        Ti_gradient = build_gradient(state_halo.Ti, self.geometry, context)
        Vi_gradient = build_gradient(state_halo.Vi, self.geometry, context)
        Ve_gradient = build_gradient(state_halo.Ve, self.geometry, context)
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
            state_halo.Ve,
            self.geometry,
            context,
        )
        Vi_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Vi,
            self.geometry,
            context,
        )
        Pe_halo = state_halo.density * state_halo.Te
        pressure_halo = (
            Pe_halo
            + self.parameters.tau * state_halo.density * state_halo.Ti
        )
        current_halo = state_halo.density * (state_halo.Vi - state_halo.Ve)
        density_flux_halo = state_halo.density * state_halo.Ve

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
            )
        else:
            parallel_div_b = fci_parallel_terms["parallel_div_b"]

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Ti = jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
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
            state_halo.Vi,
            face_bc.Vi,
            self.parameters.Vi_D_perp,
        )
        Ve_diff = self._field_perp_diffusion(
            state_halo.Ve,
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

        if bool(self.diffusion_only):
            return _mask_local_eb_state_inactive(FciDrbEBState(
                density=density_diff,
                phi=jnp.zeros_like(phi_owned),
                Te=Te_diff + Te_parallel_diff,
                Ti=Ti_diff + Ti_parallel_diff,
                Vi=Vi_diff + Vi_parallel_diff,
                Ve=Ve_diff + Ve_parallel_diff,
                vorticity=vorticity_diff + vorticity_parallel_diff,
            ), self.geometry)

        poisson_density = self._poisson_bracket_over_B(
            phi_gradient,
            density_gradient,
            phi_conservative_stencil,
            density_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.density,
        )
        poisson_Te = self._poisson_bracket_over_B(
            phi_gradient,
            Te_gradient,
            phi_conservative_stencil,
            Te_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Te,
        )
        poisson_Ti = self._poisson_bracket_over_B(
            phi_gradient,
            Ti_gradient,
            phi_conservative_stencil,
            Ti_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ti,
        )
        poisson_Vi = self._poisson_bracket_over_B(
            phi_gradient,
            Vi_gradient,
            phi_conservative_stencil,
            Vi_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Vi,
        )
        poisson_Ve = self._poisson_bracket_over_B(
            phi_gradient,
            Ve_gradient,
            phi_conservative_stencil,
            Ve_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.Ve,
        )
        poisson_vorticity = self._poisson_bracket_over_B(
            phi_gradient,
            vorticity_gradient,
            phi_conservative_stencil,
            vorticity_conservative_stencil,
            f_boundary_trace=operator_boundary.phi,
            g_boundary_trace=operator_boundary.vorticity,
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

        (
            curvature_density_contribution,
            curvature_Te_contribution,
            curvature_Ti_contribution,
            curvature_vorticity_contribution,
        ) = self._curvature_rhs_contributions(
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
            Pe_conservative_stencil=Pe_conservative_stencil,
            pressure_conservative_stencil=pressure_conservative_stencil,
            phi_conservative_stencil=phi_conservative_stencil,
            Te_conservative_stencil=Te_conservative_stencil,
            Ti_conservative_stencil=Ti_conservative_stencil,
            vorticity_conservative_stencil=vorticity_conservative_stencil,
        )

        density_rhs = (
            -(poisson_density / rho_star)
            - parallel_density_flux_divergence
            + curvature_density_contribution
            + density_diff
            + density_parallel_diff
        )
        Te_rhs = (
            -(poisson_Te / rho_star)
            - Ve * grad_parallel_Te
            + curvature_Te_contribution
            + (2.0 * Te / (3.0 * density_safe))
            * (0.71 * parallel_current_flux_divergence - density * parallel_Ve_flux_divergence)
            + Te_diff
            + Te_parallel_diff
        )
        Ti_rhs = (
            -(poisson_Ti / rho_star)
            - Vi * grad_parallel_Ti
            + curvature_Ti_contribution
            + (2.0 * Ti / (3.0 * density_safe))
            * (parallel_current_flux_divergence - density * parallel_Vi_flux_divergence)
            + Ti_diff
            + Ti_parallel_diff
        )
        Vi_rhs = (
            -(poisson_Vi / rho_star)
            - Vi * grad_parallel_Vi
            - grad_parallel_pressure / density_safe
            + Vi_diff
            + Vi_parallel_diff
        )
        Ve_poisson_term = -(poisson_Ve / rho_star)
        Ve_self_advection_term = -Ve * grad_parallel_Ve
        Ve_collision_term = mi_over_me * Ve_nu * current
        Ve_electrostatic_term = mi_over_me * grad_parallel_phi
        Ve_pressure_term = -mi_over_me * grad_parallel_Pe / density_safe
        Ve_thermal_force_term = -0.71 * mi_over_me * grad_parallel_Te
        Ve_rhs = (
            Ve_poisson_term
            + Ve_self_advection_term
            + Ve_collision_term
            + Ve_electrostatic_term
            + Ve_pressure_term
            + Ve_thermal_force_term
            + Ve_diff
            + Ve_parallel_diff
        )
        vorticity_rhs = (
            -(poisson_vorticity / rho_star)
            - Vi * grad_parallel_vorticity
            + (bmag * bmag / density_safe) * vorticity_current_flux_divergence
            + curvature_vorticity_contribution
            + vorticity_diff
            + vorticity_parallel_diff
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
                -parallel_density_flux_divergence
                + density_parallel_diff
            )
            Te_rhs = (
                -Ve * grad_parallel_Te
                + (2.0 * Te / (3.0 * density_safe))
                * (
                    0.71 * parallel_current_flux_divergence
                    - density * parallel_Ve_flux_divergence
                )
                + Te_parallel_diff
            )
            Ti_rhs = (
                -Vi * grad_parallel_Ti
                + (2.0 * Ti / (3.0 * density_safe))
                * (
                    parallel_current_flux_divergence
                    - density * parallel_Vi_flux_divergence
                )
                + Ti_parallel_diff
            )
            Vi_rhs = (
                -Vi * grad_parallel_Vi
                - grad_parallel_pressure / density_safe
                + Vi_parallel_diff
            )
            Ve_rhs = (
                Ve_self_advection_term
                + Ve_collision_term
                + Ve_electrostatic_term
                + Ve_pressure_term
                + Ve_thermal_force_term
                + Ve_parallel_diff
            )
            vorticity_rhs = (
                -Vi * grad_parallel_vorticity
                + (bmag * bmag / density_safe)
                * vorticity_current_flux_divergence
                + vorticity_parallel_diff
            )
            result = _mask_local_eb_state_inactive(FciDrbEBState(
                density=density_rhs,
                phi=jnp.zeros_like(phi_owned),
                Te=Te_rhs,
                Ti=Ti_rhs,
                Vi=Vi_rhs,
                Ve=Ve_rhs,
                vorticity=vorticity_rhs,
            ), self.geometry)
            if return_term_diagnostics or return_term_fields:
                Ve_terms = jnp.stack(
                    (
                        jnp.zeros_like(Ve),
                        Ve_self_advection_term,
                        Ve_collision_term,
                        Ve_electrostatic_term,
                        Ve_pressure_term,
                        Ve_thermal_force_term,
                        jnp.zeros_like(Ve),
                        Ve_parallel_diff,
                    ),
                    axis=0,
                )
                if return_term_fields:
                    return result, Ve_terms
                return result, jnp.max(
                    jnp.abs(Ve_terms),
                    axis=tuple(range(1, Ve_terms.ndim)),
                )
            return result
        result = _mask_local_eb_state_inactive(FciDrbEBState(
            density=density_rhs + source_owned.density,
            phi=jnp.zeros_like(phi_owned),
            Te=Te_rhs + source_owned.Te,
            Ti=Ti_rhs + source_owned.Ti,
            Vi=Vi_rhs + source_owned.Vi,
            Ve=Ve_rhs + source_owned.Ve,
            vorticity=vorticity_rhs + source_owned.vorticity,
        ), self.geometry)
        if return_term_diagnostics or return_term_fields:
            Ve_terms = jnp.stack(
                (
                    Ve_poisson_term,
                    Ve_self_advection_term,
                    Ve_collision_term,
                    Ve_electrostatic_term,
                    Ve_pressure_term,
                    Ve_thermal_force_term,
                    Ve_diff,
                    Ve_parallel_diff,
                ),
                axis=0,
            )
            if return_term_fields:
                return result, Ve_terms
            return result, jnp.max(
                jnp.abs(Ve_terms),
                axis=tuple(range(1, Ve_terms.ndim)),
            )
        return result
