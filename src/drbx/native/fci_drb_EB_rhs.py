from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import (
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from .fci_model import FciModelState
from .fci_model import inject_owned_field_to_halo, inject_owned_state_to_halo
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NONE,
    ConservativeStencil3D,
    LocalBoundaryFaceBC3D,
)
from .fci_halo import (
    HaloExchange3D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from .fci_operators import (
    LocalPerpLaplacianInverseSolver,
    build_solvax_perp_laplacian_preconditioner,
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_conservative,
    local_parallel_div_b_op,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_perp_laplacian_conservative_op,
    local_curvature_conservative_op,
    _local_axis_face_values_from_stencil,
    local_poisson_bracket_op,
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbEBImplicitState(FciModelState):
    """The five algebraic/differential unknowns of an EB IMEX stage.

    ``Ti`` and ``Vi`` deliberately do not appear here: the first ARK split
    keeps ion parallel dynamics explicit, so their stage values are known
    while the electron/acoustic--polarization block is solved implicitly.
    The field order is stable and is suitable for ``jax.linearize`` and
    sharded matrix-free Newton--Krylov methods.
    """

    density: jax.Array
    phi: jax.Array
    Te: jax.Array
    Ve: jax.Array
    vorticity: jax.Array


def implicit_state_from_eb_state(state: FciDrbEBState) -> FciDrbEBImplicitState:
    """Extract the implicit ARK stage variables from a full EB state."""

    return FciDrbEBImplicitState(
        density=state.density,
        phi=state.phi,
        Te=state.Te,
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
    ``(n, Te, Ve, omega)`` leaves untouched and applies the established local
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
        return model.evaluate_implicit_rhs(stage, phi_owned=implicit_state.phi)

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
    for axis, name, axis_stencil in zip((0, 1, 2), ("x", "y", "z"), (stencil.x, stencil.y, stencil.z)):
        value = _local_axis_face_values_from_stencil(axis_stencil, axis=axis)
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
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    curvature_face_coefficients: LocalCurvatureFaceCoefficients3D | None = None
    upwind_equilibrium_wall_projectors: UpwindEquilibriumWallProjectors | None = None
    curvature_scheme: str = "direct"
    curvature_inflow_closure: str = "central"
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

    def __post_init__(self) -> None:
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
        scalar_face_bc: LocalBoundaryFaceBC3D,
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
            face_bc=scalar_face_bc,
            axis_regular_axes=self.axis_regular_axes,
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
        ti_halo = self._prepare_scalar_halo(
            jnp.asarray(state_owned.Ti, dtype=jnp.float64),
            face_bc.Ti,
        )
        ti_squared_face_bc = _binary_local_dirichlet_face_bc(
            face_bc.Ti,
            face_bc.Ti,
            lambda left, right: left * right,
        )
        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
        )
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
        ti_face_for_curvature = face_bc.Ti
        ti_squared_face_for_curvature = ti_squared_face_bc
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
        curvature = self._conservative_curvature
        ti_curvature = curvature(ti_stencil, ti_face_for_curvature)
        ti_squared_curvature = curvature(ti_squared_stencil, ti_squared_face_for_curvature)
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
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
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
        )

    def _field_parallel_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
    ) -> jnp.ndarray:
        if float(coefficient) == 0.0:
            return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
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
        )

    def _reconstruct_phi_from_prepared(
        self,
        state_owned: FciDrbEBState,
        state_halo: FciDrbEBState,
        face_bc: LocalFciDrbEBFaceBCBundle,
        *,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
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
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            face_bc=face_bc.phi,
            axis_regular_axes=self.axis_regular_axes,
            config=self.gmres_config,
        )
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
        ``Ti`` is stage-known in the initial IMEX partition.
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
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
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
        )
        ti_laplacian = local_perp_laplacian_conservative_op(
            ti_conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
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
    ) -> FciDrbEBImplicitState:
        """Evaluate the stiff electron/acoustic part of the EB RHS.

        The complement is intentionally left in :meth:`evaluate_explicit_rhs`:
        curvature, Poisson brackets, perpendicular transport, ion dynamics,
        and both ion/electron self-advection.  The returned ``phi`` entry is
        zero because phi is constrained by :meth:`polarization_residual`, not
        evolved by an ODE.

        This is an exact term-level partition of :meth:`evaluate_stage` when
        the supplied potential is consistent with ``state_owned``.  Parallel
        diffusion/viscosity and electron collisions are included here so an
        enabled stiff coefficient is not accidentally left explicit.
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
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)

        density_flux = state_halo.density * state_halo.Ve
        current = state_halo.density * (state_halo.Vi - state_halo.Ve)
        electron_pressure = state_halo.density * state_halo.Te
        density_flux_stencil = build_local_conservative_stencil_from_field(
            density_flux, self.geometry, context
        )
        current_stencil = build_local_conservative_stencil_from_field(
            current, self.geometry, context
        )
        Te_stencil = build_local_conservative_stencil_from_field(
            state_halo.Te, self.geometry, context
        )
        phi_stencil = build_local_conservative_stencil_from_field(
            state_halo.phi, self.geometry, context
        )
        Pe_stencil = build_local_conservative_stencil_from_field(
            electron_pressure, self.geometry, context
        )
        Ve_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ve, self.geometry, context
        )
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
            density_flux_stencil, self.geometry, self.domain, **operator_kwargs
        )
        current_flux_div = local_parallel_flux_div_op(
            current_stencil, self.geometry, self.domain, **operator_kwargs
        )
        Ve_flux_div = local_parallel_flux_div_op(
            Ve_stencil, self.geometry, self.domain, **operator_kwargs
        )
        grad_Te = local_grad_parallel_op_conservative(
            Te_stencil, self.geometry, self.domain, div_b=parallel_div_b, **operator_kwargs
        )
        grad_phi = local_grad_parallel_op_conservative(
            phi_stencil, self.geometry, self.domain, div_b=parallel_div_b, **operator_kwargs
        )
        grad_Pe = local_grad_parallel_op_conservative(
            Pe_stencil, self.geometry, self.domain, div_b=parallel_div_b, **operator_kwargs
        )

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
        density_safe = jnp.maximum(density, 1.0e-30)
        bmag = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        mi_over_me = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)
        Ve_nu = jnp.asarray(self.parameters.Ve_nu, dtype=jnp.float64)
        current_owned = density * (Vi - Ve)

        density_parallel_diff = self._field_parallel_diffusion(
            state_halo.density, face_bc.density, self.parameters.density_D_parallel
        )
        Te_parallel_diff = self._field_parallel_diffusion(
            state_halo.Te, face_bc.Te, self.parameters.electron_temperature_chi_parallel
        )
        Ve_parallel_diff = self._field_parallel_diffusion(
            state_halo.Ve, face_bc.Ve, self.parameters.Ve_parallel_viscosity
        )
        vorticity_parallel_diff = self._field_parallel_diffusion(
            state_halo.vorticity, face_bc.vorticity, self.parameters.vorticity_D_parallel
        )
        return _mask_local_eb_state_inactive(
            FciDrbEBImplicitState(
                density=-density_flux_div + density_parallel_diff,
                phi=jnp.zeros_like(phi_owned),
                Te=(
                    -Ve * grad_Te
                    + (2.0 * Te / (3.0 * density_safe))
                    * (0.71 * current_flux_div - density * Ve_flux_div)
                    + Te_parallel_diff
                ),
                Ve=(
                    mi_over_me * Ve_nu * current_owned
                    + mi_over_me * grad_phi
                    - mi_over_me * grad_Pe / density_safe
                    - 0.71 * mi_over_me * grad_Te
                    + Ve_parallel_diff
                ),
                vorticity=(
                    (bmag * bmag / density_safe) * current_flux_div
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
                Ti=full.Ti,
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
        """Return the five-field DIRK stage residual for a monolithic IMEX solve.

        ``known_explicit_state`` supplies the already explicit ARK stage values
        of ``Ti`` and ``Vi``.  The returned phi component is algebraic; it is
        deliberately not multiplied by ``dt_gamma``.
        """

        stage = eb_state_with_implicit_state(known_explicit_state, implicit_stage)
        implicit_rhs = self.evaluate_implicit_rhs(stage, phi_owned=implicit_stage.phi)
        dt_gamma = jnp.asarray(dt_gamma, dtype=jnp.float64)
        return _mask_local_eb_state_inactive(
            FciDrbEBImplicitState(
                density=implicit_stage.density - implicit_predictor.density - dt_gamma * implicit_rhs.density,
                phi=self.polarization_residual(stage, phi_owned=implicit_stage.phi),
                Te=implicit_stage.Te - implicit_predictor.Te - dt_gamma * implicit_rhs.Te,
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
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        # These state halos have already been closed with each field's
        # physical face BC.  Preserve those ghost values in the direct
        # coordinate derivatives.  The one-sided physical builder is for
        # intermediate fields that do not have a physical ghost closure; using
        # it here silently discards the supplied Dirichlet/Neumann BCs.
        direct = build_local_stencil_from_field

        density_stencil = direct(state_halo.density, self.geometry, context)
        Te_stencil = direct(state_halo.Te, self.geometry, context)
        Ti_stencil = direct(state_halo.Ti, self.geometry, context)
        Vi_stencil = direct(state_halo.Vi, self.geometry, context)
        Ve_stencil = direct(state_halo.Ve, self.geometry, context)
        vorticity_stencil = direct(state_halo.vorticity, self.geometry, context)
        phi_stencil = direct(state_halo.phi, self.geometry, context)

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

        Pe_stencil = direct(Pe_halo, self.geometry, context)
        pressure_stencil = direct(pressure_halo, self.geometry, context)
        current_stencil = direct(current_halo, self.geometry, context)
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
        density_parallel_diff = self._field_parallel_diffusion(
            state_halo.density,
            face_bc.density,
            self.parameters.density_D_parallel,
        )
        Te_diff = self._field_perp_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_D_perp,
        )
        Te_parallel_diff = self._field_parallel_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_chi_parallel,
        )
        Ti_diff = self._field_perp_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_D_perp,
        )
        Ti_parallel_diff = self._field_parallel_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_chi_parallel,
        )
        Vi_diff = self._field_perp_diffusion(
            state_halo.Vi,
            face_bc.Vi,
            self.parameters.Vi_D_perp,
        )
        Vi_parallel_diff = self._field_parallel_diffusion(
            state_halo.Vi,
            face_bc.Vi,
            self.parameters.Vi_parallel_viscosity,
        )
        Ve_diff = self._field_perp_diffusion(
            state_halo.Ve,
            face_bc.Ve,
            self.parameters.Ve_D_perp,
        )
        Ve_parallel_diff = self._field_parallel_diffusion(
            state_halo.Ve,
            face_bc.Ve,
            self.parameters.Ve_parallel_viscosity,
        )
        vorticity_diff = self._field_perp_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_perp,
        )
        vorticity_parallel_diff = self._field_parallel_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_parallel,
        )

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

        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        poisson_Te = local_poisson_bracket_op(
            phi_stencil,
            Te_stencil,
            self.geometry,
        )
        poisson_Ti = local_poisson_bracket_op(
            phi_stencil,
            Ti_stencil,
            self.geometry,
        )
        poisson_Vi = local_poisson_bracket_op(
            phi_stencil,
            Vi_stencil,
            self.geometry,
        )
        poisson_Ve = local_poisson_bracket_op(
            phi_stencil,
            Ve_stencil,
            self.geometry,
        )
        poisson_vorticity = local_poisson_bracket_op(
            phi_stencil,
            vorticity_stencil,
            self.geometry,
        )

        if self.curvature_scheme == "disabled":
            curvature_Pe = jnp.zeros_like(density)
            curvature_pressure = jnp.zeros_like(density)
            curvature_phi = jnp.zeros_like(density)
            curvature_Te = jnp.zeros_like(density)
            curvature_Ti = jnp.zeros_like(density)
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
                _, Te_wall_bc, Ti_wall_bc, Pe_wall_bc, pressure_wall_bc, Ti2_wall_bc = characteristic_bcs
                phi_wall_bc = _dirichlet_face_bc_from_values(
                    _wall_candidate_values(phi_conservative_stencil, face_bc.phi),
                    self.domain.layout,
                    (face_bc.phi.mask_x, face_bc.phi.mask_y, face_bc.phi.mask_z),
                )
            else:
                Te_wall_bc = Ti_wall_bc = Pe_wall_bc = pressure_wall_bc = Ti2_wall_bc = None
                phi_wall_bc = None

            def curvature(conservative_stencil, scalar_face_bc):
                return self._conservative_curvature(
                    conservative_stencil,
                    scalar_face_bc,
                )

            curvature_Pe = self._conservative_curvature(
                Pe_conservative_stencil,
                Pe_face_bc if Pe_wall_bc is None else Pe_wall_bc,
            )
            curvature_pressure = curvature(
                pressure_conservative_stencil,
                pressure_face_bc if pressure_wall_bc is None else pressure_wall_bc,
            )
            curvature_phi = curvature(phi_conservative_stencil, face_bc.phi if phi_wall_bc is None else phi_wall_bc)
            curvature_Te = curvature(Te_conservative_stencil, face_bc.Te if Te_wall_bc is None else Te_wall_bc)
            curvature_Ti = curvature(Ti_conservative_stencil, face_bc.Ti if Ti_wall_bc is None else Ti_wall_bc)
            if self.ion_temperature_curvature_self_form == "flux":
                Ti_squared_face_bc = _binary_local_dirichlet_face_bc(
                    face_bc.Ti,
                    face_bc.Ti,
                    lambda left, right: left * right,
                )
                if Ti2_wall_bc is not None:
                    Ti_squared_face_bc = Ti2_wall_bc
                Ti_squared_conservative_stencil = (
                    build_local_conservative_stencil_from_field(
                        state_halo.Ti * state_halo.Ti,
                        self.geometry,
                        context,
                    )
                )
                curvature_Ti_self = self._conservative_curvature(
                    Ti_squared_conservative_stencil,
                    Ti_squared_face_bc,
                )
            else:
                curvature_Ti_self = curvature_Ti
        else:
            curvature_coefficients = self.curvature_coefficients_owned
            assert curvature_coefficients is not None
            curvature_Pe = local_curvature_op(
                Pe_stencil,
                self.geometry,
                curvature_coefficients=curvature_coefficients,
            )
            curvature_pressure = local_curvature_op(
                pressure_stencil,
                self.geometry,
                curvature_coefficients=curvature_coefficients,
            )
            curvature_phi = local_curvature_op(
                phi_stencil,
                self.geometry,
                curvature_coefficients=curvature_coefficients,
            )
            curvature_Te = local_curvature_op(
                Te_stencil,
                self.geometry,
                curvature_coefficients=curvature_coefficients,
            )
            curvature_Ti = local_curvature_op(
                Ti_stencil,
                self.geometry,
                curvature_coefficients=curvature_coefficients,
            )

        parallel_density_flux_divergence = local_parallel_flux_div_op(
            density_flux_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        parallel_current_flux_divergence = local_parallel_flux_div_op(
            current_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        parallel_Ve_flux_divergence = local_parallel_flux_div_op(
            Ve_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        parallel_Vi_flux_divergence = local_parallel_flux_div_op(
            Vi_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_Te = local_grad_parallel_op_conservative(
            Te_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_Ti = local_grad_parallel_op_conservative(
            Ti_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_Ve = local_grad_parallel_op_conservative(
            Ve_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_Vi = local_grad_parallel_op_conservative(
            Vi_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_phi = local_grad_parallel_op_conservative(
            phi_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_Pe = local_grad_parallel_op_conservative(
            Pe_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_pressure = local_grad_parallel_op_conservative(
            pressure_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_current = local_grad_parallel_op_conservative(
            current_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )
        grad_parallel_vorticity = local_grad_parallel_op_conservative(
            vorticity_conservative_stencil,
            self.geometry,
            self.domain,
            div_b=parallel_div_b,
            regular_face_geometry=self.geometry.regular_face_geometry,
            axis_regular_axes=self.axis_regular_axes,
        )

        curvature_density_contribution = (
            self.curvature_scale
            * (2.0 / bmag)
            * (curvature_Pe - density * curvature_phi)
            if "density" in self.curvature_equations
            else jnp.zeros_like(density)
        )
        curvature_Te_contribution = (
            self.curvature_scale
            * (4.0 * Te / (3.0 * bmag))
            * (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
            if "Te" in self.curvature_equations
            else jnp.zeros_like(Te)
        )
        if "Ti" in self.curvature_equations:
            if self.ion_temperature_curvature_self_form == "flux":
                curvature_Ti_contribution = (
                    self.curvature_scale
                    * (
                        (4.0 * Ti / (3.0 * bmag))
                        * (curvature_Pe / density_safe - curvature_phi)
                        - (5.0 * tau / (3.0 * bmag)) * curvature_Ti_self
                    )
                )
            else:
                curvature_Ti_contribution = (
                    self.curvature_scale
                    * (4.0 * Ti / (3.0 * bmag))
                    * (
                        curvature_Pe / density_safe
                        - 2.5 * tau * curvature_Ti
                        - curvature_phi
                    )
                )
        else:
            curvature_Ti_contribution = jnp.zeros_like(Ti)
        curvature_vorticity_contribution = (
            self.curvature_scale
            * (2.0 * bmag / density_safe)
            * curvature_pressure
            if "vorticity" in self.curvature_equations
            else jnp.zeros_like(density)
        )

        density_rhs = (
            -(poisson_density / (rho_star * bmag))
            - parallel_density_flux_divergence
            + curvature_density_contribution
            + density_diff
            + density_parallel_diff
        )
        Te_rhs = (
            -(poisson_Te / (rho_star * bmag))
            - Ve * grad_parallel_Te
            + curvature_Te_contribution
            + (2.0 * Te / (3.0 * density_safe))
            * (0.71 * parallel_current_flux_divergence - density * parallel_Ve_flux_divergence)
            + Te_diff
            + Te_parallel_diff
        )
        Ti_rhs = (
            -(poisson_Ti / (rho_star * bmag))
            - Vi * grad_parallel_Ti
            + curvature_Ti_contribution
            + (2.0 * Ti / (3.0 * density_safe))
            * (parallel_current_flux_divergence - density * parallel_Vi_flux_divergence)
            + Ti_diff
            + Ti_parallel_diff
        )
        Vi_rhs = (
            -(poisson_Vi / (rho_star * bmag))
            - Vi * grad_parallel_Vi
            - grad_parallel_pressure / density_safe
            + Vi_diff
            + Vi_parallel_diff
        )
        Ve_poisson_term = -(poisson_Ve / (rho_star * bmag))
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
            -(poisson_vorticity / (rho_star * bmag))
            - Vi * grad_parallel_vorticity
            + (bmag * bmag / density_safe) * parallel_current_flux_divergence
            + curvature_vorticity_contribution
            + vorticity_diff
            + vorticity_parallel_diff
        )
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
