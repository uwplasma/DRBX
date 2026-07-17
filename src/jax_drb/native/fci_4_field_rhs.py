from __future__ import annotations

import time as time_module
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import (
    ConservativeStencilBuilder,
    FciGeometry3D,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from .fci_model import FciModelState
from .fci_boundaries import BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D
from .fci_operators import (
    PerpLaplacianInverseSolver,
    PerpLaplacianMgHierarchy,
    curvature_op,
    grad_parallel_op_direct,
    perp_laplacian_conservative_op,
    poisson_bracket_op,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldState(FciModelState):
    density: jax.Array
    omega: jax.Array
    v_ion_parallel: jax.Array
    v_electron_parallel: jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldRhsParameters:
    """Physical and inversion parameters for the four-field model."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldFreeDecayParameters:
    """Four-field free-decay parameters, including perpendicular diffusion."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0
    density_perp_diffusion: float = 1.0e-2
    omega_perp_diffusion: float = 1.0e-2
    v_ion_parallel_perp_diffusion: float = 1.0e-2
    v_electron_parallel_perp_diffusion: float = 1.0e-2

    def rhs_parameters(self) -> Fci4FieldRhsParameters:
        return Fci4FieldRhsParameters(
            rho_star=self.rho_star,
            Te=self.Te,
            mi_over_me=self.mi_over_me,
            phi_inversion_tol=self.phi_inversion_tol,
            phi_inversion_maxiter=self.phi_inversion_maxiter,
            phi_inversion_restart=self.phi_inversion_restart,
            phi_inversion_regularization=self.phi_inversion_regularization,
        )

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
                self.density_perp_diffusion,
                self.omega_perp_diffusion,
                self.v_ion_parallel_perp_diffusion,
                self.v_electron_parallel_perp_diffusion,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
            density_perp_diffusion,
            omega_perp_diffusion,
            v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
            density_perp_diffusion=density_perp_diffusion,
            omega_perp_diffusion=omega_perp_diffusion,
            v_ion_parallel_perp_diffusion=v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion=v_electron_parallel_perp_diffusion,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldBlobParameters:
    """Four-field blob/interchange parameters."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0
    density_perp_diffusion: float = 1.0e-2
    omega_perp_diffusion: float = 1.0e-2
    v_ion_parallel_perp_diffusion: float = 1.0e-2
    v_electron_parallel_perp_diffusion: float = 1.0e-2

    def rhs_parameters(self) -> Fci4FieldRhsParameters:
        return Fci4FieldRhsParameters(
            rho_star=self.rho_star,
            Te=self.Te,
            mi_over_me=self.mi_over_me,
            phi_inversion_tol=self.phi_inversion_tol,
            phi_inversion_maxiter=self.phi_inversion_maxiter,
            phi_inversion_restart=self.phi_inversion_restart,
            phi_inversion_regularization=self.phi_inversion_regularization,
        )

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
                self.density_perp_diffusion,
                self.omega_perp_diffusion,
                self.v_ion_parallel_perp_diffusion,
                self.v_electron_parallel_perp_diffusion,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
            density_perp_diffusion,
            omega_perp_diffusion,
            v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
            density_perp_diffusion=density_perp_diffusion,
            omega_perp_diffusion=omega_perp_diffusion,
            v_ion_parallel_perp_diffusion=v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion=v_electron_parallel_perp_diffusion,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldRhsResult:
    rhs: Fci4FieldState

    def tree_flatten(self):
        return ((self.rhs,), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (rhs,) = children
        return cls(rhs=rhs)


def _cell_volume_weights(geometry: FciGeometry3D) -> jnp.ndarray:
    return (
        jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dx, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz, dtype=jnp.float64)
    )


def _weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _cell_volume_weights(geometry)
    return jnp.sum(weights * field) / jnp.maximum(jnp.sum(weights), 1.0e-30)


def _weighted_l2(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _cell_volume_weights(geometry)
    return jnp.sqrt(jnp.sum(weights * field * field) / jnp.maximum(jnp.sum(weights), 1.0e-30))


def _assemble_4field_non_diffusive_rhs(
    *,
    density: jnp.ndarray,
    omega: jnp.ndarray,
    v_ion_parallel: jnp.ndarray,
    v_electron_parallel: jnp.ndarray,
    rho_star: jnp.ndarray,
    te: jnp.ndarray,
    mi_over_me: jnp.ndarray,
    bmag: jnp.ndarray,
    density_safe: jnp.ndarray,
    geometry: FciGeometry3D,
    curvature_coefficients: jnp.ndarray,
    density_stencil: object,
    omega_stencil: object,
    phi_stencil: object,
    v_ion_parallel_stencil: object,
    v_electron_parallel_stencil: object,
) -> tuple[Fci4FieldState, Fci4FieldState, Fci4FieldState]:
    poisson_density = poisson_bracket_op(phi_stencil, density_stencil, geometry)
    poisson_omega = poisson_bracket_op(phi_stencil, omega_stencil, geometry)
    poisson_v_ion = poisson_bracket_op(phi_stencil, v_ion_parallel_stencil, geometry)
    poisson_v_electron = poisson_bracket_op(phi_stencil, v_electron_parallel_stencil, geometry)

    curvature_density = curvature_op(density_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_phi = curvature_op(phi_stencil, geometry, curvature_coefficients=curvature_coefficients)
    grad_parallel_density = grad_parallel_op_direct(density_stencil, geometry)
    grad_parallel_phi = grad_parallel_op_direct(phi_stencil, geometry)
    grad_parallel_v_ion = grad_parallel_op_direct(v_ion_parallel_stencil, geometry)
    grad_parallel_v_electron = grad_parallel_op_direct(v_electron_parallel_stencil, geometry)

    poisson_rhs = Fci4FieldState(
        density=-(poisson_density / (rho_star * bmag)),
        omega=-(poisson_omega / (rho_star * bmag)),
        v_ion_parallel=-(poisson_v_ion / (rho_star * bmag)),
        v_electron_parallel=-(poisson_v_electron / (rho_star * bmag)),
    )
    curvature_rhs = Fci4FieldState(
        density=(2.0 * te / bmag) * curvature_density - (2.0 * density / bmag) * curvature_phi,
        omega=(2.0 * bmag * te / density_safe) * curvature_density,
        v_ion_parallel=jnp.zeros_like(density),
        v_electron_parallel=jnp.zeros_like(density),
    )
    parallel_rhs = Fci4FieldState(
        density=-density * grad_parallel_v_electron,
        omega=(bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron),
        v_ion_parallel=-(te / density_safe) * grad_parallel_density,
        v_electron_parallel=mi_over_me * grad_parallel_phi - mi_over_me * (te / density_safe) * grad_parallel_density,
    )
    return poisson_rhs, curvature_rhs, parallel_rhs


def _assemble_4field_diffusion_rhs(
    *,
    density: jnp.ndarray,
    omega: jnp.ndarray,
    v_ion_parallel: jnp.ndarray,
    v_electron_parallel: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: Fci4FieldFreeDecayParameters,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    density_cut_wall_geometry: CutWallGeometry3D | None,
    density_cut_wall_bc: CutWallBC3D | None,
    omega_cut_wall_geometry: CutWallGeometry3D | None,
    omega_cut_wall_bc: CutWallBC3D | None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float,
    jacobian_floor: float,
) -> Fci4FieldState:
    # The perpendicular diffusion is a conservative control-volume operator, so it
    # requires conservative (face-gradient) stencils. Rebuild them from the fields
    # here rather than reuse the local advection stencils the callers pass in (which
    # carry no face-gradient data), so every caller gets a correct diffusion term.
    density_stencil = conservative_stencil_builder(
        density, geometry, periodic_axes=periodic_axes, face_bc=density_face_bc
    )
    omega_stencil = conservative_stencil_builder(
        omega, geometry, periodic_axes=periodic_axes, face_bc=omega_face_bc
    )
    v_ion_parallel_stencil = conservative_stencil_builder(
        v_ion_parallel, geometry, periodic_axes=periodic_axes, face_bc=v_ion_parallel_face_bc
    )
    v_electron_parallel_stencil = conservative_stencil_builder(
        v_electron_parallel, geometry, periodic_axes=periodic_axes, face_bc=v_electron_parallel_face_bc
    )
    density_diffusion = jnp.asarray(parameters.density_perp_diffusion, dtype=jnp.float64) * perp_laplacian_conservative_op(
        density_stencil,
        geometry,
        face_projectors=phi_face_projectors,
        face_bc=density_face_bc,
        cut_wall_geometry=density_cut_wall_geometry,
        cut_wall_bc=density_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    omega_diffusion = jnp.asarray(parameters.omega_perp_diffusion, dtype=jnp.float64) * perp_laplacian_conservative_op(
        omega_stencil,
        geometry,
        face_projectors=phi_face_projectors,
        face_bc=omega_face_bc,
        cut_wall_geometry=omega_cut_wall_geometry,
        cut_wall_bc=omega_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    v_ion_parallel_diffusion = jnp.asarray(parameters.v_ion_parallel_perp_diffusion, dtype=jnp.float64) * perp_laplacian_conservative_op(
        v_ion_parallel_stencil,
        geometry,
        face_projectors=phi_face_projectors,
        face_bc=v_ion_parallel_face_bc,
        cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        cut_wall_bc=v_ion_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    v_electron_parallel_diffusion = jnp.asarray(parameters.v_electron_parallel_perp_diffusion, dtype=jnp.float64) * perp_laplacian_conservative_op(
        v_electron_parallel_stencil,
        geometry,
        face_projectors=phi_face_projectors,
        face_bc=v_electron_parallel_face_bc,
        cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        cut_wall_bc=v_electron_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    return Fci4FieldState(
        density=density_diffusion,
        omega=omega_diffusion,
        v_ion_parallel=v_ion_parallel_diffusion,
        v_electron_parallel=v_electron_parallel_diffusion,
    )


def _compute_4field_term_components(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    parameters: Fci4FieldFreeDecayParameters,
    curvature_coefficients: jnp.ndarray,
    phi: jnp.ndarray,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None,
    phi_cut_wall_bc: CutWallBC3D | None,
    density_cut_wall_geometry: CutWallGeometry3D | None,
    density_cut_wall_bc: CutWallBC3D | None,
    omega_cut_wall_geometry: CutWallGeometry3D | None,
    omega_cut_wall_bc: CutWallBC3D | None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float,
    jacobian_floor: float,
) -> tuple[Fci4FieldState, Fci4FieldState, Fci4FieldState, Fci4FieldState]:
    density = jnp.asarray(state.density, dtype=jnp.float64)
    omega = jnp.asarray(state.omega, dtype=jnp.float64)
    v_ion_parallel = jnp.asarray(state.v_ion_parallel, dtype=jnp.float64)
    v_electron_parallel = jnp.asarray(state.v_electron_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    density_safe = jnp.maximum(density, 1.0e-30)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)

    density_stencil = stencil_builder(
        density,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=density_face_bc,
        cut_wall_geometry=density_cut_wall_geometry,
        cut_wall_bc=density_cut_wall_bc,
    )
    omega_stencil = stencil_builder(
        omega,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=omega_face_bc,
        cut_wall_geometry=omega_cut_wall_geometry,
        cut_wall_bc=omega_cut_wall_bc,
    )
    phi_stencil = stencil_builder(
        phi,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=phi_face_bc,
        cut_wall_geometry=phi_cut_wall_geometry,
        cut_wall_bc=phi_cut_wall_bc,
    )
    v_ion_parallel_stencil = stencil_builder(
        v_ion_parallel,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=v_ion_parallel_face_bc,
        cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        cut_wall_bc=v_ion_parallel_cut_wall_bc,
    )
    v_electron_parallel_stencil = stencil_builder(
        v_electron_parallel,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=v_electron_parallel_face_bc,
        cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        cut_wall_bc=v_electron_parallel_cut_wall_bc,
    )

    poisson_rhs, curvature_rhs, parallel_rhs = _assemble_4field_non_diffusive_rhs(
        density=density,
        omega=omega,
        v_ion_parallel=v_ion_parallel,
        v_electron_parallel=v_electron_parallel,
        rho_star=rho_star,
        te=te,
        mi_over_me=mi_over_me,
        bmag=bmag,
        density_safe=density_safe,
        geometry=geometry,
        curvature_coefficients=curvature_coefficients,
        density_stencil=density_stencil,
        omega_stencil=omega_stencil,
        phi_stencil=phi_stencil,
        v_ion_parallel_stencil=v_ion_parallel_stencil,
        v_electron_parallel_stencil=v_electron_parallel_stencil,
    )
    diffusion_rhs = _assemble_4field_diffusion_rhs(
        density=density,
        omega=omega,
        v_ion_parallel=v_ion_parallel,
        v_electron_parallel=v_electron_parallel,
        geometry=geometry,
        parameters=parameters,
        phi_face_projectors=phi_face_projectors,
        conservative_stencil_builder=conservative_stencil_builder,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    return poisson_rhs, curvature_rhs, parallel_rhs, diffusion_rhs


def compute_4field_rhs(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldRhsParameters = Fci4FieldRhsParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray | None] | tuple[Fci4FieldRhsResult, jnp.ndarray | None, jnp.ndarray]:
    """Assemble the normalized four-field RHS using the new stencil pipeline.

    By default the assembly is sync-free (``timings`` is ``None``) so the whole
    RHS can run inside ``jit``; ``with_diagnostics=True`` restores the
    host-synced stage timings and phi-solver diagnostics payload used by the
    validation harnesses.
    """

    density = jnp.asarray(state.density, dtype=jnp.float64)
    omega = jnp.asarray(state.omega, dtype=jnp.float64)
    v_ion_parallel = jnp.asarray(state.v_ion_parallel, dtype=jnp.float64)
    v_electron_parallel = jnp.asarray(state.v_electron_parallel, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    density_safe = jnp.maximum(density, 1.0e-30)

    if phi_inverse_solver is None:
        raise ValueError("compute_4field_rhs requires a PerpLaplacianInverseSolver instance")
    if with_diagnostics:
        phi_start = time_module.perf_counter()
        phi, phi_diagnostics = phi_inverse_solver(
            -omega,
            phi_guess=phi_guess,
            face_bc=phi_face_bc,
            cut_wall_bc=phi_cut_wall_bc,
            return_diagnostics=True,
        )
        jax.block_until_ready(phi)
        phi_time = time_module.perf_counter() - phi_start
        phi_num_steps = float(phi_diagnostics["num_steps"])
        rhs_mean_j = float(phi_diagnostics["rhs_mean_J"])
        rhs_l2_j = float(phi_diagnostics["rhs_l2_J"])
        rhs_compatibility_ratio = float(phi_diagnostics["rhs_compatibility_ratio"])
        projected_rhs_mean_j = float(phi_diagnostics["projected_rhs_mean_J"])
        projected_rhs_l2_j = float(phi_diagnostics["projected_rhs_l2_J"])
        projected_rhs_compatibility_ratio = float(phi_diagnostics["projected_rhs_compatibility_ratio"])
    else:
        phi = phi_inverse_solver(
            -omega,
            phi_guess=phi_guess,
            face_bc=phi_face_bc,
            cut_wall_bc=phi_cut_wall_bc,
        )

    stencil_start = time_module.perf_counter() if with_diagnostics else 0.0
    density_stencil = stencil_builder(
        density,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=density_face_bc,
        cut_wall_geometry=density_cut_wall_geometry,
        cut_wall_bc=density_cut_wall_bc,
    )
    omega_stencil = stencil_builder(
        omega,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=omega_face_bc,
        cut_wall_geometry=omega_cut_wall_geometry,
        cut_wall_bc=omega_cut_wall_bc,
    )
    phi_stencil = stencil_builder(
        phi,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=phi_face_bc,
        cut_wall_geometry=phi_cut_wall_geometry,
        cut_wall_bc=phi_cut_wall_bc,
    )
    v_ion_parallel_stencil = stencil_builder(
        v_ion_parallel,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=v_ion_parallel_face_bc,
        cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        cut_wall_bc=v_ion_parallel_cut_wall_bc,
    )
    v_electron_parallel_stencil = stencil_builder(
        v_electron_parallel,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=v_electron_parallel_face_bc,
        cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        cut_wall_bc=v_electron_parallel_cut_wall_bc,
    )
    if with_diagnostics:
        jax.block_until_ready(density_stencil.x.center)
        jax.block_until_ready(omega_stencil.x.center)
        jax.block_until_ready(phi_stencil.x.center)
        jax.block_until_ready(v_ion_parallel_stencil.x.center)
        jax.block_until_ready(v_electron_parallel_stencil.x.center)
    local_stencil_time = (time_module.perf_counter() - stencil_start) if with_diagnostics else 0.0

    operator_start = time_module.perf_counter() if with_diagnostics else 0.0
    poisson_rhs, curvature_rhs, parallel_rhs = _assemble_4field_non_diffusive_rhs(
        density=density,
        omega=omega,
        v_ion_parallel=v_ion_parallel,
        v_electron_parallel=v_electron_parallel,
        rho_star=rho_star,
        te=te,
        mi_over_me=mi_over_me,
        bmag=bmag,
        density_safe=density_safe,
        geometry=geometry,
        curvature_coefficients=curvature_coefficients,
        density_stencil=density_stencil,
        omega_stencil=omega_stencil,
        phi_stencil=phi_stencil,
        v_ion_parallel_stencil=v_ion_parallel_stencil,
        v_electron_parallel_stencil=v_electron_parallel_stencil,
    )

    rhs_density = jnp.asarray(
        poisson_rhs.density + curvature_rhs.density + parallel_rhs.density,
        dtype=jnp.float64,
    )
    rhs_omega = jnp.asarray(
        poisson_rhs.omega + curvature_rhs.omega + parallel_rhs.omega,
        dtype=jnp.float64,
    )
    rhs_v_ion_parallel = jnp.asarray(
        poisson_rhs.v_ion_parallel + curvature_rhs.v_ion_parallel + parallel_rhs.v_ion_parallel,
        dtype=jnp.float64,
    )
    rhs_v_electron_parallel = jnp.asarray(
        poisson_rhs.v_electron_parallel + curvature_rhs.v_electron_parallel + parallel_rhs.v_electron_parallel,
        dtype=jnp.float64,
    )
    rhs = Fci4FieldState(
        density=rhs_density,
        omega=rhs_omega,
        v_ion_parallel=rhs_v_ion_parallel,
        v_electron_parallel=rhs_v_electron_parallel,
    )
    if with_diagnostics:
        jax.block_until_ready(rhs_density)
        jax.block_until_ready(rhs_omega)
        jax.block_until_ready(rhs_v_ion_parallel)
        jax.block_until_ready(rhs_v_electron_parallel)
        operator_time = time_module.perf_counter() - operator_start
        timings = jnp.asarray(
            [
                phi_time,
                local_stencil_time,
                operator_time,
                phi_num_steps,
                rhs_mean_j,
                rhs_l2_j,
                rhs_compatibility_ratio,
                projected_rhs_mean_j,
                projected_rhs_l2_j,
                projected_rhs_compatibility_ratio,
                float(phi_diagnostics["final_residual_rel_l2"]),
            ],
            dtype=jnp.float64,
        )
    else:
        timings = None
    if return_phi:
        return Fci4FieldRhsResult(rhs=rhs), timings, phi
    return Fci4FieldRhsResult(rhs=rhs), timings


def compute_4field_free_decay_rhs(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldFreeDecayParameters = Fci4FieldFreeDecayParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray] | tuple[Fci4FieldRhsResult, jnp.ndarray, jnp.ndarray]:
    if return_phi:
        base_result, timings, phi = compute_4field_rhs(
            state,
            geometry=geometry,
            stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters.rhs_parameters(),
            curvature_coefficients=curvature_coefficients,
            phi_guess=phi_guess,
            phi_face_bc=phi_face_bc,
            density_face_bc=density_face_bc,
            omega_face_bc=omega_face_bc,
            v_ion_parallel_face_bc=v_ion_parallel_face_bc,
            v_electron_parallel_face_bc=v_electron_parallel_face_bc,
            phi_cut_wall_geometry=phi_cut_wall_geometry,
            phi_cut_wall_bc=phi_cut_wall_bc,
            density_cut_wall_geometry=density_cut_wall_geometry,
            density_cut_wall_bc=density_cut_wall_bc,
            omega_cut_wall_geometry=omega_cut_wall_geometry,
            omega_cut_wall_bc=omega_cut_wall_bc,
            v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
            v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
            v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
            v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
            phi_face_projectors=phi_face_projectors,
            phi_mg_hierarchy=phi_mg_hierarchy,
            phi_inverse_solver=phi_inverse_solver,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
            gmres_debug=gmres_debug,
            return_phi=True,
            with_diagnostics=with_diagnostics,
        )
    else:
        base_result, timings = compute_4field_rhs(
            state,
            geometry=geometry,
            stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters.rhs_parameters(),
            curvature_coefficients=curvature_coefficients,
            phi_guess=phi_guess,
            phi_face_bc=phi_face_bc,
            density_face_bc=density_face_bc,
            omega_face_bc=omega_face_bc,
            v_ion_parallel_face_bc=v_ion_parallel_face_bc,
            v_electron_parallel_face_bc=v_electron_parallel_face_bc,
            phi_cut_wall_geometry=phi_cut_wall_geometry,
            phi_cut_wall_bc=phi_cut_wall_bc,
            density_cut_wall_geometry=density_cut_wall_geometry,
            density_cut_wall_bc=density_cut_wall_bc,
            omega_cut_wall_geometry=omega_cut_wall_geometry,
            omega_cut_wall_bc=omega_cut_wall_bc,
            v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
            v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
            v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
            v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
            phi_face_projectors=phi_face_projectors,
            phi_mg_hierarchy=phi_mg_hierarchy,
            phi_inverse_solver=phi_inverse_solver,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
            gmres_debug=gmres_debug,
            return_phi=False,
            with_diagnostics=with_diagnostics,
        )

    diffusion_rhs = _assemble_4field_diffusion_rhs(
        density=state.density,
        omega=state.omega,
        v_ion_parallel=state.v_ion_parallel,
        v_electron_parallel=state.v_electron_parallel,
        geometry=geometry,
        parameters=parameters,
        phi_face_projectors=phi_face_projectors,
        conservative_stencil_builder=conservative_stencil_builder,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )

    rhs = base_result.rhs
    result = Fci4FieldRhsResult(
        rhs=Fci4FieldState(
            density=rhs.density + diffusion_rhs.density,
            omega=rhs.omega + diffusion_rhs.omega,
            v_ion_parallel=rhs.v_ion_parallel + diffusion_rhs.v_ion_parallel,
            v_electron_parallel=rhs.v_electron_parallel + diffusion_rhs.v_electron_parallel,
        )
    )
    if return_phi:
        return result, timings, phi
    return result, timings


def compute_4field_blob_rhs(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldBlobParameters = Fci4FieldBlobParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray] | tuple[Fci4FieldRhsResult, jnp.ndarray, jnp.ndarray]:
    """Assemble the four-field blob RHS.

    The blob model shares its full right-hand side with the free-decay
    wrapper (non-diffusive terms plus conservative perpendicular
    diffusion), so this simply delegates -- the two were verified to
    produce bitwise-identical RHS fields and phi.
    """

    return compute_4field_free_decay_rhs(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_guess=phi_guess,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
        gmres_debug=gmres_debug,
        return_phi=return_phi,
        with_diagnostics=with_diagnostics,
    )


def compute_4field_diffusion(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldFreeDecayParameters = Fci4FieldFreeDecayParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray] | tuple[Fci4FieldRhsResult, jnp.ndarray, jnp.ndarray]:
    base_result, timings, phi = compute_4field_free_decay_rhs(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_guess=phi_guess,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
        gmres_debug=gmres_debug,
        return_phi=True,
        with_diagnostics=with_diagnostics,
    )
    poisson_rhs, curvature_rhs, parallel_rhs, diffusion_rhs = _compute_4field_term_components(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi=phi,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    result = Fci4FieldRhsResult(
        rhs=Fci4FieldState(
            density=diffusion_rhs.density,
            omega=diffusion_rhs.omega,
            v_ion_parallel=diffusion_rhs.v_ion_parallel,
            v_electron_parallel=diffusion_rhs.v_electron_parallel,
        )
    )
    if return_phi:
        return result, timings, phi
    return result, timings


def compute_4field_poisson_diffusion(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldFreeDecayParameters = Fci4FieldFreeDecayParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray] | tuple[Fci4FieldRhsResult, jnp.ndarray, jnp.ndarray]:
    base_result, timings, phi = compute_4field_free_decay_rhs(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_guess=phi_guess,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
        gmres_debug=gmres_debug,
        return_phi=True,
        with_diagnostics=with_diagnostics,
    )
    poisson_rhs, curvature_rhs, parallel_rhs, diffusion_rhs = _compute_4field_term_components(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi=phi,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    result = Fci4FieldRhsResult(
        rhs=Fci4FieldState(
            density=poisson_rhs.density + diffusion_rhs.density + parallel_rhs.density,
            omega=poisson_rhs.omega + diffusion_rhs.omega + parallel_rhs.omega,
            v_ion_parallel=poisson_rhs.v_ion_parallel + diffusion_rhs.v_ion_parallel + parallel_rhs.v_ion_parallel,
            v_electron_parallel=poisson_rhs.v_electron_parallel
            + diffusion_rhs.v_electron_parallel
            + parallel_rhs.v_electron_parallel,
        )
    )
    if return_phi:
        return result, timings, phi
    return result, timings


def compute_4field_curvature(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: Fci4FieldFreeDecayParameters = Fci4FieldFreeDecayParameters(),
    curvature_coefficients: jnp.ndarray,
    phi_guess: jnp.ndarray | None = None,
    phi_face_bc: BoundaryFaceBC3D,
    density_face_bc: BoundaryFaceBC3D,
    omega_face_bc: BoundaryFaceBC3D,
    v_ion_parallel_face_bc: BoundaryFaceBC3D,
    v_electron_parallel_face_bc: BoundaryFaceBC3D,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    omega_cut_wall_geometry: CutWallGeometry3D | None = None,
    omega_cut_wall_bc: CutWallBC3D | None = None,
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_ion_parallel_cut_wall_bc: CutWallBC3D | None = None,
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_electron_parallel_cut_wall_bc: CutWallBC3D | None = None,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
    gmres_debug: bool = False,
    return_phi: bool = False,
    with_diagnostics: bool = False,
) -> tuple[Fci4FieldRhsResult, jnp.ndarray] | tuple[Fci4FieldRhsResult, jnp.ndarray, jnp.ndarray]:
    base_result, timings, phi = compute_4field_free_decay_rhs(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_guess=phi_guess,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
        gmres_debug=gmres_debug,
        return_phi=True,
        with_diagnostics=with_diagnostics,
    )
    poisson_rhs, curvature_rhs, parallel_rhs, diffusion_rhs = _compute_4field_term_components(
        state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi=phi,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=v_electron_parallel_face_bc,
        phi_cut_wall_geometry=phi_cut_wall_geometry,
        phi_cut_wall_bc=phi_cut_wall_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        omega_cut_wall_geometry=omega_cut_wall_geometry,
        omega_cut_wall_bc=omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    result = Fci4FieldRhsResult(
        rhs=Fci4FieldState(
            density=curvature_rhs.density + parallel_rhs.density,
            omega=curvature_rhs.omega + parallel_rhs.omega,
            v_ion_parallel=curvature_rhs.v_ion_parallel + parallel_rhs.v_ion_parallel,
            v_electron_parallel=curvature_rhs.v_electron_parallel + parallel_rhs.v_electron_parallel,
        )
    )
    if return_phi:
        return result, timings, phi
    return result, timings


__all__ = [
    "Fci4FieldBlobParameters",
    "Fci4FieldFreeDecayParameters",
    "Fci4FieldState",
    "Fci4FieldRhsParameters",
    "Fci4FieldRhsResult",
    "compute_4field_blob_rhs",
    "compute_4field_curvature",
    "compute_4field_diffusion",
    "compute_4field_free_decay_rhs",
    "compute_4field_rhs",
    "compute_4field_poisson_diffusion",
]
