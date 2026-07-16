from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import (
    ConservativeStencilBuilder,
    FciGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    StencilBuilderContext,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_local_conservative_stencil_from_field,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_local_stencil_from_field,
)
from .fci_model import FciModelState
from .fci_model import inject_owned_state_to_halo
from .fci_boundaries import (
    BoundaryFaceBC3D,
    ConservativeStencil3D,
    CutWallBC3D,
    CutWallGeometry3D,
    LocalBoundaryFaceBC3D,
    LocalStencil1D,
    LocalStencil3D,
)
from .fci_halo import HaloExchange3D, PhysicalGhostCellFiller3D
from .fci_operators import (
    LocalPerpLaplacianInverseSolver,
    curvature_op,
    grad_parallel_op_direct,
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_perp_laplacian_conservative_op,
    local_poisson_bracket_op,
    parallel_laplacian_direct_op,
    perp_laplacian_conservative_op,
    poisson_bracket_op,
)
from .fci_gmres import SpmdGmresConfig
from .fci_model import inject_owned_field_to_halo


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
class FciDrbEBBoundaryConditions:
    """Per-field boundary payload for the electrostatic Boussinesq DRB model."""

    density_face_bc: BoundaryFaceBC3D
    density_cut_wall_bc: CutWallBC3D
    potential_face_bc: BoundaryFaceBC3D
    potential_cut_wall_bc: CutWallBC3D
    vorticity_face_bc: BoundaryFaceBC3D
    vorticity_cut_wall_bc: CutWallBC3D
    Te_face_bc: BoundaryFaceBC3D
    Te_cut_wall_bc: CutWallBC3D
    Ti_face_bc: BoundaryFaceBC3D
    Ti_cut_wall_bc: CutWallBC3D
    Vi_face_bc: BoundaryFaceBC3D
    Vi_cut_wall_bc: CutWallBC3D
    Ve_face_bc: BoundaryFaceBC3D
    Ve_cut_wall_bc: CutWallBC3D

    def tree_flatten(self):
        return (
            (
                self.density_face_bc,
                self.density_cut_wall_bc,
                self.potential_face_bc,
                self.potential_cut_wall_bc,
                self.vorticity_face_bc,
                self.vorticity_cut_wall_bc,
                self.Te_face_bc,
                self.Te_cut_wall_bc,
                self.Ti_face_bc,
                self.Ti_cut_wall_bc,
                self.Vi_face_bc,
                self.Vi_cut_wall_bc,
                self.Ve_face_bc,
                self.Ve_cut_wall_bc,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FciDrbEBRhsResult:
    rhs: FciDrbEBState
    potential: jax.Array
    potential_residual_l2: jax.Array

    def tree_flatten(self):
        return ((self.rhs, self.potential, self.potential_residual_l2), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        rhs, potential, potential_residual_l2 = children
        return cls(rhs=rhs, potential=potential, potential_residual_l2=potential_residual_l2)


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


def prepare_local_fci_drb_eb_state(
    state_owned: FciDrbEBState,
    domain: LocalDomain3D,
    *,
    face_bc: LocalFciDrbEBFaceBCBundle,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
) -> FciDrbEBState:
    """Exchange topology/remote halos and fill local physical ghosts for all EB fields."""

    state_halo = inject_owned_state_to_halo(state_owned, domain.layout)
    state_halo = FciDrbEBState(
        density=topology_filler(halo_exchange(state_halo.density, domain), domain),
        phi=topology_filler(halo_exchange(state_halo.phi, domain), domain),
        Te=topology_filler(halo_exchange(state_halo.Te, domain), domain),
        Ti=topology_filler(halo_exchange(state_halo.Ti, domain), domain),
        Vi=topology_filler(halo_exchange(state_halo.Vi, domain), domain),
        Ve=topology_filler(halo_exchange(state_halo.Ve, domain), domain),
        vorticity=topology_filler(halo_exchange(state_halo.vorticity, domain), domain),
    )
    return FciDrbEBState(
        density=physical_ghost_filler(state_halo.density, domain, face_bc.density),
        phi=physical_ghost_filler(state_halo.phi, domain, face_bc.phi),
        Te=physical_ghost_filler(state_halo.Te, domain, face_bc.Te),
        Ti=physical_ghost_filler(state_halo.Ti, domain, face_bc.Ti),
        Vi=physical_ghost_filler(state_halo.Vi, domain, face_bc.Vi),
        Ve=physical_ghost_filler(state_halo.Ve, domain, face_bc.Ve),
        vorticity=physical_ghost_filler(state_halo.vorticity, domain, face_bc.vorticity),
    )


@dataclass(frozen=True)
class LocalFciDrbEBRhs:
    """SPMD/local EB RHS and phi reconstruction.

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
    curvature_coefficients_owned: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SpmdGmresConfig
    face_bc_builder: LocalFciDrbEBFaceBCBuilder
    diffusion_only: bool = False
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)

    def _face_bcs(self, state_owned: FciDrbEBState) -> LocalFciDrbEBFaceBCBundle:
        return self.face_bc_builder(
            state_owned,
            self.geometry,
            self.domain,
            self.parameters,
        )

    def _prepare_phi_halo(
        self,
        phi_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        phi_halo = inject_owned_field_to_halo(phi_owned, self.domain.layout)
        phi_halo = self.topology_filler(
            self.halo_exchange(phi_halo, self.domain),
            self.domain,
        )
        return self.physical_ghost_filler(phi_halo, self.domain, face_bc)

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
    ) -> jnp.ndarray:
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
            - state_halo.vorticity[owned]
        )
        phi_lift = jnp.broadcast_to(
            face_bc.phi.value_x[-1][None, :, :],
            self.geometry.owned_shape,
        )
        solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            regular_face_geometry=self.geometry.regular_face_geometry,
            face_bc=face_bc.phi,
            axis_regular_axes=self.axis_regular_axes,
            config=self.gmres_config,
        )
        return solver(
            phi_rhs,
            guess_owned=state_owned.phi,
            phi_lift_owned=phi_lift,
        )

    def reconstruct_phi(self, state_owned: FciDrbEBState) -> jnp.ndarray:
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
        )

    def evaluate_stage(
        self,
        state_owned: FciDrbEBState,
        source_owned: FciDrbEBState | None = None,
    ) -> FciDrbEBState:
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
        face_bc = self._face_bcs(state_owned)
        state_halo_without_phi = prepare_local_fci_drb_eb_state(
            state_owned,
            self.domain,
            face_bc=face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        phi_owned = self._reconstruct_phi_from_prepared(
            state_owned,
            state_halo_without_phi,
            face_bc,
        )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        state_halo = state_halo_without_phi.replace(phi=phi_halo)
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        direct = build_local_direct_stencil_one_sided_physical_from_halo

        density_stencil = direct(state_halo.density, self.geometry, context)
        Te_stencil = direct(state_halo.Te, self.geometry, context)
        Ti_stencil = direct(state_halo.Ti, self.geometry, context)
        Vi_stencil = direct(state_halo.Vi, self.geometry, context)
        Ve_stencil = direct(state_halo.Ve, self.geometry, context)
        vorticity_stencil = direct(state_halo.vorticity, self.geometry, context)
        phi_stencil = direct(state_halo.phi, self.geometry, context)

        Pe_halo = state_halo.density * state_halo.Te
        pressure_halo = Pe_halo + self.parameters.tau * state_halo.density * state_halo.Ti
        current_halo = state_halo.density * (state_halo.Vi - state_halo.Ve)
        density_flux_halo = state_halo.density * state_halo.Ve
        Pe_stencil = direct(Pe_halo, self.geometry, context)
        pressure_stencil = direct(pressure_halo, self.geometry, context)
        current_stencil = direct(current_halo, self.geometry, context)
        density_flux_conservative_stencil = build_local_conservative_stencil_from_field(
            density_flux_halo,
            self.geometry,
            context,
        )
        current_conservative_stencil = build_local_conservative_stencil_from_field(
            current_halo,
            self.geometry,
            context,
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
            return FciDrbEBState(
                density=density_diff,
                phi=jnp.zeros_like(phi_owned),
                Te=Te_diff + Te_parallel_diff,
                Ti=Ti_diff + Ti_parallel_diff,
                Vi=Vi_diff + Vi_parallel_diff,
                Ve=Ve_diff + Ve_parallel_diff,
                vorticity=vorticity_diff + vorticity_parallel_diff,
            )

        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        poisson_Te = local_poisson_bracket_op(phi_stencil, Te_stencil, self.geometry)
        poisson_Ti = local_poisson_bracket_op(phi_stencil, Ti_stencil, self.geometry)
        poisson_Vi = local_poisson_bracket_op(phi_stencil, Vi_stencil, self.geometry)
        poisson_Ve = local_poisson_bracket_op(phi_stencil, Ve_stencil, self.geometry)
        poisson_vorticity = local_poisson_bracket_op(
            phi_stencil,
            vorticity_stencil,
            self.geometry,
        )

        curvature_Pe = local_curvature_op(
            Pe_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_pressure = local_curvature_op(
            pressure_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_Te = local_curvature_op(
            Te_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_Ti = local_curvature_op(
            Ti_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
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
        grad_parallel_Te = local_grad_parallel_op_direct(Te_stencil, self.geometry)
        grad_parallel_Ti = local_grad_parallel_op_direct(Ti_stencil, self.geometry)
        grad_parallel_Ve = local_grad_parallel_op_direct(Ve_stencil, self.geometry)
        grad_parallel_Vi = local_grad_parallel_op_direct(Vi_stencil, self.geometry)
        grad_parallel_phi = local_grad_parallel_op_direct(phi_stencil, self.geometry)
        grad_parallel_Pe = local_grad_parallel_op_direct(Pe_stencil, self.geometry)
        grad_parallel_pressure = local_grad_parallel_op_direct(
            pressure_stencil,
            self.geometry,
        )
        grad_parallel_current = local_grad_parallel_op_direct(
            current_stencil,
            self.geometry,
        )
        grad_parallel_vorticity = local_grad_parallel_op_direct(
            vorticity_stencil,
            self.geometry,
        )

        density_rhs = (
            -(poisson_density / (rho_star * bmag))
            - parallel_density_flux_divergence
            + (2.0 / bmag) * (curvature_Pe - density * curvature_phi)
            + density_diff
            + density_parallel_diff
        )
        Te_rhs = (
            -(poisson_Te / (rho_star * bmag))
            - Ve * grad_parallel_Te
            + (4.0 * Te / (3.0 * bmag))
            * (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
            + (2.0 * Te / (3.0 * density_safe))
            * (0.71 * parallel_current_flux_divergence - density * parallel_Ve_flux_divergence)
            + Te_diff
            + Te_parallel_diff
        )
        Ti_rhs = (
            -(poisson_Ti / (rho_star * bmag))
            - Vi * grad_parallel_Ti
            + (4.0 * Ti / (3.0 * bmag))
            * (curvature_Pe / density_safe - 2.5 * tau * curvature_Ti - curvature_phi)
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
        Ve_rhs = (
            -(poisson_Ve / (rho_star * bmag))
            - Ve * grad_parallel_Ve
            + mi_over_me
            * (
                Ve_nu * current
                + grad_parallel_phi
                - grad_parallel_Pe / density_safe
                - 0.71 * grad_parallel_Te
            )
            + Ve_diff
            + Ve_parallel_diff
        )
        vorticity_rhs = (
            -(poisson_vorticity / (rho_star * bmag))
            - Vi * grad_parallel_vorticity
            + (bmag * bmag / density_safe) * parallel_current_flux_divergence
            + (2.0 * bmag / density_safe) * curvature_pressure
            + vorticity_diff
            + vorticity_parallel_diff
        )
        return FciDrbEBState(
            density=density_rhs + source_owned.density,
            phi=jnp.zeros_like(phi_owned),
            Te=Te_rhs + source_owned.Te,
            Ti=Ti_rhs + source_owned.Ti,
            Vi=Vi_rhs + source_owned.Vi,
            Ve=Ve_rhs + source_owned.Ve,
            vorticity=vorticity_rhs + source_owned.vorticity,
        )


def _multiply_local_stencils(left: LocalStencil3D, right: LocalStencil3D) -> LocalStencil3D:
    """Multiply two boundary-complete local stencils pointwise."""

    def _multiply_axis(left_axis: LocalStencil1D, right_axis: LocalStencil1D) -> LocalStencil1D:
        return left_axis.replace(
            center=left_axis.center * right_axis.center,
            minus=left_axis.minus * right_axis.minus,
            plus=left_axis.plus * right_axis.plus,
            derivative_minus_weight=left_axis.derivative_minus_weight,
            derivative_center_weight=left_axis.derivative_center_weight,
            derivative_plus_weight=left_axis.derivative_plus_weight,
        )

    return LocalStencil3D(
        x=_multiply_axis(left.x, right.x),
        y=_multiply_axis(left.y, right.y),
        z=_multiply_axis(left.z, right.z),
    )


def _density_rhs(
    *,
    density: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    Pe_stencil: LocalStencil3D,
    Ve_stencil: LocalStencil3D,
    density_conservative_stencil: ConservativeStencil3D | None,
    density_face_bc: BoundaryFaceBC3D,
    density_cut_wall_geometry: CutWallGeometry3D | None,
    density_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the density RHS with prebuilt stencils and geometry inputs."""

    density = jnp.asarray(density, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    density_D_perp = jnp.asarray(parameters.density_D_perp, dtype=jnp.float64)
    density_D_parallel = jnp.asarray(parameters.density_D_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_density = poisson_bracket_op(potential_stencil, density_stencil, geometry)
    particle_flux_parallel_stencil = _multiply_local_stencils(density_stencil, Ve_stencil)
    parallel_density_flux = grad_parallel_op_direct(particle_flux_parallel_stencil, geometry)
    curvature_Pe = curvature_op(
        Pe_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_potential = curvature_op(
        potential_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    perp_diffusion_density = jnp.zeros_like(density)
    if float(density_D_perp) != 0.0:
        if density_conservative_stencil is None:
            raise ValueError("density_conservative_stencil is required when density_D_perp is nonzero")
        perp_diffusion_density = density_D_perp * perp_laplacian_conservative_op(
            density_conservative_stencil,
            geometry,
            face_bc=density_face_bc,
            cut_wall_geometry=density_cut_wall_geometry,
            cut_wall_bc=density_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_density = jnp.zeros_like(density)
    if float(density_D_parallel) != 0.0:
        parallel_diffusion_density = density_D_parallel * parallel_laplacian_direct_op(
            density,
            geometry,
            face_bc=density_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_density / (rho_star * bmag))
        - parallel_density_flux
        + (2.0 / bmag) * (curvature_Pe - density * curvature_potential)
        + perp_diffusion_density
        + parallel_diffusion_density
    )


def _Te_rhs(
    *,
    Te: jnp.ndarray,
    density: jnp.ndarray,
    Vi: jnp.ndarray,
    Ve: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    temperature_stencil: LocalStencil3D,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    Pe_stencil: LocalStencil3D,
    current_density_stencil: LocalStencil3D,
    Ve_stencil: LocalStencil3D,
    temperature_conservative_stencil: ConservativeStencil3D | None,
    temperature_face_bc: BoundaryFaceBC3D,
    temperature_cut_wall_geometry: CutWallGeometry3D | None,
    temperature_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the electron temperature RHS with prebuilt stencils and geometry inputs."""

    Te = jnp.asarray(Te, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    Ve = jnp.asarray(Ve, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    chi_parallel = jnp.asarray(parameters.electron_temperature_chi_parallel, dtype=jnp.float64)
    D_perp = jnp.asarray(parameters.electron_temperature_D_perp, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_te = poisson_bracket_op(potential_stencil, temperature_stencil, geometry)
    parallel_advection_te = Ve * grad_parallel_op_direct(temperature_stencil, geometry)
    curvature_Pe = curvature_op(
        Pe_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_temperature = curvature_op(
        temperature_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_potential = curvature_op(
        potential_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    parallel_current_density = grad_parallel_op_direct(current_density_stencil, geometry)
    parallel_Ve = grad_parallel_op_direct(Ve_stencil, geometry)
    perp_diffusion_te = jnp.zeros_like(Te)
    if float(D_perp) != 0.0:
        if temperature_conservative_stencil is None:
            raise ValueError("temperature_conservative_stencil is required when electron_temperature_D_perp is nonzero")
        perp_diffusion_te = D_perp * perp_laplacian_conservative_op(
            temperature_conservative_stencil,
            geometry,
            face_bc=temperature_face_bc,
            cut_wall_geometry=temperature_cut_wall_geometry,
            cut_wall_bc=temperature_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_te = jnp.zeros_like(Te)
    if float(chi_parallel) != 0.0:
        parallel_diffusion_te = chi_parallel * parallel_laplacian_direct_op(
            Te,
            geometry,
            face_bc=temperature_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_te / (rho_star * bmag))
        - parallel_advection_te
        + (4.0 * Te / (3.0 * bmag))
        * (
            curvature_Pe / density
            + 2.5 * curvature_temperature
            - curvature_potential
        )
        + (2.0 * Te / (3.0 * density))
        * (
            0.71 * parallel_current_density - density * parallel_Ve
        )
        + parallel_diffusion_te
        + perp_diffusion_te
    )


def _Ti_rhs(
    *,
    Ti: jnp.ndarray,
    density: jnp.ndarray,
    Vi: jnp.ndarray,
    Ve: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    Ti_stencil: LocalStencil3D,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    Pe_stencil: LocalStencil3D,
    current_density_stencil: LocalStencil3D,
    Vi_stencil: LocalStencil3D,
    Ti_conservative_stencil: ConservativeStencil3D | None,
    Ti_face_bc: BoundaryFaceBC3D,
    Ti_cut_wall_geometry: CutWallGeometry3D | None,
    Ti_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the ion temperature RHS with prebuilt stencils and geometry inputs."""

    Ti = jnp.asarray(Ti, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    Ve = jnp.asarray(Ve, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    tau = jnp.asarray(parameters.tau, dtype=jnp.float64)
    chi_parallel = jnp.asarray(parameters.ion_temperature_chi_parallel, dtype=jnp.float64)
    D_perp = jnp.asarray(parameters.ion_temperature_D_perp, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_Ti = poisson_bracket_op(potential_stencil, Ti_stencil, geometry)
    parallel_advection_Ti = Vi * grad_parallel_op_direct(Ti_stencil, geometry)
    curvature_Pe = curvature_op(
        Pe_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_Ti = curvature_op(
        Ti_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    curvature_potential = curvature_op(
        potential_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    parallel_current_density = grad_parallel_op_direct(current_density_stencil, geometry)
    parallel_Vi = grad_parallel_op_direct(Vi_stencil, geometry)
    perp_diffusion_Ti = jnp.zeros_like(Ti)
    if float(D_perp) != 0.0:
        if Ti_conservative_stencil is None:
            raise ValueError("Ti_conservative_stencil is required when ion_temperature_D_perp is nonzero")
        perp_diffusion_Ti = D_perp * perp_laplacian_conservative_op(
            Ti_conservative_stencil,
            geometry,
            face_bc=Ti_face_bc,
            cut_wall_geometry=Ti_cut_wall_geometry,
            cut_wall_bc=Ti_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_Ti = jnp.zeros_like(Ti)
    if float(chi_parallel) != 0.0:
        parallel_diffusion_Ti = chi_parallel * parallel_laplacian_direct_op(
            Ti,
            geometry,
            face_bc=Ti_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_Ti / (rho_star * bmag))
        - parallel_advection_Ti
        + (4.0 * Ti / (3.0 * bmag))
        * (
            curvature_Pe / density
            - 2.5 * tau * curvature_Ti
            - curvature_potential
        )
        + (2.0 * Ti / (3.0 * density))
        * (
            parallel_current_density - density * parallel_Vi
        )
        + parallel_diffusion_Ti
        + perp_diffusion_Ti
    )


def _Ve_rhs(
    *,
    Ve: jnp.ndarray,
    density: jnp.ndarray,
    Te: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    Ve_stencil: LocalStencil3D,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    Pe_stencil: LocalStencil3D,
    Te_stencil: LocalStencil3D,
    current_density: jnp.ndarray,
    Ve_conservative_stencil: ConservativeStencil3D | None,
    Ve_face_bc: BoundaryFaceBC3D,
    Ve_cut_wall_geometry: CutWallGeometry3D | None,
    Ve_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the electron velocity RHS with prebuilt stencils and geometry inputs."""

    Ve = jnp.asarray(Ve, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    Te = jnp.asarray(Te, dtype=jnp.float64)
    current_density = jnp.asarray(current_density, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    Ve_nu = jnp.asarray(parameters.Ve_nu, dtype=jnp.float64)
    Ve_D_perp = jnp.asarray(parameters.Ve_D_perp, dtype=jnp.float64)
    Ve_parallel_viscosity = jnp.asarray(parameters.Ve_parallel_viscosity, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_Ve = poisson_bracket_op(potential_stencil, Ve_stencil, geometry)
    parallel_advection_Ve = Ve * grad_parallel_op_direct(Ve_stencil, geometry)
    parallel_phi = grad_parallel_op_direct(potential_stencil, geometry)
    parallel_Pe = grad_parallel_op_direct(Pe_stencil, geometry)
    parallel_Te = grad_parallel_op_direct(Te_stencil, geometry)
    perp_diffusion_Ve = jnp.zeros_like(Ve)
    if float(Ve_D_perp) != 0.0:
        if Ve_conservative_stencil is None:
            raise ValueError("Ve_conservative_stencil is required when Ve_D_perp is nonzero")
        perp_diffusion_Ve = Ve_D_perp * perp_laplacian_conservative_op(
            Ve_conservative_stencil,
            geometry,
            face_bc=Ve_face_bc,
            cut_wall_geometry=Ve_cut_wall_geometry,
            cut_wall_bc=Ve_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_Ve = jnp.zeros_like(Ve)
    if float(Ve_parallel_viscosity) != 0.0:
        parallel_diffusion_Ve = Ve_parallel_viscosity * parallel_laplacian_direct_op(
            Ve,
            geometry,
            face_bc=Ve_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_Ve / (rho_star * bmag))
        - parallel_advection_Ve
        + mi_over_me
        * (
            Ve_nu * current_density
            + parallel_phi
            - parallel_Pe / jnp.maximum(density, 1.0e-30)
            - 0.71 * parallel_Te
        )
        + parallel_diffusion_Ve
        + perp_diffusion_Ve
    )


def _Vi_rhs(
    *,
    Vi: jnp.ndarray,
    density: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    Vi_stencil: LocalStencil3D,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    pressure_stencil: LocalStencil3D,
    Vi_conservative_stencil: ConservativeStencil3D | None,
    Vi_face_bc: BoundaryFaceBC3D,
    Vi_cut_wall_geometry: CutWallGeometry3D | None,
    Vi_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the ion velocity RHS with prebuilt stencils and geometry inputs."""

    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    Vi_D_perp = jnp.asarray(parameters.Vi_D_perp, dtype=jnp.float64)
    Vi_parallel_viscosity = jnp.asarray(parameters.Vi_parallel_viscosity, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_Vi = poisson_bracket_op(potential_stencil, Vi_stencil, geometry)
    parallel_advection_Vi = Vi * grad_parallel_op_direct(Vi_stencil, geometry)
    parallel_pressure = grad_parallel_op_direct(pressure_stencil, geometry)
    perp_diffusion_Vi = jnp.zeros_like(Vi)
    if float(Vi_D_perp) != 0.0:
        if Vi_conservative_stencil is None:
            raise ValueError("Vi_conservative_stencil is required when Vi_D_perp is nonzero")
        perp_diffusion_Vi = Vi_D_perp * perp_laplacian_conservative_op(
            Vi_conservative_stencil,
            geometry,
            face_bc=Vi_face_bc,
            cut_wall_geometry=Vi_cut_wall_geometry,
            cut_wall_bc=Vi_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_Vi = jnp.zeros_like(Vi)
    if float(Vi_parallel_viscosity) != 0.0:
        parallel_diffusion_Vi = Vi_parallel_viscosity * parallel_laplacian_direct_op(
            Vi,
            geometry,
            face_bc=Vi_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_Vi / (rho_star * bmag))
        - parallel_advection_Vi
        - parallel_pressure / jnp.maximum(density, 1.0e-30)
        + parallel_diffusion_Vi
        + perp_diffusion_Vi
    )


def _vorticity_rhs(
    *,
    vorticity: jnp.ndarray,
    density: jnp.ndarray,
    Vi: jnp.ndarray,
    geometry: FciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients: jnp.ndarray,
    vorticity_stencil: LocalStencil3D,
    density_stencil: LocalStencil3D,
    potential_stencil: LocalStencil3D,
    pressure_stencil: LocalStencil3D,
    current_density_stencil: LocalStencil3D,
    vorticity_conservative_stencil: ConservativeStencil3D | None,
    vorticity_face_bc: BoundaryFaceBC3D,
    vorticity_cut_wall_geometry: CutWallGeometry3D | None,
    vorticity_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Assemble the vorticity RHS with prebuilt stencils and geometry inputs."""

    vorticity = jnp.asarray(vorticity, dtype=jnp.float64)
    density = jnp.asarray(density, dtype=jnp.float64)
    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    vorticity_D_perp = jnp.asarray(parameters.vorticity_D_perp, dtype=jnp.float64)
    vorticity_D_parallel = jnp.asarray(parameters.vorticity_D_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))

    poisson_bracket_vorticity = poisson_bracket_op(potential_stencil, vorticity_stencil, geometry)
    parallel_advection_vorticity = Vi * grad_parallel_op_direct(vorticity_stencil, geometry)
    parallel_current_density = grad_parallel_op_direct(current_density_stencil, geometry)
    curvature_pressure = curvature_op(
        pressure_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    perp_diffusion_vorticity = jnp.zeros_like(vorticity)
    if float(vorticity_D_perp) != 0.0:
        if vorticity_conservative_stencil is None:
            raise ValueError("vorticity_conservative_stencil is required when vorticity_D_perp is nonzero")
        perp_diffusion_vorticity = vorticity_D_perp * perp_laplacian_conservative_op(
            vorticity_conservative_stencil,
            geometry,
            face_bc=vorticity_face_bc,
            cut_wall_geometry=vorticity_cut_wall_geometry,
            cut_wall_bc=vorticity_cut_wall_bc,
            periodic_axes=periodic_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
        )
    parallel_diffusion_vorticity = jnp.zeros_like(vorticity)
    if float(vorticity_D_parallel) != 0.0:
        parallel_diffusion_vorticity = vorticity_D_parallel * parallel_laplacian_direct_op(
            vorticity,
            geometry,
            face_bc=vorticity_face_bc,
            periodic_axes=periodic_axes,
        )

    return (
        -(poisson_bracket_vorticity / (rho_star * bmag))
        - parallel_advection_vorticity
        + (bmag * bmag / jnp.maximum(density, 1.0e-30)) * parallel_current_density
        + (2.0 * bmag / jnp.maximum(density, 1.0e-30)) * curvature_pressure
        + parallel_diffusion_vorticity
        + perp_diffusion_vorticity
    )


def _diffusion_only_rhs(
    *,
    density: jnp.ndarray,
    Te: jnp.ndarray,
    Ti: jnp.ndarray,
    Vi: jnp.ndarray,
    Ve: jnp.ndarray,
    vorticity: jnp.ndarray,
    geometry: FciGeometry3D,
    conservative_stencil_builder: ConservativeStencilBuilder,
    parameters: FciDrbEBRhsParameters,
    density_face_bc: BoundaryFaceBC3D,
    vorticity_face_bc: BoundaryFaceBC3D,
    electron_temperature_face_bc: BoundaryFaceBC3D,
    ion_temperature_face_bc: BoundaryFaceBC3D,
    electron_velocity_parallel_face_bc: BoundaryFaceBC3D,
    ion_velocity_parallel_face_bc: BoundaryFaceBC3D,
    density_cut_wall_geometry: CutWallGeometry3D | None,
    density_cut_wall_bc: CutWallBC3D | None,
    vorticity_cut_wall_geometry: CutWallGeometry3D | None,
    vorticity_cut_wall_bc: CutWallBC3D | None,
    electron_temperature_cut_wall_geometry: CutWallGeometry3D | None,
    electron_temperature_cut_wall_bc: CutWallBC3D | None,
    ion_temperature_cut_wall_geometry: CutWallGeometry3D | None,
    ion_temperature_cut_wall_bc: CutWallBC3D | None,
    electron_velocity_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    electron_velocity_parallel_cut_wall_bc: CutWallBC3D | None,
    ion_velocity_parallel_cut_wall_geometry: CutWallGeometry3D | None,
    ion_velocity_parallel_cut_wall_bc: CutWallBC3D | None,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> FciDrbEBState:
    """Assemble only the explicit field diffusion terms for EB sanity checks."""

    density = jnp.asarray(density, dtype=jnp.float64)
    Te = jnp.asarray(Te, dtype=jnp.float64)
    Ti = jnp.asarray(Ti, dtype=jnp.float64)
    Vi = jnp.asarray(Vi, dtype=jnp.float64)
    Ve = jnp.asarray(Ve, dtype=jnp.float64)
    vorticity = jnp.asarray(vorticity, dtype=jnp.float64)

    def _field_diffusion(
        field: jnp.ndarray,
        *,
        perpendicular_coefficient: float,
        parallel_coefficient: float,
        face_bc: BoundaryFaceBC3D,
        cut_wall_geometry: CutWallGeometry3D | None,
        cut_wall_bc: CutWallBC3D | None,
    ) -> jnp.ndarray:
        rhs = jnp.zeros_like(field, dtype=jnp.float64)
        if float(perpendicular_coefficient) != 0.0:
            conservative_stencil = conservative_stencil_builder(
                field,
                geometry,
                periodic_axes=periodic_axes,
                face_bc=face_bc,
            )
            rhs = rhs + jnp.asarray(perpendicular_coefficient, dtype=jnp.float64) * perp_laplacian_conservative_op(
                conservative_stencil,
                geometry,
                face_bc=face_bc,
                cut_wall_geometry=cut_wall_geometry,
                cut_wall_bc=cut_wall_bc,
                periodic_axes=periodic_axes,
                b_floor=b_floor,
                jacobian_floor=jacobian_floor,
            )
        if float(parallel_coefficient) != 0.0:
            rhs = rhs + jnp.asarray(parallel_coefficient, dtype=jnp.float64) * parallel_laplacian_direct_op(
                field,
                geometry,
                face_bc=face_bc,
                periodic_axes=periodic_axes,
            )
        return rhs

    return FciDrbEBState(
        density=_field_diffusion(
            density,
            perpendicular_coefficient=parameters.density_D_perp,
            parallel_coefficient=parameters.density_D_parallel,
            face_bc=density_face_bc,
            cut_wall_geometry=density_cut_wall_geometry,
            cut_wall_bc=density_cut_wall_bc,
        ),
        phi=jnp.zeros_like(density, dtype=jnp.float64),
        Te=_field_diffusion(
            Te,
            perpendicular_coefficient=parameters.electron_temperature_D_perp,
            parallel_coefficient=parameters.electron_temperature_chi_parallel,
            face_bc=electron_temperature_face_bc,
            cut_wall_geometry=electron_temperature_cut_wall_geometry,
            cut_wall_bc=electron_temperature_cut_wall_bc,
        ),
        Ti=_field_diffusion(
            Ti,
            perpendicular_coefficient=parameters.ion_temperature_D_perp,
            parallel_coefficient=parameters.ion_temperature_chi_parallel,
            face_bc=ion_temperature_face_bc,
            cut_wall_geometry=ion_temperature_cut_wall_geometry,
            cut_wall_bc=ion_temperature_cut_wall_bc,
        ),
        Vi=_field_diffusion(
            Vi,
            perpendicular_coefficient=parameters.Vi_D_perp,
            parallel_coefficient=parameters.Vi_parallel_viscosity,
            face_bc=ion_velocity_parallel_face_bc,
            cut_wall_geometry=ion_velocity_parallel_cut_wall_geometry,
            cut_wall_bc=ion_velocity_parallel_cut_wall_bc,
        ),
        Ve=_field_diffusion(
            Ve,
            perpendicular_coefficient=parameters.Ve_D_perp,
            parallel_coefficient=parameters.Ve_parallel_viscosity,
            face_bc=electron_velocity_parallel_face_bc,
            cut_wall_geometry=electron_velocity_parallel_cut_wall_geometry,
            cut_wall_bc=electron_velocity_parallel_cut_wall_bc,
        ),
        vorticity=_field_diffusion(
            vorticity,
            perpendicular_coefficient=parameters.vorticity_D_perp,
            parallel_coefficient=parameters.vorticity_D_parallel,
            face_bc=vorticity_face_bc,
            cut_wall_geometry=vorticity_cut_wall_geometry,
            cut_wall_bc=vorticity_cut_wall_bc,
        ),
    )


def compute_fci_drb_eb_rhs(
    state: FciDrbEBState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    parameters: FciDrbEBRhsParameters = FciDrbEBRhsParameters(),
    curvature_coefficients: jnp.ndarray,
    density_face_bc: BoundaryFaceBC3D,
    potential_face_bc: BoundaryFaceBC3D,
    vorticity_face_bc: BoundaryFaceBC3D,
    electron_temperature_face_bc: BoundaryFaceBC3D,
    electron_velocity_parallel_face_bc: BoundaryFaceBC3D,
    ion_temperature_face_bc: BoundaryFaceBC3D | None = None,
    ion_velocity_parallel_face_bc: BoundaryFaceBC3D | None = None,
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    potential_cut_wall_geometry: CutWallGeometry3D | None = None,
    potential_cut_wall_bc: CutWallBC3D | None = None,
    vorticity_cut_wall_geometry: CutWallGeometry3D | None = None,
    vorticity_cut_wall_bc: CutWallBC3D | None = None,
    electron_temperature_cut_wall_geometry: CutWallGeometry3D | None = None,
    electron_temperature_cut_wall_bc: CutWallBC3D | None = None,
    ion_temperature_cut_wall_geometry: CutWallGeometry3D | None = None,
    ion_temperature_cut_wall_bc: CutWallBC3D | None = None,
    electron_velocity_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    electron_velocity_parallel_cut_wall_bc: CutWallBC3D | None = None,
    ion_velocity_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    ion_velocity_parallel_cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    diffusion_only: bool = False,
    density_source: jax.Array | None = None,
    electron_temperature_source: jax.Array | None = None,
) -> FciDrbEBRhsResult:
    """Assemble the electrostatic Boussinesq DRB RHS.

    Optional density and electron-temperature source terms are added only when
    ``diffusion_only`` is false.
    """

    density = jnp.asarray(state.density, dtype=jnp.float64)
    phi_state = jnp.asarray(state.phi, dtype=jnp.float64)
    vorticity = jnp.asarray(state.vorticity, dtype=jnp.float64)
    Te = jnp.asarray(state.Te, dtype=jnp.float64)
    Ti = jnp.asarray(state.Ti, dtype=jnp.float64)
    Vi = jnp.asarray(state.Vi, dtype=jnp.float64)
    Ve = jnp.asarray(state.Ve, dtype=jnp.float64)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    ion_temperature_face_bc = (
        electron_temperature_face_bc
        if ion_temperature_face_bc is None
        else ion_temperature_face_bc
    )
    ion_temperature_cut_wall_geometry = (
        electron_temperature_cut_wall_geometry
        if ion_temperature_cut_wall_geometry is None
        else ion_temperature_cut_wall_geometry
    )
    ion_temperature_cut_wall_bc = (
        electron_temperature_cut_wall_bc
        if ion_temperature_cut_wall_bc is None
        else ion_temperature_cut_wall_bc
    )
    ion_velocity_parallel_face_bc = (
        electron_velocity_parallel_face_bc
        if ion_velocity_parallel_face_bc is None
        else ion_velocity_parallel_face_bc
    )
    ion_velocity_parallel_cut_wall_geometry = (
        electron_velocity_parallel_cut_wall_geometry
        if ion_velocity_parallel_cut_wall_geometry is None
        else ion_velocity_parallel_cut_wall_geometry
    )
    ion_velocity_parallel_cut_wall_bc = (
        electron_velocity_parallel_cut_wall_bc
        if ion_velocity_parallel_cut_wall_bc is None
        else ion_velocity_parallel_cut_wall_bc
    )
    # These coefficients are Python scalars in the current simulation path, so we can
    # skip zero-diffusion branches eagerly instead of building Laplacians that multiply away.
    density_has_perp_diffusion = float(parameters.density_D_perp) != 0.0
    temperature_has_perp_diffusion = float(parameters.electron_temperature_D_perp) != 0.0
    ti_has_perp_diffusion = float(parameters.ion_temperature_D_perp) != 0.0
    ve_has_perp_diffusion = float(parameters.Ve_D_perp) != 0.0
    vi_has_perp_diffusion = float(parameters.Vi_D_perp) != 0.0
    vorticity_has_perp_diffusion = float(parameters.vorticity_D_perp) != 0.0

    if bool(diffusion_only):
        rhs = _diffusion_only_rhs(
            density=density,
            Te=Te,
            Ti=Ti,
            Vi=Vi,
            Ve=Ve,
            vorticity=vorticity,
            geometry=geometry,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters,
            density_face_bc=density_face_bc,
            vorticity_face_bc=vorticity_face_bc,
            electron_temperature_face_bc=electron_temperature_face_bc,
            ion_temperature_face_bc=ion_temperature_face_bc,
            electron_velocity_parallel_face_bc=electron_velocity_parallel_face_bc,
            ion_velocity_parallel_face_bc=ion_velocity_parallel_face_bc,
            density_cut_wall_geometry=density_cut_wall_geometry,
            density_cut_wall_bc=density_cut_wall_bc,
            vorticity_cut_wall_geometry=vorticity_cut_wall_geometry,
            vorticity_cut_wall_bc=vorticity_cut_wall_bc,
            electron_temperature_cut_wall_geometry=electron_temperature_cut_wall_geometry,
            electron_temperature_cut_wall_bc=electron_temperature_cut_wall_bc,
            ion_temperature_cut_wall_geometry=ion_temperature_cut_wall_geometry,
            ion_temperature_cut_wall_bc=ion_temperature_cut_wall_bc,
            electron_velocity_parallel_cut_wall_geometry=electron_velocity_parallel_cut_wall_geometry,
            electron_velocity_parallel_cut_wall_bc=electron_velocity_parallel_cut_wall_bc,
            ion_velocity_parallel_cut_wall_geometry=ion_velocity_parallel_cut_wall_geometry,
            ion_velocity_parallel_cut_wall_bc=ion_velocity_parallel_cut_wall_bc,
            periodic_axes=periodic_axes,
        )
        return FciDrbEBRhsResult(
            rhs=rhs,
            potential=phi_state,
            potential_residual_l2=jnp.asarray(0.0, dtype=jnp.float64),
        )

    density_stencil = stencil_builder(
        density,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=density_face_bc,
        cut_wall_geometry=density_cut_wall_geometry,
        cut_wall_bc=density_cut_wall_bc,
    )
    Pe = density * Te
    Pe_stencil = stencil_builder(
        Pe,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=electron_temperature_face_bc,
        cut_wall_geometry=electron_temperature_cut_wall_geometry,
        cut_wall_bc=electron_temperature_cut_wall_bc,
    )
    Ve_stencil = stencil_builder(
        Ve,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=electron_velocity_parallel_face_bc,
        cut_wall_geometry=electron_velocity_parallel_cut_wall_geometry,
        cut_wall_bc=electron_velocity_parallel_cut_wall_bc,
    )
    Vi_stencil = stencil_builder(
        Vi,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=ion_velocity_parallel_face_bc,
        cut_wall_geometry=ion_velocity_parallel_cut_wall_geometry,
        cut_wall_bc=ion_velocity_parallel_cut_wall_bc,
    )
    Vi_conservative_stencil = (
        conservative_stencil_builder(
            Vi,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=ion_velocity_parallel_face_bc,
        )
        if vi_has_perp_diffusion
        else None
    )
    temperature_stencil = stencil_builder(
        Te,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=electron_temperature_face_bc,
        cut_wall_geometry=electron_temperature_cut_wall_geometry,
        cut_wall_bc=electron_temperature_cut_wall_bc,
    )
    current_density = density * (Vi - Ve)
    current_density_stencil = stencil_builder(
        current_density,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=electron_velocity_parallel_face_bc,
        cut_wall_geometry=electron_velocity_parallel_cut_wall_geometry,
        cut_wall_bc=electron_velocity_parallel_cut_wall_bc,
    )
    temperature_conservative_stencil = (
        conservative_stencil_builder(
            Te,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=electron_temperature_face_bc,
        )
        if temperature_has_perp_diffusion
        else None
    )
    Ti_stencil = stencil_builder(
        Ti,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=ion_temperature_face_bc,
        cut_wall_geometry=ion_temperature_cut_wall_geometry,
        cut_wall_bc=ion_temperature_cut_wall_bc,
    )
    pressure = Pe + parameters.tau * (density * Ti)
    pressure_stencil = stencil_builder(
        pressure,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=electron_temperature_face_bc,
        cut_wall_geometry=electron_temperature_cut_wall_geometry,
        cut_wall_bc=electron_temperature_cut_wall_bc,
    )
    Ti_conservative_stencil = (
        conservative_stencil_builder(
            Ti,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=ion_temperature_face_bc,
        )
        if ti_has_perp_diffusion
        else None
    )
    potential = phi_state
    potential_residual_l2 = jnp.asarray(0.0, dtype=jnp.float64)
    potential_stencil = stencil_builder(
        potential,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=potential_face_bc,
        cut_wall_geometry=potential_cut_wall_geometry,
        cut_wall_bc=potential_cut_wall_bc,
    )
    vorticity_stencil = stencil_builder(
        vorticity,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=vorticity_face_bc,
        cut_wall_geometry=vorticity_cut_wall_geometry,
        cut_wall_bc=vorticity_cut_wall_bc,
    )
    Ve_conservative_stencil = (
        conservative_stencil_builder(
            Ve,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=electron_velocity_parallel_face_bc,
        )
        if ve_has_perp_diffusion
        else None
    )
    density_conservative_stencil = (
        conservative_stencil_builder(
            density,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=density_face_bc,
        )
        if density_has_perp_diffusion
        else None
    )
    vorticity_conservative_stencil = (
        conservative_stencil_builder(
            vorticity,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=vorticity_face_bc,
        )
        if vorticity_has_perp_diffusion
        else None
    )

    density_rhs = _density_rhs(
        density=density,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        Pe_stencil=Pe_stencil,
        Ve_stencil=Ve_stencil,
        density_conservative_stencil=density_conservative_stencil,
        density_face_bc=density_face_bc,
        density_cut_wall_geometry=density_cut_wall_geometry,
        density_cut_wall_bc=density_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    if density_source is not None:
        density_rhs = density_rhs + jnp.asarray(density_source, dtype=jnp.float64)
    Te_rhs = _Te_rhs(
        Te=Te,
        density=density,
        Vi=Vi,
        Ve=Ve,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        temperature_stencil=temperature_stencil,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        Pe_stencil=Pe_stencil,
        current_density_stencil=current_density_stencil,
        Ve_stencil=Ve_stencil,
        temperature_conservative_stencil=temperature_conservative_stencil,
        temperature_face_bc=electron_temperature_face_bc,
        temperature_cut_wall_geometry=electron_temperature_cut_wall_geometry,
        temperature_cut_wall_bc=electron_temperature_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    if electron_temperature_source is not None:
        Te_rhs = Te_rhs + jnp.asarray(electron_temperature_source, dtype=jnp.float64)
    Ti_rhs = _Ti_rhs(
        Ti=Ti,
        density=density,
        Vi=Vi,
        Ve=Ve,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        Ti_stencil=Ti_stencil,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        Pe_stencil=Pe_stencil,
        current_density_stencil=current_density_stencil,
        Vi_stencil=Vi_stencil,
        Ti_conservative_stencil=Ti_conservative_stencil,
        Ti_face_bc=ion_temperature_face_bc,
        Ti_cut_wall_geometry=ion_temperature_cut_wall_geometry,
        Ti_cut_wall_bc=ion_temperature_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    Vi_rhs = _Vi_rhs(
        Vi=Vi,
        density=density,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        Vi_stencil=Vi_stencil,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        pressure_stencil=pressure_stencil,
        Vi_conservative_stencil=Vi_conservative_stencil,
        Vi_face_bc=ion_velocity_parallel_face_bc,
        Vi_cut_wall_geometry=ion_velocity_parallel_cut_wall_geometry,
        Vi_cut_wall_bc=ion_velocity_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    vorticity_rhs = _vorticity_rhs(
        vorticity=vorticity,
        density=density,
        Vi=Vi,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        vorticity_stencil=vorticity_stencil,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        pressure_stencil=pressure_stencil,
        current_density_stencil=current_density_stencil,
        vorticity_conservative_stencil=vorticity_conservative_stencil,
        vorticity_face_bc=vorticity_face_bc,
        vorticity_cut_wall_geometry=vorticity_cut_wall_geometry,
        vorticity_cut_wall_bc=vorticity_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    Ve_rhs = _Ve_rhs(
        Ve=Ve,
        density=density,
        Te=Te,
        geometry=geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        Ve_stencil=Ve_stencil,
        density_stencil=density_stencil,
        potential_stencil=potential_stencil,
        Pe_stencil=Pe_stencil,
        Te_stencil=temperature_stencil,
        current_density=current_density,
        Ve_conservative_stencil=Ve_conservative_stencil,
        Ve_face_bc=electron_velocity_parallel_face_bc,
        Ve_cut_wall_geometry=electron_velocity_parallel_cut_wall_geometry,
        Ve_cut_wall_bc=electron_velocity_parallel_cut_wall_bc,
        periodic_axes=periodic_axes,
    )

    rhs = FciDrbEBState(
        density=density_rhs,
        phi=phi_state,
        Te=Te_rhs,
        Ti=Ti_rhs,
        Vi=Vi_rhs,
        Ve=Ve_rhs,
        vorticity=vorticity_rhs,
    )

    return FciDrbEBRhsResult(
        rhs=rhs,
        potential=potential,
        potential_residual_l2=potential_residual_l2,
    )
