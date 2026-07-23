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
    BC_DIRICHLET,
    BC_NONE,
    BoundaryFaceBC3D,
    ConservativeStencil3D,
    CutWallBC3D,
    CutWallGeometry3D,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeBoundaryBC3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalStencil1D,
    LocalStencil3D,
)
from .fci_halo import (
    HaloExchange3D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from .fci_operators import (
    LocalPerpLaplacianInverseSolver,
    build_local_control_volume_field_closure,
    build_local_control_volume_polynomial_from_field,
    curvature_op,
    expand_local_control_volume_owner_field,
    local_control_volume_product_average,
    grad_parallel_op_direct,
    local_curvature_op,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_from_gradient,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_perp_laplacian_conservative_op,
    local_poisson_bracket_op,
    local_poisson_bracket_op_from_gradients,
    _mask_inactive_owned,
    _mask_state_inactive_owned,
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


def _mask_local_eb_state_inactive(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
) -> FciDrbEBState:
    """Zero inactive owned cells for a local EB state/update payload."""

    return _mask_state_inactive_owned(state, geometry)


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


@dataclass(frozen=True)
class LocalFciDrbEBControlVolumeBCBundle:
    """Field-specific compact boundary data for the unified EB path."""

    density: LocalControlVolumeBoundaryBC3D
    phi: LocalControlVolumeBoundaryBC3D
    Te: LocalControlVolumeBoundaryBC3D
    Ti: LocalControlVolumeBoundaryBC3D
    Vi: LocalControlVolumeBoundaryBC3D
    Ve: LocalControlVolumeBoundaryBC3D
    vorticity: LocalControlVolumeBoundaryBC3D


LocalFciDrbEBFaceBCBuilder = Callable[
    [FciDrbEBState, LocalFciGeometry3D, LocalDomain3D, FciDrbEBRhsParameters],
    LocalFciDrbEBFaceBCBundle,
]
LocalFciDrbEBControlVolumeBCBuilder = Callable[
    [
        FciDrbEBState,
        LocalFciGeometry3D,
        LocalDomain3D,
        FciDrbEBRhsParameters,
        LocalEmbeddedControlVolumeGeometry3D,
    ],
    LocalFciDrbEBControlVolumeBCBundle,
]


def _binary_control_volume_dirichlet_bc(
    left: LocalControlVolumeBoundaryBC3D,
    right: LocalControlVolumeBoundaryBC3D,
    operation: Callable[[jax.Array, jax.Array], jax.Array],
) -> LocalControlVolumeBoundaryBC3D:
    """Combine two collocated Dirichlet payloads for a derived scalar field."""

    if (
        left.max_rows != right.max_rows
        or left.max_patches != right.max_patches
    ):
        raise ValueError("control-volume BC operands must share row layout")
    is_dirichlet = (
        left.active
        & right.active
        & (left.kind == BC_DIRICHLET)
        & (right.kind == BC_DIRICHLET)
    )
    return LocalControlVolumeBoundaryBC3D(
        kind=jnp.where(is_dirichlet, BC_DIRICHLET, BC_NONE),
        centroid_value=operation(
            left.centroid_value,
            right.centroid_value,
        ),
        quadrature_value=operation(
            left.quadrature_value,
            right.quadrature_value,
        ),
        active=is_dirichlet,
        max_rows=left.max_rows,
        max_patches=left.max_patches,
    )


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


def _scale_control_volume_dirichlet_bc(
    boundary_bc: LocalControlVolumeBoundaryBC3D,
    scale: float | jax.Array,
) -> LocalControlVolumeBoundaryBC3D:
    scale_value = jnp.asarray(scale, dtype=jnp.float64)
    return LocalControlVolumeBoundaryBC3D(
        kind=boundary_bc.kind,
        centroid_value=scale_value * boundary_bc.centroid_value,
        quadrature_value=scale_value * boundary_bc.quadrature_value,
        active=boundary_bc.active,
        max_rows=boundary_bc.max_rows,
        max_patches=boundary_bc.max_patches,
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
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D
    control_volume_bc_builder: LocalFciDrbEBControlVolumeBCBuilder
    diffusion_only: bool = False
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)

    def _face_bcs(self, state_owned: FciDrbEBState) -> LocalFciDrbEBFaceBCBundle:
        return self.face_bc_builder(
            state_owned,
            self.geometry,
            self.domain,
            self.parameters,
        )

    def _control_volume_bcs(
        self,
        state_owned: FciDrbEBState,
    ) -> LocalFciDrbEBControlVolumeBCBundle:
        return self.control_volume_bc_builder(
            state_owned,
            self.geometry,
            self.domain,
            self.parameters,
            self.control_volume_geometry,
        )

    def _expand_control_volume_state(
        self,
        state_owned: FciDrbEBState,
    ) -> FciDrbEBState:
        cells = self.control_volume_geometry.cells
        return FciDrbEBState(
            **{
                field_name: expand_local_control_volume_owner_field(
                    getattr(state_owned, field_name),
                    cells,
                    owner_values_halo=self.halo_exchange(
                        inject_owned_field_to_halo(getattr(state_owned, field_name), self.domain.layout), self.domain
                    ),
                )
                for field_name in (
                    "density",
                    "phi",
                    "Te",
                    "Ti",
                    "Vi",
                    "Ve",
                    "vorticity",
                )
            }
        )

    def _control_volume_polynomial(
        self,
        field_halo: jnp.ndarray,
        boundary_bc: LocalControlVolumeBoundaryBC3D,
        regular_face_bc: LocalBoundaryFaceBC3D,
    ):
        return build_local_control_volume_polynomial_from_field(
            field_halo,
            self.geometry,
            self.domain,
            StencilBuilderContext(
                layout=self.domain.layout,
                domain=self.domain,
            ),
            self.control_volume_geometry,
            boundary_bc,
            regular_face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )

    def _prepare_scalar_halo(
        self,
        values_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        owner_halo = self.halo_exchange(
            inject_owned_field_to_halo(values_owned, self.domain.layout), self.domain
        )
        storage = expand_local_control_volume_owner_field(
            values_owned, self.control_volume_geometry.cells, owner_values_halo=owner_halo,
        )
        field_halo = inject_owned_field_to_halo(
            storage,
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

    def _field_perp_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
        control_volume_bc: LocalControlVolumeBoundaryBC3D,
    ) -> jnp.ndarray:
        if float(coefficient) == 0.0:
            return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        conservative = build_local_conservative_stencil_from_field(
            field_halo,
            self.geometry,
            context,
        )
        field_closure = (
            None
            if control_volume_bc is None
            else build_local_control_volume_field_closure(
                field_halo,
                self.control_volume_geometry,
                control_volume_bc,
            )
        )
        return jnp.asarray(coefficient, dtype=jnp.float64) * local_perp_laplacian_conservative_op(
            conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc,
            regular_face_geometry=self.geometry.regular_face_geometry,
            control_volume_geometry=self.control_volume_geometry,
            field_closure=field_closure,
            axis_regular_axes=self.axis_regular_axes,
        )

    def _field_parallel_diffusion(
        self,
        field_halo: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
        coefficient: float,
        control_volume_bc: LocalControlVolumeBoundaryBC3D,
    ) -> jnp.ndarray:
        if float(coefficient) == 0.0:
            return jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        conservative = build_local_conservative_stencil_from_field(
            field_halo,
            self.geometry,
            context,
        )
        field_closure = (
            None
            if control_volume_bc is None
            else build_local_control_volume_field_closure(
                field_halo,
                self.control_volume_geometry,
                control_volume_bc,
            )
        )
        return jnp.asarray(coefficient, dtype=jnp.float64) * local_parallel_laplacian_conservative_op(
            conservative,
            self.geometry,
            self.domain,
            face_bc=face_bc,
            regular_face_geometry=self.geometry.regular_face_geometry,
            control_volume_geometry=self.control_volume_geometry,
            field_closure=field_closure,
            axis_regular_axes=self.axis_regular_axes,
        )

    def _reconstruct_phi_from_prepared(
        self,
        state_owned: FciDrbEBState,
        state_halo: FciDrbEBState,
        face_bc: LocalFciDrbEBFaceBCBundle,
        control_volume_bc: LocalFciDrbEBControlVolumeBCBundle,
    ) -> jnp.ndarray:
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        ti_conservative = build_local_conservative_stencil_from_field(
            state_halo.Ti,
            self.geometry,
            context,
        )
        ti_field_closure = build_local_control_volume_field_closure(
            state_halo.Ti,
            self.control_volume_geometry,
            control_volume_bc.Ti,
        )
        ti_laplacian = local_perp_laplacian_conservative_op(
            ti_conservative,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
            control_volume_geometry=self.control_volume_geometry,
            field_closure=ti_field_closure,
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
            control_volume_geometry=self.control_volume_geometry,
            control_volume_boundary_bc=control_volume_bc.phi,
            face_bc=face_bc.phi,
            axis_regular_axes=self.axis_regular_axes,
            config=self.gmres_config,
        )
        phi_owned = solver(
            phi_rhs,
            guess_owned=state_owned.phi,
            phi_lift_owned=phi_lift,
        )
        return _mask_inactive_owned(phi_owned, self.geometry)

    def reconstruct_phi(self, state_owned: FciDrbEBState) -> jnp.ndarray:
        face_bc = self._face_bcs(state_owned)
        control_volume_bc = self._control_volume_bcs(state_owned)
        state_halo = prepare_local_fci_drb_eb_state(
            self._expand_control_volume_state(state_owned),
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
            control_volume_bc,
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
        source_owned = _mask_local_eb_state_inactive(source_owned, self.geometry)
        face_bc = self._face_bcs(state_owned)
        control_volume_bc = self._control_volume_bcs(state_owned)
        state_halo_without_phi = prepare_local_fci_drb_eb_state(
            self._expand_control_volume_state(state_owned),
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
            control_volume_bc,
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
        if control_volume_bc is not None:
            density_polynomial = self._control_volume_polynomial(
                state_halo.density,
                control_volume_bc.density,
                face_bc.density,
            )
            phi_polynomial = self._control_volume_polynomial(
                state_halo.phi,
                control_volume_bc.phi,
                face_bc.phi,
            )
            Te_polynomial = self._control_volume_polynomial(
                state_halo.Te,
                control_volume_bc.Te,
                face_bc.Te,
            )
            Ti_polynomial = self._control_volume_polynomial(
                state_halo.Ti,
                control_volume_bc.Ti,
                face_bc.Ti,
            )
            Vi_polynomial = self._control_volume_polynomial(
                state_halo.Vi,
                control_volume_bc.Vi,
                face_bc.Vi,
            )
            Ve_polynomial = self._control_volume_polynomial(
                state_halo.Ve,
                control_volume_bc.Ve,
                face_bc.Ve,
            )
            vorticity_polynomial = self._control_volume_polynomial(
                state_halo.vorticity,
                control_volume_bc.vorticity,
                face_bc.vorticity,
            )
            Pe_control_volume_bc = _binary_control_volume_dirichlet_bc(
                control_volume_bc.density,
                control_volume_bc.Te,
                lambda left, right: left * right,
            )
            ion_pressure_control_volume_bc = (
                _binary_control_volume_dirichlet_bc(
                    control_volume_bc.density,
                    control_volume_bc.Ti,
                    lambda left, right: left * right,
                )
            )
            pressure_control_volume_bc = (
                _binary_control_volume_dirichlet_bc(
                    Pe_control_volume_bc,
                    _scale_control_volume_dirichlet_bc(
                        ion_pressure_control_volume_bc,
                        self.parameters.tau,
                    ),
                    lambda left, right: left + right,
                )
            )
            velocity_difference_control_volume_bc = (
                _binary_control_volume_dirichlet_bc(
                    control_volume_bc.Vi,
                    control_volume_bc.Ve,
                    lambda left, right: left - right,
                )
            )
            current_control_volume_bc = _binary_control_volume_dirichlet_bc(
                control_volume_bc.density,
                velocity_difference_control_volume_bc,
                lambda left, right: left * right,
            )
            density_flux_control_volume_bc = (
                _binary_control_volume_dirichlet_bc(
                    control_volume_bc.density,
                    control_volume_bc.Ve,
                    lambda left, right: left * right,
                )
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
                _scale_local_dirichlet_face_bc(
                    ion_pressure_face_bc,
                    self.parameters.tau,
                ),
                lambda left, right: left + right,
            )
            velocity_difference_face_bc = _binary_local_dirichlet_face_bc(
                face_bc.Vi,
                face_bc.Ve,
                lambda left, right: left - right,
            )
            current_face_bc = _binary_local_dirichlet_face_bc(
                face_bc.density,
                velocity_difference_face_bc,
                lambda left, right: left * right,
            )
            density_flux_face_bc = _binary_local_dirichlet_face_bc(
                face_bc.density,
                face_bc.Ve,
                lambda left, right: left * right,
            )

            cells = self.control_volume_geometry.cells
            density_owned = jnp.asarray(
                state_owned.density,
                dtype=jnp.float64,
            )
            Te_owned = jnp.asarray(state_owned.Te, dtype=jnp.float64)
            Ti_owned = jnp.asarray(state_owned.Ti, dtype=jnp.float64)
            Vi_owned = jnp.asarray(state_owned.Vi, dtype=jnp.float64)
            Ve_owned = jnp.asarray(state_owned.Ve, dtype=jnp.float64)
            Pe_owned = local_control_volume_product_average(
                density_owned,
                Te_owned,
                density_polynomial,
                Te_polynomial,
                cells,
            )
            ion_pressure_owned = local_control_volume_product_average(
                density_owned,
                Ti_owned,
                density_polynomial,
                Ti_polynomial,
                cells,
            )
            density_Vi_owned = local_control_volume_product_average(
                density_owned,
                Vi_owned,
                density_polynomial,
                Vi_polynomial,
                cells,
            )
            density_Ve_owned = local_control_volume_product_average(
                density_owned,
                Ve_owned,
                density_polynomial,
                Ve_polynomial,
                cells,
            )
            pressure_owned = (
                Pe_owned
                + jnp.asarray(self.parameters.tau, dtype=jnp.float64)
                * ion_pressure_owned
            )
            current_owned = density_Vi_owned - density_Ve_owned
            density_flux_owned = density_Ve_owned
            Pe_halo = self._prepare_scalar_halo(Pe_owned, Pe_face_bc)
            pressure_halo = self._prepare_scalar_halo(
                pressure_owned,
                pressure_face_bc,
            )
            current_halo = self._prepare_scalar_halo(
                current_owned,
                current_face_bc,
            )
            density_flux_halo = self._prepare_scalar_halo(
                density_flux_owned,
                density_flux_face_bc,
            )
            Pe_polynomial = self._control_volume_polynomial(
                Pe_halo,
                Pe_control_volume_bc,
                Pe_face_bc,
            )
            pressure_polynomial = self._control_volume_polynomial(
                pressure_halo,
                pressure_control_volume_bc,
                pressure_face_bc,
            )
            current_polynomial = self._control_volume_polynomial(
                current_halo,
                current_control_volume_bc,
                current_face_bc,
            )
            density_flux_polynomial = self._control_volume_polynomial(
                density_flux_halo,
                density_flux_control_volume_bc,
                density_flux_face_bc,
            )
        else:
            Pe_halo = state_halo.density * state_halo.Te
            pressure_halo = (
                Pe_halo
                + self.parameters.tau
                * state_halo.density
                * state_halo.Ti
            )
            current_halo = (
                state_halo.density
                * (state_halo.Vi - state_halo.Ve)
            )
            density_flux_halo = state_halo.density * state_halo.Ve
            density_polynomial = None
            phi_polynomial = None
            Te_polynomial = None
            Ti_polynomial = None
            Vi_polynomial = None
            Ve_polynomial = None
            vorticity_polynomial = None
            Pe_polynomial = None
            pressure_polynomial = None
            current_polynomial = None
            density_flux_polynomial = None
            Pe_control_volume_bc = None
            pressure_control_volume_bc = None
            current_control_volume_bc = None
            density_flux_control_volume_bc = None

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

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Ti = jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
        density_safe = jnp.maximum(density, 1.0e-30)
        bmag = jnp.maximum(
            jnp.asarray(
                (
                    self.control_volume_geometry.centroid_Bmag
                    if (
                        self.control_volume_geometry is not None
                        and self.control_volume_geometry.has_centroid_operator_geometry
                    )
                    else self.geometry.cell_bfield.Bmag_owned
                ),
                dtype=jnp.float64,
            ),
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
            None if control_volume_bc is None else control_volume_bc.density,
        )
        density_parallel_diff = self._field_parallel_diffusion(
            state_halo.density,
            face_bc.density,
            self.parameters.density_D_parallel,
            None if control_volume_bc is None else control_volume_bc.density,
        )
        Te_diff = self._field_perp_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_D_perp,
            None if control_volume_bc is None else control_volume_bc.Te,
        )
        Te_parallel_diff = self._field_parallel_diffusion(
            state_halo.Te,
            face_bc.Te,
            self.parameters.electron_temperature_chi_parallel,
            None if control_volume_bc is None else control_volume_bc.Te,
        )
        Ti_diff = self._field_perp_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_D_perp,
            None if control_volume_bc is None else control_volume_bc.Ti,
        )
        Ti_parallel_diff = self._field_parallel_diffusion(
            state_halo.Ti,
            face_bc.Ti,
            self.parameters.ion_temperature_chi_parallel,
            None if control_volume_bc is None else control_volume_bc.Ti,
        )
        Vi_diff = self._field_perp_diffusion(
            state_halo.Vi,
            face_bc.Vi,
            self.parameters.Vi_D_perp,
            None if control_volume_bc is None else control_volume_bc.Vi,
        )
        Vi_parallel_diff = self._field_parallel_diffusion(
            state_halo.Vi,
            face_bc.Vi,
            self.parameters.Vi_parallel_viscosity,
            None if control_volume_bc is None else control_volume_bc.Vi,
        )
        Ve_diff = self._field_perp_diffusion(
            state_halo.Ve,
            face_bc.Ve,
            self.parameters.Ve_D_perp,
            None if control_volume_bc is None else control_volume_bc.Ve,
        )
        Ve_parallel_diff = self._field_parallel_diffusion(
            state_halo.Ve,
            face_bc.Ve,
            self.parameters.Ve_parallel_viscosity,
            None if control_volume_bc is None else control_volume_bc.Ve,
        )
        vorticity_diff = self._field_perp_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_perp,
            None if control_volume_bc is None else control_volume_bc.vorticity,
        )
        vorticity_parallel_diff = self._field_parallel_diffusion(
            state_halo.vorticity,
            face_bc.vorticity,
            self.parameters.vorticity_D_parallel,
            None if control_volume_bc is None else control_volume_bc.vorticity,
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

        if control_volume_bc is not None:
            density_gradient = density_polynomial.as_cell_gradient()
            phi_gradient = phi_polynomial.as_cell_gradient()
            Te_gradient = Te_polynomial.as_cell_gradient()
            Ti_gradient = Ti_polynomial.as_cell_gradient()
            Vi_gradient = Vi_polynomial.as_cell_gradient()
            Ve_gradient = Ve_polynomial.as_cell_gradient()
            vorticity_gradient = vorticity_polynomial.as_cell_gradient()
            Pe_gradient = Pe_polynomial.as_cell_gradient()
            pressure_gradient = pressure_polynomial.as_cell_gradient()
            current_gradient = current_polynomial.as_cell_gradient()

            def poisson(field_gradient):
                return local_poisson_bracket_op_from_gradients(
                    phi_gradient,
                    field_gradient,
                    self.geometry,
                    control_volume_geometry=self.control_volume_geometry,
                )

            poisson_density = poisson(density_gradient)
            poisson_Te = poisson(Te_gradient)
            poisson_Ti = poisson(Ti_gradient)
            poisson_Vi = poisson(Vi_gradient)
            poisson_Ve = poisson(Ve_gradient)
            poisson_vorticity = poisson(vorticity_gradient)

            def curvature(field_gradient):
                return local_curvature_op_from_gradient(
                    field_gradient,
                    self.geometry,
                    curvature_coefficients=self.curvature_coefficients_owned,
                    control_volume_geometry=self.control_volume_geometry,
                )

            curvature_Pe = curvature(Pe_gradient)
            curvature_pressure = curvature(pressure_gradient)
            curvature_phi = curvature(phi_gradient)
            curvature_Te = curvature(Te_gradient)
            curvature_Ti = curvature(Ti_gradient)

            def parallel_flux(
                conservative_stencil,
                field_halo,
                boundary_bc,
            ):
                field_closure = build_local_control_volume_field_closure(
                    field_halo,
                    self.control_volume_geometry,
                    boundary_bc,
                )
                return local_parallel_flux_div_op(
                    conservative_stencil,
                    self.geometry,
                    self.domain,
                    control_volume_geometry=self.control_volume_geometry,
                    field_closure=field_closure,
                    axis_regular_axes=self.axis_regular_axes,
                )

            parallel_density_flux_divergence = parallel_flux(
                density_flux_conservative_stencil,
                density_flux_halo,
                density_flux_control_volume_bc,
            )
            parallel_current_flux_divergence = parallel_flux(
                current_conservative_stencil,
                current_halo,
                current_control_volume_bc,
            )
            parallel_Ve_flux_divergence = parallel_flux(
                Ve_conservative_stencil,
                state_halo.Ve,
                control_volume_bc.Ve,
            )
            parallel_Vi_flux_divergence = parallel_flux(
                Vi_conservative_stencil,
                state_halo.Vi,
                control_volume_bc.Vi,
            )

            def grad_parallel(field_gradient):
                return local_grad_parallel_op_from_gradient(
                    field_gradient,
                    self.geometry,
                    control_volume_geometry=self.control_volume_geometry,
                )

            grad_parallel_Te = grad_parallel(Te_gradient)
            grad_parallel_Ti = grad_parallel(Ti_gradient)
            grad_parallel_Ve = grad_parallel(Ve_gradient)
            grad_parallel_Vi = grad_parallel(Vi_gradient)
            grad_parallel_phi = grad_parallel(phi_gradient)
            grad_parallel_Pe = grad_parallel(Pe_gradient)
            grad_parallel_pressure = grad_parallel(pressure_gradient)
            grad_parallel_current = grad_parallel(current_gradient)
            grad_parallel_vorticity = grad_parallel(vorticity_gradient)
        else:
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
            grad_parallel_Te = local_grad_parallel_op_direct(
                Te_stencil,
                self.geometry,
            )
            grad_parallel_Ti = local_grad_parallel_op_direct(
                Ti_stencil,
                self.geometry,
            )
            grad_parallel_Ve = local_grad_parallel_op_direct(
                Ve_stencil,
                self.geometry,
            )
            grad_parallel_Vi = local_grad_parallel_op_direct(
                Vi_stencil,
                self.geometry,
            )
            grad_parallel_phi = local_grad_parallel_op_direct(
                phi_stencil,
                self.geometry,
            )
            grad_parallel_Pe = local_grad_parallel_op_direct(
                Pe_stencil,
                self.geometry,
            )
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
        return _mask_local_eb_state_inactive(FciDrbEBState(
            density=density_rhs + source_owned.density,
            phi=jnp.zeros_like(phi_owned),
            Te=Te_rhs + source_owned.Te,
            Ti=Ti_rhs + source_owned.Ti,
            Vi=Vi_rhs + source_owned.Vi,
            Ve=Ve_rhs + source_owned.Ve,
            vorticity=vorticity_rhs + source_owned.vorticity,
        ), self.geometry)


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
