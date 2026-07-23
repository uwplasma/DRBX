"""Runtime harness for the shifted-torus four-field cut-wall MMS case."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
import sys
import time as time_module

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from drbx.geometry import (
    FciGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from drbx.native import Fci4FieldRhsParameters, Fci4FieldState, SpmdGmresConfig
from drbx.native.fci_boundaries import (
    LocalBoundaryFaceBC3D,
    LocalCellGradient3D,
    LocalControlVolumeBoundaryBC3D,
    LocalEmbeddedControlVolumeGeometry3D,
)
from drbx.native.fci_halo import (
    HaloExchange3D,
    LocalHaloClosure3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_model import inject_owned_state_to_halo
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _apply_local_face_flux_bc,
    _apply_local_face_value_dirichlet_bc,
    _local_axis_face_values_from_stencil,
    _take_stencil_finite_difference,
    build_local_control_volume_field_closure,
    build_local_control_volume_polynomial_from_field,
    build_local_perp_laplacian_stencil,
    local_control_volume_product_average,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_from_gradient,
    local_parallel_flux_div_op,
    local_poisson_bracket_op_from_gradients,
)


_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


import shifted_torus_4field_mms_helpers as shifted_mms  # noqa: E402
from mms_domain_decomp_helpers import (  # noqa: E402
    assert_shape_divisible_by_shards,
    build_shifted_torus_local_domain,
    build_shifted_torus_local_geometry,
    expand_local_shard_pytree,
    extract_local_shard_pytree,
    local_shard_pytree_partition_spec,
    make_mesh_for_shard_counts,
)
from shifted_torus_4field_cutwall_geometry import (  # noqa: E402
    MESH_AXIS_NAMES,
    _build_stacked_embedded_control_volume_geometry,
    _with_embedded_control_volume_geometry,
)
from shifted_torus_4field_cutwall_mms import (  # noqa: E402
    _agglomerate_control_volume_average,
    _control_volume_exact_boundary_bc,
    _expand_control_volume_owner_values,
    _multiply_local_dirichlet_face_bc,
    _project_global_exact_state_to_control_volumes,
    _project_local_mms_source_to_control_volumes,
    _with_shifted_torus_regular_radial_face_averages,
)


@dataclass(frozen=True)
class LocalShiftedTorus4FieldCutWallRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci4FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SpmdGmresConfig
    global_shape: tuple[int, int, int]
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D

    def _prepare_phi_halo(
        self,
        phi_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        phi_halo = inject_owned_state_to_halo(
            Fci4FieldState(
                density=phi_owned,
                omega=phi_owned,
                v_ion_parallel=phi_owned,
                v_electron_parallel=phi_owned,
            ),
            self.domain.layout,
        ).density
        return LocalHaloClosure3D(
            physical_ghost_filler=self.physical_ghost_filler,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )(phi_halo, self.domain, face_bc)

    def evaluate_stage(
        self,
        state_owned: Fci4FieldState,
        stage_data: shifted_mms._ShiftedTorus4FieldStageData,
        phi_guess_owned: jnp.ndarray | None,
        *,
        phi_wall_offset: float | jax.Array = 0.0,
        solve_phi: bool = True,
    ) -> tuple[Fci4FieldState, jnp.ndarray]:
        prepared = shifted_mms._prepare_local_shifted_torus_4field_stage_state(
            state_owned,
            stage_data,
            self.domain,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        face_bc = prepared.boundary_data.face_bc
        control_volume_geometry = self.control_volume_geometry
        cells = control_volume_geometry.cells
        stage_time = jnp.asarray(
            stage_data.stage_time,
            dtype=jnp.float64,
        )
        phi_control_volume_bc = _control_volume_exact_boundary_bc(
            control_volume_geometry,
            stage_time,
            "phi",
            value_offset=phi_wall_offset,
        )
        aggregate_cell_positions_owned = cells.centroid

        def prepare_owner_field(
            values_owned: jnp.ndarray,
            field_face_bc: LocalBoundaryFaceBC3D,
        ) -> jnp.ndarray:
            return self._prepare_phi_halo(
                _expand_control_volume_owner_values(values_owned, cells),
                field_face_bc,
            )

        state_halo = Fci4FieldState(
            density=prepare_owner_field(state_owned.density, face_bc.density),
            omega=prepare_owner_field(state_owned.omega, face_bc.omega),
            v_ion_parallel=prepare_owner_field(
                state_owned.v_ion_parallel,
                face_bc.v_ion_parallel,
            ),
            v_electron_parallel=prepare_owner_field(
                state_owned.v_electron_parallel,
                face_bc.v_electron_parallel,
            ),
        )
        omega_owned = jnp.asarray(state_owned.omega, dtype=jnp.float64)
        raw_phi_lift_owned = jnp.asarray(
            stage_data.phi_halo[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        phi_lift_owned = _agglomerate_control_volume_average(
            raw_phi_lift_owned,
            cells,
        )
        if solve_phi:
            phi_solver = LocalPerpLaplacianInverseSolver(
                geometry=self.geometry,
                domain=self.domain,
                stencil_builder=build_local_conservative_stencil_from_field,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
                physical_ghost_filler=self.physical_ghost_filler,
                face_projectors=self.face_projectors,
                control_volume_geometry=control_volume_geometry,
                control_volume_boundary_bc=phi_control_volume_bc,
                face_bc=face_bc.phi,
                config=self.gmres_config,
            )
            phi_owned = phi_solver(
                -omega_owned,
                guess_owned=phi_guess_owned,
                phi_lift_owned=phi_lift_owned,
            )
        else:
            if phi_guess_owned is None:
                raise ValueError(
                    "solve_phi=False requires prescribed phi in "
                    "phi_guess_owned"
                )
            phi_owned = jnp.asarray(phi_guess_owned, dtype=jnp.float64)
        phi_owned = jnp.where(self.geometry.active_cell_mask_owned, phi_owned, 0.0)
        phi_halo = prepare_owner_field(phi_owned, face_bc.phi)

        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
        )

        def build_stencil(field_halo: jnp.ndarray):
            return build_local_stencil_from_field(
                field_halo,
                self.geometry,
                context,
            )

        def build_gradient(
            field_halo: jnp.ndarray,
            field_name: str,
            field_boundary_bc: LocalControlVolumeBoundaryBC3D | None = None,
            field_face_bc: LocalBoundaryFaceBC3D | None = None,
        ) -> tuple[LocalCellGradient3D, object, LocalControlVolumeBoundaryBC3D]:
            if field_boundary_bc is None:
                field_boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    field_name,
                )
            if field_face_bc is None:
                field_face_bc = getattr(face_bc, field_name)
            polynomial = build_local_control_volume_polynomial_from_field(
                field_halo,
                self.geometry,
                self.domain,
                StencilBuilderContext(
                    layout=context.layout,
                    domain=context.domain,
                ),
                control_volume_geometry,
                field_boundary_bc,
                field_face_bc,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
            )
            return polynomial.as_cell_gradient(), polynomial, field_boundary_bc

        density_gradient, density_polynomial, density_control_volume_bc = build_gradient(
            state_halo.density,
            "density",
        )
        omega_gradient, omega_polynomial, omega_control_volume_bc = build_gradient(
            state_halo.omega,
            "omega",
        )
        phi_gradient, phi_polynomial, _phi_control_volume_bc = build_gradient(
            phi_halo,
            "phi",
            phi_control_volume_bc,
        )
        (
            v_ion_gradient,
            v_ion_polynomial,
            v_ion_control_volume_bc,
        ) = build_gradient(
            state_halo.v_ion_parallel,
            "v_ion_parallel",
        )
        (
            v_electron_gradient,
            v_electron_polynomial,
            v_electron_control_volume_bc,
        ) = build_gradient(
            state_halo.v_electron_parallel,
            "v_electron_parallel",
        )

        density_owned = jnp.asarray(state_owned.density, dtype=jnp.float64)
        v_electron_owned = jnp.asarray(
            state_owned.v_electron_parallel,
            dtype=jnp.float64,
        )
        density_v_electron_owned = local_control_volume_product_average(
            density_owned,
            v_electron_owned,
            density_polynomial,
            v_electron_polynomial,
            cells,
        )
        density_v_electron_face_bc = _multiply_local_dirichlet_face_bc(
            face_bc.density,
            face_bc.v_electron_parallel,
        )
        density_v_electron_halo = prepare_owner_field(
            density_v_electron_owned,
            density_v_electron_face_bc,
        )
        density_v_electron_stencil = build_stencil(
            density_v_electron_halo
        )
        density_v_electron_conservative_stencil = (
            build_local_conservative_stencil_from_field(
                density_v_electron_halo,
                self.geometry,
                context,
            )
        )
        (
            density_v_electron_gradient,
            density_v_electron_polynomial,
            density_v_electron_control_volume_bc,
        ) = build_gradient(
            density_v_electron_halo,
            "density_v_electron",
            field_face_bc=density_v_electron_face_bc,
        )
        density_safe = jnp.maximum(density_owned, 1.0e-30)
        bmag_owned = jnp.maximum(
            jnp.asarray(
                (
                    control_volume_geometry.centroid_Bmag
                    if control_volume_geometry.has_centroid_operator_geometry
                    else self.geometry.cell_bfield.Bmag_owned
                ),
                dtype=jnp.float64,
            ),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        te = jnp.asarray(self.parameters.Te, dtype=jnp.float64)
        mi_over_me_value = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)

        poisson_density = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            density_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_omega = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            omega_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_v_ion = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            v_ion_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        poisson_v_electron = local_poisson_bracket_op_from_gradients(
            phi_gradient,
            v_electron_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        curvature_density = local_curvature_op_from_gradient(
            density_gradient,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
            control_volume_geometry=control_volume_geometry,
        )
        curvature_phi = local_curvature_op_from_gradient(
            phi_gradient,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_density = local_grad_parallel_op_from_gradient(
            density_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_phi = local_grad_parallel_op_from_gradient(
            phi_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_v_ion = local_grad_parallel_op_from_gradient(
            v_ion_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        grad_parallel_v_electron = local_grad_parallel_op_from_gradient(
            v_electron_gradient,
            self.geometry,
            control_volume_geometry=control_volume_geometry,
        )
        pressure_grad_parallel_density = grad_parallel_density
        pressure_density_safe = density_safe
        parallel_density_flux_divergence = local_parallel_flux_div_op(
            density_v_electron_conservative_stencil,
            self.geometry,
            self.domain,
            control_volume_geometry=control_volume_geometry,
            field_closure=build_local_control_volume_field_closure(
                density_v_electron_halo,
                control_volume_geometry,
                density_v_electron_control_volume_bc,
            ),
        )

        density_poisson_term = -(poisson_density / (rho_star_value * bmag_owned))
        density_curvature_term = (2.0 * te / bmag_owned) * curvature_density
        density_phi_curvature_term = -(2.0 * density_owned / bmag_owned) * curvature_phi
        density_parallel_term = -parallel_density_flux_divergence
        density_rhs = (
            density_poisson_term
            + density_curvature_term
            + density_phi_curvature_term
            + density_parallel_term
        )
        omega_poisson_term = -(poisson_omega / (rho_star_value * bmag_owned))
        omega_parallel_term = (
            (bmag_owned * bmag_owned / density_safe)
            * (grad_parallel_v_ion - grad_parallel_v_electron)
        )
        omega_curvature_term = (2.0 * bmag_owned * te / density_safe) * curvature_density
        omega_rhs = (
            omega_poisson_term
            + omega_parallel_term
            + omega_curvature_term
        )
        v_ion_poisson_term = -(poisson_v_ion / (rho_star_value * bmag_owned))
        v_ion_pressure_term = -(te / pressure_density_safe) * pressure_grad_parallel_density
        v_ion_rhs = (
            v_ion_poisson_term
            + v_ion_pressure_term
        )
        v_electron_poisson_term = -(poisson_v_electron / (rho_star_value * bmag_owned))
        v_electron_parallel_phi_term = mi_over_me_value * grad_parallel_phi
        v_electron_pressure_term = -mi_over_me_value * (
            te / pressure_density_safe
        ) * pressure_grad_parallel_density
        v_electron_rhs = (
            v_electron_poisson_term
            + v_electron_parallel_phi_term
            + v_electron_pressure_term
        )
        owned = self.domain.layout.owned_slices_cell
        source_density = stage_data.source_halo.density[owned]
        source_omega = stage_data.source_halo.omega[owned]
        source_v_ion = stage_data.source_halo.v_ion_parallel[owned]
        source_v_electron = stage_data.source_halo.v_electron_parallel[owned]
        source_owned = Fci4FieldState(
            density=_agglomerate_control_volume_average(source_density, cells),
            omega=_agglomerate_control_volume_average(source_omega, cells),
            v_ion_parallel=_agglomerate_control_volume_average(
                source_v_ion,
                cells,
            ),
            v_electron_parallel=_agglomerate_control_volume_average(
                source_v_electron,
                cells,
            ),
        )
        rhs = Fci4FieldState(
            density=density_rhs + source_owned.density,
            omega=omega_rhs + source_owned.omega,
            v_ion_parallel=v_ion_rhs + source_owned.v_ion_parallel,
            v_electron_parallel=v_electron_rhs + source_owned.v_electron_parallel,
        )
        rhs = _mask_4field_state_inactive(rhs, self.geometry)

        return rhs, phi_owned
def _mask_4field_state_inactive(
    state: Fci4FieldState,
    geometry: LocalFciGeometry3D,
) -> Fci4FieldState:
    """Zero inactive owned cells for test-local 4-field RHS/update payloads."""

    active = geometry.active_cell_mask_owned
    return Fci4FieldState(
        density=jnp.where(active, state.density, 0.0),
        omega=jnp.where(active, state.omega, 0.0),
        v_ion_parallel=jnp.where(active, state.v_ion_parallel, 0.0),
        v_electron_parallel=jnp.where(active, state.v_electron_parallel, 0.0),
    )
def _make_parameters(rho_star_value: float) -> Fci4FieldRhsParameters:
    return Fci4FieldRhsParameters(
        rho_star=float(rho_star_value),
        Te=float(shifted_mms.Te),
        mi_over_me=float(shifted_mms.mi_over_me),
        phi_inversion_tol=1.0e-11,
        phi_inversion_maxiter=500,
        phi_inversion_restart=100,
    )


def _make_gmres_config(parameters: Fci4FieldRhsParameters) -> SpmdGmresConfig:
    return SpmdGmresConfig(
        tol=float(parameters.phi_inversion_tol),
        atol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        stagnation_iters=0,
        acceptance_tol=float(parameters.phi_inversion_tol),
        acceptance_atol=float(parameters.phi_inversion_tol),
        regularization_epsilon=float(parameters.phi_inversion_regularization),
    )
def simulate_mms_shifted_torus_4field_cutwall(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = shifted_mms.tf,
    rho_star_value: float = shifted_mms.rho_star,
    show_progress: bool = False,
    enable_agglomeration: bool = False,
    stacked_control_volume_geometry: (
        LocalEmbeddedControlVolumeGeometry3D | None
    ) = None,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(int(size) // int(count) for size, count in zip(geometry.shape, shard_counts))
    domain = build_shifted_torus_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = shifted_mms._build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    parameters = _make_parameters(rho_star_value)
    gmres_config = _make_gmres_config(parameters)
    dt = float(final_time) / float(shifted_mms.num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    if stacked_control_volume_geometry is None:
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=geometry.shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
    initial_state, initial_phi_guess = (
        _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=0.0,
        )
    )
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    wall_step_times: list[float] = []

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = shifted_mms._put_state_on_mesh(initial_state, mesh)
        phi_guess = jax.device_put(
            jnp.asarray(initial_phi_guess, dtype=jnp.float64),
            NamedSharding(mesh, P(*MESH_AXIS_NAMES)),
        )
        state_spec = shifted_mms._state_partition_spec()
        field_spec = P(*MESH_AXIS_NAMES)
        control_volume_geometry_spec = local_shard_pytree_partition_spec(
            stacked_control_volume_geometry
        )
        control_volume_geometry_sharding = jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, spec),
            control_volume_geometry_spec,
        )
        control_volume_geometry = jax.device_put(
            stacked_control_volume_geometry,
            control_volume_geometry_sharding,
        )
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            shifted_mms._build_local_4field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        sample_stage_data = shifted_mms._build_local_4field_rk4_stage_data(
            shifted_mms._build_local_4field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            ),
            0.0,
            dt,
            parameters=parameters,
        )
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(sample_stage_data)
        )

        def invariant_kernel() -> shifted_mms._ShiftedTorus4FieldInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                shifted_mms._build_local_4field_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=geometry.shape,
                    domain=domain,
                )
            )

        def _local_geometry_for_shard(
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
        ) -> LocalFciGeometry3D:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=geometry.shape,
                shard_index=shard_index,
                x_min=shifted_mms.x_min,
                x_max=shifted_mms.x_max,
                r0=shifted_mms.r0,
                alpha_value=shifted_mms.alpha_value,
                iota=shifted_mms.iota,
                c_phi=shifted_mms.c_phi,
                sigma=shifted_mms.sigma,
            )
            return _with_embedded_control_volume_geometry(
                local_geometry,
                local_control_volume_geometry,
            )

        def source_kernel(
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> shifted_mms._ShiftedTorus4FieldRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            local_control_volume_geometry = extract_local_shard_pytree(
                local_control_volume_geometry
            )
            local_geometry = _local_geometry_for_shard(
                local_control_volume_geometry
            )
            stage_data = shifted_mms._build_local_4field_rk4_stage_data(
                local_invariants,
                step_time,
                step_timestep,
                parameters=parameters,
            )

            def project_stage(
                stage: shifted_mms._ShiftedTorus4FieldStageData,
                stage_time: jax.Array,
            ) -> shifted_mms._ShiftedTorus4FieldStageData:
                stage = _with_shifted_torus_regular_radial_face_averages(
                    stage,
                    local_geometry,
                    stage_time,
                )
                owner_source = _project_local_mms_source_to_control_volumes(
                    local_geometry,
                    local_control_volume_geometry,
                    stage_time,
                    parameters,
                )
                cells = local_control_volume_geometry.cells
                expanded_source = Fci4FieldState(
                    density=_expand_control_volume_owner_values(
                        owner_source.density,
                        cells,
                    ),
                    omega=_expand_control_volume_owner_values(
                        owner_source.omega,
                        cells,
                    ),
                    v_ion_parallel=_expand_control_volume_owner_values(
                        owner_source.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_expand_control_volume_owner_values(
                        owner_source.v_electron_parallel,
                        cells,
                    ),
                )
                return dataclass_replace(
                    stage,
                    stage_time=jnp.asarray(stage_time, dtype=jnp.float64),
                    source_halo=inject_owned_state_to_halo(
                        expanded_source,
                        domain.layout,
                    ),
                )

            half_step = 0.5 * step_timestep
            projected = shifted_mms._ShiftedTorus4FieldRk4StageData(
                stage_1=project_stage(stage_data.stage_1, step_time),
                stage_2=project_stage(
                    stage_data.stage_2,
                    step_time + half_step,
                ),
                stage_3=project_stage(
                    stage_data.stage_3,
                    step_time + half_step,
                ),
                stage_4=project_stage(
                    stage_data.stage_4,
                    step_time + step_timestep,
                ),
            )
            return expand_local_shard_pytree(projected)

        def kernel(
            state_owned: Fci4FieldState,
            phi_guess_owned: jnp.ndarray,
            local_invariants: shifted_mms._ShiftedTorus4FieldInvariantBundle,
            rk_stage_data: shifted_mms._ShiftedTorus4FieldRk4StageData,
            local_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> tuple[Fci4FieldState, jnp.ndarray]:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            local_control_volume_geometry = extract_local_shard_pytree(
                local_control_volume_geometry
            )
            local_geometry = _local_geometry_for_shard(
                local_control_volume_geometry
            )
            rhs = LocalShiftedTorus4FieldCutWallRhs(
                geometry=local_geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=local_invariants.curvature_coefficients_owned,
                face_projectors=(
                    local_invariants.face_projector_x,
                    local_invariants.face_projector_y,
                    local_invariants.face_projector_z,
                ),
                gmres_config=gmres_config,
                global_shape=geometry.shape,
                control_volume_geometry=local_control_volume_geometry,
            )

            k1, carry_1 = rhs.evaluate_stage(
                state_owned,
                rk_stage_data.stage_1,
                phi_guess_owned,
            )
            stage_1 = state_owned.axpy(k1, scale=0.5 * step_timestep)
            k2, carry_2 = rhs.evaluate_stage(
                stage_1,
                rk_stage_data.stage_2,
                carry_1,
            )
            stage_2 = state_owned.axpy(k2, scale=0.5 * step_timestep)
            k3, carry_3 = rhs.evaluate_stage(
                stage_2,
                rk_stage_data.stage_3,
                carry_2,
            )
            stage_3 = state_owned.axpy(k3, scale=step_timestep)
            k4, carry_4 = rhs.evaluate_stage(
                stage_3,
                rk_stage_data.stage_4,
                carry_3,
            )
            next_state = state_owned.axpy(
                k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0),
                scale=step_timestep / 6.0,
            )
            return next_state, carry_4

        invariants = jax.jit(
            shard_map(
                invariant_kernel,
                mesh=mesh,
                in_specs=(),
                out_specs=invariant_spec,
                check_rep=False,
            )
        )()
        compiled_source_kernel = jax.jit(
            shard_map(
                source_kernel,
                mesh=mesh,
                in_specs=(
                    invariant_spec,
                    control_volume_geometry_spec,
                    P(),
                    P(),
                ),
                out_specs=stage_data_spec,
                check_rep=False,
            )
        )
        step_kernel = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    field_spec,
                    invariant_spec,
                    stage_data_spec,
                    control_volume_geometry_spec,
                    P(),
                    P(),
                ),
                out_specs=(state_spec, field_spec),
                check_rep=False,
            )
        )
        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                f"shifted_torus_4field_cutwall RK4 progress: "
                f"{shifted_mms._format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )
        for step_index in range(steps):
            step_start = time_module.perf_counter()
            rk_stage_data = compiled_source_kernel(
                invariants,
                control_volume_geometry,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            state, phi_guess = step_kernel(
                state,
                phi_guess,
                invariants,
                rk_stage_data,
                control_volume_geometry,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            wall_step_times.append(time_module.perf_counter() - step_start)
            time_value += dt
            times.append(time_value)
            gathered_state = shifted_mms._gather_state_from_mesh(state)
            density_history.append(jnp.asarray(gathered_state.density, dtype=jnp.float32))
            omega_history.append(jnp.asarray(gathered_state.omega, dtype=jnp.float32))
            v_ion_history.append(jnp.asarray(gathered_state.v_ion_parallel, dtype=jnp.float32))
            v_electron_history.append(jnp.asarray(gathered_state.v_electron_parallel, dtype=jnp.float32))
            if show_progress:
                print(
                    "\r"
                    f"shifted_torus_4field_cutwall RK4 progress: "
                    f"{shifted_mms._format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )
        if show_progress:
            print()
        final_state = shifted_mms._gather_state_from_mesh(state)

    if wall_step_times:
        print(
            "shifted_torus_4field_cutwall mean timings per RK step: "
            f"wall={np.mean(np.asarray(wall_step_times, dtype=np.float64)):.6e} s"
        )

    return (
        final_state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(omega_history, axis=0),
        jnp.stack(v_ion_history, axis=0),
        jnp.stack(v_electron_history, axis=0),
    )
