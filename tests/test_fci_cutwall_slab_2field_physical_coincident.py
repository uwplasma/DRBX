"""Coincident cut-wall regression for the slab two-field MMS problem."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time as time_module

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    FciGeometry3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalDomain3D,
    LocalFciGeometry3D,
    StencilBuilderContext,
    build_local_stencil_from_field,
)
from jax_drb.native import Rk4Stepper
from jax_drb.native.fci_2_field_rhs import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
)
from jax_drb.native.fci_boundaries import LocalCutWallGeometry3D
from jax_drb.native.fci_halo import (
    HaloExchange3D,
    PhysicalGhostCellFiller3D,
)
from jax_drb.native.fci_operators import (
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_poisson_bracket_op,
)


jax.config.update("jax_enable_x64", True)

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))
import test_mms_slab_2_field as slab_mms  # noqa: E402


@dataclass(frozen=True)
class _CoincidentCutWallFixture:
    dependencies: LocalCoordinateStencilDependencyMap3D
    sample_i: jnp.ndarray
    sample_j: jnp.ndarray
    sample_k: jnp.ndarray
    owner_i: jnp.ndarray
    owner_j: jnp.ndarray
    owner_k: jnp.ndarray
    axis: jnp.ndarray
    side: jnp.ndarray
    distance: jnp.ndarray
    cut_wall_geometry: LocalCutWallGeometry3D | None = None


def _shape_from_resolution(resolution: int) -> tuple[int, int, int]:
    n = int(resolution)
    return (n, n, n)


def _axis_unit(axis: int, sign: float) -> jnp.ndarray:
    values = [0.0, 0.0, 0.0]
    values[int(axis)] = float(sign)
    return jnp.asarray(values, dtype=jnp.float64)


def _row_center(
    geometry: LocalFciGeometry3D,
    sample_i: int,
    sample_j: int,
    sample_k: int,
) -> jnp.ndarray:
    return jnp.asarray(
        [
            geometry.grid.x.centers_owned[int(sample_i)],
            geometry.grid.y.centers_owned[int(sample_j)],
            geometry.grid.z.centers_owned[int(sample_k)],
        ],
        dtype=jnp.float64,
    )


def _unchecked_coordinate_dependencies(
    domain: LocalDomain3D,
    *,
    target_flat: jnp.ndarray,
    axis: jnp.ndarray,
    side: jnp.ndarray,
    distance: jnp.ndarray,
    active: jnp.ndarray,
) -> LocalCoordinateStencilDependencyMap3D:
    """Create dependency metadata without value checks inside traced RHS code."""

    local = object.__new__(LocalCoordinateStencilLocalDependencyTable)
    object.__setattr__(local, "target_flat", target_flat)
    object.__setattr__(local, "axis", axis)
    object.__setattr__(local, "side", side)
    object.__setattr__(
        local,
        "value_slot",
        jnp.arange(int(target_flat.size), dtype=jnp.int32),
    )
    object.__setattr__(local, "distance", distance)
    object.__setattr__(local, "active", active)

    dependencies = object.__new__(LocalCoordinateStencilDependencyMap3D)
    object.__setattr__(dependencies, "layout", domain.layout)
    object.__setattr__(dependencies, "local", local)
    object.__setattr__(dependencies, "remote", None)
    return dependencies


def _build_coincident_cut_wall_fixture(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    include_geometry: bool = True,
) -> _CoincidentCutWallFixture:
    """Build local rows that replace each coordinate-wall-adjacent stencil leg.

    The rows are present on every local shard. On interior shard boundaries the
    sampled row value is the current local shell value, so replacing the stencil
    leg is algebraically identical to the unpatched coordinate stencil.
    """

    nx, ny, nz = domain.layout.owned_shape
    if min(nx, ny, nz) < 3:
        raise ValueError("coincident cut-wall fixture requires at least 3 cells per axis")

    owner_rows: list[tuple[int, int, int]] = []
    sample_rows: list[tuple[int, int, int]] = []
    axis_rows: list[int] = []
    side_rows: list[int] = []
    distance_rows: list[float] = []
    center_rows: list[jnp.ndarray] = []
    normal_rows: list[jnp.ndarray] = []
    area_rows: list[jnp.ndarray] = []

    global_shape = tuple(int(value) for value in domain.shard_spec.global_shape)
    spacings = tuple(1.0 / float(size - 1) for size in global_shape)

    def add_face(axis: int, side: int) -> None:
        shape = (nx, ny, nz)
        owner_axis_index = 1 if side == 0 else shape[axis] - 2
        sample_axis_index = 0 if side == 0 else shape[axis] - 1
        normal = _axis_unit(axis, -1.0 if side == 0 else 1.0)
        tangent_axes = [candidate for candidate in range(3) if candidate != axis]
        area = spacings[tangent_axes[0]] * spacings[tangent_axes[1]]
        area_covector = normal * area

        for tangent_a in range(shape[tangent_axes[0]]):
            for tangent_b in range(shape[tangent_axes[1]]):
                owner = [0, 0, 0]
                sample = [0, 0, 0]
                owner[axis] = owner_axis_index
                sample[axis] = sample_axis_index
                owner[tangent_axes[0]] = tangent_a
                sample[tangent_axes[0]] = tangent_a
                owner[tangent_axes[1]] = tangent_b
                sample[tangent_axes[1]] = tangent_b
                owner_tuple = tuple(int(value) for value in owner)
                sample_tuple = tuple(int(value) for value in sample)
                owner_rows.append(owner_tuple)
                sample_rows.append(sample_tuple)
                axis_rows.append(axis)
                side_rows.append(side)
                distance_rows.append(spacings[axis])
                if include_geometry:
                    center_rows.append(_row_center(geometry, *sample_tuple))
                    normal_rows.append(normal)
                    area_rows.append(area_covector)

    for face_axis in range(3):
        add_face(face_axis, 0)
        add_face(face_axis, 1)

    owner_i_np = np.asarray([row[0] for row in owner_rows], dtype=np.int32)
    owner_j_np = np.asarray([row[1] for row in owner_rows], dtype=np.int32)
    owner_k_np = np.asarray([row[2] for row in owner_rows], dtype=np.int32)
    sample_i = jnp.asarray([row[0] for row in sample_rows], dtype=jnp.int32)
    sample_j = jnp.asarray([row[1] for row in sample_rows], dtype=jnp.int32)
    sample_k = jnp.asarray([row[2] for row in sample_rows], dtype=jnp.int32)
    axis_array = jnp.asarray(axis_rows, dtype=jnp.int32)
    side_array = jnp.asarray(side_rows, dtype=jnp.int32)
    distance = jnp.asarray(distance_rows, dtype=jnp.float64)
    active = jnp.ones((len(owner_rows),), dtype=bool)
    target_flat = jnp.asarray((owner_i_np * ny + owner_j_np) * nz + owner_k_np)

    if include_geometry:
        dependencies = LocalCoordinateStencilDependencyMap3D(
            layout=domain.layout,
            local=LocalCoordinateStencilLocalDependencyTable(
                target_flat=target_flat,
                axis=axis_array,
                side=side_array,
                value_slot=jnp.arange(len(owner_rows), dtype=jnp.int32),
                distance=distance,
                active=active,
            ),
        )
    else:
        dependencies = _unchecked_coordinate_dependencies(
            domain,
            target_flat=target_flat,
            axis=axis_array,
            side=side_array,
            distance=distance,
            active=active,
        )

    cut_wall_geometry = None
    if include_geometry:
        max_wall_faces = len(owner_rows)
        cut_wall_geometry = LocalCutWallGeometry3D(
            owner_i=jnp.asarray(owner_i_np, dtype=jnp.int32),
            owner_j=jnp.asarray(owner_j_np, dtype=jnp.int32),
            owner_k=jnp.asarray(owner_k_np, dtype=jnp.int32),
            center=jnp.stack(center_rows, axis=0),
            normal_contra=jnp.stack(normal_rows, axis=0),
            area_covector=jnp.stack(area_rows, axis=0),
            distance=distance,
            J=jnp.ones((max_wall_faces,), dtype=jnp.float64),
            g_contra=jnp.broadcast_to(
                jnp.eye(3, dtype=jnp.float64),
                (max_wall_faces, 3, 3),
            ),
            g_cov=jnp.broadcast_to(
                jnp.eye(3, dtype=jnp.float64),
                (max_wall_faces, 3, 3),
            ),
            B_contra=jnp.broadcast_to(
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
                (max_wall_faces, 3),
            ),
            Bmag=jnp.ones((max_wall_faces,), dtype=jnp.float64),
            sign=jnp.ones((max_wall_faces,), dtype=jnp.float64),
            active=active,
            max_wall_faces=max_wall_faces,
            stencil_axis=axis_array,
            stencil_side=side_array,
            stencil_distance=distance,
        )

    return _CoincidentCutWallFixture(
        dependencies=dependencies,
        sample_i=sample_i,
        sample_j=sample_j,
        sample_k=sample_k,
        owner_i=jnp.asarray(owner_i_np, dtype=jnp.int32),
        owner_j=jnp.asarray(owner_j_np, dtype=jnp.int32),
        owner_k=jnp.asarray(owner_k_np, dtype=jnp.int32),
        axis=axis_array,
        side=side_array,
        distance=distance,
        cut_wall_geometry=cut_wall_geometry,
    )


def _sample_cut_wall_values(
    field_halo: jnp.ndarray,
    fixture: _CoincidentCutWallFixture,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    h = int(domain.layout.halo_width)
    return jnp.asarray(field_halo, dtype=jnp.float64)[
        h + fixture.sample_i,
        h + fixture.sample_j,
        h + fixture.sample_k,
    ]


def _non_pinned_interior_mask(shape: tuple[int, int, int]) -> jnp.ndarray:
    i = jnp.arange(shape[0])[:, None, None]
    j = jnp.arange(shape[1])[None, :, None]
    k = jnp.arange(shape[2])[None, None, :]
    return (
        (i > 0)
        & (i < shape[0] - 1)
        & (j > 0)
        & (j < shape[1] - 1)
        & (k > 0)
        & (k < shape[2] - 1)
    )


def _masked_state_linf(
    left: Fci2FieldState,
    right: Fci2FieldState,
    mask: jnp.ndarray,
) -> float:
    density_error = jnp.max(jnp.abs(jnp.where(mask, left.density - right.density, 0.0)))
    velocity_error = jnp.max(
        jnp.abs(jnp.where(mask, left.v_parallel - right.v_parallel, 0.0))
    )
    background_error = jnp.max(
        jnp.abs(
            jnp.where(
                mask,
                left.density_background - right.density_background,
                0.0,
            )
        )
    )
    return float(jnp.max(jnp.asarray([density_error, velocity_error, background_error])))


@dataclass(frozen=True)
class LocalSlab2FieldCutWallRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    halo_exchange: HaloExchange3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci2FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    timing_enabled: bool = False

    def __call__(
        self,
        state_owned: Fci2FieldState,
        stage_time: float | jax.Array,
        carry: None,
    ) -> tuple[Fci2FieldState, None, jnp.ndarray]:
        del carry
        prepared_stage = slab_mms._prepare_local_slab_stage_state(
            state_owned,
            self.coordinates_halo,
            self.domain,
            halo_exchange=self.halo_exchange,
            physical_ghost_filler=self.physical_ghost_filler,
            stage_time=stage_time,
        )
        density_halo = jnp.asarray(prepared_stage.state_halo.density, dtype=jnp.float64)
        v_parallel_halo = jnp.asarray(
            prepared_stage.state_halo.v_parallel,
            dtype=jnp.float64,
        )
        density_background_halo = jnp.asarray(
            prepared_stage.state_halo.density_background,
            dtype=jnp.float64,
        )
        phi_halo = jnp.log(
            jnp.maximum(density_halo, 1.0e-30)
            / jnp.maximum(density_background_halo, 1.0e-30)
        )

        fixture = _build_coincident_cut_wall_fixture(
            self.geometry,
            self.domain,
            include_geometry=False,
        )

        def build_stencil(field_halo: jnp.ndarray):
            return build_local_stencil_from_field(
                field_halo,
                self.geometry,
                StencilBuilderContext(
                    layout=self.domain.layout,
                    domain=self.domain,
                    cut_wall_stencil_dependencies=fixture.dependencies,
                    cut_wall_values=_sample_cut_wall_values(
                        field_halo,
                        fixture,
                        self.domain,
                    ),
                ),
            )

        density_stencil = build_stencil(density_halo)
        phi_stencil = build_stencil(phi_halo)
        v_parallel_stencil = build_stencil(v_parallel_halo)

        density_owned = density_halo[self.domain.layout.owned_slices_cell]
        magnetic_field = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        curvature_density = local_curvature_op(
            density_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        parallel_velocity_gradient = local_grad_parallel_op_direct(
            v_parallel_stencil,
            self.geometry,
        )
        poisson_v_parallel = local_poisson_bracket_op(
            phi_stencil,
            v_parallel_stencil,
            self.geometry,
        )

        density_rhs = (
            -(poisson_density / (rho_star_value * magnetic_field))
            + (2.0 / magnetic_field) * curvature_density
            - (2.0 * density_owned / magnetic_field) * curvature_phi
            - density_owned * parallel_velocity_gradient
        )
        v_parallel_rhs = -(poisson_v_parallel / (rho_star_value * magnetic_field))
        density_rhs = density_rhs + slab_mms._mms_local_density_source_from_coordinates(
            *self.coordinates_halo,
            stage_time,
            rho_star_value=rho_star_value,
        )[self.domain.layout.owned_slices_cell]
        v_parallel_rhs = (
            v_parallel_rhs
            + slab_mms._mms_local_v_parallel_source_from_coordinates(
                *self.coordinates_halo,
                stage_time,
                rho_star_value=rho_star_value,
            )[self.domain.layout.owned_slices_cell]
        )

        rhs = Fci2FieldState(
            density=jnp.asarray(density_rhs, dtype=jnp.float64),
            v_parallel=jnp.asarray(v_parallel_rhs, dtype=jnp.float64),
            density_background=jnp.zeros(
                self.domain.layout.owned_shape,
                dtype=jnp.float64,
            ),
        )
        aux = jnp.zeros((3,), dtype=jnp.float64)
        if self.timing_enabled:
            aux = aux + 0.0
        return rhs, None, aux


def simulate_mms_2field_slab_coincident_cutwall(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = slab_mms.tf,
    rho_star_value: float = slab_mms.rho_star,
    show_progress: bool = False,
) -> tuple[Fci2FieldState, dict[str, float]]:
    """Advance the slab MMS with coincident cut-wall coordinate-stencil rows."""

    slab_mms._assert_mms_slab_geometry(geometry)
    shard_counts = tuple(int(value) for value in shard_counts)
    slab_mms.assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    domain = slab_mms._build_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = slab_mms._build_ghost_filler(halo_width)
    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    curvature_coefficients_owned = jnp.zeros(owned_shape + (3,), dtype=jnp.float64)
    dt = (
        float(final_time) / float(slab_mms.num_steps)
        if timestep is None
        else float(timestep)
    )
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_state = slab_mms._mms_exact_state(geometry, 0.0)
    total_runtime = 0.0
    wall_step_times: list[float] = []
    prebuilt_local_geometry = None
    prebuilt_coordinates_halo = None
    if shard_counts == (1, 1, 1):
        prebuilt_local_geometry = slab_mms.build_local_slab_2field_geometry(
            owned_shape,
            halo_width,
            global_shape=geometry.shape,
            shard_index=(0, 0, 0),
        )
        prebuilt_coordinates_halo = slab_mms._mms_local_coordinates(
            prebuilt_local_geometry
        )

    with slab_mms.make_mesh_for_shard_counts(shard_counts) as mesh:
        state = slab_mms._put_state_on_mesh(initial_state, mesh)
        state_spec = slab_mms._state_partition_spec()

        def kernel(
            state_owned: Fci2FieldState,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> Fci2FieldState:
            if prebuilt_local_geometry is None:
                shard_index = tuple(lax.axis_index(name) for name in slab_mms._MESH_AXIS_NAMES)
                local_geometry = slab_mms.build_local_slab_2field_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=geometry.shape,
                    shard_index=shard_index,
                )
                coordinates_halo = slab_mms._mms_local_coordinates(local_geometry)
            else:
                local_geometry = prebuilt_local_geometry
                coordinates_halo = prebuilt_coordinates_halo
            rhs = LocalSlab2FieldCutWallRhs(
                geometry=local_geometry,
                domain=domain,
                coordinates_halo=coordinates_halo,
                halo_exchange=HaloExchange3D(),
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=curvature_coefficients_owned,
            )
            step_result = Rk4Stepper(rhs)(
                state_owned,
                time=step_time,
                timestep=step_timestep,
                carry=None,
            )
            next_exact_halo = slab_mms._mms_local_exact_state_from_coordinates(
                *coordinates_halo,
                step_time + step_timestep,
            )
            next_exact_owned = Fci2FieldState(
                density=next_exact_halo.density[domain.layout.owned_slices_cell],
                v_parallel=next_exact_halo.v_parallel[domain.layout.owned_slices_cell],
                density_background=next_exact_halo.density_background[
                    domain.layout.owned_slices_cell
                ],
            )
            next_state = slab_mms._apply_local_owned_dirichlet_to_state(
                step_result.state,
                next_exact_owned,
                domain,
            )
            return next_state

        mapped_step_kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(state_spec, P(), P()),
            out_specs=state_spec,
            check_rep=False,
        )
        step_kernel = jax.jit(mapped_step_kernel)

        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                "slab_2field coincident cut-wall RK4 progress: "
                f"{slab_mms._format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

        for step_index in range(steps):
            step_start = time_module.perf_counter()
            state = step_kernel(
                state,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            elapsed = time_module.perf_counter() - step_start
            total_runtime += elapsed
            wall_step_times.append(elapsed)
            time_value += dt
            if show_progress:
                print(
                    "\r"
                    "slab_2field coincident cut-wall RK4 progress: "
                    f"{slab_mms._format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )

        if show_progress:
            print()

        final_state = slab_mms._gather_state_from_mesh(state)

    mean_step_runtime = total_runtime / float(steps) if steps else 0.0
    timing_summary = {
        "total_runtime": float(total_runtime),
        "avg_step_runtime": float(mean_step_runtime),
        "prep_time": 0.0,
        "stencil_time": 0.0,
        "operator_time": 0.0,
    }
    if wall_step_times:
        print(
            "slab_2field coincident cut-wall mean timings per RK step: "
            f"wall={np.mean(np.asarray(wall_step_times, dtype=np.float64)):.6e} s"
        )
    return final_state, timing_summary


def run_slab_2field_coincident_cutwall_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = slab_mms.tf,
    base_steps: int = slab_mms.num_steps,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []

    print()
    print("=" * 80)
    print("Slab 2-field MMS coincident cut-wall convergence")
    print("=" * 80)
    print(f"shard_counts = {tuple(int(value) for value in shard_counts)}")
    print()

    for resolution in resolutions:
        shape = _shape_from_resolution(resolution)
        slab_mms.assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = slab_mms.build_slab_2field_geometry(*shape)
        steps = slab_mms._resolution_step_count(resolution, base_steps=base_steps)
        dt = float(final_time) / float(steps)
        final_state, timing_summary = simulate_mms_2field_slab_coincident_cutwall(
            geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=final_time,
            timestep=dt,
            rho_star_value=slab_mms.rho_star,
            show_progress=show_progress,
        )
        l2_error, median_error, linf_error = slab_mms._combined_error_statistics(
            final_state,
            geometry,
            final_time,
        )

        successful_resolutions.append(int(resolution))
        l2_errors.append(l2_error)
        linf_errors.append(linf_error)
        print(
            f"N={resolution}: shard_counts={shard_counts}, "
            f"steps={steps}, total_runtime={timing_summary['total_runtime']:.6e} s, "
            f"avg_step_runtime={timing_summary['avg_step_runtime']:.6e} s, "
            f"L2={l2_error:.6e}, median={median_error:.6e}, Linf={linf_error:.6e}"
        )

    l2_order = slab_mms._estimate_convergence_order(successful_resolutions, l2_errors)
    linf_order = slab_mms._estimate_convergence_order(successful_resolutions, linf_errors)
    if l2_order is not None:
        print(f"slab_2field coincident cut-wall L2 convergence order: {l2_order:.6f}")
    if linf_order is not None:
        print(f"slab_2field coincident cut-wall Linf convergence order: {linf_order:.6f}")

    if plot and successful_resolutions:
        slab_mms._save_convergence_plot(
            successful_resolutions,
            l2_errors,
            linf_errors,
            title=f"2-field slab coincident cut-wall MMS convergence ({shard_counts})",
            output_path=Path(plot_path or "slab_2field_coincident_cutwall_convergence.png"),
        )

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": linf_errors,
        "l2_order": l2_order,
        "linf_order": linf_order,
    }


def _compare_with_physical_convergence(
    *,
    cutwall_results: dict[str, object],
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float,
    base_steps: int,
    plot: bool,
    plot_path: str | None,
    show_progress: bool,
) -> None:
    physical_results = slab_mms.run_slab_2field_convergence(
        resolutions=resolutions,
        shard_counts=shard_counts,
        halo_width=halo_width,
        final_time=final_time,
        base_steps=base_steps,
        plot=plot,
        plot_path=plot_path,
        show_progress=show_progress,
    )
    cutwall_l2 = list(cutwall_results["l2_errors"])
    physical_l2 = list(physical_results["l2_errors"])
    print()
    print("Coincident cut-wall / physical L2 ratios")
    print("-" * 80)
    for resolution, cutwall_error, physical_error in zip(
        cutwall_results["resolutions"],
        cutwall_l2,
        physical_l2,
    ):
        ratio = float(cutwall_error) / float(physical_error)
        print(f"N={int(resolution):4d}: ratio={ratio:.6e}")


def test_coincident_cutwall_geometry_matches_physical_wall_shells() -> None:
    global_shape = (6, 8, 10)
    halo_width = 2
    owned_shape = global_shape
    geometry = slab_mms.build_local_slab_2field_geometry(
        owned_shape,
        halo_width,
        global_shape=global_shape,
    )
    domain = slab_mms._build_local_domain(
        global_shape,
        halo_width,
        (1, 1, 1),
        mesh_axis_names=(None, None, None),
    )
    fixture = _build_coincident_cut_wall_fixture(geometry, domain)
    assert fixture.cut_wall_geometry is not None

    expected_rows = 2 * (
        global_shape[1] * global_shape[2]
        + global_shape[0] * global_shape[2]
        + global_shape[0] * global_shape[1]
    )
    assert fixture.cut_wall_geometry.max_wall_faces == expected_rows
    assert bool(jnp.all(fixture.cut_wall_geometry.active))
    assert bool(jnp.all(fixture.distance > 0.0))
    assert bool(jnp.all(jnp.isfinite(fixture.cut_wall_geometry.center)))
    assert bool(jnp.any((fixture.axis == 0) & (fixture.side == 0)))
    assert bool(jnp.any((fixture.axis == 0) & (fixture.side == 1)))
    assert bool(jnp.any((fixture.axis == 1) & (fixture.side == 0)))
    assert bool(jnp.any((fixture.axis == 1) & (fixture.side == 1)))
    assert bool(jnp.any((fixture.axis == 2) & (fixture.side == 0)))
    assert bool(jnp.any((fixture.axis == 2) & (fixture.side == 1)))

    x_lower = (fixture.axis == 0) & (fixture.side == 0)
    x_upper = (fixture.axis == 0) & (fixture.side == 1)
    assert bool(jnp.all(fixture.owner_i[x_lower] == 1))
    assert bool(jnp.all(fixture.owner_i[x_upper] == global_shape[0] - 2))
    assert bool(jnp.all(fixture.sample_i[x_lower] == 0))
    assert bool(jnp.all(fixture.sample_i[x_upper] == global_shape[0] - 1))

    exact_halo = slab_mms._mms_local_exact_state(local_geometry, 0.125)
    phi_halo = jnp.log(exact_halo.density / exact_halo.density_background)
    assert bool(
        jnp.all(jnp.isfinite(_sample_cut_wall_values(exact_halo.density, fixture, domain)))
    )
    assert bool(
        jnp.all(
            jnp.isfinite(_sample_cut_wall_values(exact_halo.v_parallel, fixture, domain))
        )
    )
    assert bool(
        jnp.all(
            jnp.isfinite(
                _sample_cut_wall_values(exact_halo.density_background, fixture, domain)
            )
        )
    )
    assert bool(jnp.all(jnp.isfinite(_sample_cut_wall_values(phi_halo, fixture, domain))))

    sharded_global_shape = (8, 8, 10)
    shard_counts = (2, 1, 1)
    sharded_owned_shape = (
        sharded_global_shape[0] // shard_counts[0],
        sharded_global_shape[1],
        sharded_global_shape[2],
    )
    sharded_geometry = slab_mms.build_local_slab_2field_geometry(
        sharded_owned_shape,
        halo_width,
        global_shape=sharded_global_shape,
    )
    sharded_domain = slab_mms._build_local_domain(
        sharded_global_shape,
        halo_width,
        shard_counts,
        mesh_axis_names=(None, None, None),
    )
    sharded_fixture = _build_coincident_cut_wall_fixture(
        sharded_geometry,
        sharded_domain,
    )
    sharded_expected_rows = 2 * (
        sharded_owned_shape[1] * sharded_owned_shape[2]
        + sharded_owned_shape[0] * sharded_owned_shape[2]
        + sharded_owned_shape[0] * sharded_owned_shape[1]
    )
    assert sharded_fixture.cut_wall_geometry is not None
    assert sharded_fixture.cut_wall_geometry.max_wall_faces == sharded_expected_rows
    assert bool(jnp.all(sharded_fixture.cut_wall_geometry.active))


def test_coincident_cutwall_rhs_matches_physical_boundary_rhs() -> None:
    shape = (6, 6, 6)
    halo_width = 2
    global_geometry = slab_mms.build_slab_2field_geometry(*shape)
    local_geometry = slab_mms.build_local_slab_2field_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    domain = slab_mms._build_local_domain(
        shape,
        halo_width,
        (1, 1, 1),
        mesh_axis_names=(None, None, None),
    )
    coordinates_halo = slab_mms._mms_local_coordinates(local_geometry)
    common_kwargs = dict(
        geometry=local_geometry,
        domain=domain,
        coordinates_halo=coordinates_halo,
        halo_exchange=HaloExchange3D(),
        physical_ghost_filler=slab_mms._build_ghost_filler(halo_width),
        parameters=Fci2FieldRhsParameters(rho_star=slab_mms.rho_star),
        curvature_coefficients_owned=jnp.zeros(shape + (3,), dtype=jnp.float64),
    )
    physical_rhs = slab_mms.LocalSlab2FieldRhs(**common_kwargs)
    cutwall_rhs = LocalSlab2FieldCutWallRhs(**common_kwargs)
    initial_state = slab_mms._mms_exact_state(global_geometry, 0.0)

    physical_state, physical_carry, physical_aux = physical_rhs(initial_state, 0.0, None)
    cutwall_state, cutwall_carry, cutwall_aux = cutwall_rhs(initial_state, 0.0, None)

    assert physical_carry is None
    assert cutwall_carry is None
    np.testing.assert_allclose(np.asarray(cutwall_aux), np.asarray(physical_aux))
    mask = _non_pinned_interior_mask(shape)
    assert _masked_state_linf(cutwall_state, physical_state, mask) < 1.0e-11


def test_coincident_cutwall_slab_2field_mms_converges() -> None:
    results = run_slab_2field_coincident_cutwall_convergence(
        resolutions=[8, 12],
        shard_counts=(1, 1, 1),
        halo_width=2,
        final_time=0.02,
        base_steps=6,
        show_progress=False,
    )

    l2_errors = list(results["l2_errors"])
    linf_errors = list(results["linf_errors"])
    assert len(l2_errors) == 2
    assert all(np.isfinite(l2_errors))
    assert all(np.isfinite(linf_errors))
    assert l2_errors[-1] < l2_errors[0]
    assert linf_errors[-1] < 1.25 * linf_errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slab 2-field coincident cut-wall MMS convergence harness"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[40, 60, 120],
    )
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=slab_mms.tf)
    parser.add_argument("--base-steps", type=int, default=slab_mms.num_steps)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--compare-physical", action="store_true")
    args = parser.parse_args()

    resolutions = [int(value) for value in args.resolutions]
    shard_counts = tuple(int(value) for value in args.shard_counts)
    results = run_slab_2field_coincident_cutwall_convergence(
        resolutions=resolutions,
        shard_counts=shard_counts,
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
    )
    if bool(args.compare_physical):
        _compare_with_physical_convergence(
            cutwall_results=results,
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=int(args.halo_width),
            final_time=float(args.final_time),
            base_steps=int(args.base_steps),
            plot=bool(args.plot),
            plot_path=args.plot_path,
            show_progress=bool(args.show_progress),
        )


if __name__ == "__main__":
    main()
