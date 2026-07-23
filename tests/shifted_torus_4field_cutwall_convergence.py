"""Convergence runners and diagnostics for the shifted-torus cut-wall MMS."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
import gc
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
)
from drbx.native import Fci4FieldState
from drbx.native.fci_boundaries import (
    CV_FACE_CUT_WALL,
    CV_FACE_INTERIOR,
    CV_FACE_PARTIAL,
    CV_FACE_PHYSICAL_BOUNDARY,
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
from drbx.native.fci_model import inject_owned_field_to_halo, inject_owned_state_to_halo
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _axis_slice_nd,
    _lift_cell_field_to_faces,
    build_local_control_volume_field_closure,
    build_local_control_volume_polynomial_from_field,
    evaluate_local_control_volume_polynomial,
    local_control_volume_product_average,
    local_curvature_op_from_gradient,
    local_grad_parallel_op_from_gradient,
    local_parallel_flux_div_op,
    local_perp_laplacian_conservative_op,
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
    _shape_from_resolution,
    _shifted_torus_curvature_at_logical_points,
    _with_embedded_control_volume_geometry,
)
from shifted_torus_4field_cutwall_mms import (  # noqa: E402
    _agglomerate_control_volume_average,
    _assemble_global_control_volume_cell_data,
    _control_volume_exact_boundary_bc,
    _expand_control_volume_owner_values,
    _integrate_local_exact_state_over_fluid,
    _integrate_local_four_field_over_fluid,
    _integrate_local_scalar_over_fluid,
    _multiply_local_dirichlet_face_bc,
    _project_global_exact_state_to_control_volumes,
    _project_local_exact_time_derivative_to_control_volumes,
    _project_local_mms_source_to_control_volumes,
    _shifted_torus_analytic_rhs_at_logical_points,
    _shifted_torus_exact_time_derivative_at_logical_points,
    _shifted_torus_mms_source_at_logical_points,
    _shifted_torus_operator_reference_at_logical_points,
    _shifted_torus_regular_radial_face_average,
    _with_shifted_torus_regular_radial_face_averages,
)
from shifted_torus_4field_cutwall_runtime import (  # noqa: E402
    LocalShiftedTorus4FieldCutWallRhs,
    _make_gmres_config,
    _make_parameters,
    simulate_mms_shifted_torus_4field_cutwall,
)


def _print_control_volume_geometry_summary(
    stacked: LocalEmbeddedControlVolumeGeometry3D,
) -> None:
    cells = stacked.cells
    faces = stacked.irregular_faces
    reconstruction = stacked.reconstruction
    face_active = np.asarray(faces.active, dtype=bool)
    face_kind = np.asarray(faces.kind, dtype=np.int32)
    reconstruction_active = np.asarray(reconstruction.active, dtype=bool)
    reconstruction_order = np.asarray(
        reconstruction.polynomial_order,
        dtype=np.int32,
    )
    condition = np.asarray(
        reconstruction.condition_number,
        dtype=np.float64,
    )
    finite_condition = condition[
        reconstruction_active & np.isfinite(condition)
    ]
    print(
        "embedded control volumes: "
        f"active_owners={int(np.sum(np.asarray(cells.is_active_owner)))}, "
        f"merged_sources={int(np.sum(np.asarray(cells.is_merged_source)))}, "
        f"aggregate_targets={int(np.sum(np.asarray(cells.is_aggregate_target)))}, "
        f"irregular_faces={int(np.sum(face_active))}, "
        "interior/partial/cutwall="
        f"{int(np.sum(face_active & (face_kind == CV_FACE_INTERIOR)))}/"
        f"{int(np.sum(face_active & (face_kind == CV_FACE_PARTIAL)))}/"
        f"{int(np.sum(face_active & (face_kind == CV_FACE_CUT_WALL)))}, "
        "physical_boundary="
        f"{int(np.sum(face_active & (face_kind == CV_FACE_PHYSICAL_BOUNDARY)))}, "
        f"cubic_rows={int(np.sum(reconstruction_active & (reconstruction_order == 3)))}, "
        f"quadratic_fallbacks={int(np.sum(reconstruction_active & (reconstruction_order == 2)))}, "
        f"linear_fallbacks={int(np.sum(reconstruction_active & (reconstruction_order == 1)))}, "
        "max_condition="
        f"{float(np.max(finite_condition)) if finite_condition.size else 0.0:.6e}"
    )

def _state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
) -> dict[str, tuple[float, float, float]]:
    return shifted_mms._state_error_statistics(actual, expected)






def _masked_field_error_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[float, float, float]:
    error = jnp.asarray(actual - expected, dtype=jnp.float64)
    expected_array = jnp.asarray(expected, dtype=jnp.float64)
    mask_f = jnp.asarray(mask, dtype=jnp.float64)
    count = jnp.maximum(jnp.sum(mask_f), 1.0)
    masked_error = jnp.where(mask, error, 0.0)
    masked_expected = jnp.where(mask, expected_array, 0.0)
    l2 = float(jnp.sqrt(jnp.sum(jnp.square(masked_error)) / count))
    linf = float(jnp.max(jnp.abs(masked_error)))
    rel_l2 = float(
        jnp.sqrt(jnp.sum(jnp.square(masked_error)))
        / (jnp.sqrt(jnp.sum(jnp.square(masked_expected))) + 1.0e-30)
    )
    return l2, linf, rel_l2


def _masked_state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
    mask: jnp.ndarray,
) -> dict[str, tuple[float, float, float]]:
    return {
        "density": _masked_field_error_statistics(actual.density, expected.density, mask),
        "omega": _masked_field_error_statistics(actual.omega, expected.omega, mask),
        "v_ion_parallel": _masked_field_error_statistics(actual.v_ion_parallel, expected.v_ion_parallel, mask),
        "v_electron_parallel": _masked_field_error_statistics(
            actual.v_electron_parallel,
            expected.v_electron_parallel,
            mask,
        ),
    }


def _volume_weighted_field_error_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    volume: jnp.ndarray,
    active_owner: jnp.ndarray,
) -> tuple[float, float, float]:
    actual = jnp.asarray(actual, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    weight = jnp.where(
        active_owner,
        jnp.asarray(volume, dtype=jnp.float64),
        0.0,
    )
    error = jnp.where(active_owner, actual - expected, 0.0)
    weight_sum = jnp.maximum(jnp.sum(weight), 1.0e-30)
    l2 = jnp.sqrt(jnp.sum(weight * error * error) / weight_sum)
    linf = jnp.max(jnp.where(active_owner, jnp.abs(error), 0.0))
    expected_norm = jnp.sqrt(
        jnp.sum(weight * expected * expected) / weight_sum
    )
    return (
        float(l2),
        float(linf),
        float(l2 / jnp.maximum(expected_norm, 1.0e-30)),
    )


def _volume_weighted_state_error_statistics(
    actual: Fci4FieldState,
    expected: Fci4FieldState,
    volume: jnp.ndarray,
    active_owner: jnp.ndarray,
) -> dict[str, tuple[float, float, float]]:
    return {
        name: _volume_weighted_field_error_statistics(
            getattr(actual, name),
            getattr(expected, name),
            volume,
            active_owner,
        )
        for name in (
            "density",
            "omega",
            "v_ion_parallel",
            "v_electron_parallel",
        )
    }


def _print_state_error_statistics(
    label: str,
    statistics: dict[str, tuple[float, float, float]],
) -> None:
    print(label)
    for field_name, (l2, linf, relative_l2) in statistics.items():
        print(
            f"  {field_name}: L2={l2:.6e}, Linf={linf:.6e}, "
            f"rel_L2={relative_l2:.6e}"
        )



def _resolution_step_count(resolution: int, *, base_steps: int) -> int:
    return max(
        1,
        int(
            round(
                float(base_steps)
                * float(resolution)
                / 20.0
            )
        ),
    )









def _control_volume_operator_category_masks(
    cell_data: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    active = jnp.asarray(cell_data["is_active_owner"], dtype=bool)
    boundary_count = jnp.asarray(
        cell_data["boundary_face_count"],
        dtype=jnp.int32,
    )
    irregular_count = jnp.asarray(
        cell_data["irregular_face_count"],
        dtype=jnp.int32,
    )
    aggregate_target = jnp.asarray(
        cell_data["is_aggregate_target"],
        dtype=bool,
    )
    remote_count = jnp.asarray(
        cell_data["remote_face_count"],
        dtype=jnp.int32,
    )
    reconstruction_count = jnp.asarray(
        cell_data["reconstruction_row_count"],
        dtype=jnp.int32,
    )

    def neighbor_band(mask: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.asarray(mask, dtype=bool)
        result = jnp.zeros_like(mask)
        result = result | jnp.zeros_like(mask).at[1:, :, :].set(
            mask[:-1, :, :]
        )
        result = result | jnp.zeros_like(mask).at[:-1, :, :].set(
            mask[1:, :, :]
        )
        result = result | jnp.roll(mask, 1, axis=1)
        result = result | jnp.roll(mask, -1, axis=1)
        result = result | jnp.roll(mask, 1, axis=2)
        result = result | jnp.roll(mask, -1, axis=2)
        return result

    compact_core = active & (
        (irregular_count > 0)
        | (reconstruction_count > 0)
        | aggregate_target
    )
    dense_compact_d1 = (
        active
        & (~compact_core)
        & neighbor_band(compact_core)
    )
    dense_compact_d2 = (
        active
        & (~compact_core)
        & (~dense_compact_d1)
        & neighbor_band(compact_core | dense_compact_d1)
    )
    dense_far = active & (
        ~(compact_core | dense_compact_d1 | dense_compact_d2)
    )
    radial_index = jnp.arange(active.shape[0], dtype=jnp.int32)[:, None, None]
    radial_lower_owner = active & (radial_index == 0)
    radial_upper_owner = active & (radial_index == active.shape[0] - 1)
    radial_interior = active & (radial_index >= 2) & (
        radial_index < active.shape[0] - 2
    )
    return {
        "all_active": active,
        "bulk": active & (irregular_count == 0) & (~aggregate_target),
        "one_wall": active & (boundary_count == 1),
        "multi_wall": active & (boundary_count >= 2),
        "aggregate_target": active & aggregate_target,
        "remote_interface": active & (remote_count > 0),
        "reconstruction_row": active & (reconstruction_count > 0),
        "retained_cut_cell": (
            active
            & (boundary_count > 0)
            & (~aggregate_target)
        ),
        "dense_compact_d1": dense_compact_d1,
        "dense_compact_d2": dense_compact_d2,
        "dense_far": dense_far,
        "radial_lower_owner": radial_lower_owner,
        "radial_upper_owner": radial_upper_owner,
        "radial_interior_2plus": radial_interior,
    }


def _operator_category_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    volume: jnp.ndarray,
    categories: dict[str, jnp.ndarray],
) -> dict[str, tuple[float, float, float, int]]:
    result: dict[str, tuple[float, float, float, int]] = {}
    for category, mask in categories.items():
        count = int(jnp.sum(jnp.asarray(mask, dtype=jnp.int32)))
        if count == 0:
            result[category] = (
                float("nan"),
                float("nan"),
                float("nan"),
                0,
            )
            continue
        l2, linf, relative = _volume_weighted_field_error_statistics(
            actual,
            expected,
            volume,
            mask,
        )
        result[category] = (l2, linf, relative, count)
    return result


def _fit_operator_order(
    resolutions: list[int],
    errors: list[float],
) -> float | None:
    resolution_array = np.asarray(resolutions, dtype=np.float64)
    error_array = np.asarray(errors, dtype=np.float64)
    valid = (
        np.isfinite(resolution_array)
        & np.isfinite(error_array)
        & (resolution_array > 0.0)
        & (error_array > 0.0)
    )
    if int(np.sum(valid)) < 2:
        return None
    slope = np.polyfit(
        np.log(resolution_array[valid]),
        np.log(error_array[valid]),
        1,
    )[0]
    return float(-slope)


def run_shifted_torus_control_volume_operator_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    rho_star_value: float = shifted_mms.rho_star,
    enable_agglomeration: bool = True,
    minimum_order: float = 1.8,
    check_phi_solve: bool = True,
) -> dict[str, object]:
    """Run low-memory spatial convergence kernels for the unified CV path."""

    shard_counts = tuple(int(value) for value in shard_counts)
    parameters = _make_parameters(rho_star_value)
    gmres_config = _make_gmres_config(parameters)
    operator_names = (
        "grad_parallel_density",
        "grad_parallel_phi",
        "grad_parallel_v_ion",
        "grad_parallel_v_electron",
        "parallel_density_flux_divergence",
        "poisson_density",
        "poisson_omega",
        "poisson_v_ion",
        "poisson_v_electron",
        "curvature_density",
        "curvature_phi",
        "perp_laplacian_phi",
    )
    records: dict[
        str,
        dict[str, list[tuple[int, float, float]]],
    ] = {}
    phi_residuals: list[tuple[int, float]] = []

    for resolution in resolutions:
        resolution = int(resolution)
        shape = _shape_from_resolution(resolution)
        assert_shape_divisible_by_shards(shape, shard_counts)
        owned_shape = tuple(
            int(size) // int(count)
            for size, count in zip(shape, shard_counts)
        )
        geometry = shifted_mms.build_shifted_torus_4field_geometry(shape)
        print(
            "Preparing shifted_torus control-volume geometry: "
            f"N={resolution}, shape={shape}, shards={shard_counts}",
            flush=True,
        )
        geometry_start = time_module.perf_counter()
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
        print(
            "Prepared shifted_torus control-volume geometry: "
            f"N={resolution}, elapsed="
            f"{time_module.perf_counter() - geometry_start:.3f}s",
            flush=True,
        )
        exact_state, exact_phi = _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=0.0,
        )
        cell_data = _assemble_global_control_volume_cell_data(
            shape,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
        )
        categories = _control_volume_operator_category_masks(cell_data)
        volume = cell_data["aggregate_volume"]
        _print_control_volume_geometry_summary(
            stacked_control_volume_geometry
        )
        print(
            "Starting shifted_torus control-volume operator sweep: "
            f"N={resolution}, shape={shape}, shards={shard_counts}"
        )
        print(
            "  radial boundary contract: x-low=physical Dirichlet, "
            "x-high=physical Dirichlet, axis_regular_x=False"
        )

        domain = build_shifted_torus_local_domain(
            shape,
            halo_width,
            shard_counts,
        )
        topology_filler = TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),)
        )
        physical_ghost_filler = shifted_mms._build_ghost_filler(halo_width)

        with make_mesh_for_shard_counts(shard_counts) as mesh:
            state_spec = shifted_mms._state_partition_spec()
            field_spec = P(*MESH_AXIS_NAMES)
            state_mesh = shifted_mms._put_state_on_mesh(exact_state, mesh)
            phi_mesh = jax.device_put(
                jnp.asarray(exact_phi, dtype=jnp.float64),
                NamedSharding(mesh, field_spec),
            )
            host_domain = LocalDomain3D(
                shard_spec=domain.shard_spec,
                layout=domain.layout,
                mesh_axis_names=(None, None, None),
            )
            sample_invariants = expand_local_shard_pytree(
                shifted_mms._build_local_4field_invariants(
                    (0, 0, 0),
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=shape,
                    domain=host_domain,
                )
            )
            invariant_spec = local_shard_pytree_partition_spec(
                sample_invariants
            )
            control_volume_spec = local_shard_pytree_partition_spec(
                stacked_control_volume_geometry
            )
            control_volume_sharding = jax.tree_util.tree_map(
                lambda spec: NamedSharding(mesh, spec),
                control_volume_spec,
            )
            control_volume_mesh = jax.device_put(
                stacked_control_volume_geometry,
                control_volume_sharding,
            )

            def invariant_kernel():
                shard_index = tuple(
                    lax.axis_index(name)
                    for name in MESH_AXIS_NAMES
                )
                return expand_local_shard_pytree(
                    shifted_mms._build_local_4field_invariants(
                        shard_index,
                        owned_shape=owned_shape,
                        halo_width=halo_width,
                        global_shape=shape,
                        domain=domain,
                    )
                )

            invariants_mesh = jax.jit(
                shard_map(
                    invariant_kernel,
                    mesh=mesh,
                    in_specs=(),
                    out_specs=invariant_spec,
                    check_rep=False,
                )
            )()

            def local_geometry(
                control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
            ) -> LocalFciGeometry3D:
                shard_index = tuple(
                    lax.axis_index(name)
                    for name in MESH_AXIS_NAMES
                )
                base = build_shifted_torus_local_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=shape,
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
                    base,
                    control_volume_geometry,
                )

            def regular_face_bc(
                local_geometry_value: LocalFciGeometry3D,
                stage_time: jax.Array,
                field_name: str,
            ) -> LocalBoundaryFaceBC3D:
                lower, upper = _shifted_torus_regular_radial_face_average(
                    local_geometry_value,
                    stage_time,
                    field_name,
                )
                return (
                    shifted_mms._build_local_radial_dirichlet_face_bc_from_values(
                        lower,
                        upper,
                        domain,
                    )
                )

            def prepare_field(
                values_owned: jnp.ndarray,
                field_name: str,
                local_invariants,
                local_geometry_value: LocalFciGeometry3D,
                control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
                stage_time: jax.Array,
            ):
                face_bc = regular_face_bc(
                    local_geometry_value,
                    stage_time,
                    field_name,
                )
                storage = _expand_control_volume_owner_values(
                    values_owned,
                    control_volume_geometry.cells,
                )
                field_halo = inject_owned_field_to_halo(
                    storage,
                    domain.layout,
                )
                field_halo = LocalHaloClosure3D(
                    physical_ghost_filler=physical_ghost_filler,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                )(
                    field_halo,
                    domain,
                    face_bc,
                )
                boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    field_name,
                )
                polynomial = build_local_control_volume_polynomial_from_field(
                    field_halo,
                    local_geometry_value,
                    domain,
                    StencilBuilderContext(
                        layout=domain.layout,
                        domain=domain,
                    ),
                    control_volume_geometry,
                    boundary_bc,
                    face_bc,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                )
                return field_halo, polynomial, boundary_bc, face_bc

            def evaluate_scalar_operator(
                operator_name: str,
                state_owned: Fci4FieldState,
                phi_owned: jnp.ndarray,
                local_invariants,
                control_volume_geometry,
                stage_time: jax.Array,
            ) -> tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ]:
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                active = control_volume_geometry.cells.is_active_owner
                remote_flux_sum = jnp.asarray(0.0, dtype=jnp.float64)
                remote_flux_abs_sum = jnp.asarray(0.0, dtype=jnp.float64)
                invalid_remote_quadrature = jnp.asarray(
                    0,
                    dtype=jnp.int32,
                )
                invalid_reconstruction_rows = jnp.asarray(
                    0,
                    dtype=jnp.int32,
                )
                reconstruction_target = (
                    control_volume_geometry.reconstruction.target_row_for_cell
                    >= 0
                )

                field_suffixes = {
                    "density": "density",
                    "omega": "omega",
                    "v_ion": "v_ion_parallel",
                    "v_electron": "v_electron_parallel",
                    "phi": "phi",
                }

                if operator_name.startswith("grad_parallel_"):
                    suffix = operator_name.removeprefix(
                        "grad_parallel_"
                    )
                    field_name = field_suffixes[suffix]
                    field_owned = (
                        phi_owned
                        if field_name == "phi"
                        else getattr(state_owned, field_name)
                    )
                    _field_halo, field_poly, _bc, _face_bc = prepare_field(
                        field_owned,
                        field_name,
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    actual = local_grad_parallel_op_from_gradient(
                        field_poly.as_cell_gradient(),
                        local_geometry_value,
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~field_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = (
                        _shifted_torus_operator_reference_at_logical_points(
                            control_volume_geometry.cells.centroid,
                            stage_time,
                            operator_name,
                        )
                    )
                elif operator_name == "parallel_density_flux_divergence":
                    (
                        _density_halo,
                        density_polynomial,
                        _density_bc,
                        _density_face_bc,
                    ) = prepare_field(
                        state_owned.density,
                        "density",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    (
                        _v_electron_halo,
                        v_electron_polynomial,
                        _v_electron_bc,
                        _v_electron_face_bc,
                    ) = prepare_field(
                        state_owned.v_electron_parallel,
                        "v_electron_parallel",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    density_v_electron = local_control_volume_product_average(
                        state_owned.density,
                        state_owned.v_electron_parallel,
                        density_polynomial,
                        v_electron_polynomial,
                        control_volume_geometry.cells,
                    )
                    field_halo, polynomial, boundary_bc, face_bc = (
                        prepare_field(
                            density_v_electron,
                            "density_v_electron",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    local = build_local_conservative_stencil_from_field(
                        field_halo,
                        local_geometry_value,
                        StencilBuilderContext(
                            layout=domain.layout,
                            domain=domain,
                        ),
                    )
                    field_closure = build_local_control_volume_field_closure(
                        field_halo,
                        control_volume_geometry,
                        boundary_bc,
                    )
                    actual = local_parallel_flux_div_op(
                        local,
                        local_geometry_value,
                        domain,
                        face_bc=face_bc,
                        control_volume_geometry=control_volume_geometry,
                        field_closure=field_closure,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (
                                (~density_polynomial.valid)
                                | (~v_electron_polynomial.valid)
                                | (~polynomial.valid)
                            )
                        ).astype(jnp.int32)
                    )
                    irregular_flux = field_closure.parallel_flux
                    remote_row = (
                        control_volume_geometry.irregular_faces.active
                        & control_volume_geometry.irregular_faces.has_remote_owner
                    )
                    remote_flux_sum = jnp.sum(
                        jnp.where(remote_row, irregular_flux, 0.0)
                    )
                    remote_flux_abs_sum = jnp.sum(
                        jnp.where(
                            remote_row,
                            jnp.abs(irregular_flux),
                            0.0,
                        )
                    )
                    invalid_remote_quadrature = jnp.sum(
                        (remote_row & (~field_closure.valid)).astype(jnp.int32)
                    )
                    for mesh_axis_name in MESH_AXIS_NAMES:
                        remote_flux_sum = lax.psum(
                            remote_flux_sum,
                            mesh_axis_name,
                        )
                        remote_flux_abs_sum = lax.psum(
                            remote_flux_abs_sum,
                            mesh_axis_name,
                        )
                        invalid_remote_quadrature = lax.psum(
                            invalid_remote_quadrature,
                            mesh_axis_name,
                        )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                "parallel_density_flux_divergence",
                            )
                        ),
                    )
                elif operator_name.startswith("poisson_"):
                    suffix = operator_name.removeprefix("poisson_")
                    field_name = field_suffixes[suffix]
                    field_owned = getattr(state_owned, field_name)
                    _phi_halo, phi_poly, _bc, _face_bc = prepare_field(
                        phi_owned,
                        "phi",
                        local_invariants,
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                    _v_electron_halo, v_electron_poly, _bc, _face_bc = (
                        prepare_field(
                            field_owned,
                            field_name,
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    actual = local_poisson_bracket_op_from_gradients(
                        phi_poly.as_cell_gradient(),
                        v_electron_poly.as_cell_gradient(),
                        local_geometry_value,
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (
                                (~phi_poly.valid)
                                | (~v_electron_poly.valid)
                            )
                        ).astype(jnp.int32)
                    )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                operator_name,
                            )
                        ),
                    )
                elif operator_name.startswith("curvature_"):
                    suffix = operator_name.removeprefix("curvature_")
                    field_name = field_suffixes[suffix]
                    field_owned = (
                        phi_owned
                        if field_name == "phi"
                        else getattr(state_owned, field_name)
                    )
                    _field_halo, field_poly, _bc, _face_bc = (
                        prepare_field(
                            field_owned,
                            field_name,
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    actual = local_curvature_op_from_gradient(
                        field_poly.as_cell_gradient(),
                        local_geometry_value,
                        curvature_coefficients=(
                            local_invariants.curvature_coefficients_owned
                        ),
                        control_volume_geometry=control_volume_geometry,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~field_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = _integrate_local_scalar_over_fluid(
                        local_geometry_value,
                        control_volume_geometry,
                        lambda points: (
                            _shifted_torus_operator_reference_at_logical_points(
                                points,
                                stage_time,
                                operator_name,
                            )
                        ),
                    )
                elif operator_name == "perp_laplacian_phi":
                    phi_halo, phi_poly, boundary_bc, face_bc = (
                        prepare_field(
                            phi_owned,
                            "phi",
                            local_invariants,
                            local_geometry_value,
                            control_volume_geometry,
                            stage_time,
                        )
                    )
                    local = build_local_conservative_stencil_from_field(
                        phi_halo,
                        local_geometry_value,
                        StencilBuilderContext(
                            layout=domain.layout,
                            domain=domain,
                        ),
                    )
                    phi_closure = build_local_control_volume_field_closure(
                        phi_halo,
                        control_volume_geometry,
                        boundary_bc,
                    )
                    actual = local_perp_laplacian_conservative_op(
                        local,
                        local_geometry_value,
                        domain,
                        face_projectors=(
                            local_invariants.face_projector_x,
                            local_invariants.face_projector_y,
                            local_invariants.face_projector_z,
                        ),
                        face_bc=face_bc,
                        control_volume_geometry=control_volume_geometry,
                        field_closure=phi_closure,
                    )
                    invalid_reconstruction_rows = jnp.sum(
                        (
                            reconstruction_target
                            & (~phi_poly.valid)
                        ).astype(jnp.int32)
                    )
                    reference = state_owned.omega
                else:
                    raise ValueError(
                        f"unsupported operator kernel {operator_name!r}"
                    )
                for mesh_axis_name in MESH_AXIS_NAMES:
                    invalid_reconstruction_rows = lax.psum(
                        invalid_reconstruction_rows,
                        mesh_axis_name,
                    )
                return (
                    jnp.where(active, actual, 0.0),
                    jnp.where(active, reference, 0.0),
                    remote_flux_sum,
                    remote_flux_abs_sum,
                    invalid_remote_quadrature,
                    invalid_reconstruction_rows,
                )

            def make_scalar_kernel(operator_name: str):
                def kernel(
                    state_owned,
                    phi_owned,
                    local_invariants,
                    control_volume_geometry,
                    stage_time,
                ):
                    return evaluate_scalar_operator(
                        operator_name,
                        state_owned,
                        phi_owned,
                        local_invariants,
                        control_volume_geometry,
                        stage_time,
                    )

                return kernel

            for operator_name in operator_names:
                compiled = jax.jit(
                    shard_map(
                        make_scalar_kernel(operator_name),
                        mesh=mesh,
                        in_specs=(
                            state_spec,
                            field_spec,
                            invariant_spec,
                            control_volume_spec,
                            P(),
                        ),
                        out_specs=(
                            field_spec,
                            field_spec,
                            P(),
                            P(),
                            P(),
                            P(),
                        ),
                        check_rep=False,
                    )
                )
                start = time_module.perf_counter()
                (
                    actual_mesh,
                    reference_mesh,
                    remote_flux_sum,
                    remote_flux_abs_sum,
                    invalid_remote_quadrature,
                    invalid_reconstruction_rows,
                ) = compiled(
                    state_mesh,
                    phi_mesh,
                    invariants_mesh,
                    control_volume_mesh,
                    jnp.asarray(0.0, dtype=jnp.float64),
                )
                jax.block_until_ready(actual_mesh)
                elapsed = time_module.perf_counter() - start
                # Keep diagnostics on the host. Re-wrapping these arrays with
                # jnp.asarray would copy each result back to the accelerator
                # and retain another device allocation across the sweep.
                actual = np.asarray(
                    jax.device_get(actual_mesh),
                    dtype=np.float64,
                )
                reference = np.asarray(
                    jax.device_get(reference_mesh),
                    dtype=np.float64,
                )
                statistics = _operator_category_statistics(
                    actual,
                    reference,
                    volume,
                    categories,
                )
                print(
                    f"N={resolution} operator={operator_name} "
                    f"compile+run={elapsed:.3f}s "
                    "invalid_reconstruction_rows="
                    f"{int(np.asarray(jax.device_get(invalid_reconstruction_rows)))}"
                )
                if int(
                    np.asarray(jax.device_get(invalid_reconstruction_rows))
                ):
                    raise AssertionError(
                        f"{operator_name} produced invalid active "
                        "quadratic reconstruction rows"
                    )
                if operator_name == "parallel_density_flux_divergence":
                    remote_flux_sum_value = float(
                        np.asarray(jax.device_get(remote_flux_sum))
                    )
                    remote_flux_abs_sum_value = float(
                        np.asarray(jax.device_get(remote_flux_abs_sum))
                    )
                    invalid_remote_quadrature_value = int(
                        np.asarray(
                            jax.device_get(invalid_remote_quadrature)
                        )
                    )
                    print(
                        "  mirrored_remote_flux signed_sum="
                        f"{remote_flux_sum_value:.6e} "
                        f"abs_sum={remote_flux_abs_sum_value:.6e} "
                        "relative_imbalance="
                        f"{abs(remote_flux_sum_value) / max(remote_flux_abs_sum_value, 1.0e-30):.6e} "
                        "invalid_quadrature="
                        f"{invalid_remote_quadrature_value}"
                    )
                    remote_relative_imbalance = (
                        abs(remote_flux_sum_value)
                        / max(remote_flux_abs_sum_value, 1.0e-30)
                    )
                    if invalid_remote_quadrature_value:
                        raise AssertionError(
                            "mirrored remote interfaces contain invalid "
                            f"quadrature samples: {invalid_remote_quadrature_value}"
                        )
                    if (
                        remote_flux_abs_sum_value > 1.0e-14
                        and remote_relative_imbalance > 1.0e-12
                    ):
                        raise AssertionError(
                            "mirrored remote interface fluxes do not cancel: "
                            f"relative imbalance={remote_relative_imbalance:.6e}"
                        )
                for category, (
                    l2,
                    linf,
                    relative,
                    count,
                ) in statistics.items():
                    print(
                        f"  {category:18s} count={count:8d} "
                        f"volume_L2={l2:.6e} Linf={linf:.6e} "
                        f"rel_L2={relative:.6e}"
                    )
                    records.setdefault(operator_name, {}).setdefault(
                        category,
                        [],
                    ).append((resolution, l2, linf))
                active_mask = np.asarray(
                    cell_data["is_active_owner"],
                    dtype=bool,
                )
                absolute_error = np.where(
                    active_mask,
                    np.abs(
                        np.asarray(actual, dtype=np.float64)
                        - np.asarray(reference, dtype=np.float64)
                    ),
                    -np.inf,
                )
                top_flat = int(np.argmax(absolute_error))
                top_index = tuple(
                    int(value)
                    for value in np.unravel_index(top_flat, shape)
                )
                top_is_compact = (
                    int(
                        np.asarray(
                            cell_data["irregular_face_count"]
                        )[top_index]
                    )
                    > 0
                    or int(
                        np.asarray(
                            cell_data["reconstruction_row_count"]
                        )[top_index]
                    )
                    > 0
                    or bool(
                        np.asarray(
                            cell_data["is_aggregate_target"]
                        )[top_index]
                    )
                )
                if top_is_compact:
                    top_compact_distance = 0
                elif bool(
                    np.asarray(categories["dense_compact_d1"])[top_index]
                ):
                    top_compact_distance = 1
                elif bool(
                    np.asarray(categories["dense_compact_d2"])[top_index]
                ):
                    top_compact_distance = 2
                else:
                    top_compact_distance = 3
                print(
                    "  top_error index={} error={:.6e} actual={:.6e} "
                    "reference={:.6e} regular_physical_boundary={} "
                    "embedded_cutwall_faces={} irregular_faces={} "
                    "remote_faces={} reconstruction_rows={} aggregate={} "
                    "compact_distance={}".format(
                        top_index,
                        float(absolute_error[top_index]),
                        float(np.asarray(actual)[top_index]),
                        float(np.asarray(reference)[top_index]),
                        bool(
                            top_index[0] == 0
                            or top_index[0] == shape[0] - 1
                        ),
                        int(np.asarray(cell_data["boundary_face_count"])[top_index]),
                        int(np.asarray(cell_data["irregular_face_count"])[top_index]),
                        int(np.asarray(cell_data["remote_face_count"])[top_index]),
                        int(
                            np.asarray(
                                cell_data["reconstruction_row_count"]
                            )[top_index]
                        ),
                        bool(
                            np.asarray(
                                cell_data["is_aggregate_target"]
                            )[top_index]
                        ),
                        top_compact_distance,
                    )
                )

                # Each operator creates a distinct large shard_map executable.
                # Release its device outputs and compiled executable before
                # compiling the next operator in this diagnostic sweep.
                del (
                    compiled,
                    actual_mesh,
                    reference_mesh,
                    remote_flux_sum,
                    remote_flux_abs_sum,
                    invalid_remote_quadrature,
                    invalid_reconstruction_rows,
                    actual,
                    reference,
                    statistics,
                )
                jax.clear_caches()
                gc.collect()
            def full_rhs_kernel(
                state_owned,
                phi_owned,
                local_invariants,
                control_volume_geometry,
                stage_time,
            ):
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                stage = shifted_mms._build_local_4field_stage_data(
                    local_invariants,
                    stage_time,
                    parameters=parameters,
                )
                stage = _with_shifted_torus_regular_radial_face_averages(
                    stage,
                    local_geometry_value,
                    stage_time,
                )
                source_owner = _project_local_mms_source_to_control_volumes(
                    local_geometry_value,
                    control_volume_geometry,
                    stage_time,
                    parameters,
                )
                cells = control_volume_geometry.cells
                source_storage = Fci4FieldState(
                    density=_expand_control_volume_owner_values(
                        source_owner.density,
                        cells,
                    ),
                    omega=_expand_control_volume_owner_values(
                        source_owner.omega,
                        cells,
                    ),
                    v_ion_parallel=_expand_control_volume_owner_values(
                        source_owner.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_expand_control_volume_owner_values(
                        source_owner.v_electron_parallel,
                        cells,
                    ),
                )
                phi_storage = _expand_control_volume_owner_values(
                    phi_owned,
                    cells,
                )
                stage = dataclass_replace(
                    stage,
                    source_halo=inject_owned_state_to_halo(
                        source_storage,
                        domain.layout,
                    ),
                    phi_halo=inject_owned_field_to_halo(
                        phi_storage,
                        domain.layout,
                    ),
                )
                rhs = LocalShiftedTorus4FieldCutWallRhs(
                    geometry=local_geometry_value,
                    domain=domain,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=physical_ghost_filler,
                    parameters=parameters,
                    curvature_coefficients_owned=(
                        local_invariants.curvature_coefficients_owned
                    ),
                    face_projectors=(
                        local_invariants.face_projector_x,
                        local_invariants.face_projector_y,
                        local_invariants.face_projector_z,
                    ),
                    gmres_config=gmres_config,
                    global_shape=shape,
                    control_volume_geometry=control_volume_geometry,
                )
                actual, _phi = rhs.evaluate_stage(
                    state_owned,
                    stage,
                    phi_owned,
                    solve_phi=bool(check_phi_solve),
                )
                reference = (
                    _project_local_exact_time_derivative_to_control_volumes(
                        local_geometry_value,
                        control_volume_geometry,
                        stage_time,
                    )
                )
                source_roundtrip = Fci4FieldState(
                    density=_agglomerate_control_volume_average(
                        source_storage.density,
                        cells,
                    ),
                    omega=_agglomerate_control_volume_average(
                        source_storage.omega,
                        cells,
                    ),
                    v_ion_parallel=_agglomerate_control_volume_average(
                        source_storage.v_ion_parallel,
                        cells,
                    ),
                    v_electron_parallel=_agglomerate_control_volume_average(
                        source_storage.v_electron_parallel,
                        cells,
                    ),
                )
                active = cells.is_active_owner

                def max_active(value: jnp.ndarray) -> jnp.ndarray:
                    result = jnp.max(
                        jnp.where(active, jnp.abs(value), 0.0)
                    )
                    for mesh_axis_name in MESH_AXIS_NAMES:
                        result = lax.pmax(result, mesh_axis_name)
                    return result

                source_diagnostics = jnp.stack(
                    (
                        max_active(
                            source_owner.v_ion_parallel
                            - reference.v_ion_parallel
                        ),
                        max_active(
                            source_roundtrip.v_ion_parallel
                            - source_owner.v_ion_parallel
                        ),
                        max_active(
                            actual.v_ion_parallel
                            - source_roundtrip.v_ion_parallel
                        ),
                        max_active(
                            actual.v_ion_parallel
                            - reference.v_ion_parallel
                        ),
                    )
                )
                return actual, reference, source_diagnostics

            compiled_full_rhs = jax.jit(
                shard_map(
                    full_rhs_kernel,
                    mesh=mesh,
                    in_specs=(
                        state_spec,
                        field_spec,
                        invariant_spec,
                        control_volume_spec,
                        P(),
                    ),
                    out_specs=(state_spec, state_spec, P()),
                    check_rep=False,
                )
            )
            start = time_module.perf_counter()
            (
                actual_rhs_mesh,
                reference_rhs_mesh,
                source_diagnostics,
            ) = compiled_full_rhs(
                state_mesh,
                phi_mesh,
                invariants_mesh,
                control_volume_mesh,
                jnp.asarray(0.0, dtype=jnp.float64),
            )
            jax.block_until_ready(actual_rhs_mesh.density)
            elapsed = time_module.perf_counter() - start
            actual_rhs = shifted_mms._gather_state_from_mesh(
                actual_rhs_mesh
            )
            reference_rhs = shifted_mms._gather_state_from_mesh(
                reference_rhs_mesh
            )
            print(
                f"N={resolution} operator=full_rhs "
                f"phi_mode={'solved' if check_phi_solve else 'projected_exact'} "
                f"compile+run={elapsed:.3f}s"
            )
            source_diagnostics_host = np.asarray(
                jax.device_get(source_diagnostics),
                dtype=np.float64,
            )
            print(
                "  v_ion source consistency: "
                "source_vs_exact_t={:.6e} "
                "roundtrip_vs_source={:.6e} "
                "operator_without_source={:.6e} "
                "full_residual={:.6e}".format(
                    *source_diagnostics_host,
                )
            )
            for field_name in (
                "density",
                "omega",
                "v_ion_parallel",
                "v_electron_parallel",
            ):
                operator_name = f"full_rhs_{field_name}"
                statistics = _operator_category_statistics(
                    getattr(actual_rhs, field_name),
                    getattr(reference_rhs, field_name),
                    volume,
                    categories,
                )
                print(f"  field={field_name}")
                for category, (
                    l2,
                    linf,
                    relative,
                    count,
                ) in statistics.items():
                    print(
                        f"    {category:16s} count={count:8d} "
                        f"volume_L2={l2:.6e} Linf={linf:.6e} "
                        f"rel_L2={relative:.6e}"
                    )
                    records.setdefault(operator_name, {}).setdefault(
                        category,
                        [],
                    ).append((resolution, l2, linf))

            if not bool(check_phi_solve):
                print(f"N={resolution} phi algebraic solve skipped")
                continue

            def phi_solve_kernel(
                state_owned,
                phi_owned,
                local_invariants,
                control_volume_geometry,
                stage_time,
            ):
                local_invariants = extract_local_shard_pytree(
                    local_invariants
                )
                control_volume_geometry = extract_local_shard_pytree(
                    control_volume_geometry
                )
                local_geometry_value = local_geometry(
                    control_volume_geometry
                )
                phi_face_bc = regular_face_bc(
                    local_geometry_value,
                    stage_time,
                    "phi",
                )
                phi_boundary_bc = _control_volume_exact_boundary_bc(
                    control_volume_geometry,
                    stage_time,
                    "phi",
                )
                solver = LocalPerpLaplacianInverseSolver(
                    geometry=local_geometry_value,
                    domain=domain,
                    stencil_builder=(
                        build_local_conservative_stencil_from_field
                    ),
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=physical_ghost_filler,
                    face_projectors=(
                        local_invariants.face_projector_x,
                        local_invariants.face_projector_y,
                        local_invariants.face_projector_z,
                    ),
                    control_volume_geometry=control_volume_geometry,
                    control_volume_boundary_bc=phi_boundary_bc,
                    face_bc=phi_face_bc,
                    config=gmres_config,
                )
                solution, info = solver(
                    -state_owned.omega,
                    guess_owned=phi_owned,
                    phi_lift_owned=phi_owned,
                    return_diagnostics=True,
                )
                return (
                    solution,
                    info.final_residual_rel_l2,
                    info.converged,
                    info.failed,
                    info.num_steps,
                    info.initial_residual_l2,
                    info.final_residual_l2,
                    info.rhs_l2,
                )

            compiled_phi_solve = jax.jit(
                shard_map(
                    phi_solve_kernel,
                    mesh=mesh,
                    in_specs=(
                        state_spec,
                        field_spec,
                        invariant_spec,
                        control_volume_spec,
                        P(),
                    ),
                    out_specs=(
                        field_spec,
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    check_rep=False,
                )
            )
            (
                solved_phi,
                relative_residual,
                converged,
                phi_failed,
                phi_num_steps,
                phi_initial_residual,
                phi_final_residual,
                phi_rhs_l2,
            ) = compiled_phi_solve(
                state_mesh,
                phi_mesh,
                invariants_mesh,
                control_volume_mesh,
                jnp.asarray(0.0, dtype=jnp.float64),
            )
            jax.block_until_ready(solved_phi)
            relative_residual_value = float(
                np.asarray(jax.device_get(relative_residual))
            )
            converged_value = bool(
                np.asarray(jax.device_get(converged))
            )
            failed_value = bool(
                np.asarray(jax.device_get(phi_failed))
            )
            num_steps_value = int(
                np.asarray(jax.device_get(phi_num_steps))
            )
            initial_residual_value = float(
                np.asarray(jax.device_get(phi_initial_residual))
            )
            final_residual_value = float(
                np.asarray(jax.device_get(phi_final_residual))
            )
            rhs_l2_value = float(
                np.asarray(jax.device_get(phi_rhs_l2))
            )
            phi_residuals.append(
                (resolution, relative_residual_value)
            )
            print(
                f"N={resolution} phi algebraic residual="
                f"{relative_residual_value:.6e}, converged={converged_value}, "
                f"failed={failed_value}, steps={num_steps_value}, "
                f"initial={initial_residual_value:.6e}, "
                f"final={final_residual_value:.6e}, rhs_l2={rhs_l2_value:.6e}"
            )
            if (
                not np.isfinite(relative_residual_value)
                or relative_residual_value > 5.0e-5
            ):
                raise AssertionError(
                    "phi solve failed operator-convergence acceptance: "
                    f"N={resolution}, residual={relative_residual_value:.6e}, "
                    f"converged={converged_value}, failed={failed_value}"
                )

    order_results: dict[
        str,
        dict[str, tuple[float | None, float | None]],
    ] = {}
    failed_orders: list[str] = []
    for operator_name, category_records in records.items():
        order_results[operator_name] = {}
        for category, values in category_records.items():
            category_resolutions = [value[0] for value in values]
            exact_to_roundoff = bool(values) and all(
                np.isfinite(value[1])
                and np.isfinite(value[2])
                and abs(value[1]) <= 1.0e-12
                and abs(value[2]) <= 1.0e-12
                for value in values
            )
            l2_order = _fit_operator_order(
                category_resolutions,
                [value[1] for value in values],
            )
            linf_order = _fit_operator_order(
                category_resolutions,
                [value[2] for value in values],
            )
            order_results[operator_name][category] = (
                l2_order,
                linf_order,
            )
            l2_text = "n/a" if l2_order is None else f"{l2_order:.6f}"
            linf_text = (
                "n/a" if linf_order is None else f"{linf_order:.6f}"
            )
            if exact_to_roundoff:
                l2_text = "exact"
                linf_text = "exact"
            print(
                f"operator order {operator_name} {category}: "
                f"volume_L2={l2_text}, Linf={linf_text}"
            )
            if len(resolutions) >= 2 and category == "all_active":
                if (
                    not exact_to_roundoff
                    and (
                        l2_order is None
                        or l2_order < float(minimum_order)
                    )
                ):
                    failed_orders.append(
                        f"{operator_name}/{category} L2={l2_text}"
                    )
                if (
                    not exact_to_roundoff
                    and (
                        linf_order is None
                        or linf_order < float(minimum_order)
                    )
                ):
                    failed_orders.append(
                        f"{operator_name}/{category} Linf={linf_text}"
                    )
    if failed_orders:
        raise AssertionError(
            "operator convergence acceptance failed (minimum order "
            f"{float(minimum_order):.3f}): "
            + "; ".join(failed_orders)
        )
    return {
        "records": records,
        "orders": order_results,
        "phi_residuals": phi_residuals,
    }


def run_shifted_torus_4field_cutwall_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = shifted_mms.tf,
    base_steps: int = shifted_mms.num_steps,
    rho_star_value: float = shifted_mms.rho_star,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
    enable_agglomeration: bool = False,
    minimum_order: float | None = None,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    per_resolution_stats: list[tuple[int, dict[str, tuple[float, float, float]]]] = []

    for resolution in resolutions:
        shape = _shape_from_resolution(int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = shifted_mms.build_shifted_torus_4field_geometry(shape)
        stacked_control_volume_geometry = (
            _build_stacked_embedded_control_volume_geometry(
                global_shape=shape,
                shard_counts=shard_counts,
                halo_width=halo_width,
                enable_merging=enable_agglomeration,
            )
        )
        steps = _resolution_step_count(int(resolution), base_steps=base_steps)
        dt = float(final_time) / float(steps)
        print(
            f"Starting shifted_torus_4field_cutwall MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}, dt={dt:.6e}, "
            f"enable_agglomeration={enable_agglomeration}"
        )
        _print_control_volume_geometry_summary(
            stacked_control_volume_geometry
        )
        start = time_module.perf_counter()
        final_state, *_ = simulate_mms_shifted_torus_4field_cutwall(
            geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=final_time,
            timestep=dt,
            rho_star_value=rho_star_value,
            show_progress=show_progress,
            enable_agglomeration=enable_agglomeration,
            stacked_control_volume_geometry=stacked_control_volume_geometry,
        )
        elapsed = time_module.perf_counter() - start
        exact_state, _exact_phi = _project_global_exact_state_to_control_volumes(
            geometry,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=final_time,
        )
        control_volume_cells = _assemble_global_control_volume_cell_data(
            geometry.shape,
            stacked_control_volume_geometry,
            shard_counts=shard_counts,
        )
        active_mask = control_volume_cells["is_active_owner"]
        aggregate_volume = control_volume_cells["aggregate_volume"]
        solid_mask = ~active_mask
        abs_errors = [
            jnp.abs(final_state.density - exact_state.density),
            jnp.abs(final_state.omega - exact_state.omega),
            jnp.abs(final_state.v_ion_parallel - exact_state.v_ion_parallel),
            jnp.abs(final_state.v_electron_parallel - exact_state.v_electron_parallel),
        ]
        active_errors = [jnp.where(active_mask, error, 0.0) for error in abs_errors]
        weight_sum = jnp.maximum(jnp.sum(aggregate_volume), 1.0e-30)
        sumsq_error = sum(
            jnp.sum(aggregate_volume * jnp.square(error))
            for error in active_errors
        )
        mean_error = float(
            jnp.sqrt(
                sumsq_error
                / (weight_sum * float(len(active_errors)))
            )
        )
        active_mask_host = np.asarray(active_mask, dtype=bool)
        solid_mask_host = np.asarray(solid_mask, dtype=bool)
        masked_values = np.concatenate(
            [
                np.asarray(error, dtype=np.float64)[active_mask_host].ravel()
                for error in abs_errors
            ]
        )
        active_nonfinite = int(np.count_nonzero(~np.isfinite(masked_values)))
        solid_values = np.concatenate(
            [
                np.asarray(error, dtype=np.float64)[solid_mask_host].ravel()
                for error in abs_errors
            ]
        )
        solid_nonfinite = int(np.count_nonzero(~np.isfinite(solid_values)))
        finite_masked_values = masked_values[np.isfinite(masked_values)]
        median_error = float(np.median(finite_masked_values)) if finite_masked_values.size else float("nan")
        max_error = float(np.max(finite_masked_values)) if finite_masked_values.size else float("nan")
        per_field_stats = _volume_weighted_state_error_statistics(
            final_state,
            exact_state,
            aggregate_volume,
            active_mask,
        )
        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        per_resolution_stats.append((int(resolution), per_field_stats))
        print(
            f"N={int(resolution)}: shard_counts={shard_counts}, steps={steps}, "
            f"total_runtime={elapsed:.6e} s, avg_step_runtime={elapsed / float(steps):.6e} s, "
            f"L2={mean_error:.6e}, median={median_error:.6e}, Linf={max_error:.6e}, "
            f"active_nonfinite={active_nonfinite}, solid_nonfinite={solid_nonfinite}"
        )
        if active_nonfinite or solid_nonfinite:
            raise AssertionError(
                "shifted-torus control-volume state contains nonfinite values: "
                f"N={int(resolution)}, active={active_nonfinite}, "
                f"inactive_or_source={solid_nonfinite}"
            )
        _print_state_error_statistics(f"N={int(resolution)} per-field final errors", per_field_stats)

    l2_order: float | None = None
    max_order: float | None = None
    per_field_orders: dict[str, tuple[float | None, float | None]] = {}
    per_field_exact_to_roundoff: dict[str, bool] = {}
    if len(successful_resolutions) >= 2:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.float64)
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(np.log(plotted_resolutions), l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(np.log(plotted_resolutions), max_log_errors, 1)
        l2_order = float(-l2_slope)
        max_order = float(-max_slope)
        print(f"shifted_torus_4field_cutwall L2 convergence order: {l2_order:.6f}")
        print(f"shifted_torus_4field_cutwall Linf convergence order: {max_order:.6f}")
        for field_name in (
            "density",
            "omega",
            "v_ion_parallel",
            "v_electron_parallel",
        ):
            field_l2 = np.asarray(
                [
                    statistics[field_name][0]
                    for _resolution, statistics in per_resolution_stats
                ],
                dtype=np.float64,
            )
            field_linf = np.asarray(
                [
                    statistics[field_name][1]
                    for _resolution, statistics in per_resolution_stats
                ],
                dtype=np.float64,
            )
            exact_to_roundoff = bool(field_l2.size) and bool(field_linf.size) and (
                bool(np.all(np.isfinite(field_l2)))
                and bool(np.all(np.isfinite(field_linf)))
                and bool(np.all(np.abs(field_l2) <= 1.0e-12))
                and bool(np.all(np.abs(field_linf) <= 1.0e-12))
            )
            per_field_exact_to_roundoff[field_name] = exact_to_roundoff
            field_l2_order: float | None = None
            field_linf_order: float | None = None
            if not exact_to_roundoff:
                field_l2_order = float(
                    -np.polyfit(
                        np.log(plotted_resolutions),
                        np.log(field_l2),
                        1,
                    )[0]
                )
                field_linf_order = float(
                    -np.polyfit(
                        np.log(plotted_resolutions),
                        np.log(field_linf),
                        1,
                    )[0]
                )
            per_field_orders[field_name] = (field_l2_order, field_linf_order)
            l2_text = "exact" if exact_to_roundoff else f"{field_l2_order:.6f}"
            linf_text = "exact" if exact_to_roundoff else f"{field_linf_order:.6f}"
            print(
                "shifted_torus_4field_cutwall "
                f"{field_name} orders: volume_L2={l2_text}, "
                f"active_owner_Linf={linf_text}"
            )
        if minimum_order is not None:
            failed_orders = [
                (
                    f"{field_name}: volume_L2="
                    f"{'n/a' if orders[0] is None else f'{orders[0]:.6f}'}, "
                    f"active_owner_Linf="
                    f"{'n/a' if orders[1] is None else f'{orders[1]:.6f}'}"
                )
                for field_name, orders in per_field_orders.items()
                if (
                    not per_field_exact_to_roundoff[field_name]
                    and (
                        orders[0] is None
                        or orders[1] is None
                        or not np.isfinite(orders[0])
                        or not np.isfinite(orders[1])
                        or orders[0] < float(minimum_order)
                        or orders[1] < float(minimum_order)
                    )
                )
            ]
            if failed_orders:
                raise AssertionError(
                    "shifted-torus convergence acceptance failed "
                    f"(minimum order {float(minimum_order):.3f}): "
                    + "; ".join(failed_orders)
                )
        if plot:
            import matplotlib.pyplot as plt

            output_path = Path(plot_path or "shifted_torus_4field_cutwall_convergence.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"L2, order {l2_order:.2f}")
            ax.loglog(plotted_resolutions, max_errors, "^-", label=f"Linf, order {max_order:.2f}")
            ax.loglog(
                plotted_resolutions,
                np.exp(l2_intercept) * plotted_resolutions**l2_slope,
                "--",
                color=ax.lines[0].get_color(),
            )
            ax.loglog(
                plotted_resolutions,
                np.exp(max_intercept) * plotted_resolutions**max_slope,
                "--",
                color=ax.lines[1].get_color(),
            )
            ax.set_xlabel("resolution")
            ax.set_ylabel("absolute error")
            ax.set_title(f"Shifted-torus 4-field cut-wall MMS ({shard_counts})")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
        "l2_order": l2_order,
        "linf_order": max_order,
        "per_field": per_resolution_stats,
        "per_field_orders": per_field_orders,
    }
