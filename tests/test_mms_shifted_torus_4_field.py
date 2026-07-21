from __future__ import annotations

import argparse
import time as time_module
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    ConservativeStencilBuilder,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    Grid1D,
    LocalStencilBuilder,
    MetricGeometry,
    RegularFaceGeometry3D,
    Spacing3D,
    build_curvature_coefficients,
    build_conservative_stencil_from_field,
    build_fci_maps_from_b_contravariant,
    build_local_conservative_stencil_from_field,
    build_local_curvature_coefficients,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
    StencilBuilderContext,
)
from jax_drb.native import (
    Fci4FieldRhsParameters,
    Fci4FieldState,
    curvature_op,
    build_perp_laplacian_mg_hierarchy,
    build_perp_laplacian_face_projectors,
    compute_4field_rhs,
    poisson_bracket_op,
    perp_laplacian_conservative_op,
    SpmdGmresConfig,
)
from jax_drb.native.fci_model import FciFieldBundle, inject_owned_state_to_halo
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    PreparedLocalState3D,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
)
from jax_drb.native.fci_operators import (
    _build_global_conservative_stencil_compat,
    LocalPerpLaplacianInverseSolver,
    PerpLaplacianInverseSolver,
    build_local_perp_laplacian_face_projectors,
    grad_parallel_op_direct,
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_poisson_bracket_op,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BoundaryConditionBuilder,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
)

from mms_domain_decomp_helpers import (
    MESH_AXIS_NAMES,
    assert_shape_divisible_by_shards,
    build_shifted_torus_local_domain,
    build_shifted_torus_local_geometry,
    expand_local_shard_pytree,
    extract_local_shard_pytree,
    local_shard_pytree_partition_spec,
    make_mesh_for_shard_counts,
)


from shifted_torus_4field_mms_helpers import *  # noqa: F403

def _print_convergence_table(
    title: str,
    results: list[tuple[int, dict[str, tuple[float, float, float]]]],
    *,
    stat_index: int = 0,
) -> None:
    print(title)
    previous_resolution: int | None = None
    previous_stats: dict[str, tuple[float, float, float]] | None = None
    for resolution, stats in results:
        print(f"  resolution={resolution}")
        for field_name, field_stats in stats.items():
            order_text = ""
            if previous_resolution is not None and previous_stats is not None and field_name in previous_stats:
                order = _observed_order(
                    previous_stats[field_name][stat_index],
                    field_stats[stat_index],
                    previous_resolution,
                    resolution,
                )
                order_text = f", order={_format_order(order)}"
            l2, linf, rel_l2 = field_stats
            print(f"    {field_name}: l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}{order_text}")
        previous_resolution = resolution
        previous_stats = stats


def _discrete_minus_laplacian_phi(
    geometry: FciGeometry3D,
    time: float,
    *,
    conservative_stencil_builder: ConservativeStencilBuilder,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    periodic_axes = (False, True, True)
    exact_phi = _shifted_torus_phi(geometry, time)
    phi_face_bc, _, _, _, _ = _shifted_torus_exact_x_face_bcs(geometry, time)
    phi_stencil = _build_global_conservative_stencil_compat(
        conservative_stencil_builder,
        exact_phi,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=phi_face_bc,
    )
    return -perp_laplacian_conservative_op(
        phi_stencil,
        geometry,
        face_projectors=face_projectors,
        face_bc=phi_face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=periodic_axes,
    )


def _phi_vorticity_mismatch_statistics(
    geometry: FciGeometry3D,
    time: float,
) -> dict[str, tuple[float, float, float]]:
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    discrete_minus_lap_phi = _discrete_minus_laplacian_phi(
        geometry,
        time,
        conservative_stencil_builder=conservative_stencil_builder,
        face_projectors=face_projectors,
    )
    exact_omega_rhs = -_shifted_torus_exact_state(geometry, time).omega
    return {"-lap_perp(phi) vs -omega": _field_error_statistics(discrete_minus_lap_phi, exact_omega_rhs)}


def _discrete_4field_rhs_terms_with_phi(
    state: Fci4FieldState,
    phi: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
    face_bcs: tuple[BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D],
    conservative_stencil_builder: ConservativeStencilBuilder,
) -> dict[str, dict[str, jnp.ndarray]]:
    periodic_axes = (False, True, True)
    phi_face_bc, density_face_bc, omega_face_bc, v_ion_face_bc, v_electron_face_bc = face_bcs
    density = jnp.asarray(state.density, dtype=jnp.float64)
    omega = jnp.asarray(state.omega, dtype=jnp.float64)
    v_ion_parallel = jnp.asarray(state.v_ion_parallel, dtype=jnp.float64)
    v_electron_parallel = jnp.asarray(state.v_electron_parallel, dtype=jnp.float64)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me_value = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)

    def _stencil(field: jnp.ndarray, face_bc: BoundaryFaceBC3D):
        return _build_global_conservative_stencil_compat(
            conservative_stencil_builder,
            field,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=face_bc,
        )

    density_stencil = _stencil(density, density_face_bc)
    omega_stencil = _stencil(omega, omega_face_bc)
    phi_stencil = _stencil(phi, phi_face_bc)
    v_ion_stencil = _stencil(v_ion_parallel, v_ion_face_bc)
    v_electron_stencil = _stencil(
        v_electron_parallel,
        v_electron_face_bc,
    )
    density_v_electron_face_bc = density_face_bc.replace(
        value_x=density_face_bc.value_x * v_electron_face_bc.value_x,
        value_y=density_face_bc.value_y * v_electron_face_bc.value_y,
        value_z=density_face_bc.value_z * v_electron_face_bc.value_z,
    )
    density_v_electron_stencil = _stencil(
        density * v_electron_parallel,
        density_v_electron_face_bc,
    )

    poisson_density = poisson_bracket_op(phi_stencil, density_stencil, geometry)
    poisson_omega = poisson_bracket_op(phi_stencil, omega_stencil, geometry)
    poisson_v_ion = poisson_bracket_op(phi_stencil, v_ion_stencil, geometry)
    poisson_v_electron = poisson_bracket_op(phi_stencil, v_electron_stencil, geometry)
    curvature_density = curvature_op(density_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_phi = curvature_op(phi_stencil, geometry, curvature_coefficients=curvature_coefficients)
    grad_parallel_density = grad_parallel_op_direct(density_stencil, geometry)
    grad_parallel_phi = grad_parallel_op_direct(phi_stencil, geometry)
    grad_parallel_v_ion = grad_parallel_op_direct(v_ion_stencil, geometry)
    grad_parallel_v_electron = grad_parallel_op_direct(v_electron_stencil, geometry)
    grad_parallel_density_v_electron = grad_parallel_op_direct(
        density_v_electron_stencil,
        geometry,
    )

    return {
        "density": {
            "poisson": -(poisson_density / (rho_star_value * bmag)),
            "curvature_density": (2.0 * te / bmag) * curvature_density,
            "curvature_phi": -(2.0 * density / bmag) * curvature_phi,
            "parallel_density_v_electron": -grad_parallel_density_v_electron,
        },
        "omega": {
            "poisson": -(poisson_omega / (rho_star_value * bmag)),
            "parallel_current": (bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron),
            "curvature_density": (2.0 * bmag * te / density_safe) * curvature_density,
        },
        "v_ion_parallel": {
            "poisson": -(poisson_v_ion / (rho_star_value * bmag)),
            "grad_density": -(te / density_safe) * grad_parallel_density,
        },
        "v_electron_parallel": {
            "poisson": -(poisson_v_electron / (rho_star_value * bmag)),
            "grad_phi": mi_over_me_value * grad_parallel_phi,
            "grad_density": -mi_over_me_value * (te / density_safe) * grad_parallel_density,
        },
    }


def _exact_phi_rhs_consistency_statistics(
    geometry: FciGeometry3D,
    time: float,
    *,
    parameters: Fci4FieldRhsParameters,
) -> dict[str, tuple[float, float, float]]:
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_terms = _discrete_4field_rhs_terms_with_phi(
        exact_state,
        _shifted_torus_phi(geometry, time),
        geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        face_bcs=_shifted_torus_exact_x_face_bcs(geometry, time),
        conservative_stencil_builder=ConservativeStencilBuilder(
            build_conservative_stencil_from_field.build_fn
        ),
    )
    computed_derivative = _add_state(_sum_rhs_terms(discrete_terms), source, scale=1.0)
    return _state_error_statistics(computed_derivative, exact_derivative)


def _report_exact_phi_term_breakdown(
    geometry: FciGeometry3D,
    time: float,
    *,
    parameters: Fci4FieldRhsParameters,
) -> None:
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    continuous_terms = _continuous_4field_rhs_terms_from_exact_state(
        exact_state,
        geometry,
        time=time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_terms = _discrete_4field_rhs_terms_with_phi(
        exact_state,
        _shifted_torus_phi(geometry, time),
        geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        face_bcs=_shifted_torus_exact_x_face_bcs(geometry, time),
        stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
    )
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_rhs = _sum_rhs_terms(discrete_terms)
    computed_derivative = _add_state(discrete_rhs, source, scale=1.0)
    total_stats = _state_error_statistics(computed_derivative, exact_derivative)
    print(f"shifted_torus_4field exact-phi per-term RHS residual breakdown at resolution={geometry.shape[0]}")
    for field_name in ("density", "omega", "v_ion_parallel", "v_electron_parallel"):
        print(
            f"  {field_name}: total_l2={total_stats[field_name][0]:.6e}, "
            f"total_linf={total_stats[field_name][1]:.6e}, total_rel_l2={total_stats[field_name][2]:.6e}"
        )
        for term_name, discrete_value in discrete_terms[field_name].items():
            continuous_value = continuous_terms[field_name][term_name]
            l2, linf, rel_l2 = _field_error_statistics(discrete_value, continuous_value)
            print(f"    {term_name}: term_error_l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")
        source_value = getattr(source, field_name)
        expected_source = getattr(exact_derivative, field_name) - getattr(_sum_rhs_terms(continuous_terms), field_name)
        l2, linf, rel_l2 = _field_error_statistics(source_value, expected_source)
        print(f"    source_cancellation: term_error_l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")


def _report_rhs_consistency(
    geometry: FciGeometry3D,
    *,
    time: float,
    rho_star_value: float,
    use_multigrid_preconditioner: bool = False,
    gmres_debug: bool = False,
) -> None:
    """Compare discrete RHS plus MMS source against exact time derivatives."""

    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=1.0e-4,
    )
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_builder = BoundaryConditionBuilder(_build_dirichlet_boundary_condition_builder("density"))
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    periodic_axes = (False, True, True)

    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=periodic_axes)
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    exact_phi = _shifted_torus_phi(geometry, time)

    phi_face_bc, density_face_bc, omega_face_bc, v_ion_face_bc, v_electron_face_bc = _shifted_torus_exact_x_face_bcs(
        geometry,
        time,
    )

    mg_hierarchy = None
    if use_multigrid_preconditioner:
        mg_hierarchy = build_perp_laplacian_mg_hierarchy(
            geometry,
            conservative_stencil_builder,
            face_bc=_homogeneous_boundary_face_bc(phi_face_bc),
            regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
            cut_wall_geometry=empty_cut_wall_geometry,
            cut_wall_bc=empty_cut_wall_bc,
            periodic_axes=periodic_axes,
        )
    _report_phi_inversion_consistency(
        geometry,
        time=time,
        parameters=parameters,
        conservative_stencil_builder=conservative_stencil_builder,
        boundary_builder=boundary_builder,
        face_projectors=face_projectors,
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=periodic_axes,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )

    rhs_result, _timings = compute_4field_rhs(
        exact_state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_face_bc,
        v_electron_parallel_face_bc=v_electron_face_bc,
        phi_cut_wall_geometry=empty_cut_wall_geometry,
        phi_cut_wall_bc=empty_cut_wall_bc,
        density_cut_wall_geometry=empty_cut_wall_geometry,
        density_cut_wall_bc=empty_cut_wall_bc,
        omega_cut_wall_geometry=empty_cut_wall_geometry,
        omega_cut_wall_bc=empty_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=empty_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=empty_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=empty_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=empty_cut_wall_bc,
        phi_face_projectors=face_projectors,
        phi_mg_hierarchy=mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
    )
    computed_derivative = _add_state(rhs_result.rhs, source, scale=1.0)
    print(f"shifted_torus_4field RHS consistency at t={time:.6e}, shape={geometry.shape}")
    for field_name, actual, expected in (
        ("density", computed_derivative.density, exact_derivative.density),
        ("omega", computed_derivative.omega, exact_derivative.omega),
        ("v_ion_parallel", computed_derivative.v_ion_parallel, exact_derivative.v_ion_parallel),
        ("v_electron_parallel", computed_derivative.v_electron_parallel, exact_derivative.v_electron_parallel),
    ):
        error = jnp.asarray(actual - expected, dtype=jnp.float64)
        expected_norm = float(jnp.linalg.norm(jnp.asarray(expected, dtype=jnp.float64)))
        error_l2 = float(jnp.sqrt(jnp.mean(jnp.square(error))))
        error_linf = float(jnp.max(jnp.abs(error)))
        rel_l2 = float(jnp.linalg.norm(error) / (jnp.linalg.norm(jnp.asarray(expected, dtype=jnp.float64)) + 1.0e-30))
        print(
            f"  {field_name}: l2={error_l2:.6e}, linf={error_linf:.6e}, "
            f"rel_l2={rel_l2:.6e}, expected_l2={expected_norm:.6e}"
        )


def _report_phi_inversion_consistency(
    geometry: FciGeometry3D,
    *,
    time: float,
    parameters: Fci4FieldRhsParameters,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    mg_hierarchy: object | None = None,
    gmres_debug: bool = False,
) -> tuple[float, float, float]:
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    periodic_axes = (False, True, True)
    exact_phi = _shifted_torus_phi(geometry, time)
    exact_omega = _shifted_torus_exact_state(geometry, time).omega
    phi_face_bc, _, _, _, _ = _shifted_torus_exact_x_face_bcs(geometry, time)
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=periodic_axes,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )
    phi_from_inverse, phi_diagnostics = phi_inverse_solver(
        -exact_omega,
        face_bc=phi_face_bc,
        return_diagnostics=True,
    )
    phi_error = jnp.asarray(phi_from_inverse - exact_phi, dtype=jnp.float64)
    l2_error = float(jnp.sqrt(jnp.mean(jnp.square(phi_error))))
    linf_error = float(jnp.max(jnp.abs(phi_error)))
    rel_l2_error = float(jnp.linalg.norm(phi_error) / (jnp.linalg.norm(exact_phi) + 1.0e-30))
    print(
        "shifted_torus_4field phi inversion consistency: "
        f"shape={geometry.shape}, l2={l2_error:.6e}, "
        f"linf={linf_error:.6e}, rel_l2={rel_l2_error:.6e}, "
        f"gmres_rel_res={float(phi_diagnostics['final_residual_rel_l2']):.6e}, "
        f"gmres_steps={int(phi_diagnostics['num_steps'])}"
    )
    return l2_error, linf_error, rel_l2_error


def _run_single_rhs_diagnostic_sweeps(
    resolutions: np.ndarray,
    *,
    time: float,
    rho_star_value: float,
    phi_inversion_tol: float = 1.0e-4,
) -> None:
    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=float(phi_inversion_tol),
    )
    phi_vorticity_results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    exact_phi_rhs_results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    for resolution in resolutions:
        geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
        print(f"Diagnostic operator sweep for resolution={int(resolution)}")
        phi_vorticity_results.append(
            (
                int(resolution),
                _phi_vorticity_mismatch_statistics(geometry, time),
            )
        )
        exact_phi_rhs_results.append(
            (
                int(resolution),
                _exact_phi_rhs_consistency_statistics(
                    geometry,
                    time,
                    parameters=parameters,
                ),
            )
        )
    _print_convergence_table(
        "shifted_torus_4field discrete phi-vorticity consistency convergence",
        phi_vorticity_results,
    )
    _print_convergence_table(
        "shifted_torus_4field exact-phi RHS consistency convergence",
        exact_phi_rhs_results,
    )


def _run_phi_inversion_tolerance_sweep(
    resolutions: np.ndarray,
    *,
    time: float,
    rho_star_value: float,
    tolerances: tuple[float, ...] = (1.0e-4, 1.0e-8),
) -> None:
    for tolerance in tolerances:
        print(f"shifted_torus_4field phi inversion tolerance sweep: tol={tolerance:.1e}")
        results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
        for resolution in resolutions:
            geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
            parameters = Fci4FieldRhsParameters(
                rho_star=rho_star_value,
                Te=float(Te),
                mi_over_me=float(mi_over_me),
                phi_inversion_tol=float(tolerance),
            )
            conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
            face_projectors = build_perp_laplacian_face_projectors(geometry)
            l2, linf, rel_l2 = _report_phi_inversion_consistency(
                geometry,
                time=time,
                parameters=parameters,
                conservative_stencil_builder=conservative_stencil_builder,
                boundary_builder=BoundaryConditionBuilder(_build_dirichlet_boundary_condition_builder("phi")),
                face_projectors=face_projectors,
            )
            results.append((int(resolution), {"phi_from_inverse": (l2, linf, rel_l2)}))
        _print_convergence_table(
            f"shifted_torus_4field phi inversion convergence at tol={tolerance:.1e}",
            results,
        )


def _run_timestep_convergence(
    *,
    resolution: int,
    step_counts: tuple[int, ...],
    rho_star_value: float,
) -> None:
    results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
    for steps in step_counts:
        dt = float(tf) / float(steps)
        print(f"Starting timestep convergence run for resolution={resolution}, steps={steps}, dt={dt:.6e}")
        final_state, *_ = simulate_mms_shifted_torus_4field(
            geometry,
            final_time=tf,
            timestep=dt,
            rho_star_value=rho_star_value,
            show_progress=True,
        )
        exact_state = _shifted_torus_exact_state(geometry, tf)
        stats = _state_error_statistics(final_state, exact_state)
        _print_state_error_statistics(f"timestep convergence per-field errors: steps={steps}", stats)
        results.append((int(steps), stats))
    print("shifted_torus_4field timestep convergence at fixed resolution")
    previous_steps: int | None = None
    previous_stats: dict[str, tuple[float, float, float]] | None = None
    for steps, stats in results:
        print(f"  steps={steps}")
        for field_name, field_stats in stats.items():
            order_text = ""
            if previous_steps is not None and previous_stats is not None:
                order = _observed_order(previous_stats[field_name][0], field_stats[0], previous_steps, steps)
                order_text = f", order_vs_dt={_format_order(order)}"
            print(f"    {field_name}: l2={field_stats[0]:.6e}, linf={field_stats[1]:.6e}, rel_l2={field_stats[2]:.6e}{order_text}")
        previous_steps = steps
        previous_stats = stats




@dataclass(frozen=True)
class LocalShiftedTorus4FieldRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci4FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SpmdGmresConfig

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
        stage_data: _ShiftedTorus4FieldStageData,
        phi_guess_owned: jnp.ndarray | None,
    ) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
        prepared = _prepare_local_shifted_torus_4field_stage_state(
            state_owned,
            stage_data,
            self.domain,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        face_bc = prepared.boundary_data.face_bc
        state_halo = prepared.state_halo
        omega_owned = jnp.asarray(
            state_halo.omega[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        phi_solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            regular_face_geometry=self.geometry.regular_face_geometry,
            face_bc=face_bc.phi,
            config=self.gmres_config,
        )
        phi_owned = phi_solver(
            -omega_owned,
            guess_owned=phi_guess_owned,
            phi_lift_owned=stage_data.phi_halo[self.domain.layout.owned_slices_cell],
        )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)

        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        density_stencil = build_local_stencil_from_field(state_halo.density, self.geometry, context)
        omega_stencil = build_local_stencil_from_field(state_halo.omega, self.geometry, context)
        v_ion_stencil = build_local_stencil_from_field(state_halo.v_ion_parallel, self.geometry, context)
        v_electron_stencil = build_local_stencil_from_field(state_halo.v_electron_parallel, self.geometry, context)
        density_v_electron_stencil = build_local_stencil_from_field(
            state_halo.density * state_halo.v_electron_parallel,
            self.geometry,
            context,
        )
        phi_stencil = build_local_stencil_from_field(phi_halo, self.geometry, context)

        density_owned = jnp.asarray(
            state_halo.density[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        density_safe = jnp.maximum(density_owned, 1.0e-30)
        bmag_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        te = jnp.asarray(self.parameters.Te, dtype=jnp.float64)
        mi_over_me_value = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)

        poisson_density = local_poisson_bracket_op(phi_stencil, density_stencil, self.geometry)
        poisson_omega = local_poisson_bracket_op(phi_stencil, omega_stencil, self.geometry)
        poisson_v_ion = local_poisson_bracket_op(phi_stencil, v_ion_stencil, self.geometry)
        poisson_v_electron = local_poisson_bracket_op(phi_stencil, v_electron_stencil, self.geometry)
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
        grad_parallel_density = local_grad_parallel_op_direct(density_stencil, self.geometry)
        grad_parallel_phi = local_grad_parallel_op_direct(phi_stencil, self.geometry)
        grad_parallel_v_ion = local_grad_parallel_op_direct(v_ion_stencil, self.geometry)
        grad_parallel_v_electron = local_grad_parallel_op_direct(v_electron_stencil, self.geometry)
        grad_parallel_density_v_electron = local_grad_parallel_op_direct(
            density_v_electron_stencil,
            self.geometry,
        )

        density_rhs = (
            -(poisson_density / (rho_star_value * bmag_owned))
            + (2.0 * te / bmag_owned) * curvature_density
            - (2.0 * density_owned / bmag_owned) * curvature_phi
            - grad_parallel_density_v_electron
        )
        omega_rhs = (
            -(poisson_omega / (rho_star_value * bmag_owned))
            + (bmag_owned * bmag_owned / density_safe)
            * (grad_parallel_v_ion - grad_parallel_v_electron)
            + (2.0 * bmag_owned * te / density_safe) * curvature_density
        )
        v_ion_rhs = (
            -(poisson_v_ion / (rho_star_value * bmag_owned))
            - (te / density_safe) * grad_parallel_density
        )
        v_electron_rhs = (
            -(poisson_v_electron / (rho_star_value * bmag_owned))
            + mi_over_me_value * grad_parallel_phi
            - mi_over_me_value * (te / density_safe) * grad_parallel_density
        )
        owned = self.domain.layout.owned_slices_cell
        rhs = Fci4FieldState(
            density=density_rhs + stage_data.source_halo.density[owned],
            omega=omega_rhs + stage_data.source_halo.omega[owned],
            v_ion_parallel=v_ion_rhs + stage_data.source_halo.v_ion_parallel[owned],
            v_electron_parallel=v_electron_rhs + stage_data.source_halo.v_electron_parallel[owned],
        )
        return rhs, phi_owned, jnp.zeros((4,), dtype=jnp.float64)


def simulate_mms_shifted_torus_4field(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    use_multigrid_preconditioner: bool = False,
    disable_multigrid_on_failure: bool = True,
    gmres_debug: bool = False,
    show_progress: bool = False,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve the shifted-torus MMS system and return the final state plus stacked history."""

    del use_multigrid_preconditioner, disable_multigrid_on_failure, gmres_debug
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=1.0e-4,
        phi_inversion_maxiter=100,
        phi_inversion_restart=100,
    )
    gmres_config = SpmdGmresConfig(
        # Match the EB blob style: keep the internal GMRES solve tolerance
        # separate from the higher-level residual acceptance threshold.
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        acceptance_tol=float(parameters.phi_inversion_tol),
        acceptance_atol=float(parameters.phi_inversion_tol),
        regularization_epsilon=float(parameters.phi_inversion_regularization),
    )
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_state = _shifted_torus_exact_state(geometry, 0.0)
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    wall_step_times: list[float] = []

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = _put_state_on_mesh(initial_state, mesh)
        phi_guess = jax.device_put(
            jnp.asarray(_shifted_torus_phi(geometry, 0.0), dtype=jnp.float64),
            NamedSharding(mesh, P(*MESH_AXIS_NAMES)),
        )
        state_spec = _state_partition_spec()
        field_spec = P(*MESH_AXIS_NAMES)
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_4field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        sample_stage_data = _build_local_4field_rk4_stage_data(
            _build_local_4field_invariants(
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

        def invariant_kernel() -> _ShiftedTorus4FieldInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_4field_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorus4FieldInvariantBundle,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> _ShiftedTorus4FieldRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            return expand_local_shard_pytree(
                _build_local_4field_rk4_stage_data(
                    local_invariants,
                    step_time,
                    step_timestep,
                    parameters=parameters,
                )
            )

        def kernel(
            state_owned: Fci4FieldState,
            phi_guess_owned: jnp.ndarray,
            local_invariants: _ShiftedTorus4FieldInvariantBundle,
            rk_stage_data: _ShiftedTorus4FieldRk4StageData,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> tuple[Fci4FieldState, jnp.ndarray]:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            rhs = LocalShiftedTorus4FieldRhs(
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
            )
            k1, carry_1, _ = rhs.evaluate_stage(
                state_owned,
                rk_stage_data.stage_1,
                phi_guess_owned,
            )
            stage_1 = state_owned.axpy(k1, scale=0.5 * step_timestep)
            k2, carry_2, _ = rhs.evaluate_stage(
                stage_1,
                rk_stage_data.stage_2,
                carry_1,
            )
            stage_2 = state_owned.axpy(k2, scale=0.5 * step_timestep)
            k3, carry_3, _ = rhs.evaluate_stage(
                stage_2,
                rk_stage_data.stage_3,
                carry_2,
            )
            stage_3 = state_owned.axpy(k3, scale=step_timestep)
            k4, carry_4, _ = rhs.evaluate_stage(
                stage_3,
                rk_stage_data.stage_4,
                carry_3,
            )
            next_state = state_owned.axpy(
                k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0),
                scale=step_timestep / 6.0,
            )
            next_phi_guess = carry_4
            return next_state, next_phi_guess

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P(), P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        compiled_source_kernel = jax.jit(mapped_source_kernel)
        mapped_step_kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(state_spec, field_spec, invariant_spec, stage_data_spec, P(), P()),
            out_specs=(state_spec, field_spec),
            check_rep=False,
        )
        step_kernel = jax.jit(mapped_step_kernel)

        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                f"shifted_torus_4field RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

        for step_index in range(steps):
            step_start = time_module.perf_counter()
            rk_stage_data = compiled_source_kernel(
                invariants,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            state, phi_guess = step_kernel(
                state,
                phi_guess,
                invariants,
                rk_stage_data,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            wall_step_times.append(time_module.perf_counter() - step_start)
            time_value += dt
            times.append(time_value)
            gathered_state = _gather_state_from_mesh(state)
            density_history.append(jnp.asarray(gathered_state.density, dtype=jnp.float32))
            omega_history.append(jnp.asarray(gathered_state.omega, dtype=jnp.float32))
            v_ion_history.append(jnp.asarray(gathered_state.v_ion_parallel, dtype=jnp.float32))
            v_electron_history.append(jnp.asarray(gathered_state.v_electron_parallel, dtype=jnp.float32))
            if show_progress:
                print(
                    "\r"
                    f"shifted_torus_4field RK4 progress: "
                    f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )

        if show_progress:
            print()

        final_state = _gather_state_from_mesh(state)

    if wall_step_times:
        print(
            "shifted_torus_4field mean timings per RK step: "
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


def _shifted_torus_z_cut_indices(geometry: FciGeometry3D, count: int) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, count)
    return tuple(int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts)


def _shifted_torus_field_slices(field: jnp.ndarray, z_indices: tuple[int, ...]) -> jnp.ndarray:
    return jnp.stack([field[:, :, z_index] for z_index in z_indices], axis=0)


def _symmetric_color_limit(*arrays: np.ndarray) -> float:
    vmax = float(np.max(np.abs(np.stack(arrays, axis=0))))
    return vmax if vmax > 0.0 else 1.0


def _configure_shifted_torus_slice_axis(ax, x_values: np.ndarray) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, float(x_values[-1]))
    ax.set_yticklabels([])


def _plot_final_slices(
    state: Fci4FieldState,
    exact_state: Fci4FieldState,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 2)
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    field_specs = (
        ("density", state.density, exact_state.density, "viridis"),
        ("omega", state.omega, exact_state.omega, "coolwarm"),
        ("v_ion_parallel", state.v_ion_parallel, exact_state.v_ion_parallel, "coolwarm"),
        ("v_electron_parallel", state.v_electron_parallel, exact_state.v_electron_parallel, "coolwarm"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(14.0, 12.5), subplot_kw={"projection": "polar"}, constrained_layout=True)

    for row, (field_name, field, exact_field, cmap) in enumerate(field_specs):
        field_slices = np.asarray(
            _shifted_torus_field_slices(jnp.asarray(field, dtype=jnp.float64), z_indices),
            dtype=np.float64,
        )
        exact_slices = np.asarray(
            _shifted_torus_field_slices(jnp.asarray(exact_field, dtype=jnp.float64), z_indices),
            dtype=np.float64,
        )
        vmax = _symmetric_color_limit(field_slices, exact_slices)
        row_image = None
        for cut_index, z_index in enumerate(z_indices):
            row_image = axes[row, cut_index].pcolormesh(
                theta_grid,
                radius_grid,
                field_slices[cut_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            _configure_shifted_torus_slice_axis(axes[row, cut_index], x_values)
            axes[row, cut_index].set_title(f"{field_name} sim, zeta={z_values[z_index]:.3f}")

            row_image = axes[row, 2 + cut_index].pcolormesh(
                theta_grid,
                radius_grid,
                exact_slices[cut_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            _configure_shifted_torus_slice_axis(axes[row, 2 + cut_index], x_values)
            axes[row, 2 + cut_index].set_title(f"{field_name} exact, zeta={z_values[z_index]:.3f}")
        if row_image is not None:
            fig.colorbar(row_image, ax=axes[row, :].ravel().tolist(), shrink=0.82, pad=0.02)

    fig.suptitle(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_shifted_torus_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    omega_history: jnp.ndarray,
    v_ion_history: jnp.ndarray,
    v_electron_history: jnp.ndarray,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
    frame_stride: int = 5,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 4)
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    field_specs = (
        ("density", np.asarray(density_history, dtype=np.float64), "viridis"),
        ("omega", np.asarray(omega_history, dtype=np.float64), "coolwarm"),
        ("v_ion_parallel", np.asarray(v_ion_history, dtype=np.float64), "coolwarm"),
        ("v_electron_parallel", np.asarray(v_electron_history, dtype=np.float64), "coolwarm"),
    )
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    fig, axes = plt.subplots(4, 4, figsize=(14.0, 12.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    images = []
    for row, (field_name, field_data, cmap) in enumerate(field_specs):
        vmax = float(np.max(np.abs(field_data)))
        vmax = vmax if vmax > 0.0 else 1.0
        for col, z_index in enumerate(z_indices):
            ax = axes[row, col]
            _configure_shifted_torus_slice_axis(ax, x_values)
            ax.set_title(f"{field_name}, zeta={z_values[z_index]:.3f}")
            image = ax.pcolormesh(
                theta_grid,
                radius_grid,
                field_data[0, :, :, z_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            images.append(image)

    suptitle = fig.suptitle(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}")

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for row, (field_name, field_data, _) in enumerate(field_specs):
            for col, z_index in enumerate(z_indices):
                images[row * len(z_indices) + col].set_array(field_data[actual_index, :, :, z_index].ravel())
                axes[row, col].set_title(f"{field_name}, zeta={z_values[z_index]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def run_shifted_torus_4field_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = tf,
    base_steps: int = num_steps,
    rho_star_value: float = rho_star,
    plot: bool = False,
    plot_path: str | None = None,
    plot_slices: bool = False,
    movie: bool = False,
    movie_stride: int = 5,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    final_resolution_state: Fci4FieldState | None = None
    final_resolution_geometry: FciGeometry3D | None = None
    final_resolution: int | None = None
    final_resolution_times: jnp.ndarray | None = None
    final_resolution_density_history: jnp.ndarray | None = None
    final_resolution_omega_history: jnp.ndarray | None = None
    final_resolution_v_ion_history: jnp.ndarray | None = None
    final_resolution_v_electron_history: jnp.ndarray | None = None

    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = build_shifted_torus_4field_geometry(shape)
        steps = _resolution_step_count(int(resolution), base_steps=base_steps)
        dt = float(final_time) / float(steps)
        print(
            f"Starting shifted_torus_4field MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}, dt={dt:.6e}"
        )
        start = time_module.perf_counter()
        try:
            final_state, times, density_history, omega_history, v_ion_history, v_electron_history = simulate_mms_shifted_torus_4field(
                geometry,
                shard_counts=shard_counts,
                halo_width=halo_width,
                final_time=final_time,
                timestep=dt,
                rho_star_value=rho_star_value,
                show_progress=show_progress,
            )
            elapsed = time_module.perf_counter() - start
            mean_error, median_error, max_error = _combined_error_statistics(
                final_state,
                geometry,
                final_time,
            )
            per_field_stats = _state_error_statistics(
                final_state,
                _shifted_torus_exact_state(geometry, final_time),
            )
        except FloatingPointError as exc:
            elapsed = time_module.perf_counter() - start
            print(
                f"WARNING: resolution={int(resolution)} shard_counts={shard_counts} "
                f"failed after {elapsed:.6e} s: {exc}"
            )
            continue

        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        print(
            f"N={int(resolution)}: shard_counts={shard_counts}, steps={steps}, "
            f"total_runtime={elapsed:.6e} s, avg_step_runtime={elapsed / float(steps):.6e} s, "
            f"L2={mean_error:.6e}, median={median_error:.6e}, Linf={max_error:.6e}"
        )
        _print_state_error_statistics(f"N={int(resolution)} per-field final errors", per_field_stats)
        final_resolution_state = final_state
        final_resolution_geometry = geometry
        final_resolution = int(resolution)
        final_resolution_times = times
        final_resolution_density_history = density_history
        final_resolution_omega_history = omega_history
        final_resolution_v_ion_history = v_ion_history
        final_resolution_v_electron_history = v_electron_history

    l2_order: float | None = None
    max_order: float | None = None
    if len(successful_resolutions) >= 2:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        l2_order = float(-l2_slope)
        max_order = float(-max_slope)
        print(f"shifted_torus_4field L2 convergence order: {l2_order:.6f}")
        print(f"shifted_torus_4field Linf convergence order: {max_order:.6f}")

        if plot:
            import matplotlib.pyplot as plt

            output_path = Path(plot_path or "shifted_torus_4field_convergence.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"L2, order {l2_order:.2f}")
            ax.loglog(plotted_resolutions, max_errors, "^-", label=f"Linf, order {max_order:.2f}")
            ax.loglog(
                plotted_resolutions,
                np.exp(l2_intercept) * plotted_resolutions.astype(np.float64) ** l2_slope,
                "--",
                color=ax.lines[0].get_color(),
            )
            ax.loglog(
                plotted_resolutions,
                np.exp(max_intercept) * plotted_resolutions.astype(np.float64) ** max_slope,
                "--",
                color=ax.lines[1].get_color(),
            )
            ax.set_xlabel("resolution")
            ax.set_ylabel("absolute error")
            ax.set_title(f"Shifted-torus 4-field MMS convergence ({shard_counts})")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
    elif plot:
        print("WARNING: fewer than two successful resolutions, skipping convergence plot.")

    output_base = Path(plot_path).parent if plot_path else Path(".")
    if plot_slices and final_resolution_state is not None and final_resolution_geometry is not None and final_resolution is not None:
        final_exact_state = _shifted_torus_exact_state(final_resolution_geometry, final_time)
        _plot_final_slices(
            final_resolution_state,
            final_exact_state,
            final_resolution_geometry,
            final_resolution,
            str(output_base / "shifted_torus_4field_slices.png"),
        )

    if (
        movie
        and final_resolution_times is not None
        and final_resolution_density_history is not None
        and final_resolution_omega_history is not None
        and final_resolution_v_ion_history is not None
        and final_resolution_v_electron_history is not None
        and final_resolution_geometry is not None
        and final_resolution is not None
    ):
        _save_shifted_torus_movie(
            final_resolution_times,
            final_resolution_density_history,
            final_resolution_omega_history,
            final_resolution_v_ion_history,
            final_resolution_v_electron_history,
            final_resolution_geometry,
            final_resolution,
            str(output_base / "shifted_torus_4field_slices.gif"),
            frame_stride=movie_stride,
        )

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
        "l2_order": l2_order,
        "linf_order": max_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted-torus 4-field MMS convergence harness")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[30, 60, 120])
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=tf)
    parser.add_argument("--base-steps", type=int, default=num_steps)
    parser.add_argument("--rho-star", type=float, default=rho_star)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--plot-slices", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--movie-stride", type=int, default=5)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--run-rhs-diagnostics", action="store_true")
    parser.add_argument("--run-phi-tolerance-sweep", action="store_true")
    parser.add_argument("--run-term-breakdown", action="store_true")
    parser.add_argument("--run-timestep-convergence", action="store_true")
    args = parser.parse_args()

    resolutions = [int(value) for value in args.resolutions]
    rho_star_value = float(args.rho_star)
    if args.run_rhs_diagnostics:
        _run_single_rhs_diagnostic_sweeps(
            np.asarray(resolutions, dtype=np.int64),
            time=0.0,
            rho_star_value=rho_star_value,
        )
    if args.run_phi_tolerance_sweep:
        _run_phi_inversion_tolerance_sweep(
            np.asarray(resolutions, dtype=np.int64),
            time=0.0,
            rho_star_value=rho_star_value,
            tolerances=(1.0e-4, 1.0e-8),
        )
    if args.run_term_breakdown:
        breakdown_resolution = int(resolutions[0])
        _report_exact_phi_term_breakdown(
            build_shifted_torus_4field_geometry(
                (breakdown_resolution, breakdown_resolution, breakdown_resolution)
            ),
            0.0,
            parameters=Fci4FieldRhsParameters(
                rho_star=rho_star_value,
                Te=float(Te),
                mi_over_me=float(mi_over_me),
                phi_inversion_tol=1.0e-4,
            ),
        )
    if args.run_timestep_convergence:
        _run_timestep_convergence(
            resolution=int(resolutions[0]),
            step_counts=(25, 50, 100, 200),
            rho_star_value=rho_star_value,
        )
    if args.skip_simulation:
        return

    run_shifted_torus_4field_convergence(
        resolutions=resolutions,
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        rho_star_value=rho_star_value,
        plot=bool(args.plot),
        plot_path=args.plot_path,
        plot_slices=bool(args.plot_slices),
        movie=bool(args.movie),
        movie_stride=int(args.movie_stride),
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
