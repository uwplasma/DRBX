from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from ..config.boutinp import BoutConfig
from ..solver import (
    ImplicitStepInfo,
    active_region_from_slices,
    backward_euler_residual,
    bdf2_residual,
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    pack_active_fields,
    solve_matrix_free_newton_system,
    solve_sparse_newton_system,
    unpack_active_fields,
)
from .expression import ArrayExpressionEvaluator
from .neutral_mixed_boundaries import (
    apply_density_boundaries as _apply_density_boundaries,
    apply_dirichlet_x_boundaries as _apply_dirichlet_x_boundaries,
    apply_diffusion_boundaries as _apply_diffusion_boundaries,
    apply_momentum_boundaries as _apply_momentum_boundaries,
    apply_neumann_x_boundaries as _apply_neumann_x_boundaries,
    apply_pressure_boundaries as _apply_pressure_boundaries,
    apply_temperature_boundaries as _apply_temperature_boundaries,
    apply_velocity_boundaries as _apply_velocity_boundaries,
    soft_floor as _soft_floor,
)
from . import neutral_mixed_operators as _neutral_mixed_operators
from .neutral_mixed_operators import (
    div_a_grad_perp_flows as _div_a_grad_perp_flows,
    div_par_fvv_open as _div_par_fvv_open,
    div_par_k_grad_par_open as _div_par_k_grad_par_open,
    div_par_mod_open as _div_par_mod_open,
    grad_par_open as _grad_par_open,
    gradient_magnitude as _gradient_magnitude,
)
from .neutral_mixed_state import (
    NeutralMixedHistoryResult,
    NeutralMixedImplicitStepInfo,
    NeutralMixedRhsResult,
    NeutralMixedState,
    PreparedNeutralMixedState as _PreparedNeutralMixedState,
)
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics


def initialize_neutral_mixed_state(
    config: BoutConfig,
    *,
    section: str,
    mesh: StructuredMesh,
) -> NeutralMixedState:
    density = _evaluate_field_option(config, f"N{section}", mesh=mesh)
    pressure = _evaluate_field_option(config, f"P{section}", mesh=mesh)
    if config.has_section(f"NV{section}"):
        momentum = _evaluate_field_option(config, f"NV{section}", mesh=mesh)
    else:
        momentum = np.zeros_like(density, dtype=np.float64)
    return NeutralMixedState(
        density=_apply_neutral_density_boundaries(density, mesh),
        pressure=_apply_neutral_pressure_boundaries(pressure, mesh),
        momentum=_apply_neutral_momentum_boundaries(momentum, mesh),
    )


def advance_neutral_mixed_history(
    config: BoutConfig,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
    steps: int,
    substeps: int,
) -> NeutralMixedHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if substeps <= 0:
        raise ValueError("substeps must be positive")

    state = initialize_neutral_mixed_state(config, section=section, mesh=mesh)
    density_history = [np.asarray(state.density, dtype=np.float64)]
    pressure_history = [np.asarray(state.pressure, dtype=np.float64)]
    momentum_history = [np.asarray(state.momentum, dtype=np.float64)]

    sub_timestep = float(timestep) / float(substeps)
    for _ in range(steps):
        for _ in range(substeps):
            state = _rk4_step(
                config,
                state,
                section=section,
                mesh=mesh,
                metrics=metrics,
                meters_scale=meters_scale,
                tnorm=tnorm,
                timestep=sub_timestep,
            )
        density_history.append(np.asarray(state.density, dtype=np.float64))
        pressure_history.append(np.asarray(state.pressure, dtype=np.float64))
        momentum_history.append(np.asarray(state.momentum, dtype=np.float64))

    return NeutralMixedHistoryResult(
        density_history=np.stack(density_history, axis=0),
        pressure_history=np.stack(pressure_history, axis=0),
        momentum_history=np.stack(momentum_history, axis=0),
    )


def advance_neutral_mixed_implicit_history(
    config: BoutConfig,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
    steps: int,
    internal_substeps: int = 1,
    solver_mode: str = "sparse",
    residual_tolerance: float = 1.0e-8,
    step_tolerance: float = 1.0e-10,
    max_nonlinear_iterations: int = 8,
    linear_restart: int = 20,
    linear_maxiter: int = 200,
    linear_rtol: float = 1.0e-8,
    store_internal_substeps: bool = False,
    accepted_step_time_points: Sequence[float] | np.ndarray | None = None,
) -> NeutralMixedHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if internal_substeps <= 0:
        raise ValueError("internal_substeps must be positive")
    explicit_schedule: tuple[tuple[float, float], ...] | None = None
    if accepted_step_time_points is not None:
        supplied_times = np.asarray(accepted_step_time_points, dtype=np.float64)
        if supplied_times.ndim != 1:
            raise ValueError("accepted_step_time_points must be one-dimensional")
        if not np.all(np.isfinite(supplied_times)):
            raise ValueError("accepted_step_time_points must be finite")
        if supplied_times.size > 0 and supplied_times[0] == 0.0:
            supplied_times = supplied_times[1:]
        if supplied_times.size == 0:
            raise ValueError("accepted_step_time_points must include at least one positive time")
        if supplied_times[0] <= 0.0:
            raise ValueError("accepted_step_time_points must be positive after the initial state")
        dt_values = np.diff(np.concatenate((np.asarray([0.0]), supplied_times)))
        if np.any(dt_values <= 0.0):
            raise ValueError("accepted_step_time_points must be strictly increasing")
        explicit_schedule = tuple(
            (float(dt_value), float(time_value))
            for dt_value, time_value in zip(dt_values, supplied_times, strict=True)
        )
        store_internal_substeps = True

    state = initialize_neutral_mixed_state(config, section=section, mesh=mesh)
    density_history = [np.asarray(state.density, dtype=np.float64)]
    pressure_history = [np.asarray(state.pressure, dtype=np.float64)]
    momentum_history = [np.asarray(state.momentum, dtype=np.float64)]
    accepted_time_points = [0.0] if store_internal_substeps else None
    accepted_dt = [0.0] if store_internal_substeps else None
    accepted_order = [0] if store_internal_substeps else None
    accepted_density_history = (
        [np.asarray(state.density, dtype=np.float64)]
        if store_internal_substeps
        else None
    )
    accepted_pressure_history = (
        [np.asarray(state.pressure, dtype=np.float64)]
        if store_internal_substeps
        else None
    )
    accepted_momentum_history = (
        [np.asarray(state.momentum, dtype=np.float64)]
        if store_internal_substeps
        else None
    )
    accepted_residual_inf_norm = [0.0] if store_internal_substeps else None
    accepted_nonlinear_iterations = [0] if store_internal_substeps else None

    if steps == 0 and explicit_schedule is None:
        return NeutralMixedHistoryResult(
            density_history=np.stack(density_history, axis=0),
            pressure_history=np.stack(pressure_history, axis=0),
            momentum_history=np.stack(momentum_history, axis=0),
            accepted_step_time_points=np.asarray(accepted_time_points, dtype=np.float64)
            if accepted_time_points is not None
            else None,
            accepted_step_dt=np.asarray(accepted_dt, dtype=np.float64)
            if accepted_dt is not None
            else None,
            accepted_step_order=np.asarray(accepted_order, dtype=np.int32)
            if accepted_order is not None
            else None,
            accepted_step_density_history=np.stack(accepted_density_history, axis=0)
            if accepted_density_history is not None
            else None,
            accepted_step_pressure_history=np.stack(accepted_pressure_history, axis=0)
            if accepted_pressure_history is not None
            else None,
            accepted_step_momentum_history=np.stack(accepted_momentum_history, axis=0)
            if accepted_momentum_history is not None
            else None,
            accepted_step_residual_inf_norm=np.asarray(
                accepted_residual_inf_norm, dtype=np.float64
            )
            if accepted_residual_inf_norm is not None
            else None,
            accepted_step_nonlinear_iterations=np.asarray(
                accepted_nonlinear_iterations, dtype=np.int32
            )
            if accepted_nonlinear_iterations is not None
            else None,
        )

    previous_state: NeutralMixedState | None = None
    previous_timestep: float | None = None
    current_state = state
    sub_timestep = float(timestep) / float(internal_substeps)
    current_time = 0.0
    if explicit_schedule is None:
        output_step_schedules = (
            tuple((sub_timestep, None) for _ in range(internal_substeps))
            for _ in range(steps)
        )
    else:
        output_step_schedules = (explicit_schedule,)
    for step_schedule in output_step_schedules:
        for step_dt, target_time in step_schedule:
            if previous_state is None:
                next_state, step_info = advance_neutral_mixed_backward_euler_step(
                    config,
                    current_state,
                    section=section,
                    mesh=mesh,
                    metrics=metrics,
                    meters_scale=meters_scale,
                    tnorm=tnorm,
                    timestep=step_dt,
                    solver_mode=solver_mode,
                    residual_tolerance=residual_tolerance,
                    step_tolerance=step_tolerance,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                    linear_restart=linear_restart,
                    linear_maxiter=linear_maxiter,
                    linear_rtol=linear_rtol,
                )
                solver_order = 1
            else:
                next_state, step_info = advance_neutral_mixed_bdf2_step(
                    config,
                    current_state,
                    previous_state,
                    section=section,
                    mesh=mesh,
                    metrics=metrics,
                    meters_scale=meters_scale,
                    tnorm=tnorm,
                    timestep=step_dt,
                    previous_timestep=previous_timestep,
                    solver_mode=solver_mode,
                    residual_tolerance=residual_tolerance,
                    step_tolerance=step_tolerance,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                    linear_restart=linear_restart,
                    linear_maxiter=linear_maxiter,
                    linear_rtol=linear_rtol,
                )
                solver_order = 2
            previous_state, current_state = current_state, next_state
            current_time = (
                float(target_time)
                if target_time is not None
                else current_time + step_dt
            )
            previous_timestep = float(step_dt)
            if store_internal_substeps:
                assert accepted_time_points is not None
                assert accepted_dt is not None
                assert accepted_order is not None
                assert accepted_density_history is not None
                assert accepted_pressure_history is not None
                assert accepted_momentum_history is not None
                assert accepted_residual_inf_norm is not None
                assert accepted_nonlinear_iterations is not None
                accepted_time_points.append(float(current_time))
                accepted_dt.append(float(step_dt))
                accepted_order.append(int(solver_order))
                accepted_density_history.append(
                    np.asarray(current_state.density, dtype=np.float64)
                )
                accepted_pressure_history.append(
                    np.asarray(current_state.pressure, dtype=np.float64)
                )
                accepted_momentum_history.append(
                    np.asarray(current_state.momentum, dtype=np.float64)
                )
                accepted_residual_inf_norm.append(float(step_info.residual_inf_norm))
                accepted_nonlinear_iterations.append(
                    int(step_info.nonlinear_iterations)
                )
        density_history.append(np.asarray(current_state.density, dtype=np.float64))
        pressure_history.append(np.asarray(current_state.pressure, dtype=np.float64))
        momentum_history.append(np.asarray(current_state.momentum, dtype=np.float64))

    return NeutralMixedHistoryResult(
        density_history=np.stack(density_history, axis=0),
        pressure_history=np.stack(pressure_history, axis=0),
        momentum_history=np.stack(momentum_history, axis=0),
        accepted_step_time_points=np.asarray(accepted_time_points, dtype=np.float64)
        if accepted_time_points is not None
        else None,
        accepted_step_dt=np.asarray(accepted_dt, dtype=np.float64)
        if accepted_dt is not None
        else None,
        accepted_step_order=np.asarray(accepted_order, dtype=np.int32)
        if accepted_order is not None
        else None,
        accepted_step_density_history=np.stack(accepted_density_history, axis=0)
        if accepted_density_history is not None
        else None,
        accepted_step_pressure_history=np.stack(accepted_pressure_history, axis=0)
        if accepted_pressure_history is not None
        else None,
        accepted_step_momentum_history=np.stack(accepted_momentum_history, axis=0)
        if accepted_momentum_history is not None
        else None,
        accepted_step_residual_inf_norm=np.asarray(
            accepted_residual_inf_norm, dtype=np.float64
        )
        if accepted_residual_inf_norm is not None
        else None,
        accepted_step_nonlinear_iterations=np.asarray(
            accepted_nonlinear_iterations, dtype=np.int32
        )
        if accepted_nonlinear_iterations is not None
        else None,
    )


def pack_neutral_mixed_active_state(
    state: NeutralMixedState,
    *,
    mesh: StructuredMesh,
) -> np.ndarray:
    return pack_active_fields(
        (state.density, state.pressure, state.momentum),
        active_slices=_active_domain_slices(mesh),
    )


def unpack_neutral_mixed_active_state(
    packed: np.ndarray,
    *,
    template: NeutralMixedState,
    mesh: StructuredMesh,
) -> NeutralMixedState:
    density, pressure, momentum = unpack_active_fields(
        packed,
        templates=(template.density, template.pressure, template.momentum),
        active_slices=_active_domain_slices(mesh),
    )
    return _sanitize_neutral_state(
        NeutralMixedState(
            density=density,
            pressure=pressure,
            momentum=momentum,
        ),
        mesh,
    )


def compute_neutral_mixed_active_rhs(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
) -> np.ndarray:
    active = _active_domain_slices(mesh)
    rhs = compute_neutral_mixed_rhs(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    return np.concatenate(
        [
            np.asarray(rhs.density[active], dtype=np.float64).ravel(),
            np.asarray(rhs.pressure[active], dtype=np.float64).ravel(),
            np.asarray(rhs.momentum[active], dtype=np.float64).ravel(),
        ]
    )


def build_neutral_mixed_active_jacobian_sparsity(mesh: StructuredMesh):
    return _neutral_mixed_jacobian_sparsity(_active_domain_shape(mesh))


def build_neutral_mixed_active_jacobian_color_groups(
    mesh: StructuredMesh,
) -> tuple[tuple[int, ...], ...]:
    return _neutral_mixed_jacobian_color_groups(_active_domain_shape(mesh))


def build_neutral_mixed_sparse_residual_jacobian(
    residual,
    packed_state: np.ndarray,
    *,
    mesh: StructuredMesh,
    base_residual: np.ndarray | None = None,
    sparsity=None,
    color_groups: tuple[tuple[int, ...], ...] | None = None,
):
    if sparsity is None:
        sparsity = build_neutral_mixed_active_jacobian_sparsity(mesh)
    if color_groups is None:
        color_groups = build_neutral_mixed_active_jacobian_color_groups(mesh)
    return build_sparse_difference_quotient_jacobian(
        residual,
        packed_state,
        base_residual=base_residual,
        sparsity=sparsity,
        color_groups=color_groups,
    )


def compute_neutral_mixed_backward_euler_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    *,
    config: BoutConfig,
    template_state: NeutralMixedState,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
) -> np.ndarray:
    state = unpack_neutral_mixed_active_state(
        packed_state,
        template=template_state,
        mesh=mesh,
    )
    rhs = compute_neutral_mixed_active_rhs(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    return backward_euler_residual(
        packed_state,
        previous_packed_state,
        rhs,
        timestep=timestep,
    )


def compute_neutral_mixed_bdf2_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    previous_previous_packed_state: np.ndarray,
    *,
    config: BoutConfig,
    template_state: NeutralMixedState,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
    previous_timestep: float | None = None,
) -> np.ndarray:
    state = unpack_neutral_mixed_active_state(
        packed_state,
        template=template_state,
        mesh=mesh,
    )
    rhs = compute_neutral_mixed_active_rhs(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    return bdf2_residual(
        packed_state,
        previous_packed_state,
        previous_previous_packed_state,
        rhs,
        timestep=timestep,
        previous_timestep=previous_timestep,
    )


def advance_neutral_mixed_backward_euler_step(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
    solver_mode: str = "matrix_free",
    residual_tolerance: float = 1.0e-9,
    step_tolerance: float = 1.0e-11,
    max_nonlinear_iterations: int = 12,
    linear_restart: int = 20,
    linear_maxiter: int = 300,
    linear_rtol: float = 1.0e-8,
) -> tuple[NeutralMixedState, NeutralMixedImplicitStepInfo]:
    previous_packed_state = pack_neutral_mixed_active_state(state, mesh=mesh)

    def residual(packed_state: np.ndarray) -> np.ndarray:
        return compute_neutral_mixed_backward_euler_residual(
            packed_state,
            previous_packed_state,
            config=config,
            template_state=state,
            section=section,
            mesh=mesh,
            metrics=metrics,
            meters_scale=meters_scale,
            tnorm=tnorm,
            timestep=timestep,
        )

    if solver_mode == "sparse":
        solved, info = solve_sparse_newton_system(
            residual,
            previous_packed_state,
            active_shape=_active_domain_shape(mesh),
            sparsity=build_neutral_mixed_active_jacobian_sparsity(mesh),
            color_groups=build_neutral_mixed_active_jacobian_color_groups(mesh),
            residual_tolerance=residual_tolerance,
            step_tolerance=step_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_rtol=linear_rtol,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            previous_packed_state,
            active_shape=_active_domain_shape(mesh),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    else:
        raise ValueError(f"Unsupported neutral implicit solver_mode={solver_mode!r}.")
    next_state = unpack_neutral_mixed_active_state(
        solved,
        template=state,
        mesh=mesh,
    )
    return next_state, _as_neutral_step_info(info)


def advance_neutral_mixed_bdf2_step(
    config: BoutConfig,
    state: NeutralMixedState,
    previous_state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
    previous_timestep: float | None = None,
    solver_mode: str = "matrix_free",
    residual_tolerance: float = 1.0e-9,
    step_tolerance: float = 1.0e-11,
    max_nonlinear_iterations: int = 12,
    linear_restart: int = 20,
    linear_maxiter: int = 300,
    linear_rtol: float = 1.0e-8,
) -> tuple[NeutralMixedState, NeutralMixedImplicitStepInfo]:
    previous_packed_state = pack_neutral_mixed_active_state(state, mesh=mesh)
    previous_previous_packed_state = pack_neutral_mixed_active_state(
        previous_state, mesh=mesh
    )

    def residual(packed_state: np.ndarray) -> np.ndarray:
        return compute_neutral_mixed_bdf2_residual(
            packed_state,
            previous_packed_state,
            previous_previous_packed_state,
            config=config,
            template_state=state,
            section=section,
            mesh=mesh,
            metrics=metrics,
            meters_scale=meters_scale,
            tnorm=tnorm,
            timestep=timestep,
            previous_timestep=previous_timestep,
        )

    if solver_mode == "sparse":
        solved, info = solve_sparse_newton_system(
            residual,
            previous_packed_state,
            active_shape=_active_domain_shape(mesh),
            sparsity=build_neutral_mixed_active_jacobian_sparsity(mesh),
            color_groups=build_neutral_mixed_active_jacobian_color_groups(mesh),
            residual_tolerance=residual_tolerance,
            step_tolerance=step_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_rtol=linear_rtol,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            previous_packed_state,
            active_shape=_active_domain_shape(mesh),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    else:
        raise ValueError(f"Unsupported neutral implicit solver_mode={solver_mode!r}.")
    next_state = unpack_neutral_mixed_active_state(
        solved,
        template=state,
        mesh=mesh,
    )
    return next_state, _as_neutral_step_info(info)


def compute_neutral_mixed_rhs(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
) -> NeutralMixedRhsResult:
    prepared = _prepare_neutral_mixed_state(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    density_parallel_advection = -_div_par_mod_open(
        prepared.density,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    density_parallel_flow = _neutral_mixed_operators.last_parallel_flow().copy()
    density_perpendicular_diffusion = _div_a_grad_perp_flows(
        prepared.diffusion_density,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )
    density_terms = {
        "parallel_advection": density_parallel_advection,
        "perpendicular_diffusion": density_perpendicular_diffusion,
    }
    density_rhs = sum(
        density_terms.values(), np.zeros_like(prepared.density, dtype=np.float64)
    )

    pressure_parallel_advection = -(5.0 / 3.0) * _div_par_mod_open(
        prepared.pressure,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    pressure_parallel_flow = (
        5.0 / 2.0
    ) * _neutral_mixed_operators.last_parallel_flow().copy()
    pressure_parallel_work = (
        (2.0 / 3.0)
        * prepared.velocity
        * _grad_par_open(
            prepared.pressure,
            mesh=mesh,
            metrics=metrics,
        )
    )
    pressure_perpendicular_diffusion = (5.0 / 3.0) * _div_a_grad_perp_flows(
        prepared.diffusion_pressure,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )
    pressure_terms = {
        "parallel_advection": pressure_parallel_advection,
        "parallel_pressure_work": pressure_parallel_work,
        "perpendicular_diffusion": pressure_perpendicular_diffusion,
    }

    if (
        bool(config.parsed(section, "neutral_conduction"))
        if config.has_option(section, "neutral_conduction")
        else True
    ):
        pressure_parallel_conduction = (2.0 / 3.0) * _div_par_k_grad_par_open(
            prepared.conductivity,
            prepared.temperature,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        pressure_perpendicular_conduction = (2.0 / 3.0) * _div_a_grad_perp_flows(
            prepared.conductivity,
            prepared.temperature,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_terms["parallel_conduction"] = pressure_parallel_conduction
        pressure_terms["perpendicular_conduction"] = pressure_perpendicular_conduction

    momentum_parallel_inertia = -_section_scalar(
        config, section, "AA", default=1.0
    ) * _div_par_fvv_open(
        prepared.density_limited,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    momentum_pressure_gradient = -_grad_par_open(
        prepared.pressure, mesh=mesh, metrics=metrics
    )
    momentum_perpendicular_diffusion = _div_a_grad_perp_flows(
        prepared.diffusion_momentum,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )
    momentum_terms = {
        "parallel_inertia": momentum_parallel_inertia,
        "pressure_gradient": momentum_pressure_gradient,
        "perpendicular_diffusion": momentum_perpendicular_diffusion,
    }

    if (
        bool(config.parsed(section, "neutral_viscosity"))
        if config.has_option(section, "neutral_viscosity")
        else True
    ):
        viscosity_parallel = _div_par_k_grad_par_open(
            prepared.viscosity,
            prepared.velocity,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        viscosity_perpendicular = _div_a_grad_perp_flows(
            prepared.viscosity,
            prepared.velocity,
            mesh=mesh,
            metrics=metrics,
        )
        viscosity_source = viscosity_parallel + viscosity_perpendicular
        momentum_terms["parallel_viscosity"] = viscosity_parallel
        momentum_terms["perpendicular_viscosity"] = viscosity_perpendicular
        pressure_terms["viscous_work"] = (
            -(2.0 / 3.0) * prepared.velocity * viscosity_source
        )

    pressure_rhs = sum(
        pressure_terms.values(), np.zeros_like(prepared.pressure, dtype=np.float64)
    )
    momentum_rhs = sum(
        momentum_terms.values(), np.zeros_like(prepared.momentum, dtype=np.float64)
    )

    return NeutralMixedRhsResult(
        density=np.asarray(density_rhs, dtype=np.float64),
        pressure=np.asarray(pressure_rhs, dtype=np.float64),
        momentum=np.asarray(momentum_rhs, dtype=np.float64),
        diffusion=np.asarray(prepared.diffusion, dtype=np.float64),
        density_parallel_flow=np.asarray(density_parallel_flow, dtype=np.float64),
        pressure_parallel_flow=np.asarray(pressure_parallel_flow, dtype=np.float64),
        density_terms={
            name: np.asarray(value, dtype=np.float64)
            for name, value in density_terms.items()
        },
        pressure_terms={
            name: np.asarray(value, dtype=np.float64)
            for name, value in pressure_terms.items()
        },
        momentum_terms={
            name: np.asarray(value, dtype=np.float64)
            for name, value in momentum_terms.items()
        },
    )


def compute_neutral_mixed_diffusion(
    temperature_limited: np.ndarray,
    log_pressure: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    meters_scale: float,
    flux_limit: float,
    diffusion_limit: float = -1.0,
) -> np.ndarray:
    thermal_speed = np.sqrt(
        np.asarray(temperature_limited, dtype=np.float64) / atomic_mass
    )
    neutral_lmax = 0.1 / meters_scale
    raw_diffusion = thermal_speed * neutral_lmax

    if flux_limit > 0.0:
        grad_magnitude = _gradient_magnitude(log_pressure, mesh=mesh, metrics=metrics)
        diffusion_max = (
            flux_limit * thermal_speed / (grad_magnitude + (1.0 / neutral_lmax))
        )
        diffusion = raw_diffusion * diffusion_max / (raw_diffusion + diffusion_max)
    else:
        diffusion = raw_diffusion

    if diffusion_limit > 0.0:
        diffusion = diffusion * diffusion_limit / (diffusion + diffusion_limit)

    return _apply_neutral_diffusion_boundaries(diffusion, mesh)


def build_neutral_mixed_transport_operators(
    diffusion: np.ndarray,
    log_pressure: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> tuple[np.ndarray, ...]:
    active_nx = mesh.xend - mesh.xstart + 1
    size = active_nx * mesh.nz
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g23 = np.asarray(metrics.g23, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)

    if not np.allclose(g23, 0.0, rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError(
            "Native neutral mixed transport currently requires g23 = 0."
        )

    def index(ix: int, kz: int) -> int:
        return ix * mesh.nz + kz

    operators: list[np.ndarray] = []
    for j in range(mesh.ystart, mesh.yend + 1):
        matrix = np.zeros((size, size), dtype=np.float64)
        for i in range(mesh.xstart - 1, mesh.xend + 1):
            for k in range(mesh.nz):
                face_flux = (
                    0.5
                    * (diffusion[i, j, k] + diffusion[i + 1, j, k])
                    * (J[i, j, k] * g11[i, j, k] + J[i + 1, j, k] * g11[i + 1, j, k])
                    * (log_pressure[i + 1, j, k] - log_pressure[i, j, k])
                    / (dx[i, j, k] + dx[i + 1, j, k])
                )
                if mesh.xstart <= i <= mesh.xend:
                    row = index(i - mesh.xstart, k)
                    matrix[row, row] += 0.5 * face_flux / (dx[i, j, k] * J[i, j, k])
                if mesh.xstart <= i + 1 <= mesh.xend:
                    row = index(i + 1 - mesh.xstart, k)
                    matrix[row, row] -= (
                        0.5 * face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])
                    )
                if mesh.xstart <= i <= mesh.xend and mesh.xstart <= i + 1 <= mesh.xend:
                    row_left = index(i - mesh.xstart, k)
                    row_right = index(i + 1 - mesh.xstart, k)
                    matrix[row_left, row_right] += (
                        0.5 * face_flux / (dx[i, j, k] * J[i, j, k])
                    )
                    matrix[row_right, row_left] -= (
                        0.5 * face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])
                    )

        for i in range(mesh.xstart, mesh.xend + 1):
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                face_flux = (
                    0.25
                    * (diffusion[i, j, k] + diffusion[i, j, kp])
                    * (J[i, j, k] * g33[i, j, k] + J[i, j, kp] * g33[i, j, kp])
                    / dz[i, j, k]
                )
                row = index(i - mesh.xstart, k)
                row_periodic = index(i - mesh.xstart, kp)
                matrix[row, row] -= face_flux / (J[i, j, k] * dz[i, j, k])
                matrix[row, row_periodic] += face_flux / (J[i, j, k] * dz[i, j, k])
                matrix[row_periodic, row] += face_flux / (J[i, j, kp] * dz[i, j, kp])
                matrix[row_periodic, row_periodic] -= face_flux / (
                    J[i, j, kp] * dz[i, j, kp]
                )

        operators.append(matrix)
    return tuple(operators)


def _prepare_neutral_mixed_state(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
) -> _PreparedNeutralMixedState:
    atomic_mass = _section_scalar(config, section, "AA", default=1.0)
    flux_limit = _section_scalar(config, section, "flux_limit", default=0.2)
    diffusion_limit = _section_scalar(config, section, "diffusion_limit", default=-1.0)
    density_floor = _section_scalar(config, section, "density_floor", default=1.0e-8)
    temperature_floor = (
        _section_scalar(config, section, "temperature_floor", default=0.1) / tnorm
    )
    pressure_floor = density_floor * temperature_floor
    lax_flux = (
        bool(config.parsed(section, "lax_flux"))
        if config.has_option(section, "lax_flux")
        else True
    )

    density = _apply_neutral_density_boundaries(
        np.maximum(np.asarray(state.density, dtype=np.float64), 0.0), mesh
    )
    pressure = _apply_neutral_pressure_boundaries(
        np.maximum(np.asarray(state.pressure, dtype=np.float64), 0.0), mesh
    )
    momentum = _apply_neutral_momentum_boundaries(
        np.asarray(state.momentum, dtype=np.float64), mesh
    )

    density_limited = _apply_neutral_density_boundaries(
        _soft_floor(density, density_floor), mesh
    )
    temperature = _apply_neutral_temperature_boundaries(
        pressure / density_limited, mesh
    )
    temperature_limited = _apply_neutral_temperature_boundaries(
        _soft_floor(temperature, temperature_floor), mesh
    )
    pressure_limited = _apply_neutral_pressure_boundaries(
        _soft_floor(pressure, pressure_floor), mesh
    )
    log_pressure = _apply_neutral_pressure_boundaries(np.log(pressure_limited), mesh)
    velocity = _apply_neutral_velocity_boundaries(
        momentum / (atomic_mass * density_limited), mesh
    )

    diffusion = compute_neutral_mixed_diffusion(
        temperature_limited,
        log_pressure,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        meters_scale=meters_scale,
        flux_limit=flux_limit,
        diffusion_limit=diffusion_limit,
    )
    diffusion_density = _apply_neutral_diffusion_boundaries(
        diffusion * density_limited, mesh
    )
    diffusion_pressure = _apply_neutral_diffusion_boundaries(
        diffusion * pressure_limited, mesh
    )
    diffusion_momentum = _apply_neutral_diffusion_boundaries(diffusion * momentum, mesh)
    conductivity = _apply_neutral_diffusion_boundaries(
        (5.0 / 2.0) * diffusion_density, mesh
    )
    viscosity = _apply_neutral_diffusion_boundaries(
        atomic_mass * (2.0 / 5.0) * conductivity, mesh
    )
    sound_speed = (
        np.sqrt(np.maximum(temperature, 0.0) * (5.0 / 3.0) / atomic_mass)
        if lax_flux
        else np.zeros_like(density)
    )

    return _PreparedNeutralMixedState(
        density=density,
        pressure=pressure,
        momentum=momentum,
        density_limited=density_limited,
        pressure_limited=pressure_limited,
        temperature=temperature,
        temperature_limited=temperature_limited,
        velocity=velocity,
        diffusion=diffusion,
        diffusion_density=diffusion_density,
        diffusion_pressure=diffusion_pressure,
        diffusion_momentum=diffusion_momentum,
        conductivity=conductivity,
        viscosity=viscosity,
        log_pressure=log_pressure,
        sound_speed=sound_speed,
    )


def _evaluate_field_option(
    config: BoutConfig,
    variable_name: str,
    *,
    mesh: StructuredMesh,
) -> np.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    if config.has_option(variable_name, "function"):
        value = evaluator.resolve_option(variable_name, "function")
    elif config.has_option(variable_name, "solution"):
        value = evaluator.resolve_option(variable_name, "solution")
    else:
        raise KeyError(f"Missing function or solution for {variable_name}.")
    return np.asarray(broadcast_to_field_shape(value, mesh), dtype=np.float64)


def _section_scalar(
    config: BoutConfig, section: str, name: str, *, default: float
) -> float:
    if not config.has_option(section, name):
        return default
    return float(config.parsed(section, name))


def _has_connected_y_ends(mesh: StructuredMesh) -> bool:
    return not mesh.has_lower_y_target and not mesh.has_upper_y_target


def _apply_connected_y_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    if mesh.myg <= 0:
        return result

    interior = result[:, mesh.ystart : mesh.yend + 1, :]
    for offset in range(mesh.myg):
        result[:, mesh.ystart - 1 - offset, :] = interior[:, -(offset + 1), :]
        result[:, mesh.yend + 1 + offset, :] = interior[:, offset, :]
    return result


def _apply_neutral_density_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_neumann_x_boundaries(field, mesh), mesh
        )
    return _apply_density_boundaries(field, mesh)


def _apply_neutral_pressure_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_neumann_x_boundaries(field, mesh), mesh
        )
    return _apply_pressure_boundaries(field, mesh)


def _apply_neutral_temperature_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_neumann_x_boundaries(field, mesh), mesh
        )
    return _apply_temperature_boundaries(field, mesh)


def _apply_neutral_diffusion_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_dirichlet_x_boundaries(field, mesh), mesh
        )
    return _apply_diffusion_boundaries(field, mesh)


def _apply_neutral_momentum_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_dirichlet_x_boundaries(field, mesh), mesh
        )
    return _apply_momentum_boundaries(field, mesh)


def _apply_neutral_velocity_boundaries(
    field: np.ndarray, mesh: StructuredMesh
) -> np.ndarray:
    if _has_connected_y_ends(mesh):
        return _apply_connected_y_boundaries(
            _apply_dirichlet_x_boundaries(field, mesh), mesh
        )
    return _apply_velocity_boundaries(field, mesh)


def _sanitize_neutral_state(
    state: NeutralMixedState, mesh: StructuredMesh
) -> NeutralMixedState:
    return NeutralMixedState(
        density=_apply_neutral_density_boundaries(np.maximum(state.density, 0.0), mesh),
        pressure=_apply_neutral_pressure_boundaries(
            np.maximum(state.pressure, 0.0), mesh
        ),
        momentum=_apply_neutral_momentum_boundaries(state.momentum, mesh),
    )


def _rk4_step(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
    timestep: float,
) -> NeutralMixedState:
    k1 = compute_neutral_mixed_rhs(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    k2 = compute_neutral_mixed_rhs(
        config,
        _add_state(state, k1, scale=0.5 * timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    k3 = compute_neutral_mixed_rhs(
        config,
        _add_state(state, k2, scale=0.5 * timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    k4 = compute_neutral_mixed_rhs(
        config,
        _add_state(state, k3, scale=timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )

    next_state = NeutralMixedState(
        density=state.density
        + (timestep / 6.0)
        * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density),
        pressure=state.pressure
        + (timestep / 6.0)
        * (k1.pressure + 2.0 * k2.pressure + 2.0 * k3.pressure + k4.pressure),
        momentum=state.momentum
        + (timestep / 6.0)
        * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum),
    )
    return _sanitize_neutral_state(next_state, mesh)


def _add_state(
    state: NeutralMixedState, rhs: NeutralMixedRhsResult, *, scale: float
) -> NeutralMixedState:
    return NeutralMixedState(
        density=state.density + scale * rhs.density,
        pressure=state.pressure + scale * rhs.pressure,
        momentum=state.momentum + scale * rhs.momentum,
    )


def _active_domain_slices(mesh: StructuredMesh) -> tuple[slice, slice, slice]:
    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )


def _active_domain_shape(mesh: StructuredMesh) -> tuple[int, int, int]:
    return active_region_from_slices(
        (mesh.nx, mesh.local_ny, mesh.nz), _active_domain_slices(mesh)
    ).shape


def _as_neutral_step_info(info: ImplicitStepInfo) -> NeutralMixedImplicitStepInfo:
    return NeutralMixedImplicitStepInfo(
        residual_inf_norm=info.residual_inf_norm,
        active_shape=tuple(int(axis) for axis in info.active_shape),
        nonlinear_iterations=info.nonlinear_iterations,
        linear_iterations=info.linear_iterations,
        diagnostics={
            "residual_evaluation_count": int(
                getattr(info, "residual_evaluation_count", 0)
            ),
            "residual_evaluation_seconds": float(
                getattr(info, "residual_evaluation_seconds", 0.0)
            ),
            "jacobian_refresh_count": int(getattr(info, "jacobian_refresh_count", 0)),
            "jacobian_assembly_seconds": float(
                getattr(info, "jacobian_assembly_seconds", 0.0)
            ),
            "linear_solve_seconds": float(getattr(info, "linear_solve_seconds", 0.0)),
            "line_search_seconds": float(getattr(info, "line_search_seconds", 0.0)),
            "fallback_used": bool(getattr(info, "fallback_used", False)),
        },
    )


@lru_cache(maxsize=None)
def _neutral_mixed_jacobian_sparsity(active_shape: tuple[int, int, int]):
    return build_locality_sparsity(
        active_shape,
        field_count=3,
        radii=(2, 2, 2),
        periodic_axes=(2,),
    )


@lru_cache(maxsize=None)
def _neutral_mixed_jacobian_color_groups(
    active_shape: tuple[int, int, int],
) -> tuple[tuple[int, ...], ...]:
    return build_modulo_color_groups(
        active_shape,
        field_count=3,
        color_periods=(
            min(5, active_shape[0]),
            min(5, active_shape[1]),
            active_shape[2],
        ),
    )
