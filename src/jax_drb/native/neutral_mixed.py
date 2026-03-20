from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from ..config.boutinp import BoutConfig
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class NeutralMixedState:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray


@dataclass(frozen=True)
class NeutralMixedRhsResult:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    diffusion: np.ndarray
    density_parallel_flow: np.ndarray
    pressure_parallel_flow: np.ndarray


@dataclass(frozen=True)
class NeutralMixedHistoryResult:
    density_history: np.ndarray
    pressure_history: np.ndarray
    momentum_history: np.ndarray


@dataclass(frozen=True)
class NeutralMixedImplicitStepInfo:
    residual_inf_norm: float
    active_shape: tuple[int, int, int]
    nonlinear_iterations: int
    linear_iterations: int


@dataclass(frozen=True)
class _PreparedNeutralMixedState:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    density_limited: np.ndarray
    pressure_limited: np.ndarray
    temperature: np.ndarray
    temperature_limited: np.ndarray
    velocity: np.ndarray
    diffusion: np.ndarray
    diffusion_density: np.ndarray
    diffusion_pressure: np.ndarray
    diffusion_momentum: np.ndarray
    conductivity: np.ndarray
    viscosity: np.ndarray
    log_pressure: np.ndarray
    sound_speed: np.ndarray


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
        density=_apply_density_boundaries(density, mesh),
        pressure=_apply_pressure_boundaries(pressure, mesh),
        momentum=_apply_momentum_boundaries(momentum, mesh),
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


def pack_neutral_mixed_active_state(
    state: NeutralMixedState,
    *,
    mesh: StructuredMesh,
) -> np.ndarray:
    active = _active_domain_slices(mesh)
    return np.concatenate(
        [
            np.asarray(state.density[active], dtype=np.float64).ravel(),
            np.asarray(state.pressure[active], dtype=np.float64).ravel(),
            np.asarray(state.momentum[active], dtype=np.float64).ravel(),
        ]
    )


def unpack_neutral_mixed_active_state(
    packed: np.ndarray,
    *,
    template: NeutralMixedState,
    mesh: StructuredMesh,
) -> NeutralMixedState:
    active = _active_domain_slices(mesh)
    active_shape = template.density[active].shape
    active_size = int(np.prod(active_shape))
    density = np.array(template.density, copy=True)
    pressure = np.array(template.pressure, copy=True)
    momentum = np.array(template.momentum, copy=True)
    density[active] = np.asarray(packed[:active_size], dtype=np.float64).reshape(active_shape)
    pressure[active] = np.asarray(packed[active_size : 2 * active_size], dtype=np.float64).reshape(active_shape)
    momentum[active] = np.asarray(packed[2 * active_size :], dtype=np.float64).reshape(active_shape)
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
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Neutral Jacobian sparsity construction requires scipy.") from exc

    active_nx = mesh.xend - mesh.xstart + 1
    active_ny = mesh.yend - mesh.ystart + 1
    active_cells = active_nx * active_ny * mesh.nz
    total_size = 3 * active_cells

    def active_index(ix: int, iy: int, iz: int) -> int:
        return ((ix * active_ny) + iy) * mesh.nz + iz

    row_indices: list[int] = []
    col_indices: list[int] = []
    for equation_block in range(3):
        row_offset = equation_block * active_cells
        for ix in range(active_nx):
            for iy in range(active_ny):
                for iz in range(mesh.nz):
                    row = row_offset + active_index(ix, iy, iz)
                    neighbors: set[int] = set()
                    for dx in range(-2, 3):
                        nix = ix + dx
                        if not (0 <= nix < active_nx):
                            continue
                        for dy in range(-2, 3):
                            niy = iy + dy
                            if not (0 <= niy < active_ny):
                                continue
                            for dz in range(-2, 3):
                                niz = (iz + dz) % mesh.nz
                                neighbors.add(active_index(nix, niy, niz))
                    for variable_block in range(3):
                        col_offset = variable_block * active_cells
                        for neighbor in neighbors:
                            row_indices.append(row)
                            col_indices.append(col_offset + neighbor)

    data = np.ones(len(row_indices), dtype=bool)
    return coo_matrix((data, (row_indices, col_indices)), shape=(total_size, total_size)).tocsr()


def build_neutral_mixed_active_jacobian_color_groups(
    mesh: StructuredMesh,
) -> tuple[tuple[int, ...], ...]:
    active_nx = mesh.xend - mesh.xstart + 1
    active_ny = mesh.yend - mesh.ystart + 1
    active_cells = active_nx * active_ny * mesh.nz

    def active_index(ix: int, iy: int, iz: int) -> int:
        return ((ix * active_ny) + iy) * mesh.nz + iz

    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for variable_block in range(3):
        block_offset = variable_block * active_cells
        for ix in range(active_nx):
            for iy in range(active_ny):
                for iz in range(mesh.nz):
                    color = (variable_block, ix % 5, iy % 5, iz)
                    groups.setdefault(color, []).append(block_offset + active_index(ix, iy, iz))
    return tuple(tuple(groups[key]) for key in sorted(groups))


def build_neutral_mixed_sparse_residual_jacobian(
    residual,
    packed_state: np.ndarray,
    *,
    mesh: StructuredMesh,
    base_residual: np.ndarray | None = None,
    sparsity=None,
    color_groups: tuple[tuple[int, ...], ...] | None = None,
):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Neutral sparse Jacobian construction requires scipy.") from exc

    state = np.asarray(packed_state, dtype=np.float64)
    residual0 = (
        np.asarray(base_residual, dtype=np.float64)
        if base_residual is not None
        else np.asarray(residual(state), dtype=np.float64)
    )
    if sparsity is None:
        sparsity = build_neutral_mixed_active_jacobian_sparsity(mesh)
    if color_groups is None:
        color_groups = build_neutral_mixed_active_jacobian_color_groups(mesh)
    sparsity_csc = sparsity.tocsc()

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []

    for group in color_groups:
        perturbation = np.zeros_like(state)
        group_steps: list[tuple[int, float]] = []
        for column in group:
            step = _difference_quotient_step_size(state[column])
            perturbation[column] = step
            group_steps.append((column, step))

        perturbed_residual = np.asarray(residual(state + perturbation), dtype=np.float64)
        delta = perturbed_residual - residual0
        for column, step in group_steps:
            rows = sparsity_csc.indices[sparsity_csc.indptr[column] : sparsity_csc.indptr[column + 1]]
            row_indices.extend(rows.tolist())
            col_indices.extend([column] * len(rows))
            data.extend((delta[rows] / step).tolist())

    return coo_matrix((data, (row_indices, col_indices)), shape=sparsity.shape).tocsr()


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
    return np.asarray(packed_state, dtype=np.float64) - np.asarray(previous_packed_state, dtype=np.float64) - float(timestep) * rhs


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
    return (
        np.asarray(packed_state, dtype=np.float64)
        - (4.0 / 3.0) * np.asarray(previous_packed_state, dtype=np.float64)
        + (1.0 / 3.0) * np.asarray(previous_previous_packed_state, dtype=np.float64)
        - (2.0 / 3.0) * float(timestep) * rhs
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
        solved, info = _solve_neutral_mixed_sparse_newton_system(
            residual,
            previous_packed_state,
            mesh=mesh,
            residual_tolerance=residual_tolerance,
            step_tolerance=step_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_rtol=linear_rtol,
        )
    elif solver_mode == "matrix_free":
        solved, info = _solve_neutral_mixed_matrix_free_newton_system(
            residual,
            previous_packed_state,
            mesh=mesh,
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
    return next_state, info


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
    solver_mode: str = "matrix_free",
    residual_tolerance: float = 1.0e-9,
    step_tolerance: float = 1.0e-11,
    max_nonlinear_iterations: int = 12,
    linear_restart: int = 20,
    linear_maxiter: int = 300,
    linear_rtol: float = 1.0e-8,
) -> tuple[NeutralMixedState, NeutralMixedImplicitStepInfo]:
    previous_packed_state = pack_neutral_mixed_active_state(state, mesh=mesh)
    previous_previous_packed_state = pack_neutral_mixed_active_state(previous_state, mesh=mesh)

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
        )

    if solver_mode == "sparse":
        solved, info = _solve_neutral_mixed_sparse_newton_system(
            residual,
            previous_packed_state,
            mesh=mesh,
            residual_tolerance=residual_tolerance,
            step_tolerance=step_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_rtol=linear_rtol,
        )
    elif solver_mode == "matrix_free":
        solved, info = _solve_neutral_mixed_matrix_free_newton_system(
            residual,
            previous_packed_state,
            mesh=mesh,
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
    return next_state, info


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
    density_rhs = -_div_par_mod_open(
        prepared.density,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    density_parallel_flow = _last_parallel_flow.copy()
    density_rhs += _div_a_grad_perp_flows(
        prepared.diffusion_density,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )

    pressure_rhs = -(5.0 / 3.0) * _div_par_mod_open(
        prepared.pressure,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    pressure_parallel_flow = (5.0 / 2.0) * _last_parallel_flow.copy()
    pressure_rhs += (2.0 / 3.0) * prepared.velocity * _grad_par_open(
        prepared.pressure,
        mesh=mesh,
        metrics=metrics,
    )
    pressure_rhs += (5.0 / 3.0) * _div_a_grad_perp_flows(
        prepared.diffusion_pressure,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )

    if bool(config.parsed(section, "neutral_conduction")) if config.has_option(section, "neutral_conduction") else True:
        pressure_rhs += (2.0 / 3.0) * _div_par_k_grad_par_open(
            prepared.conductivity,
            prepared.temperature,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        pressure_rhs += (2.0 / 3.0) * _div_a_grad_perp_flows(
            prepared.conductivity,
            prepared.temperature,
            mesh=mesh,
            metrics=metrics,
        )

    momentum_rhs = -_section_scalar(config, section, "AA", default=1.0) * _div_par_fvv_open(
        prepared.density_limited,
        prepared.velocity,
        prepared.sound_speed,
        mesh=mesh,
        metrics=metrics,
    )
    momentum_rhs -= _grad_par_open(prepared.pressure, mesh=mesh, metrics=metrics)
    momentum_rhs += _div_a_grad_perp_flows(
        prepared.diffusion_momentum,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )

    if bool(config.parsed(section, "neutral_viscosity")) if config.has_option(section, "neutral_viscosity") else True:
        viscosity_source = _div_par_k_grad_par_open(
            prepared.viscosity,
            prepared.velocity,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        viscosity_source += _div_a_grad_perp_flows(
            prepared.viscosity,
            prepared.velocity,
            mesh=mesh,
            metrics=metrics,
        )
        momentum_rhs += viscosity_source
        pressure_rhs += -(2.0 / 3.0) * prepared.velocity * viscosity_source

    return NeutralMixedRhsResult(
        density=np.asarray(density_rhs, dtype=np.float64),
        pressure=np.asarray(pressure_rhs, dtype=np.float64),
        momentum=np.asarray(momentum_rhs, dtype=np.float64),
        diffusion=np.asarray(prepared.diffusion, dtype=np.float64),
        density_parallel_flow=np.asarray(density_parallel_flow, dtype=np.float64),
        pressure_parallel_flow=np.asarray(pressure_parallel_flow, dtype=np.float64),
    )


def compute_neutral_mixed_diffusion(
    density: np.ndarray,
    pressure: np.ndarray,
    log_pressure: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    meters_scale: float,
    flux_limit: float,
    diffusion_limit: float = -1.0,
) -> np.ndarray:
    temperature = pressure / density
    thermal_speed = np.sqrt(temperature / atomic_mass)
    neutral_lmax = 0.1 / meters_scale
    raw_diffusion = thermal_speed * neutral_lmax

    if flux_limit > 0.0:
        grad_magnitude = _gradient_magnitude(log_pressure, mesh=mesh, metrics=metrics)
        diffusion_max = flux_limit * thermal_speed / (grad_magnitude + (1.0 / neutral_lmax))
        diffusion = raw_diffusion * diffusion_max / (raw_diffusion + diffusion_max)
    else:
        diffusion = raw_diffusion

    if diffusion_limit > 0.0:
        diffusion = diffusion * diffusion_limit / (diffusion + diffusion_limit)

    return _apply_diffusion_boundaries(diffusion, mesh)


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
        raise NotImplementedError("Native neutral mixed transport currently requires g23 = 0.")

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
                    matrix[row, row] -= 0.5 * face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])
                if mesh.xstart <= i <= mesh.xend and mesh.xstart <= i + 1 <= mesh.xend:
                    row_left = index(i - mesh.xstart, k)
                    row_right = index(i + 1 - mesh.xstart, k)
                    matrix[row_left, row_right] += 0.5 * face_flux / (dx[i, j, k] * J[i, j, k])
                    matrix[row_right, row_left] -= 0.5 * face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])

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
                matrix[row_periodic, row_periodic] -= face_flux / (J[i, j, kp] * dz[i, j, kp])

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
    temperature_floor = _section_scalar(config, section, "temperature_floor", default=0.1) / tnorm
    pressure_floor = density_floor * temperature_floor
    lax_flux = bool(config.parsed(section, "lax_flux")) if config.has_option(section, "lax_flux") else True

    density = _apply_density_boundaries(np.maximum(np.asarray(state.density, dtype=np.float64), 0.0), mesh)
    pressure = _apply_pressure_boundaries(np.maximum(np.asarray(state.pressure, dtype=np.float64), 0.0), mesh)
    momentum = _apply_momentum_boundaries(np.asarray(state.momentum, dtype=np.float64), mesh)

    density_limited = np.maximum(density, density_floor)
    temperature = _apply_temperature_boundaries(pressure / density_limited, mesh)
    temperature_limited = np.maximum(temperature, temperature_floor)
    pressure_limited = _apply_pressure_boundaries(np.maximum(pressure, pressure_floor), mesh)
    log_pressure = _apply_pressure_boundaries(np.log(pressure_limited), mesh)
    velocity = _apply_velocity_boundaries(momentum / (atomic_mass * density_limited), mesh)

    diffusion = compute_neutral_mixed_diffusion(
        density_limited,
        pressure_limited,
        log_pressure,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        meters_scale=meters_scale,
        flux_limit=flux_limit,
        diffusion_limit=diffusion_limit,
    )
    diffusion_density = _apply_diffusion_boundaries(diffusion * density_limited, mesh)
    diffusion_pressure = _apply_diffusion_boundaries(diffusion * pressure_limited, mesh)
    diffusion_momentum = _apply_diffusion_boundaries(diffusion * momentum, mesh)
    conductivity = _apply_diffusion_boundaries((5.0 / 2.0) * diffusion_density, mesh)
    viscosity = _apply_diffusion_boundaries(atomic_mass * (2.0 / 5.0) * conductivity, mesh)
    sound_speed = np.sqrt(np.maximum(temperature, 0.0) * (5.0 / 3.0) / atomic_mass) if lax_flux else np.zeros_like(density)

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


def _section_scalar(config: BoutConfig, section: str, name: str, *, default: float) -> float:
    if not config.has_option(section, name):
        return default
    return float(config.parsed(section, name))


def _sanitize_neutral_state(state: NeutralMixedState, mesh: StructuredMesh) -> NeutralMixedState:
    return NeutralMixedState(
        density=_apply_density_boundaries(np.maximum(state.density, 0.0), mesh),
        pressure=_apply_pressure_boundaries(np.maximum(state.pressure, 0.0), mesh),
        momentum=_apply_momentum_boundaries(state.momentum, mesh),
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
        density=state.density + (timestep / 6.0) * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density),
        pressure=state.pressure
        + (timestep / 6.0) * (k1.pressure + 2.0 * k2.pressure + 2.0 * k3.pressure + k4.pressure),
        momentum=state.momentum
        + (timestep / 6.0) * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum),
    )
    return _sanitize_neutral_state(next_state, mesh)


def _add_state(state: NeutralMixedState, rhs: NeutralMixedRhsResult, *, scale: float) -> NeutralMixedState:
    return NeutralMixedState(
        density=state.density + scale * rhs.density,
        pressure=state.pressure + scale * rhs.pressure,
        momentum=state.momentum + scale * rhs.momentum,
    )


def _solve_neutral_mixed_sparse_newton_system(
    residual,
    initial_state: np.ndarray,
    *,
    mesh: StructuredMesh,
    residual_tolerance: float,
    step_tolerance: float,
    max_nonlinear_iterations: int,
    linear_restart: int,
    linear_maxiter: int,
    linear_rtol: float,
) -> tuple[np.ndarray, NeutralMixedImplicitStepInfo]:
    try:
        from scipy.optimize import newton_krylov
        from scipy.sparse.linalg import gmres, spsolve
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Neutral sparse implicit stepping requires scipy.") from exc

    state = np.asarray(initial_state, dtype=np.float64).copy()
    sparsity = build_neutral_mixed_active_jacobian_sparsity(mesh)
    color_groups = build_neutral_mixed_active_jacobian_color_groups(mesh)
    total_linear_iterations = 0

    for nonlinear_iteration in range(1, int(max_nonlinear_iterations) + 1):
        residual_value = np.asarray(residual(state), dtype=np.float64)
        residual_inf_norm = float(np.max(np.abs(residual_value)))
        if residual_inf_norm < float(residual_tolerance):
            return state, NeutralMixedImplicitStepInfo(
                residual_inf_norm=residual_inf_norm,
                active_shape=_active_domain_shape(mesh),
                nonlinear_iterations=nonlinear_iteration - 1,
                linear_iterations=total_linear_iterations,
            )

        jacobian = build_neutral_mixed_sparse_residual_jacobian(
            residual,
            state,
            mesh=mesh,
            base_residual=residual_value,
            sparsity=sparsity,
            color_groups=color_groups,
        )
        linear_iterations = 0

        def callback(_residual_norm) -> None:
            nonlocal linear_iterations
            linear_iterations += 1

        update, exit_code = gmres(
            jacobian,
            -residual_value,
            restart=int(linear_restart),
            maxiter=int(linear_maxiter),
            rtol=float(linear_rtol),
            atol=0.0,
            callback=callback,
            callback_type="pr_norm",
        )
        total_linear_iterations += linear_iterations
        if exit_code != 0:
            update = spsolve(jacobian.tocsc(), -residual_value)
            total_linear_iterations += 1

        state = state + np.asarray(update, dtype=np.float64)
        if not np.all(np.isfinite(state)):
            break
        if float(np.max(np.abs(update))) < float(step_tolerance):
            residual_value = np.asarray(residual(state), dtype=np.float64)
            residual_inf_norm = float(np.max(np.abs(residual_value)))
            return state, NeutralMixedImplicitStepInfo(
                residual_inf_norm=residual_inf_norm,
                active_shape=_active_domain_shape(mesh),
                nonlinear_iterations=nonlinear_iteration,
                linear_iterations=total_linear_iterations,
            )

    solved = newton_krylov(
        residual,
        np.asarray(initial_state, dtype=np.float64),
        f_tol=float(residual_tolerance),
        maxiter=max(int(max_nonlinear_iterations), 25),
        method="lgmres",
        verbose=0,
    )
    residual_value = np.asarray(residual(np.asarray(solved, dtype=np.float64)), dtype=np.float64)
    residual_inf_norm = float(np.max(np.abs(residual_value)))
    return np.asarray(solved, dtype=np.float64), NeutralMixedImplicitStepInfo(
        residual_inf_norm=residual_inf_norm,
        active_shape=_active_domain_shape(mesh),
        nonlinear_iterations=int(max_nonlinear_iterations),
        linear_iterations=total_linear_iterations,
    )


def _solve_neutral_mixed_matrix_free_newton_system(
    residual,
    initial_state: np.ndarray,
    *,
    mesh: StructuredMesh,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> tuple[np.ndarray, NeutralMixedImplicitStepInfo]:
    try:
        from scipy.optimize import newton_krylov
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Neutral matrix-free implicit stepping requires scipy.") from exc

    iteration_budget = max(int(max_nonlinear_iterations), 25)
    solved = newton_krylov(
        residual,
        np.asarray(initial_state, dtype=np.float64),
        f_tol=float(residual_tolerance),
        maxiter=iteration_budget,
        method="lgmres",
        verbose=0,
    )
    residual_value = np.asarray(residual(np.asarray(solved, dtype=np.float64)), dtype=np.float64)
    return np.asarray(solved, dtype=np.float64), NeutralMixedImplicitStepInfo(
        residual_inf_norm=float(np.max(np.abs(residual_value))),
        active_shape=_active_domain_shape(mesh),
        nonlinear_iterations=iteration_budget,
        linear_iterations=iteration_budget,
    )


def _active_domain_slices(mesh: StructuredMesh) -> tuple[slice, slice, slice]:
    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )


def _active_domain_shape(mesh: StructuredMesh) -> tuple[int, int, int]:
    return (
        mesh.xend - mesh.xstart + 1,
        mesh.yend - mesh.ystart + 1,
        mesh.nz,
    )


def _difference_quotient_step_size(value: float) -> float:
    scale = max(1.0, abs(float(value)))
    return np.sqrt(np.finfo(np.float64).eps) * scale


def _apply_neumann_x_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    for offset in range(1, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = result[mesh.xstart - 1 + offset, y_slice, :]
        result[mesh.xend + offset, y_slice, :] = result[mesh.xend + 1 - offset, y_slice, :]
    return result


def _apply_dirichlet_x_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[mesh.xstart - 1, y_slice, :] = -result[mesh.xstart, y_slice, :]
    result[mesh.xend + 1, y_slice, :] = -result[mesh.xend, y_slice, :]
    for offset in range(2, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = 0.0
        result[mesh.xend + offset, y_slice, :] = 0.0
    return result


def _apply_density_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_neumann_x_boundaries(field, mesh)
    return _apply_zero_gradient_y_boundaries(result, mesh)


def _apply_pressure_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_neumann_x_boundaries(field, mesh)
    return _apply_zero_gradient_y_boundaries(result, mesh)


def _apply_temperature_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_neumann_x_boundaries(field, mesh)
    return _apply_zero_gradient_y_boundaries(result, mesh)


def _apply_diffusion_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_dirichlet_x_boundaries(field, mesh)
    return _apply_antisymmetric_y_boundaries(result, mesh)


def _apply_momentum_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_dirichlet_x_boundaries(field, mesh)
    return _apply_antisymmetric_y_boundaries(result, mesh)


def _apply_velocity_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = _apply_dirichlet_x_boundaries(field, mesh)
    return _apply_antisymmetric_y_boundaries(result, mesh)


def _apply_zero_gradient_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    for offset in range(1, mesh.myg + 1):
        result[:, mesh.ystart - offset, :] = result[:, mesh.ystart - 1 + offset, :]
        result[:, mesh.yend + offset, :] = result[:, mesh.yend + 1 - offset, :]
    return result


def _apply_density_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    return _apply_zero_gradient_y_boundaries(field, mesh)


def _apply_antisymmetric_y_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    for offset in range(1, mesh.myg + 1):
        result[:, mesh.ystart - offset, :] = -result[:, mesh.ystart - 1 + offset, :]
        result[:, mesh.yend + offset, :] = -result[:, mesh.yend + 1 - offset, :]
    return result


def _gradient_magnitude(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                km = (k - 1 + mesh.nz) % mesh.nz
                kp = (k + 1) % mesh.nz
                dfdx = (field[i + 1, j, k] - field[i - 1, j, k]) / (dx[i, j, k] + dx[i - 1, j, k])
                dfdy = (field[i, j + 1, k] - field[i, j - 1, k]) / (dy[i, j, k] + dy[i, j - 1, k])
                dfdz = (field[i, j, kp] - field[i, j, km]) / (2.0 * dz[i, j, k])
                result[i, j, k] = np.sqrt(
                    g11[i, j, k] * dfdx * dfdx
                    + g33[i, j, k] * dfdz * dfdz
                    + (dfdy / J[i, j, k]) * (dfdy / J[i, j, k])
                )
    return result


def _mc_edges(center: np.ndarray, minus: np.ndarray, plus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def _minmod3_scalar(a: float, b: float, c: float) -> float:
    if (a * b <= 0.0) or (a * c <= 0.0):
        return 0.0
    magnitude = min(abs(a), abs(b), abs(c))
    return float(np.sign(a) * magnitude)


def _mc_edges_scalar(center: float, minus: float, plus: float) -> tuple[float, float]:
    slope = _minmod3_scalar(
        2.0 * (plus - center),
        0.5 * (plus - minus),
        2.0 * (center - minus),
    )
    return center - 0.5 * slope, center + 0.5 * slope


_last_parallel_flow = np.zeros((1, 1, 1), dtype=np.float64)


def _div_par_mod_open(
    field: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    if not fix_flux:
        raise NotImplementedError("Native neutral mixed advection currently supports fix_flux=True only.")
    result = np.zeros_like(field, dtype=np.float64)
    flow_ylow = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                right_common = (J[i, j, k] + J[i, j + 1, k]) / (
                    np.sqrt(g22[i, j, k]) + np.sqrt(g22[i, j + 1, k])
                )
                flux_factor_rc = right_common / (dy[i, j, k] * J[i, j, k])
                flux_factor_rp = right_common / (dy[i, j + 1, k] * J[i, j + 1, k])
                area_rp = right_common * dx[i, j + 1, k] * dz[i, j + 1, k]

                left_common = (J[i, j, k] + J[i, j - 1, k]) / (
                    np.sqrt(g22[i, j, k]) + np.sqrt(g22[i, j - 1, k])
                )
                flux_factor_lc = left_common / (dy[i, j, k] * J[i, j, k])
                flux_factor_lm = left_common / (dy[i, j - 1, k] * J[i, j - 1, k])
                area_lc = left_common * dx[i, j, k] * dz[i, j, k]

                s_left, s_right = _mc_edges_scalar(
                    float(field[i, j, k]),
                    float(field[i, j - 1, k]),
                    float(field[i, j + 1, k]),
                )
                v_left, v_right = _mc_edges_scalar(
                    float(velocity[i, j, k]),
                    float(velocity[i, j - 1, k]),
                    float(velocity[i, j + 1, k]),
                )

                if j == mesh.yend:
                    vpar = 0.5 * (velocity[i, j, k] + velocity[i, j + 1, k])
                    boundary_value = 0.5 * (field[i, j, k] + field[i, j + 1, k])
                    flux = boundary_value * vpar
                else:
                    amax = max(
                        float(wave_speed[i, j, k]),
                        float(wave_speed[i, j + 1, k]),
                        abs(float(velocity[i, j, k])),
                        abs(float(velocity[i, j + 1, k])),
                    )
                    flux = s_right * 0.5 * (v_right + amax)

                result[i, j, k] += flux * flux_factor_rc
                result[i, j + 1, k] -= flux * flux_factor_rp
                flow_ylow[i, j + 1, k] += flux * area_rp

                if j == mesh.ystart:
                    vpar = 0.5 * (velocity[i, j, k] + velocity[i, j - 1, k])
                    boundary_value = 0.5 * (field[i, j, k] + field[i, j - 1, k])
                    flux = boundary_value * vpar
                else:
                    amax = max(
                        float(wave_speed[i, j, k]),
                        float(wave_speed[i, j - 1, k]),
                        abs(float(velocity[i, j, k])),
                        abs(float(velocity[i, j - 1, k])),
                    )
                    flux = s_left * 0.5 * (v_left - amax)

                result[i, j, k] -= flux * flux_factor_lc
                result[i, j - 1, k] += flux * flux_factor_lm
                flow_ylow[i, j, k] += flux * area_lc

    global _last_parallel_flow
    _last_parallel_flow = np.asarray(flow_ylow, dtype=np.float64)
    return result


def _div_par_fvv_open(
    density: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    if not fix_flux:
        raise NotImplementedError("Native neutral mixed momentum advection currently supports fix_flux=True only.")
    result = np.zeros_like(density, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                right_common = (J[i, j, k] + J[i, j + 1, k]) / (
                    np.sqrt(g22[i, j, k]) + np.sqrt(g22[i, j + 1, k])
                )
                flux_factor_rc = right_common / (dy[i, j, k] * J[i, j, k])
                flux_factor_rp = right_common / (dy[i, j + 1, k] * J[i, j + 1, k])

                left_common = (J[i, j, k] + J[i, j - 1, k]) / (
                    np.sqrt(g22[i, j, k]) + np.sqrt(g22[i, j - 1, k])
                )
                flux_factor_lc = left_common / (dy[i, j, k] * J[i, j, k])
                flux_factor_lm = left_common / (dy[i, j - 1, k] * J[i, j - 1, k])

                s_left, s_right = _mc_edges_scalar(
                    float(density[i, j, k]),
                    float(density[i, j - 1, k]),
                    float(density[i, j + 1, k]),
                )
                v_left, v_right = _mc_edges_scalar(
                    float(velocity[i, j, k]),
                    float(velocity[i, j - 1, k]),
                    float(velocity[i, j + 1, k]),
                )
                v_mid_right = 0.5 * (velocity[i, j, k] + velocity[i, j + 1, k])
                n_mid_right = 0.5 * (density[i, j, k] + density[i, j + 1, k])
                if j == mesh.yend:
                    flux = n_mid_right * v_mid_right * v_mid_right
                else:
                    amax = max(
                        float(wave_speed[i, j, k]),
                        float(wave_speed[i, j + 1, k]),
                        abs(float(velocity[i, j, k])),
                        abs(float(velocity[i, j + 1, k])),
                    )
                    flux = s_right * 0.5 * (v_right + amax) * v_right

                result[i, j, k] += flux * flux_factor_rc
                result[i, j + 1, k] -= flux * flux_factor_rp

                v_mid_left = 0.5 * (velocity[i, j, k] + velocity[i, j - 1, k])
                n_mid_left = 0.5 * (density[i, j, k] + density[i, j - 1, k])
                if j == mesh.ystart:
                    flux = n_mid_left * v_mid_left * v_mid_left
                else:
                    amax = max(
                        float(wave_speed[i, j, k]),
                        float(wave_speed[i, j - 1, k]),
                        abs(float(velocity[i, j, k])),
                        abs(float(velocity[i, j - 1, k])),
                    )
                    flux = s_left * 0.5 * (v_left - amax) * v_left

                result[i, j, k] -= flux * flux_factor_lc
                result[i, j - 1, k] += flux * flux_factor_lm
    return result


def _div_par_k_grad_par_open(
    coefficient: np.ndarray,
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    boundary_flux: bool,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    for j in range(mesh.ystart, mesh.yend + 1):
        if boundary_flux or j != mesh.yend:
            coefficient_up = 0.5 * (
                coefficient[mesh.xstart : mesh.xend + 1, j, :] + coefficient[mesh.xstart : mesh.xend + 1, j + 1, :]
            )
            jacobian_up = 0.5 * (J[mesh.xstart : mesh.xend + 1, j, :] + J[mesh.xstart : mesh.xend + 1, j + 1, :])
            metric_up = 0.5 * (g22[mesh.xstart : mesh.xend + 1, j, :] + g22[mesh.xstart : mesh.xend + 1, j + 1, :])
            gradient_up = (
                2.0
                * (
                    field[mesh.xstart : mesh.xend + 1, j + 1, :]
                    - field[mesh.xstart : mesh.xend + 1, j, :]
                )
                / (dy[mesh.xstart : mesh.xend + 1, j, :] + dy[mesh.xstart : mesh.xend + 1, j + 1, :])
            )
            flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
            result[mesh.xstart : mesh.xend + 1, j, :] += flux_up / (
                dy[mesh.xstart : mesh.xend + 1, j, :] * J[mesh.xstart : mesh.xend + 1, j, :]
            )

        if boundary_flux or j != mesh.ystart:
            coefficient_down = 0.5 * (
                coefficient[mesh.xstart : mesh.xend + 1, j, :] + coefficient[mesh.xstart : mesh.xend + 1, j - 1, :]
            )
            jacobian_down = 0.5 * (J[mesh.xstart : mesh.xend + 1, j, :] + J[mesh.xstart : mesh.xend + 1, j - 1, :])
            metric_down = 0.5 * (g22[mesh.xstart : mesh.xend + 1, j, :] + g22[mesh.xstart : mesh.xend + 1, j - 1, :])
            gradient_down = (
                2.0
                * (
                    field[mesh.xstart : mesh.xend + 1, j, :]
                    - field[mesh.xstart : mesh.xend + 1, j - 1, :]
                )
                / (dy[mesh.xstart : mesh.xend + 1, j, :] + dy[mesh.xstart : mesh.xend + 1, j - 1, :])
            )
            flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
            result[mesh.xstart : mesh.xend + 1, j, :] -= flux_down / (
                dy[mesh.xstart : mesh.xend + 1, j, :] * J[mesh.xstart : mesh.xend + 1, j, :]
            )

    return result


def _div_a_grad_perp_flows(
    coefficient: np.ndarray,
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g23 = np.asarray(metrics.g23, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)

    if not np.allclose(g23, 0.0, rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native neutral mixed transport currently requires g23 = 0.")

    for i in range(mesh.xstart - 1, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                face_flux = (
                    0.5
                    * (coefficient[i, j, k] + coefficient[i + 1, j, k])
                    * (J[i, j, k] * g11[i, j, k] + J[i + 1, j, k] * g11[i + 1, j, k])
                    * (field[i + 1, j, k] - field[i, j, k])
                    / (dx[i, j, k] + dx[i + 1, j, k])
                )
                result[i, j, k] += face_flux / (dx[i, j, k] * J[i, j, k])
                result[i + 1, j, k] -= face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                face_flux = (
                    0.25
                    * (coefficient[i, j, k] + coefficient[i, j, kp])
                    * (J[i, j, k] * g33[i, j, k] + J[i, j, kp] * g33[i, j, kp])
                    * ((field[i, j, kp] - field[i, j, k]) / dz[i, j, k])
                )
                result[i, j, k] += face_flux / (J[i, j, k] * dz[i, j, k])
                result[i, j, kp] -= face_flux / (J[i, j, kp] * dz[i, j, kp])

    return result


def _grad_par_open(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                spacing = dy[i, j, k] + dy[i, j - 1, k]
                result[i, j, k] = (field[i, j + 1, k] - field[i, j - 1, k]) / (spacing * J[i, j, k])
    return result
