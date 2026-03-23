from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImplicitStepInfo:
    residual_inf_norm: float
    active_shape: tuple[int, ...]
    nonlinear_iterations: int
    linear_iterations: int


def difference_quotient_step_size(value: float) -> float:
    scale = max(1.0, abs(float(value)))
    return np.sqrt(np.finfo(np.float64).eps) * scale


def build_locality_sparsity(
    active_shape: tuple[int, ...],
    *,
    field_count: int,
    radii: tuple[int, ...],
    periodic_axes: tuple[int, ...] = (),
):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Sparse locality construction requires scipy.") from exc

    if len(active_shape) != len(radii):
        raise ValueError("active_shape and radii must have the same rank.")

    active_cells = int(np.prod(active_shape))
    total_size = field_count * active_cells
    periodic_axis_set = set(periodic_axes)
    row_indices: list[int] = []
    col_indices: list[int] = []

    for equation_block in range(field_count):
        row_offset = equation_block * active_cells
        for cell_index in np.ndindex(active_shape):
            row = row_offset + np.ravel_multi_index(cell_index, active_shape)
            neighbors: set[int] = set()
            for delta in np.ndindex(tuple(2 * radius + 1 for radius in radii)):
                neighbor_index: list[int] = []
                valid = True
                for axis, (coordinate, radius, raw_offset) in enumerate(zip(cell_index, radii, delta, strict=True)):
                    offset = raw_offset - radius
                    neighbor_coordinate = coordinate + offset
                    if axis in periodic_axis_set:
                        neighbor_coordinate %= active_shape[axis]
                    elif not (0 <= neighbor_coordinate < active_shape[axis]):
                        valid = False
                        break
                    neighbor_index.append(neighbor_coordinate)
                if valid:
                    neighbors.add(np.ravel_multi_index(tuple(neighbor_index), active_shape))
            for variable_block in range(field_count):
                col_offset = variable_block * active_cells
                for neighbor in neighbors:
                    row_indices.append(row)
                    col_indices.append(col_offset + neighbor)

    data = np.ones(len(row_indices), dtype=bool)
    return coo_matrix((data, (row_indices, col_indices)), shape=(total_size, total_size)).tocsr()


def build_modulo_color_groups(
    active_shape: tuple[int, ...],
    *,
    field_count: int,
    color_periods: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    if len(active_shape) != len(color_periods):
        raise ValueError("active_shape and color_periods must have the same rank.")

    active_cells = int(np.prod(active_shape))
    groups: dict[tuple[int, ...], list[int]] = {}
    for variable_block in range(field_count):
        block_offset = variable_block * active_cells
        for cell_index in np.ndindex(active_shape):
            color_key = (variable_block,) + tuple(
                coordinate % period for coordinate, period in zip(cell_index, color_periods, strict=True)
            )
            flattened = np.ravel_multi_index(cell_index, active_shape)
            groups.setdefault(color_key, []).append(block_offset + flattened)
    return tuple(tuple(groups[key]) for key in sorted(groups))


def build_sparse_difference_quotient_jacobian(
    residual,
    state: np.ndarray,
    *,
    base_residual: np.ndarray | None = None,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Sparse difference-quotient Jacobian construction requires scipy.") from exc

    state_array = np.asarray(state, dtype=np.float64)
    residual0 = (
        np.asarray(base_residual, dtype=np.float64)
        if base_residual is not None
        else np.asarray(residual(state_array), dtype=np.float64)
    )
    sparsity_csc = sparsity.tocsc()

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []

    for group in color_groups:
        perturbation = np.zeros_like(state_array)
        group_steps: list[tuple[int, float]] = []
        for column in group:
            step = difference_quotient_step_size(state_array[column])
            perturbation[column] = step
            group_steps.append((column, step))

        perturbed_residual = np.asarray(residual(state_array + perturbation), dtype=np.float64)
        delta = perturbed_residual - residual0
        for column, step in group_steps:
            rows = sparsity_csc.indices[sparsity_csc.indptr[column] : sparsity_csc.indptr[column + 1]]
            row_indices.extend(rows.tolist())
            col_indices.extend([column] * len(rows))
            data.extend((delta[rows] / step).tolist())

    return coo_matrix((data, (row_indices, col_indices)), shape=sparsity.shape).tocsr()


def backward_euler_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    rhs: np.ndarray,
    *,
    timestep: float,
) -> np.ndarray:
    return (
        np.asarray(packed_state, dtype=np.float64)
        - np.asarray(previous_packed_state, dtype=np.float64)
        - float(timestep) * np.asarray(rhs, dtype=np.float64)
    )


def bdf2_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    previous_previous_packed_state: np.ndarray,
    rhs: np.ndarray,
    *,
    timestep: float,
) -> np.ndarray:
    return (
        np.asarray(packed_state, dtype=np.float64)
        - (4.0 / 3.0) * np.asarray(previous_packed_state, dtype=np.float64)
        + (1.0 / 3.0) * np.asarray(previous_previous_packed_state, dtype=np.float64)
        - (2.0 / 3.0) * float(timestep) * np.asarray(rhs, dtype=np.float64)
    )


def solve_sparse_newton_system(
    residual,
    initial_state: np.ndarray,
    *,
    active_shape: tuple[int, ...],
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    residual_tolerance: float,
    step_tolerance: float,
    max_nonlinear_iterations: int,
    linear_restart: int,
    linear_maxiter: int,
    linear_rtol: float,
    prefer_direct_linear_solve: bool = False,
) -> tuple[np.ndarray, ImplicitStepInfo]:
    try:
        from scipy.optimize import NoConvergence, newton_krylov
        from scipy.sparse.linalg import gmres, spsolve
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Sparse implicit stepping requires scipy.") from exc

    state = np.asarray(initial_state, dtype=np.float64).copy()
    total_linear_iterations = 0
    best_state = np.array(state, copy=True)
    best_residual_inf_norm = np.inf

    for nonlinear_iteration in range(1, int(max_nonlinear_iterations) + 1):
        residual_value = np.asarray(residual(state), dtype=np.float64)
        residual_inf_norm = float(np.max(np.abs(residual_value)))
        if residual_inf_norm < best_residual_inf_norm:
            best_state = np.array(state, copy=True)
            best_residual_inf_norm = residual_inf_norm
        if residual_inf_norm < float(residual_tolerance):
            return state, ImplicitStepInfo(
                residual_inf_norm=residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration - 1,
                linear_iterations=total_linear_iterations,
            )

        jacobian = build_sparse_difference_quotient_jacobian(
            residual,
            state,
            base_residual=residual_value,
            sparsity=sparsity,
            color_groups=color_groups,
        )
        linear_iterations = 0
        if prefer_direct_linear_solve:
            update = spsolve(jacobian.tocsc(), -residual_value)
            total_linear_iterations += 1
        else:
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

        update = np.asarray(update, dtype=np.float64)
        accepted = False
        step_scale = 1.0
        candidate_state = np.array(state, copy=True)
        candidate_residual_inf_norm = residual_inf_norm
        while step_scale >= 1.0 / 64.0:
            trial_state = state + step_scale * update
            if not np.all(np.isfinite(trial_state)):
                step_scale *= 0.5
                continue
            trial_residual = np.asarray(residual(trial_state), dtype=np.float64)
            trial_residual_inf_norm = float(np.max(np.abs(trial_residual)))
            if np.isfinite(trial_residual_inf_norm) and trial_residual_inf_norm <= residual_inf_norm:
                candidate_state = trial_state
                candidate_residual_inf_norm = trial_residual_inf_norm
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            break

        state = candidate_state
        if candidate_residual_inf_norm < best_residual_inf_norm:
            best_state = np.array(state, copy=True)
            best_residual_inf_norm = candidate_residual_inf_norm
        if float(np.max(np.abs(step_scale * update))) < float(step_tolerance):
            return state, ImplicitStepInfo(
                residual_inf_norm=candidate_residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration,
                linear_iterations=total_linear_iterations,
            )
        if candidate_residual_inf_norm < float(residual_tolerance):
            return state, ImplicitStepInfo(
                residual_inf_norm=candidate_residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration,
                linear_iterations=total_linear_iterations,
            )

    try:
        solved = newton_krylov(
            residual,
            np.asarray(best_state, dtype=np.float64),
            f_tol=float(residual_tolerance),
            maxiter=max(4 * int(max_nonlinear_iterations), 25),
            method="lgmres",
            verbose=0,
        )
    except NoConvergence as exc:
        solved = np.asarray(exc.args[0], dtype=np.float64)
    residual_value = np.asarray(residual(np.asarray(solved, dtype=np.float64)), dtype=np.float64)
    residual_inf_norm = float(np.max(np.abs(residual_value)))
    return np.asarray(solved, dtype=np.float64), ImplicitStepInfo(
        residual_inf_norm=residual_inf_norm,
        active_shape=active_shape,
        nonlinear_iterations=int(max_nonlinear_iterations),
        linear_iterations=total_linear_iterations,
    )


def solve_matrix_free_newton_system(
    residual,
    initial_state: np.ndarray,
    *,
    active_shape: tuple[int, ...],
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> tuple[np.ndarray, ImplicitStepInfo]:
    try:
        from scipy.optimize import newton_krylov
    except ImportError as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError("Matrix-free implicit stepping requires scipy.") from exc

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
    return np.asarray(solved, dtype=np.float64), ImplicitStepInfo(
        residual_inf_norm=float(np.max(np.abs(residual_value))),
        active_shape=active_shape,
        nonlinear_iterations=iteration_budget,
        linear_iterations=iteration_budget,
    )
