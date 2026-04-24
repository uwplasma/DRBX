from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from threading import Lock
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ImplicitStepInfo:
    residual_inf_norm: float
    active_shape: tuple[int, ...]
    nonlinear_iterations: int
    linear_iterations: int
    residual_evaluation_count: int = 0
    residual_evaluation_seconds: float = 0.0
    jacobian_refresh_count: int = 0
    jacobian_assembly_seconds: float = 0.0
    linear_solve_seconds: float = 0.0
    line_search_seconds: float = 0.0
    fallback_used: bool = False


@dataclass(frozen=True)
class SparseDifferenceQuotientGroup:
    columns: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    starts: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class SparseDifferenceQuotientPlan:
    shape: tuple[int, int]
    groups: tuple[SparseDifferenceQuotientGroup, ...]
    nnz: int


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


def prepare_sparse_difference_quotient_plan(
    *,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    sparsity_csc=None,
) -> SparseDifferenceQuotientPlan:
    """Precompute CSC row/column slices for repeated colored FD Jacobians."""

    sparsity_csc = sparsity.tocsc() if sparsity_csc is None else sparsity_csc
    groups: list[SparseDifferenceQuotientGroup] = []
    for group in color_groups:
        columns = np.asarray(group, dtype=np.int32)
        counts = np.asarray(
            [sparsity_csc.indptr[column + 1] - sparsity_csc.indptr[column] for column in columns],
            dtype=np.int32,
        )
        starts = np.empty_like(counts)
        if counts.size:
            starts[0] = 0
            starts[1:] = np.cumsum(counts[:-1], dtype=np.int32)
        rows = np.empty(int(np.sum(counts)), dtype=np.int32)
        cols = np.empty_like(rows)
        for index, column in enumerate(columns):
            start = int(starts[index])
            stop = start + int(counts[index])
            rows[start:stop] = sparsity_csc.indices[sparsity_csc.indptr[column] : sparsity_csc.indptr[column + 1]]
            cols[start:stop] = int(column)
        groups.append(
            SparseDifferenceQuotientGroup(
                columns=columns,
                rows=rows,
                cols=cols,
                starts=starts,
                counts=counts,
            )
        )
    return SparseDifferenceQuotientPlan(
        shape=tuple(int(axis) for axis in sparsity.shape),
        groups=tuple(groups),
        nnz=int(sparsity_csc.nnz),
    )


def build_sparse_difference_quotient_jacobian(
    residual,
    state: np.ndarray,
    *,
    base_residual: np.ndarray | None = None,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    sparsity_csc=None,
    difference_plan: SparseDifferenceQuotientPlan | None = None,
    parallel_workers: int = 1,
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
    sparsity_csc = sparsity.tocsc() if sparsity_csc is None else sparsity_csc
    plan = (
        difference_plan
        if difference_plan is not None
        else prepare_sparse_difference_quotient_plan(sparsity=sparsity, color_groups=color_groups, sparsity_csc=sparsity_csc)
    )

    nnz = int(plan.nnz)
    row_indices = np.empty(nnz, dtype=np.int32)
    col_indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    offset = 0

    def _evaluate_group(group: SparseDifferenceQuotientGroup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perturbed_state = state_array.copy()
        steps = np.sqrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(state_array[group.columns]))
        perturbed_state[group.columns] += steps
        perturbed_residual = np.asarray(residual(perturbed_state), dtype=np.float64)
        delta = perturbed_residual - residual0
        group_data = np.empty_like(group.rows, dtype=np.float64)
        for index, step in enumerate(steps):
            start = int(group.starts[index])
            stop = start + int(group.counts[index])
            rows = group.rows[start:stop]
            group_data[start:stop] = delta[rows] / float(step)
        return group.rows, group.cols, group_data

    if parallel_workers > 1 and len(plan.groups) > 1:
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            group_results = tuple(executor.map(_evaluate_group, plan.groups))
    else:
        group_results = tuple(_evaluate_group(group) for group in plan.groups)

    for group_rows, group_cols, group_data in group_results:
        count = len(group_rows)
        row_indices[offset : offset + count] = group_rows
        col_indices[offset : offset + count] = group_cols
        data[offset : offset + count] = group_data
        offset += count

    return coo_matrix((data[:offset], (row_indices[:offset], col_indices[:offset])), shape=plan.shape).tocsr()


def build_sparse_jvp_jacobian(
    residual,
    state,
    *,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    sparsity_csc=None,
    difference_plan: SparseDifferenceQuotientPlan | None = None,
    batch_size: int | None = None,
):
    """Build a sparse Jacobian from grouped JAX JVPs.

    The coloring contract is the same as for the finite-difference builder:
    columns in one color group must have disjoint row support in ``sparsity``.
    Batched linearized pushes then fill those disjoint column entries without
    finite-difference perturbations. ``batch_size`` bounds the number of color
    groups pushed through ``jax.vmap`` at once; leaving it unset uses one batch
    for all groups.
    """

    try:
        import jax
        import jax.numpy as jnp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only when optional deps are unavailable
        raise ImportError("Sparse JVP Jacobian construction requires jax and scipy.") from exc

    state_array = jnp.asarray(state, dtype=jnp.float64)
    state_shape = tuple(state_array.shape)
    state_size = int(state_array.size)
    sparsity_csc = sparsity.tocsc() if sparsity_csc is None else sparsity_csc
    plan = (
        difference_plan
        if difference_plan is not None
        else prepare_sparse_difference_quotient_plan(sparsity=sparsity, color_groups=color_groups, sparsity_csc=sparsity_csc)
    )
    _, linear_map = jax.linearize(residual, state_array)

    nnz = int(plan.nnz)
    row_indices = np.empty(nnz, dtype=np.int32)
    col_indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    offset = 0
    group_count = len(plan.groups)
    if group_count == 0:
        return coo_matrix((data, (row_indices, col_indices)), shape=plan.shape).tocsr()
    resolved_batch_size = group_count if batch_size is None else max(1, int(batch_size))
    vmapped_linear_map = jax.vmap(linear_map)
    for batch_start in range(0, group_count, resolved_batch_size):
        batch_groups = plan.groups[batch_start : batch_start + resolved_batch_size]
        directions_flat = np.zeros((len(batch_groups), state_size), dtype=np.float64)
        for batch_index, group in enumerate(batch_groups):
            directions_flat[batch_index, group.columns] = 1.0
        directions = jnp.asarray(
            directions_flat.reshape((len(batch_groups), *state_shape)),
            dtype=state_array.dtype,
        )
        pushed_batch = np.asarray(
            vmapped_linear_map(directions),
            dtype=np.float64,
        ).reshape(len(batch_groups), -1)
        for batch_index, group in enumerate(batch_groups):
            pushed = pushed_batch[batch_index]
            group_data = pushed[group.rows]
            count = len(group.rows)
            row_indices[offset : offset + count] = group.rows
            col_indices[offset : offset + count] = group.cols
            data[offset : offset + count] = group_data
            offset += count

    return coo_matrix((data[:offset], (row_indices[:offset], col_indices[:offset])), shape=plan.shape).tocsr()


def backward_euler_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    rhs: np.ndarray,
    *,
    timestep: float,
) -> np.ndarray:
    packed_state_array = np.asarray(packed_state, dtype=np.float64)
    previous_array = np.asarray(previous_packed_state, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    return packed_state_array - previous_array - float(timestep) * rhs_array


def bdf2_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    previous_previous_packed_state: np.ndarray,
    rhs: np.ndarray,
    *,
    timestep: float,
) -> np.ndarray:
    packed_state_array = np.asarray(packed_state, dtype=np.float64)
    previous_array = np.asarray(previous_packed_state, dtype=np.float64)
    previous_previous_array = np.asarray(previous_previous_packed_state, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    return (
        packed_state_array
        - (4.0 / 3.0) * previous_array
        + (1.0 / 3.0) * previous_previous_array
        - (2.0 / 3.0) * float(timestep) * rhs_array
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
    jacobian_refresh_frequency: int = 1,
    jacobian_parallel_workers: int | None = None,
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
    residual_evaluation_count = 0
    residual_evaluation_seconds = 0.0
    jacobian_refresh_count = 0
    jacobian_assembly_seconds = 0.0
    linear_solve_seconds = 0.0
    line_search_seconds = 0.0
    fallback_used = False
    residual_counter_lock = Lock()

    def evaluate_residual(candidate_state: np.ndarray) -> np.ndarray:
        nonlocal residual_evaluation_count, residual_evaluation_seconds
        started_at = perf_counter()
        value = np.asarray(residual(candidate_state), dtype=np.float64)
        elapsed = perf_counter() - started_at
        with residual_counter_lock:
            residual_evaluation_count += 1
            residual_evaluation_seconds += elapsed
        return value

    def build_info(
        *,
        residual_inf_norm: float,
        nonlinear_iterations: int,
    ) -> ImplicitStepInfo:
        return ImplicitStepInfo(
            residual_inf_norm=residual_inf_norm,
            active_shape=active_shape,
            nonlinear_iterations=nonlinear_iterations,
            linear_iterations=total_linear_iterations,
            residual_evaluation_count=residual_evaluation_count,
            residual_evaluation_seconds=residual_evaluation_seconds,
            jacobian_refresh_count=jacobian_refresh_count,
            jacobian_assembly_seconds=jacobian_assembly_seconds,
            linear_solve_seconds=linear_solve_seconds,
            line_search_seconds=line_search_seconds,
            fallback_used=fallback_used,
        )

    refresh_frequency = max(1, int(jacobian_refresh_frequency))
    if jacobian_parallel_workers is None:
        env_value = os.environ.get("JAX_DRB_FD_JACOBIAN_THREADS")
        if env_value is not None:
            jacobian_parallel_workers = max(1, int(env_value))
        else:
            cpu_count = os.cpu_count() or 1
            heavy_problem = initial_state.size >= 4000 and len(color_groups) >= 8
            jacobian_parallel_workers = min(4, cpu_count) if heavy_problem else 1
    jacobian = None
    jacobian_csc = None
    sparsity_csc = sparsity.tocsc()
    difference_plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity,
        color_groups=color_groups,
        sparsity_csc=sparsity_csc,
    )

    for nonlinear_iteration in range(1, int(max_nonlinear_iterations) + 1):
        residual_value = evaluate_residual(state)
        residual_inf_norm = float(np.max(np.abs(residual_value)))
        if residual_inf_norm < best_residual_inf_norm:
            best_state = np.array(state, copy=True)
            best_residual_inf_norm = residual_inf_norm
        if residual_inf_norm < float(residual_tolerance):
            return state, build_info(
                residual_inf_norm=residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration - 1,
            )

        if jacobian is None or nonlinear_iteration == 1 or ((nonlinear_iteration - 1) % refresh_frequency == 0):
            jacobian_started_at = perf_counter()
            jacobian = build_sparse_difference_quotient_jacobian(
                evaluate_residual,
                state,
                base_residual=residual_value,
                sparsity=sparsity,
                color_groups=color_groups,
                sparsity_csc=sparsity_csc,
                difference_plan=difference_plan,
                parallel_workers=int(jacobian_parallel_workers),
            )
            jacobian_csc = jacobian.tocsc()
            jacobian_assembly_seconds += perf_counter() - jacobian_started_at
            jacobian_refresh_count += 1
        linear_iterations = 0
        linear_solve_started_at = perf_counter()
        if prefer_direct_linear_solve:
            update = spsolve(jacobian_csc, -residual_value)
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
                update = spsolve(jacobian_csc, -residual_value)
                total_linear_iterations += 1
        linear_solve_seconds += perf_counter() - linear_solve_started_at

        update = np.asarray(update, dtype=np.float64)
        accepted = False
        step_scale = 1.0
        candidate_state = np.array(state, copy=True)
        candidate_residual_inf_norm = residual_inf_norm
        line_search_started_at = perf_counter()
        while step_scale >= 1.0 / 64.0:
            trial_state = state + step_scale * update
            if not np.all(np.isfinite(trial_state)):
                step_scale *= 0.5
                continue
            trial_residual = evaluate_residual(trial_state)
            trial_residual_inf_norm = float(np.max(np.abs(trial_residual)))
            if np.isfinite(trial_residual_inf_norm) and trial_residual_inf_norm <= residual_inf_norm:
                candidate_state = trial_state
                candidate_residual_inf_norm = trial_residual_inf_norm
                accepted = True
                break
            step_scale *= 0.5
        line_search_seconds += perf_counter() - line_search_started_at

        if not accepted:
            jacobian = None
            break

        state = candidate_state
        if candidate_residual_inf_norm < best_residual_inf_norm:
            best_state = np.array(state, copy=True)
            best_residual_inf_norm = candidate_residual_inf_norm
        if float(np.max(np.abs(step_scale * update))) < float(step_tolerance):
            return state, build_info(
                residual_inf_norm=candidate_residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration,
            )
        if candidate_residual_inf_norm < float(residual_tolerance):
            return state, build_info(
                residual_inf_norm=candidate_residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration,
            )

    fallback_used = True
    fallback_started_at = perf_counter()
    try:
        solved = newton_krylov(
            evaluate_residual,
            np.asarray(best_state, dtype=np.float64),
            f_tol=float(residual_tolerance),
            maxiter=max(4 * int(max_nonlinear_iterations), 25),
            method="lgmres",
            verbose=0,
        )
    except NoConvergence as exc:
        solved = np.asarray(exc.args[0], dtype=np.float64)
    linear_solve_seconds += perf_counter() - fallback_started_at
    residual_value = evaluate_residual(np.asarray(solved, dtype=np.float64))
    residual_inf_norm = float(np.max(np.abs(residual_value)))
    return np.asarray(solved, dtype=np.float64), build_info(
        residual_inf_norm=residual_inf_norm,
        nonlinear_iterations=int(max_nonlinear_iterations),
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

    residual_evaluation_count = 0
    residual_evaluation_seconds = 0.0

    def evaluate_residual(candidate_state: np.ndarray) -> np.ndarray:
        nonlocal residual_evaluation_count, residual_evaluation_seconds
        started_at = perf_counter()
        value = np.asarray(residual(candidate_state), dtype=np.float64)
        residual_evaluation_seconds += perf_counter() - started_at
        residual_evaluation_count += 1
        return value

    iteration_budget = max(int(max_nonlinear_iterations), 25)
    solve_started_at = perf_counter()
    solved = newton_krylov(
        evaluate_residual,
        np.asarray(initial_state, dtype=np.float64),
        f_tol=float(residual_tolerance),
        maxiter=iteration_budget,
        method="lgmres",
        verbose=0,
    )
    solve_seconds = perf_counter() - solve_started_at
    residual_value = evaluate_residual(np.asarray(solved, dtype=np.float64))
    return np.asarray(solved, dtype=np.float64), ImplicitStepInfo(
        residual_inf_norm=float(np.max(np.abs(residual_value))),
        active_shape=active_shape,
        nonlinear_iterations=iteration_budget,
        linear_iterations=iteration_budget,
        residual_evaluation_count=residual_evaluation_count,
        residual_evaluation_seconds=residual_evaluation_seconds,
        linear_solve_seconds=solve_seconds,
    )


def solve_jax_linearized_newton_system(
    residual,
    initial_state: np.ndarray,
    *,
    active_shape: tuple[int, ...],
    residual_tolerance: float,
    step_tolerance: float,
    max_nonlinear_iterations: int,
    linear_restart: int = 20,
    linear_maxiter: int = 20,
) -> tuple[np.ndarray, ImplicitStepInfo]:
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import gmres
    except ImportError as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX linearized implicit stepping requires jax.") from exc

    state = jnp.asarray(initial_state, dtype=jnp.float64)
    total_linear_iterations = 0
    residual_evaluation_count = 0
    residual_evaluation_seconds = 0.0
    jacobian_refresh_count = 0
    jacobian_assembly_seconds = 0.0
    linear_solve_seconds = 0.0
    line_search_seconds = 0.0

    def _block(value):
        return jax.block_until_ready(value)

    for nonlinear_iteration in range(1, int(max_nonlinear_iterations) + 1):
        linearize_started_at = perf_counter()
        residual_value, linear_map = jax.linearize(residual, state)
        residual_value = _block(residual_value)
        elapsed = perf_counter() - linearize_started_at
        residual_evaluation_count += 1
        residual_evaluation_seconds += elapsed
        jacobian_refresh_count += 1
        jacobian_assembly_seconds += elapsed
        residual_inf_norm = float(jnp.max(jnp.abs(residual_value)))
        if residual_inf_norm < float(residual_tolerance):
            return np.asarray(state, dtype=np.float64), ImplicitStepInfo(
                residual_inf_norm=residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration - 1,
                linear_iterations=total_linear_iterations,
                residual_evaluation_count=residual_evaluation_count,
                residual_evaluation_seconds=residual_evaluation_seconds,
                jacobian_refresh_count=jacobian_refresh_count,
                jacobian_assembly_seconds=jacobian_assembly_seconds,
                linear_solve_seconds=linear_solve_seconds,
                line_search_seconds=line_search_seconds,
            )

        linear_solve_started_at = perf_counter()
        update, _ = gmres(
            linear_map,
            -residual_value,
            tol=float(residual_tolerance),
            atol=0.0,
            restart=int(linear_restart),
            maxiter=int(linear_maxiter),
        )
        update = _block(update)
        linear_solve_seconds += perf_counter() - linear_solve_started_at
        total_linear_iterations += int(linear_restart) * int(linear_maxiter)
        update = jnp.asarray(update, dtype=jnp.float64)

        accepted = False
        step_scale = 1.0
        candidate_state = state
        candidate_residual_inf_norm = residual_inf_norm
        line_search_started_at = perf_counter()
        while step_scale >= 1.0 / 64.0:
            trial_state = state + step_scale * update
            residual_started_at = perf_counter()
            trial_residual = residual(trial_state)
            trial_residual = _block(trial_residual)
            residual_evaluation_count += 1
            residual_evaluation_seconds += perf_counter() - residual_started_at
            trial_residual_inf_norm = float(jnp.max(jnp.abs(trial_residual)))
            if np.isfinite(trial_residual_inf_norm) and trial_residual_inf_norm <= residual_inf_norm:
                candidate_state = trial_state
                candidate_residual_inf_norm = trial_residual_inf_norm
                accepted = True
                break
            step_scale *= 0.5
        line_search_seconds += perf_counter() - line_search_started_at

        if not accepted:
            break

        state = candidate_state
        if float(jnp.max(jnp.abs(step_scale * update))) < float(step_tolerance):
            return np.asarray(state, dtype=np.float64), ImplicitStepInfo(
                residual_inf_norm=candidate_residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration,
                linear_iterations=total_linear_iterations,
                residual_evaluation_count=residual_evaluation_count,
                residual_evaluation_seconds=residual_evaluation_seconds,
                jacobian_refresh_count=jacobian_refresh_count,
                jacobian_assembly_seconds=jacobian_assembly_seconds,
                linear_solve_seconds=linear_solve_seconds,
                line_search_seconds=line_search_seconds,
            )
        if candidate_residual_inf_norm < float(residual_tolerance):
            return np.asarray(state, dtype=np.float64), ImplicitStepInfo(
                residual_inf_norm=candidate_residual_inf_norm,
                active_shape=active_shape,
                nonlinear_iterations=nonlinear_iteration,
                linear_iterations=total_linear_iterations,
                residual_evaluation_count=residual_evaluation_count,
                residual_evaluation_seconds=residual_evaluation_seconds,
                jacobian_refresh_count=jacobian_refresh_count,
                jacobian_assembly_seconds=jacobian_assembly_seconds,
                linear_solve_seconds=linear_solve_seconds,
                line_search_seconds=line_search_seconds,
            )

    residual_started_at = perf_counter()
    final_residual = residual(state)
    final_residual = _block(final_residual)
    residual_evaluation_count += 1
    residual_evaluation_seconds += perf_counter() - residual_started_at
    return np.asarray(state, dtype=np.float64), ImplicitStepInfo(
        residual_inf_norm=float(jnp.max(jnp.abs(final_residual))),
        active_shape=active_shape,
        nonlinear_iterations=int(max_nonlinear_iterations),
        linear_iterations=total_linear_iterations,
        residual_evaluation_count=residual_evaluation_count,
        residual_evaluation_seconds=residual_evaluation_seconds,
        jacobian_refresh_count=jacobian_refresh_count,
        jacobian_assembly_seconds=jacobian_assembly_seconds,
        linear_solve_seconds=linear_solve_seconds,
        line_search_seconds=line_search_seconds,
    )
