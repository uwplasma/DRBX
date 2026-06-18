from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from threading import Lock
from time import perf_counter
from typing import Callable

import numpy as np


def _uses_jax_backend(*values: object) -> bool:
    for value in values:
        if value is None:
            continue
        module = type(value).__module__
        if (
            hasattr(value, "aval")
            or module.startswith("jax")
            or module.startswith("jaxlib")
        ):
            return True
    return False


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
    line_search_trial_count: int = 0
    line_search_last_step_scale: float | None = None
    line_search_initial_step_scale: float = 1.0
    fallback_used: bool = False
    jacobian_mode: str = "fd"
    converged: bool | None = None
    linear_solver_backend: str | None = None
    linear_solver_tolerance: float | None = None
    linear_solver_status: int | float | str | None = None
    linear_solver_success: bool | None = None
    linear_solver_reported_iterations: int | None = None
    linear_solver_solve_method: str | None = None
    linear_operator_call_count: int = 0
    linear_operator_dispatch_seconds: float = 0.0
    linear_preconditioner: str | None = None
    linear_preconditioner_build_seconds: float = 0.0
    linear_preconditioner_build_count: int = 0
    linear_preconditioner_apply_count: int = 0
    linear_preconditioner_apply_seconds: float = 0.0
    jvp_direction_batch_count: int = 0
    jvp_direction_build_seconds: float = 0.0
    jvp_jacobian_total_seconds: float = 0.0
    jvp_jacobian_linearize_seconds: float = 0.0
    jvp_jacobian_tangent_build_seconds: float = 0.0
    jvp_jacobian_push_seconds: float = 0.0
    jvp_jacobian_device_execute_seconds: float = 0.0
    jvp_jacobian_host_transfer_seconds: float = 0.0
    jvp_jacobian_sparse_assembly_seconds: float = 0.0
    jvp_jacobian_batch_count: int = 0
    jvp_jacobian_prebuilt_direction_batch_uses: int = 0
    jvp_direction_workspace_reuses: int = 0
    residual_jitted: bool = False
    check_initial_residual: bool = True
    initial_residual_mode: str = "residual"


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


@dataclass(frozen=True)
class SparseJvpDirectionBatch:
    groups: tuple[SparseDifferenceQuotientGroup, ...]
    directions: object
    gather_batch_indices: object
    gather_rows: object
    gather_counts: tuple[int, ...]
    gather_batch_indices_device: object = None
    gather_rows_device: object = None


@dataclass(frozen=True)
class SparseJvpWorkspace:
    """Static sparse-JVP plan reusable across same-layout Newton solves."""

    sparsity_shape: tuple[int, int]
    state_shape: tuple[int, ...]
    dtype: str
    batch_size: int | None
    sparsity: object
    color_groups: tuple[tuple[int, ...], ...]
    sparsity_csc: object
    difference_plan: SparseDifferenceQuotientPlan
    direction_batches: tuple[SparseJvpDirectionBatch, ...]


@dataclass(frozen=True)
class JaxLinearizedUpdateResult:
    update: object
    backend: str
    status: object = None
    success: bool | None = None
    reported_iterations: object = None


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
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when scipy is unavailable
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
                for axis, (coordinate, radius, raw_offset) in enumerate(
                    zip(cell_index, radii, delta, strict=True)
                ):
                    offset = raw_offset - radius
                    neighbor_coordinate = coordinate + offset
                    if axis in periodic_axis_set:
                        neighbor_coordinate %= active_shape[axis]
                    elif not (0 <= neighbor_coordinate < active_shape[axis]):
                        valid = False
                        break
                    neighbor_index.append(neighbor_coordinate)
                if valid:
                    neighbors.add(
                        np.ravel_multi_index(tuple(neighbor_index), active_shape)
                    )
            for variable_block in range(field_count):
                col_offset = variable_block * active_cells
                for neighbor in neighbors:
                    row_indices.append(row)
                    col_indices.append(col_offset + neighbor)

    data = np.ones(len(row_indices), dtype=bool)
    return coo_matrix(
        (data, (row_indices, col_indices)), shape=(total_size, total_size)
    ).tocsr()


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
                coordinate % period
                for coordinate, period in zip(cell_index, color_periods, strict=True)
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
            [
                sparsity_csc.indptr[column + 1] - sparsity_csc.indptr[column]
                for column in columns
            ],
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
            rows[start:stop] = sparsity_csc.indices[
                sparsity_csc.indptr[column] : sparsity_csc.indptr[column + 1]
            ]
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


def prepare_sparse_jvp_direction_batches(
    *,
    difference_plan: SparseDifferenceQuotientPlan,
    state_shape: tuple[int, ...],
    dtype=np.float64,
    batch_size: int | None = None,
) -> tuple[SparseJvpDirectionBatch, ...]:
    """Prebuild grouped tangent directions for repeated sparse JVP Jacobians."""

    try:
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("Sparse JVP direction construction requires jax.") from exc

    groups = difference_plan.groups
    group_count = len(groups)
    if group_count == 0:
        return ()
    state_size = int(np.prod(state_shape, dtype=np.int64))
    resolved_batch_size = group_count if batch_size is None else max(1, int(batch_size))
    batches: list[SparseJvpDirectionBatch] = []
    for batch_start in range(0, group_count, resolved_batch_size):
        batch_groups = groups[batch_start : batch_start + resolved_batch_size]
        directions_flat = np.zeros((len(batch_groups), state_size), dtype=np.float64)
        for batch_index, group in enumerate(batch_groups):
            directions_flat[batch_index, group.columns] = 1.0
        directions = jnp.asarray(
            directions_flat.reshape((len(batch_groups), *state_shape)),
            dtype=dtype,
        )
        gather_batch_indices, gather_rows, gather_counts = (
            _build_sparse_jvp_batch_gather_plan(list(batch_groups))
        )
        batches.append(
            SparseJvpDirectionBatch(
                groups=tuple(batch_groups),
                directions=directions,
                gather_batch_indices=gather_batch_indices,
                gather_rows=gather_rows,
                gather_counts=gather_counts,
                gather_batch_indices_device=jnp.asarray(
                    gather_batch_indices, dtype=jnp.int32
                ),
                gather_rows_device=jnp.asarray(gather_rows, dtype=jnp.int32),
            )
        )
    return tuple(batches)


def _build_sparse_jvp_batch_gather_plan(
    batch_groups: list[SparseDifferenceQuotientGroup],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    batch_indices: list[int] = []
    rows: list[int] = []
    counts: list[int] = []
    for batch_index, group in enumerate(batch_groups):
        group_rows = np.asarray(group.rows, dtype=np.int32)
        count = int(group_rows.size)
        counts.append(count)
        if count:
            batch_indices.extend([batch_index] * count)
            rows.extend(int(row) for row in group_rows)
    return (
        np.asarray(batch_indices, dtype=np.int32),
        np.asarray(rows, dtype=np.int32),
        tuple(counts),
    )


def prepare_sparse_jvp_workspace(
    *,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    state_shape: tuple[int, ...],
    dtype=np.float64,
    batch_size: int | None = None,
    sparsity_csc=None,
    difference_plan: SparseDifferenceQuotientPlan | None = None,
) -> SparseJvpWorkspace:
    """Precompute static sparse-JVP data for repeated compatible solves."""

    resolved_state_shape = tuple(int(axis) for axis in state_shape)
    resolved_batch_size = None if batch_size is None else max(1, int(batch_size))
    resolved_dtype = np.dtype(dtype).str
    sparsity_csc = sparsity.tocsc() if sparsity_csc is None else sparsity_csc
    plan = (
        difference_plan
        if difference_plan is not None
        else prepare_sparse_difference_quotient_plan(
            sparsity=sparsity,
            color_groups=color_groups,
            sparsity_csc=sparsity_csc,
        )
    )
    direction_batches = prepare_sparse_jvp_direction_batches(
        difference_plan=plan,
        state_shape=resolved_state_shape,
        dtype=np.dtype(dtype),
        batch_size=resolved_batch_size,
    )
    return SparseJvpWorkspace(
        sparsity_shape=tuple(int(axis) for axis in sparsity.shape),
        state_shape=resolved_state_shape,
        dtype=resolved_dtype,
        batch_size=resolved_batch_size,
        sparsity=sparsity,
        color_groups=tuple(
            tuple(int(column) for column in group) for group in color_groups
        ),
        sparsity_csc=sparsity_csc,
        difference_plan=plan,
        direction_batches=direction_batches,
    )


def _sparse_jvp_workspace_is_compatible(
    workspace: SparseJvpWorkspace | None,
    *,
    sparsity,
    state_shape: tuple[int, ...],
    dtype,
    batch_size: int | None,
) -> bool:
    if workspace is None:
        return False
    resolved_batch_size = None if batch_size is None else max(1, int(batch_size))
    return (
        tuple(int(axis) for axis in workspace.sparsity_shape)
        == tuple(int(axis) for axis in sparsity.shape)
        and tuple(int(axis) for axis in workspace.state_shape)
        == tuple(int(axis) for axis in state_shape)
        and workspace.dtype == np.dtype(dtype).str
        and workspace.batch_size == resolved_batch_size
        and int(workspace.difference_plan.nnz)
        == int(getattr(sparsity, "nnz", workspace.difference_plan.nnz))
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
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when scipy is unavailable
        raise ImportError(
            "Sparse difference-quotient Jacobian construction requires scipy."
        ) from exc

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
        else prepare_sparse_difference_quotient_plan(
            sparsity=sparsity, color_groups=color_groups, sparsity_csc=sparsity_csc
        )
    )

    nnz = int(plan.nnz)
    row_indices = np.empty(nnz, dtype=np.int32)
    col_indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    offset = 0

    def _evaluate_group(
        group: SparseDifferenceQuotientGroup,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        perturbed_state = state_array.copy()
        steps = np.sqrt(np.finfo(np.float64).eps) * np.maximum(
            1.0, np.abs(state_array[group.columns])
        )
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

    return coo_matrix(
        (data[:offset], (row_indices[:offset], col_indices[:offset])), shape=plan.shape
    ).tocsr()


def build_sparse_jvp_jacobian(
    residual,
    state,
    *,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    sparsity_csc=None,
    difference_plan: SparseDifferenceQuotientPlan | None = None,
    batch_size: int | None = None,
    direction_batches: tuple[SparseJvpDirectionBatch, ...] | None = None,
    timing_callback: Callable[[dict[str, float | int]], None] | None = None,
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
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when optional deps are unavailable
        raise ImportError(
            "Sparse JVP Jacobian construction requires jax and scipy."
        ) from exc

    total_started_at = perf_counter()
    linearize_seconds = 0.0
    tangent_build_seconds = 0.0
    push_seconds = 0.0
    device_execute_seconds = 0.0
    host_transfer_seconds = 0.0
    sparse_assembly_seconds = 0.0
    sync_timing = os.environ.get(
        "JAX_DRB_SPARSE_JVP_SYNC_TIMING", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    gather_on_device = os.environ.get(
        "JAX_DRB_SPARSE_JVP_GATHER_ON_DEVICE", "0"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    state_array = jnp.asarray(state, dtype=jnp.float64)
    state_shape = tuple(state_array.shape)
    state_size = int(state_array.size)
    sparsity_csc = sparsity.tocsc() if sparsity_csc is None else sparsity_csc
    plan = (
        difference_plan
        if difference_plan is not None
        else prepare_sparse_difference_quotient_plan(
            sparsity=sparsity, color_groups=color_groups, sparsity_csc=sparsity_csc
        )
    )
    linearize_started_at = perf_counter()
    _, linear_map = jax.linearize(residual, state_array)
    linearize_seconds += perf_counter() - linearize_started_at

    nnz = int(plan.nnz)
    row_indices = np.empty(nnz, dtype=np.int32)
    col_indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    offset = 0
    group_count = len(plan.groups)
    if group_count == 0:
        if timing_callback is not None:
            timing_callback(
                {
                    "total_seconds": float(perf_counter() - total_started_at),
                    "linearize_seconds": float(linearize_seconds),
                    "tangent_build_seconds": 0.0,
                    "push_seconds": 0.0,
                    "device_execute_seconds": 0.0,
                    "host_transfer_seconds": 0.0,
                    "sparse_assembly_seconds": 0.0,
                    "group_count": 0,
                    "batch_count": 0,
                    "state_size": int(state_size),
                    "nnz": int(plan.nnz),
                    "sync_timing": int(sync_timing),
                    "gather_on_device": int(gather_on_device),
                }
            )
        return coo_matrix((data, (row_indices, col_indices)), shape=plan.shape).tocsr()
    vmapped_linear_map = jax.vmap(linear_map)
    prebuilt_directions = direction_batches is not None
    if direction_batches is None:
        tangent_started_at = perf_counter()
        direction_batches = prepare_sparse_jvp_direction_batches(
            difference_plan=plan,
            state_shape=state_shape,
            dtype=state_array.dtype,
            batch_size=batch_size,
        )
        tangent_build_seconds += perf_counter() - tangent_started_at
    batch_count = 0
    for direction_batch in direction_batches:
        batch_groups = direction_batch.groups
        batch_count += 1
        directions = jnp.asarray(direction_batch.directions, dtype=state_array.dtype)
        execute_started_at = perf_counter()
        pushed_device = vmapped_linear_map(directions)
        if sync_timing:
            pushed_device = jax.block_until_ready(pushed_device)
        device_elapsed = perf_counter() - execute_started_at
        transfer_started_at = perf_counter()
        if gather_on_device:
            pushed_flat = jnp.reshape(pushed_device, (len(batch_groups), -1))
            gather_rows = (
                direction_batch.gather_rows_device
                if direction_batch.gather_rows_device is not None
                else jnp.asarray(direction_batch.gather_rows, dtype=jnp.int32)
            )
            gather_batch_indices = (
                direction_batch.gather_batch_indices_device
                if direction_batch.gather_batch_indices_device is not None
                else jnp.asarray(
                    direction_batch.gather_batch_indices, dtype=jnp.int32
                )
            )
            gathered_device = pushed_flat[gather_batch_indices, gather_rows]
            if sync_timing:
                gathered_device = jax.block_until_ready(gathered_device)
            gathered_batch = np.asarray(gathered_device, dtype=np.float64)
            pushed_batch = None
        else:
            pushed_batch = np.asarray(pushed_device, dtype=np.float64).reshape(
                len(batch_groups), -1
            )
            gathered_batch = None
        transfer_elapsed = perf_counter() - transfer_started_at
        device_execute_seconds += device_elapsed
        host_transfer_seconds += transfer_elapsed
        push_seconds += device_elapsed + transfer_elapsed
        assembly_started_at = perf_counter()
        gather_offset = 0
        for batch_index, group in enumerate(batch_groups):
            count = int(direction_batch.gather_counts[batch_index])
            row_indices[offset : offset + count] = group.rows
            col_indices[offset : offset + count] = group.cols
            if gather_on_device:
                data[offset : offset + count] = gathered_batch[
                    gather_offset : gather_offset + count
                ]
                gather_offset += count
            else:
                pushed = pushed_batch[batch_index]
                np.take(pushed, group.rows, out=data[offset : offset + count])
            offset += count
        sparse_assembly_seconds += perf_counter() - assembly_started_at

    if timing_callback is not None:
        timing_callback(
            {
                "total_seconds": float(perf_counter() - total_started_at),
                "linearize_seconds": float(linearize_seconds),
                "tangent_build_seconds": float(tangent_build_seconds),
                "push_seconds": float(push_seconds),
                "device_execute_seconds": float(device_execute_seconds),
                "host_transfer_seconds": float(host_transfer_seconds),
                "sparse_assembly_seconds": float(sparse_assembly_seconds),
                "group_count": int(group_count),
                "batch_count": int(batch_count),
                "state_size": int(state_size),
                "nnz": int(plan.nnz),
                "prebuilt_direction_batches": int(prebuilt_directions),
                "sync_timing": int(sync_timing),
                "gather_on_device": int(gather_on_device),
            }
        )
    return coo_matrix(
        (data[:offset], (row_indices[:offset], col_indices[:offset])), shape=plan.shape
    ).tocsr()


def backward_euler_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    rhs: np.ndarray,
    *,
    timestep: float,
) -> np.ndarray:
    if _uses_jax_backend(packed_state, previous_packed_state, rhs):
        import jax.numpy as jnp

        packed_state_array = jnp.asarray(packed_state, dtype=jnp.float64)
        previous_array = jnp.asarray(previous_packed_state, dtype=jnp.float64)
        rhs_array = jnp.asarray(rhs, dtype=jnp.float64)
        return packed_state_array - previous_array - float(timestep) * rhs_array

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
    previous_timestep: float | None = None,
) -> np.ndarray:
    previous_dt = (
        float(timestep) if previous_timestep is None else float(previous_timestep)
    )
    if previous_dt <= 0.0:
        raise ValueError("previous_timestep must be positive for BDF2 residuals.")
    step_ratio = float(timestep) / previous_dt
    previous_coefficient = ((step_ratio + 1.0) ** 2) / (2.0 * step_ratio + 1.0)
    previous_previous_coefficient = (step_ratio**2) / (2.0 * step_ratio + 1.0)
    rhs_coefficient = float(timestep) * (step_ratio + 1.0) / (2.0 * step_ratio + 1.0)
    if _uses_jax_backend(
        packed_state, previous_packed_state, previous_previous_packed_state, rhs
    ):
        import jax.numpy as jnp

        packed_state_array = jnp.asarray(packed_state, dtype=jnp.float64)
        previous_array = jnp.asarray(previous_packed_state, dtype=jnp.float64)
        previous_previous_array = jnp.asarray(
            previous_previous_packed_state, dtype=jnp.float64
        )
        rhs_array = jnp.asarray(rhs, dtype=jnp.float64)
        return (
            packed_state_array
            - previous_coefficient * previous_array
            + previous_previous_coefficient * previous_previous_array
            - rhs_coefficient * rhs_array
        )

    packed_state_array = np.asarray(packed_state, dtype=np.float64)
    previous_array = np.asarray(previous_packed_state, dtype=np.float64)
    previous_previous_array = np.asarray(
        previous_previous_packed_state, dtype=np.float64
    )
    rhs_array = np.asarray(rhs, dtype=np.float64)
    return (
        packed_state_array
        - previous_coefficient * previous_array
        + previous_previous_coefficient * previous_previous_array
        - rhs_coefficient * rhs_array
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
    jacobian_mode: str = "fd",
    jvp_batch_size: int | None = None,
    sparse_jvp_workspace: SparseJvpWorkspace | None = None,
) -> tuple[np.ndarray, ImplicitStepInfo]:
    try:
        from scipy.optimize import NoConvergence, newton_krylov
        from scipy.sparse.linalg import gmres, spsolve
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when scipy is unavailable
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
    resolved_jacobian_mode = str(jacobian_mode).strip().lower()
    if resolved_jacobian_mode not in {"fd", "jvp"}:
        raise ValueError(f"Unsupported sparse Newton jacobian_mode={jacobian_mode!r}.")
    jvp_direction_batch_count = 0
    jvp_direction_build_seconds = 0.0
    jvp_jacobian_total_seconds = 0.0
    jvp_jacobian_linearize_seconds = 0.0
    jvp_jacobian_tangent_build_seconds = 0.0
    jvp_jacobian_push_seconds = 0.0
    jvp_jacobian_device_execute_seconds = 0.0
    jvp_jacobian_host_transfer_seconds = 0.0
    jvp_jacobian_sparse_assembly_seconds = 0.0
    jvp_jacobian_batch_count = 0
    jvp_jacobian_prebuilt_direction_batch_uses = 0
    jvp_direction_workspace_reuses = 0
    last_linear_solver_backend: str | None = None
    last_linear_solver_status: int | float | str | None = None
    last_linear_solver_success: bool | None = None
    last_linear_solver_reported_iterations: int | None = None

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
        converged: bool | None = None,
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
            jacobian_mode=resolved_jacobian_mode,
            converged=converged,
            linear_solver_backend=last_linear_solver_backend,
            linear_solver_tolerance=float(linear_rtol),
            linear_solver_status=last_linear_solver_status,
            linear_solver_success=last_linear_solver_success,
            linear_solver_reported_iterations=last_linear_solver_reported_iterations,
            jvp_direction_batch_count=jvp_direction_batch_count,
            jvp_direction_build_seconds=jvp_direction_build_seconds,
            jvp_jacobian_total_seconds=jvp_jacobian_total_seconds,
            jvp_jacobian_linearize_seconds=jvp_jacobian_linearize_seconds,
            jvp_jacobian_tangent_build_seconds=jvp_jacobian_tangent_build_seconds,
            jvp_jacobian_push_seconds=jvp_jacobian_push_seconds,
            jvp_jacobian_device_execute_seconds=jvp_jacobian_device_execute_seconds,
            jvp_jacobian_host_transfer_seconds=jvp_jacobian_host_transfer_seconds,
            jvp_jacobian_sparse_assembly_seconds=jvp_jacobian_sparse_assembly_seconds,
            jvp_jacobian_batch_count=jvp_jacobian_batch_count,
            jvp_jacobian_prebuilt_direction_batch_uses=jvp_jacobian_prebuilt_direction_batch_uses,
            jvp_direction_workspace_reuses=jvp_direction_workspace_reuses,
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
    workspace_is_compatible = (
        resolved_jacobian_mode == "jvp"
        and _sparse_jvp_workspace_is_compatible(
            sparse_jvp_workspace,
            sparsity=sparsity,
            state_shape=tuple(state.shape),
            dtype=np.float64,
            batch_size=jvp_batch_size,
        )
    )
    if workspace_is_compatible:
        sparsity_csc = sparse_jvp_workspace.sparsity_csc
        difference_plan = sparse_jvp_workspace.difference_plan
    else:
        sparsity_csc = sparsity.tocsc()
        difference_plan = prepare_sparse_difference_quotient_plan(
            sparsity=sparsity,
            color_groups=color_groups,
            sparsity_csc=sparsity_csc,
        )
    jvp_direction_batches = None
    if resolved_jacobian_mode == "jvp":
        if workspace_is_compatible:
            jvp_direction_batches = sparse_jvp_workspace.direction_batches
            jvp_direction_workspace_reuses = 1
        else:
            direction_started_at = perf_counter()
            jvp_direction_batches = prepare_sparse_jvp_direction_batches(
                difference_plan=difference_plan,
                state_shape=tuple(state.shape),
                dtype=np.float64,
                batch_size=jvp_batch_size,
            )
            jvp_direction_build_seconds += perf_counter() - direction_started_at
        jvp_direction_batch_count = len(jvp_direction_batches)

    def record_jvp_timing(timing: dict[str, float | int]) -> None:
        nonlocal \
            jvp_jacobian_batch_count, \
            jvp_jacobian_linearize_seconds, \
            jvp_jacobian_prebuilt_direction_batch_uses
        nonlocal \
            jvp_jacobian_device_execute_seconds, \
            jvp_jacobian_host_transfer_seconds, \
            jvp_jacobian_push_seconds
        nonlocal \
            jvp_jacobian_sparse_assembly_seconds, \
            jvp_jacobian_tangent_build_seconds
        nonlocal jvp_jacobian_total_seconds
        jvp_jacobian_total_seconds += float(timing.get("total_seconds", 0.0))
        jvp_jacobian_linearize_seconds += float(timing.get("linearize_seconds", 0.0))
        jvp_jacobian_tangent_build_seconds += float(
            timing.get("tangent_build_seconds", 0.0)
        )
        jvp_jacobian_push_seconds += float(timing.get("push_seconds", 0.0))
        jvp_jacobian_device_execute_seconds += float(
            timing.get("device_execute_seconds", 0.0)
        )
        jvp_jacobian_host_transfer_seconds += float(
            timing.get("host_transfer_seconds", 0.0)
        )
        jvp_jacobian_sparse_assembly_seconds += float(
            timing.get("sparse_assembly_seconds", 0.0)
        )
        jvp_jacobian_batch_count += int(timing.get("batch_count", 0))
        jvp_jacobian_prebuilt_direction_batch_uses += int(
            timing.get("prebuilt_direction_batches", 0)
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
                converged=True,
            )

        if (
            jacobian is None
            or nonlinear_iteration == 1
            or ((nonlinear_iteration - 1) % refresh_frequency == 0)
        ):
            jacobian_started_at = perf_counter()
            if resolved_jacobian_mode == "jvp":
                jacobian = build_sparse_jvp_jacobian(
                    residual,
                    state,
                    sparsity=sparsity,
                    color_groups=color_groups,
                    sparsity_csc=sparsity_csc,
                    difference_plan=difference_plan,
                    batch_size=jvp_batch_size,
                    direction_batches=jvp_direction_batches,
                    timing_callback=record_jvp_timing,
                )
            else:
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
            update_is_finite = bool(np.all(np.isfinite(update)))
            last_linear_solver_backend = "scipy_spsolve"
            last_linear_solver_status = "ok" if update_is_finite else "nonfinite"
            last_linear_solver_success = update_is_finite
            last_linear_solver_reported_iterations = 1
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
            gmres_update_is_finite = bool(np.all(np.isfinite(update)))
            if exit_code != 0:
                update = spsolve(jacobian_csc, -residual_value)
                total_linear_iterations += 1
                direct_update_is_finite = bool(np.all(np.isfinite(update)))
                last_linear_solver_backend = "scipy_gmres_spsolve_fallback"
                last_linear_solver_status = (
                    f"gmres_exit_{int(exit_code)}_spsolve_"
                    f"{'ok' if direct_update_is_finite else 'nonfinite'}"
                )
                last_linear_solver_success = direct_update_is_finite
                last_linear_solver_reported_iterations = int(linear_iterations) + 1
            else:
                last_linear_solver_backend = "scipy_gmres"
                last_linear_solver_status = 0 if gmres_update_is_finite else "nonfinite"
                last_linear_solver_success = gmres_update_is_finite
                last_linear_solver_reported_iterations = int(linear_iterations)
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
            if (
                np.isfinite(trial_residual_inf_norm)
                and trial_residual_inf_norm <= residual_inf_norm
            ):
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
        if candidate_residual_inf_norm < float(residual_tolerance):
            return state, build_info(
                residual_inf_norm=candidate_residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration,
                converged=True,
            )
        if float(np.max(np.abs(step_scale * update))) < float(step_tolerance):
            # A tiny update with a large residual is stagnation, not convergence.
            # Let the sparse path try its Newton-Krylov fallback below.
            break

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
        converged=residual_inf_norm < float(residual_tolerance),
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
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when scipy is unavailable
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
        converged=float(np.max(np.abs(residual_value))) < float(residual_tolerance),
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
    linear_tolerance: float | None = None,
    linear_solver_backend: str = "jax_gmres",
    linear_solver_solve_method: str = "batched",
    linear_preconditioner: Callable[[object], object] | None = None,
    linear_preconditioner_name: str | None = None,
    linear_preconditioner_context: dict[str, object] | None = None,
    check_initial_residual: bool = True,
    initial_residual_mode: str = "residual",
    jit_residual: bool = False,
    line_search_initial_step_scale: float = 1.0,
) -> tuple[np.ndarray, ImplicitStepInfo]:
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import bicgstab, gmres
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX linearized implicit stepping requires jax.") from exc

    state = jnp.asarray(initial_state, dtype=jnp.float64)
    total_linear_iterations = 0
    residual_evaluation_count = 0
    residual_evaluation_seconds = 0.0
    jacobian_refresh_count = 0
    jacobian_assembly_seconds = 0.0
    linear_solve_seconds = 0.0
    line_search_seconds = 0.0
    line_search_trial_count = 0
    line_search_last_step_scale: float | None = None
    raw_line_search_initial_step_scale = float(line_search_initial_step_scale)
    if (
        not np.isfinite(raw_line_search_initial_step_scale)
        or raw_line_search_initial_step_scale <= 0.0
    ):
        raw_line_search_initial_step_scale = 1.0
    resolved_line_search_initial_step_scale = min(
        1.0,
        max(raw_line_search_initial_step_scale, 1.0 / 64.0),
    )
    linear_preconditioner_build_seconds = 0.0
    linear_preconditioner_build_count = 0
    linear_preconditioner_apply_seconds = 0.0
    linear_preconditioner_apply_count = 0
    cached_dynamic_preconditioner = None
    dynamic_preconditioner_refresh_frequency = (
        _dynamic_jax_linear_preconditioner_refresh_frequency(
            linear_preconditioner_context
        )
    )
    linear_backend = _resolve_jax_linear_solver_backend(linear_solver_backend)
    resolved_linear_tolerance = (
        float(residual_tolerance)
        if linear_tolerance is None
        else max(float(linear_tolerance), np.finfo(np.float64).tiny)
    )
    jacobian_mode = f"jax_linearized:{linear_backend}"
    last_linear_solver_status: int | float | str | None = None
    last_linear_solver_success: bool | None = None
    last_linear_solver_reported_iterations: int | None = None
    linear_operator_call_count = 0
    linear_operator_dispatch_seconds = 0.0
    resolved_solve_method = _resolve_jax_gmres_solve_method(linear_solver_solve_method)
    resolved_initial_residual_mode = _resolve_jax_linear_initial_residual_mode(
        initial_residual_mode
    )
    residual_function = jax.jit(residual) if bool(jit_residual) else residual
    known_state_residual_inf_norm: float | None = None

    def _block(value):
        if value is None:
            return None
        blocker = getattr(value, "block_until_ready", None)
        if callable(blocker):
            return blocker()
        try:
            return jax.block_until_ready(value)
        except (TypeError, ValueError):
            return value

    def _info(
        *,
        residual_inf_norm: float,
        nonlinear_iterations: int,
        converged: bool,
    ) -> ImplicitStepInfo:
        return ImplicitStepInfo(
            residual_inf_norm=float(residual_inf_norm),
            active_shape=active_shape,
            nonlinear_iterations=int(nonlinear_iterations),
            linear_iterations=total_linear_iterations,
            residual_evaluation_count=residual_evaluation_count,
            residual_evaluation_seconds=residual_evaluation_seconds,
            jacobian_refresh_count=jacobian_refresh_count,
            jacobian_assembly_seconds=jacobian_assembly_seconds,
            linear_solve_seconds=linear_solve_seconds,
            line_search_seconds=line_search_seconds,
            line_search_trial_count=line_search_trial_count,
            line_search_last_step_scale=line_search_last_step_scale,
            line_search_initial_step_scale=resolved_line_search_initial_step_scale,
            jacobian_mode=jacobian_mode,
            converged=bool(converged),
            linear_solver_backend=linear_backend,
            linear_solver_tolerance=float(resolved_linear_tolerance),
            linear_solver_status=last_linear_solver_status,
            linear_solver_success=last_linear_solver_success,
            linear_solver_reported_iterations=last_linear_solver_reported_iterations,
            linear_solver_solve_method=(
                resolved_solve_method if linear_backend == "jax_gmres" else None
            ),
            linear_operator_call_count=linear_operator_call_count,
            linear_operator_dispatch_seconds=linear_operator_dispatch_seconds,
            linear_preconditioner=linear_preconditioner_name
            if linear_preconditioner is not None
            or _is_dynamic_jax_linear_preconditioner(linear_preconditioner_name)
            else None,
            linear_preconditioner_build_seconds=linear_preconditioner_build_seconds,
            linear_preconditioner_build_count=linear_preconditioner_build_count,
            linear_preconditioner_apply_seconds=linear_preconditioner_apply_seconds,
            linear_preconditioner_apply_count=linear_preconditioner_apply_count,
            residual_jitted=bool(jit_residual),
            check_initial_residual=bool(check_initial_residual),
            initial_residual_mode=resolved_initial_residual_mode,
        )

    if check_initial_residual and resolved_initial_residual_mode == "residual":
        residual_started_at = perf_counter()
        initial_residual = residual_function(state)
        initial_residual = _block(initial_residual)
        residual_evaluation_count += 1
        residual_evaluation_seconds += perf_counter() - residual_started_at
        initial_residual_inf_norm = float(jnp.max(jnp.abs(initial_residual)))
        known_state_residual_inf_norm = initial_residual_inf_norm
        if initial_residual_inf_norm < float(residual_tolerance):
            return np.asarray(state, dtype=np.float64), _info(
                residual_inf_norm=initial_residual_inf_norm,
                nonlinear_iterations=0,
                converged=True,
            )

    for nonlinear_iteration in range(1, int(max_nonlinear_iterations) + 1):
        linearize_started_at = perf_counter()
        residual_value, linear_map = jax.linearize(residual_function, state)
        residual_value = _block(residual_value)
        elapsed = perf_counter() - linearize_started_at
        residual_evaluation_count += 1
        residual_evaluation_seconds += elapsed
        jacobian_refresh_count += 1
        jacobian_assembly_seconds += elapsed
        residual_inf_norm = float(jnp.max(jnp.abs(residual_value)))
        known_state_residual_inf_norm = residual_inf_norm
        if residual_inf_norm < float(residual_tolerance):
            return np.asarray(state, dtype=np.float64), _info(
                residual_inf_norm=residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration - 1,
                converged=True,
            )

        effective_preconditioner = linear_preconditioner
        if _is_dynamic_jax_linear_preconditioner(linear_preconditioner_name):
            should_refresh_preconditioner = (
                cached_dynamic_preconditioner is None
                or (nonlinear_iteration - 1)
                % dynamic_preconditioner_refresh_frequency
                == 0
            )
            if should_refresh_preconditioner:
                preconditioner_started_at = perf_counter()
                cached_dynamic_preconditioner = (
                    _build_jax_linearized_dynamic_preconditioner(
                        linear_preconditioner_name,
                        linear_map,
                        state,
                        context=linear_preconditioner_context,
                    )
                )
                linear_preconditioner_build_seconds += (
                    perf_counter() - preconditioner_started_at
                )
                linear_preconditioner_build_count += 1
            effective_preconditioner = cached_dynamic_preconditioner

        def counted_linear_map(tangent):
            nonlocal linear_operator_call_count, linear_operator_dispatch_seconds
            operator_started_at = perf_counter()
            result = linear_map(tangent)
            linear_operator_dispatch_seconds += perf_counter() - operator_started_at
            linear_operator_call_count += 1
            return result

        counted_preconditioner = None
        if effective_preconditioner is not None:

            def counted_preconditioner(vector):
                nonlocal linear_preconditioner_apply_count
                nonlocal linear_preconditioner_apply_seconds
                preconditioner_apply_started_at = perf_counter()
                result = effective_preconditioner(vector)
                linear_preconditioner_apply_seconds += (
                    perf_counter() - preconditioner_apply_started_at
                )
                linear_preconditioner_apply_count += 1
                return result

        linear_solve_started_at = perf_counter()
        solve_result = _solve_jax_linearized_update(
            counted_linear_map,
            -residual_value,
            backend=linear_backend,
            residual_tolerance=float(residual_tolerance),
            linear_tolerance=float(resolved_linear_tolerance),
            linear_restart=int(linear_restart),
            linear_maxiter=int(linear_maxiter),
            jax_gmres=gmres,
            jax_bicgstab=bicgstab,
            gmres_solve_method=resolved_solve_method,
            preconditioner=counted_preconditioner,
        )
        update = solve_result.update
        update = _block(update)
        status = (
            _block(solve_result.status) if solve_result.status is not None else None
        )
        last_linear_solver_status = _normalize_linear_solver_status(status)
        last_linear_solver_success = _linear_solver_success(
            backend=linear_backend,
            status=last_linear_solver_status,
            explicit_success=solve_result.success,
        )
        last_linear_solver_reported_iterations = _normalize_linear_solver_iterations(
            solve_result.reported_iterations
        )
        linear_solve_seconds += perf_counter() - linear_solve_started_at
        total_linear_iterations += int(linear_restart) * int(linear_maxiter)
        update = jnp.asarray(update, dtype=jnp.float64)

        accepted = False
        step_scale = float(resolved_line_search_initial_step_scale)
        candidate_state = state
        candidate_residual_inf_norm = residual_inf_norm
        line_search_started_at = perf_counter()
        while step_scale >= 1.0 / 64.0:
            trial_state = state + step_scale * update
            finite_trial = _block(jnp.all(jnp.isfinite(trial_state)))
            if not bool(finite_trial):
                step_scale *= 0.5
                continue
            residual_started_at = perf_counter()
            trial_residual = residual_function(trial_state)
            trial_residual = _block(trial_residual)
            line_search_trial_count += 1
            residual_evaluation_count += 1
            residual_evaluation_seconds += perf_counter() - residual_started_at
            trial_residual_inf_norm = float(jnp.max(jnp.abs(trial_residual)))
            if (
                np.isfinite(trial_residual_inf_norm)
                and trial_residual_inf_norm <= residual_inf_norm
            ):
                candidate_state = trial_state
                candidate_residual_inf_norm = trial_residual_inf_norm
                line_search_last_step_scale = float(step_scale)
                accepted = True
                break
            step_scale *= 0.5
        line_search_seconds += perf_counter() - line_search_started_at

        if not accepted:
            break

        state = candidate_state
        known_state_residual_inf_norm = float(candidate_residual_inf_norm)
        if candidate_residual_inf_norm < float(residual_tolerance):
            return np.asarray(state, dtype=np.float64), _info(
                residual_inf_norm=candidate_residual_inf_norm,
                nonlinear_iterations=nonlinear_iteration,
                converged=True,
            )
        if float(jnp.max(jnp.abs(step_scale * update))) < float(step_tolerance):
            # Small-step termination must not mask a failed nonlinear residual.
            break

    if known_state_residual_inf_norm is None:
        residual_started_at = perf_counter()
        final_residual = residual_function(state)
        final_residual = _block(final_residual)
        residual_evaluation_count += 1
        residual_evaluation_seconds += perf_counter() - residual_started_at
        known_state_residual_inf_norm = float(jnp.max(jnp.abs(final_residual)))
    return np.asarray(state, dtype=np.float64), _info(
        residual_inf_norm=float(known_state_residual_inf_norm),
        nonlinear_iterations=int(max_nonlinear_iterations),
        converged=float(known_state_residual_inf_norm) < float(residual_tolerance),
    )


def _resolve_jax_linear_initial_residual_mode(name: str) -> str:
    normalized = str(name or "residual").strip().lower().replace("-", "_")
    aliases = {
        "": "residual",
        "default": "residual",
        "residual": "residual",
        "standalone": "residual",
        "separate": "residual",
        "linearize": "linearize",
        "linearized": "linearize",
        "linearization": "linearize",
        "jacobian": "linearize",
    }
    if normalized not in aliases:
        raise ValueError(
            "initial_residual_mode must be 'residual' or 'linearize', "
            f"got {name!r}."
        )
    return aliases[normalized]


def _resolve_jax_linear_solver_backend(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "jax": "jax_gmres",
        "jax_scipy": "jax_gmres",
        "gmres": "jax_gmres",
        "jax_gmres": "jax_gmres",
        "bicgstab": "jax_bicgstab",
        "jax_bicgstab": "jax_bicgstab",
        "lineax": "lineax_gmres",
        "lineax_gmres": "lineax_gmres",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported JAX linear solver backend {name!r}.") from exc


def _resolve_jax_gmres_solve_method(name: str | None) -> str:
    normalized = str(name or "batched").strip().lower().replace("-", "_")
    aliases = {
        "": "batched",
        "default": "batched",
        "batch": "batched",
        "batched": "batched",
        "incremental": "incremental",
        "givens": "incremental",
        "qr": "incremental",
    }
    return aliases.get(normalized, "batched")


def _is_dynamic_jax_linear_preconditioner(name: str | None) -> bool:
    return str(name or "").strip().lower().replace("-", "_") in {
        "linearized_diag",
        "jvp_diag",
        "jacobian_diag",
        "field_diag",
        "field_jacobi",
        "field_diagonal",
        "local_block_diag",
        "local_block",
        "block_jacobi",
        "cell_block",
        "physics_block",
        "parallel_line",
        "transport_line",
        "line_block",
        "physics_transport",
    }


def _dynamic_jax_linear_preconditioner_refresh_frequency(
    context: dict[str, object] | None,
) -> int:
    if context is None:
        return 1
    try:
        value = int(context.get("refresh_frequency", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _safe_diagonal_denominator(diagonal, *, floor: float):
    try:
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX diagonal preconditioning requires jax.") from exc

    abs_diagonal = jnp.abs(diagonal)
    sign = jnp.where(diagonal < 0.0, -1.0, 1.0)
    return jnp.where(abs_diagonal >= float(floor), diagonal, sign * float(floor))


def _build_jax_linearized_diagonal_preconditioner(
    linear_map,
    prototype_state,
    *,
    floor: float = 1.0e-12,
    max_size: int = 2048,
):
    """Build an opt-in inverse-Jacobian diagonal preconditioner from JVPs."""

    try:
        import jax
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX diagonal preconditioning requires jax.") from exc

    prototype = jnp.asarray(prototype_state, dtype=jnp.float64)
    flat_size = int(prototype.size)
    if flat_size == 0:
        safe_diagonal = jnp.ones_like(prototype, dtype=jnp.float64)
    else:
        if flat_size > int(max_size):
            raise ValueError(
                "linearized_diag preconditioner is bounded to systems with "
                f"at most {int(max_size)} unknowns; got {flat_size}."
            )
        basis = jnp.eye(flat_size, dtype=prototype.dtype).reshape(
            (flat_size,) + tuple(prototype.shape)
        )
        action_matrix = jax.vmap(
            lambda tangent: jnp.ravel(linear_map(tangent))
        )(basis)
        diagonal = jnp.diag(action_matrix).reshape(tuple(prototype.shape))
        safe_diagonal = _safe_diagonal_denominator(diagonal, floor=float(floor))
    safe_diagonal = jax.block_until_ready(safe_diagonal)

    def preconditioner(vector):
        return jnp.asarray(vector, dtype=jnp.float64) / safe_diagonal

    return preconditioner


def _build_jax_linearized_dynamic_preconditioner(
    name: str | None,
    linear_map,
    prototype_state,
    *,
    context: dict[str, object] | None = None,
):
    normalized = str(name or "").strip().lower().replace("-", "_")
    if normalized in {"linearized_diag", "jvp_diag", "jacobian_diag"}:
        return _build_jax_linearized_diagonal_preconditioner(linear_map, prototype_state)
    if normalized in {"field_diag", "field_jacobi", "field_diagonal"}:
        if context is None:
            raise ValueError(
                "field_diag preconditioner requires active_cell_count and "
                "field_count in linear_preconditioner_context."
            )
        return _build_jax_linearized_field_diagonal_preconditioner(
            linear_map,
            prototype_state,
            active_cell_count=int(context.get("active_cell_count", 0)),
            field_count=int(context.get("field_count", 0)),
            feedback_count=int(context.get("feedback_count", 0)),
            floor=float(context.get("floor", 1.0e-10)),
            max_unknowns=int(context.get("max_unknowns", 8192)),
        )
    if normalized in {
        "local_block_diag",
        "local_block",
        "block_jacobi",
        "cell_block",
        "physics_block",
    }:
        if context is None:
            raise ValueError(
                "local_block_diag preconditioner requires active_cell_count and "
                "field_count in linear_preconditioner_context."
            )
        return _build_jax_linearized_local_block_preconditioner(
            linear_map,
            prototype_state,
            active_cell_count=int(context.get("active_cell_count", 0)),
            field_count=int(context.get("field_count", 0)),
            feedback_count=int(context.get("feedback_count", 0)),
            floor=float(context.get("floor", 1.0e-10)),
            max_unknowns=int(context.get("max_unknowns", 4096)),
        )
    if normalized in {
        "parallel_line",
        "transport_line",
        "line_block",
        "physics_transport",
    }:
        if context is None:
            raise ValueError(
                "parallel_line preconditioner requires active_shape and field_count "
                "in linear_preconditioner_context."
            )
        active_shape = tuple(int(axis) for axis in context.get("active_shape", ()))
        return _build_jax_linearized_parallel_line_preconditioner(
            linear_map,
            prototype_state,
            active_shape=active_shape,
            field_count=int(context.get("field_count", 0)),
            feedback_count=int(context.get("feedback_count", 0)),
            parallel_axis=int(context.get("parallel_axis", 0)),
            floor=float(context.get("floor", 1.0e-10)),
            max_line_unknowns=int(context.get("max_line_unknowns", 512)),
            max_batch_unknowns=int(context.get("max_batch_unknowns", 2048)),
            max_total_unknowns=int(context.get("max_total_unknowns", 8192)),
        )
    raise ValueError(f"Unsupported dynamic JAX preconditioner {name!r}.")


def _build_jax_linearized_field_diagonal_preconditioner(
    linear_map,
    prototype_state,
    *,
    active_cell_count: int,
    field_count: int,
    feedback_count: int = 0,
    floor: float = 1.0e-10,
    max_unknowns: int = 8192,
):
    """Build a field-active diagonal preconditioner from scalar JVPs.

    This is the cheapest JVP-derived physics preconditioner for the field-major
    recycling layout. It samples only the diagonal entries of the active plasma
    and neutral field block and leaves feedback scalars unscaled.
    """

    try:
        import jax
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX field-diagonal preconditioning requires jax.") from exc

    prototype = jnp.asarray(prototype_state, dtype=jnp.float64)
    flat_size = int(prototype.size)
    active_cell_count = int(active_cell_count)
    field_count = int(field_count)
    feedback_count = int(feedback_count)
    field_unknown_count = active_cell_count * field_count
    if active_cell_count < 0 or field_count < 0 or feedback_count < 0:
        raise ValueError("field_diag preconditioner counts must be non-negative.")
    if flat_size != field_unknown_count + feedback_count:
        raise ValueError(
            "field_diag preconditioner context does not match packed state: "
            f"state has {flat_size} entries, expected "
            f"{field_unknown_count + feedback_count}."
        )
    if field_unknown_count == 0:
        return lambda vector: jnp.asarray(vector, dtype=jnp.float64)
    if field_unknown_count > int(max_unknowns):
        raise ValueError(
            "field_diag preconditioner is bounded to field blocks with at most "
            f"{int(max_unknowns)} unknowns; got {field_unknown_count}."
        )

    field_indices = jnp.arange(field_unknown_count, dtype=jnp.int32)

    def diagonal_entry(flat_index):
        tangent_flat = jnp.zeros(flat_size, dtype=prototype.dtype).at[flat_index].set(
            1.0
        )
        action = jnp.ravel(linear_map(tangent_flat.reshape(tuple(prototype.shape))))
        return action[flat_index]

    diagonal = jax.vmap(diagonal_entry)(field_indices)
    safe_diagonal = _safe_diagonal_denominator(diagonal, floor=float(floor))
    safe_diagonal = jax.block_until_ready(safe_diagonal)

    def preconditioner(vector):
        flat_vector = jnp.ravel(jnp.asarray(vector, dtype=jnp.float64))
        solved_fields = flat_vector[:field_unknown_count] / safe_diagonal
        if feedback_count:
            solved = jnp.concatenate(
                (solved_fields, flat_vector[field_unknown_count:]), axis=0
            )
        else:
            solved = solved_fields
        return solved.reshape(tuple(prototype.shape))

    return preconditioner


def _build_jax_linearized_local_block_preconditioner(
    linear_map,
    prototype_state,
    *,
    active_cell_count: int,
    field_count: int,
    feedback_count: int = 0,
    floor: float = 1.0e-10,
    max_unknowns: int = 4096,
):
    """Build a same-cell multiphysics block-Jacobi preconditioner from JVPs.

    The packed recycling state is field-major.  This preconditioner extracts the
    local field-by-equation Jacobian block at each active cell using a batched
    JVP, inverts those small dense blocks on device, and leaves global transport
    couplings to the outer Krylov iteration.
    """

    try:
        import jax
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX block preconditioning requires jax.") from exc

    prototype = jnp.asarray(prototype_state, dtype=jnp.float64)
    flat_size = int(prototype.size)
    active_cell_count = int(active_cell_count)
    field_count = int(field_count)
    feedback_count = int(feedback_count)
    field_unknown_count = active_cell_count * field_count
    if active_cell_count < 0 or field_count < 0 or feedback_count < 0:
        raise ValueError("local_block_diag preconditioner counts must be non-negative.")
    if flat_size != field_unknown_count + feedback_count:
        raise ValueError(
            "local_block_diag preconditioner context does not match packed state: "
            f"state has {flat_size} entries, expected "
            f"{field_unknown_count + feedback_count}."
        )
    if field_unknown_count == 0:
        return lambda vector: jnp.asarray(vector, dtype=jnp.float64)
    if field_unknown_count > int(max_unknowns):
        raise ValueError(
            "local_block_diag preconditioner is bounded to field blocks with at "
            f"most {int(max_unknowns)} unknowns; got {field_unknown_count}."
        )

    directions = jnp.eye(
        field_unknown_count,
        flat_size,
        dtype=prototype.dtype,
    ).reshape((field_unknown_count,) + tuple(prototype.shape))
    action_matrix = jax.vmap(lambda tangent: jnp.ravel(linear_map(tangent)))(
        directions
    )
    actions_by_cell_column = action_matrix.reshape(
        (field_count, active_cell_count, flat_size)
    ).transpose((1, 0, 2))
    row_indices = (
        jnp.arange(field_count, dtype=jnp.int32)[None, :] * active_cell_count
        + jnp.arange(active_cell_count, dtype=jnp.int32)[:, None]
    )
    block_columns = []
    for column in range(field_count):
        action_rows = actions_by_cell_column[:, column, :]
        block_columns.append(
            jnp.take_along_axis(action_rows, row_indices, axis=1)
        )
    blocks = jnp.stack(tuple(block_columns), axis=2)
    eye = jnp.eye(field_count, dtype=prototype.dtype)
    block_scale = jnp.maximum(jnp.max(jnp.abs(blocks), axis=(1, 2)), 1.0)
    regularized_blocks = blocks + (
        float(floor) * block_scale[:, None, None] * eye[None, :, :]
    )
    diagonal = jnp.diagonal(regularized_blocks, axis1=1, axis2=2)
    safe_diagonal = _safe_diagonal_denominator(diagonal, floor=float(floor))
    regularized_blocks, safe_diagonal = jax.block_until_ready(
        (regularized_blocks, safe_diagonal)
    )

    def preconditioner(vector):
        flat_vector = jnp.ravel(jnp.asarray(vector, dtype=jnp.float64))
        field_vector = flat_vector[:field_unknown_count]
        rhs_by_cell = field_vector.reshape(
            (field_count, active_cell_count)
        ).transpose((1, 0))
        solved_by_cell = jnp.linalg.solve(
            regularized_blocks,
            rhs_by_cell[..., None],
        )[..., 0]
        diagonal_fallback = rhs_by_cell / safe_diagonal
        solved_by_cell = jnp.where(
            jnp.all(jnp.isfinite(solved_by_cell), axis=1, keepdims=True),
            solved_by_cell,
            diagonal_fallback,
        )
        solved_fields = solved_by_cell.transpose((1, 0)).reshape((field_unknown_count,))
        if feedback_count:
            solved = jnp.concatenate(
                (solved_fields, flat_vector[field_unknown_count:]), axis=0
            )
        else:
            solved = solved_fields
        return solved.reshape(tuple(prototype.shape))

    return preconditioner


def _field_major_line_indices(
    *,
    active_shape: tuple[int, ...],
    field_count: int,
    parallel_axis: int,
) -> np.ndarray:
    """Return packed field-major unknown indices grouped by parallel line."""

    if not active_shape:
        if field_count == 0:
            return np.zeros((0, 0), dtype=np.int32)
        raise ValueError("parallel_line preconditioner requires a non-empty shape.")
    rank = len(active_shape)
    if not (0 <= int(parallel_axis) < rank):
        raise ValueError(
            f"parallel_axis={parallel_axis} is outside active_shape rank {rank}."
        )
    active_cell_count = int(np.prod(active_shape, dtype=np.int64))
    line_length = int(active_shape[int(parallel_axis)])
    transverse_shape = tuple(
        axis_size
        for axis, axis_size in enumerate(active_shape)
        if axis != int(parallel_axis)
    )
    transverse_indices = (
        [()]
        if not transverse_shape
        else list(np.ndindex(transverse_shape))
    )
    line_indices = np.empty(
        (len(transverse_indices), int(field_count) * line_length), dtype=np.int32
    )
    for line_number, transverse_index in enumerate(transverse_indices):
        line_columns: list[int] = []
        for field_index in range(int(field_count)):
            field_offset = field_index * active_cell_count
            for parallel_coordinate in range(line_length):
                cell_index: list[int] = []
                transverse_offset = 0
                for axis in range(rank):
                    if axis == int(parallel_axis):
                        cell_index.append(parallel_coordinate)
                    else:
                        cell_index.append(int(transverse_index[transverse_offset]))
                        transverse_offset += 1
                cell_flat = np.ravel_multi_index(tuple(cell_index), active_shape)
                line_columns.append(field_offset + int(cell_flat))
        line_indices[line_number, :] = np.asarray(line_columns, dtype=np.int32)
    return line_indices


def _build_jax_linearized_parallel_line_preconditioner(
    linear_map,
    prototype_state,
    *,
    active_shape: tuple[int, ...],
    field_count: int,
    feedback_count: int = 0,
    parallel_axis: int = 0,
    floor: float = 1.0e-10,
    max_line_unknowns: int = 512,
    max_batch_unknowns: int = 2048,
    max_total_unknowns: int = 8192,
):
    """Build a JVP-derived line-block preconditioner along the parallel axis.

    This is the matrix-free analogue of the physics preconditioners used by
    edge-fluid codes: the Krylov operator remains the true linearized residual,
    while the left preconditioner approximately inverts the stiff local
    multi-field transport block on each open-field-line segment.
    """

    try:
        import jax
        import jax.numpy as jnp
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only when jax is unavailable
        raise ImportError("JAX line-block preconditioning requires jax.") from exc

    prototype = jnp.asarray(prototype_state, dtype=jnp.float64)
    flat_size = int(prototype.size)
    field_count = int(field_count)
    feedback_count = int(feedback_count)
    active_shape = tuple(int(axis) for axis in active_shape)
    if any(axis < 0 for axis in active_shape):
        raise ValueError("parallel_line preconditioner active_shape must be non-negative.")
    active_cell_count = int(np.prod(active_shape, dtype=np.int64)) if active_shape else 0
    field_unknown_count = active_cell_count * field_count
    if field_count < 0 or feedback_count < 0:
        raise ValueError("parallel_line preconditioner counts must be non-negative.")
    if flat_size != field_unknown_count + feedback_count:
        raise ValueError(
            "parallel_line preconditioner context does not match packed state: "
            f"state has {flat_size} entries, expected "
            f"{field_unknown_count + feedback_count}."
        )
    if field_unknown_count == 0:
        return lambda vector: jnp.asarray(vector, dtype=jnp.float64)
    if field_unknown_count > int(max_total_unknowns):
        raise ValueError(
            "parallel_line preconditioner is bounded to field blocks with at most "
            f"{int(max_total_unknowns)} unknowns; got {field_unknown_count}."
        )

    line_indices_np = _field_major_line_indices(
        active_shape=active_shape,
        field_count=field_count,
        parallel_axis=int(parallel_axis),
    )
    if line_indices_np.size == 0:
        return lambda vector: jnp.asarray(vector, dtype=jnp.float64)
    line_unknown_count = int(line_indices_np.shape[1])
    if line_unknown_count > int(max_line_unknowns):
        raise ValueError(
            "parallel_line preconditioner line block exceeds max_line_unknowns: "
            f"{line_unknown_count} > {int(max_line_unknowns)}."
        )

    line_indices = jnp.asarray(line_indices_np, dtype=jnp.int32)
    blocks: list[object] = []
    max_lines_per_batch = max(1, int(max_batch_unknowns) // line_unknown_count)
    column_coordinates = jnp.arange(line_unknown_count, dtype=jnp.int32)
    for batch_start in range(0, int(line_indices_np.shape[0]), max_lines_per_batch):
        batch_indices_np = line_indices_np[
            batch_start : batch_start + max_lines_per_batch
        ]
        batch_indices = jnp.asarray(batch_indices_np, dtype=jnp.int32)
        batch_line_count = int(batch_indices_np.shape[0])
        directions = jnp.zeros(
            (batch_line_count, line_unknown_count, flat_size),
            dtype=prototype.dtype,
        ).at[
            jnp.arange(batch_line_count, dtype=jnp.int32)[:, None],
            column_coordinates[None, :],
            batch_indices,
        ].set(1.0)
        directions = directions.reshape(
            (batch_line_count * line_unknown_count,) + tuple(prototype.shape)
        )
        action_rows = jax.vmap(lambda tangent: jnp.ravel(linear_map(tangent)))(
            directions
        )
        action_rows = action_rows.reshape(
            (batch_line_count, line_unknown_count, flat_size)
        )
        selected_rows = jnp.take_along_axis(
            action_rows,
            batch_indices[:, None, :],
            axis=2,
        )
        blocks.append(jnp.transpose(selected_rows, (0, 2, 1)))

    block_array = jnp.concatenate(tuple(blocks), axis=0)
    eye = jnp.eye(line_unknown_count, dtype=prototype.dtype)
    block_scale = jnp.maximum(jnp.max(jnp.abs(block_array), axis=(1, 2)), 1.0)
    regularized_blocks = block_array + (
        float(floor) * block_scale[:, None, None] * eye[None, :, :]
    )
    diagonal = jnp.diagonal(regularized_blocks, axis1=1, axis2=2)
    safe_diagonal = _safe_diagonal_denominator(diagonal, floor=float(floor))
    regularized_blocks, safe_diagonal, line_indices = jax.block_until_ready(
        (regularized_blocks, safe_diagonal, line_indices)
    )

    def preconditioner(vector):
        flat_vector = jnp.ravel(jnp.asarray(vector, dtype=jnp.float64))
        field_vector = flat_vector[:field_unknown_count]
        rhs_by_line = field_vector[line_indices]
        solved_by_line = jnp.linalg.solve(
            regularized_blocks,
            rhs_by_line[..., None],
        )[..., 0]
        diagonal_fallback = rhs_by_line / safe_diagonal
        solved_by_line = jnp.where(
            jnp.all(jnp.isfinite(solved_by_line), axis=1, keepdims=True),
            solved_by_line,
            diagonal_fallback,
        )
        solved_fields = jnp.zeros_like(field_vector).at[line_indices].set(
            solved_by_line
        )
        if feedback_count:
            solved = jnp.concatenate(
                (solved_fields, flat_vector[field_unknown_count:]), axis=0
            )
        else:
            solved = solved_fields
        return solved.reshape(tuple(prototype.shape))

    return preconditioner


def _solve_jax_linearized_update(
    linear_map,
    rhs,
    *,
    backend: str,
    residual_tolerance: float,
    linear_tolerance: float,
    linear_restart: int,
    linear_maxiter: int,
    jax_gmres,
    jax_bicgstab,
    gmres_solve_method: str = "batched",
    preconditioner=None,
):
    if backend == "jax_gmres":
        update, status = jax_gmres(
            linear_map,
            rhs,
            tol=float(linear_tolerance),
            atol=0.0,
            restart=int(linear_restart),
            maxiter=int(linear_maxiter),
            M=preconditioner,
            solve_method=_resolve_jax_gmres_solve_method(gmres_solve_method),
        )
        return JaxLinearizedUpdateResult(update=update, backend=backend, status=status)
    if backend == "jax_bicgstab":
        update, status = jax_bicgstab(
            linear_map,
            rhs,
            tol=float(linear_tolerance),
            atol=0.0,
            maxiter=max(1, int(linear_restart) * int(linear_maxiter)),
            M=preconditioner,
        )
        return JaxLinearizedUpdateResult(update=update, backend=backend, status=status)
    if backend == "lineax_gmres":
        try:
            import jax
            import lineax as lx
        except (
            ImportError
        ) as exc:  # pragma: no cover - depends on optional local install
            raise ImportError(
                "Lineax GMRES requires the optional lineax package."
            ) from exc

        rhs_array = jax.numpy.asarray(rhs)
        structure = jax.ShapeDtypeStruct(rhs_array.shape, rhs_array.dtype)
        operator = lx.FunctionLinearOperator(linear_map, structure)
        solver = lx.GMRES(
            rtol=float(linear_tolerance),
            atol=0.0,
            restart=int(linear_restart),
            max_steps=max(1, int(linear_restart) * int(linear_maxiter)),
        )
        solution = lx.linear_solve(operator, rhs_array, solver, throw=False)
        reported_iterations = None
        stats = getattr(solution, "stats", None)
        if isinstance(stats, dict):
            reported_iterations = stats.get("num_steps")
        return JaxLinearizedUpdateResult(
            update=solution.value,
            backend=backend,
            status=getattr(solution, "result", None),
            reported_iterations=reported_iterations,
        )
    raise ValueError(f"Unsupported JAX linear solver backend {backend!r}.")


def _normalize_linear_solver_status(status: object) -> int | float | str | None:
    if status is None:
        return None
    item = getattr(status, "item", None)
    if callable(item):
        try:
            status = item()
        except (TypeError, ValueError):
            pass
    if isinstance(status, np.generic):
        status = status.item()
    if isinstance(status, (bool, int, float, str)):
        return status
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return str(status)


def _linear_solver_success(
    *,
    backend: str,
    status: int | float | str | None,
    explicit_success: bool | None = None,
) -> bool | None:
    if explicit_success is not None:
        return bool(explicit_success)
    if status is None:
        return None
    if isinstance(status, bool):
        return bool(status)
    if isinstance(status, (int, float)):
        return int(status) == 0
    normalized = str(status).strip().lower()
    if not normalized:
        return None
    if "success" in normalized or "successful" in normalized:
        return True
    if backend == "lineax_gmres" and "results<>" in normalized:
        return True
    if any(
        token in normalized
        for token in ("fail", "error", "max", "singular", "breakdown")
    ):
        return False
    if backend == "jax_gmres" and normalized.isdigit():
        return int(normalized) == 0
    return None


def _normalize_linear_solver_iterations(value: object) -> int | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
