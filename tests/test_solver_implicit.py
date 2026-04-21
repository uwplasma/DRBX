from __future__ import annotations

import numpy as np
import pytest

from jax_drb.solver import (
    backward_euler_residual,
    bdf2_residual,
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    pack_active_fields,
    solve_jax_linearized_newton_system,
    solve_matrix_free_newton_system,
    solve_sparse_newton_system,
    unpack_active_fields,
)


def test_pack_unpack_active_fields_round_trip() -> None:
    active = (slice(1, 5), slice(2, 4), slice(None))
    density = np.arange(6 * 7 * 3, dtype=np.float64).reshape(6, 7, 3)
    pressure = 10.0 + density

    packed = pack_active_fields((density, pressure), active_slices=active)
    restored_density, restored_pressure = unpack_active_fields(
        packed,
        templates=(density, pressure),
        active_slices=active,
    )

    np.testing.assert_allclose(restored_density[active], density[active])
    np.testing.assert_allclose(restored_pressure[active], pressure[active])
    np.testing.assert_allclose(restored_density[:1], density[:1])
    np.testing.assert_allclose(restored_pressure[:, :2], pressure[:, :2])


def test_build_locality_sparsity_respects_periodic_axis() -> None:
    pytest.importorskip("scipy")

    active_shape = (3, 2, 4)
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=2,
        radii=(1, 1, 1),
        periodic_axes=(2,),
    )
    active_cells = int(np.prod(active_shape))

    def row_columns(row: int) -> set[int]:
        return set(sparsity.indices[sparsity.indptr[row] : sparsity.indptr[row + 1]].tolist())

    center = np.ravel_multi_index((1, 1, 0), active_shape)
    periodic_neighbor = np.ravel_multi_index((1, 1, 3), active_shape)
    far_x_neighbor = np.ravel_multi_index((0, 0, 2), active_shape)
    columns = row_columns(center)

    assert periodic_neighbor in columns
    assert active_cells + periodic_neighbor in columns
    assert far_x_neighbor not in columns


def test_build_modulo_color_groups_partitions_state() -> None:
    active_shape = (4, 3, 2)
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=2,
        color_periods=(2, 3, 2),
    )

    flattened = sorted(column for group in color_groups for column in group)

    assert flattened == list(range(2 * np.prod(active_shape)))
    assert len(color_groups) == 2 * 2 * 3 * 2


def test_sparse_difference_quotient_jacobian_matches_single_column_difference() -> None:
    pytest.importorskip("scipy")

    active_shape = (3, 1, 4)
    state = np.linspace(0.2, 0.8, 2 * np.prod(active_shape))
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=2,
        radii=(1, 0, 1),
        periodic_axes=(2,),
    )
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=2,
        color_periods=(3, 1, 4),
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        field0 = vector[: np.prod(active_shape)].reshape(active_shape)
        field1 = vector[np.prod(active_shape) :].reshape(active_shape)
        result0 = field0**2 + 0.1 * np.roll(field0, 1, axis=2) - 0.2 * field1
        result1 = field1 + 0.3 * np.roll(field0, -1, axis=0) - 0.05 * np.roll(field1, -1, axis=2)
        return np.concatenate([result0.ravel(), result1.ravel()])

    jacobian = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
    )
    column = 5
    step = np.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(float(state[column])))
    perturbation = np.zeros_like(state)
    perturbation[column] = step
    direct = (residual(state + perturbation) - residual(state)) / step
    sparse_column = jacobian.getcol(column).toarray().ravel()

    np.testing.assert_allclose(sparse_column, direct, rtol=1e-6, atol=1e-8)


def test_sparse_difference_quotient_jacobian_parallel_matches_serial() -> None:
    pytest.importorskip("scipy")

    active_shape = (4, 1, 4)
    state = np.linspace(0.2, 0.8, 2 * np.prod(active_shape))
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=2,
        radii=(1, 0, 1),
        periodic_axes=(2,),
    )
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=2,
        color_periods=(2, 1, 4),
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        field0 = vector[: np.prod(active_shape)].reshape(active_shape)
        field1 = vector[np.prod(active_shape) :].reshape(active_shape)
        result0 = field0**2 + 0.1 * np.roll(field0, 1, axis=2) - 0.2 * field1
        result1 = field1 + 0.3 * np.roll(field0, -1, axis=0) - 0.05 * np.roll(field1, -1, axis=2)
        return np.concatenate([result0.ravel(), result1.ravel()])

    serial = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        parallel_workers=1,
    )
    parallel = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        parallel_workers=2,
    )

    np.testing.assert_allclose(serial.toarray(), parallel.toarray(), rtol=1e-10, atol=1e-12)


def test_backward_euler_and_bdf2_residual_formulas() -> None:
    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    previous = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    previous_previous = np.array([0.25, 1.0, 2.0], dtype=np.float64)
    rhs = np.array([0.2, -0.1, 0.4], dtype=np.float64)

    be = backward_euler_residual(state, previous, rhs, timestep=0.5)
    bdf = bdf2_residual(state, previous, previous_previous, rhs, timestep=0.5)

    np.testing.assert_allclose(be, np.array([0.4, 0.55, 0.3]))
    np.testing.assert_allclose(bdf, np.array([0.35, 0.3666666666666667, 0.2]))


def test_sparse_and_matrix_free_newton_solvers_recover_known_root() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(active_shape, field_count=1, color_periods=(3,))

    def residual(state: np.ndarray) -> np.ndarray:
        return state * state - target * target

    sparse_solution, sparse_info = solve_sparse_newton_system(
        residual,
        initial,
        active_shape=active_shape,
        sparsity=sparsity,
        color_groups=color_groups,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=8,
        linear_restart=5,
        linear_maxiter=20,
        linear_rtol=1.0e-10,
    )
    matrix_free_solution, matrix_free_info = solve_matrix_free_newton_system(
        residual,
        initial,
        active_shape=active_shape,
        residual_tolerance=1.0e-10,
        max_nonlinear_iterations=8,
    )

    np.testing.assert_allclose(sparse_solution, target, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(matrix_free_solution, target, rtol=1e-9, atol=1e-9)
    assert sparse_info.residual_inf_norm < 1.0e-10
    assert matrix_free_info.residual_inf_norm < 1.0e-10


def test_jax_linearized_newton_solver_recovers_known_root() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (3,)
    target = jnp.array([1.0, 0.5, 2.0], dtype=jnp.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)

    def residual(state):
        return state * state - target * target

    solution, info = solve_jax_linearized_newton_system(
        residual,
        initial,
        active_shape=active_shape,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=8,
        linear_restart=10,
        linear_maxiter=4,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-9, atol=1.0e-9)
    assert info.residual_inf_norm < 1.0e-10


def test_sparse_newton_solver_supports_direct_linear_solve_mode() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(active_shape, field_count=1, color_periods=(3,))

    def residual(state: np.ndarray) -> np.ndarray:
        return state * state - target * target

    solution, info = solve_sparse_newton_system(
        residual,
        initial,
        active_shape=active_shape,
        sparsity=sparsity,
        color_groups=color_groups,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=8,
        linear_restart=5,
        linear_maxiter=20,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, target, rtol=1.0e-9, atol=1.0e-9)
    assert info.residual_inf_norm < 1.0e-10
    assert info.linear_iterations >= 1


def test_sparse_newton_solver_supports_jacobian_reuse() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(active_shape, field_count=1, color_periods=(3,))

    def residual(state: np.ndarray) -> np.ndarray:
        return state * state - target * target

    solution, info = solve_sparse_newton_system(
        residual,
        initial,
        active_shape=active_shape,
        sparsity=sparsity,
        color_groups=color_groups,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=8,
        linear_restart=5,
        linear_maxiter=20,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
        jacobian_refresh_frequency=2,
    )

    np.testing.assert_allclose(solution, target, rtol=1.0e-9, atol=1.0e-9)
    assert info.residual_inf_norm < 1.0e-10
