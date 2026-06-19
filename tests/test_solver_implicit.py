from __future__ import annotations

import numpy as np
import pytest

import jax_drb.solver.implicit as implicit_mod
from jax_drb.solver import (
    active_region_from_slices,
    backward_euler_residual,
    bdf2_residual,
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    build_sparse_jvp_jacobian,
    difference_quotient_step_size,
    pack_active_fields,
    prepare_sparse_difference_quotient_plan,
    prepare_sparse_jvp_direction_batches,
    prepare_sparse_jvp_workspace,
    solve_jax_linearized_newton_system,
    solve_matrix_free_newton_system,
    solve_sparse_newton_system,
    unpack_active_fields,
)


def test_active_region_from_slices_reports_shape_and_size() -> None:
    region = active_region_from_slices(
        (6, 7, 3),
        (slice(1, 6, 2), slice(None, None, 3), slice(None)),
    )

    assert region.slices == (slice(1, 6, 2), slice(None, None, 3), slice(None))
    assert region.shape == (3, 3, 3)
    assert region.size == 27


def test_active_region_from_slices_rejects_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="same rank"):
        active_region_from_slices((4, 5), (slice(None),))


def test_pack_unpack_active_fields_handles_empty_field_tuple() -> None:
    active = (slice(1, 2), slice(None))

    packed = pack_active_fields((), active_slices=active)
    restored = unpack_active_fields(packed, templates=(), active_slices=active)

    assert packed.dtype == np.float64
    assert packed.size == 0
    assert restored == ()


def test_unpack_active_fields_rejects_wrong_packed_size() -> None:
    active = (slice(1, 3), slice(None))
    template = np.zeros((4, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="Packed state has size 3, expected 4"):
        unpack_active_fields(
            np.ones(3, dtype=np.float64), templates=(template,), active_slices=active
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
        return set(
            sparsity.indices[sparsity.indptr[row] : sparsity.indptr[row + 1]].tolist()
        )

    center = np.ravel_multi_index((1, 1, 0), active_shape)
    periodic_neighbor = np.ravel_multi_index((1, 1, 3), active_shape)
    far_x_neighbor = np.ravel_multi_index((0, 0, 2), active_shape)
    columns = row_columns(center)

    assert periodic_neighbor in columns
    assert active_cells + periodic_neighbor in columns
    assert far_x_neighbor not in columns


def test_difference_quotient_step_size_scales_with_state_magnitude() -> None:
    base = np.sqrt(np.finfo(np.float64).eps)

    assert difference_quotient_step_size(0.25) == pytest.approx(base)
    assert difference_quotient_step_size(-4.0) == pytest.approx(4.0 * base)


def test_sparsity_and_coloring_reject_rank_mismatch() -> None:
    pytest.importorskip("scipy")

    with pytest.raises(ValueError, match="same rank"):
        build_locality_sparsity((3, 2), field_count=1, radii=(1,))
    with pytest.raises(ValueError, match="same rank"):
        build_modulo_color_groups((3, 2), field_count=1, color_periods=(2,))


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
        result1 = (
            field1
            + 0.3 * np.roll(field0, -1, axis=0)
            - 0.05 * np.roll(field1, -1, axis=2)
        )
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
        result1 = (
            field1
            + 0.3 * np.roll(field0, -1, axis=0)
            - 0.05 * np.roll(field1, -1, axis=2)
        )
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

    np.testing.assert_allclose(
        serial.toarray(), parallel.toarray(), rtol=1e-10, atol=1e-12
    )


def test_sparse_difference_quotient_plan_matches_unplanned_jacobian() -> None:
    pytest.importorskip("scipy")

    active_shape = (4, 1, 5)
    state = np.linspace(-0.3, 0.9, 2 * np.prod(active_shape))
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=2,
        radii=(1, 0, 1),
        periodic_axes=(2,),
    )
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=2,
        color_periods=(2, 1, 5),
    )
    plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity,
        color_groups=color_groups,
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        field0 = vector[: np.prod(active_shape)].reshape(active_shape)
        field1 = vector[np.prod(active_shape) :].reshape(active_shape)
        result0 = np.sin(field0) + 0.1 * np.roll(field1, 1, axis=2)
        result1 = field1 * field1 - 0.2 * np.roll(field0, -1, axis=0)
        return np.concatenate([result0.ravel(), result1.ravel()])

    unplanned = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
    )
    planned = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        difference_plan=plan,
    )

    assert plan.nnz == sparsity.nnz
    np.testing.assert_allclose(
        planned.toarray(), unplanned.toarray(), rtol=1.0e-12, atol=1.0e-12
    )


def test_sparse_jvp_jacobian_matches_grouped_jax_derivative() -> None:
    pytest.importorskip("scipy")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (4, 1, 5)
    state = np.linspace(-0.3, 0.9, 2 * np.prod(active_shape))
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=2,
        radii=(1, 0, 1),
        periodic_axes=(0, 2),
    )
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=2,
        color_periods=(4, 1, 5),
    )

    def residual(vector):
        field0 = vector[: np.prod(active_shape)].reshape(active_shape)
        field1 = vector[np.prod(active_shape) :].reshape(active_shape)
        result0 = jnp.sin(field0) + 0.1 * jnp.roll(field1, 1, axis=2)
        result1 = field1 * field1 - 0.2 * jnp.roll(field0, -1, axis=0)
        return jnp.concatenate([result0.ravel(), result1.ravel()])

    timing_payloads: list[dict[str, float | int]] = []
    jvp_jacobian = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        timing_callback=timing_payloads.append,
    )
    residual_jvp_jacobian, residual_value = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        return_residual=True,
    )
    plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity, color_groups=color_groups
    )
    prebuilt_direction_batches = prepare_sparse_jvp_direction_batches(
        difference_plan=plan,
        state_shape=tuple(state.shape),
    )
    prebuilt_jvp_jacobian = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        difference_plan=plan,
        direction_batches=prebuilt_direction_batches,
    )
    serial_jvp_jacobian = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        batch_size=1,
    )

    expected = np.zeros(sparsity.shape, dtype=np.float64)
    active_cells = int(np.prod(active_shape))
    for column in range(sparsity.shape[1]):
        direction = np.zeros_like(state)
        direction[column] = 1.0
        expected[:, column] = np.asarray(
            jax.jvp(residual, (jnp.asarray(state),), (jnp.asarray(direction),))[1],
            dtype=np.float64,
        )

    np.testing.assert_allclose(
        jvp_jacobian.toarray(),
        expected * sparsity.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        prebuilt_jvp_jacobian.toarray(),
        jvp_jacobian.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        serial_jvp_jacobian.toarray(),
        jvp_jacobian.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        residual_jvp_jacobian.toarray(),
        jvp_jacobian.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        residual_value,
        np.asarray(residual(state), dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert active_cells > 0
    assert sum(len(batch.groups) for batch in prebuilt_direction_batches) == len(
        color_groups
    )
    assert len(timing_payloads) == 1
    timing = timing_payloads[0]
    assert timing["group_count"] == len(color_groups)
    assert timing["batch_count"] == 1
    assert timing["state_size"] == state.size
    assert timing["nnz"] == sparsity.nnz
    assert timing["total_seconds"] >= timing["linearize_seconds"] >= 0.0
    assert timing["push_seconds"] >= 0.0
    assert timing["device_execute_seconds"] >= 0.0
    assert timing["host_transfer_seconds"] >= 0.0
    assert timing["push_seconds"] == pytest.approx(
        timing["device_execute_seconds"] + timing["host_transfer_seconds"]
    )
    assert timing["sync_timing"] in {0, 1}


def test_sparse_jvp_jacobian_reuses_prebuilt_device_gather_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (3, 1, 4)
    state = np.linspace(-0.2, 0.7, 2 * np.prod(active_shape))
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

    def residual(vector):
        field0 = vector[: np.prod(active_shape)].reshape(active_shape)
        field1 = vector[np.prod(active_shape) :].reshape(active_shape)
        result0 = jnp.cos(field0) + 0.25 * jnp.roll(field1, 1, axis=2)
        result1 = 0.5 * field1 - 0.1 * jnp.roll(field0, -1, axis=0)
        return jnp.concatenate([result0.ravel(), result1.ravel()])

    plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity,
        color_groups=color_groups,
    )
    prebuilt_direction_batches = prepare_sparse_jvp_direction_batches(
        difference_plan=plan,
        state_shape=tuple(state.shape),
        batch_size=2,
    )
    assert all(
        batch.gather_batch_indices_device is not None
        and batch.gather_rows_device is not None
        for batch in prebuilt_direction_batches
    )

    timing_payloads: list[dict[str, float | int]] = []
    default_gathered = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        difference_plan=plan,
        direction_batches=prebuilt_direction_batches,
        timing_callback=timing_payloads.append,
    )
    monkeypatch.setenv("JAX_DRB_SPARSE_JVP_GATHER_ON_DEVICE", "0")
    host_gathered = build_sparse_jvp_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        difference_plan=plan,
        direction_batches=prebuilt_direction_batches,
    )

    np.testing.assert_allclose(
        default_gathered.toarray(),
        host_gathered.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert len(timing_payloads) == 1
    assert timing_payloads[0]["gather_on_device"] == 1
    assert timing_payloads[0]["prebuilt_direction_batches"] == 1
    assert timing_payloads[0]["batch_count"] == len(prebuilt_direction_batches)


def test_backward_euler_and_bdf2_residual_formulas() -> None:
    state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    previous = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    previous_previous = np.array([0.25, 1.0, 2.0], dtype=np.float64)
    rhs = np.array([0.2, -0.1, 0.4], dtype=np.float64)

    be = backward_euler_residual(state, previous, rhs, timestep=0.5)
    bdf = bdf2_residual(state, previous, previous_previous, rhs, timestep=0.5)
    variable_bdf = bdf2_residual(
        state,
        previous,
        previous_previous,
        rhs,
        timestep=0.5,
        previous_timestep=0.25,
    )

    np.testing.assert_allclose(be, np.array([0.4, 0.55, 0.3]))
    np.testing.assert_allclose(bdf, np.array([0.35, 0.3666666666666667, 0.2]))
    np.testing.assert_allclose(variable_bdf, np.array([0.24, 0.13, -0.02]))


def test_bdf2_residual_rejects_nonpositive_previous_timestep() -> None:
    state = np.array([1.0], dtype=np.float64)
    previous = np.array([0.5], dtype=np.float64)
    previous_previous = np.array([0.25], dtype=np.float64)
    rhs = np.array([0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="previous_timestep must be positive"):
        bdf2_residual(
            state,
            previous,
            previous_previous,
            rhs,
            timestep=0.5,
            previous_timestep=0.0,
        )


def test_backward_euler_and_bdf2_residuals_preserve_jax_jvp() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    previous = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float64)
    previous_previous = jnp.array([0.25, 1.0, 2.0], dtype=jnp.float64)
    rhs = jnp.array([0.2, -0.1, 0.4], dtype=jnp.float64)

    def qoi(state):
        be = backward_euler_residual(state, previous, rhs, timestep=0.5)
        bdf = bdf2_residual(state, previous, previous_previous, rhs, timestep=0.5)
        return jnp.sum(be + bdf)

    value, tangent = jax.jvp(
        qoi,
        (jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),),
        (jnp.ones(3, dtype=jnp.float64),),
    )

    assert np.isfinite(float(value))
    assert tangent == pytest.approx(6.0)


def test_sparse_and_matrix_free_newton_solvers_recover_known_root() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(
        active_shape, field_count=1, color_periods=(3,)
    )

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
    assert sparse_info.converged is True
    assert matrix_free_info.converged is True
    assert sparse_info.residual_evaluation_count >= 1
    assert sparse_info.jacobian_refresh_count >= 1
    assert sparse_info.jacobian_assembly_seconds >= 0.0
    assert sparse_info.linear_solve_seconds >= 0.0
    assert sparse_info.linear_solver_backend == "scipy_gmres"
    assert sparse_info.linear_solver_tolerance == pytest.approx(1.0e-10)
    assert sparse_info.linear_solver_status == 0
    assert sparse_info.linear_solver_success is True
    assert sparse_info.linear_solver_reported_iterations is not None
    assert sparse_info.linear_solver_reported_iterations >= 0


def test_parallel_line_preconditioner_inverts_field_line_block() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (1, 3, 1)
    field_count = 2
    active_cells = int(np.prod(active_shape))
    field_unknown_count = field_count * active_cells
    state = jnp.zeros(field_unknown_count + 1, dtype=jnp.float64)
    line_block = jnp.asarray(
        [
            [4.0, -0.5, 0.0, 0.2, 0.0, 0.0],
            [-0.25, 5.0, -0.25, 0.0, 0.3, 0.0],
            [0.0, -0.5, 4.5, 0.0, 0.0, 0.1],
            [0.15, 0.0, 0.0, 3.0, -0.4, 0.0],
            [0.0, 0.1, 0.0, -0.2, 3.5, -0.2],
            [0.0, 0.0, 0.2, 0.0, -0.3, 3.25],
        ],
        dtype=jnp.float64,
    )

    def linear_map(vector):
        field_vector = vector[:field_unknown_count]
        feedback = vector[field_unknown_count:]
        return jnp.concatenate((line_block @ field_vector, feedback))

    preconditioner = implicit_mod._build_jax_linearized_parallel_line_preconditioner(
        linear_map,
        state,
        active_shape=active_shape,
        field_count=field_count,
        feedback_count=1,
        parallel_axis=1,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 0.25, -0.75, 1.5, 9.0], dtype=jnp.float64)
    solved = preconditioner(rhs)

    np.testing.assert_allclose(
        np.asarray(linear_map(solved)),
        np.asarray(rhs),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_parallel_line_preconditioner_can_target_selected_fields() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (1, 3)
    active_cells = int(np.prod(active_shape))
    field_count = 2
    field_unknown_count = field_count * active_cells
    state = jnp.zeros(field_unknown_count + 1, dtype=jnp.float64)
    selected_block = jnp.asarray(
        [
            [4.0, -0.5, 0.0],
            [-0.25, 5.0, -0.25],
            [0.0, -0.5, 4.5],
        ],
        dtype=jnp.float64,
    )

    def linear_map(vector):
        vector = jnp.asarray(vector, dtype=jnp.float64)
        field0 = vector[:active_cells]
        field1 = vector[active_cells:field_unknown_count]
        feedback = vector[field_unknown_count:]
        return jnp.concatenate((10.0 * field0, selected_block @ field1, feedback))

    preconditioner = implicit_mod._build_jax_linearized_parallel_line_preconditioner(
        linear_map,
        state,
        active_shape=active_shape,
        field_count=field_count,
        feedback_count=1,
        parallel_axis=1,
        field_indices=(1,),
    )
    rhs = jnp.asarray([7.0, 8.0, 9.0, 1.0, -2.0, 0.5, 4.0], dtype=jnp.float64)
    solved = preconditioner(rhs)

    np.testing.assert_allclose(
        np.asarray(solved[:active_cells]),
        np.asarray(rhs[:active_cells]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(linear_map(solved)[active_cells:field_unknown_count]),
        np.asarray(rhs[active_cells:field_unknown_count]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(solved[field_unknown_count:]),
        np.asarray(rhs[field_unknown_count:]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_parallel_line_preconditioner_batches_multiple_field_lines() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (2, 3)
    field_count = 1
    active_cells = int(np.prod(active_shape))
    line_indices = implicit_mod._field_major_line_indices(
        active_shape=active_shape,
        field_count=field_count,
        parallel_axis=1,
    )
    matrix = np.eye(active_cells, dtype=np.float64)
    for indices in line_indices:
        block = np.asarray(
            [
                [4.0, -0.5, 0.0],
                [-0.25, 5.0, -0.25],
                [0.0, -0.5, 4.5],
            ],
            dtype=np.float64,
        )
        matrix[np.ix_(indices, indices)] = block
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    state = jnp.zeros(active_cells, dtype=jnp.float64)

    def linear_map(vector):
        return matrix @ jnp.ravel(vector)

    preconditioner = implicit_mod._build_jax_linearized_parallel_line_preconditioner(
        linear_map,
        state,
        active_shape=active_shape,
        field_count=field_count,
        feedback_count=0,
        parallel_axis=1,
        max_batch_unknowns=3,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 0.25, -0.75, 1.5], dtype=jnp.float64)
    solved = preconditioner(rhs)

    np.testing.assert_allclose(
        np.asarray(linear_map(solved)),
        np.asarray(rhs),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_parallel_line_preconditioner_reduces_stiff_line_operator_budget() -> None:
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (1, 12)
    line_length = active_shape[1]
    line_matrix = np.diag(4.0 * np.ones(line_length))
    line_matrix += np.diag(-1.95 * np.ones(line_length - 1), k=1)
    line_matrix += np.diag(-1.95 * np.ones(line_length - 1), k=-1)
    line_matrix = jnp.asarray(line_matrix, dtype=jnp.float64)
    target = jnp.sin(jnp.linspace(0.0, 3.0, line_length, dtype=jnp.float64))

    def residual(state):
        vector = jnp.asarray(state, dtype=jnp.float64)
        return line_matrix @ (vector - target)

    common_solver_options = dict(
        active_shape=active_shape,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=2,
        linear_restart=2,
        linear_maxiter=1,
        linear_tolerance=1.0e-10,
        check_initial_residual=False,
        line_search_mode="full_step",
    )
    unpreconditioned_solution, unpreconditioned_info = (
        solve_jax_linearized_newton_system(
            residual,
            np.zeros(line_length, dtype=np.float64),
            **common_solver_options,
        )
    )
    preconditioned_solution, preconditioned_info = solve_jax_linearized_newton_system(
        residual,
        np.zeros(line_length, dtype=np.float64),
        **common_solver_options,
        linear_preconditioner_name="parallel_line",
        linear_preconditioner_context={
            "active_shape": active_shape,
            "field_count": 1,
            "feedback_count": 0,
            "parallel_axis": 1,
        },
    )

    assert unpreconditioned_info.converged is False
    assert unpreconditioned_info.residual_inf_norm > 1.0e-3
    assert preconditioned_info.converged is True
    assert preconditioned_info.residual_inf_norm < 1.0e-10
    assert preconditioned_info.linear_operator_call_count < (
        unpreconditioned_info.linear_operator_call_count
    )
    assert preconditioned_info.linear_preconditioner == "parallel_line"
    assert preconditioned_info.linear_preconditioner_build_count == 1
    assert preconditioned_info.linear_preconditioner_apply_count > 0
    np.testing.assert_allclose(
        preconditioned_solution,
        np.asarray(target),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert np.max(np.abs(unpreconditioned_solution - np.asarray(target))) > 1.0e-3


def test_parallel_line_preconditioner_rejects_mismatched_context() -> None:
    pytest.importorskip("jax.numpy")

    with pytest.raises(ValueError, match="context does not match packed state"):
        implicit_mod._build_jax_linearized_parallel_line_preconditioner(
            lambda vector: vector,
            np.ones(5, dtype=np.float64),
            active_shape=(1, 3, 1),
            field_count=2,
            feedback_count=0,
            parallel_axis=1,
        )

    with pytest.raises(ValueError, match="outside active_shape rank"):
        implicit_mod._build_jax_linearized_parallel_line_preconditioner(
            lambda vector: vector,
            np.ones(6, dtype=np.float64),
            active_shape=(1, 3, 1),
            field_count=2,
            feedback_count=0,
            parallel_axis=3,
        )


def test_sparse_newton_solver_supports_sparse_jvp_jacobian_mode() -> None:
    pytest.importorskip("scipy")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (3,)
    target = jnp.array([1.0, 0.5, 2.0], dtype=jnp.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(
        active_shape, field_count=1, color_periods=(3,)
    )

    def residual(state):
        return (
            jnp.asarray(state, dtype=jnp.float64)
            * jnp.asarray(state, dtype=jnp.float64)
            - target * target
        )

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
        jacobian_mode="jvp",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1e-9, atol=1e-9)
    assert info.residual_inf_norm < 1.0e-10
    assert info.jacobian_mode == "jvp"
    assert info.jvp_direction_batch_count > 0
    assert info.jvp_direction_build_seconds >= 0.0
    assert info.jvp_jacobian_batch_count >= info.jvp_direction_batch_count
    assert (
        info.jvp_jacobian_prebuilt_direction_batch_uses == info.jacobian_refresh_count
    )
    assert info.jvp_jacobian_tangent_build_seconds == pytest.approx(0.0)
    assert info.jvp_jacobian_total_seconds >= info.jvp_jacobian_linearize_seconds
    assert info.jvp_jacobian_push_seconds == pytest.approx(
        info.jvp_jacobian_device_execute_seconds
        + info.jvp_jacobian_host_transfer_seconds
    )
    assert info.jvp_jacobian_gather_on_device is True
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_tolerance == pytest.approx(1.0e-10)
    assert info.linear_solver_status == "ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 1
    assert info.residual_evaluation_count == info.nonlinear_iterations
    assert info.jacobian_refresh_count == info.nonlinear_iterations


def test_sparse_newton_solver_reuses_sparse_jvp_workspace() -> None:
    pytest.importorskip("scipy")
    jnp = pytest.importorskip("jax.numpy")

    active_shape = (3,)
    target = jnp.array([1.0, 0.5, 2.0], dtype=jnp.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(
        active_shape, field_count=1, color_periods=(3,)
    )
    workspace = prepare_sparse_jvp_workspace(
        sparsity=sparsity,
        color_groups=color_groups,
        state_shape=tuple(initial.shape),
    )

    def residual(state):
        return (
            jnp.asarray(state, dtype=jnp.float64)
            * jnp.asarray(state, dtype=jnp.float64)
            - target * target
        )

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
        jacobian_mode="jvp",
        sparse_jvp_workspace=workspace,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1e-9, atol=1e-9)
    assert info.jvp_direction_workspace_reuses == 1
    assert info.jvp_direction_batch_count == len(workspace.direction_batches)
    assert info.jvp_direction_build_seconds == pytest.approx(0.0)
    assert (
        info.jvp_jacobian_prebuilt_direction_batch_uses == info.jacobian_refresh_count
    )
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_status == "ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 1


def test_sparse_newton_solver_returns_immediately_when_initial_state_satisfies_residual() -> (
    None
):
    pytest.importorskip("scipy")

    initial = np.array([1.0, 2.0], dtype=np.float64)
    sparsity = build_locality_sparsity((2,), field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups((2,), field_count=1, color_periods=(2,))

    solution, info = solve_sparse_newton_system(
        lambda state: state - initial,
        initial,
        active_shape=(2,),
        sparsity=sparsity,
        color_groups=color_groups,
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
    )

    np.testing.assert_allclose(solution, initial)
    assert info.nonlinear_iterations == 0
    assert info.linear_iterations == 0


def test_sparse_newton_solver_uses_thread_count_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    import jax_drb.solver.implicit as implicit

    captured_workers: list[int] = []

    def fake_build_jacobian(*args, parallel_workers: int, **kwargs):
        captured_workers.append(parallel_workers)
        return scipy_sparse.eye(1, format="csr")

    monkeypatch.setenv("JAX_DRB_FD_JACOBIAN_THREADS", "3")
    monkeypatch.setattr(
        implicit, "build_sparse_difference_quotient_jacobian", fake_build_jacobian
    )

    solution, info = solve_sparse_newton_system(
        lambda state: state - np.array([1.0]),
        np.array([0.0]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, np.array([1.0]))
    assert captured_workers == [3]
    assert info.linear_iterations == 1
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_status == "ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 1


def test_sparse_newton_solver_falls_back_to_direct_solve_when_gmres_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    gmres_calls: list[int] = []

    def fake_gmres(matrix, rhs, *, callback=None, **kwargs):
        gmres_calls.append(1)
        if callback is not None:
            callback(1.0)
        return np.zeros_like(rhs), 1

    monkeypatch.setattr("scipy.sparse.linalg.gmres", fake_gmres)

    solution, info = solve_sparse_newton_system(
        lambda state: state - np.array([1.0]),
        np.array([0.0]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
    )

    np.testing.assert_allclose(solution, np.array([1.0]))
    assert gmres_calls == [1]
    assert info.linear_iterations == 2
    assert info.linear_solver_backend == "scipy_gmres_spsolve_fallback"
    assert info.linear_solver_status == "gmres_exit_1_spsolve_ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 2


def test_sparse_newton_solver_can_exit_on_step_tolerance() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    solution, info = solve_sparse_newton_system(
        lambda state: state - np.array([1.0]),
        np.array([0.9]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-30,
        step_tolerance=1.0,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, np.array([1.0]))
    assert info.nonlinear_iterations == 1
    assert info.residual_inf_norm <= 1.0e-12


def test_sparse_newton_solver_does_not_converge_on_stagnated_large_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    monkeypatch.setattr(
        "scipy.sparse.linalg.spsolve", lambda *args, **kwargs: np.array([0.0])
    )
    monkeypatch.setattr(
        "scipy.optimize.newton_krylov", lambda *args, **kwargs: np.array([0.0])
    )

    solution, info = solve_sparse_newton_system(
        lambda state: np.ones_like(state),
        np.array([0.0]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, np.array([0.0]))
    assert info.residual_inf_norm == pytest.approx(1.0)
    assert info.fallback_used is True
    assert info.converged is False


def test_sparse_newton_solver_uses_newton_krylov_fallback_after_rejected_line_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_optimize = pytest.importorskip("scipy.optimize")

    def fake_spsolve(*args, **kwargs):
        return np.array([1.0], dtype=np.float64)

    def fake_newton_krylov(*args, **kwargs):
        raise scipy_optimize.NoConvergence(np.array([0.25], dtype=np.float64))

    monkeypatch.setattr("scipy.sparse.linalg.spsolve", fake_spsolve)
    monkeypatch.setattr("scipy.optimize.newton_krylov", fake_newton_krylov)

    solution, info = solve_sparse_newton_system(
        lambda state: state + 1.0,
        np.array([0.0]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=2,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, np.array([0.25]))
    assert info.residual_inf_norm == pytest.approx(1.25)
    assert info.nonlinear_iterations == 2
    assert info.converged is False


def test_sparse_newton_solver_rejects_nonfinite_trial_then_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    monkeypatch.setattr(
        "scipy.sparse.linalg.spsolve", lambda *args, **kwargs: np.array([np.inf])
    )
    monkeypatch.setattr(
        "scipy.optimize.newton_krylov", lambda *args, **kwargs: np.array([0.5])
    )

    solution, info = solve_sparse_newton_system(
        lambda state: state - 1.0,
        np.array([0.0]),
        active_shape=(1,),
        sparsity=scipy_sparse.eye(1, format="csr"),
        color_groups=((0,),),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=1,
        linear_restart=2,
        linear_maxiter=4,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    np.testing.assert_allclose(solution, np.array([0.5]))
    assert info.residual_inf_norm == pytest.approx(0.5)
    assert info.nonlinear_iterations == 1
    assert info.converged is False
    assert info.fallback_used is True
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_status == "nonfinite"
    assert info.linear_solver_success is False
    assert info.linear_solver_reported_iterations == 1


def test_jax_linearized_newton_solver_returns_immediately_for_satisfied_residual() -> (
    None
):
    jnp = pytest.importorskip("jax.numpy")

    initial = np.array([1.0, 2.0], dtype=np.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - jnp.asarray(initial),
        initial,
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
    )

    np.testing.assert_allclose(solution, initial)
    assert info.nonlinear_iterations == 0
    assert info.linear_iterations == 0
    assert info.residual_evaluation_count == 1
    assert info.jacobian_refresh_count == 0
    assert info.converged is True


def test_jax_linearized_newton_solver_can_skip_initial_residual_check() -> None:
    jnp = pytest.importorskip("jax.numpy")

    initial = np.array([1.0, 2.0], dtype=np.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - jnp.asarray(initial),
        initial,
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        check_initial_residual=False,
    )

    np.testing.assert_allclose(solution, initial)
    assert info.nonlinear_iterations == 0
    assert info.linear_iterations == 0
    assert info.residual_evaluation_count == 1
    assert info.jacobian_refresh_count == 1
    assert info.check_initial_residual is False
    assert info.converged is True


def test_jax_linearized_newton_solver_can_check_initial_residual_by_linearizing() -> (
    None
):
    jnp = pytest.importorskip("jax.numpy")

    call_count = 0

    def residual(state):
        nonlocal call_count
        call_count += 1
        return jnp.asarray(state) * jnp.asarray(state) - 2.0

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([1.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=1,
        linear_restart=4,
        linear_maxiter=4,
        initial_residual_mode="linearize",
    )

    np.testing.assert_allclose(solution, np.array([1.5]), rtol=1.0e-12)
    assert info.converged is False
    assert info.nonlinear_iterations == 1
    assert info.residual_inf_norm == pytest.approx(0.25)
    assert info.check_initial_residual is True
    assert info.initial_residual_mode == "linearize"
    assert info.residual_evaluation_count == 2
    assert info.jacobian_refresh_count == 1
    assert call_count == 2


def test_jax_linearized_newton_solver_rejects_unknown_initial_residual_mode() -> (
    None
):
    jnp = pytest.importorskip("jax.numpy")

    with pytest.raises(ValueError, match="initial_residual_mode"):
        solve_jax_linearized_newton_system(
            lambda state: jnp.asarray(state),
            np.array([1.0], dtype=np.float64),
            active_shape=(1,),
            residual_tolerance=1.0e-12,
            step_tolerance=1.0e-12,
            max_nonlinear_iterations=1,
            initial_residual_mode="unknown",
        )


def test_jax_linearized_newton_solver_reuses_known_final_residual() -> None:
    jnp = pytest.importorskip("jax.numpy")

    call_count = 0

    def residual(state):
        nonlocal call_count
        call_count += 1
        return jnp.asarray(state) * jnp.asarray(state) - 2.0

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([1.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=1,
        linear_restart=4,
        linear_maxiter=4,
    )

    np.testing.assert_allclose(solution, np.array([1.5]), rtol=1.0e-12)
    assert info.converged is False
    assert info.nonlinear_iterations == 1
    assert info.residual_inf_norm == pytest.approx(0.25)
    assert info.residual_evaluation_count == 3
    assert call_count == 3


def test_jax_linearized_newton_solver_can_exit_on_step_tolerance() -> None:
    jnp = pytest.importorskip("jax.numpy")

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - jnp.array([1.0], dtype=jnp.float64),
        np.array([0.9], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-30,
        step_tolerance=1.0,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
    )

    np.testing.assert_allclose(solution, np.array([1.0]))
    assert info.nonlinear_iterations == 1
    assert info.residual_inf_norm <= 1.0e-12
    assert info.converged is True


def test_jax_linearized_newton_solver_does_not_converge_on_stagnated_large_residual() -> (
    None
):
    jnp = pytest.importorskip("jax.numpy")

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.ones_like(state),
        np.array([0.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
    )

    np.testing.assert_allclose(solution, np.array([0.0]))
    assert info.residual_inf_norm == pytest.approx(1.0)
    assert info.converged is False


def test_jax_linearized_newton_solver_recovers_known_root() -> None:
    pytest.importorskip("jax")
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
    assert info.residual_evaluation_count >= 1
    assert info.jacobian_refresh_count >= 1
    assert info.jacobian_assembly_seconds >= 0.0
    assert info.linear_solve_seconds >= 0.0
    assert info.converged is True
    assert info.linear_solver_backend == "jax_gmres"
    assert info.linear_solver_tolerance == pytest.approx(1.0e-10)
    assert info.linear_solver_status == 0
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations is None
    assert info.linear_operator_call_count > 0
    assert info.linear_iterations == info.linear_operator_call_count


def test_jax_linearized_newton_solver_reports_custom_linear_tolerance() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0, -0.5], dtype=jnp.float64)

    def residual(state):
        return jnp.asarray(state) - target

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([0.0, 0.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        linear_tolerance=1.0e-5,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=3,
        linear_restart=4,
        linear_maxiter=4,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_solver_tolerance == pytest.approx(1.0e-5)


def test_jax_linearized_newton_solver_reports_gmres_solve_method() -> None:
    jnp = pytest.importorskip("jax.numpy")

    assert implicit_mod._resolve_jax_gmres_solve_method("incremental") == "incremental"
    assert implicit_mod._resolve_jax_gmres_solve_method("givens") == "incremental"
    assert implicit_mod._resolve_jax_gmres_solve_method("unknown") == "batched"

    target = jnp.array([1.0, -0.5], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0, 0.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=3,
        linear_restart=4,
        linear_maxiter=4,
        linear_solver_solve_method="incremental",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_solver_backend == "jax_gmres"
    assert info.linear_solver_solve_method == "incremental"


def test_jax_linearized_newton_solver_accepts_left_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0, 2.0], dtype=jnp.float64)
    weights = jnp.array([1000.0, 0.01], dtype=jnp.float64)

    def residual(state):
        return weights * (jnp.asarray(state) - target)

    def preconditioner(vector):
        return jnp.asarray(vector) / weights

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([0.5, 1.5], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        linear_preconditioner=preconditioner,
        linear_preconditioner_name="test_scale",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_preconditioner == "test_scale"
    assert info.linear_preconditioner_apply_count >= 1
    assert info.linear_preconditioner_apply_seconds >= 0.0


def test_jax_linearized_newton_solver_builds_linearized_diagonal_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0, -2.0], dtype=jnp.float64)
    weights = jnp.array([1000.0, -0.25], dtype=jnp.float64)

    def residual(state):
        return weights * (jnp.asarray(state) - target)

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([0.0, 0.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        linear_preconditioner_name="linearized_diag",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_preconditioner == "linearized_diag"
    assert info.linear_preconditioner_build_count >= 1
    assert info.linear_preconditioner_build_seconds >= 0.0


def test_field_diagonal_preconditioner_scales_active_field_block_only() -> None:
    jnp = pytest.importorskip("jax.numpy")

    active_cell_count = 2
    field_count = 2
    feedback_count = 1
    field_unknown_count = active_cell_count * field_count
    diagonal = jnp.asarray([4.0, -2.0, 8.0, 0.5, 99.0], dtype=jnp.float64)
    state = jnp.zeros(field_unknown_count + feedback_count, dtype=jnp.float64)

    def linear_map(vector):
        return diagonal * jnp.asarray(vector, dtype=jnp.float64)

    preconditioner = (
        implicit_mod._build_jax_linearized_field_diagonal_preconditioner(
            linear_map,
            state,
            active_cell_count=active_cell_count,
            field_count=field_count,
            feedback_count=feedback_count,
        )
    )
    rhs = jnp.asarray([8.0, -6.0, 4.0, 2.0, 10.0], dtype=jnp.float64)
    solved = preconditioner(rhs)

    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray([2.0, 3.0, 0.5, 4.0, 10.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_field_sample_diagonal_preconditioner_uses_one_sample_per_field() -> None:
    jnp = pytest.importorskip("jax.numpy")

    active_cell_count = 3
    field_count = 2
    feedback_count = 1
    field_unknown_count = active_cell_count * field_count
    diagonal = jnp.asarray([10.0, 4.0, 20.0, 30.0, -2.0, 40.0, 99.0])
    state = jnp.zeros(field_unknown_count + feedback_count, dtype=jnp.float64)

    def linear_map(vector):
        return diagonal * jnp.asarray(vector, dtype=jnp.float64)

    preconditioner = (
        implicit_mod._build_jax_linearized_field_sample_diagonal_preconditioner(
            linear_map,
            state,
            active_cell_count=active_cell_count,
            field_count=field_count,
            feedback_count=feedback_count,
        )
    )
    rhs = jnp.asarray([8.0, 12.0, 16.0, 4.0, 6.0, 8.0, 10.0], dtype=jnp.float64)
    solved = preconditioner(rhs)

    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray([2.0, 3.0, 4.0, -2.0, -3.0, -4.0, 10.0]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_jax_linearized_newton_solver_builds_field_sample_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.asarray([1.0, -2.0, 0.5, 3.0, 0.25], dtype=jnp.float64)
    weights = jnp.asarray([4.0, 4.0, 2.0, 2.0, 1.0], dtype=jnp.float64)

    def residual(state):
        return weights * (jnp.asarray(state, dtype=jnp.float64) - target)

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.zeros(5, dtype=np.float64),
        active_shape=(5,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=5,
        linear_maxiter=5,
        linear_preconditioner_name="field_sample_diag",
        linear_preconditioner_context={
            "active_cell_count": 2,
            "field_count": 2,
            "feedback_count": 1,
            "sample_cell_index": 0,
        },
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_preconditioner == "field_sample_diag"
    assert info.linear_preconditioner_build_count >= 1
    assert info.linear_preconditioner_build_seconds >= 0.0


def test_dynamic_linearized_diag_preconditioner_honors_context() -> None:
    jnp = pytest.importorskip("jax.numpy")

    diagonal = jnp.asarray([1.0e-12, 4.0], dtype=jnp.float64)
    state = jnp.zeros(2, dtype=jnp.float64)

    def linear_map(vector):
        return diagonal * jnp.asarray(vector, dtype=jnp.float64)

    preconditioner = implicit_mod._build_jax_linearized_dynamic_preconditioner(
        "linearized_diag",
        linear_map,
        state,
        context={"floor": 0.5, "max_unknowns": 2},
    )

    np.testing.assert_allclose(
        np.asarray(preconditioner(jnp.asarray([1.0, 8.0], dtype=jnp.float64))),
        np.asarray([2.0, 2.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="at most 1 unknowns"):
        implicit_mod._build_jax_linearized_dynamic_preconditioner(
            "linearized_diag",
            linear_map,
            state,
            context={"max_unknowns": 1},
        )


def test_jax_linearized_newton_solver_builds_field_diagonal_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.asarray([1.0, -2.0, 0.5, 3.0, 0.25], dtype=jnp.float64)
    weights = jnp.asarray([4.0, 2.0, 8.0, 0.5, 1.0], dtype=jnp.float64)

    def residual(state):
        return weights * (jnp.asarray(state, dtype=jnp.float64) - target)

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.zeros(5, dtype=np.float64),
        active_shape=(5,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=5,
        linear_maxiter=5,
        linear_preconditioner_name="field_diag",
        linear_preconditioner_context={
            "active_cell_count": 2,
            "field_count": 2,
            "feedback_count": 1,
        },
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_preconditioner == "field_diag"
    assert info.linear_preconditioner_build_count >= 1
    assert info.linear_preconditioner_build_seconds >= 0.0


def test_jax_linearized_newton_solver_builds_local_block_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    # Packed field-major state: [field0 cells..., field1 cells...].
    blocks = jnp.array(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[5.0, -1.0], [1.0, 2.0]],
        ],
        dtype=jnp.float64,
    )
    target_by_cell = jnp.array([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float64)
    target = np.asarray(target_by_cell.T.reshape((-1,)))

    def residual(state):
        by_cell = jnp.asarray(state).reshape((2, 2)).T
        local_residual = jnp.einsum("cij,cj->ci", blocks, by_cell - target_by_cell)
        return local_residual.T.reshape((-1,))

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.zeros(4, dtype=np.float64),
        active_shape=(4,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=2,
        linear_maxiter=4,
        linear_preconditioner_name="local_block_diag",
        linear_preconditioner_context={
            "active_cell_count": 2,
            "field_count": 2,
            "feedback_count": 0,
        },
    )

    np.testing.assert_allclose(solution, target, rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.linear_preconditioner == "local_block_diag"
    assert info.linear_preconditioner_build_count >= 1
    assert info.linear_preconditioner_build_seconds >= 0.0


def test_jax_linearized_newton_solver_reuses_dynamic_preconditioner() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0, -2.0], dtype=jnp.float64)
    weights = jnp.array([4.0, 2.0], dtype=jnp.float64)

    def residual(state):
        # Smooth nonlinear residual requires more than one Newton update from
        # the deliberately poor initial state.
        delta = jnp.asarray(state) - target
        return weights * delta + 0.1 * delta * delta

    solution, info = solve_jax_linearized_newton_system(
        residual,
        np.array([4.0, 3.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=6,
        linear_restart=2,
        linear_maxiter=4,
        linear_preconditioner_name="linearized_diag",
        linear_preconditioner_context={"refresh_frequency": 100},
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-7, atol=1.0e-7)
    assert info.converged is True
    assert info.nonlinear_iterations > 1
    assert info.linear_preconditioner_build_count == 1


def test_jax_linearized_newton_solver_supports_bicgstab_backend() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([0.75, 1.25], dtype=jnp.float64)
    initial = np.array([0.5, 1.0], dtype=np.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        initial,
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_solver_backend="jax_bicgstab",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.residual_inf_norm < 1.0e-12
    assert info.jacobian_mode == "jax_linearized:jax_bicgstab"
    assert info.linear_solver_backend == "jax_bicgstab"
    assert info.linear_solver_success is None
    assert info.linear_iterations == info.linear_operator_call_count


def test_jax_linearized_newton_solver_can_prejit_residual() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([0.25, 1.5], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0, 1.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        jit_residual=True,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.residual_inf_norm < 1.0e-12
    assert info.converged is True
    assert info.residual_jitted is True
    assert info.jacobian_mode == "jax_linearized:jax_gmres"


def test_jax_linearized_newton_solver_can_jit_linear_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    jit_call_count = 0
    original_jit = jax.jit

    def fake_jit(fn):
        nonlocal jit_call_count
        jit_call_count += 1
        return original_jit(fn)

    monkeypatch.setattr(jax, "jit", fake_jit)
    target = jnp.array([0.75, 1.25], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.5, 1.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        jit_linear_operator=True,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.residual_inf_norm < 1.0e-12
    assert info.converged is True
    assert info.linear_operator_jitted is True
    assert jit_call_count >= 1


def test_jax_linearized_newton_solver_reports_line_search_damping() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=1,
        linear_restart=4,
        linear_maxiter=4,
        line_search_initial_step_scale=0.5,
    )

    np.testing.assert_allclose(solution, np.array([0.5]), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is False
    assert info.residual_inf_norm == pytest.approx(0.5)
    assert info.line_search_initial_step_scale == pytest.approx(0.5)
    assert info.line_search_mode == "backtracking"
    assert info.line_search_last_step_scale == pytest.approx(0.5)
    assert info.line_search_trial_count == 1
    assert info.linear_operator_call_count > 0
    assert info.linear_operator_dispatch_seconds >= 0.0


def test_jax_linearized_newton_solver_full_step_skips_trial_residual() -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        check_initial_residual=False,
        line_search_mode="full_step",
    )

    np.testing.assert_allclose(solution, np.array([1.0]), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.residual_inf_norm < 1.0e-12
    assert info.line_search_mode == "full_step"
    assert info.line_search_last_step_scale == pytest.approx(1.0)
    assert info.line_search_trial_count == 1
    assert info.residual_evaluation_count == 2


def test_jax_linearized_newton_solver_rejects_bad_line_search_mode() -> None:
    jnp = pytest.importorskip("jax.numpy")

    with pytest.raises(ValueError, match="line_search_mode"):
        solve_jax_linearized_newton_system(
            lambda state: jnp.asarray(state),
            np.array([0.0], dtype=np.float64),
            active_shape=(1,),
            residual_tolerance=1.0e-12,
            step_tolerance=1.0e-12,
            max_nonlinear_iterations=1,
            line_search_mode="bad",
        )


@pytest.mark.parametrize("bad_scale", [float("nan"), -0.5, 0.0])
def test_jax_linearized_newton_solver_ignores_invalid_line_search_scale(
    bad_scale: float,
) -> None:
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([1.0], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=1,
        linear_restart=4,
        linear_maxiter=4,
        line_search_initial_step_scale=bad_scale,
    )

    np.testing.assert_allclose(solution, np.array([1.0]), rtol=1.0e-12, atol=1.0e-12)
    assert info.line_search_initial_step_scale == pytest.approx(1.0)
    assert info.line_search_last_step_scale == pytest.approx(1.0)


def test_jax_linearized_newton_solver_uses_prejit_residual_in_line_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    jit_call_count = 0
    original_jit = jax.jit

    def fake_jit(fn):
        compiled = original_jit(fn)

        def wrapped(*args, **kwargs):
            nonlocal jit_call_count
            jit_call_count += 1
            return compiled(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(jax, "jit", fake_jit)
    target = jnp.array([0.25, 1.5], dtype=jnp.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        np.array([0.0, 1.0], dtype=np.float64),
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_restart=4,
        linear_maxiter=4,
        jit_residual=True,
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.converged is True
    assert info.residual_jitted is True
    assert jit_call_count >= 3


def test_jax_linearized_newton_solver_reports_nonconvergence() -> None:
    jnp = pytest.importorskip("jax.numpy")

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) + 1.0,
        np.array([0.0], dtype=np.float64),
        active_shape=(1,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=0,
    )

    np.testing.assert_allclose(solution, np.array([0.0]))
    assert info.residual_inf_norm == pytest.approx(1.0)
    assert info.converged is False


def test_jax_linearized_newton_solver_supports_lineax_gmres_backend() -> None:
    pytest.importorskip("lineax")
    jnp = pytest.importorskip("jax.numpy")

    target = jnp.array([0.75, 1.25], dtype=jnp.float64)
    initial = np.array([0.5, 1.0], dtype=np.float64)

    solution, info = solve_jax_linearized_newton_system(
        lambda state: jnp.asarray(state) - target,
        initial,
        active_shape=(2,),
        residual_tolerance=1.0e-12,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=4,
        linear_solver_backend="lineax_gmres",
    )

    np.testing.assert_allclose(solution, np.asarray(target), rtol=1.0e-12, atol=1.0e-12)
    assert info.residual_inf_norm < 1.0e-12
    assert info.jacobian_mode == "jax_linearized:lineax_gmres"
    assert info.linear_solver_backend == "lineax_gmres"
    assert info.linear_solver_success is True
    assert (
        info.linear_solver_reported_iterations is None
        or info.linear_solver_reported_iterations >= 0
    )


def test_sparse_newton_solver_supports_direct_linear_solve_mode() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(
        active_shape, field_count=1, color_periods=(3,)
    )

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
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_tolerance == pytest.approx(1.0e-10)
    assert info.linear_solver_status == "ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 1


def test_sparse_newton_solver_supports_jacobian_reuse() -> None:
    pytest.importorskip("scipy")

    active_shape = (3,)
    target = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    initial = np.array([0.8, 0.7, 1.7], dtype=np.float64)
    sparsity = build_locality_sparsity(active_shape, field_count=1, radii=(0,))
    color_groups = build_modulo_color_groups(
        active_shape, field_count=1, color_periods=(3,)
    )

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
    assert info.linear_solver_backend == "scipy_spsolve"
    assert info.linear_solver_status == "ok"
    assert info.linear_solver_success is True
    assert info.linear_solver_reported_iterations == 1
