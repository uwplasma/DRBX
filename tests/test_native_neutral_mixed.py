from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.neutral_mixed import (
    _apply_density_y_boundaries,
    _div_a_grad_perp_flows,
    _div_par_k_grad_par_open,
    _gradient_magnitude,
    _grad_par_open,
    advance_neutral_mixed_implicit_history,
    _prepare_neutral_mixed_state,
    _soft_floor,
    advance_neutral_mixed_bdf2_step,
    advance_neutral_mixed_backward_euler_step,
    build_neutral_mixed_active_jacobian_color_groups,
    build_neutral_mixed_active_jacobian_sparsity,
    build_neutral_mixed_sparse_residual_jacobian,
    compute_neutral_mixed_bdf2_residual,
    build_neutral_mixed_transport_operators,
    compute_neutral_mixed_backward_euler_residual,
    compute_neutral_mixed_rhs,
    initialize_neutral_mixed_state,
    pack_neutral_mixed_active_state,
    unpack_neutral_mixed_active_state,
)
from jax_drb.native.runner import _execute_neutral_mixed_case
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_ROOT = _REPO_ROOT / "references" / "baselines"


def _neutral_mixed_input(*, nx: int = 10, ny: int = 10, nz: int = 10) -> str:
    return f"""
nout = 15
timestep = 20

[mesh]
nx = {nx}
ny = {ny}
nz = {nz}

dx = 1e-3
dy = 1e-3
dz = 1e-3

yn = y / (2π)
zn = z / (2π)

J = 1

[solver]
mxstep = 1000

[model]
components = h

[h]
type = neutral_mixed

[Nh]
function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)

[Ph]
function = 0.1 * Nh:function
"""


def _build_case(*, nx: int = 10, ny: int = 10, nz: int = 10):
    config = parse_bout_input(_neutral_mixed_input(nx=nx, ny=ny, nz=nz))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    state = initialize_neutral_mixed_state(config, section="h", mesh=mesh)
    scalars = resolved_dataset_scalars(run_config)
    rhs = compute_neutral_mixed_rhs(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    return config, run_config, mesh, metrics, state, rhs


def _build_small_implicit_case():
    return _build_case(nx=8, ny=4, nz=6)


def test_neutral_mixed_diffusion_matches_known_case_values() -> None:
    _, _, _, _, _, rhs = _build_case()
    interior = rhs.diffusion[2:8, 2:12, :]

    assert interior.min() == pytest.approx(0.4208780853793485, rel=1e-12, abs=1e-12)
    assert interior.max() == pytest.approx(1.4907267304514251, rel=1e-12, abs=1e-12)
    assert rhs.diffusion[5, 5, 5] == pytest.approx(1.170446626082471, rel=1e-12, abs=1e-12)


def test_neutral_mixed_rhs_tracks_reference_case_center_values() -> None:
    config, run_config, mesh, metrics, state, rhs = _build_case()
    expected = np.load(_BASELINE_ROOT / "reference_arrays" / "neutral_mixed_rhs.npz")
    expected_density = expected["var__Nh"][0]
    expected_pressure = expected["var__Ph"][0]
    expected_momentum = expected["var__NVh"][0]
    expected_density_rhs = expected["var__ddt(Nh)"][0]
    expected_pressure_rhs = expected["var__ddt(Ph)"][0]
    expected_momentum_rhs = expected["var__ddt(NVh)"][0]
    scalars = resolved_dataset_scalars(run_config)
    prepared = _prepare_neutral_mixed_state(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )

    assert state.density[5, 5, 5] == pytest.approx(expected_density[5, 3, 5], rel=1e-12, abs=1e-12)
    assert state.pressure[5, 5, 5] == pytest.approx(expected_pressure[5, 3, 5], rel=1e-12, abs=1e-12)
    assert state.momentum[5, 5, 5] == pytest.approx(expected_momentum[5, 3, 5], rel=1e-12, abs=1e-12)
    assert rhs.density[5, 5, 5] == pytest.approx(expected_density_rhs[5, 3, 5], rel=3e-2, abs=2e-4)
    assert rhs.pressure[5, 5, 5] == pytest.approx(expected_pressure_rhs[5, 3, 5], rel=3e-2, abs=2e-4)
    assert rhs.momentum[5, 5, 5] == pytest.approx(expected_momentum_rhs[5, 3, 5], rel=1e-12, abs=1e-12)
    assert prepared.temperature[5, 5, 5] == pytest.approx(0.1, rel=1e-12, abs=1e-12)


def test_neutral_mixed_rhs_term_decomposition_sums_to_rhs() -> None:
    _, _, _, _, _, rhs = _build_case()

    density_sum = sum(rhs.density_terms.values(), np.zeros_like(rhs.density))
    pressure_sum = sum(rhs.pressure_terms.values(), np.zeros_like(rhs.pressure))
    momentum_sum = sum(rhs.momentum_terms.values(), np.zeros_like(rhs.momentum))

    assert set(rhs.momentum_terms) == {
        "parallel_inertia",
        "pressure_gradient",
        "perpendicular_diffusion",
        "parallel_viscosity",
        "perpendicular_viscosity",
    }
    np.testing.assert_allclose(density_sum, rhs.density, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(pressure_sum, rhs.pressure, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(momentum_sum, rhs.momentum, rtol=1e-14, atol=1e-14)


def test_neutral_mixed_rhs_matches_compact_reference_diagnostics() -> None:
    config, run_config, mesh, metrics, state, rhs = _build_case()
    scalars = resolved_dataset_scalars(run_config)
    prepared = _prepare_neutral_mixed_state(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    with (_BASELINE_ROOT / "reference_metrics" / "neutral_mixed_rhs_diagnostics.json").open() as handle:
        payload = json.load(handle)
    probe = payload["probe"]
    y_slice = slice(probe["y_start"], probe["y_end"] + 1)

    np.testing.assert_allclose(state.density[5, :, 5], np.asarray(payload["density_centerline"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.pressure[5, :, 5], np.asarray(payload["pressure_centerline"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        rhs.density_parallel_flow[5, y_slice, 5],
        np.asarray(payload["density_parallel_flow_active"]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rhs.pressure_parallel_flow[5, y_slice, 5],
        np.asarray(payload["pressure_parallel_flow_active"]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        -(_div_a_grad_perp_flows(
            prepared.diffusion_density,
            prepared.log_pressure,
            mesh=mesh,
            metrics=metrics,
        )[5, y_slice, 5] - rhs.density[5, y_slice, 5]),
        np.asarray(payload["density_parallel_rhs_active"]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert prepared.sound_speed[5, 5, 5] == pytest.approx(payload["sound_speed_center"], rel=1e-12, abs=1e-12)
    assert metrics.g22[5, 5, 5] == pytest.approx(payload["g22_center"], rel=1e-12, abs=1e-18)
    assert metrics.g_22[5, 5, 5] == pytest.approx(payload["g_22_center"], rel=1e-12, abs=1e-9)


def test_neutral_mixed_rhs_keeps_guard_derivatives_zero_in_x() -> None:
    _, _, _, _, _, rhs = _build_case()

    assert rhs.density[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.density[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.pressure[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.pressure[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.momentum[:2].max() == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert rhs.momentum[8:].max() == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_neutral_mixed_transport_operators_reproduce_frozen_density_rhs() -> None:
    config, run_config, mesh, metrics, state, rhs = _build_case()
    scalars = resolved_dataset_scalars(run_config)
    prepared = _prepare_neutral_mixed_state(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    operators = build_neutral_mixed_transport_operators(
        rhs.diffusion,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )

    active_density_rhs = _div_a_grad_perp_flows(
        prepared.diffusion_density,
        prepared.log_pressure,
        mesh=mesh,
        metrics=metrics,
    )[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :]
    for j_offset, operator in enumerate(operators):
        density_slice = prepared.density_limited[mesh.xstart : mesh.xend + 1, mesh.ystart + j_offset, :]
        actual = (operator @ density_slice.reshape(-1)).reshape(density_slice.shape)
        expected = active_density_rhs[:, j_offset, :]
        error = actual - expected
        assert np.max(np.abs(error)) < 3.5e-3
        assert np.sqrt(np.mean(np.square(error))) < 1.0e-3


def test_div_a_grad_perp_flows_matches_reference_loop() -> None:
    _, _, mesh, metrics, state, _ = _build_case(nx=8, ny=4, nz=6)
    coefficient = 0.5 + 0.1 * state.density
    field = np.log1p(state.pressure)

    expected = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)
    for i in range(mesh.xstart - 1, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                x_flux = (
                    0.5
                    * (coefficient[i, j, k] + coefficient[i + 1, j, k])
                    * (J[i, j, k] * g11[i, j, k] + J[i + 1, j, k] * g11[i + 1, j, k])
                    * (field[i + 1, j, k] - field[i, j, k])
                    / (dx[i, j, k] + dx[i + 1, j, k])
                )
                expected[i, j, k] += x_flux / (dx[i, j, k] * J[i, j, k])
                expected[i + 1, j, k] -= x_flux / (dx[i + 1, j, k] * J[i + 1, j, k])

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                z_flux = (
                    0.25
                    * (coefficient[i, j, k] + coefficient[i, j, kp])
                    * (J[i, j, k] * g33[i, j, k] + J[i, j, kp] * g33[i, j, kp])
                    * ((field[i, j, kp] - field[i, j, k]) / dz[i, j, k])
                )
                expected[i, j, k] += z_flux / (J[i, j, k] * dz[i, j, k])
                expected[i, j, kp] -= z_flux / (J[i, j, kp] * dz[i, j, kp])

    actual = _div_a_grad_perp_flows(
        coefficient,
        field,
        mesh=mesh,
        metrics=metrics,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_gradient_magnitude_matches_reference_loop() -> None:
    _, _, mesh, metrics, state, _ = _build_case(nx=8, ny=4, nz=6)
    field = np.asarray(state.pressure, dtype=np.float64)

    expected = np.zeros_like(field, dtype=np.float64)
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
                expected[i, j, k] = np.sqrt(
                    g11[i, j, k] * dfdx * dfdx
                    + g33[i, j, k] * dfdz * dfdz
                    + np.square(dfdy / J[i, j, k])
                )

    actual = _gradient_magnitude(
        field,
        mesh=mesh,
        metrics=metrics,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("boundary_flux", [False, True])
def test_div_par_k_grad_par_open_matches_reference_loop(boundary_flux: bool) -> None:
    _, _, mesh, metrics, state, _ = _build_case(nx=8, ny=4, nz=6)
    coefficient = 0.5 + 0.1 * state.density
    field = np.log1p(state.pressure)

    expected = np.zeros_like(field, dtype=np.float64)
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
            gradient_up = 2.0 * (
                field[mesh.xstart : mesh.xend + 1, j + 1, :] - field[mesh.xstart : mesh.xend + 1, j, :]
            ) / (
                dy[mesh.xstart : mesh.xend + 1, j, :] + dy[mesh.xstart : mesh.xend + 1, j + 1, :]
            )
            flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
            expected[mesh.xstart : mesh.xend + 1, j, :] += flux_up / (
                dy[mesh.xstart : mesh.xend + 1, j, :] * J[mesh.xstart : mesh.xend + 1, j, :]
            )

        if boundary_flux or j != mesh.ystart:
            coefficient_down = 0.5 * (
                coefficient[mesh.xstart : mesh.xend + 1, j, :] + coefficient[mesh.xstart : mesh.xend + 1, j - 1, :]
            )
            jacobian_down = 0.5 * (J[mesh.xstart : mesh.xend + 1, j, :] + J[mesh.xstart : mesh.xend + 1, j - 1, :])
            metric_down = 0.5 * (g22[mesh.xstart : mesh.xend + 1, j, :] + g22[mesh.xstart : mesh.xend + 1, j - 1, :])
            gradient_down = 2.0 * (
                field[mesh.xstart : mesh.xend + 1, j, :] - field[mesh.xstart : mesh.xend + 1, j - 1, :]
            ) / (
                dy[mesh.xstart : mesh.xend + 1, j, :] + dy[mesh.xstart : mesh.xend + 1, j - 1, :]
            )
            flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
            expected[mesh.xstart : mesh.xend + 1, j, :] -= flux_down / (
                dy[mesh.xstart : mesh.xend + 1, j, :] * J[mesh.xstart : mesh.xend + 1, j, :]
            )

    actual = _div_par_k_grad_par_open(
        coefficient,
        field,
        mesh=mesh,
        metrics=metrics,
        boundary_flux=boundary_flux,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_neutral_mixed_active_state_round_trip_preserves_interior() -> None:
    _, _, mesh, _, state, _ = _build_case()
    packed = pack_neutral_mixed_active_state(state, mesh=mesh)
    unpacked = unpack_neutral_mixed_active_state(packed, template=state, mesh=mesh)

    np.testing.assert_allclose(unpacked.density[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :], state.density[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :])
    np.testing.assert_allclose(unpacked.pressure[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :], state.pressure[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :])
    np.testing.assert_allclose(unpacked.momentum[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :], state.momentum[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :])


def test_neutral_mixed_backward_euler_step_solves_active_residual() -> None:
    pytest.importorskip("scipy")

    config, run_config, mesh, metrics, state, _ = _build_case()
    scalars = resolved_dataset_scalars(run_config)
    previous = pack_neutral_mixed_active_state(state, mesh=mesh)
    initial_residual = compute_neutral_mixed_backward_euler_residual(
        previous,
        previous,
        config=config,
        template_state=state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=20.0,
    )

    stepped, info = advance_neutral_mixed_backward_euler_step(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=20.0,
    )
    solved = pack_neutral_mixed_active_state(stepped, mesh=mesh)
    solved_residual = compute_neutral_mixed_backward_euler_residual(
        solved,
        previous,
        config=config,
        template_state=state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=20.0,
    )

    assert np.max(np.abs(initial_residual)) > 1.0e-1
    assert info.residual_inf_norm < 1.0e-8
    assert np.max(np.abs(solved_residual)) < 1.0e-8
    assert np.all(np.isfinite(stepped.density))
    assert np.all(np.isfinite(stepped.pressure))
    assert np.all(np.isfinite(stepped.momentum))
    assert info.nonlinear_iterations >= 1
    assert info.linear_iterations >= info.nonlinear_iterations
    assert info.diagnostics["residual_evaluation_count"] >= 1


def test_neutral_mixed_sparse_backward_euler_step_solves_active_residual() -> None:
    pytest.importorskip("scipy")

    config, run_config, mesh, metrics, state, _ = _build_small_implicit_case()
    scalars = resolved_dataset_scalars(run_config)
    previous = pack_neutral_mixed_active_state(state, mesh=mesh)

    stepped, info = advance_neutral_mixed_backward_euler_step(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=5.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        step_tolerance=1.0e-10,
        max_nonlinear_iterations=8,
        linear_restart=10,
        linear_maxiter=60,
        linear_rtol=1.0e-9,
    )
    solved = pack_neutral_mixed_active_state(stepped, mesh=mesh)
    solved_residual = compute_neutral_mixed_backward_euler_residual(
        solved,
        previous,
        config=config,
        template_state=state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=5.0,
    )

    assert info.residual_inf_norm < 1.0e-7
    assert np.max(np.abs(solved_residual)) < 1.0e-7
    assert np.all(np.isfinite(stepped.density))
    assert np.all(np.isfinite(stepped.pressure))
    assert np.all(np.isfinite(stepped.momentum))
    assert info.diagnostics["residual_evaluation_count"] >= 1
    assert info.diagnostics["jacobian_refresh_count"] >= 1
    assert info.diagnostics["jacobian_assembly_seconds"] >= 0.0


def test_neutral_mixed_active_jacobian_sparsity_matches_local_stencil() -> None:
    pytest.importorskip("scipy")

    _, _, mesh, _, _, _ = _build_small_implicit_case()
    sparsity = build_neutral_mixed_active_jacobian_sparsity(mesh)
    active_nx = mesh.xend - mesh.xstart + 1
    active_ny = mesh.yend - mesh.ystart + 1
    active_cells = active_nx * active_ny * mesh.nz

    def active_index(ix: int, iy: int, iz: int) -> int:
        return ((ix * active_ny) + iy) * mesh.nz + iz

    def row_columns(row: int) -> set[int]:
        return set(sparsity.indices[sparsity.indptr[row] : sparsity.indptr[row + 1]].tolist())

    assert sparsity.shape == (3 * active_cells, 3 * active_cells)

    interior_row = active_index(1, 2, 3)
    interior_columns = row_columns(interior_row)
    required_interior = set()
    for variable_block in range(3):
        base = variable_block * active_cells
        for neighbor in (
            active_index(1, 2, 3),
            active_index(0, 2, 3),
            active_index(2, 2, 3),
            active_index(1, 1, 3),
            active_index(1, 3, 3),
            active_index(1, 2, 2),
            active_index(1, 2, 4),
            active_index(3, 2, 3),
            active_index(1, 0, 3),
            active_index(1, 2, 5),
            active_index(2, 3, 3),
        ):
            required_interior.add(base + neighbor)
    assert required_interior.issubset(interior_columns)

    boundary_row = active_index(0, 0, 0)
    boundary_columns = row_columns(boundary_row)
    required_boundary = set()
    for variable_block in range(3):
        base = variable_block * active_cells
        for neighbor in (
            active_index(0, 0, 0),
            active_index(1, 0, 0),
            active_index(2, 0, 0),
            active_index(0, 1, 0),
            active_index(0, 2, 0),
            active_index(0, 0, 1),
            active_index(0, 0, 2),
            active_index(0, 0, mesh.nz - 1),
            active_index(0, 0, mesh.nz - 2),
            active_index(1, 1, 0),
        ):
            required_boundary.add(base + neighbor)
    assert required_boundary.issubset(boundary_columns)

    far_neighbor = active_index(active_nx - 1, active_ny - 1, 0)
    assert far_neighbor not in interior_columns


def test_neutral_mixed_active_jacobian_color_groups_partition_state() -> None:
    _, _, mesh, _, state, _ = _build_small_implicit_case()
    packed = pack_neutral_mixed_active_state(state, mesh=mesh)
    color_groups = build_neutral_mixed_active_jacobian_color_groups(mesh)
    flattened = sorted(column for group in color_groups for column in group)
    active_nx = mesh.xend - mesh.xstart + 1
    active_ny = mesh.yend - mesh.ystart + 1

    assert flattened == list(range(packed.size))
    assert len(color_groups) == 3 * min(5, active_nx) * min(5, active_ny) * mesh.nz


def test_neutral_mixed_sparse_residual_jacobian_matches_single_column_difference_quotient() -> None:
    pytest.importorskip("scipy")

    config, run_config, mesh, metrics, state, _ = _build_case()
    scalars = resolved_dataset_scalars(run_config)
    packed = pack_neutral_mixed_active_state(state, mesh=mesh)

    def residual(packed_state: np.ndarray) -> np.ndarray:
        return compute_neutral_mixed_backward_euler_residual(
            packed_state,
            packed,
            config=config,
            template_state=state,
            section="h",
            mesh=mesh,
            metrics=metrics,
            meters_scale=float(scalars["rho_s0"]),
            tnorm=float(scalars["Tnorm"]),
            timestep=20.0,
        )

    jacobian = build_neutral_mixed_sparse_residual_jacobian(residual, packed, mesh=mesh)
    column = 53
    step = np.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(float(packed[column])))
    direct = (residual(packed + step * np.eye(1, packed.size, column, dtype=np.float64).ravel()) - residual(packed)) / step
    sparse_column = jacobian.getcol(column).toarray().ravel()

    np.testing.assert_allclose(sparse_column, direct, rtol=1e-6, atol=1e-8)


def test_neutral_mixed_bdf2_step_solves_active_residual() -> None:
    pytest.importorskip("scipy")

    config, run_config, mesh, metrics, state, _ = _build_case()
    scalars = resolved_dataset_scalars(run_config)
    first_step, _ = advance_neutral_mixed_backward_euler_step(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=10.0,
    )
    second_step, info = advance_neutral_mixed_bdf2_step(
        config,
        first_step,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=10.0,
    )
    solved = pack_neutral_mixed_active_state(second_step, mesh=mesh)
    previous = pack_neutral_mixed_active_state(first_step, mesh=mesh)
    previous_previous = pack_neutral_mixed_active_state(state, mesh=mesh)
    solved_residual = compute_neutral_mixed_bdf2_residual(
        solved,
        previous,
        previous_previous,
        config=config,
        template_state=first_step,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=10.0,
    )

    assert np.max(np.abs(solved_residual)) < 1.0e-7
    assert info.residual_inf_norm < 1.0e-7
    assert np.all(np.isfinite(second_step.density))
    assert np.all(np.isfinite(second_step.pressure))
    assert np.all(np.isfinite(second_step.momentum))


def test_neutral_mixed_implicit_history_returns_finite_step_sequence() -> None:
    pytest.importorskip("scipy")

    config, run_config, mesh, metrics, _, _ = _build_small_implicit_case()
    scalars = resolved_dataset_scalars(run_config)
    history = advance_neutral_mixed_implicit_history(
        config,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=5.0,
        steps=3,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        step_tolerance=1.0e-10,
        max_nonlinear_iterations=8,
        linear_restart=10,
        linear_maxiter=60,
        linear_rtol=1.0e-9,
    )

    assert history.density_history.shape == (4, mesh.nx, mesh.local_ny, mesh.nz)
    assert history.pressure_history.shape == (4, mesh.nx, mesh.local_ny, mesh.nz)
    assert history.momentum_history.shape == (4, mesh.nx, mesh.local_ny, mesh.nz)
    assert np.all(np.isfinite(history.density_history))
    assert np.all(np.isfinite(history.pressure_history))
    assert np.all(np.isfinite(history.momentum_history))


def test_execute_neutral_mixed_case_supports_one_step_and_short_window(monkeypatch: pytest.MonkeyPatch) -> None:
    config, run_config, mesh, metrics, _, _ = _build_small_implicit_case()

    class _History:
        density_history = np.full((3, mesh.nx, mesh.local_ny, mesh.nz), 1.0, dtype=np.float64)
        pressure_history = np.full((3, mesh.nx, mesh.local_ny, mesh.nz), 2.0, dtype=np.float64)
        momentum_history = np.full((3, mesh.nx, mesh.local_ny, mesh.nz), 3.0, dtype=np.float64)

    captured: list[tuple[int, str]] = []

    def _fake_history(*args, **kwargs):
        captured.append((kwargs["steps"], kwargs["solver_mode"]))
        return _History()

    monkeypatch.setattr(
        "jax_drb.native.runner.advance_neutral_mixed_implicit_history",
        _fake_history,
    )

    time_points, variables = _execute_neutral_mixed_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="one_step",
    )
    assert time_points == (0.0, 20.0)
    assert variables["Nh"].shape == (3, mesh.nx, mesh.local_ny, mesh.nz)
    assert captured[-1] == (1, "matrix_free")

    time_points, variables = _execute_neutral_mixed_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="short_window",
    )
    assert time_points[0] == 0.0
    assert time_points[-1] == run_config.time.nout * run_config.time.timestep
    assert variables["NVh"].shape == (3, mesh.nx, mesh.local_ny, mesh.nz)
    assert captured[-1] == (run_config.time.nout, "matrix_free")


def test_execute_neutral_mixed_case_honors_output_steps_override(monkeypatch: pytest.MonkeyPatch) -> None:
    config, run_config, mesh, metrics, _, _ = _build_small_implicit_case()

    class _History:
        density_history = np.full((5, mesh.nx, mesh.local_ny, mesh.nz), 1.0, dtype=np.float64)
        pressure_history = np.full((5, mesh.nx, mesh.local_ny, mesh.nz), 2.0, dtype=np.float64)
        momentum_history = np.full((5, mesh.nx, mesh.local_ny, mesh.nz), 3.0, dtype=np.float64)

    captured: list[tuple[int, str]] = []

    def _fake_history(*args, **kwargs):
        captured.append((kwargs["steps"], kwargs["solver_mode"]))
        return _History()

    monkeypatch.setattr(
        "jax_drb.native.runner.advance_neutral_mixed_implicit_history",
        _fake_history,
    )

    time_points, variables = _execute_neutral_mixed_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="short_window",
        output_steps=4,
    )

    assert captured == [(4, "matrix_free")]
    assert time_points == (0.0, 20.0, 40.0, 60.0, 80.0)
    assert variables["Nh"].shape == (5, mesh.nx, mesh.local_ny, mesh.nz)


def test_density_y_boundaries_match_reference_wall_extrapolation() -> None:
    _, _, mesh, _, state, _ = _build_case(nx=8, ny=4, nz=6)

    bounded = _apply_density_y_boundaries(state.density, mesh)

    lower_wall = np.maximum(
        0.5 * (3.0 * state.density[:, mesh.ystart, :] - state.density[:, mesh.ystart + 1, :]),
        0.0,
    )
    upper_wall = np.maximum(
        0.5 * (3.0 * state.density[:, mesh.yend, :] - state.density[:, mesh.yend - 1, :]),
        0.0,
    )

    np.testing.assert_allclose(
        bounded[:, mesh.ystart - 1, :],
        2.0 * lower_wall - state.density[:, mesh.ystart, :],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bounded[:, mesh.yend + 1, :],
        2.0 * upper_wall - state.density[:, mesh.yend, :],
        rtol=1e-12,
        atol=1e-12,
    )


def test_soft_floor_matches_reference_formula() -> None:
    values = np.asarray([-1.0, 0.0, 0.02, 0.2], dtype=np.float64)
    minimum = 0.1
    actual = _soft_floor(values, minimum)
    expected = np.maximum(values, 0.0) + minimum * np.exp(-np.maximum(values, 0.0) / minimum)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_grad_par_open_matches_centered_bout_metric_form() -> None:
    _, _, mesh, metrics, state, _ = _build_case(nx=8, ny=4, nz=6)
    field = np.asarray(state.pressure, dtype=np.float64)
    actual = _grad_par_open(field, mesh=mesh, metrics=metrics)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    i = mesh.xstart
    k = 0
    lower = mesh.ystart
    upper = mesh.yend

    expected_lower = 0.5 * (field[i, lower + 1, k] - field[i, lower - 1, k]) / (
        dy[i, lower, k] * np.sqrt(g22[i, lower, k])
    )
    expected_upper = 0.5 * (field[i, upper + 1, k] - field[i, upper - 1, k]) / (
        dy[i, upper, k] * np.sqrt(g22[i, upper, k])
    )
    interior = lower + 1
    expected_interior = 0.5 * (field[i, interior + 1, k] - field[i, interior - 1, k]) / (
        dy[i, interior, k] * np.sqrt(g22[i, interior, k])
    )

    assert actual[i, lower, k] == pytest.approx(expected_lower, rel=1e-12, abs=1e-12)
    assert actual[i, upper, k] == pytest.approx(expected_upper, rel=1e-12, abs=1e-12)
    assert actual[i, interior, k] == pytest.approx(expected_interior, rel=1e-12, abs=1e-12)
