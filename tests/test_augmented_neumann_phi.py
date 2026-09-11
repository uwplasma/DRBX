"""Reference tests for the Rung-3 all-Neumann potential solve.

These tests deliberately use a small weighted finite-volume model instead of
the production FCI geometry.  They are the algebraic oracle for the
quotient-space implementation: the production solve may use a projected
Krylov solve followed by an analytic constant shift, but it must agree with
the augmented saddle-point system below.  Keeping this oracle independent of
the production implementation also makes failures in the compatibility or
gauge algebra unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _augmented_neumann_uses_projected_pcg,
    _fixed_projected_preconditioned_cg,
    _fixed_projected_preconditioned_cg_with_corrections,
    _homogeneous_local_face_bc,
    build_local_perp_laplacian_face_projectors,
    build_solvax_perp_laplacian_preconditioner,
)
from tests.test_fci_operators_domain_decomp import (
    _build_domain,
    _build_ghost_filler,
    _build_local_geometry,
    _mms_parallel_field,
    make_mesh_for_shard_counts,
    put_scalar_field_on_mesh,
)
from tests.test_fci_projected_fine_grid_control_volume import _setup as _rlp_setup


def _weighted_periodic_laplacian(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative rank ``N-1`` Laplacian and cell weights.

    The graph has periodic nearest-neighbour edges with nonuniform positive
    cell weights.  ``A`` is represented in the volume form ``W L`` and is
    therefore symmetric, conservative, and has the constant vector as its
    one-dimensional nullspace.
    """

    n = int(size)
    if n < 3:
        raise ValueError("the oracle requires at least three cells")
    conductance = 1.0 + 0.25 * np.arange(n, dtype=float)
    graph = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        c = 0.5 * (conductance[i] + conductance[j])
        graph[i, i] += c
        graph[j, j] += c
        graph[i, j] -= c
        graph[j, i] -= c
    weights = 1.0 + 0.1 * np.arange(n, dtype=float)
    # Work with the volume-form operator used by the solver, ``L = W^-1 K``.
    # Its left null vector is W 1, while its right null vector remains 1.
    return graph / weights[:, None], weights


def _small_projected_cg_problem(size: int = 7):
    """Return a small Euclidean SPD quotient problem with constant null."""

    n = int(size)
    graph = np.zeros((n, n), dtype=float)
    for index in range(n):
        neighbor = (index + 1) % n
        graph[index, index] += 1.0
        graph[neighbor, neighbor] += 1.0
        graph[index, neighbor] -= 1.0
        graph[neighbor, index] -= 1.0
    operator = jnp.asarray(graph, dtype=jnp.float64)

    def project(values):
        values = jnp.asarray(values, dtype=jnp.float64)
        return values - jnp.mean(values)

    def apply_A(values):
        return operator @ values

    def inner_product(left, right):
        return jnp.vdot(left, right).real

    diagonal = jnp.diag(operator)

    def preconditioner(values):
        return values / diagonal

    return operator, project, apply_A, inner_product, preconditioner


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(weights, values) / np.sum(weights))


def _wall_trace_from_neumann(
    values: np.ndarray,
    neumann: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """A one-sided face trace with the affine Neumann contribution explicit."""

    return values + distances * neumann


def _augment(
    operator: np.ndarray,
    volume_weights: np.ndarray,
    wall_weights: np.ndarray,
    target: float,
) -> np.ndarray:
    """Build the dense reference saddle-point system.

    ``operator`` is the volume-form conservative operator.  The first block
    uses a constant Lagrange-multiplier column; the second row is the chosen
    wall-gauge functional.  This is equivalent to the quotient solve plus a
    constant shift when the source is compatible.
    """

    del volume_weights
    n = operator.shape[0]
    one = np.ones((n, 1), dtype=float)
    gauge = (
        np.asarray(wall_weights, dtype=float) / np.sum(wall_weights)
    )[None, :]
    return np.block([
        [operator, one],
        [gauge, np.zeros((1, 1), dtype=float)],
    ])


def _solve_augmented(
    operator: np.ndarray,
    rhs: np.ndarray,
    volume_weights: np.ndarray,
    wall_weights: np.ndarray,
    target: float,
) -> tuple[np.ndarray, float]:
    n = operator.shape[0]
    system = _augment(operator, volume_weights, wall_weights, target)
    solution = np.linalg.solve(
        system,
        np.concatenate((rhs, np.asarray((target,), dtype=float))),
    )
    return solution[:n], float(solution[n])


@dataclass(frozen=True)
class _QuotientResult:
    field: np.ndarray
    lambda_: float
    raw_compatibility: float
    iterations: int


def _solve_quotient_reference(
    operator: np.ndarray,
    rhs: np.ndarray,
    volume_weights: np.ndarray,
    wall_weights: np.ndarray,
    target: float,
    *,
    preconditioner: str = "none",
) -> _QuotientResult:
    """Reference quotient solve corresponding to production's planned path."""

    # The quotient functional need not be a left null vector.  Project both
    # the source and the operator output, solve on c(phi)=0, and recover the
    # augmented multiplier from the post-solve raw residual.
    quotient_weights = np.asarray(volume_weights, dtype=float)
    denominator = float(np.sum(quotient_weights))

    # Add one harmless rank-one term only to form a nonsingular dense oracle;
    # the resulting solution is the unique zero-volume-mean quotient result.
    # The physical operator itself is never regularized.
    n = operator.shape[0]
    one = np.ones(n)
    projection = np.eye(n) - np.outer(one, quotient_weights) / denominator
    projected_operator = projection @ operator + np.outer(
        one, quotient_weights
    ) / denominator

    diagonal = np.diag(projected_operator).copy()
    if preconditioner == "none":
        inverse = np.eye(n)
    elif preconditioner == "jacobi":
        inverse = np.diag(1.0 / diagonal)
    elif preconditioner == "line-u":
        # The 1-D line solve is the exact dense reference for this graph.
        inverse = np.linalg.inv(projected_operator)
    else:
        raise ValueError(preconditioner)

    # Solve the projected operator; preconditioners are applied only as a
    # left action in this oracle and must not alter the physical result.
    del inverse
    quotient = np.linalg.solve(projected_operator, projection @ rhs)
    quotient -= _weighted_mean(quotient, volume_weights)
    wall_mean = float(np.dot(wall_weights, quotient) / np.sum(wall_weights))
    shifted = quotient + (target - wall_mean)
    raw_residual = rhs - operator @ shifted
    raw_integral = float(np.dot(quotient_weights, raw_residual))
    lambda_ = raw_integral / denominator
    return _QuotientResult(
        field=shifted,
        lambda_=lambda_,
        raw_compatibility=raw_integral,
        iterations=1 if preconditioner == "line-u" else n,
    )


@pytest.mark.parametrize("use_preconditioner", (False, True))
def test_fixed_projected_pcg_recovers_small_spd_quotient_solution(
    use_preconditioner: bool,
) -> None:
    operator, project, apply_A, inner_product, preconditioner = (
        _small_projected_cg_problem()
    )
    exact = project(
        jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, operator.shape[0], endpoint=False))
    )
    rhs = apply_A(exact)
    solved, steps, initial_norm, final_norm, breakdown = (
        _fixed_projected_preconditioned_cg(
            apply_A,
            rhs,
            jnp.zeros_like(rhs),
            project=project,
            inner_product=inner_product,
            preconditioner=preconditioner if use_preconditioner else None,
            maxiter=20,
            tol=1.0e-12,
            atol=1.0e-13,
        )
    )

    np.testing.assert_allclose(np.asarray(solved), np.asarray(exact), atol=1.0e-11)
    assert int(steps) > 0
    assert float(final_norm) < float(initial_norm)
    assert float(final_norm) < 1.0e-11
    assert not bool(breakdown)


def test_fixed_projected_pcg_exact_zero_rhs_returns_without_breakdown() -> None:
    operator, project, apply_A, inner_product, preconditioner = (
        _small_projected_cg_problem()
    )
    rhs = jnp.zeros((operator.shape[0],), dtype=jnp.float64)
    solved, steps, initial_norm, final_norm, breakdown = (
        _fixed_projected_preconditioned_cg(
            apply_A,
            rhs,
            jnp.zeros_like(rhs),
            project=project,
            inner_product=inner_product,
            preconditioner=preconditioner,
            maxiter=20,
            tol=1.0e-12,
            atol=1.0e-13,
        )
    )

    np.testing.assert_array_equal(np.asarray(solved), np.zeros(operator.shape[0]))
    assert int(steps) == 0
    assert float(initial_norm) == 0.0
    assert float(final_norm) == 0.0
    assert not bool(breakdown)


def _dense_spd_cg_problem():
    """Return a small dense SPD problem for PCG refinement tests."""

    operator = jnp.asarray(((4.0, 1.0), (1.0, 3.0)), dtype=jnp.float64)

    def project(values):
        return jnp.asarray(values, dtype=jnp.float64)

    def apply_A(values):
        return operator @ values

    def inner_product(left, right):
        return jnp.vdot(left, right).real

    def preconditioner(values):
        return values

    return operator, project, apply_A, inner_product, preconditioner


def test_projected_pcg_residual_correction_reduces_true_residual_and_accumulates() -> None:
    """A recurrence-limited primary solve is improved by true-residual replacement."""

    _operator, project, apply_A, inner_product, preconditioner = (
        _dense_spd_cg_problem()
    )
    rhs = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    kwargs = dict(
        project=project,
        inner_product=inner_product,
        preconditioner=preconditioner,
        maxiter=1,
        tol=1.0e-12,
        atol=1.0e-13,
    )
    primary = _fixed_projected_preconditioned_cg(
        apply_A, rhs, jnp.zeros_like(rhs), **kwargs
    )
    refined = _fixed_projected_preconditioned_cg_with_corrections(
        apply_A,
        rhs,
        jnp.zeros_like(rhs),
        **kwargs,
        acceptance_threshold=1.0e-12,
        residual_correction_steps=1,
    )

    assert float(refined[3]) < float(primary[3])
    assert int(refined[1]) == int(primary[1]) + 1
    assert not bool(refined[4])


def test_projected_pcg_zero_correction_steps_preserves_primary_result() -> None:
    """Disabling refinement retains the original fixed-loop PCG result."""

    _operator, project, apply_A, inner_product, preconditioner = (
        _dense_spd_cg_problem()
    )
    rhs = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    kwargs = dict(
        project=project,
        inner_product=inner_product,
        preconditioner=preconditioner,
        maxiter=1,
        tol=1.0e-12,
        atol=1.0e-13,
    )
    primary = _fixed_projected_preconditioned_cg(
        apply_A, rhs, jnp.zeros_like(rhs), **kwargs
    )
    unchanged = _fixed_projected_preconditioned_cg_with_corrections(
        apply_A,
        rhs,
        jnp.zeros_like(rhs),
        **kwargs,
        acceptance_threshold=1.0e-12,
        residual_correction_steps=0,
    )

    for actual, expected in zip(unchanged, primary):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_projected_pcg_exact_initial_solution_skips_refinement() -> None:
    """An exact initial guess takes zero iterations, including with refinement enabled."""

    operator, project, apply_A, inner_product, preconditioner = (
        _dense_spd_cg_problem()
    )
    rhs = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    exact = jnp.linalg.solve(operator, rhs)
    solved = _fixed_projected_preconditioned_cg_with_corrections(
        apply_A,
        rhs,
        exact,
        project=project,
        inner_product=inner_product,
        preconditioner=preconditioner,
        maxiter=5,
        tol=1.0e-12,
        atol=1.0e-13,
        acceptance_threshold=1.0e-12,
        residual_correction_steps=2,
    )

    np.testing.assert_allclose(np.asarray(solved[0]), np.asarray(exact), atol=1.0e-13)
    assert int(solved[1]) == 0
    assert float(solved[2]) == 0.0
    assert float(solved[3]) == 0.0
    assert not bool(solved[4])


def test_projected_pcg_correction_breakdown_is_propagated() -> None:
    """A failed active correction cannot be reported as a healthy solve."""

    _operator, project, apply_A, inner_product, _preconditioner = (
        _dense_spd_cg_problem()
    )
    rhs = jnp.asarray((1.0, 2.0), dtype=jnp.float64)

    def fails_on_correction(values):
        # The primary residual has max magnitude 2; its true residual after
        # one step is below 1.5, so only the correction preconditioner fails.
        return jnp.where(jnp.max(jnp.abs(values)) < 1.5, jnp.nan, values)

    solved = _fixed_projected_preconditioned_cg_with_corrections(
        apply_A,
        rhs,
        jnp.zeros_like(rhs),
        project=project,
        inner_product=inner_product,
        preconditioner=fails_on_correction,
        maxiter=1,
        # A relatively loose primary tolerance marks its one valid step done,
        # so the deliberately bad preconditioner is first exercised by the
        # separately launched correction solve.
        tol=5.0e-1,
        atol=1.0e-13,
        acceptance_threshold=1.0e-12,
        residual_correction_steps=1,
    )

    assert bool(solved[4])
    assert int(solved[1]) == 1
    assert np.isfinite(float(solved[3]))


def test_neumann_operator_has_only_constant_nullspace() -> None:
    operator, weights = _weighted_periodic_laplacian(8)
    one = np.ones(operator.shape[0])

    np.testing.assert_allclose(operator @ one, 0.0, atol=1.0e-13)
    assert np.linalg.matrix_rank(operator, tol=1.0e-12) == operator.shape[0] - 1
    assert np.linalg.norm(weights @ operator) < 1.0e-12


def test_augmented_wall_gauge_removes_constant_nullspace() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(8)
    wall_weights = 0.5 + np.arange(8, dtype=float)
    target = 1.75
    system = _augment(operator, volume_weights, wall_weights, target)

    assert np.linalg.matrix_rank(system, tol=1.0e-12) == system.shape[0]
    assert np.linalg.cond(system) < 1.0e8


def test_manufactured_all_neumann_reconstruction_recovers_wall_gauge() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(9)
    wall_weights = 1.0 + 0.2 * np.arange(9, dtype=float)
    exact = np.sin(np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)) + 0.37
    target = float(np.dot(wall_weights, exact) / np.sum(wall_weights))
    rhs = operator @ exact

    solved, lambda_ = _solve_augmented(
        operator, rhs, volume_weights, wall_weights, target
    )

    np.testing.assert_allclose(solved, exact, rtol=0.0, atol=1.0e-12)
    assert abs(lambda_) < 1.0e-12
    np.testing.assert_allclose(operator @ solved, rhs, atol=1.0e-12)
    np.testing.assert_allclose(
        np.dot(wall_weights, solved) / np.sum(wall_weights), target, atol=1.0e-12
    )


def test_volume_and_wall_gauges_differ_only_by_a_constant() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(10)
    exact = np.cos(np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False))
    rhs = operator @ exact
    wall_weights = 0.8 + 0.1 * np.arange(10, dtype=float)

    volume_gauge = exact - _weighted_mean(exact, volume_weights)
    wall_target = 2.25
    wall_gauge, lambda_ = _solve_augmented(
        operator, rhs, volume_weights, wall_weights, wall_target
    )
    assert abs(lambda_) < 1.0e-12
    difference = wall_gauge - volume_gauge
    np.testing.assert_allclose(difference, difference[0], atol=1.0e-12)
    np.testing.assert_allclose(operator @ wall_gauge, operator @ volume_gauge, atol=1.0e-12)


def test_incompatible_rhs_reports_nonzero_lambda_and_raw_compatibility() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(8)
    wall_weights = np.ones(8)
    exact = np.sin(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))
    perturbation = 0.125
    rhs = operator @ exact + perturbation
    target = 0.0

    _solution, lambda_ = _solve_augmented(
        operator, rhs, volume_weights, wall_weights, target
    )
    expected = perturbation * np.sum(volume_weights) / np.sum(volume_weights)
    assert abs(lambda_) > 1.0e-8
    np.testing.assert_allclose(lambda_, expected, rtol=0.0, atol=1.0e-12)

    quotient = _solve_quotient_reference(
        operator, rhs, volume_weights, wall_weights, target
    )
    np.testing.assert_allclose(
        quotient.raw_compatibility,
        perturbation * np.sum(volume_weights),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(quotient.lambda_, perturbation, atol=1.0e-12)


def test_nonsymmetric_right_null_quotient_recovers_augmented_solution() -> None:
    """The quotient functional need not be a left null vector of ``A``."""

    operator, left_null_weights = _weighted_periodic_laplacian(7)
    one = np.ones(7)
    # Deliberately choose a quotient measure different from the actual left
    # null vector.  This is the algebraic situation exposed by the RLP test.
    quotient_weights = 0.7 + 0.03 * np.arange(7, dtype=float)
    wall_weights = 1.0 + 0.2 * np.arange(7, dtype=float)
    exact = np.sin(np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)) + 0.43
    target = float(np.dot(wall_weights, exact) / np.sum(wall_weights))
    expected_lambda = 0.137
    rhs = operator @ exact + expected_lambda * one

    np.testing.assert_allclose(operator @ one, 0.0, atol=1.0e-13)
    assert np.linalg.matrix_rank(operator, tol=1.0e-12) == 6
    assert np.linalg.norm(left_null_weights @ operator) < 1.0e-12
    assert np.linalg.norm(quotient_weights @ operator) > 1.0e-5

    dense_field, dense_lambda = _solve_augmented(
        operator,
        rhs,
        quotient_weights,
        wall_weights,
        target,
    )
    quotient = _solve_quotient_reference(
        operator,
        rhs,
        quotient_weights,
        wall_weights,
        target,
    )

    np.testing.assert_allclose(dense_field, exact, atol=1.0e-12)
    np.testing.assert_allclose(dense_lambda, expected_lambda, atol=1.0e-12)
    np.testing.assert_allclose(quotient.field, dense_field, atol=1.0e-12)
    np.testing.assert_allclose(quotient.lambda_, dense_lambda, atol=1.0e-12)
    np.testing.assert_allclose(
        quotient.raw_compatibility,
        expected_lambda * np.sum(quotient_weights),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        operator @ quotient.field + quotient.lambda_ * one,
        rhs,
        atol=1.0e-12,
    )


def test_nonzero_neumann_affine_source_enters_exactly_once() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(8)
    wall_weights = 1.0 + np.arange(8, dtype=float)
    distances = np.full(8, 0.25)
    neumann = 0.3 + 0.02 * np.arange(8, dtype=float)
    exact = np.cos(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))

    # The boundary source is represented separately from the homogeneous
    # quotient operator.  This mirrors ``rhs - boundary_source`` in the
    # production solve and prevents applying the affine halo contribution
    # once during source construction and a second time through phi_lift.
    boundary_source = volume_weights * neumann
    rhs_with_boundary = operator @ exact + boundary_source
    wall_face = _wall_trace_from_neumann(exact, neumann, distances)
    target = float(np.dot(wall_weights, wall_face) / np.sum(wall_weights))

    solved, lambda_ = _solve_augmented(
        operator,
        rhs_with_boundary - boundary_source,
        volume_weights,
        wall_weights,
        target,
    )
    assert abs(lambda_) < 1.0e-12
    # Adding boundary_source a second time would leave an O(||neumann||)
    # residual and fail this exact reconstruction.
    np.testing.assert_allclose(operator @ solved, operator @ exact, atol=1.0e-12)
    np.testing.assert_allclose(
        np.dot(wall_weights, solved) / np.sum(wall_weights), target, atol=1.0e-12
    )


@pytest.mark.parametrize("preconditioner", ("none", "jacobi", "line-u"))
def test_quotient_preconditioners_preserve_wall_gauge_and_physical_action(
    preconditioner: str,
) -> None:
    operator, volume_weights = _weighted_periodic_laplacian(9)
    wall_weights = 1.0 + 0.15 * np.arange(9, dtype=float)
    exact = np.sin(np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)) + 0.91
    rhs = operator @ exact
    target = float(np.dot(wall_weights, exact) / np.sum(wall_weights))

    result = _solve_quotient_reference(
        operator,
        rhs,
        volume_weights,
        wall_weights,
        target,
        preconditioner=preconditioner,
    )
    np.testing.assert_allclose(result.field, exact, atol=1.0e-12)
    np.testing.assert_allclose(operator @ result.field, rhs, atol=1.0e-12)
    np.testing.assert_allclose(
        np.dot(wall_weights, result.field) / np.sum(wall_weights), target, atol=1.0e-12
    )
    assert result.iterations > 0


def test_neumann_face_trace_is_the_gauge_functional_not_an_owner_mean() -> None:
    operator, volume_weights = _weighted_periodic_laplacian(8)
    wall_weights = np.asarray((1.0, 2.0, 4.0, 8.0, 7.0, 5.0, 3.0, 2.0))
    distances = np.full(8, 0.5)
    neumann = np.linspace(-0.2, 0.2, 8)
    exact = np.cos(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))
    face = _wall_trace_from_neumann(exact, neumann, distances)
    target = float(np.dot(wall_weights, face) / np.sum(wall_weights))
    affine_face_offset = float(
        np.dot(wall_weights, distances * neumann) / np.sum(wall_weights)
    )

    solved, lambda_ = _solve_augmented(
        operator,
        operator @ exact,
        volume_weights,
        wall_weights,
        target - affine_face_offset,
    )
    solved_face = _wall_trace_from_neumann(solved, neumann, distances)
    np.testing.assert_allclose(
        np.dot(wall_weights, solved_face) / np.sum(wall_weights), target, atol=1.0e-12
    )
    assert abs(lambda_) < 1.0e-12
    # A volume-owner gauge would not in general equal the physical face
    # gauge when the affine Neumann offset is nonzero.
    assert not np.isclose(_weighted_mean(solved, volume_weights), target)


def _all_neumann_radial_bc(geometry: object) -> LocalBoundaryFaceBC3D:
    """Mark the two physical radial faces as nonzero-capable Neumann faces."""

    bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    return bc.__class__(
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=bc.kind_y,
        kind_z=bc.kind_z,
        value_x=bc.value_x,
        value_y=bc.value_y,
        value_z=bc.value_z,
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
        mask_y=bc.mask_y,
        mask_z=bc.mask_z,
        layout=bc.layout,
    )


def _nonzero_neumann_filler(halo_width: int) -> PhysicalGhostCellFiller3D:
    """Use a deliberately simple affine Neumann ghost rule for manufacture."""

    weights = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.ones((halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(weights, weights, weights),
        neumann_lower=(weights, weights, weights),
        neumann_upper=(weights, weights, weights),
    )


def _nonzero_neumann_radial_bc(geometry: object) -> LocalBoundaryFaceBC3D:
    bc = _all_neumann_radial_bc(geometry)
    return bc.__class__(
        kind_x=bc.kind_x,
        kind_y=bc.kind_y,
        kind_z=bc.kind_z,
        value_x=bc.value_x.at[0].set(0.13).at[-1].set(-0.09),
        value_y=bc.value_y,
        value_z=bc.value_z,
        mask_x=bc.mask_x,
        mask_y=bc.mask_y,
        mask_z=bc.mask_z,
        layout=bc.layout,
    )


def _production_real_neumann_case(
    *,
    global_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    preconditioner: str,
    face_bc_kind: str = "zero",
    rhs_perturbation: float = 0.0,
    affine_offset: float = 0.0,
):
    """Run one real production quotient solve; intentionally not called here."""

    halo_width = 2
    domain = _build_domain(global_shape, halo_width, shard_counts)
    local_shape = domain.layout.owned_shape
    ghost_filler = (
        _nonzero_neumann_filler(halo_width)
        if face_bc_kind == "nonzero"
        else _build_ghost_filler(halo_width)
    )
    nx, ny, nz = global_shape
    rho_faces = jnp.linspace(0.2, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    toroidal = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    exact = _mms_parallel_field(rho, theta, toroidal)
    gauge_weights_global = jnp.zeros(global_shape, dtype=jnp.float64)
    gauge_weights_global = gauge_weights_global.at[0].set(1.0)
    gauge_weights_global = gauge_weights_global.at[-1].set(1.0)
    target = (
        jnp.sum(gauge_weights_global * exact)
        / jnp.sum(gauge_weights_global)
        + affine_offset
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        exact_sharded = put_scalar_field_on_mesh(exact, mesh)

        def kernel(phi_owned):
            shard_index = tuple(
                lax.axis_index(name) for name in ("x", "y", "z")
            )
            geometry = _build_local_geometry(
                local_shape,
                halo_width,
                global_shape=global_shape,
                shard_index=shard_index,
            )
            face_bc = (
                _nonzero_neumann_radial_bc(geometry)
                if face_bc_kind == "nonzero"
                else _all_neumann_radial_bc(geometry)
            )
            gauge_weights = jnp.zeros(local_shape, dtype=jnp.float64)
            gauge_weights = gauge_weights.at[0].set(1.0)
            gauge_weights = gauge_weights.at[-1].set(1.0)
            solver = LocalPerpLaplacianInverseSolver(
                geometry=geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                config=SolvaxGmresConfig(
                    tol=1.0e-9,
                    atol=1.0e-10,
                    maxiter=60,
                    restart=30,
                    preconditioner=preconditioner,
                    regularization_epsilon=0.0,
                    project_mean_zero=False,
                ),
            )
            no_cv_bc = solver._default_control_volume_boundary_bc()
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                control_volume_boundary_bc=no_cv_bc,
                project_mean_zero=False,
            ) + rhs_perturbation
            solved, info = solver.solve_neumann_with_gauge(
                rhs,
                gauge_weights_owned=gauge_weights,
                gauge_target=target,
                gauge_affine_offset=affine_offset,
                return_diagnostics=True,
            )
            weights = (
                jnp.asarray(geometry.cell_volume_geometry.volume, dtype=jnp.float64)
                * jnp.asarray(geometry.cell_volume_geometry.volume_fraction, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
            )
            return (
                solved,
                info.compatibility_multiplier,
                info.raw_compatibility_defect,
                info.final_gauge_residual,
                info.gauge_functional_valid,
                info.final_operator_residual_l2,
                weights,
            )

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(
                P("x", "y", "z"),
                P(),
                P(),
                P(),
                P(),
                P(),
                P("x", "y", "z"),
            ),
            check_rep=False,
        )
        return mapped(exact_sharded), exact, target


@pytest.mark.parametrize(
    ("preconditioner", "operator_form"),
    (
        ("none", "conservative"),
        ("jacobi", "conservative"),
        ("line-u", "conservative"),
        pytest.param(
            "line-u",
            "weighted-symmetric",
            marks=pytest.mark.skipif(
                os.environ.get("DRBX_RUN_SLOW_POLARIZATION_TESTS") != "1",
                reason="weighted production shard_map compile is an opt-in slow gate",
            ),
        ),
        pytest.param(
            "none",
            "support-paired",
            marks=pytest.mark.skipif(
                os.environ.get("DRBX_RUN_SLOW_POLARIZATION_TESTS") != "1",
                reason="support-paired production transpose compile is an opt-in slow gate",
            ),
        ),
        pytest.param(
            "line-u",
            "support-paired",
            marks=pytest.mark.skipif(
                os.environ.get("DRBX_RUN_SLOW_POLARIZATION_TESTS") != "1",
                reason="support-paired production transpose compile is an opt-in slow gate",
            ),
        ),
    ),
)
def test_production_neumann_gauge_matches_dense_augmented_oracle(
    preconditioner: str,
    operator_form: str,
) -> None:
    """Exercise the real all-Neumann quotient/Schur path on one shard.

    The independent dense oracle above covers the saddle-point algebra.  This
    test is the cheaper production integration gate: it exercises the real
    FCI ghost fill, metric operator, periodic topology, quotient solve, and
    preconditioner, then checks the manufactured state and all diagnostics.
    """

    shape = (3, 3, 3)
    shard_counts = (1, 1, 1)
    halo_width = 2
    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    nx, ny, nz = shape
    rho_faces = jnp.linspace(0.2, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    toroidal = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    exact = _mms_parallel_field(rho, theta, toroidal)
    gauge_weights = jnp.zeros(shape, dtype=jnp.float64)
    gauge_weights = gauge_weights.at[0].set(1.0)
    gauge_weights = gauge_weights.at[-1].set(1.0)
    affine_offset = jnp.asarray(0.125, dtype=jnp.float64)
    target = (
        jnp.sum(gauge_weights * exact) / jnp.sum(gauge_weights)
        + affine_offset
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        exact_sharded = put_scalar_field_on_mesh(exact, mesh)

        def kernel(phi_owned):
            shard_index = tuple(
                lax.axis_index(name) for name in ("x", "y", "z")
            )
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _all_neumann_radial_bc(geometry)
            solver = LocalPerpLaplacianInverseSolver(
                geometry=geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                operator_form=operator_form,
                config=SolvaxGmresConfig(
                    tol=1.0e-9,
                    atol=1.0e-10,
                    maxiter=60,
                    restart=30,
                    preconditioner=preconditioner,
                    regularization_epsilon=0.0,
                    project_mean_zero=False,
                ),
            )
            no_cv_bc = solver._default_control_volume_boundary_bc()
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                control_volume_boundary_bc=no_cv_bc,
                project_mean_zero=False,
            )
            solved, info = solver.solve_neumann_with_gauge(
                rhs,
                gauge_weights_owned=gauge_weights,
                gauge_target=target,
                gauge_affine_offset=affine_offset,
                return_diagnostics=True,
            )
            return (
                solved,
                rhs,
                info.compatibility_multiplier,
                info.raw_compatibility_defect,
                info.final_gauge_residual,
                info.gauge_constant_response,
                info.gauge_functional_valid,
                info.final_operator_residual_l2,
                solver._apply_A(
                    jnp.ones_like(phi_owned),
                    face_bc=face_bc,
                    control_volume_boundary_bc=no_cv_bc,
                    project_mean_zero=False,
                ),
            )

        mapped = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(
                P("x", "y", "z"),
                P("x", "y", "z"),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P("x", "y", "z"),
            ),
            check_rep=False,
        )
        (
            solved,
            rhs,
            lambda_value,
            raw_defect,
            gauge_residual,
            gauge_response,
            gauge_valid,
            operator_residual,
            constant_action,
        ) = mapped(exact_sharded)

    # ``exact`` has zero radial normal derivative, so the production
    # all-Neumann operator should annihilate its constant shift exactly.
    np.testing.assert_allclose(np.asarray(constant_action), 0.0, atol=2.0e-11)
    np.testing.assert_allclose(np.asarray(solved), np.asarray(exact), rtol=3.0e-5, atol=3.0e-5)
    np.testing.assert_allclose(float(raw_defect), 0.0, atol=3.0e-8)
    np.testing.assert_allclose(float(gauge_residual), 0.0, atol=3.0e-8)
    assert bool(gauge_valid)
    assert float(operator_residual) < 2.0e-7
    assert float(gauge_response) > 0.0


def test_production_augmented_neumann_contract_forbids_phi_lift() -> None:
    """A Neumann affine source must not be routed through a Dirichlet lift."""

    parameters = inspect.signature(
        LocalPerpLaplacianInverseSolver.solve_augmented_neumann
    ).parameters
    assert "phi_lift_owned" not in parameters
    assert "gauge_affine_offset" in parameters


@pytest.mark.parametrize(
    ("operator_form", "preconditioner", "expected"),
    (
        ("conservative", "none", False),
        ("weighted-symmetric", "none", True),
        ("weighted-symmetric", "line-u", True),
        ("support-paired", "none", False),
        ("support-paired", "jacobi", False),
        ("support-paired", "line-u", False),
    ),
)
def test_augmented_neumann_krylov_route_matches_operator_contract(
    operator_form: str,
    preconditioner: str,
    expected: bool,
) -> None:
    """Only explicitly qualified operator/preconditioner pairs enter PCG.

    Support-paired line-u is intentionally routed through FGMRES because its
    production approximate inverse does not preserve the PCG contract well
    enough, even though the underlying support-paired operator is qualified.
    """

    assert (
        _augmented_neumann_uses_projected_pcg(operator_form, preconditioner)
        is expected
    )


def test_production_incompatible_rhs_reports_lambda_and_small_augmented_residual() -> None:
    """A constant source defect is reported, not silently projected away."""

    perturbation = 0.125
    (solved, lambda_value, raw_defect, gauge_residual, valid, op_residual, weights), exact, _target = (
        _production_real_neumann_case(
            global_shape=(3, 3, 3),
            shard_counts=(1, 1, 1),
            preconditioner="none",
            rhs_perturbation=perturbation,
        )
    )
    volume = float(np.asarray(weights).sum())
    np.testing.assert_allclose(float(lambda_value), perturbation, atol=3.0e-7)
    np.testing.assert_allclose(float(raw_defect), perturbation * volume, atol=3.0e-7)
    np.testing.assert_allclose(float(gauge_residual), 0.0, atol=3.0e-7)
    assert bool(valid)
    assert float(op_residual) < 3.0e-7
    np.testing.assert_allclose(np.asarray(solved), np.asarray(exact), atol=3.0e-5)


def test_production_nonzero_neumann_affine_source_is_applied_once() -> None:
    """Manufacturing with nonzero ghost coefficients catches double entry."""

    (solved, lambda_value, raw_defect, gauge_residual, valid, op_residual, _weights), exact, _target = (
        _production_real_neumann_case(
            global_shape=(3, 3, 3),
            shard_counts=(1, 1, 1),
            preconditioner="none",
            face_bc_kind="nonzero",
        )
    )
    np.testing.assert_allclose(np.asarray(solved), np.asarray(exact), atol=3.0e-5)
    np.testing.assert_allclose(float(lambda_value), 0.0, atol=3.0e-7)
    np.testing.assert_allclose(float(raw_defect), 0.0, atol=3.0e-7)
    np.testing.assert_allclose(float(gauge_residual), 0.0, atol=3.0e-7)
    assert bool(valid)
    assert float(op_residual) < 3.0e-7


def test_production_augmented_neumann_rlp_owner_gauge_masks_inactive_owners() -> None:
    """The projected-owner path solves a nonconstant field on active owners.

    In particular, this is deliberately not a zero-RHS constant-nullspace
    check: the manufactured operator action exercises the RLP quotient and
    line-u preconditioner while the aggregate control-volume weights define
    the physical gauge over active owners.
    """

    (
        geometry,
        domain,
        _context,
        _exchange,
        _scalar_filler,
        _lowered,
        boundary_bc,
        face_bc,
        solver,
        active,
        weights,
    ) = _rlp_setup(shape=(3, 8, 6))
    active = jnp.asarray(active, dtype=bool)

    # The RLP fixture has (u, theta, eta) axes.  There is no theta dependence,
    # so the lower-polar regularity does not introduce a spurious angular
    # mismatch.  This straight-field fixture also has an additional eta-only
    # null family, so eta variation is intentionally excluded: one scalar
    # gauge cannot reconstruct that component and it is not an oracle for the
    # production one-dimensional-nullspace assumption.
    radial = geometry.grid.x.centers_owned[:, None, None]
    exact = 0.31 + 0.08 * radial
    exact = jnp.broadcast_to(exact, geometry.owned_shape)
    exact = jnp.where(active, exact, 0.0)
    rhs = solver._apply_A(
        exact,
        face_bc=face_bc,
        control_volume_boundary_bc=boundary_bc,
        project_mean_zero=False,
    )
    # ``aggregate_volume`` is zero for inactive aliases and is the same
    # global owner-volume measure used by the projected control-volume
    # operator.  The solver normalizes these weights into an average gauge.
    gauge_weights = jnp.asarray(weights, dtype=jnp.float64)
    gauge_target = jnp.sum(gauge_weights * exact) / jnp.sum(gauge_weights)
    solved, info = solver.solve_neumann_with_gauge(
        rhs,
        gauge_weights_owned=gauge_weights,
        gauge_target=gauge_target,
        return_diagnostics=True,
    )
    np.testing.assert_allclose(
        np.asarray(solved)[np.asarray(active)],
        np.asarray(exact)[np.asarray(active)],
        atol=3.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(solved)[~np.asarray(active)], 0.0, atol=0.0
    )
    np.testing.assert_allclose(float(info.final_gauge_residual), 0.0, atol=3.0e-7)
    assert bool(info.gauge_functional_valid)
    assert float(info.gauge_constant_response) > 0.0
    assert float(info.final_operator_residual_l2) < 3.0e-7
    assert solver.config.preconditioner == "line-u"
    assert np.ptp(np.asarray(exact)[np.asarray(active)]) > 1.0e-3


def test_support_paired_rlp_action_and_line_u_structural_diagnostics() -> None:
    """Check small-fixture symmetry/positivity for the qualified PCG path.

    The production-scale frozen audit separately qualifies the same
    angular-tree action on the 48^3 geometry; this test protects the routing
    contract and the small deterministic assembly oracle.
    """

    (
        geometry,
        domain,
        _context,
        _exchange,
        _scalar_filler,
        lowered,
        boundary_bc,
        face_bc,
        base_solver,
        active,
        weights,
    ) = _rlp_setup(shape=(3, 8, 6))
    solver = dataclass_replace(base_solver, operator_form="support-paired")
    active = np.asarray(active, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)

    def project(values: np.ndarray) -> np.ndarray:
        values = np.where(active, values, 0.0)
        mean = np.sum(weights * values) / np.sum(weights)
        return np.where(active, values - mean, 0.0)

    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(weights * left * right))

    rng = np.random.default_rng(20260908)
    left = project(rng.normal(size=active.shape))
    right = project(rng.normal(size=active.shape))

    def apply(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            solver._apply_A(
                jnp.asarray(values),
                face_bc=face_bc,
                control_volume_boundary_bc=boundary_bc,
                project_mean_zero=True,
                boundary_is_homogeneous=True,
            )
        )

    applied_left = apply(left)
    applied_right = apply(right)
    np.testing.assert_allclose(
        inner(left, applied_right),
        inner(applied_left, right),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert inner(left, applied_left) > 0.0

    projectors = build_local_perp_laplacian_face_projectors(
        geometry,
        domain,
        axis_regular_axes=solver.axis_regular_axes,
    )
    quotient_config = dataclass_replace(
        solver.config,
        project_mean_zero=True,
        regularization_epsilon=0.0,
        preconditioner="line-u",
    )
    raw_preconditioner = build_solvax_perp_laplacian_preconditioner(
        geometry,
        domain,
        projectors,
        _homogeneous_local_face_bc(face_bc),
        quotient_config,
        control_volume_geometry=lowered,
    )
    assert raw_preconditioner is not None
    preconditioned_left = project(
        np.asarray(raw_preconditioner(jnp.asarray(left)))
    )
    preconditioned_right = project(
        np.asarray(raw_preconditioner(jnp.asarray(right)))
    )
    np.testing.assert_allclose(
        inner(left, preconditioned_right),
        inner(preconditioned_left, right),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.all(np.isfinite(preconditioned_left))
    assert inner(left, preconditioned_left) > 0.0


def test_rlp_straight_field_eta_only_null_family_warning() -> None:
    """Document the extra eta-null family of the straight-field RLP fixture.

    This is geometry-specific: the fixture's perpendicular operator has no
    eta derivatives, so an eta-only factor multiplying an otherwise constant
    active-owner field is annihilated.  The production operator is expected to
    have only the ordinary constant nullspace; this test prevents the RLP toy
    fixture from being mistaken for that production nullspace oracle.
    """

    (
        geometry,
        _domain,
        _context,
        _exchange,
        _scalar_filler,
        _lowered,
        boundary_bc,
        face_bc,
        solver,
        active,
        _weights,
    ) = _rlp_setup(shape=(3, 8, 6))
    active = jnp.asarray(active, dtype=bool)
    eta = geometry.grid.z.centers_owned[None, None, :]
    eta_mode = jnp.broadcast_to(jnp.cos(eta), geometry.owned_shape)
    eta_mode = jnp.where(active, eta_mode, 0.0)
    rhs = solver._apply_A(
        eta_mode,
        face_bc=face_bc,
        control_volume_boundary_bc=boundary_bc,
        project_mean_zero=False,
    )
    np.testing.assert_allclose(
        np.asarray(rhs)[np.asarray(active)], 0.0, atol=3.0e-10
    )


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="requires at least two devices for a z-sharded smoke test",
)
def test_production_augmented_neumann_z_sharded_gauge() -> None:
    """A tiny z decomposition retains the global wall gauge and lambda."""

    (solved, lambda_value, raw_defect, gauge_residual, valid, op_residual, _weights), exact, _target = (
        _production_real_neumann_case(
            global_shape=(2, 2, 4),
            shard_counts=(1, 1, 2),
            preconditioner="none",
        )
    )
    np.testing.assert_allclose(np.asarray(solved), np.asarray(exact), atol=3.0e-5)
    np.testing.assert_allclose(float(lambda_value), 0.0, atol=3.0e-7)
    np.testing.assert_allclose(float(raw_defect), 0.0, atol=3.0e-7)
    np.testing.assert_allclose(float(gauge_residual), 0.0, atol=3.0e-7)
    assert bool(valid)
    assert float(op_residual) < 3.0e-7
