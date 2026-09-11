"""Cheap synthetic checks for the work-only reduced implicit harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).parents[1] / "work" / "boundary_load_audit"
sys.path.insert(0, str(AUDIT))

from reduced_implicit_stage import (  # noqa: E402
    InnerSolveResult,
    ReducedLinearSolveError,
    ReducedImplicitCallbacks,
    ReducedImplicitNewton,
    ReducedJacobianBlocks,
    ReducedNewtonConfig,
    full_newton_correction,
    reduced_newton_correction,
    _gmres_solve,
)


def _problem():
    # The known solution is (u,z)=([.5,-.3], .2).  The quadratic terms make
    # this a nonlinear test while leaving compact analytic Jacobian blocks.
    target_u = np.array([0.555, -0.322])
    target_z = 0.492

    def ru(u, z):
        u = np.asarray(u); z = float(np.asarray(z)[0])
        return u + np.array([0.15 * z, -0.2 * z]) + np.array([0.1 * u[0] ** 2, 0.2 * u[1] ** 2]) - target_u

    def rz(u, z):
        u = np.asarray(u); z = float(np.asarray(z)[0])
        return np.array([z + 0.05 * z * z + 0.4 * u[0] - 0.3 * u[1] - target_z])

    def reconstruct(u, z_guess):
        # Newton is intentionally performed from the previous trial's z so
        # the callback remains representative of an algebraic inner solve.
        z = float(np.asarray(z_guess if z_guess is not None else [0.0])[0])
        for iteration in range(40):
            residual = z + 0.05 * z * z + 0.4 * u[0] - 0.3 * u[1] - target_z
            if abs(residual) < 1.0e-13:
                return InnerSolveResult(np.array([z]), True, True, abs(residual), iteration + 1)
            z -= residual / (1.0 + 0.1 * z)
        return InnerSolveResult(np.array([z]), False, True, abs(residual), 40, "max_inner")

    def blocks(u, z):
        z = float(np.asarray(z)[0])
        M = np.diag([1.0 + 0.2 * u[0], 1.0 + 0.4 * u[1]])
        B = np.array([[0.15], [-0.2]])
        C = np.array([[0.4, -0.3]])
        A = np.array([[1.0 + 0.1 * z]])
        return ReducedJacobianBlocks(M, B, C, lambda rhs: np.linalg.solve(A, rhs), A)

    cb = ReducedImplicitCallbacks(ru, rz, reconstruct, blocks)
    return cb, ru, rz, blocks


def test_reduced_action_and_corrections_match_full_coupled_jacobian():
    cb, ru, rz, blocks_fn = _problem()
    u = np.array([0.2, 0.1]); z = np.array([0.7])
    blocks = blocks_fn(u, z)
    v = np.array([0.4, -0.2])
    expected = blocks.M @ v - blocks.B @ np.linalg.solve(blocks.A, blocks.C @ v)
    assert np.allclose(blocks.reduced_action(v), expected.ravel())

    # A nonzero algebraic residual exercises the full coupled correction term.
    ru0, rz0 = ru(u, z), rz(u, z)
    du, dz = full_newton_correction(blocks, ru0, rz0)
    full_J = np.block([[blocks.M, blocks.B], [blocks.C, blocks.A]])
    assert np.allclose(full_J @ np.r_[du, dz], -np.r_[ru0, rz0])

    # Once z is reconstructed, the reduced correction is exactly the material
    # component of the full coupled correction and reconstructs dz implicitly.
    z_projected = cb.reconstruct_z(u, z).z
    projected_blocks = blocks_fn(u, z_projected)
    du_red, dz_red = reduced_newton_correction(projected_blocks, ru(u, z_projected))
    du_full, dz_full = full_newton_correction(projected_blocks, ru(u, z_projected), rz(u, z_projected))
    assert np.allclose(du_red, du_full, atol=1.0e-12)
    assert np.allclose(dz_red, dz_full, atol=1.0e-12)


def test_every_trial_reconstructs_and_nonlinear_state_agrees():
    cb, _ru, _rz, _blocks = _problem()
    seen = []
    original = cb.reconstruct_z

    def recording_reconstruct(u, z_guess):
        out = original(u, z_guess)
        seen.append((np.asarray(u).copy(), out.converged, out.admissible, out.residual_norm))
        return out

    cb = ReducedImplicitCallbacks(
        cb.material_residual, cb.algebraic_residual, recording_reconstruct,
        cb.jacobian_blocks, material_norm=lambda value: np.linalg.norm(value),
        algebraic_norm=lambda value: np.linalg.norm(value),
    )
    result = ReducedImplicitNewton(cb, ReducedNewtonConfig(max_newton=10, full_tol=1.0e-11, inner_tol=1.0e-11)).solve(
        np.array([0.2, 0.1]), np.array([2.0])
    )
    assert result.converged, result.reason
    assert np.allclose(result.u, [0.5, -0.3], atol=1.0e-9)
    assert np.allclose(result.z, [0.2], atol=1.0e-9)
    assert seen and all(converged and admissible and residual < 1.0e-11 for _, converged, admissible, residual in seen)
    assert result.calls["reconstruct_z"] == len(seen)
    assert result.calls["material_residual"] == result.calls["algebraic_residual"]
    assert result.timings["reconstruct_z"] >= 0.0
    assert any(row["status"] == "accepted" for row in result.history)


def test_delegated_convergence_with_infinite_tolerances_preserves_armijo():
    # At u=.1, Newton's full correction for u^2-1 badly overshoots.
    # Infinite scalar tolerances delegate convergence; they must not accept it.
    tried = []
    def reconstruct(u, guess):
        tried.append(float(u[0]))
        return InnerSolveResult(np.zeros(1), True, True, 0., 0)
    def blocks(u, z):
        return ReducedJacobianBlocks(np.array([[2*u[0]]]), np.zeros((1, 1)),
            np.zeros((1, 1)), lambda rhs: rhs, np.eye(1))
    cb = ReducedImplicitCallbacks(lambda u,z: u*u-1., lambda u,z: z,
        reconstruct, blocks,
        full_convergence=lambda u,z,ru,rz: np.linalg.norm(ru) < 1.e-10)
    cfg = ReducedNewtonConfig(max_newton=1, material_tol=np.inf, full_tol=np.inf)
    result = ReducedImplicitNewton(cb, cfg).solve(np.array([.1]), np.zeros(1))
    rejected = [r for r in result.history if r['status']=='line_search_reject']
    accepted = [r for r in result.history if r['status']=='accepted']
    assert [r['alpha'] for r in rejected] == [1., .5]
    assert len(accepted) == 1 and accepted[0]['alpha'] == .25
    assert accepted[0]['full_norm'] < result.initial_full_norm
    assert len(tried) == 4
    assert not result.converged


def test_line_search_convergence_exception_uses_authoritative_callback():
    cb, *_ = _problem()
    from dataclasses import replace
    solver = ReducedImplicitNewton(replace(cb, full_convergence=lambda *args: True),
                                   ReducedNewtonConfig(full_tol=np.inf))
    from types import SimpleNamespace
    current = SimpleNamespace(full_norm=1.)
    candidate = SimpleNamespace(admissible=True, full_norm=2., u=np.zeros(2),
                                z=np.zeros(1), ru=np.zeros(2), rz=np.zeros(1))
    assert solver._line_search_accepts(current, candidate, 1.)
    solver.callbacks = replace(cb, full_convergence=lambda *args: False)
    assert not solver._line_search_accepts(current, candidate, 1.)
    candidate.admissible = False
    assert not solver._line_search_accepts(current, candidate, 1.)


def test_inner_failure_is_explicit_and_not_treated_as_full_convergence():
    cb, _ru, _rz, blocks = _problem()

    def failed_reconstruct(u, z_guess):
        return InnerSolveResult(np.array([0.0]), converged=False, admissible=True, residual_norm=1.0, reason="inner_limit")

    failed = ReducedImplicitCallbacks(cb.material_residual, cb.algebraic_residual, failed_reconstruct, blocks)
    result = ReducedImplicitNewton(failed).solve(np.array([0.2, 0.1]), np.array([0.0]))
    assert not result.converged
    assert result.reason == "inner_error"
    assert result.history[-1]["status"] == "inner_error"
    assert result.calls.get("material_residual", 0) == 0


def test_lied_about_inner_convergence_is_rejected_by_independent_algebraic_check():
    cb, _ru, _rz, blocks = _problem()

    def lied_reconstruct(u, z_guess):
        # The callback claims success, but z=0 is not the algebraic root.
        return InnerSolveResult(np.array([0.0]), converged=True, admissible=True, residual_norm=0.0)

    lied = ReducedImplicitCallbacks(cb.material_residual, cb.algebraic_residual, lied_reconstruct, blocks)
    result = ReducedImplicitNewton(lied).solve(np.array([0.2, 0.1]), np.array([0.0]))
    assert not result.converged
    assert result.reason == "inner_error"
    assert "algebraic residual" in result.history[-1]["error"]
    assert result.calls.get("material_residual", 0) == 0


def test_matrix_free_gmres_uses_multiple_directions_and_honors_budget():
    matrix = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    rhs = np.array([1.0, -2.0, 0.5])
    events = []
    low_budget = ReducedNewtonConfig(linear_maxiter=2, linear_restart=3, linear_rtol=1.0e-13, linear_atol=0.0)
    try:
        _gmres_solve(lambda value: matrix @ value, rhs, low_budget, events.append)
    except ReducedLinearSolveError:
        pass
    else:
        raise AssertionError("two Krylov directions should not solve this three dimensional system")
    assert 1 <= sum(event["status"] == "krylov" for event in events) <= 2

    high_budget = ReducedNewtonConfig(linear_maxiter=6, linear_restart=3, linear_rtol=1.0e-13, linear_atol=0.0)
    events = []
    correction = _gmres_solve(lambda value: matrix @ value, rhs, high_budget, events.append)
    assert sum(event["status"] == "krylov" for event in events) > 1
    assert np.allclose(matrix @ correction, rhs, atol=1.0e-11)


def test_true_linear_residual_uses_absolute_tolerance_for_tiny_rhs():
    cb, _ru, _rz, _blocks = _problem()
    config = ReducedNewtonConfig(linear_rtol=1.0e-3, linear_atol=1.0e-8)
    solver = ReducedImplicitNewton(cb, config)
    blocks = ReducedJacobianBlocks(
        np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)), lambda value: np.asarray(value), np.eye(1)
    )
    rhs = np.array([1.0e-6])
    # The relative defect is 5e-3 (> rtol), while its absolute defect is
    # 5e-9 (< atol).  The configured absolute criterion must accept it.
    accepted = solver._check_linear_solution(blocks, rhs, np.array([1.005e-6]), "tiny-rhs")
    assert np.allclose(accepted, [1.005e-6])


def test_custom_linear_gate_uses_material_scaling_for_residual_and_rhs():
    cb, _ru, _rz, _blocks = _problem()
    scaled = ReducedImplicitCallbacks(
        cb.material_residual, cb.algebraic_residual, cb.reconstruct_z, cb.jacobian_blocks,
        material_norm=lambda value: float(np.linalg.norm(np.asarray(value) * np.array([1.0e-6, 1.0]))),
    )
    solver = ReducedImplicitNewton(scaled, ReducedNewtonConfig(linear_rtol=1.0e-6, linear_atol=0.0))
    blocks = ReducedJacobianBlocks(
        np.eye(2), np.zeros((2, 1)), np.zeros((1, 2)), lambda value: np.asarray(value), np.eye(1)
    )
    rhs = np.array([0.0, 1.0])
    solution = np.array([1.0e-3, 1.0])
    # Euclidean relative defect is 1e-3 and would fail; the configured
    # material norm downweights the first owner lane to a 1e-9 defect.
    assert np.isclose(np.linalg.norm(blocks.reduced_action(solution) - rhs), 1.0e-3)
    accepted = solver._check_linear_solution(blocks, rhs, solution, "scaled-custom")
    assert np.allclose(accepted, solution)


def test_strong_block_feedback_breaks_fixed_point_but_reduced_newton_converges():
    # The material and algebraic solves each look harmless, but their
    # alternating feedback has gain 1.2**2 > 1.  The Schur derivative keeps
    # the coupling and solves the coupled problem in one Newton correction.
    def material(u, z):
        return np.asarray(u) + 1.2 * np.asarray(z) - 1.0

    def algebraic(u, z):
        return 1.2 * np.asarray(u) + np.asarray(z) - 1.0

    def reconstruct(u, z_guess):
        z = 1.0 - 1.2 * np.asarray(u)
        return InnerSolveResult(z, True, True, 0.0, 1)

    def blocks(_u, _z):
        M = np.array([[1.0]]); B = np.array([[1.2]])
        C = np.array([[1.2]]); A = np.array([[1.0]])
        return ReducedJacobianBlocks(M, B, C, lambda rhs: np.asarray(rhs), A)

    # This is the ordinary alternating material/algebraic fixed-point map.
    z = 0.0
    errors = []
    exact = 1.0 / 2.2
    for _ in range(8):
        u = 1.0 - 1.2 * z
        z = 1.0 - 1.2 * u
        errors.append(abs(z - exact))
    assert errors[-1] > errors[0]

    callbacks = ReducedImplicitCallbacks(material, algebraic, reconstruct, blocks)
    result = ReducedImplicitNewton(callbacks, ReducedNewtonConfig(max_newton=3, full_tol=1.0e-12)).solve(
        np.array([0.0]), np.array([0.0])
    )
    assert result.converged, result.reason
    assert np.allclose(result.u, [exact], atol=1.0e-12)
    assert np.allclose(result.z, [exact], atol=1.0e-12)
