"""Exercise the actual capture reconstruction cache path without geometry setup."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))
from run_reduced_implicit_capture import CaptureReducedProblem, AlgebraicSolve


def make_problem(*, cached=True, correction_fraction=1., inverse_passed=True,
                 initial_residual=9.66e-13, admissible=True):
    # A scalar polarization equation plus a gauge row, with six fixed material
    # fields. The initial state satisfies production acceptance but can fail
    # the actual capture runner's strict absolute gate, as in nonlinear v1.
    problem = object.__new__(CaptureReducedProblem)
    material = np.ones((6, 1))
    z = np.array([initial_residual, 0.])
    state = SimpleNamespace(phi=z[:1].copy())
    vector = np.r_[material.ravel(), z]
    def residual(x):
        return np.r_[np.zeros(6), x[-2:]]
    def parts(r):
        return {"polarization": r[-2:-1], "gauge": r[-1]}
    ctx = SimpleNamespace(
        polarization_diagnostics=lambda x, residual: {
            "raw_rhs_l2": 7.7e-12,
            "polarization_l2": float(np.linalg.norm(residual["polarization"]))},
        admissibility=lambda x: {"admissible": admissible},
        norm=np.linalg.norm,
    )
    trial = AlgebraicSolve(material, state, vector, state.phi, z[-1], z,
        residual(vector)[-2:], {"algebraic_converged": True,
        "admissibility": {"admissible": admissible}, "inner_iterations": 0})
    reconstruction_calls = []
    def reconstruct(u):
        reconstruction_calls.append(np.array(u))
        return trial
    problem.adapter = SimpleNamespace(n_active=1, augmented=True,
        reconstruct_capture=reconstruct, full_residual=residual,
        full_vector=lambda u, z: (np.r_[u.ravel(), z], SimpleNamespace(phi=z[:1])),
        _parts_from_full=parts, _algebraic_converged=lambda x, r: True,
        model=SimpleNamespace(gmres_config=SimpleNamespace()))
    problem.ctx = ctx
    problem.jnp = np
    problem.args = SimpleNamespace(inner_preconditioner="multiplicative-line-u-coarse-1",
        inner_corrections=1, inner_atol=1.e-13, inner_rtol=1.e-10)
    problem.algebraic_rows = np.array([6, 7])
    problem.algebraic_metric = np.ones(2)
    problem._projection_cache = (material.tobytes(), trial) if cached else None
    problem._full_cache = None
    problem.calls = {}
    problem.timings = {}
    problem.diagnostic_arrays = {}
    problem.inner_history = []
    inverse_calls = []
    def inverse(rhs, u, current_trial, *, purpose):
        inverse_calls.append((rhs.copy(), u.copy(), purpose))
        return correction_fraction * rhs, {"passed": inverse_passed,
                                          "total_inner_iterations": 1}
    problem.refined_algebraic_inverse = inverse
    return problem, material, inverse_calls, reconstruction_calls


@pytest.mark.parametrize("cached", [True, False])
def test_strict_correction_qualifies_both_fresh_and_cached_projection(cached):
    problem, material, inverse_calls, reconstruction_calls = make_problem(cached=cached)
    result = problem.reconstruct_z(material, None)
    assert result.converged and result.admissible
    assert len(inverse_calls) == 1
    assert len(reconstruction_calls) == (0 if cached else 1)
    np.testing.assert_array_equal(inverse_calls[0][1], material)
    assert inverse_calls[0][2] == "primal_reconstruction"
    assert problem.inner_history[-1]["context_gate"]["polarization_target"] == 1.e-13
    assert problem.latest_trial.diagnostics["bounded_primal_corrections"] == 1
    again = problem.reconstruct_z(material, None)
    assert again.converged
    assert len(inverse_calls) == 1
    np.testing.assert_array_equal(again.z, result.z)


@pytest.mark.parametrize("fraction,inverse_passed", [(0., True), (1., False)])
def test_failed_correction_stays_failed_on_cache_hit_without_extra_budget(fraction, inverse_passed):
    problem, material, inverse_calls, _ = make_problem(
        correction_fraction=fraction, inverse_passed=inverse_passed)
    assert not problem.reconstruct_z(material, None).converged
    assert not problem.reconstruct_z(material, None).converged
    assert len(inverse_calls) == 1


def test_already_strict_cache_entry_needs_no_correction():
    problem, material, inverse_calls, _ = make_problem(initial_residual=5.e-14)
    assert problem.reconstruct_z(material, None).converged
    assert inverse_calls == []


def test_inadmissible_cache_entry_is_not_corrected():
    problem, material, inverse_calls, _ = make_problem(admissible=False)
    assert not problem.reconstruct_z(material, None).admissible
    assert inverse_calls == []
