"""Nonuniform-mass regression for the actual host inverse stopping policy."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import gmres

sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))
from run_reduced_implicit_capture import CaptureReducedProblem, _weighted_krylov_stopping


@pytest.mark.parametrize("scale", [1.e-12, 1.])
def test_host_inverse_meets_both_norms_with_nonuniform_mass(scale):
    weights = np.array([4.e-6, 9.e-6, 25.e-6, 16.e-6])
    d = np.sqrt(weights)
    projector = np.eye(4) - np.outer(d, d) / np.dot(d, d)
    aq = projector @ np.diag([1., 2., 4., 7.]) @ projector
    physical_a = aq * d[None, :] / d[:, None]
    rhs = np.r_[np.array([1., -2., .5, 3.]) * scale, .1 * scale]
    problem = object.__new__(CaptureReducedProblem)
    problem.active_weights = weights
    problem.nz = 5
    problem.args = SimpleNamespace(inner_rtol=1.e-10, inner_atol=1.e-13, inner_maxiter=100)
    problem._host_algebraic_action = lambda q: aq @ q
    problem._host_algebraic_preconditioner = lambda q: projector @ q
    problem._host_algebraic_metadata = {"test": "projected identity"}
    problem.adapter = SimpleNamespace(augmented=True, shape=(4,), active_flat=np.arange(4),
        _capture_homogeneous_A=lambda phi: physical_a @ np.asarray(phi),
        model=SimpleNamespace(_face_bcs=lambda state: None,
            _simplified_gbs_mpe_phi_gauge_data=lambda state, face: (weights, None, None)))
    trial = SimpleNamespace(state=None)
    weighted_rhs = projector @ (rhs[:-1] * d)
    if scale < 1.e-10:
        # The old absolute threshold accepts zero for this RHS even though
        # the physical unweighted defect is two orders of magnitude too large.
        old, info = gmres(aq, weighted_rhs, rtol=1.e-10, atol=1.e-13)
        assert info == 0 and np.count_nonzero(old) == 0
        assert np.linalg.norm(weighted_rhs / d) > 1.e-13
    z, report = problem._host_algebraic_inverse(rhs, trial)
    image = np.r_[physical_a @ z[:-1] + z[-1], np.dot(weights, z[:-1])/weights.sum()]
    defect = image - rhs
    assert np.linalg.norm(defect) <= max(1.e-13, 1.e-10*np.linalg.norm(rhs))
    metric = np.r_[d, 1.]
    assert np.linalg.norm(metric*defect) <= max(1.e-13, 1.e-10*np.linalg.norm(metric*rhs))
    assert report["solver_info"]["num_steps"] > 0
    assert report["solver_info"]["converged"]


def test_weighted_budget_bounds_both_norms_and_handles_zero_rhs():
    weights = np.array([4.e-6, 9.e-6])
    budget = _weighted_krylov_stopping(np.zeros(3), weights, rtol=1.e-10, atol=1.e-13)
    assert budget["rtol"] == 0.
    assert budget["atol"] / np.sqrt(weights.min()) <= 0.5e-13
    assert budget["atol"] <= 0.5e-13


@pytest.mark.parametrize("weights", [[0., 1.], [-1., 1.], [np.nan, 1.]])
def test_invalid_mass_rejected(weights):
    with pytest.raises(ValueError):
        _weighted_krylov_stopping(np.zeros(3), weights, rtol=1.e-10, atol=1.e-13)
