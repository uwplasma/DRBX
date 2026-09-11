"""Synthetic contracts for the work-only frozen-stage diagnostics."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
AUDIT = ROOT / "work" / "boundary_load_audit"
sys.path.insert(0, str(AUDIT))

from host_fgmres_lower import flexible_gmres
from frozen_outer_fgmres_48_control import _atomic_json, _atomic_npz, _lane_diagnostics, _needs_s0_setup
from frozen_outer_fgmres_lower import run_outer_whitened_fgmres
from frozen_outer_fgmres_lower import build_coupling_aware_schur_preconditioner


def test_fgmres_four_argument_callback_reuses_true_residual_vector():
    matrix = np.array([[2.0, 0.2], [-0.1, 1.5]])
    rhs = np.array([1.0, -2.0])
    events = []
    out = flexible_gmres(matrix, rhs, lambda value: value, budget=4,
                         iteration_callback=lambda i, rel, correction, residual:
                         events.append((i, rel, correction, residual)))
    assert events
    assert np.allclose(events[-1][3], out["residual_vector"])
    assert np.isclose(events[-1][1], np.linalg.norm(events[-1][3]) / np.linalg.norm(rhs))


def test_fgmres_legacy_three_argument_callback_is_preserved():
    events = []
    flexible_gmres(np.eye(2), np.array([1.0, 2.0]), lambda value: value,
                   budget=2, iteration_callback=lambda i, rel, correction:
                   events.append((i, rel, correction)))
    assert events and events[-1][2].shape == (2,)


def test_lane_diagnostics_and_atomic_artifacts(tmp_path):
    labels = ("n", "Te", "Ti", "Vi", "Ve", "omega", "phi", "lambda")
    norms, ratios = _lane_diagnostics(np.arange(15.0), (2, 2, 2, 2, 2, 2, 2, 1), labels,
                                      {label: 1.0 for label in labels})
    assert norms["n"] == np.sqrt(1.0)
    assert ratios["lambda"] == 14.0
    json_path = tmp_path / "progress.json"
    npz_path = tmp_path / "progress.npz"
    _atomic_json(json_path, {"status": "running", "outer_iteration": 1})
    _atomic_npz(npz_path, correction=np.ones(3), current_whitened_residual=np.zeros(3))
    assert json.loads(json_path.read_text())["status"] == "running"
    with np.load(npz_path) as payload:
        assert np.all(payload["correction"] == 1.0)


def test_outer_final_residual_is_not_reduced_twice():
    class Fields:
        density = Te = Ti = Vi = Ve = vorticity = np.ones((2,))

    class Context:
        base = Fields()
        active_owned = np.ones((2,), dtype=bool)
        mass_weights = np.ones((2,))
        dt = 1.0
        pack = staticmethod(lambda *args: np.zeros(15))
        unpack = staticmethod(lambda value: (value,) * 7)
        admissibility = staticmethod(lambda value: {"admissible": True})

    ctx = Context()
    x = np.zeros(15)
    target = np.arange(15.0) + 1.0
    residual = x - target
    out = run_outer_whitened_fgmres(
        ctx, x, residual, lambda value: np.asarray(value) - target,
        lambda value: (np.asarray(value), {}), budget=1)
    assert out["final_whitened_residual"].shape == (15,)


def test_outer_uses_injected_linear_jacobian_action():
    class Fields:
        density = Te = Ti = Vi = Ve = vorticity = np.ones((1,))
    class Context:
        base = Fields(); active_owned = np.ones((1,), dtype=bool)
        mass_weights = np.ones((1,)); dt = 1.0
        pack = staticmethod(lambda *args: np.zeros(8))
        unpack = staticmethod(lambda value: (value,) * 7)
        admissibility = staticmethod(lambda value: {"admissible": True})
    calls = []
    def injected(value):
        calls.append(np.asarray(value).copy())
        return 2.0 * np.asarray(value)
    ctx = Context(); x = np.zeros(8); residual = -np.ones(8)
    import frozen_outer_fgmres_lower as lower
    captured = {}
    def fake_fgmres(action, b, pre, *, budget, iteration_callback=None):
        probe = np.arange(b.size, dtype=float) + 1.0
        captured["action"] = np.asarray(action(probe))
        return {"x": np.zeros_like(b), "history": [1.0], "true_residual": 1.0,
                "residual_vector": -np.asarray(b), "iterations": 1}
    old = lower.flexible_gmres; lower.flexible_gmres = fake_fgmres
    try:
        out = run_outer_whitened_fgmres(
            ctx, x, residual, lambda value: np.asarray(value) - np.ones(8),
            lambda value: (np.asarray(value), {}), budget=2,
            jacobian_action=injected)
    finally:
        lower.flexible_gmres = old
    assert calls and np.allclose(captured["action"], 2.0 * (np.arange(8) + 1.0))


def test_coupling_aware_schur_matches_dense_block_ldu():
    rng = np.random.default_rng(41); m, a = 3, 2
    M = np.eye(m) + .1*rng.standard_normal((m,m)); A = np.eye(a) + .1*rng.standard_normal((a,a))
    B = .2*rng.standard_normal((m,a)); C = .2*rng.standard_normal((a,m))
    J = np.block([[M,B],[C,A]]); pm=lambda z: np.r_[z[:m],np.zeros(a)]; pa=lambda z: np.r_[np.zeros(m),z[m:]]
    S = A - C @ np.linalg.solve(M, B)
    pre = build_coupling_aware_schur_preconditioner(
        lambda z: np.r_[np.linalg.solve(M,z[:m]),np.zeros(a)],
        lambda z: np.r_[np.zeros(m),np.linalg.solve(S,z[m:])],
        lambda z: J@z, project_material=pm, project_algebraic=pa,
        record_diagnostics=True)
    rhs = rng.standard_normal(m+a); out = pre(rhs)
    assert np.linalg.norm(J@out-rhs) < 1e-12
    assert pre.diagnostics["algebraic_inverse_calls"] == 1
    assert pre.diagnostics["complete_action_calls"] > 0
    assert {"schur_rhs", "algebraic_correction", "material_correction", "full_defect"}.issubset(pre.diagnostics)
    fast = build_coupling_aware_schur_preconditioner(
        lambda z: np.r_[np.linalg.solve(M,z[:m]),np.zeros(a)],
        lambda z: np.r_[np.zeros(m),np.linalg.solve(S,z[m:])],
        lambda z: J@z, project_material=pm, project_algebraic=pa)
    fast(rhs)
    assert fast.diagnostics["complete_action_calls"] == 2


def test_coupling_aware_schur_can_use_fixed_budget_inner_fgmres():
    rng = np.random.default_rng(42); m, a = 4, 3
    M = np.eye(m) + .1*rng.standard_normal((m,m))
    A = np.eye(a) + .1*rng.standard_normal((a,a))
    B = .2*rng.standard_normal((m,a)); C = .2*rng.standard_normal((a,m))
    J = np.block([[M,B],[C,A]])
    pm = lambda z: np.r_[z[:m],np.zeros(a)]
    pa = lambda z: np.r_[np.zeros(m),z[m:]]
    # Deliberately crude A inverse: inner FGMRES must apply the true Schur
    # action rather than accepting this as a one-shot inverse.
    pre = build_coupling_aware_schur_preconditioner(
        lambda z: np.r_[np.linalg.solve(M,z[:m]),np.zeros(a)],
        lambda z: np.r_[np.zeros(m),np.linalg.solve(A,z[m:])],
        lambda z: J@z, project_material=pm, project_algebraic=pa,
        schur_krylov_budget=a, record_diagnostics=True)
    rhs = rng.standard_normal(m+a); out = pre(rhs)
    assert np.linalg.norm(J@out-rhs) < 1e-11
    inner = pre.diagnostics["schur_krylov"]
    assert inner["iterations"] <= a
    assert inner["true_relative_residual"] < 1e-11
    assert pre.diagnostics["algebraic_inverse_calls"] == inner["iterations"]


def test_inner_schur_fgmres_honors_supplied_reduced_coordinates():
    rng = np.random.default_rng(43); m, a = 3, 2
    M = np.eye(m) + .1*rng.standard_normal((m,m))
    A = np.eye(a) + .1*rng.standard_normal((a,a))
    B = .2*rng.standard_normal((m,a)); C = .2*rng.standard_normal((a,m))
    J = np.block([[M,B],[C,A]])
    pm = lambda z: np.r_[z[:m],np.zeros(a)]
    pa = lambda z: np.r_[np.zeros(m),z[m:]]
    weights = np.array([1.0e-3, 4.0])
    to_reduced = lambda z: np.asarray(z)[m:] * weights
    to_full = lambda z: np.r_[np.zeros(m), np.asarray(z) / weights]
    pre = build_coupling_aware_schur_preconditioner(
        lambda z: np.r_[np.linalg.solve(M,z[:m]),np.zeros(a)],
        lambda z: np.r_[np.zeros(m),np.linalg.solve(A,z[m:])],
        lambda z: J@z, project_material=pm, project_algebraic=pa,
        schur_krylov_budget=a, schur_to_reduced=to_reduced,
        schur_to_full=to_full, record_diagnostics=True)
    rhs = rng.standard_normal(m+a); out = pre(rhs)
    assert np.linalg.norm(J@out-rhs) < 1e-10
    assert pre.diagnostics["schur_krylov"]["true_relative_residual"] < 1e-11


def test_physics_s0_setup_is_shared_by_probe_and_schur_modes():
    class Args:
        physics_leading_s0_probe = False; schur_base = "physics-leading-s0"
    assert _needs_s0_setup(Args())
    Args.physics_leading_s0_probe = True; Args.schur_base = "none"
    assert _needs_s0_setup(Args())


def test_s0_control_keeps_real_residual_for_schur_mode():
    source = (ROOT / "work" / "boundary_load_audit" /
              "frozen_outer_fgmres_48_control.py").read_text()
    assert "if a.physics_leading_s0_probe:\n                residual=np.zeros" in source
    assert "if _needs_s0_setup(a):" in source
    assert "for k in range(12 if a.physics_leading_s0_probe else 0):" in source


def test_coupling_factorization_uses_selected_ordered_material_inverse():
    source = (ROOT / "work" / "boundary_load_audit" /
              "frozen_outer_fgmres_48_control.py").read_text()
    assert "factorization_material_inverse = ordered_material_inverse" in source
    assert "fixed = build_coupling_aware_schur_preconditioner(\n                    factorization_material_inverse" in source
