"""Fast synthetic tests of ``drbx.native.fci_perpendicular_phi_solver`` and ``solve_p07_dirichlet_with``.

The synthetic P07 operator is the anisotropic flux-form operator of ``test_p08_step5_preconditioners`` on an
``n x n x n`` (ring, theta, plane) grid (``raw_to_owner = identity``, so ``ring = i``, ``theta = j``, ``plane = k``),
with weak plane coupling, a skew-symmetric perturbation and wall conductances, plus a nonzero boundary-data block.
The exported operator of the synthetic *plan* of ``tests/perpendicular_synthetic`` is random (not positive), so
``build_phi_solver`` is exercised through a patched export.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import drbx.native.fci_perpendicular_phi_solver as phi_mod                                            # noqa: E402
from drbx.native.fci_perpendicular_p07_solve import (                                                 # noqa: E402
    P07SolveConfig, direct_solve_p07, p07_linear_system, solve_p07_dirichlet_with,
    solve_p07_dirichlet_with_jit)
from drbx.native.fci_perpendicular_p07_sparse import P07SparseOperator                               # noqa: E402
from drbx.native.fci_perpendicular_phi_solver import (                                                # noqa: E402
    PHI_RTOL_DEFAULT, PhiSolver, build_phi_solver, phi_boundary_term, phi_solver_from_operator, solve_phi)
from drbx.native.fci_perpendicular_plane_preconditioner import apply_plane_preconditioner            # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData                           # noqa: E402

NG = 6                       # grid size n
N = NG ** 3
QD = 5                       # Dirichlet trace points


def build_operator(n=NG, eps_eta=2e-2, skew=0.15, seed=0):
    rng = np.random.default_rng(seed)
    nn = n ** 3
    cell = lambda i, j, k: (i * n + j) * n + k                     # == raw cell index (ring i, theta j, plane k)
    vol = rng.uniform(0.6, 1.4, nn)
    coef = (1.0, 0.7, eps_eta)
    kr, kc, kv = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for axis, d in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                    i2, j2, k2 = i + d[0], j + d[1], k + d[2]
                    if i2 >= n or j2 >= n or k2 >= n:
                        continue
                    a, b = cell(i, j, k), cell(i2, j2, k2)
                    c = coef[axis] * rng.uniform(0.8, 1.2)
                    kr += [a, a, b, b, a, b]
                    kc += [a, b, a, b, b, a]
                    kv += [c, -c, -c, c, skew * c, -skew * c]
    wall = [cell(i, j, k) for i in range(n) for j in range(n) for k in range(n)
            if i in (0, n - 1) or j in (0, n - 1) or k in (0, n - 1)]
    kr += wall
    kc += wall
    kv += list(rng.uniform(0.8, 1.2, len(wall)))
    kmat = sp.csr_matrix((kv, (kr, kc)), shape=(nn, nn))
    a = (sp.diags(1.0 / vol) @ kmat).tocsr()
    a.sort_indices()
    rows = rng.choice(wall, size=3 * QD, replace=False)
    bval = sp.csr_matrix((rng.uniform(0.2, 0.8, 3 * QD), (rows, np.tile(np.arange(QD), 3))), shape=(nn, QD))
    btan = sp.csr_matrix((rng.uniform(-0.2, 0.2, 2 * QD), (rows[:2 * QD], np.arange(2 * QD))), shape=(nn, 2 * QD))
    op = P07SparseOperator("dirichlet", a, bval, btan, sp.csr_matrix((nn, 0)), vol, np.zeros((QD, 3)),
                           np.zeros((0, 3)))
    return op, np.arange(nn, dtype=np.int64)


def make_bc(seed=3, fields=1):
    rng = np.random.default_rng(seed)
    return BoundaryData(rng.normal(size=(QD, fields)), rng.normal(size=(QD, 2, fields)), np.zeros((0, fields)))


@pytest.fixture(scope="module")
def world():
    op, raw = build_operator()
    solver = phi_solver_from_operator(op, raw, NG)
    rng = np.random.default_rng(5)
    phi = rng.normal(size=N)
    return op, raw, solver, phi, op.matrix @ phi


def _m_norm(op, x):
    return float(np.sqrt(np.sum(op.owner_volume * x * x)))


def test_default_tolerance_and_config(world):
    _, _, solver, _, _ = world
    assert PHI_RTOL_DEFAULT == 1e-8
    assert isinstance(solver, PhiSolver)
    cfg = solver.config
    assert (cfg.rtol, cfg.restart, cfg.max_restarts) == (1e-8, 50, 40)
    assert solver.prec.dtype == "float32"
    assert set(solver.setup_seconds) == {"system", "layout", "preconditioner"}
    assert all(v >= 0 for v in solver.setup_seconds.values())
    custom = phi_solver_from_operator(solver.op, np.arange(N), NG, factor_dtype="float64", rtol=1e-6, restart=30,
                                      max_restarts=3)
    assert custom.prec.dtype == "float64" and custom.config.rtol == 1e-6 and custom.config.restart == 30


def test_recovers_known_solution(world):
    op, _, solver, phi, rhs = world
    x, info = solve_phi(solver, rhs)
    assert info["converged"] and info["relative_residual"] <= 1e-8
    assert info["residual_norm"] <= 1e-8 * info["rhs_norm"] * (1 + 1e-9)
    assert 0 < info["iterations"] < 40 and info["seconds"] > 0
    assert x.dtype == np.float64 and x.shape == (N,)
    cond_bound = 1e3                                                    # generous bound of the operator conditioning
    assert np.linalg.norm(x - phi) / np.linalg.norm(phi) < 1e-8 * cond_bound
    ref = direct_solve_p07(op, rhs)
    assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-6
    # independent residual in the M norm
    res = rhs - op.matrix @ x
    assert _m_norm(op, res) <= 1e-8 * _m_norm(op, rhs) * (1 + 1e-6)


def test_tighter_tolerance_is_honoured(world):
    op, raw, _, phi, rhs = world
    tight = phi_solver_from_operator(op, raw, NG, rtol=1e-12, factor_dtype="float64")
    x, info = solve_phi(tight, rhs)
    assert info["converged"] and info["relative_residual"] <= 1e-12
    assert np.linalg.norm(x - phi) / np.linalg.norm(phi) < 1e-9


def test_warm_start(world):
    op, _, solver, phi, rhs = world
    x, cold = solve_phi(solver, rhs)
    exact = np.linalg.solve(op.matrix.toarray(), rhs)
    xw, warm = solve_phi(solver, rhs, x0=exact)
    assert warm["iterations"] == 0 and warm["converged"]
    np.testing.assert_allclose(xw, exact, rtol=1e-12, atol=1e-13 * np.abs(exact).max())
    # a perturbed previous solution converges in fewer iterations than a cold start
    rng = np.random.default_rng(6)
    x0 = exact + 1e-3 * np.linalg.norm(exact) / np.sqrt(N) * rng.normal(size=N)
    xp, near = solve_phi(solver, rhs, x0=x0)
    assert near["converged"] and near["iterations"] < cold["iterations"]
    with pytest.raises(ValueError, match="x0"):
        solve_phi(solver, rhs, x0=np.zeros(N - 1))
    with pytest.raises(ValueError, match="rhs"):
        solve_phi(solver, rhs[:-1])


def test_boundary_term_path(world):
    op, _, solver, phi, _ = world
    bc = make_bc()
    bt = phi_boundary_term(solver, bc)
    assert bt.shape == (N,) and np.abs(bt).max() > 0
    expected = np.asarray(op.dirichlet_value @ bc.dirichlet_value[:, 0] + op.dirichlet_tangential @ bc.dirichlet_tangential.reshape(2 * QD, 1)[:, 0])
    np.testing.assert_allclose(bt, expected, rtol=1e-13, atol=1e-14)
    rhs = op.matrix @ phi + bt                                       # A phi + B g = rhs  =>  solving A x = rhs - B g gives phi
    x, info = solve_phi(solver, rhs, bc)
    assert info["converged"]
    assert np.linalg.norm(x - phi) / np.linalg.norm(phi) < 1e-5
    np.testing.assert_allclose(x, direct_solve_p07(op, rhs, bc), rtol=1e-6, atol=1e-7 * np.abs(phi).max())
    # without bc the same rhs solves a different problem
    xnb, _ = solve_phi(solver, rhs)
    assert np.linalg.norm(xnb - phi) / np.linalg.norm(phi) > 1e-3
    with pytest.raises(ValueError, match="one field"):
        phi_boundary_term(solver, make_bc(fields=2))


def test_solver_with_data_argument_matches_closure_solve(world):
    op, _, solver, _, rhs = world
    system = p07_linear_system(op)
    cfg = P07SolveConfig(rtol=1e-10, restart=50, max_restarts=40)
    bt = jnp.asarray(0.1 * np.random.default_rng(8).normal(size=N))
    x, info = solve_p07_dirichlet_with_jit(system, jnp.asarray(rhs), solver.prec, apply=apply_plane_preconditioner,
                                           boundary_term=bt, config=cfg)
    xe, infoe = solve_p07_dirichlet_with(system, jnp.asarray(rhs), solver.prec, apply=apply_plane_preconditioner,
                                         boundary_term=bt, config=cfg)
    from drbx.native.fci_perpendicular_p07_solve import solve_p07_dirichlet_jit
    xc, infoc = solve_p07_dirichlet_jit(system, jnp.asarray(rhs), boundary_term=bt, config=cfg,
                                        preconditioner=lambda r: apply_plane_preconditioner(solver.prec, r))
    for other in (xe, xc):
        np.testing.assert_allclose(np.asarray(x), np.asarray(other), rtol=1e-12, atol=1e-13 * float(jnp.abs(x).max()))
    assert int(info["iterations"]) == int(infoc["iterations"]) and bool(info["converged"])
    assert set(info) == set(infoc)
    # the factors are an argument of the jitted function, not constants
    closed = jax.make_jaxpr(lambda s, r, p: solve_p07_dirichlet_with(
        s, r, p, apply=apply_plane_preconditioner, config=cfg))(system, jnp.asarray(rhs), solver.prec)
    factor_bytes = sum(int(x.nbytes) for x in (solver.prec.lo, solver.prec.up, solver.prec.dinv))
    assert sum(int(np.asarray(c).nbytes) for c in closed.consts) < 0.05 * factor_bytes


def test_build_phi_solver_from_plan(world, monkeypatch):
    op, raw, _, phi, rhs = world
    monkeypatch.setattr(phi_mod, "export_p07_sparse", lambda plan, kind: (op if (plan, kind) == ("plan", "dirichlet")
                                                                          else pytest.fail("wrong arguments")))
    solver = build_phi_solver("plan", raw_to_owner=raw, n=NG)
    assert set(solver.setup_seconds) == {"export", "system", "layout", "preconditioner"}
    assert solver.config.rtol == PHI_RTOL_DEFAULT
    x, info = solve_phi(solver, rhs)
    assert info["converged"] and np.linalg.norm(x - phi) / np.linalg.norm(phi) < 1e-5


def test_validation(world):
    op, raw, _, _, _ = world
    neumann = P07SparseOperator("neumann", op.matrix, op.dirichlet_value, op.dirichlet_tangential,
                                op.neumann_normal, op.owner_volume, op.dirichlet_points, op.neumann_points)
    with pytest.raises(ValueError, match="dirichlet"):
        phi_solver_from_operator(neumann, raw, NG)
    with pytest.raises(ValueError, match="raw_to_owner"):
        phi_solver_from_operator(op, raw[:-1], NG)
    short = raw.copy()
    short[-1] = -1                                                   # a valid map with one owner fewer
    with pytest.raises(ValueError, match="owners"):
        phi_solver_from_operator(op, short, NG)
