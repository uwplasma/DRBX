"""Fast tests for the lean P07 Dirichlet GMRES solve (synthetic non-symmetric operator)."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from drbx.native.fci_perpendicular_p07_solve import (
    P07SolveConfig, direct_solve_p07, p07_linear_system, solve_p07_dirichlet,
    solve_p07_dirichlet_jit)

try:  # real implementation when the parallel module exists
    from drbx.native.fci_perpendicular_p07_sparse import boundary_source as _real_boundary_source
except ImportError:  # pragma: no cover
    _real_boundary_source = None

QD, QN = 10, 4


def _local_boundary_source(op, bc):
    gv = np.asarray(bc.dirichlet_value)
    gt = np.asarray(bc.dirichlet_tangential).reshape(2 * gv.shape[0], -1)
    gn = np.asarray(bc.neumann_normal)
    return op.dirichlet_value @ gv + op.dirichlet_tangential @ gt + op.neumann_normal @ gn


def _make_operator(seed=0, nx=12, ny=10, scale_spread=0.0, kind="dirichlet"):
    rng = np.random.default_rng(seed)
    n = nx * ny
    volume = rng.uniform(0.5, 2.0, n)
    ex, ey = sp.eye(nx), sp.eye(ny)
    d1 = lambda m: sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    lap = sp.kron(d1(nx), ey) + sp.kron(ex, d1(ny)) + 0.1 * sp.eye(n)
    skew = sp.kron(sp.diags([1.0, -1.0], [1, -1], shape=(nx, nx)), ey)
    mat = sp.diags(1.0 / volume) @ (lap + 0.05 * skew)
    if scale_spread:
        mat = sp.diags(np.exp(scale_spread * rng.standard_normal(n))) @ mat
    blk = lambda cols: sp.random(n, cols, density=0.1, random_state=seed, format="csr")
    op = SimpleNamespace(
        kind=kind, matrix=sp.csr_matrix(mat), owner_volume=volume,
        dirichlet_value=blk(QD), dirichlet_tangential=blk(2 * QD), neumann_normal=blk(QN),
        dirichlet_points=np.zeros((QD, 3)), neumann_points=np.zeros((QN, 3)))
    bc = SimpleNamespace(
        dirichlet_value=rng.standard_normal((QD, 1)),
        dirichlet_tangential=rng.standard_normal((QD, 2, 1)),
        neumann_normal=rng.standard_normal((QN, 1)))
    return op, bc, rng.standard_normal(n)


def _bsrc(op, bc):
    return (_real_boundary_source or _local_boundary_source)(op, bc)


def _rel(a, b):
    return np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b)


def _direct(op, rhs, bc):
    """Oracle with a locally computed boundary term (independent of the real module)."""
    return direct_solve_p07(op, rhs - _local_boundary_source(op, bc)[:, 0], None)


@pytest.mark.parametrize("pc", ["none", "jacobi"])
def test_gmres_matches_direct(pc):
    op, bc, rhs = _make_operator()
    system = p07_linear_system(op)
    cfg = P07SolveConfig(rtol=1e-11, preconditioner=pc)
    x, info = solve_p07_dirichlet(system, jnp.asarray(rhs), config=cfg)
    assert _rel(x, direct_solve_p07(op, rhs)) < 1e-8
    assert bool(info["converged"]) and int(info["iterations"]) > 0
    assert float(info["residual_norm"]) <= 1e-11 * float(info["rhs_norm"]) * (1 + 1e-6) + 1e-300
    assert float(info["relative_residual"]) <= 1e-11 * (1 + 1e-6)


def test_boundary_term_path():
    op, bc, rhs = _make_operator(seed=1)
    system = p07_linear_system(op)
    bterm = _local_boundary_source(op, bc)[:, 0]
    assert np.linalg.norm(bterm) > 0
    cfg = P07SolveConfig(rtol=1e-11)
    x, info = solve_p07_dirichlet(system, jnp.asarray(rhs), boundary_term=jnp.asarray(bterm), config=cfg)
    x_ref, _ = solve_p07_dirichlet(system, jnp.asarray(rhs - bterm), config=cfg)
    assert bool(info["converged"])
    assert _rel(x, np.asarray(x_ref)) < 1e-9
    assert _rel(x, _direct(op, rhs, bc)) < 1e-8
    if _real_boundary_source is not None:
        assert _rel(direct_solve_p07(op, rhs, bc), _direct(op, rhs, bc)) < 1e-12
        np.testing.assert_allclose(_bsrc(op, bc), _local_boundary_source(op, bc), atol=1e-12)


def test_jacobi_not_worse_on_badly_scaled():
    op, _, rhs = _make_operator(seed=2, scale_spread=1.5)
    system = p07_linear_system(op)
    base = dict(rtol=1e-10, restart=30, max_restarts=40)
    _, i0 = solve_p07_dirichlet(system, jnp.asarray(rhs), config=P07SolveConfig(preconditioner="none", **base))
    xj, i1 = solve_p07_dirichlet(system, jnp.asarray(rhs), config=P07SolveConfig(preconditioner="jacobi", **base))
    assert bool(i1["converged"])
    assert int(i1["iterations"]) <= int(i0["iterations"])
    assert _rel(xj, direct_solve_p07(op, rhs)) < 1e-6


def test_eager_vs_jit():
    op, bc, rhs = _make_operator(seed=3)
    system = p07_linear_system(op)
    bterm = jnp.asarray(_local_boundary_source(op, bc)[:, 0])
    cfg = P07SolveConfig(rtol=1e-12)
    x0, i0 = solve_p07_dirichlet(system, jnp.asarray(rhs), boundary_term=bterm, config=cfg)
    x1, i1 = solve_p07_dirichlet_jit(system, jnp.asarray(rhs), boundary_term=bterm, config=cfg)
    assert float(jnp.max(jnp.abs(x0 - x1))) < 1e-13 * max(1.0, float(jnp.max(jnp.abs(x0))))
    assert int(i0["iterations"]) == int(i1["iterations"])
    assert bool(i1["converged"])


def test_custom_preconditioner_exact_inverse():
    op, _, rhs = _make_operator(seed=4)
    system = p07_linear_system(op)
    inv = jnp.asarray(np.linalg.inv(op.matrix.toarray()))
    cfg = P07SolveConfig(rtol=1e-10)
    x, info = solve_p07_dirichlet_jit(
        system, jnp.asarray(rhs), config=cfg, preconditioner=lambda r: inv @ r)
    assert bool(info["converged"]) and 0 < int(info["iterations"]) <= 2
    assert _rel(x, direct_solve_p07(op, rhs)) < 1e-9


def test_x0_warm_start_reduces_iterations():
    op, _, rhs = _make_operator(seed=5)
    system = p07_linear_system(op)
    cfg = P07SolveConfig(rtol=1e-10)
    x, i0 = solve_p07_dirichlet(system, jnp.asarray(rhs), config=cfg)
    _, i1 = solve_p07_dirichlet(system, jnp.asarray(rhs), config=cfg, x0=x)
    assert int(i1["iterations"]) <= int(i0["iterations"])
    assert bool(i1["converged"])


def test_rejects_neumann_and_nonpositive_diagonal():
    op, _, _ = _make_operator(kind="neumann")
    with pytest.raises(ValueError, match="dirichlet"):
        p07_linear_system(op)
    op, _, _ = _make_operator()
    bad = op.matrix.tolil()
    bad[3, 3] = 0.0
    op.matrix = bad.tocsr()
    with pytest.raises(ValueError, match="diagonal"):
        p07_linear_system(op)
    with pytest.raises(ValueError, match="preconditioner"):
        P07SolveConfig(preconditioner="ilu")
