"""Core-Schur plane preconditioner of the nodal SBP Laplacian and the windowed in-plane assembly: the preconditioner is the exact
(symmetric positive definite) inverse of every in-plane block, CG takes the iterations of the previous layout, and the windowed
assembly gives bitwise the blocks of the full host assembly."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.geometry.sbp_laplacian_assembly import LaplacianAssembly, iter_inplane_blocks, plane_group_size
from drbx.native import fci_perpendicular_sbp_laplacian as lap
from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol
from drbx.native.fci_perpendicular_plane_preconditioner import (
    CoreSchurPreconditioner, apply_core_schur_preconditioner, build_core_schur_preconditioner)
from drbx.validation import sbp_laplacian_audit
from tests import sbp_laplacian_testbed as tb


@pytest.fixture(scope="module")
def c16():
    return tb.case(16, 8)


@pytest.fixture(scope="module")
def blocks(c16):
    """Full host in-plane matrices (constant and varying coefficient) and the per-plane exact solves of a random vector."""
    out = {}
    r = np.random.default_rng(1).standard_normal(c16.E * c16.P)
    P = c16.P
    for name, coeff in (("constant", None), ("varying", c16.coeff)):
        M = LaplacianAssembly(c16.plan, coeff).matrix("dirichlet", inplane=True).tocsr()
        ref = np.concatenate([spla.spsolve(M[k * P:(k + 1) * P, k * P:(k + 1) * P].tocsc(), r[k * P:(k + 1) * P])
                              for k in range(c16.E)])
        out[name] = (coeff, M, ref)
    return r, out


def _same_csr(a, b):
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices()
    b.sort_indices()
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


# ------------------------------------------------------------------------------------------------ windowed assembly
@pytest.mark.parametrize("name", ["constant", "varying"])
def test_windowed_inplane_assembly_equals_the_full_host_assembly_bitwise(c16, blocks, name):
    coeff, M, _ = blocks[1][name]
    P = c16.P
    for group in (1, 3, 8):
        seen = 0
        for k0, k1, B in iter_inplane_blocks(c16.plan, coeff, group_planes=group):
            assert k0 == seen and k1 - k0 <= group
            assert _same_csr(B, M[k0 * P:k1 * P, k0 * P:k1 * P]), (name, group, k0)
            seen = k1
        assert seen == c16.E
    # Neumann blocks too, a non-default c_kappa, and a single window through the class
    Mn = LaplacianAssembly(c16.plan, coeff, 2.5).matrix("neumann", inplane=True)
    win = LaplacianAssembly(c16.plan, coeff, 2.5, window=(2, 5)).matrix("neumann", inplane=True)
    assert _same_csr(win, Mn[2 * P:5 * P, 2 * P:5 * P])


def test_windowed_assembly_rejects_bad_windows_and_full_matrices(c16):
    with pytest.raises(ValueError, match="window"):
        LaplacianAssembly(c16.plan, window=(3, 3))
    with pytest.raises(ValueError, match="window"):
        LaplacianAssembly(c16.plan, window=(0, c16.E + 1))
    win = LaplacianAssembly(c16.plan, window=(0, 2))
    with pytest.raises(ValueError, match="in-plane"):
        win.matrix("dirichlet")
    with pytest.raises(ValueError, match="in-plane"):
        win.data_vector(np.zeros((2, c16.N)))
    assert 1 <= plane_group_size(c16.plan.structure) <= c16.E


def test_production_solver_does_not_import_from_validation_and_the_audit_reexports_the_assembly():
    src = Path(sol.__file__).read_text()
    assert "drbx.validation" not in src
    assert sbp_laplacian_audit.LaplacianAssembly is LaplacianAssembly


# ------------------------------------------------------------------------------------------------ the preconditioner
@pytest.mark.parametrize("name", ["constant", "varying"])
@pytest.mark.parametrize("rpb", ["auto", 1, 2, 3, 5])
def test_core_schur_preconditioner_is_the_exact_per_plane_inverse(c16, blocks, name, rpb):
    r, mats = blocks
    coeff, _, ref = mats[name]
    prec = sol.build_dirichlet_preconditioner(c16.plan, coeff, rings_per_block=rpb, group_planes=3)
    assert isinstance(prec, CoreSchurPreconditioner) and prec.info["n_core"] == c16.plan.structure.Nc
    m = c16.plan.structure.m
    assert prec.meta.B == prec.meta.rings_per_block * c16.N and prec.meta.S * prec.meta.rings_per_block >= m
    assert prec.info["max_block_cond1"] < 1e6
    z = np.asarray(jax.jit(apply_core_schur_preconditioner)(prec, r))
    assert np.abs(z - ref).max() <= 1e-11 * np.abs(ref).max()
    if rpb == "auto":                                                    # float32 factors: a close approximation
        p32 = sol.build_dirichlet_preconditioner(c16.plan, coeff, factor_dtype="float32")
        z32 = np.asarray(sol.apply_preconditioner(p32, r))
        assert np.abs(z32 - ref).max() <= 1e-5 * np.abs(ref).max() and p32.nbytes < prec.nbytes
        assert p32.meta.dtype == "float32" and z32.dtype == np.float64


def test_core_schur_preconditioner_from_a_given_assembly_and_the_group_size_do_not_matter(c16, blocks):
    r, mats = blocks
    coeff, _, ref = mats["varying"]
    asm = LaplacianAssembly(c16.plan, coeff)
    z = []
    for kw in (dict(assembly=asm, group_planes=3), dict(group_planes=1), dict(group_planes=8)):
        prec = sol.build_dirichlet_preconditioner(c16.plan, coeff, rings_per_block=2, **kw)
        z.append(np.asarray(sol.apply_preconditioner(prec, r)))
    assert all(np.array_equal(z[0], zi) for zi in z[1:])                  # bitwise: same blocks, same arithmetic
    assert np.abs(z[0] - ref).max() <= 1e-11 * np.abs(ref).max()


@pytest.mark.parametrize("dtype,tol", [("float64", 1e-12), ("float32", 1e-6)])
def test_core_schur_preconditioner_is_symmetric_positive_definite(c16, dtype, tol):
    prec = sol.build_dirichlet_preconditioner(c16.plan, c16.coeff, factor_dtype=dtype, rings_per_block=2, group_planes=4)
    n = c16.E * c16.P
    Z = np.asarray(jax.jit(jax.vmap(lambda v: apply_core_schur_preconditioner(prec, v)))(np.eye(n)))   # rows = M^-1 e_i
    assert np.abs(Z - Z.T).max() <= tol * np.abs(Z).max()
    assert np.linalg.eigvalsh(0.5 * (Z + Z.T)).min() > 0.0


def test_cg_iterations_equal_those_of_the_core_super_ring_preconditioner(c16):
    plan, E, P = c16.plan, c16.E, c16.P
    s, g = c16.lap[..., 0], c16.wall_val[..., 0]
    solve = jax.jit(lambda lp, s, g, pr, coeff, x0: sol.solve_dirichlet(
        lp, s, lap.LaplacianBoundaryData(value=(g,)), pr, coeff=coeff, x0=x0, rtol=1e-11, maxit=60))
    zero, ones = np.zeros((E, P)), np.ones((E, P))
    for dtype in ("float64", "float32"):
        for coeff in (ones, c16.coeff):
            old = sol.build_dirichlet_preconditioner(plan, coeff, method="core_super_ring", factor_dtype=dtype)
            new = sol.build_dirichlet_preconditioner(plan, coeff, factor_dtype=dtype)
            xo, io = solve(plan, s, g, old, coeff, zero)
            xn, inn = solve(plan, s, g, new, coeff, zero)
            assert bool(io["converged"]) and bool(inn["converged"])
            assert int(inn["iterations"]) == int(io["iterations"]), (dtype, int(inn["iterations"]), int(io["iterations"]))
            assert np.abs(np.asarray(xn) - np.asarray(xo)).max() <= 1e-8 * np.abs(np.asarray(xo)).max()


def test_builders_reject_bad_input(c16):
    with pytest.raises(ValueError, match="method"):
        sol.build_dirichlet_preconditioner(c16.plan, method="jacobi")
    with pytest.raises(ValueError, match="max_block"):
        sol.build_dirichlet_preconditioner(c16.plan, max_block=500)
    with pytest.raises(ValueError, match="rings_per_block"):
        sol.build_dirichlet_preconditioner(c16.plan, rings_per_block=0)
    st = c16.plan.structure
    kw = dict(n_planes=st.n_eta, n_core=st.Nc, n_rings=st.m, ring_size=st.N)
    groups = list(iter_inplane_blocks(c16.plan, group_planes=4))
    with pytest.raises(ValueError, match="consecutive"):
        build_core_schur_preconditioner(groups[1:], **kw)
    with pytest.raises(ValueError, match="cover"):
        build_core_schur_preconditioner(groups[:1], **kw)
    with pytest.raises(ValueError, match="between planes|couples different planes"):
        full = LaplacianAssembly(c16.plan).matrix("dirichlet").tocsr()
        build_core_schur_preconditioner([(0, st.n_eta, full)], **kw)
