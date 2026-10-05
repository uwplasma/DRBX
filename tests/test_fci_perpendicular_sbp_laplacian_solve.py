"""Plane-block preconditioned CG Dirichlet solve of the nodal SBP Laplacian: the previous layout with the dense core as the first
super-ring (``method="core_super_ring"``), CG against a direct solve (constant and varying polarization coefficient, lagged
preconditioner, warm start) with the default core-Schur preconditioner, and JAX hygiene. The core-Schur preconditioner itself is
tested in ``test_fci_perpendicular_sbp_laplacian_precond.py``."""
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

from drbx.native import fci_perpendicular_sbp_laplacian as lap
from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol
from drbx.native.fci_perpendicular_plane_preconditioner import apply_plane_preconditioner
from drbx.validation.sbp_laplacian_audit import LaplacianAssembly
from tests import sbp_laplacian_testbed as tb


@pytest.fixture(scope="module")
def c16():
    return tb.case(16, 8)


def test_plane_preconditioner_accepts_the_dense_core_as_the_first_super_ring(c16):
    plan = c16.plan
    asm = LaplacianAssembly(plan, c16.coeff)
    prec = sol.build_dirichlet_preconditioner(plan, c16.coeff, assembly=asm, method="core_super_ring")
    info = prec.info
    st = plan.structure
    assert info["n_owners"] == c16.E * c16.P and info["n_planes"] == c16.E and info["n_rings"] == 1 + st.m
    # the core (30 nodes, the widest block) is super-ring 0 and every ring of 16 nodes gets its own super-ring
    assert info["B"] == st.Nc and info["S"] == 1 + st.m and info["w"] >= 5 and info["max_block_cond1"] < 1e6
    # M^-1 of the in-plane blocks, applied by the JAX scans, equals the host block solves
    M = asm.matrix("dirichlet", inplane=True).tocsr()
    r = np.random.default_rng(0).standard_normal(c16.E * c16.P)
    z = np.asarray(jax.jit(apply_plane_preconditioner)(prec, r))
    P = c16.P
    ref = np.concatenate([spla.spsolve(M[k * P:(k + 1) * P, k * P:(k + 1) * P].tocsc(), r[k * P:(k + 1) * P])
                          for k in range(c16.E)])
    assert np.abs(z - ref).max() <= 1e-10 * np.abs(ref).max()
    # a float32 factorisation is a close approximation (flexible CG tolerates it)
    prec32 = sol.build_dirichlet_preconditioner(plan, c16.coeff, assembly=asm, factor_dtype="float32", method="core_super_ring")
    z32 = np.asarray(apply_plane_preconditioner(prec32, r))
    assert np.abs(z32 - ref).max() <= 1e-4 * np.abs(ref).max() and prec32.nbytes < prec.nbytes


def test_cg_matches_a_direct_solve_for_constant_and_varying_coefficients_with_lagged_preconditioner_and_warm_start(c16):
    plan, E, P = c16.plan, c16.E, c16.P
    traces = []

    def fn(lp, s, g, prec, coeff, ck, x0, maxit):
        traces.append(1)
        return sol.solve_dirichlet(lp, s, lap.LaplacianBoundaryData(value=(g,)), prec, coeff=coeff, c_kappa=ck, x0=x0,
                                   rtol=1e-11, maxit=maxit)

    solve = jax.jit(fn)
    s = c16.lap[..., 0]
    g = c16.wall_val[..., 0]
    ones = np.ones((E, P))
    zero = np.zeros((E, P))
    prec1 = sol.build_dirichlet_preconditioner(plan)
    precc = sol.build_dirichlet_preconditioner(plan, c16.coeff)
    for name, coeff, prec in (("constant", ones, prec1), ("varying", c16.coeff, precc), ("lagged", c16.coeff, prec1)):
        asm = LaplacianAssembly(plan, coeff)
        x, info = solve(plan, s, g, prec, coeff, 1.0, zero, 60)
        x = np.asarray(x)
        b = asm.data_vector(g)
        xd = spla.spsolve(asm.matrix("dirichlet").tocsc(), b - asm.H * s.ravel()).reshape(E, P)
        assert bool(info["converged"]) and float(info["relative_residual"]) <= 1e-11, name
        assert np.abs(x - xd).max() <= 1e-7 * np.abs(xd).max(), name              # the conditioning of M, not the solver
        assert int(info["iterations"]) <= (12 if name != "lagged" else 30), (name, int(info["iterations"]))
        # the discrete problem solved: L x = s (host matrix; the residual is relative to |rhs| = |b - H s|, above |H s|)
        res = (asm.matrix("dirichlet") @ x.ravel() - b + asm.H * s.ravel()).reshape(E, P)
        assert np.sqrt(np.sum(res**2 / c16.H) / np.sum(c16.H * s**2)) <= 1e-7, name
        if name == "varying":
            _, warm = solve(plan, s, g, prec, coeff, 1.0, x, 60)                       # warm start at the solution
            assert int(warm["iterations"]) <= 1
    _, short = solve(plan, s, g, prec1, ones, 1.0, zero, 1)                        # maxit is traced too
    assert not bool(short["converged"]) and int(short["iterations"]) == 1
    assert len(traces) == 1                                                       # new coefficient / preconditioner: no retrace
    # the solution of L f = L_exact(f_exact) is f_exact up to the static discretisation error
    x = np.asarray(solve(plan, s, g, prec1, ones, 1.0, zero, 60)[0])
    assert c16.h_rel_error(x - c16.vals[..., 0], c16.vals[..., 0]) <= 0.05


def test_solver_rejects_unknown_residual_norm(c16):
    prec = sol.build_dirichlet_preconditioner(c16.plan)
    with pytest.raises(ValueError, match="residual_norm"):
        sol.solve_dirichlet(c16.plan, c16.lap[..., 0], lap.LaplacianBoundaryData(value=(c16.wall_val[..., 0],)), prec,
                            residual_norm="max")
