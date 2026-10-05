"""Plane-block preconditioned CG Dirichlet solve of the nodal SBP Laplacian: the previous layout with the dense core as the first
super-ring (``method="core_super_ring"``), CG against a direct solve (constant and varying polarization coefficient, lagged
preconditioner, warm start) with the default core-Schur preconditioner, and JAX hygiene. The core-Schur preconditioner itself is
tested in ``test_fci_perpendicular_sbp_laplacian_precond.py``."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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


# ----------------------------------------------------------------------------------------------- implicit derivatives
_RTOL = 1e-13


def _implicit(lp, s, g, prec, coeff, x0):
    f, _ = sol.solve_dirichlet_implicit(lp, s, lap.LaplacianBoundaryData(value=(g,)), prec, coeff=coeff, x0=x0, rtol=_RTOL,
                                        maxit=100)
    return f


@pytest.fixture(scope="module")
def imp(c16):
    """Jitted implicit solve ``f(plan, s, g, prec, coeff, x0)`` with the lagged (unit-coefficient) preconditioner and the data."""
    prec = sol.build_dirichlet_preconditioner(c16.plan)
    return {"f": jax.jit(_implicit), "prec": prec, "s": c16.lap[..., 0], "g": c16.wall_val[..., 0], "coeff": np.asarray(c16.coeff),
            "x0": np.zeros((c16.E, c16.P)), "plan": c16.plan, "H": np.asarray(c16.H)}


def _args(imp, **over):
    d = {**imp, **over}
    return d["plan"], d["s"], d["g"], d["prec"], d["coeff"], d["x0"]


def _functional(imp):
    """Scalar functional ``sum(H f^2)/2 + sum(w f)`` of the implicit solution, as a function of ``(s, g, coeff)``."""
    w = np.random.default_rng(7).standard_normal(imp["s"].shape)
    H = imp["H"]

    def J(s, g, coeff):
        f = _implicit(imp["plan"], s, g, imp["prec"], coeff, imp["x0"])
        return 0.5 * jnp.sum(H * f * f) + jnp.sum(w * f)
    return jax.jit(J)


def test_implicit_solve_values_equal_the_cg_solve(c16, imp):
    plan, s, g, prec, coeff, x0 = _args(imp)
    f, info = sol.solve_dirichlet_implicit(plan, s, lap.LaplacianBoundaryData(value=(g,)), prec, coeff=coeff, x0=x0, rtol=1e-11,
                                           maxit=60)
    f0, info0 = sol.solve_dirichlet(plan, s, lap.LaplacianBoundaryData(value=(g,)), prec, coeff=coeff, x0=x0, rtol=1e-11, maxit=60)
    assert np.abs(np.asarray(f) - np.asarray(f0)).max() <= 1e-10 * np.abs(np.asarray(f0)).max()
    assert bool(info["converged"]) and int(info["iterations"]) == int(info0["iterations"])
    assert float(info["relative_residual"]) <= 1e-11
    # the direct solve of the discrete problem
    asm = LaplacianAssembly(plan, coeff)
    xd = spla.spsolve(asm.matrix("dirichlet").tocsc(), asm.data_vector(g) - asm.H * np.asarray(s).ravel()).reshape(c16.E, c16.P)
    assert np.abs(np.asarray(f) - xd).max() <= 1e-7 * np.abs(xd).max()
    with pytest.raises(ValueError, match="residual_norm"):
        sol.solve_dirichlet_implicit(plan, s, lap.LaplacianBoundaryData(value=(g,)), prec, residual_norm="max")


def test_implicit_jvp_and_grad_match_central_differences_in_s_and_the_dirichlet_data(imp):
    J = _functional(imp)
    s, g, coeff = (jnp.asarray(imp[k]) for k in ("s", "g", "coeff"))
    rng = np.random.default_rng(3)
    ds, dg = rng.standard_normal(s.shape), rng.standard_normal(g.shape)
    grad_s, grad_g = jax.grad(J, argnums=(0, 1))(s, g, coeff)
    h = 1e-5
    for name, d, grad in (("s", ds, grad_s), ("g", dg, grad_g)):
        def at(e):
            return J(s + e * d, g, coeff) if name == "s" else J(s, g + e * d, coeff)
        fd = (float(at(h)) - float(at(-h))) / (2 * h)
        jvp = float(jax.jvp(at, (0.0,), (1.0,))[1])
        vjp = float(jnp.sum(grad * d))
        scale = abs(fd) + 1e-3
        assert abs(jvp - fd) <= 1e-6 * scale, (name, jvp, fd)
        assert abs(vjp - fd) <= 1e-6 * scale, (name, vjp, fd)
        assert abs(jvp - vjp) <= 1e-9 * scale, (name, jvp, vjp)


def test_implicit_jvp_vjp_consistency_for_the_vector_valued_solution(imp):
    plan, prec, coeff, x0 = imp["plan"], imp["prec"], imp["coeff"], imp["x0"]

    def F(s, g):
        return _implicit(plan, s, g, prec, coeff, x0)

    F = jax.jit(F)
    s, g = jnp.asarray(imp["s"]), jnp.asarray(imp["g"])
    rng = np.random.default_rng(5)
    vs, vg = rng.standard_normal(s.shape), rng.standard_normal(g.shape)
    w = rng.standard_normal(s.shape)
    f, Jv = jax.jvp(F, (s, g), (vs, vg))
    f2, pull = jax.vjp(F, s, g)
    ws, wg = pull(w)
    lhs = float(jnp.sum(Jv * w))
    rhs = float(jnp.sum(vs * ws) + jnp.sum(vg * wg))
    assert np.abs(np.asarray(f) - np.asarray(f2)).max() == 0.0
    assert abs(lhs - rhs) <= 1e-9 * (abs(lhs) + 1e-3), (lhs, rhs)
    # the s-tangent is the linear solve itself: d f = M^-1 (-H ds), so Jv(vs only) equals the solve with rhs -H vs
    _, Jvs = jax.jvp(F, (s, g), (vs, np.zeros_like(vg)))
    asm = LaplacianAssembly(plan, coeff)
    ref = spla.spsolve(asm.matrix("dirichlet").tocsc(), -asm.H * vs.ravel()).reshape(vs.shape)
    assert np.abs(np.asarray(Jvs) - ref).max() <= 1e-7 * np.abs(ref).max()


def test_implicit_derivative_is_independent_of_the_warm_start_and_the_coefficient_derivative_is_exact(imp):
    J = _functional(imp)
    s, g, coeff = (jnp.asarray(imp[k]) for k in ("s", "g", "coeff"))
    # x0 enters only as an initial guess: the gradients do not depend on it (up to the solve tolerance), its own gradient is 0 ...
    w = np.random.default_rng(11).standard_normal(s.shape)

    def Jx0(s, g, x0):
        f = _implicit(imp["plan"], s, g, imp["prec"], jnp.asarray(imp["coeff"]), x0)
        return 0.5 * jnp.sum(imp["H"] * f * f) + jnp.sum(w * f)

    x0 = jnp.asarray(np.random.default_rng(13).standard_normal(s.shape))
    gB = jax.grad(Jx0, argnums=(0, 1, 2))(s, g, x0)
    gC = jax.grad(Jx0, argnums=(0, 1, 2))(s, g, jnp.zeros_like(s))
    for a, b in zip(gB[:2], gC[:2]):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() <= 1e-8 * np.abs(np.asarray(b)).max()
    assert float(jnp.abs(gB[2]).max()) == 0.0 and float(jnp.abs(gC[2]).max()) == 0.0   # d/d x0 = 0 exactly
    # ... and the derivative with respect to the polarization coefficient (the operator) is the exact implicit one. The
    # testbed coefficient has exactly tied extrema (mirror-symmetric nodes) in the max-based penalty scalings, where the
    # operator has a kink; a small random perturbation of the base point breaks the ties.
    coeff = coeff + 1e-2 * jnp.asarray(np.random.default_rng(19).standard_normal(coeff.shape))
    d = np.random.default_rng(17).standard_normal(coeff.shape)
    h = 1e-5
    fd = (float(J(s, g, coeff + h * d)) - float(J(s, g, coeff - h * d))) / (2 * h)
    jvp = float(jax.jvp(lambda e: J(s, g, coeff + e * d), (0.0,), (1.0,))[1])
    vjp = float(jnp.sum(jax.grad(J, argnums=2)(s, g, coeff) * d))
    assert abs(jvp - fd) <= 1e-6 * (abs(fd) + 1e-3) and abs(vjp - jvp) <= 1e-9 * (abs(jvp) + 1e-3), (fd, jvp, vjp)


def test_implicit_solve_under_jit_with_plan_and_preconditioner_as_arguments(c16, imp):
    traces = []

    def fn(lp, s, g, prec, coeff, x0):
        traces.append(1)
        return jnp.sum(_implicit(lp, s, g, prec, coeff, x0) ** 2)

    plan, s, g, prec, coeff, x0 = _args(imp)
    val_grad = jax.jit(jax.value_and_grad(fn, argnums=(1, 2)))
    v, (gs, gg) = val_grad(plan, s, g, prec, coeff, x0)
    precc = sol.build_dirichlet_preconditioner(plan, coeff)                 # a different (exact-coefficient) preconditioner
    v2, (gs2, gg2) = val_grad(plan, s, g, precc, coeff, x0)
    assert len(traces) == 1                                                  # same pytree structure: no retrace
    assert abs(float(v) - float(v2)) <= 1e-9 * abs(float(v2))                # the preconditioner does not change the solution
    for a, b in ((gs, gs2), (gg, gg2)):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() <= 1e-8 * np.abs(np.asarray(b)).max()
    # the jitted public wrapper equals the eager call
    f, info = sol.solve_dirichlet_implicit_jit(plan, s, lap.LaplacianBoundaryData(value=(g,)), prec, coeff=coeff, x0=x0, rtol=1e-11)
    assert bool(info["converged"])
    assert np.abs(np.asarray(f) - np.asarray(_implicit(plan, s, g, prec, coeff, x0))).max() <= 1e-9 * float(jnp.abs(f).max())
    # zero data and zero source: exactly zero solution and zero tangent
    z = jnp.zeros_like(s)
    f0, tang = jax.jvp(lambda a: _implicit(plan, a, jnp.zeros_like(g), prec, coeff, x0), (z,), (z,))
    assert float(jnp.abs(f0).max()) == 0.0 and float(jnp.abs(tang).max()) == 0.0
