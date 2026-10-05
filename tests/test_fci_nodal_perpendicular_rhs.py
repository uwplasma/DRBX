"""Single-device nodal perpendicular RHS and potential solve (chunk C1): composition of the standalone SBP operators, the
``psi`` solve against the two-term form, prescribed ``phi``, constants, the bracket wall rule, jit hygiene and validation.

Family A, ``n = 16``, ``n_eta = 8``; the nodal plan and the Laplacian plan are built from one jacobian so that ``Hp`` agrees
bitwise (the context builder asserts it).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.geometry.nodal_layout import node_points, wall_points
from drbx.native import fci_nodal_perpendicular_rhs as rhs
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket
from drbx.native.fci_perpendicular_sbp_curvature import sbp_curvature
from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action
from drbx.native.fci_perpendicular_sbp_laplacian_solve import solve_dirichlet
from drbx.native.fci_perpendicular_sbp_ops import trace
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable
from tests import sbp_laplacian_testbed as tb

ALL = rhs.FIELD_NAMES
KINDS4 = ("dirichlet", "dirichlet", "neumann", "dirichlet")        # density, Te, Ti, vorticity
KINDS6 = ("dirichlet", "dirichlet", "neumann", "dirichlet", "neumann", "dirichlet")


def _nodal_metric_fn(pts):
    """Smooth ``(h, jac, B, K)`` at logical points; the jacobian is the testbed's, so ``Hp`` matches the Laplacian plan's."""
    _A, J, _G = tb._batched(tb._geometry_fn, pts)
    u, th, et = np.asarray(pts).T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    K = np.stack([0.3 * np.cos(th), 0.1 * u + 0.05, 0.2 * np.sin(et) + 0.1], -1)
    return h, np.abs(J), 1.0 + 0.2 * u, K


class Geo:
    def __init__(self):
        self.case = tb.case(16, 8, False)
        lay = self.layout = self.case.layout
        self.lplan = self.case.plan
        self.plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, _nodal_metric_fn))
        self.E, self.P, self.N = lay.n_eta, lay.P, self.case.N
        self.pts = node_points(lay)
        self.wpts = wall_points(lay, lay.walls[0])
        self.ctx = rhs.build_nodal_perpendicular_context(self.plan, self.lplan)
        self.ctx_nosolve = rhs.build_nodal_perpendicular_context(self.plan, self.lplan, build_preconditioner=False)


@pytest.fixture(scope="module")
def geo():
    return Geo()


def field_values(pts, k):
    """Smooth, positive (density, Te, Ti) analytic fields ``k = 0..5`` at points ``(..., 3)`` (u, theta, eta)."""
    u, th, et = pts[..., 0], pts[..., 1], pts[..., 2]
    x, y = u * np.cos(th), u * np.sin(th)
    s = 2 * np.pi * (0.9 * x + 0.4 * y) + et
    return [1.0 + 0.3 * np.cos(s), 0.9 + 0.2 * np.sin(s + 0.3), 1.1 + 0.25 * np.cos(2 * np.pi * (0.5 * x - 0.8 * y) + et),
            0.2 * np.sin(s), 0.15 * np.cos(s + 1.0), 0.3 * np.sin(2 * np.pi * x + et) + 0.2 * np.cos(2 * np.pi * y)][k]


def make_state(geo, fields):
    idx = [ALL.index(f) for f in fields]
    state = np.stack([field_values(geo.pts, k) for k in idx], -1)
    wall = np.stack([field_values(geo.wpts, k) for k in idx], -1)
    return jnp.asarray(state), jnp.asarray(wall)


def phi_field(geo):
    return jnp.asarray(0.4 * np.sin(2 * np.pi * geo.pts[..., 0] * np.cos(geo.pts[..., 1]) + geo.pts[..., 2]) + 0.1)


def phi_wall(geo):
    return jnp.asarray(0.4 * np.sin(2 * np.pi * geo.wpts[..., 0] * np.cos(geo.wpts[..., 1]) + geo.wpts[..., 2]) + 0.1)


def make_inputs(geo, fields, *, normal=True):
    state, value = make_state(geo, fields)
    normal_data = jnp.asarray(0.3 * np.cos(3.0 * geo.wpts[..., 1])[..., None] + 0.1 * np.asarray(value)) if normal else None
    params = rhs.NodalPerpendicularParams(jnp.asarray(0.7), jnp.asarray(0.6), jnp.asarray(0.01 * (1 + np.arange(len(fields)))))
    return state, params, rhs.NodalWallData(value, normal_data, None)


def exact_equal(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def direct_terms(geo, opts, params, state, wall, phi):
    """The standalone operators called with the same arguments."""
    bracket = sbp_bracket(geo.plan, phi, state, SatBoundaryData((wall.value,)), params.rho_star, opts.bracket_c_kappa)
    idx = list(opts.curvature_index) if "curvature" in opts.terms else None
    curv = jnp.zeros_like(state)
    if idx is not None:
        q = jnp.concatenate([state[..., idx], phi[..., None]], -1)
        c = sbp_curvature(geo.plan, q, SatBoundaryData((wall.value[..., idx],)), tau=params.tau, psi=opts.psi,
                          absolute_method=opts.absolute_method, jump_dissipation=opts.curvature_jump_dissipation,
                          c_kappa=opts.curvature_c_kappa)
        curv = curv.at[..., idx].set(c)
    diff = jnp.zeros_like(state)
    if "diffusion" in opts.terms:
        key = "normal_derivative" if opts.neumann_mode == "physical" else "conormal"
        bcd = LaplacianBoundaryData(value=(wall.value,), **{key: (wall.normal,)})
        diff = params.D * laplacian_action(geo.lplan, state, bcd, opts.diffusion_kinds, None, opts.laplacian_c_kappa,
                                           neumann_mode=opts.neumann_mode)
    return bracket, curv, diff


# ----------------------------------------------------------------------------------------------- geometry helper
def test_node_points_layout(geo):
    pts = geo.pts
    assert pts.shape == (geo.E, geo.P, 3)
    assert np.array_equal(pts[0, :, 0], geo.layout.node_u) and np.array_equal(pts[3, :, 1], geo.layout.node_theta)
    assert np.allclose(pts[:, 0, 2], (np.arange(geo.E) + 0.5) * geo.layout.deta)
    assert tb.node_points is node_points


def test_context_builds_and_checks_hp(geo):
    ctx = geo.ctx
    assert ctx.prec is not None and geo.ctx_nosolve.prec is None
    assert ctx.curvature_flux.shape == (geo.E, geo.P, 3)
    bad = dataclasses.replace(geo.lplan, Hp=np.asarray(geo.lplan.Hp) * (1.0 + 1e-12))
    with pytest.raises(ValueError, match="Hp"):
        rhs.build_nodal_perpendicular_context(geo.plan, bad, build_preconditioner=False)
    with pytest.raises(ValueError, match="either"):
        rhs.build_nodal_perpendicular_context(geo.plan, geo.lplan, prec=ctx.prec, rings_per_block=2)


# ----------------------------------------------------------------------------------------------- composition
@pytest.mark.parametrize("psi", ["phi_plus_tau_pi", "phi_plus_tau_ti"])
@pytest.mark.parametrize("neumann_mode", ["physical", "conormal"])
def test_composition_is_bitwise_the_standalone_operators(geo, psi, neumann_mode):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed", psi=psi,
                                         neumann_mode=neumann_mode)
    state, params, wall = make_inputs(geo, fields)
    phi = phi_field(geo)
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=phi)
    b, c, d = direct_terms(geo, opts, params, state, wall, phi)
    assert exact_equal(out.bracket, b) and exact_equal(out.curvature, c) and exact_equal(out.diffusion, d)
    assert exact_equal(out.total, b + c + d) and exact_equal(out.phi, phi)
    # every term is active and nontrivial
    for term in (b, c, d):
        assert float(jnp.abs(term).max()) > 1e-3


def test_curvature_options_pass_through(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=("curvature",), phi_mode="prescribed",
                                         curvature_jump_dissipation=True, curvature_c_kappa=0.5, bracket_c_kappa=0.3,
                                         absolute_method="lapack4")
    state, params, wall = make_inputs(geo, fields, normal=False)
    phi = phi_field(geo)
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=phi)
    idx = list(opts.curvature_index)
    q = jnp.concatenate([state, phi[..., None]], -1)
    ref = sbp_curvature(geo.plan, q, SatBoundaryData((wall.value,)), tau=params.tau, psi=opts.psi, absolute_method="lapack4",
                        jump_dissipation=True, c_kappa=0.5)
    assert idx == [0, 1, 2, 3] and exact_equal(out.curvature, ref)
    default = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, dataclasses.replace(opts, curvature_jump_dissipation=False,
                                                                                curvature_c_kappa=0.0, absolute_method="closed_form"),
                                          params, state, wall, phi=phi)
    assert float(jnp.abs(default.curvature - ref).max()) > 1e-6        # the options do something


@pytest.mark.parametrize("terms", [("bracket",), ("curvature",), ("diffusion",), ("bracket", "diffusion"),
                                   ("curvature", "diffusion")])
def test_term_subsets(geo, terms):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=terms, diffusion_kinds=KINDS4, phi_mode="prescribed")
    state, params, wall = make_inputs(geo, fields)
    phi = phi_field(geo)
    source = jnp.asarray(0.01 * np.cos(np.asarray(state)))
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=phi, source=source)
    b, c, d = direct_terms(geo, opts, params, state, wall, phi)
    full = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed")
    for name, got, ref in (("bracket", out.bracket, b), ("curvature", out.curvature, c), ("diffusion", out.diffusion, d)):
        if name in terms:
            assert exact_equal(got, ref)
        else:
            assert exact_equal(got, jnp.zeros_like(state))
    assert exact_equal(out.total, out.bracket + out.curvature + out.diffusion + source)
    assert full.terms == ("bracket", "curvature", "diffusion")


@pytest.mark.parametrize("fields,terms", [(("density",), ("bracket", "diffusion")),
                                          (("density", "Ti"), ("bracket", "diffusion")),
                                          (("Te", "Vi", "Ve"), ("bracket",)),
                                          (ALL, ("bracket", "curvature", "diffusion"))])
def test_field_subsets(geo, fields, terms):
    kinds = tuple(KINDS6[ALL.index(f)] for f in fields)
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=terms, diffusion_kinds=kinds, phi_mode="prescribed")
    state, params, wall = make_inputs(geo, fields)
    phi = phi_field(geo)
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=phi)
    assert out.total.shape == state.shape
    b, c, d = direct_terms(geo, opts, params, state, wall, phi)
    assert exact_equal(out.bracket, b) and exact_equal(out.curvature, c) and exact_equal(out.diffusion, d)
    # a field subset is the restriction of the full-field result (the operators act field by field)
    if "curvature" in terms:
        for k in (3, 4):                                              # Vi, Ve slots are zero in the curvature term
            assert exact_equal(out.curvature[..., k], jnp.zeros_like(out.curvature[..., k]))
    full_fields = ALL
    full_opts = dataclasses.replace(opts, fields=full_fields, diffusion_kinds=KINDS6, terms=("bracket", "diffusion"))
    fstate, fparams, fwall = make_inputs(geo, full_fields)
    f_out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, full_opts, fparams, fstate, fwall, phi=phi)
    sel = [ALL.index(f) for f in fields]
    sub = rhs.NodalPerpendicularOptions(fields=fields, terms=("bracket", "diffusion"), diffusion_kinds=kinds, phi_mode="prescribed")
    sub_params = fparams._replace(D=fparams.D[np.asarray(sel)])
    sub_out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, sub, sub_params, fstate[..., sel], rhs.NodalWallData(
        fwall.value[..., sel], fwall.normal[..., sel], None), phi=phi)
    np.testing.assert_allclose(np.asarray(sub_out.total), np.asarray(f_out.total[..., sel]), rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------------------------- psi solve
def two_term_phi(geo, opts, params, omega, n, Ti, phi_w, n_w, Ti_w, sigma, rtol):
    """``L phi = omega - sigma - tau L p`` with Dirichlet data ``phi_w`` and the pressure data ``p_w``."""
    p, p_w = rhs.pressure_variable(opts, n, Ti), rhs.pressure_variable(opts, n_w, Ti_w)
    Lp = laplacian_action(geo.lplan, p, LaplacianBoundaryData(value=(p_w,)), "dirichlet", None, opts.laplacian_c_kappa)
    s = omega - params.tau * Lp - (0.0 if sigma is None else sigma)
    phi, info = solve_dirichlet(geo.lplan, s, LaplacianBoundaryData(value=(phi_w,)), geo.ctx.prec,
                                c_kappa=opts.laplacian_c_kappa, rtol=rtol, maxit=400)
    return phi, info


@pytest.mark.parametrize("psi", ["phi_plus_tau_pi", "phi_plus_tau_ti"])
def test_psi_solve_equals_two_term_form(geo, psi):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=("bracket",), diffusion_kinds=KINDS4, psi=psi, phi_rtol=1e-12,
                                         phi_maxit=400)
    state, params, wall = make_inputs(geo, fields)
    n, Ti, omega = state[..., 0], state[..., 2], state[..., 3]
    phi_w = phi_wall(geo)
    psi_w = rhs.psi_wall_data(opts, params.tau, phi_w, wall.value[..., 0], wall.value[..., 2])
    sigma = jnp.asarray(0.2 * np.sin(2.0 * np.asarray(geo.pts[..., 2]) + np.asarray(geo.pts[..., 1])))
    psi_f, phi, info = rhs.solve_potential(geo.ctx, opts, params, omega, n, Ti, psi_w, sigma=sigma)
    assert bool(info["converged"]) and int(info["iterations"]) > 0
    ref, ref_info = two_term_phi(geo, opts, params, omega, n, Ti, phi_w, wall.value[..., 0], wall.value[..., 2], sigma, 1e-12)
    assert bool(ref_info["converged"])
    scale = float(jnp.abs(ref).max())
    err = float(jnp.abs(phi - ref).max()) / scale
    assert err < 1e-9, err
    # the solved psi is phi + tau p, and the boundary datum is honoured by the wall trace
    assert float(jnp.abs(psi_f - phi - params.tau * rhs.pressure_variable(opts, n, Ti)).max()) < 1e-13 * max(1.0, scale)
    # the full RHS reports the same potential
    out = rhs.nodal_perpendicular_rhs(geo.ctx, opts, params, state, rhs.NodalWallData(wall.value, wall.normal, psi_w), sigma=sigma)
    assert exact_equal(out.phi, phi) and exact_equal(out.psi, psi_f)
    assert bool(out.solve_info["converged"])


def test_sigma_none_is_sigma_zero_and_warm_start(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=("bracket",))
    state, params, wall = make_inputs(geo, fields)
    psi_w = rhs.psi_wall_data(opts, params.tau, phi_wall(geo), wall.value[..., 0], wall.value[..., 2])
    args = (geo.ctx, opts, params, state[..., 3], state[..., 0], state[..., 2], psi_w)
    psi0, phi0, _ = rhs.solve_potential(*args)
    psiz, phiz, _ = rhs.solve_potential(*args, sigma=jnp.zeros_like(state[..., 0]))
    assert exact_equal(psi0, psiz) and exact_equal(phi0, phiz)
    # a converged warm start needs no (or almost no) further iterations
    psiw, _phiw, info = rhs.solve_potential(*args, x0=psi0)
    assert int(info["iterations"]) <= 1 and float(jnp.abs(psiw - psi0).max()) < 1e-9


def test_solve_mode_errors(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, terms=("bracket",))
    state, params, wall = make_inputs(geo, fields)
    psi_w = jnp.zeros((geo.E, geo.N))
    with pytest.raises(ValueError, match="wall.psi"):
        rhs.nodal_perpendicular_rhs(geo.ctx, opts, params, state, wall)
    with pytest.raises(ValueError, match="preconditioner"):
        rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall._replace(psi=psi_w))
    with pytest.raises(ValueError, match="phi_mode"):
        rhs.nodal_perpendicular_rhs(geo.ctx, opts, params, state, wall._replace(psi=psi_w), phi=jnp.zeros((geo.E, geo.P)))
    with pytest.raises(ValueError, match="density, Ti and vorticity"):
        o2 = rhs.NodalPerpendicularOptions(fields=("density", "Te"), terms=("bracket",))
        rhs.nodal_perpendicular_rhs(geo.ctx, o2, params._replace(D=params.D[:2]), state[..., :2], rhs.NodalWallData(
            wall.value[..., :2], None, psi_w))


# ----------------------------------------------------------------------------------------------- prescribed phi
def test_prescribed_phi_path(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed")
    state, params, wall = make_inputs(geo, fields)
    phi = phi_field(geo)
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=phi)      # no preconditioner needed
    assert exact_equal(out.phi, phi)
    np.testing.assert_allclose(np.asarray(out.psi), np.asarray(phi + params.tau * state[..., 0] * state[..., 2]), rtol=0, atol=1e-15)
    assert int(out.solve_info["iterations"]) == 0 and bool(out.solve_info["converged"])
    legacy = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, dataclasses.replace(opts, psi="phi_plus_tau_ti"), params, state, wall, phi=phi)
    np.testing.assert_allclose(np.asarray(legacy.psi), np.asarray(phi + params.tau * state[..., 2]), rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="phi is required"):
        rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall)


# ----------------------------------------------------------------------------------------------- constants
def test_constant_state_with_matching_wall_data_gives_zero_terms(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed")
    c = jnp.asarray([1.3, 0.8, 1.1, 0.25])
    state = jnp.broadcast_to(c, (geo.E, geo.P, 4))
    value = jnp.broadcast_to(c, (geo.E, geo.N, 4))
    wall = rhs.NodalWallData(value, jnp.zeros_like(value), None)             # Neumann datum n.grad f = 0
    params = rhs.NodalPerpendicularParams(jnp.asarray(0.7), jnp.asarray(0.6), jnp.asarray([0.01, 0.02, 0.03, 0.04]))
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall, phi=jnp.full((geo.E, geo.P), 0.37))
    tol = 5e-13
    for name in ("bracket", "curvature", "diffusion", "total"):
        err = float(jnp.abs(getattr(out, name)).max())
        assert err < tol, (name, err)
    # solve mode with zero vorticity and constant psi data: a constant potential and zero terms
    state0 = state.at[..., 3].set(0.0)
    wall0 = rhs.NodalWallData(value.at[..., 3].set(0.0), jnp.zeros_like(value), jnp.full((geo.E, geo.N), 0.9))
    sopts = dataclasses.replace(opts, phi_mode="solve", phi_rtol=1e-13)
    sout = rhs.nodal_perpendicular_rhs(geo.ctx, sopts, params, state0, wall0)
    assert bool(sout.solve_info["converged"])
    assert float(jnp.abs(sout.psi - 0.9).max()) < 1e-10, float(jnp.abs(sout.psi - 0.9).max())
    assert float(jnp.abs(sout.total).max()) < 1e-10, float(jnp.abs(sout.total).max())


# ----------------------------------------------------------------------------------------------- bracket wall rule
def test_bracket_rule_inflow(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed")
    state, _params, wall = make_inputs(geo, fields)
    value = wall.value + 0.5                                          # data unlike the field's own trace
    out = np.asarray(rhs.bracket_rule_inflow(geo.ctx, opts, state, value))
    assert out.shape == (geo.E, geo.N, 4)
    # the bracket's own three-point SAT trace, so the inflow mismatch a - data vanishes for Neumann fields
    st = geo.plan.structure
    plan_trace = np.asarray(trace(geo.plan, st.walls[0][0], st.walls[0][1], state))
    for k, kind in enumerate(KINDS4):
        ref = plan_trace[..., k] if kind == "neumann" else np.asarray(value)[..., k]
        assert np.array_equal(out[..., k], ref)
    # all-Dirichlet kinds: the value unchanged
    dopts = dataclasses.replace(opts, diffusion_kinds=("dirichlet",) * 4)
    assert exact_equal(rhs.bracket_rule_inflow(geo.ctx, dopts, state, value), value)
    with pytest.raises(ValueError, match="diffusion_kinds"):
        rhs.bracket_rule_inflow(geo.ctx, dataclasses.replace(opts, diffusion_kinds=()), state, value)


# ----------------------------------------------------------------------------------------------- jit
def test_jit_with_context_argument_does_not_retrace(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_rtol=1e-8)
    state, params, wall = make_inputs(geo, fields)
    psi_w = rhs.psi_wall_data(opts, params.tau, phi_wall(geo), wall.value[..., 0], wall.value[..., 2])
    wall = wall._replace(psi=psi_w)
    sigma = jnp.zeros((geo.E, geo.P))
    source = jnp.zeros_like(state)
    calls = []

    def f(ctx, params, state, wall, sigma, source, psi_x0):
        calls.append(1)
        return rhs.nodal_perpendicular_rhs(ctx, opts, params, state, wall, sigma=sigma, source=source, psi_x0=psi_x0)

    jf = jax.jit(f)
    out = jf(geo.ctx, params, state, wall, sigma, source, jnp.zeros((geo.E, geo.P)))
    assert len(calls) == 1
    ref = rhs.nodal_perpendicular_rhs(geo.ctx, opts, params, state, wall, sigma=sigma, source=source)
    assert float(jnp.abs(out.total - ref.total).max()) < 1e-9 * max(1.0, float(jnp.abs(ref.total).max()))
    # changed traced values (state, params, wall, sigma, source, warm start) reuse the compiled function
    state2 = state * 1.05
    params2 = params._replace(rho_star=jnp.asarray(0.9), tau=jnp.asarray(0.4), D=params.D * 2)
    wall2 = wall._replace(value=wall.value * 0.97, normal=wall.normal * 1.1)
    out2 = jf(geo.ctx, params2, state2, wall2, sigma + 0.1, source + 0.2, out.psi)
    out3 = jf(geo.ctx, params, state, wall, sigma, source, out.psi)
    assert len(calls) == 1
    assert float(jnp.abs(out2.total - out.total).max()) > 1e-6
    assert float(jnp.abs(out3.total - out.total).max()) < 1e-8 * max(1.0, float(jnp.abs(out.total).max()))
    # the library entry point takes the options as a static argument
    jit_out = rhs.nodal_perpendicular_rhs_jit(geo.ctx, opts, params, state, wall)
    assert bool(jit_out.solve_info["converged"])


# ----------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("kwargs,match", [
    (dict(fields=("Te", "density")), "order"),
    (dict(fields=("density", "density")), "order"),
    (dict(fields=("density", "pressure")), "unknown fields"),
    (dict(fields=()), "empty"),
    (dict(terms=("bracket", "viscosity")), "terms"),
    (dict(terms=("bracket", "bracket")), "terms"),
    (dict(fields=("density", "Te"), terms=("curvature",)), "curvature term needs"),
    (dict(terms=("diffusion",), diffusion_kinds=("dirichlet",)), "one entry per field"),
    (dict(terms=("bracket", "diffusion")), "one entry per field"),
    (dict(diffusion_kinds=("dirichlet", "robin", "dirichlet", "dirichlet")), "dirichlet"),
    (dict(neumann_mode="normal"), "neumann_mode"),
    (dict(phi_mode="given"), "phi_mode"),
    (dict(psi="phi"), "psi"),
    (dict(absolute_method="svd"), "absolute_method"),
    (dict(phi_maxit=0), "phi_maxit"),
    (dict(fields="density"), "sequence"),
])
def test_option_validation(kwargs, match):
    base = dict(diffusion_kinds=KINDS4) if "terms" not in kwargs else {}
    with pytest.raises(ValueError, match=match):
        rhs.NodalPerpendicularOptions(**{**base, **kwargs})


def test_options_are_hashable_static_keys():
    a = rhs.NodalPerpendicularOptions(diffusion_kinds=list(KINDS4))
    b = rhs.NodalPerpendicularOptions(diffusion_kinds=KINDS4)
    assert a == b and hash(a) == hash(b) and isinstance(a.fields, tuple)


def test_input_shape_and_missing_wall_data_errors(geo):
    fields = ("density", "Te", "Ti", "vorticity")
    opts = rhs.NodalPerpendicularOptions(fields=fields, diffusion_kinds=KINDS4, phi_mode="prescribed")
    state, params, wall = make_inputs(geo, fields)
    phi = phi_field(geo)
    with pytest.raises(ValueError, match="state must have shape"):
        rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state[..., :3], wall, phi=phi)
    with pytest.raises(ValueError, match="wall.value"):
        rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall._replace(value=wall.value[:, :-1]), phi=phi)
    with pytest.raises(ValueError, match="wall.normal is required"):
        rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, opts, params, state, wall._replace(normal=None), phi=phi)
    # all-Dirichlet diffusion needs no normal datum
    dopts = dataclasses.replace(opts, diffusion_kinds=("dirichlet",) * 4)
    out = rhs.nodal_perpendicular_rhs(geo.ctx_nosolve, dopts, params, state, wall._replace(normal=None), phi=phi)
    assert out.diffusion.shape == state.shape
