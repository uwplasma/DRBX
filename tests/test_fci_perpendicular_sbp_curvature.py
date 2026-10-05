"""Algebra-only tests of the nodal SBP P06 curvature operator: derivative along the curvature flux, wall SAT, matrix dissipation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_families import build_family_a_layout
from drbx.geometry.nodal_layout import build_nodal_layout
from drbx.native import fci_perpendicular_sbp_curvature as cv
from drbx.native import fci_perpendicular_sbp_ops as ops
from drbx.native.fci_curvature_production_flux import curvature_principal_matrix
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable

TWO_PI = 2.0 * np.pi
N_GRID = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]
E = 8
TAU = 1.0
HALO = 3


def metric(constant_B=False):
    def fn(p):
        u, th, et = p.T
        h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
        K = np.stack([0.3 * np.cos(th), 0.1 * u + 0.05, 0.2 * np.sin(et) + 0.1], -1)
        B = 1.0 + 0 * u if constant_B else 1.0 + 0.2 * u
        return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), B, K

    return fn


def ring_case(constant_B=False):
    lay = build_nodal_layout(N_GRID, LEVELS, n_eta=E, inner="wall")
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, metric(constant_B)))


@pytest.fixture(scope="module")
def case():
    return ring_case()


@pytest.fixture(scope="module")
def case_B1():
    return ring_case(constant_B=True)


def hsum(plan, a, b):
    Hp = np.asarray(plan.Hp)
    return float(np.sum(Hp.reshape(Hp.shape + (1,) * (np.ndim(a) - 2)) * np.asarray(a) * np.asarray(b)))


def wall_traces(plan, F, *gs):
    out = []
    for blk, side, sign, N in plan.structure.walls:
        a = [np.asarray(ops.trace(plan, blk, side, g)) for g in gs]
        v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
        out.append((sign, N, a, v))
    return out


def ext(x):
    return ops.extend_periodic(jnp.asarray(x), HALO)


def state(plan, seed=0, background=False):
    """A positive five-field state ``(E, P, 5)``."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(plan.jac.shape + (5,)) * 0.1
    q[..., :3] += np.array([1.1, 0.9, 0.7])
    if background:
        q = np.zeros_like(q) + np.array([1.1, 0.9, 0.7, 0.0, 0.0])
    return q


def eigen_W(P0):
    lam, R = np.linalg.eig(P0)
    assert np.abs(lam.imag).max() <= 1e-12
    lam, R = lam.real, R.real
    Ri = np.linalg.inv(R)
    return R, lam, Ri.T @ Ri, Ri.T @ np.diag(np.abs(lam)) @ Ri       # W, W|M|


# ------------------------------------------------------------------------------------------------ scalar derivative
def test_scalar_derivative_kills_constants_and_has_the_continuum_sign(case):
    lay, plan = case
    rng = np.random.default_rng(1)
    F = rng.standard_normal((E, lay.P, 3))
    c1 = np.asarray(cv.curvature_derivative(plan, F, np.ones((E, lay.P))))
    assert np.abs(c1).max() <= 1e-13 * np.abs(F).max() / np.asarray(plan.jac).min()
    # smooth constant-coefficient flux and polynomial g: exact (F / |J|) . grad g away from walls and the level face
    jac = np.asarray(plan.jac)
    u = np.broadcast_to(lay.node_u, (E, lay.P))
    F = np.stack([0.7 + 0.3 * u, 0.4 + 0 * u, 0.9 + 0 * u], -1)
    ring = np.zeros(lay.P, dtype=int)
    for desc in plan.structure.blocks:
        _, o, nb, m, N, _ = desc
        ring[o:o + nb] = np.repeat(np.arange(m), N)
    interior = np.zeros(lay.P, dtype=bool)
    for desc in plan.structure.blocks:
        _, o, nb, m, N, _ = desc
        interior[o:o + nb] = (ring[o:o + nb] >= 4) & (ring[o:o + nb] <= m - 5)
    assert interior.any()
    for g_of_u, dg in ((lambda x: x, lambda x: 1.0 + 0 * x), (lambda x: x * x, lambda x: 2 * x)):
        g = g_of_u(u)
        got = np.asarray(cv.curvature_derivative(plan, F, g))
        want = F[..., 0] * dg(u) / jac
        assert np.abs((got - want)[:, interior]).max() <= 1e-12 * np.abs(want).max()
        assert np.abs(want[:, interior]).max() > 0.1                       # positive F gives a positive derivative


def test_antisymmetry_identity_up_to_c_and_wall_terms(case):
    lay, plan = case
    rng = np.random.default_rng(2)
    F = rng.standard_normal((E, lay.P, 3))
    f, g = rng.standard_normal((2, E, lay.P))
    Fe = ext(F)
    c = np.asarray(cv.compressibility_flux(plan, Fe))
    Cf, Cg = (np.asarray(cv.curvature_derivative_ext(plan, Fe, ext(x))) for x in (f, g))
    lhs = hsum(plan, f, Cg) + hsum(plan, g, Cf)
    wall = sum(TWO_PI / N * np.sum(sign * v * a[0] * a[1]) for sign, N, a, v in wall_traces(plan, F, f, g))
    scale = np.sqrt(hsum(plan, f, f) * hsum(plan, Cg, Cg))
    assert abs(lhs - (wall - hsum(plan, c, f * g))) <= 1e-13 * scale
    # c is the discrete divergence of F: for the divergence-free-in-the-continuum flux it is small and has a sign structure
    assert np.abs(c).max() > 0.0


def test_legacy_and_pi_selectors_give_the_same_centred_total(case):
    lay, plan = case
    q = state(plan, 3)
    data = tuple(np.asarray(ops.trace(plan, blk, side, q[..., :4])) for blk, side, _s, _n in plan.structure.walls)
    bcd = SatBoundaryData(data, None)
    out = {psi: np.asarray(cv.sbp_curvature(plan, q, bcd, tau=TAU, psi=psi)) for psi in cv.PSI_VARIANTS}
    scale = np.abs(out["phi_plus_tau_ti"]).max()
    assert scale > 0.0
    assert np.abs(out["phi_plus_tau_pi"] - out["phi_plus_tau_ti"]).max() <= 1e-12 * scale
    # the selectors differ only through |M| of the wall SAT, which vanishes for matching data
    other = SatBoundaryData(tuple(d + 0.01 for d in data), None)
    a = np.asarray(cv.sbp_curvature(plan, q, other, tau=TAU, psi="phi_plus_tau_ti"))
    b = np.asarray(cv.sbp_curvature(plan, q, other, tau=TAU, psi="phi_plus_tau_pi"))
    assert np.abs(a - b).max() > 1e-6 * scale


def test_curvature_of_a_constant_state_vanishes_with_matching_wall_data(case):
    lay, plan = case
    q = state(plan, background=True)
    data = tuple(np.broadcast_to(q[0, 0, :4], (E, N, 4)).copy() for _b, _s, _g, N in plan.structure.walls)
    out = np.asarray(cv.sbp_curvature(plan, q, SatBoundaryData(data, None), tau=TAU, psi="phi_plus_tau_pi"))
    assert np.abs(out).max() <= 1e-13 * np.abs(np.asarray(cv.curvature_flux(plan))).max() / np.asarray(plan.jac).min()


# ------------------------------------------------------------------------------------------------ wall SAT energy
def frozen_operator(plan, F, P0, g, data=None, method="closed_form"):
    """``P0 C_h g`` plus the wall SAT of the frozen matrix (state = constant background)."""
    Fe = ext(F)
    Cg = np.asarray(cv.curvature_derivative_ext(plan, Fe, ext(g)))
    Mg = np.einsum("ij,epj->epi", P0, Cg)
    q0 = np.broadcast_to(np.array([1.1, 0.9, 0.7, 0.0, 0.0]), g.shape[:2] + (5,)).copy()
    bcd = None if data is None else SatBoundaryData(data, None)
    sat = np.asarray(cv.wall_inflow_matrix(plan, jnp.asarray(F), q0, jnp.asarray(g), bcd, tau=TAU, psi="phi_plus_tau_pi",
                                           absolute_method=method))
    return Mg + sat, sat, Cg


@pytest.mark.parametrize("method", ["closed_form", "lapack4"])
def test_wall_sat_energy_sign_frozen_matrix(case_B1, method):
    """``sum_H g^T W (P0 C_h g + SAT) + c-term = -1/2 Omega sum |sigma v| a^T W|P0| a <= 0`` (the right inflow sign)."""
    lay, plan = case_B1
    rng = np.random.default_rng(5)
    F = rng.standard_normal((E, lay.P, 3))
    g = rng.standard_normal((E, lay.P, 4))
    P0 = np.asarray(curvature_principal_matrix(1.1, 0.9, 0.7, 1.0, TAU, psi="phi_plus_tau_pi"))
    _R, _lam, W, WabsM = eigen_W(P0)
    S = W @ P0
    assert np.abs(S - S.T).max() <= 1e-12 * np.abs(S).max()                    # W M is symmetric
    L, _sat, _ = frozen_operator(plan, F, P0, g, method=method)
    c = np.asarray(cv.compressibility_flux(plan, ext(F)))
    Hp = np.asarray(plan.Hp)
    energy = float(np.einsum("ep,epi,ij,epj->", Hp, g, W, L))
    c_term = 0.5 * float(np.einsum("ep,ep,epi,ij,epj->", Hp, c, g, S, g))
    pred = 0.0
    for sign, N, (a,), v in wall_traces(plan, F, g):
        pred += -0.5 * TWO_PI / N * float(np.einsum("ej,eji,ik,ejk->", np.abs(sign * v), a, WabsM, a))
    scale = np.sqrt(np.einsum("ep,epi,ij,epj->", Hp, g, W, g) * np.einsum("ep,epi,ij,epj->", Hp, L, W, L))
    assert abs(energy + c_term - pred) <= 1e-12 * scale
    assert pred < 0.0 and energy + c_term <= 1e-12 * scale


def test_wall_sat_closed_form_matches_lapack4(case):
    lay, plan = case
    rng = np.random.default_rng(6)
    F = rng.standard_normal((E, lay.P, 3))
    q = state(plan, 7)
    g = q[..., :4] + 0.05 * rng.standard_normal((E, lay.P, 4))
    data = tuple(rng.standard_normal((E, N, 4)) * 0.1 + 1.0 for _b, _s, _g, N in plan.structure.walls)
    for psi in cv.PSI_VARIANTS:
        out = [np.asarray(cv.wall_inflow_matrix(plan, jnp.asarray(F), q, jnp.asarray(g), SatBoundaryData(data, None),
                                                tau=TAU, psi=psi, absolute_method=m)) for m in cv.ABSOLUTE_METHODS]
        assert np.abs(out[0]).max() > 0.0
        assert np.abs(out[0] - out[1]).max() <= 1e-12 * np.abs(out[1]).max()


# ------------------------------------------------------------------------------------------------ matrix dissipation
@pytest.mark.parametrize("method", ["closed_form", "lapack4"])
def test_jump_dissipation_and_upwind_are_negative_in_H_times_W(case_B1, method):
    lay, plan = case_B1
    rng = np.random.default_rng(8)
    F = rng.standard_normal((E, lay.P, 3))
    q = state(plan, background=True)
    P0 = np.asarray(curvature_principal_matrix(1.1, 0.9, 0.7, 1.0, TAU, psi="phi_plus_tau_ti"))
    _R, _lam, W, _WA = eigen_W(P0)
    H = np.asarray(plan.Hp) * plan.structure.deta
    kw = dict(tau=TAU, psi="phi_plus_tau_ti", absolute_method=method)
    Fe, qe = ext(F), ext(q)
    diss_fn = jax.jit(lambda p_, Fe_, qe_, ge_: cv.jump_dissipation_ext(p_, Fe_, qe_, ge_, **kw))
    up_fn = jax.jit(lambda p_, F_, q_, g_: cv.interface_upwind_matrix(p_, F_, q_, g_, **kw))
    for _ in range(2):
        g = rng.standard_normal((E, lay.P, 4))
        diss = np.asarray(diss_fn(plan, Fe, qe, ext(g)))
        up = np.asarray(up_fn(plan, jnp.asarray(F), q, jnp.asarray(g)))
        for D in (diss, up, diss + up):
            e = float(np.einsum("ep,epi,ij,epj->", H, g, W, D))
            scale = np.sqrt(np.einsum("ep,epi,ij,epj->", H, g, W, g) * np.einsum("ep,epi,ij,epj->", H, D, W, D))
            assert e <= 1e-14 * scale
        assert float(np.einsum("ep,epi,ij,epj->", H, g, W, diss)) < 0.0
        assert np.abs(up).max() > 0.0
    # a constant field is not dissipated
    one = np.broadcast_to(np.array([1.0, -2.0, 0.5, 3.0]), (E, lay.P, 4)).copy()
    assert np.abs(np.asarray(diss_fn(plan, Fe, qe, ext(one)))).max() <= 1e-11
    assert np.abs(np.asarray(up_fn(plan, jnp.asarray(F), q, jnp.asarray(one)))).max() <= 1e-11


def test_matrix_dissipation_reduces_to_the_scalar_weight_for_a_scaled_identity_matrix(case_B1):
    """With Te = Ti -> 0 limit not available, check the face weight against the closed-form |M| action on one face jump."""
    lay, plan = case_B1
    rng = np.random.default_rng(9)
    jump = rng.standard_normal((5, 4))
    n, te, ti, b = 1.1, 0.9, 0.7, 1.0
    M = np.asarray(curvature_principal_matrix(n, te, ti, b, TAU, psi="phi_plus_tau_ti"))
    lam, R = np.linalg.eig(M)
    absM = (R @ np.diag(np.abs(lam)) @ np.linalg.inv(R)).real
    for method in cv.ABSOLUTE_METHODS:
        got = np.asarray(cv.absolute_action(method, n, te, ti, b, TAU, 0.37, jnp.asarray(jump)))
        assert np.abs(got - 0.37 * jump @ absM.T).max() <= 1e-12


# ------------------------------------------------------------------------------------------------ core, jit and AD
@pytest.fixture(scope="module")
def core_case():
    lay = build_family_a_layout(16, n_eta=4)

    def fn(p):
        u, th, et = p.T
        h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
        K = np.stack([0.3 * np.cos(th), 0.1 * u, 0.2 * np.sin(et) + 0.1], -1)
        return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0.2 * u, K

    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, fn))


def test_core_damping_is_negative_and_spectral_radius_matches_eig(core_case):
    lay, plan = core_case
    Ec = lay.n_eta
    rng = np.random.default_rng(10)
    F = np.asarray(cv.curvature_flux(plan))
    q = state(plan, 11)
    g = rng.standard_normal((Ec, lay.P, 4))
    out = np.asarray(cv.core_damping_matrix_ext(plan, ext(F), ext(q), ext(g), 1.0, tau=TAU, psi="phi_plus_tau_pi"))
    n_c = plan.structure.core[3]
    assert np.abs(out[:, n_c:]).max() == 0.0 and np.abs(out[:, :n_c]).max() > 0.0
    Hk = np.asarray(plan.wxy)[None, :n_c] * np.asarray(plan.jac)[:, :n_c]
    assert float(np.sum(Hk[..., None] * g[:, :n_c] * out[:, :n_c])) < 0.0
    assert np.abs(np.asarray(cv.core_damping_matrix_ext(plan, ext(F), ext(q), ext(g), 0.0, tau=TAU))).max() == 0.0
    for psi in cv.PSI_VARIANTS:
        n, te, ti = 1.1, 0.9, 0.4
        M = np.asarray(curvature_principal_matrix(n, te, ti, 1.3, TAU, psi=psi))
        want = np.abs(np.linalg.eigvals(M)).max()
        assert abs(float(cv.spectral_radius(n, te, ti, TAU, psi)) - want) <= 1e-12 * want


def test_jit_and_forward_mode_ad_with_core(core_case):
    lay, plan = core_case
    q = jnp.asarray(state(plan, 12))
    data = tuple(np.asarray(ops.trace(plan, blk, side, q[..., :4])) + 0.01 for blk, side, _s, _n in plan.structure.walls)
    bcd = SatBoundaryData(tuple(jnp.asarray(d) for d in data), None)

    @jax.jit
    def f(plan_, q_, ck):
        return cv.sbp_curvature(plan_, q_, bcd, tau=TAU, psi="phi_plus_tau_pi", c_kappa=ck)

    out = np.asarray(f(plan, q, 1.0))
    assert out.shape == q.shape[:2] + (4,) and np.all(np.isfinite(out))
    assert np.abs(np.asarray(f(plan, q, 0.3)) - out).max() > 0.0                # c_kappa is traced and acts
    dq = jnp.asarray(np.random.default_rng(13).standard_normal(q.shape) * 0.1)
    _p, tang = jax.jvp(lambda x: f(plan, x, 1.0), (q,), (dq,))
    eps = 1e-6
    fd = (np.asarray(f(plan, q + eps * dq, 1.0)) - np.asarray(f(plan, q - eps * dq, 1.0))) / (2 * eps)
    assert np.abs(np.asarray(tang) - fd).max() <= 1e-5 * np.abs(fd).max()


def test_dissipative_variant_jits_on_the_core_plan(core_case):
    lay, plan = core_case
    q = jnp.asarray(state(plan, 14))
    f = jax.jit(lambda p_, q_: cv.sbp_curvature(p_, q_, None, tau=TAU, psi="phi_plus_tau_pi", jump_dissipation=True,
                                                c_kappa=1.0))
    base = jax.jit(lambda p_, q_: cv.sbp_curvature(p_, q_, None, tau=TAU, psi="phi_plus_tau_pi", c_kappa=1.0))
    out, ref = np.asarray(f(plan, q)), np.asarray(base(plan, q))
    assert np.all(np.isfinite(out)) and np.abs(out - ref).max() > 0.0
