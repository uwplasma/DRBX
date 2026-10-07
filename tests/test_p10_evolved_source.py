"""P10 evolved MMS, chunk C2 (source): continuum kernels, the discrete source pairing with ``nodal_perpendicular_rhs`` at the RK stage
times, the polarization source and the wall data, on the n = 16 family-A synthetic bundle (``p10_evolved_mms.synthetic``)."""
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
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native import fci_nodal_perpendicular_rhs as rhs        # noqa: E402
from p10_evolved_mms import fields as F                           # noqa: E402
from p10_evolved_mms import source as S                           # noqa: E402
from p10_evolved_mms import synthetic as syn                      # noqa: E402
from tests import sbp_laplacian_testbed as tb                      # noqa: E402

# the provisional configuration (rho* = 4.5e-4, time_scale = 1, T = 1/30) and a fast-time variant that exercises the advection strongly
P_CONFIG = F.default_params()
# a_omega * rho_star^2 = 1: the O(1) vorticity of the fast sets (the config set has the consistent rho_star^2 amplitude)
P_FAST = F.default_params(rho_star=0.05, time_scale=1.0, a_phi=0.5, a_omega=1.0 / 0.05 ** 2)
CASES = {"config": (P_CONFIG, 0.01, 0.01), "fast": (P_FAST, 0.2, 0.02)}      # config: stage times inside T = 1/30


@pytest.fixture(scope="module")
def bundle():
    return syn.synthetic_bundle()


# ----------------------------------------------------------------------------------------------------- kernels
def test_point_bracket_is_the_frozen_reference_triple_product():
    """The frozen references write ``-(h x grad a) . grad b / (|J| rho*)``; ``source.point_bracket`` is the same without ``rho*``."""
    rng = np.random.default_rng(1)
    h, ga, gb = rng.normal(size=(3, 40, 3))
    jac = rng.uniform(0.1, 2.0, 40)
    frozen = lambda rho: -np.sum(np.cross(h, ga) * gb, axis=-1) / np.abs(jac) / rho       # references.point_bracket
    got = np.asarray(S.point_bracket(jnp.asarray(h), jnp.asarray(jac), jnp.asarray(ga), jnp.asarray(gb)[:, None, :]))[:, 0]
    np.testing.assert_allclose(got, frozen(1.0), rtol=1e-14, atol=1e-14)
    # convention conversion: the single-length term rho * pb equals rho * rho0 * (frozen reference at rho0)
    np.testing.assert_allclose(0.7 * got, 0.7 * 0.05 * frozen(0.05), rtol=1e-14, atol=1e-14)


def test_laplacian_kernel_matches_the_testbed_exact_laplacian(bundle):
    """``(divA . grad f + A : grad grad f) / |J|`` with the extracted ``A, divA, |J|`` equals the exact flux-form Laplacian
    ``J^-1 d_i(A^{ij} d_j f)`` of the testbed (an independent autodiff of the flux)."""
    ref = bundle.ref
    pts = ref.points.reshape(-1, 3)
    fns = list(tb.FIELDS.values())

    @jax.jit
    def derivs(pts):
        return (jnp.stack([jax.vmap(jax.grad(f))(pts) for f in fns], axis=1),
                jnp.stack([jax.vmap(jax.hessian(f))(pts) for f in fns], axis=1))

    G, H = derivs(pts)
    E, P = ref.jac.shape
    got = S.laplacian_cont(ref.A, ref.divA, ref.jac, G.reshape(E, P, -1, 3), H.reshape(E, P, -1, 3, 3))
    exact = jnp.asarray(tb.laplacians(np.asarray(pts))).reshape(E, P, -1)
    assert float(jnp.abs(got - exact).max()) < 1e-11 * float(jnp.abs(exact).max())


def test_psi_derivatives_equal_autodiff_of_psi(bundle):
    pts = bundle.ref.points.reshape(-1, 3)[::97]
    mp = P_FAST

    def run(pts):
        V, G, H = F.FIELDS.values_grad_hess(pts, 0.3, mp)
        out = {}
        for variant in ("phi_plus_tau_pi", "phi_plus_tau_ti"):
            psi = lambda x: (lambda v: v[4] + mp.tau * S.pressure(variant, v[0], v[2]))(F.FIELDS.f(x, 0.3, mp))
            out[variant] = (S.psi_derivatives(variant, mp.tau, V, G, H),
                            (jax.vmap(psi)(pts), jax.vmap(jax.grad(psi))(pts), jax.vmap(jax.hessian(psi))(pts)))
        return out

    results = jax.jit(run)(pts)
    for variant in ("phi_plus_tau_pi", "phi_plus_tau_ti"):
        (v, g, h), (v2, g2, h2) = results[variant]
        np.testing.assert_allclose(np.asarray(v), np.asarray(v2), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g2), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(h), np.asarray(h2), rtol=1e-11, atol=1e-11)


# ----------------------------------------------------------------------------------------------------- continuum
def test_continuum_structure_and_sigma_vanishes_for_synthetic_omega(bundle):
    mp = P_FAST._replace(rho_star=0.7, tau=0.6)
    c = S.continuum_jit(bundle, 0.3, mp)
    E, P = bundle.E, bundle.P
    assert c.q.shape == c.R.shape == c.S.shape == (E, P, 4) and c.sigma.shape == c.phi.shape == (E, P)
    np.testing.assert_allclose(np.asarray(c.R), np.asarray(c.bracket + c.curvature + c.diffusion), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(c.S), np.asarray(c.dq - c.R), rtol=1e-13, atol=1e-13)
    for term in (c.bracket, c.curvature, c.diffusion / jnp.asarray(mp.D), c.dq, c.sigma):
        assert float(jnp.abs(term).max()) > 1e-4                         # all pieces are active (diffusion per unit D)
    # psi = phi + tau n Ti at the nodes
    np.testing.assert_allclose(np.asarray(c.psi), np.asarray(c.phi + mp.tau * c.q[..., 0] * c.q[..., 2]), rtol=1e-14, atol=1e-14)
    # the polarization source is sigma = Omega - rho*^2 L_cont psi (the Omega of the fields), and vanishes for the synthetic
    # vorticity Omega := rho*^2 L_cont psi built from an independent autodiff of the flux J^-1 d_i (A^{ij} d_j psi)
    np.testing.assert_allclose(np.asarray(c.sigma), np.asarray(c.q[..., 3] - mp.rho_star ** 2 * c.lap_psi), rtol=1e-14, atol=1e-14)

    def psi_fn(x):
        v = F.FIELDS.f(x, 0.3, mp)
        return v[4] + mp.tau * v[0] * v[2]

    def lap_exact(x):
        A, J, _G = tb.tensor(x)
        flux = lambda y: tb.tensor(y)[0] @ jax.grad(psi_fn)(y)
        return jnp.trace(jax.jacfwd(flux)(x)) / J

    omega_syn = mp.rho_star ** 2 * jax.jit(jax.vmap(lap_exact))(bundle.ref.points.reshape(-1, 3)).reshape(E, P)
    sigma_syn = omega_syn - mp.rho_star ** 2 * c.lap_psi
    assert float(jnp.abs(sigma_syn).max()) < 1e-11 * float(jnp.abs(omega_syn).max())


def test_continuum_jit_matches_eager_and_does_not_retrace_across_stage_times(bundle):
    jitted = jax.jit(lambda t: S.continuum(bundle, t, P_FAST))
    a, b = jitted(0.1), jitted(0.25)
    assert jitted._cache_size() == 1                                      # t is traced: one compilation for every stage time
    eager = S.continuum(bundle, 0.25, P_FAST)
    np.testing.assert_allclose(np.asarray(b.S), np.asarray(eager.S), rtol=1e-11, atol=1e-12)
    assert float(jnp.abs(a.S - b.S).max()) > 1e-6                         # the source is time dependent


# ----------------------------------------------------------------------------------------------------- wall data
def test_wall_data_values_normal_derivative_and_psi(bundle):
    mp, t = P_FAST, 0.37
    w = S.wall_data_jit(bundle, t, mp, "NNN-D")
    E, N = bundle.E, bundle.N
    assert w.value.shape == w.normal.shape == (E, N, 4) and w.psi.shape == (E, N)
    pts = bundle.ref.wall_points.reshape(-1, 3)
    V = jax.vmap(lambda x: F.FIELDS.f(x, t, mp))(pts).reshape(E, N, 5)
    np.testing.assert_allclose(np.asarray(w.value), np.asarray(V[..., :4]), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.asarray(w.psi), np.asarray(V[..., 4] + mp.tau * V[..., 0] * V[..., 2]), rtol=1e-14, atol=1e-14)
    # physical normal derivative n . grad f = g^{u j} d_j f / sqrt(g^{uu}) with the analytic testbed inverse metric
    def normal(x):
        _A, _J, G = tb.tensor(x)
        grad = jax.jacfwd(lambda y: F.FIELDS.f(y, t, mp))(x)[:4]                # (4, 3)
        return grad @ G[0] / jnp.sqrt(G[0, 0])

    ref = jax.jit(jax.vmap(normal))(pts).reshape(E, N, 4)
    np.testing.assert_allclose(np.asarray(w.normal), np.asarray(ref), rtol=1e-11, atol=1e-11)
    assert S.wall_data(bundle, t, mp, "DDDD").normal is None
    np.testing.assert_array_equal(np.asarray(S.wall_data(bundle, t, mp, "DDDD").value), np.asarray(w.value))
    # time dependence of the data
    w2 = S.wall_data_jit(bundle, t + 0.05, mp, "NNN-D")
    assert float(jnp.abs(w2.value - w.value).max()) > 1e-4 and float(jnp.abs(w2.psi - w.psi).max()) > 1e-5


# ----------------------------------------------------------------------------------------------------- the pairing
@pytest.mark.parametrize("pattern", ["NNN-D", "DDDD"])
@pytest.mark.parametrize("case", ["config", "fast"])
def test_pairing_exact_state_with_discrete_source_reproduces_dt_q(bundle, pattern, case):
    """At the exact nodal state, ``R_h(q, wall(t)) + S_h == d_t q`` to round-off at the four RK stage times, prescribed phi; and the
    psi solve with ``sigma_h`` returns the exact ``psi`` to the solve tolerance."""
    mp, t0, dt = CASES[case]
    params = mp.nodal()
    opts = S.nodal_options(pattern)
    opts_solve = S.nodal_options(pattern, phi_mode="solve", phi_rtol=1e-12, phi_maxit=100)
    rhs_jit, solve_jit = rhs.nodal_perpendicular_rhs_jit, rhs.solve_potential_jit
    seen = []
    for t in (t0, t0 + dt / 2, t0 + dt / 2, t0 + dt):
        wall = S.wall_data_jit(bundle, t, mp, pattern)
        d = S.discrete_jit(bundle, None, opts, None, t, mp)
        out = rhs_jit(bundle.ctx, opts, params, d.q, wall, phi=d.phi, source=d.S)
        scale = float(jnp.abs(d.dq).max())
        assert float(jnp.abs(out.total - d.dq).max()) <= 1e-13 * scale, (float(jnp.abs(out.total - d.dq).max()), scale)
        # nontrivial: the RHS and the source are both active and the truncation is visible (S_h != d_t q)
        # (at physical rho_star the perpendicular terms are ~1e-4 of d_t q, so the RHS threshold is 1e-6 of it)
        assert float(jnp.abs(d.R).max()) > 1e-6 * scale and float(jnp.abs(d.S).max()) > 1e-3 * scale
        # the wall data of the discrete source are those of wall_data (same arrays in, same arrays out)
        d2 = S.discrete_jit(bundle, None, opts, None, t, mp, wall=wall)
        np.testing.assert_array_equal(np.asarray(d2.S), np.asarray(d.S))
        # polarization: L psi = (Omega - sigma_h) / rho*^2 with the Dirichlet psi_w returns the exact psi
        psi, phi, info = solve_jit(bundle.ctx, opts_solve, params, d.q[..., 3], d.q[..., 0], d.q[..., 2], wall.psi, sigma=d.sigma)
        assert bool(info["converged"])
        psi_err = float(jnp.abs(psi - d.psi).max()) / float(jnp.abs(d.psi).max())
        phi_err = float(jnp.abs(phi - d.phi).max()) / float(jnp.abs(d.phi).max())
        # the solve conditions the data Omega - sigma_h = rho*^2 L_h psi against Omega: round-off is amplified by Omega / (rho*^2 L psi)
        amp = max(1.0, float(jnp.abs(d.q[..., 3]).max()) / (float(params.rho_star) ** 2 * float(jnp.abs(d.lap_psi).max())))
        assert psi_err < 1e-9 * amp and phi_err < 1e-8 * amp, (psi_err, phi_err, amp)
        seen.append(np.asarray(wall.value))
    assert np.array_equal(seen[1], seen[2]) and float(np.abs(seen[0] - seen[3]).max()) > 0.0       # stage times differ


@pytest.mark.slow
def test_discrete_source_prescribed_equals_solve_mode_total(bundle):
    """The solve-mode RHS of the exact state with ``sigma_h`` is the prescribed-phi RHS up to the solve error."""
    mp, t, _ = CASES["fast"]
    params = mp.nodal()
    opts = S.nodal_options("NNN-D")
    opts_solve = S.nodal_options("NNN-D", phi_mode="solve", phi_rtol=1e-12, phi_maxit=100)
    wall = S.wall_data(bundle, t, mp, "NNN-D")
    d = S.discrete(bundle, None, opts, None, t, mp, wall=wall)
    out = rhs.nodal_perpendicular_rhs_jit(bundle.ctx, opts_solve, params, d.q, wall, sigma=d.sigma, source=d.S)
    assert bool(out.solve_info["converged"])
    assert float(jnp.abs(out.total - d.dq).max()) < 1e-8 * float(jnp.abs(d.dq).max())


def test_discrete_polarization_source_is_omega_minus_rho_star_squared_lap_psi(bundle):
    """``sigma_h = Omega - rho*^2 L_h psi`` (single-length)."""
    mp, t, _ = CASES["fast"]
    d = S.discrete(bundle, None, S.nodal_options("DDDD"), None, t, mp)
    np.testing.assert_allclose(np.asarray(d.sigma), np.asarray(d.q[..., 3] - mp.rho_star ** 2 * d.lap_psi), rtol=1e-14, atol=1e-14)


# ----------------------------------------------------------------------------------------------------- truncation
def _truncation(n, n_eta, t=0.3):
    b = syn.synthetic_bundle() if n == 16 else syn.synthetic_bundle(n, n_eta, "raw", False)    # no preconditioner needed for n = 24
    mp = P_FAST
    opts = S.nodal_options("NNN-D")
    c = S.continuum_jit(b, t, mp)
    d = S.discrete_jit(b, None, opts, None, t, mp)

    def rel(a, ref):
        H = b.H[..., None] if a.ndim == 3 else b.H
        return float(jnp.sqrt(jnp.sum(H * (a - ref) ** 2)) / jnp.sqrt(jnp.sum(H * ref ** 2)))

    return {"bracket": rel(d.bracket, c.bracket), "curvature": rel(d.curvature, c.curvature),
            "diffusion": rel(d.diffusion, c.diffusion), "lap_psi": rel(d.lap_psi, c.lap_psi), "S": rel(d.S, c.S)}


def test_continuum_vs_discrete_truncation_decreases_with_n():
    e16, e24 = _truncation(16, 8), _truncation(24, 12)
    print("truncation n=16", e16, "n=24", e24)
    for key in e16:
        assert e24[key] < e16[key] / 1.3, (key, e16[key], e24[key])
    assert e16["S"] < 0.5                                                  # the continuum source is a meaningful reference
