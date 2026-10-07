"""P10 evolved MMS, chunk C5 (preflight): spectral radius, the RK4 step rule, term shares, the bracket energy abscissa and the
integration with the stage RHS of ``model.py`` (skipped until it exists), on the n = 16 family-A synthetic bundle."""
from __future__ import annotations

import json
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

from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData                 # noqa: E402
from drbx.native.fci_perpendicular_sbp_bracket import compressibility, sbp_bracket, velocity_flux   # noqa: E402
from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action      # noqa: E402
from drbx.native.fci_perpendicular_sbp_norms import h_weights                          # noqa: E402
from p10_evolved_mms import fields as F                                                # noqa: E402
from p10_evolved_mms import preflight as PF                                            # noqa: E402
from p10_evolved_mms import source as S                                                # noqa: E402
from p10_evolved_mms import synthetic as syn                                           # noqa: E402

# a_omega * rho_star^2 = 1: the O(1) vorticity of the fast sets (the config set has the consistent rho_star^2 amplitude)
P_FAST = F.default_params(rho_star=0.05, time_scale=1.0, a_phi=0.5, a_omega=1.0 / 0.05 ** 2)


@pytest.fixture(scope="module")
def bundle():
    return syn.synthetic_bundle()


# ----------------------------------------------------------------------------------------------------- spectral radius
def _block_operator(n_blocks=40, r=3.0, theta=0.7, seed=0):
    """Block diagonal: one rotation-scaling block (eigenvalues ``r e^{+-i theta}``) plus 2x2 blocks of smaller modulus."""
    rng = np.random.default_rng(seed)
    n = 2 * n_blocks
    A = np.zeros((n, n))
    A[:2, :2] = r * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for b in range(1, n_blocks):
        rb, th = rng.uniform(0.2, 0.7) * r, rng.uniform(0.0, np.pi)
        A[2 * b:2 * b + 2, 2 * b:2 * b + 2] = rb * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ A @ Q.T, r


def _nonnormal_operator(n_blocks=50, re=1.8, im=2.1, seed=4):
    """Nonsymmetric, non-normal real matrix ``Q (D + U) Q^T`` with the known spectrum of the block diagonal ``D`` (``U`` strictly
    block upper triangular): the rightmost pair is ``re +- i im``; a real eigenvalue ``-57.3`` has the largest modulus and the
    other eigenvalues have real parts in ``[-40, 0.9]``."""
    rng = np.random.default_rng(seed)
    n = 2 * n_blocks
    A = np.zeros((n, n))
    A[:2, :2] = [[re, -im], [im, re]]
    A[2, 2] = -57.3
    for b in range(2, n_blocks):
        a, w = rng.uniform(-40.0, 0.9), rng.uniform(0.0, 3.0)
        A[2 * b:2 * b + 2, 2 * b:2 * b + 2] = [[a, -w], [w, a]]
    A[3, 3] = -3.0
    A += np.triu(rng.normal(scale=0.3, size=(n, n)), k=2) * (np.arange(n)[None, :] // 2 > np.arange(n)[:, None] // 2)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ A @ Q.T


def test_rightmost_eigenvalue_of_a_nonsymmetric_matrix():
    A = _nonnormal_operator()
    assert float(np.abs(A - A.T).max()) > 1.0                                    # genuinely nonsymmetric
    ev = np.linalg.eigvals(A)
    assert abs(ev.real.max() - 1.8) < 1e-9 and abs(np.abs(ev).max() - 57.3) < 1e-9            # the known spectrum
    rng = np.random.default_rng(1)
    res = PF.rightmost_eigenvalue(lambda x: A @ x, rng.normal(size=A.shape[0]))
    assert res["converged"] and "error" not in res and res["matvecs"] > 6
    re, im = res["rightmost"]
    assert abs(re - 1.8) < 1e-6 and abs(abs(im) - 2.1) < 1e-6
    assert res["eigenvalues"][0][0] == re and len(res["eigenvalues"]) == 6
    assert all(res["eigenvalues"][i][0] >= res["eigenvalues"][i + 1][0] for i in range(5))      # descending real part
    pair = {(round(e[0], 6), round(abs(e[1]), 6)) for e in res["eigenvalues"][:2]}
    assert pair == {(1.8, 2.1)}                                                  # the conjugate pair comes first
    # the largest modulus (spectral_radius) is a different eigenvalue
    assert abs(PF.spectral_radius(lambda x: A @ x, rng.normal(size=A.shape[0]))["lambda_max"] - 57.3) < 1e-4
    # tiny operators are assembled densely
    small = PF.rightmost_eigenvalue(lambda x: A[:5, :5] @ x, np.ones(5))
    assert small["converged"] and small["rightmost"][0] == pytest.approx(np.linalg.eigvals(A[:5, :5]).real.max(), abs=1e-12)


def test_rightmost_eigenvalue_records_an_arpack_failure():
    A = _nonnormal_operator()
    res = PF.rightmost_eigenvalue(lambda x: A @ x, np.ones(A.shape[0]), ncv=14, tol=1e-14, maxiter=1)
    assert res["converged"] is False and "ArpackNoConvergence" in res["error"]
    assert set(res) >= {"rightmost", "eigenvalues", "matvecs", "converged", "error"}


def test_spectral_radius_diagonal_exact():
    rng = np.random.default_rng(3)
    d = np.concatenate([[-57.3], rng.uniform(-40.0, 30.0, 199)])                 # the largest modulus is negative
    res = PF.spectral_radius(lambda x: d * x, rng.normal(size=d.size))
    assert res["method"] == "arpack" and res["converged"]
    assert abs(res["lambda_max"] - 57.3) < 1e-6 * 57.3
    assert len(res["eigenvalues"]) == 6 and res["matvecs"] > 6
    assert abs(res["eigenvalues"][0][0] + 57.3) < 1e-5 and abs(res["eigenvalues"][0][1]) < 1e-8
    mags = [np.hypot(*e) for e in res["eigenvalues"]]
    assert all(a >= b - 1e-12 for a, b in zip(mags, mags[1:]))                   # descending modulus
    assert all(isinstance(x, float) for e in res["eigenvalues"] for x in e)


def test_spectral_radius_complex_pair_exact():
    A, r = _block_operator()
    res = PF.spectral_radius(lambda x: A @ x, np.random.default_rng(1).normal(size=A.shape[0]))
    assert res["converged"] and abs(res["lambda_max"] - r) < 1e-6 * r
    top = np.array(res["eigenvalues"][:2])                                         # the pair r e^{+-i theta}
    np.testing.assert_allclose(sorted(top[:, 1]), [-r * np.sin(0.7), r * np.sin(0.7)], atol=1e-5)
    np.testing.assert_allclose(top[:, 0], r * np.cos(0.7), atol=1e-5)
    # a random nonsymmetric matrix against the dense eigenvalues
    rng = np.random.default_rng(5)
    M = rng.normal(size=(150, 150)) / np.sqrt(150) + np.diag(np.linspace(0.0, 0.6, 150))
    res = PF.spectral_radius(lambda x: M @ x, rng.normal(size=150), tol=1e-8)
    assert abs(res["lambda_max"] - np.abs(np.linalg.eigvals(M)).max()) < 1e-5


def test_spectral_radius_power_fallback_agrees_with_arpack():
    A, r = _block_operator(seed=2)
    x0 = np.random.default_rng(4).normal(size=A.shape[0])
    a = PF.spectral_radius(lambda x: A @ x, x0)
    p = PF.spectral_radius(lambda x: A @ x, x0, method="power")
    assert a["method"] == "arpack" and p["method"] == "power" and p["converged"]
    assert abs(p["lambda_max"] - a["lambda_max"]) < 1e-4 * r                        # complex pair resolved by the Ritz estimate
    # real dominant eigenvalue (negative) and a dominant real pair
    d = np.concatenate([[-12.0, 11.0], np.random.default_rng(6).uniform(-8.0, 8.0, 98)])
    a = PF.spectral_radius(lambda x: d * x, x0[:100] if x0.size >= 100 else np.ones(100))
    p = PF.spectral_radius(lambda x: d * x, np.ones(100) + np.arange(100) / 100.0, method="power")
    assert p["converged"] and abs(p["lambda_max"] - 12.0) < 1e-4 and abs(a["lambda_max"] - 12.0) < 1e-6
    # ARPACK failure falls back to the power iteration (maxiter = 1 cannot converge)
    fb = PF.spectral_radius(lambda x: A @ x, x0, maxiter=1, ncv=8, k=2)
    assert fb["method"] == "power" and "arpack_error" in fb and abs(fb["lambda_max"] - r) < 1e-4 * r


def test_spectral_radius_small_operator_is_dense_and_validates_input():
    J = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    res = PF.spectral_radius(lambda x: J @ x, np.ones(3))
    assert res["method"] == "dense" and res["matvecs"] == 3 and abs(res["lambda_max"] - 2.0) < 1e-14
    with pytest.raises(ValueError):
        PF.spectral_radius(lambda x: x, np.zeros(10))
    with pytest.raises(ValueError):
        PF.spectral_radius(lambda x: x, np.ones(10), method="lanczos")


# ----------------------------------------------------------------------------------------------------- dt rule
def test_dt_rule_arithmetic():
    for lam, T in ((1.0, 1.0), (10.0, 1.0), (3.7e4, 74.07407407407408), (2.0e-3, 5.0), (1.3, 1.0), (50.0, 0.2)):
        dt, n = PF.dt_rule(lam, T)
        assert isinstance(n, int) and n >= 1
        assert abs(dt * n - T) <= 4e-16 * T * n                                   # dt * nsteps == T to round-off
        dt_max = 0.5 * 2.6 / lam
        assert dt <= dt_max * (1 + 1e-12)                                        # inside the safe step
        if n > 1:
            assert T / (n - 1) > dt_max * (1 - 1e-12)                            # and nsteps is the smallest such integer
    assert PF.dt_rule(1.0, 1.0) == (1.0, 1)                                      # 0.5 * 2.6 = 1.3 > 1
    dt, n = PF.dt_rule(10.0, 1.0)
    assert n == 8 and dt == 1.0 / 8                                              # ceil(1 / 0.13) = 8
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PF.dt_rule(bad, 1.0)
    with pytest.raises(ValueError):
        PF.dt_rule(1.0, 0.0)


# ----------------------------------------------------------------------------------------------------- term shares
def test_term_shares_reproduce_direct_h_norms(bundle):
    t = 0.1
    H = np.asarray(h_weights(bundle.ctx.plan))
    assert H.shape == (bundle.E, bundle.P)                                      # (E, P) convention, as the state
    c = S.continuum_jit(bundle, t, P_FAST)
    masks = PF.region_masks(bundle)
    assert "all" in masks and "core" in masks
    for mode, terms in PF.MODE_TERMS.items():
        sh = PF.term_shares(bundle, P_FAST, mode, t)
        assert tuple(sh) == PF.FIELD_LABELS
        for f, label in enumerate(PF.FIELD_LABELS):
            assert tuple(sh[label]) == terms
            for term in terms:
                assert set(sh[label][term]) == set(masks)
                arr = np.asarray(getattr(c, term))[..., f]
                dq = np.asarray(c.dq)[..., f]
                for name, m in masks.items():
                    direct = np.sqrt(np.sum(H[m] * arr[m] ** 2) / np.sum(H[m] * dq[m] ** 2))
                    v = sh[label][term][name]
                    assert v is not None and v >= 0.0 and abs(v - direct) <= 1e-12 * max(direct, 1e-300)
        # all-nodes value: the plain ratio of H norms over every node
        a = sh["Ti"][terms[0]]["all"]
        arr, dq = np.asarray(getattr(c, terms[0]))[..., 2], np.asarray(c.dq)[..., 2]
        assert abs(a - np.sqrt(np.sum(H * arr ** 2) / np.sum(H * dq ** 2))) <= 1e-12 * a
    # the exact RHS closes: d_t q = bracket + curvature + diffusion + S, and the shares of the full set are O(1) in the fast case
    full = PF.term_shares(bundle, P_FAST, "coupled", t)
    assert max(full[f][term]["all"] for f in PF.FIELD_LABELS for term in PF.MODE_TERMS["coupled"]) > 1e-3


def test_term_shares_none_where_the_time_derivative_vanishes(bundle):
    frozen = P_FAST._replace(time_scale=0.0)
    sh = PF.term_shares(bundle, frozen, "hyperbolic", 0.3)
    assert all(v is None for f in sh.values() for term in f.values() for v in term.values())
    with pytest.raises(ValueError):
        PF.term_shares(bundle, P_FAST, "bogus", 0.0)


# ----------------------------------------------------------------------------------------------------- max (c / 2)
def test_max_half_c_scaling_and_direct_compressibility(bundle):
    t, plan = 0.2, bundle.ctx.plan
    phi = PF.potential(bundle, P_FAST, t)
    np.testing.assert_allclose(np.asarray(phi), np.asarray(F.FIELDS.values(bundle.ref.points, t, P_FAST)[..., 4]), rtol=0, atol=0)
    rho = float(P_FAST.rho_star)
    got = PF.max_half_c(bundle, P_FAST, t, T=3.0)
    # direct: the compressibility of the flux of the scaled bracket (divisor 1 / rho_star = flux rho_star h x grad phi)
    c_direct = np.asarray(compressibility(plan, velocity_flux(plan, phi, 1.0 / rho)))
    assert got["max_half_c"] != 0.0 and abs(got["max_half_c"] - 0.5 * c_direct.max()) <= 1e-12 * abs(got["max_half_c"])
    assert abs(got["min_half_c"] - 0.5 * c_direct.min()) <= 1e-12 * abs(got["min_half_c"])
    assert abs(got["max_half_c_T"] - 3.0 * got["max_half_c"]) <= 1e-14 * abs(got["max_half_c_T"])
    assert PF.max_half_c(bundle, P_FAST, t)["max_half_c_T"] is None
    # homogeneity: rho_star times the value at rho_star = 1 (the fields do not depend on rho_star)
    unit = PF.max_half_c(bundle, P_FAST._replace(rho_star=1.0), t)
    assert abs(got["max_half_c"] - rho * unit["max_half_c"]) <= 1e-12 * abs(got["max_half_c"])
    c_unit = np.asarray(compressibility(plan, velocity_flux(plan, phi, 1.0)))
    np.testing.assert_allclose(c_direct, rho * c_unit, rtol=1e-12, atol=1e-14 * np.abs(c_unit).max())


def test_scaled_bracket_is_homogeneous_in_the_flux_scale(bundle):
    """``rho_star * sbp_bracket(plan, phi, g, bcd, 1.0)`` (the harness bracket) equals ``sbp_bracket(plan, phi, g, bcd, 1/rho_star)``
    including the dissipation and the SATs, which is the evidence that ``c`` scales by ``rho_star``."""
    t, plan = 0.2, bundle.ctx.plan
    phi = PF.potential(bundle, P_FAST, t)
    g = jnp.asarray(np.random.default_rng(0).normal(size=(bundle.E, bundle.P)))
    data = SatBoundaryData((jnp.asarray(np.random.default_rng(1).normal(size=(bundle.E, bundle.N))),))
    rho = float(P_FAST.rho_star)
    a = rho * sbp_bracket(plan, phi, g, data, 1.0, 1.0)
    b = sbp_bracket(plan, phi, g, data, 1.0 / rho, 1.0)
    assert float(jnp.abs(a - b).max()) <= 1e-11 * float(jnp.abs(a).max())


# ----------------------------------------------------------------------------------------------------- needs model.py
@pytest.fixture(scope="module")
def model():
    return pytest.importorskip("p10_evolved_mms.model")


def _tight(model):
    """Tight solve options for finite-difference checks (the production ``rtol = 1e-11`` is the noise floor otherwise)."""
    return {"phi_rtol": 1e-14, "phi_maxit": 400}


@pytest.mark.parametrize("mode", ["hyperbolic", "coupled"])
def test_stage_linearization_jvp_matches_central_finite_differences(bundle, model, mode):
    t = 0.1
    mv = PF.stage_linearization(bundle, mode, "continuum", P_FAST, "NNN-D", t, opts_override=_tight(model))
    x0, rhs = mv.x0, jax.jit(mv.rhs)
    rng = np.random.default_rng(7)
    v = jnp.asarray(rng.normal(size=mv.size))
    # the RHS is strongly nonlinear in the fast case (central-difference truncation error ~ 1e4 eps^2 relative), hence a small step
    eps = 1e-6 * float(jnp.linalg.norm(x0)) / float(jnp.linalg.norm(v))
    fd = (rhs(x0 + eps * v) - rhs(x0 - eps * v)) / (2.0 * eps)
    jv = mv(v)
    assert jv.shape == (mv.size,) and float(jnp.abs(jv).max()) > 0.0
    rel = float(jnp.linalg.norm(jv - fd) / jnp.linalg.norm(fd))
    assert rel <= 1e-6, rel
    # linear in v
    np.testing.assert_allclose(np.asarray(mv(2.5 * v)), 2.5 * np.asarray(jv), rtol=1e-12, atol=1e-12 * float(jnp.abs(jv).max()))


def test_diffusion_lambda_max_equals_dmax_times_laplacian_radius(bundle, model):
    pattern = "DDDD"
    mv = PF.stage_linearization(bundle, "diffusion", "continuum", P_FAST, pattern, 0.1)
    rng = np.random.default_rng(8)
    res = PF.spectral_radius(lambda x: np.asarray(mv(x)), rng.normal(size=mv.size), tol=1e-8)
    lplan, ck = bundle.ctx.lplan, S.nodal_options(pattern).laplacian_c_kappa
    zero = LaplacianBoundaryData(value=(jnp.zeros((bundle.E, bundle.N)),))
    lap = jax.jit(lambda f: laplacian_action(lplan, f, zero, "dirichlet", None, ck))
    shape = (bundle.E, bundle.P)
    direct = PF.spectral_radius(lambda x: np.asarray(lap(jnp.asarray(x.reshape(shape)))).reshape(-1),
                                rng.normal(size=bundle.E * bundle.P), tol=1e-8)
    expect = float(np.max(np.asarray(P_FAST.D))) * direct["lambda_max"]
    assert res["converged"] and direct["converged"]
    assert abs(res["lambda_max"] - expect) <= 3e-2 * expect, (res["lambda_max"], expect)
    assert direct["eigenvalues"][0][0] < 0.0 and abs(direct["eigenvalues"][0][1]) < 1e-6 * abs(direct["eigenvalues"][0][0])


def test_cg_iterations_cold_and_warm(bundle, model):
    out = PF.cg_iterations(bundle, P_FAST, "NNN-D", 0.1, 0.01)
    assert out["cold"] >= 1 and out["warm"] >= 0 and out["cold_converged"] and out["warm_converged"]
    assert out["cold"] <= 50 and out["warm"] <= 50                             # within the production budget (maxit = 50)


@pytest.mark.parametrize("mode", ["diffusion", "coupled"])
def test_preflight_writes_a_schema_valid_json(bundle, model, mode, tmp_path):
    T = 0.2
    res = PF.preflight(bundle, mode, "NNN-D", P_FAST, T, times=(0.0, T), out=tmp_path, k=3, tol=1e-4)
    on_disk = json.loads((tmp_path / "preflight.json").read_text())
    assert on_disk["schema"] == PF.SCHEMA and on_disk == json.loads(json.dumps(res))
    for key in ("lambda_max", "method", "dt_rule", "dt", "nsteps", "term_shares", "max_half_c", "max_half_c_T", "rightmost",
                "max_rightmost_real", "max_rightmost_real_T"):
        assert key in on_disk
    if mode == "diffusion":                                                    # skipped by default (clustered near-null modes)
        assert on_disk["rightmost_skipped"] is True and on_disk["rightmost"] == [None, None]
        assert on_disk["max_rightmost_real"] is None and on_disk["rightmost_converged"] is None
    else:
        assert on_disk["rightmost_skipped"] is False
        assert len(on_disk["rightmost"]) == 2 and all(len(r) == 2 for r in on_disk["rightmost"])
        assert on_disk["max_rightmost_real"] == max(r[0] for r in on_disk["rightmost"])
        assert on_disk["max_rightmost_real_T"] == pytest.approx(on_disk["max_rightmost_real"] * T, rel=1e-14)
        assert on_disk["rightmost_converged"] is True
        assert all(r[0] <= lam * (1 + 1e-6) for r, lam in zip(on_disk["rightmost"], on_disk["lambda_max"]))  # Re lambda <= |lambda|
    assert len(on_disk["lambda_max"]) == 2 and all(x > 0 for x in on_disk["lambda_max"])
    assert on_disk["dt_rule"] == "T / ceil(T / (0.5 * 2.6 / lambda_max))"
    assert on_disk["dt"] == pytest.approx(T / on_disk["nsteps"], rel=1e-14) and isinstance(on_disk["nsteps"], int)
    assert on_disk["dt"] <= 0.5 * 2.6 / max(on_disk["lambda_max"]) * (1 + 1e-12)
    assert set(on_disk["term_shares"]) == {repr(0.0), repr(T)}
    for per_t in on_disk["term_shares"].values():
        assert set(per_t) == set(PF.FIELD_LABELS)
        assert all(set(per_t[f]) == set(PF.MODE_TERMS[mode]) for f in per_t)
    assert on_disk["max_half_c_T"] == pytest.approx(on_disk["max_half_c"] * T, rel=1e-12)
    assert on_disk["identity"]["bundle_sha256"] == bundle.identity["bundle_sha256"] and on_disk["identity"]["arm"] == "raw"
    if mode == "coupled":
        assert set(on_disk["cg_iterations"]) == {"cold", "warm"}
        assert all(isinstance(v, int) for v in on_disk["cg_iterations"].values())
    else:
        assert "cg_iterations" not in on_disk
