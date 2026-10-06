"""P10 evolved MMS, chunk C2 (fields): time derivatives, the static reduction, Hessian symmetry, the harness parameters and the ``jnp``
curvature continuum port. Fast (no geometry bundle)."""
from __future__ import annotations

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

from p10_evolved_mms import fields as F          # noqa: E402
from p10_evolved_mms import source as S          # noqa: E402

PTS = np.random.default_rng(3).uniform([0.05, 0.0, 0.0], [1.0, 2 * np.pi, 2 * np.pi], size=(48, 3))
PTS_J = jnp.asarray(PTS)


def _static_reference(pts):
    """Values, gradients, Hessians of the pinned static step-6 transverse set (``p08_step6_global.fields``)."""
    from p08_step6_global import fields as t6

    names = t6.PINNED_FIELD_SETS["transverse"]
    st = t6.TransverseState(names, [t6.PINNED_FIELDS[k] for k in names], 1.0)
    v, g, h = st.values_gradients_hessians(pts)
    return np.asarray(v).T, np.moveaxis(np.asarray(g), 0, 1), np.moveaxis(np.asarray(h), 0, 1)


@pytest.mark.parametrize("mp", [F.default_params(time_scale=1.0, a_phi=1.0), F.default_params(),
                                F.default_params(time_scale=1.0, w1=1.0)], ids=["ts1", "config", "w1"])
def test_time_derivative_matches_central_difference(mp):
    t, h = 0.0317, 1e-6
    v, dv = F.FIELDS.values_dt(PTS_J, t, mp)
    fd = (F.FIELDS.values(PTS_J, t + h, mp) - F.FIELDS.values(PTS_J, t - h, mp)) / (2 * h)
    scale = float(jnp.abs(dv).max())
    assert scale > 1e-3 * float(mp.time_scale)                      # the fields do evolve
    assert float(jnp.abs(dv - fd).max()) < 1e-6 * scale, float(jnp.abs(dv - fd).max()) / scale
    # d_t carries the time scale: the field at (t, s) is the base field at s t
    base = F.default_params(time_scale=1.0, a_phi=float(mp.a_phi), w1=float(mp.w1))
    _, dbase = F.FIELDS.values_dt(PTS_J, float(mp.time_scale) * t, base)
    np.testing.assert_allclose(np.asarray(dv), float(mp.time_scale) * np.asarray(dbase), rtol=1e-12, atol=1e-14)


def test_time_scale_zero_is_the_static_set_bitwise():
    v0, g0, h0 = _static_reference(PTS)
    mp = F.default_params(time_scale=0.0, a_phi=1.0)
    for t in (0.0, 0.37, 5.0):
        v, g, h = F.FIELDS.values_grad_hess(PTS_J, t, mp)
        assert np.array_equal(np.asarray(v), v0) and np.array_equal(np.asarray(h), h0)       # values, Hessians: bitwise
        np.testing.assert_allclose(np.asarray(g), g0, rtol=0, atol=1e-15)                      # gradients: fused to the last ulp
        _, dv = F.FIELDS.values_dt(PTS_J, t, mp)
        assert not np.any(np.asarray(dv))                           # frozen: exactly zero time derivative
    # the same as a traced zero (one jit, the time operations present and evaluating to zero)
    traced = jax.jit(lambda t, ts: F.FIELDS.values_grad_hess(PTS_J, t, mp._replace(time_scale=ts)))(0.37, 0.0)
    for got, want in zip(traced, (v0, g0, h0)):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-14)         # different fusion: a few ulp
    # and at t = 0 the full time dependence reduces to the static set as well
    v, g, h = F.FIELDS.values_grad_hess(PTS_J, 0.0, F.default_params(time_scale=1.0, a_phi=1.0))
    np.testing.assert_allclose(np.asarray(v), v0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(g), g0, rtol=0, atol=1e-13)


def test_hessian_symmetry_and_gradient_vs_difference():
    mp = F.default_params(time_scale=1.0)
    v, g, h = F.FIELDS.values_grad_hess(PTS_J, 0.21, mp)
    h = np.asarray(h)
    assert float(np.abs(h - np.swapaxes(h, -1, -2)).max()) < 1e-11 * float(np.abs(h).max())
    # gradient: central difference of the values along each logical coordinate
    eps = 1e-6
    for d in range(3):
        e = np.zeros(3)
        e[d] = eps
        fd = (np.asarray(F.FIELDS.values(PTS_J + e, 0.21, mp)) - np.asarray(F.FIELDS.values(PTS_J - e, 0.21, mp))) / (2 * eps)
        assert float(np.abs(np.asarray(g)[..., d] - fd).max()) < 1e-6 * float(np.abs(fd).max())
        gp = np.asarray(F.FIELDS.values_grad(PTS_J + e, 0.21, mp)[1])
        gm = np.asarray(F.FIELDS.values_grad(PTS_J - e, 0.21, mp)[1])
        assert float(np.abs(h[..., d] - (gp - gm) / (2 * eps)).max()) < 1e-5 * float(np.abs(h).max())


def test_harness_parameters_a_phi_and_w1():
    base = F.default_params(time_scale=1.0, a_phi=1.0)
    v1, g1, _ = F.FIELDS.values_grad_hess(PTS_J, 0.3, base)
    # a_phi scales phi only
    v2, g2, _ = F.FIELDS.values_grad_hess(PTS_J, 0.3, base._replace(a_phi=0.1))
    np.testing.assert_array_equal(np.asarray(v2[..., :4]), np.asarray(v1[..., :4]))
    np.testing.assert_allclose(np.asarray(v2[..., 4]), 0.1 * np.asarray(v1[..., 4]), rtol=1e-15, atol=0)
    np.testing.assert_allclose(np.asarray(g2[..., 4, :]), 0.1 * np.asarray(g1[..., 4, :]), rtol=1e-14, atol=1e-16)
    # w1 = 0 is the base family bitwise (python flag and traced flag); w1 = 1 changes Omega only
    v0, _, _ = F.FIELDS.values_grad_hess(PTS_J, 0.3, base._replace(w1=0.0))
    assert np.array_equal(np.asarray(v0), np.asarray(v1))
    tr = jax.jit(lambda w: F.FIELDS.values(PTS_J, 0.3, base._replace(w1=w)))
    np.testing.assert_allclose(np.asarray(tr(0.0)), np.asarray(v1), rtol=0, atol=1e-15)
    vw = np.asarray(tr(1.0))
    np.testing.assert_allclose(vw[..., [0, 1, 2, 4]], np.asarray(v1)[..., [0, 1, 2, 4]], rtol=0, atol=2e-15)
    extra = vw[..., 3] - np.asarray(v1)[..., 3]
    assert float(np.abs(extra).max()) > 1e-3
    # the added term has eta harmonic 2: invariant under eta -> eta + pi, while the base Omega (harmonic 1) flips sign
    shift = np.array([0.0, 0.0, np.pi])
    vw2 = np.asarray(F.FIELDS.values(PTS_J + shift, 0.3, base._replace(w1=1.0)))
    v12 = np.asarray(F.FIELDS.values(PTS_J + shift, 0.3, base))
    np.testing.assert_allclose(vw2[..., 3] - v12[..., 3], extra, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v12[..., 3], -np.asarray(v1)[..., 3], rtol=1e-12, atol=1e-14)


def test_positivity_of_state_fields_over_time():
    mp = F.default_params(time_scale=1.0)
    for t in np.linspace(0.0, 1.0, 11):
        v = np.asarray(F.FIELDS.values(PTS_J, t, mp))
        assert v[..., :3].min() > 0.39


@pytest.mark.parametrize("psi", ["phi_plus_tau_pi", "phi_plus_tau_ti"])
def test_jnp_curvature_port_equals_numpy_continuum_rhs(psi):
    from drbx.native.fci_perpendicular_sbp_curvature import continuum_rhs

    rng = np.random.default_rng(11)
    Q = 400
    vals = np.concatenate([rng.uniform(0.5, 1.6, (3, Q)), rng.normal(size=(2, Q))])
    grads = rng.normal(size=(5, Q, 3))
    K = rng.normal(size=(Q, 3))
    B = rng.uniform(0.7, 1.5, Q)
    for tau in (1.0, 0.37):
        ref = continuum_rhs(vals, grads, K, B, tau, psi)
        got = np.asarray(S.continuum_curvature_rhs(vals, grads, K, B, tau, psi))
        assert got.shape == ref.shape == (Q, 4)
        assert float(np.abs(got - ref).max()) <= 1e-13 * float(np.abs(ref).max())
    # batched node layout (E, P): the leading axes are free
    ref = continuum_rhs(vals.reshape(5, 20, 20), grads.reshape(5, 20, 20, 3), K.reshape(20, 20, 3), B.reshape(20, 20), 1.0, psi)
    got = np.asarray(jax.jit(lambda *a: S.continuum_curvature_rhs(*a, 1.0, psi))(vals.reshape(5, 20, 20), grads.reshape(5, 20, 20, 3),
                                                                                 K.reshape(20, 20, 3), B.reshape(20, 20)))
    assert float(np.abs(got - ref).max()) <= 1e-13 * float(np.abs(ref).max())


def test_curvature_split_variants_agree_in_the_continuum():
    """The chain rule makes the continuum RHS the same for both splits (to rounding)."""
    rng = np.random.default_rng(5)
    vals = np.concatenate([rng.uniform(0.5, 1.6, (3, 50)), rng.normal(size=(2, 50))])
    grads, K, B = rng.normal(size=(5, 50, 3)), rng.normal(size=(50, 3)), rng.uniform(0.7, 1.5, 50)
    a = np.asarray(S.continuum_curvature_rhs(vals, grads, K, B, 1.0, "phi_plus_tau_pi"))
    b = np.asarray(S.continuum_curvature_rhs(vals, grads, K, B, 1.0, "phi_plus_tau_ti"))
    assert float(np.abs(a - b).max()) < 1e-12 * float(np.abs(a).max())


def test_configuration_is_marked_provisional_and_consistent():
    c = F.CONFIG
    assert c["provisional"] is True and c["psi"] == "phi_plus_tau_pi" and c["rho_star_convention"] == "single-length"
    assert c["time_scale"] == c["rho_star"] == 4.5e-4 and abs(c["T"] - 1.0 / (30 * c["rho_star"])) < 1e-9
    assert c["a_phi"] == 0.1 and c["D"] == [1.0e-7, 1.2e-7, 1.4e-7, 0.8e-7] and c["tau"] == 1.0
    assert set(c["patterns"]) == {"NNN-D", "DDDD"} and c["curvature"]["c_kappa"] == 0.0 and not c["curvature"]["jump_dissipation"]
    opts = S.nodal_options("NNN-D")
    assert opts.diffusion_kinds == ("neumann", "neumann", "neumann", "dirichlet") and opts.psi == "phi_plus_tau_pi"
    assert S.nodal_options("DDDD").diffusion_kinds == ("dirichlet",) * 4
    with pytest.raises(ValueError, match="pattern"):
        S.nodal_options("XXXX")
