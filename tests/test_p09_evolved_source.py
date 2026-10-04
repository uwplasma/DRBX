"""P09.1a: time-dependent step-6 fields and the JAX evolved-MMS source evaluator (synthetic, no environment)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p05_direct_midpoint_global import direct_operator                              # noqa: E402
from p06_structured_global import numerics as p06numerics                           # noqa: E402
from p08_step6_global import fields as old_fields                                    # noqa: E402
from p09_evolved_mms import fields as F                                              # noqa: E402
from p09_evolved_mms import source as S                                              # noqa: E402
from p09_evolved_mms.geometry import Geometry                                        # noqa: E402

PINNED = [F.PINNED_FIELDS[n] for n in F.FIELD_SET]
TIME = [F.time_fields()[n] for n in F.FIELD_SET]


def _points(q, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(0.0, 1.0, q), rng.uniform(0.0, 2 * np.pi, q), rng.uniform(0.0, 2 * np.pi, q)]


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(b))), 1e-300))


@pytest.mark.parametrize("t", [0.0, 0.013, 0.4])
def test_static_specs_are_the_step6_fields_at_any_t(t):
    pts = _points(64)
    v_old, g_old = old_fields.TransverseState(F.FIELD_SET, PINNED, 1.0).values_gradients(pts)
    v, g = F.TimeState(PINNED, 1.0).jit_value_grad(pts, t)
    assert np.array_equal(np.asarray(v), v_old.T)
    assert _rel(np.moveaxis(np.asarray(g), 0, 1), g_old) < 1e-14
    dt = F.TimeState(PINNED, 1.0).jit_value_grad_dt(pts, t)[2]
    assert np.all(np.asarray(dt) == 0.0)


def test_time_fields_reduce_to_step6_at_t0_and_keep_positivity():
    pts = _points(256, 1)
    st = F.TimeState(TIME, 1.0)
    assert np.array_equal(np.asarray(st.jit_value(pts, 0.0)), np.asarray(F.TimeState(PINNED, 1.0).jit_value(pts, 0.0)))
    low = min(float(np.min(np.asarray(st.jit_value(pts, t))[:, :3])) for t in np.linspace(0.0, 0.3, 301))
    assert low >= 0.4 - 1e-12


def test_time_derivatives_vs_central_differences():
    pts = _points(48, 2)
    st = F.TimeState(TIME, 1.0)
    h = 1e-5
    for t in (0.0, 0.02, 0.07):
        _v, g, dt = st.jit_value_grad_dt(pts, t)
        fd = (np.asarray(st.jit_value(pts, t + h)) - np.asarray(st.jit_value(pts, t - h))) / (2 * h)
        assert _rel(dt, fd) < 1e-7
        gfd = (np.asarray(st.jit_value_grad(pts, t + h)[1]) - np.asarray(st.jit_value_grad(pts, t - h)[1])) / (2 * h)
        assert _rel(st.jit_grad_dt(pts, t), gfd) < 1e-7
    assert float(np.max(np.abs(dt))) > 1.0                    # the time dependence is really there


def test_wave_phase_travels_with_omega():
    spec = {"kind": "wave", "direction_deg": 0.0, "wavelength": 2.0, "part": "re", "offset": 1.0, "amplitude": 0.5,
            "Omega": 30.0}
    f = F.build_function(spec, 1.0)
    p = jnp.asarray([0.3, 0.2, 0.9])
    # phase = k x + eta - Omega t: shifting t by dt equals shifting eta by -Omega dt
    assert abs(float(f(p, 0.05)) - float(f(p.at[2].add(-30.0 * 0.05), 0.0))) < 1e-13


def test_point_bracket_port_matches_numpy():
    rng = np.random.default_rng(3)
    h, ga, gb = (rng.normal(size=(200, 3)) for _ in range(3))
    jac = -rng.uniform(0.2, 3.0, size=200)
    ref = direct_operator.point_bracket(h, jac, ga, gb)
    assert _rel(S.point_bracket(h, jac, ga, gb), ref) <= 1e-15


def test_continuum_terms_port_matches_numpy(monkeypatch):
    rng = np.random.default_rng(4)
    q = 300
    values = np.vstack([rng.uniform(0.4, 1.6, (3, q)), rng.normal(size=(2, q))])
    grads = rng.normal(size=(5, q, 3))
    prep = SimpleNamespace(B=rng.uniform(0.8, 1.2, q), K=rng.normal(size=(q, 3)))
    for tau in (1.0, 0.7):
        monkeypatch.setattr(p06numerics, "TAU", tau)
        ref = p06numerics._continuum_terms(values, grads, prep)[:3]
        got = S.continuum_terms(values, grads, prep.B, prep.K, tau)
        for r, g in zip(ref, got):
            assert _rel(g, r) <= 1e-15


def test_owner_reductions_match_numpy_loops():
    rng = np.random.default_rng(5)
    rows, count = 500, 17
    owner = rng.integers(0, count, rows)
    w, x = rng.uniform(0.1, 2.0, rows), rng.normal(size=(rows, 4))
    ref = np.zeros((count, 4))
    for i in range(rows):
        ref[owner[i]] += w[i] * x[i]
    assert _rel(S.owner_sum(x, w, owner, count), ref) <= 1e-14
    lower, upper = rng.integers(-1, count, 90), rng.integers(-1, count, 90)
    val = rng.normal(size=(90, 3))
    ref = np.zeros((count, 3))
    for f in range(90):
        if lower[f] >= 0:
            ref[lower[f]] -= val[f]
        if upper[f] >= 0:
            ref[upper[f]] += val[f]
    assert _rel(S.face_scatter(val, lower, upper, count), ref) <= 1e-14


def _synthetic_geometry(seed=6, q=60, n_own=7, faces=15, qd=11, qn=9):
    rng = np.random.default_rng(seed)
    owner = np.concatenate([np.arange(n_own), rng.integers(0, n_own, q - n_own)])
    volume = rng.uniform(0.5, 1.5, q)
    owner_volume = np.zeros(n_own)
    np.add.at(owner_volume, owner, volume)
    geo = dict(points=_points(q, seed), raw_owner=owner, raw_volume=volume, h=rng.normal(size=(q, 3)),
               jac=rng.uniform(0.3, 2.0, q), B=rng.uniform(0.8, 1.2, q), K=rng.normal(size=(q, 3)),
               evolution_weight=rng.uniform(0.5, 1.5, q), owner_volume=owner_volume,
               face_points=_points(faces * 9, seed + 1).reshape(faces, 9, 3), face_integrand=rng.normal(size=(faces, 9, 3)),
               face_lower=rng.integers(-1, n_own, faces), face_upper=rng.integers(0, n_own, faces),
               dirichlet_points=_points(qd, seed + 2), neumann_points=_points(qn, seed + 3),
               neumann_a=rng.normal(size=(qn, 3)))
    return Geometry(**{k: jnp.asarray(v) for k, v in geo.items()})


def _numpy_reference(geo, st, t, params, variable="phi_plus_tau_pi"):
    """The host definitions of ``reference_rhs`` (numpy ``point_bracket`` / ``_continuum_terms`` / loops).

    ``psi = phi + tau n Ti`` (hot-ion polarization; ``phi + tau Ti`` for ``variable="phi_plus_tau_ti"``)."""
    g = {k: np.asarray(v) for k, v in geo._asdict().items()}
    n_own = len(g["owner_volume"])
    v, gr, dt = (np.asarray(a) for a in st.jit_value_grad_dt(g["points"], t))
    own, vol = g["raw_owner"], g["raw_volume"]

    def mean(arr, w, den):
        out = np.zeros((n_own,) + arr.shape[1:])
        for i in range(len(own)):
            out[own[i]] += w[i] * arr[i]
        return out / den.reshape((-1,) + (1,) * (arr.ndim - 1))

    qbar, dqbar = mean(v, vol, g["owner_volume"]), mean(dt, vol, g["owner_volume"])
    gall = np.moveaxis(gr, 1, 2)                                           # (Q, 3, 5)
    bracket = np.stack([direct_operator.point_bracket(g["h"], g["jac"], gall[:, :, 4], gall[:, :, i])
                        for i in range(4)], axis=1)
    bracket = mean(bracket, vol, g["owner_volume"]) / float(params.rho_star)
    old_tau, p06numerics.TAU = p06numerics.TAU, float(params.tau)
    try:
        total = p06numerics._continuum_terms(v.T, np.moveaxis(gr, 0, 1), SimpleNamespace(B=g["B"], K=g["K"]))[2]
    finally:
        p06numerics.TAU = old_tau
    wden = np.maximum(mean(g["evolution_weight"][:, None], np.ones(len(own)), np.ones(n_own))[:, 0], 1e-300)
    curv = mean(total, g["evolution_weight"], wden)
    vf, gf = st.jit_value_grad(g["face_points"].reshape(-1, 3), t)
    gf = np.asarray(gf).reshape(len(g["face_lower"]), 9, 5, 3)
    vf = np.asarray(vf).reshape(len(g["face_lower"]), 9, 5)
    tau = float(params.tau)
    if variable == "phi_plus_tau_pi":      # grad(phi + tau n Ti) = grad phi + tau (Ti grad n + n grad Ti)
        grad_psi = gf[:, :, 4] + tau * (vf[:, :, 2, None] * gf[:, :, 0] + vf[:, :, 0, None] * gf[:, :, 2])
    else:
        grad_psi = gf[:, :, 4] + tau * gf[:, :, 2]
    gf = np.concatenate([gf[:, :, :4], grad_psi[:, :, None]], axis=2)
    face_o = np.einsum("fqa,fqka->fk", g["face_integrand"], gf)
    o = np.zeros((n_own, 5))
    for f in range(len(face_o)):
        if g["face_lower"][f] >= 0:
            o[g["face_lower"][f]] -= face_o[f]
        if g["face_upper"][f] >= 0:
            o[g["face_upper"][f]] += face_o[f]
    o /= g["owner_volume"][:, None]
    diffusion = -o[:, :4] * np.asarray(params.D)
    rbar = bracket + curv + diffusion
    return {"qbar": qbar, "dqbar_dt": dqbar, "bracket": bracket, "curv_total": curv, "diffusion": diffusion,
            "Rbar": rbar, "S": dqbar[:, :4] - rbar, "Dbar_psi": -o[:, 4], "sigma": qbar[:, 3] + o[:, 4]}


@pytest.mark.parametrize("variable", ["phi_plus_tau_pi", "phi_plus_tau_ti"])
def test_evaluate_matches_numpy_pipeline_and_time_derivatives(variable):
    geo = _synthetic_geometry()
    params = S.default_params(0.05, 1.0, np.asarray([0.01, 0.02, 0.03, 0.04]))
    # block smaller than the point count: padding is exercised
    src = S.Source(1.0, TIME, block=32, polarization_variable=variable)
    st = F.TimeState(TIME, 1.0)
    t = 0.017
    out = src.evaluate(geo, t, params)
    ref = _numpy_reference(geo, st, t, params, variable)
    for key, r in ref.items():
        assert _rel(out[key], r) <= 1e-13, key
    # dqbar_dt vs central differences of qbar(t); S is built from it
    h = 1e-5
    fd = (np.asarray(src.evaluate(geo, t + h, params)["qbar"]) - np.asarray(src.evaluate(geo, t - h, params)["qbar"])) / (2 * h)
    assert _rel(out["dqbar_dt"], fd) < 1e-7


def test_polarization_variable_changes_only_the_psi_functional():
    geo = _synthetic_geometry()
    params = S.default_params(0.05, 0.7, np.asarray([0.01, 0.02, 0.03, 0.04]))
    new = S.Source(1.0, TIME, block=32).evaluate(geo, 0.017, params)
    old = S.Source(1.0, TIME, block=32, polarization_variable="phi_plus_tau_ti").evaluate(geo, 0.017, params)
    for key in new:                       # the curvature split cpsi and every evolved-field lane are unchanged
        if key not in ("Dbar_psi", "sigma"):
            assert np.array_equal(np.asarray(new[key]), np.asarray(old[key])), key
    assert _rel(new["Dbar_psi"], old["Dbar_psi"]) > 1e-6
    with pytest.raises(ValueError, match="polarization_variable"):
        S.Source(1.0, TIME, polarization_variable="phi_plus_tau_te")


def test_boundary_data_and_time_derivative():
    geo = _synthetic_geometry()
    params = S.default_params()
    src = S.Source(1.0, TIME, block=4)
    st = F.TimeState(TIME, 1.0)
    t = 0.011
    bc, psi = src.boundary(geo, t, params)
    v, g = (np.asarray(a) for a in st.jit_value_grad(np.asarray(geo.dirichlet_points), t))
    assert np.array_equal(np.asarray(bc.dirichlet_value), v)
    assert _rel(bc.dirichlet_tangential, np.moveaxis(g[:, :, 1:], 1, 2)) <= 1e-15
    gn = np.asarray(st.jit_value_grad(np.asarray(geo.neumann_points), t)[1])
    assert _rel(bc.neumann_normal, np.einsum("qa,qfa->qf", np.asarray(geo.neumann_a), gn)) <= 1e-14
    tau = float(params.tau)
    assert _rel(psi.dirichlet_value[:, 0], v[:, 4] + tau * v[:, 0] * v[:, 2]) <= 1e-15 and psi.neumann_normal is None
    tang = np.moveaxis(g[:, :, 1:], 1, 2)                                   # (Qd, 2, 5)
    expected = tang[:, :, 4] + tau * (v[:, None, 2] * tang[:, :, 0] + v[:, None, 0] * tang[:, :, 2])
    assert _rel(psi.dirichlet_tangential[:, :, 0], expected) <= 1e-14
    legacy_bc, legacy_psi = S.Source(1.0, TIME, block=4, polarization_variable="phi_plus_tau_ti").boundary(geo, t, params)
    assert _rel(legacy_psi.dirichlet_value[:, 0], v[:, 4] + tau * v[:, 2]) <= 1e-15
    assert _rel(legacy_psi.dirichlet_tangential[:, :, 0], tang[:, :, 4] + tau * tang[:, :, 2]) <= 1e-15
    (b0, p0), (db, dp) = src.boundary_dt(geo, t, params)
    h = 1e-5
    bp, pp = src.boundary(geo, t + h, params)
    bm, pm = src.boundary(geo, t - h, params)
    for a, b_, c in ((db.dirichlet_value, bp.dirichlet_value, bm.dirichlet_value),
                     (db.dirichlet_tangential, bp.dirichlet_tangential, bm.dirichlet_tangential),
                     (db.neumann_normal, bp.neumann_normal, bm.neumann_normal),
                     (dp.dirichlet_value, pp.dirichlet_value, pm.dirichlet_value),
                     (dp.dirichlet_tangential, pp.dirichlet_tangential, pm.dirichlet_tangential)):
        assert _rel(a, (np.asarray(b_) - np.asarray(c)) / (2 * h)) < 1e-7


def test_evaluate_is_traceable_in_t_and_compiles_once():
    geo = _synthetic_geometry()
    params = S.default_params()
    src = S.Source(1.0, TIME)
    calls = []
    fn = jax.jit(lambda t: (calls.append(1), src._evaluate(geo, t, params)["S"])[1])
    a, b = fn(0.0), fn(0.01)
    assert len(calls) == 1 and not np.array_equal(np.asarray(a), np.asarray(b))
