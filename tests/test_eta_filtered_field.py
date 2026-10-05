"""The eta-filtered logical magnetic field (``drbx.geometry.eta_filtered_field``) and its scripts-side option.

Fast tests on a synthetic toroidal map with an analytic, divergence-free logical flux density ``F^i = J B^i = curl A + F0``
of known eta harmonics (per field period ``m = 0, 1, 2, 5, 7``, ``nfp = 4``):

* the column-exact form: ``M = inf`` reproduces the input, ``m > M`` is removed exactly, ``d_i (J B~^i) = 0`` (finite
  differences), nfp symmetry, the filter acts column by column;
* the tabulated form: B-spline exactness, table vs column agreement and its fourth-order convergence, the JAX twin vs
  the NumPy table (values, jit, derivatives);
* the ``curvature_autodiff`` hook: ``logical_field=None`` traces the identical program as the historic expressions,
  and the autodiff ``K`` of the filtered field (table + JAX metric) against the finite-difference ``K`` of the frozen
  reference on the column-exact field;
* ``p_shared.eta_filter`` (option check, identity, the reference swap, the provider/curvature wiring).

Slow, skip-gated on the local HSX inputs: the nodal ``h`` of the filtered N32 provider against the scoping prototype
(``work/p09_eta_filter_20261004/nodal_h_N32.npz``), the removal of the ripple line (k = 48), and the table-based autodiff
``K`` against the column-exact finite-difference ``K``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent
SCRIPTS = REPO / "scripts"
for _path in (str(REPO), str(SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.geometry import curvature_autodiff as ca  # noqa: E402
from drbx.geometry import eta_filtered_field as eff  # noqa: E402
from drbx.geometry.MetricEvaluator import MetricEvaluation  # noqa: E402

NFP = 4
PERIOD = 2.0 * np.pi / NFP
R0, A0 = 3.0, 0.5
B0 = 2.0

# (component, per-period eta harmonic m, theta harmonic l, u power p, amplitude, phase): A_comp = c u^p cos(l theta - m nfp eta + ph)
TERMS = (
    (0, 1, 1, 2, 0.040, 0.3), (1, 1, 2, 2, 0.030, 1.1), (2, 1, 1, 2, 0.020, -0.4),
    (0, 2, 2, 2, 0.020, 0.7), (1, 2, 0, 2, 0.015, 0.2), (2, 2, 1, 3, 0.020, 2.0),
    (0, 5, 1, 2, 0.006, 0.9), (1, 5, 2, 3, 0.004, -1.3), (2, 5, 0, 2, 0.005, 0.1),
    (0, 7, 2, 2, 0.003, 1.7), (2, 7, 1, 3, 0.003, -0.8),
    (0, 0, 1, 2, 0.050, 0.5), (1, 0, 1, 3, 0.040, -0.2),
)
F0 = (0.0, 0.3, 2.0)         # mean part of F (divergence free): F^theta = 0.3 u, F^eta = 2 u times the factor below


# ---------------------------------------------------------------------------
# The synthetic map, flux density and evaluators.
# ---------------------------------------------------------------------------
def _map(q):
    u, t, e = q[..., 0], q[..., 1], q[..., 2]
    R = R0 + A0 * u * jnp.cos(t) + 0.05 * u * u * jnp.cos(t - NFP * e)
    Z = -A0 * u * jnp.sin(t) * (1.0 + 0.1 * jnp.cos(NFP * e))
    phi = e + 0.04 * u * jnp.sin(t - NFP * e)
    return jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)


def _potential(q, mmax):
    u, t, e = q[0], q[1], q[2]
    comps = [0.0, 0.0, 0.0]
    for comp, m, l, p, c, ph in TERMS:
        if mmax is None or m <= mmax:
            comps[comp] = comps[comp] + c * u ** p * jnp.cos(l * t - m * NFP * e + ph)
    return jnp.stack(comps)


def _flux(q, mmax=None):
    """``F^i = J B^i``: curl of the potential (``d_i F^i = 0`` identically) plus the mean part."""
    d = jax.jacfwd(lambda x: _potential(x, mmax))(q)                  # d[comp, axis]
    curl = jnp.stack((d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]))
    return curl + jnp.stack((0.0 * q[0], F0[1] * q[0], F0[2] * q[0]))


_flux_batch = jax.jit(jax.vmap(_flux))
_flux_low = jax.jit(jax.vmap(lambda q: _flux(q, 3)))
_jac = jax.jit(jax.vmap(jax.jacfwd(_map)))
_pos = jax.jit(_map)


def _invert_one(x):
    """Logical point of a position, by Newton iterations from the torus guess (any eta representative)."""
    R, Z = jnp.hypot(x[0], x[1]), x[2]
    q = jnp.stack((jnp.hypot(R - R0, Z) / A0, jnp.arctan2(-Z, R - R0), jnp.arctan2(x[1], x[0])))

    def step(q, _):
        return q - jnp.linalg.solve(jax.jacfwd(_map)(q), _map(q) - x), None

    return jax.lax.scan(step, q, None, length=14)[0]


_invert = jax.jit(jax.vmap(_invert_one))


class _NumpyMetric:
    period = PERIOD

    def evaluate(self, q, reject_nonpositive_J=True):
        q = jnp.asarray(q, dtype=jnp.float64)
        jac = np.asarray(_jac(q))
        g = np.einsum("...ki,...kj->...ij", jac, jac)
        return MetricEvaluation(np.asarray(_pos(q)), jac, np.linalg.det(jac), g, np.linalg.inv(g),
                                np.zeros(len(jac)), np.ones(len(jac), bool))


class _JaxMetric:
    def position_and_jacobian(self, q):
        q = jnp.asarray(q, dtype=jnp.float64)
        return _map(q), jax.vmap(jax.jacfwd(_map))(q)

    def evaluate(self, q, *, reject_nonpositive_J=True):
        q = jnp.asarray(q, dtype=jnp.float64)
        jac = jax.vmap(jax.jacfwd(_map))(q)
        g = jnp.einsum("...ki,...kj->...ij", jac, jac)
        return SimpleNamespace(position=_map(q), jacobian_matrix=jac, g_cov=g, g_contra=jnp.linalg.inv(g),
                               J=jnp.linalg.det(jac))


class _BField:
    """The Cartesian field ``(dX/dq) F / J`` of the synthetic logical flux density (unfiltered, all harmonics)."""

    def evaluate_cartesian(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        flat = x.reshape(-1, 3)
        q = _invert(flat)
        jac = _jac(q)
        contra = _flux_batch(q) / jnp.linalg.det(jac)[:, None]
        return np.asarray(jnp.einsum("nij,nj->ni", jac, contra)).reshape(x.shape)


def _points(n, seed=0, u=(0.1, 1.0)):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(*u, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)], axis=1)


@pytest.fixture(scope="module")
def metric():
    return _NumpyMetric()


@pytest.fixture(scope="module")
def raw_field(metric):
    return eff.EtaFilteredColumnField(metric, _BField(), eff.EtaFilterSpec(max_harmonic_per_period=None))


@pytest.fixture(scope="module")
def field(metric):
    return eff.EtaFilteredColumnField(metric, _BField(), eff.EtaFilterSpec())


# ---------------------------------------------------------------------------
# The spec and the harmonic analysis.
# ---------------------------------------------------------------------------
def test_spec_defaults_validation_and_roundtrip():
    spec = eff.EtaFilterSpec()
    assert spec.as_dict() == {"quantity": "J*B^i", "max_harmonic_per_period": 3, "nfp": 4, "samples_per_period": 64}
    assert eff.EtaFilterSpec.from_mapping(spec.as_dict()) == spec
    assert spec.n_coefficients == 7 and np.isclose(spec.period, np.pi / 2)
    assert eff.EtaFilterSpec(max_harmonic_per_period=None).max_harmonic_per_period == 32
    for bad in ({"max_harmonic_per_period": 33}, {"max_harmonic_per_period": -1}, {"nfp": 0}, {"samples_per_period": 3},
                {"quantity": "B^i"}, {"nfp": 2.5}):
        with pytest.raises(ValueError):
            eff.EtaFilterSpec(**bad)
    with pytest.raises(ValueError, match="exactly the keys"):
        eff.EtaFilterSpec.from_mapping({"nfp": 4})


@pytest.mark.parametrize("ns", [64, 16, 15])
def test_harmonic_analysis_synthesis_roundtrip(ns):
    rng = np.random.default_rng(1)
    top = ns // 2
    spec = eff.EtaFilterSpec(max_harmonic_per_period=top, samples_per_period=ns)
    coef = rng.normal(size=(5, spec.n_coefficients, 3))
    if ns % 2 == 0:
        coef[:, top + top, :] = 0.0                      # sin at the Nyquist is not representable
    eta = lambda s: s * spec.period / ns                 # noqa: E731
    samples = eff._synthesize(np.repeat(coef, ns, axis=0), np.tile(eta(np.arange(ns)), 5), spec).reshape(5, ns, 3)
    back = eff._harmonic_coefficients(samples, spec)
    np.testing.assert_allclose(back, coef, atol=1e-13)
    # low-pass: the analysis with M = 3 keeps exactly the leading coefficients
    low = eff.EtaFilterSpec(max_harmonic_per_period=3, samples_per_period=ns)
    np.testing.assert_allclose(eff._harmonic_coefficients(samples, low), _take_low(coef, top, 3), atol=1e-13)
    # evaluation between the samples is the trigonometric interpolant
    q = rng.uniform(0, 2 * np.pi, 5)
    np.testing.assert_allclose(eff._synthesize(coef, q, spec),
                               eff._synthesize(_take_low(coef, top, top), q, spec), atol=1e-13)


def _take_low(coef, top, M):
    return np.concatenate([coef[:, : M + 1], coef[:, top + 1: top + 1 + M]], axis=1)


# ---------------------------------------------------------------------------
# (a) The column-exact form.
# ---------------------------------------------------------------------------
def test_synthetic_flux_is_divergence_free_and_has_period_symmetry():
    q = _points(6, 3)
    step = 1e-5
    div = 0.0
    scale = 0.0
    for axis in range(3):
        sh = np.zeros(3); sh[axis] = step
        term = (np.asarray(_flux_batch(q + sh))[:, axis] - np.asarray(_flux_batch(q - sh))[:, axis]) / (2 * step)
        div = div + term
        scale = scale + np.abs(term)
    assert np.all(np.abs(div) < 1e-8 * scale.max())
    np.testing.assert_allclose(np.asarray(_flux_batch(q + [0, 0, PERIOD])), np.asarray(_flux_batch(q)), atol=1e-12)


def test_unfiltered_reproduces_the_input(raw_field, metric):
    """M = infinity (every sampled harmonic): the trigonometric interpolant of 64 samples is the input (content m <= 7)."""
    q = _points(40, 5)
    F = raw_field.flux_density(q)
    np.testing.assert_allclose(F, np.asarray(_flux_batch(q)), rtol=0, atol=1e-12 * np.abs(F).max())
    contra = raw_field.contravariant(q)
    m = metric.evaluate(q)
    np.testing.assert_allclose(contra, np.asarray(_flux_batch(q)) / m.signed_J[:, None], rtol=1e-11)
    np.testing.assert_allclose(raw_field.cartesian(q), _BField().evaluate_cartesian(m.position), rtol=1e-10, atol=1e-11)
    me = raw_field.project_magnetic_field(q, m)
    np.testing.assert_allclose(me.B_cartesian, _BField().evaluate_cartesian(m.position), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(me.magnitude, np.linalg.norm(me.B_cartesian, axis=-1), rtol=1e-14)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", m.jacobian_matrix, me.B_contravariant), me.B_cartesian, rtol=1e-13)


def test_harmonics_above_M_are_removed_and_lower_are_kept(field):
    q = _points(40, 6)
    F = field.flux_density(q)
    want = np.asarray(_flux_low(q))                 # the analytic flux with the m <= 3 terms only (curl is linear)
    np.testing.assert_allclose(F, want, rtol=0, atol=1e-12 * np.abs(want).max())
    # spectrum of the filtered column along eta
    u, t = 0.6, 1.3
    eta = np.arange(64) * PERIOD / 64
    col = field.flux_density(np.stack([np.full(64, u), np.full(64, t), eta], axis=1))
    spectrum = np.abs(np.fft.rfft(col, axis=0)) / 64
    assert spectrum[4:].max() < 1e-13 * spectrum.max()
    assert spectrum[1:4].max() > 1e-3 * spectrum.max()
    # for M = 0 only the mean survives and for M = 5, m = 7 alone is gone
    spec5 = eff.EtaFilterSpec(max_harmonic_per_period=5)
    f5 = eff.EtaFilteredColumnField(field.metric_evaluator, field.bfield_evaluator, spec5)
    want5 = np.asarray(jax.jit(jax.vmap(lambda x: _flux(x, 5)))(q))
    np.testing.assert_allclose(f5.flux_density(q), want5, rtol=0, atol=1e-12 * np.abs(want5).max())
    f0 = eff.EtaFilteredColumnField(field.metric_evaluator, field.bfield_evaluator, eff.EtaFilterSpec(max_harmonic_per_period=0))
    want0 = np.asarray(jax.jit(jax.vmap(lambda x: _flux(x, 0)))(q))
    np.testing.assert_allclose(f0.flux_density(q), want0, rtol=0, atol=1e-12 * np.abs(want0).max())


def test_filtered_flux_is_divergence_free(field):
    """d_i (J B~^i) = 0 by central differences: the filter commutes with d_u, d_theta, d_eta."""
    q = _points(5, 7, u=(0.2, 0.95))
    step = 2e-4

    def partial(axis, order4=True):
        sh = np.zeros(3); sh[axis] = step
        f = lambda s: field.flux_density(q + s * sh)[:, axis]          # noqa: E731
        return (f(-2) - 8 * f(-1) + 8 * f(1) - f(2)) / (12 * step)

    terms = [partial(a) for a in range(3)]
    assert np.all(np.abs(sum(terms)) < 1e-8 * max(np.abs(t).max() for t in terms))
    # the raw divergence is not changed by dropping content, but the content removed is itself divergence free
    removed = np.asarray(_flux_batch(q)) - field.flux_density(q)
    assert np.abs(removed).max() > 1e-3                                   # a non-vacuous filter


def test_nfp_symmetry_and_column_locality(field, metric):
    q = _points(10, 8)
    for shift in (PERIOD, -3 * PERIOD, 7 * PERIOD):
        np.testing.assert_allclose(field.flux_density(q + [0, 0, shift]), field.flux_density(q), atol=1e-11)
    # B~^i at (u, theta, eta + period): J is symmetric as well
    np.testing.assert_allclose(field.contravariant(q + [0, 0, PERIOD]), field.contravariant(q), rtol=1e-9, atol=1e-11)
    # a column's coefficients do not depend on what else is asked (and the cache returns identical numbers)
    solo = eff.EtaFilteredColumnField(metric, _BField(), eff.EtaFilterSpec())
    one = solo.flux_density(q[3:4])
    np.testing.assert_array_equal(field.flux_density(q)[3:4], one)
    np.testing.assert_array_equal(field.flux_density(q[3:4]), one)
    # many points on one column cost one column
    cols = eff.EtaFilteredColumnField(metric, _BField(), eff.EtaFilterSpec())
    qq = np.stack([np.full(30, 0.4), np.full(30, 2.0), np.linspace(0, 6, 30)], axis=1)
    cols.flux_density(qq)
    assert len(cols._cache) == 1


def test_metric_period_must_match_nfp(metric):
    class Wrong(_NumpyMetric):
        period = 2 * np.pi

    with pytest.raises(ValueError, match="period"):
        eff.EtaFilteredColumnField(Wrong(), _BField(), eff.EtaFilterSpec())


def test_column_cache_is_bounded(metric):
    f = eff.EtaFilteredColumnField(metric, _BField(), eff.EtaFilterSpec(), column_chunk=4, max_cached_columns=10)
    q = _points(30, 9)
    out = f.flux_density(q)
    assert len(f._cache) <= 10
    np.testing.assert_allclose(f.flux_density(q), out, atol=1e-14)


# ---------------------------------------------------------------------------
# (b) The tabulated form.
# ---------------------------------------------------------------------------
def test_bspline_prefilter_exact_on_cubic_u_and_trig_theta():
    nu, nt = 12, 16
    u = np.linspace(0, 1.3, nu)
    th = np.arange(nt) * 2 * np.pi / nt
    U, T = np.meshgrid(u, th, indexing="ij")
    vals = ((1 + 2 * U - 0.7 * U ** 2 + 0.3 * U ** 3) * (1 + 0.4 * np.cos(T) - 0.2 * np.sin(2 * T)))[..., None]
    coef = eff._bspline_prefilter(vals)
    assert coef.shape == (nu + 2, nt, 1)
    spec = eff.EtaFilterSpec(max_harmonic_per_period=0)
    full = np.zeros((nu, nt, 1, 3)); full[..., 0, 0] = vals[..., 0]
    table = eff.EtaFilterTable(full, spec, u_max=1.3)
    uq = np.array([0.0, 0.05, 0.31, 0.77, 1.0, 1.29, 1.3])
    # exact at the nodes
    got_nodes = table.coefficients(U.ravel(), T.ravel())[:, 0, 0]
    np.testing.assert_allclose(got_nodes, vals.ravel(), atol=1e-13)
    # cubic in u is reproduced exactly between nodes (not-a-knot); theta-trig content converges (not exact at 16 points)
    for uu in uq:
        got = table.coefficients(np.full(3, uu), np.array([0.0, 2 * np.pi * 3 / nt, 2 * np.pi * 9 / nt]))[:, 0, 0]
        want = (1 + 2 * uu - 0.7 * uu ** 2 + 0.3 * uu ** 3) * (1 + 0.4 * np.cos([0.0, 2 * np.pi * 3 / nt, 2 * np.pi * 9 / nt])
                                                                  - 0.2 * np.sin(2 * np.array([0.0, 2 * np.pi * 3 / nt, 2 * np.pi * 9 / nt])))
        np.testing.assert_allclose(got, want, atol=1e-13)
    # theta is periodic: theta and theta + 2 pi are the same
    a = table.coefficients(np.array([0.5]), np.array([0.3]))
    np.testing.assert_allclose(table.coefficients(np.array([0.5]), np.array([0.3 + 2 * np.pi])), a, atol=1e-13)
    np.testing.assert_allclose(table.coefficients(np.array([0.5]), np.array([0.3 - 4 * np.pi])), a, atol=1e-13)


def _table(raw_metric, grid):
    cf = eff.EtaFilteredColumnField(raw_metric, _BField(), eff.EtaFilterSpec())
    nu, nt = grid
    return cf, eff.EtaFilterTable.from_column_field(cf, nu=nu, ntheta=nt, u_max=1.02, chunk=700)


@pytest.fixture(scope="module")
def tables(metric):
    return {grid: _table(metric, grid) for grid in ((17, 16), (33, 32), (65, 64))}


def test_table_agrees_with_columns_and_converges_fourth_order(tables):
    q = _points(120, 10, u=(0.2, 1.0))
    errors = {}
    for grid, (cf, table) in tables.items():
        check = table.check_against_columns(cf, q)
        errors[grid] = check["max_abs_over_max_ref"]
        assert set(check) == {"points", "max_abs_over_max_ref", "rms_point_relative", "max_point_relative"}
    assert errors[(65, 64)] < 5e-7, errors
    assert errors[(33, 32)] < errors[(17, 16)] / 8 and errors[(65, 64)] < errors[(33, 32)] / 8, errors   # 4th order: ~16x
    # the contravariant and Cartesian forms follow from the same coefficients
    cf, table = tables[(65, 64)]
    m = _NumpyMetric().evaluate(q)
    np.testing.assert_allclose(table.contravariant(q, m), cf.contravariant(q, m), rtol=0, atol=5e-7 * np.abs(cf.contravariant(q, m)).max())
    np.testing.assert_allclose(table.cartesian(q, m), cf.cartesian(q, m), rtol=0, atol=5e-7 * np.abs(cf.cartesian(q, m)).max())
    with pytest.raises(ValueError, match="no metric evaluator"):
        eff.EtaFilterTable(table.values, table.spec, table.u_max).contravariant(q)
    np.testing.assert_allclose(table.contravariant(q), table.contravariant(q, m), rtol=0, atol=0)   # evaluates its own metric


def test_table_beyond_the_wall_is_sampled_through_a_metric_view(metric):
    """``MetricEvaluator`` rejects u > 1 (the real map does); the table extends to u_max by sampling a ``JaxMetricView``."""

    class Strict(_NumpyMetric):
        def evaluate(self, q, reject_nonpositive_J=True):
            if np.any(np.asarray(q)[:, 0] > 1.0):
                raise ValueError("u and v queries must lie in [0, 1]")
            return super().evaluate(q, reject_nonpositive_J)

    strict = eff.EtaFilteredColumnField(Strict(), _BField(), eff.EtaFilterSpec())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        eff.EtaFilterTable.from_column_field(strict, nu=9, ntheta=8, u_max=1.02)
    view = eff.JaxMetricView(_JaxMetric(), block=64)
    table = eff.EtaFilterTable.from_column_field(strict, nu=9, ntheta=8, u_max=1.02, sample_metric=view)
    ref = eff.EtaFilterTable.from_column_field(
        eff.EtaFilteredColumnField(_NumpyMetric(), _BField(), eff.EtaFilterSpec()), nu=9, ntheta=8, u_max=1.02)
    np.testing.assert_allclose(table.values, ref.values, rtol=0, atol=1e-12 * np.abs(ref.values).max())
    q = _points(70, 21)
    m = _NumpyMetric().evaluate(q)
    v = view.evaluate(q)
    np.testing.assert_allclose(v.position, m.position, atol=1e-13)
    np.testing.assert_allclose(v.jacobian_matrix, m.jacobian_matrix, atol=1e-13)
    np.testing.assert_allclose(v.signed_J, m.signed_J, rtol=1e-13)
    assert len(view.evaluate(q[:3]).signed_J) == 3


def test_table_save_load_roundtrip(tables, tmp_path):
    cf, table = tables[(17, 16)]
    table.save(tmp_path / "t.npz", {"arm": "x"})
    back = eff.EtaFilterTable.load(tmp_path / "t.npz", metric_evaluator=cf.metric_evaluator)
    assert back.spec == table.spec and back.u_max == table.u_max and back.meta["arm"] == "x"
    np.testing.assert_array_equal(back.values, table.values)
    q = _points(5, 11)
    np.testing.assert_array_equal(back.flux_density(q), table.flux_density(q))


def test_jax_twin_matches_numpy_table_values_jit_and_derivatives(tables):
    cf, table = tables[(33, 32)]
    twin = table.to_jax()
    q = _points(60, 12, u=(0.05, 1.01))
    np.testing.assert_allclose(np.asarray(twin.flux_density(q)), table.flux_density(q), rtol=0, atol=1e-13 * np.abs(table.flux_density(q)).max())
    m = _NumpyMetric().evaluate(q)
    jm = SimpleNamespace(J=jnp.asarray(m.signed_J), jacobian_matrix=jnp.asarray(m.jacobian_matrix))
    np.testing.assert_allclose(np.asarray(twin.contravariant(q, jm)), table.contravariant(q, m), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(twin.cartesian(q, jm)), table.cartesian(q, m), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(twin.flux_density)(q)), np.asarray(twin.flux_density(q)), atol=1e-13)
    # PyTree: the coefficient array is the only leaf, jit with the table as an argument works
    leaves = jax.tree_util.tree_leaves(twin)
    assert len(leaves) == 1 and leaves[0].shape == table._coef.shape
    np.testing.assert_allclose(np.asarray(jax.jit(lambda t, x: t.flux_density(x))(twin, q)), np.asarray(twin.flux_density(q)), atol=1e-13)
    # derivatives: autodiff of the JAX twin against central differences of the NumPy table (both piecewise-cubic in u, theta)
    jac = np.asarray(jax.vmap(jax.jacfwd(twin.flux_density))(jnp.asarray(q)))             # (n, comp, axis)
    for axis in range(3):
        sh = np.zeros(3); sh[axis] = 1e-6
        fd = (table.flux_density(q + sh) - table.flux_density(q - sh)) / 2e-6
        np.testing.assert_allclose(jac[:, :, axis], fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())


# ---------------------------------------------------------------------------
# The curvature_autodiff hook.
# ---------------------------------------------------------------------------
def test_hook_default_traces_the_historic_program():
    """``logical_field=None`` is the historic expression, jaxpr-identical (hence bitwise); also the other entry points."""

    class Bf:
        def evaluate_cartesian(self, x):
            return jnp.stack((1.5 + jnp.sin(x[..., 1]), 0.2 + 0.3 * jnp.cos(x[..., 2]), 0.4 + 0.2 * jnp.sin(x[..., 0])), axis=-1)

    jm, bf = _JaxMetric(), Bf()
    floor = 1.0e-30

    def historic_parts(q):                                   # the pre-hook body of covariant_over_B_parts
        m = jm.evaluate(q[None], reject_nonpositive_J=False)
        Bc = bf.evaluate_cartesian(m.position)
        Bcontra = jnp.linalg.solve(m.jacobian_matrix, Bc[..., None])[..., 0] / B0
        bmag = jnp.maximum(jnp.linalg.norm(Bc, axis=-1) / B0, floor)
        bunit = Bcontra / bmag[..., None]
        bcov = jnp.einsum("...ij,...j->...i", m.g_cov, bunit)
        return (bcov / bmag[..., None])[0], bmag[0], m.J[0]

    def historic_tensor(q):
        m = jm.evaluate(q[None], reject_nonpositive_J=False)
        Bc = bf.evaluate_cartesian(m.position)
        Bcontra = jnp.linalg.solve(m.jacobian_matrix, Bc[..., None])[..., 0] / B0
        bmag = jnp.maximum(jnp.linalg.norm(Bc, axis=-1) / B0, floor)
        bunit = Bcontra / bmag[..., None]
        projector = m.g_contra - jnp.einsum("...i,...j->...ij", bunit, bunit)
        return (m.J[..., None, None] * projector)[0]

    q = jnp.asarray([0.4, 1.2, 0.7])
    assert str(jax.make_jaxpr(ca.covariant_over_B_parts(jm, bf, B0))(q)) == str(jax.make_jaxpr(historic_parts)(q))
    assert str(jax.make_jaxpr(ca.covariant_over_B_parts(jm, bf, B0, None))(q)) == str(jax.make_jaxpr(historic_parts)(q))
    assert str(jax.make_jaxpr(ca.perpendicular_flux_tensor_one(jm, bf, B0))(q)) == str(jax.make_jaxpr(historic_tensor)(q))
    np.testing.assert_array_equal(np.asarray(ca.covariant_over_B_parts(jm, bf, B0)(q)[0]), np.asarray(historic_parts(q)[0]))
    k_default = ca.autodiff_curvature(jm, bf, B0, block=4)(np.asarray(q)[None])
    k_none = ca.autodiff_curvature(jm, bf, B0, block=4, logical_field=None)(np.asarray(q)[None])
    np.testing.assert_array_equal(k_default, k_none)


def test_hook_with_a_logical_field_uses_its_contravariant_field():
    class Logical:
        """B^i = (0.3 + 0.1 sin(theta), 0.2 u, 1.4 + 0.1 cos(eta)) / J: a field given in logical coordinates."""

        def contravariant(self, q, m):
            f = jnp.stack((0.3 + 0.1 * jnp.sin(q[..., 1]), 0.2 * q[..., 0], 1.4 + 0.1 * jnp.cos(q[..., 2])), axis=-1)
            return f / m.J[..., None]

    jm = _JaxMetric()
    parts = ca.covariant_over_B_parts(jm, None, B0, Logical())
    q = jnp.asarray([0.4, 1.2, 0.7])
    A, bmag, J = parts(q)
    m = jm.evaluate(q[None])
    contra = np.array([0.3 + 0.1 * np.sin(1.2), 0.2 * 0.4, 1.4 + 0.1 * np.cos(0.7)]) / float(m.J[0])
    Bc = np.asarray(m.jacobian_matrix[0]) @ contra
    np.testing.assert_allclose(float(bmag), np.linalg.norm(Bc) / B0, rtol=1e-13)
    bunit = contra / B0 / float(bmag)
    np.testing.assert_allclose(np.asarray(A), np.asarray(m.g_cov[0]) @ bunit / float(bmag), rtol=1e-13)
    np.testing.assert_allclose(float(J), float(m.J[0]), rtol=1e-14)
    tensor = ca.perpendicular_flux_tensor_one(jm, None, B0, Logical())(q)
    np.testing.assert_allclose(np.asarray(tensor), float(m.J[0]) * (np.asarray(m.g_contra[0]) - np.outer(bunit, bunit)), rtol=1e-13)
    # all the entry points take it
    q1 = np.asarray(q)[None]
    k = ca.autodiff_curvature(jm, None, B0, block=2, logical_field=Logical())
    assert np.all(np.isfinite(k(q1)))
    assert np.all(np.isfinite(ca.curvature_divergence_identity(jm, None, B0, q1, logical_field=Logical())))
    tensor_pg, div = ca.autodiff_perpendicular_geometry(jm, None, B0, block=2, logical_field=Logical())(q1)
    np.testing.assert_allclose(tensor_pg[0], np.asarray(tensor), rtol=1e-12)
    assert np.all(np.isfinite(div))


def _reference(metric_evaluator=None):
    """A real ``ContinuumMmsReference`` (hash-pinned module) on the synthetic map/field."""
    from hsx_mms_continuum_reference import ContinuumMmsReference

    class Metric(_NumpyMetric):
        def project_magnetic_field(self, metrics, bfield_evaluator):
            from drbx.geometry.MetricEvaluator import MetricEvaluator
            return MetricEvaluator.project_magnetic_field(self, metrics, bfield_evaluator)

    return ContinuumMmsReference(metric_evaluator or Metric(), _BField(), B0, eta_period=2 * np.pi, enable_generalized_potential=True)


def _sidecar(tmp_path, **override):
    payload = {"metric_cache": {"sha256": "a" * 64}, "makegrid": {"sha256": "b" * 64}, "makegrid_currents": [1.0, -1.0, 0.5]}
    payload.update(override)
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(payload))
    return path


def test_autodiff_K_of_the_filtered_field_matches_finite_difference_K(tmp_path):
    """The table + JAX metric K (the hook) against the frozen finite-difference K of the column-exact filtered field."""
    from p_shared import eta_filter as pef

    reference = _reference()
    pef.apply_eta_filter(reference, _sidecar(tmp_path), eff.EtaFilterSpec().as_dict())
    arm = reference.eta_filter_arm
    q = _points(6, 13, u=(0.3, 0.95))
    fd = reference._curvature(q)                                  # frozen 4th-order stencil on the column-exact field
    scale = np.linalg.norm(fd, axis=1)
    assert np.all(scale > 1e-3)
    errors = {}
    for grid in ((33, 32), (65, 64)):
        table = eff.EtaFilterTable.from_column_field(arm.column_field, nu=grid[0], ntheta=grid[1], u_max=1.02, chunk=700)
        ad = ca.autodiff_curvature(_JaxMetric(), None, B0, block=4, logical_field=table.to_jax())(q)
        errors[grid] = (np.linalg.norm(ad - fd, axis=1) / scale).max()
    assert errors[(65, 64)] < 1e-4, errors
    assert errors[(33, 32)] > 3.0 * errors[(65, 64)], errors      # the difference is the table's interpolation error (converging)
    # the raw field has a visibly different curvature (a non-vacuous filter)
    raw = _reference()
    assert np.linalg.norm(raw._curvature(q) - fd) > 1e-3 * np.linalg.norm(fd)


# ---------------------------------------------------------------------------
# Scripts side: p_shared.eta_filter.
# ---------------------------------------------------------------------------
def test_option_check_and_identity(tmp_path):
    from p_shared import eta_filter as pef

    assert pef.DEFAULT_ETA_FILTER is None and pef.check_eta_filter(None) is None
    option = {"quantity": "J*B^i", "max_harmonic_per_period": 3, "nfp": 4, "samples_per_period": 64}
    assert pef.check_eta_filter(option) == option
    with pytest.raises(ValueError):
        pef.check_eta_filter({**option, "nfp": 0})
    with pytest.raises(ValueError):
        pef.check_eta_filter({"nfp": 4})
    base = pef.arm_identity(_sidecar(tmp_path), "compact_c3", option)
    assert len(base) == 64 and base == pef.arm_identity(_sidecar(tmp_path), "compact_c3", dict(reversed(list(option.items()))))
    variants = {
        "bfield_toroidal": pef.arm_identity(_sidecar(tmp_path), "spline", option),
        "max_harmonic": pef.arm_identity(_sidecar(tmp_path), "compact_c3", {**option, "max_harmonic_per_period": 4}),
        "samples": pef.arm_identity(_sidecar(tmp_path), "compact_c3", {**option, "samples_per_period": 128}),
        "map": pef.arm_identity(_sidecar(tmp_path, metric_cache={"sha256": "c" * 64}), "compact_c3", option),
        "makegrid": pef.arm_identity(_sidecar(tmp_path, makegrid={"sha256": "c" * 64}), "compact_c3", option),
        "currents": pef.arm_identity(_sidecar(tmp_path, makegrid_currents=[1.0, -1.0, 0.25]), "compact_c3", option),
    }
    assert len({base, *variants.values()}) == 1 + len(variants), "every identity component must move the arm identity"


def test_apply_none_leaves_the_reference_untouched(tmp_path):
    from p_shared import eta_filter as pef

    reference = _reference()
    reference.provenance = {"bfield_toroidal": "spline"}
    before = dict(reference.__dict__)
    assert pef.apply_eta_filter(reference, _sidecar(tmp_path), None) is reference
    assert reference.__dict__.keys() == before.keys() and reference.provenance == {"bfield_toroidal": "spline"}
    assert "_metric_batch" not in reference.__dict__ and not hasattr(reference, "eta_filter_arm")
    assert pef.reference_eta_filter(reference) is None


def test_apply_swaps_the_metric_batch_and_records_the_arm(tmp_path):
    from p_shared import eta_filter as pef

    reference = _reference()
    reference.provenance = {}
    raw = {k: reference._metric(_points(7, 14)) for k in ("m",)}["m"]
    option = eff.EtaFilterSpec().as_dict()
    sidecar = _sidecar(tmp_path)
    pef.apply_eta_filter(reference, sidecar, option, "compact_c3")
    arm = reference.eta_filter_arm
    assert pef.reference_eta_filter(reference) == option
    assert reference.provenance["eta_filter"] == {**option, "arm_sha256": pef.arm_identity(sidecar, "compact_c3", option)} == arm.meta()
    with pytest.raises(ValueError, match="already"):
        pef.apply_eta_filter(reference, sidecar, option)
    q = _points(7, 14)
    got = reference._metric(q)
    assert set(got) == set(raw) and all(got[k].dtype == np.float64 for k in got)
    # geometry arrays are those of the raw map; the field is the filtered one
    for key in ("J", "gcov", "gcontra"):
        np.testing.assert_array_equal(got[key], raw[key])
    contra = arm.column_field.contravariant(q) / B0
    mag = np.linalg.norm(np.einsum("nij,nj->ni", np.asarray(_NumpyMetric().evaluate(q).jacobian_matrix), contra * B0), axis=1) / B0
    np.testing.assert_allclose(got["B"], mag, rtol=1e-13)
    np.testing.assert_allclose(got["b"], contra / mag[:, None], rtol=1e-13)
    np.testing.assert_allclose(got["bcov"], np.einsum("nij,nj->ni", got["gcov"], got["b"]), rtol=1e-13)
    assert np.max(np.abs(got["b"] - raw["b"])) > 1e-4                   # the filter removed content
    # chunked queries (the reference's metric batching) agree
    reference.metric_query_batch_size = 3
    again = reference._metric(q)
    for key in got:
        np.testing.assert_allclose(again[key], got[key], rtol=1e-13)
    # an arm cannot be switched off on a filtered reference
    with pytest.raises(ValueError, match="carries"):
        pef.apply_eta_filter(reference, sidecar, None)


def test_provider_option_wiring(tmp_path):
    from p_shared import eta_filter as pef
    from p_shared.provider import ScriptsGeometryProvider

    option = eff.EtaFilterSpec().as_dict()
    assert ScriptsGeometryProvider(_reference(), curvature="fd").eta_filter is None
    filtered = pef.apply_eta_filter(_reference(), _sidecar(tmp_path), option)
    assert ScriptsGeometryProvider(filtered, curvature="fd", eta_filter=option).eta_filter == option
    with pytest.raises(ValueError, match="does not match"):
        ScriptsGeometryProvider(filtered, curvature="fd")                       # raw option, filtered reference
    with pytest.raises(ValueError, match="does not match"):
        ScriptsGeometryProvider(_reference(), curvature="fd", eta_filter=option)  # filtered option, raw reference
    with pytest.raises(ValueError, match="does not match"):
        ScriptsGeometryProvider(filtered, curvature="fd", eta_filter={**option, "max_harmonic_per_period": 4})
    prov = ScriptsGeometryProvider(filtered, curvature="fd", eta_filter=option)
    q = _points(5, 15)
    h, jac = prov.p05_metric(q)
    ref = filtered._metric(q)
    np.testing.assert_array_equal(h, ref["bcov"] / ref["B"][:, None])
    np.testing.assert_array_equal(jac, np.abs(ref["J"]))
    import inspect
    assert inspect.signature(ScriptsGeometryProvider.from_sidecar).parameters["eta_filter"].default is None


def test_curvature_reference_feeds_the_table_twin_to_the_autodiff_evaluators(tmp_path, monkeypatch):
    from p_shared import curvature_reference as cr
    from p_shared import eta_filter as pef

    class FakeJaxB:
        built = []

        @classmethod
        def from_evaluator(cls, evaluator):
            cls.built.append(evaluator)
            return "jax-bfield"

    raw = _reference()
    assert cr._jax_field(raw, FakeJaxB) == ("jax-bfield", {}) and FakeJaxB.built == [raw.bfield_evaluator]
    filtered = pef.apply_eta_filter(_reference(), _sidecar(tmp_path), eff.EtaFilterSpec().as_dict())
    sentinel = object()
    monkeypatch.setattr(filtered.eta_filter_arm, "jax_field", lambda: sentinel)
    assert cr._jax_field(filtered, FakeJaxB) == (None, {"logical_field": sentinel}) and len(FakeJaxB.built) == 1


def test_arm_table_is_built_once_cached_and_checked(tmp_path, monkeypatch):
    from p_shared import eta_filter as pef

    monkeypatch.setattr(pef, "DEFAULT_TABLE_GRID", (17, 16, 1.02), raising=False)
    monkeypatch.setattr(eff.EtaFilterTable, "from_column_field", classmethod(
        lambda cls, cf, **kw: _orig_from(cls, cf, nu=17, ntheta=16, u_max=1.02, chunk=700, sample_metric=kw.get("sample_metric"))))
    monkeypatch.setenv(pef.TABLE_CACHE_ENV, str(tmp_path / "cache"))
    option = eff.EtaFilterSpec().as_dict()
    first = pef.apply_eta_filter(_reference(), _sidecar(tmp_path), option).eta_filter_arm
    first.table_metric = eff.JaxMetricView(_JaxMetric(), block=512)             # the synthetic map has no cache payload
    twin = first.jax_field()
    assert first.jax_field() is twin and first.table_from_cache is False
    assert first.table_check["max_abs_over_max_ref"] < 1e-2 and first.table_check["points"] == pef.EQUIVALENCE_POINTS
    cached = sorted((tmp_path / "cache").glob("eta_filter_table_*.npz"))
    assert len(cached) == 1
    second = pef.apply_eta_filter(_reference(), _sidecar(tmp_path), option).eta_filter_arm
    second.table()                                                              # loaded: no sampling metric needed
    assert second.table_from_cache is True and second.table_check == first.table_check
    np.testing.assert_array_equal(second.table().values, first.table().values)
    # a table of another arm is refused
    other = pef.apply_eta_filter(_reference(), _sidecar(tmp_path, makegrid={"sha256": "d" * 64}), option).eta_filter_arm
    cached[0].rename(cached[0].with_name(f"eta_filter_table_{other.identity[:16]}.npz"))
    with pytest.raises(ValueError, match="another arm"):
        other.table()


_orig_from = eff.EtaFilterTable.from_column_field.__func__


# ---------------------------------------------------------------------------
# Real HSX geometry (slow, skip-gated on the local inputs).
# ---------------------------------------------------------------------------
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
PROTOTYPE = WORKSPACE / "work/p09_eta_filter_20261004/nodal_h_N32.npz"
OPTION = {"quantity": "J*B^i", "max_harmonic_per_period": 3, "nfp": 4, "samples_per_period": 64}


@pytest.fixture(scope="module")
def hsx_filtered_provider():
    if not (SIDECAR.is_file() and PROTOTYPE.is_file()):
        pytest.skip("HSX sidecar / scoping prototype are unavailable")
    from p_shared.provider import ScriptsGeometryProvider

    try:
        return ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="fd",
                                                    bfield_toroidal="compact_c3", eta_filter=OPTION)
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"HSX inputs unavailable: {error}")


@pytest.mark.slow
def test_hsx_n32_nodal_h_matches_the_prototype_and_removes_the_ripple_line(hsx_filtered_provider):
    from drbx.geometry.nodal_families import build_family_a_layout

    prov = hsx_filtered_provider
    assert prov.eta_filter == OPTION and prov.reference.provenance["eta_filter"]["max_harmonic_per_period"] == 3
    layout = build_family_a_layout(32, n_eta=32)
    eta = (np.arange(32) + 0.5) * layout.deta
    P = layout.P
    pts = np.stack([np.broadcast_to(layout.node_u, (32, P)), np.broadcast_to(layout.node_theta, (32, P)),
                    np.broadcast_to(eta[:, None], (32, P))], axis=-1).reshape(-1, 3)
    h, jac = prov.p05_metric(pts)
    with np.load(PROTOTYPE) as z:
        proto = np.asarray(z["h_3"]).reshape(-1, 3)
        raw_h = np.asarray(z["h_direct"]).reshape(-1, 3)
        proto_jac = np.asarray(z["jac_check"]).reshape(-1)
    np.testing.assert_allclose(h, proto, rtol=0, atol=1e-10 * np.abs(proto).max())
    np.testing.assert_allclose(jac, np.abs(proto_jac), rtol=1e-12)
    assert np.abs(h - raw_h).max() > 1e-4                       # differs from the raw arm
    # k = 48 per turn (per-period m = 12) line of h_eta on wall columns, 256 samples per turn
    ns, k48 = 256, 48
    thetas = (0.3, 1.7, 3.1, 4.6)
    ref = prov.reference
    filtered, raw = [], []
    for theta in thetas:
        q = np.stack([np.full(ns, 0.999), np.full(ns, theta), np.arange(ns) * 2 * np.pi / ns], axis=1)
        filtered.append(prov.p05_metric(q)[0][:, 2])
        metric = ref.metric_evaluator.evaluate(q, reject_nonpositive_J=False)       # the raw field of the same map
        magnetic = ref.metric_evaluator.project_magnetic_field(metric, ref.bfield_evaluator)
        bmag = np.maximum(np.asarray(magnetic.magnitude) / ref.B0, 1e-30)
        bunit = np.asarray(magnetic.B_contravariant) / ref.B0 / bmag[:, None]
        raw.append((np.einsum("nij,nj->ni", np.asarray(metric.covariant_metric), bunit) / bmag[:, None])[:, 2])
    line = lambda rows: np.array([np.abs(np.fft.rfft(r))[k48] / ns for r in rows])        # noqa: E731
    ratio = line(raw) / line(filtered)
    assert ratio.min() > 100.0, ratio                           # the scoping report: about 300x at the wall


@pytest.mark.slow
def test_hsx_table_autodiff_K_matches_the_column_exact_finite_difference_K(hsx_filtered_provider):
    from p_shared.curvature_reference import AutodiffCurvatureReference

    ref = hsx_filtered_provider.reference
    wrapped = AutodiffCurvatureReference(ref)
    rng = np.random.default_rng(3)
    q = np.stack([rng.uniform(0.15, 0.97, 12), rng.uniform(0, 2 * np.pi, 12), rng.uniform(0, 2 * np.pi, 12)], axis=1)
    ad = wrapped._curvature(q)                                     # builds the table (cached with DRBX_ETA_FILTER_TABLE_CACHE)
    fd = ref._curvature(q)
    scale = np.linalg.norm(fd, axis=1)
    err = np.linalg.norm(ad - fd, axis=1) / scale
    arm = ref.eta_filter_arm
    assert arm.table_check["max_abs_over_max_ref"] < 1e-5, arm.table_check
    assert err.max() < 1e-3, err
