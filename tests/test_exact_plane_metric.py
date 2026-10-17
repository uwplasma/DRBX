"""Exact level-set plane metric: a toroidal map whose constant-eta planes solve eta_hat(R, phi, Z) = eta exactly.

Synthetic setup (fast): a stellarator-symmetric rotating-ellipse vessel, a harmonic field fitted by the spline scalar
potential, the toroidal Fourier--Zernike base map built on that eta's own wall curves, and the exact-plane evaluator
on top.  Checks: level-set exactness and plane normals, analytic Jacobian vs finite differences, NumPy-facing vs JAX
parity (eager/jit), nested jacfwd to third order vs finite differences, bitwise batch independence of the autodiff
curvature, the extension to u = 1.02, cache payload round trips (incl. the eta payload) and the
``JaxMetricEvaluator`` dispatch.  The real-HSX end-to-end build is a slow test that skips without the inputs.
"""
from __future__ import annotations

import math
import os
import resource
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.geometry import curvature_autodiff as ca  # noqa: E402
from drbx.geometry import spline_scalar_potential as ssp  # noqa: E402
from drbx.geometry.MetricEvaluator import MetricEvaluator, build_metric_evaluator  # noqa: E402
from drbx.geometry.WallEvaluator import WallEvaluator  # noqa: E402
from drbx.geometry.eta_filtered_field import JaxMetricView  # noqa: E402
from drbx.geometry.exact_plane_metric import (  # noqa: E402
    ExactPlaneMetricEvaluator,
    JaxExactPlaneMetricEvaluator,
)
from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator  # noqa: E402

NFP = 4
PERIOD = 2.0 * math.pi / NFP


def _wall(R0=1.2, a=0.12, b=0.09, d=0.02, nt=33, npol=129):
    phi = np.linspace(0.0, PERIOD, nt)
    theta = np.linspace(0.0, 2.0 * np.pi, npol)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    rz = np.stack((R0 + a * np.cos(T) + d * np.cos(T - NFP * P), b * np.sin(T) + d * np.sin(T - NFP * P)), -1)
    rz[:, -1] = rz[:, 0]
    rz[-1] = rz[0]
    return WallEvaluator({"phi": phi, "RZ": rz, "nfp": NFP})


class _HarmonicField:
    """B = grad(G0 phi + a (R^2 - 2 Z^2) + b R^4 cos(4 phi) + c ln R)."""

    def __init__(self, G0=1.2, a=0.02, b=0.007, c=0.06):
        self.G0, self.a, self.b, self.c = G0, a, b, c
        self.period, self.nfp = PERIOD, NFP
        self.R = np.linspace(0.835, 1.625, 316)
        self.Z = np.linspace(-0.444, 0.444, 356)
        self.phi = np.arange(90) * PERIOD / 90

    def evaluate_cylindrical(self, points):
        R, ph, Z = np.asarray(points, float).T
        return np.column_stack((2 * self.a * R + 4 * self.b * R ** 3 * np.cos(4 * ph) + self.c / R,
                                (self.G0 - 4 * self.b * R ** 4 * np.sin(4 * ph)) / R, -4 * self.a * Z))


@jax.tree_util.register_pytree_node_class
class _JaxField:
    """The same harmonic field in Cartesian components, as a JAX PyTree (for the autodiff consumers)."""

    def __init__(self, field):
        self.params = jnp.asarray([field.G0, field.a, field.b, field.c])

    def tree_flatten(self):
        return (self.params,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        instance = object.__new__(cls)
        instance.params = children[0]
        return instance

    def evaluate_cartesian(self, x):
        G0, a, b, c = self.params
        R = jnp.hypot(x[..., 0], x[..., 1])
        ph = jnp.arctan2(x[..., 1], x[..., 0])
        BR = 2 * a * R + 4 * b * R ** 3 * jnp.cos(4 * ph) + c / R
        Bp = (G0 - 4 * b * R ** 4 * jnp.sin(4 * ph)) / R
        cp, sp = jnp.cos(ph), jnp.sin(ph)
        return jnp.stack((BR * cp - Bp * sp, BR * sp + Bp * cp, -4 * a * x[..., 2]), axis=-1)


def _tan_angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.linalg.norm(np.cross(a, b), axis=-1) / np.abs(np.sum(a * b, -1))


@pytest.fixture(scope="module")
def geometry():
    wall = _wall()
    field = _HarmonicField()
    eta = ssp.spline_scalar_potential_evaluator_from_bfield(field, wall, h=0.04, nphi=12, coarse=dict(maxiter=300),
                                                            maxiter=1500)
    base = build_metric_evaluator(eta, topology="toroidal", wall_evaluator=wall, mesh_shape=(9, 16, 16),
                                  radial_degree=7, poloidal_modes=5, toroidal_modes=4, projection_iterations=2)
    metric = ExactPlaneMetricEvaluator(base, eta)
    metric.validate()
    return wall, field, eta, base, metric


def _points(n, seed, u=(0.05, 1.0)):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(*u, n), rng.uniform(0.0, 2.0 * np.pi, n), rng.uniform(0.0, PERIOD, n)))


# ---------------------------------------------------------------------------------------------------------------------
def test_planes_are_exact_level_sets(geometry):
    _, _, eta, base, metric = geometry
    q = _points(400, 0)
    residual = metric.level_set_residual(q)
    assert np.max(np.abs(residual)) < 1e-14
    ev = metric.evaluate(q)
    labels = eta.evaluate_cartesian(ev.position)
    np.testing.assert_allclose(np.angle(np.exp(2j * np.pi * (labels - q[:, 2]) / PERIOD)), 0.0, rtol=0, atol=1e-13)
    normal = np.cross(ev.jacobian_matrix[..., 0], ev.jacobian_matrix[..., 1])
    _, gradient = eta.evaluate_and_gradient_cartesian(ev.position)
    assert _tan_angle(normal, gradient).max() < 1e-12
    # the base map's planes are only approximately level sets; the exact map keeps its R and Z
    base_ev = base.evaluate(q)
    np.testing.assert_allclose(np.hypot(ev.position[:, 0], ev.position[:, 1]),
                               np.hypot(base_ev.position[:, 0], base_ev.position[:, 1]), rtol=0, atol=1e-14)
    np.testing.assert_allclose(ev.position[:, 2], base_ev.position[:, 2], rtol=0, atol=1e-15)
    assert _tan_angle(np.cross(base_ev.jacobian_matrix[..., 0], base_ev.jacobian_matrix[..., 1]),
                      eta.evaluate_and_gradient_cartesian(base_ev.position)[1]).max() > 1e-9
    assert np.all(ev.J > 0)


def test_analytic_jacobian_matches_finite_differences(geometry):
    metric = geometry[-1]
    q = _points(200, 1, u=(0.05, 0.999))
    analytic = metric.jacobian_matrix(q)
    h = 1e-5
    fd = np.stack([(metric.position(q + h * e) - metric.position(q - h * e)) / (2 * h) for e in np.eye(3)], -1)
    assert np.max(np.abs(fd - analytic)) <= 1e-8 * np.max(np.abs(analytic))


def test_jax_dispatch_and_parity_with_the_numpy_facing_evaluator(geometry):
    _, _, _, base, metric = geometry
    runtime = JaxMetricEvaluator.from_metric_evaluator(metric)
    assert isinstance(runtime, JaxExactPlaneMetricEvaluator)
    assert type(JaxMetricEvaluator.from_metric_evaluator(base)) is JaxMetricEvaluator
    assert runtime.topology == "toroidal" and runtime.nfp == NFP and runtime.radial_degree == base.radial_degree
    q = _points(37, 2)
    expected = metric.evaluate(q)
    np.testing.assert_allclose(np.asarray(runtime.position(q)), expected.position, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(runtime.jacobian_matrix(q)), expected.jacobian_matrix, rtol=2e-12,
                               atol=2e-12)
    np.testing.assert_allclose(np.asarray(runtime.evaluate(q).J), expected.J, rtol=2e-12, atol=2e-12)
    jit_pj = jax.jit(lambda state, x: state.position_and_jacobian(x))
    position, jacobian = jit_pj(runtime, jnp.asarray(q))
    np.testing.assert_allclose(np.asarray(position), expected.position, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jacobian), expected.jacobian_matrix, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(runtime.evaluate)(jnp.asarray(q)).g_contra), expected.g_contra,
                               rtol=2e-12, atol=2e-12)
    leaves = jax.tree_util.tree_leaves(runtime)
    assert len(leaves) == len(jax.tree_util.tree_leaves(runtime.base)) + len(jax.tree_util.tree_leaves(runtime.spline))
    # regularized frame, including the axis
    axis = np.array([[0.0, 0.3, 0.2], [0.0, 2.1, 0.9], [0.4, 1.0, 0.7]])
    reg_np = metric.evaluate_regularized(axis)
    reg_jax = runtime.evaluate_regularized(axis)
    assert np.all(reg_np.valid)
    np.testing.assert_allclose(np.asarray(reg_jax.jacobian_matrix), reg_np.jacobian_matrix, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(reg_jax.position), reg_np.position, rtol=2e-12, atol=2e-12)
    on_axis = metric.position(np.array([[0.0, t, 0.4] for t in np.linspace(0, 2 * np.pi, 7)]))
    assert np.max(np.ptp(on_axis, axis=0)) < 1e-12
    # channels of the exact map: delta_phi = phi - eta
    values, du, dt, de = runtime.channel_values_and_derivatives(q)
    phi = np.arctan2(expected.position[:, 1], expected.position[:, 0])
    np.testing.assert_allclose(np.angle(np.exp(1j * (np.asarray(values)[:, 2] + q[:, 2] - phi))), 0.0, atol=1e-13)


def test_nested_jacfwd_matches_finite_differences(geometry):
    runtime = JaxMetricEvaluator.from_metric_evaluator(geometry[-1])

    def jacobian(state, x):
        return state.jacobian_matrix(x[None])[0]

    first = jax.jit(jacobian)
    second = jax.jit(jax.jacfwd(jacobian, argnums=1))
    third = jax.jit(jax.jacfwd(jax.jacfwd(jacobian, argnums=1), argnums=1))
    position_second = jax.jit(jax.jacfwd(jax.jacfwd(lambda state, y: state.position(y[None])[0], argnums=1),
                                         argnums=1))
    h = 1e-4
    eye = jnp.eye(3)
    for q in _points(3, 3, u=(0.2, 0.95)):
        x = jnp.asarray(q)
        d2 = np.asarray(second(runtime, x))
        fd2 = np.stack([(np.asarray(first(runtime, x + h * e)) - np.asarray(first(runtime, x - h * e))) / (2 * h)
                        for e in eye], -1)
        assert np.max(np.abs(d2 - fd2)) <= 1e-7 * np.max(np.abs(fd2))
        d3 = np.asarray(third(runtime, x))
        fd3 = np.stack([(np.asarray(second(runtime, x + h * e)) - np.asarray(second(runtime, x - h * e))) / (2 * h)
                        for e in eye], -1)
        assert np.max(np.abs(d3 - fd3)) <= 1e-6 * np.max(np.abs(fd3))
        # the position's second derivative (custom_root through phi) equals the explicit Jacobian's derivative
        np.testing.assert_allclose(np.asarray(position_second(runtime, x)), d2, rtol=0,
                                   atol=1e-12 * np.max(np.abs(d2)))


def test_autodiff_curvature_is_bitwise_batch_independent(geometry):
    _, field, _, _, metric = geometry
    runtime = JaxMetricEvaluator.from_metric_evaluator(metric)
    kernel = ca.autodiff_curvature(runtime, _JaxField(field), 1.0, block=8)
    points = _points(19, 4, u=(0.1, 1.0))
    whole = kernel(points)
    assert np.all(np.isfinite(whole))
    np.testing.assert_array_equal(np.concatenate([kernel(points[i:i + 1]) for i in range(len(points))]), whole)
    np.testing.assert_array_equal(np.concatenate([kernel(points[i:i + 3]) for i in range(0, len(points), 3)]), whole)
    np.testing.assert_array_equal(kernel(points[::-1])[::-1], whole)
    # third map derivatives: the divergence identity d_i(|J| K^i / B) = 0 holds to round-off
    terms = kernel.divergence_identity_terms(points[:8])
    assert np.all(np.abs(terms.sum(-1)) <= 1e-9 * np.abs(terms).sum(-1))


def test_evaluation_extends_to_u_1p02(geometry):
    metric = geometry[-1]
    runtime = JaxMetricEvaluator.from_metric_evaluator(metric)
    view = JaxMetricView(runtime, block=64)
    q = _points(100, 5, u=(0.98, 1.02))
    ev = view.evaluate(q)
    assert np.all(np.isfinite(ev.position)) and np.all(ev.signed_J > 0)
    assert np.max(np.abs(np.asarray(runtime.level_set_residual(q)))) < 1e-13
    inside = q[:, 0] <= 1.0
    np.testing.assert_allclose(ev.position[inside], metric.position(q[inside]), rtol=0, atol=2e-12)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metric.evaluate(q[~inside])
    with pytest.raises(ValueError, match="u=0"):
        metric.evaluate(np.array([[0.0, 0.2, 0.3]]))


def test_cache_payload_round_trip_includes_the_eta(geometry, tmp_path):
    _, _, eta, base, metric = geometry
    payload = metric.to_cache_payload(prefix="metric_evaluator_")
    assert str(payload["metric_evaluator_plane_construction"]) == "exact_level_set"
    assert "metric_evaluator_exact_plane_eta_coefficients" in payload
    path = tmp_path / "metric.npz"
    np.savez(path, **payload)
    q = _points(64, 6)
    with np.load(path, allow_pickle=False) as loaded:
        restored = MetricEvaluator.from_cache_payload(loaded, prefix="metric_evaluator_")
        runtime = JaxMetricEvaluator.from_cache_payload(loaded, prefix="metric_evaluator_")
        plain = MetricEvaluator._from_plain_cache_payload(loaded, prefix="metric_evaluator_")
    assert isinstance(restored, ExactPlaneMetricEvaluator)
    assert isinstance(runtime, JaxExactPlaneMetricEvaluator)
    np.testing.assert_array_equal(restored.eta_potential.coefficients, eta.coefficients)
    assert restored.eta_potential.G == eta.G and restored.newton_iterations == metric.newton_iterations
    np.testing.assert_array_equal(restored.position(q), metric.position(q))
    np.testing.assert_array_equal(restored.jacobian_matrix(q), metric.jacobian_matrix(q))
    np.testing.assert_allclose(plain.position(q), base.position(q), rtol=0, atol=1e-14)   # NumPy base: ulp-level
    np.testing.assert_allclose(np.asarray(runtime.position(q)), metric.position(q), rtol=2e-12, atol=2e-12)
    # a fitted payload stays a fitted MetricEvaluator
    assert type(MetricEvaluator.from_cache_payload(base.to_cache_payload())) is MetricEvaluator
    bad = dict(payload)
    bad["metric_evaluator_plane_construction"] = np.asarray("something_else")
    with pytest.raises(ValueError, match="plane construction"):
        MetricEvaluator.from_cache_payload(bad, prefix="metric_evaluator_")


def test_constructor_validation(geometry):
    _, _, eta, base, metric = geometry
    with pytest.raises(TypeError):
        ExactPlaneMetricEvaluator(metric, eta)
    with pytest.raises(TypeError):
        ExactPlaneMetricEvaluator(base, object())
    assert metric.plane_construction == "exact_level_set" and metric.base_metric is base
    assert metric.topology == "toroidal" and metric.period == base.period and metric._channels == ()
    np.testing.assert_array_equal(metric.u, base.u)
    np.testing.assert_array_equal(metric.eta, base.eta)


# ---------------------------------------------------------------------------------------------------------------------
# Real HSX (slow; skipped without the MAKEGRID/vessel inputs)
# ---------------------------------------------------------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("DRBX_HSX_WORKSPACE", Path(__file__).resolve().parents[2]))
MAKEGRID = WORKSPACE / "mgrid_res2p5cm_180pln.nc"
VESSEL = WORKSPACE / "vessel_hsx_flare.txt"
needs_hsx = pytest.mark.skipif(not (MAKEGRID.is_file() and VESSEL.is_file()),
                               reason="HSX MAKEGRID/vessel inputs are unavailable")


@pytest.mark.slow
@needs_hsx
def test_real_hsx_spline_exact_plane_metric_end_to_end():
    from drbx.geometry import hsx_fci_builder as hsx

    metric, eta, bfield, nfp = hsx.build_hsx_metric_evaluator(
        makegrid_path=MAKEGRID, vessel_path=VESSEL, resolution=(16, 16, 32), fit_sample_shape=(8, 9, 8),
        radial_degree=3, vertical_degree=3, toroidal_modes=2, metric_spline_degree=1, mmpde_iterations=0,
        axis_core_radius=0.03, makegrid_currents=hsx.DEFAULT_HSX_QHS_MAKEGRID_CURRENTS, topology="toroidal",
        metric_mesh_shape=(64, 64, 64), metric_radial_degree=17, metric_poloidal_modes=15, metric_toroidal_modes=15,
        eta_projection_iterations=3, eta_representation="spline", plane_construction="exact")
    assert isinstance(metric, ExactPlaneMetricEvaluator) and nfp == 4
    assert isinstance(eta, ssp.SplineScalarPotentialEvaluator)
    assert eta.diagnostics["misfit_tan_angle_all_data"]["rms"] < 1e-4
    q = _points(4096, 7, u=(1e-3, 1.0))
    assert np.max(np.abs(metric.level_set_residual(q))) < 1e-13
    ev = metric.evaluate(q)
    normal = np.cross(ev.jacobian_matrix[..., 0], ev.jacobian_matrix[..., 1])
    assert _tan_angle(normal, eta.evaluate_and_gradient_cartesian(ev.position)[1]).max() < 1e-11
    B = bfield.evaluate_cartesian(ev.position)
    core = q[:, 0] <= 0.5
    assert np.sqrt(np.mean(_tan_angle(normal[core], B[core]) ** 2)) < 1e-5
    payload = metric.to_cache_payload(prefix="metric_evaluator_")
    restored = MetricEvaluator.from_cache_payload(payload, prefix="metric_evaluator_")
    np.testing.assert_array_equal(restored.position(q), ev.position)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if os.uname().sysname == "Darwin" else 1e6)
    assert peak_gb < 3.5
