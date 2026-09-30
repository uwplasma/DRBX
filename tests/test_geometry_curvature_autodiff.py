"""Fast tests of the autodiff curvature (K1 of the P08 bundle).

* ``drbx.geometry.curvature_autodiff`` on a synthetic curvilinear map with an analytic Cartesian field:
  ``K`` against an independent NumPy Cartesian-curl finite difference, the divergence identity, and
  batch-size independence of the sequential mode (bitwise) and of the fixed-block mode.
* ``scripts/p_shared/curvature_reference.AutodiffCurvatureReference``: delegation, write-through,
  the ``metric_reuse`` interplay, and the frozen call sites (``curvature_geometry``).
* ``ScriptsGeometryProvider(curvature=...)``: the raw and face paths, wall nodes, and that the option is
  threaded (keyword, default ``"fd"``) and recorded in the build policy/identity.

The real-HSX checks are in ``test_p_shared_curvature_reference_real.py`` (slow).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.geometry import curvature_autodiff as ca  # noqa: E402

B0 = 1.7


# ---------------------------------------------------------------------------
# A synthetic curvilinear map and Cartesian field (JAX side and independent NumPy side).
# ---------------------------------------------------------------------------
def _map(q):
    u, t, e = q[..., 0], q[..., 1], q[..., 2]
    return jnp.stack((1.0 + 0.5 * u + 0.1 * jnp.sin(t) * u + 0.05 * e * u,
                      0.4 * t + 0.08 * u * u + 0.1 * jnp.cos(e),
                      0.7 * e + 0.2 * u * jnp.cos(t) + 0.05 * t), axis=-1)


def _field_jax(x):
    return jnp.stack((1.5 + jnp.sin(x[..., 1]), 0.2 + 0.3 * jnp.cos(x[..., 2]) + 0.2 * x[..., 0],
                      0.4 + 0.2 * jnp.sin(x[..., 0]) + 0.1 * x[..., 1] * x[..., 2]), axis=-1)


def _field_np(x):
    return np.stack((1.5 + np.sin(x[..., 1]), 0.2 + 0.3 * np.cos(x[..., 2]) + 0.2 * x[..., 0],
                     0.4 + 0.2 * np.sin(x[..., 0]) + 0.1 * x[..., 1] * x[..., 2]), axis=-1)


class _Metric:
    def evaluate(self, q, *, reject_nonpositive_J=True):
        q = jnp.asarray(q, dtype=jnp.float64)
        position = _map(q)
        jac = jax.vmap(jax.jacfwd(_map))(q)          # columns d X / d q_i
        g = jnp.einsum("...ki,...kj->...ij", jac, jac)
        return SimpleNamespace(position=position, jacobian_matrix=jac, g_cov=g, J=jnp.linalg.det(jac))


class _BField:
    def evaluate_cartesian(self, x):
        return _field_jax(jnp.asarray(x, dtype=jnp.float64))


POINTS = np.array([[0.31, 0.7, 0.2], [0.55, 2.1, 1.3], [0.9, 4.0, 0.4], [1.0, 5.5, 2.2], [0.05, 1.0, 0.9],
                   [0.62, 3.3, 0.05], [0.77, 0.2, 1.9]])


def _expected_K(q):
    """Independent NumPy reference: K_logical = 0.5 * bmag * J_mat^-1 curl_cart(A_cart), A_cart = bhat / bmag."""
    q = np.asarray(q, dtype=np.float64)
    x = np.asarray(_map(jnp.asarray(q)))
    jac = np.asarray(jax.vmap(jax.jacfwd(_map))(jnp.asarray(q)))

    def A(y):
        b = _field_np(y)
        norm = np.linalg.norm(b, axis=-1, keepdims=True)
        return b / norm / (norm / B0)

    h = 1.0e-3
    d = np.zeros((len(x), 3, 3))                      # d[:, comp, axis]
    for axis in range(3):
        step = np.zeros(3); step[axis] = h
        # 6th-order central difference
        d[:, :, axis] = (45 * (A(x + step) - A(x - step)) - 9 * (A(x + 2 * step) - A(x - 2 * step))
                         + (A(x + 3 * step) - A(x - 3 * step))) / (60 * h)
    curl = np.stack((d[:, 2, 1] - d[:, 1, 2], d[:, 0, 2] - d[:, 2, 0], d[:, 1, 0] - d[:, 0, 1]), axis=-1)
    bmag = np.linalg.norm(_field_np(x), axis=-1) / B0
    return 0.5 * bmag[:, None] * np.linalg.solve(jac, curl[..., None])[..., 0]


@pytest.fixture(scope="module")
def kernel():
    return ca.autodiff_curvature(_Metric(), _BField(), B0, block=8)


def test_autodiff_curvature_matches_independent_cartesian_curl(kernel):
    K = kernel(POINTS)
    assert K.shape == (len(POINTS), 3) and K.dtype == np.float64
    expected = _expected_K(POINTS)
    scale = np.linalg.norm(expected, axis=1)
    assert np.all(np.linalg.norm(K - expected, axis=1) <= 1.0e-8 * scale)
    assert np.all(scale > 1.0e-3)            # a nontrivial field: not a vacuous comparison


def test_autodiff_curvature_exact_at_u_equal_one(kernel):
    q = POINTS[POINTS[:, 0] == 1.0]
    assert len(q) == 1
    K = kernel(q)
    assert np.all(np.isfinite(K))
    np.testing.assert_allclose(K, _expected_K(q), rtol=1e-8)


def test_divergence_identity_is_roundoff(kernel):
    div = kernel.divergence_identity(POINTS)
    terms = kernel.divergence_identity_terms(POINTS)
    assert div.shape == (len(POINTS),)
    np.testing.assert_array_equal(div, terms.sum(axis=-1))
    scale = np.abs(terms).sum(axis=-1)
    assert np.all(scale > 1.0e-3)
    assert np.all(np.abs(div) <= 1.0e-10 * scale)
    free = ca.curvature_divergence_identity(_Metric(), _BField(), B0, POINTS)
    assert np.all(np.abs(free) <= 1.0e-10 * scale)


def test_default_mode_is_block():
    assert ca.DEFAULT_MODE == "block"
    assert ca.autodiff_curvature(_Metric(), _BField(), B0).mode == "block"


@pytest.fixture(scope="module")
def sequential():
    return ca.autodiff_curvature(_Metric(), _BField(), B0, mode="sequential", block=8)


@pytest.mark.parametrize("which", ["default", "sequential"])
def test_results_are_bitwise_independent_of_batching(which, kernel, sequential):
    k = kernel if which == "default" else sequential
    whole = k(POINTS)
    ones = np.concatenate([k(POINTS[i:i + 1]) for i in range(len(POINTS))])
    threes = np.concatenate([k(POINTS[i:i + 3]) for i in range(0, len(POINTS), 3)])
    np.testing.assert_array_equal(ones, whole)
    np.testing.assert_array_equal(threes, whole)
    np.testing.assert_array_equal(k(POINTS[::-1])[::-1], whole)
    # more than one block (block=8): 19 points
    many = np.concatenate([POINTS, POINTS + 0.001, POINTS[:5] + 0.002])
    assert len(many) > 2 * k.block
    all_at_once = k(many)
    np.testing.assert_array_equal(np.concatenate([k(many[i:i + 4]) for i in range(0, len(many), 4)]), all_at_once)


def test_block_and_sequential_modes_agree_to_roundoff(kernel, sequential):
    np.testing.assert_allclose(kernel(POINTS), sequential(POINTS), rtol=1e-9)


def test_input_validation_and_empty(kernel):
    assert kernel(np.zeros((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        kernel(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        ca.autodiff_curvature(_Metric(), _BField(), B0, mode="nope")
    with pytest.raises(ValueError):
        ca.autodiff_curvature(_Metric(), _BField(), B0, block=0)


# ---------------------------------------------------------------------------
# The reference wrapper.
# ---------------------------------------------------------------------------
from p_shared import curvature_reference as cr  # noqa: E402


class _FakeReference:
    """Stands in for the scripts continuum reference: distinct FD ``_curvature``, a metric, mutable attributes."""

    finite_difference_step = 2.0e-4

    def __init__(self):
        self.B0 = 1.0
        self.calls = []

    def _metric(self, q):
        q = np.asarray(q, dtype=np.float64)
        self.calls.append("metric")
        return {"J": 1.0 + q[:, 0], "B": 2.0 + q[:, 1], "bcov": np.ones((len(q), 3))}

    def _curvature(self, q):
        return np.full((len(q), 3), -1.0)           # the "fd" value

    def prepare(self, q):
        return self._curvature(q)                   # internal self._curvature stays finite-difference

    def other(self):
        return "delegated"


def _wrapped(value=7.0):
    ref = _FakeReference()
    wrapper = cr.AutodiffCurvatureReference(ref)
    object.__setattr__(wrapper, "_autodiff_k", lambda q: np.full((len(q), 3), value))   # skip building JAX evaluators
    return ref, wrapper


def test_wrapper_overrides_only_curvature_and_delegates_everything_else():
    ref, wrapper = _wrapped()
    q = np.array([[0.5, 1.0, 2.0], [1.0, 0.2, 0.3]])
    np.testing.assert_array_equal(wrapper._curvature(q), np.full((2, 3), 7.0))
    np.testing.assert_array_equal(ref._curvature(q), np.full((2, 3), -1.0))
    assert wrapper.other() == "delegated"
    assert wrapper.wrapped is ref and wrapper.B0 == 1.0
    np.testing.assert_array_equal(wrapper._metric(q)["J"], ref._metric(q)["J"])
    np.testing.assert_array_equal(wrapper.prepare(q), np.full((2, 3), -1.0))    # documented: internal callers stay FD
    with pytest.raises(AttributeError):
        wrapper.no_such_attribute
    assert cr.AutodiffCurvatureReference(wrapper).wrapped is ref               # no double wrapping


def test_wrapper_writes_reach_the_wrapped_reference():
    ref, wrapper = _wrapped()
    wrapper.finite_difference_step = 1.0e-3            # the campaigns' control sweeps do this on ``ref``
    assert ref.finite_difference_step == 1.0e-3 and wrapper.finite_difference_step == 1.0e-3
    del wrapper.finite_difference_step
    assert "finite_difference_step" not in vars(ref)   # instance attribute removed; class default is back
    assert wrapper.finite_difference_step == 2.0e-4


def test_curvature_geometry_uses_autodiff_k_and_restores_metric():
    from perpendicular_structured.reference_geometry import curvature_geometry

    ref, wrapper = _wrapped(value=3.0)
    q = np.array([[0.4, 0.5, 0.6], [0.6, 0.1, 0.2]])
    prepared = curvature_geometry(wrapper, q)
    np.testing.assert_array_equal(prepared.K, np.full((2, 3), 3.0))
    np.testing.assert_array_equal(prepared.J, ref._metric(q)["J"])
    np.testing.assert_array_equal(prepared.B, ref._metric(q)["B"])
    assert "_metric" not in vars(wrapper)               # metric_reuse released its per-call cache
    plain = curvature_geometry(ref, q)                  # the unwrapped reference: fd K, unchanged
    np.testing.assert_array_equal(plain.K, np.full((2, 3), -1.0))


def test_wrap_reference_and_choice_validation():
    ref = _FakeReference()
    assert cr.wrap_reference(ref, "fd") is ref
    assert isinstance(cr.wrap_reference(ref, "autodiff"), cr.AutodiffCurvatureReference)
    for bad in ("FD", "auto", None):
        with pytest.raises(ValueError):
            cr.check_curvature(bad)


# ---------------------------------------------------------------------------
# Provider option.
# ---------------------------------------------------------------------------
from p_shared import provider as ps_provider  # noqa: E402


def test_provider_default_is_autodiff_and_fd_is_the_plain_reference():
    ref = _FakeReference()
    assert cr.DEFAULT_CURVATURE == "autodiff"
    prov = ps_provider.ScriptsGeometryProvider(ref)
    assert prov.curvature == "autodiff" and isinstance(prov.reference, cr.AutodiffCurvatureReference)
    prov = ps_provider.ScriptsGeometryProvider(ref, curvature="fd")
    assert prov.curvature == "fd" and prov.reference is ref
    with pytest.raises(ValueError):
        ps_provider.ScriptsGeometryProvider(ref, curvature="sideways")


def test_provider_autodiff_raw_and_face_paths_use_autodiff_everywhere():
    ref = _FakeReference()                       # has no ``_derivative``: the one-sided wall rule would raise
    prov = ps_provider.ScriptsGeometryProvider(ref, curvature="autodiff")
    assert isinstance(prov.reference, cr.AutodiffCurvatureReference) and prov.curvature == "autodiff"
    object.__setattr__(prov.reference, "_autodiff_k", lambda q: np.full((len(q), 3), 5.0))
    q = np.array([[0.3, 0.2, 0.1], [1.0, 0.4, 0.5], [1.0 - 4 * np.finfo(float).eps, 0.4, 0.5], [0.9, 0.1, 0.2]])
    J, B, K = prov.p06_face_curvature(q)
    np.testing.assert_array_equal(J, 1.0 + q[:, 0])
    np.testing.assert_array_equal(B, 2.0 + q[:, 1])
    np.testing.assert_array_equal(K, np.full((4, 3), 5.0))              # wall nodes included
    J2, B2, K2 = prov.p06_curvature(q)
    np.testing.assert_array_equal(J2, J)
    np.testing.assert_array_equal(B2, B)
    np.testing.assert_array_equal(K2, np.full((4, 3), 5.0))


def test_provider_fd_face_path_is_the_frozen_face_geometry():
    import p06_structured_global.numerics as p06numerics

    ref = _FakeReference()
    prov = ps_provider.ScriptsGeometryProvider(ref, curvature="fd")
    q = np.array([[0.3, 0.2, 0.1], [0.5, 0.4, 0.5], [0.9, 0.1, 0.2]])      # interior nodes only
    got = prov.p06_face_curvature(q)
    want = p06numerics._face_geometry(ref, q)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(got[2], np.full((3, 3), -1.0))


# ---------------------------------------------------------------------------
# Option threading and identity.
# ---------------------------------------------------------------------------
def test_curvature_option_is_threaded_with_default_autodiff():
    from p_shared import build_artifact, jax_replay, owner_closure, perpendicular_reference_rhs, replay_support
    from p_shared import step3_gates

    targets = [replay_support.build_environment, owner_closure.load_provider_for_env,
               owner_closure.run_owner_closure_check, jax_replay.run_jax_owner_closure_check,
               step3_gates.build_setup, step3_gates.run_g32, step3_gates.run_g33,
               build_artifact.build_identity, build_artifact.build_geometry_only, build_artifact.run_full_build,
               build_artifact._init_geometry_worker, build_artifact._init_worker,
               ps_provider.ScriptsGeometryProvider.from_sidecar, ps_provider.ScriptsGeometryProvider.__init__]
    for fn in targets:
        parameter = inspect.signature(fn).parameters["curvature"]
        assert parameter.default == "autodiff", fn
        assert parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD), fn
    assert replay_support.Environment.__dataclass_fields__["curvature"].default == "autodiff"
    assert step3_gates.Step3Setup.__dataclass_fields__["curvature"].default == "autodiff"
    assert perpendicular_reference_rhs.env_curvature(SimpleNamespace()) == "fd"      # an unwrapped legacy env
    assert perpendicular_reference_rhs.env_curvature(SimpleNamespace(curvature="autodiff")) == "autodiff"


def test_build_owner_rows_rejects_a_provider_that_disagrees_with_the_environment():
    from p_shared import owner_closure

    with pytest.raises(ValueError, match="curvature"):
        owner_closure.build_owner_rows(SimpleNamespace(curvature="fd"), [0],
                                       provider=SimpleNamespace(curvature="autodiff"))
    with pytest.raises(ValueError, match="curvature"):
        owner_closure.build_owner_rows(SimpleNamespace(curvature="autodiff"), [0],
                                       provider=SimpleNamespace(curvature="fd"))


def test_build_policy_and_identity_distinguish_curvature(monkeypatch):
    from p_shared import build_artifact as ba

    assert ba.build_policy("fd", "q3") == ba.POLICY and "curvature" not in ba.build_policy("fd")
    assert ba.build_policy("autodiff", "q3") == {**ba.POLICY, "curvature": "autodiff"}
    assert "curvature" not in ba.POLICY                          # the module constant (fd identities) is untouched
    with pytest.raises(ValueError):
        ba.build_policy("bogus")

    monkeypatch.setattr(ba, "_geometry_component_hashes", lambda root, n: {"geometry": "g"})
    monkeypatch.setattr(ba, "_sidecar_component_hashes", lambda path: {"sidecar": "s"})
    default = ba.build_identity(n=32, input_root=Path("."), sidecar_path=Path("."))
    assert default == ba.build_identity(n=32, input_root=Path("."), sidecar_path=Path("."), curvature="autodiff")
    fd = ba.build_identity(n=32, input_root=Path("."), sidecar_path=Path("."), curvature="fd", face_quadrature="q3")
    ad = ba.build_identity(n=32, input_root=Path("."), sidecar_path=Path("."), curvature="autodiff",
                           face_quadrature="q3")
    assert fd["policy"] == ba.POLICY
    assert ad != fd and ad["policy"]["curvature"] == "autodiff"
    assert set(fd["source_hashes"]) == set(ba.SOURCE_FILES)
    assert set(ad["source_hashes"]) == set(ba.SOURCE_FILES) | set(ba.AUTODIFF_SOURCE_FILES)
    for rel in ba.AUTODIFF_SOURCE_FILES:
        assert (REPO / rel).is_file()
    assert ba.parse_args(["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o",
                          "--workers", "1"]).curvature == "autodiff"
    assert ba.parse_args(["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o", "--workers", "1",
                          "--curvature", "fd"]).curvature == "fd"
    assert ba.parse_args(["--n", "32", "--input-root", ".", "--sidecar", "s", "--output", "o", "--workers", "1",
                          "--curvature", "autodiff"]).curvature == "autodiff"


def test_face_geometry_helper_and_wrapper_reuse():
    ref = _FakeReference()                       # no ``_derivative``: the frozen wall rule would fail on a wall node
    wrapper = cr.AutodiffCurvatureReference(ref, mode="sequential", block=4)
    assert cr.wrap_reference(wrapper, "autodiff") is wrapper             # a chosen mode/block is kept
    object.__setattr__(wrapper, "_autodiff_k", lambda q: np.full((len(q), 3), 9.0))
    q = np.array([[1.0, 0.4, 0.5], [0.5, 0.1, 0.2]])
    J, B, K = cr.face_geometry(wrapper, q)
    np.testing.assert_array_equal(J, 1.0 + q[:, 0])
    np.testing.assert_array_equal(K, np.full((2, 3), 9.0))
    interior = q[1:]
    import p06_structured_global.numerics as p06numerics
    for got, want in zip(cr.face_geometry(ref, interior), p06numerics._face_geometry(ref, interior)):
        np.testing.assert_array_equal(got, want)                          # plain reference: the frozen call
