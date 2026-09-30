"""Autodiff curvature K on the real HSX N32 geometry (K1 of the P08 bundle; slow).

On ~200 real raw midpoints and real wall q3 nodes:

* autodiff ``K`` equals the prototype formula (``work/p08_autodiff_curvature_20260928/compare.py``:
  ``jax.vmap(jax.jacfwd)`` of ``b_cov/B``) to roundoff, and the frozen 4th-order finite-difference ``K``
  to a median of ~3e-11 (a few 1e-6 at worst points, where the finite difference is the inaccurate side);
* nodes at ``u = 1`` evaluate finitely, and ``ScriptsGeometryProvider(curvature="autodiff")`` returns ``J``/``B``
  bitwise as the frozen ``_face_geometry`` and autodiff ``K`` at every node including the wall;
* the default ``"fd"`` provider is bitwise the frozen calls;
* ``d_i (|J| K^i / B)`` is at roundoff, and the result is bitwise independent of batching;
* near the axis autodiff stays finite for ``u > 0`` (K grows like ``1/u``) and is NaN at ``u = 0``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32

pytestmark = pytest.mark.slow

_STATE: dict = {}


def _state() -> dict:
    if not _STATE:
        directory = GEOMETRY / f"{N}x{N}x{N}"
        if not ((directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()
                and SIDECAR.is_file()):
            pytest.skip(f"HSX N{N} geometry/sidecar inputs are unavailable")
        try:
            json.loads(SIDECAR.read_text())
            from perpendicular_structured.reconstruction import load_context
            from p_shared.provider import ScriptsGeometryProvider

            t = load_context(N, str(WORKSPACE))
            ad = ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="autodiff")
            fd = ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False, curvature="fd")
        except (FileNotFoundError, OSError) as error:
            pytest.skip(f"HSX inputs unavailable: {error}")
        from p07_diffusion_global.numerics import quadrature

        rng = np.random.default_rng(0)
        raw = np.asarray(t.pts)[rng.choice(len(t.pts), 200, replace=False)]
        # real wall q3 nodes: radial faces at i = n (u = 1), and an interior face batch of the same size
        wall_keys = np.array([[0, N, j, k] for j, k in rng.integers(0, N, size=(10, 2))], dtype=np.int64)
        wall = quadrature(t.faces, wall_keys, 3, face=True)[0].reshape(-1, 3)
        inner_keys = np.array([[a, *rng.integers(1, N, size=3)] for a in rng.integers(0, 3, size=10)], dtype=np.int64)
        inner = quadrature(t.faces, inner_keys, 3, face=True)[0].reshape(-1, 3)
        _STATE.update(t=t, ad=ad, fd=fd, raw=raw, wall=wall, inner=inner)
    return _STATE


def _prototype_K(ref, points):
    """The prototype's formula, verbatim: vmap(jacfwd) of b_cov/B through the JAX evaluators."""
    import jax
    import jax.numpy as jnp
    from drbx.geometry.jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
    from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator

    jm = JaxMetricEvaluator.from_metric_evaluator(ref.metric_evaluator)
    jb = JaxComponentSplineBFieldEvaluator.from_evaluator(ref.bfield_evaluator)
    B0 = float(ref.B0)

    def A_and_parts(q):
        m = jm.evaluate(q[None], reject_nonpositive_J=False)
        Bc = jb.evaluate_cartesian(m.position)
        Bcontra = jnp.linalg.solve(m.jacobian_matrix, Bc[..., None])[..., 0] / B0
        bmag = jnp.maximum(jnp.linalg.norm(Bc, axis=-1) / B0, 1e-30)
        bunit = Bcontra / bmag[..., None]
        bcov = jnp.einsum("...ij,...j->...i", m.g_cov, bunit)
        return (bcov / bmag[..., None])[0], bmag[0], m.J[0]

    A_only = lambda q: A_and_parts(q)[0]  # noqa: E731

    @jax.jit
    def K_ad(qs):
        def one(q):
            d = jax.jacfwd(A_only)(q)
            _, bmag, J = A_and_parts(q)
            curl = jnp.stack((d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]))
            return 0.5 * bmag * curl / jnp.maximum(jnp.abs(J), 1e-30)
        return jax.vmap(one)(qs)

    return np.asarray(K_ad(jnp.asarray(points)))


def _rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)


def test_autodiff_k_matches_prototype_and_finite_difference():
    s = _state()
    ref_ad = s["ad"].reference
    K = ref_ad._curvature(s["raw"])
    proto = _prototype_K(ref_ad.wrapped, s["raw"])
    assert np.max(np.abs(K - proto)) <= 1.0e-9 * np.max(np.abs(proto))        # roundoff (vmap vs sequential program)
    K_fd = ref_ad.wrapped._curvature(s["raw"])
    rel = _rel(K, K_fd)
    assert np.median(rel) < 5.0e-10 and np.quantile(rel, 0.99) < 1.0e-5 and rel.max() < 1.0e-3


def test_wall_nodes_evaluate_and_provider_face_path():
    import p06_structured_global.numerics as p06numerics

    s = _state()
    ad, fd = s["ad"], s["fd"]
    q = np.concatenate([s["wall"], s["inner"]])
    assert np.all(np.isclose(s["wall"][:, 0], 1.0, rtol=0.0, atol=8 * np.finfo(float).eps))
    J, B, K = ad.p06_face_curvature(q)
    assert np.all(np.isfinite(K)) and K.shape == (len(q), 3)
    J_fd, B_fd, K_fd = p06numerics._face_geometry(fd.reference, q)            # the frozen call, plain reference
    np.testing.assert_array_equal(J, J_fd)                                    # J, B exactly as _face_geometry
    np.testing.assert_array_equal(B, B_fd)
    np.testing.assert_array_equal(K, ad.reference._curvature(q))              # autodiff at every node
    n_wall = len(s["wall"])
    rel = _rel(K, K_fd)
    assert np.median(rel[n_wall:]) < 5.0e-10                                  # interior: as the raw points
    assert np.max(rel[:n_wall]) < 1.0e-3                                      # wall: the frozen one-sided rule is the coarse side


def test_default_provider_is_bitwise_the_frozen_calls():
    import p06_structured_global.numerics as p06numerics
    from perpendicular_structured.reference_geometry import curvature_geometry

    s = _state()
    fd = s["fd"]
    assert fd.curvature == "fd" and not hasattr(fd.reference, "wrapped")
    q = np.concatenate([s["wall"], s["inner"]])
    for got, want in zip(fd.p06_face_curvature(q), p06numerics._face_geometry(fd.reference, q)):
        np.testing.assert_array_equal(got, want)
    prepared = curvature_geometry(fd.reference, s["raw"])
    for got, want in zip(fd.p06_curvature(s["raw"]), (prepared.J, prepared.B, prepared.K)):
        np.testing.assert_array_equal(got, want)
    J, B, K = s["ad"].p06_curvature(s["raw"])                                 # raw path: the same call, autodiff K
    np.testing.assert_array_equal(J, prepared.J)
    np.testing.assert_array_equal(B, prepared.B)
    np.testing.assert_array_equal(K, s["ad"].reference._curvature(s["raw"]))
    prepared_ad = curvature_geometry(s["ad"].reference, s["raw"])
    np.testing.assert_array_equal(prepared_ad.K, K)


def test_divergence_identity_and_batch_independence():
    s = _state()
    ref_ad = s["ad"].reference
    kernel = ref_ad.autodiff()
    q = np.concatenate([s["raw"], s["wall"]])
    terms = kernel.divergence_identity_terms(q)
    div = kernel.divergence_identity(q)
    scale = np.abs(terms).sum(axis=-1)
    assert np.all(np.abs(div) <= 1.0e-9 * scale)
    K = kernel(q)
    np.testing.assert_array_equal(np.concatenate([kernel(q[i:i + 37]) for i in range(0, len(q), 37)]), K)
    np.testing.assert_array_equal(kernel(q[::-1])[::-1], K)
    np.testing.assert_array_equal(kernel(q[5:6]), K[5:6])


def test_axis_and_wall_behaviour():
    s = _state()
    ref_ad = s["ad"].reference
    base = s["raw"][:20].copy()
    wall = base.copy(); wall[:, 0] = 1.0
    assert np.all(np.isfinite(ref_ad._curvature(wall)))
    near = base.copy(); near[:, 0] = 1.0e-6
    K = ref_ad._curvature(near)
    assert np.all(np.isfinite(K))
    assert np.median(np.linalg.norm(K, axis=1)) > 1.0e5                        # ~1/u growth of the contravariant K
    assert np.median(_rel(K, ref_ad.wrapped._curvature(near))) < 1.0e-6         # the finite difference still agrees
    axis = base.copy(); axis[:, 0] = 0.0
    assert not np.any(np.isfinite(ref_ad._curvature(axis)))                     # the singular axis itself is NaN
