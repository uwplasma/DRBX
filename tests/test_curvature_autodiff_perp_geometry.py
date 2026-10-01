"""Autodiff perpendicular flux tensor and divergence vs the finite-difference reference on the real N32 geometry (slow).

``AutodiffPerpendicularGeometry`` returns ``(J (g^ij - b^i b^j), sum_i d_i [J (g^ij - b^i b^j)])`` by ``jax.jacfwd``.
The tensor must equal the FD reference tensor (same metric/B pipeline, no derivative) to <= 1e-12 relative; the
divergence must match the FD divergence evaluated with a small, converged step (5e-5 / 2e-5, the closer is taken) to
<= 1e-7 relative, at random interior points and at wall-near points (u = 0.99), and must not depend on the batching.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
N = 32
TENSOR_TOL = 1e-12
DIV_TOL = 1e-7
REPORT: dict = {}

pytestmark = pytest.mark.slow

try:
    from p_shared.replay_support import DEFAULT_SIDECAR as SIDECAR
except Exception:  # pragma: no cover - scripts tree unavailable
    SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and Path(SIDECAR).is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def ref():
    from p_shared.replay_support import build_environment

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="autodiff",
                            face_quadrature="q2", inner_support="fixed_radius")
    return env.ref


def _points(fd, u, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([u, rng.uniform(0.0, 2.0 * np.pi, len(u)), rng.uniform(0.0, fd.eta_period, len(u))])


@pytest.fixture(scope="module")
def interior(ref):
    return _points(ref.wrapped, np.random.default_rng(1).uniform(0.05, 0.95, 200), 2)


@pytest.fixture(scope="module")
def wall(ref):
    return _points(ref.wrapped, np.full(40, 0.99), 3)


def _fd_divergence_converged(fd, q, autodiff_div):
    """FD divergence at the converged steps; returns the one closest to ``autodiff_div`` (and the step)."""
    h0 = fd.finite_difference_step
    best = None
    try:
        for h in (5e-5, 2e-5):
            fd.finite_difference_step = h
            _, div = fd._perpendicular_geometry(q)
            err = np.linalg.norm(div - autodiff_div) / np.linalg.norm(div)
            if best is None or err < best[0]:
                best = (err, h, div)
    finally:
        fd.finite_difference_step = h0
    return best[2], best[1]


def _per_point_rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


@needs_inputs
@pytest.mark.parametrize("name", ["interior", "wall"])
def test_tensor_and_divergence_match_finite_differences(ref, interior, wall, name):
    from p_shared.curvature_reference import perpendicular_geometry

    q = {"interior": interior, "wall": wall}[name]
    fd = ref.wrapped
    tensor, div = perpendicular_geometry(ref, q, "autodiff")
    assert tensor.shape == (len(q), 3, 3) and div.shape == (len(q), 3)
    fd_tensor, fd_div_default = perpendicular_geometry(ref, q, "fd")
    fd_tensor_plain, _ = perpendicular_geometry(fd, q, "fd")                # plain reference works too
    np.testing.assert_array_equal(fd_tensor, fd_tensor_plain)
    tensor_rel = float(np.max(np.abs(tensor - fd_tensor)) / np.max(np.abs(fd_tensor)))
    fd_div, h = _fd_divergence_converged(fd, q, div)
    div_rel = float(_per_point_rel(div, fd_div).max())
    div_scale_rel = float(np.max(np.abs(div - fd_div)) / np.max(np.abs(fd_div)))
    default_rel = float(_per_point_rel(div, fd_div_default).max())
    REPORT[name] = dict(tensor_rel=tensor_rel, div_per_point_rel=div_rel, div_global_rel=div_scale_rel,
                        best_step=h, vs_default_step=default_rel)
    print(f"\nautodiff perpendicular geometry vs FD [{name}]: {REPORT[name]}")
    assert tensor_rel <= TENSOR_TOL
    assert div_rel <= DIV_TOL
    # the FD reference at its default step (2e-4) is only ~1e-7..1e-8 accurate; autodiff is the more accurate one
    assert default_rel <= 1e-6


@needs_inputs
def test_sign_convention_matches_reference_J(ref, interior):
    fd = ref.wrapped
    q = interior[:16]
    tensor, _ = ref.perpendicular_geometry_autodiff(q)
    metric = fd._metric(q)
    expected = metric["J"][:, None, None] * (metric["gcontra"] - np.einsum("ni,nj->nij", metric["b"], metric["b"]))
    np.testing.assert_allclose(tensor, expected, rtol=1e-12, atol=1e-13 * np.abs(expected).max())
    np.testing.assert_allclose(tensor, tensor.transpose(0, 2, 1), rtol=1e-12, atol=1e-13 * np.abs(tensor).max())


@needs_inputs
def test_block_mode_is_chunk_batch_and_order_independent(ref, interior, wall):
    geometry = ref.autodiff_perpendicular()
    assert geometry.mode == "block" and ref.autodiff_perpendicular() is geometry          # lazily built once
    q = np.concatenate([interior, wall])                                                   # 240 points (< block = 256)
    tensor, div = geometry(q)
    # reversed / permuted order
    perm = np.random.default_rng(4).permutation(len(q))
    t_p, d_p = geometry(q[perm])
    np.testing.assert_array_equal(t_p, tensor[perm])
    np.testing.assert_array_equal(d_p, div[perm])
    # chunked by the caller (including a single point and a ragged tail)
    for size in (1, 37, 100):
        chunks = [geometry(q[s:s + size]) for s in range(0, len(q), size)]
        np.testing.assert_array_equal(np.concatenate([c[0] for c in chunks]), tensor)
        np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), div)
    # more points than one block (internal chunking) and the same points embedded among other points
    big = np.concatenate([q, q[:100], interior[:50]])
    t_b, d_b = geometry(big)
    assert t_b.shape == (len(big), 3, 3)
    np.testing.assert_array_equal(t_b[:len(q)], tensor)
    np.testing.assert_array_equal(d_b[:len(q)], div)
    t0, d0 = geometry(np.zeros((0, 3)))
    assert t0.shape == (0, 3, 3) and d0.shape == (0, 3)


@needs_inputs
def test_sequential_mode_agrees_with_block(ref, interior):
    from drbx.geometry.curvature_autodiff import autodiff_perpendicular_geometry
    from drbx.geometry.jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
    from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator

    fd = ref.wrapped
    jm = JaxMetricEvaluator.from_metric_evaluator(fd.metric_evaluator)
    jb = JaxComponentSplineBFieldEvaluator.from_evaluator(fd.bfield_evaluator)
    seq = autodiff_perpendicular_geometry(jm, jb, float(fd.B0), mode="sequential", block=16)
    q = interior[:16]
    t_s, d_s = seq(q)
    t_b, d_b = ref.perpendicular_geometry_autodiff(q)
    np.testing.assert_allclose(t_s, t_b, rtol=1e-12, atol=1e-13 * np.abs(t_b).max())
    assert float(_per_point_rel(d_s, d_b).max()) < 1e-9
    with pytest.raises(ValueError):
        autodiff_perpendicular_geometry(jm, jb, float(fd.B0), mode="bogus")
    with pytest.raises(ValueError):
        autodiff_perpendicular_geometry(jm, jb, float(fd.B0), block=0)


@needs_inputs
def test_perpendicular_geometry_dispatch_and_fd_is_not_overridden(ref, interior):
    from p_shared.curvature_reference import AutodiffCurvatureReference, perpendicular_geometry

    q = interior[:8]
    # the wrapper's FD geometry is the frozen reference's (artifact builds keep their FD p07_raw_divergence)
    t_w, d_w = ref._perpendicular_geometry(q)
    t_f, d_f = ref.wrapped._perpendicular_geometry(q)
    np.testing.assert_array_equal(t_w, t_f)
    np.testing.assert_array_equal(d_w, d_f)
    # a plain reference is wrapped for method="autodiff"
    t_a, d_a = perpendicular_geometry(ref.wrapped, q, "autodiff")
    t_r, d_r = ref.perpendicular_geometry_autodiff(q)
    np.testing.assert_allclose(t_a, t_r, rtol=0, atol=0)
    np.testing.assert_allclose(d_a, d_r, rtol=0, atol=0)
    assert not isinstance(ref.wrapped, AutodiffCurvatureReference)
    with pytest.raises(ValueError, match="method"):
        perpendicular_geometry(ref, q, "bogus")
