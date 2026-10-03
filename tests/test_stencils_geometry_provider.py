"""Tests for ``drbx.stencils.geometry_arrays`` and its research adapter.

Two independent groups:

* ``GeometryArrays`` save/load round-trip and identity rejection, against a
  small synthetic in-process provider -- no HSX workspace inputs needed.
* Bitwise equality of ``scripts/p_shared/provider.py``'s
  ``ScriptsGeometryProvider`` against direct calls of the frozen functions it
  wraps (P05's inline ``h``/``|J|`` formula, P06's ``curvature_geometry`` and
  ``_face_geometry``, P07's ``_perpendicular_flux_tensor``/
  ``_perpendicular_geometry``, and the shared q1/q3 ``quadrature``
  primitive), on a real N32 sample of raw midpoints and face nodes spanning
  axis, transition, ordinary, and wall regions. Skips cleanly (module-wide)
  if the local HSX workspace inputs the accepted campaigns read from are not
  present, the same way ``p06n_field_derived_global/preflight_fixtures/
  extract.py`` locates them (``work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/
  runtime_inputs`` and ``work/p07n_extraction_hotspot_audit_20260926/
  localized_sidecar.json``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from drbx.stencils.geometry_arrays import GeometryArrays, GeometryProvider, SCHEMA  # noqa: E402

RUNTIME_INPUTS = WORKSPACE / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/runtime_inputs"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32


def _workspace_inputs_available() -> bool:
    if not RUNTIME_INPUTS.is_dir() or not SIDECAR.is_file():
        return False
    geometry_root = RUNTIME_INPUTS / "geometry_artifacts/rlp_convergence_32_48_64_20260917" / f"{N}x{N}x{N}"
    if not (geometry_root / "base_geometry.npz").is_file():
        return False
    try:
        side = json.loads(SIDECAR.read_text())
        for key in ("metric_cache", "makegrid"):
            if not Path(side[key]["path"]).is_file():
                return False
        if not Path(side["artifact"]["path"]).is_dir():
            return False
    except Exception:
        return False
    return True


WORKSPACE_AVAILABLE = _workspace_inputs_available()
needs_workspace = pytest.mark.skipif(
    not WORKSPACE_AVAILABLE,
    reason="HSX workspace geometry inputs (runtime_inputs / localized sidecar / metric cache) are unavailable",
)


# ---------------------------------------------------------------------------
# GeometryArrays save/load + identity, against a small synthetic provider.
# No HSX workspace inputs required.
# ---------------------------------------------------------------------------
class _SyntheticProvider:
    """A tiny, deterministic stand-in satisfying ``GeometryProvider`` for the
    container-level save/load/identity tests, so those do not depend on real
    HSX workspace inputs being present."""

    def p05_metric(self, points):
        points = np.asarray(points, dtype=np.float64)
        h = np.stack([points[:, 0], points[:, 1], points[:, 2]], axis=-1) * 0.5
        jac = 1.0 + points[:, 0]
        return h, jac

    def p06_curvature(self, points):
        points = np.asarray(points, dtype=np.float64)
        J = 2.0 + points[:, 0]
        B = 1.0 + 0.1 * points[:, 1]
        K = np.stack([points[:, 0], -points[:, 1], points[:, 2]], axis=-1)
        return J, B, K

    def p06_face_curvature(self, points):
        return self.p06_curvature(points)

    def p07_perpendicular_tensor(self, points):
        points = np.asarray(points, dtype=np.float64)
        eye = np.eye(3)
        return eye[None, :, :] * (1.0 + points[:, 0])[:, None, None]

    def p07_perpendicular_tensor_and_divergence(self, points):
        tensor = self.p07_perpendicular_tensor(points)
        divergence = np.asarray(points, dtype=np.float64) * 0.25
        return tensor, divergence

    def raw_cell_weight(self, faces, keys):
        keys = np.asarray(keys, dtype=np.int64)
        points = np.zeros((len(keys), 1, 3), dtype=np.float64)
        for axis, face_array in enumerate(faces):
            face_array = np.asarray(face_array, dtype=np.float64)
            lo, hi = face_array[keys[:, axis]], face_array[keys[:, axis] + 1]
            points[:, 0, axis] = 0.5 * (lo + hi)
        weight = np.prod(
            [np.asarray(faces[a])[keys[:, a] + 1] - np.asarray(faces[a])[keys[:, a]] for a in range(3)],
            axis=0,
        ).reshape(-1, 1)
        return points, weight

    def face_node_weight(self, faces, keys):
        keys = np.asarray(keys, dtype=np.int64)
        base_points, base_weight = self.raw_cell_weight(faces, keys)
        points = np.repeat(base_points, 9, axis=1)
        weight = np.repeat(base_weight, 9, axis=1) / 9.0
        return points, weight


def _synthetic_geometry_arrays(n=6, n_raw=10, n_faces=4, seed=1):
    rng = np.random.default_rng(seed)
    faces = tuple(np.sort(rng.uniform(0.0, 1.0, size=n + 1)) for _ in range(3))
    raw_keys = np.column_stack([rng.integers(0, n, n_raw) for _ in range(3)])
    face_keys = np.column_stack(
        [rng.integers(0, 3, n_faces), rng.integers(0, n, n_faces), rng.integers(0, n, n_faces), rng.integers(0, n, n_faces)]
    )
    provider = _SyntheticProvider()
    return GeometryArrays.build(provider, faces=faces, raw_keys=raw_keys, face_keys=face_keys)


def test_provider_protocol_is_structurally_satisfied_by_synthetic_provider():
    assert isinstance(_SyntheticProvider(), GeometryProvider)


def test_geometry_arrays_build_shapes_and_schema():
    arrays = _synthetic_geometry_arrays()
    assert arrays.schema == SCHEMA
    assert arrays.raw_points.shape == (10, 3)
    assert arrays.p05_raw_h.shape == (10, 3)
    assert arrays.p06_raw_K.shape == (10, 3)
    assert arrays.face_points.shape == (4, 9, 3)
    assert arrays.p07_face_tensor.shape == (4, 9, 3, 3)
    arrays.verify()  # does not raise


def test_geometry_arrays_save_load_round_trip_is_bitwise_equal(tmp_path):
    arrays = _synthetic_geometry_arrays()
    path = tmp_path / "geometry.npz"
    arrays.save(path)
    loaded = GeometryArrays.load(path)
    assert loaded.schema == arrays.schema
    assert loaded.identity == arrays.identity
    for name in (
        "raw_points", "p05_raw_h", "p05_raw_jacobian", "p06_raw_J", "p06_raw_B", "p06_raw_K",
        "p06_raw_weight", "p07_raw_tensor", "p07_raw_divergence", "face_points", "p05_face_h",
        "p05_face_jacobian", "p06_face_J", "p06_face_B", "p06_face_K", "p06_face_weight", "p07_face_tensor",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(arrays, name))


def test_geometry_arrays_load_rejects_a_mismatched_expected_identity(tmp_path):
    arrays = _synthetic_geometry_arrays()
    path = tmp_path / "geometry.npz"
    arrays.save(path)
    GeometryArrays.load(path, expected_identity=arrays.identity)  # does not raise
    with pytest.raises(ValueError, match="does not match the expected identity"):
        GeometryArrays.load(path, expected_identity="0" * 64)


def test_geometry_arrays_load_rejects_hand_edited_arrays(tmp_path):
    arrays = _synthetic_geometry_arrays()
    path = tmp_path / "geometry.npz"
    arrays.save(path)
    payload = dict(np.load(path, allow_pickle=False))
    payload["p06_raw_K"] = payload["p06_raw_K"] + 1.0  # corrupt one array in place
    corrupt = tmp_path / "geometry_corrupt.npz"
    np.savez(corrupt, **payload)
    with pytest.raises(ValueError, match="identity does not match its arrays"):
        GeometryArrays.load(corrupt)


def test_geometry_arrays_load_rejects_an_unknown_schema(tmp_path):
    arrays = _synthetic_geometry_arrays()
    path = tmp_path / "geometry.npz"
    arrays.save(path)
    payload = dict(np.load(path, allow_pickle=False))
    payload["schema"] = np.array("drbx.p-stencil-geometry-arrays.v0")
    stale = tmp_path / "geometry_stale_schema.npz"
    np.savez(stale, **payload)
    with pytest.raises(ValueError, match="unsupported geometry-arrays schema"):
        GeometryArrays.load(stale)


def test_geometry_arrays_identity_changes_with_content_but_not_across_rebuilds():
    a = _synthetic_geometry_arrays(seed=1)
    b = _synthetic_geometry_arrays(seed=1)
    c = _synthetic_geometry_arrays(seed=2)
    assert a.identity == b.identity
    assert a.identity != c.identity


# ---------------------------------------------------------------------------
# Bitwise equality against the frozen campaign calls, on a real N32 sample.
# ---------------------------------------------------------------------------
def _bounds(axis, n):
    i_max = n + 1 if axis == 0 else n
    j_max = n + 1 if axis == 1 else n
    k_max = n + 1 if axis == 2 else n
    return i_max, j_max, k_max


def _random_face_keys(rng, axis, count, n, *, i_lo=0, i_hi=None):
    i_max, j_max, k_max = _bounds(axis, n)
    i_hi = i_max if i_hi is None else i_hi
    i = rng.integers(i_lo, i_hi, size=count)
    j = rng.integers(0, j_max, size=count)
    k = rng.integers(0, k_max, size=count)
    return np.column_stack([np.full(count, axis, dtype=np.int64), i, j, k])


@pytest.fixture(scope="module")
def campaign_context():
    from p07_combined_global import kernels as k

    k.configure(RUNTIME_INPUTS)
    return k.load(N)


@pytest.fixture(scope="module")
def frozen_reference():
    from p07_diffusion_global import numerics as refnum

    return refnum.reference(SIDECAR, verify_hashes=False)


@pytest.fixture(scope="module")
def geometry_provider(frozen_reference):
    from p_shared import provider as p_shared_provider

    return p_shared_provider.ScriptsGeometryProvider(frozen_reference)


@pytest.fixture(scope="module")
def sample_keys():
    rng = np.random.default_rng(20260928)
    n = N

    raw_categories = [
        # axis: first radial ring (raw-cell centers there are interior points,
        # not the singular u == 0 face, so the metric is well-defined).
        np.column_stack([np.zeros(20, dtype=np.int64), rng.integers(0, n, 20), rng.integers(0, n, 20)]),
        # wall: last radial ring.
        np.column_stack([np.full(20, n - 1, dtype=np.int64), rng.integers(0, n, 20), rng.integers(0, n, 20)]),
        # transition: an interior radial band away from both edges.
        np.column_stack([np.full(20, n // 3, dtype=np.int64), rng.integers(0, n, 20), rng.integers(0, n, 20)]),
        # ordinary: broad interior coverage.
        np.column_stack([rng.integers(1, n - 1, 150), rng.integers(0, n, 150), rng.integers(0, n, 150)]),
    ]
    raw_keys = np.unique(np.vstack(raw_categories), axis=0)

    face_categories = [
        # axis-ring theta/eta faces (radial cell index 0); true axis
        # (radial-normal, i == 0) faces are singular and are excluded here,
        # exactly as the accepted P06 campaign's own axis_collapsed mask
        # excludes them from _face_geometry.
        _random_face_keys(rng, 1, 15, n, i_lo=0, i_hi=1),
        _random_face_keys(rng, 2, 15, n, i_lo=0, i_hi=1),
        # wall: the outer radial face.
        _random_face_keys(rng, 0, 20, n, i_lo=n, i_hi=n + 1),
        # transition: an interior radial face band.
        _random_face_keys(rng, 0, 20, n, i_lo=n // 3, i_hi=n // 3 + 1),
        # ordinary: broad coverage across all three axes, excluding the
        # collapsed axis=0, i=0 rows.
        _random_face_keys(rng, 0, 30, n, i_lo=1, i_hi=n),
        _random_face_keys(rng, 1, 20, n),
        _random_face_keys(rng, 2, 20, n),
    ]
    face_keys = np.unique(np.vstack(face_categories), axis=0)
    return raw_keys, face_keys


@needs_workspace
@pytest.mark.slow
def test_provider_conforms_to_geometry_provider_protocol(geometry_provider):
    assert isinstance(geometry_provider, GeometryProvider)


@needs_workspace
@pytest.mark.slow
def test_p05_metric_is_bitwise_equal_to_the_direct_campaign_formula(
    campaign_context, frozen_reference, geometry_provider, sample_keys
):
    t = campaign_context
    raw_keys, _face_keys = sample_keys
    raw_ids = np.ravel_multi_index(raw_keys.T, (N, N, N))
    points = t.pts[raw_ids]

    h, jac = geometry_provider.p05_metric(points)

    # The exact two lines scripts/p05_direct_midpoint_global/campaign.py uses
    # at both compute_owner:308-310 and compute_raw_direct:436.
    metric = frozen_reference._metric(points)
    h_direct = metric["bcov"] / metric["B"][:, None]
    jac_direct = np.abs(metric["J"])

    np.testing.assert_array_equal(h, h_direct)
    np.testing.assert_array_equal(jac, jac_direct)


@needs_workspace
@pytest.mark.slow
def test_p07_perpendicular_tensor_is_bitwise_equal_at_face_nodes(
    campaign_context, frozen_reference, geometry_provider, sample_keys
):
    import p06_structured_global.numerics as p06numerics

    t = campaign_context
    _raw_keys, face_keys = sample_keys
    points, _weight = p06numerics.base._face_quadrature(_Shim(t.faces), face_keys)
    flat = points.reshape(-1, 3)

    tensor = geometry_provider.p07_perpendicular_tensor(flat)
    tensor_direct = frozen_reference._perpendicular_flux_tensor(flat)

    np.testing.assert_array_equal(tensor, tensor_direct)


@needs_workspace
@pytest.mark.slow
def test_p07_perpendicular_tensor_and_divergence_is_bitwise_equal_at_raw_midpoints(
    campaign_context, frozen_reference, geometry_provider, sample_keys
):
    t = campaign_context
    raw_keys, _face_keys = sample_keys
    raw_ids = np.ravel_multi_index(raw_keys.T, (N, N, N))
    points = t.pts[raw_ids]

    tensor, divergence = geometry_provider.p07_perpendicular_tensor_and_divergence(points)
    tensor_direct, divergence_direct = frozen_reference._perpendicular_geometry(points)

    np.testing.assert_array_equal(tensor, tensor_direct)
    np.testing.assert_array_equal(divergence, divergence_direct)


class _Shim:
    """The same ``x_faces``/``y_faces``/``z_faces`` adapter
    ``p06n_field_derived_global.core._cell_geometry_shim`` uses to hand the
    accepted P06 quadrature helpers a plain-attribute context."""

    def __init__(self, faces):
        self.x_faces, self.y_faces, self.z_faces = faces


@needs_workspace
@pytest.mark.slow
def test_raw_cell_weight_matches_p06s_own_q1_quadrature_call(
    campaign_context, geometry_provider, sample_keys
):
    import p06_structured_global.numerics as p06numerics

    t = campaign_context
    raw_keys, _face_keys = sample_keys
    shim = _Shim(t.faces)

    points, weight = geometry_provider.raw_cell_weight(t.faces, raw_keys)
    points_direct, weight_direct = p06numerics._cell_quadrature(shim, raw_keys, 1)

    np.testing.assert_array_equal(points, points_direct)
    np.testing.assert_array_equal(weight, weight_direct)

    # And the raw q1 node is exactly the raw-cell center used everywhere else.
    raw_ids = np.ravel_multi_index(raw_keys.T, (N, N, N))
    np.testing.assert_array_equal(points.reshape(-1, 3), t.pts[raw_ids])


@needs_workspace
@pytest.mark.slow
def test_face_node_weight_matches_p06s_own_q3_quadrature_call(
    campaign_context, geometry_provider, sample_keys
):
    import p06_structured_global.numerics as p06numerics

    t = campaign_context
    _raw_keys, face_keys = sample_keys
    shim = _Shim(t.faces)

    points, weight = geometry_provider.face_node_weight(t.faces, face_keys)
    points_direct, weight_direct = p06numerics.base._face_quadrature(shim, face_keys)

    np.testing.assert_array_equal(points, points_direct)
    np.testing.assert_array_equal(weight, weight_direct)


@needs_workspace
@pytest.mark.slow
def test_geometry_arrays_from_real_sample_saves_and_loads_bitwise_equal(
    tmp_path, campaign_context, geometry_provider, sample_keys
):
    t = campaign_context
    raw_keys, face_keys = sample_keys
    arrays = GeometryArrays.build(geometry_provider, faces=t.faces, raw_keys=raw_keys, face_keys=face_keys)
    path = tmp_path / "N32_geometry_sample.npz"
    arrays.save(path)
    loaded = GeometryArrays.load(path, expected_identity=arrays.identity)
    np.testing.assert_array_equal(loaded.p06_raw_K, arrays.p06_raw_K)
    np.testing.assert_array_equal(loaded.p07_face_tensor, arrays.p07_face_tensor)
