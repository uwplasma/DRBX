"""CSR pack/unpack and identity rejection for the P08 step-1 row artifact.

Covers task 3 of ``work/p08_step1_consolidation_design_20260928/design.md``:
``drbx.stencils.artifact`` packs the host row objects (``PointRows``,
``NeumannPointRows``, ``IntegratedFaceRow``) into a per-target CSR, saves a
chunked ``row_artifact/N{n}/`` directory with a manifest identity and
per-chunk sha256, and ``expand()``s the CSR back into the same row objects,
bitwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_reconstruction import (
    PointRowContext, PointRows, StructuredReconstruction)
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
from drbx.geometry.fci_perpendicular_integrated_rows import (
    IntegratedFaceRow, prepare_integrated_face_rows)
from drbx.native.fci_perpendicular_point_rows import load_point_row_plan

from drbx.stencils.artifact import (
    _common_dtype, build_identity, expand_integrated_rows, expand_neumann_rows, expand_point_rows,
    hash_bytes, hash_file, hash_source, load_row_artifact, pack_integrated_rows,
    pack_neumann_rows, pack_point_rows, save_row_artifact,
)
import drbx.stencils.artifact as artifact_module

POINT_ROW_DATA = Path(__file__).parent / "data/p_shared_point_rows"
FACE_ROW_DATA = Path(__file__).parent / "data/p_shared_face_rows"


# --------------------------------------------------------------------------
# Toy geometry (mirrors the pattern used by the existing point/neumann tests)
# --------------------------------------------------------------------------

def _toy_context(n=12):
    faces = (np.linspace(0, 1, n + 1), np.linspace(0, 2 * np.pi, n + 1), np.linspace(0, 2 * np.pi, n + 1))
    centers = tuple((axis[:-1] + axis[1:]) / 2 for axis in faces)
    ijk = np.array(np.unravel_index(np.arange(n ** 3), (n, n, n))).T
    pts = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
    centroids = np.column_stack((pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1])))
    return PointRowContext.from_arrays(faces=faces, centers=centers, raw_to_owner=np.arange(n ** 3),
        raw_volume=np.ones(n ** 3), owner_volume=np.ones(n ** 3), owner_centroid_xy=centroids,
        eta_period=2 * np.pi, dr=1 / n, dtheta=2 * np.pi / n, deta=2 * np.pi / n)


def _assert_point_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.gradient, expected.gradient)
    assert actual.boundary_conditioned == expected.boundary_conditioned
    np.testing.assert_array_equal(actual.trace_donor_points, expected.trace_donor_points)
    np.testing.assert_array_equal(actual.trace_target_points, expected.trace_target_points)
    assert actual.diagnostics == expected.diagnostics


def _assert_neumann_chunk_tags(chunk, *, request, radial_degree):
    np.testing.assert_array_equal(chunk.request, np.array(request, dtype="<U8"))
    np.testing.assert_array_equal(chunk.radial_degree, np.array(radial_degree, dtype=np.int8))


def _assert_neumann_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.gradient, expected.gradient)
    np.testing.assert_array_equal(actual.boundary_points, expected.boundary_points)
    np.testing.assert_array_equal(actual.boundary_value, expected.boundary_value)
    np.testing.assert_array_equal(actual.boundary_gradient, expected.boundary_gradient)
    assert actual.condition == expected.condition
    assert actual.constraint_residual == expected.constraint_residual


def _assert_integrated_rows_equal(actual, expected):
    np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
    np.testing.assert_array_equal(actual.weights, expected.weights)
    assert actual.boundary_conditioned == expected.boundary_conditioned
    np.testing.assert_array_equal(actual.trace_donor_points, expected.trace_donor_points)
    np.testing.assert_array_equal(actual.trace_target_points, expected.trace_target_points)
    np.testing.assert_array_equal(actual.value_loading, expected.value_loading)
    np.testing.assert_array_equal(actual.tangential_loading, expected.tangential_loading)
    assert actual.family == expected.family


# --------------------------------------------------------------------------
# PointRows: fresh, toy-built rows (R1 cell / R2 face common / R3 face side)
# --------------------------------------------------------------------------

def _toy_point_rows(n=12):
    """One of each family the host builder produces, with real toy geometry."""
    context = _toy_context(n)
    builder = StructuredReconstruction(context)
    rows, request, entity_id, bc_variant, radial_degree = [], [], [], [], []

    p = context.pts[np.ravel_multi_index((5, 1, 0), (n, n, n))][None, :]
    rows.append(builder.rows((5, 1, 0), p, location="cell"))
    request.append("R1"); entity_id.append(510); bc_variant.append(""); radial_degree.append(0)

    p = context.pts[np.ravel_multi_index((n - 1, 1, 0), (n, n, n))][None, :]
    rows.append(builder.rows((n - 1, 1, 0), p, location="cell"))
    request.append("R1"); entity_id.append(911); bc_variant.append("D"); radial_degree.append(3)

    p = np.stack([context.pts[0], context.pts[1]])
    rows.append(builder.rows((0, 0, 3, 0), p, location="face"))  # collapsed_r0: zero donors
    request.append("R2"); entity_id.append(3); bc_variant.append(""); radial_degree.append(0)

    p = context.pts[np.ravel_multi_index((5, 3, 0), (n, n, n))][None, :]
    rows.append(builder.rows((1, 3, 5, 0), p, location="face"))
    request.append("R2"); entity_id.append(135); bc_variant.append(""); radial_degree.append(0)

    p = context.pts[np.ravel_multi_index((n - 1, 3, 0), (n, n, n))][None, :]
    rows.append(builder.rows((0, n - 1, 3, 0), p, location="face"))
    request.append("R2"); entity_id.append(731); bc_variant.append("D"); radial_degree.append(4)

    p = context.pts[np.ravel_multi_index((5, 1, 0), (n, n, n))][None, :]
    lower, upper = builder.side_rows((0, 5, 1, 0), p)
    rows.append(lower); request.append("R3"); entity_id.append(51); bc_variant.append(""); radial_degree.append(3)
    rows.append(upper); request.append("R3"); entity_id.append(52); bc_variant.append(""); radial_degree.append(3)

    return context, rows, request, entity_id, bc_variant, radial_degree


def test_point_row_chunk_roundtrip_toy():
    _, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    families = {row.diagnostics.get("family") for row in rows}
    assert families == {"singleton", "boundary_transverse", "collapsed_r0", "quartic_wall"}
    assert any(row.boundary_conditioned for row in rows)
    assert any(len(row.donor_ids) == 0 for row in rows)

    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    assert chunk.donor.dtype == np.int32
    expanded = expand_point_rows(chunk)
    assert len(expanded) == len(rows)
    for actual, expected in zip(expanded, rows, strict=True):
        _assert_point_rows_equal(actual, expected)
    total_points = sum(len(row.trace_target_points) for row in rows)
    assert len(chunk.entity_id) == len(chunk.request) == len(chunk.quad_node) == total_points
    # every target row within one source shares that source's entity id/request tag
    expected_entity = np.repeat(np.asarray(entity_id, dtype=np.int64),
                                [len(row.trace_target_points) for row in rows])
    np.testing.assert_array_equal(chunk.entity_id, expected_entity)


def test_point_row_dropped_gradient_is_a_documented_empty_sentinel():
    context, rows, *_ = _toy_point_rows()
    side = rows[-1]  # an R3 face-side row
    chunk = pack_point_rows([side], request="R3", entity_id=52, store_gradient=False)
    assert not bool(chunk.has_gradient[0])
    expanded = expand_point_rows(chunk)[0]
    np.testing.assert_array_equal(expanded.donor_ids, side.donor_ids)
    np.testing.assert_array_equal(expanded.value, side.value)
    assert expanded.gradient.shape == (side.value.shape[0], 3, 0)


def test_point_row_artifact_save_load_roundtrip(tmp_path):
    _, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    identity = build_identity(component_hashes={"geometry": "abc123"},
                              source_hashes={"reconstruction": "def456"},
                              policy={"svd": 1e-4, "residual": 1e-9})
    save_row_artifact(tmp_path, 32, identity=identity, cells=[chunk])
    loaded = load_row_artifact(tmp_path, 32, identity)
    assert loaded.n == 32
    assert len(loaded.cells) == 1 and loaded.faces == () and loaded.neumann == () and loaded.p07 == ()
    for actual, expected in zip(expand_point_rows(loaded.cells[0]), rows, strict=True):
        _assert_point_rows_equal(actual, expected)


# --------------------------------------------------------------------------
# PointRows: real-HSX fixture round trip (tests/data/p_shared_point_rows)
# --------------------------------------------------------------------------

def _point_rows_from_fixture(plan):
    """One PointRows per already-lowered target, kept at its full padded width.

    No attempt is made to trim the padding a bucket may carry beyond a
    row's true donor count: an untrimmed (donor id 0, weight 0) tail is
    still a perfectly valid row to pack, and using it verbatim removes any
    guesswork about where the real data ends — the point of this test is
    that our own CSR reproduces whatever it was given, bitwise.
    """
    rows, entity_id, bc_variant = [], [], []
    for batch in plan.payload.batches:
        for q in range(len(batch.target_ids)):
            donor = batch.donor_ids[q].astype(np.int64)
            value = batch.value_weights[q][None, :]
            gradient = batch.gradient_weights[q][None, :, :]
            conditioned = bool(batch.conditioned[q])
            target_id = int(batch.target_ids[q])
            trace_target_points = plan.target_points[target_id][None, :]
            if conditioned:
                trace_donor_points = plan.boundary_points[batch.donor_query_ids[q]]
            else:
                trace_donor_points = np.empty((0, 3))
            diagnostics = dict(plan.row_metadata[target_id])
            rows.append(PointRows(donor, value, gradient, conditioned,
                                  trace_donor_points, trace_target_points, diagnostics))
            entity_id.append(target_id)
            bc_variant.append("D" if conditioned else "")
    return rows, entity_id, bc_variant


@pytest.mark.parametrize("n", (32, 48, 64))
def test_point_rows_real_hsx_fixture_roundtrip(n):
    with np.load(POINT_ROW_DATA / f"N{n}.plan.npz", allow_pickle=False) as source:
        identity = json.loads(str(source["metadata_json"].item()))["identity"]
    plan = load_point_row_plan(POINT_ROW_DATA / f"N{n}.plan.npz", identity)
    rows, entity_id, bc_variant = _point_rows_from_fixture(plan)
    assert len(rows) == 34

    chunk = pack_point_rows(rows, request="R1", entity_id=entity_id, bc_variant=bc_variant)
    expanded = expand_point_rows(chunk)
    for actual, expected in zip(expanded, rows, strict=True):
        _assert_point_rows_equal(actual, expected)


# --------------------------------------------------------------------------
# NeumannPointRows: fresh, toy-built wall-lattice rows
# --------------------------------------------------------------------------

def test_neumann_row_chunk_roundtrip_toy():
    n = 12
    context = _toy_context(n)
    centers = context.centers
    coeff = np.array((np.sqrt(1.04), -.2 / np.sqrt(1.04), 0.))
    points = np.array([[.98, centers[1][5], centers[2][5]],
                       [.97, centers[1][6], centers[2][2]]])
    rows = prepare_neumann_point_rows(context, points,
        normal_coefficients=lambda q: np.broadcast_to(coeff, (len(q), 3)))
    assert len(rows) == 2

    chunk = pack_neumann_rows(rows, entity_id=[0, 1], quad_node=[0, 1],
                              request=["R2", "R3"], radial_degree=[4, 3])
    assert chunk.donor.dtype == np.int32
    assert chunk.request.dtype == np.dtype("<U8")
    assert chunk.radial_degree.dtype == np.int8
    # boundary_value/boundary_gradient keep their native (builder) dtype,
    # untouched by the request/radial_degree tags added alongside them.
    assert chunk.boundary_value.dtype == _common_dtype([r.boundary_value for r in rows])
    assert chunk.boundary_gradient.dtype == _common_dtype([r.boundary_gradient for r in rows])
    _assert_neumann_chunk_tags(chunk, request=["R2", "R3"], radial_degree=[4, 3])
    expanded = expand_neumann_rows(chunk)
    for actual, expected in zip(expanded, rows, strict=True):
        _assert_neumann_rows_equal(actual, expected)


def test_neumann_row_chunk_scalar_request_and_degree_broadcast():
    n = 12
    context = _toy_context(n)
    centers = context.centers
    coeff = np.array((1., 0., 0.))
    points = np.array([[.99, centers[1][4], centers[2][4]],
                       [.98, centers[1][5], centers[2][5]]])
    rows = prepare_neumann_point_rows(context, points,
        normal_coefficients=lambda q: np.broadcast_to(coeff, (len(q), 3)))
    chunk = pack_neumann_rows(rows, entity_id=[3, 4], request="R4", radial_degree=3)
    _assert_neumann_chunk_tags(chunk, request=["R4", "R4"], radial_degree=[3, 3])


def test_neumann_row_chunk_request_and_radial_degree_are_required():
    n = 12
    context = _toy_context(n)
    centers = context.centers
    coeff = np.array((1., 0., 0.))
    points = np.array([[.99, centers[1][4], centers[2][4]]])
    rows = prepare_neumann_point_rows(context, points,
        normal_coefficients=lambda q: np.broadcast_to(coeff, (len(q), 3)))
    with pytest.raises(TypeError):
        pack_neumann_rows(rows, entity_id=[0])
    with pytest.raises(TypeError):
        pack_neumann_rows(rows, entity_id=[0], request=["R1"])
    with pytest.raises(TypeError):
        pack_neumann_rows(rows, entity_id=[0], radial_degree=[3])
    with pytest.raises(ValueError, match="unsupported Neumann source request"):
        pack_neumann_rows(rows, entity_id=[0], request=["neumann"], radial_degree=[3])


def test_neumann_row_artifact_save_load_roundtrip(tmp_path):
    n = 12
    context = _toy_context(n)
    centers = context.centers
    coeff = np.array((1., 0., 0.))
    points = np.array([[.99, centers[1][4], centers[2][4]]])
    rows = prepare_neumann_point_rows(context, points,
        normal_coefficients=lambda q: np.broadcast_to(coeff, (len(q), 3)))
    chunk = pack_neumann_rows(rows, entity_id=[7], request=["R1"], radial_degree=[3])
    identity = build_identity(component_hashes={}, source_hashes={}, policy={"max_condition": 1e8})
    save_row_artifact(tmp_path, 16, identity=identity, neumann=[chunk])
    loaded = load_row_artifact(tmp_path, 16, identity)
    _assert_neumann_rows_equal(expand_neumann_rows(loaded.neumann[0])[0], rows[0])
    _assert_neumann_chunk_tags(loaded.neumann[0], request=["R1"], radial_degree=[3])


# --------------------------------------------------------------------------
# IntegratedFaceRow: fresh, toy-built P07 rows (families 0 / 1-4 / 5)
# --------------------------------------------------------------------------

def _toy_integrated_rows(n=12):
    context = _toy_context(n)

    def q3(i, j, k):
        base = context.pts[np.ravel_multi_index((i % n, j % n, k % n), (n, n, n))]
        return np.tile(base, (9, 1))

    keys = np.array([(0, 0, 3, 0), (0, n - 1, 3, 0), (1, 3, 5, 0)])
    points = np.array([q3(0, 3, 0), q3(n - 1, 3, 0), q3(5, 3, 0)])
    family = np.array([0, 1, 5])
    integrand = np.ones((3, 9, 3))
    rows = prepare_integrated_face_rows(context, keys, family, points, integrand)
    return context, rows


def test_integrated_row_chunk_roundtrip_toy():
    _, rows = _toy_integrated_rows()
    families = {row.family for row in rows}
    assert families == {0, 1, 5}
    assert any(row.boundary_conditioned for row in rows)
    assert any(len(row.donor_ids) == 0 for row in rows)

    chunk = pack_integrated_rows(rows, entity_id=[100, 101, 102])
    assert chunk.donor.dtype == np.int32
    expanded = expand_integrated_rows(chunk)
    for actual, expected in zip(expanded, rows, strict=True):
        _assert_integrated_rows_equal(actual, expected)


def test_integrated_row_artifact_save_load_roundtrip(tmp_path):
    _, rows = _toy_integrated_rows()
    chunk = pack_integrated_rows(rows, entity_id=[1, 2, 3])
    identity = build_identity(component_hashes={"topology": "xyz"}, source_hashes={},
                              policy={"quadrature": "q3"})
    save_row_artifact(tmp_path, 8, identity=identity, p07=[chunk])
    loaded = load_row_artifact(tmp_path, 8, identity)
    for actual, expected in zip(expand_integrated_rows(loaded.p07[0]), rows, strict=True):
        _assert_integrated_rows_equal(actual, expected)


# --------------------------------------------------------------------------
# IntegratedFaceRow: real-HSX fixture round trip (tests/data/p_shared_face_rows)
# --------------------------------------------------------------------------

def _integrated_rows_from_fixture(arrays):
    """Real donor ids/weights/tangential loadings; the coordinates this
    already-lowered fixture never carried (``IntegratedFaceBatch`` has no
    point-coordinate field) are deterministic placeholders. The CSR pack
    code is agnostic to what the floats mean, so this still exercises the
    query-table/CSR mechanics faithfully on real donor/weight data.
    """
    rows, entity_id = [], []
    family_table = arrays["family"]
    for b in range(int(arrays["batch_count"])):
        face_ids = arrays[f"batch_{b}_face_ids"]
        donor_ids = arrays[f"batch_{b}_donor_ids"]
        weights = arrays[f"batch_{b}_weights"]
        tangential_weights = arrays[f"batch_{b}_tangential_weights"]
        conditioned = arrays[f"batch_{b}_conditioned"]
        for q, face in enumerate(face_ids):
            face = int(face)
            donor = donor_ids[q].astype(np.int64)
            weight = weights[q].copy()
            cond = bool(conditioned[q])
            target_points = np.arange(27, dtype=np.float64).reshape(9, 3) * 1e-3 + face
            if cond:
                d = len(donor)
                trace_donor_points = np.arange(d * 3, dtype=np.float64).reshape(d, 3) * 1e-4 + face + .5
                value_loading = -weight
                tangential_loading = tangential_weights[q].copy()
            else:
                trace_donor_points = np.empty((0, 3))
                value_loading = np.empty(0)
                tangential_loading = np.empty((0, 2))
            rows.append(IntegratedFaceRow(donor, weight, cond, trace_donor_points, target_points,
                                          value_loading, tangential_loading, int(family_table[face])))
            entity_id.append(face)
    return rows, entity_id


@pytest.mark.parametrize("n", (32, 48, 64))
def test_integrated_rows_real_hsx_fixture_roundtrip(n):
    with np.load(FACE_ROW_DATA / f"P07_N{n}.npz", allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    rows, entity_id = _integrated_rows_from_fixture(arrays)
    assert len(rows) == int(arrays["face_count"])
    assert any(row.boundary_conditioned for row in rows)
    assert any(not row.boundary_conditioned for row in rows)

    chunk = pack_integrated_rows(rows, entity_id=entity_id)
    expanded = expand_integrated_rows(chunk)
    for actual, expected in zip(expanded, rows, strict=True):
        _assert_integrated_rows_equal(actual, expected)


# --------------------------------------------------------------------------
# Freshly built on a small real N32 set — skip cleanly if unavailable
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_real_n32_point_rows_if_geometry_inputs_are_available():
    env_path = os.environ.get("DRBX_P08_REAL_N32_CONTEXT")
    if not env_path or not Path(env_path).exists():
        pytest.skip("no real N32 PointRowContext geometry inputs available in this environment "
                    "(set DRBX_P08_REAL_N32_CONTEXT to an npz with from_arrays() inputs)")
    with np.load(env_path, allow_pickle=False) as z:
        context = PointRowContext.from_arrays(
            faces=(z["faces_0"], z["faces_1"], z["faces_2"]),
            centers=(z["centers_0"], z["centers_1"], z["centers_2"]),
            raw_to_owner=z["raw_to_owner"], raw_volume=z["raw_volume"], owner_volume=z["owner_volume"],
            owner_centroid_xy=z["owner_centroid_xy"], eta_period=float(z["eta_period"]),
            dr=float(z["dr"]), dtheta=float(z["dtheta"]), deta=float(z["deta"]))
    builder = StructuredReconstruction(context)
    p = context.pts[:1]
    row = builder.rows((5, 1, 0), p, location="cell")
    chunk = pack_point_rows([row], request="R1", entity_id=0)
    expanded = expand_point_rows(chunk)[0]
    _assert_point_rows_equal(expanded, row)


# --------------------------------------------------------------------------
# Identity mismatch and corrupted-chunk rejection
# --------------------------------------------------------------------------

def test_identity_mismatch_is_rejected(tmp_path):
    _, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    identity = build_identity(component_hashes={"geometry": "abc"}, source_hashes={}, policy={"svd": 1e-4})
    save_row_artifact(tmp_path, 32, identity=identity, cells=[chunk])
    load_row_artifact(tmp_path, 32, identity)  # sanity: the matching identity still loads
    with pytest.raises(ValueError, match="identity"):
        load_row_artifact(tmp_path, 32, {**identity, "policy": {"svd": 2e-4}})
    with pytest.raises(ValueError, match="identity"):
        load_row_artifact(tmp_path, 32, {**identity, "component_hashes": {"geometry": "changed"}})


def test_corrupted_chunk_is_rejected(tmp_path):
    _, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    identity = build_identity(component_hashes={}, source_hashes={}, policy={})
    grid_dir = save_row_artifact(tmp_path, 40, identity=identity, cells=[chunk])
    chunk_path = grid_dir / "rows" / "cells_0.npz"
    data = bytearray(chunk_path.read_bytes())
    data[-1] ^= 0xFF
    chunk_path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupted"):
        load_row_artifact(tmp_path, 40, identity)


def test_missing_manifest_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_row_artifact(tmp_path, 999, {})


def test_legacy_v1_schema_is_rejected_with_a_clear_error(tmp_path):
    """The pre-tag v1 schema (no ``request``/``radial_degree`` on Neumann chunks)
    must not silently load."""
    _, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    identity = build_identity(component_hashes={}, source_hashes={}, policy={})
    grid_dir = save_row_artifact(tmp_path, 41, identity=identity, cells=[chunk])
    manifest_path = grid_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "drbx.p-row-artifact.v1"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_row_artifact(tmp_path, 41, identity)


# --------------------------------------------------------------------------
# Empty-row and mixed-width ("padded") cases within one chunk
# --------------------------------------------------------------------------

def test_empty_and_mixed_width_rows_in_one_chunk():
    context, rows, request, entity_id, bc_variant, radial_degree = _toy_point_rows()
    widths = {len(row.donor_ids) for row in rows}
    assert 0 in widths and len(widths) > 2  # a genuine mix of empty and differently sized rows

    chunk = pack_point_rows(rows, request=request, entity_id=entity_id,
                            bc_variant=bc_variant, radial_degree=radial_degree)
    expanded = expand_point_rows(chunk)
    for actual, expected in zip(expanded, rows, strict=True):
        assert len(actual.donor_ids) == len(expected.donor_ids)
        _assert_point_rows_equal(actual, expected)


# --------------------------------------------------------------------------
# Identity/hash helper sanity
# --------------------------------------------------------------------------

def test_hash_and_identity_helpers(tmp_path):
    payload = tmp_path / "component.bin"
    payload.write_bytes(b"some geometry bytes")
    file_hash = hash_file(payload)
    assert file_hash == hash_bytes(payload.read_bytes())
    assert len(file_hash) == 64

    module_hash = hash_source(artifact_module)
    assert len(module_hash) == 64

    identity = build_identity(component_hashes={"geometry": file_hash},
                              source_hashes={"artifact": module_hash},
                              policy={"svd": 1e-4, "residual": 1e-9})
    assert identity["component_hashes"]["geometry"] == file_hash
    assert "numpy_version" in identity and "platform" in identity
    # round trips through JSON exactly (the manifest stores it the same way)
    assert json.loads(json.dumps(identity, sort_keys=True)) == identity
