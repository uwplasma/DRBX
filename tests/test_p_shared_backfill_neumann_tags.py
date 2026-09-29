"""Tests for ``scripts/p_shared/backfill_neumann_tags.py``, the tool that
upgrades an existing P08 step-1 row artifact's Neumann chunks in place to
carry the ``request``/``radial_degree`` tags ``drbx.stencils.artifact``'s
``pack_neumann_rows`` used to drop (see that module's ``NeumannRowChunk``).

Two independent groups:

* **Pure derivation tests** (no geometry needed): feed small, hand-built
  sibling point/integrated chunk arrays into ``_derive_cells``/
  ``_derive_faces``/``_derive_p07`` and check the request/degree sequence
  they infer, including the R2-vs-R3 same-``(entity_id, quad_node)``-key
  split a faces unit can produce, and that a tampered/mismatched Neumann key
  sequence is rejected loudly.
* **End-to-end dry-run** on a tiny real artifact built from actual N32
  geometry inputs (skipped cleanly if the local HSX workspace inputs are
  unavailable, the same way ``tests/test_stencils_builder.py`` does):
  exercises the full CLI path, including the bitwise-recompute verification
  against ``prepare_neumann_point_rows``.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from drbx.stencils import artifact as artifact_mod  # noqa: E402
from drbx.stencils import builder  # noqa: E402
from drbx.stencils.census import FaceCensus  # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays  # noqa: E402

from p_shared import backfill_neumann_tags as bft  # noqa: E402
from p_shared import build_artifact  # noqa: E402

N = 32
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"


def _workspace_inputs_available() -> bool:
    if not (GEOMETRY / f"{N}x{N}x{N}" / "rlp_topology.npz").is_file():
        return False
    if not SIDECAR.is_file():
        return False
    try:
        side = json.loads(SIDECAR.read_text())
        for key in ("metric_cache", "makegrid"):
            if not Path(side[key]["path"]).is_file():
                return False
    except Exception:
        return False
    return True


needs_workspace = pytest.mark.skipif(
    not _workspace_inputs_available(),
    reason="HSX workspace geometry_artifacts / localized sidecar / metric cache are unavailable",
)


# ---------------------------------------------------------------------------
# Pure derivation tests: synthetic sibling-chunk arrays, no geometry needed.
# ---------------------------------------------------------------------------
def _point_arrays(*, request, entity_id, bc_variant, radial_degree, quad_node=None):
    n = len(request)
    if quad_node is None:
        quad_node = [0] * n
    return {
        "request": np.array(request, dtype="<U8"),
        "entity_id": np.array(entity_id, dtype=np.int64),
        "quad_node": np.array(quad_node, dtype=np.int16),
        "bc_variant": np.array(bc_variant, dtype="<U1"),
        "radial_degree": np.array(radial_degree, dtype=np.int16),
    }


def _integrated_arrays(*, entity_id, family, conditioned):
    return {
        "entity_id": np.array(entity_id, dtype=np.int64),
        "family": np.array(family, dtype=np.int16),
        "conditioned": np.array(conditioned, dtype=bool),
    }


def _neumann_arrays(*, entity_id, quad_node):
    return {"entity_id": np.array(entity_id, dtype=np.int64), "quad_node": np.array(quad_node, dtype=np.int16)}


def test_derive_cells_selects_boundary_rows_at_the_builder_constant_degree():
    point_arrays = _point_arrays(
        request=["R1"] * 6, entity_id=[10, 11, 12, 13, 14, 15],
        bc_variant=["", "D", "", "D", "D", ""], radial_degree=[0, 3, 0, 3, 3, 0])
    exp_request, exp_entity, exp_quad, exp_degree, blocks = bft._derive_cells(point_arrays, label="t")
    np.testing.assert_array_equal(exp_entity, [11, 13, 14])
    np.testing.assert_array_equal(exp_quad, [0, 0, 0])
    np.testing.assert_array_equal(exp_degree, [3, 3, 3])
    assert all(r == "R1" for r in exp_request)
    assert blocks == [("R1", 0, 1), ("R1", 1, 2), ("R1", 2, 3)]
    # cross-check against a matching neumann key sequence: no error
    bft._check_sequence("t", _neumann_arrays(entity_id=[11, 13, 14], quad_node=[0, 0, 0]), exp_entity, exp_quad)


def test_derive_cells_rejects_a_degree_that_disagrees_with_the_builder_constant():
    point_arrays = _point_arrays(request=["R1"], entity_id=[5], bc_variant=["D"], radial_degree=[4])
    with pytest.raises(ValueError, match="_CELL_NEUMANN_DEGREE"):
        bft._derive_cells(point_arrays, label="t")


def test_check_sequence_rejects_a_mismatched_key_sequence():
    point_arrays = _point_arrays(request=["R1", "R1"], entity_id=[11, 13], bc_variant=["D", "D"],
                                 radial_degree=[3, 3])
    exp_request, exp_entity, exp_quad, exp_degree, _blocks = bft._derive_cells(point_arrays, label="t")
    with pytest.raises(ValueError, match="disagrees with the builder's deterministic enumeration"):
        bft._check_sequence("t", _neumann_arrays(entity_id=[11, 99], quad_node=[0, 0]), exp_entity, exp_quad)
    with pytest.raises(ValueError, match="neumann row count"):
        bft._check_sequence("t", _neumann_arrays(entity_id=[11], quad_node=[0]), exp_entity, exp_quad)


def _faces_point_arrays_two_rows(*, r2_boundary, r3_side0_boundary, r3_side1_boundary):
    """One R2 census row (ridx=100) and one R3 side-pair (ridx=100, doubled
    ids 200/201), each contributing a whole 9-quad-node block -- exactly the
    shape ``build_r2_face_rows``/``build_r3_side_rows`` produce."""
    request, entity_id, bc_variant, radial_degree, quad_node = [], [], [], [], []

    def add(req, eid, boundary, degree):
        for q in range(9):
            request.append(req); entity_id.append(eid)
            bc_variant.append("D" if boundary else ""); radial_degree.append(degree if boundary else 0)
            quad_node.append(q)

    add("R2", 100, r2_boundary, 4)
    add("R3", 200, r3_side0_boundary, 3)
    add("R3", 201, r3_side1_boundary, 3)
    return _point_arrays(request=request, entity_id=entity_id, bc_variant=bc_variant,
                         radial_degree=radial_degree, quad_node=quad_node)


def test_derive_faces_r2_and_r3_share_a_key_but_carry_different_degrees():
    """The exact bug this tool fixes: R2 (degree 4, quartic_wall) and R3
    (degree 3, always) both produce a Neumann row tagged at census row 100,
    quad_node 0..8 -- identical keys, different degrees."""
    point_arrays = _faces_point_arrays_two_rows(r2_boundary=True, r3_side0_boundary=True, r3_side1_boundary=False)
    exp_request, exp_entity, exp_quad, exp_degree, blocks = bft._derive_faces(point_arrays, label="t")
    assert len(exp_request) == 18  # one R2 block + one R3 block, 9 rows each
    np.testing.assert_array_equal(exp_request[:9], ["R2"] * 9)
    np.testing.assert_array_equal(exp_request[9:], ["R3"] * 9)
    np.testing.assert_array_equal(exp_entity, [100] * 18)  # R3 tagged at the census row, not the doubled id
    np.testing.assert_array_equal(exp_quad, list(range(9)) * 2)
    np.testing.assert_array_equal(exp_degree[:9], [4] * 9)
    np.testing.assert_array_equal(exp_degree[9:], [3] * 9)
    assert blocks == [("R2", 0, 9), ("R3", 9, 18)]

    # The neumann chunk's own (entity_id, quad_node) cannot tell these two
    # 9-row blocks apart (same key twice) -- exactly the reported bug.
    neumann_entity = list(exp_entity)
    neumann_quad = list(exp_quad)
    bft._check_sequence("t", _neumann_arrays(entity_id=neumann_entity, quad_node=neumann_quad), exp_entity, exp_quad)


def test_derive_faces_r3_triggered_by_either_side_alone():
    # side0 not boundary, side1 boundary: the shared Neumann row still appears once.
    point_arrays = _faces_point_arrays_two_rows(r2_boundary=False, r3_side0_boundary=False, r3_side1_boundary=True)
    exp_request, exp_entity, exp_quad, exp_degree, blocks = bft._derive_faces(point_arrays, label="t")
    assert len(exp_request) == 9
    assert all(r == "R3" for r in exp_request)
    np.testing.assert_array_equal(exp_entity, [100] * 9)
    np.testing.assert_array_equal(exp_degree, [3] * 9)


def test_derive_faces_neither_side_boundary_and_r2_not_boundary_yields_nothing():
    point_arrays = _faces_point_arrays_two_rows(r2_boundary=False, r3_side0_boundary=False, r3_side1_boundary=False)
    exp_request, exp_entity, exp_quad, exp_degree, blocks = bft._derive_faces(point_arrays, label="t")
    assert len(exp_request) == 0
    assert blocks == []


def test_derive_p07_selects_boundary_families_at_their_own_degree():
    integrated_arrays = _integrated_arrays(entity_id=[1, 2, 3, 4], family=[0, 1, 2, 4],
                                          conditioned=[False, True, True, True])
    exp_request, exp_entity, exp_quad, exp_degree, blocks = bft._derive_p07(integrated_arrays, label="t")
    assert len(exp_request) == 27  # 3 boundary faces (families 1, 2, 4) x 9 nodes
    np.testing.assert_array_equal(np.unique(exp_entity), [2, 3, 4])
    # family 1 -> degree 4, family 2 -> degree 4, family 4 -> degree 3
    by_entity_degree = dict(zip(exp_entity.tolist(), exp_degree.tolist()))
    assert by_entity_degree[2] == 4 and by_entity_degree[3] == 4 and by_entity_degree[4] == 3


def test_derive_p07_rejects_a_boundary_family_that_is_not_conditioned():
    integrated_arrays = _integrated_arrays(entity_id=[7], family=[1], conditioned=[False])
    with pytest.raises(ValueError, match="boundary_conditioned"):
        bft._derive_p07(integrated_arrays, label="t")


# ---------------------------------------------------------------------------
# End-to-end dry-run on a tiny real artifact.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_geometry():
    if not _workspace_inputs_available():
        pytest.skip("workspace inputs unavailable")
    from perpendicular_structured.reconstruction import load_context
    from p_shared import provider as p_shared_provider

    t = load_context(N, str(WORKSPACE))
    context = build_artifact._make_context(t)
    census = FaceCensus.build(N, t.ro)
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False)
    normal_coefficients = build_artifact._normal_coefficients_fn(provider.reference)
    return {"t": t, "context": context, "census": census, "provider": provider,
            "normal_coefficients": normal_coefficients}


def _sparse_face_geometry(provider, t, face_row_indices, census) -> "GeometryArrays":
    """A ``GeometryArrays`` covering *every* row ``backfill_neumann_tags``'s
    ``_point_for`` needs to be able to look up (so its ``face_row_indices``
    positional mapping is correct over the whole grid), but built without
    ever calling the provider's physics methods (``p05_metric``,
    ``p06_curvature``, ``p07_perpendicular_tensor(_and_divergence)``) --
    those are the expensive per-point reference evaluations, and
    ``_point_for`` only ever reads ``face_points`` (the plain quadrature-node
    coordinates, cheap to produce for every face at once). Every other field
    is a zero-filled placeholder of the right shape; ``GeometryArrays``'s own
    identity is computed from these exact arrays, so ``.verify()`` still
    passes on load."""
    from drbx.stencils.geometry_arrays import SCHEMA as GEOMETRY_SCHEMA

    n = t.n
    raw_keys = np.array(np.unravel_index(np.arange(n ** 3, dtype=np.int64), (n, n, n))).T.astype(np.int64)
    face_keys = census.keys()[np.asarray(face_row_indices, dtype=np.int64)]
    raw_node_points, raw_weight = provider.raw_cell_weight(t.faces, raw_keys)
    raw_points = np.asarray(raw_node_points).reshape(-1, 3)
    raw_weight = np.asarray(raw_weight).reshape(-1)
    face_points, face_weight = provider.face_node_weight(t.faces, face_keys)
    face_points = np.asarray(face_points)
    n_faces = face_points.shape[0]
    face_weight = np.asarray(face_weight).reshape(n_faces, 9)
    arrays = dict(
        raw_points=raw_points,
        p05_raw_h=np.zeros((len(raw_points), 3)), p05_raw_jacobian=np.zeros(len(raw_points)),
        p06_raw_J=np.zeros(len(raw_points)), p06_raw_B=np.zeros(len(raw_points)),
        p06_raw_K=np.zeros((len(raw_points), 3)), p06_raw_weight=raw_weight,
        p07_raw_tensor=np.zeros((len(raw_points), 3, 3)), p07_raw_divergence=np.zeros((len(raw_points), 3)),
        face_points=face_points,
        p05_face_h=np.zeros((n_faces, 9, 3)), p05_face_jacobian=np.zeros((n_faces, 9)),
        p06_face_J=np.zeros((n_faces, 9)), p06_face_B=np.zeros((n_faces, 9)),
        p06_face_K=np.zeros((n_faces, 9, 3)), p06_face_weight=face_weight,
        p07_face_tensor=np.zeros((n_faces, 9, 3, 3)),
    )
    identity = GeometryArrays._compute_identity(arrays)
    return GeometryArrays(schema=GEOMETRY_SCHEMA, identity=identity, **arrays)


def _write_stage(grid_dir: Path, manifest_chunks: dict, stage: str, index: int, *, point_chunk_arrays,
                 neumann_chunk_arrays, group: str):
    rows_dir = grid_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    def _save(name, arrays):
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        data = buffer.getvalue()
        (rows_dir / name).write_bytes(data)
        return artifact_mod.hash_bytes(data)

    chunk_name = f"{group}_{stage}_{index:05d}.npz"
    neumann_name = f"neumann_{stage}_{index:05d}.npz"
    manifest_chunks.setdefault(group, []).append(
        {"file": f"rows/{chunk_name}", "sha256": _save(chunk_name, point_chunk_arrays)})
    manifest_chunks.setdefault("neumann", []).append(
        {"file": f"rows/{neumann_name}", "sha256": _save(neumann_name, neumann_chunk_arrays)})


@needs_workspace
@pytest.mark.slow
def test_backfill_dry_run_end_to_end_on_a_tiny_real_artifact(tmp_path, real_geometry):
    """Build a tiny (few-row) legacy-schema artifact directly from real N32
    geometry -- one boundary cell, one boundary face (with both R2 and R3
    Neumann companions at the same key), one P07 boundary face -- and run
    ``backfill_neumann_tags`` end to end in dry-run mode."""
    t, context, census = real_geometry["t"], real_geometry["context"], real_geometry["census"]
    provider, nc = real_geometry["provider"], real_geometry["normal_coefficients"]
    from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction
    S = StructuredReconstruction(context)

    n = N
    grid_dir = tmp_path / f"N{n}"
    grid_dir.mkdir(parents=True)
    patch_cache: dict = {}

    # --- cells: a tiny slice covering at least one boundary raw cell.
    raw_ids = np.arange(n * n * (n - 1), n * n * (n - 1) + 8, dtype=np.int64)  # i == n-2 layer
    cell_point_rows, cell_neumann_rows = builder.build_r1_cell_rows(
        S, context, raw_ids, normal_coefficients=nc, patch_cache=patch_cache)
    assert len(cell_neumann_rows) > 0
    cell_point_chunk = artifact_mod.pack_point_rows(
        [r.row for r in cell_point_rows], request=[r.request for r in cell_point_rows],
        entity_id=[r.entity_id for r in cell_point_rows], bc_variant=[r.bc_variant for r in cell_point_rows],
        radial_degree=[r.radial_degree for r in cell_point_rows])
    cell_neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in cell_neumann_rows], entity_id=[r.entity_id for r in cell_neumann_rows],
        quad_node=[r.quad_node for r in cell_neumann_rows], request=[r.source for r in cell_neumann_rows],
        radial_degree=[r.radial_degree for r in cell_neumann_rows])
    cell_neumann_legacy = artifact_mod._neumann_chunk_to_arrays(cell_neumann_chunk)
    del cell_neumann_legacy["request"], cell_neumann_legacy["radial_degree"]

    # --- faces: a boundary census row (physical_wall_quartic_BC, i == n),
    # so R2 gets degree 4 and its R3 companion (degree 3, always) collides at
    # the exact same (entity_id, quad_node) key.
    face_row_indices = build_artifact.face_row_selection(census)
    keys = census.keys()
    wall_mask = (keys[face_row_indices, 0] == 0) & (keys[face_row_indices, 1] == n)
    assert np.any(wall_mask), "sample grid has no physical-wall radial face to test R2/R3 collision against"
    ridx = int(face_row_indices[np.flatnonzero(wall_mask)[0]])
    provider_geom = build_artifact.builder.build_geometry_arrays(
        provider, context, raw_ids=np.arange(0, dtype=np.int64), face_row_indices=[ridx], census=census)
    face_points = provider_geom.face_points

    point_rows_2, neumann_rows_2 = builder.build_r2_face_rows(
        S, context, census, [ridx], face_points, normal_coefficients=nc, patch_cache=patch_cache)
    point_rows_3, neumann_rows_3 = builder.build_r3_side_rows(
        S, context, census, [ridx], face_points, normal_coefficients=nc, patch_cache=patch_cache)
    face_point_rows = point_rows_2 + point_rows_3
    face_neumann_rows = neumann_rows_2 + neumann_rows_3
    assert len(neumann_rows_2) > 0 and len(neumann_rows_3) > 0, "need both an R2 and an R3 Neumann companion"
    face_point_chunk = artifact_mod.pack_point_rows(
        [r.row for r in face_point_rows], request=[r.request for r in face_point_rows],
        entity_id=[r.entity_id for r in face_point_rows], bc_variant=[r.bc_variant for r in face_point_rows],
        radial_degree=[r.radial_degree for r in face_point_rows],
        store_gradient=[not str(r.request).startswith("R3") for r in face_point_rows])
    face_neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in face_neumann_rows], entity_id=[r.entity_id for r in face_neumann_rows],
        quad_node=[r.quad_node for r in face_neumann_rows], request=[r.source for r in face_neumann_rows],
        radial_degree=[r.radial_degree for r in face_neumann_rows])
    face_neumann_legacy = artifact_mod._neumann_chunk_to_arrays(face_neumann_chunk)
    del face_neumann_legacy["request"], face_neumann_legacy["radial_degree"]

    # --- p07: the same wall face's P07 topology row, if it carries one.
    p07_row_indices = builder.p07_row_selection(census)
    p07_hit = np.flatnonzero(p07_row_indices == ridx)
    manifest_chunks: dict = {}
    _write_stage(grid_dir, manifest_chunks, "cells", 0, point_chunk_arrays=artifact_mod._point_chunk_to_arrays(cell_point_chunk),
                neumann_chunk_arrays=cell_neumann_legacy, group="cells")
    _write_stage(grid_dir, manifest_chunks, "faces", 0, point_chunk_arrays=artifact_mod._point_chunk_to_arrays(face_point_chunk),
                neumann_chunk_arrays=face_neumann_legacy, group="faces")
    if len(p07_hit):
        p07_geom = build_artifact.builder.build_geometry_arrays(
            provider, context, raw_ids=np.arange(0, dtype=np.int64), face_row_indices=[ridx], census=census)
        family = census.family[[ridx]].astype(np.int64)
        weight = p07_geom.p06_face_weight
        tensor = p07_geom.p07_face_tensor
        from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor
        integrand = contract_face_tensor(weight, tensor, np.array([0]))
        integrated_rows, p07_neumann_rows = builder.build_r4_p07_rows(
            context, census, [ridx], p07_geom.face_points, weight, tensor,
            normal_coefficients=nc, patch_cache=patch_cache)
        p07_chunk = artifact_mod.pack_integrated_rows(
            [r.row for r in integrated_rows], entity_id=[r.entity_id for r in integrated_rows])
        p07_neumann_chunk = artifact_mod.pack_neumann_rows(
            [r.row for r in p07_neumann_rows], entity_id=[r.entity_id for r in p07_neumann_rows],
            quad_node=[r.quad_node for r in p07_neumann_rows], request=[r.source for r in p07_neumann_rows],
            radial_degree=[r.radial_degree for r in p07_neumann_rows])
        p07_neumann_legacy = artifact_mod._neumann_chunk_to_arrays(p07_neumann_chunk)
        del p07_neumann_legacy["request"], p07_neumann_legacy["radial_degree"]
        _write_stage(grid_dir, manifest_chunks, "p07", 0, point_chunk_arrays=artifact_mod._integrated_chunk_to_arrays(p07_chunk),
                    neumann_chunk_arrays=p07_neumann_legacy, group="p07")

    identity = {"component_hashes": {}, "source_hashes": {}, "policy": {}}
    manifest = {"schema": bft.LEGACY_SCHEMA, "identity": identity, "chunks": manifest_chunks}
    (grid_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    census.save(grid_dir / "census.npz")
    full_geometry = _sparse_face_geometry(provider, t, build_artifact.face_row_selection(census), census)
    full_geometry.save(grid_dir / "geometry.npz")

    receipt = bft.main([
        "--root", str(tmp_path), "--n", str(n),
        "--input-root", str(WORKSPACE), "--sidecar", str(SIDECAR), "--dry-run",
    ])

    assert receipt["dry_run"] is True and receipt["finalized"] is False
    assert receipt["files_processed"] == len(manifest_chunks.get("neumann", []))
    assert receipt["row_counts_by_request"].get("R1", 0) > 0
    assert receipt["row_counts_by_request"].get("R2", 0) > 0
    assert receipt["row_counts_by_request"].get("R3", 0) > 0
    assert receipt["verification"]["max_abs_diff_at_own_degree"] == 0.0
    # the R2/R3 collision at the wall face must show up as distinguishable
    # (different degrees, different rows) -- the whole point of this tool.
    assert receipt["verification"]["distinguishable_at_other_degree"] > 0
    # dry-run must never have written anything back
    assert json.loads((grid_dir / "manifest.json").read_text())["schema"] == bft.LEGACY_SCHEMA
    assert not (grid_dir / "backfill_receipt.json").exists()


@needs_workspace
@pytest.mark.slow
def test_backfill_rejects_a_tampered_neumann_key_sequence(tmp_path, real_geometry):
    """A hand-corrupted neumann chunk (entity_id sequence that disagrees with
    the sibling point chunk's own deterministic enumeration) must fail
    loudly, never silently guess."""
    t, context, census = real_geometry["t"], real_geometry["context"], real_geometry["census"]
    nc = real_geometry["normal_coefficients"]
    from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction
    S = StructuredReconstruction(context)
    n = N
    raw_ids = np.arange(n * n * (n - 1), n * n * (n - 1) + 4, dtype=np.int64)
    point_rows, neumann_rows = builder.build_r1_cell_rows(
        S, context, raw_ids, normal_coefficients=nc, patch_cache={})
    assert len(neumann_rows) > 0
    point_chunk = artifact_mod.pack_point_rows(
        [r.row for r in point_rows], request=[r.request for r in point_rows],
        entity_id=[r.entity_id for r in point_rows], bc_variant=[r.bc_variant for r in point_rows],
        radial_degree=[r.radial_degree for r in point_rows])
    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    neumann_legacy = artifact_mod._neumann_chunk_to_arrays(neumann_chunk)
    del neumann_legacy["request"], neumann_legacy["radial_degree"]
    # Corrupt the key sequence.
    neumann_legacy["entity_id"] = neumann_legacy["entity_id"].copy()
    neumann_legacy["entity_id"][0] += 999999

    grid_dir = tmp_path / f"N{n}"
    grid_dir.mkdir(parents=True)
    manifest_chunks: dict = {}
    _write_stage(grid_dir, manifest_chunks, "cells", 0, point_chunk_arrays=artifact_mod._point_chunk_to_arrays(point_chunk),
                neumann_chunk_arrays=neumann_legacy, group="cells")
    manifest = {"schema": bft.LEGACY_SCHEMA, "identity": {}, "chunks": manifest_chunks}
    (grid_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    census.save(grid_dir / "census.npz")

    with pytest.raises(ValueError, match="disagrees with the builder's deterministic enumeration"):
        bft.main(["--root", str(tmp_path), "--n", str(n), "--dry-run", "--skip-verification"])
