"""Schema v3 of the P08 row artifact: lossless source-major point chunks,
conditioned-only P07 side arrays, and the v2 reader.

See ``work/p08_step2_layout_loader_design_20260929/design.md`` section 2. The fast
tests use synthetic rows; the ``slow`` test re-encodes the real local N32 dev chunks.
"""
from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
from drbx.geometry.fci_perpendicular_neumann_trace import NeumannPointRows
from drbx.geometry.fci_perpendicular_reconstruction import PointRows
from drbx.stencils import artifact as art
from drbx.stencils.artifact import (
    BC_VARIANTS, REQUEST_KINDS, SCHEMA, SCHEMA_V2, IntegratedRowChunk, PointRowChunk, chunk_mismatches,
    decode_chunk, encode_chunk, expand_integrated_rows, expand_point_rows, hash_bytes, load_row_artifact,
    pack_integrated_rows, pack_neumann_rows, pack_point_rows, save_row_artifact,
)

REPO = Path(__file__).resolve().parents[1]

FAMILIES = ("singleton", "ringwise", "coupled_quartic", "boundary_transverse", "quartic_wall", "centered_radial")
IDENTITY = art.build_identity(component_hashes={"geometry": "abc"}, source_hashes={"x": "y"}, policy={"p": 1})


def _encode_v2(group, chunk):
    """The npz bytes of ``chunk`` in the v2 member layout (the writer only writes v3)."""
    to_arrays = {"cells": art._point_chunk_to_arrays_v2, "faces": art._point_chunk_to_arrays_v2,
                 "neumann": art._neumann_chunk_to_arrays, "p07": art._integrated_chunk_to_arrays_v2}[group]
    buffer = io.BytesIO()
    np.savez(buffer, **to_arrays(chunk))
    return buffer.getvalue()


def _save_v2_artifact(root, n, *, identity, **groups):
    """Write a v2 ``row_artifact/N{n}`` (manifest entries carry only ``file`` and ``sha256``)."""
    grid_dir = Path(root) / f"N{int(n)}"
    manifest_chunks = {}
    for group, chunks in groups.items():
        entries = []
        for index, chunk in enumerate(chunks):
            data = _encode_v2(group, chunk)
            relative = f"rows/{group}_{index}.npz"
            art._atomic_write_bytes(grid_dir / relative, data)
            entries.append({"file": relative, "sha256": hash_bytes(data)})
        manifest_chunks[group] = entries
    art._atomic_write_bytes(grid_dir / "manifest.json", json.dumps(
        {"schema": SCHEMA_V2, "identity": art._json_safe(identity), "chunks": manifest_chunks},
        sort_keys=True, indent=2).encode("utf-8"))
    return grid_dir


# --------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------

def _point_rows(seed, count=14, *, force_conditioned=None):
    """A mix of sources: 1 or 9 quadrature nodes, 0..12 donors, every family / request /
    BC variant / conditioning / gradient storage, with a shared pool of trace points."""
    rng = np.random.default_rng(seed)
    pool = rng.normal(size=(8, 3))
    rows, request, entity_id, bc_variant, degree, store_gradient = [], [], [], [], [], []
    for s in range(count):
        q = int(rng.choice((1, 9)))
        d = int(rng.integers(0, 13)) if s else 0     # the first source is a zero-donor one
        conditioned = bool(rng.random() < 0.4) if force_conditioned is None else force_conditioned
        donors = rng.choice(500, size=d, replace=False).astype(np.int64)
        rows.append(PointRows(
            donors, rng.normal(size=(q, d)), rng.normal(size=(q, 3, d)), conditioned,
            pool[rng.integers(0, len(pool), size=d)] if conditioned else np.empty((0, 3)),
            rng.normal(size=(q, 3)), {"family": str(rng.choice(FAMILIES)), "s": s}))
        request.append(str(rng.choice(("R1", "R2", "R3"))))
        entity_id.append(int(rng.integers(0, 10 ** 6)))
        bc_variant.append(str(rng.choice(BC_VARIANTS)))
        degree.append(int(rng.integers(0, 5)))
        store_gradient.append(bool(rng.random() < 0.6))
    return rows, dict(request=request, entity_id=entity_id, bc_variant=bc_variant,
                      radial_degree=degree, store_gradient=store_gradient)


def _point_chunk(seed=0, **kwargs):
    rows, tags = _point_rows(seed, **kwargs)
    return pack_point_rows(rows, **tags), rows


def _integrated_rows(seed, count=12, *, conditioned=None):
    rng = np.random.default_rng(seed)
    pool = rng.normal(size=(6, 3))
    rows = []
    for _ in range(count):
        d = int(rng.integers(0, 10))
        cond = bool(rng.random() < 0.5) if conditioned is None else conditioned
        rows.append(IntegratedFaceRow(
            rng.choice(400, size=d, replace=False).astype(np.int64), rng.normal(size=d), cond,
            pool[rng.integers(0, len(pool), size=d)] if cond else np.empty((0, 3)),
            rng.normal(size=(9, 3)),
            rng.normal(size=d) if cond else np.empty(0),
            rng.normal(size=(9, 2)) if cond else np.empty((0, 2)),
            int(rng.integers(0, 6))))
    return rows


def _integrated_chunk(seed=0, **kwargs):
    rows = _integrated_rows(seed, **kwargs)
    return pack_integrated_rows(rows, entity_id=list(range(100, 100 + len(rows)))), rows


def _neumann_chunk(seed=0, count=5):
    rng = np.random.default_rng(seed)
    rows = [NeumannPointRows(
        (donors := rng.choice(300, size=int(rng.integers(1, 9)), replace=False).astype(np.int64)),
        rng.normal(size=len(donors)), rng.normal(size=(3, len(donors))), rng.normal(size=(28, 3)),
        rng.normal(size=28), rng.normal(size=(3, 28)), float(rng.random()), float(rng.random()))
        for _ in range(count)]
    return pack_neumann_rows(rows, entity_id=list(range(count)), request="R2", radial_degree=3)


def _assert_identical(actual, expected):
    assert chunk_mismatches(actual, expected) == []
    for field in dataclasses.fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if not isinstance(b, str):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), field.name


# --------------------------------------------------------------------------
# Point chunks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_point_chunk_v3_roundtrip_is_bitwise(seed):
    chunk, rows = _point_chunk(seed)
    assert chunk.conditioned.any() and not chunk.conditioned.all()
    assert chunk.has_gradient.any() and not chunk.has_gradient.all()
    decoded = decode_chunk("faces", encode_chunk("faces", chunk))
    _assert_identical(decoded, chunk)
    # the exact dtypes the decoder must reproduce
    assert decoded.family.dtype == np.dtype("<U32") and decoded.request.dtype == np.dtype("<U8")
    assert decoded.bc_variant.dtype == np.dtype("<U1")
    assert decoded.donor_ptr.dtype == decoded.gradient_ptr.dtype == np.dtype(np.int64)
    assert decoded.donor_query.dtype == np.dtype(np.int32)
    assert np.all(decoded.donor_query[np.repeat(~chunk.conditioned, np.diff(chunk.donor_ptr))] == -1)
    # and the host expansion still recovers the original rows
    for actual, expected in zip(expand_point_rows(decoded), rows, strict=True):
        np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
        np.testing.assert_array_equal(actual.value, expected.value)


@pytest.mark.parametrize("conditioned", (True, False))
@pytest.mark.parametrize("gradients", (True, False))
def test_point_chunk_v3_roundtrip_uniform_conditioning_and_gradient(conditioned, gradients):
    rows, tags = _point_rows(11, force_conditioned=conditioned)
    tags["store_gradient"] = gradients
    chunk = pack_point_rows(rows, **tags)
    assert bool(chunk.conditioned.all()) == conditioned or not conditioned
    _assert_identical(decode_chunk("cells", encode_chunk("cells", chunk)), chunk)


def test_point_chunk_v3_all_kinds_variants_and_families_survive():
    rows, tags = _point_rows(3, count=60)
    tags["request"] = [REQUEST_KINDS[i % len(REQUEST_KINDS)] for i in range(len(rows))]
    tags["bc_variant"] = [BC_VARIANTS[i % len(BC_VARIANTS)] for i in range(len(rows))]
    chunk = pack_point_rows(rows, **tags)
    assert set(chunk.request.tolist()) == set(REQUEST_KINDS) and set(chunk.bc_variant.tolist()) == set(BC_VARIANTS)
    assert set(chunk.family.tolist()) == set(FAMILIES)
    _assert_identical(decode_chunk("faces", encode_chunk("faces", chunk)), chunk)


def test_point_chunk_v3_empty_chunk_roundtrips():
    chunk = pack_point_rows([], request="R1", entity_id=[])
    assert len(chunk.source_ptr) == 1
    _assert_identical(decode_chunk("cells", encode_chunk("cells", chunk)), chunk)


def test_point_chunk_v3_stores_one_donor_list_per_source():
    chunk, _ = _point_chunk(2)
    arrays = art._point_chunk_to_arrays(chunk)
    sources = len(chunk.source_ptr) - 1
    assert set(arrays) == {
        "src_request", "src_entity_id", "src_family", "family_table", "src_conditioned", "src_bc_variant",
        "src_radial_degree", "src_has_gradient", "source_ptr", "src_donor_ptr", "src_donor",
        "src_donor_query_ptr", "src_donor_query", "quad_node", "target_point", "value", "gradient",
        "query_table", "source_diagnostics_json"}
    assert arrays["src_request"].dtype == np.int8 and arrays["src_bc_variant"].dtype == np.int8
    assert arrays["src_family"].dtype == np.int16 and arrays["src_donor"].dtype == np.int32
    assert len(arrays["src_donor_ptr"]) == sources + 1
    assert set(json.loads(str(arrays["family_table"]))) == set(chunk.family.tolist())
    assert len(arrays["src_donor"]) < len(chunk.donor)
    assert len(arrays["src_donor_query"]) == int(np.diff(arrays["src_donor_ptr"])[arrays["src_conditioned"]].sum())
    assert len(encode_chunk("faces", chunk)) < len(_encode_v2("faces", chunk))


@pytest.mark.parametrize("violation", (
    "donor_list", "donor_query", "request", "entity_id", "family", "conditioned", "bc_variant",
    "radial_degree", "has_gradient", "donor_count", "unconditioned_donor_query", "gradient_ptr",
    "family_dtype", "empty_source"))
def test_point_chunk_v3_encoder_raises_on_invariant_violation(violation):
    chunk, _ = _point_chunk(4, force_conditioned=True)
    # a source with at least two targets and at least two donors
    widths = np.diff(chunk.donor_ptr)
    counts = np.diff(chunk.source_ptr)
    s = int(np.flatnonzero((counts > 1) & (widths[chunk.source_ptr[:-1]] > 1))[0])
    t = int(chunk.source_ptr[s]) + 1                      # the second target of that source
    p = int(chunk.donor_ptr[t])
    changes = {}
    if violation == "donor_list":
        donor = chunk.donor.copy(); donor[p] += 1
        changes["donor"] = donor
    elif violation == "donor_query":
        donor_query = chunk.donor_query.copy(); donor_query[p] += 1
        changes["donor_query"] = donor_query
    elif violation in ("request", "family", "bc_variant"):
        other = {"request": "R1" if chunk.request[t] != "R1" else "R2", "family": "other_family",
                 "bc_variant": "N" if chunk.bc_variant[t] != "N" else "D"}[violation]
        array = getattr(chunk, violation).copy()
        array[t] = other
        changes[violation] = array
    elif violation in ("entity_id", "radial_degree"):
        array = getattr(chunk, violation).copy(); array[t] += 1
        changes[violation] = array
    elif violation in ("conditioned", "has_gradient"):
        array = getattr(chunk, violation).copy(); array[t] = ~array[t]
        changes[violation] = array
    elif violation == "donor_count":
        donor_ptr = chunk.donor_ptr.copy(); donor_ptr[t + 1:] -= 1     # one target loses a donor
        changes.update(donor_ptr=donor_ptr, donor=np.delete(chunk.donor, p), value=np.delete(chunk.value, p),
                       donor_query=np.delete(chunk.donor_query, p))
    elif violation == "unconditioned_donor_query":
        chunk, _ = _point_chunk(4, force_conditioned=False)
        changes["donor_query"] = np.zeros_like(chunk.donor_query)
    elif violation == "gradient_ptr":
        gradient_ptr = chunk.gradient_ptr.copy(); gradient_ptr[-1] += 1
        changes["gradient_ptr"] = gradient_ptr
    elif violation == "family_dtype":
        changes["family"] = chunk.family.astype("<U16")
    elif violation == "empty_source":
        changes["source_ptr"] = np.insert(chunk.source_ptr, 1, 0)
    with pytest.raises(ValueError):
        art._point_chunk_to_arrays(dataclasses.replace(chunk, **changes))


def test_point_chunk_v3_decoder_rejects_inconsistent_arrays():
    chunk, _ = _point_chunk(5)
    arrays = art._point_chunk_to_arrays(chunk)
    arrays["src_donor_query"] = arrays["src_donor_query"][:-1] if len(arrays["src_donor_query"]) else np.zeros(1, np.int32)
    with pytest.raises(ValueError, match="corrupted"):
        art._arrays_to_point_chunk(arrays)


# --------------------------------------------------------------------------
# Integrated (P07) chunks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("conditioned", (None, True, False))
def test_integrated_chunk_v3_roundtrip_is_bitwise(conditioned):
    chunk, rows = _integrated_chunk(1, conditioned=conditioned)
    if conditioned is None:
        assert chunk.conditioned.any() and not chunk.conditioned.all()
    decoded = decode_chunk("p07", encode_chunk("p07", chunk))
    _assert_identical(decoded, chunk)
    assert not np.any(decoded.donor_query[np.repeat(~decoded.conditioned, np.diff(decoded.donor_ptr))] != -1)
    for actual, expected in zip(expand_integrated_rows(decoded), rows, strict=True):
        np.testing.assert_array_equal(actual.donor_ids, expected.donor_ids)
        np.testing.assert_array_equal(actual.value_loading, expected.value_loading)


def test_integrated_chunk_v3_stores_conditioned_rows_only():
    chunk, _ = _integrated_chunk(2)
    arrays = art._integrated_chunk_to_arrays(chunk)
    assert not {"donor_query", "value_loading", "tangential_loading"} & set(arrays)
    nnz = int(np.diff(chunk.donor_ptr)[chunk.conditioned].sum())
    assert len(arrays["cond_donor_query"]) == len(arrays["cond_value_loading"]) == nnz < len(chunk.donor)
    assert arrays["cond_tangential_loading"].shape == (int(chunk.conditioned.sum()), 9, 2)
    assert len(arrays["cond_donor_ptr"]) == int(chunk.conditioned.sum()) + 1


def test_integrated_chunk_v3_empty_chunk_roundtrips():
    chunk = pack_integrated_rows([], entity_id=[])
    _assert_identical(decode_chunk("p07", encode_chunk("p07", chunk)), chunk)


@pytest.mark.parametrize("violation", ("donor_query", "value_loading", "negative_zero", "tangential", "dtype"))
def test_integrated_chunk_v3_encoder_raises_on_invariant_violation(violation):
    chunk, _ = _integrated_chunk(3)
    row = int(np.flatnonzero(~chunk.conditioned & (np.diff(chunk.donor_ptr) > 0))[0])
    p = int(chunk.donor_ptr[row])
    if violation == "donor_query":
        donor_query = chunk.donor_query.copy(); donor_query[p] = 0
        changes = {"donor_query": donor_query}
    elif violation == "value_loading":
        loading = chunk.value_loading.copy(); loading[p] = 1.0
        changes = {"value_loading": loading}
    elif violation == "negative_zero":
        loading = chunk.value_loading.copy(); loading[p] = -0.0
        changes = {"value_loading": loading}
    elif violation == "tangential":
        tangential = chunk.tangential_loading.copy(); tangential[row, 0, 0] = 1.0
        changes = {"tangential_loading": tangential}
    else:
        changes = {"value_loading": chunk.value_loading.astype(np.float32)}
    with pytest.raises(ValueError):
        art._integrated_chunk_to_arrays(dataclasses.replace(chunk, **changes))


def test_integrated_chunk_v3_decoder_rejects_inconsistent_arrays():
    chunk, _ = _integrated_chunk(3)
    arrays = art._integrated_chunk_to_arrays(chunk)
    arrays["cond_value_loading"] = arrays["cond_value_loading"][:-1]
    with pytest.raises(ValueError, match="corrupted"):
        art._arrays_to_integrated_chunk(arrays)


# --------------------------------------------------------------------------
# Neumann chunks, manifest, and both schema versions on disk
# --------------------------------------------------------------------------

def test_neumann_chunk_layout_is_unchanged_in_v3():
    chunk = _neumann_chunk()
    assert art._neumann_chunk_to_arrays(chunk).keys() == {
        "entity_id", "quad_node", "request", "radial_degree", "donor_ptr", "donor", "value", "gradient",
        "wall_query", "boundary_value", "boundary_gradient", "condition", "constraint_residual", "query_table"}
    assert encode_chunk("neumann", chunk).count(b"PK") == _encode_v2("neumann", chunk).count(b"PK")
    _assert_identical(decode_chunk("neumann", encode_chunk("neumann", chunk)), chunk)


def _synthetic_groups(seed=0):
    point = [_point_chunk(seed + i)[0] for i in range(2)]
    return dict(cells=[point[0]], faces=point, neumann=[_neumann_chunk(seed), _neumann_chunk(seed + 1, 3)],
                p07=[_integrated_chunk(seed)[0], _integrated_chunk(seed + 1, conditioned=False)[0]])


def _assert_artifacts_equal(actual, expected):
    for group in ("cells", "faces", "neumann", "p07"):
        left, right = getattr(actual, group), getattr(expected, group)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_identical(a, b)


def test_save_row_artifact_writes_v3_manifest_fields_and_loads(tmp_path):
    groups = _synthetic_groups()
    grid_dir = save_row_artifact(tmp_path, 32, identity=IDENTITY, **groups)
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    assert manifest["schema"] == SCHEMA == "drbx.p-row-artifact.v3"
    assert manifest["identity"] == art._json_safe(IDENTITY)
    for group, chunks in groups.items():
        assert len(manifest["chunks"][group]) == len(chunks)
        for entry, chunk in zip(manifest["chunks"][group], chunks, strict=True):
            data = (grid_dir / entry["file"]).read_bytes()
            assert entry["sha256"] == hash_bytes(data) and entry["bytes"] == len(data)
            sources, targets = art.chunk_counts(group, chunk)
            assert (entry["sources"], entry["targets"]) == (sources, targets)
            assert art.chunk_file_stats(group, grid_dir / entry["file"]) == {
                "sources": sources, "targets": targets, "bytes": len(data)}
    assert manifest["chunks"]["faces"][0]["targets"] == int(groups["faces"][0].source_ptr[-1])
    loaded = load_row_artifact(tmp_path, 32, IDENTITY)
    _assert_artifacts_equal(loaded, art.RowArtifact(32, IDENTITY, **groups))


def test_v3_chunk_corruption_and_identity_are_still_rejected(tmp_path):
    grid_dir = save_row_artifact(tmp_path, 8, identity=IDENTITY, **_synthetic_groups())
    with pytest.raises(ValueError, match="identity"):
        load_row_artifact(tmp_path, 8, {**IDENTITY, "policy": {"p": 2}})
    path = grid_dir / "rows" / "faces_0.npz"
    data = bytearray(path.read_bytes()); data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupted"):
        load_row_artifact(tmp_path, 8, IDENTITY)


def test_v2_artifacts_still_load_and_decode_like_v3(tmp_path):
    groups = _synthetic_groups(2)
    _save_v2_artifact(tmp_path / "v2", 32, identity=IDENTITY, **groups)
    save_row_artifact(tmp_path / "v3", 32, identity=IDENTITY, **groups)
    manifest = json.loads((tmp_path / "v2" / "N32" / "manifest.json").read_text())
    assert manifest["schema"] == SCHEMA_V2
    assert set(manifest["chunks"]["faces"][0]) == {"file", "sha256"}
    v2 = load_row_artifact(tmp_path / "v2", 32, IDENTITY)
    v3 = load_row_artifact(tmp_path / "v3", 32, IDENTITY)
    _assert_artifacts_equal(v2, art.RowArtifact(32, IDENTITY, **groups))
    _assert_artifacts_equal(v3, v2)
    with np.load(tmp_path / "v2" / "N32" / "rows" / "faces_0.npz") as z:
        assert "src_request" not in z.files and "donor_query" in z.files


def test_unknown_schema_is_rejected(tmp_path):
    grid_dir = save_row_artifact(tmp_path, 9, identity=IDENTITY, cells=[_point_chunk(1)[0]])
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    for schema in ("drbx.p-row-artifact.v1", "drbx.p-row-artifact.v4"):
        (grid_dir / "manifest.json").write_text(json.dumps({**manifest, "schema": schema}))
        with pytest.raises(ValueError, match="schema mismatch"):
            load_row_artifact(tmp_path, 9, IDENTITY)


# --------------------------------------------------------------------------
# Slow: the real local N32 dev chunks
# --------------------------------------------------------------------------

_WORK = REPO.parent / "work"
DEV_GRIDS = tuple(_WORK / name / "artifact" / "N32" for name in (
    "p08_step1_campaign_dev_20260929T002907Z", "p08_step1_campaign_dev_20260928T201159Z"))


@pytest.mark.slow
@pytest.mark.parametrize("grid", DEV_GRIDS, ids=lambda p: p.parents[1].name)
def test_real_n32_dev_chunks_reencode_bitwise(grid):
    if not (grid / "manifest.json").exists():
        pytest.skip(f"no local dev artifact at {grid}")
    old = json.loads((grid / "manifest.json").read_text())
    if old["schema"] != SCHEMA_V2:
        pytest.skip(f"{grid} is not a v2 artifact")
    for group, entries in old["chunks"].items():
        for entry in entries:
            v2 = decode_chunk(group, (grid / entry["file"]).read_bytes())
            try:
                data = encode_chunk(group, v2)
            except KeyError as error:
                pytest.skip(f"{grid.parents[1].name} predates a field the current codec needs: {error}")
            assert chunk_mismatches(decode_chunk(group, data), v2) == []
