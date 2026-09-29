"""Real-geometry gate for the tensor-factored v3 point chunks (P08 step 2, task D1; slow).

Builds a few real N32 build units (the same cell / face units ``build_artifact`` would plan, point rows only,
geometry only for those units -- no full-grid stage) chosen from the owner profile so that together they cover
every factored family and request: R1 cells (ringwise, singleton), R2 faces (ringwise, singleton,
centered_radial) and R3 sides (ringwise, singleton), next to CSR-only sources (coupled_quartic, ...) in the
same chunks. Each unit is encoded once CSR-only and once with tensor factors; the tensor chunk must decode
bitwise to the CSR chunk (every field, hence every source), with no encoder fallback.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction  # noqa: E402
from drbx.stencils import artifact as art  # noqa: E402
from drbx.stencils import builder  # noqa: E402
from drbx.stencils.census import FaceCensus  # noqa: E402

N = 32
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917" / f"{N}x{N}x{N}"
needs_geometry = pytest.mark.skipif(
    not ((GEOMETRY / "rlp_topology.npz").is_file() and (GEOMETRY / "base_geometry.npz").is_file()),
    reason="HSX workspace N32 geometry_artifacts are unavailable")

#: (stage, unit index) of ``runner.chunk_units`` at the production chunk sizes (cells 4096, faces 2048), found
#: from the N32 owner profile [1, 4, 4, 8, 8, 16 x 6, 32 x 21]: cells 1 = radial 4-7 (ringwise), cells 3 =
#: radial 12-15 (singleton); faces 1 = radial-face i 3-4 (coupled_quartic R2 with ringwise R3 sides: a mixed
#: CSR / tensor chunk), faces 2 = i 5-6 (ringwise), faces 12 = i 25-26 (singleton + centered_radial)
UNITS = (("cells", 1), ("cells", 3), ("faces", 1), ("faces", 2), ("faces", 12))
FACTORED = {("R1", "singleton"), ("R1", "ringwise"), ("R2", "singleton"), ("R2", "ringwise"),
            ("R2", "centered_radial"), ("R3", "singleton"), ("R3", "ringwise")}
IDENTITY = art.build_identity(component_hashes={"geometry": "d1-real-gate"}, source_hashes={}, policy={"d1": 1})


def _no_normals(points):
    raise AssertionError("the chosen units have no boundary rows, so no Neumann companion should be requested")


def _build_unit(stage, index, *, capture, S, context, census, face_rows, quadrature, runner, build_artifact,
                limit=None):
    if stage == "cells":
        unit = runner.chunk_units("cells", N, N ** 3, 4096)[index]
        raw_ids = np.arange(unit["start"], unit["stop"], dtype=np.int64)[:limit]
        rows, _ = builder.build_r1_cell_rows(S, context, raw_ids, normal_coefficients=_no_normals, patch_cache={},
                                             capture_factors=capture)
        return rows, ([], rows)
    unit = runner.chunk_units("faces", N, len(face_rows), 2048)[index]
    selected = face_rows[unit["start"]:unit["stop"]][:limit]
    points, _ = quadrature(context.faces, census.keys()[selected], 3, face=True)
    r2, n2 = builder.build_r2_face_rows(S, context, census, selected, points, normal_coefficients=_no_normals,
                                        patch_cache={}, capture_factors=capture)
    r3, n3 = builder.build_r3_side_rows(S, context, census, selected, points, normal_coefficients=_no_normals,
                                        patch_cache={}, capture_factors=capture)
    assert not n2 and not n3
    return r2 + r3, (r2, r3)


@pytest.fixture(scope="module")
def real_units():
    from perpendicular_structured.reconstruction import load_context
    from p07_diffusion_global.numerics import quadrature
    from p_shared import build_artifact, runner

    t = load_context(N, str(WORKSPACE))
    context = build_artifact._make_context(t)
    S = StructuredReconstruction(context)
    census = FaceCensus.build(N, t.ro)
    face_rows = build_artifact.face_row_selection(census)
    tools = dict(S=S, context=context, census=census, face_rows=face_rows, quadrature=quadrature, runner=runner,
                 build_artifact=build_artifact)
    built = {}
    for stage, index in UNITS:
        rows, _ = _build_unit(stage, index, capture=True, **tools)
        built[(stage, index)] = rows
    return built, tools


@needs_geometry
@pytest.mark.slow
def test_capture_is_bitwise_identical_to_plain_rows_on_real_units(real_units):
    built, tools = real_units
    S = StructuredReconstruction(tools["context"])                   # fresh caches: the plain path
    checked = 0
    for (stage, index), rows in built.items():
        _, (plain2, plain3) = _build_unit(stage, index, capture=False, limit=24, **{**tools, "S": S})
        captured = _build_unit(stage, index, capture=True, limit=24, **tools)[1]
        for plain, with_factors in ((plain2, captured[0]), (plain3, captured[1])):
            assert len(plain) == len(with_factors)
            for a, b in zip(plain, with_factors, strict=True):
                assert (a.request, a.entity_id, a.bc_variant, a.radial_degree) == (b.request, b.entity_id, b.bc_variant, b.radial_degree)
                for name in ("donor_ids", "value", "gradient", "trace_donor_points", "trace_target_points"):
                    x, y = getattr(a.row, name), getattr(b.row, name)
                    assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
                assert a.row.diagnostics == b.row.diagnostics and a.factors is None
                factored = a.row.diagnostics["family"] in ("singleton", "ringwise", "centered_radial") and not a.row.boundary_conditioned
                assert (b.factors is not None) == factored
                checked += 1
    assert checked > 200


@needs_geometry
@pytest.mark.slow
def test_real_units_tensor_chunks_decode_bitwise_to_csr(real_units, tmp_path, capsys):
    built, tools = real_units
    build_artifact = tools["build_artifact"]
    totals = collections.Counter()
    covered = collections.Counter()
    sources_by = collections.Counter()
    per_unit = []
    chunks = {"cells": [], "faces": []}
    factors_by_group = {"cells": [], "faces": []}
    for (stage, index), rows in built.items():
        chunk = build_artifact._pack_point_rows(rows)
        stats = {}
        csr = art.encode_chunk(stage, chunk)
        tensor = art.encode_chunk(stage, chunk, factors=[r.factors for r in rows], stats=stats)
        from_csr, from_tensor = art.decode_chunk(stage, csr), art.decode_chunk(stage, tensor)
        assert art.chunk_mismatches(from_csr, chunk) == []
        assert art.chunk_mismatches(from_tensor, from_csr) == []      # every field of every source, dtype and bytes
        for name in from_csr.__dataclass_fields__:
            a, b = getattr(from_tensor, name), getattr(from_csr, name)
            assert isinstance(b, str) and a == b or (a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
        assert stats["fallback_sources"] == 0, stats
        with np.load(__import__("io").BytesIO(tensor)) as z:
            encoding = z["src_encoding"]
        first = from_csr.source_ptr[:-1]
        for request, family, coded in zip(from_csr.request[first], from_csr.family[first], encoding):
            sources_by[(str(request), str(family))] += 1
            if coded:
                covered[(str(request), str(family))] += 1
        assert int(encoding.sum()) == stats["tensor_sources"]
        totals.update(sources=len(first), tensor=stats["tensor_sources"], csr_bytes=len(csr), tensor_bytes=len(tensor),
                      fallback=stats["fallback_sources"])
        per_unit.append((stage, index, len(first), stats["tensor_sources"], len(csr), len(tensor)))
        chunks[stage].append(chunk)
        factors_by_group[stage].append([r.factors for r in rows])
    assert set(covered) == FACTORED, sorted(set(covered) ^ FACTORED)
    assert all(n > 0 for n in covered.values())

    # the same units through save_row_artifact / load_row_artifact, CSR-only and tensor
    csr_dir = art.save_row_artifact(tmp_path / "csr", N, identity=IDENTITY, cells=chunks["cells"], faces=chunks["faces"])
    stats = {}
    tensor_dir = art.save_row_artifact(tmp_path / "tensor", N, identity=IDENTITY, cells=chunks["cells"],
                                       faces=chunks["faces"], point_factors=factors_by_group, stats=stats)
    csr_art = art.load_row_artifact(tmp_path / "csr", N, IDENTITY)
    tensor_art = art.load_row_artifact(tmp_path / "tensor", N, IDENTITY)
    for group in ("cells", "faces"):
        for a, b in zip(tensor_art.__dict__[group], csr_art.__dict__[group], strict=True):
            assert art.chunk_mismatches(a, b) == []
    assert stats["fallback_sources"] == 0 and stats["tensor_sources"] == totals["tensor"]
    csr_manifest = json.loads((csr_dir / "manifest.json").read_text())
    tensor_manifest = json.loads((tensor_dir / "manifest.json").read_text())
    csr_bytes = sum(e["bytes"] for g in csr_manifest["chunks"].values() for e in g)
    tensor_bytes = sum(e["bytes"] for g in tensor_manifest["chunks"].values() for e in g)
    assert tensor_bytes < csr_bytes

    with capsys.disabled():
        print("\n[D1 real N32 gate] sources per (request, family): tensor / total")
        for key in sorted(sources_by):
            print(f"  {key[0]} {key[1]:20s} {covered.get(key, 0):6d} / {sources_by[key]:6d}")
        print(f"  fallbacks: {totals['fallback']}; tensor sources {totals['tensor']} of {totals['sources']}")
        for stage, index, sources, tensor_sources, cb, tb in per_unit:
            print(f"  {stage} unit {index}: {sources} sources, {tensor_sources} tensor, CSR-v3 {cb / 1e6:.2f} MB -> tensor-v3 {tb / 1e6:.2f} MB")
        print(f"  total CSR-v3 {csr_bytes / 1e6:.2f} MB -> tensor-v3 {tensor_bytes / 1e6:.2f} MB ({csr_bytes / tensor_bytes:.2f}x)")
