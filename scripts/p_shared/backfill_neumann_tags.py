#!/usr/bin/env python3
"""Upgrade an existing P08 step-1 row artifact's Neumann chunks in place.

Fixes a format gap: ``drbx.stencils.artifact.pack_neumann_rows`` used to drop
the ``request`` ("R1"/"R2"/"R3"/"R4", the *originating* request that needed
this Neumann wall-lattice row) and ``radial_degree`` tags that
``scripts/p_shared/build_artifact.py`` already had on hand (see
``drbx.stencils.builder.NeumannRowRequest``). In a faces unit, an R2 row and
an R3 row can share the exact same ``(entity_id, quad_node)`` key -- R3's
Neumann companion is tagged at the face's own census row index, not the
side-doubled R3 point-row id (see ``build_r3_side_rows``'s docstring) -- while
using different degrees, so a consumer reading only ``(entity_id, quad_node)``
cannot tell which stored row is which.

This tool never recomputes a row's arithmetic (donor ids, value, gradient,
boundary_value/boundary_gradient, condition, residual are all copied
byte-for-byte from the existing chunk); it only *infers* each row's
``request``/``radial_degree`` from the builder's own deterministic packing
order, cross-checks that inference against the sibling point/integrated row
chunk already saved alongside it (which never had this bug -- ``PointRowChunk``
and ``IntegratedRowChunk`` already carried ``request``/``radial_degree``), and
then verifies a sample of rows by an independent bitwise recompute through
``prepare_neumann_point_rows`` at the inferred degree.

Run from ``DRBX/scripts`` (never from inside a scripts package directory --
shadows the stdlib ``operator``):

    python -m p_shared.backfill_neumann_tags --root <artifact output dir> \\
        --n 32 --input-root "/Users/yxie/Desktop/HSX drbx" \\
        --sidecar "/Users/yxie/Desktop/HSX drbx/work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json" \\
        [--dry-run] [--stages cells,faces,p07] [--limit-files N] \\
        [--sample-block-cap N] [--skip-verification]

Deterministic packing order this tool relies on (mirrors ``build_artifact.py``
and ``drbx.stencils.builder`` exactly -- see those modules' docstrings):

* **cells** units: every row is ``request='R1'``; a Neumann companion exists
  for exactly the boundary (``bc_variant == 'D'``) rows of the sibling point
  chunk, one row each (``quad_node`` always 0), at
  ``builder._CELL_NEUMANN_DEGREE``.
* **faces** units: the sibling point chunk is ``R2`` rows (each a whole
  9-quad-node block per census row) followed by ``R3`` rows (each census row
  contributing one 9-node block per present side, tagged at the *doubled*
  entity id ``census_row*2 + side``). The Neumann chunk is
  ``neumann_rows_2 + neumann_rows_3``: an R2 block (degree taken from that
  row's own stored ``radial_degree``, 3 or 4) for every boundary census row,
  in row order, followed by one R3 block (degree always
  ``builder._SIDE_NEUMANN_DEGREE``) for every census row where *either* side
  was boundary-conditioned, in row order.
* **p07** units: a Neumann block (degree from
  ``builder._P07_NEUMANN_DEGREE`` by family) for every row whose P07 family is
  in ``{1, 2, 4}`` (always boundary-conditioned), in row order.

Verification is mandatory to *finalize* an upgrade (bump the manifest schema
and overwrite files) but not to inspect a candidate one: ``--dry-run`` never
writes, and ``--skip-verification`` may only be combined with ``--dry-run``
(a structural-only check with no geometry inputs needed).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = Path(__file__).resolve().parent           # .../DRBX/scripts/p_shared
SCRIPTS = HERE.parent                             # .../DRBX/scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import build_artifact as _build_artifact             # noqa: E402
from p_shared import provider as p_shared_provider                  # noqa: E402
from perpendicular_structured.reconstruction import load_context    # noqa: E402

from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows  # noqa: E402
from drbx.stencils.census import FaceCensus            # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays  # noqa: E402
from drbx.stencils import builder                       # noqa: E402
from drbx.stencils import artifact as artifact_mod        # noqa: E402

LEGACY_SCHEMA = artifact_mod.SCHEMA_V1_NEUMANN_UNTAGGED
CURRENT_SCHEMA = artifact_mod.SCHEMA
_STAGE_ORDER = ("cells", "faces", "p07")
_NEUMANN_NAME_RE = re.compile(r"^neumann_(cells|faces|p07)_(\d+)\.npz$")


# ---------------------------------------------------------------------------
# Manifest / npz plumbing (mirrors artifact.py's own verified-read/atomic-write
# convention, without going through load_row_artifact -- which refuses this
# artifact's legacy schema by design).
# ---------------------------------------------------------------------------
def _read_manifest(grid_dir: Path) -> dict:
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no row artifact manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    if schema == CURRENT_SCHEMA:
        raise ValueError(f"{manifest_path} is already schema {CURRENT_SCHEMA!r}; nothing to backfill")
    if schema != LEGACY_SCHEMA:
        raise ValueError(f"backfill_neumann_tags only understands schema {LEGACY_SCHEMA!r}; "
                          f"found {schema!r}, which this tool does not know how to upgrade")
    return manifest


def _manifest_entry(manifest: dict, group: str, relative: str) -> dict:
    for entry in manifest["chunks"].get(group, []):
        if entry["file"] == relative:
            return entry
    raise ValueError(f"manifest has no entry for {relative!r} in group {group!r}")


def _read_verified_npz(grid_dir: Path, entry: dict) -> dict:
    path = grid_dir / entry["file"]
    data = path.read_bytes()
    actual = artifact_mod.hash_bytes(data)
    if actual != entry["sha256"]:
        raise ValueError(f"chunk corrupted before backfill: {entry['file']} "
                          f"(sha256 {actual} != manifest {entry['sha256']})")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _discover_neumann_files(manifest: dict, *, stages=None) -> list[tuple[str, int, dict]]:
    out = []
    for entry in manifest["chunks"].get("neumann", []):
        name = Path(entry["file"]).name
        m = _NEUMANN_NAME_RE.match(name)
        if not m:
            raise ValueError(f"unrecognized neumann chunk filename: {name!r}")
        stage, index = m.group(1), int(m.group(2))
        if stages is not None and stage not in stages:
            continue
        out.append((stage, index, entry))
    out.sort(key=lambda item: (_STAGE_ORDER.index(item[0]), item[1]))
    return out


# ---------------------------------------------------------------------------
# Deterministic re-derivation of request/radial_degree per stage, from the
# sibling point/integrated chunk (never from the neumann keys alone).
# ---------------------------------------------------------------------------
def _iter_source_blocks(request_p: np.ndarray, entity_p: np.ndarray):
    """Maximal runs of the sibling point chunk sharing (request, entity_id)."""
    n = len(entity_p)
    i = 0
    while i < n:
        j = i + 1
        while j < n and entity_p[j] == entity_p[i] and request_p[j] == request_p[i]:
            j += 1
        yield i, j
        i = j


def _derive_cells(point_arrays: dict, *, label: str):
    request_p = point_arrays["request"]
    if len(request_p) and not np.all(request_p == "R1"):
        raise ValueError(f"{label}: sibling point chunk contains a non-R1 request; unexpected for the cells group")
    entity_p = point_arrays["entity_id"].astype(np.int64)
    bcv_p = point_arrays["bc_variant"]
    deg_p = point_arrays["radial_degree"].astype(np.int64)
    mask = bcv_p == "D"
    exp_entity = entity_p[mask]
    exp_quad = np.zeros(len(exp_entity), dtype=np.int64)
    exp_degree = deg_p[mask]
    if len(exp_degree) and not np.all(exp_degree == builder._CELL_NEUMANN_DEGREE):
        raise ValueError(f"{label}: a boundary R1 row's own radial_degree does not match "
                          f"builder._CELL_NEUMANN_DEGREE ({builder._CELL_NEUMANN_DEGREE})")
    exp_request = np.full(len(exp_entity), "R1", dtype="<U8")
    blocks = [("R1", i, i + 1) for i in range(len(exp_entity))]
    return exp_request, exp_entity, exp_quad, exp_degree.astype(np.int8), blocks


def _derive_faces(point_arrays: dict, *, label: str):
    request_p = point_arrays["request"]
    entity_p = point_arrays["entity_id"].astype(np.int64)
    quad_p = point_arrays["quad_node"].astype(np.int64)
    bcv_p = point_arrays["bc_variant"]
    deg_p = point_arrays["radial_degree"].astype(np.int64)

    out_request: list[str] = []
    out_entity: list[int] = []
    out_quad: list[int] = []
    out_degree: list[int] = []
    blocks: list[tuple[str, int, int]] = []

    def flush_r3(ridx: int, sides: list[tuple[int, int]]) -> None:
        conditioned_sides = [(s, e) for s, e in sides if bcv_p[s] == "D"]
        if not conditioned_sides:
            return
        for s, e in conditioned_sides:
            if not np.all(bcv_p[s:e] == "D"):
                raise ValueError(f"{label}: an R3 side's own targets disagree on bc_variant (ridx={ridx})")
        degrees = {int(deg_p[s]) for s, e in conditioned_sides}
        if degrees != {builder._SIDE_NEUMANN_DEGREE}:
            raise ValueError(f"{label}: an R3 conditioned side's radial_degree {degrees} != "
                              f"builder._SIDE_NEUMANN_DEGREE ({builder._SIDE_NEUMANN_DEGREE}) at ridx={ridx}")
        start = len(out_entity)
        out_request.extend(["R3"] * 9)
        out_entity.extend([ridx] * 9)
        out_quad.extend(range(9))
        out_degree.extend([builder._SIDE_NEUMANN_DEGREE] * 9)
        blocks.append(("R3", start, start + 9))

    current_ridx = None
    current_sides: list[tuple[int, int]] = []
    for s, e in _iter_source_blocks(request_p, entity_p):
        kind = str(request_p[s])
        if e - s != 9:
            raise ValueError(f"{label}: a {kind} row has {e - s} quad nodes, expected 9")
        if kind == "R2":
            if bcv_p[s] == "D":
                if not np.all(bcv_p[s:e] == "D"):
                    raise ValueError(f"{label}: an R2 row's targets disagree on bc_variant (entity_id={int(entity_p[s])})")
                degree = int(deg_p[s])
                if not np.all(deg_p[s:e] == degree):
                    raise ValueError(f"{label}: an R2 row's targets disagree on radial_degree (entity_id={int(entity_p[s])})")
                if degree not in (3, 4):
                    raise ValueError(f"{label}: an R2 boundary row has unexpected radial_degree {degree}")
                start = len(out_entity)
                out_request.extend(["R2"] * 9)
                out_entity.extend([int(entity_p[s])] * 9)
                out_quad.extend(quad_p[s:e].tolist())
                out_degree.extend([degree] * 9)
                blocks.append(("R2", start, start + 9))
        elif kind == "R3":
            ridx = int(entity_p[s]) // 2
            if current_ridx is not None and ridx != current_ridx:
                flush_r3(current_ridx, current_sides)
                current_sides = []
            current_ridx = ridx
            current_sides.append((s, e))
        else:
            raise ValueError(f"{label}: unexpected request kind {kind!r} in a faces point chunk")
    if current_ridx is not None:
        flush_r3(current_ridx, current_sides)

    exp_request = np.array(out_request, dtype="<U8")
    exp_entity = np.array(out_entity, dtype=np.int64)
    exp_quad = np.array(out_quad, dtype=np.int64)
    exp_degree = np.array(out_degree, dtype=np.int8)
    return exp_request, exp_entity, exp_quad, exp_degree, blocks


def _derive_p07(integrated_arrays: dict, *, label: str):
    entity_i = integrated_arrays["entity_id"].astype(np.int64)
    family_i = integrated_arrays["family"].astype(np.int64)
    conditioned_i = integrated_arrays["conditioned"].astype(bool)
    degree_map = builder._P07_NEUMANN_DEGREE
    mask = np.isin(family_i, list(degree_map))
    if mask.any() and not np.all(conditioned_i[mask]):
        raise ValueError(f"{label}: a P07 family expected to carry a Neumann companion is not boundary_conditioned")

    out_request: list[str] = []
    out_entity: list[int] = []
    out_quad: list[int] = []
    out_degree: list[int] = []
    blocks: list[tuple[str, int, int]] = []
    for eid, fam in zip(entity_i[mask].tolist(), family_i[mask].tolist(), strict=True):
        degree = degree_map[fam]
        start = len(out_entity)
        out_request.extend(["R4"] * 9)
        out_entity.extend([eid] * 9)
        out_quad.extend(range(9))
        out_degree.extend([degree] * 9)
        blocks.append(("R4", start, start + 9))

    exp_request = np.array(out_request, dtype="<U8")
    exp_entity = np.array(out_entity, dtype=np.int64)
    exp_quad = np.array(out_quad, dtype=np.int64)
    exp_degree = np.array(out_degree, dtype=np.int8)
    return exp_request, exp_entity, exp_quad, exp_degree, blocks


def _check_sequence(label: str, neumann_arrays: dict, exp_entity: np.ndarray, exp_quad: np.ndarray) -> None:
    got_entity = neumann_arrays["entity_id"].astype(np.int64)
    got_quad = neumann_arrays["quad_node"].astype(np.int64)
    if len(got_entity) != len(exp_entity):
        raise ValueError(f"{label}: neumann row count {len(got_entity)} != derived expectation {len(exp_entity)}")
    if len(got_entity):
        mismatch = np.flatnonzero((got_entity != exp_entity) | (got_quad != exp_quad))
        if len(mismatch):
            first = int(mismatch[0])
            raise ValueError(
                f"{label}: neumann (entity_id, quad_node) key sequence disagrees with the builder's "
                "deterministic enumeration derived from the sibling chunk file at row "
                f"{first} (stored ({int(got_entity[first])}, {int(got_quad[first])}) vs "
                f"derived ({int(exp_entity[first])}, {int(exp_quad[first])}))")


# ---------------------------------------------------------------------------
# Verification: bitwise recompute of a deterministic row sample.
# ---------------------------------------------------------------------------
@dataclass
class GeometryContext:
    context: object
    normal_coefficients: object
    census: FaceCensus
    geometry: GeometryArrays
    face_row_indices: np.ndarray
    p07_row_indices: np.ndarray
    p07_ids_by_row_pos: np.ndarray
    patch_cache: dict = field(default_factory=dict)


def build_geometry_context(grid_dir: Path, n: int, input_root: Path, sidecar_path: Path) -> GeometryContext:
    t = load_context(n, str(input_root))
    context = _build_artifact._make_context(t)
    ref = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(sidecar_path), verify_hashes=False).reference
    normal_coefficients = _build_artifact._normal_coefficients_fn(ref)
    census = FaceCensus.load(grid_dir / "census.npz")
    geometry = GeometryArrays.load(grid_dir / "geometry.npz")
    face_row_indices = _build_artifact.face_row_selection(census)
    p07_row_indices = builder.p07_row_selection(census)
    p07_ids_by_row_pos = census.p07_id[p07_row_indices]
    return GeometryContext(context, normal_coefficients, census, geometry,
                           face_row_indices, p07_row_indices, p07_ids_by_row_pos)


def _point_for(stage: str, entity_id: int, quad_node: int, *, geometry_ctx: GeometryContext) -> np.ndarray:
    if stage == "cells":
        return geometry_ctx.context.pts[entity_id]
    if stage == "faces":
        pos = int(np.searchsorted(geometry_ctx.face_row_indices, entity_id))
        if pos >= len(geometry_ctx.face_row_indices) or geometry_ctx.face_row_indices[pos] != entity_id:
            raise ValueError(f"faces entity_id {entity_id} not found in the artifact's face_row_indices")
        return geometry_ctx.geometry.face_points[pos, quad_node]
    if stage == "p07":
        matches = np.flatnonzero(geometry_ctx.p07_ids_by_row_pos == entity_id)
        if len(matches) != 1:
            raise ValueError(f"p07 entity_id {entity_id} does not map to exactly one census row")
        census_row = int(geometry_ctx.p07_row_indices[matches[0]])
        pos = int(np.searchsorted(geometry_ctx.face_row_indices, census_row))
        if pos >= len(geometry_ctx.face_row_indices) or geometry_ctx.face_row_indices[pos] != census_row:
            raise ValueError(f"p07 census row {census_row} not found in the artifact's face_row_indices")
        return geometry_ctx.geometry.face_points[pos, quad_node]
    raise ValueError(f"unknown stage {stage!r}")


def _row_bitwise_diff(recomputed, neumann_arrays: dict, row: int) -> float:
    p0, p1 = int(neumann_arrays["donor_ptr"][row]), int(neumann_arrays["donor_ptr"][row + 1])
    donor_ids = neumann_arrays["donor"][p0:p1].astype(np.int64)
    if not np.array_equal(np.asarray(recomputed.donor_ids, dtype=np.int64), donor_ids):
        return float("inf")
    value = neumann_arrays["value"][p0:p1]
    gradient = neumann_arrays["gradient"][:, p0:p1]
    boundary_value = np.asarray(neumann_arrays["boundary_value"][row], dtype=np.float64)
    boundary_gradient = np.asarray(neumann_arrays["boundary_gradient"][row], dtype=np.float64)
    diffs = [
        float(np.max(np.abs(recomputed.value - value))) if value.size else 0.0,
        float(np.max(np.abs(recomputed.gradient - gradient))) if gradient.size else 0.0,
        float(np.max(np.abs(np.asarray(recomputed.boundary_value, dtype=np.float64) - boundary_value))),
        float(np.max(np.abs(np.asarray(recomputed.boundary_gradient, dtype=np.float64) - boundary_gradient))),
    ]
    return max(diffs)


def _sample_row_indices(blocks: list[tuple[str, int, int]], n_rows: int, sample_cap: int | None) -> list[int]:
    if n_rows == 0:
        return []
    chosen: set[int] = {0, n_rows - 1}
    block_iter = blocks
    if sample_cap is not None and sample_cap >= 0:
        if sample_cap == 0:
            block_iter = []
        else:
            block_iter = blocks[:sample_cap] + blocks[-sample_cap:]
    for _kind, s, e in block_iter:
        chosen.add(s)
        chosen.add(e - 1)
    return sorted(chosen)


def _verify_sample(label: str, stage: str, blocks, neumann_arrays: dict, exp_request: np.ndarray,
                   exp_degree: np.ndarray, key_lookup: dict, *, geometry_ctx: GeometryContext | None,
                   sample_cap: int | None) -> dict:
    n_rows = len(exp_request)
    if n_rows == 0:
        return {"sampled": 0, "max_abs_diff_own_degree": 0.0, "distinguishable_at_other_degree": 0,
                "untestable_same_degree": 0, "no_key_collision": 0, "rows": []}
    if geometry_ctx is None:
        return {"sampled": 0, "skipped_reason": "no geometry inputs supplied (--skip-verification)",
                "max_abs_diff_own_degree": None, "distinguishable_at_other_degree": 0,
                "untestable_same_degree": 0, "no_key_collision": 0, "rows": []}

    sample_indices = _sample_row_indices(blocks, n_rows, sample_cap)
    max_abs_diff = 0.0
    distinguishable = 0
    untestable_same_degree = 0
    no_collision = 0
    rows_report = []
    for row in sample_indices:
        eid = int(neumann_arrays["entity_id"][row])
        q = int(neumann_arrays["quad_node"][row])
        own_kind = str(exp_request[row])
        own_degree = int(exp_degree[row])
        point = _point_for(stage, eid, q, geometry_ctx=geometry_ctx)[None, :]
        (recomputed,) = prepare_neumann_point_rows(
            geometry_ctx.context, point, normal_coefficients=geometry_ctx.normal_coefficients,
            radial_degree=own_degree, patch_cache=geometry_ctx.patch_cache)
        diff = _row_bitwise_diff(recomputed, neumann_arrays, row)
        if diff != 0.0:
            raise ValueError(f"{label}: sample row {row} ({own_kind}, entity_id={eid}, quad_node={q}) "
                              f"recomputed at its own inferred degree {own_degree} does not match the stored "
                              f"row bitwise (max abs diff {diff:.3e})")
        max_abs_diff = max(max_abs_diff, diff)

        other_degree = None
        if stage == "faces":
            counterpart = "R3" if own_kind == "R2" else "R2"
            other_degree = key_lookup.get(eid, {}).get(counterpart)
        if other_degree is None:
            no_collision += 1
        elif other_degree == own_degree:
            untestable_same_degree += 1
        else:
            (recomputed_other,) = prepare_neumann_point_rows(
                geometry_ctx.context, point, normal_coefficients=geometry_ctx.normal_coefficients,
                radial_degree=other_degree, patch_cache=geometry_ctx.patch_cache)
            other_diff = _row_bitwise_diff(recomputed_other, neumann_arrays, row)
            if other_diff == 0.0:
                raise ValueError(f"{label}: sample row {row} is bitwise identical whether recomputed at its own "
                                  f"degree {own_degree} or the other request's degree {other_degree} at the same "
                                  "key -- the two candidate degrees should not coincide when they differ")
            distinguishable += 1
        rows_report.append({"row": row, "request": own_kind, "entity_id": eid, "quad_node": q,
                            "own_degree": own_degree, "other_degree": other_degree})
    return {"sampled": len(sample_indices), "max_abs_diff_own_degree": max_abs_diff,
            "distinguishable_at_other_degree": distinguishable, "untestable_same_degree": untestable_same_degree,
            "no_key_collision": no_collision, "rows": rows_report}


# ---------------------------------------------------------------------------
# Per-file processing: derive -> cross-check -> verify -> (maybe) rewrite.
# ---------------------------------------------------------------------------
@dataclass
class FileResult:
    stage: str
    index: int
    relative: str
    row_count: int
    request_counts: dict
    verification: dict
    rewritten: bool


def process_file(grid_dir: Path, manifest: dict, stage: str, index: int, *,
                 geometry_ctx: GeometryContext | None, sample_cap: int | None, dry_run: bool) -> FileResult:
    neumann_relative = f"rows/neumann_{stage}_{index:05d}.npz"
    neumann_entry = _manifest_entry(manifest, "neumann", neumann_relative)
    neumann_arrays = _read_verified_npz(grid_dir, neumann_entry)
    if "request" in neumann_arrays:
        raise ValueError(f"{neumann_relative}: already carries a request field; refusing to reprocess "
                          "(expected the legacy, pre-tag schema)")

    sibling_relative = f"rows/{stage}_{stage}_{index:05d}.npz"
    sibling_entry = _manifest_entry(manifest, stage, sibling_relative)
    sibling_arrays = _read_verified_npz(grid_dir, sibling_entry)

    label = neumann_relative
    key_lookup: dict[int, dict[str, int]] = {}
    if stage == "cells":
        exp_request, exp_entity, exp_quad, exp_degree, blocks = _derive_cells(sibling_arrays, label=label)
    elif stage == "faces":
        exp_request, exp_entity, exp_quad, exp_degree, blocks = _derive_faces(sibling_arrays, label=label)
        for kind, s, _e in blocks:
            ridx = int(exp_entity[s])
            key_lookup.setdefault(ridx, {})[kind] = int(exp_degree[s])
    elif stage == "p07":
        exp_request, exp_entity, exp_quad, exp_degree, blocks = _derive_p07(sibling_arrays, label=label)
    else:
        raise ValueError(f"unknown stage {stage!r}")

    _check_sequence(label, neumann_arrays, exp_entity, exp_quad)
    verification = _verify_sample(label, stage, blocks, neumann_arrays, exp_request, exp_degree,
                                  key_lookup, geometry_ctx=geometry_ctx, sample_cap=sample_cap)
    request_counts = ({kind: int((exp_request == kind).sum()) for kind in np.unique(exp_request)}
                      if len(exp_request) else {})

    rewritten = False
    if not dry_run:
        new_chunk = artifact_mod.NeumannRowChunk(
            entity_id=neumann_arrays["entity_id"], quad_node=neumann_arrays["quad_node"],
            request=exp_request, radial_degree=exp_degree,
            donor_ptr=neumann_arrays["donor_ptr"], donor=neumann_arrays["donor"],
            value=neumann_arrays["value"], gradient=neumann_arrays["gradient"],
            wall_query=neumann_arrays["wall_query"], boundary_value=neumann_arrays["boundary_value"],
            boundary_gradient=neumann_arrays["boundary_gradient"], condition=neumann_arrays["condition"],
            constraint_residual=neumann_arrays["constraint_residual"], query_table=neumann_arrays["query_table"],
        )
        new_arrays = artifact_mod._neumann_chunk_to_arrays(new_chunk)
        buffer = io.BytesIO()
        np.savez(buffer, **new_arrays)
        data = buffer.getvalue()
        artifact_mod._atomic_write_bytes(grid_dir / neumann_relative, data)
        neumann_entry["sha256"] = artifact_mod.hash_bytes(data)
        rewritten = True

    return FileResult(stage, index, neumann_relative, len(exp_entity), request_counts, verification, rewritten)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="the artifact output root (parent of N{n})")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--input-root", type=Path, default=None, help="required unless --skip-verification")
    p.add_argument("--sidecar", type=Path, default=None, help="required unless --skip-verification")
    p.add_argument("--dry-run", action="store_true", help="do everything except write")
    p.add_argument("--skip-verification", action="store_true",
                   help="UNSAFE structural-only check; only valid together with --dry-run")
    p.add_argument("--stages", type=str, default=None, help="comma-separated subset of cells,faces,p07")
    p.add_argument("--limit-files", type=int, default=None,
                   help="process only the first N neumann files (bounded partial run; forces --dry-run "
                        "semantics for the manifest/schema, see below)")
    p.add_argument("--sample-block-cap", type=int, default=None,
                   help="verify only the first/last N R2/R3/R4 blocks per file, plus the file's global "
                        "first/last row (default: every block)")
    return p.parse_args(argv)


def _build_receipt(results: list[FileResult], *, root, n, dry_run, finalized) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        for kind, c in r.request_counts.items():
            counts[kind] = counts.get(kind, 0) + c
    sampled = [r.verification for r in results if r.verification.get("sampled", 0)]
    max_diff = max((v["max_abs_diff_own_degree"] for v in sampled), default=0.0)
    return {
        "root": str(root), "n": n, "dry_run": dry_run, "finalized": finalized,
        "files_processed": len(results),
        "row_counts_by_request": counts,
        "verification": {
            "sample_size": sum(v.get("sampled", 0) for v in (r.verification for r in results)),
            "max_abs_diff_at_own_degree": max_diff,
            "distinguishable_at_other_degree": sum(r.verification.get("distinguishable_at_other_degree", 0) for r in results),
            "untestable_same_degree_both_candidates": sum(r.verification.get("untestable_same_degree", 0) for r in results),
            "no_key_collision": sum(r.verification.get("no_key_collision", 0) for r in results),
        },
        "files": [{"file": r.relative, "row_count": r.row_count, "request_counts": r.request_counts,
                   "sampled": r.verification.get("sampled", 0),
                   "max_abs_diff_own_degree": r.verification.get("max_abs_diff_own_degree")}
                  for r in results],
        "time": time.time(),
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not args.dry_run and args.skip_verification:
        raise ValueError("--skip-verification may only be combined with --dry-run; "
                          "verification is mandatory before finalizing a backfill")

    grid_dir = Path(args.root).resolve() / f"N{args.n}"
    manifest = _read_manifest(grid_dir)

    stages = set(args.stages.split(",")) if args.stages else None
    all_files = _discover_neumann_files(manifest, stages=None)
    files = _discover_neumann_files(manifest, stages=stages)
    if args.limit_files is not None:
        files = files[:args.limit_files]
    partial = len(files) != len(all_files)
    if partial and not args.dry_run:
        raise ValueError("refusing to finalize: --stages/--limit-files selected only a subset of the "
                          "artifact's neumann files; a real (non-dry-run) backfill must cover every file "
                          "so the manifest schema can be bumped safely")

    geometry_ctx = None
    if not args.skip_verification:
        if args.input_root is None or args.sidecar is None:
            raise ValueError("verification is mandatory; pass --input-root and --sidecar, or explicitly "
                              "opt out with --skip-verification together with --dry-run")
        geometry_ctx = build_geometry_context(grid_dir, args.n, args.input_root.resolve(), args.sidecar.resolve())

    results = []
    for stage, index, _entry in files:
        result = process_file(grid_dir, manifest, stage, index, geometry_ctx=geometry_ctx,
                              sample_cap=args.sample_block_cap, dry_run=args.dry_run)
        results.append(result)
        v = result.verification
        print(f"{result.relative}: rows={result.row_count} by_request={result.request_counts} "
              f"verified={v.get('sampled', 0)} max_diff={v.get('max_abs_diff_own_degree')} "
              f"distinguishable={v.get('distinguishable_at_other_degree', 0)} "
              f"untestable_same_degree={v.get('untestable_same_degree', 0)} "
              f"no_key_collision={v.get('no_key_collision', 0)}", file=sys.stderr)

    finalized = False
    if not args.dry_run:
        manifest["schema"] = CURRENT_SCHEMA
        (grid_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        finalized = True

    receipt = _build_receipt(results, root=args.root, n=args.n, dry_run=args.dry_run, finalized=finalized)
    if finalized:
        (grid_dir / "backfill_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
