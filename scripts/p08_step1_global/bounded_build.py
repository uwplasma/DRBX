"""Bounded (partial) row-artifact builds: geometry run to full completion
(required -- ``GeometryArrays``/``census.npz`` are indexed by every raw id
/face row, so a partial geometry stage is not meaningful), then an
explicitly-chosen, possibly non-contiguous-from-zero subset of cells/faces
/p07 units -- used by this package's ``preflight`` command (a few interior
*and* a few boundary/wall units, plus P07 families 1/2/4) and by local
validation (task 4: "Build fresh N32 units into your scratch folder... via
the existing build functions").

Why this exists rather than calling ``build_artifact.run_full_build`` with
its own ``--max-units``: ``run_full_build`` always assembles *every* planned
unit of every stage after running it (``_assemble``/``_assemble_geometry``
iterate the stage's full ``plan[stage]``, not just the units actually
executed) -- so a ``max_units``-truncated call to it raises
``FileNotFoundError`` at assembly for any unit beyond the truncation, for
both the geometry and row stages (confirmed empirically while building this
package's own local validation artifact). This module instead calls
``build_artifact``'s per-stage primitives directly (``runner.run_stage``
with its own ``_COMPUTE``/``_init_worker``/``_GEOMETRY_COMPUTE``/
``_init_geometry_worker``) and assembles only the units it actually built --
read-only use of ``build_artifact.py``, never an edit to it (off-limits
while another agent parallelizes its geometry stage).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared import runner                      # noqa: E402
from p_shared import build_artifact as ba          # noqa: E402
from drbx.stencils.census import FaceCensus        # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays  # noqa: E402
from drbx.stencils import artifact as artifact_mod  # noqa: E402
from perpendicular_structured.reconstruction import load_context  # noqa: E402


def ensure_geometry(*, n: int, input_root: Path, sidecar_path: Path, output: Path, identity: dict,
                    workers: int, geometry_raw_chunk_size: int, geometry_face_chunk_size: int,
                    max_tasks_per_worker=None) -> None:
    """Run the geometry stage (``geometry_raw`` + ``geometry_face``) to full
    completion if not already present -- resumable exactly like
    ``run_full_build``'s own copy of this logic (it is a deliberate,
    read-only duplication: ``build_artifact.py`` is off-limits to edit)."""
    grid_dir = Path(output) / f"N{n}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    census_path = grid_dir / "census.npz"
    geometry_path = grid_dir / "geometry.npz"
    if census_path.exists() and geometry_path.exists():
        GeometryArrays.load(geometry_path)  # validate, never trust a stale file silently
        return

    if census_path.exists():
        census = FaceCensus.load(census_path)
    else:
        t = load_context(n, str(input_root))
        census = FaceCensus.build(n, t.ro)
        census.save(census_path)

    face_row_indices = ba.face_row_selection(census)
    plan = {
        "geometry_raw": runner.chunk_units("geometry_raw", n, n ** 3, geometry_raw_chunk_size),
        "geometry_face": runner.chunk_units("geometry_face", n, len(face_row_indices), geometry_face_chunk_size),
    }
    initargs = (str(input_root), str(sidecar_path), str(output), n, identity)
    for stage in ("geometry_raw", "geometry_face"):
        runner.run_stage(output, stage, plan[stage], identity, compute=ba._GEOMETRY_COMPUTE[stage],
                         initializer=ba._init_geometry_worker, initargs=initargs, workers=workers,
                         parts=("chunk",), max_tasks_per_worker=max_tasks_per_worker)
    ba._assemble_geometry(output, grid_dir, plan)


def build_units(*, n: int, input_root: Path, sidecar_path: Path, output: Path, identity: dict,
               workers: int, units_by_stage: dict, max_tasks_per_worker=None) -> dict:
    """Build exactly ``units_by_stage`` (``{"cells": [...], "faces": [...],
    "p07": [...]}``, any subset/order of each stage's real build units --
    see :func:`stage_units`) and assemble *only* those into
    ``<output>/N{n}/manifest.json``/``plan.json``. Geometry must already be
    present (see :func:`ensure_geometry`). Returns the same shape of receipt
    ``build_artifact.run_full_build`` does, restricted to what was built."""
    output = Path(output)
    grid_dir = output / f"N{n}"
    identity_path = grid_dir / "build_identity.json"
    if identity_path.exists():
        saved = json.loads(identity_path.read_text())
        if saved != identity:
            raise ValueError("build identity changed for an existing output directory; use a new --output")
    else:
        runner.write_json(identity_path, identity)

    initargs = (str(input_root), str(sidecar_path), str(output), n, identity)
    started = time.time()
    summaries = {}
    for stage, units in units_by_stage.items():
        if not units:
            continue
        summaries[stage] = runner.run_stage(output, stage, units, identity, compute=ba._COMPUTE[stage],
                                            initializer=ba._init_worker, initargs=initargs, workers=workers,
                                            parts=("chunk", "neumann"), max_tasks_per_worker=max_tasks_per_worker)

    plan = {stage: units for stage, units in units_by_stage.items() if units}
    plan_path = grid_dir / "plan.json"
    runner.write_json(plan_path, {"identity": identity, "plan": plan})
    diagnostics = ba._aggregate_receipts(output, plan)
    manifest = ba._assemble(output, grid_dir, identity, plan)
    import shutil
    shutil.rmtree(output / "_chunks", ignore_errors=True)
    loaded = artifact_mod.load_row_artifact(output, n, identity)
    receipt = {
        "n": n, "identity": identity, "wall_seconds": time.time() - started,
        "cpu_seconds": sum(s["worker_seconds"] for s in summaries.values()),
        "peak_rss_gib": max((s["peak_worker_rss_gib"] for s in summaries.values()), default=0.0),
        "stage_summaries": summaries,
        "row_counts": {
            "cells": sum(len(c.request) for c in loaded.cells), "faces": sum(len(c.request) for c in loaded.faces),
            "neumann": sum(len(c.entity_id) for c in loaded.neumann), "p07": sum(len(c.entity_id) for c in loaded.p07),
        },
        "diagnostics": diagnostics, "bounded": True,
    }
    runner.write_json(grid_dir / "build_receipt.json", receipt)
    return receipt


def stage_units(*, n: int, output: Path, cell_chunk_size: int, face_chunk_size: int, p07_chunk_size: int) -> dict:
    """The real, full build-unit lists for ``n`` (identical to what
    ``run_full_build`` would plan) -- callers slice these (a prefix, a
    suffix, or an explicit subset) rather than inventing their own unit
    boundaries, so a bounded build's chunk files are indistinguishable from
    a full build's own (same chunk sizes, same ``{start, stop}`` ranges)."""
    grid_dir = Path(output) / f"N{n}"
    census = FaceCensus.load(grid_dir / "census.npz")
    face_row_indices = ba.face_row_selection(census)
    p07_row_indices = ba.builder.p07_row_selection(census)
    return {
        "cells": runner.chunk_units("cells", n, n ** 3, cell_chunk_size),
        "faces": runner.chunk_units("faces", n, len(face_row_indices), face_chunk_size),
        "p07": runner.chunk_units("p07", n, len(p07_row_indices), p07_chunk_size),
    }


def boundary_and_family_units(*, n: int, output: Path, stage_units_map: dict, tail_count: int = 2) -> dict:
    """For preflight coverage: the *last* ``tail_count`` units of each of
    ``cells``/``faces``/``p07`` (the census's own row order,
    ``radial_block_then_theta_then_eta``, puts the wall/boundary layers --
    and every P07 family, including 1/2/4 -- at the end of each domain, so a
    tail slice is the cheap way to reach them without building the whole
    interior first)."""
    return {stage: units[-tail_count:] if len(units) > tail_count else units
           for stage, units in stage_units_map.items()}


# ---------------------------------------------------------------------------
# Lightweight, subset-only geometry+rows: never touches build_artifact.py's
# chunked geometry_raw/geometry_face stage or its full-grid GeometryArrays
# file. Used by preflight at a grid with no existing (full) geometry.npz --
# "preflight must never require a full geometry stage" (hard constraint:
# never run a full-grid geometry stage locally, so this path builds
# ``GeometryArrays`` only for the handful of raw ids / census rows the
# preflight actually needs, via ``drbx.stencils.builder.build_geometry_arrays``
# called directly with that small subset -- the same function
# ``build_artifact.build_geometry_only``/``_compute_geometry_raw`` calls, just
# never chunked/parallelized/persisted across the whole grid).
# ---------------------------------------------------------------------------
def light_subset_requests(*, n: int, input_root: Path, sidecar_path: Path, census, raw_ids, face_row_indices,
                          p07_row_indices):
    """Build R1/R2/R3/R4 (+ Neumann companion) row requests in-memory for
    exactly ``raw_ids``/``face_row_indices``/``p07_row_indices`` -- no
    on-disk artifact, no chunk files, no full-grid geometry array. Returns
    ``(point_requests, neumann_requests, integrated_requests, geometry)``.
    ``p07_row_indices`` must be a subset of ``face_row_indices`` (or the
    collapsed r=0 rows, which get their own quadrature points -- not
    supported here; callers should restrict to non-collapsed p07 rows,
    which is what every family-1/2/4 case needs anyway)."""
    from p_shared import provider as p_shared_provider
    from p07n_field_derived_global.fields import normal as p07n_normal
    from drbx.stencils import builder
    from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, StructuredReconstruction

    t = load_context(n, str(input_root))
    context = PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro, raw_volume=t.rv,
                                         owner_volume=t.vol, owner_centroid_xy=t.g.owner_centroid_xy,
                                         eta_period=t.g.eta_period, dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)
    S = StructuredReconstruction(context)
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(sidecar_path), verify_hashes=False, curvature="fd")
    ref = provider.reference
    geometry = builder.build_geometry_arrays(provider, context, raw_ids=raw_ids,
                                             face_row_indices=face_row_indices, census=census)

    def normal_coefficients(q):
        return p07n_normal(ref, __import__("numpy").asarray(q, dtype="float64"))

    patch_cache: dict = {}
    point_requests, neumann_requests = builder.build_r1_cell_rows(
        S, context, raw_ids, normal_coefficients=normal_coefficients, patch_cache=patch_cache)
    r2_requests, r2_neumann = builder.build_r2_face_rows(
        S, context, census, face_row_indices, geometry.face_points,
        normal_coefficients=normal_coefficients, patch_cache=patch_cache)
    r3_requests, r3_neumann = builder.build_r3_side_rows(
        S, context, census, face_row_indices, geometry.face_points,
        normal_coefficients=normal_coefficients, patch_cache=patch_cache)
    point_requests = point_requests + r2_requests + r3_requests
    neumann_requests = neumann_requests + r2_neumann + r3_neumann

    import numpy as np
    face_pos = np.searchsorted(face_row_indices, p07_row_indices)
    if not np.array_equal(face_row_indices[face_pos], p07_row_indices):
        raise ValueError("light_subset_requests: p07_row_indices must be a subset of face_row_indices")
    integrated_requests, p07_neumann = builder.build_r4_p07_rows(
        context, census, p07_row_indices, geometry.face_points[face_pos], geometry.p06_face_weight[face_pos],
        geometry.p07_face_tensor[face_pos], normal_coefficients=normal_coefficients, patch_cache=patch_cache)
    neumann_requests = neumann_requests + p07_neumann
    return point_requests, neumann_requests, integrated_requests, geometry, t, context, ref
