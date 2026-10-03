#!/usr/bin/env python3
"""CLI entry point: build the P08 step-1 per-grid row artifact (task 4 --
see ``work/p08_step1_consolidation_design_20260928/design.md`` sections 3
("Row artifact") and 6 (task 4)).

Run from ``DRBX/scripts`` (never from inside a scripts package directory --
``p06n_field_derived_global`` and friends shadow the stdlib ``operator``):

    python -m p_shared.build_artifact --n 32 \\
        --input-root "/Users/yxie/Desktop/HSX drbx" \\
        --sidecar "/Users/yxie/Desktop/HSX drbx/work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json" \\
        --output "/Users/yxie/Desktop/HSX drbx/work/p08_row_artifact_N32_<stamp>" \\
        --workers 6

Produces ``<output>/N{n}/`` with ``manifest.json``, ``census.npz``,
``geometry.npz`` and ``rows/*.npz`` (design section 3's directory layout,
loadable by ``drbx.stencils.artifact.load_row_artifact``), plus this
build's own resumable bookkeeping under ``<output>/N{n}/_chunks`` (per-unit
checkpoints; never read by ``drbx.stencils.artifact``) and
``<output>/N{n}/build_receipt.json`` (wall/CPU time, peak RSS, bytes/row
counts per kind, max residual/condition -- design section 6, task 4).

Every row request is built by :mod:`drbx.stencils.builder` (field-
independent, no ``scripts`` import); this module supplies the concrete
geometry (a loaded ``PointRowContext``, the frozen continuum reference, the
field-independent physical-normal callback) and drives
:mod:`p_shared.runner`'s chunked, resumable process pool.

Geometry stage (parallel)
-------------------------
``GeometryArrays`` (face points, P06 face weights, the P07 face tensor, the
4th-order finite-difference curvature ``K``, ...) used to be built serially
in the controller process (``build_geometry_only``), one call over the
*entire* ``n**3`` raw-cell batch and one over the entire face batch. That
scaled as ``n**3`` and, at N32, took a large share of the ~20-minute build;
at N64 it would dominate. It is now two more :func:`p_shared.runner.run_stage`
stages, ``geometry_raw`` and ``geometry_face``, chunked exactly like
``cells``/``faces``/``p07`` (configurable chunk size, per-unit receipts,
resumable). Each unit calls
:func:`drbx.stencils.geometry_arrays.build_raw_geometry_arrays` /
``build_face_geometry_arrays`` -- the same provider methods, same finite-
difference ``K``, no autodiff, just a sub-batch instead of the whole grid --
and writes its own slice via ``runner.write_unit``. Once every unit is valid,
``_assemble_geometry`` concatenates the slices (in plan order, so it
reproduces the same raw-id / census-row order the single-call serial build
used) into one ``GeometryArrays``, recomputes its identity from the
assembled arrays, and saves it to the same ``geometry.npz`` the row stages
already expect -- ``_init_worker`` (below) loads it exactly as before, and
the on-disk schema is unchanged.

Numerically this is expected to be bitwise identical to the old serial
geometry build for the bulk of the grid, with a residual at the
``1e-13``-relative level on a minority of points: the frozen continuum
reference's ``_metric`` (``hsx_mms_continuum_reference.py``) internally
re-batches any query longer than ``metric_query_batch_size`` (4096) into
fixed-size sub-calls, and P06's face wall-rule (``p06_structured_global.
numerics._face_geometry``) further splits a face batch into a wall/non-wall
boolean partition before calling the metric. A parallel unit's own call is
its own, smaller batch, so it does not always land on the same internal
sub-batch boundaries as the one-shot serial call -- this is pre-existing
batch-size-dependent floating-point roundoff in those evaluators, not a
numerics change here. The default geometry chunk sizes
(``--geometry-raw-chunk-size``/``--geometry-face-chunk-size``, both 4096)
are chosen so most units' start offsets land on a multiple of 4096 raw
points (respectively ``9 * multiple-of-4096`` flattened face-node points),
which reproduces the serial internal chunk boundaries for interior
(non-wall) units and keeps the wall-adjacent residual within the ``1e-12``
relative acceptance checked by the one-off N32 equivalence run (see
``tests/test_p_shared_geometry_stage.py`` for the unit-splitting/assembly/
resume tests; the N32-vs-serial numeric comparison itself is a one-off,
run once per code change -- not part of the automated suite).

Calling the full build
-----------------------
A campaign package (or any other caller) that wants the whole build --
geometry stage, then the cells/faces/p07 row stages, then row-manifest
assembly -- as one call should use :func:`run_full_build`::

    from p_shared import build_artifact

    receipt = build_artifact.run_full_build(
        n=64,
        input_root=Path("/Users/yxie/Desktop/HSX drbx"),
        sidecar_path=Path(".../localized_sidecar.json"),
        output=Path(".../p08_row_artifact_N64_<stamp>"),
        workers=32,
        memory_budget_gib=200.0,       # optional; None skips the cap
        worker_memory_gib=4.0,         # required together with memory_budget_gib
        memory_reserve_gib=4.0,        # default 1.0
        cell_chunk_size=4096, face_chunk_size=2048, p07_chunk_size=2048,
        geometry_raw_chunk_size=4096, geometry_face_chunk_size=4096,
        max_tasks_per_worker=None, max_units=None,
    )

Returns the same dict written to ``build_receipt.json``. ``main()``/the CLI
below is a thin argument-parsing wrapper around this one function; every
existing CLI flag keeps its old name, default and meaning.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = Path(__file__).resolve().parent           # .../DRBX/scripts/p_shared
SCRIPTS = HERE.parent                             # .../DRBX/scripts
REPO = SCRIPTS.parent                             # .../DRBX
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import runner                                       # noqa: E402
from p_shared import provider as p_shared_provider                 # noqa: E402
from p_shared.bfield import (                        # noqa: E402
    BFIELD_TOROIDAL_CHOICES, DEFAULT_BFIELD_TOROIDAL, check_bfield_toroidal)
from p_shared.curvature_reference import check_curvature           # noqa: E402
from p_shared.curvature_reference import DEFAULT_CURVATURE  # noqa: E402
from p_shared.face_quadrature import (               # noqa: E402
    DEFAULT_FACE_QUADRATURE, FACE_QUADRATURE_CHOICES, check_face_quadrature, face_order as _face_order)
from p_shared.inner_support import (                 # noqa: E402
    DEFAULT_INNER_SUPPORT, INNER_SUPPORT_CHOICES, check_inner_support)
from p07_diffusion_global.numerics import quadrature as _p07_quadrature  # noqa: E402
from perpendicular_structured.reconstruction import load_context   # noqa: E402
from p07n_field_derived_global.fields import normal as _p07n_normal  # noqa: E402

from drbx.geometry.fci_perpendicular_reconstruction import (        # noqa: E402
    CELL_STENCILS, PointRowContext, StructuredReconstruction)
from drbx.stencils.census import FaceCensus          # noqa: E402
from drbx.stencils.geometry_arrays import (            # noqa: E402
    GeometryArrays, SCHEMA as GEOMETRY_SCHEMA,
    build_raw_geometry_arrays, build_face_geometry_arrays)
from drbx.stencils import builder                     # noqa: E402
from drbx.stencils import artifact as artifact_mod      # noqa: E402

# The raw/face field-name split of GeometryArrays._ARRAY_FIELDS (kept as a
# literal tuple here, not imported, since it is part of the on-disk schema
# this module assembles into -- a change there should be a deliberate,
# visible edit in both places, not a silent import-time rename).
RAW_GEOMETRY_FIELDS = (
    "raw_points", "p05_raw_h", "p05_raw_jacobian", "p06_raw_J", "p06_raw_B",
    "p06_raw_K", "p06_raw_weight", "p07_raw_tensor", "p07_raw_divergence",
)
FACE_GEOMETRY_FIELDS = (
    "face_points", "p05_face_h", "p05_face_jacobian", "p06_face_J",
    "p06_face_B", "p06_face_K", "p06_face_weight", "p07_face_tensor",
)
#: the additional face fields of a ``face_quadrature="q2"`` build (the P07 q3 set; see ``GeometryArrays``)
FACE_GEOMETRY_FIELDS_Q2 = FACE_GEOMETRY_FIELDS + ("p07_face_points", "p07_face_weight")

GEOMETRY_SUBDIR = "geometry_artifacts/rlp_convergence_32_48_64_20260917"

POLICY = {
    "svd_threshold": 1.0e-4,
    "residual_tolerance": 1.0e-9,
    "quadrature": {"raw": "q1", "face": "q3"},
    "wall_threshold_i_ge_n_minus_2": True,
    "centered_radial_threshold_i_ge_n_minus_6": True,
    "neumann_max_condition": 1.0e8,
    "neumann_degree_wall_quartic": 4,
    "neumann_degree_transverse_or_side_cubic": 3,
    "p07_neumann_degree_by_family": {"1": 4, "2": 4, "4": 3},
    "seam_policy": "p06_slot_space_dedupe_theta_eta_slot_n_alias",
    "census_row_order": "radial_block_then_theta_then_eta_C_order",
    # storage only (decoded rows are bit-identical either way): unconditioned singleton/ringwise/centered_radial
    # sources are written as tensor factors, each verified by bitwise expansion, else CSR (design section 5-6)
    "point_row_encoding": "v3_tensor_factored_unconditioned_with_csr_fallback",
}

# Every builder/provider source contributing to this artifact's arithmetic
# (design section 3, "Identity": "blob hashes of every builder/provider
# source"), relative to REPO (DRBX/). Explicitly includes
# src/drbx/native/fci_operators.py: "P06's wall-state function lives there"
# (not called by this build, but its identity must still be pinned, per the
# task instructions).
SOURCE_FILES = [
    "src/drbx/stencils/builder.py",
    "src/drbx/stencils/census.py",
    "src/drbx/stencils/geometry_arrays.py",
    "src/drbx/stencils/artifact.py",
    "src/drbx/stencils/tensor_rows.py",
    "src/drbx/geometry/fci_perpendicular_reconstruction.py",
    "src/drbx/geometry/fci_perpendicular_neumann_trace.py",
    "src/drbx/geometry/fci_perpendicular_integrated_rows.py",
    "src/drbx/geometry/_fci_perpendicular_point_primitives.py",
    "src/drbx/native/fci_operators.py",
    "scripts/p_shared/provider.py",
    "scripts/p_shared/runner.py",
    "scripts/p_shared/build_artifact.py",
    "scripts/p07n_field_derived_global/fields.py",
]


# ---------------------------------------------------------------------------
# Identity.
# ---------------------------------------------------------------------------
def check_cell_stencil(cell_stencil: str) -> str:
    if cell_stencil not in CELL_STENCILS:
        raise ValueError(f"cell_stencil must be one of {CELL_STENCILS}, got {cell_stencil!r}")
    return cell_stencil


def _geometry_component_hashes(input_root: Path, n: int) -> dict:
    geometry_dir = Path(input_root) / GEOMETRY_SUBDIR / f"{n}x{n}x{n}"
    hashes = {}
    for name in ("base_geometry.npz", "rlp_topology.npz"):
        hashes[f"geometry:{name}"] = artifact_mod.hash_file(geometry_dir / name)
    return hashes


def _sidecar_component_hashes(sidecar_path: Path) -> dict:
    sidecar = json.loads(Path(sidecar_path).read_text())
    hashes = {"sidecar": artifact_mod.hash_file(sidecar_path)}
    hashes["metric_cache"] = artifact_mod.hash_file(sidecar["metric_cache"]["path"])
    hashes["makegrid"] = artifact_mod.hash_file(sidecar["makegrid"]["path"])
    manifest = Path(sidecar["artifact"]["path"]) / "manifest.json"
    if manifest.is_file():
        hashes["makegrid_artifact_manifest"] = artifact_mod.hash_file(manifest)
    return hashes


#: extra sources pinned only for ``curvature="autodiff"`` builds (the ``"fd"`` identity is unchanged)
AUTODIFF_SOURCE_FILES = [
    "src/drbx/geometry/curvature_autodiff.py",
    "scripts/p_shared/curvature_reference.py",
]


#: extra source pinned only for ``face_quadrature="q2"`` builds (the ``"q3"`` identity is unchanged)
Q2_SOURCE_FILES = [
    "scripts/p_shared/face_quadrature.py",
]


#: extra source pinned only for ``inner_support != "profile7"`` builds (the ``"profile7"`` identity is unchanged)
INNER_SUPPORT_SOURCE_FILES = [
    "scripts/p_shared/inner_support.py",
]


#: extra sources pinned only for ``bfield_toroidal != "spline"`` builds (the ``"spline"`` identity is unchanged)
BFIELD_SOURCE_FILES = [
    "scripts/p_shared/bfield.py",
    "src/drbx/geometry/Bfield_evaluator.py",
    "src/drbx/geometry/compact_toroidal.py",
    "src/drbx/geometry/jax_bfield_evaluator.py",
]


def build_policy(curvature: str = DEFAULT_CURVATURE, face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                 inner_support: str = DEFAULT_INNER_SUPPORT, bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL,
                 cell_stencil: str = "biased") -> dict:
    """``POLICY`` for ``curvature="fd"`` and ``face_quadrature="q3"`` (exactly, so existing identities are
    unchanged); with ``"autodiff"`` the same policy plus ``curvature`` (distinct identity); with ``"q2"`` the
    P05/P06 face rule is recorded as ``quadrature = {"raw": "q1", "face": "q2", "p07_face": "q3"}``."""
    check_curvature(curvature)
    check_face_quadrature(face_quadrature)
    check_inner_support(inner_support)
    check_bfield_toroidal(bfield_toroidal)
    check_cell_stencil(cell_stencil)
    policy = dict(POLICY) if curvature == "fd" else {**POLICY, "curvature": "autodiff"}
    if face_quadrature != "q3":
        policy["quadrature"] = {"raw": "q1", "face": face_quadrature, "p07_face": "q3"}
    if inner_support != "profile7":        # the literal, not the default: the default may move, the frozen identity may not
        policy["inner_support"] = inner_support
    if bfield_toroidal != "spline":        # likewise the literal: the frozen spline identity must not move
        policy["bfield_toroidal"] = bfield_toroidal
    if cell_stencil != "biased":           # likewise the literal: the frozen biased identity must not move
        policy["cell_stencil"] = cell_stencil
    return policy


def build_identity(*, n: int, input_root: Path, sidecar_path: Path, curvature: str = DEFAULT_CURVATURE,
                   face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                   inner_support: str = DEFAULT_INNER_SUPPORT,
                   bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL,
                   cell_stencil: str = "biased") -> dict:
    component_hashes = {**_geometry_component_hashes(input_root, n), **_sidecar_component_hashes(sidecar_path)}
    sources = (SOURCE_FILES + (AUTODIFF_SOURCE_FILES if curvature != "fd" else [])
               + (Q2_SOURCE_FILES if face_quadrature != "q3" else [])
               + (INNER_SUPPORT_SOURCE_FILES if inner_support != "profile7" else [])
               + (BFIELD_SOURCE_FILES if bfield_toroidal != "spline" else []))
    source_hashes = {rel: artifact_mod.hash_file(REPO / rel) for rel in sources}
    return artifact_mod.build_identity(component_hashes=component_hashes, source_hashes=source_hashes,
                                       policy=build_policy(curvature, face_quadrature, inner_support,
                                                                           bfield_toroidal, cell_stencil))


# ---------------------------------------------------------------------------
# Geometry-only, one-shot build products: PointRowContext / FaceCensus / GeometryArrays.
# ---------------------------------------------------------------------------
def _make_context(t) -> PointRowContext:
    return PointRowContext.from_arrays(faces=t.faces, centers=t.centers, raw_to_owner=t.ro,
                                       raw_volume=t.rv, owner_volume=t.vol,
                                       owner_centroid_xy=t.g.owner_centroid_xy, eta_period=t.g.eta_period,
                                       dr=t.g.dr, dtheta=t.g.dtheta, deta=t.g.deta)


def _normal_coefficients_fn(ref):
    """The field-independent physical outward normal, bitwise identical to
    ``p07n_field_derived_global.fields.normal`` (used directly, not
    re-derived -- see that module for the one-line ``gcontra`` formula)."""
    def normal_coefficients(q):
        return _p07n_normal(ref, np.asarray(q, dtype=np.float64))
    return normal_coefficients


def face_row_selection(census: FaceCensus) -> np.ndarray:
    """R2/R3 census rows: every slot except the collapsed r=0 face and the
    periodic alias slots (design section 2: "collapsed r=0 and alias slots
    excluded; internal faces included")."""
    return np.flatnonzero(~(census.collapsed_r0 | census.legacy_alias_slots))


def build_geometry_only(*, n: int, input_root: Path, sidecar_path: Path, grid_dir: Path, curvature: str = DEFAULT_CURVATURE,
                        face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                        bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL):
    """Load ``t``/build ``context``/``census``/``GeometryArrays`` once, all
    in this single process (no parallelism) -- the original, still-supported
    one-shot geometry build. Saves ``census.npz``/``geometry.npz`` under
    ``grid_dir`` and returns ``(t, context, census, geometry)``.

    Kept for smoke-testing and for callers that want the whole geometry
    build in one call without a process pool; :func:`run_full_build` (this
    module's real entry point for a full N32/N48/N64 build) uses the
    parallel ``geometry_raw``/``geometry_face`` stages below instead -- see
    this module's docstring.
    """
    t = load_context(n, str(input_root))
    context = _make_context(t)
    census = FaceCensus.build(n, t.ro)
    census.save(grid_dir / "census.npz")

    face_row_indices = face_row_selection(census)
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(sidecar_path), verify_hashes=False,
                                                                     curvature=curvature,
                                                                     face_quadrature=face_quadrature,
                                                                     bfield_toroidal=bfield_toroidal)
    extra = {} if face_quadrature == "q3" else {"face_order": _face_order(face_quadrature)}
    geometry = builder.build_geometry_arrays(provider, context, raw_ids=np.arange(n ** 3, dtype=np.int64),
                                             face_row_indices=face_row_indices, census=census, **extra)
    geometry.save(grid_dir / "geometry.npz")
    return t, context, census, geometry


# ---------------------------------------------------------------------------
# Geometry stage (parallel): geometry_raw / geometry_face units through
# runner.run_stage, assembled into the same geometry.npz build_geometry_only
# used to write serially -- same GeometryArrays schema, same provider
# methods, same finite-difference K, no autodiff (see this module's
# docstring, "Geometry stage (parallel)").
# ---------------------------------------------------------------------------
GEOMETRY_STATE: dict = {}


def _init_geometry_worker(input_root: str, sidecar_path: str, output: str, n: int, identity: dict,
                          curvature: str = DEFAULT_CURVATURE, face_quadrature: str = DEFAULT_FACE_QUADRATURE,
                          bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL):
    global GEOMETRY_STATE
    runner.require_cpu_backend()
    grid_dir = Path(output) / f"N{n}"
    t = load_context(n, str(input_root))
    census = FaceCensus.load(grid_dir / "census.npz")
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(sidecar_path, verify_hashes=False,
                                                                     curvature=curvature,
                                                                     face_quadrature=face_quadrature,
                                                                     bfield_toroidal=bfield_toroidal)
    GEOMETRY_STATE = {
        "faces": t.faces, "census": census, "provider": provider,
        "face_row_indices": face_row_selection(census), "n": n,
        "output": Path(output), "identity": identity, "face_order": _face_order(face_quadrature),
    }


def _compute_geometry_raw(unit: dict) -> dict:
    s = GEOMETRY_STATE
    _require_free_disk(s["output"])
    started = time.time()
    n = s["n"]
    raw_ids = np.arange(unit["start"], unit["stop"], dtype=np.int64)
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    arrays = build_raw_geometry_arrays(s["provider"], s["faces"], raw_keys)
    return runner.write_unit(s["output"], unit, s["identity"], chunks={"chunk": arrays},
                              started=started, extra={"count": int(len(raw_ids))})


def _compute_geometry_face(unit: dict) -> dict:
    s = GEOMETRY_STATE
    _require_free_disk(s["output"])
    started = time.time()
    sl = slice(unit["start"], unit["stop"])
    row_indices = s["face_row_indices"][sl]
    face_keys = builder.census_face_keys(s["census"], row_indices)
    face_order = s.get("face_order", 3)                    # (test states set up by hand have no "face_order")
    arrays = (build_face_geometry_arrays(s["provider"], s["faces"], face_keys) if face_order == 3
              else build_face_geometry_arrays(s["provider"], s["faces"], face_keys, face_order))
    return runner.write_unit(s["output"], unit, s["identity"], chunks={"chunk": arrays},
                              started=started, extra={"count": int(len(row_indices))})


_GEOMETRY_COMPUTE = {"geometry_raw": _compute_geometry_raw, "geometry_face": _compute_geometry_face}


def _assemble_geometry(output: Path, grid_dir: Path, plan: dict, face_order: int = 3) -> GeometryArrays:
    """Concatenate every geometry_raw/geometry_face unit's own slice, in
    plan order -- the same raw-id order (``np.arange(n**3)``) and the same
    face-row order (``face_row_selection(census)``) the one-shot serial
    build used, so this reproduces its array layout exactly -- into one
    ``GeometryArrays``, recompute its identity from the assembled arrays,
    and save it to ``geometry.npz`` (same path/schema
    ``_init_worker``/``build_geometry_only`` already read). ``face_order`` 2 (the ``q2`` face rule) adds the
    P07 q3 face set (``p07_face_points`` / ``p07_face_weight``) to the assembled arrays."""
    def _load(stage_units, fields):
        parts = []
        for unit in stage_units:
            with np.load(runner.unit_path(output, unit, "chunk")) as data:
                parts.append({name: np.asarray(data[name]) for name in fields})
        return {name: np.concatenate([p[name] for p in parts], axis=0) for name in fields}

    arrays = {**_load(plan["geometry_raw"], RAW_GEOMETRY_FIELDS),
              **_load(plan["geometry_face"], FACE_GEOMETRY_FIELDS if face_order == 3 else FACE_GEOMETRY_FIELDS_Q2)}
    if face_order != 3:
        arrays["face_order"] = face_order
    identity = GeometryArrays._compute_identity(arrays)
    geometry = GeometryArrays(schema=GEOMETRY_SCHEMA, identity=identity, **arrays)
    geometry.save(grid_dir / "geometry.npz")
    return geometry


# ---------------------------------------------------------------------------
# Worker state (spawned process pool; module-level so `spawn` can pickle
# just the (function, args) pair, per p05n_field_derived_global/campaign.py's
# own convention).
# ---------------------------------------------------------------------------
STATE: dict = {}


def _row_diagnostics_summary(point_rows) -> dict:
    families: dict[str, int] = {}
    max_residual = 0.0
    for req in point_rows:
        fam = str(req.row.diagnostics.get("family", ""))
        families[fam] = families.get(fam, 0) + 1
        max_residual = max(max_residual, float(req.row.diagnostics.get("max_residual", 0.0) or 0.0))
    return {"families": families, "max_residual": max_residual}


def _neumann_diagnostics_summary(neumann_rows) -> dict:
    max_condition = 0.0
    max_residual = 0.0
    for req in neumann_rows:
        max_condition = max(max_condition, float(req.row.condition))
        max_residual = max(max_residual, float(req.row.constraint_residual))
    return {"count": len(neumann_rows), "max_condition": max_condition, "max_residual": max_residual}


def _integrated_diagnostics_summary(integrated_rows) -> dict:
    families: dict[str, int] = {}
    for req in integrated_rows:
        key = str(int(req.row.family))
        families[key] = families.get(key, 0) + 1
    return {"families": families}


def _init_worker(input_root: str, sidecar_path: str, output: str, n: int, identity: dict, curvature: str = DEFAULT_CURVATURE,
                 inner_support: str = DEFAULT_INNER_SUPPORT, bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL,
                 cell_stencil: str = "biased"):
    global STATE
    runner.require_cpu_backend()
    grid_dir = Path(output) / f"N{n}"
    t = load_context(n, str(input_root))
    context = _make_context(t)
    S = (StructuredReconstruction(context) if inner_support == "profile7" and cell_stencil == "biased"
         else StructuredReconstruction(context, inner_support=inner_support, cell_stencil=cell_stencil))
    census = FaceCensus.load(grid_dir / "census.npz")
    geometry = GeometryArrays.load(grid_dir / "geometry.npz")
    ref = p_shared_provider.ScriptsGeometryProvider.from_sidecar(sidecar_path, verify_hashes=False,
                                                                 curvature=curvature,
                                                                 bfield_toroidal=bfield_toroidal).reference
    normal_coefficients = _normal_coefficients_fn(ref)

    face_row_indices = face_row_selection(census)
    p07_row_indices = builder.p07_row_selection(census)
    # P07 family 0 is the collapsed r=0 face, which R2/R3 (and so the face
    # geometry arrays) exclude. Those rows get their own quadrature points with
    # a zero integrand, exactly as p07n_field_derived_global.core.face_chunk does
    # (it fills the integrand only where family != 0). Every other P07 row must
    # be an R2/R3 row, so its geometry can be taken from the face arrays.
    p07_collapsed = census.collapsed_r0[p07_row_indices]
    p07_to_face_pos = np.full(len(p07_row_indices), -1, dtype=np.int64)
    regular = ~p07_collapsed
    p07_to_face_pos[regular] = np.searchsorted(face_row_indices, p07_row_indices[regular])
    if not np.array_equal(face_row_indices[p07_to_face_pos[regular]], p07_row_indices[regular]):
        raise ValueError("P07 census rows (other than collapsed r=0) are not a subset of the R2/R3 face selection")

    STATE = {
        "t": t, "context": context, "S": S, "census": census, "geometry": geometry, "n": n,
        "normal_coefficients": normal_coefficients, "patch_cache": {},
        "face_row_indices": face_row_indices, "p07_row_indices": p07_row_indices,
        "p07_to_face_pos": p07_to_face_pos, "output": Path(output), "identity": identity,
        "inner_support": inner_support, "cell_stencil": cell_stencil,
    }


MIN_FREE_GIB_PER_UNIT = 2.0

#: set to a non-empty, non-"0" value (inherited by the spawned workers) to write plain CSR point chunks
CSR_ONLY_ENV = "P_SHARED_CSR_ONLY"


def tensor_encoding_enabled() -> bool:
    return os.environ.get(CSR_ONLY_ENV, "") in ("", "0")


def _pack_point_rows(point_rows):
    """Pack tagged R1/R2/R3 ``PointRowRequest``s into one ``PointRowChunk``. R3 side rows are consumed as
    values only (P05N/P06N batched_side_values), so their gradient weights are not stored (design section 3):
    they were 57% of the face-row bytes in the first full N32 build."""
    return artifact_mod.pack_point_rows(
        [r.row for r in point_rows], request=[r.request for r in point_rows],
        entity_id=[r.entity_id for r in point_rows], bc_variant=[r.bc_variant for r in point_rows],
        radial_degree=[r.radial_degree for r in point_rows],
        store_gradient=[not str(r.request).startswith("R3") for r in point_rows])


def _point_chunk_arrays(point_chunk, point_rows, *, tensor: bool) -> tuple[dict, dict]:
    """The v3 members of a packed point chunk, storing the unconditioned singleton / ringwise /
    centered_radial sources as verified tensor factors when ``tensor`` (the rows need to have been built with
    ``capture_factors=True``), plus the ``artifact.py`` tensor/fallback counts."""
    stats: dict = {}
    factors = [r.factors for r in point_rows] if tensor else None
    return artifact_mod._point_chunk_to_arrays(point_chunk, factors=factors, stats=stats), stats


def _require_free_disk(output) -> None:
    """Stop a worker before it writes a chunk onto a nearly full volume."""
    free = shutil.disk_usage(output).free / 2**30
    if free < MIN_FREE_GIB_PER_UNIT:
        raise RuntimeError(f"only {free:.2f} GiB free on the output volume; stopping before writing another chunk")


def _compute_cells(unit: dict) -> dict:
    s = STATE
    _require_free_disk(s["output"])
    started = time.time()
    raw_ids = np.arange(unit["start"], unit["stop"], dtype=np.int64)
    tensor = tensor_encoding_enabled()
    point_rows, neumann_rows = builder.build_r1_cell_rows(
        s["S"], s["context"], raw_ids,
        normal_coefficients=s["normal_coefficients"], patch_cache=s["patch_cache"], capture_factors=tensor)
    point_chunk = _pack_point_rows(point_rows)
    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    point_arrays, tensor_stats = _point_chunk_arrays(point_chunk, point_rows, tensor=tensor)
    chunks = {"chunk": point_arrays, "neumann": artifact_mod._neumann_chunk_to_arrays(neumann_chunk)}
    extra = {"point_diagnostics": _row_diagnostics_summary(point_rows),
             "neumann_diagnostics": _neumann_diagnostics_summary(neumann_rows),
             "point_row_count": len(point_rows), "neumann_row_count": len(neumann_rows),
             "tensor_encoding": tensor_stats}
    return runner.write_unit(s["output"], unit, s["identity"], chunks=chunks, started=started, extra=extra)


def _compute_faces(unit: dict) -> dict:
    s = STATE
    _require_free_disk(s["output"])
    started = time.time()
    sl = slice(unit["start"], unit["stop"])
    row_indices = s["face_row_indices"][sl]
    face_points = s["geometry"].face_points[sl]
    tensor = tensor_encoding_enabled()
    point_rows_2, neumann_rows_2 = builder.build_r2_face_rows(
        s["S"], s["context"], s["census"], row_indices, face_points,
        normal_coefficients=s["normal_coefficients"], patch_cache=s["patch_cache"], capture_factors=tensor)
    point_rows_3, neumann_rows_3 = builder.build_r3_side_rows(
        s["S"], s["context"], s["census"], row_indices, face_points,
        normal_coefficients=s["normal_coefficients"], patch_cache=s["patch_cache"], capture_factors=tensor)
    point_rows = point_rows_2 + point_rows_3
    neumann_rows = neumann_rows_2 + neumann_rows_3
    point_chunk = _pack_point_rows(point_rows)
    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    point_arrays, tensor_stats = _point_chunk_arrays(point_chunk, point_rows, tensor=tensor)
    chunks = {"chunk": point_arrays, "neumann": artifact_mod._neumann_chunk_to_arrays(neumann_chunk)}
    extra = {"point_diagnostics": _row_diagnostics_summary(point_rows),
             "neumann_diagnostics": _neumann_diagnostics_summary(neumann_rows),
             "point_row_count": len(point_rows), "neumann_row_count": len(neumann_rows),
             "tensor_encoding": tensor_stats}
    return runner.write_unit(s["output"], unit, s["identity"], chunks=chunks, started=started, extra=extra)


def _compute_p07(unit: dict) -> dict:
    s = STATE
    _require_free_disk(s["output"])
    started = time.time()
    sl = slice(unit["start"], unit["stop"])
    row_indices = s["p07_row_indices"][sl]
    face_pos = s["p07_to_face_pos"][sl]
    geometry = s["geometry"]
    regular = face_pos >= 0
    count = len(row_indices)
    p07_points, p07_weight = geometry.p07_points, geometry.p07_weight      # the q3 set (= the face set at q3)
    face_points = np.empty((count,) + p07_points.shape[1:], dtype=p07_points.dtype)
    face_weight = np.zeros((count,) + p07_weight.shape[1:], dtype=p07_weight.dtype)
    face_tensor = np.zeros((count,) + geometry.p07_face_tensor.shape[1:], dtype=geometry.p07_face_tensor.dtype)
    face_points[regular] = p07_points[face_pos[regular]]
    face_weight[regular] = p07_weight[face_pos[regular]]
    face_tensor[regular] = geometry.p07_face_tensor[face_pos[regular]]
    if np.any(~regular):
        collapsed_keys = s["census"].keys()[row_indices[~regular]]
        points, _weights = _p07_quadrature(s["t"].faces, collapsed_keys, 3, face=True)
        face_points[~regular] = points
    integrated_rows, neumann_rows = builder.build_r4_p07_rows(
        s["context"], s["census"], row_indices, face_points, face_weight, face_tensor,
        normal_coefficients=s["normal_coefficients"], patch_cache=s["patch_cache"],
        **({} if s.get("inner_support", "profile7") == "profile7" else {"inner_support": s["inner_support"]}))
    integrated_chunk = artifact_mod.pack_integrated_rows(
        [r.row for r in integrated_rows], entity_id=[r.entity_id for r in integrated_rows])
    neumann_chunk = artifact_mod.pack_neumann_rows(
        [r.row for r in neumann_rows], entity_id=[r.entity_id for r in neumann_rows],
        quad_node=[r.quad_node for r in neumann_rows], request=[r.source for r in neumann_rows],
        radial_degree=[r.radial_degree for r in neumann_rows])
    chunks = {"chunk": artifact_mod._integrated_chunk_to_arrays(integrated_chunk),
              "neumann": artifact_mod._neumann_chunk_to_arrays(neumann_chunk)}
    extra = {"integrated_diagnostics": _integrated_diagnostics_summary(integrated_rows),
             "neumann_diagnostics": _neumann_diagnostics_summary(neumann_rows),
             "integrated_row_count": len(integrated_rows), "neumann_row_count": len(neumann_rows)}
    return runner.write_unit(s["output"], unit, s["identity"], chunks=chunks, started=started, extra=extra)


_COMPUTE = {"cells": _compute_cells, "faces": _compute_faces, "p07": _compute_p07}


# ---------------------------------------------------------------------------
# Manifest assembly: rows/{group}_<index>.npz + manifest.json, without ever
# holding the whole grid's chunks in memory (each unit's already-written,
# already-sha256'd file is moved -- not copied -- into its final `rows/`
# name, so this step never doubles the artifact's disk footprint). The
# now-empty ``_chunks/`` scratch tree is removed once every file has been
# moved and the manifest is written.
# ---------------------------------------------------------------------------
def _chunk_stats(group: str, path: Path) -> dict:
    """The v3 manifest entry's ``sources``/``targets``/``bytes`` for one moved chunk
    file (index member only). A file without the row-chunk index members -- only the
    re-entry test harness's stand-in files -- records just its size."""
    try:
        return artifact_mod.chunk_file_stats(group, path)
    except KeyError:
        return {"bytes": path.stat().st_size}


def _assemble(output: Path, grid_dir: Path, identity: dict, plan: dict) -> dict:
    rows_dir = grid_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    manifest_chunks: dict[str, list] = {"cells": [], "faces": [], "neumann": [], "p07": []}
    group_by_stage_part = {
        ("cells", "chunk"): "cells", ("cells", "neumann"): "neumann",
        ("faces", "chunk"): "faces", ("faces", "neumann"): "neumann",
        ("p07", "chunk"): "p07", ("p07", "neumann"): "neumann",
    }
    for stage in ("cells", "faces", "p07"):
        units = plan[stage]
        for index, unit in enumerate(units):
            for part in ("chunk", "neumann"):
                group = group_by_stage_part[(stage, part)]
                src = runner.unit_path(output, unit, part)
                relative = f"rows/{group}_{stage}_{index:05d}.npz"
                dest = grid_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                manifest_chunks[group].append({"file": relative, "sha256": runner.sha256_file(dest),
                                               **_chunk_stats(group, dest)})
    manifest = {"schema": artifact_mod.SCHEMA, "identity": artifact_mod._json_safe(identity), "chunks": manifest_chunks}
    (grid_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def _verify_assembled(output: Path, grid_dir: Path, identity: dict, plan: dict) -> tuple[dict | None, str | None, bool]:
    """Check whether ``grid_dir`` already holds a finished, intact build for
    ``identity``/``plan``: ``build_receipt.json`` with this identity,
    ``geometry.npz`` whose sha256 equals the receipt's ``geometry_sha256``,
    and a manifest (this identity and schema, one entry per planned unit
    part) whose every row file matches its recorded sha256 -- checked by
    streaming hashes, never by decoding the rows (N64 rows do not fit in
    memory). Returns ``(receipt, reason, geometry_mismatch)``: the receipt
    when everything verifies; ``(None, None, False)`` when there is nothing
    to verify (no receipt, or a bounded local-testing build's receipt, which
    is not a finished grid); otherwise ``reason`` names the first failed
    check and ``geometry_mismatch`` says ``geometry.npz`` itself disagrees
    with the receipt."""
    receipt_path = grid_dir / "build_receipt.json"
    manifest_path = grid_dir / "manifest.json"
    geometry_path = grid_dir / "geometry.npz"
    if not receipt_path.exists():
        return None, None, False
    try:
        receipt = json.loads(receipt_path.read_text())
    except ValueError:
        return None, "build_receipt.json is unreadable", False
    if receipt.get("bounded") or "geometry_sha256" not in receipt:
        return None, None, False
    if receipt.get("identity") != json.loads(json.dumps(identity, default=runner._json_default)):
        return None, "build_receipt.json identity differs from this build", False
    if not geometry_path.exists() or runner.sha256_file(geometry_path) != receipt["geometry_sha256"]:
        return None, "geometry.npz is missing or does not match the receipt's geometry_sha256", True
    if not manifest_path.exists():
        return None, "manifest.json is missing", False
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError:
        return None, "manifest.json is unreadable", False
    if manifest.get("schema") != artifact_mod.SCHEMA or manifest.get("identity") != artifact_mod._json_safe(identity):
        return None, "manifest.json schema/identity differs from this build", False
    expected = {"cells": len(plan["cells"]), "faces": len(plan["faces"]), "p07": len(plan["p07"])}
    expected["neumann"] = sum(expected.values())
    chunks = manifest.get("chunks", {})
    if {group: len(chunks.get(group, ())) for group in expected} != expected:
        return None, "manifest.json chunk counts differ from the plan", False
    for entries in chunks.values():
        for entry in entries:
            path = grid_dir / entry["file"]
            if not path.exists() or runner.sha256_file(path) != entry["sha256"]:
                return None, f"row file {entry['file']} is missing or corrupted", False
    return receipt, None, False


def _disk_free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (2 ** 30)


def _aggregate_receipts(output: Path, plan: dict) -> dict:
    """Fold every unit's ``extra`` diagnostics (written by ``_compute_*``,
    still on disk under ``_chunks/`` at this point) into one build-wide
    summary: row counts, family histograms, and max residual/condition."""
    families: dict[str, int] = {}
    p07_families: dict[str, int] = {}
    max_point_residual = 0.0
    max_neumann_residual = 0.0
    max_condition = 0.0
    point_rows = neumann_rows = integrated_rows = 0
    tensor_encoding: dict = {}
    # Geometry units carry no row diagnostics, and their receipts are gone
    # once a previous build has assembled (or a caller pruned) ``_chunks/``
    # -- never require them here.
    for stage in ("cells", "faces", "p07"):
        for unit in plan.get(stage, ()):
            item = json.loads(runner.receipt_path(output, unit).read_text())
            pd = item.get("point_diagnostics")
            if pd:
                for fam, count in pd["families"].items():
                    families[fam] = families.get(fam, 0) + count
                max_point_residual = max(max_point_residual, pd["max_residual"])
            idg = item.get("integrated_diagnostics")
            if idg:
                for fam, count in idg["families"].items():
                    p07_families[fam] = p07_families.get(fam, 0) + count
            nd = item.get("neumann_diagnostics")
            if nd:
                max_neumann_residual = max(max_neumann_residual, nd["max_residual"])
                max_condition = max(max_condition, nd["max_condition"])
            for key, value in (item.get("tensor_encoding") or {}).items():
                if isinstance(value, dict):
                    by = tensor_encoding.setdefault(key, {})
                    for name, count in value.items():
                        by[name] = by.get(name, 0) + count
                else:
                    tensor_encoding[key] = tensor_encoding.get(key, 0) + value
            point_rows += item.get("point_row_count", 0)
            neumann_rows += item.get("neumann_row_count", 0)
            integrated_rows += item.get("integrated_row_count", 0)
    return {
        "point_row_families": families,
        "p07_integrated_row_families": p07_families,
        "max_point_row_residual": max_point_residual,
        "max_neumann_constraint_residual": max_neumann_residual,
        "max_neumann_condition": max_condition,
        "total_point_rows": point_rows,
        "total_neumann_rows": neumann_rows,
        "total_integrated_rows": integrated_rows,
        # sources stored as tensor factors / CSR fallbacks of a failed bitwise expansion (expect 0 fallbacks)
        "tensor_encoding": tensor_encoding,
    }


# ---------------------------------------------------------------------------
# Full build: geometry stage (parallel), then cells/faces/p07 (parallel),
# then row-manifest assembly -- one function, so a campaign package can call
# the whole build without going through argparse/the CLI. See this module's
# docstring, "Calling the full build", for a worked example.
# ---------------------------------------------------------------------------
def _effective_workers(workers: int, memory_budget_gib, worker_memory_gib, memory_reserve_gib: float) -> int:
    """Cap ``workers`` by a memory budget, mirroring
    ``p05n_field_derived_global``/``p06n_field_derived_global``'s own
    ``effective()`` helper. Both ``memory_budget_gib`` and
    ``worker_memory_gib`` are optional (default ``None``); passing either
    one without the other is a mistake, not a silent no-op, so it raises."""
    if memory_budget_gib is None and worker_memory_gib is None:
        return workers
    if not memory_budget_gib or not worker_memory_gib or memory_budget_gib <= 0 or worker_memory_gib <= 0:
        raise ValueError("memory_budget_gib and worker_memory_gib must both be given, and positive, together")
    count = min(workers, int((memory_budget_gib - memory_reserve_gib) // worker_memory_gib))
    if count < 1:
        raise ValueError("memory budget cannot fit one worker plus reserve")
    return count


def run_full_build(
    *,
    n: int,
    input_root: Path,
    sidecar_path: Path,
    output: Path,
    workers: int,
    memory_budget_gib: float | None = None,
    worker_memory_gib: float | None = None,
    memory_reserve_gib: float = 1.0,
    cell_chunk_size: int = 4096,
    face_chunk_size: int = 2048,
    p07_chunk_size: int = 2048,
    geometry_raw_chunk_size: int = 4096,
    geometry_face_chunk_size: int = 4096,
    max_tasks_per_worker: int | None = None,
    max_units: int | None = None,
    curvature: str = DEFAULT_CURVATURE,
    face_quadrature: str = DEFAULT_FACE_QUADRATURE,
    inner_support: str = DEFAULT_INNER_SUPPORT,
    bfield_toroidal: str = DEFAULT_BFIELD_TOROIDAL,
    cell_stencil: str = "biased",
) -> dict:
    """Build the full N{n} row artifact and return the same dict written to
    ``<output>/N{n}/build_receipt.json``.

    ``curvature`` (``"autodiff"``, the default, or ``"fd"``) selects the curvature ``K`` of the geometry
    arrays and of the reference used by the row builders (see ``p_shared.provider``); it is recorded in
    the build policy, so an autodiff build has a different identity from an fd build.

    ``face_quadrature`` (``"q3"``, the default, or ``"q2"``) is the P05/P06 face-node rule: q2 builds R2/R3 rows
    and the face geometry at 2x2 Gauss (four nodes per face); P07 (R4) stays q3. It is recorded in the build
    policy (``quadrature.face``), so a q2 build has a distinct identity; ``"q3"`` is bitwise the historic build.

    ``bfield_toroidal`` (``"spline"``, the default, or ``"compact_c3"``) is the toroidal interpolation of the B
    evaluator behind the geometry and the reference (see ``p_shared.bfield``); recorded in the build policy (and the
    pinned sources) only when not ``"spline"``, so a spline build is bitwise the historic one.

    ``cell_stencil`` (``"biased"``, the default, or ``"symmetric"``) is the R1 cell-row stencil of the structured
    reconstruction (the fourth-order centred ``1/2 (A + B)`` cell rows); recorded in the build policy only when not
    ``"biased"``, so a biased build is bitwise the historic one. Symmetric cell rows are stored as CSR (they are not
    one tensor factorization).

    ``workers`` is the requested process-pool size for every stage; if both
    ``memory_budget_gib`` and ``worker_memory_gib`` are given, the effective
    worker count is additionally capped to
    ``min(workers, (memory_budget_gib - memory_reserve_gib) // worker_memory_gib)``
    (``memory_reserve_gib`` defaults to 1.0), the same convention
    ``p05n_field_derived_global``/``p06n_field_derived_global`` use. Leave
    both ``None`` (the default) to use ``workers`` as given -- required for
    the disk-tight, RAM-constrained local machine this was developed on,
    where the caller instead just passes a small ``workers``.

    Resumable exactly like the row stages already were: an interrupted
    ``geometry_raw``/``geometry_face``/``cells``/``faces``/``p07`` stage
    picks back up from its own per-unit receipts on the next call with the
    same ``output`` (and the same build identity/plan -- a changed input
    raises rather than silently reusing stale units).

    Re-entering a grid that is already assembled (``_chunks/`` -- every unit
    receipt -- is deleted once the rows are assembled) does nothing: if
    ``build_receipt.json`` (same identity), ``geometry.npz`` (sha256 equal to
    the receipt's ``geometry_sha256``) and every manifest row file verify, the
    stored receipt is returned with ``"reentry": "already_assembled"``.
    Otherwise the stale receipt/manifest (and a mismatching ``geometry.npz``)
    are discarded and the grid is rebuilt; the new receipt then carries
    ``rebuilt_after_failed_verification``.
    """
    check_curvature(curvature)
    check_face_quadrature(face_quadrature)
    check_inner_support(inner_support)
    check_bfield_toroidal(bfield_toroidal)
    check_cell_stencil(cell_stencil)
    face_order = _face_order(face_quadrature)
    input_root = Path(input_root).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    output = Path(output).resolve()
    grid_dir = output / f"N{n}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    effective_workers = _effective_workers(workers, memory_budget_gib, worker_memory_gib, memory_reserve_gib)

    with runner.lock(output):
        identity = build_identity(n=n, input_root=input_root, sidecar_path=sidecar_path, curvature=curvature,
                                  face_quadrature=face_quadrature, inner_support=inner_support,
                                  bfield_toroidal=bfield_toroidal, cell_stencil=cell_stencil)
        identity_path = grid_dir / "build_identity.json"
        if identity_path.exists():
            saved = json.loads(identity_path.read_text())
            if saved != identity:
                raise ValueError("build identity changed for an existing output directory; use a new --output")
        else:
            runner.write_json(identity_path, identity)

        overall_started = time.time()
        census_path = grid_dir / "census.npz"
        geometry_path = grid_dir / "geometry.npz"
        if census_path.exists():
            census = FaceCensus.load(census_path)
        else:
            t = load_context(n, str(input_root))
            census = FaceCensus.build(n, t.ro)
            census.save(census_path)

        face_row_indices = face_row_selection(census)
        p07_row_indices = builder.p07_row_selection(census)

        plan = {
            "geometry_raw": runner.chunk_units("geometry_raw", n, n ** 3, geometry_raw_chunk_size),
            "geometry_face": runner.chunk_units("geometry_face", n, len(face_row_indices), geometry_face_chunk_size),
            "cells": runner.chunk_units("cells", n, n ** 3, cell_chunk_size),
            "faces": runner.chunk_units("faces", n, len(face_row_indices), face_chunk_size),
            "p07": runner.chunk_units("p07", n, len(p07_row_indices), p07_chunk_size),
        }
        plan_path = grid_dir / "plan.json"
        plan_payload = {"identity": identity, "plan": plan}
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan_payload, default=runner._json_default)):
                raise ValueError("plan changed for an existing output directory; use a new --output")
        else:
            runner.write_json(plan_path, plan_payload)

        # Re-entry after a finished build: ``_chunks/`` (every unit receipt)
        # is deleted once the rows are assembled, so an assembled grid must
        # not be re-run through the stages. If it verifies, return its
        # receipt as is; if not, discard the stale receipt/manifest (and a
        # geometry.npz that disagrees with the receipt) and rebuild.
        assembled, rebuilt_reason, geometry_mismatch = _verify_assembled(output, grid_dir, identity, plan)
        if assembled is not None:
            return {**assembled, "reentry": "already_assembled"}
        if rebuilt_reason is not None:
            (grid_dir / "build_receipt.json").unlink(missing_ok=True)
            (grid_dir / "manifest.json").unlink(missing_ok=True)
            if geometry_mismatch:
                geometry_path.unlink(missing_ok=True)

        # Disk-space guard: abort rather than run a build that could exhaust the
        # volume. The estimate is measured, not the design's: the first full N32
        # build (side-row gradients included) was 13.4 GiB of rows; without them
        # ~10 GiB in total. Scale with n^3 and require 8 GiB to remain. Each unit also
        # checks free space before writing (MIN_FREE_GIB_PER_UNIT).
        estimated_gib = 10.0 * (n / 32) ** 3
        free_before = _disk_free_gib(output)
        if free_before - estimated_gib < 8.0:
            raise RuntimeError(
                f"refusing to build N{n}: {free_before:.2f} GiB free, estimated artifact "
                f"~{estimated_gib:.2f} GiB would leave <8 GiB free")

        initargs = (str(input_root), str(sidecar_path), str(output), n, identity, curvature)
        # always explicit: a worker must never fall back to a (changed) module default
        geometry_initargs = initargs + (face_quadrature, bfield_toroidal)
        row_initargs = initargs + (inner_support, bfield_toroidal, cell_stencil)
        summaries = {}

        geometry_complete = geometry_path.exists()
        if geometry_complete:
            GeometryArrays.load(geometry_path)  # validate, never trust a stale file silently
        else:
            for stage in ("geometry_raw", "geometry_face"):
                summaries[stage] = runner.run_stage(
                    output, stage, plan[stage], identity,
                    compute=_GEOMETRY_COMPUTE[stage], initializer=_init_geometry_worker, initargs=geometry_initargs,
                    workers=effective_workers, parts=("chunk",),
                    max_tasks_per_worker=max_tasks_per_worker, max_units=max_units)
            # `max_units` (a bounded, resumable invocation -- e.g. a
            # time-limited remote job stepping through the geometry stage
            # across several submissions) may leave some geometry_raw/
            # geometry_face units not yet built. `_assemble_geometry`
            # concatenates *every* planned unit's own chunk file
            # unconditionally, so calling it before every unit is present
            # raised a confusing FileNotFoundError on the units this
            # invocation's `max_units` skipped -- guard it on every unit of
            # both stages actually being valid first; a later call (with the
            # remaining units, same `output`) resumes and completes it.
            geometry_complete = all(
                runner.valid_unit(output, unit, identity, parts=("chunk",))
                for stage in ("geometry_raw", "geometry_face") for unit in plan[stage])
            if geometry_complete:
                (_assemble_geometry(output, grid_dir, plan) if face_order == 3
                 else _assemble_geometry(output, grid_dir, plan, face_order))

        if not geometry_complete:
            return {
                "n": n, "identity": identity, "geometry_complete": False,
                "wall_seconds": time.time() - overall_started,
                "stage_summaries": summaries,
                "note": "geometry stage incomplete under max_units; resume with a later call before row stages can run",
            }

        for stage in ("cells", "faces", "p07"):
            summaries[stage] = runner.run_stage(
                output, stage, plan[stage], identity,
                compute=_COMPUTE[stage], initializer=_init_worker, initargs=row_initargs,
                workers=effective_workers, parts=("chunk", "neumann"),
                max_tasks_per_worker=max_tasks_per_worker, max_units=max_units)

        diagnostics = _aggregate_receipts(output, plan)
        manifest = _assemble(output, grid_dir, identity, plan)
        shutil.rmtree(output / "_chunks", ignore_errors=True)

        # Verify: reload the full artifact via the frozen artifact.py API,
        # which checks every chunk's sha256 and the manifest identity.
        loaded = artifact_mod.load_row_artifact(output, n, identity)

        receipt = {
            "n": n, "identity": identity,
            "wall_seconds": time.time() - overall_started,
            "cpu_seconds": sum(s["worker_seconds"] for s in summaries.values()),
            "peak_rss_gib": max((s["peak_worker_rss_gib"] for s in summaries.values()), default=0.0),
            "effective_workers": effective_workers,
            "stage_summaries": summaries,
            "row_counts": {
                "cells": sum(len(c.request) for c in loaded.cells),
                "faces": sum(len(c.request) for c in loaded.faces),
                "neumann": sum(len(c.entity_id) for c in loaded.neumann),
                "p07": sum(len(c.entity_id) for c in loaded.p07),
            },
            "bytes_per_row_kind": {
                group: sum((grid_dir / entry["file"]).stat().st_size for entry in manifest["chunks"][group])
                for group in ("cells", "faces", "neumann", "p07")
            },
            "chunk_files": sum(len(v) for v in manifest["chunks"].values()),
            "diagnostics": diagnostics,
            "disk_free_gib_before": free_before,
            "disk_free_gib_after": _disk_free_gib(output),
            "geometry_sha256": runner.sha256_file(geometry_path),
        }
        if rebuilt_reason is not None:
            receipt["rebuilt_after_failed_verification"] = rebuilt_reason
        runner.write_json(grid_dir / "build_receipt.json", receipt)
        return receipt


# ---------------------------------------------------------------------------
# CLI: a thin argparse wrapper around run_full_build. Every flag keeps its
# old name, default and meaning; --geometry-*-chunk-size and the memory-
# budget flags are new and optional.
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--cell-chunk-size", type=int, default=4096)
    p.add_argument("--face-chunk-size", type=int, default=2048)
    p.add_argument("--p07-chunk-size", type=int, default=2048)
    p.add_argument("--geometry-raw-chunk-size", type=int, default=4096)
    p.add_argument("--geometry-face-chunk-size", type=int, default=4096)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
    p.add_argument("--max-units", type=int, default=None, help="for smoke-testing a partial build")
    p.add_argument("--curvature", choices=("fd", "autodiff"), default=DEFAULT_CURVATURE,
                   help="curvature K of the geometry and reference (recorded in the build identity)")
    p.add_argument("--inner-support", choices=INNER_SUPPORT_CHOICES, default=DEFAULT_INNER_SUPPORT,
                   help="inner donor support of the P05/P06/P07 rows: profile7 (C0) or last_aggregate (C1); "
                        "recorded in the build identity when not profile7")
    p.add_argument("--bfield-toroidal", choices=BFIELD_TOROIDAL_CHOICES, default=DEFAULT_BFIELD_TOROIDAL,
                   help="toroidal interpolation of the B evaluator: spline (frozen) or compact_c3 "
                        "(recorded in the build identity when not spline)")
    p.add_argument("--cell-stencil", choices=CELL_STENCILS, default="biased",
                   help="R1 cell-row stencil: biased (frozen) or symmetric (fourth-order centred 1/2 (A + B)); "
                        "recorded in the build identity when not biased")
    p.add_argument("--face-quadrature", choices=FACE_QUADRATURE_CHOICES, default=DEFAULT_FACE_QUADRATURE,
                   help="P05/P06 face-node rule: q3 (9 nodes) or q2 (4 nodes); P07 stays q3 "
                        "(recorded in the build identity)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    return run_full_build(
        n=args.n, input_root=args.input_root, sidecar_path=args.sidecar, output=args.output,
        workers=args.workers, memory_budget_gib=args.memory_budget_gib, worker_memory_gib=args.worker_memory_gib,
        memory_reserve_gib=args.memory_reserve_gib,
        cell_chunk_size=args.cell_chunk_size, face_chunk_size=args.face_chunk_size, p07_chunk_size=args.p07_chunk_size,
        geometry_raw_chunk_size=args.geometry_raw_chunk_size, geometry_face_chunk_size=args.geometry_face_chunk_size,
        max_tasks_per_worker=args.max_tasks_per_worker, max_units=args.max_units, curvature=args.curvature,
        face_quadrature=args.face_quadrature, inner_support=args.inner_support,
        bfield_toroidal=args.bfield_toroidal, cell_stencil=args.cell_stencil)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=runner._json_default))
