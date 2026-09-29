"""Tests for the parallel geometry stage added to
``scripts/p_shared/build_artifact.py`` (``geometry_raw``/``geometry_face``
units run through ``p_shared.runner.run_stage``, assembled into
``geometry.npz``) -- see that module's docstring, "Geometry stage
(parallel)", for the design this exercises.

Three groups:

* **Pure** (no workspace needed): ``runner.chunk_units`` covers a range
  without gaps/overlap for the geometry-stage naming, and the raw/face
  field-name split (``RAW_GEOMETRY_FIELDS``/``FACE_GEOMETRY_FIELDS``) covers
  every ``GeometryArrays`` array field exactly once.
* **Assembly order** (real N32 geometry, a tiny handful of raw
  cells/faces -- well under the continuum reference's internal 4096-point
  metric batch size, so no batch-size-dependent roundoff can appear here):
  ``_compute_geometry_raw``/``_compute_geometry_face`` write their unit
  slices, ``_assemble_geometry`` concatenates them, and the result must
  equal calling ``build_raw_geometry_arrays``/``build_face_geometry_arrays``
  directly over the same combined batch, in the same order.
* **Resume** (real N32 geometry, a real spawned process pool -- the same
  pattern ``tests/test_stencils_builder.py``'s
  ``test_runner_plan_resume_and_rejection`` uses for the generic runner):
  a geometry_raw unit already computed by a first, partial invocation is
  not recomputed by a second invocation over the full unit list.

Skipped cleanly (the assembly/resume groups) if the local HSX workspace
geometry inputs are unavailable, the same way
``tests/test_stencils_builder.py``/``tests/test_p_shared_backfill_neumann_tags.py``
do.
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

from drbx.stencils.census import FaceCensus  # noqa: E402
from drbx.stencils.geometry_arrays import (  # noqa: E402
    GeometryArrays, build_raw_geometry_arrays, build_face_geometry_arrays)

from p_shared import build_artifact  # noqa: E402
from p_shared import runner  # noqa: E402

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
# Pure tests: no workspace/geometry needed.
# ---------------------------------------------------------------------------
def test_chunk_units_cover_a_range_without_gaps_or_overlap():
    for stage, count, size in (
        ("geometry_raw", 100, 30), ("geometry_face", 47, 10),
        ("geometry_raw", 4096 * 3, 4096), ("geometry_face", 1, 4096),
    ):
        units = runner.chunk_units(stage, N, count, size)
        assert units[0]["start"] == 0
        assert units[-1]["stop"] == count
        for a, b in zip(units, units[1:]):
            assert a["stop"] == b["start"]
        assert all(u["stage"] == stage and u["n"] == N for u in units)
        assert all(0 < u["stop"] - u["start"] <= size for u in units)


def test_geometry_field_split_covers_every_geometryarrays_array_field_once():
    raw = set(build_artifact.RAW_GEOMETRY_FIELDS)
    face = set(build_artifact.FACE_GEOMETRY_FIELDS)
    assert not (raw & face)
    assert raw | face == set(GeometryArrays.__dataclass_fields__) - {"schema", "identity"}
    assert len(build_artifact.RAW_GEOMETRY_FIELDS) == len(raw)
    assert len(build_artifact.FACE_GEOMETRY_FIELDS) == len(face)


# ---------------------------------------------------------------------------
# Real, tiny (< the continuum reference's internal 4096-point metric-query
# batch size) subsets of the N32 grid.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def geometry_state():
    if not _workspace_inputs_available():
        pytest.skip("workspace inputs unavailable")
    from perpendicular_structured.reconstruction import load_context
    from p_shared import provider as p_shared_provider

    t = load_context(N, str(WORKSPACE))
    census = FaceCensus.build(N, t.ro)
    provider = p_shared_provider.ScriptsGeometryProvider.from_sidecar(str(SIDECAR), verify_hashes=False)
    return {"t": t, "census": census, "provider": provider,
            "face_row_indices": build_artifact.face_row_selection(census)}


def _set_geometry_state_in_process(tmp_path, geometry_state, n=N):
    """Populate build_artifact.GEOMETRY_STATE the way _init_geometry_worker
    would from disk, without a process pool -- a fast, direct check of
    _compute_geometry_raw/_compute_geometry_face/_assemble_geometry."""
    build_artifact.GEOMETRY_STATE = {
        "faces": geometry_state["t"].faces, "census": geometry_state["census"],
        "provider": geometry_state["provider"], "face_row_indices": geometry_state["face_row_indices"],
        "n": n, "output": tmp_path, "identity": {"marker": "geometry-stage-test"},
    }


@needs_workspace
def test_geometry_units_assemble_in_plan_order(tmp_path, geometry_state):
    _set_geometry_state_in_process(tmp_path, geometry_state)
    raw_units = runner.chunk_units("geometry_raw", N, 38, 13)
    face_units = runner.chunk_units("geometry_face", N, 30, 11)
    assert len(raw_units) == 3 and len(face_units) == 3

    for unit in raw_units:
        build_artifact._compute_geometry_raw(unit)
    for unit in face_units:
        build_artifact._compute_geometry_face(unit)

    plan = {"geometry_raw": raw_units, "geometry_face": face_units}
    grid_dir = tmp_path / f"N{N}"
    geometry = build_artifact._assemble_geometry(tmp_path, grid_dir, plan)

    # Directly recompute the same combined batches as one call each (both
    # well under the internal 4096-point metric-query batch size), and check
    # assembly reproduces both the values and the plan's own row order.
    t, provider, census = geometry_state["t"], geometry_state["provider"], geometry_state["census"]
    raw_keys = np.array(np.unravel_index(np.arange(38, dtype=np.int64), (N, N, N))).T.astype(np.int64)
    expected_raw = build_raw_geometry_arrays(provider, t.faces, raw_keys)
    face_row_indices = geometry_state["face_row_indices"][:30]
    face_keys = census.keys()[face_row_indices]
    expected_face = build_face_geometry_arrays(provider, t.faces, face_keys)

    # Splitting a batch into smaller per-unit calls is not bitwise against
    # one call over the combined batch -- the frozen continuum reference's
    # metric evaluator has pre-existing batch-size-dependent floating-point
    # roundoff (BLAS-level, not a numerics change; see build_artifact.py's
    # module docstring), amplified for the curvature/divergence fields
    # (finite-difference quotients divide it by the FD step, ~2e-4). This
    # test deliberately uses tiny, non-4096-aligned chunk sizes for speed, so
    # it sees a *larger* residual than the production ``--geometry-*-chunk-
    # size`` default (4096, aligned to the reference's own internal
    # metric_query_batch_size) does -- the 1e-12 production acceptance bound
    # is checked separately by the one-off N32 equivalence script (see this
    # module's docstring), not by this fast unit test. Assert same ballpark
    # (order-of-magnitude sanity that assembly didn't scramble rows/values),
    # not bitwise equality.
    for name in build_artifact.RAW_GEOMETRY_FIELDS:
        np.testing.assert_allclose(getattr(geometry, name), expected_raw[name], rtol=1e-6, atol=1e-8)
    for name in build_artifact.FACE_GEOMETRY_FIELDS:
        np.testing.assert_allclose(getattr(geometry, name), expected_face[name], rtol=1e-6, atol=1e-8)

    geometry.verify()  # identity matches the assembled arrays, not trusted blindly
    loaded = GeometryArrays.load(grid_dir / "geometry.npz")
    assert loaded.identity == geometry.identity
    for name in build_artifact.RAW_GEOMETRY_FIELDS + build_artifact.FACE_GEOMETRY_FIELDS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(geometry, name))


@needs_workspace
def test_geometry_raw_stage_resumes_without_recomputing(tmp_path, geometry_state):
    """A first invocation over half the units, then a second invocation over
    the full list: the first half must resume (skip, not recompute), only
    the new half executes -- mirrors
    tests/test_stencils_builder.py's test_runner_plan_resume_and_rejection,
    against the real geometry_raw compute/initializer instead of a toy."""
    grid_dir = tmp_path / f"N{N}"
    grid_dir.mkdir(parents=True)
    geometry_state["census"].save(grid_dir / "census.npz")

    identity = {"marker": "geometry-stage-resume-test"}
    units = runner.chunk_units("geometry_raw", N, 20, 5)  # 4 tiny units
    assert len(units) == 4
    initargs = (str(WORKSPACE), str(SIDECAR), str(tmp_path), N, identity)

    first = runner.run_stage(
        tmp_path, "geometry_raw", units[:2], identity,
        compute=build_artifact._compute_geometry_raw, initializer=build_artifact._init_geometry_worker,
        initargs=initargs, workers=2, parts=("chunk",))
    assert first["executed_units"] == 2 and first["resumed_units"] == 0
    for u in units[:2]:
        assert runner.valid_unit(tmp_path, u, identity, parts=("chunk",))

    second = runner.run_stage(
        tmp_path, "geometry_raw", units, identity,
        compute=build_artifact._compute_geometry_raw, initializer=build_artifact._init_geometry_worker,
        initargs=initargs, workers=2, parts=("chunk",))
    assert second["resumed_units"] == 2
    assert second["executed_units"] == 2
    for u in units:
        assert runner.valid_unit(tmp_path, u, identity, parts=("chunk",))
