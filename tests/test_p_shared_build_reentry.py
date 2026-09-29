"""Re-entering ``build_artifact.run_full_build`` on a grid whose rows are
already assembled (``_chunks/`` -- every unit receipt -- deleted) must skip
every stage, and must rebuild (not crash) when the assembled outputs do not
verify. Fully synthetic: the geometry/census/row compute and the process pool
are replaced by tiny in-process fakes that write real ``runner`` unit files,
so ``_assemble``, ``_aggregate_receipts`` and the receipt/verification logic
run for real."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p_shared import build_artifact as ba  # noqa: E402
from p_shared import runner  # noqa: E402

N = 2
IDENTITY = {"marker": "reentry-test"}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Patch run_full_build's external pieces; return (call, calls) where
    ``call()`` runs a full build against ``tmp_path/out`` and ``calls`` lists
    the stage names actually executed (in order)."""
    calls: list[str] = []
    output = tmp_path / "out"
    (output / f"N{N}").mkdir(parents=True)
    (output / f"N{N}" / "census.npz").write_bytes(b"census")

    class _Census:
        @classmethod
        def load(cls, path):
            return object()

    counter = {"geometry": 0}

    def fake_run_stage(out, stage, units, identity, *, parts=("chunk",), **_kw):
        calls.append(stage)
        for u in units:
            if runner.valid_unit(out, u, identity, parts=parts):
                continue
            extra = {} if stage.startswith("geometry") else {"point_row_count": 1}
            runner.write_unit(out, u, identity, started=time.time(), extra=extra,
                              chunks={part: {"x": np.array([u["start"]])} for part in parts})
        return {"peak_worker_rss_gib": 0.0, "worker_seconds": 0.0}

    def fake_assemble_geometry(out, grid_dir, plan):
        counter["geometry"] += 1
        (grid_dir / "geometry.npz").write_bytes(f"geometry-{counter['geometry']}".encode())

    monkeypatch.setattr(ba, "build_identity", lambda **kw: dict(IDENTITY))
    monkeypatch.setattr(ba, "FaceCensus", _Census)
    monkeypatch.setattr(ba, "face_row_selection", lambda census: np.arange(10))
    monkeypatch.setattr(ba.builder, "p07_row_selection", lambda census: np.arange(5))
    monkeypatch.setattr(ba, "_disk_free_gib", lambda path: 1.0e6)
    monkeypatch.setattr(ba.GeometryArrays, "load", staticmethod(lambda path: None))
    monkeypatch.setattr(ba, "_assemble_geometry", fake_assemble_geometry)
    monkeypatch.setattr(ba.artifact_mod, "load_row_artifact",
                        lambda *a, **k: SimpleNamespace(cells=(), faces=(), neumann=(), p07=()))
    monkeypatch.setattr(runner, "run_stage", fake_run_stage)

    def call():
        return ba.run_full_build(
            n=N, input_root=tmp_path, sidecar_path=tmp_path / "sidecar.json", output=output, workers=1,
            cell_chunk_size=4, face_chunk_size=4, p07_chunk_size=4,
            geometry_raw_chunk_size=4, geometry_face_chunk_size=4)

    return SimpleNamespace(call=call, calls=calls, output=output, grid=output / f"N{N}")


def _snapshot(grid):
    return {p.relative_to(grid).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(grid.rglob("*")) if p.is_file()}


def test_reentering_an_assembled_grid_skips_every_stage_and_changes_nothing(harness):
    first = harness.call()
    assert harness.calls == ["geometry_raw", "geometry_face", "cells", "faces", "p07"]
    assert not (harness.output / "_chunks").exists()  # the post-assembly state: no unit receipts at all
    before = _snapshot(harness.grid)

    harness.calls.clear()
    second = harness.call()  # previously: FileNotFoundError on _chunks/N2/geometry_raw/...json

    assert harness.calls == []
    assert second["reentry"] == "already_assembled"
    assert {k: v for k, v in second.items() if k != "reentry"} == json.loads(json.dumps(first, default=runner._json_default))
    assert _snapshot(harness.grid) == before  # no rebuild, no rewritten file
    assert not (harness.output / "_chunks").exists()


def test_corrupt_row_file_triggers_a_row_rebuild_not_a_crash(harness):
    harness.call()
    manifest = json.loads((harness.grid / "manifest.json").read_text())
    victim = harness.grid / manifest["chunks"]["faces"][0]["file"]
    victim.write_bytes(b"corrupted")
    geometry_before = (harness.grid / "geometry.npz").read_bytes()

    harness.calls.clear()
    rebuilt = harness.call()

    assert harness.calls == ["cells", "faces", "p07"]  # geometry verified, so it is kept
    assert (harness.grid / "geometry.npz").read_bytes() == geometry_before
    assert "row file" in rebuilt["rebuilt_after_failed_verification"]
    assert "reentry" not in rebuilt

    harness.calls.clear()
    assert harness.call()["reentry"] == "already_assembled"  # the rebuilt grid verifies
    assert harness.calls == []


def test_geometry_that_disagrees_with_the_receipt_is_rebuilt(harness):
    harness.call()
    (harness.grid / "geometry.npz").write_bytes(b"tampered")

    harness.calls.clear()
    rebuilt = harness.call()

    assert harness.calls == ["geometry_raw", "geometry_face", "cells", "faces", "p07"]
    assert "geometry" in rebuilt["rebuilt_after_failed_verification"]
    assert (harness.grid / "geometry.npz").read_bytes() != b"tampered"

    harness.calls.clear()
    assert harness.call()["reentry"] == "already_assembled"


def test_bounded_local_testing_receipt_is_not_treated_as_an_assembled_grid(harness):
    harness.call()
    receipt_path = harness.grid / "build_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("geometry_sha256")
    receipt["bounded"] = True
    receipt_path.write_text(json.dumps(receipt))
    geometry_before = (harness.grid / "geometry.npz").read_bytes()

    harness.calls.clear()
    harness.call()

    assert harness.calls == ["cells", "faces", "p07"]  # geometry.npz untouched and reused
    assert (harness.grid / "geometry.npz").read_bytes() == geometry_before


def test_aggregate_receipts_needs_only_row_stage_receipts(tmp_path):
    output = tmp_path / "out"
    plan = {
        "geometry_raw": runner.chunk_units("geometry_raw", N, 8, 4),
        "cells": runner.chunk_units("cells", N, 8, 4),
    }
    for unit in plan["cells"]:
        runner.write_unit(output, unit, IDENTITY, started=time.time(), extra={"point_row_count": 3},
                          chunks={"chunk": {"x": np.zeros(1)}})
    # No geometry_raw receipts on disk (as after a finished build): must not raise.
    assert ba._aggregate_receipts(output, plan)["total_point_rows"] == 6
