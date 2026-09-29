"""Tests for ``scripts/p08_step1_global/campaign.py`` (CLI, identity,
preflight gating, resume, reduction on synthetic units) and
``scripts/p_shared/oracle_manifest.py``. Fully synthetic / tmp_path-based:
no real geometry, no multi-GB row artifact, no real oracle data -- matches
this package's README.md testing commitment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p08_step1_global import campaign  # noqa: E402
from p_shared import oracle_manifest as om  # noqa: E402
from p_shared import replay_units as ru  # noqa: E402
from p_shared import runner  # noqa: E402


# ---------------------------------------------------------------------------
# Remote-ready inputs: the same immutable input root P05N/P06N/P07N use
# (input_manifest.json, byte-identical) plus the canonical sidecar
# campaign.verify/campaign.localize_sidecar localize into <output>/
# localized_sidecar.json. Tests that exercise campaign.verify()/main() build
# a synthetic workspace: campaign._input_manifest is monkeypatched to an
# empty file list (no real ~9 GB immutable inputs needed on disk for these
# CLI/identity/resume tests), but the canonical sidecar itself is a real,
# minimal, schema-shaped fixture file so localize_sidecar's own read/rewrite
# logic runs for real.
# ---------------------------------------------------------------------------
def _fake_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_input_manifest", lambda: {"files": []})
    workspace = tmp_path / "workspace"
    cfg = campaign.config()
    sidecar_path = workspace / cfg["canonical_sidecar_relative"]
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps({
        "schema": "drbx.hsx-continuous-mms-reference",
        "metric_cache": {"path": "placeholder"},
        "makegrid": {"path": "placeholder"},
        "artifact": {"path": "placeholder"},
    }))
    return workspace


# ---------------------------------------------------------------------------
# Configuration / identity.
# ---------------------------------------------------------------------------
def test_config_matches_campaign_catalogue():
    cfg = campaign.config()
    from p_shared.replay_support import CAMPAIGN_FUNCS
    assert cfg["campaigns"] == list(CAMPAIGN_FUNCS)
    assert cfg["resolutions"] == [32, 48, 64]
    assert cfg["cell_chunk_size"] > 0 and cfg["face_chunk_size"] > 0 and cfg["p07_chunk_size"] > 0


def test_source_hashes_cover_existing_files():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES)
    for rel, digest in hashes.items():
        assert (REPO / rel).is_file(), f"listed source file missing: {rel}"
        assert len(digest) == 64  # sha256 hex


def test_oracle_paths_default_vs_override(tmp_path):
    default = campaign.oracle_paths(None, tmp_path)
    override = campaign.oracle_paths(tmp_path / "oracle_copy", tmp_path)
    cfg = campaign.config()
    for name, rel in cfg["oracle_default_paths"].items():
        assert default[name] == tmp_path / rel
        assert override[name] == (tmp_path / "oracle_copy") / rel


def test_verify_is_stable_across_two_calls_and_sensitive_to_source_change(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"

    identity_1 = campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)
    identity_2 = campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)
    assert identity_1 == identity_2
    assert (output / "campaign_manifest.json").is_file()
    assert (output / "oracle_manifest.json").is_file()
    assert (output / "localized_sidecar.json").is_file()
    saved = json.loads((output / "campaign_manifest.json").read_text())
    assert saved["localized_sidecar_sha256"] == campaign.sha(output / "localized_sidecar.json")

    # Simulate a source change (e.g. an edit to replay_units.py): a
    # different recorded identity for the SAME output must raise, not
    # silently overwrite.
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"fake/path.py": "0" * 64})
    with pytest.raises(ValueError, match="campaign identity changed"):
        campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)


def test_verify_refuses_if_localized_sidecar_changed_on_disk(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)

    # Tamper the localized copy itself (not the canonical source): caught
    # directly by localize_sidecar's own content check, before verify()'s
    # separate recorded-hash check ever runs.
    local = output / "localized_sidecar.json"
    tampered = json.loads(local.read_text())
    tampered["metric_query_batch_size"] = 1
    local.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="localized reference sidecar changed"):
        campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)


def test_verify_refuses_if_recorded_sidecar_hash_changed(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)

    # Tamper only the recorded hash in campaign_manifest.json (the localized
    # copy on disk is left untouched and still matches the canonical
    # source): caught by verify()'s separate recorded-hash check.
    manifest_path = output / "campaign_manifest.json"
    saved = json.loads(manifest_path.read_text())
    saved["localized_sidecar_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="localized sidecar hash changed"):
        campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)


def test_verify_refuses_missing_immutable_input(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    monkeypatch.setattr(campaign, "_input_manifest",
                        lambda: {"files": [{"path": "does/not/exist.npz", "bytes": 1, "sha256": "0" * 64}]})
    with pytest.raises(ValueError, match="missing or changed immutable input"):
        campaign.verify(input_root=fake_workspace, output=output, oracle_root=None)


def test_localize_sidecar_rewrites_paths_under_input_root(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    output.mkdir()
    local = campaign.localize_sidecar(fake_workspace, output)
    payload = json.loads(local.read_text())
    assert payload["metric_cache"]["path"] == str((fake_workspace / "hsx_metric_d58d392545fd3917efeb83b6.npz").resolve())
    assert payload["makegrid"]["path"] == str((fake_workspace / "mgrid_res2p5cm_180pln.nc").resolve())
    assert payload["artifact"]["path"] == str((fake_workspace / "prototype_runs/geometry/hsx_fci_64x64x64").resolve())
    assert payload["metric_query_batch_size"] == 4096

    # A second call against the same (unchanged) canonical sidecar is a
    # no-op; a call after the canonical sidecar's own content changed raises.
    campaign.localize_sidecar(fake_workspace, output)
    cfg = campaign.config()
    canonical_path = fake_workspace / cfg["canonical_sidecar_relative"]
    canonical = json.loads(canonical_path.read_text())
    # Change a field localize_sidecar does not itself overwrite (it always
    # rewrites metric_cache/makegrid/artifact's own "path" and
    # metric_query_batch_size, so tampering one of those would not change
    # the recomputed `side`).
    canonical["schema"] = "changed"
    canonical_path.write_text(json.dumps(canonical))
    with pytest.raises(ValueError, match="localized reference sidecar changed"):
        campaign.localize_sidecar(fake_workspace, output)


def test_input_manifest_matches_p07n_byte_identically():
    p07n = REPO / "scripts/p07n_field_derived_global/input_manifest.json"
    p08 = REPO / "scripts/p08_step1_global/input_manifest.json"
    assert p08.read_bytes() == p07n.read_bytes()


def test_sidecar_path_is_localized_under_output_not_a_work_scratch_path(tmp_path):
    import argparse
    args = argparse.Namespace(output=tmp_path / "some_output", input_root=tmp_path / "workspace")
    assert campaign._sidecar_path(args) == args.output / "localized_sidecar.json"


def test_verify_with_oracle_root_reports_missing_files_clearly(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    empty_oracle_root = tmp_path / "nowhere"
    with pytest.raises(ValueError, match="oracle verification failed"):
        campaign.verify(input_root=fake_workspace, output=output, oracle_root=empty_oracle_root)


# ---------------------------------------------------------------------------
# pack-oracles (tiny synthetic subset only, per the coordinator's explicit
# instruction -- never the real ~9 GB oracle data).
# ---------------------------------------------------------------------------
def test_pack_oracles_tar_contains_exactly_the_manifest_files(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "work/fake_campaign").mkdir(parents=True)
    np.savez(workspace / "work/fake_campaign/N32.owner_values.npz", values=np.arange(4.0))

    def fake_campaign_files(name, n, paths):
        return [paths["fake"] / f"N{n}.owner_values.npz"]

    monkeypatch.setattr(om, "campaign_files", fake_campaign_files)
    monkeypatch.setattr(om, "_WORKSPACE", workspace)
    _orig_config = campaign.config()
    monkeypatch.setattr(campaign, "config", lambda: {**_orig_config,
                                                     "campaigns": ["fake"],
                                                     "oracle_default_paths": {"fake": "work/fake_campaign"}})

    output = tmp_path / "campaign_output"
    output.mkdir()
    tar_path = tmp_path / "oracle.tar"
    result = campaign.pack_oracles(input_root=workspace, output=output, tar_path=tar_path, grids=[32])
    assert result["file_count"] == 1
    assert tar_path.is_file()

    import tarfile
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert "work/fake_campaign/N32.owner_values.npz" in names
    assert "oracle_manifest.json" in names
    assert len(names) == 2  # exactly the manifest's files plus the manifest itself


# ---------------------------------------------------------------------------
# Preflight gating: `run` refuses to start without a matching preflight.
# ---------------------------------------------------------------------------
def test_run_refuses_without_matching_preflight(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    argv = ["campaign.py", "run", "--input-root", str(fake_workspace), "--output", str(output),
           "--resolutions", "32", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="run requires a matching preflight"):
        campaign.main()


def test_validate_refuses_without_preflight(tmp_path, monkeypatch):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    argv = ["campaign.py", "validate", "--input-root", str(fake_workspace), "--output", str(output),
           "--resolutions", "32"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="a matching preflight is required"):
        campaign.main()


def test_verify_inputs_command_completes_and_locks_output(tmp_path, monkeypatch, capsys):
    fake_workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    argv = ["campaign.py", "verify-inputs", "--input-root", str(fake_workspace), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["command"] == "verify-inputs"
    assert payload["status"] == "complete"
    assert (output / ".campaign.lock").exists()


# ---------------------------------------------------------------------------
# Resume semantics: a valid unit is skipped, a corrupted one raises (reuses
# p_shared.runner's own contract; exercised here through the same
# convention this campaign's stages use).
# ---------------------------------------------------------------------------
def test_replay_unit_receipt_resume_and_corruption(tmp_path):
    output = tmp_path / "replay_out"
    unit = {"stage": "cells", "n": 8, "start": 0, "stop": 4}
    identity = {"id": 1}

    def compute(u):
        arrays = ru._pack_unit_arrays({"dummy": (np.array([0], dtype=np.int64), np.array([[1.0]]))})
        return runner.write_unit(output, u, identity, chunks={"chunk": arrays}, started=0.0)

    assert not runner.valid_unit(output, unit, identity)
    compute(unit)
    assert runner.valid_unit(output, unit, identity)

    # Corrupt the chunk file in place -> the same identity/unit now raises
    # rather than silently being treated as invalid-and-rebuildable.
    path = runner.unit_path(output, unit, "chunk")
    with path.open("r+b") as f:
        f.seek(0)
        f.write(b"\x00" * 8)
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        runner.valid_unit(output, unit, identity)


# ---------------------------------------------------------------------------
# Reduction on synthetic units: two fake "cells" unit outputs, hand-summed,
# must match a single combined call -- the same additivity
# replay_units.reduce_grid relies on, exercised end to end through the
# on-disk unit-output format (not just the in-memory sparse-scatter helpers
# tests/test_p_shared_replay_units.py already covers).
# ---------------------------------------------------------------------------
def test_reduction_sums_two_synthetic_cells_units_correctly(tmp_path):
    output = tmp_path / "replay_out"
    identity = {"id": 7}
    owners = 3

    def write_unit(start, stop, owner_ids, values):
        unit = {"stage": "cells", "n": 8, "start": start, "stop": stop}
        arrays = ru._pack_unit_arrays({"p05_centered": (np.asarray(owner_ids, dtype=np.int64),
                                                        np.asarray(values, dtype=np.float64))})
        runner.write_unit(output, unit, identity, chunks={"chunk": arrays}, started=0.0)
        return unit

    unit_a = write_unit(0, 2, [0, 1], [[1.0, 1.0], [2.0, 2.0]])
    unit_b = write_unit(2, 4, [1, 2], [[3.0, 3.0], [4.0, 4.0]])

    acc = ru.OwnerAccumulator(owners, (2,))
    for unit in (unit_a, unit_b):
        path = runner.unit_path(output, unit, "chunk")
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        data = ru._unpack_unit_arrays(arrays)
        uniq, values = data["p05_centered"]
        acc.add(uniq, values)

    expected = np.array([[1.0, 1.0], [5.0, 5.0], [4.0, 4.0]])  # owner 1 = 2+3 from both units
    np.testing.assert_allclose(acc.total, expected)
