"""Focused tests for current-source extended-IMEX benchmark provenance."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))
from current_source_benchmark import (  # noqa: E402
    BenchmarkProvenanceError,
    EXTENDED_IMPLICIT_FIELDS,
    make_benchmark_manifest,
    load_benchmark_checkpoint,
    state_sha256,
    timestep_metadata,
    validate_benchmark_manifest,
    write_benchmark_checkpoint,
)


def _state():
    return {name: np.arange(6, dtype=np.float64).reshape(2, 3) + i
            for i, name in enumerate(("density", *EXTENDED_IMPLICIT_FIELDS, "phi"))}


def _manifest():
    return make_benchmark_manifest(
        case="test-current-source",
        dt=2.34375e-4,
        gamma=1.0 - 1.0 / np.sqrt(2.0),
        initialization_provenance={
            "mode": "boundary-compatible-restart",
            "description": "test state with independently reconstructed potential",
        },
        state=_state(),
    )


def test_manifest_records_five_fields_and_distinguishes_full_and_stage_dt():
    manifest = _manifest()
    assert manifest["selected_material_fields"] == list(EXTENDED_IMPLICIT_FIELDS)
    assert manifest["explicit_material_fields"] == ["density"]
    assert manifest["dt"] == 2.34375e-4
    assert manifest["gamma_dt"] == pytest.approx(manifest["gamma"] * manifest["dt"])
    assert manifest["time_step"]["dt_role"] == "full simulation timestep"
    assert "implicit SSP222 stage coefficient" in manifest["time_step"]["gamma_dt_role"]
    assert manifest["state_provenance"]["compatible_potential"] is True
    assert any("fci_rlp_diffusion.py" in key for key in manifest["source_sha256"])
    assert any("fci_curvature_production_flux.py" in key for key in manifest["source_sha256"])
    assert validate_benchmark_manifest(manifest)["case"] == "test-current-source"


def test_state_hash_is_named_and_shape_sensitive():
    state = _state()
    assert state_sha256(state) != state_sha256(state, fields=("Te", "Ti", "Vi", "Ve", "vorticity"))
    reshaped = dict(state)
    reshaped["Te"] = reshaped["Te"].reshape(3, 2)
    assert state_sha256(state) != state_sha256(reshaped)


def test_stale_source_and_incompatible_dt_are_rejected():
    stale = copy.deepcopy(_manifest())
    key = next(iter(stale["source_sha256"]))
    stale["source_sha256"][key] = "0" * 64
    with pytest.raises(BenchmarkProvenanceError, match="source hash mismatch"):
        validate_benchmark_manifest(stale)

    bad_dt = copy.deepcopy(_manifest())
    bad_dt["gamma_dt"] *= 2.0
    with pytest.raises(BenchmarkProvenanceError, match="gamma_dt"):
        validate_benchmark_manifest(bad_dt)

    bad_fields = copy.deepcopy(_manifest())
    bad_fields["selected_material_fields"] = ["Te", "Vi", "Ve", "vorticity"]
    with pytest.raises(BenchmarkProvenanceError, match="selected_material_fields"):
        validate_benchmark_manifest(bad_fields)


def test_checkpoint_round_trip_and_tamper_rejection(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest()
    write_benchmark_checkpoint(manifest_path, manifest, {"Te": _state()["Te"], "step": np.array(0, dtype=np.int64)})
    loaded, arrays = load_benchmark_checkpoint(manifest_path)
    assert loaded["checkpoint"]["arrays"]["Te"]["dtype"] == "<f8"
    np.testing.assert_array_equal(arrays["Te"], _state()["Te"])

    artifact = manifest_path.with_suffix(".npz")
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(BenchmarkProvenanceError, match="checkpoint artifact hash mismatch"):
        load_benchmark_checkpoint(manifest_path)


def test_timestep_metadata_rejects_stage_coefficient_misuse():
    with pytest.raises(ValueError):
        timestep_metadata(0.0, 0.3)
    with pytest.raises(ValueError):
        timestep_metadata(1.0, 1.0)
