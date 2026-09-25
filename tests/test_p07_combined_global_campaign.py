"""Operational gates for the tracked combined P07 research campaign."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CAMPAIGN = Path(__file__).resolve().parents[1] / "scripts/p07_combined_global"
sys.path.insert(0, str(CAMPAIGN))
import campaign as p07  # noqa: E402


def _support_unit(output, unit, identity, face_id, family):
    path = p07.unit_path(output, unit)
    p07.atomic_npz(path, ids=np.array([face_id]), family=np.array([family]),
                   supported=np.array([True]), full_rank=np.array([True]),
                   max_residual=np.array([1e-14]), min_actual_rank=np.array([7]),
                   min_uniform_rank=np.array([7]), cubic_expansion=np.array([0]),
                   quartic_expansion=np.array([0]), max_plane_donors=np.array([7]),
                   row_offsets=np.array([0, 2]), donor_ids=np.array([0, 1]),
                   coefficients=np.array([1.0, -1.0]))
    p07.atomic_json(path.with_suffix(".json"), {"identity": identity, "unit": unit,
                                                "sha256": p07.sha(path)})


def test_support_reducer_rejects_missing_duplicate_and_stale_chunks(tmp_path, monkeypatch):
    identity = "frozen-test-identity"
    monkeypatch.setattr(p07, "stage_identity", lambda output, stage: identity)
    monkeypatch.setattr(p07, "settings", lambda: {"resolutions": [2], "target_residual_tolerance": 1e-9,
                                                  "original_cubic_max_expansions": 3,
                                                  "quartic_additional_max_expansions": 2})
    monkeypatch.setattr(p07.k, "configure", lambda root: None)
    monkeypatch.setattr(p07.k, "load", lambda n: SimpleNamespace(vol=np.ones(3)))
    monkeypatch.setattr(p07, "selected_ids", lambda stage, n, output: np.array([11, 12]))
    np.savez(tmp_path / "N2.topology.npz", face_ids=np.array([11, 12]),
             family=np.array([6, 7]), endpoints=np.array([[0, 1], [1, 2]]))
    first = {"n": 2, "stage": "support", "start": 0, "stop": 1}
    second = {"n": 2, "stage": "support", "start": 1, "stop": 2}
    plan = {"stages": {"support": {"2": [first, second]}}}
    _support_unit(tmp_path, first, identity, 11, 6)
    _support_unit(tmp_path, second, identity, 12, 7)
    result = p07.reduce_stage(tmp_path, tmp_path, "support", plan)
    assert result["resolutions"]["2"]["ids"] == 2
    receipt = p07.unit_path(tmp_path, second).with_suffix(".json")
    saved = receipt.read_text(); receipt.unlink()
    with pytest.raises(ValueError, match="missing support chunk"):
        p07.reduce_stage(tmp_path, tmp_path, "support", plan)
    receipt.write_text(saved)
    with pytest.raises(ValueError, match="missing or duplicate"):
        p07.reduce_stage(tmp_path, tmp_path, "support", {"stages": {"support": {"2": [first, first]}}})
    changed = json.loads(saved); changed["identity"] = "stale"; receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stale receipt"):
        p07.reduce_stage(tmp_path, tmp_path, "support", plan)


def test_boundary_value_and_tangential_channels_are_both_applied(monkeypatch):
    def fields(reference, points):
        values = np.tile(points[:, 0, None], (1, 4))
        gradients = np.zeros((len(points), 4, 3))
        gradients[:, :, 1] = 2.0
        gradients[:, :, 2] = 3.0
        return values, gradients, None
    monkeypatch.setattr(p07.k.num, "fields", fields)
    bc = {"value_loading": np.array([-0.25, -0.75]),
          "tangential_loading": np.array([[1.0, 0.5], [0.25, -0.5]]),
          "trace_donor_points": np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.2]]),
          "trace_target_points": np.array([[1.0, 0.2, 0.3], [1.0, 0.4, 0.5]])}
    result = p07.boundary_data(None, [(np.array([0, 1]), np.array([1.0, 2.0]), bc)])
    assert result.shape == (1, 4)
    assert np.allclose(result[0], -1.0 + (2.0 + 1.5) + (0.5 - 1.5))


def test_assembly_reducer_rejects_missing_boundary_value_channel(tmp_path, monkeypatch):
    identity = "assembly-test-identity"
    monkeypatch.setattr(p07, "stage_identity", lambda output, stage: identity)
    monkeypatch.setattr(p07, "settings", lambda: {"resolutions": [2]})
    monkeypatch.setattr(p07.k, "configure", lambda root: None)
    monkeypatch.setattr(p07.k, "load", lambda n: SimpleNamespace(vol=np.ones(3)))
    monkeypatch.setattr(p07, "selected_ids", lambda stage, n, output: np.array([11]))
    np.savez(tmp_path / "N2.topology.npz", face_ids=np.array([11]), family=np.array([1]),
             endpoints=np.array([[0, -1]]))
    unit = {"n": 2, "stage": "assembly", "start": 0, "stop": 1}
    path = p07.unit_path(tmp_path, unit)
    p07.atomic_npz(path, ids=np.array([11]), family=np.array([1]), endpoints=np.array([[0, -1]]),
                   row_offsets=np.array([0, 1]), donor_ids=np.array([0]), coefficients=np.array([1.0]),
                   BC_value_coefficients=np.array([0.0]), BC_trace_donor_points=np.ones((1, 3)),
                   BC_face_positions=np.array([0]), BC_tangential=np.zeros((1, 9, 2)),
                   BC_trace_target_points=np.ones((1, 9, 3)),
                   flux=np.zeros((1, 4)), oracle_flux=np.zeros((1, 4)))
    p07.atomic_json(path.with_suffix(".json"), {"identity": identity, "unit": unit,
                                                "sha256": p07.sha(path)})
    with pytest.raises(ValueError, match="BC value channel"):
        p07.reduce_stage(tmp_path, tmp_path, "assembly", {"stages": {"assembly": {"2": [unit]}}})


def test_summary_uses_assembled_flux_and_independent_controls(tmp_path, monkeypatch):
    fields = ["phi_mms", "Ti_mms", "regular_neumann", "mixed_eta_neumann"]
    monkeypatch.setattr(p07, "settings", lambda: {"resolutions": [32, 48, 64], "fields": fields,
                                                  "global_l2_order_minimum": 1.8,
                                                  "bounded_reference_fraction_maximum": 0.1})
    monkeypatch.setattr(p07, "current_identity", lambda output: "source")
    monkeypatch.setattr(p07, "stage_identity", lambda output, stage: stage)
    monkeypatch.setattr(p07, "valid_unit", lambda output, unit, identity: True)
    monkeypatch.setattr(p07.k, "configure", lambda root: None)
    monkeypatch.setattr(p07.k, "load", lambda n: SimpleNamespace(n=n, vol=np.ones(1),
                        g=SimpleNamespace(owner_flat_ids=np.array([0]))))
    monkeypatch.setattr(p07, "region_masks", lambda t, ep, fa: {"all": np.array([True])})
    for stage in p07.STAGES:
        p07.atomic_json(tmp_path / f"{stage}_reduction.json", {"identity": stage, "all_complete": True,
                                                              "unsupported_count": 0})
    plan = {"stages": {"assembly": {}}}
    reference = {}; controls = {}; face_controls = {}
    for n in (32, 48, 64):
        error = (32 / n)**2
        unit = {"n": n, "stage": "assembly", "start": 0, "stop": 1}
        plan["stages"]["assembly"][str(n)] = [unit]
        p07.atomic_npz(p07.unit_path(tmp_path, unit), endpoints=np.array([[-1, 0]]),
                       flux=np.full((1, 4), error), oracle_flux=np.full((1, 4), error / 2))
        np.savez(tmp_path / f"N{n}.topology.npz", endpoints=np.array([[-1, 0]]), family=np.array([1]))
        reference[f"N{n}.numerator_q3"] = np.zeros((1, 4))
        reference[f"N{n}.volume_q3"] = np.ones(1)
        controls[f"N{n}.owner_ids"] = np.array([0])
        face_controls[f"N{n}.owner_ids"] = np.array([0])
        for q in (3, 5, 7): controls[f"N{n}.average_q{q}"] = np.zeros((1, 4))
        for q in (3, 7): face_controls[f"N{n}.oracle_q{q}"] = np.full((1, 4), error / 2)
    p07.atomic_npz(tmp_path / "reference_q3.npz", **reference)
    p07.atomic_npz(tmp_path / "reference_controls.npz", **controls)
    p07.atomic_npz(tmp_path / "face_controls.npz", **face_controls)
    result = p07.summarize(tmp_path, tmp_path, plan)
    assert result["computation_completed"] and result["global_order_pass"]
    assert result["reference_qualified_by_bounded_checks"]
    assert np.allclose(result["fields"]["phi_mms"]["orders"], [2.0, 2.0])
    assert result["resolutions"]["32"]["stats"]["phi_mms"]["reconstruction_q3_l2"] == 0.5
