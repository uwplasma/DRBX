import argparse
import json

import numpy as np
import pytest

from scripts.p05_direct_midpoint_global import campaign, direct_operator


def test_coordinate_bracket_is_antisymmetric_and_uses_abs_jacobian():
    rng = np.random.default_rng(6041)
    h = rng.normal(size=(9, 3))
    ga = rng.normal(size=(9, 3))
    gb = rng.normal(size=(9, 3))
    jacobian = -rng.uniform(0.2, 3.0, size=9)
    ab = direct_operator.point_bracket(h, jacobian, ga, gb)
    ba = direct_operator.point_bracket(h, jacobian, gb, ga)
    explicit = -np.einsum("ij,ij->i", np.cross(h, ga), gb) / np.abs(jacobian)
    np.testing.assert_array_equal(ab, explicit)
    np.testing.assert_allclose(ab + ba, 0.0, rtol=0.0, atol=2e-15)


def test_owner_projection_uses_every_raw_member_and_physical_volume():
    action = np.array([[1.0, -1.0], [3.0, 2.0], [5.0, 4.0], [7.0, 6.0]])
    raw_volume = np.array([1.0, 3.0, 2.0, 2.0])
    raw_owner = np.array([0, 0, 1, 1])
    owner_volume = np.array([4.0, 4.0])
    got = direct_operator.project_raw_to_owners(action, raw_volume, raw_owner, owner_volume)
    expected = np.array([[2.5, 1.25], [6.0, 5.0]])
    np.testing.assert_array_equal(got, expected)


def test_saved_u_minus_a_is_added_as_a_distinct_candidate():
    centered = np.array([[2.0, -3.0], [0.5, 0.25]])
    saved_jump = np.array([[0.1, 0.4], [-0.2, 0.5]])
    candidates = direct_operator.candidate_variants(centered, saved_jump)
    np.testing.assert_array_equal(candidates[:, :, 0], centered)
    np.testing.assert_array_equal(candidates[:, :, 1], centered + saved_jump)
    assert not np.shares_memory(candidates[:, :, 0], centered)


def test_regional_masks_keep_axis_first_wall_adjacent_and_aggregate_owners():
    n = 12
    raw = np.arange(n**3, dtype=np.int64)
    raw_owner = raw.copy()
    a = np.ravel_multi_index((4, 0, 0), (n, n, n))
    b = np.ravel_multi_index((4, 1, 0), (n, n, n))
    raw_owner[b] = raw_owner[a]
    raw_ijk = np.array(np.unravel_index(raw, (n, n, n))).T
    owner_count = int(raw_owner.max()) + 1
    masks, meta = direct_operator.regional_owner_masks(raw_owner, raw_ijk, owner_count, n)
    axis_owner = int(raw_owner[0])
    first_owner = int(raw_owner[np.ravel_multi_index((1, 0, 0), (n,n,n))])
    adjacent_owner = int(raw_owner[np.ravel_multi_index((10, 0, 0), (n,n,n))])
    wall_owner = int(raw_owner[np.ravel_multi_index((11, 0, 0), (n,n,n))])
    aggregate_owner = int(raw_owner[a])
    assert masks["axis_core"][axis_owner]
    assert masks["first_ring"][first_owner]
    assert masks["adjacent_reconstruction_band"][adjacent_owner]
    assert masks["wall"][wall_owner]
    assert masks["aggregate"][aggregate_owner]
    assert meta["aggregate_owner_count"] == 1


def test_global_order_gate_and_regional_orders_are_descriptive():
    resolutions = (32, 48, 64)
    rms = np.array([[1/32**2], [1/48**2], [1/64**2]])
    orders = direct_operator.orders_from_rms(rms, resolutions)
    np.testing.assert_allclose(orders[:, 0], 2.0, rtol=0.0, atol=1e-14)
    assert np.isnan(direct_operator.orders_from_rms(np.zeros((3, 1)), resolutions)).all()


def test_worker_memory_cap_and_chunk_resume_identity(tmp_path):
    args = argparse.Namespace(workers=20, memory_budget_gib=10.0,
                               worker_memory_gib=2.0, memory_reserve_gib=2.0,
                               max_tasks_per_worker=8)
    assert campaign.workers(args) == 4
    identity = "frozen-test-identity"
    unit = campaign.unit_record("run", 32, 0)
    result = {"arrays": {"ids": np.arange(unit["start"], unit["stop"]),
                         "action": np.ones((unit["stop"]-unit["start"], 8))},
              "seconds": 0.01, "peak_rss_gib": 0.2}
    campaign.atomic_chunk(tmp_path, unit, identity, result)
    assert campaign.valid_chunk(tmp_path, unit, identity)
    assert not campaign.valid_chunk(tmp_path, unit, "different-identity")
    campaign.save_npz(campaign.chunk_path(tmp_path, unit), ids=np.array([99]))
    assert not campaign.valid_chunk(tmp_path, unit, identity)


def test_fresh_output_adoption_allows_runner_lock_but_rejects_unknown_files(tmp_path):
    (tmp_path / ".campaign.lock").touch()
    for name in ("chunks", "executions", "logs", "cache", "scratch", "reuse_inputs", "runtime_inputs"):
        (tmp_path / name).mkdir()
    (tmp_path / "reference_sidecar.json").write_text("{}")
    campaign.assert_adoptable_output(tmp_path)
    (tmp_path / "unexpected.txt").write_text("not campaign state")
    with pytest.raises(ValueError, match="nonempty output directory"):
        campaign.assert_adoptable_output(tmp_path)


def test_chunked_parent_reduction_matches_serial_owner_projection():
    rng = np.random.default_rng(11)
    raw_action = rng.normal(size=(37, 8))
    volume = rng.uniform(0.1, 2.0, size=37)
    owners = rng.integers(0, 7, size=37)
    owner_volume = np.bincount(owners, weights=volume, minlength=7)
    serial = direct_operator.project_raw_to_owners(raw_action, volume, owners, owner_volume)
    partial = np.zeros_like(serial)
    for start in range(0, len(owners), 5):
        stop = min(start+5, len(owners))
        for pair in range(raw_action.shape[1]):
            np.add.at(partial[:, pair], owners[start:stop], volume[start:stop]*raw_action[start:stop, pair])
    parallel_style = partial / owner_volume[:, None]
    np.testing.assert_allclose(parallel_style, serial, rtol=0.0, atol=5e-15)


def test_frozen_mapping_and_scoped_actual_omega_replay_tolerance():
    cfg = campaign.config()
    reuse = json.loads(campaign.REUSE_MANIFEST.read_text())
    assert cfg["cases"] == reuse["cases"]
    assert cfg["pairs"] == reuse["pairs"]
    assert cfg["fields"] == reuse["fields"]
    assert cfg["actual_omega_batched_replay_abs_tolerance"] == pytest.approx(1.6e-8)
    assert cfg["smooth_reference_replay_abs_tolerance"] == pytest.approx(1.0e-10)
    assert "actual-vorticity analytic reference only" in cfg["old_omega_batch_noise_provenance"]["tolerance_scope"]
    assert "actual-vorticity B/C remain unavailable" in reuse["old_exact_input_O_policy"]
    assert all("resolved_path" not in entry for entry in reuse["runtime_geometry_inputs"])
    assert all(not str(value).startswith("/") for value in reuse["source_data"].values())
