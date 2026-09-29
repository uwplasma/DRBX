"""Tests for the P08 step-1 frozen replay-owner selection (design task 6).

See ``work/p08_step1_consolidation_design_20260928/design.md`` sections 5-6.
These tests only exercise geometry/topology loading (the frozen HSX owner
map) and small non-result config/fixture JSON files -- never a saved
campaign action/result array.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import selection  # noqa: E402

# Every test in this module loads real HSX geometry/topology (via
# selection.build_selection/verify_frozen/freeze) behind an inline
# _require_inputs skip gate -- the whole file is real-data.
pytestmark = pytest.mark.slow

FROZEN_ROOT = WORKSPACE / "work/p08_step1_replay_selection_20260928"


def _geometry_available(n: int) -> bool:
    directory = GEOMETRY / f"{n}x{n}x{n}"
    return (directory / "base_geometry.npz").is_file() and (directory / "rlp_topology.npz").is_file()


def _fixtures_available(n: int) -> bool:
    for fixtures in (selection.P05N_FIXTURES, selection.P06N_FIXTURES):
        if not (fixtures / f"N{n}.selection.json").is_file():
            return False
    return selection.P05_DIRECT_CONFIG.is_file() and selection.P06_STRUCTURED_CONFIG.is_file()


def _require_inputs(n: int) -> None:
    if not (_geometry_available(n) and _fixtures_available(n)):
        pytest.skip(f"geometry/fixture inputs for N{n} are unavailable")


@pytest.mark.parametrize("n", selection.GRIDS)
def test_build_selection_is_deterministic(n: int) -> None:
    """Same inputs, same explicit timestamp -> byte-for-byte identical payload."""
    _require_inputs(n)
    first = selection.build_selection(n, timestamp="2026-09-28T00:00:00+00:00")
    second = selection.build_selection(n, timestamp="2026-09-28T00:00:00+00:00")
    assert first == second
    # And the exact serialized form (what actually gets hashed/frozen) matches too.
    assert selection._serialize(first) == selection._serialize(second)


@pytest.mark.parametrize("n", selection.GRIDS)
def test_every_coverage_category_is_nonempty(n: int) -> None:
    """Every design-section-5 coverage category must contribute at least one owner."""
    _require_inputs(n)
    payload = selection.build_selection(n, timestamp="2026-09-28T00:00:00+00:00")
    categories = payload["categories"]
    expected = {
        "radial_axis_0_1",
        "profile_transitions_pm1",
        "singleton_ring_and_before",
        "aggregate_layer",
        "midplane_n_over_2",
        "wall_band",
        "theta_eta_seams",
        "existing_preflight_owners",
        "p05_hotspots",
    }
    assert expected <= categories.keys()
    for name in expected:
        assert len(categories[name]["owners"]) > 0, f"category {name!r} is empty for N{n}"

    closure = payload["closure"]
    assert len(closure["owners"]) > 0
    assert len(closure["raw_member_ids"]) > 0
    assert len(closure["p07_census_face_ids"]) > 0
    assert len(closure["p06_slot_census_face_ids"]) > 0
    assert len(closure["legacy_alias_slot_ids"]) > 0
    # Alias slots (P06's periodic duplicate theta/eta slot n) are a subset of
    # the full P06 slot-census closure.
    assert set(closure["legacy_alias_slot_ids"]) <= set(closure["p06_slot_census_face_ids"])
    # Every raw member's owner must be one of the selected closure owners.
    assert closure["owners"] == sorted(set(closure["owners"]))


@pytest.mark.parametrize("n", selection.GRIDS)
def test_frozen_file_matches_recomputation(n: int) -> None:
    """The committed frozen selection.json / SELECTION_SHA256 pair, if present,
    must reproduce byte-for-byte under the same recorded timestamp."""
    grid_dir = FROZEN_ROOT / f"N{n}"
    if not ((grid_dir / "selection.json").is_file() and (grid_dir / "SELECTION_SHA256").is_file()):
        pytest.skip(f"no frozen selection recorded yet for N{n}")
    _require_inputs(n)
    matches, recomputed, recorded = selection.verify_frozen(grid_dir)
    assert matches, f"frozen N{n} selection no longer reproduces from current geometry/fixture inputs"
    assert recomputed["n"] == recorded["n"] == n
    assert recomputed["closure"]["owners"] == recorded["closure"]["owners"]


def test_freeze_writes_sha256_of_exact_serialized_file(tmp_path) -> None:
    """freeze() must record the sha256 of the literal bytes written to disk."""
    _require_inputs(selection.GRIDS[0])
    written = selection.freeze(tmp_path, grids=(selection.GRIDS[0],), timestamp="2026-09-28T00:00:00+00:00")
    info = written[selection.GRIDS[0]]
    on_disk = info["selection_path"].read_text()
    assert selection._sha256_bytes(on_disk.encode()) == info["sha256"]
    assert info["sha256_path"].read_text().strip() == info["sha256"]
