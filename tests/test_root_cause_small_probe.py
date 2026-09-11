"""Focused contracts for the work-only representative boundary probe."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work" / "boundary_load_audit"))

from root_cause_small_probe import (  # noqa: E402
    ProbeConfig,
    audit_bohm_branches,
    audit_physical_neumann_ghost_response,
    audit_topology,
    build_representative_fixture,
    stage_admission,
)


@pytest.fixture(scope="module")
def representative_fixture():
    return build_representative_fixture(ProbeConfig())


def test_representative_fixture_has_real_wall_and_ordinary_ghost_dependencies(
    representative_fixture,
):
    fixture = representative_fixture
    topology, _arrays = audit_topology(fixture)
    assert topology["augmented_global_polarization"]
    assert topology["polarization_operator_form"] == "support-paired"
    assert fixture.model.neumann_normal_scheme == "physical"
    assert fixture.construction["physical_ghost_filler_type"] == "MetricAwarePhysicalGhostCellFiller3D"
    assert fixture.construction["conflicting_wall_owner_count"] > 0
    for direction in ("backward", "forward"):
        info = topology["directions"][direction]
        assert info["physical_wall_endpoint_count"] > 0
        assert info["ordinary_mapped_endpoint_count"] > 0
        assert info["wall_targets_with_radial_ghost_support"] == info["physical_wall_endpoint_count"]
        assert info["ordinary_targets_with_radial_ghost_support"] > 0
        assert info["wall_targets_on_lower_radial_owner_plane"] == 0
        assert info["wall_normalized_B_abs_min"] > 0.0


def test_true_endpoint_bohm_groups_and_physical_neumann_response(
    representative_fixture,
):
    branch, _arrays = audit_bohm_branches(representative_fixture)
    for direction in ("backward", "forward"):
        counts = branch["directions"][direction]["branch_counts"]
        assert counts["subsonic"] > 0
        assert counts["tie"] > 0
        assert counts["supersonic"] > 0
        assert branch["directions"][direction]["resolved_vi_identity_max_abs"] < 1.0e-12
    response, _filled = audit_physical_neumann_ghost_response(representative_fixture)
    assert response["neumann_normal_scheme"] == "physical"
    assert response["upper_radial_ghost_response_nonzero"]
    assert response["upper_radial_ghost_response_max_abs"] > 0.0


def test_stage_admission_uses_consistent_absolute_or_relative_gate():
    config = ProbeConfig()
    base = {
        "converged": True,
        "initial_residual_l2": 1.0e-8,
        "final_residual_l2": config.stage_absolute_tolerance,
        "linear_history": [{"iterations": 1, "residual_l2": 1.0e-20}],
        "final_diagnostics": {"admissibility": {"admissible": True, "reason": "ok"}},
    }
    accepted = stage_admission(base, config)
    assert accepted["admitted"]
    assert accepted["scale_aware_target"] == config.stage_absolute_tolerance

    rejected = stage_admission({
        **base,
        "final_diagnostics": {"admissibility": {"admissible": False, "reason": "negative_sheath_drop_x"}},
    }, config)
    assert not rejected["admitted"]
    assert "final_stage_iterate_not_physically_admissible" in rejected["reasons"]


def test_config_rejects_nonrepresentative_two_cell_radial_extent():
    with pytest.raises(ValueError, match="at least 3"):
        ProbeConfig(shape=(2, 4, 3))
