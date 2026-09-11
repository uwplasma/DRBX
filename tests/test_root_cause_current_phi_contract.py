"""Recount the persisted true-hit current/phi contract audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "boundary_load_audit"


def test_true_hit_current_phi_and_production_force_contracts() -> None:
    audit = json.loads(
        (WORK / "root_cause_current_phi_contract.json").read_text()
    )
    fixture = audit["fixture"]
    homogeneous = audit["homogeneous_pair"]
    force = audit["production_electron_force"]
    control = audit["interior_zero_trace_control"]
    green = audit["green_remainder"]

    # This is the preserved pre-fix experiment, not a test of today's RHS.
    # Pin its recorded source identity and raw arrays while allowing the live
    # production source to be repaired. Live current-closure regressions are
    # in test_current_phi_boundary_payload_isolation.py.
    assert fixture["rhs_source_sha256"] == (
        "113ce5219f9cc901171da5618bf2974f4c3432da881d94a96bc89dfa0aafb7e2"
    )
    assert fixture["fixture_source_sha256"] == (
        "5a6184c94f45487f718ccc6c44e3fb63fff399e9039fef81b3e347e046378692"
    )
    assert hashlib.sha256(
        (WORK / "root_cause_current_phi_contract.npz").read_bytes()
    ).hexdigest() == "2852c0a9d3fe4ff510de9e78a2ea5da5fec80459f01d5eb696db45475a180b40"
    assert fixture["physical_hit_count"] > 0
    assert fixture["support_core_count"] > 0
    assert fixture["wall_normalized_B_abs_min"] > 1.0e-3
    assert fixture["plasma_phi_face_abs_max"] > 1.0e-2
    assert fixture["phi_neumann_payload_abs_max"] > 1.0e-3
    assert fixture["density_neumann_payload_abs_max"] > 1.0e-3
    assert fixture["physical_ghost_filler_type"] == (
        "MetricAwarePhysicalGhostCellFiller3D"
    )
    ghost = fixture["physical_neumann_ghost_response"]
    assert ghost["neumann_normal_scheme"] == "physical"
    assert ghost["upper_radial_ghost_response_nonzero"]
    assert ghost["upper_radial_ghost_response_max_abs"] > 1.0e-3

    # In this corrected fixture the copied density Neumann template produces
    # no affine current offset. This is deliberately fixture-scoped evidence.
    assert homogeneous["D0_zero_abs_max"] < 2.0e-10
    assert homogeneous["D0_zero_with_zero_density_payload_abs_max"] < 2.0e-10
    assert homogeneous["characteristic_density_template_lift_abs_max"] < 2.0e-10
    assert homogeneous["D0_affine_linearity_error_abs_max"] < 2.0e-10
    assert homogeneous["G0_one_wall_abs_max"] > 1.0e-3
    assert homogeneous["weighted_adjoint_residual_abs"] < 2.0e-10
    assert homogeneous["random_weighted_adjoint_residual_abs"] < 2.0e-10
    assert homogeneous["common_shift_response_prediction_error_abs_max"] < 2.0e-10

    assert force["composite_reconstruction_error_abs_max"] < 2.0e-10
    assert force["Gphys_minus_G0_abs_max"] > 1.0e-3
    assert force["explicit_phi_face_lift_abs_max"] > 1.0e-3
    assert force["support_Gphys_minus_support_abs_max"] < 2.0e-10
    assert force["joint_plasma_wall_shift_composite_abs_max"] < 2.0e-10
    assert force["joint_plasma_wall_shift_Ve_force_abs_max"] < 5.0e-7

    assert control["wall_trace_response_abs_max"] < 2.0e-10
    assert control["current_probe_coordinate_boundary_abs_max"] < 2.0e-10
    assert control["current_probe_mapped_endpoint_trace_abs_max"] < 2.0e-10
    assert control["homogeneous_G0_D0_green_residual_abs"] < 2.0e-10
    assert control["production_Gphys_D0_green_residual_abs"] > 1.0e-6
    assert control["production_green_residual_relative_to_work_sum"] > 1.0e-4
    assert control["mismatch_nonzero_support_core_count"] > 0
    assert control["mismatch_nonzero_physical_hit_count"] > 0
    assert green["decomposition_error_abs"] < 2.0e-10
    assert not green["physical_face_quadrature_power_verified"]

    # Independently recount both Green residuals from the saved arrays. This
    # lets reviewers verify the decisive scalar without rebuilding the tracer.
    with np.load(WORK / "root_cause_current_phi_contract.npz") as raw:
        mass = raw["mass"]
        phi_probe = raw["phi_probe"]
        current_probe = raw["current_probe"]
        d0_probe = raw["D0_current_probe_centered"]
        g0_probe = raw["G0_phi_probe"]
        gphys_probe = raw["Gphys_phi_probe"]
        d_work = float(np.sum(mass * phi_probe * d0_probe))
        g0_work = float(np.sum(mass * g0_probe * current_probe))
        gphys_work = float(np.sum(mass * gphys_probe * current_probe))

    np.testing.assert_allclose(d_work, control["D0_work"], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(g0_work, control["G0_work"], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        gphys_work, control["Gphys_work"], rtol=0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(d_work + g0_work, 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        abs(d_work + gphys_work),
        control["production_Gphys_D0_green_residual_abs"],
        rtol=0.0,
        atol=2.0e-13,
    )
