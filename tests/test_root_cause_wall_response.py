"""Focused tests for the work-only scalar wall-response audit."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))

from drbx.native.fci_halo import neumann_face_trace_physical_affine
from drbx.native.fci_physical_wall import physical_wall_model_from_name
from test_rung3_physical_normal_contract import _case

from root_cause_wall_response import (  # noqa: E402
    branch_crossing_audit,
    build_report,
    captured_bohm_audit,
    finite_secant,
    gauge_audit,
    representative_case,
    trace_distance_audit,
    wall_directional_derivative,
    wall_response,
)


def test_report_fails_clearly_when_a_saved_input_is_missing(tmp_path):
    missing = tmp_path / "missing_bohm_capture.npz"
    try:
        build_report(bohm_capture=missing)
    except FileNotFoundError as error:
        assert str(missing) in str(error)
        assert "required audit input is missing" in str(error)
    else:
        raise AssertionError("missing input was silently accepted")


def test_hand_wall_derivative_matches_branch_local_secants_with_ti_and_trace_inputs():
    for branch in ("bohm", "outflow"):
        values, direction = representative_case(branch=branch)
        hand = wall_directional_derivative(values, direction, branch=branch)
        coarse = finite_secant(values, direction, 1.0e-4, branch=branch)
        fine = finite_secant(values, direction, 1.0e-6, branch=branch)
        outputs = ("cb", "vi_face", "gvi", "gphi", "n_face", "phi_face", "ve_face", "current")
        coarse_error = max(abs(coarse[name] - hand[name]) for name in outputs)
        fine_error = max(abs(fine[name] - hand[name]) for name in outputs)
        assert fine_error < 2.0e-2 * coarse_error

        ti_column = wall_directional_derivative(values, {"ti": 1.0}, branch=branch)
        np.testing.assert_allclose(ti_column["cb"], values["tau"] / (2.0 * wall_response(values)["cb"]))
        assert abs(ti_column["gphi"]) > 0.0
        assert abs(ti_column["current"]) > 0.0

        vi_trace_column = wall_directional_derivative(values, {"vi_base": 1.0}, branch=branch)
        assert abs(vi_trace_column["gvi"]) > 0.0
        assert abs(vi_trace_column["current"]) > 0.0


def test_uniform_distance_scaling_cancels_from_reconstructed_face_state_and_current():
    audit = trace_distance_audit()
    for row in audit["rows"]:
        np.testing.assert_allclose(row["gvi_times_scale_over_reference"], 1.0, atol=2.0e-15)
        np.testing.assert_allclose(row["gphi_times_scale_over_reference"], 1.0, atol=2.0e-15)
        np.testing.assert_allclose(row["n_face_difference"], 0.0, atol=2.0e-15)
        np.testing.assert_allclose(row["phi_face_difference"], 0.0, atol=2.0e-15)
        np.testing.assert_allclose(row["current_difference"], 0.0, atol=2.0e-14)


def test_finite_max_crossing_breaks_the_base_branch_linearization_only_after_crossing():
    audit = branch_crossing_audit()
    rows = audit["rows"]
    assert audit["predicted_crossing_alpha"] == 0.05
    assert all(row["endpoint_branch"] == "outflow" for row in rows[:2])
    assert all(row["error_against_base_outflow_tangent"] < 0.05 for row in rows[:2])
    assert rows[3]["endpoint_branch"] == "bohm"
    assert rows[-1]["error_against_base_outflow_tangent"] > rows[2]["error_against_base_outflow_tangent"]


def test_common_wall_and_plasma_potential_shift_leaves_electron_target_and_current_invariant():
    audit = gauge_audit()
    np.testing.assert_allclose(audit["common_phi_and_wall_shift_phi_face_derivative"], 1.0)
    np.testing.assert_allclose(audit["common_phi_and_wall_shift_ve_derivative"], 0.0, atol=2.0e-14)
    np.testing.assert_allclose(audit["common_phi_and_wall_shift_current_derivative"], 0.0, atol=2.0e-14)
    assert abs(audit["fixed_wall_phi_shift_current_derivative"]) > 1.0e-3


def test_saved_initial_capture_recounts_direction_selected_bohm_faces_without_jax():
    path = Path(__file__).parents[1] / "work/boundary_load_audit/bohm_switch_derivative_v1.npz"
    audit = captured_bohm_audit(path)
    assert audit["active_faces"] == 2304
    assert audit["near_ties"] == 2304
    positive = next(row for row in audit["rows"] if row["alpha"] == 1.0e-5)
    negative = next(row for row in audit["rows"] if row["alpha"] == -1.0e-5)
    assert (positive["entered_bohm_faces"], positive["entered_outflow_faces"]) == (1162, 1142)
    assert (negative["entered_bohm_faces"], negative["entered_outflow_faces"]) == (1142, 1162)
    assert positive["secant_vs_entered_tangent_relative"] < 2.0e-7
    assert positive["jax_max_tangent_vs_entered_relative"] > 0.1


def test_scalar_transcription_matches_current_production_wall_fixture():
    _layout, domain, geometry, _filler, state, params, halos = _case()
    model = replace(
        physical_wall_model_from_name("simplified-gbs-mpe"),
        conducting_sheath_wall_potential=0.0,
    )
    bundle = model(state, geometry, domain, params, topology_halos=halos)
    side, index, lane = "upper", -1, (2, 1)

    te, _ = neumann_face_trace_physical_affine(halos["Te"], geometry, domain, 0, side)
    ti, _ = neumann_face_trace_physical_affine(halos["Ti"], geometry, domain, 0, side)
    vi_base, vi_response = neumann_face_trace_physical_affine(halos["Vi"], geometry, domain, 0, side)
    n_base, n_response = neumann_face_trace_physical_affine(halos["density"], geometry, domain, 0, side)
    phi_base, phi_response = neumann_face_trace_physical_affine(halos["phi"], geometry, domain, 0, side)
    values = {
        "te": float(te[lane]),
        "ti": float(ti[lane]),
        "tau": params.tau,
        "sigma": 1.0,
        "vi_owner": float(state.Vi[index][lane]),
        "vi_base": float(vi_base[lane]),
        "vi_response": float(vi_response[lane]),
        "n_base": float(n_base[lane]),
        "n_response": float(n_response[lane]),
        "phi_base": float(phi_base[lane]),
        "phi_response": float(phi_response[lane]),
        "mu": params.mi_over_me,
        "wall": 0.0,
    }
    transcribed = wall_response(values)
    np.testing.assert_allclose(transcribed["vi_face"], bundle.Vi.value_x[index][lane], atol=2.0e-13)
    np.testing.assert_allclose(-transcribed["k"] * transcribed["n_face"], bundle.density.value_x[index][lane], atol=2.0e-13)
    np.testing.assert_allclose(transcribed["gphi"], bundle.phi.value_x[index][lane], atol=2.0e-13)
    np.testing.assert_allclose(transcribed["ve_face"], bundle.Ve.value_x[index][lane], atol=2.0e-13)
