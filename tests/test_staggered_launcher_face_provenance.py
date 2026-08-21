"""Static contracts for the isolated staggered launcher transformation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py"
    spec = importlib.util.spec_from_file_location("staggered_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_transformation_carries_actual_face_topology_to_output_and_diagnostics():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )
    compile(transformed, str(launcher.SHARED_DRIVER), "exec")

    assert "def staggered_face_preflight(" in transformed
    assert "build_local_outgoing_fci_face_topology_from_geometry(" in transformed
    assert '"face_owner_map_sha256"' in transformed
    assert '"face_measure_sha256"' in transformed
    assert '"face_provenance_sha256"' in transformed
    assert "OUTGOING_FCI_FACE_OWNERSHIP_POLICY" in transformed
    assert '"cell_velocity_projection": "PcRc-after-face-to-center"' in transformed
    assert '"face_native_parallel_forces": "direct-Gc-and-compatible-Dc"' in transformed
    assert '"center_force_to_face_transfer": "Pe-L-Rc-mass-adjoint-f2c"' in transformed
    assert '"initial_velocity_projection": "center-to-outgoing-face-Re"' in transformed
    assert "edge_destination_support" in transformed
    assert "outgoing_face_topology_host" in transformed
    assert "prolong_local_outgoing_fci_face_owner_field(current_state.Vi, face)" in transformed
    assert 'f"face_topology_{name}"' in transformed
    assert "values[owner_i, owner_j, owner_k]" in transformed
    assert "_materialize_face_owner_array" in transformed
    assert "face_mask if name in (\"Vi\", \"Ve\") else cell_mask" in transformed
    assert "face_owner_count_basis" not in transformed

    diagnostic_calls = [
        node for node in ast.walk(ast.parse(transformed))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "diagnostic_state"
    ]
    assert diagnostic_calls
    assert all(
        len(node.args) == 3
        and isinstance(node.args[2], ast.Attribute)
        and node.args[2].attr == "outgoing_face_topology"
        for node in diagnostic_calls
    )

def test_staggered_initializer_preserves_centered_velocities_until_local_c2f_restriction():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )

    # Scalars retain the shared host P_cR_c initialization.  Vi/Ve deliberately
    # bypass it so their actual local FCI map and remote halo rows can perform
    # c2f then R_e; this is essential for any future nonzero velocity seed.
    assert 'and name in ("Vi", "Ve")' in transformed
    assert "result[name] = raw" in transformed
    assert "project_initial_staggered_velocities" in transformed
    assert "model._center_owned_to_outgoing_face(values, bc, context)" in transformed
    assert "model._restrict_fine_face_field(" in transformed
    assert "model._owner_face_field(" in transformed
    assert "_assert_owner_sparse(\n                state, owner_host_geometry, outgoing_face_topology_host" in transformed


def test_rhs_term_history_diagnostic_is_read_only_and_uses_mixed_materialization():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )

    assert "DRBX_RHS_TERM_HISTORY" in transformed
    assert "return_rhs_term_fields=True" in transformed
    assert "phi_owned=local_state.phi" in transformed
    assert "prolong_local_outgoing_fci_face_owner_field(value, model.outgoing_face_topology)" in transformed
    assert "return materialized.at[3].set(vi_face_terms).at[4].set(ve_face_terms)" in transformed
    assert "_vi_near_band_report(vi_terms, saved_vi, near_start)" in transformed
    assert "return state" in transformed


def test_rhs_term_history_arguments_are_launcher_only_and_guarded():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--rhs-term-history", type=Path)' in source
    assert 'parser.add_argument("--rhs-term-frames", default="100,180,225")' in source
    assert 'parser.add_argument("--rhs-term-output", type=Path)' in source
    assert "--rhs-term-history requires --parallel-velocity-layout fci-staggered" in source
    assert "--rhs-term-history requires --rhs-term-output" in source


def test_vi_near_band_report_has_consistent_energy_and_complex_inner_products():
    launcher = _launcher_module()
    # theta=4; retain modes 1 and 2.  The first term equals the state, while
    # the second is its negative, so their sum cancels exactly in the band.
    state = np.asarray([[[1.0], [-1.0], [1.0], [-1.0]]])
    terms = np.stack((state, -state), axis=0)
    report = launcher._vi_near_band_report(terms, state, near_start=1)
    state_spectrum = np.fft.rfft(state, axis=1)[:, 1:, :]
    expected_energy = float(np.sum(np.abs(state_spectrum) ** 2))
    assert report["rfft_normalization"] == "numpy-unnormalized"
    np.testing.assert_allclose(report["term_near_band_energy"], (expected_energy, expected_energy))
    assert report["term_near_band_inner_product_with_saved_Vi"][0] == {
        "real": expected_energy, "imag": 0.0
    }
    assert report["term_near_band_inner_product_with_saved_Vi"][1] == {
        "real": -expected_energy, "imag": 0.0
    }
    assert report["sum_term_near_band_energy"] == 0.0
    assert report["sum_term_near_band_inner_product_with_saved_Vi"] == {"real": 0.0, "imag": 0.0}
    with pytest.raises(ValueError, match="match state"):
        launcher._vi_near_band_report(np.zeros((2, 1, 4, 1)), np.zeros((1, 3, 1)), 1)
