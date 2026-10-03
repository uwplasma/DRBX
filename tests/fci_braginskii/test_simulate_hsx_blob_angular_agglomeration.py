"""Driver-level production angular-RLP tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import simulate_hsx_blob as driver


def test_removed_axis_experiment_cli_options_are_absent():
    parser = driver._build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "axis_treatment" not in destinations
    assert "pole_owner_profile" not in destinations
    assert "pole_collapsed_radial_rings" not in destinations
    assert "phi_solver_space" not in destinations
    assert "axis_core_state_space" not in destinations
    assert "gmres_preconditioner" not in destinations


def test_parser_removes_producer_geometry_controls():
    destinations = {action.dest for action in driver._build_parser()._actions}
    for removed in (
        "angular_group_profile",
        "topology",
        "resolution",
        "makegrid",
        "vessel",
        "fci_trace_substeps",
        "metric_mesh_shape",
        "metric_cache_dir",
    ):
        assert removed not in destinations


def test_driver_has_one_canonical_toroidal_lowering():
    source = open(driver.__file__, encoding="utf-8").read()
    assert "build_sharded_polar_angular_agglomeration_payload" in source
    assert "assemble_local_polar_angular_agglomeration_geometry" in source
    main_source = source[source.index("def main("):]
    assert "lower_polar_angular_agglomeration_geometry" not in main_source
    assert "lower_pole_control_volume_geometry(" not in main_source
    assert "control_volume_operator_mode" not in source
