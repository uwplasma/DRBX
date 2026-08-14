"""Production toroidal-topology driver contract."""

import sys

import pytest

sys.path.insert(0, str(__file__.split("/DRBX/tests")[0]))

import simulate_hsx_blob as hsx


def test_topology_descriptors_distinguish_square_and_toroidal():
    square = hsx.topology_descriptor("square")
    toroidal = hsx.topology_descriptor("toroidal")
    assert square.axis_regular_axes == (False, False, False)
    assert square.periodic_axes == (False, False, True)
    assert toroidal.axis_regular_axes == (True, False, False)
    assert toroidal.periodic_axes == (False, True, True)


def test_parser_defaults_use_production_operator_pair():
    args = hsx._build_parser().parse_args(())
    assert args.topology == "square"
    assert args.curvature_scheme == "conservative"
    assert args.poisson_bracket_scheme == "compatible-flux"
    assert args.gmres_preconditioner == "line-u"


def test_removed_axis_experiment_options_are_rejected():
    parser = hsx._build_parser()
    for option in (
        "--axis-treatment",
        "--pole-owner-profile",
        "--pole-collapsed-radial-rings",
        "--phi-solver-space",
        "--axis-core-state-space",
        "--axis-core-gradient-degree",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args((option, "legacy-cartesian"))


def test_composite_even_ntheta_is_accepted_by_parser():
    args = hsx._build_parser().parse_args(
        ("--topology", "toroidal", "--resolution", "8", "48", "16")
    )
    assert args.resolution == [8, 48, 16]


def test_toroidal_production_requirements_are_explicit_in_main_source():
    source = open(hsx.__file__, encoding="utf-8").read()
    main = source[source.index("def main("):]
    assert 'args.topology == "toroidal"' in main
    assert "build_metric_aware_polar_angular_agglomeration_geometry" in main
    assert "lower_polar_angular_agglomeration_geometry" in main
    assert "lower_pole_control_volume_geometry(" not in main
