"""Production toroidal-topology driver contract."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import simulate_hsx_blob as hsx


def test_topology_descriptor_supports_only_toroidal():
    toroidal = hsx.topology_descriptor("toroidal")
    assert toroidal.axis_regular_axes == (True, False, False)
    assert toroidal.periodic_axes == (False, True, True)
    with pytest.raises(
        ValueError, match="the HSX backend requires a toroidal geometry artifact"
    ):
        hsx.topology_descriptor("square")


def test_parser_defaults_use_production_operator_pair():
    args = hsx._build_parser().parse_args(())
    assert args.geometry is None
    assert not hasattr(args, "topology")
    assert not hasattr(args, "resolution")
    assert not hasattr(args, "curvature_scheme")
    assert not hasattr(args, "poisson_bracket_scheme")
    assert not hasattr(args, "gmres_preconditioner")


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


def _main_error(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as error:
        hsx.main(argv)
    assert error.value.code == 2
    return capsys.readouterr().err


def test_main_requires_a_geometry_artifact(capsys):
    assert "--geometry" in _main_error(capsys)


def test_toroidal_artifact_requires_even_ntheta(fake_geometry_artifact, capsys):
    path, _ = fake_geometry_artifact(hsx, shape=(8, 47, 16))
    assert "NTHETA must be even" in _main_error(capsys, "--geometry", str(path))


def test_composite_even_ntheta_artifact_passes_the_ntheta_check(
    fake_geometry_artifact, capsys
):
    path, _ = fake_geometry_artifact(hsx, shape=(8, 48, 16))
    message = _main_error(capsys, "--geometry", str(path), "--shard-counts", "2", "1", "1")
    assert "NTHETA" not in message
    assert "eta-only" in message


def test_production_driver_rejects_radial_and_poloidal_sharding(
    fake_geometry_artifact, capsys
):
    path, _ = fake_geometry_artifact(hsx)
    for shard_counts in ((2, 1, 1), (1, 2, 1)):
        message = _main_error(
            capsys,
            "--geometry",
            str(path),
            "--shard-counts",
            *(str(value) for value in shard_counts),
        )
        assert "eta-only" in message


def test_parser_exposes_eta_sharding_without_a_topology_specific_option():
    args = hsx._build_parser().parse_args(
        ("--shard-counts", "1", "1", "2")
    )
    assert args.shard_counts == [1, 1, 2]


def test_toroidal_production_requirements_are_explicit_in_main_source():
    source = open(hsx._driver.__file__, encoding="utf-8").read()
    main = source[source.index("def main("):]
    # The redundant post-hoc topology check is gone; topology_descriptor()
    # itself is the single place that rejects a non-toroidal artifact, and
    # main() surfaces that failure through its --geometry load try/except.
    assert 'descriptor.name != "toroidal"' not in main
    assert "could not load --geometry artifact" in main
    assert "the HSX backend requires a toroidal geometry artifact" in source
    assert "simulation_geometry.polar_angular_geometry" in main
    assert "build_metric_aware_polar_angular_agglomeration_geometry" not in main
    assert "build_sharded_polar_angular_agglomeration_payload" in main
    assert "lower_polar_angular_agglomeration_geometry" not in main
    assert "lower_pole_control_volume_geometry(" not in main
