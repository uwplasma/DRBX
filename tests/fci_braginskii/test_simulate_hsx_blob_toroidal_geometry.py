"""Production toroidal-topology driver contract."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drbx.fci_braginskii import run as hsx


def test_topology_descriptor_supports_only_toroidal():
    toroidal = hsx.topology_descriptor("toroidal")
    assert toroidal.axis_regular_axes == (True, False, False)
    assert toroidal.periodic_axes == (False, True, True)
    with pytest.raises(
        ValueError, match="the HSX backend requires a toroidal geometry artifact"
    ):
        hsx.topology_descriptor("square")






def _main_error(hsx, **config) -> str:
    def fail(message):
        raise SystemExit(message)

    with pytest.raises(SystemExit) as error:
        hsx.run(hsx.HsxBlobConfig(**config), fail=fail)
    return str(error.value)


def test_toroidal_artifact_requires_even_ntheta(fake_geometry_artifact):
    path, _ = fake_geometry_artifact(hsx, shape=(8, 47, 16))
    assert "NTHETA must be even" in _main_error(hsx, geometry=path)


def test_composite_even_ntheta_artifact_passes_the_ntheta_check(fake_geometry_artifact):
    path, _ = fake_geometry_artifact(hsx, shape=(8, 48, 16))
    message = _main_error(hsx, geometry=path, shard_counts=(2, 1, 1))
    assert "NTHETA" not in message and "eta-only" in message


def test_production_driver_rejects_radial_and_poloidal_sharding(fake_geometry_artifact):
    path, _ = fake_geometry_artifact(hsx)
    for shard_counts in ((2, 1, 1), (1, 2, 1)):
        assert "eta-only" in _main_error(hsx, geometry=path, shard_counts=shard_counts)


def test_toroidal_production_requirements_are_explicit_in_main_source():
    source = open(hsx.__file__, encoding="utf-8").read()
    main = source[source.index("def run("):]
    # The redundant post-hoc topology check is gone; topology_descriptor()
    # itself is the single place that rejects a non-toroidal artifact, and
    # main() surfaces that failure through its --geometry load try/except.
    assert 'descriptor.name != "toroidal"' not in main
    assert "could not load geometry artifact" in main
    assert "the HSX backend requires a toroidal geometry artifact" in source
    assert "simulation_geometry.polar_angular_geometry" in main
    assert "build_metric_aware_polar_angular_agglomeration_geometry" not in main
    assert "build_sharded_polar_angular_agglomeration_payload" in main
    assert "lower_polar_angular_agglomeration_geometry" not in main
    assert "lower_pole_control_volume_geometry(" not in main
