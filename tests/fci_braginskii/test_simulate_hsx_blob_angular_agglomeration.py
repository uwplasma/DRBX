"""Driver-level production angular-RLP tests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drbx.fci_braginskii import run as driver






def test_driver_has_one_canonical_toroidal_lowering():
    source = open(driver.__file__, encoding="utf-8").read()
    assert "build_sharded_polar_angular_agglomeration_payload" in source
    assert "assemble_local_polar_angular_agglomeration_geometry" in source
    main_source = source[source.index("def run("):]
    assert "lower_polar_angular_agglomeration_geometry" not in main_source
    assert "lower_pole_control_volume_geometry(" not in main_source
    assert "control_volume_operator_mode" not in source
