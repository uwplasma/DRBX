"""Low-cost contract tests for the work-only actual-HSX fixture builder."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))

from root_cause_hsx_fixture import (  # noqa: E402
    _capture_driver_inputs,
    _field_change_record,
    _source_hashes,
    _validate_resolution,
)


class _FakeSimulation:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.original_calls = 0

        def original(*_args, **_kwargs):
            self.original_calls += 1

        self.run_full_eb = original

    def main(self, argv):
        assert argv == ["--resolution", "4", "6", "8"]
        self.run_full_eb("state", timestep=0.25)
        if self.fail:
            raise RuntimeError("driver failed after intercept")


def test_capture_restores_production_hook_and_returns_setup_kwargs():
    simulation = _FakeSimulation()
    original = simulation.run_full_eb
    state, kwargs = _capture_driver_inputs(
        simulation, ["--resolution", "4", "6", "8"]
    )
    assert simulation.run_full_eb is original
    assert state == "state"
    assert kwargs == {"timestep": 0.25}
    assert simulation.original_calls == 0


def test_capture_restores_production_hook_when_driver_raises():
    simulation = _FakeSimulation(fail=True)
    original = simulation.run_full_eb
    with pytest.raises(RuntimeError, match="driver failed"):
        _capture_driver_inputs(simulation, ["--resolution", "4", "6", "8"])
    assert simulation.run_full_eb is original


def test_resolution_guard_bounds_full_packed_upper_limit():
    assert _validate_resolution((4, 6, 8)) == (4, 6, 8)
    with pytest.raises(ValueError, match="1500 packed-unknown"):
        _validate_resolution((6, 8, 8))
    with pytest.raises(ValueError, match="three positive"):
        _validate_resolution((4, 0, 8))


def test_wall_initializer_change_record_keeps_each_field_distinct():
    initial = SimpleNamespace(
        **{name: np.ones((2, 1, 1), dtype=float) for name in (
            "density", "Te", "Ti", "Vi", "Ve", "vorticity", "phi"
        )}
    )
    changed_values = {
        name: np.asarray(getattr(initial, name)).copy()
        for name in ("density", "Te", "Ti", "Vi", "Ve", "vorticity", "phi")
    }
    changed_values["Vi"][1, 0, 0] = 3.0
    changed_values["vorticity"][:] = 0.5
    record = _field_change_record(initial, SimpleNamespace(**changed_values))
    assert record["changed_fields"] == ["Vi", "vorticity"]
    assert record["fields"]["Vi"]["changed_entry_count"] == 1
    assert record["fields"]["Vi"]["max_abs_change"] == 2.0
    assert record["fields"]["phi"]["changed"] is False


def test_cache_identity_hashes_cover_fixture_and_wall_operator_chain():
    hashes = _source_hashes()
    required = {
        "work/boundary_load_audit/root_cause_hsx_fixture.py",
        "src/drbx/native/fci_boundary_imex_model.py",
        "src/drbx/native/fci_drb_EB_rhs.py",
        "src/drbx/native/fci_halo.py",
        "src/drbx/native/fci_operators.py",
        "src/drbx/native/fci_physical_wall.py",
    }
    assert required.issubset(hashes)
    assert all(len(value) == 64 for value in hashes.values())
