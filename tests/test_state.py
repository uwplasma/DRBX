from __future__ import annotations

from drbx.runtime.state import SimulationState


def test_simulation_state_updates_immutably() -> None:
    state = SimulationState(time=1.0)

    next_state = state.with_field("Ne", 3.0).with_diagnostic("rhs_norm", 1.2).advance_time(0.5)

    assert state.fields == {}
    assert state.diagnostics == {}
    assert next_state.fields == {"Ne": 3.0}
    assert next_state.diagnostics == {"rhs_norm": 1.2}
    assert next_state.time == 1.5
