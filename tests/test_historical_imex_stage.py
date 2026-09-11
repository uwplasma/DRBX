"""Contracts for the reusable historical SSP222 split orchestration."""

import jax.numpy as jnp
import numpy as np

from drbx.native import FciDrbEBState
from simulate_hsx_blob import historical_imex_ssp222_stage


def _state(value):
    a = jnp.asarray(value, dtype=jnp.float64)
    return FciDrbEBState(*(a + i for i in range(7)))


def test_historical_helper_matches_reference_sequence():
    current = _state(1.0)
    source = _state(0.0).replace(**{
        name: jnp.asarray([0.1, 0.2]) for name in _state(0.0).field_names()
    })
    dt = 0.04
    gamma = 0.25
    calls = []

    def implicit(base, model, solve_dt, selection_dt):
        calls.append(("implicit", float(solve_dt), float(selection_dt)))
        rate = base.map_fields(lambda value: 0.1 * value)
        return base.axpy(rate, scale=solve_dt), rate, float(solve_dt)

    def explicit(stage, phi, model, source):
        calls.append(("explicit", float(source.density)))
        return stage.map_fields(lambda value: 0.2 * value)

    def reconstruct(state, model):
        return state.phi + 0.0, float(state.density)

    result = historical_imex_ssp222_stage(
        current, object(), source, dt, gamma=gamma,
        implicit_stage=implicit, explicit_operator=explicit,
        reconstruct_phi=reconstruct,
    )
    next_state, stages, rates, infos = result

    # Independently spell out the same two-stage SSP222 algebra.
    gdt = gamma * dt
    s1, i1, _ = implicit(current, object(), gdt, dt)
    e1 = explicit(s1, s1.phi, object(), source.replace(**{
        n: getattr(source, n)[0] for n in source.field_names()
    }))
    b2 = current.axpy(e1, scale=dt).axpy(i1, scale=(1 - 2 * gamma) * dt)
    b2 = b2.replace(phi=b2.phi)
    s2, i2, _ = implicit(b2, object(), gdt, dt)
    e2 = explicit(s2, s2.phi, object(), source.replace(**{
        n: getattr(source, n)[1] for n in source.field_names()
    }))
    expected = current.axpy(
        e1.axpy(e2, scale=1.0).axpy(i1, scale=1.0).axpy(i2, scale=1.0).map_fields(lambda value: 0.5 * value),
        scale=dt,
    )
    for name in current.field_names():
        np.testing.assert_allclose(getattr(next_state, name), getattr(expected, name))
    assert [c[0] for c in calls[:4]] == ["implicit", "explicit", "implicit", "explicit"]
    assert len(stages) == len(rates) == 5
    assert len(infos) == 4


def test_historical_helper_applies_each_wall_callback_once():
    current = _state(1.0)
    source = _state(0.0).replace(**{
        name: jnp.asarray([0.0, 0.0]) for name in _state(0.0).field_names()
    })
    counts = {"implicit": 0, "explicit": 0, "reconstruct": 0}

    def implicit(base, model, solve_dt, selection_dt):
        counts["implicit"] += 1
        return base, base.map_fields(lambda value: 0.0), None

    def explicit(stage, phi, model, source):
        counts["explicit"] += 1
        return stage.map_fields(lambda value: 0.0)

    def reconstruct(state, model):
        counts["reconstruct"] += 1
        return state.phi, None

    historical_imex_ssp222_stage(
        current, None, source, 0.01, implicit_stage=implicit,
        explicit_operator=explicit, reconstruct_phi=reconstruct,
    )
    assert counts == {"implicit": 2, "explicit": 2, "reconstruct": 2}
