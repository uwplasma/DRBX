from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_time_integrator import (
    ARK2_B,
    ARK2_B_EMBEDDED,
    ARK2_C,
    ARK2_EXPLICIT_A,
    ARK2_GAMMA,
    ARK2_IMPLICIT_A,
    Ark2ImexStepper,
)
from drbx.native.fci_model import FciModelState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ScalarState(FciModelState):
    field: jax.Array


def _linear_split_stepper(
    explicit_rate: float,
    implicit_rate: float,
    *,
    stage_records: list[tuple[float, float, float]] | None = None,
) -> Ark2ImexStepper:
    """Return an ARK stepper with the exact scalar DIRK stage solve."""

    def explicit_rhs(state, stage_time, carry):
        return (
            jax.tree_util.tree_map(lambda value: explicit_rate * value, state),
            carry + 1,
            jnp.asarray(stage_time),
        )

    def implicit_rhs(state, stage_time, carry):
        return (
            jax.tree_util.tree_map(lambda value: implicit_rate * value, state),
            carry + 1,
            jnp.asarray(stage_time),
        )

    def solve(predictor, stage_time, diagonal_timestep, carry):
        if stage_records is not None:
            stage_records.append(
                (
                    float(stage_time),
                    float(diagonal_timestep),
                    float(predictor["density"]),
                )
            )
        stage = jax.tree_util.tree_map(
            lambda value: value / (1.0 - diagonal_timestep * implicit_rate),
            predictor,
        )
        implicit = jax.tree_util.tree_map(lambda value: implicit_rate * value, stage)
        return stage, implicit, carry + 1, jnp.asarray(diagonal_timestep)

    return Ark2ImexStepper(explicit_rhs, implicit_rhs, solve)


def test_ark2_tableau_matches_arkode_ark2_312() -> None:
    sqrt2 = math.sqrt(2.0)
    gamma = 1.0 - 1.0 / sqrt2
    assert ARK2_C == (0.0, 2.0 - sqrt2, 1.0)
    assert np.allclose(ARK2_GAMMA, gamma)
    assert np.allclose(ARK2_EXPLICIT_A[1], (2.0 - sqrt2, 0.0, 0.0))
    assert np.allclose(
        ARK2_EXPLICIT_A[2],
        (1.0 - (3.0 + 2.0 * sqrt2) / 6.0, (3.0 + 2.0 * sqrt2) / 6.0, 0.0),
    )
    assert np.allclose(ARK2_IMPLICIT_A[1], (gamma, gamma, 0.0))
    assert np.allclose(
        ARK2_IMPLICIT_A[2],
        (1.0 / (2.0 * sqrt2), 1.0 / (2.0 * sqrt2), gamma),
    )
    assert np.allclose(
        ARK2_B,
        (1.0 / (2.0 * sqrt2), 1.0 / (2.0 * sqrt2), gamma),
    )
    assert np.allclose(
        ARK2_B_EMBEDDED,
        ((4.0 - sqrt2) / 8.0, (4.0 - sqrt2) / 8.0, 1.0 / (2.0 * sqrt2)),
    )


def test_ark2_stage_predictors_follow_additive_tableau() -> None:
    explicit_rate = -0.4
    implicit_rate = -1.3
    timestep = 0.2
    records: list[tuple[float, float, float]] = []
    stepper = _linear_split_stepper(
        explicit_rate,
        implicit_rate,
        stage_records=records,
    )
    initial = {"density": jnp.asarray(1.25), "temperature": jnp.asarray(-0.5)}
    out = stepper(initial, time=0.3, timestep=timestep, carry=jnp.asarray(0))

    y1 = float(initial["density"])
    f_e1 = explicit_rate * y1
    f_i1 = implicit_rate * y1
    expected_predictor_2 = y1 + timestep * (
        ARK2_EXPLICIT_A[1][0] * f_e1 + ARK2_IMPLICIT_A[1][0] * f_i1
    )
    y2 = expected_predictor_2 / (1.0 - timestep * ARK2_GAMMA * implicit_rate)
    expected_predictor_3 = y1 + timestep * (
        ARK2_EXPLICIT_A[2][0] * f_e1
        + ARK2_EXPLICIT_A[2][1] * explicit_rate * y2
        + ARK2_IMPLICIT_A[2][0] * f_i1
        + ARK2_IMPLICIT_A[2][1] * implicit_rate * y2
    )

    assert len(records) == 2
    assert np.allclose(records[0], (0.3 + ARK2_C[1] * timestep, ARK2_GAMMA * timestep, expected_predictor_2))
    assert np.allclose(records[1], (0.3 + timestep, ARK2_GAMMA * timestep, expected_predictor_3))
    assert int(out.carry) == 6
    assert len(out.explicit_stage_aux) == 3
    assert len(out.implicit_stage_aux) == 3
    assert out.solve_stage_aux[0] is None


def test_ark2_is_second_order_for_scalar_additive_split() -> None:
    explicit_rate = -0.35
    implicit_rate = -1.15
    final_time = 0.8
    initial = {"density": jnp.asarray(1.0), "temperature": jnp.asarray(-2.0)}

    def advance(timestep: float) -> float:
        stepper = _linear_split_stepper(explicit_rate, implicit_rate)
        state = initial
        steps = round(final_time / timestep)
        for step in range(steps):
            state = stepper(
                state,
                time=step * timestep,
                timestep=timestep,
                carry=jnp.asarray(0),
            ).state
        return float(state["density"])

    exact = math.exp((explicit_rate + implicit_rate) * final_time)
    coarse_error = abs(advance(0.1) - exact)
    fine_error = abs(advance(0.05) - exact)
    assert fine_error < coarse_error / 3.5


def test_ark2_accepts_and_jits_nested_pytrees() -> None:
    stepper = _linear_split_stepper(-0.25, -0.5)
    initial = {
        "fluid": (jnp.ones((2, 3)), {"electron": 2.0 * jnp.ones((2, 3))}),
    }

    @jax.jit
    def advance(state):
        return stepper(
            state,
            time=jnp.asarray(0.0),
            timestep=jnp.asarray(0.125),
            carry=jnp.asarray(0),
        ).state

    actual = advance(initial)
    expected_factor = math.exp((-0.25 - 0.5) * 0.125)
    # A single second-order step is close to, but intentionally not identical
    # to, the exponential solution.
    assert jnp.allclose(actual["fluid"][0], actual["fluid"][1]["electron"] / 2.0)
    assert jnp.max(jnp.abs(actual["fluid"][0] - expected_factor)) < 2.0e-4


def test_ark2_uses_fci_model_state_axpy_path() -> None:
    explicit_rate = -0.2
    implicit_rate = -0.7

    def explicit_rhs(state, _time, carry):
        return _ScalarState(explicit_rate * state.field), carry, jnp.asarray(1)

    def implicit_rhs(state, _time, carry):
        return _ScalarState(implicit_rate * state.field), carry, jnp.asarray(2)

    def solve(predictor, _time, diagonal_timestep, carry):
        state = _ScalarState(
            predictor.field / (1.0 - diagonal_timestep * implicit_rate)
        )
        return (
            _ScalarState(state.field),
            _ScalarState(implicit_rate * state.field),
            carry,
            jnp.asarray(3),
        )

    stepper = Ark2ImexStepper(explicit_rhs, implicit_rhs, solve)
    initial = _ScalarState(jnp.arange(4.0).reshape(2, 2))
    actual = stepper(initial, time=0.0, timestep=0.1, carry=None).state
    assert isinstance(actual, _ScalarState)
    assert jnp.all(jnp.isfinite(actual.field))
