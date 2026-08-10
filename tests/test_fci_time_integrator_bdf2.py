from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_model import FciModelState
from drbx.native.fci_time_integrator import (
    ImexBdf2StepResult,
    ImexBdf2Stepper,
    imex_bdf2_step,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ScalarState(FciModelState):
    field: jax.Array


def _tree_scale(state, factor):
    return jax.tree_util.tree_map(lambda value: factor * value, state)


def test_bdf2_contract_coefficients_and_single_callback_calls() -> None:
    implicit_calls = []
    explicit_calls = []

    def implicit_solve(predictor, extrapolated, next_time, alpha, carry):
        implicit_calls.append((predictor, extrapolated, next_time, alpha, carry))
        return predictor, carry + 10, {"alpha": alpha}

    def explicit_rhs(state, next_time, carry):
        explicit_calls.append((state, next_time, carry))
        return _tree_scale(state, 3.0), carry + 1, {"time": next_time}

    result = ImexBdf2Stepper(explicit_rhs, implicit_solve)(
        {"u": jnp.asarray(2.0)},
        {"u": jnp.asarray(4.0)},
        {"u": jnp.asarray(0.5)},
        {"u": jnp.asarray(1.5)},
        time=0.3,
        timestep=0.2,
        carry=jnp.asarray(7),
    )

    assert isinstance(result, ImexBdf2StepResult)
    assert len(implicit_calls) == 1
    assert len(explicit_calls) == 1
    predictor, extrapolated, next_time, alpha, solve_carry = implicit_calls[0]
    assert np.allclose(predictor["u"], 4.0 / 3.0 * 4.0 - 2.0 / 3.0 +
                       (2.0 / 3.0) * 0.2 * (2.0 * 1.5 - 0.5))
    assert np.allclose(extrapolated["u"], 2.0 * 4.0 - 2.0)
    assert np.allclose(next_time, 0.5)
    assert np.allclose(alpha, 2.0 * 0.2 / 3.0)
    assert int(solve_carry) == 7
    assert np.allclose(result.state["u"], predictor["u"])
    assert np.allclose(result.explicit_rhs["u"], 3.0 * predictor["u"])
    assert int(result.carry) == 18
    assert np.allclose(result.predictor["u"], predictor["u"])
    assert np.allclose(result.extrapolated_state["u"], extrapolated["u"])


def test_bdf2_is_second_order_for_scalar_additive_split() -> None:
    explicit_rate = -0.35
    implicit_rate = -1.15
    final_time = 0.8

    def explicit_rhs(state, _time, carry):
        return _tree_scale(state, explicit_rate), carry, None

    def implicit_solve(predictor, _extrapolated, _time, alpha, carry):
        state = _tree_scale(
            predictor,
            1.0 / (1.0 - alpha * implicit_rate),
        )
        return state, carry, None

    def advance(timestep: float) -> float:
        stepper = ImexBdf2Stepper(explicit_rhs, implicit_solve)
        steps = round(final_time / timestep)
        # Exact U0/U1 provide the required second-order startup history.
        exact = lambda time: jnp.asarray(
            math.exp((explicit_rate + implicit_rate) * time)
        )
        state_nm1 = {"u": exact(0.0)}
        state_n = {"u": exact(timestep)}
        rhs_nm1 = explicit_rhs(state_nm1, 0.0, None)[0]
        rhs_n = explicit_rhs(state_n, timestep, None)[0]
        for step in range(1, steps):
            result = stepper(
                state_nm1,
                state_n,
                rhs_nm1,
                rhs_n,
                time=step * timestep,
                timestep=timestep,
                carry=None,
            )
            state_nm1, state_n = state_n, result.state
            rhs_nm1, rhs_n = rhs_n, result.explicit_rhs
        return float(state_n["u"])

    exact_final = math.exp((explicit_rate + implicit_rate) * final_time)
    coarse_error = abs(advance(0.1) - exact_final)
    fine_error = abs(advance(0.05) - exact_final)
    assert fine_error < coarse_error / 3.5


def test_bdf2_jits_nested_pytree_and_registered_result() -> None:
    def explicit_rhs(state, _time, carry):
        return _tree_scale(state, -0.25), carry + 1, {"rhs": carry}

    def implicit_solve(predictor, extrapolated, _time, alpha, carry):
        del extrapolated
        state = _tree_scale(predictor, 1.0 / (1.0 + 0.5 * alpha))
        return state, carry + 2, {"alpha": alpha}

    stepper = ImexBdf2Stepper(explicit_rhs, implicit_solve)
    state_nm1 = {"fluid": (jnp.ones((2, 3)), {"electron": 2.0 * jnp.ones((2, 3))})}
    state_n = _tree_scale(state_nm1, 1.1)
    rhs_nm1 = _tree_scale(state_nm1, -0.25)
    rhs_n = _tree_scale(state_n, -0.25)

    @jax.jit
    def advance(state_nm1, state_n, rhs_nm1, rhs_n):
        return stepper(
            state_nm1,
            state_n,
            rhs_nm1,
            rhs_n,
            time=jnp.asarray(0.2),
            timestep=jnp.asarray(0.125),
            carry=jnp.asarray(3),
        )

    result = advance(state_nm1, state_n, rhs_nm1, rhs_n)
    assert isinstance(result, ImexBdf2StepResult)
    assert int(result.carry) == 6
    assert jnp.allclose(result.state["fluid"][1]["electron"],
                        2.0 * result.state["fluid"][0])
    assert jnp.all(jnp.isfinite(result.explicit_rhs["fluid"][0]))


def test_bdf2_uses_fci_model_state_path_and_functional_wrapper() -> None:
    def explicit_rhs(state, _time, carry):
        return _ScalarState(-0.2 * state.field), carry, jnp.asarray(11)

    def implicit_solve(predictor, extrapolated, _time, alpha, carry):
        del extrapolated
        state = _ScalarState(predictor.field / (1.0 + 0.7 * alpha))
        return state, carry, jnp.asarray(22)

    state_nm1 = _ScalarState(jnp.arange(4.0).reshape(2, 2))
    state_n = _ScalarState(state_nm1.field + 0.1)
    rhs_nm1 = _ScalarState(-0.2 * state_nm1.field)
    rhs_n = _ScalarState(-0.2 * state_n.field)
    result = imex_bdf2_step(
        state_nm1,
        state_n,
        rhs_nm1,
        rhs_n,
        time=0.0,
        timestep=0.1,
        explicit_rhs_fn=explicit_rhs,
        implicit_solve_fn=implicit_solve,
        carry=None,
    )

    assert isinstance(result.state, _ScalarState)
    assert isinstance(result.predictor, _ScalarState)
    assert isinstance(result.extrapolated_state, _ScalarState)
    assert jnp.all(jnp.isfinite(result.state.field))
    assert jnp.allclose(result.explicit_rhs.field, -0.2 * result.state.field)


def test_bdf2_explicit_history_uses_the_state_returned_by_implicit_solve() -> None:
    """The explicit-history callback must see the constrained implicit state."""

    explicit_seen = []

    def explicit_rhs(state, _time, carry):
        explicit_seen.append(state.field)
        return state, carry, None

    def implicit_solve(_predictor, _extrapolated, _time, _alpha, carry):
        constrained = _ScalarState(jnp.full((2, 2), 7.0))
        return constrained, carry, None

    state_nm1 = _ScalarState(jnp.zeros((2, 2)))
    state_n = _ScalarState(jnp.ones((2, 2)))
    result = imex_bdf2_step(
        state_nm1,
        state_n,
        state_nm1,
        state_n,
        time=0.0,
        timestep=0.1,
        explicit_rhs_fn=explicit_rhs,
        implicit_solve_fn=implicit_solve,
        carry=None,
    )

    assert len(explicit_seen) == 1
    assert jnp.all(explicit_seen[0] == 7.0)
    assert jnp.all(result.explicit_rhs.field == 7.0)
