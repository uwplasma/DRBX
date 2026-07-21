from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Callable, Generic, TypeVar

import jax
import jax.numpy as jnp

from .fci_model import FciModelState


StateT = TypeVar("StateT", bound=FciModelState)
CarryT = TypeVar("CarryT")
AuxT = TypeVar("AuxT")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Rk4StepResult(Generic[StateT, CarryT, AuxT]):
    """Container for a single RK4 advance.

    ``stage_aux`` stores the four auxiliary payloads returned by the stage RHS
    calls. The auxiliary payload is intentionally opaque so that callers can
    thread timings, solver diagnostics, warm-start carries, or any other
    model-specific stage information without the RK4 core knowing about it.
    """

    state: StateT
    carry: CarryT
    stage_aux: tuple[AuxT, AuxT, AuxT, AuxT]

    def tree_flatten(self):
        aux_1, aux_2, aux_3, aux_4 = self.stage_aux
        return (self.state, self.carry, aux_1, aux_2, aux_3, aux_4), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        state, carry, aux_1, aux_2, aux_3, aux_4 = children
        return cls(state=state, carry=carry, stage_aux=(aux_1, aux_2, aux_3, aux_4))


def _rk4_weighted_rhs(k1: StateT, k2: StateT, k3: StateT, k4: StateT) -> StateT:
    """Return ``k1 + 2*k2 + 2*k3 + k4`` using the shared state algebra."""

    return k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0)


def _assert_rhs_compatible(reference: StateT, rhs: StateT) -> None:
    if not isinstance(rhs, FciModelState):
        raise TypeError("rhs_fn must return an FciModelState RHS")
    if type(reference) is not type(rhs):
        raise TypeError(
            "rhs_fn must return the same state type it receives; "
            f"got state={type(reference).__name__}, rhs={type(rhs).__name__}"
        )
    field_items = reference.field_items()
    if not field_items:
        return
    expected_shape = tuple(jnp.asarray(field_items[0][1]).shape)
    rhs.assert_field_shape(expected_shape)


@dataclass(frozen=True)
class Rk4Stepper(Generic[StateT, CarryT, AuxT]):
    """Model-agnostic classical RK4 stepper.

    The RK4 algebra is model-agnostic:

    - ``rhs_fn`` computes the stage RHS for the current stage state and returns
      the RHS, the next carry value, and an arbitrary auxiliary payload.
    - The carry is threaded from stage to stage so models can warm-start local
      solves, keep stage caches, or propagate other stage-local context.
    - The final carry returned by the step is the carry produced by the fourth
      stage evaluation.

    Domain-decomposed models should put their stage preparation, communication,
    boundary construction, and operator calls inside ``rhs_fn``.
    """

    rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT],
        tuple[StateT, CarryT, AuxT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.rhs_fn):
            raise TypeError("rhs_fn must be callable")

    def __call__(
        self,
        state: StateT,
        *,
        time: float | jax.Array,
        timestep: float | jax.Array,
        carry: CarryT,
    ) -> Rk4StepResult[StateT, CarryT, AuxT]:
        if not isinstance(state, FciModelState):
            raise TypeError("state must be an FciModelState instance")

        k1, carry_1, aux_1 = self.rhs_fn(state, time, carry)
        _assert_rhs_compatible(state, k1)
        stage_1 = state.axpy(k1, scale=0.5 * timestep)

        k2, carry_2, aux_2 = self.rhs_fn(
            stage_1,
            time + 0.5 * timestep,
            carry_1,
        )
        _assert_rhs_compatible(state, k2)
        stage_2 = state.axpy(k2, scale=0.5 * timestep)

        k3, carry_3, aux_3 = self.rhs_fn(
            stage_2,
            time + 0.5 * timestep,
            carry_2,
        )
        _assert_rhs_compatible(state, k3)
        stage_3 = state.axpy(k3, scale=timestep)

        k4, carry_4, aux_4 = self.rhs_fn(
            stage_3,
            time + timestep,
            carry_3,
        )
        _assert_rhs_compatible(state, k4)
        next_state = state.axpy(
            _rk4_weighted_rhs(k1, k2, k3, k4),
            scale=timestep / 6.0,
        )
        return Rk4StepResult(
            state=next_state,
            carry=carry_4,
            stage_aux=(aux_1, aux_2, aux_3, aux_4),
        )


def sum_stage_outputs(stage_outputs: tuple[AuxT, AuxT, AuxT, AuxT]) -> AuxT:
    """Reduce four stage payloads by addition.

    This is handy when each stage returns a timing vector or another additive
    PyTree. Models with non-additive diagnostics can ignore this helper and
    reduce their stage payloads manually.
    """

    def _add(left: AuxT, right: AuxT) -> AuxT:
        return jax.tree_util.tree_map(lambda lhs, rhs: lhs + rhs, left, right)

    return reduce(_add, stage_outputs[1:], stage_outputs[0])


__all__ = ["Rk4StepResult", "Rk4Stepper", "sum_stage_outputs"]
