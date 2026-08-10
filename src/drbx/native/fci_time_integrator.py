from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import sqrt
from typing import Any, Callable, Generic, TypeVar

import jax
import jax.numpy as jnp

from .fci_model import FciModelState


StateT = TypeVar("StateT")
CarryT = TypeVar("CarryT")
AuxT = TypeVar("AuxT")
ImplicitAuxT = TypeVar("ImplicitAuxT")


# ARKODE_ARK2_ERK_3_1_2 / ARKODE_ARK2_DIRK_3_1_2.  Keep the
# coefficients here, rather than hidden in the stage code, so an IMEX caller
# can audit exactly which additive Runge--Kutta pair is in use.
_SQRT2 = sqrt(2.0)
ARK2_GAMMA = 1.0 - 1.0 / _SQRT2
ARK2_C = (0.0, 2.0 - _SQRT2, 1.0)
ARK2_EXPLICIT_A = (
    (0.0, 0.0, 0.0),
    (2.0 - _SQRT2, 0.0, 0.0),
    (1.0 - (3.0 + 2.0 * _SQRT2) / 6.0, (3.0 + 2.0 * _SQRT2) / 6.0, 0.0),
)
ARK2_IMPLICIT_A = (
    (0.0, 0.0, 0.0),
    (ARK2_GAMMA, ARK2_GAMMA, 0.0),
    (1.0 / (2.0 * _SQRT2), 1.0 / (2.0 * _SQRT2), ARK2_GAMMA),
)
ARK2_B = (1.0 / (2.0 * _SQRT2), 1.0 / (2.0 * _SQRT2), ARK2_GAMMA)
ARK2_B_EMBEDDED = (
    (4.0 - _SQRT2) / 8.0,
    (4.0 - _SQRT2) / 8.0,
    1.0 / (2.0 * _SQRT2),
)


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


def rk4_step(
    state: StateT,
    *,
    time: float | jax.Array,
    timestep: float | jax.Array,
    rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT],
        tuple[StateT, CarryT, AuxT],
    ],
    carry: CarryT,
) -> Rk4StepResult[StateT, CarryT, AuxT]:
    """Compatibility function for callers not retaining an ``Rk4Stepper``."""

    return Rk4Stepper(rhs_fn)(
        state,
        time=time,
        timestep=timestep,
        carry=carry,
    )


def _assert_tree_compatible(reference: Any, value: Any, *, name: str) -> None:
    """Validate an additive-RK operand without requiring an FCI state type.

    The IMEX core is deliberately usable by small scalar/PyTree problems as
    well as ``FciModelState`` subclasses.  FCI states retain their stricter
    named-field checks; generic PyTrees must at least preserve tree structure
    and leaf shapes.
    """

    if isinstance(reference, FciModelState):
        _assert_rhs_compatible(reference, value)
        return
    reference_def = jax.tree_util.tree_structure(reference)
    value_def = jax.tree_util.tree_structure(value)
    if reference_def != value_def:
        raise TypeError(f"{name} must have the same PyTree structure as state")
    for reference_leaf, value_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(value),
    ):
        if jnp.asarray(reference_leaf).shape != jnp.asarray(value_leaf).shape:
            raise ValueError(
                f"{name} leaf shape must match state; got "
                f"{jnp.asarray(value_leaf).shape}, expected "
                f"{jnp.asarray(reference_leaf).shape}"
            )


def _tree_axpy(state: StateT, rhs: StateT, *, scale: float | jax.Array) -> StateT:
    """Return ``state + scale * rhs`` for FCI states or generic PyTrees."""

    if isinstance(state, FciModelState):
        return state.axpy(rhs, scale=scale)
    return jax.tree_util.tree_map(lambda lhs, value: lhs + scale * value, state, rhs)


def _tree_linear_combination(
    reference: StateT,
    terms: tuple[tuple[float | jax.Array, StateT], ...],
    *,
    scale: float | jax.Array,
) -> StateT:
    """Return ``reference + scale * sum(coeff * term)`` for a shared PyTree."""

    result = reference
    for coefficient, term in terms:
        if coefficient != 0.0:
            result = _tree_axpy(result, term, scale=scale * coefficient)
    return result


def _tree_bdf2_predictor(
    state_nm1: StateT,
    state_n: StateT,
    explicit_rhs_nm1: StateT,
    explicit_rhs_n: StateT,
    *,
    timestep: float | jax.Array,
) -> StateT:
    """Assemble the constant-step IMEX-BDF2 explicit predictor."""

    alpha = (2.0 / 3.0) * timestep
    predictor = _tree_axpy(state_n, state_nm1, scale=-1.0 / 3.0)
    predictor = _tree_axpy(predictor, state_n, scale=1.0 / 3.0)
    predictor = _tree_axpy(predictor, explicit_rhs_n, scale=2.0 * alpha)
    return _tree_axpy(predictor, explicit_rhs_nm1, scale=-alpha)


def _tree_extrapolate(state_nm1: StateT, state_n: StateT) -> StateT:
    """Return the second-order state extrapolation ``2 U_n - U_nm1``."""

    return _tree_axpy(_tree_axpy(state_n, state_nm1, scale=-1.0), state_n, scale=1.0)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImexBdf2StepResult(Generic[StateT, CarryT, AuxT, ImplicitAuxT]):
    """Result of one fixed-timestep IMEX-BDF2/SBDF2 advance.

    ``predictor`` is the BDF2/explicit value on the right-hand side of the
    implicit solve.  ``extrapolated_state`` is ``2 U_n - U_{n-1}``, retained
    for initial guesses and algebraic fields.  ``explicit_rhs`` is evaluated
    at the accepted new state and is therefore ready to become the newest
    explicit-RHS history value.
    """

    state: StateT
    explicit_rhs: StateT
    predictor: StateT
    extrapolated_state: StateT
    carry: CarryT
    implicit_solve_aux: ImplicitAuxT
    explicit_rhs_aux: AuxT

    def tree_flatten(self):
        return (
            self.state,
            self.explicit_rhs,
            self.predictor,
            self.extrapolated_state,
            self.carry,
            self.implicit_solve_aux,
            self.explicit_rhs_aux,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            state,
            explicit_rhs,
            predictor,
            extrapolated_state,
            carry,
            implicit_solve_aux,
            explicit_rhs_aux,
        ) = children
        return cls(
            state=state,
            explicit_rhs=explicit_rhs,
            predictor=predictor,
            extrapolated_state=extrapolated_state,
            carry=carry,
            implicit_solve_aux=implicit_solve_aux,
            explicit_rhs_aux=explicit_rhs_aux,
        )


@dataclass(frozen=True)
class ImexBdf2Stepper(Generic[StateT, CarryT, AuxT, ImplicitAuxT]):
    """Model-agnostic fixed-step IMEX-BDF2/SBDF2 core.

    The implicit callback receives ``(predictor, extrapolated_state,
    next_time, alpha, carry)`` and returns ``(accepted_state, carry, aux)``.
    Here ``alpha = 2 * timestep / 3``.  The explicit callback is evaluated
    exactly once, at the accepted state, and returns the explicit RHS history
    value for the next BDF2 step.

    This core does not perform startup, adaptive timestepping, rejection, or
    model-specific algebraic reconstruction.  Those policies belong to the
    caller or a higher-level driver.
    """

    explicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, AuxT]
    ]
    implicit_solve_fn: Callable[
        [StateT, StateT, float | jax.Array, float | jax.Array, CarryT],
        tuple[StateT, CarryT, ImplicitAuxT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.explicit_rhs_fn):
            raise TypeError("explicit_rhs_fn must be callable")
        if not callable(self.implicit_solve_fn):
            raise TypeError("implicit_solve_fn must be callable")

    def __call__(
        self,
        state_nm1: StateT,
        state_n: StateT,
        explicit_rhs_nm1: StateT,
        explicit_rhs_n: StateT,
        *,
        time: float | jax.Array,
        timestep: float | jax.Array,
        carry: CarryT,
    ) -> ImexBdf2StepResult[StateT, CarryT, AuxT, ImplicitAuxT]:
        _assert_tree_compatible(state_n, state_nm1, name="state_nm1")
        _assert_tree_compatible(
            state_n, explicit_rhs_nm1, name="explicit_rhs_nm1"
        )
        _assert_tree_compatible(state_n, explicit_rhs_n, name="explicit_rhs_n")

        predictor = _tree_bdf2_predictor(
            state_nm1,
            state_n,
            explicit_rhs_nm1,
            explicit_rhs_n,
            timestep=timestep,
        )
        extrapolated_state = _tree_extrapolate(state_nm1, state_n)
        next_time = time + timestep
        alpha = (2.0 / 3.0) * timestep

        accepted_state, solve_carry, implicit_solve_aux = self.implicit_solve_fn(
            predictor,
            extrapolated_state,
            next_time,
            alpha,
            carry,
        )
        _assert_tree_compatible(state_n, accepted_state, name="implicit solve state")

        explicit_rhs, next_carry, explicit_rhs_aux = self.explicit_rhs_fn(
            accepted_state,
            next_time,
            solve_carry,
        )
        _assert_tree_compatible(
            state_n, explicit_rhs, name="explicit_rhs_fn result"
        )
        return ImexBdf2StepResult(
            state=accepted_state,
            explicit_rhs=explicit_rhs,
            predictor=predictor,
            extrapolated_state=extrapolated_state,
            carry=next_carry,
            implicit_solve_aux=implicit_solve_aux,
            explicit_rhs_aux=explicit_rhs_aux,
        )


def imex_bdf2_step(
    state_nm1: StateT,
    state_n: StateT,
    explicit_rhs_nm1: StateT,
    explicit_rhs_n: StateT,
    *,
    time: float | jax.Array,
    timestep: float | jax.Array,
    explicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, AuxT]
    ],
    implicit_solve_fn: Callable[
        [StateT, StateT, float | jax.Array, float | jax.Array, CarryT],
        tuple[StateT, CarryT, ImplicitAuxT],
    ],
    carry: CarryT,
) -> ImexBdf2StepResult[StateT, CarryT, AuxT, ImplicitAuxT]:
    """Functional wrapper for :class:`ImexBdf2Stepper`."""

    return ImexBdf2Stepper(
        explicit_rhs_fn=explicit_rhs_fn,
        implicit_solve_fn=implicit_solve_fn,
    )(
        state_nm1,
        state_n,
        explicit_rhs_nm1,
        explicit_rhs_n,
        time=time,
        timestep=timestep,
        carry=carry,
    )


# Naming aliases keep the method discoverable beside ``Ark2ImexStepper`` and
# also expose the natural ``imex_bdf2_*`` spelling used in documentation.
Bdf2ImexStepResult = ImexBdf2StepResult
Bdf2ImexStepper = ImexBdf2Stepper
bdf2_imex_step = imex_bdf2_step


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Ark2ImexStepResult(Generic[StateT, CarryT, AuxT, ImplicitAuxT]):
    """Result of one ARK2(3,1,2) additive IMEX advance.

    ``state`` is the second-order solution and ``embedded_state`` is the
    first-order embedded estimate.  The difference is available to a driver
    that later adds adaptive timestep control.  This stepper itself has no
    acceptance/rejection policy and is intentionally fixed-step.
    """

    state: StateT
    embedded_state: StateT
    carry: CarryT
    stage_states: tuple[StateT, StateT, StateT]
    explicit_stage_aux: tuple[AuxT, AuxT, AuxT]
    implicit_stage_aux: tuple[ImplicitAuxT, ImplicitAuxT, ImplicitAuxT]
    solve_stage_aux: tuple[ImplicitAuxT | None, ImplicitAuxT, ImplicitAuxT]

    def tree_flatten(self):
        stage_1, stage_2, stage_3 = self.stage_states
        explicit_1, explicit_2, explicit_3 = self.explicit_stage_aux
        implicit_1, implicit_2, implicit_3 = self.implicit_stage_aux
        solve_1, solve_2, solve_3 = self.solve_stage_aux
        return (
            self.state,
            self.embedded_state,
            self.carry,
            stage_1,
            stage_2,
            stage_3,
            explicit_1,
            explicit_2,
            explicit_3,
            implicit_1,
            implicit_2,
            implicit_3,
            solve_1,
            solve_2,
            solve_3,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            state,
            embedded_state,
            carry,
            stage_1,
            stage_2,
            stage_3,
            explicit_1,
            explicit_2,
            explicit_3,
            implicit_1,
            implicit_2,
            implicit_3,
            solve_1,
            solve_2,
            solve_3,
        ) = children
        return cls(
            state=state,
            embedded_state=embedded_state,
            carry=carry,
            stage_states=(stage_1, stage_2, stage_3),
            explicit_stage_aux=(explicit_1, explicit_2, explicit_3),
            implicit_stage_aux=(implicit_1, implicit_2, implicit_3),
            solve_stage_aux=(solve_1, solve_2, solve_3),
        )


@dataclass(frozen=True)
class Ark2ImexStepper(Generic[StateT, CarryT, AuxT, ImplicitAuxT]):
    """Fixed-step ARKODE ARK2(3,1,2) additive IMEX stepper.

    The caller supplies additive explicit/implicit RHS functions and an
    implicit stage solve.  At implicit stages the solve callback receives the
    already assembled predictor and ``gamma * timestep`` and must return the
    converged stage state together with its implicit RHS.  It can therefore
    use a matrix-free Newton--Krylov solve, an analytic solve, or a model
    specific algebraic constraint without coupling those choices to this
    tableau implementation.

    ``implicit_rhs_fn`` is evaluated only at stage one, whose DIRK diagonal is
    zero.  Stages two and three obtain their implicit RHS from
    ``implicit_stage_solve_fn`` so the residual is evaluated only by the
    nonlinear solver.
    """

    explicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, AuxT]
    ]
    implicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, ImplicitAuxT]
    ]
    implicit_stage_solve_fn: Callable[
        [StateT, float | jax.Array, float | jax.Array, CarryT],
        tuple[StateT, StateT, CarryT, ImplicitAuxT],
    ]

    def __post_init__(self) -> None:
        if not callable(self.explicit_rhs_fn):
            raise TypeError("explicit_rhs_fn must be callable")
        if not callable(self.implicit_rhs_fn):
            raise TypeError("implicit_rhs_fn must be callable")
        if not callable(self.implicit_stage_solve_fn):
            raise TypeError("implicit_stage_solve_fn must be callable")

    def __call__(
        self,
        state: StateT,
        *,
        time: float | jax.Array,
        timestep: float | jax.Array,
        carry: CarryT,
    ) -> Ark2ImexStepResult[StateT, CarryT, AuxT, ImplicitAuxT]:
        # Stage 1: both ARK tables have a zero first row.
        stage_1 = state
        explicit_1, carry_1, explicit_aux_1 = self.explicit_rhs_fn(
            stage_1, time + ARK2_C[0] * timestep, carry
        )
        _assert_tree_compatible(state, explicit_1, name="explicit_rhs_fn result")
        implicit_1, carry_2, implicit_aux_1 = self.implicit_rhs_fn(
            stage_1, time + ARK2_C[0] * timestep, carry_1
        )
        _assert_tree_compatible(state, implicit_1, name="implicit_rhs_fn result")

        predictor_2 = _tree_linear_combination(
            state,
            (
                (ARK2_EXPLICIT_A[1][0], explicit_1),
                (ARK2_IMPLICIT_A[1][0], implicit_1),
            ),
            scale=timestep,
        )
        stage_2, implicit_2, carry_3, solve_aux_2 = self.implicit_stage_solve_fn(
            predictor_2,
            time + ARK2_C[1] * timestep,
            ARK2_IMPLICIT_A[1][1] * timestep,
            carry_2,
        )
        _assert_tree_compatible(state, stage_2, name="implicit stage state")
        _assert_tree_compatible(state, implicit_2, name="implicit stage RHS")
        explicit_2, carry_4, explicit_aux_2 = self.explicit_rhs_fn(
            stage_2, time + ARK2_C[1] * timestep, carry_3
        )
        _assert_tree_compatible(state, explicit_2, name="explicit_rhs_fn result")

        predictor_3 = _tree_linear_combination(
            state,
            (
                (ARK2_EXPLICIT_A[2][0], explicit_1),
                (ARK2_EXPLICIT_A[2][1], explicit_2),
                (ARK2_IMPLICIT_A[2][0], implicit_1),
                (ARK2_IMPLICIT_A[2][1], implicit_2),
            ),
            scale=timestep,
        )
        stage_3, implicit_3, carry_5, solve_aux_3 = self.implicit_stage_solve_fn(
            predictor_3,
            time + ARK2_C[2] * timestep,
            ARK2_IMPLICIT_A[2][2] * timestep,
            carry_4,
        )
        _assert_tree_compatible(state, stage_3, name="implicit stage state")
        _assert_tree_compatible(state, implicit_3, name="implicit stage RHS")
        explicit_3, carry_6, explicit_aux_3 = self.explicit_rhs_fn(
            stage_3, time + ARK2_C[2] * timestep, carry_5
        )
        _assert_tree_compatible(state, explicit_3, name="explicit_rhs_fn result")

        stages_explicit = (explicit_1, explicit_2, explicit_3)
        stages_implicit = (implicit_1, implicit_2, implicit_3)
        weighted_terms = tuple(
            (ARK2_B[index], stages_explicit[index])
            for index in range(3)
        ) + tuple(
            (ARK2_B[index], stages_implicit[index])
            for index in range(3)
        )
        next_state = _tree_linear_combination(state, weighted_terms, scale=timestep)
        embedded_terms = tuple(
            (ARK2_B_EMBEDDED[index], stages_explicit[index])
            for index in range(3)
        ) + tuple(
            (ARK2_B_EMBEDDED[index], stages_implicit[index])
            for index in range(3)
        )
        embedded_state = _tree_linear_combination(
            state,
            embedded_terms,
            scale=timestep,
        )
        return Ark2ImexStepResult(
            state=next_state,
            embedded_state=embedded_state,
            carry=carry_6,
            stage_states=(stage_1, stage_2, stage_3),
            explicit_stage_aux=(explicit_aux_1, explicit_aux_2, explicit_aux_3),
            implicit_stage_aux=(implicit_aux_1, solve_aux_2, solve_aux_3),
            solve_stage_aux=(None, solve_aux_2, solve_aux_3),
        )


def ark2_imex_step(
    state: StateT,
    *,
    time: float | jax.Array,
    timestep: float | jax.Array,
    explicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, AuxT]
    ],
    implicit_rhs_fn: Callable[
        [StateT, float | jax.Array, CarryT], tuple[StateT, CarryT, ImplicitAuxT]
    ],
    implicit_stage_solve_fn: Callable[
        [StateT, float | jax.Array, float | jax.Array, CarryT],
        tuple[StateT, StateT, CarryT, ImplicitAuxT],
    ],
    carry: CarryT,
) -> Ark2ImexStepResult[StateT, CarryT, AuxT, ImplicitAuxT]:
    """Functional compatibility wrapper for :class:`Ark2ImexStepper`."""

    return Ark2ImexStepper(
        explicit_rhs_fn=explicit_rhs_fn,
        implicit_rhs_fn=implicit_rhs_fn,
        implicit_stage_solve_fn=implicit_stage_solve_fn,
    )(state, time=time, timestep=timestep, carry=carry)


def sum_stage_outputs(stage_outputs: tuple[AuxT, AuxT, AuxT, AuxT]) -> AuxT:
    """Reduce four stage payloads by addition.

    This is handy when each stage returns a timing vector or another additive
    PyTree. Models with non-additive diagnostics can ignore this helper and
    reduce their stage payloads manually.
    """

    def _add(left: AuxT, right: AuxT) -> AuxT:
        return jax.tree_util.tree_map(lambda lhs, rhs: lhs + rhs, left, right)

    return reduce(_add, stage_outputs[1:], stage_outputs[0])


__all__ = [
    "ARK2_B",
    "ARK2_B_EMBEDDED",
    "ARK2_C",
    "ARK2_EXPLICIT_A",
    "ARK2_GAMMA",
    "ARK2_IMPLICIT_A",
    "Ark2ImexStepResult",
    "Ark2ImexStepper",
    "Bdf2ImexStepResult",
    "Bdf2ImexStepper",
    "ImexBdf2StepResult",
    "ImexBdf2Stepper",
    "Rk4StepResult",
    "Rk4Stepper",
    "ark2_imex_step",
    "bdf2_imex_step",
    "imex_bdf2_step",
    "rk4_step",
    "sum_stage_outputs",
]
