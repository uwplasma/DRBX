from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .recycling_layout import RecyclingPackedStateLayout


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RecyclingFixedState:
    """JAX PyTree representation of a fixed-layout recycling active state.

    The heavy Hermès-compatible recycling path still carries dictionaries and
    full guard-cell arrays. This compact representation is the promoted
    transformable lane: all dynamic values are JAX arrays with static field
    ordering from ``RecyclingPackedStateLayout``.
    """

    field_values: tuple[jax.Array, ...]
    feedback_values: jax.Array

    def tree_flatten(self):
        return (self.field_values, self.feedback_values), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        field_values, feedback_values = children
        return cls(field_values=tuple(field_values), feedback_values=feedback_values)


def fixed_state_from_fields(
    fields: dict[str, object],
    *,
    feedback_integrals: dict[str, object],
    layout: RecyclingPackedStateLayout,
) -> RecyclingFixedState:
    """Extract active recycling fields into a transformable PyTree state."""

    return RecyclingFixedState(
        field_values=tuple(jnp.asarray(fields[name][layout.active_slices], dtype=jnp.float64) for name in layout.field_names),
        feedback_values=jnp.asarray(
            [feedback_integrals.get(name, 0.0) for name in layout.feedback_names],
            dtype=jnp.float64,
        ),
    )


def pack_fixed_state(state: RecyclingFixedState) -> jax.Array:
    """Pack a fixed-layout PyTree state into a single JAX vector."""

    field_blocks = tuple(jnp.ravel(jnp.asarray(value, dtype=jnp.float64)) for value in state.field_values)
    if field_blocks and state.feedback_values.size:
        return jnp.concatenate((*field_blocks, jnp.ravel(jnp.asarray(state.feedback_values, dtype=jnp.float64))))
    if field_blocks:
        return jnp.concatenate(field_blocks)
    return jnp.ravel(jnp.asarray(state.feedback_values, dtype=jnp.float64))


def unpack_fixed_state(packed: object, *, layout: RecyclingPackedStateLayout) -> RecyclingFixedState:
    """Unpack a vector into active arrays using static recycling layout metadata."""

    packed_array = jnp.asarray(packed, dtype=jnp.float64)
    active_cell_count = int(layout.field_size // max(len(layout.field_names), 1)) if layout.field_names else 0
    field_values = []
    offset = 0
    for _name in layout.field_names:
        field_values.append(packed_array[offset : offset + active_cell_count].reshape(layout.active_shape))
        offset += active_cell_count
    feedback_values = packed_array[offset : offset + len(layout.feedback_names)]
    return RecyclingFixedState(field_values=tuple(field_values), feedback_values=feedback_values)


def fixed_state_to_field_dict(state: RecyclingFixedState, *, layout: RecyclingPackedStateLayout) -> dict[str, jax.Array]:
    """Return active-domain field arrays keyed by the static field names."""

    return {name: value for name, value in zip(layout.field_names, state.field_values, strict=True)}


def build_fixed_backward_euler_residual(
    rhs_function: Callable[[RecyclingFixedState], RecyclingFixedState],
    *,
    layout: RecyclingPackedStateLayout,
    previous_packed_state: object,
    timestep: float,
) -> Callable[[object], jax.Array]:
    """Build a JAX-transformable backward-Euler residual on the fixed layout."""

    previous = jnp.asarray(previous_packed_state, dtype=jnp.float64)

    def residual(packed_state: object) -> jax.Array:
        state = unpack_fixed_state(packed_state, layout=layout)
        rhs_state = rhs_function(state)
        return pack_fixed_state(state) - previous - float(timestep) * pack_fixed_state(rhs_state)

    return residual


def build_fixed_bdf2_residual(
    rhs_function: Callable[[RecyclingFixedState], RecyclingFixedState],
    *,
    layout: RecyclingPackedStateLayout,
    previous_packed_state: object,
    previous_previous_packed_state: object,
    timestep: float,
) -> Callable[[object], jax.Array]:
    """Build a JAX-transformable BDF2 residual on the fixed layout."""

    previous = jnp.asarray(previous_packed_state, dtype=jnp.float64)
    previous_previous = jnp.asarray(previous_previous_packed_state, dtype=jnp.float64)

    def residual(packed_state: object) -> jax.Array:
        state = unpack_fixed_state(packed_state, layout=layout)
        rhs_state = rhs_function(state)
        return (
            pack_fixed_state(state)
            - (4.0 / 3.0) * previous
            + (1.0 / 3.0) * previous_previous
            - (2.0 / 3.0) * float(timestep) * pack_fixed_state(rhs_state)
        )

    return residual
