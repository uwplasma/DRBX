from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend
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


def pack_fixed_state(state: RecyclingFixedState) -> object:
    """Pack a fixed-layout PyTree state into a single JAX vector."""

    if use_jax_backend(*state.field_values, state.feedback_values):
        field_blocks = tuple(jnp.ravel(jnp.asarray(value, dtype=jnp.float64)) for value in state.field_values)
        feedback_values = jnp.ravel(jnp.asarray(state.feedback_values, dtype=jnp.float64))
        if field_blocks and feedback_values.size:
            return jnp.concatenate((*field_blocks, feedback_values))
        if field_blocks:
            return jnp.concatenate(field_blocks)
        return feedback_values

    field_blocks = tuple(np.ravel(np.asarray(value, dtype=np.float64)) for value in state.field_values)
    feedback_values = np.ravel(np.asarray(state.feedback_values, dtype=np.float64))
    if field_blocks and feedback_values.size:
        return np.concatenate((*field_blocks, feedback_values))
    if field_blocks:
        return np.concatenate(field_blocks)
    return feedback_values


def unpack_fixed_state(packed: object, *, layout: RecyclingPackedStateLayout) -> RecyclingFixedState:
    """Unpack a vector into active arrays using static recycling layout metadata."""

    if use_jax_backend(packed):
        packed_array = jnp.asarray(packed, dtype=jnp.float64)
    else:
        packed_array = np.asarray(packed, dtype=np.float64)
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


def fixed_state_to_full_fields(state: RecyclingFixedState, *, layout: RecyclingPackedStateLayout) -> dict[str, object]:
    """Restore full guard-cell fields from a fixed-layout active state.

    This is the compatibility seam between the fixed PyTree lane and the
    existing Hermès-compatible recycling RHS, which still expects full arrays
    with guard cells. When the active state is traced, the reconstructed arrays
    stay on the JAX backend; host-only callers receive NumPy arrays.
    """

    fields: dict[str, object] = {}
    for name, active_value, template in zip(layout.field_names, state.field_values, layout.field_templates, strict=True):
        if use_jax_backend(active_value):
            active_array = jnp.asarray(active_value, dtype=jnp.float64)
            full_field = jnp.asarray(template, dtype=jnp.float64).at[layout.active_slices].set(active_array)
        else:
            active_array = np.asarray(active_value, dtype=np.float64)
            full_field = np.array(template, dtype=np.float64, copy=True)
            full_field[layout.active_slices] = active_array
        fields[name] = full_field
    return fields


def fixed_state_to_feedback_integrals(
    state: RecyclingFixedState,
    *,
    layout: RecyclingPackedStateLayout,
    base_feedback_integrals: dict[str, object] | None = None,
) -> dict[str, object]:
    """Restore controller integrals keyed by the fixed feedback ordering."""

    integrals = dict(base_feedback_integrals or {})
    for index, name in enumerate(layout.feedback_names):
        value = state.feedback_values[index]
        integrals[name] = value if use_jax_backend(value) else float(value)
    return integrals


def build_fixed_host_rhs_bridge(
    packed_rhs_function: Callable[[dict[str, object], dict[str, object]], object],
    *,
    layout: RecyclingPackedStateLayout,
    base_feedback_integrals: dict[str, object] | None = None,
) -> Callable[[RecyclingFixedState], RecyclingFixedState]:
    """Adapt the current full-field packed RHS into the fixed-state interface.

    The returned function is a parity bridge, not the final production solver
    path: the wrapped RHS may still perform host-side NumPy/SciPy work. Keeping
    this adapter explicit lets tests compare the fixed-layout lane against the
    current validated RHS before each term is ported to pure JAX kernels.
    """

    def rhs(state: RecyclingFixedState) -> RecyclingFixedState:
        packed_rhs = packed_rhs_function(
            fixed_state_to_full_fields(state, layout=layout),
            fixed_state_to_feedback_integrals(
                state,
                layout=layout,
                base_feedback_integrals=base_feedback_integrals,
            ),
        )
        return unpack_fixed_state(packed_rhs, layout=layout)

    return rhs


def build_fixed_array_rhs(
    field_rhs_function: Callable[[dict[str, jax.Array], jax.Array], dict[str, object]],
    *,
    layout: RecyclingPackedStateLayout,
    feedback_rhs_function: Callable[[dict[str, jax.Array], jax.Array], object] | None = None,
) -> Callable[[RecyclingFixedState], RecyclingFixedState]:
    """Build a transformable RHS from active-domain array kernels.

    This is the production-facing counterpart to ``build_fixed_host_rhs_bridge``:
    callers provide RHS arrays already keyed by the fixed field layout, so the
    residual path does not need to reconstruct full guard-cell dictionaries.
    Missing field RHS entries default to zero, which makes staged term-by-term
    ports possible while preserving a fixed output layout.
    """

    def rhs(state: RecyclingFixedState) -> RecyclingFixedState:
        fields = fixed_state_to_field_dict(state, layout=layout)
        field_rhs = field_rhs_function(fields, state.feedback_values)
        rhs_values = tuple(
            jnp.asarray(field_rhs.get(name, jnp.zeros_like(value)), dtype=jnp.float64)
            for name, value in zip(layout.field_names, state.field_values, strict=True)
        )
        if feedback_rhs_function is None:
            feedback_rhs = jnp.zeros_like(state.feedback_values, dtype=jnp.float64)
        else:
            feedback_rhs = jnp.asarray(feedback_rhs_function(fields, state.feedback_values), dtype=jnp.float64)
        return RecyclingFixedState(field_values=rhs_values, feedback_values=feedback_rhs)

    return rhs


def build_fixed_full_field_array_rhs(
    full_field_rhs_function: Callable[[dict[str, object], jax.Array], dict[str, object]],
    *,
    layout: RecyclingPackedStateLayout,
    feedback_rhs_function: Callable[[dict[str, jax.Array], jax.Array], object] | None = None,
) -> Callable[[RecyclingFixedState], RecyclingFixedState]:
    """Stage guard-cell kernels through the fixed-layout active-array RHS.

    This adapter is the migration seam for recycling terms that still need
    target or guard-cell values, such as sheath recycling, neutral diffusion,
    and collision closures. The wrapped kernel receives reconstructed full
    fields, but the public residual interface stays a static PyTree of active
    arrays, so JAX transformations continue to work when the wrapped kernel is
    itself JAX-compatible.
    """

    def field_rhs(active_fields: dict[str, jax.Array], feedback_values: jax.Array) -> dict[str, object]:
        state = RecyclingFixedState(
            field_values=tuple(active_fields[name] for name in layout.field_names),
            feedback_values=feedback_values,
        )
        full_fields = fixed_state_to_full_fields(state, layout=layout)
        full_rhs = full_field_rhs_function(full_fields, feedback_values)
        rhs_values: dict[str, object] = {}
        for name in layout.field_names:
            if name not in full_rhs:
                continue
            value = jnp.asarray(full_rhs[name], dtype=jnp.float64)
            rhs_values[name] = value if value.shape == layout.active_shape else value[layout.active_slices]
        return rhs_values

    return build_fixed_array_rhs(field_rhs, layout=layout, feedback_rhs_function=feedback_rhs_function)


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
