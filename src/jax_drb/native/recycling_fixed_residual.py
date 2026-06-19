from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend
from .recycling_layout import RecyclingPackedStateLayout


def _active_cell_count(layout: RecyclingPackedStateLayout) -> int:
    return int(np.prod(layout.active_shape, dtype=np.int64))


def _expected_field_size(layout: RecyclingPackedStateLayout) -> int:
    return _active_cell_count(layout) * len(layout.field_names)


def _expected_packed_size(layout: RecyclingPackedStateLayout) -> int:
    return _expected_field_size(layout) + len(layout.feedback_names)


def _static_shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    return tuple(int(axis) for axis in shape)


def _validate_packed_vector(
    packed_array: object,
    *,
    layout: RecyclingPackedStateLayout,
) -> None:
    if packed_array.ndim != 1:
        raise ValueError(
            f"Packed fixed state must be one-dimensional, got shape {tuple(packed_array.shape)}."
        )
    expected_field_size = _expected_field_size(layout)
    if int(layout.field_size) != expected_field_size:
        raise ValueError(
            "Layout field_size does not match active_shape and field_names: "
            f"got {int(layout.field_size)}, expected {expected_field_size}."
        )
    expected_size = _expected_packed_size(layout)
    if int(packed_array.size) != expected_size:
        raise ValueError(
            f"Packed fixed state has size {int(packed_array.size)}, expected {expected_size}."
        )


def _validate_fixed_state_shapes(
    state: "RecyclingFixedState",
    *,
    layout: RecyclingPackedStateLayout,
) -> None:
    if len(state.field_values) != len(layout.field_names):
        raise ValueError(
            "Fixed state field count does not match layout: "
            f"got {len(state.field_values)}, expected {len(layout.field_names)}."
        )
    expected_field_shape = tuple(layout.active_shape)
    for name, value in zip(layout.field_names, state.field_values, strict=True):
        shape = _static_shape(value)
        if shape != expected_field_shape:
            raise ValueError(
                f"Fixed state field {name!r} has shape {shape}, expected {expected_field_shape}."
            )
    expected_feedback_shape = (len(layout.feedback_names),)
    feedback_shape = _static_shape(state.feedback_values)
    if feedback_shape != expected_feedback_shape:
        raise ValueError(
            "Fixed state feedback_values has shape "
            f"{feedback_shape}, expected {expected_feedback_shape}."
        )


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


def unpack_fixed_state(
    packed: object,
    *,
    layout: RecyclingPackedStateLayout,
    validate: bool = True,
) -> RecyclingFixedState:
    """Unpack a vector into active arrays using static recycling layout metadata."""

    if use_jax_backend(packed):
        packed_array = jnp.asarray(packed, dtype=jnp.float64)
    else:
        packed_array = np.asarray(packed, dtype=np.float64)
    if validate:
        _validate_packed_vector(packed_array, layout=layout)
    active_cell_count = _active_cell_count(layout)
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
        _validate_fixed_state_shapes(state, layout=layout)
        fields = fixed_state_to_field_dict(state, layout=layout)
        field_rhs = field_rhs_function(fields, state.feedback_values)
        unknown_fields = set(field_rhs) - set(layout.field_names)
        if unknown_fields:
            unknown = ", ".join(repr(name) for name in sorted(unknown_fields))
            raise ValueError(f"Field RHS returned unknown layout entries: {unknown}.")
        rhs_values = []
        for name, value in zip(layout.field_names, state.field_values, strict=True):
            if name in field_rhs:
                rhs_array = jnp.asarray(field_rhs[name], dtype=jnp.float64)
                if tuple(rhs_array.shape) != tuple(layout.active_shape):
                    raise ValueError(
                        f"Field RHS for {name!r} has shape {tuple(rhs_array.shape)}, "
                        f"expected {tuple(layout.active_shape)}."
                    )
            else:
                rhs_array = jnp.zeros_like(value, dtype=jnp.float64)
            rhs_values.append(rhs_array)
        if feedback_rhs_function is None:
            feedback_rhs = jnp.zeros_like(state.feedback_values, dtype=jnp.float64)
        else:
            feedback_rhs = jnp.asarray(feedback_rhs_function(fields, state.feedback_values), dtype=jnp.float64)
            expected_feedback_shape = (len(layout.feedback_names),)
            if tuple(feedback_rhs.shape) != expected_feedback_shape:
                raise ValueError(
                    "Feedback RHS has shape "
                    f"{tuple(feedback_rhs.shape)}, expected {expected_feedback_shape}."
                )
        return RecyclingFixedState(field_values=tuple(rhs_values), feedback_values=feedback_rhs)

    return rhs


def build_fixed_array_state_rhs(
    state_rhs_function: Callable[[dict[str, jax.Array], jax.Array], RecyclingFixedState],
    *,
    layout: RecyclingPackedStateLayout,
    validate_shapes: bool = True,
) -> Callable[[RecyclingFixedState], RecyclingFixedState]:
    """Build a transformable RHS whose field and feedback terms share one kernel.

    ``build_fixed_array_rhs`` is convenient for term-by-term ports where field
    and feedback derivatives are independent.  Some recycling terms, however,
    compute controller feedback from the same expensive collision, neutral and
    sheath source evaluation used by the field RHS.  This adapter keeps the same
    fixed-layout PyTree contract while letting those shared kernels run once per
    residual evaluation.
    """

    def rhs(state: RecyclingFixedState) -> RecyclingFixedState:
        if validate_shapes:
            _validate_fixed_state_shapes(state, layout=layout)
        fields = fixed_state_to_field_dict(state, layout=layout)
        rhs_state = state_rhs_function(fields, state.feedback_values)
        if not isinstance(rhs_state, RecyclingFixedState):
            raise TypeError(
                "State RHS must return a RecyclingFixedState, got "
                f"{type(rhs_state).__name__}."
            )
        if validate_shapes:
            _validate_fixed_state_shapes(rhs_state, layout=layout)
        return RecyclingFixedState(
            field_values=tuple(
                jnp.asarray(value, dtype=jnp.float64)
                for value in rhs_state.field_values
            ),
            feedback_values=jnp.asarray(rhs_state.feedback_values, dtype=jnp.float64),
        )

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
        packed = jnp.asarray(packed_state, dtype=jnp.float64)
        state = unpack_fixed_state(packed, layout=layout, validate=False)
        rhs_state = rhs_function(state)
        return packed - previous - float(timestep) * pack_fixed_state(rhs_state)

    return residual


def build_fixed_bdf2_residual(
    rhs_function: Callable[[RecyclingFixedState], RecyclingFixedState],
    *,
    layout: RecyclingPackedStateLayout,
    previous_packed_state: object,
    previous_previous_packed_state: object,
    timestep: float,
    previous_timestep: float | None = None,
) -> Callable[[object], jax.Array]:
    """Build a JAX-transformable BDF2 residual on the fixed layout."""

    previous = jnp.asarray(previous_packed_state, dtype=jnp.float64)
    previous_previous = jnp.asarray(previous_previous_packed_state, dtype=jnp.float64)
    previous_dt = float(timestep) if previous_timestep is None else float(previous_timestep)
    if previous_dt <= 0.0:
        raise ValueError("previous_timestep must be positive for BDF2 residuals.")
    step_ratio = float(timestep) / previous_dt
    previous_coefficient = ((step_ratio + 1.0) ** 2) / (2.0 * step_ratio + 1.0)
    previous_previous_coefficient = (step_ratio**2) / (2.0 * step_ratio + 1.0)
    rhs_coefficient = float(timestep) * (step_ratio + 1.0) / (2.0 * step_ratio + 1.0)
    history_state = (
        previous_coefficient * previous
        - previous_previous_coefficient * previous_previous
    )

    def residual(packed_state: object) -> jax.Array:
        packed = jnp.asarray(packed_state, dtype=jnp.float64)
        state = unpack_fixed_state(packed, layout=layout, validate=False)
        rhs_state = rhs_function(state)
        return packed - history_state - rhs_coefficient * pack_fixed_state(rhs_state)

    return residual


@dataclass
class FixedResidualLinearizedAction:
    """Instrumented reusable Jacobian action for a fixed-layout residual.

    This object is the narrow matrix-free seam used by recycling solver probes:
    one call to :func:`jax.linearize` gives the residual and a reusable
    Jacobian-vector action, while host-side tests and profile scripts can
    inspect how many serial or batched actions were dispatched. The counters are
    intentionally diagnostic only; pass ``apply`` or ``apply_batch`` to solvers,
    not the mutable object itself, when tracing with JAX.
    """

    residual_value: jax.Array
    linear_action: Callable[[object], jax.Array]
    state_shape: tuple[int, ...]
    call_count: int = 0
    batched_call_count: int = 0
    dispatch_seconds: float = 0.0
    batched_dispatch_seconds: float = 0.0

    def _validate_tangent(self, tangent: object) -> jax.Array:
        tangent_array = jnp.asarray(tangent, dtype=jnp.float64)
        tangent_shape = tuple(tangent_array.shape)
        if tangent_shape != self.state_shape:
            raise ValueError(
                f"Residual tangent has shape {tangent_shape}, expected {self.state_shape}."
            )
        return tangent_array

    def _validate_tangent_batch(self, tangent_batch: object) -> jax.Array:
        tangent_batch_array = jnp.asarray(tangent_batch, dtype=jnp.float64)
        tangent_batch_shape = tuple(tangent_batch_array.shape)
        if len(tangent_batch_shape) != len(self.state_shape) + 1:
            raise ValueError(
                "Batched residual tangent array must include exactly one leading "
                f"batch axis, got shape {tangent_batch_shape} for state shape "
                f"{self.state_shape}."
            )
        if tangent_batch_shape[1:] != self.state_shape:
            raise ValueError(
                "Batched residual tangent entries have shape "
                f"{tangent_batch_shape[1:]}, expected {self.state_shape}."
            )
        return tangent_batch_array

    def apply(self, tangent: object) -> jax.Array:
        """Apply the linearized residual Jacobian to one tangent vector."""

        tangent_array = self._validate_tangent(tangent)
        started_at = perf_counter()
        result = self.linear_action(tangent_array)
        self.dispatch_seconds += perf_counter() - started_at
        self.call_count += 1
        return result

    def apply_batch(self, tangent_batch: object) -> jax.Array:
        """Apply the same linearization to a leading batch of tangent vectors."""

        tangent_batch_array = self._validate_tangent_batch(tangent_batch)
        started_at = perf_counter()
        result = jax.vmap(self.linear_action)(tangent_batch_array)
        self.batched_dispatch_seconds += perf_counter() - started_at
        self.batched_call_count += 1
        return result

    def diagnostics(self) -> dict[str, float | int | tuple[int, ...]]:
        """Return host-side dispatch counters for profile artifacts."""

        return {
            "state_shape": self.state_shape,
            "call_count": int(self.call_count),
            "batched_call_count": int(self.batched_call_count),
            "dispatch_seconds": float(self.dispatch_seconds),
            "batched_dispatch_seconds": float(self.batched_dispatch_seconds),
        }


@dataclass(frozen=True)
class FixedResidualLinearizedSolveResult:
    """Result of one matrix-free fixed-residual Newton update."""

    update: jax.Array
    residual_value: jax.Array
    solver_status: object
    solver_success: bool | None
    linear_update_residual_inf_norm: float
    linear_update_relative_residual: float
    diagnostics: dict[str, object]


def build_fixed_residual_linearized_action(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
) -> FixedResidualLinearizedAction:
    """Linearize a fixed-layout residual once and return an instrumented action."""

    state_array = jnp.asarray(packed_state, dtype=jnp.float64)
    residual_value, linear_action = jax.linearize(residual_function, state_array)
    return FixedResidualLinearizedAction(
        residual_value=residual_value,
        linear_action=linear_action,
        state_shape=tuple(state_array.shape),
    )


def solve_fixed_residual_linearized_update(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
    *,
    rhs: object | None = None,
    linear_tolerance: float = 1.0e-8,
    linear_restart: int = 20,
    linear_maxiter: int = 20,
    solve_method: str = "batched",
    preconditioner: Callable[[object], object] | None = None,
    jit_linear_operator: bool = False,
) -> FixedResidualLinearizedSolveResult:
    """Solve one matrix-free Newton update for a fixed-layout residual.

    The stable production recycling solver still owns nonlinear globalization,
    history management and compatibility fallback. This helper is deliberately
    narrower: it exposes a reusable fixed-layout JAX linearization as a
    solver-facing primitive that profile and promotion gates can test without
    constructing sparse finite-difference Jacobians.
    """

    from jax.scipy.sparse.linalg import gmres

    action = build_fixed_residual_linearized_action(
        residual_function,
        packed_state,
    )
    linear_rhs = (
        -jnp.asarray(action.residual_value, dtype=jnp.float64)
        if rhs is None
        else jnp.asarray(rhs, dtype=jnp.float64)
    )
    if tuple(linear_rhs.shape) != tuple(action.state_shape):
        raise ValueError(
            f"Linearized residual RHS has shape {tuple(linear_rhs.shape)}, "
            f"expected {action.state_shape}."
        )
    if bool(jit_linear_operator):
        linear_operator = jax.jit(action.linear_action)
    else:
        linear_operator = action.apply
    update, status = gmres(
        linear_operator,
        linear_rhs,
        tol=max(float(linear_tolerance), jnp.finfo(jnp.float64).tiny),
        atol=0.0,
        restart=max(1, int(linear_restart)),
        maxiter=max(1, int(linear_maxiter)),
        M=preconditioner,
        solve_method=_canonical_gmres_solve_method(solve_method),
    )
    update = jax.block_until_ready(jnp.asarray(update, dtype=jnp.float64))
    linear_update_residual = linear_operator(update) - linear_rhs
    linear_update_residual = jax.block_until_ready(linear_update_residual)
    residual_norm = float(jnp.max(jnp.abs(linear_update_residual)))
    rhs_norm = max(float(jnp.max(jnp.abs(linear_rhs))), jnp.finfo(jnp.float64).tiny)
    normalized_status = _normalize_solver_status(status)
    diagnostics = {
        **action.diagnostics(),
        "linear_operator_jitted": bool(jit_linear_operator),
        "linear_tolerance": float(linear_tolerance),
        "linear_restart": int(linear_restart),
        "linear_maxiter": int(linear_maxiter),
        "solve_method": _canonical_gmres_solve_method(solve_method),
        "preconditioner_used": preconditioner is not None,
    }
    return FixedResidualLinearizedSolveResult(
        update=update,
        residual_value=action.residual_value,
        solver_status=normalized_status,
        solver_success=_solver_status_success(normalized_status),
        linear_update_residual_inf_norm=residual_norm,
        linear_update_relative_residual=residual_norm / rhs_norm,
        diagnostics=diagnostics,
    )


def _canonical_gmres_solve_method(name: str | None) -> str:
    normalized = str(name or "batched").strip().lower().replace("-", "_")
    aliases = {
        "": "batched",
        "default": "batched",
        "batch": "batched",
        "batched": "batched",
        "incremental": "incremental",
        "givens": "incremental",
        "qr": "incremental",
    }
    return aliases.get(normalized, "batched")


def _normalize_solver_status(status: object) -> int | float | str | None:
    if status is None:
        return None
    item = getattr(status, "item", None)
    if callable(item):
        try:
            status = item()
        except (TypeError, ValueError):
            pass
    if isinstance(status, np.generic):
        status = status.item()
    if isinstance(status, (bool, int, float, str)):
        return status
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return str(status)


def _solver_status_success(status: object) -> bool | None:
    if status is None:
        return None
    if isinstance(status, bool):
        return bool(status)
    if isinstance(status, (int, float)):
        return int(status) == 0
    normalized = str(status).strip().lower()
    if normalized in {"0", "success", "successful", "ok", "none"}:
        return True
    if normalized in {"1", "false", "failed", "failure"}:
        return False
    return None


def linearize_fixed_residual_action(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
) -> tuple[jax.Array, Callable[[object], jax.Array]]:
    """Return residual value and a reusable JAX linearized Jacobian action."""

    residual_value, linear_action = jax.linearize(
        residual_function,
        jnp.asarray(packed_state, dtype=jnp.float64),
    )

    def apply_action(tangent: object) -> jax.Array:
        return linear_action(jnp.asarray(tangent, dtype=jnp.float64))

    return residual_value, apply_action


def linearize_fixed_residual_batched_action(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
) -> tuple[jax.Array, Callable[[object], jax.Array]]:
    """Return residual value and a batched linearized Jacobian action.

    The returned action expects a leading batch axis on the tangent array and
    pushes all tangents through the same ``jax.linearize`` result with
    ``jax.vmap``. This is the fixed-layout residual counterpart to the solver's
    colored sparse-JVP batches, but without materializing a sparse matrix.
    """

    state_array = jnp.asarray(packed_state, dtype=jnp.float64)
    residual_value, linear_action = jax.linearize(residual_function, state_array)
    state_shape = tuple(state_array.shape)

    def apply_batched_action(tangent_batch: object) -> jax.Array:
        tangent_batch_array = jnp.asarray(tangent_batch, dtype=jnp.float64)
        tangent_batch_shape = tuple(tangent_batch_array.shape)
        if len(tangent_batch_shape) != len(state_shape) + 1:
            raise ValueError(
                "Batched residual tangent array must include exactly one leading "
                f"batch axis, got shape {tangent_batch_shape} for state shape {state_shape}."
            )
        if tangent_batch_shape[1:] != state_shape:
            raise ValueError(
                "Batched residual tangent entries have shape "
                f"{tangent_batch_shape[1:]}, expected {state_shape}."
            )
        return jax.vmap(linear_action)(tangent_batch_array)

    return residual_value, apply_batched_action


def fixed_residual_jvp_action(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
    tangent: object,
) -> jax.Array:
    """Apply the residual Jacobian to ``tangent`` with a JAX JVP."""

    _, tangent_out = jax.jvp(
        residual_function,
        (jnp.asarray(packed_state, dtype=jnp.float64),),
        (jnp.asarray(tangent, dtype=jnp.float64),),
    )
    return tangent_out


def fixed_residual_jvp_batch_action(
    residual_function: Callable[[object], jax.Array],
    packed_state: object,
    tangent_batch: object,
) -> jax.Array:
    """Apply the residual Jacobian to a batch of tangents with one linearization."""

    _, batched_action = linearize_fixed_residual_batched_action(
        residual_function,
        packed_state,
    )
    return batched_action(tangent_batch)
