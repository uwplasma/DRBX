"""JAX-native continuous-field tracing for FCI points.

This module contains the point-tracing part of the callback FCI path without
any dependency on a materialized cell-centred field.  The callback is called
with an ``(N, 3)`` array of logical coordinates and must return
``(B_contravariant, Bmag)``.  Both callback evaluations and the complete
integrator therefore remain inside a JAX transformation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp


def _field_values(field_evaluator: Callable, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract and shape-check a JAX callback result."""

    result = field_evaluator(points)
    if isinstance(result, Mapping):
        b = result["B_contravariant"]
        bmag = result["magnitude"]
    elif hasattr(result, "B_contravariant") and hasattr(result, "magnitude"):
        b = result.B_contravariant
        bmag = result.magnitude
    else:
        b, bmag = result
    b = jnp.asarray(b, dtype=jnp.float64)
    bmag = jnp.asarray(bmag, dtype=jnp.float64)
    if b.shape != (points.shape[0], 3):
        raise ValueError(
            "field_evaluator B_contravariant must have shape "
            f"{(points.shape[0], 3)}, got {b.shape}"
        )
    if bmag.shape not in ((points.shape[0],), (points.shape[0], 1)):
        raise ValueError(
            "field_evaluator Bmag must have shape "
            f"{(points.shape[0],)}, got {bmag.shape}"
        )
    return b, bmag.reshape((points.shape[0],))


def trace_fci_points_to_plane_jax(
    grid_bounds,
    field_evaluator: Callable,
    seed_points,
    eta_step,
    *,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    min_abs_bz: float = 1.0e-30,
    axis_epsilon: float | None = None,
    valid_mask=None,
    seed_valid_mask=None,
) -> dict[str, jnp.ndarray]:
    """Trace a batch of logical points through one signed eta interval.

    Parameters
    ----------
    grid_bounds:
        Three ``(lower, upper)`` coordinate bounds, as an array of shape
        ``(3, 2)`` (lists and tuples are accepted too).
    field_evaluator:
        JAX-compatible callable receiving ``(N, 3)`` points and returning a
        two-tuple ``(B_contravariant, Bmag)``.  A mapping with keys
        ``B_contravariant`` and ``magnitude`` or an object exposing those
        attributes is also accepted.
    seed_points:
        Array of shape ``(N, 3)``.
    eta_step:
        A scalar, or one signed step per seed with shape ``(N,)``.  The
        returned connection length is always nonnegative.
    valid_mask, seed_valid_mask:
        Optional boolean shape ``(N,)`` mask for padded/inactive seed rows.
        ``seed_valid_mask`` is an alias for ``valid_mask``; passing both is an
        error.  Inactive rows remain unchanged and have zero length and a
        false boundary flag.  Padding is local to each batch/shard; callers
        using a sharded leading dimension should pad each shard as needed and
        keep the global batch dimension divisible by its device count.

    The topology arguments are Python/static configuration.  The numerical
    arrays, callback, and the whole fixed-substep RK4 integration are JAX
    compatible and can be wrapped in ``jax.jit``.  Lower-radial axis
    regularity follows the signed-radius identification: a negative radius is
    sampled at its absolute radius and ``theta + pi``, with the radial field
    component changing sign.  Only the lower-radial x axis is currently
    supported, matching the callback tracer's topology.
    """

    bounds = jnp.asarray(grid_bounds, dtype=jnp.float64)
    points = jnp.asarray(seed_points, dtype=jnp.float64)
    if bounds.ndim != 2 or bounds.shape != (3, 2):
        raise ValueError(f"grid_bounds must have shape (3, 2), got {bounds.shape}")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"seed_points must have shape (n, 3), got {points.shape}")
    if valid_mask is not None and seed_valid_mask is not None:
        raise ValueError("pass only one of valid_mask and seed_valid_mask")
    if seed_valid_mask is not None:
        valid_mask = seed_valid_mask
    if valid_mask is None:
        row_valid = jnp.ones((points.shape[0],), dtype=bool)
    else:
        row_valid = jnp.asarray(valid_mask, dtype=bool)
        if row_valid.ndim != 1 or row_valid.shape != (points.shape[0],):
            raise ValueError(
                f"valid_mask must have shape ({points.shape[0]},), got {row_valid.shape}"
            )

    if len(periodic_axes) != 3 or len(axis_regular_axes) != 3:
        raise ValueError("periodic_axes and axis_regular_axes must have length 3")
    periodic_axes = tuple(bool(v) for v in periodic_axes)
    axis_regular_axes = tuple(bool(v) for v in axis_regular_axes)
    if any(axis_regular_axes[1:]):
        raise ValueError(
            "FCI axis regularity currently supports only the lower-radial x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    axis_regular_x = axis_regular_axes[0]
    if axis_regular_x and (periodic_axes[0] or not periodic_axes[1] or not periodic_axes[2]):
        raise ValueError(
            "lower-radial axis regularity requires x nonperiodic and y/z periodic; "
            f"got periodic_axes={periodic_axes}"
        )

    # Shape is static even while values may be tracers.  Keeping this check
    # here catches accidental broadcasting before the callback is entered.
    n_points = points.shape[0]
    eta = jnp.asarray(eta_step, dtype=jnp.float64)
    if eta.ndim == 0:
        eta = jnp.broadcast_to(eta, (n_points,))
    elif eta.ndim == 1 and eta.shape == (n_points,):
        pass
    else:
        raise ValueError(f"eta_step must be a scalar or shape ({n_points},), got {eta.shape}")

    x_lower, x_upper = bounds[0, 0], bounds[0, 1]
    y_lower, y_upper = bounds[1, 0], bounds[1, 1]
    z_lower, z_upper = bounds[2, 0], bounds[2, 1]
    y_period = y_upper - y_lower

    # These controls are deliberately static Python values.  This keeps
    # topology branches out of the traced numerical graph and avoids dynamic
    # indexing/conditionals in the RK loop.
    periodic_x, periodic_y, periodic_z = periodic_axes
    min_bz = jnp.asarray(min_abs_bz, dtype=jnp.float64)
    if axis_regular_x and axis_epsilon is None:
        # The grid-aware production wrapper passes the NumPy tracer's exact
        # max(1e-12, 1e-8 * first_radial_cell_width) value.  Retain a
        # transformation-safe bounds-only fallback for the standalone kernel.
        sample_axis_epsilon = jnp.maximum(
            1.0e-12,
            1.0e-12 * jnp.maximum(jnp.abs(x_upper - x_lower), 1.0),
        )
    else:
        sample_axis_epsilon = jnp.asarray(
            0.0 if axis_epsilon is None else axis_epsilon,
            dtype=jnp.float64,
        )

    def _wrap_coordinate(value, lower, upper, periodic):
        if periodic:
            return jnp.mod(value - lower, upper - lower) + lower
        return value

    def _wrap_points(values):
        return jnp.stack(
            (
                _wrap_coordinate(values[:, 0], x_lower, x_upper, periodic_x),
                _wrap_coordinate(values[:, 1], y_lower, y_upper, periodic_y),
                _wrap_coordinate(values[:, 2], z_lower, z_upper, periodic_z),
            ),
            axis=-1,
        )

    def _signed_wrap_points(values):
        """Wrap tangential coordinates while retaining signed radius."""

        return jnp.stack(
            (
                _wrap_coordinate(values[:, 0], x_lower, x_upper, periodic_x)
                if not axis_regular_x
                else values[:, 0],
                _wrap_coordinate(values[:, 1], y_lower, y_upper, periodic_y),
                _wrap_coordinate(values[:, 2], z_lower, z_upper, periodic_z),
            ),
            axis=-1,
        )

    def _regularize_points(values):
        values = _signed_wrap_points(values)
        if axis_regular_x:
            crossed = values[:, 0] < x_lower
            x = jnp.where(crossed, 2.0 * x_lower - values[:, 0], values[:, 0])
            theta = jnp.where(crossed, values[:, 1] + 0.5 * y_period, values[:, 1])
            values = jnp.stack((x, theta, values[:, 2]), axis=-1)
        return _wrap_points(values)

    def _sample(values):
        signed = _signed_wrap_points(values)
        if axis_regular_x:
            negative = signed[:, 0] < x_lower
            x = jnp.abs(signed[:, 0] - x_lower) + x_lower
            theta = jnp.where(negative, signed[:, 1] + 0.5 * y_period, signed[:, 1])
            sample = jnp.stack((x, theta, signed[:, 2]), axis=-1)
            sample = _wrap_points(sample)
            # Avoid an exact singular coordinate in callbacks that use a
            # coordinate Jacobian.  The value is below numerical significance
            # for the signed-radius chart, as in the NumPy callback tracer.
            sample = sample.at[:, 0].set(
                jnp.maximum(sample[:, 0], sample_axis_epsilon)
            )
        else:
            sample = _wrap_points(signed)

        sample = jnp.stack(
            (
                sample[:, 0] if periodic_x else jnp.clip(sample[:, 0], x_lower, x_upper),
                sample[:, 1] if periodic_y else jnp.clip(sample[:, 1], y_lower, y_upper),
                sample[:, 2] if periodic_z else jnp.clip(sample[:, 2], z_lower, z_upper),
            ),
            axis=-1,
        )
        b, bmag = _field_values(field_evaluator, sample)
        if axis_regular_x:
            negative = signed[:, 0] < x_lower
            b = b.at[:, 0].set(jnp.where(negative, -b[:, 0], b[:, 0]))
        return b, bmag

    def _safe_bz(bz):
        return jnp.where(jnp.abs(bz) < min_bz, jnp.where(bz < 0.0, -min_bz, min_bz), bz)

    def _rhs_from_field(b):
        bz = _safe_bz(b[:, 2])
        return jnp.stack((b[:, 0] / bz, b[:, 1] / bz, jnp.ones_like(bz)), axis=-1)

    def _speed_from_field(b, bmag):
        bz = _safe_bz(b[:, 2])
        return bmag / jnp.maximum(jnp.abs(bz), 1.0e-30)

    def _rhs(values):
        b, _ = _sample(values)
        return _rhs_from_field(b)

    def _speed(values):
        b, bmag = _sample(values)
        return _speed_from_field(b, bmag)

    def _valid(values):
        result = jnp.all(jnp.isfinite(values), axis=1)
        if not periodic_x:
            if axis_regular_x:
                result = result & (jnp.abs(values[:, 0] - x_lower) <= (x_upper - x_lower))
            else:
                result = result & (values[:, 0] >= x_lower) & (values[:, 0] <= x_upper)
        if not periodic_y:
            result = result & (values[:, 1] >= y_lower) & (values[:, 1] <= y_upper)
        if not periodic_z:
            result = result & (values[:, 2] >= z_lower) & (values[:, 2] <= z_upper)
        return result

    def _boundary_hit(old, new, valid_new):
        fractions = jnp.full((n_points,), jnp.inf, dtype=jnp.float64)
        for axis, lower, upper, periodic, ignore_lower in (
            (0, x_lower, x_upper, periodic_x, axis_regular_x),
            (1, y_lower, y_upper, periodic_y, False),
            (2, z_lower, z_upper, periodic_z, False),
        ):
            if periodic:
                continue
            delta = new[:, axis] - old[:, axis]
            safe_delta = jnp.where(jnp.abs(delta) < 1.0e-300, 1.0, delta)
            candidate = jnp.full((n_points,), jnp.inf, dtype=jnp.float64)
            if axis == 0 and axis_regular_x:
                candidate = jnp.where(
                    new[:, axis] < 2.0 * lower - upper,
                    (2.0 * lower - upper - old[:, axis]) / safe_delta,
                    candidate,
                )
            elif not ignore_lower:
                candidate = jnp.where(
                    new[:, axis] < lower,
                    (lower - old[:, axis]) / safe_delta,
                    candidate,
                )
            candidate = jnp.minimum(
                candidate,
                jnp.where(new[:, axis] > upper, (upper - old[:, axis]) / safe_delta, jnp.inf),
            )
            candidate = jnp.where((candidate >= 0.0) & (candidate <= 1.0), candidate, jnp.inf)
            fractions = jnp.minimum(fractions, candidate)
        has_hit = (~valid_new) & jnp.isfinite(fractions)
        fraction = jnp.where(has_hit, fractions, 1.0)
        hit = old + fraction[:, None] * (new - old)
        hit = jnp.stack(
            (
                jnp.clip(hit[:, 0], 2.0 * x_lower - x_upper if axis_regular_x else x_lower, x_upper)
                if not periodic_x else hit[:, 0],
                jnp.clip(hit[:, 1], y_lower, y_upper) if not periodic_y else hit[:, 1],
                jnp.clip(hit[:, 2], z_lower, z_upper) if not periodic_z else hit[:, 2],
            ),
            axis=-1,
        )
        return _signed_wrap_points(hit), fraction

    # ``fori_loop`` permits a dynamic loop bound for callers that choose to
    # stage it as a JAX scalar, while a normal Python integer remains a fixed
    # compile-time topology choice.  Eager calls still get the useful input
    # validation below.
    try:
        python_substeps = int(substeps)
    except (TypeError, ValueError):
        python_substeps = None
    if python_substeps is not None and python_substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    loop_upper = substeps if python_substeps is None else python_substeps
    step = eta / jnp.asarray(loop_upper, dtype=jnp.float64)

    def _rk4(values, h, k1):
        h = h[:, None]
        k2 = _rhs(_signed_wrap_points(values + 0.5 * h * k1))
        k3 = _rhs(_signed_wrap_points(values + 0.5 * h * k2))
        k4 = _rhs(_signed_wrap_points(values + h * k3))
        return values + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _body(_, carry):
        state, lengths, alive, boundary, field0, field0_magnitude = carry
        state = _signed_wrap_points(state)
        speed0 = _speed_from_field(field0, field0_magnitude)
        raw_next = _rk4(state, step, _rhs_from_field(field0))
        finite_next = jnp.all(jnp.isfinite(raw_next), axis=1)
        unwrapped_next = jnp.where(finite_next[:, None], raw_next, state)
        next_state = _signed_wrap_points(unwrapped_next)
        valid_next = _valid(next_state)
        active_full = alive & finite_next & valid_next
        active_exit = alive & finite_next & (~valid_next)
        hit_state, fraction = _boundary_hit(state, unwrapped_next, valid_next)
        field1, field1_magnitude = _sample(next_state)
        hit_field, hit_field_magnitude = _sample(hit_state)
        speed1 = _speed_from_field(field1, field1_magnitude)
        hit_speed = _speed_from_field(hit_field, hit_field_magnitude)
        lengths = lengths + jnp.where(
            active_full, 0.5 * jnp.abs(step) * (speed0 + speed1), 0.0
        )
        lengths = lengths + jnp.where(
            active_exit, 0.5 * jnp.abs(step) * fraction * (speed0 + hit_speed), 0.0
        )
        state = jnp.where(active_full[:, None], next_state, state)
        state = jnp.where(active_exit[:, None], hit_state, state)
        state = _signed_wrap_points(state)
        field0 = jnp.where(active_full[:, None], field1, field0)
        field0 = jnp.where(active_exit[:, None], hit_field, field0)
        field0_magnitude = jnp.where(
            active_full, field1_magnitude, field0_magnitude
        )
        field0_magnitude = jnp.where(
            active_exit, hit_field_magnitude, field0_magnitude
        )
        boundary = boundary | active_exit | (alive & ~finite_next)
        alive = alive & finite_next & valid_next
        return None, (
            state,
            lengths,
            alive,
            boundary,
            field0,
            field0_magnitude,
        )

    initial_state = _signed_wrap_points(points)
    initial_field, initial_field_magnitude = _sample(initial_state)
    init = (
        initial_state,
        jnp.zeros((n_points,), dtype=jnp.float64),
        row_valid,
        jnp.zeros((n_points,), dtype=bool),
        initial_field,
        initial_field_magnitude,
    )
    state, lengths, _, boundary, _, _ = jax.lax.fori_loop(
        0, loop_upper, lambda _, carry: _body(None, carry)[1], init
    )
    endpoint = _regularize_points(state)
    endpoint_b, endpoint_bmag = _sample(endpoint)
    return {
        "endpoint": endpoint,
        "length": lengths,
        "boundary": boundary,
        "endpoint_b_contravariant": endpoint_b,
        "endpoint_bmag": endpoint_bmag,
    }


# Short aliases make the standalone kernel convenient while retaining the
# descriptive name used by the existing callback tracer.
trace_fci_points_jax = trace_fci_points_to_plane_jax
trace_fci_points = trace_fci_points_to_plane_jax
trace_fci_points_to_plane_from_callbacks_jax = trace_fci_points_to_plane_jax


__all__ = [
    "trace_fci_points_to_plane_jax",
    "trace_fci_points_jax",
    "trace_fci_points",
    "trace_fci_points_to_plane_from_callbacks_jax",
]
