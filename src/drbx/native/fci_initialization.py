"""Boundary-compatible initialization helpers for the Rung-3 wall model.

These routines construct an initial state that satisfies the existing
coordinate-wall closures.  They do not alter the boundary conditions used by
the evolution operators.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import SIDE_PHYSICAL
from .fci_halo import neumann_face_trace_physical_affine
from .fci_model import inject_owned_field_to_halo


BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES = (
    "runtime_invalid_count",
    "minimum_density",
    "minimum_Te",
    "minimum_Ti",
    "density_neumann_trace_defect_max",
    "phi_neumann_trace_defect_max",
    "Te_neumann_trace_defect_max",
    "Ti_neumann_trace_defect_max",
    "Vi_neumann_trace_defect_max",
    "Ve_neumann_trace_defect_max",
    "density_normal_bc_max",
    "phi_normal_bc_max",
    "Vi_wall_target_defect_max",
    "Ve_wall_target_defect_max",
    "gauge_residual_abs",
    "core_change_max",
    "inactive_alias_abs_max",
    "vorticity_abs_max",
    "phi_gauge_shift",
    "fixed_outer_owner_defect_max",
    "outer_inactive_owner_count",
    "nonfinite_active_value_count",
    "gauge_constant_response_abs",
    "trace_response_min_abs",
)


_TRACE_RESPONSE_FLOOR = 1.0e-14


def _validate_outer_patch_layout(domain, layer_count: int) -> None:
    shape = tuple(int(value) for value in domain.layout.owned_shape)
    if len(shape) != 3:
        raise ValueError(f"boundary-compatible initialization requires 3-D fields, got {shape}")
    nx = shape[0]
    layers = int(layer_count)
    if layers < 2 or layers >= nx:
        raise ValueError(
            "boundary-compatible initialization requires "
            "2 <= layer_count < owned radial extent"
        )
    halo_width = int(domain.layout.halo_width)
    if halo_width < 2:
        raise ValueError(
            "boundary-compatible initialization requires halo_width >= 2 so "
            "the mirrored face trace responds to the penultimate owner"
        )
    if halo_width > nx:
        raise ValueError(
            "boundary-compatible initialization requires owned radial extent "
            "at least halo_width"
        )
    shard_spec = domain.shard_spec
    if int(shard_spec.shard_counts[0]) != 1:
        raise ValueError(
            "boundary-compatible initialization currently requires one radial shard"
        )


def validate_boundary_compatible_initialization_support(
    domain,
    layer_count: int,
) -> None:
    """Validate the static geometry supported by the complete initializer."""

    _validate_outer_patch_layout(domain, layer_count)
    shard_spec = domain.shard_spec
    if tuple(bool(value) for value in shard_spec.periodic_axes[1:]) != (True, True):
        raise ValueError(
            "boundary-compatible initialization currently supports toroidal "
            "domains with periodic angular axes"
        )
    for axis in range(3):
        for side in ("lower", "upper"):
            kind = (
                shard_spec.lower_side_kind(axis)
                if side == "lower"
                else shard_spec.upper_side_kind(axis)
            )
            if kind == SIDE_PHYSICAL and not (axis == 0 and side == "upper"):
                raise ValueError(
                    "boundary-compatible initialization supports only the "
                    "physical upper-radial coordinate wall"
                )
    if shard_spec.upper_side_kind(0) != SIDE_PHYSICAL:
        raise ValueError(
            "boundary-compatible initialization requires a physical upper-radial wall"
        )


def _spmd_reduce(value: jax.Array, domain, operation: str) -> jax.Array:
    result = jnp.asarray(value)
    for axis_name in domain.mesh_axis_names:
        if axis_name is None:
            continue
        if operation == "sum":
            result = jax.lax.psum(result, axis_name=axis_name)
        elif operation == "max":
            result = jax.lax.pmax(result, axis_name=axis_name)
        elif operation == "min":
            result = jax.lax.pmin(result, axis_name=axis_name)
        else:  # pragma: no cover - private callers use the fixed operations above.
            raise ValueError(f"unknown SPMD reduction {operation!r}")
    return result


def _topology_halo(model, value: jax.Array) -> jax.Array:
    halo = inject_owned_field_to_halo(value, model.domain.layout)
    if model.halo_exchange is not None:
        halo = model.halo_exchange(halo, model.domain)
    if model.topology_filler is not None:
        halo = model.topology_filler(halo, model.domain)
    return halo


def _quiet_outer_trace_patch(
    model,
    field_owned: jax.Array,
    target_face: jax.Array,
    *,
    layer_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Match the upper owner and its zero-normal reconstructed face trace."""

    domain = model.domain
    geometry = model.geometry
    value = jnp.asarray(field_owned, dtype=jnp.float64)
    target = jnp.asarray(target_face, dtype=jnp.float64)
    shape = tuple(int(extent) for extent in domain.layout.owned_shape)
    if value.shape != shape:
        raise ValueError(f"field_owned must have shape {shape}, got {value.shape}")
    if target.shape != shape[1:]:
        raise ValueError(f"target_face must have shape {shape[1:]}, got {target.shape}")
    _validate_outer_patch_layout(domain, layer_count)

    nx = shape[0]
    h = int(domain.layout.halo_width)
    centers = jnp.asarray(
        geometry.grid.x.centers_halo[h : h + nx], dtype=jnp.float64
    )
    anchor = nx - int(layer_count) - 1
    radial_span = centers[-1] - centers[anchor]
    span_valid = jnp.isfinite(radial_span) & (radial_span > 0.0)
    safe_span = jnp.where(span_valid, radial_span, 1.0)
    s = jnp.clip((centers - centers[anchor]) / safe_span, 0.0, 1.0)
    w0 = jnp.where(
        s > 0.0,
        s**3 * (10.0 - 15.0 * s + 6.0 * s**2),
        0.0,
    )[:, None, None]
    raw_w1 = s**3 * (1.0 - s)
    w1_scale = raw_w1[nx - 2]
    scale_valid = jnp.isfinite(w1_scale) & (jnp.abs(w1_scale) >= _TRACE_RESPONSE_FLOOR)
    safe_scale = jnp.where(scale_valid, w1_scale, 1.0)
    w1 = jnp.broadcast_to((raw_w1 / safe_scale)[:, None, None], value.shape)

    candidate = value + w0 * (target[None, ...] - value[-1][None, ...])
    base, _ = neumann_face_trace_physical_affine(
        _topology_halo(model, candidate), geometry, domain, 0, "upper"
    )
    response, _ = neumann_face_trace_physical_affine(
        _topology_halo(model, w1), geometry, domain, 0, "upper"
    )
    response_valid = jnp.isfinite(response) & (
        jnp.abs(response) >= _TRACE_RESPONSE_FLOOR
    )
    safe_response = jnp.where(response_valid, response, 1.0)
    result = candidate + w1 * ((target - base) / safe_response)[None, ...]
    invalid_count = (
        (~span_valid).astype(jnp.float64)
        + (~scale_valid).astype(jnp.float64)
        + jnp.sum((~response_valid).astype(jnp.float64))
        + jnp.sum((~jnp.isfinite(result)).astype(jnp.float64))
    )
    response_min_abs = jnp.min(
        jnp.where(jnp.isfinite(response), jnp.abs(response), jnp.inf)
    )
    return result, invalid_count, response_min_abs


def quiet_outer_trace_patch(
    model,
    field_owned: jax.Array,
    target_face: jax.Array,
    *,
    layer_count: int = 8,
) -> jax.Array:
    """Return a compact upper-radial patch with a zero-normal target trace."""

    result, _invalid_count, _response_min_abs = _quiet_outer_trace_patch(
        model, field_owned, target_face, layer_count=layer_count
    )
    return result


def initialize_boundary_compatible_rung3(
    model,
    current,
    *,
    layer_count: int = 8,
) -> tuple[object, jax.Array]:
    """Apply the coordinate-wall compatibility constraints and derive vorticity.

    ``current`` is expected to have already passed through the live directional
    FCI velocity-layer blend.  Physical endpoint resolution remains unchanged;
    this pass only chooses a compatible initial owner state for the existing
    upper-radial wall laws.
    """

    if getattr(model, "physical_wall_model_name", None) != "simplified-gbs-mpe":
        raise ValueError(
            "boundary-compatible Rung-3 initialization requires the "
            "simplified-gbs-mpe physical wall model"
        )
    validate_boundary_compatible_initialization_support(model.domain, layer_count)
    domain = model.domain
    geometry = model.geometry
    nx = int(domain.layout.owned_shape[0])
    layers = int(layer_count)
    anchor_stop = nx - layers

    initial_owner = model._owner_state(current)
    initial = model._materialized_wall_state(initial_owner)
    active = jnp.asarray(
        (
            model.control_volume_geometry.cells.is_active_owner
            if model.control_volume_geometry is not None
            else geometry.active_cell_mask_owned
        ),
        dtype=bool,
    )
    if active.shape != tuple(domain.layout.owned_shape):
        raise ValueError("active owner mask must match the local owned shape")

    primitive_names = ("density", "phi", "Te", "Ti", "Vi", "Ve")
    before = {name: jnp.asarray(getattr(initial, name), dtype=jnp.float64) for name in primitive_names}
    fixed_targets = {name: before[name][-1] for name in ("density", "phi", "Te", "Ti")}
    quiet = initial
    invalid_count = jnp.asarray(0.0, dtype=jnp.float64)
    trace_response_min = jnp.asarray(jnp.inf, dtype=jnp.float64)

    def patch(name: str, target: jax.Array) -> None:
        nonlocal quiet, invalid_count, trace_response_min
        result, invalid, response_min = _quiet_outer_trace_patch(
            model, getattr(quiet, name), target, layer_count=layers
        )
        quiet = quiet.replace(**{name: result})
        invalid_count = invalid_count + invalid
        trace_response_min = jnp.minimum(trace_response_min, response_min)

    for name, target in fixed_targets.items():
        patch(name, target)

    b_normal = jnp.asarray(
        geometry.face_bfield.axes[0].B_contra_owned[..., 0][-1],
        dtype=jnp.float64,
    )
    invalid_count = invalid_count + jnp.sum((~jnp.isfinite(b_normal)).astype(jnp.float64))
    te_trace, _ = neumann_face_trace_physical_affine(
        _topology_halo(model, quiet.Te), geometry, domain, 0, "upper"
    )
    ti_trace, _ = neumann_face_trace_physical_affine(
        _topology_halo(model, quiet.Ti), geometry, domain, 0, "upper"
    )
    sound_speed_argument = te_trace + model.parameters.tau * ti_trace
    sound_speed_valid = jnp.isfinite(sound_speed_argument) & (
        sound_speed_argument > 0.0
    )
    sound_speed = jnp.sqrt(jnp.where(sound_speed_valid, sound_speed_argument, 1.0))
    sigma = jnp.where(
        b_normal >= 1.0e-12,
        1.0,
        jnp.where(b_normal <= -1.0e-12, -1.0, 0.0),
    )
    vi_target = jnp.where(jnp.abs(b_normal) <= 1.0e-12, quiet.Vi[-1], sigma * sound_speed)
    invalid_count = invalid_count + jnp.sum((~sound_speed_valid).astype(jnp.float64))
    patch("Vi", vi_target)

    face_before_gauge = model._face_bcs(quiet)
    gauge_weights, gauge_affine, gauge_target = model._simplified_gbs_mpe_phi_gauge_data(
        quiet, face_before_gauge
    )
    gauge_weight_sum = _spmd_reduce(jnp.sum(gauge_weights), domain, "sum")
    gauge_phi = _spmd_reduce(jnp.sum(gauge_weights * quiet.phi), domain, "sum")
    gauge_valid = jnp.isfinite(gauge_weight_sum) & (
        jnp.abs(gauge_weight_sum) >= _TRACE_RESPONSE_FLOOR
    )
    safe_gauge_response = jnp.where(gauge_valid, gauge_weight_sum, 1.0)
    gauge_shift = (gauge_target - gauge_affine - gauge_phi) / safe_gauge_response
    invalid_count = invalid_count + (~gauge_valid).astype(jnp.float64)
    quiet = quiet.replace(phi=quiet.phi + gauge_shift)

    face_after_gauge = model._face_bcs(quiet)
    ve_target = face_after_gauge.Ve.value_x[-1]
    patch("Ve", ve_target)

    result = model._owner_state(quiet)
    final_face = model._face_bcs(result)
    result = model._owner_state(
        result.replace(
            vorticity=model._vorticity_from_polarization(
                result.phi,
                result.Ti,
                final_face.phi,
                final_face.Ti,
            )
        )
    )
    final_materialized = model._materialized_wall_state(result)
    final_face = model._face_bcs(result)

    trace_defects = []
    for name in primitive_names:
        trace, _ = neumann_face_trace_physical_affine(
            _topology_halo(model, getattr(final_materialized, name)),
            geometry,
            domain,
            0,
            "upper",
        )
        trace_defects.append(jnp.max(jnp.abs(trace - getattr(final_materialized, name)[-1])))

    active_count = _spmd_reduce(jnp.sum(active.astype(jnp.float64)), domain, "sum")
    outer_inactive = _spmd_reduce(
        jnp.sum((~active[-layers:]).astype(jnp.float64)), domain, "sum"
    )
    minima = []
    for name in ("density", "Te", "Ti"):
        local_minimum = jnp.min(
            jnp.where(active, getattr(result, name), jnp.inf)
        )
        minima.append(_spmd_reduce(local_minimum, domain, "min"))

    nonfinite_active = jnp.asarray(0.0, dtype=jnp.float64)
    inactive_alias_abs = jnp.asarray(0.0, dtype=jnp.float64)
    for name, values in result.field_items():
        values = jnp.asarray(values, dtype=jnp.float64)
        nonfinite_active = nonfinite_active + jnp.sum(
            (active & ~jnp.isfinite(values)).astype(jnp.float64)
        )
        inactive_alias_abs = jnp.maximum(
            inactive_alias_abs,
            jnp.max(jnp.where(~active, jnp.abs(values), 0.0)),
        )
    nonfinite_active = _spmd_reduce(nonfinite_active, domain, "sum")
    inactive_alias_abs = _spmd_reduce(inactive_alias_abs, domain, "max")

    radial_core = jnp.arange(nx)[:, None, None] < anchor_stop
    core_mask = active & radial_core
    core_change = jnp.asarray(0.0, dtype=jnp.float64)
    for name in primitive_names:
        expected_shift = gauge_shift if name == "phi" else 0.0
        difference = (
            getattr(final_materialized, name) - before[name] - expected_shift
        )
        core_change = jnp.maximum(
            core_change,
            jnp.max(jnp.where(core_mask, jnp.abs(difference), 0.0)),
        )
    core_change = _spmd_reduce(core_change, domain, "max")

    fixed_owner_defect = jnp.asarray(0.0, dtype=jnp.float64)
    for name in ("density", "phi", "Te", "Ti"):
        expected_shift = gauge_shift if name == "phi" else 0.0
        fixed_owner_defect = jnp.maximum(
            fixed_owner_defect,
            jnp.max(
                jnp.abs(
                    getattr(final_materialized, name)[-1]
                    - expected_shift
                    - fixed_targets[name]
                )
            ),
        )
    fixed_owner_defect = _spmd_reduce(fixed_owner_defect, domain, "max")
    gauge_residual = (
        _spmd_reduce(jnp.sum(gauge_weights * result.phi), domain, "sum")
        + gauge_affine
        - gauge_target
    )
    vorticity_abs = _spmd_reduce(
        jnp.max(jnp.where(active, jnp.abs(result.vorticity), 0.0)), domain, "max"
    )

    invalid_count = _spmd_reduce(invalid_count, domain, "sum")
    trace_defects = [_spmd_reduce(value, domain, "max") for value in trace_defects]
    density_normal = _spmd_reduce(jnp.max(jnp.abs(final_face.density.value_x[-1])), domain, "max")
    phi_normal = _spmd_reduce(jnp.max(jnp.abs(final_face.phi.value_x[-1])), domain, "max")
    vi_target_defect = _spmd_reduce(
        jnp.max(jnp.abs(final_face.Vi.value_x[-1] - final_materialized.Vi[-1])), domain, "max"
    )
    ve_target_defect = _spmd_reduce(
        jnp.max(jnp.abs(final_face.Ve.value_x[-1] - final_materialized.Ve[-1])), domain, "max"
    )
    trace_response_min = _spmd_reduce(trace_response_min, domain, "min")
    invalid_count = invalid_count + outer_inactive + (active_count <= 0.0).astype(jnp.float64)

    diagnostics = jnp.asarray(
        (
            invalid_count,
            minima[0], minima[1], minima[2],
            trace_defects[0], trace_defects[1], trace_defects[2],
            trace_defects[3], trace_defects[4], trace_defects[5],
            density_normal, phi_normal,
            vi_target_defect, ve_target_defect,
            jnp.abs(gauge_residual), core_change, inactive_alias_abs,
            vorticity_abs, gauge_shift, fixed_owner_defect,
            outer_inactive, nonfinite_active,
            jnp.abs(gauge_weight_sum), trace_response_min,
        ),
        dtype=jnp.float64,
    )
    return result, diagnostics


def boundary_compatibility_acceptance_failures(
    diagnostics: Sequence[float],
) -> tuple[str, ...]:
    """Return host-side rejection reasons for compiled diagnostics."""

    values = np.asarray(diagnostics, dtype=np.float64)
    if values.shape != (len(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES),):
        return (
            "diagnostic vector has shape "
            f"{values.shape}, expected {(len(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES),)}",
        )
    named = dict(zip(BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES, values, strict=True))
    failures: list[str] = []
    if not np.all(np.isfinite(values)):
        failures.append("diagnostics contain nonfinite values")
    if named["runtime_invalid_count"] != 0.0:
        failures.append(f"runtime_invalid_count={named['runtime_invalid_count']:.0f}")
    for name in ("minimum_density", "minimum_Te", "minimum_Ti"):
        if not named[name] > 0.0:
            failures.append(f"{name}={named[name]:.6e}")
    for name in (
        "density_neumann_trace_defect_max", "phi_neumann_trace_defect_max",
        "Te_neumann_trace_defect_max", "Ti_neumann_trace_defect_max",
        "Vi_neumann_trace_defect_max", "Ve_neumann_trace_defect_max",
        "density_normal_bc_max", "phi_normal_bc_max",
    ):
        if named[name] > 1.0e-8:
            failures.append(f"{name}={named[name]:.6e} > 1.0e-8")
    for name in ("Vi_wall_target_defect_max", "Ve_wall_target_defect_max"):
        if named[name] > 1.0e-9:
            failures.append(f"{name}={named[name]:.6e} > 1.0e-9")
    for name, tolerance in (
        ("gauge_residual_abs", 1.0e-10),
        ("core_change_max", 1.0e-11),
        ("inactive_alias_abs_max", 1.0e-12),
        ("fixed_outer_owner_defect_max", 1.0e-11),
        ("outer_inactive_owner_count", 0.0),
        ("nonfinite_active_value_count", 0.0),
    ):
        if named[name] > tolerance:
            failures.append(f"{name}={named[name]:.6e} > {tolerance:.1e}")
    if named["gauge_constant_response_abs"] < _TRACE_RESPONSE_FLOOR:
        failures.append("gauge constant response is singular")
    if named["trace_response_min_abs"] < _TRACE_RESPONSE_FLOOR:
        failures.append("outer trace response is singular")
    return tuple(failures)


__all__ = [
    "BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES",
    "boundary_compatibility_acceptance_failures",
    "initialize_boundary_compatible_rung3",
    "quiet_outer_trace_patch",
    "validate_boundary_compatible_initialization_support",
]
