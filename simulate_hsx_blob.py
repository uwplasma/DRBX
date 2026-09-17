#!/usr/bin/env python3
"""Run the HSX seven-field EB model from an explicit geometry artifact.

Physical geometry production is intentionally outside this driver.  The
consumer loads one producer-qualified ``FciSimulationGeometry3D`` directory,
lowers it for the selected device layout, assembles operators, and advances
the state.  It never fits metrics, traces field lines, searches caches, or
regenerates missing geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
_drbx_source_override = os.environ.get("DRBX_SOURCE_ROOT")
DRBX_SRC = (
    Path(_drbx_source_override).expanduser().resolve()
    if _drbx_source_override
    else SCRIPT_DIR / "src"
)
if not DRBX_SRC.is_dir() or not (DRBX_SRC / "drbx").is_dir():
    source_origin = "DRBX_SOURCE_ROOT" if _drbx_source_override else "default"
    raise RuntimeError(
        f"{source_origin} DRBX source root must be a src directory containing "
        f"a drbx package, got {DRBX_SRC}"
    )
if str(DRBX_SRC) not in sys.path:
    sys.path.insert(0, str(DRBX_SRC))

from drbx.runtime import configure_jax_runtime  # noqa: E402

configure_jax_runtime(precision="float64")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402
from drbx.geometry import (  # noqa: E402
    FciGeometry3D,
    LocalDomain3D,
    LocalCurvatureFaceCoefficients3D,
    LocalFciGeometry3D,
    build_local_curvature_face_coefficients,
)
from drbx.native import (  # noqa: E402
    FciDrbEBRhsParameters,
    FciDrbEBState,
    GhostFillWeights1D,
    HaloExchange3D,
    LocalBoundaryFaceBC3D,
    LocalFciDrbEBPhysicalWallBundle,
    LocalFciDrbEBRhs,
    PHYSICAL_WALL_MODEL_NAMES,
    LocalPeriodicTopologyRule3D,
    MetricAwarePhysicalGhostCellFiller3D,
    PhysicalGhostCellFiller3D,
    ShardedFciGeometry3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    build_local_fci_drb_eb_operator_boundary_bundle,
    make_default_topology_halo_filler_3d,
    make_shard_mesh,
    physical_wall_model_from_name,
)
from drbx.native.fci_angular_agglomeration import (  # noqa: E402
    RLP_PACKED_FIELD_COUNT,
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
    empty_angular_agglomeration_boundary_bc,
)
from drbx.native.fci_owner_agglomeration import (  # noqa: E402
    CORNER_EDGE_PACKED_FIELD_COUNT,
    assemble_local_plane_local_owner_map_geometry,
    build_sharded_plane_local_owner_map_payload,
)
from drbx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN  # noqa: E402
from drbx.native.fci_polarization_coarse import build_coarse_data  # noqa: E402
from drbx.native.fci_initialization import (  # noqa: E402
    BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES,
    boundary_compatibility_acceptance_failures,
    initialize_boundary_compatible_rung3,
    validate_boundary_compatible_initialization_support,
)
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    curvature_component_diagnostic_names,
    parallel_characteristic_matrix,
    prepare_local_fci_drb_eb_state,
)

ELECTRON_FORCE_TERM_NAMES = (
    "parallel_self_advection", "collision", "electrostatic",
    "electron_pressure", "thermal_force", "characteristic_leg_upwind",
    "vorticity_current_flux_divergence",
)
ELECTRON_FORCE_LEG_TERM_NAMES = (
    "parallel_self_advection", "electrostatic", "electron_pressure",
    "thermal_force", "characteristic_leg_upwind",
)


def historical_imex_ssp222_stage(
    current,
    model,
    source_stages,
    dt,
    *,
    implicit_stage,
    explicit_operator,
    reconstruct_phi,
    gamma=None,
):
    """Run the historical SSP222 split with explicit phi reconstruction.

    This small orchestration helper deliberately contains no physics.  The
    callbacks supply the selected-wall implicit material solve, explicit RHS,
    and algebraic polarization reconstruction.  Keeping this sequence
    reusable makes the legacy split available to eager diagnostics while the
    production driver retains its existing default behavior.
    """
    if gamma is None:
        gamma = 1.0 - 1.0 / np.sqrt(2.0)
    gamma_dt = jnp.asarray(gamma, dtype=jnp.float64) * jnp.asarray(dt, dtype=jnp.float64)

    def source_at(index):
        return source_stages.replace(**{
            name: getattr(source_stages, name)[index]
            for name in source_stages.field_names()
        })

    stage_1, implicit_1, phi_info_1 = implicit_stage(
        current, model, gamma_dt, dt
    )
    explicit_1 = explicit_operator(stage_1, stage_1.phi, model, source_at(0))
    stage_2_base = current.axpy(explicit_1, scale=dt).axpy(
        implicit_1, scale=(1.0 - 2.0 * gamma) * dt
    )
    stage_2_base_phi, phi_info_2_base = reconstruct_phi(stage_2_base, model)
    stage_2_base = stage_2_base.replace(phi=stage_2_base_phi)
    stage_2, implicit_2, phi_info_2 = implicit_stage(
        stage_2_base, model, gamma_dt, dt
    )
    explicit_2 = explicit_operator(stage_2, stage_2.phi, model, source_at(1))
    weighted_rate = explicit_1.axpy(explicit_2, scale=1.0).axpy(
        implicit_1, scale=1.0
    ).axpy(implicit_2, scale=1.0).map_fields(lambda value: 0.5 * value)
    next_state = current.axpy(weighted_rate, scale=dt)
    next_phi, phi_info_next = reconstruct_phi(next_state, model)
    next_state = next_state.replace(phi=next_phi)
    return next_state, (
        current, stage_1, stage_2_base, stage_2, next_state
    ), (
        implicit_1, explicit_1, implicit_2, explicit_2, weighted_rate
    ), (phi_info_1, phi_info_2_base, phi_info_2, phi_info_next)
ELECTRON_FORCE_GRADIENT_NAMES = ("Ve", "phi", "Pe", "Te")
ELECTRON_FORCE_ENDPOINT_FIELD_NAMES = (
    "density", "Te", "Ti", "Vi", "Ve", "phi", "Pe",
)
ELECTRON_FORCE_STENCIL_DIRECTION_NAMES = ("backward", "center", "forward")
ELECTRON_FORCE_ENDPOINT_DIRECTION_NAMES = ("backward", "forward")
ELECTRON_FORCE_CHARACTERISTIC_PRINCIPAL_NAMES = (
    "centered_principal", "upwind_principal",
)
ELECTRON_FORCE_CHARACTERISTIC_PRIMITIVE_FIELD_NAMES = (
    "density", "Te", "Ti", "Vi", "Ve",
)
from drbx.native.fci_operators import (  # noqa: E402
    build_local_perp_laplacian_face_projectors,
    expand_local_control_volume_owner_field,
    local_curvature_conservative_components_op,
)
GMRES_TARGET_TOLERANCE = 1.0e-8


def _vi_near_band_report(
    vi_terms: np.ndarray,
    vi_state: np.ndarray,
    near_start: int,
) -> dict[str, object]:
    """Return unnormalized-RFFT near-band energies and state inner products."""

    terms = np.asarray(vi_terms, dtype=np.float64)
    state = np.asarray(vi_state, dtype=np.float64)
    if terms.ndim != 4 or state.shape != terms.shape[1:]:
        raise ValueError("Vi terms must be (term, radial, theta, eta) and match state")
    term_spectrum = np.fft.rfft(terms, axis=2)[:, :, near_start:, :]
    state_spectrum = np.fft.rfft(state, axis=1)[:, near_start:, :]
    term_energy = np.sum(np.abs(term_spectrum) ** 2, axis=(1, 2, 3))
    term_inner = np.sum(
        term_spectrum * np.conj(state_spectrum)[None], axis=(1, 2, 3)
    )
    sum_spectrum = np.sum(term_spectrum, axis=0)
    sum_energy = np.sum(np.abs(sum_spectrum) ** 2)
    sum_inner = np.sum(sum_spectrum * np.conj(state_spectrum))

    def pair(value):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}

    return {
        "rfft_normalization": "numpy-unnormalized",
        "term_near_band_energy": [float(value) for value in term_energy],
        "term_near_band_inner_product_with_saved_Vi": [
            pair(value) for value in term_inner
        ],
        "sum_term_near_band_energy": float(sum_energy),
        "sum_term_near_band_inner_product_with_saved_Vi": pair(sum_inner),
    }

# This is the map schema consumed by runtime lowering.  Physical map
# production and qualification live in the explicit geometry artifact.
FCI_MAP_FIELDS = (
    "forward_x",
    "forward_y",
    "backward_x",
    "backward_y",
    "forward_endpoint_x",
    "forward_endpoint_y",
    "forward_endpoint_z",
    "backward_endpoint_x",
    "backward_endpoint_y",
    "backward_endpoint_z",
    "forward_endpoint_b_contra_x",
    "forward_endpoint_b_contra_y",
    "forward_endpoint_b_contra_z",
    "forward_endpoint_bmag",
    "backward_endpoint_b_contra_x",
    "backward_endpoint_b_contra_y",
    "backward_endpoint_b_contra_z",
    "backward_endpoint_bmag",
    "forward_length",
    "backward_length",
    "forward_boundary",
    "backward_boundary",
)


@dataclass(frozen=True)
class TopologyDescriptor:
    """Logical-coordinate contract shared by geometry and runtime metadata."""

    name: str
    coordinate_names: tuple[str, str, str]
    periodic_axes: tuple[bool, bool, bool]
    axis_regular_axes: tuple[bool, bool, bool]
    logical_extents: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def topology_descriptor(topology: str) -> TopologyDescriptor:
    selected = str(topology).lower()
    if selected == "square":
        return TopologyDescriptor(
            name="square",
            coordinate_names=("u", "v", "eta"),
            periodic_axes=(False, False, True),
            axis_regular_axes=(False, False, False),
            logical_extents=((0.0, 1.0), (0.0, 1.0), (0.0, 2.0 * np.pi)),
        )
    if selected == "toroidal":
        return TopologyDescriptor(
            name="toroidal",
            coordinate_names=("u", "theta", "eta"),
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
            logical_extents=((0.0, 1.0), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)),
        )
    raise ValueError("topology must be 'square' or 'toroidal'")


def load_fci_simulation_geometry(path: Path):
    """Load a producer-owned FCI simulation geometry artifact.

    This small forwarding function is intentionally the only geometry entry
    point used by the simulation CLI.  In particular, it does not inspect
    MAKEGRID/vessel inputs, consult metric caches, or regenerate missing
    payloads; those are producer responsibilities.
    """

    from drbx.geometry import fci_simulation_geometry

    return fci_simulation_geometry.load_fci_simulation_geometry(Path(path))


def _artifact_topology_descriptor(topology) -> TopologyDescriptor:
    """Normalize an artifact topology record without rebuilding geometry."""

    if isinstance(topology, str):
        return topology_descriptor(topology)
    if isinstance(topology, Mapping):
        name = topology.get("name", topology.get("topology"))
        if name is not None:
            return topology_descriptor(str(name))
    name = getattr(topology, "name", None)
    if name is not None:
        try:
            return topology_descriptor(str(name))
        except ValueError:
            pass
    if all(
        hasattr(topology, field)
        for field in (
            "name", "coordinate_names", "periodic_axes", "axis_regular_axes",
            "logical_extents",
        )
    ):
        return TopologyDescriptor(
            name=str(topology.name),
            coordinate_names=tuple(topology.coordinate_names),
            periodic_axes=tuple(bool(v) for v in topology.periodic_axes),
            axis_regular_axes=tuple(bool(v) for v in topology.axis_regular_axes),
            logical_extents=tuple(tuple(float(x) for x in pair) for pair in topology.logical_extents),
        )
    raise ValueError("simulation geometry artifact has no valid topology record")


_SQUARE_TOPOLOGY = topology_descriptor("square")
# Backward-compatible symbols used by the existing square runtime path.
PERIODIC_AXES = _SQUARE_TOPOLOGY.periodic_axes
AXIS_REGULAR_AXES = _SQUARE_TOPOLOGY.axis_regular_axes


def _aggregate_initial_owner_state(
    state: FciDrbEBState,
    host_geometry,
) -> FciDrbEBState:
    """Volume-average raw cells into owners and zero all aliases."""

    topology = host_geometry.topology
    raw_volume = np.asarray(host_geometry.raw_volume, dtype=np.float64)
    aggregate_volume = np.asarray(host_geometry.aggregate_chart_volume, dtype=np.float64)
    owner_mask = np.asarray(topology.is_active_owner, dtype=bool)
    result = {}
    aggregate_ids = np.asarray(topology.aggregate_id, dtype=np.int64).ravel()
    owner_flat_ids = np.flatnonzero(owner_mask.ravel())
    aggregate_volume_flat = aggregate_volume.ravel()
    for name, value in state.field_items():
        raw = np.asarray(value, dtype=np.float64)
        weighted = raw * raw_volume
        owner_sums = np.zeros(raw.size, dtype=np.float64)
        np.add.at(owner_sums, aggregate_ids, weighted.ravel())
        averaged_flat = np.zeros(raw.size, dtype=np.float64)
        averaged_flat[owner_flat_ids] = owner_sums[owner_flat_ids] / np.maximum(
            aggregate_volume_flat[owner_flat_ids], np.finfo(float).tiny
        )
        averaged = averaged_flat.reshape(raw.shape)
        # The explicit mask keeps source slots exactly zero in the runtime
        # state.  All production fields, including Vi and Ve, use the
        # canonical cell-owner basis.
        result[name] = np.where(owner_mask, averaged, 0.0)
    return FciDrbEBState(**result)


def _restore_materialized_cell_owner_state(
    state: FciDrbEBState,
    host_geometry,
) -> tuple[FciDrbEBState, bool]:
    """Invert checkpoint owner materialization exactly when it validates.

    Output checkpoints prolong every canonical cell-owner value to all of its
    member cells.  When every saved member still equals that owner, retaining
    only active-owner slots is the exact inverse and avoids recomputing a
    volume average.  Noncanonical/legacy inputs are returned unchanged so the
    caller can use the general aggregation path.
    """

    topology = host_geometry.topology
    owner_index = np.asarray(topology.owner_index, dtype=np.int32)
    owner_coordinates = tuple(np.moveaxis(owner_index, -1, 0))
    owner_mask = np.asarray(topology.is_active_owner, dtype=bool)
    restored: dict[str, np.ndarray] = {}
    for name, value in state.field_items():
        materialized = np.asarray(value, dtype=np.float64)
        owner_values = materialized[owner_coordinates]
        if not np.array_equal(materialized, owner_values, equal_nan=True):
            return state, False
        restored[name] = np.where(owner_mask, materialized, 0.0)
    return FciDrbEBState(**restored), True


def _materialize_owner_state(state: FciDrbEBState, host_geometry) -> FciDrbEBState:
    """Prolong owner values to the fine grid at output boundaries."""

    topology = host_geometry.topology
    owner_index = np.asarray(topology.owner_index, dtype=np.int32)
    result = {}
    for name, value in state.field_items():
        array = np.asarray(value, dtype=np.float64)
        result[name] = array[tuple(np.moveaxis(owner_index, -1, 0))]
    return FciDrbEBState(**result)


def _materialize_owner_array(array: np.ndarray, host_geometry) -> np.ndarray:
    """Expand a leading-term-axis cell-owner diagnostic array."""

    value = np.asarray(array, dtype=np.float64)
    if host_geometry is None or value.ndim != 4:
        return value
    owner_index = np.asarray(host_geometry.topology.owner_index, dtype=np.int32)
    return value[(slice(None),) + tuple(np.moveaxis(owner_index, -1, 0))]

def _assert_owner_sparse(state: FciDrbEBState, host_geometry) -> None:
    cell_mask = ~np.asarray(host_geometry.topology.is_active_owner, dtype=bool)
    maximum = max(
        float(np.max(np.abs(np.asarray(value)[mask])) if np.any(mask) else 0.0)
        for name, value in state.field_items()
        for mask in (cell_mask,)
    )
    if maximum > 1.0e-12:
        raise FloatingPointError(
            f"RLP owner-sparse invariant violated: alias magnitude={maximum:.3e}"
        )


def _blend_rung3_fci_wall_layer(
    vi_owned: jax.Array,
    ve_owned: jax.Array,
    backward_target: jax.Array,
    forward_target: jax.Array,
    selected_backward: jax.Array,
    selected_forward: jax.Array,
    backward_alpha: jax.Array,
    forward_alpha: jax.Array,
    dx_minus: jax.Array,
    dx_plus: jax.Array,
    active_owner_mask: jax.Array,
    layer_count: int,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Blend owner values toward all selected FCI wall endpoint targets.

    Physical FCI endpoints are row-local and can occur away from the
    coordinate upper-x face.  A double-hit owner has two directional targets;
    when they disagree, the startup owner value is the stiffness-weighted
    compromise while the production flux retains both directional targets.
    Influence is extended only inward in logical ``u`` and never across eta
    shards or inactive RLP aliases.
    """
    values = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            vi_owned, ve_owned, backward_target, forward_target,
            backward_alpha, forward_alpha, dx_minus, dx_plus,
        )
    )
    vi_owned, ve_owned, backward_target, forward_target, backward_alpha, forward_alpha, dx_minus, dx_plus = values
    selected_backward = jnp.asarray(selected_backward, dtype=bool)
    selected_forward = jnp.asarray(selected_forward, dtype=bool)
    active_owner_mask = jnp.asarray(active_owner_mask, dtype=bool)
    if vi_owned.ndim != 3 or ve_owned.shape != vi_owned.shape:
        raise ValueError("Vi and Ve owner fields must have identical 3-D shapes")
    if backward_target.shape != vi_owned.shape + (2,) or forward_target.shape != vi_owned.shape + (2,):
        raise ValueError("wall targets must have owner shape plus final (Vi, Ve) components")
    for name, value in (
        ("selected_backward", selected_backward), ("selected_forward", selected_forward),
        ("backward_alpha", backward_alpha), ("forward_alpha", forward_alpha),
        ("dx_minus", dx_minus), ("dx_plus", dx_plus),
        ("active_owner_mask", active_owner_mask),
    ):
        if value.shape != vi_owned.shape:
            raise ValueError(f"{name} must have owner shape {vi_owned.shape}, got {value.shape}")
    if int(layer_count) < 1:
        raise ValueError("layer_count must be positive")

    selected_backward &= active_owner_mask
    selected_forward &= active_owner_mask
    # Targets are passed as (..., 2), one component for Vi and Ve.
    if backward_target.shape[-1:] != (2,) or forward_target.shape[-1:] != (2,):
        raise ValueError("wall targets must have final dimension (Vi, Ve)")
    finite_backward = jnp.all(jnp.isfinite(backward_target), axis=-1)
    finite_forward = jnp.all(jnp.isfinite(forward_target), axis=-1)
    valid_backward = selected_backward & finite_backward
    valid_forward = selected_forward & finite_forward
    backward_weight = jnp.abs(backward_alpha) / jnp.maximum(jnp.abs(dx_minus), 1.0e-30)
    forward_weight = jnp.abs(forward_alpha) / jnp.maximum(jnp.abs(dx_plus), 1.0e-30)
    weight_sum = jnp.maximum(backward_weight + forward_weight, 1.0e-30)
    double_target = (
        backward_weight[..., None] * backward_target
        + forward_weight[..., None] * forward_target
    ) / weight_sum[..., None]
    single_target = jnp.where(valid_backward[..., None], backward_target, vi_owned[..., None])
    single_target = jnp.where(valid_forward[..., None], forward_target, single_target)
    both_target = jnp.where(valid_backward[..., None] & valid_forward[..., None], double_target, single_target)
    seed_mask = valid_backward | valid_forward
    conflict = valid_backward & valid_forward & (
        jnp.max(jnp.abs(backward_target - forward_target), axis=-1) > 1.0e-12
    )
    nearest_distance = jnp.full(vi_owned.shape, int(layer_count), dtype=jnp.int32)
    nearest_target = jnp.broadcast_to(vi_owned[..., None], vi_owned.shape + (2,))
    for distance in range(int(layer_count)):
        if distance == 0:
            candidate_mask = seed_mask
            candidate_target = both_target
        else:
            candidate_mask = jnp.concatenate(
                (seed_mask[distance:], jnp.zeros((distance,) + seed_mask.shape[1:], dtype=bool)),
                axis=0,
            )
            candidate_target = jnp.concatenate(
                (both_target[distance:], jnp.broadcast_to(vi_owned[-1:, ..., None], (distance,) + vi_owned.shape[1:] + (2,))),
                axis=0,
            )
        better = candidate_mask & (nearest_distance == int(layer_count))
        nearest_distance = jnp.where(better, distance, nearest_distance)
        nearest_target = jnp.where(better[..., None], candidate_target, nearest_target)
    depth = nearest_distance.astype(jnp.float64)
    normalized = jnp.clip((float(layer_count) - depth) / float(layer_count), 0.0, 1.0)
    smooth_weight = normalized * normalized * (3.0 - 2.0 * normalized)
    blend_weight = jnp.where(
        (nearest_distance < int(layer_count)) & active_owner_mask,
        smooth_weight,
        0.0,
    )
    vi = vi_owned + blend_weight * (nearest_target[..., 0] - vi_owned)
    ve = ve_owned + blend_weight * (nearest_target[..., 1] - ve_owned)
    diagnostics = {
        "selected_backward": selected_backward,
        "selected_forward": selected_forward,
        "double_hit": valid_backward & valid_forward,
        "double_hit_conflict": conflict,
        "target_mismatch": jnp.max(jnp.abs(backward_target - forward_target), axis=-1),
        "modified_owner": blend_weight > 0.0,
    }
    return vi, ve, diagnostics


def _initialize_rung3_wall_layer_local(
    model: LocalFciDrbEBRhs,
    local_state: FciDrbEBState,
    *,
    timestep: float,
    layer_count: int,
) -> tuple[FciDrbEBState, jax.Array]:
    """Execute one local Rung-3 FCI wall-layer initialization pass."""
    local_state = model._owner_state(local_state)
    face_bc = model._face_bcs(local_state)
    active_owner_mask = (
        model.control_volume_geometry.cells.is_active_owner
        if model.control_volume_geometry is not None
        else model.geometry.active_cell_mask_owned
    )
    state_halo = model._prepare_state_halo(local_state, face_bc)
    operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
        state_halo,
        model.geometry,
        model.domain,
        face_bc,
        tau=model.parameters.tau,
    )
    parallel_boundary = model._parallel_operator_boundary(
        state_halo=state_halo,
        operator_boundary=operator_boundary,
    )
    characteristic_data = model._fci_parallel_characteristic_wall_data(
        state_halo=state_halo,
        face_bc=face_bc,
        parallel_boundary=parallel_boundary,
        context=model._stencil_builder_context(),
        short_leg_selection_dt=timestep,
        evaluate_wall_data=True,
    )
    wall_data = characteristic_data["wall_data"]
    if wall_data is None:
        raise RuntimeError("Rung-3 initializer requires live wall data")
    backward_target = jnp.stack(
        (wall_data["backward_endpoint_state"][..., 3],
         wall_data["backward_endpoint_state"][..., 4]), axis=-1
    )
    forward_target = jnp.stack(
        (wall_data["forward_endpoint_state"][..., 3],
         wall_data["forward_endpoint_state"][..., 4]), axis=-1
    )
    primitive_stencils = characteristic_data["primitive_stencils"]
    vi, ve, layer_diagnostics = _blend_rung3_fci_wall_layer(
        local_state.Vi,
        local_state.Ve,
        backward_target,
        forward_target,
        wall_data["selected_backward_wall"],
        wall_data["selected_forward_wall"],
        wall_data["backward_alpha"],
        wall_data["forward_alpha"],
        primitive_stencils[0].dx_min,
        primitive_stencils[0].dx_plus,
        active_owner_mask,
        layer_count,
    )
    adjusted = model._owner_state(local_state.replace(Vi=vi, Ve=ve))

    def spmd_reduce(value, operation):
        result = jnp.asarray(value)
        for axis_name in model.domain.mesh_axis_names:
            if axis_name is None:
                continue
            result = (
                jax.lax.psum(result, axis_name=axis_name)
                if operation == "sum"
                else jax.lax.pmax(result, axis_name=axis_name)
            )
        return result

    diagnostic_vector = jnp.asarray(
        (
            spmd_reduce(jnp.sum(layer_diagnostics["selected_backward"]), "sum"),
            spmd_reduce(jnp.sum(layer_diagnostics["selected_forward"]), "sum"),
            spmd_reduce(jnp.sum(layer_diagnostics["double_hit"]), "sum"),
            spmd_reduce(jnp.sum(layer_diagnostics["double_hit_conflict"]), "sum"),
            spmd_reduce(jnp.max(layer_diagnostics["target_mismatch"]), "max"),
            spmd_reduce(jnp.sum(layer_diagnostics["modified_owner"]), "sum"),
        ),
        dtype=jnp.float64,
    )
    compatible, compatibility_diagnostics = initialize_boundary_compatible_rung3(
        model,
        adjusted,
        layer_count=layer_count,
    )
    return compatible, jnp.concatenate(
        (diagnostic_vector, compatibility_diagnostics), axis=0
    )


def build_face_bc_bundle(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    parameters: FciDrbEBRhsParameters,
    *,
    parallel_velocity_wall_bc: str = "neumann",
    physical_wall_model: str = "legacy-velocity-trace",
    conducting_sheath_wall_potential: float | None = None,
    topology_halos: dict[str, jax.Array] | None = None,
) -> LocalFciDrbEBPhysicalWallBundle:
    """Build one stage-local physical wall bundle.

    ``parallel_velocity_wall_bc`` is retained only for the historical
    ``legacy-velocity-trace`` adapter. New wall rungs select a named physical
    model and own the complete bundle.
    """

    model = physical_wall_model_from_name(
        physical_wall_model,
        legacy_parallel_velocity_wall_bc=parallel_velocity_wall_bc,
        conducting_sheath_wall_potential=conducting_sheath_wall_potential,
    )
    if physical_wall_model == "simplified-gbs-mpe":
        return model(
            state, geometry, domain, parameters, topology_halos=topology_halos
        )
    return model(state, geometry, domain, parameters)


def build_initial_state(
    geometry: FciGeometry3D,
    *,
    initialization: str,
    density_amplitude: float,
    temperature_amplitude: float,
    blob_center: tuple[float, float],
    blob_width: float,
    toroidal_perturbation_amplitude: float = 0.0,
    toroidal_perturbation_mode: int = 1,
    toroidal_perturbation_phase: float = 0.0,
    periodic_axes: tuple[bool, bool, bool] = PERIODIC_AXES,
    axis_regular_axes: tuple[bool, bool, bool] = AXIS_REGULAR_AXES,
) -> FciDrbEBState:
    if initialization == "logical":
        u = jnp.asarray(
            geometry.grid.x.centers,
            dtype=jnp.float64,
        )[:, None, None]
        v = jnp.asarray(
            geometry.grid.y.centers,
            dtype=jnp.float64,
        )[None, :, None]
        if axis_regular_axes[0]:
            distance_squared = (
                u**2
                + float(blob_center[0]) ** 2
                - 2.0
                * u
                * float(blob_center[0])
                * jnp.cos(v - float(blob_center[1]))
            )
        else:
            distance_squared = (
                (u - float(blob_center[0])) ** 2
                + (v - float(blob_center[1])) ** 2
            )
        profile_2d = jnp.exp(
            -distance_squared / (2.0 * float(blob_width) ** 2)
        )
        eta = jnp.asarray(
            geometry.grid.z.centers,
            dtype=jnp.float64,
        )[None, None, :]
        toroidal_modulation = 1.0 + float(
            toroidal_perturbation_amplitude
        ) * jnp.cos(
            int(toroidal_perturbation_mode) * eta
            + float(toroidal_perturbation_phase)
        )
        profile = jnp.broadcast_to(
            profile_2d * toroidal_modulation,
            geometry.shape,
        )
    else:
        raise ValueError(f"unknown initialization mode {initialization!r}")

    zeros = jnp.zeros(geometry.shape, dtype=jnp.float64)
    electron_temperature = 1.0 + float(temperature_amplitude) * profile
    return FciDrbEBState(
        density=1.0 + float(density_amplitude) * profile,
        phi=zeros,
        Te=electron_temperature,
        Ti=jnp.ones(geometry.shape, dtype=jnp.float64),
        Vi=zeros,
        Ve=zeros,
        vorticity=zeros,
    )


def build_local_eb_model(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    parameters: FciDrbEBRhsParameters,
    *,
    gmres_target_tolerance: float,
    gmres_acceptance_tolerance: float,
    gmres_max_iterations: int,
    gmres_restart: int = 100,
    gmres_preconditioner: str = "none",
    gmres_residual_correction_steps: int = 0,
    neumann_ghost_scheme: str = "physical",
    parallel_velocity_wall_bc: str = "neumann",
    physical_wall_model: str = "legacy-velocity-trace",
    conducting_sheath_wall_potential: float | None = None,
    parallel_operator_scheme: str = "coordinate",
    poisson_bracket_scheme: str = "direct",
    polarization_operator_form: str = "conservative",
    polarization_coarse_data=None,
    parallel_material_scheme: str | None = None,
    parallel_vorticity_advection_scheme: str | None = None,
    parallel_material_fallback_representation: str | None = None,
    parallel_material_div_b_fallback_scheme: str | None = None,
    control_volume_geometry=None,
    control_volume_boundary_bc=None,
    curvature_face_coefficients_override: LocalCurvatureFaceCoefficients3D | None = None,
) -> LocalFciDrbEBRhs:
    if gmres_restart < 1:
        raise ValueError("gmres_restart must be positive")
    if gmres_residual_correction_steps < 0:
        raise ValueError("gmres_residual_correction_steps must be non-negative")
    if parallel_operator_scheme not in ("coordinate", "fci"):
        raise ValueError(
            "parallel_operator_scheme must be 'coordinate' or 'fci', got "
            f"{parallel_operator_scheme!r}"
        )
    if neumann_ghost_scheme not in ("logical", "physical"):
        raise ValueError(
            "neumann_ghost_scheme must be 'logical' or 'physical', got "
            f"{neumann_ghost_scheme!r}"
        )
    if parallel_velocity_wall_bc not in (
        "dirichlet-zero",
        "neumann",
        "bohm",
    ):
        raise ValueError(
            "parallel_velocity_wall_bc must be 'dirichlet-zero', "
            f"'neumann', or 'bohm', got {parallel_velocity_wall_bc!r}"
        )
    if physical_wall_model not in PHYSICAL_WALL_MODEL_NAMES:
        raise ValueError(
            f"physical_wall_model must be one of {PHYSICAL_WALL_MODEL_NAMES}, "
            f"got {physical_wall_model!r}"
        )
    if (
        physical_wall_model != "legacy-velocity-trace"
        and parameters.parallel_characteristic_wall_law != "physical-boundary-state"
    ):
        raise ValueError(
            "named physical wall models require "
            "parallel_characteristic_wall_law='physical-boundary-state'"
        )
    if (
        physical_wall_model == "legacy-velocity-trace"
        and parameters.parallel_characteristic_wall_law == "physical-boundary-state"
    ):
        raise ValueError(
            "parallel_characteristic_wall_law='physical-boundary-state' "
            "requires a named physical wall model"
        )
    if poisson_bracket_scheme not in (
        "direct",
        "compatible-flux",
        "compatible-third-order-upwind",
        "material-scalar-third-order-upwind",
        "material-scalar-vorticity-compatible-upwind",
    ):
        raise ValueError(
            "poisson_bracket_scheme must be 'direct', 'compatible-flux', or "
            "'compatible-third-order-upwind', or "
            "'material-scalar-third-order-upwind', or "
            "'material-scalar-vorticity-compatible-upwind', "
            f"got {poisson_bracket_scheme!r}"
        )
    if polarization_operator_form not in (
        "conservative",
        "weighted-symmetric",
        "support-paired",
    ):
        raise ValueError(
            "polarization_operator_form must be 'conservative', "
            "'weighted-symmetric', or 'support-paired', got "
            f"{polarization_operator_form!r}"
        )
    if parallel_material_scheme is None:
        parallel_material_scheme = os.environ.get(
            "DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"
        )
    if parallel_vorticity_advection_scheme is None:
        parallel_vorticity_advection_scheme = os.environ.get(
            "DRBX_PARALLEL_VORTICITY_ADVECTION_SCHEME", "first-order"
        )
    if parallel_material_fallback_representation is None:
        parallel_material_fallback_representation = os.environ.get(
            "DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION", "legacy-p"
        )
    if parallel_material_div_b_fallback_scheme is None:
        parallel_material_div_b_fallback_scheme = os.environ.get(
            "DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME", "legacy-p"
        )
    halo_exchange = HaloExchange3D()
    topology_filler = (
        make_default_topology_halo_filler_3d(
            angle_axis_name=domain.mesh_axis_names[1],
            radial_axis_lower_regular=True,
            radial_axis_upper_regular=False,
            fill_periodic_axes=domain.periodic_axes,
        )
        if domain.axis_regular_axes[0]
        else TopologyHaloFiller3D(
            rules=(
                LocalPeriodicTopologyRule3D(
                    fill_axes=domain.periodic_axes,
                ),
            )
        )
    )
    halo_width = geometry.layout.halo_width
    def paired_neumann_weights(axis: int, side: str) -> GhostFillWeights1D:
        """Logical-normal Neumann weights paired layer-by-layer with owners."""
        grid = (geometry.grid.x, geometry.grid.y, geometry.grid.z)[axis]
        centers = jnp.asarray(grid.centers_halo, dtype=jnp.float64)
        h = int(halo_width)
        n = int(geometry.owned_shape[axis])
        if side == "lower":
            owner = centers[h : h + h]
            ghost = centers[h - 1 :: -1][:h]
        else:
            owner = centers[h + n - h : h + n][::-1]
            ghost = centers[h + n : h + n + h]
        displacement = ghost - owner
        return GhostFillWeights1D(
            owned_weights=jnp.eye(h, dtype=jnp.float64),
            bc_weights=displacement,
        )

    dirichlet_ghost_weights = GhostFillWeights1D(
        owned_weights=-jnp.eye(halo_width, dtype=jnp.float64),
        bc_weights=jnp.full(
            (halo_width,),
            2.0,
            dtype=jnp.float64,
        ),
    )
    neumann_lower_weights = tuple(
        paired_neumann_weights(axis, "lower") for axis in range(3)
    )
    neumann_upper_weights = tuple(
        paired_neumann_weights(axis, "upper") for axis in range(3)
    )
    ghost_filler_kwargs = dict(
        dirichlet=(
            dirichlet_ghost_weights,
            dirichlet_ghost_weights,
            dirichlet_ghost_weights,
        ),
        neumann_lower=(
            *neumann_lower_weights,
        ),
        neumann_upper=(
            *neumann_upper_weights,
        ),
    )
    physical_ghost_filler = (
        MetricAwarePhysicalGhostCellFiller3D(
            **ghost_filler_kwargs,
            geometry=geometry,
        )
        if neumann_ghost_scheme == "physical"
        else PhysicalGhostCellFiller3D(**ghost_filler_kwargs)
    )
    curvature_face_coefficients = (
        curvature_face_coefficients_override
        if curvature_face_coefficients_override is not None
        else build_local_curvature_face_coefficients(geometry, domain)
    )
    def face_bc_builder(
        state, local_geometry, local_domain, local_parameters, *, topology_halos=None
    ):
        return build_face_bc_bundle(
            state,
            local_geometry,
            local_domain,
            local_parameters,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            physical_wall_model=physical_wall_model,
            conducting_sheath_wall_potential=conducting_sheath_wall_potential,
            topology_halos=topology_halos,
        )

    rhs_kwargs = dict(
        geometry=geometry,
        domain=domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        parameters=parameters,
        face_projectors=build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            axis_regular_axes=domain.axis_regular_axes,
        ),
        gmres_config=SolvaxGmresConfig(
            tol=float(gmres_target_tolerance),
            atol=float(gmres_target_tolerance),
            maxiter=int(gmres_max_iterations),
            restart=min(int(gmres_restart), int(gmres_max_iterations)),
            acceptance_tol=float(gmres_acceptance_tolerance),
            acceptance_atol=float(gmres_acceptance_tolerance),
            project_mean_zero=False,
            regularization_epsilon=float(
                parameters.phi_inversion_regularization
            ),
            preconditioner=str(gmres_preconditioner),
            residual_correction_steps=int(gmres_residual_correction_steps),
        ),
        parallel_operator_scheme=str(parallel_operator_scheme),
        parallel_material_scheme=str(parallel_material_scheme),
        parallel_vorticity_advection_scheme=str(
            parallel_vorticity_advection_scheme
        ),
        parallel_material_fallback_representation=str(
            parallel_material_fallback_representation
        ),
        parallel_material_div_b_fallback_scheme=str(
            parallel_material_div_b_fallback_scheme
        ),
        face_bc_builder=face_bc_builder,
        physical_wall_model_name=str(physical_wall_model),
        conducting_sheath_wall_potential=conducting_sheath_wall_potential,
        axis_regular_axes=domain.axis_regular_axes,
        curvature_face_coefficients=curvature_face_coefficients,
        poisson_bracket_scheme=poisson_bracket_scheme,
        polarization_operator_form=polarization_operator_form,
        polarization_coarse_data=polarization_coarse_data,
        control_volume_geometry=control_volume_geometry,
        control_volume_boundary_bc=control_volume_boundary_bc,
    )
    model = LocalFciDrbEBRhs(
        **rhs_kwargs,
    )
    return model


class _JittedPhaseTimer:
    """Collect ordered host timestamps emitted by one compiled advance."""

    def __init__(self, *, expected_markers: int = 8, label: str = "RK4") -> None:
        self._expected_markers = int(expected_markers)
        self._label = str(label)
        self._lock = threading.Lock()
        self._step_start: float | None = None
        self._last_marker: float | None = None
        self._operator_seconds = 0.0
        self._gmres_seconds = 0.0
        self._marker_count = 0

    def begin_step(self) -> None:
        now = time.perf_counter()
        with self._lock:
            self._step_start = now
            self._last_marker = now
            self._operator_seconds = 0.0
            self._gmres_seconds = 0.0
            self._marker_count = 0

    def mark_operator(self, *_dependencies: object) -> None:
        self._mark("operator")

    def mark_gmres(self, *_dependencies: object) -> None:
        self._mark("gmres")

    def _mark(self, phase: str) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._last_marker is None:
                return
            elapsed = now - self._last_marker
            if phase == "operator":
                self._operator_seconds += elapsed
            else:
                self._gmres_seconds += elapsed
            self._last_marker = now
            self._marker_count += 1

    def finish_step(self) -> tuple[float, float]:
        with self._lock:
            if self._step_start is None:
                raise RuntimeError("phase timer was not started")
            if self._marker_count != self._expected_markers:
                raise RuntimeError(
                    f"compiled {self._label} timing markers were incomplete: "
                    f"expected {self._expected_markers}, got {self._marker_count}"
                )
            return self._operator_seconds, self._gmres_seconds


def _state_marker_dependencies(state: FciDrbEBState) -> tuple[jax.Array, ...]:
    """Return scalar dependencies that make a timing marker await every field."""

    return tuple(jnp.ravel(value)[0] for _, value in state.field_items())


IMEX_SSP222_GAMMA = 1.0 - 1.0 / np.sqrt(2.0)


def _tree_axpy(left, right, scale):
    """Return ``left + scale*right`` for an array or matching PyTree."""

    return jax.tree_util.tree_map(
        lambda x, y: x + jnp.asarray(scale, dtype=jnp.float64) * y,
        left,
        right,
    )


def _imex_ssp222_step(current, dt, explicit_rhs, implicit_stage):
    """Advance one additive IMEX-SSP2(2,2,2) step.

    ``implicit_stage(base, stage_dt)`` returns both the solved stage and the
    implicit rate represented by its increment.  Keeping the rate explicit
    avoids differencing the diagnostic/algebraic ``phi`` leaf.  The method is
    the two-stage L-stable SDIRK/SSP explicit pair with
    ``gamma = 1 - 1/sqrt(2)``.
    """

    dt = jnp.asarray(dt, dtype=jnp.float64)
    gamma_dt = jnp.asarray(IMEX_SSP222_GAMMA, dtype=jnp.float64) * dt

    stage_1, implicit_1 = implicit_stage(current, gamma_dt)
    explicit_1 = explicit_rhs(stage_1)

    stage_2_base = _tree_axpy(current, explicit_1, dt)
    stage_2_base = _tree_axpy(
        stage_2_base,
        implicit_1,
        (1.0 - 2.0 * IMEX_SSP222_GAMMA) * dt,
    )
    stage_2, implicit_2 = implicit_stage(stage_2_base, gamma_dt)
    explicit_2 = explicit_rhs(stage_2)

    weighted_rate = jax.tree_util.tree_map(
        lambda e1, e2, i1, i2: 0.5 * (e1 + e2 + i1 + i2),
        explicit_1,
        explicit_2,
        implicit_1,
        implicit_2,
    )
    next_state = _tree_axpy(current, weighted_rate, dt)
    return (
        next_state,
        (stage_1, stage_2_base, stage_2),
        (implicit_1, explicit_1, implicit_2, explicit_2, weighted_rate),
    )


def _explicit_source_stage_times(
    time_integrator: str, start_time: float, timestep: float
) -> tuple[float, ...]:
    """Return source times in the same order as the compiled advance stages."""

    t = float(start_time)
    dt = float(timestep)
    if time_integrator == "rk4":
        return (t, t + 0.5 * dt, t + 0.5 * dt, t + dt)
    if time_integrator == "imex-ssp222":
        return (t, t + dt)
    raise ValueError(f"unsupported time integrator {time_integrator!r}")


def _resolve_execution_mode(
    requested: str,
    *,
    work_items: int,
    auto_short_mode: str = "eager",
) -> str:
    """Resolve the requested eager, staged, or monolithic JIT mode."""

    if requested not in ("auto", "compiled", "staged-compiled", "eager"):
        raise ValueError(
            "execution mode must be 'auto', 'compiled', 'staged-compiled', "
            "or 'eager', got "
            f"{requested!r}"
        )
    if work_items < 1:
        raise ValueError("execution mode resolution requires positive work_items")
    if auto_short_mode not in ("compiled", "staged-compiled", "eager"):
        raise ValueError(
            "auto_short_mode must be 'compiled', 'staged-compiled', or "
            f"'eager', got {auto_short_mode!r}"
        )
    if requested == "auto":
        return auto_short_mode if work_items < 100 else "compiled"
    return requested


def _pack_curvature_face_coefficients(
    coefficients: LocalCurvatureFaceCoefficients3D,
) -> np.ndarray:
    """Store each cell's lower/upper invariant face values as six channels."""

    packed = []
    for axis, values in enumerate(coefficients.axes):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        packed.extend((values[tuple(lower)], values[tuple(upper)]))
    return np.stack([np.asarray(value) for value in packed], axis=-1)


def _unpack_curvature_face_coefficients(
    packed: jax.Array,
    layout,
) -> LocalCurvatureFaceCoefficients3D:
    """Rebuild shard-local owned-face arrays from cell-aligned channels."""

    if packed.ndim != 4 or packed.shape[-1] != 6:
        raise ValueError(
            "packed curvature coefficients must have shape (nx, ny, nz, 6)"
        )

    def unpack_face(axis: int) -> jax.Array:
        lower = packed[..., 2 * axis]
        upper = packed[..., 2 * axis + 1]
        last = [slice(None)] * 3
        last[axis] = slice(-1, None)
        return jnp.concatenate((lower, upper[tuple(last)]), axis=axis)

    return LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=unpack_face(0),
        y=unpack_face(1),
        z=unpack_face(2),
    )


def _progress_line(
    *,
    step: int,
    num_steps: int,
    simulation_time: float,
    density_min: float,
    density_max: float,
    step_seconds: float,
    operator_seconds: float | None,
    gmres_seconds: float | None,
    gmres_iterations: float,
    gmres_relative_residual: float,
    elapsed_seconds: float,
    solver_label: str = "gmres-iters(avg4)",
) -> str:
    width = 24
    fraction = step / num_steps
    filled = min(width, int(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    eta_seconds = max(0.0, elapsed_seconds / step * (num_steps - step))
    phase_text = ""
    if operator_seconds is not None and gmres_seconds is not None:
        phase_text = (
            f" op={operator_seconds:.2f}s"
            f" gmres={gmres_seconds:.2f}s"
        )
    solver_text = (
        f"gmres-iters(avg4)={gmres_iterations:.2f}"
        if solver_label == "gmres-iters(avg4)"
        else f"{solver_label}={gmres_iterations:.2f}"
    )
    return (
        f"[{bar}] {step:5d}/{num_steps} "
        f"t={simulation_time:.6e} step={step_seconds:.2f}s"
        f"{phase_text} {solver_text} "
        f"gmres-relres(max4)={gmres_relative_residual:.3e} "
        f"ETA={eta_seconds:.1f}s "
        f"n=[{density_min:.6e}, {density_max:.6e}]"
    )


def _format_state_diagnostics(
    field_names: Sequence[str],
    diagnostics: np.ndarray,
) -> str:
    """Format the static compiled state diagnostics for a host-side log line."""

    parts = []
    for name, (minimum, maximum, absolute_maximum) in zip(
        field_names,
        np.asarray(diagnostics),
        strict=True,
    ):
        parts.append(
            f"{name}[{minimum:.3e},{maximum:.3e},|.|={absolute_maximum:.3e}]"
        )
    return " ".join(parts)


def _rk_stage_diagnostics_have_finite_bit(
    rk_stage_diagnostics: np.ndarray,
) -> bool:
    """Return the host-side acceptance bit from the compiled stage payload.

    Newer payloads carry an explicit finiteness slot in the last column so that
    the host no longer needs to infer safety from extrema reductions alone.  We
    keep a backward-compatible fallback for older four-slot payloads.
    """

    diagnostics = np.asarray(rk_stage_diagnostics)
    if diagnostics.ndim >= 3 and diagnostics.shape[-1] >= 5:
        return bool(np.all(diagnostics[..., 4] > 0.5))
    return bool(np.all(np.isfinite(diagnostics[..., :3])))


_PHI_DIAGNOSTIC_RHS_INCOMPATIBLE_REL = 4
_PHI_DIAGNOSTIC_FINAL_INCOMPATIBLE_REL = 5
_PHI_DIAGNOSTIC_FINAL_FULL_REL = 6
_PHI_DIAGNOSTIC_WIDTH = 19
_PHI_DIAGNOSTIC_NORM_FLOOR = 1.0e-30
_PHI_SOLVER_DIAGNOSTIC_NAMES = (
    "gmres_num_steps",
    "gmres_final_residual_rel_l2",
    "gmres_failed",
    "gmres_converged",
    "phi_lambda",
    "phi_raw_compatibility_defect",
    "phi_final_gauge_residual",
    "phi_initial_residual_l2",
    "phi_final_residual_l2",
    "phi_rhs_l2",
    "phi_projected_rhs_l2",
    "phi_solution_finite",
    "phi_rhs_finite",
    "phi_guess_finite",
    "phi_final_operator_residual_l2",
    "phi_input_rhs_finite",
    "phi_boundary_source_finite",
    "phi_input_rhs_max_abs",
    "phi_boundary_source_max_abs",
)

# These slots are replicated by ``reconstruct_stage_phi`` and are therefore
# safe to inspect between the separately compiled staged kernels.  The
# ``converged`` bit is the solver's acceptance bit (it is deliberately not
# synonymous with reaching the tighter target tolerance), while ``failed``
# is its complement.  The additional finiteness bits protect the dependent
# explicit RHS from consuming a bad phi result even when a Krylov backend
# returns a numerically finite-looking residual.
_PHI_DIAGNOSTIC_FINITE_SLOTS = (11, 12, 13, 15, 16)


def _coupled_phi_diagnostics_accepted(info: object) -> bool:
    """Return the strict acceptance predicate for coupled final phi data."""
    values = np.asarray(info, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] >= _PHI_DIAGNOSTIC_WIDTH
        and np.all(np.isfinite(values))
        and values[3] > 0.5
        and np.all(values[list(_PHI_DIAGNOSTIC_FINITE_SLOTS)] > 0.5)
        and abs(values[6]) <= 1.0e-10
    )


def _validate_staged_phi_solver_diagnostics(
    info: object,
    stage_name: str,
) -> np.ndarray:
    """Validate a staged phi solve before its result is consumed.

    Staged IMEX advances are orchestrated in Python between compiled kernels,
    so this host-side check can stop immediately after each inversion.  The
    monolithic/fused paths retain their existing end-of-step validation.
    """

    jax.block_until_ready(info)
    diagnostics = np.asarray(info, dtype=np.float64)
    if diagnostics.ndim != 1 or diagnostics.shape[0] < _PHI_DIAGNOSTIC_WIDTH:
        raise FloatingPointError(
            f"staged {stage_name} phi inversion returned malformed diagnostics: "
            f"shape={diagnostics.shape}"
        )
    finite = bool(np.all(np.isfinite(diagnostics)))
    finite_flags = bool(
        np.all(diagnostics[list(_PHI_DIAGNOSTIC_FINITE_SLOTS)] > 0.5)
    )
    failed = bool(diagnostics[2] > 0.5)
    converged = bool(diagnostics[3] > 0.5)
    if not finite or not finite_flags or failed or not converged:
        raise FloatingPointError(
            f"staged {stage_name} phi inversion rejected: "
            f"iters={diagnostics[0]:.0f}, relres={diagnostics[1]:.3e}, "
            f"failed={failed}, converged={converged}, "
            f"diagnostics_finite={finite}, payload_finite={finite_flags}"
        )
    return diagnostics


def _format_phi_solver_diagnostics(
    info: object,
) -> jax.Array:
    """Pack fixed-shape phi diagnostics while preserving the first four slots."""

    first_four = jnp.stack(
        (
            jnp.asarray(info.num_steps, dtype=jnp.float64),
            jnp.asarray(info.final_residual_rel_l2, dtype=jnp.float64),
            jnp.asarray(info.failed, dtype=jnp.float64),
            jnp.asarray(info.converged, dtype=jnp.float64),
        )
    )
    base_info = getattr(info, "base_info", info)
    return jnp.concatenate(
        (
            first_four,
            jnp.stack(
                (
                    jnp.asarray(
                        getattr(info, "compatibility_multiplier", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "raw_compatibility_defect", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "final_gauge_residual", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "initial_residual_l2", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "final_residual_l2", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "rhs_l2", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "projected_rhs_l2", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "phi_is_finite", True),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "rhs_is_finite", True),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(base_info, "guess_is_finite", True),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "final_operator_residual_l2", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "input_rhs_is_finite", True),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "boundary_source_is_finite", True),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "input_rhs_max_abs", 0.0),
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        getattr(info, "boundary_source_max_abs", 0.0),
                        dtype=jnp.float64,
                    ),
                )
            ),
        )
    )


def _print_rk_stage_diagnostics(
    field_names: Sequence[str],
    rk_stage_diagnostics: np.ndarray,
    *,
    integrator: str = "rk4",
) -> None:
    """Print complete stage-state and rate diagnostics for a failure path."""

    labels = (
        ("current/k1", "stage2/k2", "stage3/k3", "stage4/k4", "next/weighted")
        if integrator == "rk4"
        else (
            "current/implicit1",
            "imex-stage1/explicit1",
            "stage2-base/implicit2",
            "imex-stage2/explicit2",
            "next/weighted",
        )
    )
    for rk_name, rk_values in zip(
        labels,
        np.asarray(rk_stage_diagnostics),
        strict=True,
    ):
        rhs_values = rk_values[:, 3]
        dominant_rhs_index = int(
            np.argmax(np.where(np.isfinite(rhs_values), rhs_values, -np.inf))
        )
        state_abs_values = rk_values[:, 2]
        state_absmax = (
            np.nanmax(state_abs_values)
            if np.any(np.isfinite(state_abs_values))
            else np.nan
        )
        print(
            f"[diagnostics] {rk_name}: "
            f"state_absmax={state_absmax:.3e}, "
            f"rhs_absmax={rk_values[dominant_rhs_index, 3]:.3e} "
            f"({field_names[dominant_rhs_index]})",
            flush=True,
        )
        for field_name, values in zip(field_names, rk_values, strict=True):
            print(
                f"[diagnostics]   {field_name}: "
                f"state=[{values[0]:.3e},{values[1]:.3e}], "
                f"rhs_absmax={values[3]:.3e}",
                flush=True,
            )


def _atomic_save_npz(path: Path, **payload: object) -> None:
    """Write an NPZ checkpoint beside ``path`` and atomically publish it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _snapshot_metric_payload(global_geometry: FciGeometry3D) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for name in (
        "J", "g11", "g22", "g33", "g12", "g13", "g23",
        "g_11", "g_22", "g_33", "g_12", "g_13", "g_23",
    ):
        value = getattr(global_geometry.cell_metric, name, None)
        if value is not None:
            payload[f"metric_{name}"] = np.asarray(value, dtype=np.float64)
    payload["jacobian"] = np.asarray(global_geometry.cell_metric.J, dtype=np.float64)
    payload["Bmag"] = np.asarray(global_geometry.cell_bfield.Bmag, dtype=np.float64)
    payload["B_contravariant"] = np.asarray(
        global_geometry.cell_bfield.B_contra,
        dtype=np.float64,
    )
    return payload


def _snapshot_parallel_coefficient_payload(
    state_payload: dict[str, np.ndarray],
    *,
    Bmag: np.ndarray,
    tau: float,
    mi_over_me: float,
) -> dict[str, np.ndarray]:
    """Materialize the state-dependent coefficients used by parallel EB terms.

    These arrays are derivable from a complete state, but storing them makes a
    frozen-snapshot audit self-describing and prevents an analysis driver from
    silently substituting equilibrium coefficients.
    """

    density = np.asarray(state_payload["density"], dtype=np.float64)
    Te = np.asarray(state_payload["Te"], dtype=np.float64)
    Ti = np.asarray(state_payload["Ti"], dtype=np.float64)
    Vi = np.asarray(state_payload["Vi"], dtype=np.float64)
    Ve = np.asarray(state_payload["Ve"], dtype=np.float64)
    density_safe = np.maximum(density, 1.0e-30)
    inverse_density = 1.0 / density_safe
    electron_pressure = density * Te
    return {
        "parallel_density_safe": density_safe,
        "parallel_inverse_density": inverse_density,
        "parallel_electron_pressure": electron_pressure,
        "parallel_total_pressure": electron_pressure
        + float(tau) * density * Ti,
        "parallel_current": density * (Vi - Ve),
        "parallel_density_Ve_flux": density * Ve,
        "parallel_Te_compression_multiplier": 2.0
        * Te
        * inverse_density
        / 3.0,
        "parallel_Ti_compression_multiplier": 2.0
        * Ti
        * inverse_density
        / 3.0,
        "parallel_Ve_pressure_multiplier": float(mi_over_me)
        * inverse_density,
        "parallel_vorticity_current_multiplier": np.square(
            np.asarray(Bmag, dtype=np.float64)
        )
        * inverse_density,
    }


def _load_restart_state(
    path: Path,
    *,
    resolution: tuple[int, int, int],
    frame: int,
) -> tuple[FciDrbEBState, float]:
    """Load one 3-D snapshot or one frame from a history NPZ."""

    with np.load(path, allow_pickle=False) as data:
        field_names = tuple(FciDrbEBState.__dataclass_fields__.keys())
        arrays: dict[str, np.ndarray] = {}
        selected_frame = 0
        for name in field_names:
            if name not in data:
                raise ValueError(
                    f"restart file {path} is missing required field {name!r}"
                )
            value = np.asarray(data[name])
            if value.ndim == 4:
                selected_frame = frame
                if not (-value.shape[0] <= frame < value.shape[0]):
                    raise ValueError(
                        f"restart frame {frame} is outside {name!r} history "
                        f"with {value.shape[0]} frames"
                    )
                value = value[frame]
            if tuple(value.shape) != resolution:
                raise ValueError(
                    f"restart field {name!r} has shape {value.shape}; "
                    f"expected resolution {resolution}"
                )
            arrays[name] = value.astype(np.float64, copy=False)
        if "times" in data:
            times = np.asarray(data["times"], dtype=np.float64).reshape(-1)
            if np.asarray(data[field_names[0]]).ndim == 4:
                if not (-times.size <= selected_frame < times.size):
                    raise ValueError(
                        f"restart frame {frame} is outside times with "
                        f"{times.size} entries"
                    )
                restart_time = float(times[selected_frame])
            else:
                restart_time = 0.0
        else:
            restart_time = float(np.asarray(data.get("time", 0.0)).reshape(-1)[0])
    return FciDrbEBState(**arrays), restart_time


def _format_snapshot_time(value: float) -> str:
    return f"{value:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")


@dataclass(frozen=True)
class FrozenEbDiagnosticRequest:
    """Request one production-split frozen-state diagnostic evaluation.

    The source is already expressed in owner space.  The two time scales are
    kept explicit because the local backward-Euler solve uses ``solve_dt``
    while the production short-leg selector is defined with ``selection_dt``.
    """

    source_state: FciDrbEBState
    implicit_solve_dt: float
    implicit_selection_dt: float
    raw_reference_state: FciDrbEBState | None = None
    execution: str = "compiled"


@dataclass(frozen=True)
class FrozenEbDiagnosticResult:
    """Globally assembled arrays from a sharded frozen EB evaluation."""

    exact_explicit: FciDrbEBState
    exact_rhs_term_fields: jax.Array
    sourced_explicit: FciDrbEBState
    sourced_rhs_term_fields: jax.Array
    reconstructed_phi: jax.Array
    phi_solver_diagnostics: jax.Array
    reconstructed_explicit: FciDrbEBState
    reconstructed_rhs_term_fields: jax.Array
    exact_implicit_complete_residual_owner: jax.Array
    exact_selected_wall: jax.Array
    reconstructed_implicit_complete_residual_owner: jax.Array
    reconstructed_selected_wall: jax.Array
    material_counterfactual_fields: jax.Array | None = None
    material_ti_force_fields: jax.Array | None = None
    poisson_operand_counterfactual_fields: jax.Array | None = None
    generalized_potential_control_fields: jax.Array | None = None


def run_full_eb(
    initial_state: FciDrbEBState,
    *,
    simulation_geometry,
    sharded_geometry: ShardedFciGeometry3D,
    mesh: Mesh,
    parameters: FciDrbEBRhsParameters,
    gmres_target_tolerance: float,
    gmres_acceptance_tolerance: float,
    gmres_max_iterations: int,
    gmres_restart: int = 100,
    gmres_preconditioner: str,
    gmres_residual_correction_steps: int = 0,
    time_integrator: str,
    imex_split: str = "historical",
    advance_execution: str = "compiled",
    num_steps: int,
    timestep: float,
    start_time: float,
    output_path: Path,
    save_every: int,
    phase_timing: bool = True,
    diagnostic_every: int = 0,
    checkpoint_every: int = 0,
    snapshot_times: tuple[float, ...] = (),
    snapshot_dir: Path | None = None,
    snapshot_term_fields: bool = False,
    track_rhs_terms: bool = False,
    rhs_replay_history: Path | None = None,
    rhs_replay_frames: tuple[int, ...] = (),
    rhs_replay_output: Path | None = None,
    rhs_replay_electron_force_wall_audit: bool = False,
    rhs_replay_execution: str = "compiled",
    run_metadata: dict[str, object] | None = None,
    reconstruct_initial_phi: bool = True,
    neumann_ghost_scheme: str = "physical",
    parallel_velocity_wall_bc: str = "neumann",
    physical_wall_model: str = "legacy-velocity-trace",
    conducting_sheath_wall_potential: float | None = None,
    parallel_operator_scheme: str = "coordinate",
    poisson_bracket_scheme: str = "direct",
    polarization_operator_form: str = "conservative",
    polarization_coarse_data=None,
    parallel_material_scheme: str | None = None,
    parallel_vorticity_advection_scheme: str | None = None,
    parallel_material_fallback_representation: str | None = None,
    parallel_material_div_b_fallback_scheme: str | None = None,
    track_curvature_chain_rule_defect: bool = False,
    control_volume_descriptor=None,
    control_volume_fields_host=None,
    control_volume_boundary_bc=None,
    control_volume_assembler=None,
    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,
    source_evaluator: Callable[[float], FciDrbEBState] | None = None,
    history_dtype: str = "float32",
    frozen_diagnostic: FrozenEbDiagnosticRequest | None = None,
    staged_audit_cells: tuple[tuple[int, int, int], ...] = (),
    staged_audit_output: Path | None = None,
    staged_audit_explicit_ablation: str = "none",
    initialize_rung3_wall_layer: bool = False,
    rung3_wall_layer_cells: int = 8,
) -> FciDrbEBState | FrozenEbDiagnosticResult:
    """Advance the global EB state or evaluate its sharded frozen diagnostic."""

    # The producer owns qualification.  This consumer deliberately performs
    # no geometry audit, cache lookup, fitting, tracing, or regeneration.
    # Ordinary attribute/array failures during lowering are the only
    # compatibility checks on this path.
    global_geometry = simulation_geometry.global_geometry
    cell_positions = simulation_geometry.cell_positions
    nfp = simulation_geometry.nfp
    curvature_edge_one_form = simulation_geometry.curvature_edge_one_form
    owner_host_geometry = simulation_geometry.owner_geometry
    geometry_metadata = dict(simulation_geometry.metadata)

    shard_counts = tuple(int(value) for value in sharded_geometry.shard_counts)
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        raise ValueError(
            "run_full_eb supports eta-only decomposition; radial and "
            "poloidal shard counts must both be one"
        )

    solver_space = (
        "owner-grid-RLP" if control_volume_descriptor is not None else "full-grid"
    )
    if control_volume_descriptor is not None and control_volume_assembler is None:
        raise ValueError(
            "control_volume_assembler is required with a control-volume descriptor"
        )
    if int(control_volume_field_count) < 1:
        raise ValueError("control_volume_field_count must be positive")
    if time_integrator not in ("rk4", "imex-ssp222"):
        raise ValueError("time_integrator must be 'rk4' or 'imex-ssp222'")
    if imex_split not in ("historical", "coupled-boundary"):
        raise ValueError("imex_split must be 'historical' or 'coupled-boundary'")
    if imex_split == "coupled-boundary":
        if time_integrator != "imex-ssp222":
            raise ValueError("coupled-boundary requires time_integrator='imex-ssp222'")
        if advance_execution != "eager":
            raise ValueError("coupled-boundary currently requires eager execution")
        if shard_counts != (1, 1, 1):
            raise ValueError("coupled-boundary currently supports one device only")
        if parallel_operator_scheme != "fci":
            raise ValueError("coupled-boundary requires parallel_operator_scheme='fci'")
    if advance_execution not in ("compiled", "staged-compiled", "eager"):
        raise ValueError(
            "advance_execution must be 'compiled', 'staged-compiled', or 'eager'"
        )
    if advance_execution == "staged-compiled" and time_integrator != "imex-ssp222":
        raise ValueError(
            "advance_execution='staged-compiled' currently requires "
            "time_integrator='imex-ssp222'"
        )
    if staged_audit_cells and advance_execution != "staged-compiled":
        raise ValueError(
            "staged_audit_cells require advance_execution='staged-compiled'"
        )
    if staged_audit_cells and staged_audit_output is None:
        raise ValueError("staged_audit_cells require staged_audit_output")
    if staged_audit_output is not None and not staged_audit_cells:
        raise ValueError("staged_audit_output requires staged_audit_cells")
    if staged_audit_explicit_ablation not in (
        "none",
        "phi-current-pair",
        "vorticity-parallel-advection",
        "vorticity-advection-phi-current",
        "curvature",
        "parallel-material",
        "curvature-parallel-material",
    ):
        raise ValueError(
            "staged_audit_explicit_ablation must be 'none', "
            "'phi-current-pair', 'vorticity-parallel-advection', "
            "'vorticity-advection-phi-current', 'curvature', "
            "'parallel-material', or "
            "'curvature-parallel-material'"
        )
    if staged_audit_explicit_ablation != "none" and not staged_audit_cells:
        raise ValueError(
            "staged_audit_explicit_ablation requires staged_audit_cells"
        )
    if staged_audit_cells and shard_counts != (1, 1, 1):
        raise ValueError(
            "selected-cell staged audits currently require shard_counts=(1, 1, 1)"
        )
    if history_dtype not in ("float32", "float64"):
        raise ValueError("history_dtype must be 'float32' or 'float64'")
    if physical_wall_model not in PHYSICAL_WALL_MODEL_NAMES:
        raise ValueError(
            f"physical_wall_model must be one of {PHYSICAL_WALL_MODEL_NAMES}, "
            f"got {physical_wall_model!r}"
        )
    if int(rung3_wall_layer_cells) < 1:
        raise ValueError("rung3_wall_layer_cells must be positive")
    if initialize_rung3_wall_layer and physical_wall_model != "simplified-gbs-mpe":
        raise ValueError(
            "initialize_rung3_wall_layer requires physical_wall_model="
            "'simplified-gbs-mpe'"
        )
    if initialize_rung3_wall_layer:
        validate_boundary_compatible_initialization_support(
            sharded_geometry.domain,
            int(rung3_wall_layer_cells),
        )
    run_metadata = {
        **(run_metadata or {}),
        "rung3_wall_layer_initialization_requested": bool(
            initialize_rung3_wall_layer
        ),
        "rung3_wall_layer_initialization_effective": bool(
            initialize_rung3_wall_layer
            and physical_wall_model == "simplified-gbs-mpe"
        ),
        "rung3_wall_layer_cells_requested": int(rung3_wall_layer_cells),
        "rung3_wall_layer_cells_effective": (
            int(rung3_wall_layer_cells)
            if initialize_rung3_wall_layer
            else 0
        ),
        "rung3_wall_layer_initialization_algorithm": (
            "cubic-smoothstep-owner-rings-to-live-directional-fci-wall-targets-"
            "then-compact-quintic-boundary-compatible-upper-wall-traces-"
            "wall-area-phi-gauge-and-polarization-derived-vorticity"
            if initialize_rung3_wall_layer
            else None
        ),
    }
    if (
        physical_wall_model != "legacy-velocity-trace"
        and parameters.parallel_characteristic_wall_law != "physical-boundary-state"
    ):
        raise ValueError(
            "named physical wall models require "
            "parallel_characteristic_wall_law='physical-boundary-state'"
        )
    if (
        physical_wall_model == "legacy-velocity-trace"
        and parameters.parallel_characteristic_wall_law == "physical-boundary-state"
    ):
        raise ValueError(
            "parallel_characteristic_wall_law='physical-boundary-state' "
            "requires a named physical wall model"
        )
    if frozen_diagnostic is not None:
        if not isinstance(frozen_diagnostic, FrozenEbDiagnosticRequest):
            raise TypeError(
                "frozen_diagnostic must be FrozenEbDiagnosticRequest or None"
            )
        if frozen_diagnostic.execution not in ("compiled", "eager"):
            raise ValueError(
                "frozen diagnostic execution must be 'compiled' or 'eager'"
            )
        if float(frozen_diagnostic.implicit_solve_dt) <= 0.0:
            raise ValueError("frozen diagnostic implicit_solve_dt must be positive")
        if float(frozen_diagnostic.implicit_selection_dt) <= 0.0:
            raise ValueError(
                "frozen diagnostic implicit_selection_dt must be positive"
            )
        if rhs_replay_history is not None:
            raise ValueError(
                "frozen_diagnostic and rhs_replay_history are mutually exclusive"
            )
    history_numpy_dtype = (
        np.float32 if history_dtype == "float32" else np.float64
    )
    short_leg_treatment = os.environ.get(
        "DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"
    )
    if imex_split == "historical" and short_leg_treatment == "local-backward-euler" and time_integrator != "imex-ssp222":
        raise ValueError(
            "local-backward-euler short legs require the stage-wise "
            "time_integrator='imex-ssp222'; post-step RK4 splitting is not "
            "a consistent handoff"
        )
    if imex_split == "historical" and time_integrator == "imex-ssp222" and short_leg_treatment != "local-backward-euler":
        raise ValueError(
            "time_integrator='imex-ssp222' currently requires "
            "parallel_short_leg_treatment='local-backward-euler'"
        )
    if gmres_restart < 1:
        raise ValueError("gmres_restart must be positive")
    if int(checkpoint_every) < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if rhs_replay_execution not in ("compiled", "eager"):
        raise ValueError("rhs_replay_execution must be 'compiled' or 'eager'")
    setup_execution = (
        rhs_replay_execution
        if rhs_replay_history is not None
        else advance_execution
    )
    if parallel_operator_scheme not in ("coordinate", "fci"):
        raise ValueError(
            "parallel_operator_scheme must be 'coordinate' or 'fci', got "
            f"{parallel_operator_scheme!r}"
        )
    if parallel_operator_scheme == "fci":
        if not sharded_geometry.domain.axis_regular_axes[0]:
            raise ValueError(
                "parallel_operator_scheme='fci' requires toroidal topology"
            )
        if not sharded_geometry.maps_valid or sharded_geometry.map_fields is None:
            raise ValueError(
                "parallel_operator_scheme='fci' requires valid sharded FCI maps"
            )

    domain = sharded_geometry.domain
    if staged_audit_cells:
        global_shape = tuple(int(value) for value in sharded_geometry.global_shape)
        normalized_audit_cells = tuple(
            tuple(int(index) for index in cell) for cell in staged_audit_cells
        )
        if len(set(normalized_audit_cells)) != len(normalized_audit_cells):
            raise ValueError("staged_audit_cells must not contain duplicates")
        for cell in normalized_audit_cells:
            if len(cell) != 3 or any(
                index < 0 or index >= extent
                for index, extent in zip(cell, global_shape, strict=True)
            ):
                raise ValueError(
                    f"staged audit cell {cell!r} lies outside global shape "
                    f"{global_shape}"
                )
        staged_audit_cells = normalized_audit_cells
    spatial_spec = P("x", "y", "z")
    source_spec = P(None, "x", "y", "z")
    geometry_spec = P("x", "y", "z", None)
    replicated_spec = P()
    state_spec = initial_state.map_fields(lambda _value: spatial_spec)
    state_sharding = NamedSharding(mesh, spatial_spec)
    geometry_sharding = NamedSharding(mesh, geometry_spec)
    state = initial_state.map_fields(
        lambda value: jax.device_put(
            np.asarray(value, dtype=np.float64),
            state_sharding,
        )
    )
    source_stage_count = 4 if time_integrator == "rk4" else 2
    source_sharding = NamedSharding(mesh, source_spec)
    zero_source_stages = initial_state.map_fields(
        lambda value: jax.device_put(
            np.zeros(
                (source_stage_count,) + np.asarray(value).shape,
                dtype=np.float64,
            ),
            source_sharding,
        )
    )

    def source_stages_for_step(
        step_start_time: float,
    ) -> FciDrbEBState:
        """Evaluate and shard all explicit source stages for one timestep."""

        if source_evaluator is None:
            return zero_source_stages
        values_by_name = {name: [] for name in initial_state.field_names()}
        evaluated_sources: dict[float, FciDrbEBState] = {}
        for stage_time in _explicit_source_stage_times(
            time_integrator, step_start_time, float(timestep)
        ):
            stage_key = float(stage_time)
            source = evaluated_sources.get(stage_key)
            if source is None:
                source = source_evaluator(stage_key)
                evaluated_sources[stage_key] = source
            if not isinstance(source, FciDrbEBState):
                raise TypeError(
                    "source_evaluator must return FciDrbEBState, got "
                    f"{type(source).__name__}"
                )
            for name in initial_state.field_names():
                value = np.asarray(getattr(source, name), dtype=np.float64)
                expected_shape = tuple(int(v) for v in global_geometry.shape)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"source_evaluator field {name!r} has shape "
                        f"{value.shape}, expected {expected_shape}"
                    )
                if not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"source_evaluator field {name!r} contains non-finite values"
                    )
                values_by_name[name].append(value)
        return FciDrbEBState(**{
            name: jax.device_put(
                np.stack(values, axis=0), source_sharding
            )
            for name, values in values_by_name.items()
        })

    def materialized_state(current_state: FciDrbEBState) -> FciDrbEBState:
        materialized = (
            current_state if owner_host_geometry is None
            else _materialize_owner_state(current_state, owner_host_geometry)
        )
        return materialized

    def diagnostic_state(
        current_state: FciDrbEBState,
        local_control_volume_geometry,
    ) -> FciDrbEBState:
        if local_control_volume_geometry is None:
            return current_state
        return current_state.replace(
            density=expand_local_control_volume_owner_field(
                current_state.density, local_control_volume_geometry.cells
            ),
            phi=expand_local_control_volume_owner_field(
                current_state.phi, local_control_volume_geometry.cells
            ),
            Te=expand_local_control_volume_owner_field(
                current_state.Te, local_control_volume_geometry.cells
            ),
            Ti=expand_local_control_volume_owner_field(
                current_state.Ti, local_control_volume_geometry.cells
            ),
            Vi=expand_local_control_volume_owner_field(
                current_state.Vi, local_control_volume_geometry.cells
            ),
            Ve=expand_local_control_volume_owner_field(
                current_state.Ve, local_control_volume_geometry.cells
            ),
            vorticity=expand_local_control_volume_owner_field(
                current_state.vorticity, local_control_volume_geometry.cells
            ),
        )
    geometry_field_count = int(sharded_geometry.cell_fields.shape[-1])
    curvature_face_field_count = 0
    cell_fields_host = np.asarray(
        sharded_geometry.cell_fields, dtype=np.float64
    )
    print(
        "[simulation] precomputing invariant full-torus curvature face "
        "coefficients",
        flush=True,
    )
    curvature_precompute_start = time.perf_counter()
    host_sharded_geometry = build_local_fci_geometries(
        global_geometry,
        (1, 1, 1),
        halo_width=int(sharded_geometry.domain.layout.halo_width),
        periodic_axes=sharded_geometry.domain.periodic_axes,
        axis_regular_axes=sharded_geometry.domain.axis_regular_axes,
    )
    host_local_geometry = assemble_single_device_local_fci_geometry(
        host_sharded_geometry
    )
    host_domain = replace(
        host_sharded_geometry.domain,
        mesh_axis_names=(None, None, None),
    )
    # Do not run this sizeable JAX geometry calculation primitive by primitive
    # in eager mode. It is an invariant setup kernel reused by every RHS stage.
    if curvature_edge_one_form is None:
        # Preserve the historical call contract exactly for the default path,
        # including frozen diagnostics that substitute a minimal builder.
        curvature_face_setup = jax.jit(
            lambda: build_local_curvature_face_coefficients(
                host_local_geometry,
                host_domain,
            ).axes
        )
    else:
        curvature_face_setup = jax.jit(
            lambda: build_local_curvature_face_coefficients(
                host_local_geometry,
                host_domain,
                shared_edge_one_form=curvature_edge_one_form,
            ).axes
        )
    # Coupled eager execution still permits this invariant setup kernel to
    # compile in isolation; never widen the scope to the advance/Newton path.
    if imex_split == "coupled-boundary" and advance_execution == "eager":
        with jax.disable_jit(False):
            curvature_face_axes = curvature_face_setup()
    else:
        curvature_face_axes = curvature_face_setup()
    jax.block_until_ready(curvature_face_axes)
    host_curvature_faces = LocalCurvatureFaceCoefficients3D(
        layout=host_local_geometry.layout,
        x=curvature_face_axes[0],
        y=curvature_face_axes[1],
        z=curvature_face_axes[2],
    )
    curvature_face_fields_host = _pack_curvature_face_coefficients(
        host_curvature_faces
    )
    curvature_face_field_count = int(curvature_face_fields_host.shape[-1])
    cell_fields_host = np.concatenate(
        (cell_fields_host, curvature_face_fields_host), axis=-1
    )
    print(
        "[simulation] invariant curvature face coefficients packed into "
        f"{curvature_face_field_count} shard-local channels in "
        f"{time.perf_counter() - curvature_precompute_start:.3f} s",
        flush=True,
    )
    cell_fields = jax.device_put(cell_fields_host, geometry_sharding)
    map_fields_host = (
        np.asarray(sharded_geometry.map_fields, dtype=np.float64)
        if sharded_geometry.map_fields is not None
        else np.zeros(
            sharded_geometry.global_shape + (len(FCI_MAP_FIELDS),),
            dtype=np.float64,
        )
    )
    map_fields = jax.device_put(map_fields_host, geometry_sharding)
    if control_volume_descriptor is None:
        control_volume_fields_host = np.zeros(
            sharded_geometry.global_shape + (int(control_volume_field_count),),
            dtype=np.float64,
        )
    elif control_volume_fields_host is None:
        raise ValueError(
            "control_volume_fields_host is required with an RLP descriptor"
        )
    control_volume_fields = jax.device_put(
        np.asarray(control_volume_fields_host, dtype=np.float64),
        geometry_sharding,
    )

    def diagnostic_state(current_state: FciDrbEBState, local_control_volume_geometry) -> FciDrbEBState:
        if local_control_volume_geometry is None:
            return current_state
        cells = local_control_volume_geometry.cells
        scalar = lambda value: expand_local_control_volume_owner_field(value, cells)
        return current_state.replace(
            density=scalar(current_state.density), phi=scalar(current_state.phi),
            Te=scalar(current_state.Te), Ti=scalar(current_state.Ti),
            Vi=scalar(current_state.Vi), Ve=scalar(current_state.Ve),
            vorticity=scalar(current_state.vorticity),
        )

    shard_count = int(np.prod(sharded_geometry.shard_counts))
    if phase_timing and advance_execution == "eager":
        print(
            "[simulation] operator/GMRES host-callback timing disabled for "
            "eager advancement; total step timing remains enabled",
            flush=True,
        )
        phase_timing = False
    if phase_timing and shard_count > 1:
        print(
            "[simulation] operator/GMRES host-callback timing disabled for "
            "multi-device shard_map; total step timing remains enabled",
            flush=True,
        )
        phase_timing = False

    def unpack_local_curvature_face_coefficients(
        cell_fields_owned: jax.Array,
        layout,
    ) -> LocalCurvatureFaceCoefficients3D:
        packed = cell_fields_owned[
            ...,
            geometry_field_count : geometry_field_count
            + curvature_face_field_count,
        ]
        return _unpack_curvature_face_coefficients(
            packed,
            layout,
        )

    runtime_polarization_coarse_data = polarization_coarse_data

    def build_local_model(
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        preconditioner_override: str | None = None,
        host: bool = False,
    ) -> LocalFciDrbEBRhs:
        geometry_fields_owned = cell_fields_owned[..., :geometry_field_count]
        if host:
            local_geometry = host_local_geometry
            local_domain = host_domain
        else:
            local_geometry = assemble_local_fci_geometry(
                sharded_geometry,
                geometry_fields_owned,
                map_fields_owned if parallel_operator_scheme == "fci" else None,
            )
            local_domain = domain
        local_curvature_face_coefficients = (
            unpack_local_curvature_face_coefficients(
                cell_fields_owned,
                local_geometry.layout,
            )
        )
        local_control_volume_geometry = (
            None
            if control_volume_descriptor is None
            else control_volume_assembler(
                control_volume_descriptor,
                control_volume_fields_owned,
                local_geometry,
            )
        )
        return build_local_eb_model(
            local_geometry,
            local_domain,
            parameters,
            gmres_target_tolerance=float(gmres_target_tolerance),
            gmres_acceptance_tolerance=float(gmres_acceptance_tolerance),
            gmres_max_iterations=int(gmres_max_iterations),
            gmres_restart=int(gmres_restart),
            gmres_preconditioner=str(gmres_preconditioner if preconditioner_override is None else preconditioner_override),
            gmres_residual_correction_steps=int(
                gmres_residual_correction_steps
            ),
            neumann_ghost_scheme=neumann_ghost_scheme,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            physical_wall_model=physical_wall_model,
            conducting_sheath_wall_potential=conducting_sheath_wall_potential,
            parallel_operator_scheme=parallel_operator_scheme,
            parallel_material_scheme=parallel_material_scheme,
            parallel_vorticity_advection_scheme=(
                parallel_vorticity_advection_scheme
            ),
            parallel_material_fallback_representation=(
                parallel_material_fallback_representation
            ),
            parallel_material_div_b_fallback_scheme=(
                parallel_material_div_b_fallback_scheme
            ),
            poisson_bracket_scheme=poisson_bracket_scheme,
            polarization_operator_form=polarization_operator_form,
            polarization_coarse_data=runtime_polarization_coarse_data,
            control_volume_geometry=local_control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            curvature_face_coefficients_override=(
                local_curvature_face_coefficients
            ),
        )

    if gmres_preconditioner in ("coarse-additive", "coarse-multiplicative"):
        if polarization_coarse_data is not None:
            raise ValueError("coarse preconditioning builds its polarization coarse data internally")
        if polarization_operator_form != "support-paired" or physical_wall_model != "simplified-gbs-mpe":
            raise ValueError("coarse preconditioning requires support-paired simplified-gbs-mpe")
        from drbx.native import fci_operators as _polar_ops
        coarse_start = time.perf_counter()
        # Build one host model with the ordinary Jacobi route; this avoids
        # recursively requesting the coarse payload while constructing A.
        host_control_volume_geometry = (
            None if control_volume_descriptor is None else control_volume_assembler(
                control_volume_descriptor,
                jnp.asarray(control_volume_fields_host, dtype=jnp.float64),
                host_local_geometry,
            )
        )
        host_model = build_local_eb_model(
            host_local_geometry, host_domain, parameters,
            gmres_target_tolerance=float(gmres_target_tolerance),
            gmres_acceptance_tolerance=float(gmres_acceptance_tolerance),
            gmres_max_iterations=int(gmres_max_iterations), gmres_restart=int(gmres_restart),
            gmres_preconditioner="jacobi",
            gmres_residual_correction_steps=int(gmres_residual_correction_steps),
            neumann_ghost_scheme=neumann_ghost_scheme,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            physical_wall_model=physical_wall_model,
            conducting_sheath_wall_potential=conducting_sheath_wall_potential,
            parallel_operator_scheme=parallel_operator_scheme,
            parallel_material_scheme=parallel_material_scheme,
            parallel_vorticity_advection_scheme=(
                parallel_vorticity_advection_scheme
            ),
            parallel_material_fallback_representation=(
                parallel_material_fallback_representation
            ),
            parallel_material_div_b_fallback_scheme=(
                parallel_material_div_b_fallback_scheme
            ),
            poisson_bracket_scheme=poisson_bracket_scheme,
            polarization_operator_form=polarization_operator_form,
            control_volume_geometry=host_control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            curvature_face_coefficients_override=host_curvature_faces,
        )
        host_state = FciDrbEBState(**{name: jnp.asarray(getattr(initial_state, name), dtype=jnp.float64) for name in initial_state.field_names()})
        host_face = host_model._face_bcs(host_state)
        host_solver = host_model._polarization_solver(
            _polar_ops._homogeneous_local_face_bc(host_face.phi),
            config=replace(host_model.gmres_config, regularization_epsilon=0.0),
        )
        active_host, weights_host = host_solver._operator_mass_weights()
        active_host = jnp.asarray(active_host, dtype=bool)
        weights_host = jnp.asarray(weights_host, dtype=jnp.float64)
        host_hface = _polar_ops._homogeneous_local_face_bc(host_face.phi)
        host_hcv = (None if host_solver.control_volume_boundary_bc is None else
                    _polar_ops._homogeneous_local_control_volume_boundary_bc(host_solver.control_volume_boundary_bc))
        # The coarse operator follows the full physical action, including its
        # matched Neumann surface term; the resulting coarse solve is LU.
        apply_host_operator = jax.jit(lambda values: host_solver._apply_A(
            values, face_bc=host_hface, control_volume_boundary_bc=host_hcv,
            project_mean_zero=True, boundary_is_homogeneous=True,
        ))
        if imex_split == "coupled-boundary" and advance_execution == "eager":
            with jax.disable_jit(False):
                coarse_data = build_coarse_data(
                    lambda values: apply_host_operator(values), active_host, weights_host,
                    global_shape=tuple(int(v) for v in sharded_geometry.global_shape),
                )
        else:
            coarse_data = build_coarse_data(
                lambda values: apply_host_operator(values), active_host, weights_host,
                global_shape=tuple(int(v) for v in sharded_geometry.global_shape),
            )
        if not np.all(np.isfinite(np.asarray(coarse_data.H_lu))):
            raise ValueError("coarse setup produced non-finite H factor")
        runtime_polarization_coarse_data = coarse_data
        run_metadata.update({
            "polarization_coarse_preconditioner": str(gmres_preconditioner),
            "polarization_coarse_multiplicative_cycles": 1 if gmres_preconditioner == "coarse-multiplicative" else 0,
            "polarization_coarse_setup_rank": int(coarse_data.rank),
            "polarization_coarse_setup_seconds": float(time.perf_counter() - coarse_start),
            "polarization_coarse_setup_count": 1,
        })
        print(
            f"[simulation] {gmres_preconditioner} setup complete: "
            f"rank={int(coarse_data.rank)}, seconds={time.perf_counter()-coarse_start:.3f}",
            flush=True,
        )

    rung3_wall_layer_effective_cells = int(rung3_wall_layer_cells)
    rung3_wall_layer_initialize_sharded = None
    rung3_wall_layer_derive_sharded = None
    if initialize_rung3_wall_layer:
        if owner_host_geometry is not None:
            outer_active = np.asarray(
                owner_host_geometry.topology.is_active_owner, dtype=bool
            )[-rung3_wall_layer_effective_cells:]
            if not np.all(outer_active):
                raise ValueError(
                    "boundary-compatible Rung-3 initialization requires every "
                    "owner in the outer patch band to be canonical and active"
                )

        def initialize_rung3_wall_layer_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ) -> tuple[FciDrbEBState, jax.Array]:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            return _initialize_rung3_wall_layer_local(
                model,
                local_state,
                timestep=timestep,
                layer_count=rung3_wall_layer_effective_cells,
            )

        rung3_wall_layer_initialize_sharded = jax.shard_map(
            initialize_rung3_wall_layer_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(state_spec, replicated_spec),
            check_vma=False,
        )

        def derive_rung3_vorticity_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ) -> FciDrbEBState:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            face_bc = model._face_bcs(local_state)
            vorticity = model._vorticity_from_polarization(
                local_state.phi,
                local_state.Ti,
                face_bc.phi,
                face_bc.Ti,
            )
            return model._owner_state(local_state.replace(vorticity=vorticity))

        rung3_wall_layer_derive_sharded = jax.shard_map(
            derive_rung3_vorticity_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=state_spec,
            check_vma=False,
        )

        rung3_wall_layer_start = time.perf_counter()
        print(
            "[rung3-init] compiling FCI blend and boundary-compatible wall-layer initializer "
            f"(requested_cells={int(rung3_wall_layer_cells)}, "
            f"effective_cells={rung3_wall_layer_effective_cells})",
            flush=True,
        )
        rung3_wall_layer_initialize = jax.jit(
            rung3_wall_layer_initialize_sharded
        )
        state, rung3_wall_layer_diagnostics = rung3_wall_layer_initialize(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        jax.block_until_ready((state, rung3_wall_layer_diagnostics))
        rung3_wall_layer_diagnostics_host = np.asarray(
            rung3_wall_layer_diagnostics
        )
        compatibility_diagnostics_host = rung3_wall_layer_diagnostics_host[6:]
        compatibility_failures = boundary_compatibility_acceptance_failures(
            compatibility_diagnostics_host
        )
        if compatibility_failures:
            raise FloatingPointError(
                "boundary-compatible Rung-3 initialization rejected: "
                + "; ".join(compatibility_failures)
            )
        compatibility_metadata = {
            f"rung3_wall_layer_{name}": float(value)
            for name, value in zip(
                BOUNDARY_COMPATIBILITY_DIAGNOSTIC_NAMES,
                compatibility_diagnostics_host,
                strict=True,
            )
        }
        run_metadata.update(
            {
                "rung3_wall_layer_selected_backward_hits": int(
                    rung3_wall_layer_diagnostics_host[0]
                ),
                "rung3_wall_layer_selected_forward_hits": int(
                    rung3_wall_layer_diagnostics_host[1]
                ),
                "rung3_wall_layer_double_hit_owners": int(
                    rung3_wall_layer_diagnostics_host[2]
                ),
                "rung3_wall_layer_double_hit_conflicts": int(
                    rung3_wall_layer_diagnostics_host[3]
                ),
                "rung3_wall_layer_max_double_hit_target_mismatch": float(
                    rung3_wall_layer_diagnostics_host[4]
                ),
                "rung3_wall_layer_modified_owners": int(
                    rung3_wall_layer_diagnostics_host[5]
                ),
                "rung3_wall_layer_compatibility_accepted": True,
                **compatibility_metadata,
            }
        )
        print(
            "[rung3-init] wall-layer initializer completed in "
            f"{time.perf_counter() - rung3_wall_layer_start:.3f} s; "
            "Vi/Ve targets began from live FCI wall endpoint values; "
            f"modified_owners={int(rung3_wall_layer_diagnostics_host[5])}, "
            f"double_hit_conflicts={int(rung3_wall_layer_diagnostics_host[3])}; "
            "upper coordinate-wall density/phi normal data and velocity targets "
            "passed compatibility checks; vorticity is derived from the adjusted "
            "polarization state",
            flush=True,
        )

    phi_start = time.perf_counter()
    if frozen_diagnostic is None:
        print(
            "[simulation] compiling and "
            + ("reconstructing" if reconstruct_initial_phi else "reusing")
            + " initial sharded phi",
            flush=True,
        )
    else:
        print(
            "[frozen-diagnostic] preparing sharded explicit, implicit, and "
            "phi-reconstruction operators",
            flush=True,
        )

    def reconstruct_initial_phi_kernel(
        local_state: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        phi, info = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
        ).reconstruct_phi(local_state, return_diagnostics=True)
        return phi, _format_phi_solver_diagnostics(info)

    reconstruct_phi_sharded = jax.shard_map(
        reconstruct_initial_phi_kernel,
        mesh=mesh,
        in_specs=(
            state_spec,
            geometry_spec,
            geometry_spec,
            geometry_spec,
        ),
        out_specs=(spatial_spec, replicated_spec),
        check_vma=False,
    )
    reconstruct_phi = jax.jit(reconstruct_phi_sharded)
    if frozen_diagnostic is not None:
        expected_shape = tuple(int(value) for value in sharded_geometry.global_shape)
        source_state = frozen_diagnostic.source_state
        if not isinstance(source_state, FciDrbEBState):
            raise TypeError(
                "frozen diagnostic source_state must be FciDrbEBState"
            )
        for name, value in source_state.field_items():
            host_value = np.asarray(value, dtype=np.float64)
            if host_value.shape != expected_shape:
                raise ValueError(
                    f"frozen diagnostic source field {name!r} has shape "
                    f"{host_value.shape}, expected {expected_shape}"
                )
            if not np.all(np.isfinite(host_value)):
                raise ValueError(
                    f"frozen diagnostic source field {name!r} contains "
                    "non-finite values"
                )
        sharded_source = source_state.map_fields(
            lambda value: jax.device_put(
                np.asarray(value, dtype=np.float64), state_sharding
            )
        )
        raw_reference_state = frozen_diagnostic.raw_reference_state
        if raw_reference_state is not None:
            if not isinstance(raw_reference_state, FciDrbEBState):
                raise TypeError(
                    "frozen diagnostic raw_reference_state must be FciDrbEBState"
                )
            for name, value in raw_reference_state.field_items():
                host_value = np.asarray(value, dtype=np.float64)
                if host_value.shape != expected_shape:
                    raise ValueError(
                        f"frozen diagnostic raw reference field {name!r} has "
                        f"shape {host_value.shape}, expected {expected_shape}"
                    )
                if not np.all(np.isfinite(host_value)):
                    raise ValueError(
                        f"frozen diagnostic raw reference field {name!r} "
                        "contains non-finite values"
                    )
            sharded_raw_reference = raw_reference_state.map_fields(
                lambda value: jax.device_put(
                    np.asarray(value, dtype=np.float64), state_sharding
                )
            )
        else:
            sharded_raw_reference = None
        zero_source = state.zeros_like()
        solve_dt = jnp.asarray(
            float(frozen_diagnostic.implicit_solve_dt), dtype=jnp.float64
        )
        selection_dt = jnp.asarray(
            float(frozen_diagnostic.implicit_selection_dt), dtype=jnp.float64
        )

        def frozen_stage_kernel(
            local_state: FciDrbEBState,
            local_source: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            return model.evaluate_stage(
                local_state,
                source_owned=local_source,
                phi_owned=local_state.phi,
                short_leg_selection_dt=selection_dt,
                return_rhs_term_fields=True,
            )

        frozen_stage_sharded = jax.shard_map(
            frozen_stage_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(state_spec, P(None, None, "x", "y", "z")),
            check_vma=False,
        )

        # A qualification run must not compare results from differently
        # specialized stage executables.  In particular, a large N=64 HSX
        # graph once produced a wrong Ve electrostatic term only in the lean
        # specialization while this audited specialization was correct.
        # Accept the source as a dynamic input so exact, source-paired, and
        # reconstructed evaluations all reuse this one executable shape.
        def frozen_audited_stage_kernel(
            local_state: FciDrbEBState,
            local_source: FciDrbEBState,
            local_raw_reference: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            return model.evaluate_stage(
                local_state,
                source_owned=local_source,
                phi_owned=local_state.phi,
                short_leg_selection_dt=selection_dt,
                return_rhs_term_fields=True,
                return_mms_counterfactual_fields=True,
                diagnostic_raw_state=local_raw_reference,
            )

        frozen_audited_stage_sharded = jax.shard_map(
            frozen_audited_stage_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                state_spec,
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(
                state_spec,
                P(None, None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            ),
            check_vma=False,
        )

        def frozen_implicit_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            _updated, _increment, info = (
                model.apply_short_leg_implicit_material_step(
                    local_state,
                    solve_dt=solve_dt,
                    selection_dt=selection_dt,
                    phi_owned=local_state.phi,
                    return_increment=True,
                )
            )
            return (
                info["selected_complete_residual_owner"],
                info["selected_wall"],
            )

        frozen_implicit_sharded = jax.shard_map(
            frozen_implicit_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(P("x", "y", "z", None), spatial_spec),
            check_vma=False,
        )

        def frozen_reconstruct_phi_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            phi, info = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            ).reconstruct_phi(local_state, return_diagnostics=True)
            return phi, _format_phi_solver_diagnostics(info)

        frozen_reconstruct_phi_sharded = jax.shard_map(
            frozen_reconstruct_phi_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(spatial_spec, replicated_spec),
            check_vma=False,
        )
        if frozen_diagnostic.execution == "compiled":
            # ``shard_map`` owns the logical PartitionSpecs, while the outer
            # JIT must still publish concrete global shardings for returned
            # arrays.  Without explicit output shardings, sufficiently large
            # compiled graphs can escape with JAX's internal
            # ``UnspecifiedValue`` layout.  Such an array completes device
            # execution but cannot participate in ordinary JAX arithmetic or
            # even be transferred to NumPy for MMS postprocessing.
            state_output_sharding = initial_state.map_fields(
                lambda _value: state_sharding
            )
            term_output_sharding = NamedSharding(
                mesh, P(None, None, "x", "y", "z")
            )
            material_force_output_sharding = NamedSharding(
                mesh, P(None, "x", "y", "z")
            )
            implicit_output_sharding = NamedSharding(
                mesh, P("x", "y", "z", None)
            )
            replicated_output_sharding = NamedSharding(mesh, replicated_spec)
            frozen_stage = jax.jit(
                frozen_stage_sharded,
                out_shardings=(
                    state_output_sharding,
                    term_output_sharding,
                ),
            )
            frozen_audited_stage = jax.jit(
                frozen_audited_stage_sharded,
                out_shardings=(
                    state_output_sharding,
                    term_output_sharding,
                    term_output_sharding,
                    material_force_output_sharding,
                    term_output_sharding,
                    material_force_output_sharding,
                ),
            )
            frozen_implicit = jax.jit(
                frozen_implicit_sharded,
                out_shardings=(
                    implicit_output_sharding,
                    state_sharding,
                ),
            )
            frozen_reconstruct_phi = jax.jit(
                frozen_reconstruct_phi_sharded,
                out_shardings=(
                    state_sharding,
                    replicated_output_sharding,
                ),
            )
        else:
            frozen_stage = frozen_stage_sharded
            frozen_audited_stage = frozen_audited_stage_sharded
            frozen_implicit = frozen_implicit_sharded
            frozen_reconstruct_phi = frozen_reconstruct_phi_sharded

        def execute(callable_, *values):
            with jax.disable_jit(frozen_diagnostic.execution == "eager"):
                result = callable_(*values)
            jax.block_until_ready(result)
            return result

        common_geometry = (cell_fields, map_fields, control_volume_fields)
        material_counterfactuals = None
        material_ti_forces = None
        poisson_operand_counterfactuals = None
        generalized_potential_controls = None
        if sharded_raw_reference is None:
            exact_explicit, exact_terms = execute(
                frozen_stage,
                state,
                zero_source,
                *common_geometry,
            )
        else:
            # Deliberately use the same augmented executable as the exact
            # evaluation.  The extra outputs are discarded after completion;
            # their presence keeps compiler specialization identical.
            (
                exact_explicit,
                exact_terms,
                material_counterfactuals,
                material_ti_forces,
                poisson_operand_counterfactuals,
                generalized_potential_controls,
            ) = execute(
                frozen_audited_stage,
                state,
                zero_source,
                sharded_raw_reference,
                *common_geometry,
            )
        if sharded_raw_reference is None:
            sourced_explicit, sourced_terms = execute(
                frozen_stage,
                state,
                sharded_source,
                *common_geometry,
            )
        else:
            (
                sourced_explicit,
                sourced_terms,
                _sourced_material_counterfactuals,
                _sourced_material_ti_forces,
                _sourced_poisson_operand_counterfactuals,
                _sourced_generalized_potential_controls,
            ) = execute(
                frozen_audited_stage,
                state,
                sharded_source,
                sharded_raw_reference,
                *common_geometry,
            )
            del (
                _sourced_material_counterfactuals,
                _sourced_material_ti_forces,
                _sourced_poisson_operand_counterfactuals,
                _sourced_generalized_potential_controls,
            )
        exact_implicit, exact_selected_wall = execute(
            frozen_implicit,
            state,
            *common_geometry,
        )
        reconstructed_phi, phi_diagnostics = execute(
            frozen_reconstruct_phi,
            state,
            *common_geometry,
        )
        reconstructed_state = state.replace(phi=reconstructed_phi)
        if sharded_raw_reference is None:
            reconstructed_explicit, reconstructed_terms = execute(
                frozen_stage,
                reconstructed_state,
                zero_source,
                *common_geometry,
            )
        else:
            (
                reconstructed_explicit,
                reconstructed_terms,
                _reconstructed_material_counterfactuals,
                _reconstructed_material_ti_forces,
                _reconstructed_poisson_operand_counterfactuals,
                _reconstructed_generalized_potential_controls,
            ) = execute(
                frozen_audited_stage,
                reconstructed_state,
                zero_source,
                sharded_raw_reference,
                *common_geometry,
            )
            del (
                _reconstructed_material_counterfactuals,
                _reconstructed_material_ti_forces,
                _reconstructed_poisson_operand_counterfactuals,
                _reconstructed_generalized_potential_controls,
            )
        reconstructed_implicit, reconstructed_selected_wall = execute(
            frozen_implicit,
            reconstructed_state,
            *common_geometry,
        )
        print(
            "[frozen-diagnostic] sharded evaluation completed in "
            f"{time.perf_counter() - phi_start:.3f} s",
            flush=True,
        )
        return FrozenEbDiagnosticResult(
            exact_explicit=exact_explicit,
            exact_rhs_term_fields=exact_terms,
            sourced_explicit=sourced_explicit,
            sourced_rhs_term_fields=sourced_terms,
            reconstructed_phi=reconstructed_phi,
            phi_solver_diagnostics=phi_diagnostics,
            reconstructed_explicit=reconstructed_explicit,
            reconstructed_rhs_term_fields=reconstructed_terms,
            exact_implicit_complete_residual_owner=exact_implicit,
            exact_selected_wall=exact_selected_wall,
            reconstructed_implicit_complete_residual_owner=(
                reconstructed_implicit
            ),
            reconstructed_selected_wall=reconstructed_selected_wall,
            material_counterfactual_fields=material_counterfactuals,
            material_ti_force_fields=material_ti_forces,
            poisson_operand_counterfactual_fields=(
                poisson_operand_counterfactuals
            ),
            generalized_potential_control_fields=(
                generalized_potential_controls
            ),
        )
    if rhs_replay_history is not None:
        if rhs_replay_output is None or not rhs_replay_frames:
            raise ValueError(
                "RHS replay requires a nonempty frame list and output path"
            )
        if parallel_operator_scheme != "fci":
            raise ValueError("RHS replay currently requires the FCI production path")

        def replay_rhs_terms(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ) -> tuple[jax.Array, ...]:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            phi, info = model.reconstruct_phi(local_state, return_diagnostics=True)
            reconstructed = local_state.replace(phi=phi)
            rhs, term_fields, curvature_component_fields = model.evaluate_stage(
                reconstructed,
                phi_owned=phi,
                short_leg_selection_dt=(
                    jnp.asarray(float(timestep), dtype=jnp.float64)
                    if os.environ.get(
                        "DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"
                    )
                    == "local-backward-euler"
                    else None
                ),
                return_rhs_term_fields=True,
                return_curvature_component_fields=True,
            )
            polarization_terms = model.polarization_balance_terms(
                reconstructed,
                phi_owned=phi,
            )
            # Different curvature wall closures can leave the visible state
            # almost unchanged while injecting a hard, grid-scale component
            # into the next polarization solve.  Apply the exact production
            # Ti Laplacian to the stage RHS so replay files expose the source
            # tendency tau*Lperp(Ti_t)-omega_t directly.
            rhs_polarization_terms = model.polarization_balance_terms(
                rhs.replace(phi=jnp.zeros_like(rhs.phi)),
                phi_owned=jnp.zeros_like(rhs.phi),
            )
            polarization_source_tendency = (
                rhs_polarization_terms[1] + rhs_polarization_terms[2]
            )
            state_fields = jnp.stack(
                tuple(value for _name, value in reconstructed.field_items()),
                axis=0,
            )
            rhs_fields = jnp.stack(
                tuple(getattr(rhs, name) for name in RHS_TERM_FIELD_NAMES),
                axis=0,
            )
            if model.control_volume_geometry is not None:
                cells = model.control_volume_geometry.cells
                prolong = lambda value: expand_local_control_volume_owner_field(
                    value, cells
                )
                state_fields = jax.vmap(prolong)(state_fields)
                rhs_fields = jax.vmap(prolong)(rhs_fields)
                term_fields = jax.vmap(jax.vmap(prolong))(term_fields)
                curvature_component_fields = jax.vmap(jax.vmap(prolong))(
                    curvature_component_fields
                )
                polarization_terms = jax.vmap(prolong)(polarization_terms)
                polarization_source_tendency = prolong(
                    polarization_source_tendency
                )
            base_outputs = (
                state_fields,
                rhs_fields,
                term_fields,
                curvature_component_fields,
                polarization_terms,
                polarization_source_tendency,
                _format_phi_solver_diagnostics(info),
            )
            if not rhs_replay_electron_force_wall_audit:
                return base_outputs
            electron_force_outputs = model.electron_parallel_force_diagnostics(
                reconstructed,
                phi_owned=phi,
            )
            if model.control_volume_geometry is not None:
                cells = model.control_volume_geometry.cells
                prolong = lambda value: expand_local_control_volume_owner_field(
                    value, cells
                )
                electron_force_outputs = (
                    jax.vmap(prolong)(electron_force_outputs[0]),
                    jax.vmap(jax.vmap(prolong))(electron_force_outputs[1]),
                    jax.vmap(jax.vmap(prolong))(electron_force_outputs[2]),
                    *electron_force_outputs[3:6],
                    jax.vmap(prolong)(electron_force_outputs[6]),
                    *electron_force_outputs[7:],
                )
            return base_outputs + electron_force_outputs

        replay_out_specs = (
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, None, "x", "y", "z"),
            P(None, None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P("x", "y", "z"),
            replicated_spec,
        )
        if rhs_replay_electron_force_wall_audit:
            replay_out_specs = replay_out_specs + (
                P(None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            )

        replay_sharded = jax.shard_map(
            replay_rhs_terms,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=replay_out_specs,
            check_vma=False,
        )
        replay = (
            jax.jit(replay_sharded)
            if rhs_replay_execution == "compiled"
            else replay_sharded
        )
        replay_states = []
        replay_times = []
        replay_rhs = []
        replay_terms = []
        replay_curvature_components = []
        replay_polarization = []
        replay_polarization_source_tendency = []
        replay_phi_diagnostics = []
        replay_electron_force_terms = []
        replay_electron_force_leg_terms = []
        replay_electron_force_gradients = []
        replay_electron_force_endpoint_values = []
        replay_electron_force_wall_masks = []
        replay_electron_force_leg_lengths = []
        replay_electron_force_characteristic_principals = []
        replay_electron_force_characteristic_primitive_traces = []
        replay_electron_force_endpoint_kinds = []
        replay_action = (
            f"compiling on frame {rhs_replay_frames[0]}"
            if rhs_replay_execution == "compiled"
            else "running eagerly with outer jax.jit disabled"
        )
        print(
            f"[rhs-replay] {replay_action} and evaluating "
            f"{len(rhs_replay_frames)} frozen states",
            flush=True,
        )
        replay_start = time.perf_counter()
        for frame in rhs_replay_frames:
            host_state, frame_time = _load_restart_state(
                rhs_replay_history,
                resolution=tuple(int(value) for value in sharded_geometry.global_shape),
                frame=int(frame),
            )
            if owner_host_geometry is not None:
                host_state = _aggregate_initial_owner_state(
                    host_state, owner_host_geometry
                )
            sharded_state = host_state.map_fields(
                lambda value: jax.device_put(
                    jnp.asarray(value, dtype=jnp.float64), state_sharding
                )
            )
            with jax.disable_jit(rhs_replay_execution == "eager"):
                outputs = replay(
                    sharded_state,
                    cell_fields,
                    map_fields,
                    control_volume_fields,
                )
            jax.block_until_ready(outputs)
            (
                state_values,
                rhs_values,
                terms,
                curvature_components,
                polarization,
                polarization_source_tendency,
                phi_diagnostics,
            ) = tuple(
                np.asarray(value, dtype=np.float64) for value in outputs[:7]
            )
            if rhs_replay_electron_force_wall_audit:
                (
                    electron_force_terms,
                    electron_force_leg_terms,
                    electron_force_gradients,
                    electron_force_endpoint_values,
                    electron_force_wall_masks,
                    electron_force_leg_lengths,
                    electron_force_characteristic_principals,
                    electron_force_characteristic_primitive_traces,
                    electron_force_endpoint_kinds,
                ) = tuple(
                    np.asarray(value, dtype=np.float64) for value in outputs[7:]
                )
                replay_electron_force_terms.append(electron_force_terms)
                replay_electron_force_leg_terms.append(electron_force_leg_terms)
                replay_electron_force_gradients.append(electron_force_gradients)
                replay_electron_force_endpoint_values.append(
                    electron_force_endpoint_values
                )
                replay_electron_force_wall_masks.append(electron_force_wall_masks)
                replay_electron_force_leg_lengths.append(electron_force_leg_lengths)
                replay_electron_force_characteristic_principals.append(
                    electron_force_characteristic_principals
                )
                replay_electron_force_characteristic_primitive_traces.append(
                    electron_force_characteristic_primitive_traces
                )
                replay_electron_force_endpoint_kinds.append(
                    electron_force_endpoint_kinds
                )
            replay_states.append(state_values)
            replay_times.append(frame_time)
            replay_rhs.append(rhs_values)
            replay_terms.append(terms)
            replay_curvature_components.append(curvature_components)
            replay_polarization.append(polarization)
            replay_polarization_source_tendency.append(
                polarization_source_tendency
            )
            replay_phi_diagnostics.append(phi_diagnostics)
            print(
                f"[rhs-replay] frame={frame} time={frame_time:.8e} "
                f"phi_iterations={int(phi_diagnostics[0])} "
                f"phi_rel_residual={phi_diagnostics[1]:.3e}",
                flush=True,
            )

        mass_weights = (
            np.asarray(owner_host_geometry.raw_volume, dtype=np.float64)
            if owner_host_geometry is not None
            else np.asarray(global_geometry.cell_metric.J, dtype=np.float64)
        )
        metadata = dict(run_metadata or {})
        metadata.update(
            {
                "diagnostic": "frozen-state-spatial-rhs-replay",
                "rhs_replay_history": str(rhs_replay_history),
                "rhs_replay_frames": [int(value) for value in rhs_replay_frames],
                "rhs_replay_execution": str(rhs_replay_execution),
                "rhs_term_field_names": list(RHS_TERM_FIELD_NAMES),
                "rhs_term_names": {
                    field: list(names)
                    for field, names in zip(
                        RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES, strict=True
                    )
                },
                "polarization_term_names": [
                    "minus_Lperp_phi",
                    "tau_Lperp_Ti",
                    "minus_vorticity",
                ],
                "curvature_component_equation_names": [
                    "density", "Te", "Ti", "vorticity"
                ],
                "curvature_component_direction_names": list(curvature_component_diagnostic_names()),
                "electron_force_wall_audit": bool(
                    rhs_replay_electron_force_wall_audit
                ),
            }
        )
        rhs_replay_output.parent.mkdir(parents=True, exist_ok=True)
        replay_payload = {
            "frames": np.asarray(rhs_replay_frames, dtype=np.int64),
            "times": np.asarray(replay_times, dtype=np.float64),
            "state_fields": np.stack(replay_states),
            "rhs_fields": np.stack(replay_rhs),
            "rhs_term_fields": np.stack(replay_terms),
            "curvature_component_fields": np.stack(replay_curvature_components),
            "polarization_terms": np.stack(replay_polarization),
            "polarization_source_tendency": np.stack(
                replay_polarization_source_tendency
            ),
            "phi_solver_diagnostics": np.stack(replay_phi_diagnostics),
            "mass_weights": mass_weights,
            "field_names_json": np.asarray(
                json.dumps(tuple(FciDrbEBState.__dataclass_fields__.keys()))
            ),
            "rhs_term_field_names_json": np.asarray(json.dumps(RHS_TERM_FIELD_NAMES)),
            "rhs_term_names_json": np.asarray(json.dumps(RHS_TERM_NAMES)),
            "curvature_component_equation_names_json": np.asarray(
                json.dumps(("density", "Te", "Ti", "vorticity"))
            ),
            "curvature_component_direction_names_json": np.asarray(
                json.dumps(curvature_component_diagnostic_names())
            ),
            "polarization_term_names_json": np.asarray(
                json.dumps(("minus_Lperp_phi", "tau_Lperp_Ti", "minus_vorticity"))
            ),
            "run_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        if rhs_replay_electron_force_wall_audit:
            replay_payload.update(
                {
                    "electron_force_terms": np.stack(
                        replay_electron_force_terms
                    ),
                    "electron_force_leg_terms": np.stack(
                        replay_electron_force_leg_terms
                    ),
                    "electron_force_gradients": np.stack(
                        replay_electron_force_gradients
                    ),
                    "electron_force_endpoint_values": np.stack(
                        replay_electron_force_endpoint_values
                    ),
                    "electron_force_wall_masks": np.stack(
                        replay_electron_force_wall_masks
                    ),
                    "electron_force_leg_lengths": np.stack(
                        replay_electron_force_leg_lengths
                    ),
                    "electron_force_characteristic_principals": np.stack(
                        replay_electron_force_characteristic_principals
                    ),
                    "electron_force_characteristic_primitive_traces": np.stack(
                        replay_electron_force_characteristic_primitive_traces
                    ),
                    "electron_force_endpoint_kinds": np.stack(
                        replay_electron_force_endpoint_kinds
                    ),
                    "electron_force_term_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_TERM_NAMES)
                    ),
                    "electron_force_leg_term_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_LEG_TERM_NAMES)
                    ),
                    "electron_force_gradient_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_GRADIENT_NAMES)
                    ),
                    "electron_force_endpoint_field_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_ENDPOINT_FIELD_NAMES)
                    ),
                    "electron_force_stencil_direction_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_STENCIL_DIRECTION_NAMES)
                    ),
                    "electron_force_endpoint_direction_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_ENDPOINT_DIRECTION_NAMES)
                    ),
                    "electron_force_characteristic_principal_names_json": np.asarray(
                        json.dumps(ELECTRON_FORCE_CHARACTERISTIC_PRINCIPAL_NAMES)
                    ),
                    "electron_force_characteristic_primitive_field_names_json": np.asarray(
                        json.dumps(
                            ELECTRON_FORCE_CHARACTERISTIC_PRIMITIVE_FIELD_NAMES
                        )
                    ),
                }
            )
        np.savez_compressed(rhs_replay_output, **replay_payload)
        print(
            f"[rhs-replay] wrote {rhs_replay_output} in "
            f"{time.perf_counter() - replay_start:.3f} s",
            flush=True,
        )
        return state
    if reconstruct_initial_phi:
        initial_phi, initial_phi_diagnostics = reconstruct_phi(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        jax.block_until_ready((initial_phi, initial_phi_diagnostics))
        initial_phi_diagnostics_host = np.asarray(initial_phi_diagnostics)
        initial_phi_failed = bool(initial_phi_diagnostics_host[2] > 0.5)
        initial_phi_iteration_text = (
            f"GMRES iterations={int(initial_phi_diagnostics_host[0])}"
            f" relres={initial_phi_diagnostics_host[1]:.3e}"
        )
        if initial_phi_failed:
            print(
                "[diagnostics] initial phi reconstruction rejected: "
                f"{initial_phi_iteration_text}",
                flush=True,
            )
            raise FloatingPointError(
                "initial phi reconstruction did not satisfy the solver's "
                "acceptance semantics"
            )
        state = state.replace(phi=initial_phi)
    else:
        jax.block_until_ready(state)
        initial_phi_iteration_text = "GMRES reconstruction skipped"
    if (
        initialize_rung3_wall_layer
        and reconstruct_initial_phi
        and rung3_wall_layer_derive_sharded is not None
    ):
        # The first initializer pass derives omega from the restart phi before
        # the normal phi reconstruction.  Re-derive it once more after that
        # reconstruction so the tuple entering the first IMEX stage remains
        # exactly compatible with the selected Rung-3 polarization operator.
        rung3_wall_layer_derive = jax.jit(rung3_wall_layer_derive_sharded)
        state = rung3_wall_layer_derive(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        jax.block_until_ready(state)
        print(
            "[rung3-init] re-derived vorticity after initial phi "
            "reconstruction for discrete polarization consistency",
            flush=True,
        )
    print(
        f"[simulation] initial sharded phi ready in "
        f"{time.perf_counter() - phi_start:.3f} s; "
        f"{initial_phi_iteration_text}",
        flush=True,
    )

    phase_timer = (
        _JittedPhaseTimer(
            expected_markers=8 if time_integrator == "rk4" else 6,
            label=time_integrator,
        )
        if phase_timing
        else None
    )
    operator_marker = None if phase_timer is None else phase_timer.mark_operator
    gmres_marker = None if phase_timer is None else phase_timer.mark_gmres
    dt = jnp.asarray(float(timestep), dtype=jnp.float64)

    def mark_operator(rhs: FciDrbEBState) -> None:
        if operator_marker is not None:
            jax.debug.callback(
                operator_marker,
                *_state_marker_dependencies(rhs),
                ordered=True,
            )

    def reconstruct_stage_phi(
        stage_state: FciDrbEBState,
        model: LocalFciDrbEBRhs,
    ) -> tuple[jax.Array, jax.Array]:
        with jax.named_scope("gmres"):
            phi, info = model.reconstruct_phi(
                stage_state,
                return_diagnostics=True,
            )
        if gmres_marker is not None:
            jax.debug.callback(
                gmres_marker,
                jnp.ravel(phi)[0],
                ordered=True,
            )
        diagnostics = _format_phi_solver_diagnostics(info)
        return phi, diagnostics

    def evaluate_operators(
        stage_state: FciDrbEBState,
        phi: jax.Array,
        model: LocalFciDrbEBRhs,
        source_owned: FciDrbEBState | None = None,
    ) -> FciDrbEBState:
        with jax.named_scope("operators"):
            rhs = model.evaluate_stage(
                stage_state,
                source_owned=source_owned,
                phi_owned=phi,
                short_leg_selection_dt=(
                    dt
                    if os.environ.get(
                        "DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"
                    )
                    == "local-backward-euler"
                    else None
                ),
            )
        rhs = model.project_galerkin_state(rhs)
        mark_operator(rhs)
        return rhs

    def finalize_advance(
        next_state: FciDrbEBState,
        model: LocalFciDrbEBRhs,
        stage_states: tuple[FciDrbEBState, ...],
        stage_rates: tuple[FciDrbEBState, ...],
        gmres_infos: tuple[jax.Array, ...],
    ):
        """Build the common fixed-shape diagnostics for either integrator."""

        gmres_stage_diagnostics = jnp.stack(gmres_infos, axis=0)
        gmres_iterations = jnp.mean(gmres_stage_diagnostics[:, 0])
        diagnostic_states = tuple(
            diagnostic_state(
                stage,
                model.control_volume_geometry,
            )
            for stage in stage_states
        )
        state_mins = jnp.stack(tuple(
            jnp.stack(tuple(jnp.min(value) for _, value in stage.field_items()))
            for stage in diagnostic_states
        ))
        state_maxs = jnp.stack(tuple(
            jnp.stack(tuple(jnp.max(value) for _, value in stage.field_items()))
            for stage in diagnostic_states
        ))
        state_abs_maxs = jnp.stack(tuple(
            jnp.stack(tuple(
                jnp.max(jnp.abs(value)) for _, value in stage.field_items()
            ))
            for stage in diagnostic_states
        ))
        state_finites = jnp.stack(tuple(
            jnp.stack(tuple(
                jnp.all(jnp.isfinite(value)).astype(jnp.int32)
                for _, value in stage.field_items()
            ))
            # Finiteness is a validity property of canonical owner storage,
            # not merely of its materialized diagnostic view.  Checking the
            # raw stage catches poisoned owner/alias entries before expansion
            # can replace or hide them.
            for stage in stage_states
        ))
        rhs_abs_maxs = jnp.stack(tuple(
            jnp.stack(tuple(
                jnp.max(jnp.abs(value)) for _, value in rhs.field_items()
            ))
            for rhs in stage_rates
        ))
        rhs_finites = jnp.stack(tuple(
            jnp.stack(tuple(
                jnp.all(jnp.isfinite(value)).astype(jnp.int32)
                for _, value in rhs.field_items()
            ))
            for rhs in stage_rates
        ))
        for mesh_axis_name in tuple(
            name for name in model.domain.mesh_axis_names if name is not None
        ):
            state_mins = jax.lax.pmin(state_mins, mesh_axis_name)
            state_maxs = jax.lax.pmax(state_maxs, mesh_axis_name)
            state_abs_maxs = jax.lax.pmax(state_abs_maxs, mesh_axis_name)
            state_finites = jax.lax.pmin(state_finites, mesh_axis_name)
            rhs_abs_maxs = jax.lax.pmax(rhs_abs_maxs, mesh_axis_name)
            rhs_finites = jax.lax.pmin(rhs_finites, mesh_axis_name)
        stage_finites = jnp.logical_and(state_finites > 0, rhs_finites > 0)
        stage_diagnostics = jnp.stack(
            (
                state_mins,
                state_maxs,
                state_abs_maxs,
                rhs_abs_maxs,
                stage_finites.astype(jnp.float64),
            ),
            axis=-1,
        )

        # Keep this a fixed-shape compiled payload.  Each reduction is local
        # to the shard first and then made global over all three shard_map
        # mesh axes.
        field_values = tuple(
            value
            for _, value in diagnostic_state(
                next_state,
                model.control_volume_geometry,
            ).field_items()
        )
        field_mins = jnp.stack(tuple(jnp.min(value) for value in field_values))
        field_maxs = jnp.stack(tuple(jnp.max(value) for value in field_values))
        field_abs_maxs = jnp.stack(
            tuple(jnp.max(jnp.abs(value)) for value in field_values)
        )
        for mesh_axis_name in tuple(
            name for name in model.domain.mesh_axis_names if name is not None
        ):
            field_mins = jax.lax.pmin(field_mins, mesh_axis_name)
            field_maxs = jax.lax.pmax(field_maxs, mesh_axis_name)
            field_abs_maxs = jax.lax.pmax(field_abs_maxs, mesh_axis_name)
        diagnostics = jnp.stack((field_mins, field_maxs, field_abs_maxs), axis=1)
        if track_curvature_chain_rule_defect:
            curvature_diagnostics = (
                model.ion_temperature_curvature_chain_rule_diagnostics(next_state)
            )
            for mesh_axis_name in tuple(
                name for name in model.domain.mesh_axis_names if name is not None
            ):
                curvature_diagnostics = jax.lax.pmax(
                    curvature_diagnostics, mesh_axis_name
                )
            return (
                next_state,
                diagnostics,
                curvature_diagnostics,
                gmres_iterations,
                gmres_stage_diagnostics,
                stage_diagnostics,
            )
        return (
            next_state,
            diagnostics,
            gmres_iterations,
            gmres_stage_diagnostics,
            stage_diagnostics,
        )

    def full_rk4_advance(
        current: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        source_stages: FciDrbEBState,
        current_time: jax.Array,
    ) -> tuple[FciDrbEBState, jax.Array, jax.Array] | tuple[
        FciDrbEBState, jax.Array, jax.Array, jax.Array
    ]:
        del current_time
        model = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
        )

        # `current.phi` was reconstructed at the end of the previous advance,
        # so stage one does not need another identical elliptic solve.
        def stage_source(index: int) -> FciDrbEBState:
            return source_stages.replace(
                density=source_stages.density[index],
                phi=source_stages.phi[index],
                Te=source_stages.Te[index],
                Ti=source_stages.Ti[index],
                Vi=source_stages.Vi[index],
                Ve=source_stages.Ve[index],
                vorticity=source_stages.vorticity[index],
            )

        k1 = evaluate_operators(current, current.phi, model, stage_source(0))
        stage_2 = current.axpy(k1, scale=0.5 * dt)

        phi_2, gmres_info_2 = reconstruct_stage_phi(stage_2, model)
        k2 = evaluate_operators(stage_2, phi_2, model, stage_source(1))
        stage_3 = current.axpy(k2, scale=0.5 * dt)

        phi_3, gmres_info_3 = reconstruct_stage_phi(stage_3, model)
        k3 = evaluate_operators(stage_3, phi_3, model, stage_source(2))
        stage_4 = current.axpy(k3, scale=dt)

        phi_4, gmres_info_4 = reconstruct_stage_phi(stage_4, model)
        k4 = evaluate_operators(stage_4, phi_4, model, stage_source(3))
        weighted_rhs = k1.axpy(k2, scale=2.0).axpy(
            k3,
            scale=2.0,
        ).axpy(k4, scale=1.0)
        next_state = current.axpy(weighted_rhs, scale=dt / 6.0)
        next_phi, gmres_info_next = reconstruct_stage_phi(next_state, model)
        next_state = next_state.replace(phi=next_phi)
        return finalize_advance(
            next_state,
            model,
            (current, stage_2, stage_3, stage_4, next_state),
            (k1, k2, k3, k4, weighted_rhs),
            (gmres_info_2, gmres_info_3, gmres_info_4, gmres_info_next),
        )

    def full_imex_advance(
        current: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        source_stages: FciDrbEBState,
        current_time: jax.Array,
    ):
        """Advance with the complete short-wall residual at every IMEX stage."""

        del current_time
        model = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
        )
        def implicit_stage(base: FciDrbEBState, _model, solve_dt, selection_dt):
            updated, increment, _info = (
                model.apply_short_leg_implicit_material_step(
                    base,
                    solve_dt=solve_dt,
                    selection_dt=selection_dt,
                    phi_owned=base.phi,
                    return_increment=True,
                )
            )
            stage_phi, phi_info = reconstruct_stage_phi(updated, model)
            stage = updated.replace(phi=stage_phi)
            implicit_rate = increment.map_fields(
                lambda value: value / solve_dt
            )
            return stage, implicit_rate, phi_info
        next_state, stage_states, stage_rates, phi_infos = historical_imex_ssp222_stage(
            current, model, source_stages, dt,
            implicit_stage=implicit_stage,
            explicit_operator=evaluate_operators,
            reconstruct_phi=reconstruct_stage_phi,
        )
        return finalize_advance(
            next_state,
            model,
            stage_states,
            stage_rates,
            phi_infos,
        )

    def coupled_host_advance(
        current, cell_fields_owned, map_fields_owned,
        control_volume_fields_owned, source_stages, current_time,
    ):
        """Eager single-device host orchestration for the coupled contract."""
        del current_time
        from drbx.native.fci_boundary_imex import (
            advance_ssp222_coupled, solve_coupled_boundary_stage,
        )
        from drbx.native.fci_boundary_imex_model import CoupledBoundaryStageContext
        from drbx.native.fci_boundary_imex_split import evaluate_boundary_imex_split
        from drbx.native.fci_boundary_imex_model import evaluate_coupled_stage_admissibility
        if not hasattr(coupled_host_advance, "_resources"):
            from drbx.native.fci_boundary_imex_kernels import build_coupled_residual_kernel
            resource_model = build_local_model(
                cell_fields_owned, map_fields_owned, control_volume_fields_owned,
                host=True,
            )
            coupled_host_advance._resources = (
                resource_model, build_coupled_residual_kernel(resource_model)
            )
        model, residual_kernel = coupled_host_advance._resources
        stage_states = []
        stage_rates = []
        stage_infos = []
        stage_multipliers = []
        stage_bases = [current]
        explicit_rates = []
        coupled_phi_infos = []
        final_phi_info = [None]
        solve_index = [0]
        explicit_index = [0]

        def source_at(index):
            return source_stages.replace(**{
                name: getattr(source_stages, name)[index]
                for name in source_stages.field_names()
            })

        def solve_stage(base, stage_dt):
            stage_bases.append(base)
            source = source_at(min(solve_index[0], 1))
            solve_index[0] += 1
            stage, multiplier, info = solve_coupled_boundary_stage(
                model, base, solve_dt=stage_dt, source_owned=source,
                residual_kernel=residual_kernel,
                progress=lambda event: print(
                    f"[coupled stage {solve_index[0]}] {event}", flush=True),
            )
            if not info.converged:
                coupled_stage_records.append({
                    "success": False, "stage": solve_index[0],
                    "info": asdict(info), "failure_reason": str(info.reason),
                })
                raise RuntimeError(f"coupled boundary stage failed: {info.reason}")
            split = evaluate_boundary_imex_split(
                model, stage, source_owned=source,
                polarization_multiplier=multiplier,
            )
            rate = split.implicit
            stage_states.append(stage)
            stage_rates.append(rate)
            stage_multipliers.append(multiplier)
            context = CoupledBoundaryStageContext(model, base, stage_dt, source)
            pd = context.polarization_diagnostics(
                context.pack(stage, multiplier)
            )
            diag = jnp.zeros((_PHI_DIAGNOSTIC_WIDTH,), dtype=jnp.float64)
            diag = diag.at[0].set(info.linear_iterations)
            diag = diag.at[1].set(pd["polarization_relative"])
            diag = diag.at[2].set(0.0)
            diag = diag.at[3].set(1.0)
            diag = diag.at[4].set(multiplier)
            diag = diag.at[5].set(pd["raw_compatibility_defect"])
            diag = diag.at[6].set(pd["gauge"])
            diag = diag.at[8].set(pd["polarization_l2"])
            diag = diag.at[9].set(pd["raw_rhs_l2"])
            diag = diag.at[11].set(jnp.all(jnp.isfinite(stage.phi)))
            diag = diag.at[12].set(jnp.all(jnp.isfinite(pd["raw_rhs_l2"])))
            diag = diag.at[13].set(1.0)
            diag = diag.at[15].set(1.0)
            diag = diag.at[16].set(1.0)
            coupled_phi_infos.append(diag)
            stage_infos.append(info)
            return stage, rate, {"converged": True, "multiplier": float(multiplier)}

        def explicit(stage, source):
            multiplier = stage_multipliers[min(explicit_index[0], len(stage_multipliers) - 1)]
            explicit_index[0] += 1
            explicit_rate = evaluate_boundary_imex_split(
                model, stage, source_owned=source,
                polarization_multiplier=multiplier,
            ).explicit
            explicit_rates.append(explicit_rate)
            return explicit_rate

        def reconstruct(final):
            phi, phi_info = reconstruct_stage_phi(final, model)
            final_phi_info[0] = phi_info
            accepted = _coupled_phi_diagnostics_accepted(phi_info)
            gauge_ok = bool(np.asarray(np.abs(phi_info[6]) <= 1.0e-10))
            admissibility = evaluate_coupled_stage_admissibility(
                model, final.replace(phi=phi)
            )
            return final.replace(phi=phi), {
                "converged": accepted and gauge_ok and bool(admissibility["admissible"]),
                "phi_diagnostics": np.asarray(phi_info).tolist(),
                "admissibility": admissibility,
            }

        result, orchestration_info = advance_ssp222_coupled(
            current, float(timestep), solve_stage=solve_stage,
            explicit=explicit, source_stages=source_stages,
            reconstruct_final=reconstruct,
        )
        if not orchestration_info.get("converged", False):
            raise RuntimeError(f"coupled boundary advance failed: {orchestration_info}")
        weighted = stage_rates[0].axpy(explicit_rates[0], scale=1.0).axpy(
            stage_rates[1], scale=1.0
        ).axpy(explicit_rates[1], scale=1.0).map_fields(
            lambda value: 0.5 * value
        )
        phi_infos = (
            coupled_phi_infos[0], coupled_phi_infos[0], coupled_phi_infos[1],
            final_phi_info[0],
        )
        coupled_stage_records.append({
            "stages": [asdict(info) for info in stage_infos],
            "multipliers": [float(np.asarray(value)) for value in stage_multipliers],
            "final_phi_diagnostics": np.asarray(final_phi_info[0]).tolist(),
            "success": True,
        })
        return finalize_advance(
            result, model,
            (current, stage_states[0], stage_bases[2], stage_states[1], result),
            (stage_rates[0], explicit_rates[0], stage_rates[1], explicit_rates[1], weighted),
            phi_infos,
        )

    full_advance = (
        full_rk4_advance if time_integrator == "rk4" else full_imex_advance
    )
    if imex_split == "coupled-boundary":
        full_advance = coupled_host_advance
    stage_description = (
        "4 operator stages, 4 SOLVAX FGMRES solves"
        if time_integrator == "rk4"
        else "2 explicit operator stages, 2 complete short-wall solves, "
        "4 SOLVAX FGMRES solves"
    )
    advance_action = (
        "lowering shard-local geometry and compiling one complete"
        if advance_execution == "compiled"
        else (
            "compiling reusable staged"
            if advance_execution == "staged-compiled"
            else "running without the outer jax.jit compiled advance for the"
        )
    )
    print(
        f"[simulation] {advance_action} shard_map {time_integrator} advance "
        f"({stage_description})",
        flush=True,
    )
    if phase_timing:
        print(
            "[simulation] phase timing enabled; ordered host markers create "
            "a distinct instrumented executable and add runtime overhead, "
            "but the executable is reused for every step in this run",
            flush=True,
    )
    compile_start = time.perf_counter()
    staged_audit_records: list[dict[str, object]] = []
    advance_out_specs = (
        (
            state_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
        )
        if track_curvature_chain_rule_defect
        else (
            state_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
        )
    )
    sharded_advance = jax.shard_map(
        full_advance,
        mesh=mesh,
        in_specs=(
            state_spec,
            geometry_spec,
            geometry_spec,
            geometry_spec,
            source_spec,
            replicated_spec,
        ),
        out_specs=advance_out_specs,
        check_vma=False,
    )
    if advance_execution == "compiled":
        compiled_advance = jax.jit(sharded_advance).lower(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            zero_source_stages,
            jnp.asarray(start_time, dtype=jnp.float64),
        ).compile()
        print(
            f"[simulation] compiled sharded {time_integrator} advance in "
            f"{time.perf_counter() - compile_start:.3f} s",
            flush=True,
        )
    elif advance_execution == "eager":
        compiled_advance = sharded_advance
        print(
            "[simulation] eager advance ready; outer jax.jit compilation "
            "disabled",
            flush=True,
        )
    else:
        # Keep the existing monolithic compiled/eager paths unchanged.  The
        # staged path is an IMEX short-diagnostic mode: each reusable kernel
        # is compiled once, then the SSP222 algebra is orchestrated with
        # device-side array operations between kernel calls.
        scalar_spec = replicated_spec

        def staged_implicit_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            solve_dt: jax.Array,
            selection_dt: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            updated, increment, _info = (
                model.apply_short_leg_implicit_material_step(
                    local_state,
                    solve_dt=solve_dt,
                    selection_dt=selection_dt,
                    phi_owned=local_state.phi,
                    return_increment=True,
                )
            )
            stage_phi, phi_info = reconstruct_stage_phi(updated, model)
            return updated.replace(phi=stage_phi), increment, phi_info

        staged_implicit_sharded = jax.shard_map(
            staged_implicit_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
                scalar_spec,
                scalar_spec,
            ),
            out_specs=(state_spec, state_spec, replicated_spec),
            check_vma=False,
        )

        def staged_explicit_kernel(
            local_state: FciDrbEBState,
            local_source: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            selection_dt: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            if staged_audit_explicit_ablation == "none":
                rhs = model.evaluate_stage(
                    local_state,
                    source_owned=local_source,
                    phi_owned=local_state.phi,
                    short_leg_selection_dt=selection_dt,
                )
            else:
                rhs, term_fields = model.evaluate_stage(
                    local_state,
                    source_owned=local_source,
                    phi_owned=local_state.phi,
                    return_rhs_term_fields=True,
                    short_leg_selection_dt=selection_dt,
                )
                if staged_audit_explicit_ablation in (
                    "phi-current-pair",
                    "vorticity-advection-phi-current",
                ):
                    vorticity_rhs = (
                        rhs.vorticity
                        - term_fields[
                            RHS_TERM_FIELD_NAMES.index("vorticity"),
                            RHS_TERM_NAMES[
                                RHS_TERM_FIELD_NAMES.index("vorticity")
                            ].index("parallel_current"),
                        ]
                    )
                    if (
                        staged_audit_explicit_ablation
                        == "vorticity-advection-phi-current"
                    ):
                        vorticity_rhs = (
                            vorticity_rhs
                            - term_fields[
                                RHS_TERM_FIELD_NAMES.index("vorticity"),
                                RHS_TERM_NAMES[
                                    RHS_TERM_FIELD_NAMES.index("vorticity")
                                ].index("parallel_advection"),
                            ]
                        )
                    rhs = rhs.replace(
                        Ve=(
                            rhs.Ve
                            - term_fields[
                                RHS_TERM_FIELD_NAMES.index("Ve"),
                                RHS_TERM_NAMES[
                                    RHS_TERM_FIELD_NAMES.index("Ve")
                                ].index("electrostatic"),
                            ]
                        ),
                        vorticity=vorticity_rhs,
                    )
                else:
                    def audit_term(field_name: str, term_name: str):
                        field_index = RHS_TERM_FIELD_NAMES.index(field_name)
                        return term_fields[
                            field_index,
                            RHS_TERM_NAMES[field_index].index(term_name),
                        ]

                    remove_curvature = staged_audit_explicit_ablation in (
                        "curvature", "curvature-parallel-material"
                    )
                    remove_parallel_material = (
                        staged_audit_explicit_ablation in (
                            "parallel-material",
                            "curvature-parallel-material",
                        )
                    )
                    remove_vorticity_parallel_advection = (
                        staged_audit_explicit_ablation
                        == "vorticity-parallel-advection"
                    )
                    density_rhs = rhs.density
                    Te_rhs = rhs.Te
                    Ti_rhs = rhs.Ti
                    Vi_rhs = rhs.Vi
                    Ve_rhs = rhs.Ve
                    vorticity_rhs = rhs.vorticity
                    if remove_curvature:
                        density_rhs = density_rhs - audit_term(
                            "density", "curvature"
                        )
                        Te_rhs = Te_rhs - audit_term("Te", "curvature")
                        Ti_rhs = Ti_rhs - audit_term("Ti", "curvature")
                        vorticity_rhs = vorticity_rhs - audit_term(
                            "vorticity", "curvature"
                        )
                    if remove_vorticity_parallel_advection:
                        vorticity_rhs = vorticity_rhs - audit_term(
                            "vorticity", "parallel_advection"
                        )
                    if remove_parallel_material:
                        for field_name, term_name in (
                            ("density", "parallel_density_flux_divergence"),
                            ("Te", "parallel_advection"),
                            ("Ti", "parallel_advection"),
                            ("Vi", "parallel_self_advection"),
                            ("Ve", "parallel_self_advection"),
                        ):
                            term = audit_term(field_name, term_name)
                            if field_name == "density":
                                density_rhs = density_rhs - term
                            elif field_name == "Te":
                                Te_rhs = Te_rhs - term
                            elif field_name == "Ti":
                                Ti_rhs = Ti_rhs - term
                            elif field_name == "Vi":
                                Vi_rhs = Vi_rhs - term
                            else:
                                Ve_rhs = Ve_rhs - term
                    rhs = rhs.replace(
                        density=density_rhs,
                        Te=Te_rhs,
                        Ti=Ti_rhs,
                        Vi=Vi_rhs,
                        Ve=Ve_rhs,
                        vorticity=vorticity_rhs,
                    )
            rhs = model.project_galerkin_state(rhs)
            mark_operator(rhs)
            return rhs

        staged_explicit_sharded = jax.shard_map(
            staged_explicit_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
                scalar_spec,
            ),
            out_specs=state_spec,
            check_vma=False,
        )

        def staged_phi_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            return reconstruct_stage_phi(local_state, model)

        staged_phi_sharded = jax.shard_map(
            staged_phi_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=(spatial_spec, replicated_spec),
            check_vma=False,
        )

        def staged_finalize_kernel(
            current: FciDrbEBState,
            stage_1: FciDrbEBState,
            stage_2_base: FciDrbEBState,
            stage_2: FciDrbEBState,
            next_state: FciDrbEBState,
            implicit_1: FciDrbEBState,
            explicit_1: FciDrbEBState,
            implicit_2: FciDrbEBState,
            explicit_2: FciDrbEBState,
            weighted_rate: FciDrbEBState,
            gmres_info_1: jax.Array,
            gmres_info_2_base: jax.Array,
            gmres_info_2: jax.Array,
            gmres_info_next: jax.Array,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            return finalize_advance(
                next_state,
                model,
                (current, stage_1, stage_2_base, stage_2, next_state),
                (implicit_1, explicit_1, implicit_2, explicit_2, weighted_rate),
                (gmres_info_1, gmres_info_2_base, gmres_info_2, gmres_info_next),
            )

        staged_finalize_sharded = jax.shard_map(
            staged_finalize_kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                state_spec,
                replicated_spec,
                replicated_spec,
                replicated_spec,
                replicated_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=advance_out_specs,
            check_vma=False,
        )

        def compile_staged_kernel(label: str, sharded_kernel, *example_args):
            """Compile one staged kernel and report lowering/cache separately."""

            lower_start = time.perf_counter()
            lowered = jax.jit(sharded_kernel).lower(*example_args)
            lower_seconds = time.perf_counter() - lower_start
            compile_start = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - compile_start
            print(
                f"[simulation] staged kernel {label}: lowering="
                f"{lower_seconds:.3f} s, compile/cache="
                f"{compile_seconds:.3f} s",
                flush=True,
            )
            return executable

        staged_compile_start = time.perf_counter()
        staged_implicit = compile_staged_kernel(
            "implicit+phi",
            staged_implicit_sharded,
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            jnp.asarray(
                IMEX_SSP222_GAMMA * float(timestep), dtype=jnp.float64
            ),
            jnp.asarray(float(timestep), dtype=jnp.float64),
        )
        staged_explicit = compile_staged_kernel(
            "explicit-rhs",
            staged_explicit_sharded,
            state,
            state.zeros_like(),
            cell_fields,
            map_fields,
            control_volume_fields,
            jnp.asarray(float(timestep), dtype=jnp.float64),
        )
        staged_phi = compile_staged_kernel(
            "standalone-phi",
            staged_phi_sharded,
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        zero_info = jnp.zeros((_PHI_DIAGNOSTIC_WIDTH,), dtype=jnp.float64)
        staged_finalize = compile_staged_kernel(
            "stage-diagnostics",
            staged_finalize_sharded,
            state,
            state,
            state,
            state,
            state,
            state,
            state,
            state,
            state,
            state,
            zero_info,
            zero_info,
            zero_info,
            zero_info,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        staged_audit_select_state = None
        staged_audit_explicit_terms = None
        staged_audit_wall_current = None
        if staged_audit_cells:
            audit_indices = np.asarray(staged_audit_cells, dtype=np.int32)
            audit_u = jnp.asarray(audit_indices[:, 0], dtype=jnp.int32)
            audit_theta = jnp.asarray(audit_indices[:, 1], dtype=jnp.int32)
            audit_eta = jnp.asarray(audit_indices[:, 2], dtype=jnp.int32)

            def select_audit_state(global_state: FciDrbEBState) -> jax.Array:
                packed = jnp.stack(
                    tuple(value for _, value in global_state.field_items()),
                    axis=0,
                )
                return jnp.transpose(
                    packed[:, audit_u, audit_theta, audit_eta], (1, 0)
                )

            staged_audit_select_state = compile_staged_kernel(
                "audit-state-gather",
                select_audit_state,
                state,
            )

            def staged_explicit_term_audit_kernel(
                local_state: FciDrbEBState,
                local_source: FciDrbEBState,
                cell_fields_owned: jax.Array,
                map_fields_owned: jax.Array,
                control_volume_fields_owned: jax.Array,
                selection_dt: jax.Array,
            ):
                model = build_local_model(
                    cell_fields_owned,
                    map_fields_owned,
                    control_volume_fields_owned,
                )
                (
                    rhs,
                    term_fields,
                    curvature_component_fields,
                    parallel_material_component_fields,
                ) = model.evaluate_stage(
                    local_state,
                    source_owned=local_source,
                    phi_owned=local_state.phi,
                    return_rhs_term_fields=True,
                    return_curvature_component_fields=True,
                    return_parallel_material_component_fields=True,
                    short_leg_selection_dt=selection_dt,
                )
                rhs = model.project_galerkin_state(rhs)
                packed_rhs = jnp.stack(
                    tuple(getattr(rhs, name) for name in RHS_TERM_FIELD_NAMES),
                    axis=0,
                )
                selected_rhs = jnp.transpose(
                    packed_rhs[:, audit_u, audit_theta, audit_eta], (1, 0)
                )
                selected_terms = jnp.transpose(
                    term_fields[:, :, audit_u, audit_theta, audit_eta],
                    (2, 0, 1),
                )
                selected_curvature_components = jnp.transpose(
                    curvature_component_fields[
                        :, :, audit_u, audit_theta, audit_eta
                    ],
                    (2, 0, 1),
                )
                selected_parallel_material_components = jnp.transpose(
                    parallel_material_component_fields[
                        :, :, audit_u, audit_theta, audit_eta
                    ],
                    (2, 1, 0),
                )
                return (
                    selected_rhs,
                    selected_terms,
                    selected_curvature_components,
                    selected_parallel_material_components,
                )

            staged_explicit_term_audit_sharded = jax.shard_map(
                staged_explicit_term_audit_kernel,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    scalar_spec,
                ),
                out_specs=(
                    replicated_spec,
                    replicated_spec,
                    replicated_spec,
                    replicated_spec,
                ),
                check_vma=False,
            )
            staged_audit_explicit_terms = compile_staged_kernel(
                "audit-explicit-term-lanes",
                staged_explicit_term_audit_sharded,
                state,
                state.zeros_like(),
                cell_fields,
                map_fields,
                control_volume_fields,
                jnp.asarray(float(timestep), dtype=jnp.float64),
            )

            def staged_wall_current_audit_kernel(
                local_state: FciDrbEBState,
                cell_fields_owned: jax.Array,
                map_fields_owned: jax.Array,
                control_volume_fields_owned: jax.Array,
                selection_dt: jax.Array,
            ):
                model = build_local_model(
                    cell_fields_owned,
                    map_fields_owned,
                    control_volume_fields_owned,
                )
                (
                    raw_states,
                    effective_states,
                    currents,
                    particle_fluxes,
                    metadata,
                    current_divergences,
                    leg_lengths,
                ) = model.parallel_wall_current_diagnostics(
                    local_state,
                    phi_owned=local_state.phi,
                    selection_dt=selection_dt,
                )

                def select_directional(values):
                    return jnp.transpose(
                        values[:, :, audit_u, audit_theta, audit_eta],
                        (2, 1, 0),
                    )

                return (
                    select_directional(raw_states),
                    select_directional(effective_states),
                    select_directional(currents),
                    select_directional(particle_fluxes),
                    select_directional(metadata),
                    jnp.transpose(
                        current_divergences[
                            :, audit_u, audit_theta, audit_eta
                        ],
                        (1, 0),
                    ),
                    jnp.transpose(
                        leg_lengths[:, audit_u, audit_theta, audit_eta],
                        (1, 0),
                    ),
                )

            staged_wall_current_audit_sharded = jax.shard_map(
                staged_wall_current_audit_kernel,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    scalar_spec,
                ),
                out_specs=(replicated_spec,) * 7,
                check_vma=False,
            )
            staged_audit_wall_current = compile_staged_kernel(
                "audit-wall-current",
                staged_wall_current_audit_sharded,
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                jnp.asarray(float(timestep), dtype=jnp.float64),
            )
        print(
            "[simulation] compiled staged IMEX kernels (implicit+phi, "
            "explicit, phi, diagnostics) in "
            f"{time.perf_counter() - staged_compile_start:.3f} s",
            flush=True,
        )

        def staged_execute_advance(*advance_args):
            (
                current,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                source_stages,
                _current_time,
            ) = advance_args
            audit_active = staged_audit_select_state is not None
            dt_dynamic = jnp.asarray(float(timestep), dtype=jnp.float64)
            gamma_dt = jnp.asarray(IMEX_SSP222_GAMMA, dtype=jnp.float64) * dt_dynamic

            stage_1, increment_1, gmres_info_1 = staged_implicit(
                current,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                gamma_dt,
                dt_dynamic,
            )
            _validate_staged_phi_solver_diagnostics(gmres_info_1, "imex1")
            implicit_1 = increment_1.map_fields(
                lambda value: value / gamma_dt
            )
            source_1 = source_stages.replace(
                density=source_stages.density[0], phi=source_stages.phi[0],
                Te=source_stages.Te[0], Ti=source_stages.Ti[0],
                Vi=source_stages.Vi[0], Ve=source_stages.Ve[0],
                vorticity=source_stages.vorticity[0],
            )
            source_2 = source_stages.replace(
                density=source_stages.density[1], phi=source_stages.phi[1],
                Te=source_stages.Te[1], Ti=source_stages.Ti[1],
                Vi=source_stages.Vi[1], Ve=source_stages.Ve[1],
                vorticity=source_stages.vorticity[1],
            )
            explicit_1 = staged_explicit(
                stage_1,
                source_1,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                dt_dynamic,
            )
            explicit_probe_1 = term_fields_1 = curvature_components_1 = None
            parallel_material_components_1 = None
            wall_current_1 = None
            if audit_active:
                (
                    explicit_probe_1,
                    term_fields_1,
                    curvature_components_1,
                    parallel_material_components_1,
                ) = staged_audit_explicit_terms(
                        stage_1,
                        source_1,
                        cell_fields_owned,
                        map_fields_owned,
                        control_volume_fields_owned,
                        dt_dynamic,
                    )
                wall_current_1 = staged_audit_wall_current(
                    stage_1,
                    cell_fields_owned,
                    map_fields_owned,
                    control_volume_fields_owned,
                    dt_dynamic,
                )
            stage_2_base_before_phi = current.axpy(
                explicit_1, scale=dt_dynamic
            ).axpy(
                implicit_1,
                scale=(1.0 - 2.0 * IMEX_SSP222_GAMMA) * dt_dynamic,
            )
            stage_2_base_phi, gmres_info_2_base = staged_phi(
                stage_2_base_before_phi,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            _validate_staged_phi_solver_diagnostics(
                gmres_info_2_base, "stage2-base"
            )
            stage_2_base = stage_2_base_before_phi.replace(phi=stage_2_base_phi)
            stage_2, increment_2, gmres_info_2 = staged_implicit(
                stage_2_base,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                gamma_dt,
                dt_dynamic,
            )
            _validate_staged_phi_solver_diagnostics(gmres_info_2, "imex2")
            implicit_2 = increment_2.map_fields(
                lambda value: value / gamma_dt
            )
            explicit_2 = staged_explicit(
                stage_2,
                source_2,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                dt_dynamic,
            )
            explicit_probe_2 = term_fields_2 = curvature_components_2 = None
            parallel_material_components_2 = None
            wall_current_2 = None
            if audit_active:
                (
                    explicit_probe_2,
                    term_fields_2,
                    curvature_components_2,
                    parallel_material_components_2,
                ) = staged_audit_explicit_terms(
                        stage_2,
                        source_2,
                        cell_fields_owned,
                        map_fields_owned,
                        control_volume_fields_owned,
                        dt_dynamic,
                    )
                wall_current_2 = staged_audit_wall_current(
                    stage_2,
                    cell_fields_owned,
                    map_fields_owned,
                    control_volume_fields_owned,
                    dt_dynamic,
                )
            weighted_rate = explicit_1.axpy(explicit_2, scale=1.0).axpy(
                implicit_1, scale=1.0
            ).axpy(implicit_2, scale=1.0).map_fields(
                lambda value: 0.5 * value
            )
            next_state = current.axpy(weighted_rate, scale=dt_dynamic)
            next_phi, gmres_info_next = staged_phi(
                next_state,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            _validate_staged_phi_solver_diagnostics(gmres_info_next, "next")
            next_state = next_state.replace(phi=next_phi)
            if audit_active:
                audit_stage_names = (
                    "current",
                    "implicit_rate_1",
                    "stage_1",
                    "explicit_rate_1",
                    "stage_2_base_before_phi",
                    "stage_2_base",
                    "implicit_rate_2",
                    "stage_2",
                    "explicit_rate_2",
                    "weighted_rate",
                    "final",
                )
                audit_stage_states = (
                    current,
                    implicit_1,
                    stage_1,
                    explicit_1,
                    stage_2_base_before_phi,
                    stage_2_base,
                    implicit_2,
                    stage_2,
                    explicit_2,
                    weighted_rate,
                    next_state,
                )
                selected_stages = tuple(
                    staged_audit_select_state(stage)
                    for stage in audit_stage_states
                )
                jax.block_until_ready(
                    (
                        selected_stages,
                        explicit_probe_1,
                        term_fields_1,
                        explicit_probe_2,
                        term_fields_2,
                        curvature_components_1,
                        curvature_components_2,
                        parallel_material_components_1,
                        parallel_material_components_2,
                        wall_current_1,
                        wall_current_2,
                    )
                )
                staged_audit_records.append(
                    {
                        "start_time": float(np.asarray(_current_time)),
                        "stage_names": audit_stage_names,
                        "stage_values": np.stack(
                            tuple(np.asarray(value, dtype=np.float64)
                                  for value in selected_stages),
                            axis=0,
                        ),
                        "explicit_probe_rhs": np.stack(
                            (
                                np.asarray(explicit_probe_1, dtype=np.float64),
                                np.asarray(explicit_probe_2, dtype=np.float64),
                            ),
                            axis=0,
                        ),
                        "explicit_term_values": np.stack(
                            (
                                np.asarray(term_fields_1, dtype=np.float64),
                                np.asarray(term_fields_2, dtype=np.float64),
                            ),
                            axis=0,
                        ),
                        "curvature_component_values": np.stack(
                            (
                                np.asarray(
                                    curvature_components_1, dtype=np.float64
                                ),
                                np.asarray(
                                    curvature_components_2, dtype=np.float64
                                ),
                            ),
                            axis=0,
                        ),
                        "parallel_material_component_values": np.stack(
                            (
                                np.asarray(
                                    parallel_material_components_1,
                                    dtype=np.float64,
                                ),
                                np.asarray(
                                    parallel_material_components_2,
                                    dtype=np.float64,
                                ),
                            ),
                            axis=0,
                        ),
                        "wall_current_values": tuple(
                            np.stack(
                                (
                                    np.asarray(first, dtype=np.float64),
                                    np.asarray(second, dtype=np.float64),
                                ),
                                axis=0,
                            )
                            for first, second in zip(
                                wall_current_1, wall_current_2, strict=True
                            )
                        ),
                        "gmres_stage_diagnostics": np.stack(
                            tuple(
                                np.asarray(value, dtype=np.float64)
                                for value in (
                                    gmres_info_1,
                                    gmres_info_2_base,
                                    gmres_info_2,
                                    gmres_info_next,
                                )
                            ),
                            axis=0,
                        ),
                    }
                )
            return staged_finalize(
                current,
                stage_1,
                stage_2_base,
                stage_2,
                next_state,
                implicit_1,
                explicit_1,
                implicit_2,
                explicit_2,
                weighted_rate,
                gmres_info_1,
                gmres_info_2_base,
                gmres_info_2,
                gmres_info_next,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )

        compiled_advance = staged_execute_advance

    # The coupled solver is intentionally a host/eager transaction: Python
    # Newton control must never be traced through shard_map or the fused path.
    if imex_split == "coupled-boundary":
        compiled_advance = coupled_host_advance

    rhs_term_inspection = None
    if track_rhs_terms:
        radial_centers_owned = jnp.asarray(
            global_geometry.grid.x.centers, dtype=jnp.float64
        ).reshape((-1, 1, 1))

        def inspect_rhs_terms(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ) -> jax.Array:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            _, term_fields = model.evaluate_stage(
                local_state,
                phi_owned=local_state.phi,
                return_rhs_term_fields=True,
            )
            if model.control_volume_geometry is not None:
                cells = model.control_volume_geometry.cells
                term_fields = jax.vmap(
                    jax.vmap(
                        lambda value: expand_local_control_volume_owner_field(
                            value, cells
                        )
                    )
                )(term_fields)
            jacobian = jnp.asarray(
                model.geometry.cell_metric.J_owned, dtype=jnp.float64
            )
            spatial_axes = tuple(range(term_fields.ndim - 3, term_fields.ndim))
            global_weight = jax.lax.psum(
                jnp.sum(jacobian), ("x", "y", "z")
            )
            weighted_mean = jax.lax.psum(
                jnp.sum(term_fields * jacobian, axis=spatial_axes),
                ("x", "y", "z"),
            ) / global_weight
            weighted_rms = jnp.sqrt(
                jax.lax.psum(
                    jnp.sum(term_fields * term_fields * jacobian, axis=spatial_axes),
                    ("x", "y", "z"),
                )
                / global_weight
            )
            maximum_absolute = jax.lax.pmax(
                jnp.max(jnp.abs(term_fields), axis=spatial_axes),
                ("x", "y", "z"),
            )
            weighted_radial_moment = jax.lax.psum(
                jnp.sum(
                    term_fields * jacobian * radial_centers_owned,
                    axis=spatial_axes,
                ),
                ("x", "y", "z"),
            ) / global_weight
            return jnp.stack(
                (
                    weighted_mean,
                    weighted_rms,
                    maximum_absolute,
                    weighted_radial_moment,
                ),
                axis=0,
            )

        rhs_compile_start = time.perf_counter()
        rhs_term_inspection_sharded = jax.shard_map(
            inspect_rhs_terms,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=replicated_spec,
            check_vma=False,
        )
        if advance_execution in ("compiled", "staged-compiled"):
            rhs_term_inspection = jax.jit(
                rhs_term_inspection_sharded
            ).lower(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
            ).compile()
            print(
                "[simulation] compiled all-equation RHS term inspection in "
                f"{time.perf_counter() - rhs_compile_start:.3f} s",
                flush=True,
            )
        else:
            rhs_term_inspection = rhs_term_inspection_sharded
            print(
                "[simulation] all-equation RHS term inspection will execute "
                "eagerly",
                flush=True,
            )

    def inspect_rhs_terms_host(current_state: FciDrbEBState) -> np.ndarray:
        if rhs_term_inspection is None:
            raise RuntimeError("RHS term inspection was not compiled")
        with jax.disable_jit(advance_execution == "eager"):
            result = rhs_term_inspection(
                current_state,
                cell_fields,
                map_fields,
                control_volume_fields,
            )
        jax.block_until_ready(result)
        return np.asarray(result, dtype=np.float64)

    # Periodic checkpoints can be state-only.  Do not force compilation of
    # the comparatively expensive spatial inspection path unless diagnostics
    # or explicitly scheduled diagnostic snapshots already require it.
    wall_term_count = 4
    inspection_enabled = bool(diagnostic_every > 0 or snapshot_times)
    inspection = None
    if inspection_enabled:
        term_spec = P(None, "x", "y", "z")
        wall_spec = P(None, None, "x", "y", "z")
        owned_shape = tuple(int(value) for value in domain.layout.owned_shape)
        global_shape = tuple(int(value) for value in sharded_geometry.global_shape)
        halo_width = int(domain.layout.halo_width)
        def inspect_state(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
            )
            polarization_residual = model.polarization_residual(
                local_state,
                phi_owned=local_state.phi,
            )
            diagnostic_local_state = diagnostic_state(local_state, model.control_volume_geometry)
            face_bc = model._face_bcs(local_state)
            state_halo = prepare_local_fci_drb_eb_state(
                diagnostic_local_state,
                domain,
                face_bc=face_bc,
                halo_exchange=model.halo_exchange,
                topology_filler=model.topology_filler,
                physical_ghost_filler=model.physical_ghost_filler,
            )

            local_ve = jnp.abs(diagnostic_local_state.Ve)
            local_max = jnp.max(local_ve)
            local_flat = jnp.argmax(local_ve)
            local_coords = jnp.unravel_index(local_flat, owned_shape)
            shard_indices = tuple(
                jax.lax.axis_index(axis_name) for axis_name in ("x", "y", "z")
            )
            global_coords = tuple(
                local_coords[axis]
                + shard_indices[axis] * owned_shape[axis]
                for axis in range(3)
            )
            local_linear = (
                global_coords[0] * global_shape[1] * global_shape[2]
                + global_coords[1] * global_shape[2]
                + global_coords[2]
            )
            global_max = jax.lax.pmax(local_max, ("x", "y", "z"))
            candidate = jnp.where(
                local_max == global_max,
                local_linear,
                jnp.asarray(np.prod(global_shape), dtype=jnp.int64),
            )
            global_index = jax.lax.pmin(candidate, ("x", "y", "z"))

            global_coordinates = tuple(
                jnp.arange(owned_shape[axis], dtype=jnp.int64)
                + shard_indices[axis] * owned_shape[axis]
                for axis in range(3)
            )
            # Mark only cells adjacent to runtime physical sides.  The
            # runtime predicates are SPMD-safe scalar JAX values: on a
            # toroidal mesh this selects only the physical outer radial side,
            # while the axis and periodic theta/eta seams remain bulk.
            wall_masks = jnp.zeros(owned_shape, dtype=bool)
            for axis in range(3):
                coordinate_shape = [1, 1, 1]
                coordinate_shape[axis] = owned_shape[axis]
                coordinates = global_coordinates[axis].reshape(coordinate_shape)
                lower_physical = jnp.asarray(
                    domain.runtime_has_physical_lower(axis),
                    dtype=bool,
                )
                upper_physical = jnp.asarray(
                    domain.runtime_has_physical_upper(axis),
                    dtype=bool,
                )
                side_mask = (
                    (coordinates < wall_term_count) & lower_physical
                ) | (
                    (coordinates >= global_shape[axis] - wall_term_count)
                    & upper_physical
                )
                wall_masks = wall_masks | side_mask
            bulk_masks = ~wall_masks
            high_pass = []
            for _, field in state_halo.field_items():
                wall_energy = jnp.asarray(0.0, dtype=jnp.float64)
                bulk_energy = jnp.asarray(0.0, dtype=jnp.float64)
                for axis in range(3):
                    center = [slice(halo_width, halo_width + size) for size in owned_shape]
                    lower = list(center)
                    upper = list(center)
                    lower[axis] = slice(halo_width - 1, halo_width - 1 + owned_shape[axis])
                    upper[axis] = slice(halo_width + 1, halo_width + 1 + owned_shape[axis])
                    center_values = field[tuple(center)]
                    second_difference = field[tuple(upper)] - 2.0 * center_values + field[tuple(lower)]
                    wall_energy = wall_energy + jnp.sum(
                        jnp.square(second_difference) * wall_masks
                    )
                    bulk_energy = bulk_energy + jnp.sum(
                        jnp.square(second_difference) * bulk_masks
                    )
                high_pass.append(jnp.stack((wall_energy, bulk_energy)))
            high_pass = jnp.stack(high_pass)
            for axis_name in ("x", "y", "z"):
                high_pass = jax.lax.psum(high_pass, axis_name)

            wall_ghost_fields = []
            for _, field in state_halo.field_items():
                owned = jnp.zeros(owned_shape, dtype=jnp.float64)
                lower_x = field[halo_width - 1, halo_width:halo_width + owned_shape[1], halo_width:halo_width + owned_shape[2]]
                upper_x = field[halo_width + owned_shape[0], halo_width:halo_width + owned_shape[1], halo_width:halo_width + owned_shape[2]]
                lower_y = field[halo_width:halo_width + owned_shape[0], halo_width - 1, halo_width:halo_width + owned_shape[2]]
                upper_y = field[halo_width:halo_width + owned_shape[0], halo_width + owned_shape[1], halo_width:halo_width + owned_shape[2]]
                x_lower_values = jnp.zeros_like(owned).at[0, :, :].set(lower_x)
                x_upper_values = jnp.zeros_like(owned).at[-1, :, :].set(upper_x)
                y_lower_values = jnp.zeros_like(owned).at[:, 0, :].set(lower_y)
                y_upper_values = jnp.zeros_like(owned).at[:, -1, :].set(upper_y)
                nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
                x_lower_values = jnp.where(jax.lax.axis_index("x") == 0, x_lower_values, nan)
                x_upper_values = jnp.where(jax.lax.axis_index("x") == sharded_geometry.shard_counts[0] - 1, x_upper_values, nan)
                y_lower_values = jnp.where(jax.lax.axis_index("y") == 0, y_lower_values, nan)
                y_upper_values = jnp.where(jax.lax.axis_index("y") == sharded_geometry.shard_counts[1] - 1, y_upper_values, nan)
                wall_ghost_fields.append(
                    jnp.stack((x_lower_values, x_upper_values, y_lower_values, y_upper_values))
                )
            wall_ghost_fields = jnp.stack(wall_ghost_fields)
            inspection_diagnostics = jnp.concatenate(
                (jnp.asarray((global_max, global_index), dtype=jnp.float64), high_pass.reshape(-1))
            )
            if snapshot_term_fields:
                _, term_fields = model.evaluate_stage(
                    local_state,
                    phi_owned=local_state.phi,
                    return_term_fields=True,
                )
                return (
                    inspection_diagnostics,
                    term_fields,
                    wall_ghost_fields,
                    polarization_residual,
                )
            return inspection_diagnostics, wall_ghost_fields, polarization_residual

        inspection_out_specs = (
            (replicated_spec, term_spec, wall_spec, spatial_spec)
            if snapshot_term_fields
            else (replicated_spec, wall_spec, spatial_spec)
        )
        inspection_sharded = jax.shard_map(
            inspect_state,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
            ),
            out_specs=inspection_out_specs,
            check_vma=False,
        )
        if advance_execution in ("compiled", "staged-compiled"):
            inspection = jax.jit(inspection_sharded).lower(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
            ).compile()
            inspection_action = "compiled"
        else:
            inspection = inspection_sharded
            inspection_action = "eager"
        print(
            f"[simulation] {inspection_action} snapshot/grid-scale "
            f"inspection path (terms={'on' if snapshot_term_fields else 'off'})",
            flush=True,
        )

    def inspect_host(current_state: FciDrbEBState):
        if inspection is None:
            return None
        with jax.disable_jit(advance_execution == "eager"):
            result = inspection(
                current_state,
                cell_fields,
                map_fields,
                control_volume_fields,
            )
        jax.block_until_ready(result)
        return tuple(np.asarray(value) for value in result)

    base_output_payload = {
        "u": np.asarray(global_geometry.grid.x.centers, dtype=np.float64),
        "v": np.asarray(global_geometry.grid.y.centers, dtype=np.float64),
        "theta": np.asarray(global_geometry.grid.y.centers, dtype=np.float64),
        "eta": np.asarray(global_geometry.grid.z.centers, dtype=np.float64),
        "cartesian": np.asarray(cell_positions, dtype=np.float64),
        "nfp": np.asarray(int(nfp), dtype=np.int32),
        "simulated_field_periods": np.asarray(int(nfp), dtype=np.int32),
        "toroidal_extent": np.asarray(2.0 * np.pi, dtype=np.float64),
        "geometry_metadata_json": np.asarray(
            json.dumps(geometry_metadata, sort_keys=True, default=str)
        ),
        "shard_counts": np.asarray(sharded_geometry.shard_counts, dtype=np.int32),
        "periodic_axes": np.asarray(
            (run_metadata or {}).get("periodic_axes", PERIODIC_AXES), dtype=bool
        ),
        "axis_regular_axes": np.asarray(
            (run_metadata or {}).get("axis_regular_axes", AXIS_REGULAR_AXES),
            dtype=bool,
        ),
        "phi_solver_space": np.asarray(solver_space),
        "parallel_operator_scheme": np.asarray(str(parallel_operator_scheme)),
        "fci_trace_substeps": np.asarray(
            int((run_metadata or {}).get("fci_trace_substeps", 64)),
            dtype=np.int64,
        ),
        "axis_treatment": np.asarray(
            str((run_metadata or {}).get("axis_treatment", "none"))
        ),
        "angular_owner_count": np.asarray(
            int(
                -1
                if (run_metadata or {}).get("angular_owner_count") is None
                else (run_metadata or {})["angular_owner_count"]
            ),
            dtype=np.int64,
        ),
        "angular_alias_count": np.asarray(
            int(
                -1
                if (run_metadata or {}).get("angular_alias_count") is None
                else (run_metadata or {})["angular_alias_count"]
            ),
            dtype=np.int64,
        ),
        "angular_owner_profile": np.asarray(
            str((run_metadata or {}).get("angular_owner_profile", "none"))
        ),
        "angular_group_sizes": np.asarray(
            (run_metadata or {}).get("angular_group_sizes") or (), dtype=np.int32
        ),
        "angular_profile_safety_ratio": np.asarray(
            float((run_metadata or {}).get("angular_profile_safety_ratio") or -1.0), dtype=np.float64
        ),
    }
    output_topology = topology_descriptor(
        str((run_metadata or {}).get("topology", "square"))
    )
    base_output_payload.update(
        {
            "topology": np.asarray(output_topology.name),
            "coordinate_names_json": np.asarray(
                json.dumps(output_topology.coordinate_names)
            ),
            "logical_extents": np.asarray(
                output_topology.logical_extents, dtype=np.float64
            ),
            "metric_mesh_shape": np.asarray(
                (run_metadata or {}).get("metric_mesh_shape") or (-1, -1, -1),
                dtype=np.int64,
            ),
            "metric_radial_degree": np.asarray(
                int((run_metadata or {}).get("metric_radial_degree", -1)),
                dtype=np.int64,
            ),
            "metric_poloidal_modes": np.asarray(
                int((run_metadata or {}).get("metric_poloidal_modes", -1)),
                dtype=np.int64,
            ),
            "metric_toroidal_modes": np.asarray(
                int((run_metadata or {}).get("metric_toroidal_modes", -1)),
                dtype=np.int64,
            ),
            "eta_projection_iterations": np.asarray(
                int((run_metadata or {}).get("eta_projection_iterations", -1)),
                dtype=np.int64,
            ),
        }
    )
    base_output_payload.update(_snapshot_metric_payload(global_geometry))
    snapshot_schedule = tuple(sorted(float(value) for value in snapshot_times))
    snapshot_root = Path(snapshot_dir) if snapshot_dir is not None else output_path.parent
    metadata = dict(run_metadata or {})
    coupled_stage_records: list[dict[str, object]] = []
    metadata["coupled_stage_diagnostics"] = coupled_stage_records
    metadata["imex_split"] = str(imex_split)
    metadata["coupled_residual_execution"] = (
        "isolated-jit-kernel" if imex_split == "coupled-boundary" else "historical"
    )
    if imex_split == "coupled-boundary":
        metadata["coupled_phi_diagnostic_semantics"] = {
            "slot_0": "coupled_linear_iterations",
            "slot_1": "polarization_relative_residual",
            "slot_4": "compatibility_multiplier",
            "slot_5": "raw_compatibility_defect",
            "slot_6": "gauge_residual",
            "slot_8": "polarization_residual_l2",
            "slot_9": "raw_polarization_rhs_l2",
        }
    metadata.update(
        {
            "time_integrator": str(time_integrator),
            "resolution": list(sharded_geometry.global_shape),
            "shard_counts": list(sharded_geometry.shard_counts),
            "phi_solver_space": solver_space,
            "parallel_operator_scheme": str(parallel_operator_scheme),
            "fci_trace_substeps": int(
                (run_metadata or {}).get("fci_trace_substeps", 64)
            ),
            "snapshot_term_fields": bool(snapshot_term_fields),
            "checkpoint_every": int(checkpoint_every),
            "track_rhs_terms": bool(track_rhs_terms),
            "rhs_replay_execution": str(rhs_replay_execution),
            "field_names": list(initial_state.field_names()),
            "ve_term_names": [
                "poisson_bracket",
                "parallel_self_advection",
                "collision",
                "electrostatic",
                "electron_pressure",
                "thermal_force",
                "perpendicular_diffusion",
                "parallel_viscosity",
                "characteristic_leg_upwind",
            ],
            "rhs_term_field_names": list(RHS_TERM_FIELD_NAMES),
            "rhs_term_names": {
                field: list(names)
                for field, names in zip(
                    RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES, strict=True
                )
            },
            "rhs_term_statistic_names": [
                "volume_weighted_mean",
                "volume_weighted_rms",
                "maximum_absolute",
                "volume_weighted_radial_moment",
            ],
            "parallel_coefficient_names": [
                "parallel_density_safe",
                "parallel_inverse_density",
                "parallel_electron_pressure",
                "parallel_total_pressure",
                "parallel_current",
                "parallel_density_Ve_flux",
                "parallel_Te_compression_multiplier",
                "parallel_Ti_compression_multiplier",
                "parallel_Ve_pressure_multiplier",
                "parallel_vorticity_current_multiplier",
            ],
        }
    )
    if imex_split == "coupled-boundary":
        metadata.update({
            "coupled_implicit_terms": [
                "complete-parallel-material-and-geometric",
                "generalized-electron-force-phi-plus-tau-Ti",
                "vorticity-parallel-advection-current",
                "enabled-parallel-diffusion-and-collisions",
                "physical-minus-reference-perpendicular-PB-diffusion-curvature",
            ],
            "coupled_explicit_terms": [
                "plasma-support-perpendicular-bulk", "prescribed-sources",
            ],
            "historical_local_wall_update_applied": False,
            "parallel_short_leg_explicit_energy_pair": None,
            "parallel_short_leg_time_handoff": "inactive-coupled-stage",
            "advance_execution_kernel_layout": (
                "host-eager-newton-isolated-residual-fgmres",
            ),
        })

    def save_snapshot(
        requested_time: float,
        actual_time: float,
        step: int,
        *,
        inspected: tuple[np.ndarray, ...] | None = None,
        phi_solver_diagnostics: np.ndarray | None = None,
        failure_reason: str | None = None,
        periodic_checkpoint: bool = False,
    ) -> None:
        if inspected is None:
            inspected = inspect_host(state)
        snapshot_state = materialized_state(state)
        payload = dict(base_output_payload)
        payload.update(
            {
                name: np.asarray(value, dtype=np.float64)
                for name, value in snapshot_state.field_items()
            }
        )
        payload.update(
            _snapshot_parallel_coefficient_payload(
                payload,
                Bmag=payload["Bmag"],
                tau=float(parameters.tau),
                mi_over_me=float(parameters.mi_over_me),
            )
        )
        payload["time"] = np.asarray(actual_time, dtype=np.float64)
        payload["requested_snapshot_time"] = np.asarray(requested_time, dtype=np.float64)
        payload["step"] = np.asarray(step, dtype=np.int64)
        snapshot_metadata = dict(metadata)
        snapshot_metadata.update(
            {
                "actual_time": actual_time,
                "requested_time": requested_time,
                "step": step,
                "failure_reason": failure_reason,
                "diagnostic_definition": (
                    "sum of squared three-point second differences; wall is "
                    f"within {wall_term_count} cells of runtime physical sides"
                ),
            }
        )
        if phi_solver_diagnostics is not None:
            phi_solver_diagnostics = np.asarray(
                phi_solver_diagnostics, dtype=np.float64
            )
            payload["phi_solver_diagnostics"] = phi_solver_diagnostics
            if phi_solver_diagnostics.ndim == 1:
                phi_solver_diagnostics = phi_solver_diagnostics[None, :]
            snapshot_metadata["phi_solver_diagnostics"] = {
                "field_names": list(_PHI_SOLVER_DIAGNOSTIC_NAMES),
                "stage_values": phi_solver_diagnostics.tolist(),
                "max_abs": {
                    name: float(np.max(np.abs(phi_solver_diagnostics[:, index])))
                    for index, name in enumerate(_PHI_SOLVER_DIAGNOSTIC_NAMES)
                },
            }
            print(
                "[snapshot] phi solver: "
                + ", ".join(
                    (
                        f"{name}="
                        f"{snapshot_metadata['phi_solver_diagnostics']['max_abs'][name]:.3e}"
                    )
                    for name in _PHI_SOLVER_DIAGNOSTIC_NAMES[4:]
                ),
                flush=True,
            )
        if inspected is not None:
            if snapshot_term_fields:
                (
                    diagnostic_values,
                    term_fields,
                    wall_ghost_fields,
                    polarization_residual,
                ) = inspected
                payload["Ve_rhs_terms"] = np.asarray(term_fields, dtype=np.float64)
            else:
                (
                    diagnostic_values,
                    wall_ghost_fields,
                    polarization_residual,
                ) = inspected
            payload["wall_ghost_states"] = wall_ghost_fields.astype(np.float64)
            payload["polarization_residual"] = _materialize_owner_array(
                np.asarray(polarization_residual, dtype=np.float64)[None, ...],
                owner_host_geometry,
            )[0]
            payload["grid_scale_diagnostics"] = diagnostic_values.astype(np.float64)
            high_pass_values = diagnostic_values[2:].reshape(7, 2)
            snapshot_metadata["grid_scale_diagnostics"] = {
                "max_abs_Ve": float(diagnostic_values[0]),
                "max_abs_Ve_global_linear_index": int(diagnostic_values[1]),
                "high_pass_wall_bulk_sum_sq": {
                    name: [float(values[0]), float(values[1])]
                    for name, values in zip(
                        initial_state.field_names(), high_pass_values, strict=True
                    )
                },
            }
            snapshot_metadata["polarization_residual"] = {
                "maximum_absolute": float(
                    np.max(np.abs(payload["polarization_residual"]))
                ),
                "root_mean_square": float(
                    np.sqrt(np.mean(np.square(payload["polarization_residual"])))
                ),
            }
            print(
                "[snapshot] grid-scale: "
                f"max|Ve|={diagnostic_values[0]:.6e} "
                f"global_index={int(diagnostic_values[1])}; "
                + ", ".join(
                    f"{name}=({values[0]:.3e},{values[1]:.3e})"
                    for name, values in zip(
                        initial_state.field_names(), high_pass_values, strict=True
                    )
                )
                + " [wall,bulk]",
                flush=True,
            )
        payload["run_metadata_json"] = np.asarray(json.dumps(snapshot_metadata, sort_keys=True))
        if failure_reason is not None:
            checkpoint_path = output_path.with_name(
                f"{output_path.stem}.failure_step{int(step):06d}.npz"
            )
        elif periodic_checkpoint:
            checkpoint_path = snapshot_root / (
                f"{output_path.stem}.checkpoint_step{int(step):06d}.npz"
            )
        else:
            checkpoint_path = snapshot_root / (
                f"{output_path.stem}.snapshot_t"
                f"{_format_snapshot_time(requested_time)}.npz"
            )
        _atomic_save_npz(checkpoint_path, **payload)
        if periodic_checkpoint:
            print(
                f"[checkpoint] saved t={actual_time:.6e}, step={step}: "
                f"{checkpoint_path}",
                flush=True,
            )
        elif failure_reason is None:
            print(
                f"[snapshot] saved requested t={requested_time:.6e}, "
                f"actual t={actual_time:.6e}, step={step}: {checkpoint_path}",
                flush=True,
            )
        else:
            print(
                f"[failure-checkpoint] saved t={actual_time:.6e}, "
                f"step={step}, reason={failure_reason}: {checkpoint_path}",
                flush=True,
            )

    initial_output_state = materialized_state(state)
    history: dict[str, list[np.ndarray]] = {
        name: [np.asarray(value, dtype=history_numpy_dtype)]
        for name, value in initial_output_state.field_items()
    }
    saved_times = [float(start_time)]
    rhs_term_statistics_history = (
        [inspect_rhs_terms_host(state)] if track_rhs_terms else []
    )
    next_snapshot = 0
    while next_snapshot < len(snapshot_schedule) and snapshot_schedule[next_snapshot] <= start_time + 1.0e-14:
        save_snapshot(snapshot_schedule[next_snapshot], float(start_time), 0)
        next_snapshot += 1
    simulation_start = time.perf_counter()
    accumulated_step_seconds = 0.0
    accumulated_operator_seconds = 0.0
    accumulated_gmres_seconds = 0.0
    accumulated_gmres_iterations = 0.0

    def execute_advance(*advance_args):
        try:
            with jax.disable_jit(advance_execution == "eager"):
                return compiled_advance(*advance_args)
        except Exception as exc:
            if imex_split == "coupled-boundary":
                coupled_stage_records.append({
                    "success": False,
                    "failure_reason": repr(exc),
                })
                save_snapshot(
                    step_time, step_time, max(int(step) - 1, 0),
                    failure_reason="coupled-stage-failed",
                )
            raise

    for step in range(1, int(num_steps) + 1):
        step_start = time.perf_counter()
        step_time = float(start_time) + (step - 1) * float(timestep)
        source_stages = source_stages_for_step(step_time)
        if phase_timer is not None:
            phase_timer.begin_step()
        if track_curvature_chain_rule_defect:
            (
                state,
                diagnostics,
                curvature_diagnostics,
                gmres_iterations,
                gmres_stage_diagnostics,
                rk_stage_diagnostics,
            ) = execute_advance(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                source_stages,
                jnp.asarray(
                    step_time,
                    dtype=jnp.float64,
                ),
            )
            jax.block_until_ready(
                (
                    state,
                    diagnostics,
                    curvature_diagnostics,
                    gmres_iterations,
                    gmres_stage_diagnostics,
                    rk_stage_diagnostics,
                )
            )
        else:
            (
                state,
                diagnostics,
                gmres_iterations,
                gmres_stage_diagnostics,
                rk_stage_diagnostics,
            ) = execute_advance(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                source_stages,
                jnp.asarray(
                    step_time,
                    dtype=jnp.float64,
                ),
            )
            jax.block_until_ready(
                (
                    state,
                    diagnostics,
                    gmres_iterations,
                    gmres_stage_diagnostics,
                    rk_stage_diagnostics,
                )
            )
        if owner_host_geometry is not None:
            _assert_owner_sparse(state, owner_host_geometry)
        step_seconds = time.perf_counter() - step_start
        if phase_timer is None:
            operator_seconds = None
            gmres_seconds = None
        else:
            operator_seconds, gmres_seconds = phase_timer.finish_step()
            accumulated_operator_seconds += operator_seconds
            accumulated_gmres_seconds += gmres_seconds
        accumulated_step_seconds += step_seconds
        current_time = float(start_time) + step * float(timestep)

        if step % int(save_every) == 0 or step == int(num_steps):
            saved_times.append(current_time)
            output_state = materialized_state(state)
            for name, value in output_state.field_items():
                history[name].append(
                    np.asarray(value, dtype=history_numpy_dtype)
                )

        diagnostics_host = np.asarray(diagnostics)
        gmres_stage_diagnostics_host = np.asarray(gmres_stage_diagnostics)
        rk_stage_diagnostics_host = np.asarray(rk_stage_diagnostics)
        gmres_iterations_host = float(np.asarray(gmres_iterations))
        gmres_relative_residual_host = float(
            np.max(gmres_stage_diagnostics_host[:, 1])
        )
        gmres_failed_host = bool(
            np.any(gmres_stage_diagnostics_host[:, 2] > 0.5)
        )
        accumulated_gmres_iterations += gmres_iterations_host
        field_names = initial_state.field_names()
        density_index = field_names.index("density")
        Te_index = field_names.index("Te")
        Ti_index = field_names.index("Ti")
        density_min = float(diagnostics_host[density_index, 0])
        density_max = float(diagnostics_host[density_index, 1])
        Te_min = float(diagnostics_host[Te_index, 0])
        Te_max = float(diagnostics_host[Te_index, 1])
        Ti_min = float(diagnostics_host[Ti_index, 0])
        Ti_max = float(diagnostics_host[Ti_index, 1])
        temperature_min = min(Te_min, Ti_min)
        state_diagnostics = _format_state_diagnostics(field_names, diagnostics_host)
        stage_finite = _rk_stage_diagnostics_have_finite_bit(
            rk_stage_diagnostics_host
        )
        stage_density_min = float(
            np.min(rk_stage_diagnostics_host[:, density_index, 0])
        )
        stage_Te_min = float(
            np.min(rk_stage_diagnostics_host[:, Te_index, 0])
        )
        stage_Ti_min = float(
            np.min(rk_stage_diagnostics_host[:, Ti_index, 0])
        )
        if (
            not stage_finite
            or stage_density_min <= 0.0
            or stage_Te_min <= 0.0
            or stage_Ti_min <= 0.0
        ):
            print(
                f"[diagnostics] step={step} invalid "
                f"{time_integrator} stage: "
                f"finite={stage_finite}, n_min={stage_density_min:.6e}, "
                f"Te_min={stage_Te_min:.6e}, Ti_min={stage_Ti_min:.6e}",
                flush=True,
            )
            _print_rk_stage_diagnostics(
                field_names,
                rk_stage_diagnostics_host,
                integrator=time_integrator,
            )
            save_snapshot(
                current_time,
                current_time,
                step,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
                failure_reason=f"invalid-{time_integrator}-stage",
            )
            raise FloatingPointError(
                f"invalid {time_integrator} stage after step {step}"
            )
        if gmres_failed_host:
            stage_text = ", ".join(
                (
                    f"{name}:iters={int(values[0])},"
                    f"relres={values[1]:.3e},accepted={bool(values[3] > 0.5)}"
                )
                for name, values in zip(
                    (
                        ("rk2", "rk3", "rk4", "next")
                        if time_integrator == "rk4"
                        else ("imex1", "stage2-base", "imex2", "next")
                    ),
                    gmres_stage_diagnostics_host,
                    strict=True,
                )
            )
            print(
                f"[diagnostics] step={step} rejected phi inversion: "
                f"{stage_text}; state={state_diagnostics}",
                flush=True,
            )
            _print_rk_stage_diagnostics(
                field_names,
                rk_stage_diagnostics_host,
                integrator=time_integrator,
            )
            save_snapshot(
                current_time,
                current_time,
                step,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
                failure_reason="unaccepted-phi-inversion",
            )
            raise FloatingPointError(
                f"unaccepted phi inversion after step {step}"
            )
        density_finite = np.isfinite(density_min) and np.isfinite(density_max)
        temperature_finite = all(
            np.isfinite(value) for value in (Te_min, Te_max, Ti_min, Ti_max)
        )
        if not density_finite or not temperature_finite:
            print(
                f"[diagnostics] step={step} nonfinite: {state_diagnostics}",
                flush=True,
            )
            _print_rk_stage_diagnostics(
                field_names,
                rk_stage_diagnostics_host,
                integrator=time_integrator,
            )
            save_snapshot(
                current_time,
                current_time,
                step,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
                failure_reason="nonfinite-eb-state",
            )
            raise FloatingPointError(f"nonfinite EB state after step {step}")
        if density_min <= 0.0 or temperature_min <= 0.0:
            print(
                f"[diagnostics] step={step} positivity failure: "
                f"{state_diagnostics}",
                flush=True,
            )
            _print_rk_stage_diagnostics(
                field_names,
                rk_stage_diagnostics_host,
                integrator=time_integrator,
            )
            save_snapshot(
                current_time,
                current_time,
                step,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
                failure_reason="nonpositive-eb-state",
            )
            raise FloatingPointError(
                f"nonpositive density/temperature after step {step}: "
                f"n_min={density_min:.6e}, T_min={temperature_min:.6e}"
            )
        if track_rhs_terms:
            rhs_term_statistics_history.append(inspect_rhs_terms_host(state))
        inspection_host = None
        snapshot_due = (
            next_snapshot < len(snapshot_schedule)
            and snapshot_schedule[next_snapshot] <= current_time + 1.0e-14
        )
        periodic_checkpoint_due = (
            checkpoint_every > 0 and step % int(checkpoint_every) == 0
        )
        if inspection_enabled and (
            (diagnostic_every > 0 and step % int(diagnostic_every) == 0)
            or snapshot_due
            or periodic_checkpoint_due
        ):
            inspection_host = inspect_host(state)
        if diagnostic_every > 0 and step % int(diagnostic_every) == 0:
            print(
                f"[diagnostics] step={step}: {state_diagnostics}",
                flush=True,
            )
            if track_curvature_chain_rule_defect:
                chain_rule_host = np.asarray(curvature_diagnostics)
                print(
                    "[diagnostics] ion-temperature curvature self-form: "
                    f"product={chain_rule_host[0]:.6e}, "
                    f"flux={chain_rule_host[1]:.6e}, "
                    f"defect={chain_rule_host[2]:.6e}",
                    flush=True,
                )
            if inspection_host is not None:
                diagnostic_values = inspection_host[0]
                high_pass = diagnostic_values[2:].reshape(7, 2)
                print(
                    "[diagnostics] grid-scale: "
                    f"max|Ve|={diagnostic_values[0]:.6e} "
                    f"global_index={int(diagnostic_values[1])}; "
                    + ", ".join(
                        f"{name}=({values[0]:.3e},{values[1]:.3e})"
                        for name, values in zip(
                            initial_state.field_names(), high_pass, strict=True
                        )
                    )
                    + " [wall,bulk]",
                    flush=True,
                )
        while next_snapshot < len(snapshot_schedule) and snapshot_schedule[next_snapshot] <= current_time + 1.0e-14:
            save_snapshot(
                snapshot_schedule[next_snapshot],
                current_time,
                step,
                inspected=inspection_host,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
            )
            next_snapshot += 1
        if periodic_checkpoint_due:
            save_snapshot(
                current_time,
                current_time,
                step,
                inspected=inspection_host,
                phi_solver_diagnostics=gmres_stage_diagnostics_host,
                periodic_checkpoint=True,
            )
        line = _progress_line(
            step=step,
            num_steps=int(num_steps),
            simulation_time=current_time,
            density_min=density_min,
            density_max=density_max,
            step_seconds=step_seconds,
            operator_seconds=operator_seconds,
            gmres_seconds=gmres_seconds,
            gmres_iterations=gmres_iterations_host,
            gmres_relative_residual=gmres_relative_residual_host,
            elapsed_seconds=time.perf_counter() - simulation_start,
            solver_label="gmres-iters(avg4)",
        )
        if sys.stdout.isatty():
            print(f"\r{line}", end="\n" if step == int(num_steps) else "", flush=True)
        else:
            print(line, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        times=np.asarray(saved_times, dtype=np.float64),
        u=np.asarray(global_geometry.grid.x.centers, dtype=np.float64),
        v=np.asarray(global_geometry.grid.y.centers, dtype=np.float64),
        theta=np.asarray(global_geometry.grid.y.centers, dtype=np.float64),
        eta=np.asarray(global_geometry.grid.z.centers, dtype=np.float64),
        jacobian=np.asarray(
            global_geometry.cell_metric.J,
            dtype=np.float64,
        ),
        **(
            {
                # These are output-only owner measures.  They let a
                # login-node analyzer compare final production histories at
                # fixed resolution without rebuilding geometry or replacing
                # the production advance with a test-specific operator.
                "owner_active": np.asarray(
                    owner_host_geometry.topology.is_active_owner,
                    dtype=bool,
                ),
                "owner_aggregate_volume": np.asarray(
                    owner_host_geometry.aggregate_chart_volume,
                    dtype=np.float64,
                ),
            }
            if owner_host_geometry is not None
            else {}
        ),
        Bmag=np.asarray(
            global_geometry.cell_bfield.Bmag,
            dtype=np.float64,
        ),
        B_contravariant=np.asarray(
            global_geometry.cell_bfield.B_contra,
            dtype=np.float64,
        ),
        cartesian=np.asarray(cell_positions, dtype=np.float64),
        nfp=np.asarray(int(nfp), dtype=np.int32),
        simulated_field_periods=np.asarray(int(nfp), dtype=np.int32),
        toroidal_extent=np.asarray(2.0 * np.pi, dtype=np.float64),
        geometry_metadata_json=np.asarray(
            json.dumps(geometry_metadata, sort_keys=True, default=str)
        ),
        shard_counts=np.asarray(sharded_geometry.shard_counts, dtype=np.int32),
        periodic_axes=np.asarray(
            (metadata or {}).get("periodic_axes", PERIODIC_AXES), dtype=bool
        ),
        axis_regular_axes=np.asarray(
            (metadata or {}).get("axis_regular_axes", AXIS_REGULAR_AXES),
            dtype=bool,
        ),
        topology=np.asarray(str((metadata or {}).get("topology", "square"))),
        coordinate_names_json=np.asarray(
            json.dumps(
                (metadata or {}).get(
                    "coordinate_names", ("u", "v", "eta")
                )
            )
        ),
        logical_extents=np.asarray(
            (metadata or {}).get(
                "logical_extents", ((0.0, 1.0), (0.0, 1.0), (0.0, 2.0 * np.pi))
            ),
            dtype=np.float64,
        ),
        metric_mesh_shape=np.asarray(
            (metadata or {}).get("metric_mesh_shape") or (-1, -1, -1),
            dtype=np.int64,
        ),
        metric_radial_degree=np.asarray(
            int((metadata or {}).get("metric_radial_degree", -1)),
            dtype=np.int64,
        ),
        metric_poloidal_modes=np.asarray(
            int((metadata or {}).get("metric_poloidal_modes", -1)),
            dtype=np.int64,
        ),
        metric_toroidal_modes=np.asarray(
            int((metadata or {}).get("metric_toroidal_modes", -1)),
            dtype=np.int64,
        ),
        eta_projection_iterations=np.asarray(
            int((metadata or {}).get("eta_projection_iterations", -1)),
            dtype=np.int64,
        ),
        parallel_operator_scheme=np.asarray(
            str((metadata or {}).get("parallel_operator_scheme", parallel_operator_scheme))
        ),
        fci_trace_substeps=np.asarray(
            int((metadata or {}).get("fci_trace_substeps", 64)),
            dtype=np.int64,
        ),
        history_dtype=np.asarray(history_dtype),
        **{
            name: np.stack(values, axis=0)
            for name, values in history.items()
        },
        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **(
            {
                "rhs_term_times": np.asarray(
                    [
                        float(start_time) + index * float(timestep)
                        for index in range(int(num_steps) + 1)
                    ],
                    dtype=np.float64,
                ),
                "rhs_term_statistics": np.stack(
                    rhs_term_statistics_history, axis=0
                ),
                "rhs_term_field_names_json": np.asarray(
                    json.dumps(RHS_TERM_FIELD_NAMES)
                ),
                "rhs_term_names_json": np.asarray(
                    json.dumps(
                        {
                            field: list(names)
                            for field, names in zip(
                                RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES, strict=True
                            )
                        },
                        sort_keys=True,
                    )
                ),
                "rhs_term_statistic_names_json": np.asarray(
                    json.dumps(
                        (
                            "volume_weighted_mean",
                            "volume_weighted_rms",
                            "maximum_absolute",
                            "volume_weighted_radial_moment",
                        )
                    )
                ),
            }
            if track_rhs_terms
            else {}
        ),
    )
    if staged_audit_records:
        if staged_audit_output is None:  # Defensive; validated above.
            raise RuntimeError("missing staged_audit_output")
        staged_audit_output = Path(staged_audit_output)
        staged_audit_output.parent.mkdir(parents=True, exist_ok=True)
        stage_names = tuple(staged_audit_records[0]["stage_names"])
        stage_values = np.stack(
            tuple(record["stage_values"] for record in staged_audit_records),
            axis=0,
        )
        explicit_probe_rhs = np.stack(
            tuple(
                record["explicit_probe_rhs"]
                for record in staged_audit_records
            ),
            axis=0,
        )
        explicit_term_values = np.stack(
            tuple(
                record["explicit_term_values"]
                for record in staged_audit_records
            ),
            axis=0,
        )
        curvature_component_values = np.stack(
            tuple(
                record["curvature_component_values"]
                for record in staged_audit_records
            ),
            axis=0,
        )
        parallel_material_component_values = np.stack(
            tuple(
                record["parallel_material_component_values"]
                for record in staged_audit_records
            ),
            axis=0,
        )
        wall_current_values = tuple(
            np.stack(
                tuple(record["wall_current_values"][index]
                      for record in staged_audit_records),
                axis=0,
            )
            for index in range(7)
        )
        gmres_stage_diagnostics = np.stack(
            tuple(
                record["gmres_stage_diagnostics"]
                for record in staged_audit_records
            ),
            axis=0,
        )
        state_field_names = tuple(initial_state.field_names())
        evolved_state_indices = np.asarray(
            tuple(state_field_names.index(name) for name in RHS_TERM_FIELD_NAMES),
            dtype=np.int32,
        )
        stage_index = {name: index for index, name in enumerate(stage_names)}
        evolved = stage_values[..., evolved_state_indices]
        audit_dt = float(timestep)
        gamma_dt = IMEX_SSP222_GAMMA * audit_dt
        implicit_1_closure = (
            evolved[:, stage_index["stage_1"]]
            - evolved[:, stage_index["current"]]
            - gamma_dt * evolved[:, stage_index["implicit_rate_1"]]
        )
        explicit_probe_closure = np.stack(
            (
                evolved[:, stage_index["explicit_rate_1"]]
                - explicit_probe_rhs[:, 0],
                evolved[:, stage_index["explicit_rate_2"]]
                - explicit_probe_rhs[:, 1],
            ),
            axis=1,
        )
        explicit_ablation_values = np.zeros_like(explicit_probe_rhs)
        if staged_audit_explicit_ablation in (
            "phi-current-pair",
            "vorticity-advection-phi-current",
        ):
            for field_name, term_name in (
                ("Ve", "electrostatic"),
                ("vorticity", "parallel_current"),
            ):
                field_index = RHS_TERM_FIELD_NAMES.index(field_name)
                term_index = RHS_TERM_NAMES[field_index].index(term_name)
                explicit_ablation_values[..., field_index] = (
                    explicit_term_values[..., field_index, term_index]
                )
        if staged_audit_explicit_ablation in (
            "vorticity-parallel-advection",
            "vorticity-advection-phi-current",
        ):
            field_index = RHS_TERM_FIELD_NAMES.index("vorticity")
            term_index = RHS_TERM_NAMES[field_index].index(
                "parallel_advection"
            )
            explicit_ablation_values[..., field_index] += (
                explicit_term_values[..., field_index, term_index]
            )
        if staged_audit_explicit_ablation in (
            "curvature", "curvature-parallel-material"
        ):
            for field_name in ("density", "Te", "Ti", "vorticity"):
                field_index = RHS_TERM_FIELD_NAMES.index(field_name)
                term_index = RHS_TERM_NAMES[field_index].index("curvature")
                explicit_ablation_values[..., field_index] = (
                    explicit_term_values[..., field_index, term_index]
                )
        if staged_audit_explicit_ablation in (
            "parallel-material", "curvature-parallel-material"
        ):
            for field_name, term_name in (
                ("density", "parallel_density_flux_divergence"),
                ("Te", "parallel_advection"),
                ("Ti", "parallel_advection"),
                ("Vi", "parallel_self_advection"),
                ("Ve", "parallel_self_advection"),
            ):
                field_index = RHS_TERM_FIELD_NAMES.index(field_name)
                term_index = RHS_TERM_NAMES[field_index].index(term_name)
                explicit_ablation_values[..., field_index] += (
                    explicit_term_values[..., field_index, term_index]
                )
        explicit_ablation_closure = np.stack(
            (
                evolved[:, stage_index["explicit_rate_1"]]
                - (explicit_probe_rhs[:, 0] - explicit_ablation_values[:, 0]),
                evolved[:, stage_index["explicit_rate_2"]]
                - (explicit_probe_rhs[:, 1] - explicit_ablation_values[:, 1]),
            ),
            axis=1,
        )
        explicit_term_closure = (
            np.sum(explicit_term_values, axis=-1) - explicit_probe_rhs
        )
        curvature_fields = ("density", "Te", "Ti", "vorticity")
        curvature_term_values = np.stack(
            tuple(
                explicit_term_values[
                    ...,
                    RHS_TERM_FIELD_NAMES.index(field_name),
                    RHS_TERM_NAMES[
                        RHS_TERM_FIELD_NAMES.index(field_name)
                    ].index("curvature"),
                ]
                for field_name in curvature_fields
            ),
            axis=-1,
        )
        curvature_component_closure = (
            np.sum(curvature_component_values, axis=-1)
            - curvature_term_values
        )
        parallel_material_fields = ("density", "Te", "Ti", "Vi", "Ve")
        parallel_material_term_names = (
            "parallel_density_flux_divergence",
            "parallel_advection",
            "parallel_advection",
            "parallel_self_advection",
            "parallel_self_advection",
        )
        parallel_material_term_values = np.stack(
            tuple(
                explicit_term_values[
                    ...,
                    RHS_TERM_FIELD_NAMES.index(field_name),
                    RHS_TERM_NAMES[
                        RHS_TERM_FIELD_NAMES.index(field_name)
                    ].index(term_name),
                ]
                for field_name, term_name in zip(
                    parallel_material_fields,
                    parallel_material_term_names,
                    strict=True,
                )
            ),
            axis=-1,
        )
        parallel_material_component_closure = (
            np.sum(parallel_material_component_values, axis=-1)
            - parallel_material_term_values
        )
        stage_2_base_closure = (
            evolved[:, stage_index["stage_2_base_before_phi"]]
            - evolved[:, stage_index["current"]]
            - audit_dt * evolved[:, stage_index["explicit_rate_1"]]
            - (1.0 - 2.0 * IMEX_SSP222_GAMMA)
            * audit_dt
            * evolved[:, stage_index["implicit_rate_1"]]
        )
        implicit_2_closure = (
            evolved[:, stage_index["stage_2"]]
            - evolved[:, stage_index["stage_2_base"]]
            - gamma_dt * evolved[:, stage_index["implicit_rate_2"]]
        )
        weighted_rate_closure = (
            evolved[:, stage_index["weighted_rate"]]
            - 0.5
            * (
                evolved[:, stage_index["explicit_rate_1"]]
                + evolved[:, stage_index["explicit_rate_2"]]
                + evolved[:, stage_index["implicit_rate_1"]]
                + evolved[:, stage_index["implicit_rate_2"]]
            )
        )
        final_closure = (
            evolved[:, stage_index["final"]]
            - evolved[:, stage_index["current"]]
            - audit_dt * evolved[:, stage_index["weighted_rate"]]
        )
        np.savez_compressed(
            staged_audit_output,
            cell_indices=np.asarray(staged_audit_cells, dtype=np.int32),
            cell_u=np.asarray(
                [global_geometry.grid.x.centers[cell[0]] for cell in staged_audit_cells],
                dtype=np.float64,
            ),
            cell_theta=np.asarray(
                [global_geometry.grid.y.centers[cell[1]] for cell in staged_audit_cells],
                dtype=np.float64,
            ),
            cell_eta=np.asarray(
                [global_geometry.grid.z.centers[cell[2]] for cell in staged_audit_cells],
                dtype=np.float64,
            ),
            start_times=np.asarray(
                tuple(record["start_time"] for record in staged_audit_records),
                dtype=np.float64,
            ),
            timestep=np.asarray(audit_dt, dtype=np.float64),
            imex_ssp222_gamma=np.asarray(
                IMEX_SSP222_GAMMA, dtype=np.float64
            ),
            state_field_names_json=np.asarray(json.dumps(state_field_names)),
            rhs_field_names_json=np.asarray(json.dumps(RHS_TERM_FIELD_NAMES)),
            rhs_term_names_json=np.asarray(
                json.dumps(
                    {
                        name: list(terms)
                        for name, terms in zip(
                            RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES, strict=True
                        )
                    },
                    sort_keys=True,
                )
            ),
            stage_names_json=np.asarray(json.dumps(stage_names)),
            stage_values=stage_values,
            explicit_probe_rhs=explicit_probe_rhs,
            explicit_term_values=explicit_term_values,
            curvature_field_names_json=np.asarray(
                json.dumps(curvature_fields)
            ),
            curvature_direction_names_json=np.asarray(
                json.dumps(curvature_component_diagnostic_names())
            ),
            curvature_component_values=curvature_component_values,
            curvature_component_closure=curvature_component_closure,
            parallel_material_field_names_json=np.asarray(
                json.dumps(parallel_material_fields)
            ),
            parallel_material_direction_names_json=np.asarray(
                json.dumps(("backward", "center_geometric", "forward"))
            ),
            parallel_material_component_values=(
                parallel_material_component_values
            ),
            parallel_material_component_closure=(
                parallel_material_component_closure
            ),
            wall_current_stage_names_json=np.asarray(
                json.dumps(("stage_1", "stage_2"))
            ),
            wall_current_direction_names_json=np.asarray(
                json.dumps(("backward", "forward"))
            ),
            wall_current_state_field_names_json=np.asarray(
                json.dumps(("density", "Te", "Ti", "Vi", "Ve"))
            ),
            wall_current_channel_names_json=np.asarray(
                json.dumps(
                    (
                        "owner",
                        "raw_wall",
                        "effective_nonlinear",
                        "effective_linearized",
                        "exported_sat",
                        "material_owner_rate",
                    )
                )
            ),
            wall_current_particle_flux_names_json=np.asarray(
                json.dumps(("ion", "electron"))
            ),
            wall_current_metadata_names_json=np.asarray(
                json.dumps(("wall", "incoming_count", "cfl", "selected"))
            ),
            wall_current_divergence_names_json=np.asarray(
                json.dumps(
                    (
                        "homogeneous",
                        "affine",
                        "total",
                        "effective_nonlinear_total",
                        "effective_linearized_total",
                    )
                )
            ),
            wall_current_raw_endpoint_states=wall_current_values[0],
            wall_current_effective_face_states=wall_current_values[1],
            wall_current_channels=wall_current_values[2],
            wall_current_particle_fluxes=wall_current_values[3],
            wall_current_metadata=wall_current_values[4],
            wall_current_divergences=wall_current_values[5],
            wall_current_leg_lengths=wall_current_values[6],
            explicit_ablation=np.asarray(staged_audit_explicit_ablation),
            explicit_ablation_values=explicit_ablation_values,
            gmres_stage_diagnostics=gmres_stage_diagnostics,
            implicit_1_closure=implicit_1_closure,
            explicit_probe_closure=explicit_probe_closure,
            explicit_ablation_closure=explicit_ablation_closure,
            explicit_term_closure=explicit_term_closure,
            stage_2_base_closure=stage_2_base_closure,
            implicit_2_closure=implicit_2_closure,
            weighted_rate_closure=weighted_rate_closure,
            final_closure=final_closure,
            run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        closure_arrays = (
            implicit_1_closure,
            (
                explicit_probe_closure
                if staged_audit_explicit_ablation == "none"
                else explicit_ablation_closure
            ),
            explicit_ablation_closure,
            explicit_term_closure,
            curvature_component_closure,
            parallel_material_component_closure,
            stage_2_base_closure,
            implicit_2_closure,
            weighted_rate_closure,
            final_closure,
        )
        print(
            f"[staged-audit] wrote {staged_audit_output}; "
            f"max algebra/term closure="
            f"{max(float(np.max(np.abs(value))) for value in closure_arrays):.3e}",
            flush=True,
        )
    print(
        f"sharded EB advance completed in "
        f"{time.perf_counter() - simulation_start:.3f} s; "
        f"history written to {output_path}",
        flush=True,
    )
    if phase_timer is not None:
        timed_total = accumulated_operator_seconds + accumulated_gmres_seconds
        print(
            "[simulation] average compiled-step timing: "
            f"total={accumulated_step_seconds / num_steps:.3f} s, "
            f"operators={accumulated_operator_seconds / num_steps:.3f} s "
            f"({100.0 * accumulated_operator_seconds / max(timed_total, 1.0e-30):.1f}%), "
            f"GMRES={accumulated_gmres_seconds / num_steps:.3f} s "
            f"({100.0 * accumulated_gmres_seconds / max(timed_total, 1.0e-30):.1f}%)",
            flush=True,
        )
    print(
        f"[simulation] average {time_integrator} GMRES iterations: "
        f"{accumulated_gmres_iterations / num_steps:.2f} "
        "(four solves per timestep)",
        flush=True,
    )
    return materialized_state(state)


def _validate_flux_framework(args: argparse.Namespace) -> None:
    """Validate native production/diagnostic selectors before compilation."""

    framework = str(args.flux_framework)
    if getattr(args, "parallel_current_pairing", "reference") == "live-gradient-prototype":
        if not (
            framework == "production-split"
            and args.parallel_operator_scheme == "fci"
            and args.parallel_flux_pairing == "support-core"
            and args.parallel_boundary_pairing == "characteristic-sat"
            and args.physical_wall_model == "simplified-gbs-mpe"
        ):
            raise ValueError(
                "live-gradient-prototype requires production-split / fci / "
                "support-core / characteristic-sat / simplified-gbs-mpe"
            )
    if args.physical_wall_model != "legacy-velocity-trace":
        if args.parallel_characteristic_wall_law != "physical-boundary-state":
            raise ValueError(
                "named physical wall models require "
                "--parallel-characteristic-wall-law physical-boundary-state"
            )
        if args.parallel_velocity_wall_bc != "neumann":
            raise ValueError(
                "--parallel-velocity-wall-bc is only valid with "
                "--physical-wall-model legacy-velocity-trace"
            )
    if args.parallel_short_leg_selection == "all-physical-walls":
        if args.parallel_short_leg_treatment != "local-backward-euler":
            raise ValueError(
                "all-physical-walls short-leg selection requires "
                "--parallel-short-leg-treatment local-backward-euler"
            )
        if framework != "production-split" or args.parallel_operator_scheme != "fci":
            raise ValueError("all-physical-walls requires production FCI configuration")
        if args.parallel_flux_pairing != "support-core":
            raise ValueError("all-physical-walls requires support-core pairing")
        if args.parallel_boundary_pairing != "characteristic-sat":
            raise ValueError("all-physical-walls requires characteristic-sat pairing")
    if args.parallel_characteristic_wall_law == "energy-absorbing":
        if framework != "production-split":
            raise ValueError(
                "energy-absorbing parallel characteristic wall law requires "
                "the production-path parallel material scheme"
            )
        if args.parallel_boundary_pairing != "characteristic-sat":
            raise ValueError(
                "energy-absorbing parallel characteristic wall law requires "
                "characteristic-sat boundary pairing"
            )
    if args.parallel_characteristic_wall_law == "physical-boundary-state":
        if framework != "production-split":
            raise ValueError(
                "physical-boundary-state parallel characteristic wall law "
                "requires the production-path parallel material scheme"
            )
        if args.parallel_boundary_pairing != "characteristic-sat":
            raise ValueError(
                "physical-boundary-state parallel characteristic wall law "
                "requires characteristic-sat boundary pairing"
            )
    if not np.isfinite(args.parallel_short_leg_cfl_limit) or (
        args.parallel_short_leg_cfl_limit <= 0.0
    ):
        raise ValueError("--parallel-short-leg-cfl-limit must be finite and positive")
    if (
        getattr(args, "imex_split", "historical") == "historical"
        and args.parallel_short_leg_treatment == "local-backward-euler"
        and framework != "production-split"
    ):
        raise ValueError(
            "--parallel-short-leg-treatment local-backward-euler requires "
            "--flux-framework production-split"
        )
    if (
        getattr(args, "imex_split", "historical") == "historical"
        and args.parallel_short_leg_treatment == "local-backward-euler"
        and args.time_integrator != "imex-ssp222"
    ):
        raise ValueError(
            "--parallel-short-leg-treatment local-backward-euler requires "
            "--time-integrator imex-ssp222 so the complete selected residual "
            "is solved at every stage"
        )
    if (
        getattr(args, "imex_split", "historical") == "historical"
        and args.time_integrator == "imex-ssp222"
        and args.parallel_short_leg_treatment != "local-backward-euler"
    ):
        raise ValueError(
            "--time-integrator imex-ssp222 currently requires "
            "--parallel-short-leg-treatment local-backward-euler"
        )
    if args.parallel_flux_pairing == "support-core":
        if args.parallel_operator_scheme != "fci":
            raise ValueError("support-core requires --parallel-operator-scheme fci")
    if framework == "legacy":
        return
    if framework != "production-split":
        raise ValueError(f"unsupported flux framework {framework!r}")
    if args.time_integrator not in ("rk4", "imex-ssp222"):
        raise ValueError(
            "production-split requires --time-integrator rk4 or imex-ssp222"
        )
    if args.parallel_operator_scheme != "fci":
        raise ValueError("production-split requires --parallel-operator-scheme fci")
    if args.parallel_flux_pairing != "support-core":
        raise ValueError("production-split requires support-core current pairing")
    if (
        args.parallel_boundary_pairing == "legacy"
        and args.rhs_replay_history is None
    ):
        raise ValueError(
            "production-split trajectories require current-phi or characteristic-sat "
            "boundary pairing"
        )
    if args.poisson_bracket_scheme not in (
        "compatible-flux",
        "compatible-third-order-upwind",
        "material-scalar-third-order-upwind",
        "material-scalar-vorticity-compatible-upwind",
    ):
        raise ValueError("production-split requires compatible Poisson brackets")
    if args.physical_wall_model == "simplified-gbs-mpe" and not _simplified_gbs_mpe_selector_bundle_is_exact(args):
        raise ValueError(
            "simplified-gbs-mpe requires the exact rung-3 selector bundle: "
            "production-split / fci / support-core / characteristic-sat / "
            "local-backward-euler / all-physical-walls / "
            "physical-boundary-state / imex-ssp222 / "
            "material-scalar-vorticity-compatible-upwind / "
            "support-paired / jacobi, line-u, coarse-additive, or coarse-multiplicative"
        )


def _configure_runtime_selectors(args: argparse.Namespace) -> None:
    """Export native CLI selectors consumed by LocalFciDrbEBRhs factories."""

    os.environ["DRBX_FLUX_FRAMEWORK"] = str(args.flux_framework)
    os.environ["DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"] = str(args.parallel_characteristic_wall_law)
    os.environ["DRBX_PARALLEL_FLUX_PAIRING"] = str(args.parallel_flux_pairing)
    os.environ["DRBX_PARALLEL_CURRENT_PAIRING"] = str(
        getattr(args, "parallel_current_pairing", "reference")
    )
    os.environ["DRBX_PARALLEL_BOUNDARY_PAIRING"] = (
        str(args.parallel_boundary_pairing)
        if args.parallel_flux_pairing == "support-core"
        else "legacy"
    )
    os.environ["DRBX_PARALLEL_SHORT_LEG_TREATMENT"] = str(
        args.parallel_short_leg_treatment
    )
    os.environ["DRBX_PARALLEL_SHORT_LEG_SELECTION"] = str(
        args.parallel_short_leg_selection
    )
    os.environ["DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT"] = str(
        args.parallel_short_leg_cfl_limit
    )
    for name in (
        "DRBX_CHARACTERISTIC_SAT_AFFINE_CURRENT_LIFT",
        "DRBX_PARALLEL_CURRENT_PHI_PAIR",
        "DRBX_CURVATURE_EVOLUTION_COMPONENT",
        "DRBX_CURVATURE_RADIAL_ABLATION",
        "DRBX_CURVATURE_CHARACTERISTIC_AXES",
        "DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME",
        "DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME",
        "DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME",
    ):
        os.environ.pop(name, None)
    if args.flux_framework == "production-split":
        os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] = "production-path"
        os.environ["DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"] = (
            _parallel_characteristic_wall_metadata(
                str(args.parallel_characteristic_wall_law)
            )["parallel_material_wall_flux_closure"]
        )
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY", None)
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE", None)
    else:
        for name in (
            "DRBX_PARALLEL_MATERIAL_SCHEME",
            "DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE",
        ):
            os.environ.pop(name, None)
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY", None)
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE", None)
    for name in (
        "DRBX_RHS_TERM_HISTORY",
        "DRBX_RHS_TERM_FRAMES",
        "DRBX_RHS_TERM_OUTPUT",
    ):
        os.environ.pop(name, None)


def _parallel_characteristic_wall_metadata(wall_law: str) -> dict[str, object]:
    """Describe the selected wall law without inheriting stale environment state."""

    if wall_law == "primitive-least-residual":
        return {
            "parallel_material_wall_flux_closure": (
                "characteristic-projected-operator-trace-canonical-face-state"
            ),
            "parallel_material_wall_flux_closure_source": (
                "DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"
            ),
            "parallel_characteristic_wall_equilibrium_reference": None,
            "parallel_characteristic_wall_equilibrium_reference_source": None,
            "parallel_characteristic_wall_provenance": "primitive-least-residual",
            "parallel_characteristic_wall_energy_normalizer": None,
            "parallel_characteristic_wall_energy_normalizer_source": None,
        }
    if wall_law == "energy-absorbing":
        return {
            "parallel_material_wall_flux_closure": (
                "maximally-dissipative-energy-absorbing-normalized-equilibrium"
            ),
            "parallel_material_wall_flux_closure_source": (
                "simulate_hsx_blob.py:--parallel-characteristic-wall-law"
            ),
            "parallel_characteristic_wall_equilibrium_reference": [
                1.0, 1.0, 1.0, 0.0, 0.0
            ],
            "parallel_characteristic_wall_equilibrium_reference_source": (
                "normalized-equilibrium-contract"
            ),
            "parallel_characteristic_wall_provenance": (
                "experimental-normalized-equilibrium-absorber"
            ),
            "parallel_characteristic_wall_energy_normalizer": (
                "unit-modal-mathematical"
            ),
            "parallel_characteristic_wall_energy_normalizer_source": (
                "characteristic-wall-residual.py:unit-modal-energy"
            ),
        }
    if wall_law == "physical-boundary-state":
        return {
            "parallel_material_wall_flux_closure": (
                "live-characteristic-physical-boundary-state"
            ),
            "parallel_material_wall_flux_closure_source": (
                "simulate_hsx_blob.py:--parallel-characteristic-wall-law"
            ),
            "parallel_characteristic_wall_equilibrium_reference": None,
            "parallel_characteristic_wall_equilibrium_reference_source": None,
            "parallel_characteristic_wall_provenance": (
                "physical-face-trace-live-characteristic-split"
            ),
            "parallel_characteristic_wall_energy_normalizer": None,
            "parallel_characteristic_wall_energy_normalizer_source": None,
        }
    raise ValueError(f"unknown parallel characteristic wall law: {wall_law!r}")


def _physical_wall_model_provenance(args: argparse.Namespace) -> str:
    """Name the selected wall-model rung/preset without changing the CLI."""

    if args.physical_wall_model == "simplified-gbs-mpe":
        return "simplified-gbs-mpe"
    if args.physical_wall_model == "simple-conducting-sheath":
        return "production-rung2-simple-conducting-sheath"
    if args.physical_wall_model == "no-flow":
        return "no-flow-rung1"
    return "legacy-compatibility"


def _simplified_gbs_mpe_selector_bundle_is_exact(args: argparse.Namespace) -> bool:
    """Return whether the explicit rung-3 selector bundle is fully enabled."""

    return (
        args.flux_framework == "production-split"
        and args.parallel_operator_scheme == "fci"
        and args.parallel_flux_pairing == "support-core"
        and args.parallel_boundary_pairing == "characteristic-sat"
        and args.parallel_short_leg_treatment == "local-backward-euler"
        and args.parallel_short_leg_selection == "all-physical-walls"
        and args.parallel_characteristic_wall_law == "physical-boundary-state"
        and args.time_integrator == "imex-ssp222"
        and args.poisson_bracket_scheme
        == "material-scalar-vorticity-compatible-upwind"
        # Rung 3 requires the support-paired homogeneous Neumann action so the
        # polarization solve, the Ti term, and the derived vorticity trace use
        # one metric-weighted gather/scatter contract.
        and args.polarization_operator_form == "support-paired"
        and args.gmres_preconditioner in (
            "jacobi",
            "line-u",
            "coarse-additive",
            "coarse-multiplicative",
        )
    )


def _build_parser(*, require_geometry: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a producer-owned HSX FCI geometry artifact and run the full "
            "local/sharding-compatible seven-field electrostatic Boussinesq model."
        )
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        required=bool(require_geometry),
        help=(
            "Directory containing the precomputed FCI simulation geometry "
            "artifact. Geometry generation and metric-cache options are not "
            "part of the simulation consumer."
        ),
    )
    parser.add_argument(
        "--parallel-operator-scheme",
        choices=("coordinate", "fci"),
        default="coordinate",
        help=(
            "Parallel derivative/operator implementation. 'fci' uses the "
            "traced toroidal field-line maps and requires --topology=toroidal."
        ),
    )
    parser.add_argument(
        "--parallel-flux-pairing",
        choices=("legacy", "support-core"),
        default="legacy",
        help=(
            "Pairing used by mapped parallel gradient/divergence operators. "
            "The production path requires support-core."
        ),
    )
    parser.add_argument(
        "--parallel-boundary-pairing",
        choices=("legacy", "current-phi", "characteristic-sat"),
        default="current-phi",
        help=(
            "Physical-wall closure for support-core FCI fluxes. "
            "characteristic-sat uses the projected characteristic wall "
            "state for the affine current flux and the homogeneous paired "
            "gradient."
        ),
    )
    parser.add_argument(
        "--parallel-current-pairing",
        choices=("reference", "live-gradient-prototype"),
        default="reference",
        help=(
            "Current divergence paired with the reference or complete live "
            "generalized-potential gradient. The prototype retains the "
            "canonical endpoint-current lift and requires simplified-gbs-mpe."
        ),
    )
    parser.add_argument(
        "--parallel-characteristic-wall-law",
        choices=(
            "primitive-least-residual",
            "energy-absorbing",
            "physical-boundary-state",
        ),
        default="primitive-least-residual",
        help=(
            "Characteristic parallel material wall law. "
            "'primitive-least-residual' retains the primitive incoming "
            "trace; 'energy-absorbing' selects the experimental mathematical "
            "characteristic normalized-equilibrium absorber with unit modal "
            "weights (reference [1,1,1,0,0]); 'physical-boundary-state' "
            "passes the complete physical face trace through the live "
            "characteristic split without a fixed incoming-mode solve."
        ),
    )
    parser.add_argument(
        "--parallel-short-leg-treatment",
        choices=("explicit", "local-backward-euler"),
        default="explicit",
        help=(
            "Treatment of selected short FCI wall legs. local-backward-euler "
            "hands the complete characteristic material plus "
            "mu*tau*grad_parallel(Ti) row residual to every imex-ssp222 "
            "stage; the weighted-adjoint current/phi pair stays explicit."
        ),
    )
    parser.add_argument(
        "--parallel-short-leg-selection",
        choices=("cfl", "all-physical-walls"),
        default="cfl",
        help=(
            "Select material wall legs for the local backward-Euler split. "
            "'cfl' preserves threshold selection; 'all-physical-walls' "
            "uses no CFL threshold and splits all physical wall material "
            "legs to local backward Euler. The vorticity current-divergence "
            "part of the characteristic-SAT pair remains explicit."
        ),
    )
    parser.add_argument(
        "--parallel-short-leg-cfl-limit",
        type=float,
        default=2.5,
        help="CFL threshold selecting short wall legs for the local implicit split.",
    )
    parser.add_argument(
        "--shard-counts",
        "--shards",
        nargs=3,
        type=int,
        metavar=("SU", "SV", "SETA"),
        default=(1, 1, 1),
        help=(
            "Production JAX decomposition in u, the second logical "
            "coordinate, and eta. Only eta decomposition is supported, so "
            "the first two entries must be one. Neta must be divisible by "
            "SETA, whose value must not exceed the available JAX device count."
        ),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument(
        "--neumann-ghost-scheme",
        choices=("logical", "physical"),
        default="physical",
        help=(
            "Neumann ghost closure. 'physical' interprets the data as a "
            "physical-normal derivative using the inverse metric; 'logical' "
            "retains the copied-ghost coordinate-normal closure."
        ),
    )
    parser.add_argument(
        "--physical-wall-model",
        choices=PHYSICAL_WALL_MODEL_NAMES,
        default="legacy-velocity-trace",
        help=(
            "Physical wall bundle model. 'no-flow' supplies Vi=Ve=0; "
            "'simple-conducting-sheath' supplies the grounded conducting-sheath "
            "trace with warm-ion Bohm outflow and electron saturation response; "
            "'simplified-gbs-mpe' selects the explicit rung-3 production "
            "bundle with the physical-boundary-state characteristic wall law. "
            "Named models require the physical-boundary-state characteristic law. "
            "The legacy adapter preserves --parallel-velocity-wall-bc for old runs."
        ),
    )
    parser.add_argument(
        "--conducting-sheath-wall-potential",
        type=float,
        default=None,
        help=(
            "Grounded/simple conducting-sheath wall potential in normalized "
            "phi units; None uses the wall-model default."
        ),
    )
    parser.add_argument(
        "--parallel-velocity-wall-bc",
        choices=("dirichlet-zero", "neumann", "bohm"),
        default="neumann",
        help=(
            "Legacy primitive Vi/Ve condition on physical vessel faces, "
            "used only with --physical-wall-model legacy-velocity-trace. "
            "'dirichlet-zero' supplies Vi=Ve=0 primitive face traces; "
            "'neumann' extrapolates both parallel velocities; 'bohm' sets "
            "outward Vi=Ve=sign(B.n)*sqrt(Te+tau*Ti), a zero-current "
            "sheath-entry diagnostic without a magnetic-presheath model."
        ),
    )
    parser.add_argument(
        "--poisson-bracket-scheme",
        choices=(
            "direct",
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
            "material-scalar-vorticity-compatible-upwind",
        ),
        default="compatible-flux",
        help=(
            "Poisson-bracket discretization. 'compatible-flux' is the "
            "production antisymmetrized shared-face flux form and includes "
            "the RHS 1/B factor. 'compatible-third-order-upwind' evaluates "
            "one compatible characteristic bracket for every equation: it "
            "keeps the compatible skew core and replaces the physical "
            "A_phi(q) channel by the complete third-order upwind action, with "
            "first-order wall/RLP fallbacks and retained D(Uq)-qD(U)."
            " 'material-scalar-third-order-upwind' uses pure third-order "
            "A_phi(q) transport for material fields and the centered "
            "compatible bracket for vorticity. "
            "'material-scalar-vorticity-compatible-upwind' keeps that "
            "material transport and adds the physical A_phi upwind "
            "correction to the compatible vorticity bracket."
        ),
    )
    parser.add_argument(
        "--polarization-operator-form",
        choices=("conservative", "weighted-symmetric", "support-paired"),
        default="conservative",
        help=(
            "Perpendicular polarization operator. 'conservative' preserves "
            "the historical independently assembled face gradient/divergence. "
            "'weighted-symmetric' uses the exact physical-volume weighted "
            "self-adjoint homogeneous action for both phi and Ti while "
            "retaining affine boundary sources exactly once. "
            "'support-paired' constructs the homogeneous action directly as "
            "the metric-weighted face-gradient Gram operator G^dagger W G."
        ),
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1.0e-8,
        help="Final normalized simulation time.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of equal RK4 steps used to reach --final-time.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Write an atomic restartable checkpoint after every N completed "
            "steps; 0 disables periodic checkpoints. Checkpoints are separate "
            "step-indexed NPZ files and survive a later run failure."
        ),
    )
    parser.add_argument(
        "--snapshot-times",
        nargs="+",
        type=float,
        default=(),
        metavar="T",
        help=(
            "Absolute physical times at which durable atomic checkpoint NPZ "
            "files are written during the run. A checkpoint is written at "
            "the first completed step at or after each requested time."
        ),
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory for scheduled snapshot NPZ files; defaults to the output directory.",
    )
    parser.add_argument(
        "--snapshot-term-fields",
        action="store_true",
        help="Include the eight spatial Ve RHS term fields in each snapshot.",
    )
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help="Restart from a snapshot NPZ or a saved history NPZ.",
    )
    parser.add_argument(
        "--restart-frame",
        type=int,
        default=-1,
        help="Frame to load from a history NPZ; ignored for a single snapshot.",
    )
    parser.add_argument(
        "--reconstruct-restart-phi",
        action="store_true",
        help=(
            "Reconstruct phi with the current polarization operator and wall "
            "closure after loading a restart. Use this when the restart was "
            "produced by a different boundary/operator configuration; by "
            "default restart phi is preserved for exact continuation."
        ),
    )
    parser.add_argument(
        "--initialize-rung3-wall-layer",
        action="store_true",
        help=(
            "For simplified-gbs-mpe, smoothly initialize Vi and Ve over "
            "the inward owner rings from every selected FCI physical-hit target, "
            "enforce compatible upper-radial wall traces and the wall-area phi "
            "gauge, then derive vorticity from the polarization state. Disabled "
            "by default so unrelated restarts are unchanged."
        ),
    )
    parser.add_argument(
        "--rung3-wall-layer-cells",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Number of outer radial owner rings used by "
            "--initialize-rung3-wall-layer (default: 8)."
        ),
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Print compiled min/max/max-absolute diagnostics for every state "
            "field every N steps; 0 disables periodic detailed output."
        ),
    )
    parser.add_argument(
        "--staged-audit-cell",
        action="append",
        nargs=3,
        type=int,
        default=[],
        metavar=("IU", "ITHETA", "IETA"),
        help=(
            "Record the complete IMEX stage sequence and all explicit RHS "
            "term lanes at one selected global cell. Repeat for multiple "
            "cells. This diagnostic requires staged-compiled execution, "
            "one shard, and --staged-audit-output."
        ),
    )
    parser.add_argument(
        "--staged-audit-output",
        type=Path,
        default=None,
        help="NPZ output for --staged-audit-cell stage and term data.",
    )
    parser.add_argument(
        "--staged-audit-explicit-ablation",
        choices=(
            "none",
            "phi-current-pair",
            "vorticity-parallel-advection",
            "vorticity-advection-phi-current",
            "curvature",
            "parallel-material",
            "curvature-parallel-material",
        ),
        default="none",
        help=(
            "Diagnostic-only paired explicit-term ablation for a selected-cell "
            "staged audit. 'phi-current-pair' removes electron electrostatic "
            "force together with vorticity current divergence; "
            "'vorticity-parallel-advection' removes only parallel vorticity "
            "advection; 'vorticity-advection-phi-current' removes it together "
            "with the electrostatic/current pair; 'curvature' "
            "removes the curvature lanes from density, Te, Ti, and vorticity; "
            "'parallel-material' removes the complete five-field production "
            "parallel-material residual; the combined choice removes both."
        ),
    )
    parser.add_argument(
        "--track-curvature-chain-rule-defect",
        action="store_true",
        help=(
            "Track global max-absolute ion-temperature product-form, "
            "flux-form, and product-rule-defect terms in the compiled "
            "RK4 advance. Values print with --diagnostic-every."
        ),
    )
    parser.add_argument(
        "--track-rhs-terms",
        action="store_true",
        help=(
            "Evaluate the complete six-equation RHS decomposition at the "
            "initial state and every accepted timestep, storing global "
            "mean/RMS/max/radial-moment statistics in the output NPZ."
        ),
    )
    parser.add_argument(
        "--rhs-replay-history",
        type=Path,
        default=None,
        help=(
            "Evaluate and export the complete spatial RHS decomposition at "
            "selected frames of an existing history, without advancing time."
        ),
    )
    parser.add_argument(
        "--rhs-replay-frames",
        default="",
        metavar="I,J,...",
        help="Comma-separated history frame indices for --rhs-replay-history.",
    )
    parser.add_argument(
        "--rhs-replay-output",
        type=Path,
        default=None,
        help="NPZ output for the frozen-state spatial RHS replay.",
    )
    parser.add_argument(
        "--rhs-replay-execution",
        choices=("auto", "compiled", "eager"),
        default="auto",
        help=(
            "Execution mode for frozen RHS replays. 'auto' uses eager "
            "execution for fewer than 100 frames and compilation for larger "
            "production batches. "
            "'eager' disables the outer jax.jit and avoids building the "
            "large fused replay executable, although the JAX backend may "
            "still compile small primitive kernels."
        ),
    )
    parser.add_argument(
        "--rhs-replay-electron-force-wall-audit",
        action="store_true",
        help=(
            "Include exact wall-face electron parallel-force traces, "
            "directional stencil contributions, masks, and leg lengths in "
            "an RHS replay archive."
        ),
    )
    parser.add_argument(
        "--blob-initialization",
        choices=("logical",),
        default="logical",
        help="Initialize the perturbation from logical cell centers.",
    )
    parser.add_argument("--density-amplitude", type=float, default=0.05)
    parser.add_argument(
        "--temperature-amplitude",
        type=float,
        default=0.0,
        help=(
            "Electron-temperature perturbation for the logical initialization."
        ),
    )
    parser.add_argument(
        "--blob-center",
        nargs=2,
        type=float,
        metavar=("U0", "V0"),
        default=(0.65, 0.50),
    )
    parser.add_argument("--blob-width", type=float, default=0.10)
    parser.add_argument(
        "--toroidal-perturbation-amplitude",
        type=float,
        default=0.0,
        help=(
            "Relative cosine modulation of the logical blob used to break "
            "field-period symmetry."
        ),
    )
    parser.add_argument(
        "--toroidal-perturbation-mode",
        type=int,
        default=1,
        help="Full-torus integer mode number used by the initial perturbation.",
    )
    parser.add_argument(
        "--toroidal-perturbation-phase",
        type=float,
        default=0.0,
        help="Initial toroidal perturbation phase in radians.",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--rho-star", type=float, default=1.0)
    parser.add_argument("--mi-over-me", type=float, default=1836.0)
    parser.add_argument("--perp-diffusion", type=float, default=1.0e-5)
    parser.add_argument("--parallel-diffusion", type=float, default=1.0e-5)
    parser.add_argument("--electron-collision-frequency", type=float, default=0.0)
    parser.add_argument(
        "--time-integrator",
        choices=("rk4", "imex-ssp222"),
        default="rk4",
        help=(
            "Time integrator. Classical RK4 is used for fully explicit "
            "configurations. 'imex-ssp222' is the stage-wise two-stage IMEX "
            "method required by local backward-Euler short wall legs."
        ),
    )
    parser.add_argument(
        "--imex-split",
        choices=("historical", "coupled-boundary"),
        default="historical",
        help=(
            "IMEX stage contract. 'historical' selects the existing split; "
            "'coupled-boundary' is the guarded eager, single-device nonlinear "
            "boundary-stage path."
        ),
    )
    parser.add_argument(
        "--advance-execution",
        choices=("auto", "compiled", "staged-compiled", "eager"),
        default="auto",
        help=(
            "Execution mode for time advancement. 'auto' uses staged "
            "compilation for fewer than 100 IMEX diagnostic steps, eager "
            "execution for short RK4 diagnostics, and fused compilation for "
            "longer production runs. 'compiled' builds one fused "
            "advance executable. 'staged-compiled' (IMEX-SSP222 only) "
            "compiles reusable implicit, explicit, phi, and diagnostic "
            "shard-map kernels separately. 'eager' disables the outer "
            "jax.jit, although the JAX backend may still compile kernels."
        ),
    )
    parser.add_argument(
        "--flux-framework",
        choices=("legacy", "production-split"),
        default="legacy",
        help=(
            "High-level flux wiring. 'legacy' preserves the established "
            "path; 'production-split' selects the production curvature and "
            "parallel material paths with compatibility guards."
        ),
    )
    parser.add_argument(
        "--gmres-acceptance-tolerance",
        "--gmres-tolerance",
        dest="gmres_acceptance_tolerance",
        type=float,
        default=5.0e-5,
        help=(
            "Relative and absolute residual accepted after GMRES stops. "
            "--gmres-tolerance is a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--gmres-target-tolerance",
        type=float,
        default=GMRES_TARGET_TOLERANCE,
        help="Relative and absolute residual target used to stop GMRES.",
    )
    parser.add_argument("--gmres-max-iterations", type=int, default=500)
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=100,
        help="GMRES restart length; capped at --gmres-max-iterations.",
    )
    parser.add_argument(
        "--gmres-preconditioner",
        choices=(
            "none",
            "jacobi",
            "line-u",
            "line-v",
            "line-uv",
            "coarse-additive",
            "coarse-multiplicative",
        ),
        default="line-u",
        help=(
            "SOLVAX right preconditioner for the phi inversion. Line "
            "preconditioners use local complete u and/or v grid lines. "
            "coarse-multiplicative uses one line-u/coarse/line-u cycle."
        ),
    )
    parser.add_argument(
        "--gmres-residual-correction-steps",
        type=int,
        default=1,
        help=(
            "Number of reliable true-residual correction solves attempted "
            "only when the primary GMRES result would otherwise be rejected."
        ),
    )
    parser.add_argument(
        "--no-phase-timing",
        action="store_true",
        help=(
            "Disable ordered in-executable timing markers for the operator "
            "and GMRES portions of an advance. Phase timing is disabled "
            "automatically for eager execution and multi-device shard_map."
        ),
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help=(
            "Stop after global FciGeometry3D assembly and shard-local "
            "geometry lowering."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "hsx_blob_history.npz",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser(require_geometry=True)
    args = parser.parse_args(argv)
    if args.geometry is None:
        parser.error("--geometry is required")
    if not args.geometry.is_dir():
        parser.error(f"--geometry must be an existing directory: {args.geometry}")
    try:
        simulation_geometry = load_fci_simulation_geometry(args.geometry)
        global_geometry = simulation_geometry.global_geometry
        cell_positions = simulation_geometry.cell_positions
        nfp = int(simulation_geometry.nfp)
        artifact_topology = simulation_geometry.topology
        descriptor = _artifact_topology_descriptor(artifact_topology)
    except (OSError, TypeError, ValueError, AttributeError) as error:
        parser.error(f"could not load --geometry artifact: {error}")
    # The artifact is authoritative for mesh shape/topology and all producer
    # geometry options below are intentionally ignored by this consumer.
    args.topology = descriptor.name
    args.resolution = tuple(int(value) for value in global_geometry.shape)
    args.curvature_edge_one_form = (
        getattr(simulation_geometry, "curvature_edge_one_form", None) is not None
    )
    owner_host_geometry = getattr(simulation_geometry, "owner_geometry", None)
    owner_overlap = getattr(simulation_geometry, "owner_overlap", None)
    artifact_metadata = getattr(simulation_geometry, "metadata", {})
    if not isinstance(artifact_metadata, Mapping):
        artifact_metadata = {}
    # These options belonged to the producer and are deliberately absent
    # from the consumer parser.  Keep local metadata defaults for the legacy
    # reporting fields below.
    angular_group_profile = ()
    args.square_agglomeration = "none"
    args.agglomeration_volume_ratio = 1.2
    args.agglomeration_rate_threshold = None
    args.agglomeration_rk4_safety = 0.85
    if args.imex_split == "coupled-boundary":
        if args.time_integrator != "imex-ssp222":
            parser.error("--imex-split coupled-boundary requires --time-integrator imex-ssp222")
        if args.advance_execution not in ("auto", "eager"):
            parser.error("--imex-split coupled-boundary rejects compiled/staged execution")
        if tuple(int(v) for v in args.shard_counts) != (1, 1, 1):
            parser.error("--imex-split coupled-boundary currently requires one device")
        args.advance_execution = "eager"
    try:
        _validate_flux_framework(args)
    except ValueError as error:
        parser.error(str(error))
    _configure_runtime_selectors(args)
    physical_wall_model_provenance = _physical_wall_model_provenance(args)
    print(
        "[simulation] flux_framework="
        f"{args.flux_framework}; "
        "parallel_velocities=cell-centered; "
        f"parallel_flux_pairing={args.parallel_flux_pairing}; "
        f"parallel_current_pairing={args.parallel_current_pairing}; "
        f"parallel_characteristic_wall_law={args.parallel_characteristic_wall_law}; "
        f"physical_wall_model={args.physical_wall_model}; "
        "parallel_boundary_pairing="
        f"{os.environ['DRBX_PARALLEL_BOUNDARY_PAIRING']}; "
        f"parallel_short_leg_treatment={args.parallel_short_leg_treatment}; "
        f"parallel_short_leg_selection={args.parallel_short_leg_selection}",
        flush=True,
    )
    resolution = tuple(int(value) for value in args.resolution)
    if args.topology == "toroidal" and resolution[1] % 2:
        parser.error("toroidal global NTHETA must be even")
    if args.parallel_operator_scheme == "fci" and args.topology != "toroidal":
        parser.error("--parallel-operator-scheme=fci requires --topology=toroidal")
    if args.rung3_wall_layer_cells < 1:
        parser.error("--rung3-wall-layer-cells must be positive")
    if (
        args.initialize_rung3_wall_layer
        and args.physical_wall_model != "simplified-gbs-mpe"
    ):
        parser.error(
            "--initialize-rung3-wall-layer requires "
            "--physical-wall-model=simplified-gbs-mpe"
        )
    if args.initialize_rung3_wall_layer and args.topology != "toroidal":
        parser.error(
            "--initialize-rung3-wall-layer currently requires --topology=toroidal"
        )
    if args.initialize_rung3_wall_layer and args.halo_width < 2:
        parser.error(
            "--initialize-rung3-wall-layer requires --halo-width at least 2"
        )
    shard_counts = tuple(int(value) for value in args.shard_counts)
    if any(value < 1 for value in shard_counts):
        parser.error("--shard-counts entries must be positive")
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        parser.error(
            "production sharding is eta-only; use --shard-counts 1 1 NETA_SHARDS"
        )
    curvature_edge_one_form = None
    if args.topology == "toroidal":
        if args.gmres_preconditioner not in (
            "none",
            "jacobi",
            "line-u",
            "coarse-additive",
            "coarse-multiplicative",
        ):
            parser.error(
                "toroidal RLP supports only --gmres-preconditioner "
                "none, jacobi, line-u, coarse-additive, or coarse-multiplicative"
            )
        if args.gmres_preconditioner in ("coarse-additive", "coarse-multiplicative") and (
            args.polarization_operator_form != "support-paired"
            or args.physical_wall_model != "simplified-gbs-mpe"
        ):
            parser.error(
                "coarse preconditioning requires "
                "--polarization-operator-form=support-paired and "
                "--physical-wall-model=simplified-gbs-mpe"
            )
        if args.poisson_bracket_scheme not in (
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
            "material-scalar-vorticity-compatible-upwind",
        ):
            parser.error("toroidal RLP requires a compatible Poisson-bracket scheme")
    if args.square_agglomeration == "corner-edge":
        if args.time_integrator != "rk4":
            parser.error("square corner-edge agglomeration currently requires --time-integrator=rk4")
        if args.gmres_preconditioner != "line-u":
            parser.error("square corner-edge agglomeration currently requires --gmres-preconditioner=line-u")
        if args.poisson_bracket_scheme not in (
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
            "material-scalar-vorticity-compatible-upwind",
        ):
            parser.error(
                "square corner-edge agglomeration requires a compatible "
                "Poisson-bracket scheme"
            )
    for axis, (cell_count, shard_count) in enumerate(
        zip(resolution, shard_counts)
    ):
        if cell_count % shard_count:
            parser.error(
                f"resolution axis {axis} ({cell_count}) is not divisible by "
                f"shard count {shard_count}"
            )
    try:
        mesh = make_shard_mesh(shard_counts)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    if args.final_time <= 0.0:
        parser.error("--final-time must be positive")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if any(float(value) < 0.0 for value in args.snapshot_times):
        parser.error("--snapshot-times must be nonnegative")
    if any(float(value) > float(args.final_time) for value in args.snapshot_times):
        parser.error("--snapshot-times must not exceed --final-time")
    if len(set(float(value) for value in args.snapshot_times)) != len(args.snapshot_times):
        parser.error("--snapshot-times must not contain duplicates")
    if args.gmres_target_tolerance < 0.0:
        parser.error("--gmres-target-tolerance must be nonnegative")
    if args.gmres_acceptance_tolerance < 0.0:
        parser.error("--gmres-acceptance-tolerance must be nonnegative")
    if args.gmres_max_iterations < 1:
        parser.error("--gmres-max-iterations must be positive")
    if args.gmres_restart < 1:
        parser.error("--gmres-restart must be positive")
    if args.gmres_residual_correction_steps < 0:
        parser.error("--gmres-residual-correction-steps must be nonnegative")
    if args.halo_width < 1:
        parser.error("--halo-width must be positive")
    if args.density_amplitude < 0.0:
        parser.error("--density-amplitude must be nonnegative")
    if args.temperature_amplitude < 0.0:
        parser.error("--temperature-amplitude must be nonnegative")
    if args.blob_width <= 0.0:
        parser.error("--blob-width must be positive")
    if args.diagnostic_every < 0:
        parser.error("--diagnostic-every must be nonnegative")
    try:
        rhs_replay_frames = tuple(
            int(value)
            for value in str(args.rhs_replay_frames).split(",")
            if value.strip()
        )
    except ValueError as error:
        parser.error("--rhs-replay-frames must be comma-separated integers")
    if args.rhs_replay_history is not None:
        if not args.rhs_replay_history.is_file():
            parser.error("--rhs-replay-history must name an existing NPZ")
        if not rhs_replay_frames or any(value < 0 for value in rhs_replay_frames):
            parser.error(
                "--rhs-replay-history requires nonnegative --rhs-replay-frames"
            )
        if args.rhs_replay_output is None:
            parser.error("--rhs-replay-history requires --rhs-replay-output")
        if args.parallel_operator_scheme != "fci":
            parser.error("--rhs-replay-history requires --parallel-operator-scheme=fci")
    elif rhs_replay_frames or args.rhs_replay_output is not None:
        parser.error(
            "--rhs-replay-frames/--rhs-replay-output require --rhs-replay-history"
        )
    if (
        args.rhs_replay_history is None
        and args.rhs_replay_execution == "eager"
    ):
        parser.error("--rhs-replay-execution eager requires --rhs-replay-history")
    if (
        args.rhs_replay_electron_force_wall_audit
        and args.rhs_replay_history is None
    ):
        parser.error(
            "--rhs-replay-electron-force-wall-audit requires --rhs-replay-history"
        )
    requested_advance_execution = str(args.advance_execution)
    if args.imex_split == "coupled-boundary":
        requested_advance_execution = "eager"
        args.advance_execution = "eager"
    requested_rhs_replay_execution = str(args.rhs_replay_execution)
    args.advance_execution = _resolve_execution_mode(
        requested_advance_execution,
        work_items=int(args.num_steps),
        auto_short_mode=(
            "staged-compiled"
            if args.time_integrator == "imex-ssp222"
            else "eager"
        ),
    )
    args.rhs_replay_execution = (
        _resolve_execution_mode(
            requested_rhs_replay_execution,
            work_items=len(rhs_replay_frames),
        )
        if args.rhs_replay_history is not None
        else "compiled"
    )
    setup_execution = (
        args.rhs_replay_execution
        if args.rhs_replay_history is not None
        else args.advance_execution
    )
    if (
        args.advance_execution == "staged-compiled"
        and args.time_integrator != "imex-ssp222"
    ):
        parser.error(
            "--advance-execution staged-compiled requires "
            "--time-integrator imex-ssp222"
        )
    staged_audit_cells = tuple(
        tuple(int(index) for index in cell)
        for cell in args.staged_audit_cell
    )
    if staged_audit_cells:
        if args.staged_audit_output is None:
            parser.error(
                "--staged-audit-cell requires --staged-audit-output"
            )
        if args.advance_execution != "staged-compiled":
            parser.error(
                "--staged-audit-cell requires --advance-execution "
                "staged-compiled"
            )
        if tuple(int(value) for value in shard_counts) != (1, 1, 1):
            parser.error("--staged-audit-cell currently requires one shard")
        if len(set(staged_audit_cells)) != len(staged_audit_cells):
            parser.error("--staged-audit-cell entries must be unique")
        for cell in staged_audit_cells:
            if any(
                index < 0 or index >= extent
                for index, extent in zip(cell, resolution, strict=True)
            ):
                parser.error(
                    f"--staged-audit-cell {cell!r} lies outside resolution "
                    f"{tuple(resolution)}"
                )
    elif args.staged_audit_output is not None:
        parser.error(
            "--staged-audit-output requires at least one --staged-audit-cell"
        )
    if args.staged_audit_explicit_ablation != "none" and not staged_audit_cells:
        parser.error(
            "--staged-audit-explicit-ablation requires --staged-audit-cell"
        )
    if requested_advance_execution == "auto":
        print(
            "[simulation] auto-selected advance execution: "
            f"{args.advance_execution} for {int(args.num_steps)} step(s)",
            flush=True,
        )
    if (
        args.rhs_replay_history is not None
        and requested_rhs_replay_execution == "auto"
    ):
        print(
            "[rhs-replay] auto-selected execution: "
            f"{args.rhs_replay_execution} for {len(rhs_replay_frames)} frame(s)",
            flush=True,
        )
    if not 0.0 <= args.toroidal_perturbation_amplitude < 1.0:
        parser.error(
            "--toroidal-perturbation-amplitude must lie in [0, 1)"
        )
    if args.toroidal_perturbation_mode < 1:
        parser.error("--toroidal-perturbation-mode must be positive")
    print(
        "loading HSX simulation geometry artifact: "
        f"cells={resolution}, shards={shard_counts}, "
        f"devices={int(np.prod(shard_counts))}, path={args.geometry}",
        flush=True,
    )
    geometry_start = time.perf_counter()
    curvature_edge_one_form = getattr(
        simulation_geometry, "curvature_edge_one_form", None
    )
    lowering_start = time.perf_counter()
    print(
        f"[geometry] preparing shard-local geometry inputs "
        f"(shards={shard_counts}, halo_width={int(args.halo_width)})",
        flush=True,
    )
    sharded_geometry = build_local_fci_geometries(
        global_geometry,
        shard_counts,
        halo_width=int(args.halo_width),
        periodic_axes=descriptor.periodic_axes,
        axis_regular_axes=descriptor.axis_regular_axes,
    )
    control_volume_descriptor = None
    control_volume_fields = None
    control_volume_boundary_bc = None
    control_volume_assembler = None
    control_volume_field_count = RLP_PACKED_FIELD_COUNT
    angular_profile_safety_ratio = None
    if owner_host_geometry is not None:
        if args.topology == "toroidal":
            profile = getattr(owner_host_geometry, "angular_group_size", None)
            if profile is not None:
                print(
                    f"[angular-rlp-host] artifact profile={np.asarray(profile).tolist()}",
                    flush=True,
                )
        (
            control_volume_descriptor,
            control_volume_fields,
        ) = build_sharded_polar_angular_agglomeration_payload(
            owner_host_geometry,
            sharded_geometry.domain,
            compile_compact_transition_faces=False,
        )
        compact_transition_face_count = int(
            getattr(control_volume_descriptor, "compact_face_count", 0)
        )
        control_volume_boundary_bc = empty_angular_agglomeration_boundary_bc(
            max_rows=compact_transition_face_count
        )
        control_volume_assembler = assemble_local_polar_angular_agglomeration_geometry
        print(
            "[geometry] production eta-shardable angular RLP payload ready: "
            f"owners={int(np.count_nonzero(owner_host_geometry.topology.is_active_owner))}, "
            f"aliases={int(np.count_nonzero(owner_host_geometry.topology.is_merge_source))}, "
            f"runtime_channels={RLP_PACKED_FIELD_COUNT}, "
            f"compact_transition_faces={compact_transition_face_count}",
            flush=True,
        )
    if args.parallel_operator_scheme == "fci":
        if not sharded_geometry.maps_valid or sharded_geometry.map_fields is None:
            parser.error(
                "FCI parallel operators require finite generated maps; "
                "map generation or sharded lowering was invalid"
            )
    domain = sharded_geometry.domain
    print(
        f"sharded geometry inputs ready in "
        f"{time.perf_counter() - geometry_start:.3f} s: "
        f"global={sharded_geometry.global_shape}, "
        f"owned_per_shard={domain.layout.owned_shape}, "
        f"halo_per_shard={domain.layout.cell_halo_shape}, "
        f"periodic_axes={domain.periodic_axes}, "
        f"axis_regular_axes={domain.axis_regular_axes}; "
        f"lowering={time.perf_counter() - lowering_start:.3f} s",
        flush=True,
    )
    if args.geometry_only:
        return

    restart_time = 0.0
    restart_used = args.restart_from is not None
    if restart_used:
        try:
            initial_state, restart_time = _load_restart_state(
                args.restart_from,
                resolution=resolution,
                frame=int(args.restart_frame),
            )
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        if restart_time < 0.0 or restart_time >= float(args.final_time):
            parser.error(
                "restart time must be nonnegative and strictly earlier than "
                "--final-time"
            )
        if any(float(value) < restart_time - 1.0e-14 for value in args.snapshot_times):
            parser.error(
                "restart runs cannot schedule snapshots earlier than the "
                "restart time"
            )
        print(
            f"[restart] loaded all seven fields from {args.restart_from}; "
            f"frame={int(args.restart_frame)}, start_time={restart_time:.6e}",
            flush=True,
        )
    timestep = (float(args.final_time) - restart_time) / float(args.num_steps)
    print(
        f"time integration: final_time={float(args.final_time):.6e}, "
        f"num_steps={int(args.num_steps)}, dt={timestep:.6e}",
        flush=True,
    )
    diffusion = float(args.perp_diffusion)
    parallel_diffusion = float(args.parallel_diffusion)
    parameters = FciDrbEBRhsParameters(
        tau=float(args.tau),
        mi_over_me=float(args.mi_over_me),
        rho_star=float(args.rho_star),
        phi_inversion_iterations=int(args.gmres_max_iterations),
        phi_inversion_regularization=0.0,
        density_D_perp=diffusion,
        density_D_parallel=parallel_diffusion,
        electron_temperature_chi_parallel=parallel_diffusion,
        electron_temperature_D_perp=diffusion,
        ion_temperature_chi_parallel=parallel_diffusion,
        ion_temperature_D_perp=diffusion,
        Ve_nu=float(args.electron_collision_frequency),
        Ve_D_perp=diffusion,
        Ve_parallel_viscosity=parallel_diffusion,
        Vi_D_perp=diffusion,
        Vi_parallel_viscosity=parallel_diffusion,
        vorticity_D_perp=diffusion,
        vorticity_D_parallel=parallel_diffusion,
    )
    corner_edge_rate_threshold = None
    corner_edge_characteristic_speed = None
    if not restart_used:
        initial_state = build_initial_state(
            global_geometry,
            initialization=str(args.blob_initialization),
            density_amplitude=float(args.density_amplitude),
            temperature_amplitude=float(args.temperature_amplitude),
            blob_center=tuple(float(value) for value in args.blob_center),
            blob_width=float(args.blob_width),
            toroidal_perturbation_amplitude=float(
                args.toroidal_perturbation_amplitude
            ),
            toroidal_perturbation_mode=int(args.toroidal_perturbation_mode),
            toroidal_perturbation_phase=float(
                args.toroidal_perturbation_phase
            ),
            periodic_axes=descriptor.periodic_axes,
            axis_regular_axes=descriptor.axis_regular_axes,
        )
    if owner_host_geometry is not None:
        reused_materialized_owners = False
        if restart_used:
            (
                initial_state,
                reused_materialized_owners,
            ) = _restore_materialized_cell_owner_state(
                initial_state,
                owner_host_geometry,
            )
        if not reused_materialized_owners:
            initial_state = _aggregate_initial_owner_state(
                initial_state, owner_host_geometry
            )
        _assert_owner_sparse(initial_state, owner_host_geometry)
        print(
            "[simulation] initial cell state "
            + (
                "restored exactly from checkpoint materialized owners"
                if reused_materialized_owners
                else "volume-aggregated into canonical owners"
            )
            + "; "
            + "all seven fields use the canonical cell-owner basis",
            flush=True,
        )
    if restart_used:
        print(
            "[simulation] using restart state; preserving all seven saved fields "
            "including phi",
            flush=True,
        )
        print(
            "[simulation] restart phi policy: "
            + (
                "reconstruct with the current polarization/wall closure"
                if args.reconstruct_restart_phi
                else "preserve saved phi for exact continuation"
            )
            + " (source=--reconstruct-restart-phi)",
            flush=True,
        )
    elif args.toroidal_perturbation_amplitude > 0.0:
        symmetry = (
            "field-period symmetric"
            if int(args.toroidal_perturbation_mode) % int(nfp) == 0
            else "breaks field-period symmetry"
        )
        print(
            "[simulation] initial toroidal seed: "
            f"amplitude={float(args.toroidal_perturbation_amplitude):.3e}, "
            f"mode={int(args.toroidal_perturbation_mode)} ({symmetry})",
            flush=True,
        )
    print(
        "[simulation] flux framework: "
        f"{str(args.flux_framework)}"
        + (
            "; curvature split=production-path; parallel material=production-path"
            "; characteristic solver=canonical-face-state"
            if args.flux_framework == "production-split"
            else ""
        ),
        flush=True,
    )
    print(
        "[simulation] curvature: production characteristic owner-face; "
        "wall closure: BC-characteristic operator trace",
        flush=True,
    )
    print(
        "[simulation] parallel operator scheme: "
        f"{str(args.parallel_operator_scheme)}; FCI trace substeps: "
        f"{artifact_metadata.get('fci_trace_substeps', 'artifact')}",
        flush=True,
    )
    print(
        "[simulation] Poisson bracket scheme: "
        f"{str(args.poisson_bracket_scheme)}",
        flush=True,
    )
    print(
        "[simulation] Neumann ghost scheme: "
        f"{str(args.neumann_ghost_scheme)}",
        flush=True,
    )
    print(
        "[simulation] physical wall model: "
        f"{str(args.physical_wall_model)} ("
        f"{physical_wall_model_provenance}"
        ")",
        flush=True,
    )
    print(
        "[simulation] Rung-3 wall-layer initialization: "
        + (
            f"enabled ({int(args.rung3_wall_layer_cells)} requested radial "
            "owner cells; live FCI velocity blend followed by compatible "
            "upper-wall traces, phi gauge, and derived vorticity)"
            if args.initialize_rung3_wall_layer
            else "disabled"
        ),
        flush=True,
    )
    print(
        "[simulation] parallel velocity wall BC: "
        f"{str(args.parallel_velocity_wall_bc)}",
        flush=True,
    )
    print(
        "[simulation] parallel characteristic wall law: "
        f"{str(args.parallel_characteristic_wall_law)} "
        "(source=simulate_hsx_blob.py:--parallel-characteristic-wall-law)",
        flush=True,
    )
    print(
        "[simulation] parallel short-leg selection: "
        f"{str(args.parallel_short_leg_selection)} "
        "(all-physical-walls uses no CFL threshold; selected material and "
        "electron Ti-force are one IMEX stage residual)",
        flush=True,
    )
    print(
        "[simulation] GMRES settings: "
        f"target={float(args.gmres_target_tolerance):.3e}, "
        f"acceptance={float(args.gmres_acceptance_tolerance):.3e}, "
        f"max_iterations={int(args.gmres_max_iterations)}, "
        f"restart={min(int(args.gmres_restart), int(args.gmres_max_iterations))}, "
        f"residual_corrections={int(args.gmres_residual_correction_steps)}, "
        f"preconditioner={str(args.gmres_preconditioner)}, "
        + (
            "solver_space=owner-grid-RLP"
            if control_volume_descriptor is not None
            else "solver_space=full-grid"
        ),
        flush=True,
    )
    print(
        f"[simulation] time integrator: {str(args.time_integrator)}",
        flush=True,
    )
    run_full_eb(
        initial_state,
        simulation_geometry=simulation_geometry,
        sharded_geometry=sharded_geometry,
        mesh=mesh,
        parameters=parameters,
        gmres_target_tolerance=float(args.gmres_target_tolerance),
        gmres_acceptance_tolerance=float(args.gmres_acceptance_tolerance),
        gmres_max_iterations=int(args.gmres_max_iterations),
        gmres_restart=int(args.gmres_restart),
        gmres_preconditioner=str(args.gmres_preconditioner),
        gmres_residual_correction_steps=int(
            args.gmres_residual_correction_steps
        ),
        parallel_operator_scheme=str(args.parallel_operator_scheme),
        time_integrator=str(args.time_integrator),
        imex_split=str(args.imex_split),
        advance_execution=str(args.advance_execution),
        num_steps=int(args.num_steps),
        timestep=timestep,
        start_time=restart_time,
        output_path=args.output,
        save_every=int(args.save_every),
        phase_timing=not bool(args.no_phase_timing),
        diagnostic_every=int(args.diagnostic_every),
        checkpoint_every=int(args.checkpoint_every),
        snapshot_times=tuple(float(value) for value in args.snapshot_times),
        snapshot_dir=args.snapshot_dir,
        snapshot_term_fields=bool(args.snapshot_term_fields),
        track_rhs_terms=bool(args.track_rhs_terms),
        rhs_replay_history=args.rhs_replay_history,
        rhs_replay_frames=rhs_replay_frames,
        rhs_replay_output=args.rhs_replay_output,
        rhs_replay_electron_force_wall_audit=bool(
            args.rhs_replay_electron_force_wall_audit
        ),
        rhs_replay_execution=str(args.rhs_replay_execution),
        initialize_rung3_wall_layer=bool(args.initialize_rung3_wall_layer),
        rung3_wall_layer_cells=int(args.rung3_wall_layer_cells),
        run_metadata={
            "command": " ".join(sys.argv),
            "drbx_source_root": str(DRBX_SRC),
            **_topology_metadata(descriptor),
            "restart_from": None if args.restart_from is None else str(args.restart_from),
            "restart_frame": int(args.restart_frame),
            "reconstruct_restart_phi_requested": bool(
                args.reconstruct_restart_phi
            ),
            "reconstruct_initial_phi_effective": bool(
                (not restart_used) or args.reconstruct_restart_phi
            ),
            "reconstruct_initial_phi_source": (
                "new-initial-state"
                if not restart_used
                else (
                    "simulate_hsx_blob.py:--reconstruct-restart-phi"
                    if args.reconstruct_restart_phi
                    else "restart-state-preserved"
                )
            ),
            "rung3_wall_layer_initialization_requested": bool(
                args.initialize_rung3_wall_layer
            ),
            "rung3_wall_layer_initialization_effective": bool(
                args.initialize_rung3_wall_layer
                and args.physical_wall_model == "simplified-gbs-mpe"
            ),
            "rung3_wall_layer_cells_requested": int(
                args.rung3_wall_layer_cells
            ),
            "rung3_wall_layer_cells_effective": (
                int(args.rung3_wall_layer_cells)
                if args.initialize_rung3_wall_layer
                else 0
            ),
            "rung3_wall_layer_initialization_algorithm": (
                "cubic-smoothstep-owner-rings-to-live-directional-fci-wall-targets-"
                "then-compact-quintic-boundary-compatible-upper-wall-traces-"
                "wall-area-phi-gauge-and-polarization-derived-vorticity"
                if args.initialize_rung3_wall_layer
                else None
            ),
            "final_time": float(args.final_time),
            "num_steps": int(args.num_steps),
            "dt": float(timestep),
            "sharding_policy": "eta-only",
            "shard_counts": [int(value) for value in shard_counts],
            "halo_width": int(args.halo_width),
            "fit_sample_shape": artifact_metadata.get("fit_sample_shape"),
            "radial_degree": artifact_metadata.get("radial_degree"),
            "vertical_degree": artifact_metadata.get("vertical_degree"),
            "toroidal_modes": artifact_metadata.get("toroidal_modes"),
            "metric_mesh_shape": artifact_metadata.get("metric_mesh_shape"),
            "metric_radial_degree": artifact_metadata.get("metric_radial_degree"),
            "metric_poloidal_modes": artifact_metadata.get("metric_poloidal_modes"),
            "metric_toroidal_modes": artifact_metadata.get("metric_toroidal_modes"),
            "eta_projection_iterations": artifact_metadata.get("eta_projection_iterations"),
            "parallel_operator_scheme": str(args.parallel_operator_scheme),
            "parallel_flux_pairing": os.environ.get("DRBX_PARALLEL_FLUX_PAIRING", "legacy"),
            "parallel_current_pairing": str(args.parallel_current_pairing),
            "parallel_current_pairing_source": "simulate_hsx_blob.py:--parallel-current-pairing",
            "parallel_characteristic_wall_law": str(args.parallel_characteristic_wall_law),
            "parallel_characteristic_wall_law_env": os.environ.get("DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"),
            "parallel_characteristic_wall_law_source": "simulate_hsx_blob.py:--parallel-characteristic-wall-law",
            "parallel_boundary_pairing": os.environ.get("DRBX_PARALLEL_BOUNDARY_PAIRING", "legacy"),
            "parallel_boundary_pairing_source": "simulate_hsx_blob.py:--parallel-boundary-pairing",
            "parallel_short_leg_treatment": os.environ.get("DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"),
            "parallel_short_leg_treatment_source": "simulate_hsx_blob.py:--parallel-short-leg-treatment",
            "parallel_short_leg_selection": str(args.parallel_short_leg_selection),
            "parallel_short_leg_selection_source": "simulate_hsx_blob.py:--parallel-short-leg-selection",
            "parallel_short_leg_cfl_limit": float(os.environ.get("DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT", "2.5")),
            "parallel_short_leg_cfl_limit_source": "simulate_hsx_blob.py:--parallel-short-leg-cfl-limit",
            "parallel_short_leg_implicit_terms": (
                [] if args.imex_split == "coupled-boundary" else [
                    "selected-characteristic-material-action",
                    "selected-mu-tau-grad-parallel-Ti",
                ]
                if args.parallel_short_leg_treatment == "local-backward-euler"
                else []
            ),
            "parallel_short_leg_explicit_energy_pair": (
                None if args.imex_split == "coupled-boundary" else
                "mu-grad-parallel-phi<->weighted-adjoint-current-divergence"
            ),
            "parallel_short_leg_time_handoff": (
                "inactive-coupled-stage"
                if args.imex_split == "coupled-boundary" else
                "imex-ssp222-stage-wise"
                if args.parallel_short_leg_treatment == "local-backward-euler"
                else "none"
            ),
            "curvature_wall_flux_closure": (
                "bc-characteristic-operator-trace-canonical-face-state"
            ),
            "curvature_wall_flux_closure_source": "fixed production method",
            "curvature_wall_characteristic_jump": "direct-boundary-minus-interior",
            **_parallel_characteristic_wall_metadata(str(args.parallel_characteristic_wall_law)),
            "field_locations": {"Vi": "cell-center", "Ve": "cell-center"},
            "perpendicular_velocity_geometry": "face-to-center-perpendicular-center-to-face",
            "fci_trace_substeps": artifact_metadata.get("fci_trace_substeps"),
            "metric_spline_degree": artifact_metadata.get("metric_spline_degree"),
            "mmpde_iterations": artifact_metadata.get("mmpde_iterations"),
            "axis_core_radius": artifact_metadata.get("axis_core_radius"),
            "reference_magnetic_field": (
                artifact_metadata.get("reference_magnetic_field")
            ),
            "tau": float(args.tau),
            "rho_star": float(args.rho_star),
            "mi_over_me": float(args.mi_over_me),
            "perp_diffusion": float(args.perp_diffusion),
            "parallel_diffusion": float(args.parallel_diffusion),
            "electron_collision_frequency": float(
                args.electron_collision_frequency
            ),
            "time_integrator": str(args.time_integrator),
            "imex_split": str(args.imex_split),
            "advance_execution": str(args.advance_execution),
            "advance_execution_requested": requested_advance_execution,
            "staged_audit_cells": [
                [int(index) for index in cell]
                for cell in staged_audit_cells
            ],
            "staged_audit_output": (
                None
                if args.staged_audit_output is None
                else str(args.staged_audit_output)
            ),
            "staged_audit_explicit_ablation": str(
                args.staged_audit_explicit_ablation
            ),
            "advance_execution_kernel_layout": (
                ("host-eager-coupled-boundary-stage", "complete-E-plus-I-split", "stage-diagnostics")
                if args.imex_split == "coupled-boundary" else (
                    (
                    "implicit-short-leg-plus-phi",
                    "explicit-rhs",
                    "standalone-phi",
                    "stage-diagnostics",
                    )
                if args.advance_execution == "staged-compiled"
                else ("monolithic-advance",)
                )
            ),
            "rhs_replay_execution": str(args.rhs_replay_execution),
            "rhs_replay_execution_requested": requested_rhs_replay_execution,
            "flux_framework": str(args.flux_framework),
            "flux_framework_env": os.environ.get("DRBX_FLUX_FRAMEWORK", "legacy"),
            "flux_framework_source": "simulate_hsx_blob.py:--flux-framework",
            "production_characteristic_solver": (
                "canonical-face-state"
                if args.flux_framework == "production-split"
                else None
            ),
            "production_characteristic_solver_source": (
                "fixed production method"
                if args.flux_framework == "production-split"
                else None
            ),
            "curvature_operator": "production-characteristic-owner-face",
            "curvature_operator_source": "fixed production method",
            "curvature_edge_one_form": (
                "direct-continuous-shared-edge"
                if args.curvature_edge_one_form
                else "cell-centered-edge-average"
            ),
            "parallel_material_scheme": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME"),
            "parallel_material_scheme_env": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME"),
            "parallel_material_scheme_source": (
                "DRBX_PARALLEL_MATERIAL_SCHEME"
                if os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME") is not None
                else None
            ),
            "parallel_vorticity_advection_scheme": os.environ.get(
                "DRBX_PARALLEL_VORTICITY_ADVECTION_SCHEME", "first-order"
            ),
            "parallel_vorticity_advection_scheme_source": (
                "DRBX_PARALLEL_VORTICITY_ADVECTION_SCHEME"
                if os.environ.get("DRBX_PARALLEL_VORTICITY_ADVECTION_SCHEME")
                is not None
                else "default"
            ),
            "parallel_material_fallback_representation": os.environ.get(
                "DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION", "legacy-p"
            ),
            "parallel_material_fallback_representation_source": (
                "DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION"
                if os.environ.get(
                    "DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION"
                )
                is not None
                else "default"
            ),
            "parallel_material_div_b_fallback_scheme": os.environ.get(
                "DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME", "legacy-p"
            ),
            "parallel_material_div_b_fallback_scheme_source": (
                "DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME"
                if os.environ.get(
                    "DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME"
                )
                is not None
                else "default"
            ),
            "gmres_target_tolerance": float(
                args.gmres_target_tolerance
            ),
            "gmres_acceptance_tolerance": float(
                args.gmres_acceptance_tolerance
            ),
            "gmres_max_iterations": int(args.gmres_max_iterations),
            "gmres_restart": int(args.gmres_restart),
            "gmres_residual_correction_steps": int(
                args.gmres_residual_correction_steps
            ),
            "gmres_preconditioner": str(args.gmres_preconditioner),
            "phi_solver_diagnostic_width": int(_PHI_DIAGNOSTIC_WIDTH),
            "phi_solver_diagnostic_names": list(_PHI_SOLVER_DIAGNOSTIC_NAMES),
            "phi_inversion_regularization": float(
                parameters.phi_inversion_regularization
            ),
            "phi_inversion_regularization_source": (
                "simulate_hsx_blob.py:parameters.phi_inversion_regularization"
            ),
            "phi_solver_space": (
                "owner-grid-RLP"
                if control_volume_descriptor is not None
                else "full-grid"
            ),
            "neumann_ghost_scheme": str(args.neumann_ghost_scheme),
            "physical_wall_model": str(args.physical_wall_model),
            "conducting_sheath_wall_potential": (
                None
                if args.conducting_sheath_wall_potential is None
                else float(args.conducting_sheath_wall_potential)
            ),
            "physical_wall_model_provenance": physical_wall_model_provenance,
            "physical_wall_model_provenance_source": (
                "simulate_hsx_blob.py:selector-derived"
            ),
            "parallel_velocity_wall_bc": str(
                args.parallel_velocity_wall_bc
            ),
            "poisson_bracket_scheme": str(args.poisson_bracket_scheme),
            "polarization_operator_form": str(
                args.polarization_operator_form
            ),
            "axis_treatment": (
                "radius-dependent-angular-rlp"
                if args.topology == "toroidal"
                else (
                    "square-corner-edge-rlp"
                    if args.square_agglomeration == "corner-edge"
                    else "none"
                )
            ),
            "angular_owner_profile": (
                (
                    "explicit-diagnostic"
                    if angular_group_profile
                    else "radius-dependent"
                )
                if args.topology == "toroidal"
                else "none"
            ),
            "angular_group_profile_override": (
                list(angular_group_profile) if angular_group_profile else None
            ),
            "angular_group_sizes": (
                None
                if args.topology != "toroidal"
                else [
                    int(v)
                    for v in np.asarray(
                        owner_host_geometry.angular_group_size
                    ).tolist()
                ]
            ),
            "angular_profile_safety_ratio": angular_profile_safety_ratio,
            "angular_owner_count": (
                None
                if args.topology != "toroidal"
                else int(
                    np.count_nonzero(owner_host_geometry.topology.is_active_owner)
                )
            ),
            "angular_alias_count": (
                None
                if args.topology != "toroidal"
                else int(
                    np.count_nonzero(owner_host_geometry.topology.is_merge_source)
                )
            ),
            "square_agglomeration": str(args.square_agglomeration),
            "corner_edge_volume_ratio": float(args.agglomeration_volume_ratio),
            "corner_edge_rate_threshold": corner_edge_rate_threshold,
            "corner_edge_characteristic_speed": corner_edge_characteristic_speed,
            "corner_edge_seed_count": (
                None
                if args.square_agglomeration != "corner-edge"
                else int(np.count_nonzero(owner_host_geometry.seed_mask))
            ),
            "corner_edge_owner_count": (
                None
                if args.square_agglomeration != "corner-edge"
                else int(
                    np.count_nonzero(owner_host_geometry.topology.is_active_owner)
                )
            ),
            "corner_edge_alias_count": (
                None
                if args.square_agglomeration != "corner-edge"
                else int(
                    np.count_nonzero(owner_host_geometry.topology.is_merge_source)
                )
            ),
        },
        reconstruct_initial_phi=(not restart_used) or bool(
            args.reconstruct_restart_phi
        ),
        neumann_ghost_scheme=str(args.neumann_ghost_scheme),
        parallel_velocity_wall_bc=str(args.parallel_velocity_wall_bc),
        physical_wall_model=str(args.physical_wall_model),
        conducting_sheath_wall_potential=(
            None
            if args.conducting_sheath_wall_potential is None
            else float(args.conducting_sheath_wall_potential)
        ),
        poisson_bracket_scheme=str(args.poisson_bracket_scheme),
        polarization_operator_form=str(args.polarization_operator_form),
        parallel_material_scheme=(
            "production-path"
            if str(args.flux_framework) == "production-split"
            else "legacy"
        ),
        parallel_material_fallback_representation=os.environ.get(
            "DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION", "legacy-p"
        ),
        parallel_material_div_b_fallback_scheme=os.environ.get(
            "DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME", "legacy-p"
        ),
        track_curvature_chain_rule_defect=bool(
            args.track_curvature_chain_rule_defect
        ),
        control_volume_descriptor=control_volume_descriptor,
        control_volume_fields_host=control_volume_fields,
        control_volume_boundary_bc=control_volume_boundary_bc,
        control_volume_assembler=control_volume_assembler,
        control_volume_field_count=control_volume_field_count,
        staged_audit_cells=staged_audit_cells,
        staged_audit_output=args.staged_audit_output,
        staged_audit_explicit_ablation=str(
            args.staged_audit_explicit_ablation
        ),
    )


if __name__ == "__main__":
    main()
