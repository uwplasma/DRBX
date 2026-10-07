#!/usr/bin/env python3
"""Run a full seven-field EB blob on a precomputed HSX FCI geometry artifact.

Entry points: ``drbx run <deck.toml>`` (deck with ``[model] backend =
"fci_braginskii"``) and the compatibility script ``simulate_hsx_blob.py``.
Both build one :class:`HsxBlobConfig` and call :func:`run`.

The construction path is:

    FCI simulation geometry artifact (--geometry)
      -> FciGeometry3D + RLP owner geometry
      -> LocalFciGeometry3D + LocalDomain3D
      -> LocalFciDrbEBRhs.

Geometry is generated separately; this driver only loads it.  The driver
requires a toroidal geometry artifact and uses its radius-dependent angular
agglomeration with the projected fine-grid ``R A_f P`` formulation.
Parallel derivatives always use the artifact's axis-regular FCI maps, and
time advancement always uses IMEX-SSP222.  Physical-wall FCI endpoints
sample operator-specific ghost/leg fills.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Callable, Sequence
import zipfile

# Defaults resolve against the repository checkout, as they did when this
# module was the root ``simulate_hsx_blob.py`` script.
SCRIPT_DIR = Path(__file__).resolve().parents[3]
DRBX_SRC = Path(__file__).resolve().parents[2]

from drbx.runtime import configure_jax_runtime  # noqa: E402

configure_jax_runtime(precision="float64")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402
from drbx.fci_braginskii.geometry.fci_geometry import (  # noqa: E402
    FciGeometry3D,
    LocalCurvatureFaceCoefficients3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    build_local_curvature_face_coefficients,
    interpolate_B_contravariant,
)
from drbx.fci_braginskii.geometry.fci_simulation_geometry import (  # noqa: E402
    load_fci_simulation_geometry,
)
from drbx.fci_braginskii.native.fci_halo import (  # noqa: E402
    GhostFillWeights1D,
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    MetricAwarePhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
    make_default_topology_halo_filler_3d,
)
from drbx.fci_braginskii.native.fci_sharding import (  # noqa: E402
    ShardedFciGeometry3D,
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)
from drbx.fci_braginskii.native.fci_gmres import SolvaxGmresConfig  # noqa: E402
from drbx.fci_braginskii.native.fci_angular_agglomeration import (  # noqa: E402
    RLP_PACKED_FIELD_COUNT,
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
    empty_angular_agglomeration_boundary_bc,
)
from drbx.fci_braginskii.native.fci_boundaries import (  # noqa: E402
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)
from drbx.fci_braginskii.native.fci_drb_EB_rhs import (  # noqa: E402
    FciDrbEBRhsParameters,
    FciDrbEBState,
    LocalFciDrbEBPhysicalWallBundle,
    LocalFciDrbEBRhs,
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    prepare_local_fci_drb_eb_state,
)
from drbx.fci_braginskii.native.fci_operators import (  # noqa: E402
    build_local_perp_laplacian_face_projectors,
    expand_local_control_volume_owner_field,
)
DEFAULT_FILAMENT_CACHE_DIR = SCRIPT_DIR / ".hsx_filament_cache"
PLASMA_OPERATOR_BASELINE = "d340f4d638e0d9495546d1fd2e2c66c1c206704a"
GEOMETRY_PRODUCER_METADATA_KEYS = (
    "fit_sample_shape",
    "radial_degree",
    "vertical_degree",
    "toroidal_modes",
    "metric_mesh_shape",
    "metric_radial_degree",
    "metric_poloidal_modes",
    "metric_toroidal_modes",
    "eta_projection_iterations",
    "metric_spline_degree",
    "mmpde_iterations",
    "axis_core_radius",
    "reference_magnetic_field",
)
FILAMENT_CACHE_FORMAT_VERSION = 2
GMRES_TARGET_TOLERANCE = 1.0e-8
FCI_BRAGINSKII_BACKEND = "fci_braginskii"
METRIC_FIELDS = (
    "J",
    "g11",
    "g22",
    "g33",
    "g12",
    "g13",
    "g23",
    "g_11",
    "g_22",
    "g_33",
    "g_12",
    "g_13",
    "g_23",
)


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
    if selected == "toroidal":
        return TopologyDescriptor(
            name="toroidal",
            coordinate_names=("u", "theta", "eta"),
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
            logical_extents=((0.0, 1.0), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)),
        )
    raise ValueError(
        f"the HSX backend requires a toroidal geometry artifact, got topology {selected!r}"
    )


_TOROIDAL_TOPOLOGY = topology_descriptor("toroidal")
# Backward-compatible symbols used by the production toroidal runtime path.
PERIODIC_AXES = _TOROIDAL_TOPOLOGY.periodic_axes
AXIS_REGULAR_AXES = _TOROIDAL_TOPOLOGY.axis_regular_axes


def _topology_metadata(descriptor: TopologyDescriptor) -> dict[str, object]:
    return {
        "topology": descriptor.name,
        "coordinate_names": list(descriptor.coordinate_names),
        "periodic_axes": list(descriptor.periodic_axes),
        "axis_regular_axes": list(descriptor.axis_regular_axes),
        "logical_extents": [list(extent) for extent in descriptor.logical_extents],
    }


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


def build_face_bc_bundle(
    state: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    parameters: FciDrbEBRhsParameters,
) -> LocalFciDrbEBPhysicalWallBundle:
    """Build the physical wall bundle on the wall-fitted chart sides.

    Parallel velocities always use no-flow Dirichlet wall traces
    (Vi=Ve=0 primitive face traces); this is the fixed production
    configuration.
    """

    empty = LocalBoundaryFaceBC3D.empty(geometry.layout)
    mask_x = (
        empty.mask_x.at[0]
        .set(domain.runtime_has_physical_lower(0))
        .at[-1]
        .set(domain.runtime_has_physical_upper(0))
    )
    mask_y = (
        empty.mask_y.at[:, 0, :]
        .set(domain.runtime_has_physical_lower(1))
        .at[:, -1, :]
        .set(domain.runtime_has_physical_upper(1))
    )
    neumann = replace(
        empty,
        kind_x=empty.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=(
            empty.kind_y.at[:, 0, :]
            .set(BC_NEUMANN)
            .at[:, -1, :]
            .set(BC_NEUMANN)
        ),
        mask_x=mask_x,
        mask_y=mask_y,
    )
    dirichlet = replace(
        empty,
        kind_x=(
            empty.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET)
        ),
        kind_y=(
            empty.kind_y.at[:, 0, :]
            .set(BC_DIRICHLET)
            .at[:, -1, :]
            .set(BC_DIRICHLET)
        ),
        mask_x=mask_x,
        mask_y=mask_y,
    )
    return LocalFciDrbEBPhysicalWallBundle(
        density=neumann,
        phi=dirichlet,
        Te=neumann,
        Ti=neumann,
        Vi=dirichlet,
        Ve=dirichlet,
        vorticity=dirichlet,
    )


def _wrapped_periodic_offset(
    values: np.ndarray,
    reference: float,
    period: float,
) -> np.ndarray:
    """Return signed shortest offsets from ``reference`` on a periodic axis."""

    return (
        np.mod(
            np.asarray(values, dtype=np.float64)
            - float(reference)
            + 0.5 * float(period),
            float(period),
        )
        - 0.5 * float(period)
    )


def _filament_cache_path(
    geometry: FciGeometry3D,
    *,
    cache_dir: Path | None,
    blob_center: tuple[float, float],
    blob_width: float,
    reference_eta: float,
    parallel_half_length: float,
    tracing_substeps_per_plane: int,
) -> Path | None:
    if cache_dir is None:
        return None

    digest = hashlib.sha256()
    digest.update(f"field-aligned-filament-v{FILAMENT_CACHE_FORMAT_VERSION}".encode())
    spec = {
        "shape": list(geometry.shape),
        "blob_center": [float(value) for value in blob_center],
        "blob_width": float(blob_width),
        "reference_eta": float(reference_eta),
        "parallel_half_length": float(parallel_half_length),
        "tracing_substeps_per_plane": int(tracing_substeps_per_plane),
    }
    digest.update(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    )
    for values in (
        geometry.grid.x.centers,
        geometry.grid.y.centers,
        geometry.grid.z.centers,
        geometry.cell_bfield.B_contra,
    ):
        contiguous = np.ascontiguousarray(
            np.asarray(values, dtype=np.float64)
        )
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())

    cache_dir = cache_dir.resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"[filament-cache] disabled: {error}", flush=True)
        return None
    return cache_dir / f"hsx_filament_{digest.hexdigest()[:24]}.npz"


def _trace_logical_labels_to_eta_plane(
    geometry: FciGeometry3D,
    *,
    reference_eta: float,
    tracing_substeps_per_plane: int,
    periodic_axes: tuple[bool, bool, bool] = PERIODIC_AXES,
    axis_regular_axes: tuple[bool, bool, bool] = AXIS_REGULAR_AXES,
    min_abs_b_eta: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Backtrace every cell center to one eta plane in a single JAX solve."""

    grid = geometry.grid
    shape = geometry.shape
    eta_lower = float(grid.z.faces[0])
    eta_upper = float(grid.z.faces[-1])
    eta_period = eta_upper - eta_lower
    reference_eta = (
        np.mod(float(reference_eta) - eta_lower, eta_period) + eta_lower
    )

    u, v, eta = np.meshgrid(
        np.asarray(grid.x.centers, dtype=np.float64),
        np.asarray(grid.y.centers, dtype=np.float64),
        np.asarray(grid.z.centers, dtype=np.float64),
        indexing="ij",
    )
    offset_from_reference = _wrapped_periodic_offset(
        eta,
        reference_eta,
        eta_period,
    )
    trace_delta = -offset_from_reference.reshape(-1)
    points = np.stack((u, v, eta), axis=-1).reshape(-1, 3)

    minimum_eta_width = float(
        np.min(np.asarray(grid.z.widths, dtype=np.float64))
    )
    maximum_plane_count = max(
        1,
        int(np.ceil(np.max(np.abs(trace_delta)) / minimum_eta_width)),
    )
    integration_steps = max(
        1,
        maximum_plane_count * int(tracing_substeps_per_plane),
    )
    step_per_cell = trace_delta / float(integration_steps)

    u_lower = jnp.asarray(grid.x.faces[0], dtype=jnp.float64)
    u_upper = jnp.asarray(grid.x.faces[-1], dtype=jnp.float64)
    v_lower = jnp.asarray(grid.y.faces[0], dtype=jnp.float64)
    v_upper = jnp.asarray(grid.y.faces[-1], dtype=jnp.float64)
    v_period = v_upper - v_lower
    b_eta_floor = jnp.asarray(float(min_abs_b_eta), dtype=jnp.float64)

    def trace_all(
        initial_points: jax.Array,
        per_cell_step: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        initial_alive = jnp.ones(initial_points.shape[0], dtype=bool)

        def normalize_topology(values: jax.Array) -> jax.Array:
            result = values
            if axis_regular_axes[0]:
                reflected = result[..., 0] < u_lower
                result = result.at[..., 0].set(
                    jnp.where(reflected, 2.0 * u_lower - result[..., 0], result[..., 0])
                )
                result = result.at[..., 1].set(
                    jnp.where(reflected, result[..., 1] + jnp.pi, result[..., 1])
                )
            if periodic_axes[1]:
                result = result.at[..., 1].set(
                    jnp.mod(result[..., 1] - v_lower, v_period) + v_lower
                )
            return result

        def rhs(values: jax.Array) -> jax.Array:
            values = normalize_topology(values)
            b_contra = interpolate_B_contravariant(
                geometry,
                values,
                periodic_axes=periodic_axes,
                boundary_value=jnp.nan,
            )
            b_eta = b_contra[..., 2]
            safe_b_eta = jnp.where(
                jnp.abs(b_eta) < b_eta_floor,
                jnp.where(b_eta < 0.0, -b_eta_floor, b_eta_floor),
                b_eta,
            )
            return jnp.stack(
                (
                    b_contra[..., 0] / safe_b_eta,
                    b_contra[..., 1] / safe_b_eta,
                    jnp.ones_like(safe_b_eta),
                ),
                axis=-1,
            )

        def body(
            _index: int,
            carry: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            state, alive = carry
            step = per_cell_step[:, None]
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * step * k1)
            k3 = rhs(state + 0.5 * step * k2)
            k4 = rhs(state + step * k3)
            candidate = state + (step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            candidate = normalize_topology(candidate)
            finite = jnp.all(jnp.isfinite(candidate), axis=-1)
            in_cross_section = (
                (candidate[:, 0] >= u_lower)
                & (candidate[:, 0] <= u_upper)
                & (
                    periodic_axes[1]
                    | (
                        (candidate[:, 1] >= v_lower)
                        & (candidate[:, 1] <= v_upper)
                    )
                )
            )
            next_alive = alive & finite & in_cross_section
            next_state = jnp.where(next_alive[:, None], candidate, state)
            return next_state, next_alive

        return jax.lax.fori_loop(
            0,
            integration_steps,
            body,
            (initial_points, initial_alive),
        )

    print(
        "[initialization] compiling one-time full-torus field-line label trace "
        f"({integration_steps} RK4 substeps, {int(np.prod(shape))} cells)",
        flush=True,
    )
    trace_start = time.perf_counter()
    traced_points, valid = jax.jit(trace_all)(
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(step_per_cell, dtype=jnp.float64),
    )
    jax.block_until_ready((traced_points, valid))
    print(
        f"[initialization] field-line labels traced in "
        f"{time.perf_counter() - trace_start:.3f} s",
        flush=True,
    )
    return (
        np.asarray(traced_points, dtype=np.float64).reshape(shape + (3,)),
        np.asarray(valid, dtype=bool).reshape(shape),
    )


def build_field_aligned_filament_profile(
    geometry: FciGeometry3D,
    *,
    blob_center: tuple[float, float],
    blob_width: float,
    reference_eta: float,
    parallel_half_length: float,
    tracing_substeps_per_plane: int,
    cache_dir: Path | None,
    rebuild_cache: bool,
    periodic_axes: tuple[bool, bool, bool] = PERIODIC_AXES,
    axis_regular_axes: tuple[bool, bool, bool] = AXIS_REGULAR_AXES,
) -> jnp.ndarray:
    """Build a finite, seam-safe filament from backtraced field-line labels."""

    eta_lower = float(geometry.grid.z.faces[0])
    eta_upper = float(geometry.grid.z.faces[-1])
    eta_period = eta_upper - eta_lower
    reference_eta = (
        np.mod(float(reference_eta) - eta_lower, eta_period) + eta_lower
    )
    cache_path = _filament_cache_path(
        geometry,
        cache_dir=cache_dir,
        blob_center=blob_center,
        blob_width=blob_width,
        reference_eta=reference_eta,
        parallel_half_length=parallel_half_length,
        tracing_substeps_per_plane=tracing_substeps_per_plane,
    )

    profile = None
    if cache_path is not None and cache_path.is_file() and not rebuild_cache:
        try:
            load_start = time.perf_counter()
            with np.load(cache_path, allow_pickle=False) as cached:
                if (
                    int(cached["format_version"].item())
                    != FILAMENT_CACHE_FORMAT_VERSION
                ):
                    raise ValueError("filament cache format mismatch")
                candidate = np.asarray(cached["profile"], dtype=np.float64)
            if candidate.shape != geometry.shape:
                raise ValueError(
                    f"cached profile shape {candidate.shape} != {geometry.shape}"
                )
            if not np.all(np.isfinite(candidate)):
                raise ValueError("cached profile contains nonfinite values")
            profile = candidate
            print(
                f"[filament-cache] loaded {cache_path} in "
                f"{time.perf_counter() - load_start:.3f} s",
                flush=True,
            )
        except (
            EOFError,
            KeyError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            print(
                f"[filament-cache] ignored ({error}); rebuilding",
                flush=True,
            )

    if profile is None:
        traced_points, valid = _trace_logical_labels_to_eta_plane(
            geometry,
            reference_eta=reference_eta,
            tracing_substeps_per_plane=tracing_substeps_per_plane,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
        )
        eta = np.asarray(geometry.grid.z.centers, dtype=np.float64)
        parallel_offset = _wrapped_periodic_offset(
            eta,
            reference_eta,
            eta_period,
        )
        absolute_offset = np.abs(parallel_offset)
        envelope = np.where(
            absolute_offset < float(parallel_half_length),
            np.cos(
                0.5
                * np.pi
                * absolute_offset
                / float(parallel_half_length)
            )
            ** 2,
            0.0,
        )
        u_centers = np.asarray(
            geometry.grid.x.centers,
            dtype=np.float64,
        )
        v_centers = np.asarray(
            geometry.grid.y.centers,
            dtype=np.float64,
        )
        u_mesh, v_mesh = np.meshgrid(
            u_centers,
            v_centers,
            indexing="ij",
        )
        center_fit_width = max(
            0.5 * float(blob_width),
            1.5
            * max(
                float(np.max(np.asarray(geometry.grid.x.widths))),
                float(np.max(np.asarray(geometry.grid.y.widths))),
            ),
        )
        if axis_regular_axes[0]:
            center_distance_squared = (
                traced_points[..., 0] ** 2
                + float(blob_center[0]) ** 2
                - 2.0
                * traced_points[..., 0]
                * float(blob_center[0])
                * np.cos(traced_points[..., 1] - float(blob_center[1]))
            )
        else:
            center_distance_squared = (
                (traced_points[..., 0] - float(blob_center[0])) ** 2
                + (traced_points[..., 1] - float(blob_center[1])) ** 2
            )
        center_weights = np.exp(
            -center_distance_squared / (2.0 * center_fit_width**2)
        ) * valid
        weight_sum = np.sum(center_weights, axis=(0, 1))
        if np.any(weight_sum <= np.finfo(np.float64).tiny):
            raise RuntimeError(
                "the requested filament center could not be traced to every "
                "supported eta plane"
            )
        center_u = np.sum(
            center_weights * u_mesh[:, :, None],
            axis=(0, 1),
        ) / weight_sum
        if axis_regular_axes[0]:
            center_v = np.arctan2(
                np.sum(center_weights * np.sin(v_mesh[:, :, None]), axis=(0, 1)),
                np.sum(center_weights * np.cos(v_mesh[:, :, None]), axis=(0, 1)),
            )
            perpendicular_distance_squared = (
                u_centers[:, None, None] ** 2
                + center_u[None, None, :] ** 2
                - 2.0
                * u_centers[:, None, None]
                * center_u[None, None, :]
                * np.cos(v_centers[None, :, None] - center_v[None, None, :])
            )
        else:
            center_v = np.sum(
                center_weights * v_mesh[:, :, None],
                axis=(0, 1),
            ) / weight_sum
            perpendicular_distance_squared = (
                (u_centers[:, None, None] - center_u[None, None, :]) ** 2
                + (v_centers[None, :, None] - center_v[None, None, :]) ** 2
            )
        perpendicular_profile = np.exp(
            -perpendicular_distance_squared / (2.0 * float(blob_width) ** 2)
        )
        profile = perpendicular_profile * envelope[None, None, :]
        print(
            "[initialization] field-aligned filament: "
            f"reference_eta={reference_eta:.6e}, "
            f"parallel_half_length={float(parallel_half_length):.6e}, "
            f"trace_valid={100.0 * float(np.mean(valid)):.2f}%, "
            f"profile_max={float(np.max(profile)):.6e}, "
            f"center_u=[{float(np.min(center_u)):.3f}, "
            f"{float(np.max(center_u)):.3f}], "
            f"center_v=[{float(np.min(center_v)):.3f}, "
            f"{float(np.max(center_v)):.3f}]",
            flush=True,
        )
        if cache_path is not None:
            try:
                np.savez_compressed(
                    cache_path,
                    format_version=np.asarray(
                        FILAMENT_CACHE_FORMAT_VERSION,
                        dtype=np.int64,
                    ),
                    profile=np.asarray(profile, dtype=np.float64),
                )
                print(f"[filament-cache] written to {cache_path}", flush=True)
            except OSError as error:
                print(f"[filament-cache] write failed: {error}", flush=True)

    b_contra = np.asarray(
        geometry.cell_bfield.B_contra,
        dtype=np.float64,
    )
    bmag = np.asarray(geometry.cell_bfield.Bmag, dtype=np.float64)
    gradient = np.stack(
        np.gradient(
            profile,
            np.asarray(geometry.grid.x.centers, dtype=np.float64),
            np.asarray(geometry.grid.y.centers, dtype=np.float64),
            np.asarray(geometry.grid.z.centers, dtype=np.float64),
            edge_order=2,
        ),
        axis=-1,
    )
    grad_parallel = np.sum((b_contra / bmag[..., None]) * gradient, axis=-1)
    print(
        "[initialization] sampled filament parallel gradient: "
        f"rms={float(np.sqrt(np.mean(grad_parallel**2))):.6e}, "
        f"max={float(np.max(np.abs(grad_parallel))):.6e}",
        flush=True,
    )
    return jnp.asarray(profile, dtype=jnp.float64)


def build_initial_state(
    geometry: FciGeometry3D,
    *,
    initialization: str,
    density_amplitude: float,
    temperature_amplitude: float,
    blob_center: tuple[float, float],
    blob_width: float,
    blob_reference_eta: float,
    blob_parallel_half_length: float,
    fieldline_substeps_per_plane: int,
    filament_cache_dir: Path | None,
    rebuild_filament_cache: bool,
    toroidal_perturbation_amplitude: float = 0.0,
    toroidal_perturbation_mode: int = 1,
    toroidal_perturbation_phase: float = 0.0,
    periodic_axes: tuple[bool, bool, bool] = PERIODIC_AXES,
    axis_regular_axes: tuple[bool, bool, bool] = AXIS_REGULAR_AXES,
) -> FciDrbEBState:
    if initialization == "field-aligned":
        profile = build_field_aligned_filament_profile(
            geometry,
            blob_center=blob_center,
            blob_width=blob_width,
            reference_eta=blob_reference_eta,
            parallel_half_length=blob_parallel_half_length,
            tracing_substeps_per_plane=fieldline_substeps_per_plane,
            cache_dir=filament_cache_dir,
            rebuild_cache=rebuild_filament_cache,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
        )
    elif initialization == "logical":
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
    electron_temperature = (
        jnp.ones(geometry.shape, dtype=jnp.float64)
        if initialization == "field-aligned"
        else 1.0 + float(temperature_amplitude) * profile
    )
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
    gmres_residual_correction_steps: int = 0,
    control_volume_geometry=None,
    control_volume_boundary_bc=None,
    curvature_face_coefficients_override: LocalCurvatureFaceCoefficients3D | None = None,
    implicit_current_phi_pair: bool = False,
) -> LocalFciDrbEBRhs:
    if gmres_restart < 1:
        raise ValueError("gmres_restart must be positive")
    if gmres_residual_correction_steps < 0:
        raise ValueError("gmres_residual_correction_steps must be non-negative")
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
    physical_ghost_filler = MetricAwarePhysicalGhostCellFiller3D(
        **ghost_filler_kwargs,
        geometry=geometry,
    )
    curvature_face_coefficients = (
        curvature_face_coefficients_override
        if curvature_face_coefficients_override is not None
        else build_local_curvature_face_coefficients(geometry, domain)
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
            preconditioner="line-u",
            residual_correction_steps=int(gmres_residual_correction_steps),
        ),
        face_bc_builder=build_face_bc_bundle,
        axis_regular_axes=domain.axis_regular_axes,
        curvature_face_coefficients=curvature_face_coefficients,
        control_volume_geometry=control_volume_geometry,
        control_volume_boundary_bc=control_volume_boundary_bc,
        implicit_current_phi_pair=bool(implicit_current_phi_pair),
    )
    model = LocalFciDrbEBRhs(
        **rhs_kwargs,
    )
    return model


class _JittedPhaseTimer:
    """Collect ordered host timestamps emitted by one compiled advance."""

    def __init__(self, *, expected_markers: int = 8, label: str = "imex-ssp222") -> None:
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


def _format_phi_solver_diagnostics(
    info: object,
) -> jax.Array:
    """Pack fixed-shape phi diagnostics; slot 4 is the strict-tolerance flag."""

    first_five = jnp.stack(
        (
            jnp.asarray(info.num_steps, dtype=jnp.float64),
            jnp.asarray(info.final_residual_rel_l2, dtype=jnp.float64),
            jnp.asarray(info.failed, dtype=jnp.float64),
            jnp.asarray(info.converged, dtype=jnp.float64),
            jnp.asarray(info.strict_converged, dtype=jnp.float64),
        )
    )
    return jnp.concatenate((first_five, jnp.zeros(2, dtype=jnp.float64)))


def _print_rk_stage_diagnostics(
    field_names: Sequence[str],
    rk_stage_diagnostics: np.ndarray,
) -> None:
    """Print complete stage-state and rate diagnostics for a failure path."""

    labels = (
        "current/implicit1",
        "imex-stage1/explicit1",
        "stage2-base/implicit2",
        "imex-stage2/explicit2",
        "next/weighted",
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
    for name in METRIC_FIELDS:
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


def _restart_step_offset(path: Path) -> int:
    """Return the global step stored in a snapshot/checkpoint, else 0.

    Restarted runs number their steps from this offset so that periodic
    checkpoints keep global step labels and cannot overwrite the checkpoint
    the run restarted from.  History files carry no step and give 0.
    """

    with np.load(path, allow_pickle=False) as data:
        if "step" not in data:
            return 0
        step = np.asarray(data["step"])
        return int(step.reshape(-1)[0]) if step.size == 1 else 0


def _format_snapshot_time(value: float) -> str:
    return f"{value:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")


def run_full_eb(
    initial_state: FciDrbEBState,
    *,
    global_geometry: FciGeometry3D,
    cell_positions: np.ndarray,
    nfp: int,
    sharded_geometry: ShardedFciGeometry3D,
    mesh: Mesh,
    parameters: FciDrbEBRhsParameters,
    gmres_target_tolerance: float,
    gmres_acceptance_tolerance: float,
    gmres_max_iterations: int,
    gmres_restart: int = 100,
    gmres_residual_correction_steps: int = 0,
    time_integrator: str = "imex-ssp222",
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
    run_metadata: dict[str, object] | None = None,
    reconstruct_initial_phi: bool = True,
    start_step: int = 0,
    parallel_operator_scheme: str = "fci",
    control_volume_descriptor=None,
    control_volume_fields_host=None,
    control_volume_boundary_bc=None,
    control_volume_assembler=None,
    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,
    owner_host_geometry=None,
    history_dtype: str = "float32",
    implicit_current_phi_pair: bool = False,
) -> FciDrbEBState:
    """Advance the global EB state."""

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
    if time_integrator != "imex-ssp222":
        raise ValueError("time_integrator must be 'imex-ssp222'")
    if advance_execution not in ("compiled", "staged-compiled", "eager"):
        raise ValueError(
            "advance_execution must be 'compiled', 'staged-compiled', or 'eager'"
        )
    if history_dtype not in ("float32", "float64"):
        raise ValueError("history_dtype must be 'float32' or 'float64'")
    history_numpy_dtype = (
        np.float32 if history_dtype == "float32" else np.float64
    )
    if gmres_restart < 1:
        raise ValueError("gmres_restart must be positive")
    if int(checkpoint_every) < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if parallel_operator_scheme != "fci":
        raise ValueError("parallel_operator_scheme must be 'fci'")
    if not sharded_geometry.domain.axis_regular_axes[0]:
        raise ValueError(
            "the FCI parallel operator requires toroidal topology"
        )
    if not sharded_geometry.maps_valid or sharded_geometry.map_fields is None:
        raise ValueError(
            "the FCI parallel operator requires valid sharded FCI maps"
        )

    domain = sharded_geometry.domain
    spatial_spec = P("x", "y", "z")
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
    curvature_face_setup = jax.jit(
        lambda: build_local_curvature_face_coefficients(
            host_local_geometry,
            host_domain,
        ).axes
    )
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

    def build_local_model(
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
    ) -> LocalFciDrbEBRhs:
        geometry_fields_owned = cell_fields_owned[..., :geometry_field_count]
        local_geometry = assemble_local_fci_geometry(
            sharded_geometry,
            geometry_fields_owned,
            map_fields_owned,
        )
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
            domain,
            parameters,
            gmres_target_tolerance=float(gmres_target_tolerance),
            gmres_acceptance_tolerance=float(gmres_acceptance_tolerance),
            gmres_max_iterations=int(gmres_max_iterations),
            gmres_restart=int(gmres_restart),
            gmres_residual_correction_steps=int(
                gmres_residual_correction_steps
            ),
            control_volume_geometry=local_control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            curvature_face_coefficients_override=(
                local_curvature_face_coefficients
            ),
            implicit_current_phi_pair=implicit_current_phi_pair,
        )

    phi_start = time.perf_counter()
    print(
        "[simulation] compiling and "
        + ("reconstructing" if reconstruct_initial_phi else "reusing")
        + " initial sharded phi",
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
        return phi, info.num_steps

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
    if reconstruct_initial_phi:
        initial_phi, initial_phi_iterations = reconstruct_phi(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
        )
        jax.block_until_ready((initial_phi, initial_phi_iterations))
        state = state.replace(phi=initial_phi)
        initial_phi_iteration_text = (
            f"GMRES iterations={int(np.asarray(initial_phi_iterations))}"
        )
    else:
        jax.block_until_ready(state)
        initial_phi_iteration_text = "GMRES reconstruction skipped"
    print(
        f"[simulation] initial sharded phi ready in "
        f"{time.perf_counter() - phi_start:.3f} s; "
        f"{initial_phi_iteration_text}",
        flush=True,
    )

    phase_timer = (
        _JittedPhaseTimer(
            expected_markers=6,
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

    def mark_pair_solve_phi(phi: jax.Array) -> None:
        """Record the same ordered gmres host marker ``reconstruct_stage_phi``
        would have recorded, for the coupled current/phi pair solve.

        ``_JittedPhaseTimer`` expects a fixed number of ordered markers per
        compiled step; the implicit current/phi pair path replaces
        ``reconstruct_stage_phi`` (which records its own marker) with
        ``solve_implicit_current_phi_pair`` (which does not), so the marker
        must be recorded explicitly here to keep the count matched.
        """
        if gmres_marker is not None:
            jax.debug.callback(
                gmres_marker,
                jnp.ravel(phi)[0],
                ordered=True,
            )

    def evaluate_operators(
        stage_state: FciDrbEBState,
        phi: jax.Array,
        model: LocalFciDrbEBRhs,
    ) -> FciDrbEBState:
        with jax.named_scope("operators"):
            rhs = model.evaluate_stage(
                stage_state,
                phi_owned=phi,
                short_leg_selection_dt=dt,
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
        rhs_abs_maxs = jnp.stack(tuple(
            jnp.stack(tuple(
                jnp.max(jnp.abs(value)) for _, value in rhs.field_items()
            ))
            for rhs in stage_rates
        ))
        for mesh_axis_name in ("x", "y", "z"):
            state_mins = jax.lax.pmin(state_mins, mesh_axis_name)
            state_maxs = jax.lax.pmax(state_maxs, mesh_axis_name)
            state_abs_maxs = jax.lax.pmax(state_abs_maxs, mesh_axis_name)
            rhs_abs_maxs = jax.lax.pmax(rhs_abs_maxs, mesh_axis_name)
        stage_diagnostics = jnp.stack(
            (state_mins, state_maxs, state_abs_maxs, rhs_abs_maxs), axis=-1
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
        for mesh_axis_name in ("x", "y", "z"):
            field_mins = jax.lax.pmin(field_mins, mesh_axis_name)
            field_maxs = jax.lax.pmax(field_maxs, mesh_axis_name)
            field_abs_maxs = jax.lax.pmax(field_abs_maxs, mesh_axis_name)
        diagnostics = jnp.stack((field_mins, field_maxs, field_abs_maxs), axis=1)
        return (
            next_state,
            diagnostics,
            gmres_iterations,
            gmres_stage_diagnostics,
            stage_diagnostics,
        )

    def full_imex_advance(
        current: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        current_time: jax.Array,
    ):
        """Advance with the complete short-wall residual at every IMEX stage."""

        del current_time
        model = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
        )
        gamma_dt = jnp.asarray(IMEX_SSP222_GAMMA, dtype=jnp.float64) * dt

        def implicit_stage(base: FciDrbEBState):
            updated, increment, _info = (
                model.apply_short_leg_implicit_material_step(
                    base,
                    solve_dt=gamma_dt,
                    selection_dt=dt,
                    phi_owned=base.phi,
                    return_increment=True,
                )
            )
            if model.implicit_current_phi_pair:
                with jax.named_scope("gmres"):
                    stage, pair_increment, pair_solver_info = (
                        model.solve_implicit_current_phi_pair(
                            updated, solve_dt=gamma_dt,
                        )
                    )
                mark_pair_solve_phi(stage.phi)
                phi_info = _format_phi_solver_diagnostics(pair_solver_info)
                total_increment = increment.axpy(pair_increment, scale=1.0)
            else:
                stage_phi, phi_info = reconstruct_stage_phi(updated, model)
                stage = updated.replace(phi=stage_phi)
                total_increment = increment
            implicit_rate = total_increment.map_fields(
                lambda value: value / gamma_dt
            )
            return stage, implicit_rate, phi_info

        # The persisted current state already carries its consistent algebraic
        # potential.  Solve the complete selected-wall residual before the
        # first explicit evaluation, not after a finished timestep.
        stage_1, implicit_1, gmres_info_1 = implicit_stage(current)
        explicit_1 = evaluate_operators(
            stage_1, stage_1.phi, model
        )

        stage_2_base = current.axpy(explicit_1, scale=dt).axpy(
            implicit_1,
            scale=(1.0 - 2.0 * IMEX_SSP222_GAMMA) * dt,
        )
        stage_2_base_phi, gmres_info_2_base = reconstruct_stage_phi(
            stage_2_base, model
        )
        stage_2_base = stage_2_base.replace(phi=stage_2_base_phi)
        stage_2, implicit_2, gmres_info_2 = implicit_stage(stage_2_base)
        explicit_2 = evaluate_operators(
            stage_2, stage_2.phi, model
        )

        weighted_rate = explicit_1.axpy(explicit_2, scale=1.0).axpy(
            implicit_1, scale=1.0
        ).axpy(implicit_2, scale=1.0).map_fields(lambda value: 0.5 * value)
        next_state = current.axpy(weighted_rate, scale=dt)
        next_phi, gmres_info_next = reconstruct_stage_phi(next_state, model)
        next_state = next_state.replace(phi=next_phi)
        return finalize_advance(
            next_state,
            model,
            (current, stage_1, stage_2_base, stage_2, next_state),
            (implicit_1, explicit_1, implicit_2, explicit_2, weighted_rate),
            (gmres_info_1, gmres_info_2_base, gmres_info_2, gmres_info_next),
        )

    full_advance = full_imex_advance
    stage_description = (
        "2 explicit operator stages, 2 complete short-wall solves, "
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
    advance_out_specs = (
        state_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
    )
    sharded_advance = jax.shard_map(
        full_advance,
        mesh=mesh,
        in_specs=(
            state_spec,
            geometry_spec,
            geometry_spec,
            geometry_spec,
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
            if model.implicit_current_phi_pair:
                with jax.named_scope("gmres"):
                    stage, pair_increment, pair_solver_info = (
                        model.solve_implicit_current_phi_pair(
                            updated, solve_dt=solve_dt,
                        )
                    )
                mark_pair_solve_phi(stage.phi)
                phi_info = _format_phi_solver_diagnostics(pair_solver_info)
                total_increment = increment.axpy(pair_increment, scale=1.0)
                return stage, total_increment, phi_info
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
            rhs = model.evaluate_stage(
                local_state,
                phi_owned=local_state.phi,
                short_leg_selection_dt=selection_dt,
            )
            rhs = model.project_galerkin_state(rhs)
            mark_operator(rhs)
            return rhs

        staged_explicit_sharded = jax.shard_map(
            staged_explicit_kernel,
            mesh=mesh,
            in_specs=(
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
        zero_info = jnp.zeros((7,), dtype=jnp.float64)
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
                _current_time,
            ) = advance_args
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
            implicit_1 = increment_1.map_fields(
                lambda value: value / gamma_dt
            )
            explicit_1 = staged_explicit(
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
            stage_2_base = stage_2_base_before_phi.replace(phi=stage_2_base_phi)
            stage_2, increment_2, gmres_info_2 = staged_implicit(
                stage_2_base,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                gamma_dt,
                dt_dynamic,
            )
            implicit_2 = increment_2.map_fields(
                lambda value: value / gamma_dt
            )
            explicit_2 = staged_explicit(
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
            next_state = next_state.replace(phi=next_phi)
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

    # Periodic checkpoints can be state-only.  Do not force compilation of
    # the comparatively expensive spatial inspection path unless diagnostics
    # or explicitly scheduled diagnostic snapshots already require it.
    wall_term_count = 4
    inspection_enabled = bool(diagnostic_every > 0 or snapshot_times)
    inspection = None
    if inspection_enabled:
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
            return inspection_diagnostics, wall_ghost_fields, polarization_residual

        inspection_out_specs = (replicated_spec, wall_spec, spatial_spec)
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
            "inspection path",
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
            int((run_metadata or {}).get("fci_trace_substeps", 4)),
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
    metadata.update(
        {
            "time_integrator": str(time_integrator),
            "resolution": list(sharded_geometry.global_shape),
            "shard_counts": list(sharded_geometry.shard_counts),
            "phi_solver_space": solver_space,
            "parallel_operator_scheme": str(parallel_operator_scheme),
            "fci_trace_substeps": int(
                (run_metadata or {}).get("fci_trace_substeps", 4)
            ),
            "checkpoint_every": int(checkpoint_every),
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

    def save_snapshot(
        requested_time: float,
        actual_time: float,
        step: int,
        *,
        inspected: tuple[np.ndarray, ...] | None = None,
        failure_reason: str | None = None,
        periodic_checkpoint: bool = False,
    ) -> None:
        step = int(step) + int(start_step)
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
        if inspected is not None:
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
    next_snapshot = 0
    while next_snapshot < len(snapshot_schedule) and snapshot_schedule[next_snapshot] <= start_time + 1.0e-14:
        save_snapshot(snapshot_schedule[next_snapshot], float(start_time), 0)
        next_snapshot += 1
    simulation_start = time.perf_counter()
    accumulated_step_seconds = 0.0
    accumulated_operator_seconds = 0.0
    accumulated_gmres_seconds = 0.0
    accumulated_gmres_iterations = 0.0
    relaxed_phi_acceptances = 0

    def execute_advance(*advance_args):
        with jax.disable_jit(advance_execution == "eager"):
            return compiled_advance(*advance_args)

    for step in range(1, int(num_steps) + 1):
        step_start = time.perf_counter()
        step_time = float(start_time) + (step - 1) * float(timestep)
        if phase_timer is not None:
            phase_timer.begin_step()
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
        relaxed_phi_acceptances += int(
            np.sum(
                (gmres_stage_diagnostics_host[:, 3] > 0.5)
                & (gmres_stage_diagnostics_host[:, 4] < 0.5)
            )
        )
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
        stage_finite = bool(
            np.all(np.isfinite(rk_stage_diagnostics_host[:, :, :3]))
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
            )
            save_snapshot(
                current_time,
                current_time,
                step,
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
                    ("imex1", "stage2-base", "imex2", "next"),
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
            )
            save_snapshot(
                current_time,
                current_time,
                step,
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
            )
            save_snapshot(
                current_time,
                current_time,
                step,
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
            )
            save_snapshot(
                current_time,
                current_time,
                step,
                failure_reason="nonpositive-eb-state",
            )
            raise FloatingPointError(
                f"nonpositive density/temperature after step {step}: "
                f"n_min={density_min:.6e}, T_min={temperature_min:.6e}"
            )
        inspection_host = None
        snapshot_due = (
            next_snapshot < len(snapshot_schedule)
            and snapshot_schedule[next_snapshot] <= current_time + 1.0e-14
        )
        periodic_checkpoint_due = (
            checkpoint_every > 0
            and (step + int(start_step)) % int(checkpoint_every) == 0
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
            )
            next_snapshot += 1
        if periodic_checkpoint_due:
            save_snapshot(
                current_time,
                current_time,
                step,
                inspected=inspection_host,
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
            int((metadata or {}).get("fci_trace_substeps", 4)),
            dtype=np.int64,
        ),
        history_dtype=np.asarray(history_dtype),
        **{
            name: np.stack(values, axis=0)
            for name, values in history.items()
        },
        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
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
    print(
        f"[simulation] phi solves accepted only under the relaxed acceptance "
        f"tolerance (strict tol missed): {relaxed_phi_acceptances} of "
        f"{4 * num_steps}",
        flush=True,
    )
    return materialized_state(state)


def _parallel_characteristic_wall_metadata() -> dict[str, object]:
    """Describe the fixed production physical-boundary-state wall law."""

    return {
        "parallel_material_wall_flux_closure": (
            "live-characteristic-physical-boundary-state"
        ),
        "parallel_material_wall_flux_closure_source": (
            "fixed production configuration"
        ),
        "parallel_characteristic_wall_equilibrium_reference": None,
        "parallel_characteristic_wall_equilibrium_reference_source": None,
        "parallel_characteristic_wall_provenance": (
            "physical-face-trace-live-characteristic-split"
        ),
        "parallel_characteristic_wall_energy_normalizer": None,
        "parallel_characteristic_wall_energy_normalizer_source": None,
    }


@dataclass(frozen=True)
class HsxBlobConfig:
    """One HSX FCI blob run.

    Field names are the ``simulate_hsx_blob.py`` option destinations and the
    keys of a deck's ``[fci_braginskii]`` table.  These defaults are the only
    default table: the argument parser and the TOML loader both read them.
    """

    geometry: Path | None = None
    shard_counts: tuple[int, int, int] = (1, 1, 1)
    halo_width: int = 2
    filament_cache_dir: Path = DEFAULT_FILAMENT_CACHE_DIR
    no_filament_cache: bool = False
    rebuild_filament_cache: bool = False
    final_time: float = 0.15
    num_steps: int = 200
    save_every: int = 1
    checkpoint_every: int = 0
    snapshot_times: tuple[float, ...] = ()
    snapshot_dir: Path | None = None
    restart_from: Path | None = None
    restart_frame: int = -1
    diagnostic_every: int = 0
    blob_initialization: str = "field-aligned"
    density_amplitude: float = 0.05
    temperature_amplitude: float = 0.0
    blob_center: tuple[float, float] = (0.65, 0.50)
    blob_width: float = 0.10
    blob_reference_eta: float = float(np.pi)
    blob_parallel_half_length: float = float(np.pi)
    fieldline_substeps_per_plane: int = 4
    toroidal_perturbation_amplitude: float = 0.0
    toroidal_perturbation_mode: int = 1
    toroidal_perturbation_phase: float = 0.0
    tau: float = 1.0
    rho_star: float = 5.0e-4
    mi_over_me: float = 1836.0
    perp_diffusion: float = 1.0e-5
    parallel_diffusion: float = 0.0
    electron_collision_frequency: float = 0.0
    implicit_current_phi_pair: bool = True
    advance_execution: str = "staged-compiled"
    gmres_acceptance_tolerance: float = 5.0e-5
    gmres_target_tolerance: float = GMRES_TARGET_TOLERANCE
    gmres_max_iterations: int = 500
    gmres_restart: int = 100
    gmres_residual_correction_steps: int = 1
    no_phase_timing: bool = False
    geometry_only: bool = False
    output: Path = SCRIPT_DIR / "hsx_blob_history.npz"


_DEFAULTS = HsxBlobConfig()
_PATH_FIELDS = frozenset(
    ("geometry", "filament_cache_dir", "snapshot_dir", "restart_from", "output")
)
_SEQUENCE_ITEM_TYPES = {
    "shard_counts": (int, 3),
    "snapshot_times": (float, None),
    "blob_center": (float, 2),
}


def _coerce_deck_value(name: str, value: object, base_dir: Path) -> object:
    def scalar(kind: type, item: object) -> object:
        if kind is float and isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
        if isinstance(item, kind) and (kind is bool or not isinstance(item, bool)):
            return item
        raise ValueError(f"[fci_braginskii] {name} must be {kind.__name__}, got {item!r}")

    if name in _PATH_FIELDS:
        path = Path(str(scalar(str, value))).expanduser()
        return path if path.is_absolute() else base_dir / path
    if name in _SEQUENCE_ITEM_TYPES:
        kind, length = _SEQUENCE_ITEM_TYPES[name]
        if not isinstance(value, list) or (length is not None and len(value) != length):
            raise ValueError(f"[fci_braginskii] {name} must be a list of {length or 'any number of'} values")
        return tuple(scalar(kind, item) for item in value)
    return scalar(type(getattr(_DEFAULTS, name)), value)


def load_hsx_blob_deck(path: str | Path) -> HsxBlobConfig:
    """Read an ``[model] backend = "fci_braginskii"`` TOML deck.

    Relative paths resolve against the deck directory; unknown tables or keys
    are errors.
    """

    from ..config.boutinp import tomllib

    source = Path(path)
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    unknown_tables = sorted(set(data) - {"model", "fci_braginskii"})
    if unknown_tables:
        raise ValueError(f"unknown table(s) in fci_braginskii deck: {unknown_tables}")
    model = data.get("model", {})
    if set(model) != {"backend"} or model["backend"] != FCI_BRAGINSKII_BACKEND:
        raise ValueError(f'[model] must contain only backend = "{FCI_BRAGINSKII_BACKEND}"')
    table = data.get("fci_braginskii", {})
    known = {field.name for field in fields(HsxBlobConfig)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [fci_braginskii] key(s): {unknown}")
    base_dir = source.resolve().parent
    config = HsxBlobConfig(
        **{name: _coerce_deck_value(name, value, base_dir) for name, value in table.items()}
    )
    if config.geometry is None:
        raise ValueError("[fci_braginskii] geometry is required")
    if config.blob_initialization not in ("field-aligned", "logical"):
        raise ValueError("[fci_braginskii] blob_initialization must be 'field-aligned' or 'logical'")
    if config.advance_execution not in ("compiled", "staged-compiled", "eager"):
        raise ValueError(
            "[fci_braginskii] advance_execution must be 'compiled', 'staged-compiled' or 'eager'"
        )
    return config


def _deck_error(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _build_parser(*, require_geometry: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full local/sharding-compatible seven-field electrostatic "
            "Boussinesq model on a precomputed FCI simulation geometry artifact."
        )
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=_DEFAULTS.geometry,
        required=require_geometry,
        help=(
            "Directory containing the precomputed FCI simulation geometry "
            "artifact. It sets the topology, resolution and RLP owner profile."
        ),
    )
    parser.add_argument(
        "--shard-counts",
        "--shards",
        nargs=3,
        type=int,
        metavar=("SU", "SV", "SETA"),
        default=_DEFAULTS.shard_counts,
        help=(
            "Production JAX decomposition in u, the second logical "
            "coordinate, and eta. Only eta decomposition is supported, so "
            "the first two entries must be one. Neta must be divisible by "
            "SETA, whose value must not exceed the available JAX device count."
        ),
    )
    parser.add_argument("--halo-width", type=int, default=_DEFAULTS.halo_width)
    parser.add_argument(
        "--filament-cache-dir",
        type=Path,
        default=_DEFAULTS.filament_cache_dir,
        help="Directory for reusable initial-filament field-line label caches.",
    )
    parser.add_argument(
        "--no-filament-cache",
        action="store_true",
        help="Disable loading and writing initial-filament label caches.",
    )
    parser.add_argument(
        "--rebuild-filament-cache",
        action="store_true",
        help="Ignore a matching filament cache and replace it after tracing.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=_DEFAULTS.final_time,
        help="Final normalized simulation time.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=_DEFAULTS.num_steps,
        help=(
            "Number of equal IMEX-SSP222 steps used to reach --final-time. "
            "The default gives dt = 7.5e-4 (0.15/200), validated with the "
            "implicit current/phi pair at the default rho*."
        ),
    )
    parser.add_argument("--save-every", type=int, default=_DEFAULTS.save_every)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=_DEFAULTS.checkpoint_every,
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
        default=_DEFAULTS.snapshot_times,
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
        default=_DEFAULTS.snapshot_dir,
        help="Directory for scheduled snapshot NPZ files; defaults to the output directory.",
    )
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=_DEFAULTS.restart_from,
        help="Restart from a snapshot NPZ or a saved history NPZ.",
    )
    parser.add_argument(
        "--restart-frame",
        type=int,
        default=_DEFAULTS.restart_frame,
        help="Frame to load from a history NPZ; ignored for a single snapshot.",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=_DEFAULTS.diagnostic_every,
        metavar="N",
        help=(
            "Print compiled min/max/max-absolute diagnostics for every state "
            "field every N steps; 0 disables periodic detailed output."
        ),
    )
    parser.add_argument(
        "--blob-initialization",
        choices=("field-aligned", "logical"),
        default=_DEFAULTS.blob_initialization,
        help=(
            "Use a backtraced, finite field-aligned density filament by "
            "default. 'logical' restores the eta-copied Gaussian."
        ),
    )
    parser.add_argument("--density-amplitude", type=float, default=_DEFAULTS.density_amplitude)
    parser.add_argument(
        "--temperature-amplitude",
        type=float,
        default=_DEFAULTS.temperature_amplitude,
        help=(
            "Electron-temperature perturbation used only by the legacy "
            "logical initialization. The field-aligned filament is "
            "density-only."
        ),
    )
    parser.add_argument(
        "--blob-center",
        nargs=2,
        type=float,
        metavar=("U0", "V0"),
        default=_DEFAULTS.blob_center,
    )
    parser.add_argument("--blob-width", type=float, default=_DEFAULTS.blob_width)
    parser.add_argument(
        "--blob-reference-eta",
        type=float,
        default=_DEFAULTS.blob_reference_eta,
        help="Toroidal reference plane of the field-aligned filament.",
    )
    parser.add_argument(
        "--blob-parallel-half-length",
        type=float,
        default=_DEFAULTS.blob_parallel_half_length,
        help=(
            "Half-length in eta radians of the compact cos^2 filament "
            "envelope. It must be no larger than pi so the perturbation "
            "vanishes at or before the full-torus periodic seam."
        ),
    )
    parser.add_argument(
        "--fieldline-substeps-per-plane",
        type=int,
        default=_DEFAULTS.fieldline_substeps_per_plane,
        help=(
            "RK4 tracing substeps per toroidal plane spacing, used only to "
            "construct the initial field-line labels."
        ),
    )
    parser.add_argument(
        "--toroidal-perturbation-amplitude",
        type=float,
        default=_DEFAULTS.toroidal_perturbation_amplitude,
        help=(
            "Relative cosine modulation used only by the legacy logical "
            "blob. The finite field-aligned filament already breaks "
            "field-period symmetry."
        ),
    )
    parser.add_argument(
        "--toroidal-perturbation-mode",
        type=int,
        default=_DEFAULTS.toroidal_perturbation_mode,
        help="Full-torus integer mode number used by the initial perturbation.",
    )
    parser.add_argument(
        "--toroidal-perturbation-phase",
        type=float,
        default=_DEFAULTS.toroidal_perturbation_phase,
        help="Initial toroidal perturbation phase in radians.",
    )
    parser.add_argument("--tau", type=float, default=_DEFAULTS.tau)
    parser.add_argument(
        "--rho-star",
        type=float,
        default=_DEFAULTS.rho_star,
        help=(
            "rho_s / L_ref with L_ref = 1 m (lengths in the geometry are in "
            "metres). Scales E x B and curvature drifts by rho*, "
            "polarization by rho*^2. The default is hydrogen at Te ~ 24 eV "
            "and B = 1 T; rho* scales as sqrt(Te)/B. --rho-star 1 "
            "--no-implicit-current-phi-pair reproduces the base-commit "
            "operators."
        ),
    )
    parser.add_argument("--mi-over-me", type=float, default=_DEFAULTS.mi_over_me)
    parser.add_argument("--perp-diffusion", type=float, default=_DEFAULTS.perp_diffusion)
    parser.add_argument("--parallel-diffusion", type=float, default=_DEFAULTS.parallel_diffusion)
    parser.add_argument("--electron-collision-frequency", type=float, default=_DEFAULTS.electron_collision_frequency)
    parser.add_argument(
        "--implicit-current-phi-pair",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.implicit_current_phi_pair,
        help=(
            "Treat mu*grad_par(phi) and the homogeneous parallel current "
            "divergence implicitly through a coupled potential solve in "
            "each implicit IMEX stage (default). "
            "--no-implicit-current-phi-pair restores the explicit "
            "treatment, which at rho* = 5e-4 limits dt to about 1e-4."
        ),
    )
    parser.add_argument(
        "--advance-execution",
        choices=("compiled", "staged-compiled", "eager"),
        default=_DEFAULTS.advance_execution,
        help=(
            "Execution mode for time advancement. 'compiled' builds one "
            "fused advance executable. 'staged-compiled' compiles reusable "
            "implicit, explicit, phi, and diagnostic shard-map kernels "
            "separately. 'eager' disables the outer jax.jit, although the "
            "JAX backend may still compile kernels."
        ),
    )
    parser.add_argument(
        "--gmres-acceptance-tolerance",
        "--gmres-tolerance",
        dest="gmres_acceptance_tolerance",
        type=float,
        default=_DEFAULTS.gmres_acceptance_tolerance,
        help=(
            "Relative and absolute residual accepted after GMRES stops. "
            "--gmres-tolerance is a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--gmres-target-tolerance",
        type=float,
        default=_DEFAULTS.gmres_target_tolerance,
        help="Relative and absolute residual target used to stop GMRES.",
    )
    parser.add_argument("--gmres-max-iterations", type=int, default=_DEFAULTS.gmres_max_iterations)
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=_DEFAULTS.gmres_restart,
        help="GMRES restart length; capped at --gmres-max-iterations.",
    )
    parser.add_argument(
        "--gmres-residual-correction-steps",
        type=int,
        default=_DEFAULTS.gmres_residual_correction_steps,
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
            "Stop after loading the geometry artifact and shard-local "
            "geometry lowering."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULTS.output,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser(require_geometry=True)
    run(HsxBlobConfig(**vars(parser.parse_args(argv))), fail=parser.error)


def run(args: HsxBlobConfig, *, fail: Callable[[str], object] = _deck_error) -> None:
    """Load geometry, build the initial state and advance one configured run."""

    geometry_start = time.perf_counter()
    try:
        simulation_geometry = load_fci_simulation_geometry(args.geometry)
        descriptor = topology_descriptor(simulation_geometry.topology_name)
        geometry_manifest_sha256 = hashlib.sha256(
            (args.geometry / "manifest.json").read_bytes()
        ).hexdigest()
    except (OSError, KeyError, TypeError, ValueError) as error:
        fail(f"could not load --geometry artifact: {error}")
    if simulation_geometry.polar_angular_geometry is None:
        fail("toroidal geometry artifacts must include the RLP topology")
    geometry_metadata = simulation_geometry.metadata
    resolution = tuple(int(value) for value in simulation_geometry.geometry.shape)
    print(
        f"[geometry] artifact={args.geometry}; topology={descriptor.name}; "
        f"shape={resolution}; manifest_sha256={geometry_manifest_sha256}; "
        f"trace_substeps={geometry_metadata.get('trace_substeps')}",
        flush=True,
    )
    print(
        "[simulation] flux_framework=production-split; "
        "parallel_velocities=cell-centered; "
        "parallel_flux_pairing=support-core; "
        "parallel_characteristic_wall_law=physical-boundary-state; "
        "parallel_boundary_pairing=characteristic-sat; "
        "parallel_short_leg_treatment=local-backward-euler; "
        "parallel_short_leg_selection=all-physical-walls",
        flush=True,
    )
    if resolution[1] % 2:
        fail("toroidal global NTHETA must be even")
    shard_counts = tuple(int(value) for value in args.shard_counts)
    if any(value < 1 for value in shard_counts):
        fail("--shard-counts entries must be positive")
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        fail(
            "production sharding is eta-only; use --shard-counts 1 1 NETA_SHARDS"
        )
    for axis, (cell_count, shard_count) in enumerate(
        zip(resolution, shard_counts)
    ):
        if cell_count % shard_count:
            fail(
                f"resolution axis {axis} ({cell_count}) is not divisible by "
                f"shard count {shard_count}"
            )
    try:
        mesh = make_shard_mesh(shard_counts)
    except (RuntimeError, ValueError) as error:
        fail(str(error))
    if args.num_steps < 1:
        fail("--num-steps must be positive")
    if args.final_time <= 0.0:
        fail("--final-time must be positive")
    if args.save_every < 1:
        fail("--save-every must be positive")
    if args.checkpoint_every < 0:
        fail("--checkpoint-every must be nonnegative")
    if any(float(value) < 0.0 for value in args.snapshot_times):
        fail("--snapshot-times must be nonnegative")
    if any(float(value) > float(args.final_time) for value in args.snapshot_times):
        fail("--snapshot-times must not exceed --final-time")
    if len(set(float(value) for value in args.snapshot_times)) != len(args.snapshot_times):
        fail("--snapshot-times must not contain duplicates")
    if args.gmres_target_tolerance < 0.0:
        fail("--gmres-target-tolerance must be nonnegative")
    if args.gmres_acceptance_tolerance < 0.0:
        fail("--gmres-acceptance-tolerance must be nonnegative")
    if args.gmres_max_iterations < 1:
        fail("--gmres-max-iterations must be positive")
    if args.gmres_restart < 1:
        fail("--gmres-restart must be positive")
    if args.gmres_residual_correction_steps < 0:
        fail("--gmres-residual-correction-steps must be nonnegative")
    if args.halo_width < 1:
        fail("--halo-width must be positive")
    if args.density_amplitude < 0.0:
        fail("--density-amplitude must be nonnegative")
    if args.temperature_amplitude < 0.0:
        fail("--temperature-amplitude must be nonnegative")
    if args.blob_width <= 0.0:
        fail("--blob-width must be positive")
    if not 0.0 < args.blob_parallel_half_length <= np.pi:
        fail(
            "--blob-parallel-half-length must lie in (0, pi]"
        )
    if args.fieldline_substeps_per_plane < 1:
        fail("--fieldline-substeps-per-plane must be positive")
    if args.diagnostic_every < 0:
        fail("--diagnostic-every must be nonnegative")
    if not 0.0 <= args.toroidal_perturbation_amplitude < 1.0:
        fail(
            "--toroidal-perturbation-amplitude must lie in [0, 1)"
        )
    if args.toroidal_perturbation_mode < 1:
        fail("--toroidal-perturbation-mode must be positive")
    if args.blob_initialization == "field-aligned":
        if args.temperature_amplitude != 0.0:
            fail(
                "--temperature-amplitude must be zero for the density-only "
                "field-aligned filament"
            )
        if args.toroidal_perturbation_amplitude != 0.0:
            fail(
                "--toroidal-perturbation-amplitude must be zero for the "
                "field-aligned filament; its finite parallel envelope "
                "already breaks field-period symmetry"
            )
    global_geometry = simulation_geometry.geometry
    cell_positions = simulation_geometry.cell_positions
    nfp = simulation_geometry.nfp
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
    control_volume_field_count = RLP_PACKED_FIELD_COUNT
    owner_host_geometry = simulation_geometry.polar_angular_geometry
    angular_profile_safety_ratio = float(
        geometry_metadata["angular_profile_safety_ratio"]
    )
    print(
        f"[angular-rlp-host] profile={owner_host_geometry.angular_group_size.tolist()} "
        f"minimum_width_ratio={angular_profile_safety_ratio:.6g}",
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
    if not sharded_geometry.maps_valid or sharded_geometry.map_fields is None:
        fail(
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
            fail(str(error))
        if restart_time < 0.0 or restart_time >= float(args.final_time):
            fail(
                "restart time must be nonnegative and strictly earlier than "
                "--final-time"
            )
        if any(float(value) < restart_time - 1.0e-14 for value in args.snapshot_times):
            fail(
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
    if not restart_used:
        initial_state = build_initial_state(
            global_geometry,
            initialization=str(args.blob_initialization),
            density_amplitude=float(args.density_amplitude),
            temperature_amplitude=float(args.temperature_amplitude),
            blob_center=tuple(float(value) for value in args.blob_center),
            blob_width=float(args.blob_width),
            blob_reference_eta=float(args.blob_reference_eta),
            blob_parallel_half_length=float(args.blob_parallel_half_length),
            fieldline_substeps_per_plane=int(
                args.fieldline_substeps_per_plane
            ),
            filament_cache_dir=(
                None if args.no_filament_cache else args.filament_cache_dir
            ),
            rebuild_filament_cache=bool(args.rebuild_filament_cache),
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
    elif args.blob_initialization == "field-aligned":
        print(
            "[simulation] initialized density-only field-aligned filament; "
            "Te=Ti=1, Vi=Ve=phi=vorticity=0; its finite full-torus "
            "parallel envelope breaks field-period symmetry",
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
        "[simulation] flux framework: production-split"
        "; curvature split=production-path; parallel material=production-path"
        "; characteristic solver=canonical-face-state",
        flush=True,
    )
    print(
        "[simulation] curvature: production characteristic owner-face; "
        "wall closure: BC-characteristic operator trace",
        flush=True,
    )
    print(
        "[simulation] parallel operator scheme: fci",
        flush=True,
    )
    print(
        "[simulation] Poisson bracket scheme: "
        "material-scalar-third-order-upwind (fixed production configuration)",
        flush=True,
    )
    print(
        "[simulation] Neumann ghost scheme: physical (fixed production configuration)",
        flush=True,
    )
    print(
        "[simulation] parallel velocity wall BC: dirichlet-zero "
        "(fixed production configuration)",
        flush=True,
    )
    print(
        "[simulation] parallel characteristic wall law: "
        "physical-boundary-state (fixed production configuration)",
        flush=True,
    )
    print(
        "[simulation] parallel short-leg selection: "
        "all-physical-walls (fixed production configuration; selected "
        "material and electron Ti-force are one IMEX stage residual)",
        flush=True,
    )
    print(
        "[simulation] GMRES settings: "
        f"target={float(args.gmres_target_tolerance):.3e}, "
        f"acceptance={float(args.gmres_acceptance_tolerance):.3e}, "
        f"max_iterations={int(args.gmres_max_iterations)}, "
        f"restart={min(int(args.gmres_restart), int(args.gmres_max_iterations))}, "
        f"residual_corrections={int(args.gmres_residual_correction_steps)}, "
        "preconditioner=line-u, "
        + (
            "solver_space=owner-grid-RLP"
            if control_volume_descriptor is not None
            else "solver_space=full-grid"
        ),
        flush=True,
    )
    print(
        "[simulation] time integrator: imex-ssp222",
        flush=True,
    )
    run_full_eb(
        initial_state,
        global_geometry=global_geometry,
        cell_positions=cell_positions,
        nfp=nfp,
        sharded_geometry=sharded_geometry,
        mesh=mesh,
        parameters=parameters,
        gmres_target_tolerance=float(args.gmres_target_tolerance),
        gmres_acceptance_tolerance=float(args.gmres_acceptance_tolerance),
        gmres_max_iterations=int(args.gmres_max_iterations),
        gmres_restart=int(args.gmres_restart),
        gmres_residual_correction_steps=int(
            args.gmres_residual_correction_steps
        ),
        parallel_operator_scheme="fci",
        time_integrator="imex-ssp222",
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
        implicit_current_phi_pair=bool(args.implicit_current_phi_pair),
        run_metadata={
            "command": " ".join(sys.argv),
            "drbx_source_root": str(DRBX_SRC),
            "geometry": str(args.geometry.resolve()),
            "geometry_manifest_sha256": geometry_manifest_sha256,
            "plasma_operator_baseline": PLASMA_OPERATOR_BASELINE,
            **_topology_metadata(descriptor),
            "restart_from": None if args.restart_from is None else str(args.restart_from),
            "restart_frame": int(args.restart_frame),
            "final_time": float(args.final_time),
            "num_steps": int(args.num_steps),
            "dt": float(timestep),
            "sharding_policy": "eta-only",
            "shard_counts": [int(value) for value in shard_counts],
            "halo_width": int(args.halo_width),
            **{key: geometry_metadata.get(key) for key in GEOMETRY_PRODUCER_METADATA_KEYS},
            "fci_trace_substeps": geometry_metadata.get("trace_substeps"),
            "parallel_operator_scheme": "fci",
            "parallel_flux_pairing": "support-core",
            "parallel_flux_pairing_source": "fixed production configuration",
            "parallel_characteristic_wall_law": "physical-boundary-state",
            "parallel_characteristic_wall_law_source": "fixed production configuration",
            "parallel_boundary_pairing": "characteristic-sat",
            "parallel_boundary_pairing_source": "fixed production configuration",
            "parallel_short_leg_treatment": "local-backward-euler",
            "parallel_short_leg_treatment_source": "fixed production configuration",
            "parallel_short_leg_selection": "all-physical-walls",
            "parallel_short_leg_selection_source": "fixed production configuration",
            "parallel_short_leg_implicit_terms": [
                "selected-characteristic-material-action",
                "selected-mu-tau-grad-parallel-Ti",
            ],
            "parallel_short_leg_explicit_energy_pair": (
                "mu-grad-parallel-phi<->weighted-adjoint-current-divergence"
            ),
            "parallel_short_leg_time_handoff": "imex-ssp222-stage-wise",
            "curvature_wall_flux_closure": (
                "bc-characteristic-operator-trace-canonical-face-state"
            ),
            "curvature_wall_flux_closure_source": "fixed production method",
            "curvature_wall_characteristic_jump": "direct-boundary-minus-interior",
            **_parallel_characteristic_wall_metadata(),
            "field_locations": {"Vi": "cell-center", "Ve": "cell-center"},
            "perpendicular_velocity_geometry": "face-to-center-perpendicular-center-to-face",
            "tau": float(args.tau),
            "rho_star": float(args.rho_star),
            "mi_over_me": float(args.mi_over_me),
            "perp_diffusion": float(args.perp_diffusion),
            "parallel_diffusion": float(args.parallel_diffusion),
            "electron_collision_frequency": float(
                args.electron_collision_frequency
            ),
            "implicit_current_phi_pair": bool(args.implicit_current_phi_pair),
            "time_integrator": "imex-ssp222",
            "advance_execution": str(args.advance_execution),
            "advance_execution_kernel_layout": (
                (
                    "implicit-short-leg-plus-phi",
                    "explicit-rhs",
                    "standalone-phi",
                    "stage-diagnostics",
                )
                if args.advance_execution == "staged-compiled"
                else ("monolithic-advance",)
            ),
            "flux_framework": "production-split",
            "flux_framework_source": "fixed production configuration",
            "production_characteristic_solver": "canonical-face-state",
            "production_characteristic_solver_source": "fixed production method",
            "curvature_operator": "production-characteristic-owner-face",
            "curvature_operator_source": "fixed production method",
            "parallel_material_scheme": "production-path",
            "parallel_material_scheme_source": "fixed production configuration",
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
            "gmres_preconditioner": "line-u",
            "phi_solver_space": (
                "owner-grid-RLP"
                if control_volume_descriptor is not None
                else "full-grid"
            ),
            "neumann_ghost_scheme": "physical",
            "parallel_velocity_wall_bc": "dirichlet-zero",
            "poisson_bracket_scheme": "material-scalar-third-order-upwind",
            "poisson_bracket_scheme_source": "fixed production configuration",
            "axis_treatment": "radius-dependent-angular-rlp",
            "angular_owner_profile": "geometry-artifact",
            "angular_group_sizes": [
                int(v)
                for v in np.asarray(
                    owner_host_geometry.angular_group_size
                ).tolist()
            ],
            "angular_profile_safety_ratio": angular_profile_safety_ratio,
            "angular_owner_count": int(
                np.count_nonzero(owner_host_geometry.topology.is_active_owner)
            ),
            "angular_alias_count": int(
                np.count_nonzero(owner_host_geometry.topology.is_merge_source)
            ),
        },
        reconstruct_initial_phi=not restart_used,
        start_step=(
            _restart_step_offset(args.restart_from) if restart_used else 0
        ),
        control_volume_descriptor=control_volume_descriptor,
        control_volume_fields_host=control_volume_fields,
        control_volume_boundary_bc=control_volume_boundary_bc,
        control_volume_assembler=control_volume_assembler,
        control_volume_field_count=control_volume_field_count,
        owner_host_geometry=owner_host_geometry,
    )


if __name__ == "__main__":
    main()
