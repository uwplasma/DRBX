#!/usr/bin/env python3
"""Run a full seven-field EB blob over the complete wall-fitted HSX torus.

The construction path is:

    MAKEGRID + vessel
      -> scalar-potential coordinate and wall-fitted MetricEvaluator
      -> transient FciGeometry3D
      -> LocalFciGeometry3D + LocalDomain3D
      -> LocalFciDrbEBRhs.

Toroidal runs use radius-dependent angular agglomeration with the projected
fine-grid ``R A_f P`` formulation.  Square runs can optionally use a
metric-driven corner/edge owner agglomeration through the same projection.
Parallel derivatives are selectable between coordinate
conservative stencils and cached, axis-regular FCI maps. Physical-wall FCI
endpoints sample operator-specific ghost/leg fills.
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
from types import SimpleNamespace
from typing import Mapping, Sequence
import zipfile

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
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    LocalDomain3D,
    LocalFciGeometry3D,
    MetricEvaluator,
    MetricGeometry,
    Spacing3D,
    WallEvaluator,
    bfield_evaluator_from_makegrid,
    build_fci_maps_from_callbacks,
    build_local_conservative_stencil_from_field,
    build_local_curvature_coefficients,
    build_local_curvature_face_coefficients,
    build_metric_aware_polar_angular_agglomeration_geometry,
    build_metric_evaluator,
    interpolate_B_contravariant,
    scalar_potential_evaluator_from_bfield,
)
from drbx.geometry.solve_MMPDE import MMPDEOptions  # noqa: E402
from drbx.geometry.fci_corner_edge_agglomeration import (  # noqa: E402
    build_corner_edge_agglomeration,
)
from drbx.native import (  # noqa: E402
    FciDrbEBRhsParameters,
    FciDrbEBState,
    GhostFillWeights1D,
    HaloExchange3D,
    LocalBoundaryFaceBC3D,
    LocalFciDrbEBFaceBCBundle,
    LocalFciDrbEBRhs,
    LocalPeriodicTopologyRule3D,
    MetricAwarePhysicalGhostCellFiller3D,
    PhysicalGhostCellFiller3D,
    ShardedFciGeometry3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    make_default_topology_halo_filler_3d,
    make_shard_mesh,
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
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    UpwindEquilibriumWallProjectors,
    build_upwind_equilibrium_wall_projectors,
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
    OUTGOING_FCI_FACE_OWNERSHIP_POLICY,
    build_local_outgoing_fci_face_topology_from_geometry,
    build_local_perp_laplacian_face_projectors,
    expand_local_control_volume_owner_field,
    local_curvature_conservative_components_op,
    prolong_local_outgoing_fci_face_owner_field,
)
DEFAULT_WORKSPACE_DATA_DIR = SCRIPT_DIR.parent
DEFAULT_MAKEGRID = DEFAULT_WORKSPACE_DATA_DIR / "mgrid_res2p5cm_180pln.nc"
DEFAULT_VESSEL = DEFAULT_WORKSPACE_DATA_DIR / "vessel_hsx_flare.txt"
DEFAULT_METRIC_CACHE_DIR = DEFAULT_WORKSPACE_DATA_DIR / ".hsx_metric_cache"
HSX_QHS_MAIN_CURRENT_AMPERES = 10722.0
DEFAULT_HSX_QHS_MAKEGRID_CURRENTS = (
    (HSX_QHS_MAIN_CURRENT_AMPERES,) * 6 + (0.0,) * 6
)
METRIC_CACHE_FORMAT_VERSION = 7
FILAMENT_CACHE_FORMAT_VERSION = 2
GMRES_TARGET_TOLERANCE = 1.0e-8
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

# This is intentionally duplicated as a small driver-level contract.  The
# global FciMaps3D object is the map-generation/cache boundary, so adding a
# field to that object must also update the cache and quality checks here.
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
FCI_MAP_FLOAT_FIELDS = FCI_MAP_FIELDS[:12]
FCI_MAP_BOOL_FIELDS = FCI_MAP_FIELDS[12:]
FCI_MAP_CACHE_PREFIX = "fci_maps_"
FCI_MAP_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class HSXMetricContext:
    """Reusable continuous HSX metric and magnetic-field representation.

    The evaluator is independent of the final PDE cell resolution.  Passing
    this context to :func:`build_hsx_fci_geometry` therefore avoids refitting
    the continuous metric when several PDE resolutions are sampled.  The
    sampled cell/face geometry and any FCI maps remain resolution-specific.
    """

    metric_evaluator: MetricEvaluator
    bfield: object
    nfp: int


def _validate_hsx_metric_context(
    context: HSXMetricContext,
    *,
    topology: str,
) -> HSXMetricContext:
    """Validate the explicit continuous representation reuse contract."""

    if not isinstance(context, HSXMetricContext):
        raise TypeError(
            "metric_context must be an HSXMetricContext produced from the "
            "HSX metric builder"
        )
    evaluator = context.metric_evaluator
    if not isinstance(evaluator, MetricEvaluator):
        raise TypeError("metric_context.metric_evaluator must be a MetricEvaluator")
    if evaluator.topology != str(topology).lower():
        raise ValueError(
            "metric_context topology does not match the requested geometry: "
            f"{evaluator.topology!r} != {str(topology).lower()!r}"
        )
    if isinstance(context.nfp, (bool, np.bool_)) or int(context.nfp) != context.nfp:
        raise ValueError("metric_context.nfp must be a positive integer")
    if int(context.nfp) < 1:
        raise ValueError("metric_context.nfp must be a positive integer")
    evaluator_nfp = evaluator.nfp
    if evaluator_nfp is not None and int(evaluator_nfp) != int(context.nfp):
        raise ValueError(
            "metric_context evaluator and field-period counts disagree: "
            f"{evaluator_nfp} != {context.nfp}"
        )
    try:
        bfield_nfp = int(context.bfield.nfp)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "metric_context.bfield must expose a positive integer nfp"
        ) from error
    if bfield_nfp != int(context.nfp):
        raise ValueError(
            "metric_context magnetic-field and field-period counts disagree: "
            f"{bfield_nfp} != {context.nfp}"
        )
    return context


def _hsx_fci_map_source_fingerprint() -> str:
    """Fingerprint the driver-visible FCI map schema and tracer source.

    The metric cache deliberately does not include ``fci_geometry.py`` in its
    metric-cache key: changing the tracer should not force an expensive metric
    rebuild.  Cached maps carry this separate fingerprint instead.
    """

    source_path = DRBX_SRC / "drbx" / "geometry" / "fci_geometry.py"
    try:
        stat = source_path.stat()
        source = {
            "path": str(source_path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        source = {"path": str(source_path), "unavailable": True}
    contract = {
        "cache_format": FCI_MAP_CACHE_FORMAT_VERSION,
        "fields": list(FCI_MAP_FIELDS),
        "endpoint_interpolation_order": 2,
        "source": source,
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically replace an NPZ cache file."""

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez(temporary_path, **payload)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_or_load_hsx_fci_maps(
    *,
    grid: CellCenteredGrid3D,
    topology: str,
    construct_fci_maps: bool,
    fci_trace_substeps: int,
    cache_payload: Mapping[str, np.ndarray] | None,
    cache_path: Path | None,
    metric_evaluator: MetricEvaluator | None,
    bfield: object | None,
    makegrid_path: Path,
    makegrid_currents: np.ndarray | None = None,
) -> tuple[FciMaps3D | None, dict[str, np.ndarray] | None, object | None]:
    """Load validated HSX maps or trace them from the continuous B callback."""

    if not construct_fci_maps:
        return None, None if cache_payload is None else dict(cache_payload), bfield
    if str(topology).lower() != "toroidal":
        raise ValueError(
            "construct_fci_maps=True is currently supported only for "
            "topology='toroidal'"
        )
    if int(fci_trace_substeps) < 1:
        raise ValueError(
            f"fci_trace_substeps must be >= 1, got {fci_trace_substeps}"
        )
    expected_shape = tuple(int(value) for value in grid.shape)
    source_fingerprint = _hsx_fci_map_source_fingerprint()
    payload = None if cache_payload is None else dict(cache_payload)
    maps = None
    if payload is not None:
        try:
            cached_substeps = int(
                np.asarray(payload["fci_maps_trace_substeps"]).item()
            )
            cached_source = str(
                np.asarray(payload["fci_maps_source_fingerprint"]).item()
            )
            if (
                cached_substeps != int(fci_trace_substeps)
                or cached_source != source_fingerprint
            ):
                raise ValueError("cached FCI map tracer metadata is stale")
            maps = fci_maps_from_metric_cache_payload(
                payload,
                expected_shape=expected_shape,
            )
            validate_hsx_fci_maps(maps, grid, topology="toroidal")
            print(
                "[fci-map-cache] validated cached full-torus HSX maps",
                flush=True,
            )
            return maps, payload, bfield
        except (KeyError, TypeError, ValueError, OSError) as error:
            print(
                f"[fci-map-cache] ignored cached maps ({error}); regenerating",
                flush=True,
            )

    if metric_evaluator is None:
        raise RuntimeError(
            "a MetricEvaluator is required to generate HSX FCI maps"
        )
    if bfield is None:
        print(
            "[fci-map-cache] loading MAKEGRID magnetic field for map tracing",
            flush=True,
        )
        bfield = bfield_evaluator_from_makegrid(
            makegrid_path,
            currents=makegrid_currents,
            method="cubic",
        )

    trace_start = time.perf_counter()

    def continuous_magnetic_field(points: np.ndarray):
        # MetricEvaluator handles the one-field-period Fourier wrapping in eta;
        # no cell-centered/materialized B field participates in tracing.
        return metric_evaluator.evaluate_magnetic_field(points, bfield)

    map_payload = build_fci_maps_from_callbacks(
        grid,
        continuous_magnetic_field,
        substeps=int(fci_trace_substeps),
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        endpoint_interpolation_order=2,
    )
    missing = [name for name in FCI_MAP_FIELDS if name not in map_payload]
    if missing:
        raise ValueError(
            f"HSX FCI map builder returned incomplete fields: {missing}"
        )
    maps = FciMaps3D(
        **{
            name: jnp.asarray(map_payload[name])
            for name in FCI_MAP_FIELDS
        }
    )
    report = validate_hsx_fci_maps(maps, grid, topology="toroidal")
    print(
        f"[fci-map-cache] traced full-torus HSX maps in "
        f"{time.perf_counter() - trace_start:.3f} s; "
        f"forward_boundary={report['counts']['forward_boundary']}, "
        f"backward_boundary={report['counts']['backward_boundary']}",
        flush=True,
    )

    if cache_path is not None and payload is not None:
        payload = add_fci_maps_to_metric_cache_payload(payload, maps)
        payload["fci_maps_source_fingerprint"] = np.asarray(source_fingerprint)
        payload["fci_maps_trace_substeps"] = np.asarray(
            int(fci_trace_substeps), dtype=np.int64
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_npz_atomic(cache_path, payload)
            print(
                f"[fci-map-cache] atomically added maps to {cache_path}",
                flush=True,
            )
        except OSError as error:
            print(f"[fci-map-cache] map-cache write failed: {error}", flush=True)
    return maps, payload, bfield


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


_SQUARE_TOPOLOGY = topology_descriptor("square")
# Backward-compatible symbols used by the existing square runtime path.
PERIODIC_AXES = _SQUARE_TOPOLOGY.periodic_axes
AXIS_REGULAR_AXES = _SQUARE_TOPOLOGY.axis_regular_axes


def fci_maps_to_metric_cache_payload(
    maps: FciMaps3D,
    *,
    prefix: str = FCI_MAP_CACHE_PREFIX,
) -> dict[str, np.ndarray]:
    """Serialize one global ``FciMaps3D`` into metric-cache arrays.

    Map generation is intentionally not performed here.  A future geometry
    builder can call this after tracing, then merge the returned arrays into
    the existing metric cache payload.  The explicit schema/version fields
    make stale or partial map payloads fail closed on reload.
    """

    if not isinstance(maps, FciMaps3D):
        raise TypeError(f"maps must be FciMaps3D, got {type(maps).__name__}")
    prefix = str(prefix)
    payload: dict[str, np.ndarray] = {
        f"{prefix}format_version": np.asarray(
            FCI_MAP_CACHE_FORMAT_VERSION,
            dtype=np.int64,
        ),
        f"{prefix}shape": np.asarray(maps.shape, dtype=np.int64),
    }
    for name in FCI_MAP_FLOAT_FIELDS:
        payload[f"{prefix}{name}"] = np.array(
            getattr(maps, name),
            dtype=np.float64,
            copy=True,
        )
    for name in FCI_MAP_BOOL_FIELDS:
        payload[f"{prefix}{name}"] = np.array(
            getattr(maps, name),
            dtype=bool,
            copy=True,
        )
    return payload


def add_fci_maps_to_metric_cache_payload(
    cache_payload: Mapping[str, np.ndarray],
    maps: FciMaps3D,
    *,
    prefix: str = FCI_MAP_CACHE_PREFIX,
) -> dict[str, np.ndarray]:
    """Return an existing cache payload extended with serialized FCI maps."""

    payload = dict(cache_payload)
    payload.update(fci_maps_to_metric_cache_payload(maps, prefix=prefix))
    return payload


def fci_maps_from_metric_cache_payload(
    cache_payload: Mapping[str, np.ndarray],
    *,
    expected_shape: tuple[int, int, int] | None = None,
    prefix: str = FCI_MAP_CACHE_PREFIX,
) -> FciMaps3D:
    """Deserialize and structurally validate cached global FCI maps."""

    prefix = str(prefix)
    required = (
        f"{prefix}format_version",
        f"{prefix}shape",
        *(f"{prefix}{name}" for name in FCI_MAP_FIELDS),
    )
    missing = [name for name in required if name not in cache_payload]
    if missing:
        raise KeyError(f"cached FCI map payload is missing {missing}")
    version = int(np.asarray(cache_payload[f"{prefix}format_version"]).item())
    if version != FCI_MAP_CACHE_FORMAT_VERSION:
        raise ValueError(
            "cached FCI map format version mismatch: "
            f"got {version}, expected {FCI_MAP_CACHE_FORMAT_VERSION}"
        )
    stored_shape_array = np.asarray(cache_payload[f"{prefix}shape"])
    if stored_shape_array.shape != (3,):
        raise ValueError(
            f"cached FCI map shape metadata must have shape (3,), "
            f"got {stored_shape_array.shape}"
        )
    stored_shape = tuple(int(value) for value in stored_shape_array)
    if any(value <= 0 for value in stored_shape):
        raise ValueError(f"cached FCI map shape must be positive, got {stored_shape}")
    if expected_shape is not None:
        expected_shape = tuple(int(value) for value in expected_shape)
        if stored_shape != expected_shape:
            raise ValueError(
                f"cached FCI map shape {stored_shape} does not match "
                f"expected {expected_shape}"
            )

    arrays: dict[str, np.ndarray] = {}
    for name in FCI_MAP_FLOAT_FIELDS:
        value = np.asarray(cache_payload[f"{prefix}{name}"])
        if value.shape != stored_shape:
            raise ValueError(
                f"cached FCI map {name} has shape {value.shape}, "
                f"expected {stored_shape}"
            )
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"cached FCI map {name} must be numeric")
        arrays[name] = np.array(value, dtype=np.float64, copy=True)
    for name in FCI_MAP_BOOL_FIELDS:
        value = np.asarray(cache_payload[f"{prefix}{name}"])
        if value.shape != stored_shape:
            raise ValueError(
                f"cached FCI map {name} has shape {value.shape}, "
                f"expected {stored_shape}"
            )
        if not np.issubdtype(value.dtype, np.bool_):
            raise TypeError(f"cached FCI map {name} must have boolean dtype")
        arrays[name] = np.array(value, dtype=bool, copy=True)
    return FciMaps3D(**arrays)


def fci_map_quality_report(
    maps: FciMaps3D | Mapping[str, np.ndarray],
    grid: CellCenteredGrid3D,
    *,
    topology: str = "toroidal",
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> dict[str, object]:
    """Report strict structural/topological checks for a full-torus map set.

    The validator is deliberately independent of map generation.  In the
    toroidal topology the lower radial axis is an identified coordinate
    singularity, not a wall; only the upper ``u`` face may be marked as a
    physical cross-section boundary.
    """

    errors: list[str] = []
    checks: dict[str, bool] = {}
    shape = tuple(int(value) for value in grid.shape)
    arrays: dict[str, np.ndarray] = {}

    def get_field(name: str):
        if isinstance(maps, Mapping):
            if name not in maps:
                errors.append(f"missing field {name}")
                return None
            return maps[name]
        if not hasattr(maps, name):
            errors.append(f"missing field {name}")
            return None
        return getattr(maps, name)

    shape_ok = True
    for name in FCI_MAP_FIELDS:
        value = get_field(name)
        if value is None:
            shape_ok = False
            continue
        array = np.asarray(value)
        arrays[name] = array
        if array.shape != shape:
            shape_ok = False
            errors.append(
                f"{name} has shape {array.shape}, expected {shape}"
            )
    checks["shapes"] = shape_ok
    if not shape_ok:
        return {
            "valid": False,
            "topology": str(topology),
            "shape": shape,
            "checks": checks,
            "counts": {},
            "errors": tuple(errors),
        }

    float_finite_ok = all(
        np.all(np.isfinite(arrays[name])) for name in FCI_MAP_FLOAT_FIELDS
    )
    checks["finite_float_fields"] = bool(float_finite_ok)
    if not float_finite_ok:
        errors.append("one or more floating-point map fields contain NaN or inf")
    bool_dtype_ok = all(
        np.issubdtype(arrays[name].dtype, np.bool_)
        for name in FCI_MAP_BOOL_FIELDS
    )
    checks["boolean_boundary_fields"] = bool(bool_dtype_ok)
    if not bool_dtype_ok:
        errors.append("boundary masks must have boolean dtype")

    lengths_ok = all(
        np.all(arrays[name] > 0.0)
        for name in ("forward_length", "backward_length")
    )
    checks["positive_connection_lengths"] = bool(lengths_ok)
    if not lengths_ok:
        errors.append("connection lengths must be finite and strictly positive")

    selected_topology = str(topology).lower()
    try:
        descriptor = topology_descriptor(selected_topology)
    except ValueError as error:
        descriptor = None
        errors.append(str(error))
    topology_ok = descriptor is not None and selected_topology == "toroidal"
    checks["full_torus_toroidal_topology"] = bool(topology_ok)
    if descriptor is not None and selected_topology != "toroidal":
        errors.append("FCI map quality validation requires topology='toroidal'")

    x_faces = np.asarray(grid.x.faces, dtype=float)
    y_faces = np.asarray(grid.y.faces, dtype=float)
    z_centers = np.asarray(grid.z.centers, dtype=float)
    z_faces = np.asarray(grid.z.faces, dtype=float)
    nx, ny, nz = shape
    z_period = float(z_faces[-1] - z_faces[0])
    period_ok = np.isfinite(z_period) and z_period > 0.0
    checks["positive_eta_period"] = bool(period_ok)
    if not period_ok:
        errors.append("eta faces must define a finite positive periodic period")

    def periodic_difference(values, expected):
        if not period_ok:
            return np.full_like(np.asarray(values, dtype=float), np.inf)
        raw = np.asarray(values, dtype=float) - np.asarray(expected, dtype=float)
        return np.abs(
            np.mod(raw + 0.5 * z_period, z_period) - 0.5 * z_period
        )

    coordinate_ok = True
    for name in (
        "forward_endpoint_x",
        "forward_endpoint_y",
        "forward_endpoint_z",
        "backward_endpoint_x",
        "backward_endpoint_y",
        "backward_endpoint_z",
    ):
        coordinate_ok &= bool(np.all(np.isfinite(arrays[name])))
    coordinate_ok &= bool(
        np.all((arrays["forward_endpoint_x"] >= x_faces[0] - atol)
               & (arrays["forward_endpoint_x"] <= x_faces[-1] + atol))
    )
    coordinate_ok &= bool(
        np.all((arrays["backward_endpoint_x"] >= x_faces[0] - atol)
               & (arrays["backward_endpoint_x"] <= x_faces[-1] + atol))
    )
    coordinate_ok &= bool(
        np.all((arrays["forward_endpoint_y"] >= y_faces[0] - atol)
               & (arrays["forward_endpoint_y"] <= y_faces[-1] + atol))
    )
    coordinate_ok &= bool(
        np.all((arrays["backward_endpoint_y"] >= y_faces[0] - atol)
               & (arrays["backward_endpoint_y"] <= y_faces[-1] + atol))
    )
    checks["endpoint_coordinates"] = bool(coordinate_ok)
    if not coordinate_ok:
        errors.append("endpoint coordinates leave the toroidal logical domain")

    fraction_ok = True
    # Axis-regular interpolation may use one ghost layer on the radial
    # coordinate: index -0.5 is the lower-axis ghost-center limit and
    # nx-0.5 is the outer-wall ghost-center limit.  Theta is periodic and is
    # represented on the half-open index interval [0, ny), up to tolerance at
    # the seam.
    for name in ("forward_x", "backward_x"):
        fraction_ok &= bool(
            np.all((arrays[name] >= -0.5 - atol)
                   & (arrays[name] <= float(nx) - 0.5 + atol))
        )
    for name in ("forward_y", "backward_y"):
        fraction_ok &= bool(
            np.all((arrays[name] >= -atol)
                   & (arrays[name] < float(ny) + atol))
        )
    checks["cell_centered_fractional_coordinates"] = bool(fraction_ok)
    if not fraction_ok:
        errors.append(
            "fractional interpolation coordinates are outside axis-regular "
            "cell-centered bounds"
        )

    forward_boundary = arrays["forward_boundary"].astype(bool, copy=False)
    backward_boundary = arrays["backward_boundary"].astype(bool, copy=False)
    forward_nonboundary = ~forward_boundary
    backward_nonboundary = ~backward_boundary
    expected_forward_z = z_centers[np.mod(np.arange(nz) + 1, nz)]
    expected_backward_z = z_centers[np.mod(np.arange(nz) - 1, nz)]
    expected_forward_z_grid = np.broadcast_to(
        expected_forward_z.reshape(1, 1, nz), shape
    )
    expected_backward_z_grid = np.broadcast_to(
        expected_backward_z.reshape(1, 1, nz), shape
    )
    forward_z_error = periodic_difference(
        arrays["forward_endpoint_z"], expected_forward_z_grid
    )
    backward_z_error = periodic_difference(
        arrays["backward_endpoint_z"], expected_backward_z_grid
    )
    forward_tolerance = atol + rtol * np.maximum(1.0, np.abs(expected_forward_z_grid))
    backward_tolerance = atol + rtol * np.maximum(1.0, np.abs(expected_backward_z_grid))
    forward_z_ok = np.all((~forward_nonboundary) | (forward_z_error <= forward_tolerance))
    backward_z_ok = np.all((~backward_nonboundary) | (backward_z_error <= backward_tolerance))
    adjacent_plane_ok = bool(forward_z_ok and backward_z_ok)
    checks["nonboundary_endpoints_on_adjacent_eta_plane"] = adjacent_plane_ok
    if not adjacent_plane_ok:
        errors.append(
            "a nonboundary endpoint does not land on the adjacent periodic eta plane"
        )

    # Every physical boundary must be the outer u wall.  In particular, a
    # lower-axis hit is a topology error even if all numerical fields are finite.
    outer_u = x_faces[-1]
    forward_outer_ok = np.all(
        ~forward_boundary
        | (np.abs(arrays["forward_endpoint_x"] - outer_u) <= atol + rtol * max(1.0, abs(outer_u)))
    )
    backward_outer_ok = np.all(
        ~backward_boundary
        | (np.abs(arrays["backward_endpoint_x"] - outer_u) <= atol + rtol * max(1.0, abs(outer_u)))
    )
    boundary_topology_ok = bool(forward_outer_ok and backward_outer_ok)
    checks["only_outer_u_is_physical_boundary"] = boundary_topology_ok
    if not boundary_topology_ok:
        errors.append(
            "toroidal maps classify a lower-axis or non-outer-u endpoint as physical boundary"
        )

    seam_ok = bool(
        np.all(
            forward_boundary[..., -1]
            | (
                periodic_difference(
                    arrays["forward_endpoint_z"][..., -1], z_centers[0]
                )
                <= atol + rtol * max(1.0, abs(float(z_centers[0])))
            )
        )
        and np.all(
            backward_boundary[..., 0]
            | (
                periodic_difference(
                    arrays["backward_endpoint_z"][..., 0], z_centers[-1]
                )
                <= atol + rtol * max(1.0, abs(float(z_centers[-1])))
            )
        )
    )
    checks["periodic_eta_seam"] = seam_ok
    if not seam_ok:
        errors.append("periodic eta seam is not mapped to the opposite endpoint plane")

    return {
        "valid": not errors,
        "topology": selected_topology,
        "shape": shape,
        "checks": checks,
        "counts": {
            "forward_boundary": int(np.count_nonzero(forward_boundary)),
            "backward_boundary": int(np.count_nonzero(backward_boundary)),
            "total_cells": int(np.prod(shape)),
        },
        "errors": tuple(errors),
    }


def validate_hsx_fci_maps(
    maps: FciMaps3D | Mapping[str, np.ndarray],
    grid: CellCenteredGrid3D,
    *,
    topology: str = "toroidal",
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> dict[str, object]:
    """Raise on any strict full-torus FCI map quality failure."""

    report = fci_map_quality_report(
        maps,
        grid,
        topology=topology,
        atol=atol,
        rtol=rtol,
    )
    if not bool(report["valid"]):
        errors = "; ".join(str(error) for error in report["errors"])
        raise ValueError(f"invalid HSX toroidal FCI maps: {errors}")
    return report


def _mode_cutoff(value: object) -> int:
    """Normalize scalar/array/tuple mode properties to max(abs(mode))."""

    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError("cached mode set is empty")
    return max(abs(int(mode)) for mode in values)


def _topology_metadata(descriptor: TopologyDescriptor) -> dict[str, object]:
    return {
        "topology": descriptor.name,
        "coordinate_names": list(descriptor.coordinate_names),
        "periodic_axes": list(descriptor.periodic_axes),
        "axis_regular_axes": list(descriptor.axis_regular_axes),
        "logical_extents": [list(extent) for extent in descriptor.logical_extents],
    }


def _padded_clipped_bounds(
    values: np.ndarray,
    domain: tuple[float, float],
    *,
    fraction: float = 0.02,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    domain_values = np.asarray(domain, dtype=float)
    padding = max(
        fraction * max(float(np.ptp(values)), np.finfo(float).eps),
        1.0e-4,
    )
    lower = max(float(domain_values[0]), float(values.min()) - padding)
    upper = min(float(domain_values[1]), float(values.max()) + padding)
    lower = max(
        lower,
        float(np.nextafter(domain_values[0], domain_values[1])),
    )
    upper = min(
        upper,
        float(np.nextafter(domain_values[1], domain_values[0])),
    )
    if not lower < upper:
        raise ValueError("vessel extrema leave no nonempty MAKEGRID fit interval")
    return lower, upper


def _repeat_field_period_cells(
    values: np.ndarray,
    nfp: int,
) -> np.ndarray:
    """Repeat one endpoint-exclusive field period around the full torus."""

    values = np.asarray(values)
    return np.concatenate([values] * int(nfp), axis=2)


def _repeat_field_period_eta_faces(
    values: np.ndarray,
    nfp: int,
) -> np.ndarray:
    """Repeat eta-face data while retaining only the final 2π endpoint."""

    values = np.asarray(values)
    return np.concatenate(
        [values[:, :, :-1]] * int(nfp) + [values[:, :, -1:]],
        axis=2,
    )


def _rotate_field_period_positions(
    positions: np.ndarray,
    nfp: int,
) -> np.ndarray:
    """Rotate one Cartesian field-period embedding through the full torus."""

    positions = np.asarray(positions, dtype=np.float64)
    periods = []
    for period_index in range(int(nfp)):
        angle = period_index * 2.0 * np.pi / int(nfp)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotated = np.empty_like(positions)
        rotated[..., 0] = (
            cosine * positions[..., 0] - sine * positions[..., 1]
        )
        rotated[..., 1] = (
            sine * positions[..., 0] + cosine * positions[..., 1]
        )
        rotated[..., 2] = positions[..., 2]
        periods.append(rotated)
    return np.concatenate(periods, axis=2)


def build_hsx_metric_evaluator(
    *,
    makegrid_path: Path,
    vessel_path: Path,
    resolution: tuple[int, int, int],
    fit_sample_shape: tuple[int, int, int],
    radial_degree: int,
    vertical_degree: int,
    toroidal_modes: int,
    metric_spline_degree: int,
    mmpde_iterations: int,
    axis_core_radius: float,
    makegrid_currents: object | None = None,
    topology: str = "square",
    metric_mesh_shape: tuple[int, int, int] | None = None,
    metric_radial_degree: int = 17,
    metric_poloidal_modes: int = 15,
    metric_toroidal_modes: int = 16,
    eta_projection_iterations: int = 0,
) -> tuple[MetricEvaluator, object, object, int]:
    """Build the exact one-period metric representation used by the solver.

    ``fit_sample_shape`` controls the physical scalar-potential fit samples
    and retains its historical ``(nR, nphi, nZ)`` ordering. For the default
    square topology it also controls the wall-fitted MMPDE node
    representation, with MMPDE ordering ``(u, v, eta) = (nR, nZ, nphi)``.
    For toroidal topology, ``metric_mesh_shape`` independently specifies
    ``(NU, NTHETA, NETA_PER_PERIOD)`` for the axis-regular Fourier--Zernike
    representation. ``resolution`` remains reserved for the final full-torus
    PDE cell grid.
    """

    nu, nv, neta = (int(value) for value in resolution)
    if nu < 3 or nv < 3 or neta < 4:
        raise ValueError("resolution must satisfy NU >= 3, NV >= 3, NETA >= 4")
    makegrid_path = Path(makegrid_path).resolve()
    vessel_path = Path(vessel_path).resolve()
    fit_sample_shape = tuple(int(value) for value in fit_sample_shape)
    if len(fit_sample_shape) != 3 or any(value < 2 for value in fit_sample_shape):
        raise ValueError(
            "fit_sample_shape must contain three integers of at least two"
        )
    nR_fit, nphi_fit, nZ_fit = fit_sample_shape
    selected_topology = str(topology).lower()
    if selected_topology not in {"square", "toroidal"}:
        raise ValueError("topology must be 'square' or 'toroidal'")
    if selected_topology == "toroidal":
        if metric_mesh_shape is None:
            raise ValueError(
                "metric_mesh_shape=(NU, NTHETA, NETA_PER_PERIOD) is required "
                "for topology='toroidal'"
            )
        try:
            raw_metric_mesh_shape = tuple(metric_mesh_shape)
        except TypeError as error:
            raise ValueError(
                "metric_mesh_shape must contain (NU, NTHETA, NETA_PER_PERIOD)"
            ) from error
        if len(raw_metric_mesh_shape) != 3:
            raise ValueError(
                "metric_mesh_shape must contain (NU, NTHETA, NETA_PER_PERIOD)"
            )
        if any(
            isinstance(value, (bool, np.bool_)) or int(value) != value
            for value in raw_metric_mesh_shape
        ):
            raise ValueError("metric_mesh_shape entries must be integers")
        metric_mesh_shape = tuple(int(value) for value in raw_metric_mesh_shape)
        metric_nu, metric_ntheta, metric_neta = metric_mesh_shape
        if metric_nu < 3 or metric_ntheta < 3 or metric_neta < 4:
            raise ValueError(
                "metric_mesh_shape must satisfy NU >= 3, NTHETA >= 3, "
                "NETA_PER_PERIOD >= 4"
            )
        integer_options = (
            ("metric_radial_degree", metric_radial_degree),
            ("metric_poloidal_modes", metric_poloidal_modes),
            ("metric_toroidal_modes", metric_toroidal_modes),
            ("eta_projection_iterations", eta_projection_iterations),
        )
        for name, value in integer_options:
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if int(metric_radial_degree) < 2:
            raise ValueError("metric_radial_degree must be at least 2")
        if int(metric_poloidal_modes) < 1:
            raise ValueError("metric_poloidal_modes must be at least 1")
        if int(metric_poloidal_modes) > int(metric_radial_degree):
            raise ValueError(
                "metric_poloidal_modes cannot exceed metric_radial_degree: "
                f"got {metric_poloidal_modes} > {metric_radial_degree}"
            )
        if int(metric_radial_degree) // 2 + 1 > int(metric_nu):
            raise ValueError(
                "metric_mesh_shape has too few radial nodes for "
                "metric_radial_degree: need NU >= radial_degree//2 + 1, "
                f"got NU={metric_nu}, radial_degree={metric_radial_degree}"
            )
        if int(metric_poloidal_modes) > int(metric_ntheta) // 2:
            raise ValueError(
                "metric_poloidal_modes is not resolvable by metric_mesh_shape: "
                f"need modes <= floor(NTHETA/2), got NTHETA={metric_ntheta} for "
                f"modes={metric_poloidal_modes}"
            )
        if int(metric_toroidal_modes) > int(metric_neta) // 2:
            raise ValueError(
                "metric_toroidal_modes is not resolvable by "
                "metric_mesh_shape: need modes <= floor(NETA_PER_PERIOD/2), "
                f"got NETA_PER_PERIOD={metric_neta} for "
                f"modes={metric_toroidal_modes}"
            )
    else:
        # These options are intentionally ignored on the historical square
        # path so existing callers retain exactly the same construction.
        metric_mesh_shape = None

    stage_start = time.perf_counter()
    print(
        f"[geometry] loading MAKEGRID magnetic field ({makegrid_path})",
        flush=True,
    )
    bfield = bfield_evaluator_from_makegrid(
        makegrid_path,
        currents=makegrid_currents,
        method="cubic",
    )
    print(
        f"[geometry] MAKEGRID loaded in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    stage_start = time.perf_counter()
    print(f"[geometry] loading vessel ({vessel_path})", flush=True)
    wall = WallEvaluator.from_file(vessel_path)
    print(
        f"[geometry] vessel loaded in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    if wall.nfp != bfield.nfp:
        raise ValueError(
            "MAKEGRID and vessel field-period counts disagree: "
            f"{bfield.nfp} != {wall.nfp}"
        )
    nfp = int(wall.nfp)
    if neta % nfp:
        raise ValueError(
            f"full-torus NETA={neta} must be divisible by HSX nfp={nfp}"
        )
    neta_per_period = neta // nfp
    if neta_per_period < 2:
        raise ValueError(
            "full-torus resolution must provide at least two eta cells "
            f"per field period; got NETA={neta}, nfp={nfp}"
        )

    raw_rz = np.asarray(wall.raw["RZ"], dtype=float)
    fit_R_bounds = _padded_clipped_bounds(
        raw_rz[..., 0],
        (float(bfield.R[0]), float(bfield.R[-1])),
    )
    fit_Z_bounds = _padded_clipped_bounds(
        raw_rz[..., 1],
        (float(bfield.Z[0]), float(bfield.Z[-1])),
    )

    def fit_mask(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        axis_R, axis_Z, _, _ = wall.reference_axis(points[:, 1])
        outside_axis_core = (
            np.hypot(points[:, 0] - axis_R, points[:, 2] - axis_Z)
            > float(axis_core_radius)
        )
        return wall.contains_cylindrical(points) & outside_axis_core

    stage_start = time.perf_counter()
    print(
        "[geometry] fitting scalar-potential coordinate "
        f"(samples={fit_sample_shape}, degrees="
        f"({int(radial_degree)}, {int(vertical_degree)}), "
        f"toroidal_modes={int(toroidal_modes)})",
        flush=True,
    )
    eta_evaluator = scalar_potential_evaluator_from_bfield(
        bfield,
        radial_degree=int(radial_degree),
        vertical_degree=int(vertical_degree),
        toroidal_modes=int(toroidal_modes),
        sample_shape=fit_sample_shape,
        R_bounds=fit_R_bounds,
        Z_bounds=fit_Z_bounds,
        mask=fit_mask,
        reference_axis=wall.reference_axis,
    )
    print(
        f"[geometry] scalar-potential fit completed in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    stage_start = time.perf_counter()
    if selected_topology == "square":
        print(
            "[geometry] constructing wall-fitted MetricEvaluator "
            f"(MMPDE nodes=({nR_fit}, {nZ_fit}, {nphi_fit}) from "
            f"fit-sample-shape={fit_sample_shape}, "
            f"full-torus NETA={neta}, "
            f"MMPDE iterations={int(mmpde_iterations)})",
            flush=True,
        )
        metric_evaluator = build_metric_evaluator(
            eta_evaluator,
            wall_evaluator=wall,
            mesh_shape=(nR_fit, nZ_fit, nphi_fit),
            options=MMPDEOptions(
                max_iterations=int(mmpde_iterations),
                progress_interval=(
                    max(1, int(mmpde_iterations) // 10)
                    if int(mmpde_iterations) > 0
                    else 0
                ),
            ),
            metric_spline_degree=int(metric_spline_degree),
        )
    else:
        print(
            "[geometry] constructing toroidal MetricEvaluator "
            f"(mesh-shape={metric_mesh_shape}, "
            f"radial-degree={int(metric_radial_degree)}, "
            f"poloidal-modes={int(metric_poloidal_modes)}, "
            f"toroidal-modes={int(metric_toroidal_modes)}, "
            f"eta-projection-iterations={int(eta_projection_iterations)}, "
            f"NETA_PER_PERIOD={metric_mesh_shape[2]}, nfp={nfp})",
            flush=True,
        )
        metric_evaluator = build_metric_evaluator(
            eta_evaluator,
            topology="toroidal",
            wall_evaluator=wall,
            mesh_shape=metric_mesh_shape,
            radial_degree=int(metric_radial_degree),
            poloidal_modes=int(metric_poloidal_modes),
            toroidal_modes=int(metric_toroidal_modes),
            projection_iterations=int(eta_projection_iterations),
        )
    print(
        f"[geometry] MetricEvaluator constructed in "
        f"{time.perf_counter() - stage_start:.3f} s",
        flush=True,
    )
    return metric_evaluator, eta_evaluator, bfield, nfp


def build_hsx_fci_geometry(
    *,
    makegrid_path: Path,
    vessel_path: Path,
    resolution: tuple[int, int, int],
    fit_sample_shape: tuple[int, int, int],
    radial_degree: int,
    vertical_degree: int,
    toroidal_modes: int,
    metric_spline_degree: int,
    mmpde_iterations: int,
    axis_core_radius: float,
    reference_magnetic_field: float | None,
    makegrid_currents: object | None = None,
    topology: str = "square",
    metric_mesh_shape: tuple[int, int, int] | None = None,
    metric_radial_degree: int = 17,
    metric_poloidal_modes: int = 15,
    metric_toroidal_modes: int = 16,
    eta_projection_iterations: int = 0,
    construct_fci_maps: bool = False,
    fci_trace_substeps: int = 4,
    metric_cache_dir: Path | None = DEFAULT_METRIC_CACHE_DIR,
    rebuild_metric_cache: bool = False,
    metric_context: HSXMetricContext | None = None,
    return_metric_evaluator: bool = False,
) -> tuple[FciGeometry3D, np.ndarray, int, Path | None] | tuple[
    FciGeometry3D, np.ndarray, int, Path | None, MetricEvaluator
]:
    """Build the global HSX geometry and its Cartesian cell embedding."""

    descriptor = topology_descriptor(topology)
    topology = descriptor.name
    if construct_fci_maps and topology != "toroidal":
        raise ValueError(
            "construct_fci_maps=True is currently supported only for "
            "topology='toroidal'"
        )
    if int(fci_trace_substeps) < 1:
        raise ValueError(
            f"fci_trace_substeps must be >= 1, got {fci_trace_substeps}"
        )
    nu, nv, neta = (int(value) for value in resolution)
    if nu < 3 or nv < 3 or neta < 4:
        raise ValueError("resolution must satisfy NU >= 3, NV >= 3, NETA >= 4")
    if topology == "toroidal" and nv % 2:
        raise ValueError("toroidal global NTHETA must be even for axis half-turn parity")
    if metric_context is not None:
        _validate_hsx_metric_context(metric_context, topology=topology)

    makegrid_path = makegrid_path.resolve()
    vessel_path = vessel_path.resolve()
    if makegrid_currents is None:
        makegrid_current_array = None
    else:
        makegrid_current_array = np.asarray(
            makegrid_currents, dtype=np.float64
        ).reshape(-1)
        if (
            makegrid_current_array.size == 0
            or not np.all(np.isfinite(makegrid_current_array))
        ):
            raise ValueError("makegrid_currents must contain finite values")
    fit_sample_shape = tuple(int(value) for value in fit_sample_shape)
    cache_path = None
    cache_spec_json = ""
    cache_payload = None
    metric_evaluator = None
    bfield = None
    reuse_metric_context = metric_context is not None
    if reuse_metric_context:
        # A context carries the fitted representation itself.  Do not read or
        # write a resolution-keyed metric cache in this explicit path: doing
        # so would make the cache metadata claim that the evaluator was fit
        # for the target resolution.  Default callers retain the exact cache
        # behavior below.
        metric_evaluator = metric_context.metric_evaluator
        bfield = metric_context.bfield
        nfp = int(metric_context.nfp)
    if metric_cache_dir is not None and not reuse_metric_context:
        source_paths = (
            DRBX_SRC / "drbx" / "geometry" / "Bfield_evaluator.py",
            DRBX_SRC / "drbx" / "geometry" / "ScalarPotential_evaluator.py",
            DRBX_SRC / "drbx" / "geometry" / "MetricEvaluator.py",
            DRBX_SRC / "drbx" / "geometry" / "WallEvaluator.py",
            DRBX_SRC / "drbx" / "geometry" / "solve_MMPDE.py",
        )
        cache_spec = {
            "format_version": METRIC_CACHE_FORMAT_VERSION,
            "makegrid": {
                "path": str(makegrid_path),
                "size": makegrid_path.stat().st_size,
                "mtime_ns": makegrid_path.stat().st_mtime_ns,
            },
            "makegrid_currents": (
                None
                if makegrid_current_array is None
                else makegrid_current_array.tolist()
            ),
            "vessel": {
                "path": str(vessel_path),
                "size": vessel_path.stat().st_size,
                "mtime_ns": vessel_path.stat().st_mtime_ns,
            },
            "geometry_sources": {
                str(path.relative_to(DRBX_SRC)): {
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in source_paths
            },
            "resolution": [nu, nv, neta],
            "topology": descriptor.name,
            "coordinate_names": list(descriptor.coordinate_names),
            "periodic_axes": list(descriptor.periodic_axes),
            "axis_regular_axes": list(descriptor.axis_regular_axes),
            "logical_extents": [list(extent) for extent in descriptor.logical_extents],
            "toroidal_domain": "full_2pi",
            "fit_sample_shape": list(fit_sample_shape),
            "radial_degree": int(radial_degree),
            "vertical_degree": int(vertical_degree),
            "toroidal_modes": int(toroidal_modes),
            "metric_spline_degree": int(metric_spline_degree),
            "mmpde_iterations": int(mmpde_iterations),
            "metric_mesh_shape": (
                None if metric_mesh_shape is None else list(metric_mesh_shape)
            ),
            "metric_radial_degree": int(metric_radial_degree),
            "metric_poloidal_modes": int(metric_poloidal_modes),
            "metric_toroidal_modes": int(metric_toroidal_modes),
            "eta_projection_iterations": int(eta_projection_iterations),
            "axis_core_radius": float(axis_core_radius),
            "reference_magnetic_field": (
                None
                if reference_magnetic_field is None
                else float(reference_magnetic_field)
            ),
        }
        cache_spec_json = json.dumps(
            cache_spec,
            sort_keys=True,
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(cache_spec_json.encode()).hexdigest()[:24]
        metric_cache_dir = metric_cache_dir.resolve()
        try:
            metric_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = metric_cache_dir / f"hsx_metric_{cache_key}.npz"
        except OSError as error:
            print(f"metric cache disabled: {error}")

    locations = ("cell", "x_face", "y_face", "z_face")
    expected_shapes = {
        "cell": (nu, nv, neta),
        "x_face": (nu + 1, nv, neta),
        "y_face": (nu, nv + 1, neta),
        "z_face": (nu, nv, neta + 1),
    }
    if (
        not reuse_metric_context
        and cache_path is not None
        and cache_path.is_file()
        and not rebuild_metric_cache
    ):
        try:
            cache_load_start = time.perf_counter()
            print(
                f"[metric-cache] loading {cache_path}",
                flush=True,
            )
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_payload = {
                    name: np.array(cached[name], copy=True)
                    for name in cached.files
                }
            if (
                int(cache_payload["format_version"].item())
                != METRIC_CACHE_FORMAT_VERSION
            ):
                raise ValueError("cache format version mismatch")
            if json.loads(str(cache_payload["cache_spec"].item())) != cache_spec:
                raise ValueError("cache specification mismatch")
            cached_metadata = {
                "topology": str(cache_payload["topology"].item()),
                "coordinate_names": tuple(
                    json.loads(str(cache_payload["coordinate_names_json"].item()))
                ),
                "periodic_axes": tuple(
                    bool(value) for value in cache_payload["periodic_axes"]
                ),
                "axis_regular_axes": tuple(
                    bool(value) for value in cache_payload["axis_regular_axes"]
                ),
                "logical_extents": tuple(
                    tuple(float(value) for value in extent)
                    for extent in np.asarray(cache_payload["logical_extents"])
                ),
            }
            if cached_metadata != {
                "topology": descriptor.name,
                "coordinate_names": descriptor.coordinate_names,
                "periodic_axes": descriptor.periodic_axes,
                "axis_regular_axes": descriptor.axis_regular_axes,
                "logical_extents": descriptor.logical_extents,
            }:
                raise ValueError("cached topology descriptor is inconsistent")
            for name, expected_shape in (
                ("u_faces", (nu + 1,)),
                ("v_faces", (nv + 1,)),
                ("eta_faces", (neta + 1,)),
                ("cell_position", (nu, nv, neta, 3)),
            ):
                if cache_payload[name].shape != expected_shape:
                    raise ValueError(
                        f"cached {name} has shape {cache_payload[name].shape}, "
                        f"expected {expected_shape}"
                    )
            cached_nfp = int(cache_payload["nfp"].item())
            if cached_nfp < 1 or neta % cached_nfp:
                raise ValueError("cached nfp is incompatible with full-torus NETA")
            cached_metric_evaluator = MetricEvaluator.from_cache_payload(
                cache_payload,
                prefix="metric_evaluator_",
            )
            if cached_metric_evaluator.nfp != cached_nfp:
                raise ValueError("cached evaluator nfp is inconsistent")
            if cached_metric_evaluator.topology != topology:
                raise ValueError("cached evaluator topology is inconsistent")
            if not np.isclose(cached_metric_evaluator.u[0], 0.0) or not np.isclose(
                cached_metric_evaluator.u[-1], 1.0
            ):
                raise ValueError("cached evaluator u extent is inconsistent")
            expected_v_extent = (0.0, 1.0) if topology == "square" else (0.0, 2.0 * np.pi)
            if topology == "square":
                actual_v_extent = (cached_metric_evaluator.v[0], cached_metric_evaluator.v[-1])
                if not np.allclose(actual_v_extent, expected_v_extent):
                    raise ValueError("cached evaluator v extent is inconsistent")
            else:
                if not np.isclose(cached_metric_evaluator.v[0], 0.0):
                    raise ValueError("cached evaluator theta origin is inconsistent")
                if not np.isclose(
                    cached_metric_evaluator.v.size * np.diff(cached_metric_evaluator.v)[0],
                    2.0 * np.pi,
                ):
                    raise ValueError("cached evaluator theta extent is inconsistent")
                if cached_metric_evaluator.radial_degree != int(metric_radial_degree):
                    raise ValueError("cached evaluator radial degree is inconsistent")
                if _mode_cutoff(cached_metric_evaluator.poloidal_modes) != int(metric_poloidal_modes):
                    raise ValueError("cached evaluator poloidal modes are inconsistent")
                if _mode_cutoff(cached_metric_evaluator.toroidal_modes) != int(metric_toroidal_modes):
                    raise ValueError("cached evaluator toroidal modes are inconsistent")
            # Retain the continuous evaluator for an optional FCI map build.
            # A metric-cache hit should not require rebuilding the expensive
            # Fourier--Zernike representation.
            metric_evaluator = cached_metric_evaluator
            for location in locations:
                expected_shape = expected_shapes[location]
                for field in METRIC_FIELDS:
                    name = f"{location}_metric_{field}"
                    if cache_payload[name].shape != expected_shape:
                        raise ValueError(
                            f"cached {name} has shape "
                            f"{cache_payload[name].shape}, expected "
                            f"{expected_shape}"
                        )
                if cache_payload[f"{location}_Bmag"].shape != expected_shape:
                    raise ValueError(f"cached {location} Bmag shape mismatch")
                if (
                    cache_payload[f"{location}_B_contra"].shape
                    != expected_shape + (3,)
                ):
                    raise ValueError(
                        f"cached {location} B_contra shape mismatch"
                    )
            print(
                f"[metric-cache] loaded in "
                f"{time.perf_counter() - cache_load_start:.3f} s",
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
                f"[metric-cache] ignored ({error}); rebuilding {cache_path}",
                flush=True,
            )
            cache_payload = None

    if cache_payload is not None:
        u_faces = np.asarray(cache_payload["u_faces"], dtype=np.float64)
        v_faces = np.asarray(cache_payload["v_faces"], dtype=np.float64)
        eta_faces = np.asarray(cache_payload["eta_faces"], dtype=np.float64)
        reference_magnetic_field = float(
            cache_payload["reference_magnetic_field"].item()
        )
        nfp = int(cache_payload["nfp"].item())
        if neta % nfp:
            raise ValueError(
                f"full-torus NETA={neta} must be divisible by nfp={nfp}"
            )
        if not np.isclose(
            eta_faces[-1] - eta_faces[0],
            2.0 * np.pi,
            rtol=2.0e-10,
            atol=2.0e-12,
        ):
            raise ValueError("cached eta axis does not span the full 2π torus")
        cell_positions = np.asarray(
            cache_payload["cell_position"],
            dtype=np.float64,
        )
        metric_geometries = tuple(
            MetricGeometry(
                **{
                    field: jnp.asarray(
                        cache_payload[f"{location}_metric_{field}"]
                    )
                    for field in METRIC_FIELDS
                }
            )
            for location in locations
        )
        bfield_geometries = tuple(
            BFieldGeometry(
                B_contra=jnp.asarray(
                    cache_payload[f"{location}_B_contra"]
                ),
                Bmag=jnp.asarray(cache_payload[f"{location}_Bmag"]),
            )
            for location in locations
        )
        # Keep the loaded payload available so a missing/stale map set can be
        # regenerated and atomically added without rebuilding the metric.
    else:
        if not reuse_metric_context and cache_path is not None:
            print(f"[metric-cache] miss: {cache_path}", flush=True)
        if not reuse_metric_context:
            metric_evaluator, eta_evaluator, bfield, nfp = (
                build_hsx_metric_evaluator(
                    makegrid_path=makegrid_path,
                    vessel_path=vessel_path,
                    resolution=(nu, nv, neta),
                    fit_sample_shape=fit_sample_shape,
                    radial_degree=radial_degree,
                    vertical_degree=vertical_degree,
                    toroidal_modes=toroidal_modes,
                    metric_spline_degree=metric_spline_degree,
                    mmpde_iterations=mmpde_iterations,
                    axis_core_radius=axis_core_radius,
                    makegrid_currents=makegrid_current_array,
                    topology=topology,
                    metric_mesh_shape=metric_mesh_shape,
                    metric_radial_degree=metric_radial_degree,
                    metric_poloidal_modes=metric_poloidal_modes,
                    metric_toroidal_modes=metric_toroidal_modes,
                    eta_projection_iterations=eta_projection_iterations,
                )
            )
        nfp = int(nfp)
        if nfp < 1:
            raise ValueError(f"HSX nfp must be a positive integer; got {nfp}")
        if neta % nfp:
            raise ValueError(
                f"full-torus NETA={neta} must be divisible by HSX nfp={nfp}"
            )
        neta_per_period = neta // nfp

        # The evaluator's node grid is controlled by fit_sample_shape. The
        # PDE grid is an independent uniform logical grid controlled only by
        # --resolution, so all center/face arrays below have the requested
        # final computational shape.
        u_faces = np.linspace(0.0, 1.0, nu + 1, dtype=np.float64)
        v_faces = (
            np.linspace(0.0, 2.0 * np.pi, nv + 1, dtype=np.float64)
            if topology == "toroidal"
            else np.linspace(0.0, 1.0, nv + 1, dtype=np.float64)
        )
        eta_faces_period = (
            float(metric_evaluator.eta[0])
            + np.arange(neta_per_period + 1, dtype=np.float64)
            * float(metric_evaluator.period)
            / float(neta_per_period)
        )
        u_centers = 0.5 * (u_faces[:-1] + u_faces[1:])
        v_centers = 0.5 * (v_faces[:-1] + v_faces[1:])
        eta_centers_period = 0.5 * (
            eta_faces_period[:-1] + eta_faces_period[1:]
        )
        cell_points = np.stack(
            np.meshgrid(u_centers, v_centers, eta_centers_period, indexing="ij"),
            axis=-1,
        )
        if topology == "toroidal":
            positive_u_faces = u_faces[1:]
            x_face_points = np.stack(
                np.meshgrid(
                    positive_u_faces, v_centers, eta_centers_period, indexing="ij"
                ),
                axis=-1,
            )
        else:
            x_face_points = np.stack(
                np.meshgrid(u_faces, v_centers, eta_centers_period, indexing="ij"),
                axis=-1,
            )
        y_face_points = np.stack(
            np.meshgrid(u_centers, v_faces, eta_centers_period, indexing="ij"),
            axis=-1,
        )
        z_face_points = np.stack(
            np.meshgrid(u_centers, v_centers, eta_faces_period, indexing="ij"),
            axis=-1,
        )
        metric_results = []
        for location, points in (
            ("cell centers", cell_points),
            ("u faces", x_face_points),
            ("v faces", y_face_points),
            ("eta faces", z_face_points),
        ):
            stage_start = time.perf_counter()
            print(
                f"[geometry] evaluating metric coefficients at {location}",
                flush=True,
            )
            metric_results.append(metric_evaluator.evaluate(points))
            print(
                f"[geometry] metric coefficients at {location} completed in "
                f"{time.perf_counter() - stage_start:.3f} s",
                flush=True,
            )
        cell_metric_eval, *metric_evaluations = metric_results
        cell_positions = _rotate_field_period_positions(
            np.asarray(cell_metric_eval.position, dtype=np.float64),
            nfp,
        )
        eta_faces = (
            float(eta_faces_period[0])
            + np.arange(neta + 1, dtype=np.float64)
            * 2.0
            * np.pi
            / float(neta)
        )
        stage_start = time.perf_counter()
        print(
            "[geometry] evaluating magnetic coefficients at cell centers",
            flush=True,
        )
        cell_b_eval = metric_evaluator.evaluate_magnetic_field(
            cell_points,
            bfield,
        )
        print(
            f"[geometry] magnetic coefficients at cell centers completed in "
            f"{time.perf_counter() - stage_start:.3f} s",
            flush=True,
        )
        b_evaluations = []
        for location, points in (
            ("u faces", x_face_points),
            ("v faces", y_face_points),
            ("eta faces", z_face_points),
        ):
            stage_start = time.perf_counter()
            print(
                f"[geometry] evaluating magnetic coefficients at {location}",
                flush=True,
            )
            b_evaluations.append(
                metric_evaluator.evaluate_magnetic_field(points, bfield)
            )
            print(
                f"[geometry] magnetic coefficients at {location} completed "
                f"in {time.perf_counter() - stage_start:.3f} s",
                flush=True,
            )

        if topology == "toroidal":
            # The u=0 face is a finite-volume collapsed-face representation,
            # not an ordinary pointwise polar metric.  Build it from the first
            # positive-u radial layer using (-u,theta,eta)=(u,theta+pi,eta)
            # and T=diag(-1,1,1).  NTHETA is even, so the half-turn is exact.
            half_turn = nv // 2
            proxy_metric = metric_evaluator.evaluate(
                np.stack(
                    np.meshgrid(
                        [u_centers[0]], v_centers, eta_centers_period, indexing="ij"
                    ),
                    axis=-1,
                )
            )
            proxy_b = metric_evaluator.evaluate_magnetic_field(
                np.stack(
                    np.meshgrid(
                        [u_centers[0]], v_centers, eta_centers_period, indexing="ij"
                    ),
                    axis=-1,
                ),
                bfield,
            )
            def _axis_tensor(values):
                mirrored = np.roll(np.asarray(values), half_turn, axis=1)
                signs = np.asarray((-1.0, 1.0, 1.0))
                transformed = mirrored * signs.reshape(1, 1, 1, 3, 1)
                transformed = transformed * signs.reshape(1, 1, 1, 1, 3)
                return 0.5 * (np.asarray(values) + transformed)

            def _axis_vector(values):
                mirrored = np.roll(np.asarray(values), half_turn, axis=1)
                return 0.5 * (
                    np.asarray(values)
                    + mirrored * np.asarray((-1.0, 1.0, 1.0)).reshape(1, 1, 1, 3)
                )

            axis_metric = {
                "signed_J": np.zeros((1, nv, neta_per_period), dtype=np.float64),
                "g_contra": _axis_tensor(proxy_metric.g_contra),
                "g_cov": _axis_tensor(proxy_metric.g_cov),
            }
            axis_b = {
                "B_contravariant": _axis_vector(proxy_b.B_contravariant),
                "magnitude": np.mean(
                    np.stack(
                        (np.asarray(proxy_b.magnitude),
                         np.roll(np.asarray(proxy_b.magnitude), half_turn, axis=1)),
                        axis=0,
                    ),
                    axis=0,
                ),
            }
            if any(
                not np.all(np.isfinite(values))
                for values in (*axis_metric.values(), *axis_b.values())
            ):
                raise ValueError("collapsed toroidal axis-face representation is not finite")
            metric_evaluations[0] = SimpleNamespace(
                **{
                    key: np.concatenate((axis_metric[key], np.asarray(value)), axis=0)
                    for key, value in {
                    "signed_J": metric_evaluations[0].signed_J,
                    "g_contra": metric_evaluations[0].g_contra,
                    "g_cov": metric_evaluations[0].g_cov,
                    }.items()
                }
            )
            b_evaluations[0] = SimpleNamespace(
                **{
                    key: np.concatenate((axis_b[key], np.asarray(value)), axis=0)
                    for key, value in {
                    "B_contravariant": b_evaluations[0].B_contravariant,
                    "magnitude": b_evaluations[0].magnitude,
                    }.items()
                }
            )

        if reference_magnetic_field is None:
            reference_magnetic_field = float(
                np.median(
                    np.asarray(cell_b_eval.magnitude, dtype=np.float64)
                )
            )
        reference_magnetic_field = float(reference_magnetic_field)
        if (
            not np.isfinite(reference_magnetic_field)
            or reference_magnetic_field <= 0.0
        ):
            raise ValueError(
                "reference magnetic field must be positive and finite"
            )

        metric_geometries = []
        for location, evaluation in zip(
            locations,
            (cell_metric_eval, *metric_evaluations),
        ):
            repeat_periods = (
                _repeat_field_period_eta_faces
                if location == "z_face"
                else _repeat_field_period_cells
            )
            signed_J = repeat_periods(
                np.asarray(evaluation.signed_J, dtype=np.float64),
                nfp,
            )
            g_contra = repeat_periods(
                np.asarray(evaluation.g_contra, dtype=np.float64),
                nfp,
            )
            g_cov = repeat_periods(
                np.asarray(evaluation.g_cov, dtype=np.float64),
                nfp,
            )
            metric_geometries.append(
                MetricGeometry(
                    J=jnp.asarray(signed_J),
                    g11=jnp.asarray(g_contra[..., 0, 0]),
                    g22=jnp.asarray(g_contra[..., 1, 1]),
                    g33=jnp.asarray(g_contra[..., 2, 2]),
                    g12=jnp.asarray(g_contra[..., 0, 1]),
                    g13=jnp.asarray(g_contra[..., 0, 2]),
                    g23=jnp.asarray(g_contra[..., 1, 2]),
                    g_11=jnp.asarray(g_cov[..., 0, 0]),
                    g_22=jnp.asarray(g_cov[..., 1, 1]),
                    g_33=jnp.asarray(g_cov[..., 2, 2]),
                    g_12=jnp.asarray(g_cov[..., 0, 1]),
                    g_13=jnp.asarray(g_cov[..., 0, 2]),
                    g_23=jnp.asarray(g_cov[..., 1, 2]),
                )
            )
        bfield_geometries = []
        for location, evaluation in zip(
            locations,
            (cell_b_eval, *b_evaluations),
        ):
            repeat_periods = (
                _repeat_field_period_eta_faces
                if location == "z_face"
                else _repeat_field_period_cells
            )
            bfield_geometries.append(
                BFieldGeometry(
                    B_contra=(
                        jnp.asarray(
                            repeat_periods(
                                np.asarray(
                                    evaluation.B_contravariant,
                                    dtype=np.float64,
                                ),
                                nfp,
                            )
                        )
                        / reference_magnetic_field
                    ),
                    Bmag=(
                        jnp.asarray(
                            repeat_periods(
                                np.asarray(
                                    evaluation.magnitude,
                                    dtype=np.float64,
                                ),
                                nfp,
                            )
                        )
                        / reference_magnetic_field
                    ),
                )
            )
        bfield_geometries = tuple(bfield_geometries)

        if cache_path is not None:
            cache_write_start = time.perf_counter()
            print(
                f"[metric-cache] serializing evaluated geometry to "
                f"{cache_path}",
                flush=True,
            )
            cache_payload = {
                "format_version": np.asarray(
                    METRIC_CACHE_FORMAT_VERSION,
                    dtype=np.int64,
                ),
                "cache_spec": np.asarray(cache_spec_json),
                "topology": np.asarray(descriptor.name),
                "coordinate_names_json": np.asarray(
                    json.dumps(descriptor.coordinate_names)
                ),
                "periodic_axes": np.asarray(descriptor.periodic_axes, dtype=bool),
                "axis_regular_axes": np.asarray(
                    descriptor.axis_regular_axes, dtype=bool
                ),
                "logical_extents": np.asarray(
                    descriptor.logical_extents, dtype=np.float64
                ),
                "u_faces": u_faces,
                "v_faces": v_faces,
                "eta_faces": eta_faces,
                "reference_magnetic_field": np.asarray(
                    reference_magnetic_field,
                    dtype=np.float64,
                ),
                "nfp": np.asarray(nfp, dtype=np.int64),
                "cell_position": cell_positions,
            }
            cache_payload.update(
                metric_evaluator.to_cache_payload(
                    prefix="metric_evaluator_"
                )
            )
            for location, metric in zip(locations, metric_geometries):
                for field in METRIC_FIELDS:
                    cache_payload[f"{location}_metric_{field}"] = np.asarray(
                        getattr(metric, field),
                        dtype=np.float64,
                    )
            for location, magnetic_field in zip(
                locations,
                bfield_geometries,
            ):
                cache_payload[f"{location}_B_contra"] = np.asarray(
                    magnetic_field.B_contra,
                    dtype=np.float64,
                )
                cache_payload[f"{location}_Bmag"] = np.asarray(
                    magnetic_field.Bmag,
                    dtype=np.float64,
                )
            try:
                _write_npz_atomic(cache_path, cache_payload)
                print(
                    f"[metric-cache] written in "
                    f"{time.perf_counter() - cache_write_start:.3f} s "
                    f"({cache_path.stat().st_size / 2**20:.1f} MiB)",
                    flush=True,
                )
            except OSError as error:
                print(
                    f"[metric-cache] write failed: {error}",
                    flush=True,
                )

    assembly_start = time.perf_counter()
    print("[geometry] assembling FciGeometry3D", flush=True)
    u_centers = 0.5 * (u_faces[:-1] + u_faces[1:])
    v_centers = 0.5 * (v_faces[:-1] + v_faces[1:])
    eta_centers = 0.5 * (eta_faces[:-1] + eta_faces[1:])
    grid = CellCenteredGrid3D(
        x=Grid1D(centers=jnp.asarray(u_centers), faces=jnp.asarray(u_faces)),
        y=Grid1D(centers=jnp.asarray(v_centers), faces=jnp.asarray(v_faces)),
        z=Grid1D(
            centers=jnp.asarray(eta_centers),
            faces=jnp.asarray(eta_faces),
        ),
    )
    shape = grid.shape
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
    )
    cell_metric, x_face_metric, y_face_metric, z_face_metric = metric_geometries
    face_metric = FaceMetricGeometry(
        x=x_face_metric,
        y=y_face_metric,
        z=z_face_metric,
    )
    cell_bfield, x_face_bfield, y_face_bfield, z_face_bfield = (
        bfield_geometries
    )
    face_bfield = FaceBFieldGeometry(
        x=x_face_bfield,
        y=y_face_bfield,
        z=z_face_bfield,
    )
    maps, cache_payload, bfield = _build_or_load_hsx_fci_maps(
        grid=grid,
        topology=topology,
        construct_fci_maps=bool(construct_fci_maps),
        fci_trace_substeps=int(fci_trace_substeps),
        cache_payload=cache_payload,
        cache_path=cache_path,
        metric_evaluator=metric_evaluator,
        bfield=bfield,
        makegrid_path=makegrid_path,
        makegrid_currents=makegrid_current_array,
    )
    if maps is None:
        # The direct local operators do not consume traced FCI maps, and the
        # local lowering installs its own inactive LocalFciMaps3D. Keep an
        # explicitly invalid shape-only payload for the legacy path.
        invalid_map_value = jnp.full(shape, jnp.nan, dtype=jnp.float64)
        boundary_map = jnp.ones(shape, dtype=bool)
        maps = FciMaps3D(
            forward_x=invalid_map_value,
            forward_y=invalid_map_value,
            backward_x=invalid_map_value,
            backward_y=invalid_map_value,
            forward_endpoint_x=invalid_map_value,
            forward_endpoint_y=invalid_map_value,
            forward_endpoint_z=invalid_map_value,
            backward_endpoint_x=invalid_map_value,
            backward_endpoint_y=invalid_map_value,
            backward_endpoint_z=invalid_map_value,
            forward_length=invalid_map_value,
            backward_length=invalid_map_value,
            forward_boundary=boundary_map,
            backward_boundary=boundary_map,
        )
    geometry = FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )
    print(
        f"global full-torus geometry: shape={geometry.shape}, nfp={nfp}, "
        f"eta_extent={float(eta_faces[-1] - eta_faces[0]):.6e}, "
        f"B0={reference_magnetic_field:.6e} T, "
        f"J=[{float(jnp.min(cell_metric.J)):.6e}, "
        f"{float(jnp.max(cell_metric.J)):.6e}], "
        f"|B|=[{float(jnp.min(cell_bfield.Bmag)):.6e}, "
        f"{float(jnp.max(cell_bfield.Bmag)):.6e}], "
        f"FCI map construction={'enabled' if construct_fci_maps else 'disabled'}; "
        f"assembly={time.perf_counter() - assembly_start:.3f} s",
        flush=True,
    )
    usable_cache_path = (
        cache_path
        if cache_path is not None and cache_path.is_file()
        else None
    )
    if return_metric_evaluator:
        if metric_evaluator is None:
            raise RuntimeError(
                "return_metric_evaluator=True requires a continuous evaluator"
            )
        return geometry, cell_positions, nfp, usable_cache_path, metric_evaluator
    return geometry, cell_positions, nfp, usable_cache_path


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
        # The explicit mask keeps source slots exactly zero in the runtime state.
        # Vi/Ve begin as centered physical data.  In the staggered
        # layout they must remain fine until the local FCI c2f/Re
        # initialization pass below; applying PcRc here would erase
        # their source-edge support before that projection.
        if (
            os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered"
            and name in ("Vi", "Ve")
        ):
            result[name] = raw
        else:
            result[name] = np.where(owner_mask, averaged, 0.0)
    return FciDrbEBState(**result)


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

def _materialize_face_owner_array(array: np.ndarray, face_topology_host) -> np.ndarray:
    """Expand a leading-term-axis outgoing-face owner diagnostic array."""

    value = np.asarray(array, dtype=np.float64)
    if face_topology_host is None or value.ndim != 4:
        return value
    result = value[(slice(None),) + (
        np.asarray(face_topology_host["edge_owner_i"], dtype=np.int32),
        np.asarray(face_topology_host["edge_owner_j"], dtype=np.int32),
        np.asarray(face_topology_host["edge_owner_k"], dtype=np.int32),
    )]
    return np.where(np.asarray(face_topology_host["edge_active"], dtype=bool)[None], result, 0.0)


def _assert_owner_sparse(state: FciDrbEBState, host_geometry, outgoing_face_topology_host=None) -> None:
    cell_mask = ~np.asarray(host_geometry.topology.is_active_owner, dtype=bool)
    face_mask = (cell_mask if outgoing_face_topology_host is None else
                 ~np.asarray(outgoing_face_topology_host["is_active_owner"], dtype=bool))
    maximum = max(
        float(np.max(np.abs(np.asarray(value)[mask])) if np.any(mask) else 0.0)
        for name, value in state.field_items()
        for mask in (face_mask if name in ("Vi", "Ve") else cell_mask,)
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
    *,
    parallel_velocity_wall_bc: str = "neumann",
) -> LocalFciDrbEBFaceBCBundle:
    """Simple static vessel BCs on the four wall-fitted chart sides."""

    if parallel_velocity_wall_bc not in (
        "dirichlet-zero",
        "neumann",
        "bohm",
    ):
        raise ValueError(
            "parallel_velocity_wall_bc must be 'dirichlet-zero', "
            f"'neumann', or 'bohm', got {parallel_velocity_wall_bc!r}"
        )

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
    velocity_bc = dirichlet
    if parallel_velocity_wall_bc == "neumann":
        velocity_bc = neumann
    elif parallel_velocity_wall_bc == "bohm":
        tau = jnp.asarray(parameters.tau, dtype=jnp.float64)

        def bohm_velocity(axis: int, side: str) -> jax.Array:
            owner_index = 0 if side == "lower" else -1
            Te_owner = jnp.take(state.Te, owner_index, axis=axis)
            Ti_owner = jnp.take(state.Ti, owner_index, axis=axis)
            sound_speed = jnp.sqrt(
                jnp.maximum(Te_owner + tau * Ti_owner, 1.0e-12)
            )
            face_bfield = geometry.face_bfield.axes[axis]
            face_index = 0 if side == "lower" else -1
            B_normal = jnp.take(
                face_bfield.B_contra_owned[..., axis],
                face_index,
                axis=axis,
            )
            outward_B_normal = (-1.0 if side == "lower" else 1.0) * B_normal
            return jnp.where(
                outward_B_normal > 0.0,
                sound_speed,
                jnp.where(outward_B_normal < 0.0, -sound_speed, 0.0),
            )

        velocity_bc = replace(
            dirichlet,
            value_x=(
                dirichlet.value_x.at[0].set(bohm_velocity(0, "lower"))
                .at[-1]
                .set(bohm_velocity(0, "upper"))
            ),
            value_y=(
                dirichlet.value_y.at[:, 0, :].set(bohm_velocity(1, "lower"))
                .at[:, -1, :]
                .set(bohm_velocity(1, "upper"))
            ),
        )
    return LocalFciDrbEBFaceBCBundle(
        density=neumann,
        phi=dirichlet,
        Te=neumann,
        Ti=neumann,
        Vi=velocity_bc,
        Ve=velocity_bc,
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
    gmres_preconditioner: str = "none",
    gmres_residual_correction_steps: int = 0,
    curvature_scheme: str = "direct",
    curvature_scale: float = 1.0,
    curvature_rlp_face_scheme: str = "projected-fine",
    curvature_rlp_fine_glue_penalty: float = 1.0,
    curvature_rlp_fine_glue_transition_face: int | None = None,
    curvature_equations: tuple[str, ...] = (
        "density",
        "Te",
        "Ti",
        "vorticity",
    ),
    ion_temperature_curvature_self_form: str = "product",
    neumann_ghost_scheme: str = "physical",
    parallel_velocity_wall_bc: str = "neumann",
    parallel_inflow_closure: str = "central",
    vorticity_current_inflow_trace: str = "operator",
    parallel_operator_scheme: str = "coordinate",
    fci_parallel_leg_scheme: str = "centered",
    parallel_subsystem_only: bool = False,
    curvature_inflow_closure: str = "central",
    poisson_bracket_scheme: str = "direct",
    curvature_split_scheme: str | None = None,
    parallel_material_scheme: str | None = None,
    upwind_equilibrium_wall_projectors: UpwindEquilibriumWallProjectors | None = None,
    control_volume_geometry=None,
    control_volume_boundary_bc=None,
    outgoing_face_topology=None,
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
    if fci_parallel_leg_scheme not in (
        "centered",
        "boundary-characteristic-upwind",
    ):
        raise ValueError(
            "fci_parallel_leg_scheme must be 'centered' or "
            "'boundary-characteristic-upwind'"
        )
    if curvature_scheme not in ("direct", "conservative", "disabled"):
        raise ValueError(
            "curvature_scheme must be 'direct', 'conservative', or "
            "'disabled', got "
            f"{curvature_scheme!r}"
        )
    if curvature_scale < 0.0:
        raise ValueError("curvature_scale must be nonnegative")
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
    if parallel_inflow_closure not in (
        "central",
        "local-characteristic",
        "equilibrium-characteristic",
    ):
        raise ValueError(
            "parallel_inflow_closure must be 'central' or "
            "'local-characteristic' or 'equilibrium-characteristic', got "
            f"{parallel_inflow_closure!r}"
        )
    if vorticity_current_inflow_trace not in (
        "operator",
        "parallel-characteristic",
    ):
        raise ValueError(
            "vorticity_current_inflow_trace must be 'operator' or "
            "'parallel-characteristic', got "
            f"{vorticity_current_inflow_trace!r}"
        )
    if curvature_inflow_closure not in ("central", "upwind-equilibrium"):
        raise ValueError(
            "curvature_inflow_closure must be 'central' or "
            "'upwind-equilibrium', "
            f"got {curvature_inflow_closure!r}"
        )
    if poisson_bracket_scheme not in ("direct", "compatible-flux"):
        raise ValueError(
            "poisson_bracket_scheme must be 'direct' or 'compatible-flux', "
            f"got {poisson_bracket_scheme!r}"
        )
    if ion_temperature_curvature_self_form not in ("product", "flux"):
        raise ValueError(
            "ion_temperature_curvature_self_form must be 'product' or 'flux', "
            f"got {ion_temperature_curvature_self_form!r}"
        )
    if (
        ion_temperature_curvature_self_form == "flux"
        and curvature_scheme != "conservative"
    ):
        raise ValueError(
            "ion_temperature_curvature_self_form='flux' requires "
            "curvature_scheme='conservative'"
        )
    # Resolve production selectors here so callers can pass them explicitly;
    # retain the environment-backed defaults for legacy direct callers.
    if curvature_split_scheme is None:
        curvature_split_scheme = os.environ.get(
            "DRBX_CURVATURE_SPLIT_SCHEME", "legacy"
        )
    if parallel_material_scheme is None:
        parallel_material_scheme = os.environ.get(
            "DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"
        )
    curvature_equations = tuple(curvature_equations)
    valid_curvature_equations = {"density", "Te", "Ti", "vorticity"}
    if len(set(curvature_equations)) != len(curvature_equations):
        raise ValueError(
            "curvature_equations must not contain duplicates, got "
            f"{curvature_equations!r}"
        )
    invalid_curvature_equations = set(curvature_equations).difference(
        valid_curvature_equations
    )
    if invalid_curvature_equations:
        raise ValueError(
            "curvature_equations contains invalid equations: "
            f"{sorted(invalid_curvature_equations)!r}"
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
    curvature_coefficients = (
        build_local_curvature_coefficients(
            geometry,
            domain,
            periodic_axes=domain.periodic_axes,
            axis_regular_axes=domain.axis_regular_axes,
        )
        if curvature_scheme == "direct"
        else None
    )
    curvature_face_coefficients = (
        build_local_curvature_face_coefficients(geometry, domain)
        if curvature_scheme == "conservative"
        else None
    )
    def face_bc_builder(state, local_geometry, local_domain, local_parameters):
        return build_face_bc_bundle(
            state,
            local_geometry,
            local_domain,
            local_parameters,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
        )

    rhs_kwargs = dict(
        geometry=geometry,
        domain=domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        parameters=parameters,
        curvature_coefficients_owned=curvature_coefficients,
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
        fci_parallel_leg_scheme=str(fci_parallel_leg_scheme),
        face_bc_builder=face_bc_builder,
        diffusion_only=False,
        axis_regular_axes=domain.axis_regular_axes,
        curvature_face_coefficients=curvature_face_coefficients,
        upwind_equilibrium_wall_projectors=upwind_equilibrium_wall_projectors,
        curvature_scheme=curvature_scheme,
        curvature_scale=float(curvature_scale),
        curvature_split_scheme=str(curvature_split_scheme),
        curvature_rlp_face_scheme=curvature_rlp_face_scheme,
        curvature_rlp_fine_glue_penalty=float(
            curvature_rlp_fine_glue_penalty
        ),
        curvature_rlp_fine_glue_transition_face=(
            curvature_rlp_fine_glue_transition_face
        ),
        curvature_inflow_closure=curvature_inflow_closure,
        poisson_bracket_scheme=poisson_bracket_scheme,
        curvature_equations=curvature_equations,
        ion_temperature_curvature_self_form=ion_temperature_curvature_self_form,
        control_volume_geometry=control_volume_geometry,
        control_volume_boundary_bc=control_volume_boundary_bc,
        outgoing_face_topology=outgoing_face_topology,
    )
    model = LocalFciDrbEBRhs(
        **rhs_kwargs,
        parallel_inflow_closure=parallel_inflow_closure,
        vorticity_current_inflow_trace=vorticity_current_inflow_trace,
        parallel_subsystem_only=parallel_subsystem_only,
    )
    return model


class _JittedPhaseTimer:
    """Collect ordered host timestamps emitted by one compiled RK4 advance."""

    _EXPECTED_MARKERS = 8

    def __init__(self) -> None:
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
            if self._marker_count != self._EXPECTED_MARKERS:
                raise RuntimeError(
                    "compiled RK4 timing markers were incomplete: "
                    f"expected {self._EXPECTED_MARKERS}, got {self._marker_count}"
                )
            return self._operator_seconds, self._gmres_seconds


def _state_marker_dependencies(state: FciDrbEBState) -> tuple[jax.Array, ...]:
    """Return scalar dependencies that make a timing marker await every field."""

    return tuple(jnp.ravel(value)[0] for _, value in state.field_items())


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


_PHI_DIAGNOSTIC_RHS_INCOMPATIBLE_REL = 4
_PHI_DIAGNOSTIC_FINAL_INCOMPATIBLE_REL = 5
_PHI_DIAGNOSTIC_FINAL_FULL_REL = 6
_PHI_DIAGNOSTIC_WIDTH = 7
_PHI_DIAGNOSTIC_NORM_FLOOR = 1.0e-30


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
    return jnp.concatenate((first_four, jnp.zeros(3, dtype=jnp.float64)))


def _print_rk_stage_diagnostics(
    field_names: Sequence[str],
    rk_stage_diagnostics: np.ndarray,
) -> None:
    """Print complete RK-stage state and RHS diagnostics for a failure path."""

    for rk_name, rk_values in zip(
        ("current/k1", "stage2/k2", "stage3/k3", "stage4/k4", "next/weighted"),
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


def _format_snapshot_time(value: float) -> str:
    return f"{value:.12e}".replace("+", "p").replace("-", "m").replace(".", "d")


def _curvature_transition_audit_baseline(
    candidate: LocalFciDrbEBRhs,
) -> LocalFciDrbEBRhs:
    """Return the projected-fine comparator for a selected-face audit."""

    return replace(
        candidate,
        curvature_rlp_face_scheme="projected-fine",
        curvature_rlp_fine_glue_transition_face=None,
    )


def run_full_eb(
    initial_state: FciDrbEBState,
    *,
    global_geometry: FciGeometry3D,
    cell_positions: np.ndarray,
    nfp: int,
    sharded_geometry: ShardedFciGeometry3D,
    mesh: Mesh,
    parameters: FciDrbEBRhsParameters,
    metric_cache_path: Path | None,
    gmres_target_tolerance: float,
    gmres_acceptance_tolerance: float,
    gmres_max_iterations: int,
    gmres_restart: int = 100,
    gmres_preconditioner: str,
    gmres_residual_correction_steps: int = 0,
    time_integrator: str,
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
    curvature_manufactured_output: Path | None = None,
    curvature_transition_audit_output: Path | None = None,
    run_metadata: dict[str, object] | None = None,
    reconstruct_initial_phi: bool = True,
    curvature_scheme: str = "direct",
    curvature_scale: float = 1.0,
    curvature_rlp_face_scheme: str = "projected-fine",
    curvature_rlp_fine_glue_penalty: float = 1.0,
    curvature_rlp_fine_glue_transition_face: int | None = None,
    curvature_equations: tuple[str, ...] = (
        "density",
        "Te",
        "Ti",
        "vorticity",
    ),
    ion_temperature_curvature_self_form: str = "product",
    neumann_ghost_scheme: str = "physical",
    parallel_velocity_wall_bc: str = "neumann",
    parallel_inflow_closure: str = "central",
    vorticity_current_inflow_trace: str = "operator",
    parallel_operator_scheme: str = "coordinate",
    fci_parallel_leg_scheme: str = "centered",
    parallel_subsystem_only: bool = False,
    curvature_inflow_closure: str = "central",
    poisson_bracket_scheme: str = "direct",
    curvature_split_scheme: str | None = None,
    parallel_material_scheme: str | None = None,
    track_curvature_chain_rule_defect: bool = False,
    control_volume_descriptor=None,
    control_volume_fields_host=None,
    control_volume_boundary_bc=None,
    control_volume_assembler=None,
    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,
    owner_host_geometry=None,
    outgoing_face_topology_host=None,
) -> FciDrbEBState:
    """Advance the global seven-field EB state with classical RK4."""

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
    if time_integrator != "rk4":
        raise ValueError("time_integrator must be 'rk4'")
    if gmres_restart < 1:
        raise ValueError("gmres_restart must be positive")
    if int(checkpoint_every) < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if parallel_subsystem_only and time_integrator != "rk4":
        raise ValueError(
            "parallel_subsystem_only is currently supported only with "
            "time_integrator='rk4'"
        )
    if parallel_operator_scheme not in ("coordinate", "fci"):
        raise ValueError(
            "parallel_operator_scheme must be 'coordinate' or 'fci', got "
            f"{parallel_operator_scheme!r}"
        )
    if fci_parallel_leg_scheme not in (
        "centered",
        "boundary-characteristic-upwind",
    ):
        raise ValueError(
            "fci_parallel_leg_scheme must be 'centered' or "
            "'boundary-characteristic-upwind'"
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
    spatial_spec = P("x", "y", "z")
    geometry_spec = P("x", "y", "z", None)
    replicated_spec = P()
    state_spec = initial_state.map_fields(lambda _value: spatial_spec)
    state_sharding = NamedSharding(mesh, spatial_spec)
    geometry_sharding = NamedSharding(mesh, geometry_spec)
    state = initial_state.map_fields(
        lambda value: jax.device_put(
            jnp.asarray(value, dtype=jnp.float64),
            state_sharding,
        )
    )

    def materialized_state(current_state: FciDrbEBState) -> FciDrbEBState:
        materialized = (
            current_state if owner_host_geometry is None
            else _materialize_owner_state(current_state, owner_host_geometry)
        )
        if outgoing_face_topology_host is None:
            return materialized
        face_active = jnp.asarray(outgoing_face_topology_host["edge_active"], dtype=bool)
        owner_i = jnp.asarray(outgoing_face_topology_host["edge_owner_i"], dtype=jnp.int32)
        owner_j = jnp.asarray(outgoing_face_topology_host["edge_owner_j"], dtype=jnp.int32)
        owner_k = jnp.asarray(outgoing_face_topology_host["edge_owner_k"], dtype=jnp.int32)
        def materialize_face(values):
            return jnp.where(face_active, values[owner_i, owner_j, owner_k], 0.0)
        return materialized.replace(
            Vi=materialize_face(current_state.Vi),
            Ve=materialize_face(current_state.Ve),
        )

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
    cell_fields = jax.device_put(
        jnp.asarray(sharded_geometry.cell_fields, dtype=jnp.float64),
        geometry_sharding,
    )
    map_fields_host = (
        jnp.asarray(sharded_geometry.map_fields, dtype=jnp.float64)
        if sharded_geometry.map_fields is not None
        else jnp.zeros(
            sharded_geometry.global_shape + (len(FCI_MAP_FIELDS),),
            dtype=jnp.float64,
        )
    )
    map_fields = jax.device_put(map_fields_host, geometry_sharding)
    if control_volume_descriptor is None:
        control_volume_fields_host = jnp.zeros(
            sharded_geometry.global_shape + (int(control_volume_field_count),),
            dtype=jnp.float64,
        )
    elif control_volume_fields_host is None:
        raise ValueError(
            "control_volume_fields_host is required with an RLP descriptor"
        )
    control_volume_fields = jax.device_put(
        jnp.asarray(control_volume_fields_host, dtype=jnp.float64),
        geometry_sharding,
    )

    def diagnostic_state(current_state: FciDrbEBState, local_control_volume_geometry, outgoing_face_topology=None) -> FciDrbEBState:
        if local_control_volume_geometry is None:
            return current_state
        cells = local_control_volume_geometry.cells
        scalar = lambda value: expand_local_control_volume_owner_field(value, cells)
        face = outgoing_face_topology
        if face is None:
            return current_state.replace(
                density=scalar(current_state.density), phi=scalar(current_state.phi),
                Te=scalar(current_state.Te), Ti=scalar(current_state.Ti),
                Vi=scalar(current_state.Vi), Ve=scalar(current_state.Ve),
                vorticity=scalar(current_state.vorticity),
            )
        return current_state.replace(
            density=scalar(current_state.density), phi=scalar(current_state.phi),
            Te=scalar(current_state.Te), Ti=scalar(current_state.Ti),
            Vi=prolong_local_outgoing_fci_face_owner_field(current_state.Vi, face),
            Ve=prolong_local_outgoing_fci_face_owner_field(current_state.Ve, face),
            vorticity=scalar(current_state.vorticity),
        )

    shard_count = int(np.prod(sharded_geometry.shard_counts))
    if phase_timing and shard_count > 1:
        print(
            "[simulation] operator/GMRES host-callback timing disabled for "
            "multi-device shard_map; total step timing remains enabled",
            flush=True,
        )
        phase_timing = False

    def build_local_model(
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        wall_projectors: UpwindEquilibriumWallProjectors | None,
    ) -> LocalFciDrbEBRhs:
        local_geometry = assemble_local_fci_geometry(
            sharded_geometry,
            cell_fields_owned,
            map_fields_owned if parallel_operator_scheme == "fci" else None,
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
        local_outgoing_face_topology = None
        if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered":
            if local_geometry.maps is None:
                raise ValueError("fci-staggered requires local FCI maps")
            if (
                local_control_volume_geometry is None
                or not local_control_volume_geometry.has_angular_agglomeration
            ):
                raise ValueError("fci-staggered requires angular-RLP control-volume geometry")
            local_outgoing_face_topology = (
                build_local_outgoing_fci_face_topology_from_geometry(
                    local_control_volume_geometry.cells, local_geometry.maps
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
            gmres_preconditioner=str(gmres_preconditioner),
            gmres_residual_correction_steps=int(
                gmres_residual_correction_steps
            ),
            curvature_scheme=curvature_scheme,
            curvature_scale=float(curvature_scale),
            curvature_split_scheme=curvature_split_scheme,
            curvature_rlp_face_scheme=curvature_rlp_face_scheme,
            curvature_rlp_fine_glue_penalty=float(
                curvature_rlp_fine_glue_penalty
            ),
            curvature_rlp_fine_glue_transition_face=(
                curvature_rlp_fine_glue_transition_face
            ),
            curvature_equations=curvature_equations,
            ion_temperature_curvature_self_form=ion_temperature_curvature_self_form,
            neumann_ghost_scheme=neumann_ghost_scheme,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            parallel_inflow_closure=parallel_inflow_closure,
            vorticity_current_inflow_trace=vorticity_current_inflow_trace,
            parallel_operator_scheme=parallel_operator_scheme,
            parallel_material_scheme=parallel_material_scheme,
            fci_parallel_leg_scheme=fci_parallel_leg_scheme,
            parallel_subsystem_only=parallel_subsystem_only,
            curvature_inflow_closure=curvature_inflow_closure,
            poisson_bracket_scheme=poisson_bracket_scheme,
            upwind_equilibrium_wall_projectors=wall_projectors,
            control_volume_geometry=local_control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            outgoing_face_topology=local_outgoing_face_topology,
        )

    wall_projectors = None
    wall_projector_specs = None
    if curvature_inflow_closure == "upwind-equilibrium":
        projector_axis_specs = (
            P(None, "y", "z"),
            P("x", None, "z"),
            P("x", "y", None),
        )
        wall_projector_specs = UpwindEquilibriumWallProjectors(
            axes=tuple(
                tuple(
                    projector_axis_specs[axis]
                    for _side in range(2)
                )
                for axis in range(3)
            )
        )

        def precompute_wall_projectors(
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
        ) -> UpwindEquilibriumWallProjectors:
            local_geometry = assemble_local_fci_geometry(
                sharded_geometry,
                cell_fields_owned,
                map_fields_owned if parallel_operator_scheme == "fci" else None,
            )
            local_domain_coefficients = build_local_curvature_face_coefficients(
                local_geometry,
                domain,
            )
            return build_upwind_equilibrium_wall_projectors(
                local_geometry,
                domain,
                local_domain_coefficients,
                parameters.tau,
            )

        print(
            "[simulation] precomputing invariant upwind-equilibrium wall "
            "projectors",
            flush=True,
        )
        projector_start = time.perf_counter()
        precompute_projectors = jax.jit(
            jax.shard_map(
                precompute_wall_projectors,
                mesh=mesh,
                in_specs=(geometry_spec, geometry_spec),
                out_specs=wall_projector_specs,
                check_vma=True,
            )
        )
        wall_projectors = precompute_projectors(cell_fields, map_fields)
        jax.block_until_ready(jax.tree_util.tree_leaves(wall_projectors))
        print(
            "[simulation] invariant wall projectors ready in "
            f"{time.perf_counter() - projector_start:.3f} s",
            flush=True,
        )

    if outgoing_face_topology_host is not None:
        # The host initializer supplies centered physical Vi/Ve.  Do
        # this conversion after local FCI geometry/halos exist so the
        # initial staggered values are c2f followed by R_e, not the
        # generic cell P_cR_c aggregation used by scalar leaves.
        def project_initial_staggered_velocities(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
        ) -> FciDrbEBState:
            model = build_local_model(
                cell_fields_owned, map_fields_owned,
                control_volume_fields_owned, local_wall_projectors,
            )
            face_bc = model._face_bcs(local_state)
            context = model._stencil_builder_context()
            def project(values, bc):
                return model._owner_face_field(
                    model._restrict_fine_face_field(
                        model._center_owned_to_outgoing_face(values, bc, context)
                    )
                )
            return local_state.replace(
                Vi=project(local_state.Vi, face_bc.Vi),
                Ve=project(local_state.Ve, face_bc.Ve),
            )
        project_initial_staggered = jax.jit(jax.shard_map(
            project_initial_staggered_velocities, mesh=mesh,
            in_specs=(state_spec, geometry_spec, geometry_spec, geometry_spec, wall_projector_specs),
            out_specs=state_spec, check_vma=False,
        ))
        state = project_initial_staggered(
            state, cell_fields, map_fields, control_volume_fields, wall_projectors
        )
        jax.block_until_ready(state)
        if owner_host_geometry is not None:
            _assert_owner_sparse(
                state, owner_host_geometry, outgoing_face_topology_host
            )
    phi_start = time.perf_counter()
    print(
        "[simulation] compiling and "
        + ("reconstructing" if reconstruct_initial_phi else "reusing")
        + " initial sharded phi",
        flush=True,
    )

    def reconstruct_initial_phi(
        local_state: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        local_wall_projectors: UpwindEquilibriumWallProjectors | None,
    ) -> tuple[jax.Array, jax.Array]:
        phi, info = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
            local_wall_projectors,
        ).reconstruct_phi(local_state, return_diagnostics=True)
        return phi, info.num_steps

    reconstruct_phi = jax.jit(
        jax.shard_map(
            reconstruct_initial_phi,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
                wall_projector_specs,
            ),
            out_specs=(spatial_spec, replicated_spec),
            check_vma=False,
        )
    )
    if curvature_transition_audit_output is not None:
        if curvature_scheme != "conservative":
            raise ValueError(
                "curvature transition audit requires conservative curvature"
            )
        if owner_host_geometry is None or control_volume_descriptor is None:
            raise ValueError(
                "curvature transition audit requires the toroidal angular RLP"
            )

        def raw_curvature_pair(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            field: jax.Array,
        ) -> jax.Array:
            candidate = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            baseline = _curvature_transition_audit_baseline(candidate)
            scalar_bc = candidate._face_bcs(local_state).density

            def one(model):
                field_halo = model._prepare_scalar_halo(field, scalar_bc)
                stencil = build_local_conservative_stencil_from_field(
                    field_halo,
                    model.geometry,
                    model._stencil_builder_context(),
                )
                components = model._conservative_curvature_components(
                    stencil,
                    scalar_bc,
                    field_halo=field_halo,
                )
                components = jax.vmap(model._restrict_fine_field)(components)
                return jax.vmap(model._owner_result)(components)

            return jnp.stack((one(candidate), one(baseline)), axis=0)

        def raw_curvature_linearized(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            tangent: jax.Array,
        ) -> jax.Array:
            function = lambda value: raw_curvature_pair(
                local_state,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
                value,
            )
            return jax.jvp(function, (local_state.density,), (tangent,))[1]

        raw_specs = (
            state_spec,
            geometry_spec,
            geometry_spec,
            geometry_spec,
            wall_projector_specs,
            spatial_spec,
        )
        raw_output_spec = P(None, None, "x", "y", "z")
        raw_affine = jax.jit(
            jax.shard_map(
                raw_curvature_pair,
                mesh=mesh,
                in_specs=raw_specs,
                out_specs=raw_output_spec,
                check_vma=False,
            )
        )
        raw_linearized = jax.jit(
            jax.shard_map(
                raw_curvature_linearized,
                mesh=mesh,
                in_specs=raw_specs,
                out_specs=raw_output_spec,
                check_vma=False,
            )
        )

        def bound_raw(tangent):
            return raw_linearized(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                tangent,
            )

        raw_zero = jax.device_put(
            jnp.zeros(sharded_geometry.global_shape, dtype=jnp.float64),
            state_sharding,
        )
        raw_transpose = jax.jit(jax.linear_transpose(bound_raw, raw_zero))

        def coupled_curvature_pair(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            primitive_fields: jax.Array,
        ) -> jax.Array:
            varied_state = local_state.replace(
                density=primitive_fields[0],
                Te=primitive_fields[1],
                Ti=primitive_fields[2],
                phi=primitive_fields[3],
            )
            candidate = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            baseline = _curvature_transition_audit_baseline(candidate)

            def one(model):
                _rhs, components = model.evaluate_stage(
                    varied_state,
                    phi_owned=varied_state.phi,
                    return_curvature_component_fields=True,
                )
                return components

            return jnp.stack((one(candidate), one(baseline)), axis=0)

        def coupled_curvature_linearized(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            tangent: jax.Array,
        ) -> jax.Array:
            base = jnp.stack(
                (local_state.density, local_state.Te, local_state.Ti, local_state.phi),
                axis=0,
            )
            function = lambda value: coupled_curvature_pair(
                local_state,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
                value,
            )
            return jax.jvp(function, (base,), (tangent,))[1]

        coupled_input_spec = P(None, "x", "y", "z")
        coupled_output_spec = P(None, None, None, "x", "y", "z")
        coupled_specs = (
            state_spec,
            geometry_spec,
            geometry_spec,
            geometry_spec,
            wall_projector_specs,
            coupled_input_spec,
        )
        coupled_affine = jax.jit(
            jax.shard_map(
                coupled_curvature_pair,
                mesh=mesh,
                in_specs=coupled_specs,
                out_specs=coupled_output_spec,
                check_vma=False,
            )
        )
        coupled_linearized = jax.jit(
            jax.shard_map(
                coupled_curvature_linearized,
                mesh=mesh,
                in_specs=coupled_specs,
                out_specs=coupled_output_spec,
                check_vma=False,
            )
        )
        coupled_base_input = jnp.stack(
            (state.density, state.Te, state.Ti, state.phi), axis=0
        )

        def bound_coupled(tangent):
            return coupled_linearized(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                tangent,
            )

        coupled_zero = jax.device_put(
            jnp.zeros((4,) + sharded_geometry.global_shape, dtype=jnp.float64),
            NamedSharding(mesh, coupled_input_spec),
        )
        coupled_transpose = jax.jit(
            jax.linear_transpose(bound_coupled, coupled_zero)
        )

        # Optional Stage-1 audit reduction.  The ordinary transition audit is
        # rectangular in z=(n,Te,Ti,phi) and qdot=(n,Te,Ti,omega), so it cannot
        # support an energy symmetric-part claim.  This audit-only path maps a
        # physical q tangent through the exact production polarization solve
        # before applying the frozen curvature JVP.  It is compiled and used
        # only by helpers that explicitly advertise the reduced-energy hook.
        def reduced_curvature_linearized(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            tangent_q: jax.Array,
        ) -> jax.Array:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            zero = jnp.zeros_like(local_state.phi)
            polarization_tangent = local_state.replace(
                Ti=tangent_q[2],
                phi=zero,
                vorticity=tangent_q[3],
            )
            phi_tangent = model.reconstruct_phi(polarization_tangent)
            tangent_z = jnp.stack(
                (tangent_q[0], tangent_q[1], tangent_q[2], phi_tangent),
                axis=0,
            )
            return coupled_curvature_linearized(
                local_state,
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
                tangent_z,
            )

        def reduced_energy_metric_gradient(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            tangent_q: jax.Array,
        ) -> jax.Array:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            tau_value = jnp.asarray(model.parameters.tau, dtype=jnp.float64)
            thermal_hessian = jnp.asarray(
                (
                    (1.0 + tau_value, -3.0 * tau_value / 5.0, 3.0 * tau_value / 5.0),
                    (-3.0 * tau_value / 5.0, 3.0 * (3.0 * tau_value + 5.0) / 10.0, 0.0),
                    (3.0 * tau_value / 5.0, 0.0, 3.0 * tau_value / 5.0),
                ),
                dtype=jnp.float64,
            )
            thermal = jnp.einsum("ab,b...->a...", thermal_hessian, tangent_q[:3])
            bmag_owned = jnp.asarray(
                model.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64
            )
            thermal = thermal * bmag_owned[None, ...] ** 2

            # With Ti=0, the exact production reconstruction returns
            # chi=-Aphi^dagger*omega.  Therefore -chi is the omega gradient
            # Aphi^dagger*omega of the quadratic polarization energy.
            zero = jnp.zeros_like(local_state.phi)
            omega_tangent = local_state.replace(
                Ti=zero,
                phi=zero,
                vorticity=tangent_q[3],
            )
            chi = model.reconstruct_phi(omega_tangent)
            return jnp.concatenate((thermal, (-chi)[None, ...]), axis=0)

        reduced_linearized = jax.jit(
            jax.shard_map(
                reduced_curvature_linearized,
                mesh=mesh,
                in_specs=coupled_specs,
                out_specs=coupled_output_spec,
                check_vma=False,
            )
        )
        reduced_metric = jax.jit(
            jax.shard_map(
                reduced_energy_metric_gradient,
                mesh=mesh,
                in_specs=coupled_specs,
                out_specs=coupled_input_spec,
                check_vma=False,
            )
        )

        print(
            "[curvature-transition-audit] compiling raw and coupled Jacobians",
            flush=True,
        )
        audit_start = time.perf_counter()
        raw_base_outputs = raw_affine(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
            state.density,
        )
        coupled_base_outputs = coupled_affine(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
            coupled_base_input,
        )
        raw_warmup = bound_raw(raw_zero)
        coupled_warmup = bound_coupled(coupled_zero)
        jax.block_until_ready(
            (raw_base_outputs, coupled_base_outputs, raw_warmup, coupled_warmup)
        )
        print(
            "[curvature-transition-audit] Jacobians ready; running matrix-free analysis",
            flush=True,
        )

        raw_cotangent_sharding = NamedSharding(mesh, raw_output_spec)
        coupled_cotangent_sharding = NamedSharding(mesh, coupled_output_spec)

        def apply_raw_pair_host(value):
            tangent = jax.device_put(jnp.asarray(value), state_sharding)
            result = bound_raw(tangent)
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        def transpose_raw_pair_host(value):
            cotangent = jax.device_put(
                jnp.asarray(value), raw_cotangent_sharding
            )
            (result,) = raw_transpose(cotangent)
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        def apply_coupled_pair_host(value):
            tangent = jax.device_put(
                jnp.asarray(value), NamedSharding(mesh, coupled_input_spec)
            )
            result = bound_coupled(tangent)
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        def transpose_coupled_pair_host(value):
            cotangent = jax.device_put(
                jnp.asarray(value), coupled_cotangent_sharding
            )
            (result,) = coupled_transpose(cotangent)
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        def apply_reduced_pair_host(value):
            tangent = jax.device_put(
                jnp.asarray(value), NamedSharding(mesh, coupled_input_spec)
            )
            result = reduced_linearized(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                tangent,
            )
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        def apply_reduced_metric_host(value):
            tangent = jax.device_put(
                jnp.asarray(value), NamedSharding(mesh, coupled_input_spec)
            )
            result = reduced_metric(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                tangent,
            )
            jax.block_until_ready(result)
            return np.asarray(result, dtype=np.float64)

        topology = owner_host_geometry.topology
        active_owner = np.asarray(topology.is_active_owner, dtype=bool)
        owner_index = np.asarray(topology.owner_index, dtype=np.int32)
        raw_volume = np.asarray(owner_host_geometry.raw_volume, dtype=np.float64)
        aggregate_volume = np.asarray(
            owner_host_geometry.aggregate_chart_volume, dtype=np.float64
        )
        bmag = np.asarray(global_geometry.cell_bfield.Bmag, dtype=np.float64)
        aggregate_ids = np.asarray(topology.aggregate_id, dtype=np.int64).ravel()
        natural_flat = np.zeros(active_owner.size, dtype=np.float64)
        np.add.at(
            natural_flat,
            aggregate_ids,
            (raw_volume / np.maximum(bmag, np.finfo(float).tiny)).ravel(),
        )
        natural_volume = natural_flat.reshape(active_owner.shape)
        natural_volume[~active_owner] = 0.0

        helper_path = (
            curvature_transition_audit_output.parent
            / "analyze_transition_operators.py"
        )
        if not helper_path.is_file():
            raise FileNotFoundError(
                f"curvature transition audit helper is missing: {helper_path}"
            )
        import importlib.util

        helper_spec = importlib.util.spec_from_file_location(
            "hsx_curvature_transition_audit", helper_path
        )
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError(f"cannot load curvature audit helper {helper_path}")
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        audit_metadata = dict(run_metadata or {})
        audit_metadata.update(
            {
                "diagnostic": "curvature-transition-compatibility-audit",
                "candidate_scheme": curvature_rlp_face_scheme,
                "baseline_scheme": "projected-fine",
                "raw_scalar_linearization_state": "frozen density",
                "coupled_inputs": ["density", "Te", "Ti", "phi"],
                "coupled_outputs": ["density", "Te", "Ti", "vorticity"],
            }
        )
        audit_kwargs = dict(
            output=curvature_transition_audit_output,
            scheme=curvature_rlp_face_scheme,
            apply_raw_pair=apply_raw_pair_host,
            transpose_raw_pair=transpose_raw_pair_host,
            raw_base_outputs=np.asarray(raw_base_outputs, dtype=np.float64),
            apply_coupled_pair=apply_coupled_pair_host,
            transpose_coupled_pair=transpose_coupled_pair_host,
            coupled_base_outputs=np.asarray(coupled_base_outputs, dtype=np.float64),
            coupled_base_inputs=np.asarray(coupled_base_input, dtype=np.float64),
            coupled_transfer_state=np.asarray(
                jnp.stack(
                    (state.density, state.Te, state.Ti, state.vorticity), axis=0
                ),
                dtype=np.float64,
            ),
            active=active_owner,
            owner_index=owner_index,
            raw_volume=raw_volume,
            aggregate_volume=aggregate_volume,
            natural_volume=natural_volume,
            angular_group_profile=np.asarray(
                owner_host_geometry.angular_group_size, dtype=np.int32
            ),
            metadata=audit_metadata,
        )
        if getattr(helper, "SUPPORTS_REDUCED_ENERGY_AUDIT", False):
            print(
                "[curvature-transition-audit] compiling exact polarization-reduced "
                "energy callbacks",
                flush=True,
            )
            reduced_warmup = reduced_linearized(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                coupled_zero,
            )
            metric_warmup = reduced_metric(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                coupled_zero,
            )
            jax.block_until_ready((reduced_warmup, metric_warmup))
            audit_kwargs.update(
                apply_reduced_pair=apply_reduced_pair_host,
                apply_reduced_metric=apply_reduced_metric_host,
            )
        helper.run_transition_audit(**audit_kwargs)
        print(
            f"[curvature-transition-audit] wrote {curvature_transition_audit_output} "
            f"in {time.perf_counter() - audit_start:.3f} s",
            flush=True,
        )
        return state
    if curvature_manufactured_output is not None:
        if curvature_scheme != "conservative":
            raise ValueError(
                "manufactured curvature audit requires conservative curvature"
            )

        u_values = np.asarray(global_geometry.grid.x.centers, dtype=np.float64)
        theta_values = np.asarray(global_geometry.grid.y.centers, dtype=np.float64)
        eta_values = np.asarray(global_geometry.grid.z.centers, dtype=np.float64)
        u_grid, theta_grid, eta_grid = np.meshgrid(
            u_values, theta_values, eta_values, indexing="ij"
        )
        radial = np.cos(np.pi * u_grid)
        radial_gradient = np.stack(
            (-np.pi * np.sin(np.pi * u_grid), np.zeros_like(u_grid), np.zeros_like(u_grid)),
            axis=-1,
        )
        poloidal_envelope = np.sin(0.5 * np.pi * u_grid)
        poloidal = poloidal_envelope * np.cos(theta_grid)
        poloidal_gradient = np.stack(
            (
                0.5 * np.pi * np.cos(0.5 * np.pi * u_grid) * np.cos(theta_grid),
                -poloidal_envelope * np.sin(theta_grid),
                np.zeros_like(u_grid),
            ),
            axis=-1,
        )
        helical_envelope = poloidal_envelope * poloidal_envelope
        helical_phase = 2.0 * theta_grid - float(nfp) * eta_grid
        helical = helical_envelope * np.cos(helical_phase)
        helical_gradient = np.stack(
            (
                np.pi
                * poloidal_envelope
                * np.cos(0.5 * np.pi * u_grid)
                * np.cos(helical_phase),
                -2.0 * helical_envelope * np.sin(helical_phase),
                float(nfp) * helical_envelope * np.sin(helical_phase),
            ),
            axis=-1,
        )
        manufactured_names = ("constant", "radial", "poloidal_m1", "helical_m2_nfp")
        manufactured_fields_host = np.stack(
            (np.ones_like(u_grid), radial, poloidal, helical), axis=0
        )
        manufactured_gradients_host = np.stack(
            (
                np.zeros(u_grid.shape + (3,), dtype=np.float64),
                radial_gradient,
                poloidal_gradient,
                helical_gradient,
            ),
            axis=0,
        )
        manufactured_fields = jax.device_put(
            jnp.asarray(manufactured_fields_host),
            NamedSharding(mesh, P(None, "x", "y", "z")),
        )
        manufactured_gradients = jax.device_put(
            jnp.asarray(manufactured_gradients_host),
            NamedSharding(mesh, P(None, "x", "y", "z", None)),
        )

        def manufactured_curvature_kernel(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
            fields_local: jax.Array,
            gradients_local: jax.Array,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            scalar_bc = model._face_bcs(local_state).density
            direct_coefficients = build_local_curvature_coefficients(
                model.geometry,
                model.domain,
                periodic_axes=model.domain.periodic_axes,
                axis_regular_axes=model.domain.axis_regular_axes,
            )
            context = model._stencil_builder_context()

            def one(field, exact_gradient):
                fine_halo = model._prepare_fine_storage_halo(field, scalar_bc)
                fine_stencil = build_local_conservative_stencil_from_field(
                    fine_halo, model.geometry, context
                )
                # Keep the fine reference on the unmodified production
                # operator.  The moment-shared trace is defined from RLP
                # aggregate averages and is meaningful only for the
                # projected input below.
                fine_components = local_curvature_conservative_components_op(
                    fine_stencil,
                    model.geometry,
                    model.curvature_face_coefficients,
                    domain=model.domain,
                    face_bc=scalar_bc,
                    axis_regular_axes=model.axis_regular_axes,
                )
                projected_input = model._project_fine_center_to_cell_rlp(field)
                projected_halo = model._prepare_fine_storage_halo(
                    projected_input, scalar_bc
                )
                projected_stencil = build_local_conservative_stencil_from_field(
                    projected_halo, model.geometry, context
                )
                projected_components = model._conservative_curvature_components(
                    projected_stencil, scalar_bc, field_halo=projected_halo
                )
                fine_operator = jnp.sum(fine_components, axis=0)
                projected_operator = jnp.sum(projected_components, axis=0)
                exact_reference = jnp.einsum(
                    "...i,...i->...", direct_coefficients, exact_gradient
                )
                restricted_fine = model._restrict_fine_field(fine_operator)
                restricted_projected = model._restrict_fine_field(projected_operator)
                restricted_reference = model._restrict_fine_field(exact_reference)
                if model.control_volume_geometry is not None:
                    cells = model.control_volume_geometry.cells
                    prolong = lambda value: expand_local_control_volume_owner_field(
                        value, cells
                    )
                    restricted_fine = prolong(restricted_fine)
                    restricted_projected = prolong(restricted_projected)
                    restricted_reference = prolong(restricted_reference)
                return (
                    fine_operator,
                    projected_operator,
                    restricted_fine,
                    restricted_projected,
                    exact_reference,
                    restricted_reference,
                    fine_components,
                    projected_components,
                    projected_input,
                )

            return jax.vmap(one)(fields_local, gradients_local)

        output_specs = (
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, "x", "y", "z"),
            P(None, None, "x", "y", "z"),
            P(None, None, "x", "y", "z"),
            P(None, "x", "y", "z"),
        )
        audit = jax.jit(
            jax.shard_map(
                manufactured_curvature_kernel,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    wall_projector_specs,
                    P(None, "x", "y", "z"),
                    P(None, "x", "y", "z", None),
                ),
                out_specs=output_specs,
                check_vma=False,
            )
        )
        print(
            "[curvature-manufactured] compiling fine and R A_f P audit",
            flush=True,
        )
        audit_start = time.perf_counter()
        outputs = audit(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
            manufactured_fields,
            manufactured_gradients,
        )
        jax.block_until_ready(outputs)
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in outputs)
        mass_weights = (
            np.asarray(owner_host_geometry.raw_volume, dtype=np.float64)
            if owner_host_geometry is not None
            else np.asarray(global_geometry.cell_metric.J, dtype=np.float64)
        )
        metadata = dict(run_metadata or {})
        metadata.update(
            {
                "diagnostic": "manufactured-conservative-curvature-audit",
                "manufactured_field_names": list(manufactured_names),
                "curvature_component_direction_names": ["u", "theta", "eta"],
                "comparison": [
                    "A_f f",
                    "A_f^{shared-trace} P R f",
                    "R A_f f",
                    "R A_f^{shared-trace} P R f",
                    "cell-centered K dot analytic logical gradient",
                    "R reference",
                ],
            }
        )
        curvature_manufactured_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            curvature_manufactured_output,
            manufactured_fields=manufactured_fields_host,
            manufactured_gradients=manufactured_gradients_host,
            fine_operator=arrays[0],
            projected_operator=arrays[1],
            restricted_fine_operator=arrays[2],
            restricted_projected_operator=arrays[3],
            exact_gradient_reference=arrays[4],
            restricted_reference=arrays[5],
            fine_directional_components=arrays[6],
            projected_directional_components=arrays[7],
            projected_input=arrays[8],
            mass_weights=mass_weights,
            u=u_values,
            theta=theta_values,
            eta=eta_values,
            manufactured_field_names_json=np.asarray(json.dumps(manufactured_names)),
            curvature_component_direction_names_json=np.asarray(
                json.dumps(("u", "theta", "eta"))
            ),
            run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        print(
            f"[curvature-manufactured] wrote {curvature_manufactured_output} "
            f"in {time.perf_counter() - audit_start:.3f} s",
            flush=True,
        )
        return state
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
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
        ) -> tuple[jax.Array, ...]:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
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

        replay = jax.jit(
            jax.shard_map(
                replay_rhs_terms,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    wall_projector_specs,
                ),
                out_specs=replay_out_specs,
                check_vma=False,
            )
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
        print(
            f"[rhs-replay] compiling on frame {rhs_replay_frames[0]} and "
            f"evaluating {len(rhs_replay_frames)} frozen states",
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
            outputs = replay(
                sharded_state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
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
        initial_phi, initial_phi_iterations = reconstruct_phi(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
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

    rhs_term_history_path = os.environ.get("DRBX_RHS_TERM_HISTORY")
    if rhs_term_history_path is not None:
        if outgoing_face_topology_host is None or owner_host_geometry is None:
            raise ValueError("RHS term history diagnostics require staggered cell and face topology")
        frame_text = os.environ.get("DRBX_RHS_TERM_FRAMES", "")
        frames = tuple(int(value) for value in frame_text.split(",") if value)
        output_text = os.environ.get("DRBX_RHS_TERM_OUTPUT")
        if not frames or output_text is None:
            raise ValueError("RHS term history diagnostic requires frames and output")
        with np.load(rhs_term_history_path, allow_pickle=False) as history:
            frame_count = int(history["Vi"].shape[0])
            history_times = np.asarray(history["times"], dtype=np.float64)
            if any(frame < 0 or frame >= frame_count for frame in frames):
                raise ValueError(f"RHS term frame outside [0, {frame_count})")
            saved_states = []
            saved_materialized_vi = []
            face_owner = tuple(np.asarray(outgoing_face_topology_host[name], dtype=np.int32) for name in ("edge_owner_i", "edge_owner_j", "edge_owner_k"))
            face_active_owner = np.asarray(outgoing_face_topology_host["is_active_owner"], dtype=bool)
            for frame in frames:
                raw = FciDrbEBState(**{name: jnp.asarray(history[name][frame], dtype=jnp.float64) for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")})
                recovered = _aggregate_initial_owner_state(raw, owner_host_geometry)
                recovered = recovered.replace(
                    Vi=jnp.asarray(np.where(face_active_owner, np.asarray(raw.Vi)[face_owner], 0.0)),
                    Ve=jnp.asarray(np.where(face_active_owner, np.asarray(raw.Ve)[face_owner], 0.0)),
                )
                saved_states.append(recovered)
                saved_materialized_vi.append(np.asarray(raw.Vi, dtype=np.float64))
        def materialize_rhs_term_fields(local_state, cell_fields_owned, map_fields_owned, control_volume_fields_owned, local_wall_projectors):
            model = build_local_model(cell_fields_owned, map_fields_owned, control_volume_fields_owned, local_wall_projectors)
            _, terms = model.evaluate_stage(local_state, phi_owned=local_state.phi, return_rhs_term_fields=True)
            cells = model.control_volume_geometry.cells
            materialized = jax.vmap(jax.vmap(lambda value: expand_local_control_volume_owner_field(value, cells)))(terms)
            vi_face_terms = jax.vmap(lambda value: prolong_local_outgoing_fci_face_owner_field(value, model.outgoing_face_topology))(terms[3])
            ve_face_terms = jax.vmap(lambda value: prolong_local_outgoing_fci_face_owner_field(value, model.outgoing_face_topology))(terms[4])
            return materialized.at[3].set(vi_face_terms).at[4].set(ve_face_terms)
        rhs_term_materializer = jax.jit(jax.shard_map(
            materialize_rhs_term_fields, mesh=mesh,
            in_specs=(state_spec, geometry_spec, geometry_spec, geometry_spec, wall_projector_specs),
            out_specs=P(None, None, "x", "y", "z"), check_vma=False,
        ))
        vi_names = RHS_TERM_NAMES[3]
        report_frames = []
        near_start = int(np.ceil(0.75 * (sharded_geometry.global_shape[1] // 2)))
        for frame, recovered, saved_vi in zip(frames, saved_states, saved_materialized_vi, strict=True):
            sharded = recovered.map_fields(lambda value: jax.device_put(jnp.asarray(value, dtype=jnp.float64), state_sharding))
            terms = np.asarray(rhs_term_materializer(sharded, cell_fields, map_fields, control_volume_fields, wall_projectors), dtype=np.float64)
            vi_terms = terms[3, :len(vi_names)]
            spectral = _vi_near_band_report(vi_terms, saved_vi, near_start)
            report_frames.append({
                "frame": int(frame), "time": float(history_times[frame]),
                "Vi_theta_near_band_start": near_start,
                "Vi_term_near_band_energy": {name: value for name, value in zip(vi_names, spectral["term_near_band_energy"], strict=True)},
                "Vi_term_near_band_inner_product_with_saved_Vi": {name: value for name, value in zip(vi_names, spectral["term_near_band_inner_product_with_saved_Vi"], strict=True)},
                "Vi_sum_term_near_band_energy": spectral["sum_term_near_band_energy"],
                "Vi_sum_term_near_band_inner_product_with_saved_Vi": spectral["sum_term_near_band_inner_product_with_saved_Vi"],
                "rfft_normalization": spectral["rfft_normalization"],
            })
        output = Path(output_text)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"history": rhs_term_history_path, "frames": report_frames, "field": "Vi", "term_names": list(vi_names)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[rhs-term-history] wrote {output}", flush=True)
        return state
    phase_timer = _JittedPhaseTimer() if phase_timing else None
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
    ) -> FciDrbEBState:
        with jax.named_scope("operators"):
            rhs = model.evaluate_stage(
                stage_state,
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

    def full_rk4_advance(
        current: FciDrbEBState,
        cell_fields_owned: jax.Array,
        map_fields_owned: jax.Array,
        control_volume_fields_owned: jax.Array,
        local_wall_projectors: UpwindEquilibriumWallProjectors | None,
        current_time: jax.Array,
    ) -> tuple[FciDrbEBState, jax.Array, jax.Array] | tuple[
        FciDrbEBState, jax.Array, jax.Array, jax.Array
    ]:
        del current_time
        model = build_local_model(
            cell_fields_owned,
            map_fields_owned,
            control_volume_fields_owned,
            local_wall_projectors,
        )

        # `current.phi` was reconstructed at the end of the previous advance,
        # so stage one does not need another identical elliptic solve.
        k1 = evaluate_operators(current, current.phi, model)
        stage_2 = current.axpy(k1, scale=0.5 * dt)

        phi_2, gmres_info_2 = reconstruct_stage_phi(stage_2, model)
        k2 = evaluate_operators(stage_2, phi_2, model)
        stage_3 = current.axpy(k2, scale=0.5 * dt)

        phi_3, gmres_info_3 = reconstruct_stage_phi(stage_3, model)
        k3 = evaluate_operators(stage_3, phi_3, model)
        stage_4 = current.axpy(k3, scale=dt)

        phi_4, gmres_info_4 = reconstruct_stage_phi(stage_4, model)
        k4 = evaluate_operators(stage_4, phi_4, model)
        weighted_rhs = k1.axpy(k2, scale=2.0).axpy(
            k3,
            scale=2.0,
        ).axpy(k4, scale=1.0)
        next_state = current.axpy(weighted_rhs, scale=dt / 6.0)
        if (
            os.environ.get("DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit")
            == "local-backward-euler"
        ):
            next_state = model.apply_short_leg_implicit_material_step(
                next_state,
                solve_dt=dt,
                selection_dt=dt,
            )
        next_phi, gmres_info_next = reconstruct_stage_phi(next_state, model)
        next_state = next_state.replace(phi=next_phi)
        gmres_stage_diagnostics = jnp.stack(
            (
                gmres_info_2,
                gmres_info_3,
                gmres_info_4,
                gmres_info_next,
            ),
            axis=0,
        )
        gmres_iterations = jnp.mean(gmres_stage_diagnostics[:, 0])

        rk_states = tuple(
            diagnostic_state(stage, model.control_volume_geometry, model.outgoing_face_topology)
            for stage in (current, stage_2, stage_3, stage_4, next_state)
        )
        rk_rhs_values = (k1, k2, k3, k4, weighted_rhs)
        rk_state_mins = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(jnp.min(value) for _, value in stage.field_items())
                )
                for stage in rk_states
            )
        )
        rk_state_maxs = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(jnp.max(value) for _, value in stage.field_items())
                )
                for stage in rk_states
            )
        )
        rk_state_abs_maxs = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        jnp.max(jnp.abs(value))
                        for _, value in stage.field_items()
                    )
                )
                for stage in rk_states
            )
        )
        rk_rhs_abs_maxs = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        jnp.max(jnp.abs(value))
                        for _, value in rhs.field_items()
                    )
                )
                for rhs in rk_rhs_values
            )
        )
        for mesh_axis_name in ("x", "y", "z"):
            rk_state_mins = jax.lax.pmin(rk_state_mins, mesh_axis_name)
            rk_state_maxs = jax.lax.pmax(rk_state_maxs, mesh_axis_name)
            rk_state_abs_maxs = jax.lax.pmax(
                rk_state_abs_maxs,
                mesh_axis_name,
            )
            rk_rhs_abs_maxs = jax.lax.pmax(
                rk_rhs_abs_maxs,
                mesh_axis_name,
            )
        rk_stage_diagnostics = jnp.stack(
            (
                rk_state_mins,
                rk_state_maxs,
                rk_state_abs_maxs,
                rk_rhs_abs_maxs,
            ),
            axis=-1,
        )

        # Keep this a fixed-shape compiled payload.  Each reduction is local
        # to the shard first and then made global over all three shard_map
        # mesh axes.  The host only materializes the 7x3 result when it needs
        # to print it or validate the positivity invariant.
        field_values = tuple(
            value
            for _, value in diagnostic_state(next_state, model.control_volume_geometry, model.outgoing_face_topology).field_items()
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
        if track_curvature_chain_rule_defect:
            curvature_diagnostics = (
                model.ion_temperature_curvature_chain_rule_diagnostics(next_state)
            )
            for mesh_axis_name in ("x", "y", "z"):
                curvature_diagnostics = jax.lax.pmax(
                    curvature_diagnostics,
                    mesh_axis_name,
                )
            return (
                next_state,
                diagnostics,
                curvature_diagnostics,
                gmres_iterations,
                gmres_stage_diagnostics,
                rk_stage_diagnostics,
            )
        return (
            next_state,
            diagnostics,
            gmres_iterations,
            gmres_stage_diagnostics,
            rk_stage_diagnostics,
        )

    print(
        "[simulation] lowering shard-local geometry and compiling one complete "
        "shard_map RK4 advance (4 operator stages, 4 SOLVAX FGMRES solves)",
        flush=True,
    )
    if phase_timing:
        print(
            "[simulation] phase timing enabled; ordered host markers prevent "
            "persistent disk caching of this outer executable, but the "
            "compiled executable is reused for every step in this run",
            flush=True,
        )
    compile_start = time.perf_counter()
    rk4_out_specs = (
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
    compiled_advance = jax.jit(
        jax.shard_map(
            full_rk4_advance,
            mesh=mesh,
            in_specs=(
                state_spec,
                geometry_spec,
                geometry_spec,
                geometry_spec,
                wall_projector_specs,
                replicated_spec,
            ),
            out_specs=rk4_out_specs,
            check_vma=False,
        )
    ).lower(
        state,
        cell_fields,
        map_fields,
        control_volume_fields,
        wall_projectors,
        jnp.asarray(start_time, dtype=jnp.float64),
    ).compile()
    print(
        f"[simulation] compiled sharded RK4 advance in "
        f"{time.perf_counter() - compile_start:.3f} s",
        flush=True,
    )

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
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
        ) -> jax.Array:
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
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
        rhs_term_inspection = jax.jit(
            jax.shard_map(
                inspect_rhs_terms,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    wall_projector_specs,
                ),
                out_specs=replicated_spec,
                check_vma=False,
            )
        ).lower(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
        ).compile()
        print(
            "[simulation] compiled all-equation RHS term inspection in "
            f"{time.perf_counter() - rhs_compile_start:.3f} s",
            flush=True,
        )

    def inspect_rhs_terms_host(current_state: FciDrbEBState) -> np.ndarray:
        if rhs_term_inspection is None:
            raise RuntimeError("RHS term inspection was not compiled")
        result = rhs_term_inspection(
            current_state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
        )
        jax.block_until_ready(result)
        return np.asarray(result, dtype=np.float64)

    # Periodic checkpoints can be state-only.  Do not force compilation of
    # the comparatively expensive spatial inspection path unless diagnostics
    # or explicitly scheduled diagnostic snapshots already require it.
    inspection_enabled = bool(diagnostic_every > 0 or snapshot_times)
    inspection = None
    if inspection_enabled:
        term_spec = P(None, "x", "y", "z")
        wall_spec = P(None, None, "x", "y", "z")
        owned_shape = tuple(int(value) for value in domain.layout.owned_shape)
        global_shape = tuple(int(value) for value in sharded_geometry.global_shape)
        halo_width = int(domain.layout.halo_width)
        wall_term_count = 4

        def inspect_state(
            local_state: FciDrbEBState,
            cell_fields_owned: jax.Array,
            map_fields_owned: jax.Array,
            control_volume_fields_owned: jax.Array,
            local_wall_projectors: UpwindEquilibriumWallProjectors | None,
        ):
            model = build_local_model(
                cell_fields_owned,
                map_fields_owned,
                control_volume_fields_owned,
                local_wall_projectors,
            )
            polarization_terms = model.polarization_balance_terms(
                local_state,
                phi_owned=local_state.phi,
            )
            polarization_residual = jnp.sum(polarization_terms, axis=0)
            diagnostic_local_state = diagnostic_state(local_state, model.control_volume_geometry, model.outgoing_face_topology)
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
        inspection = jax.jit(
            jax.shard_map(
                inspect_state,
                mesh=mesh,
                in_specs=(
                    state_spec,
                    geometry_spec,
                    geometry_spec,
                    geometry_spec,
                    wall_projector_specs,
                ),
                out_specs=inspection_out_specs,
                check_vma=False,
            )
        ).lower(
            state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
        ).compile()
        print(
            "[simulation] compiled snapshot/grid-scale inspection path "
            f"(terms={'on' if snapshot_term_fields else 'off'})",
            flush=True,
        )

    def inspect_host(current_state: FciDrbEBState):
        if inspection is None:
            return None
        result = inspection(
            current_state,
            cell_fields,
            map_fields,
            control_volume_fields,
            wall_projectors,
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
        "metric_cache_path": np.asarray(
            "" if metric_cache_path is None else str(metric_cache_path)
        ),
        "metric_cache_format_version": np.asarray(
            METRIC_CACHE_FORMAT_VERSION,
            dtype=np.int64,
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
        "fci_parallel_leg_scheme": np.asarray(str(fci_parallel_leg_scheme)),
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
    if outgoing_face_topology_host is not None:
        base_output_payload.update({
            f"face_topology_{name}": np.asarray(value)
            for name, value in outgoing_face_topology_host.items()
        })
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
            "fci_parallel_leg_scheme": str(fci_parallel_leg_scheme),
            "fci_trace_substeps": int(
                (run_metadata or {}).get("fci_trace_substeps", 4)
            ),
            "curvature_scale": float(curvature_scale),
            "parallel_subsystem_only": bool(parallel_subsystem_only),
            "snapshot_term_fields": bool(snapshot_term_fields),
            "checkpoint_every": int(checkpoint_every),
            "track_rhs_terms": bool(track_rhs_terms),
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
            if snapshot_term_fields:
                (
                    diagnostic_values,
                    term_fields,
                    wall_ghost_fields,
                    polarization_residual,
                ) = inspected
                payload["Ve_rhs_terms"] = _materialize_face_owner_array(
                    term_fields, outgoing_face_topology_host
                ).astype(np.float64)
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
        name: [np.asarray(value, dtype=np.float32)]
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
    for step in range(1, int(num_steps) + 1):
        step_start = time.perf_counter()
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
            ) = compiled_advance(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                jnp.asarray(
                    float(start_time) + (step - 1) * float(timestep),
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
            ) = compiled_advance(
                state,
                cell_fields,
                map_fields,
                control_volume_fields,
                wall_projectors,
                jnp.asarray(
                    float(start_time) + (step - 1) * float(timestep),
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
            _assert_owner_sparse(state, owner_host_geometry, outgoing_face_topology_host)
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
                history[name].append(np.asarray(value, dtype=np.float32))

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
                "RK4 stage: "
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
                failure_reason="invalid-rk4-stage",
            )
            raise FloatingPointError(
                f"invalid RK4 stage after step {step}"
            )
        if gmres_failed_host:
            stage_text = ", ".join(
                (
                    f"{name}:iters={int(values[0])},"
                    f"relres={values[1]:.3e},accepted={bool(values[3] > 0.5)}"
                )
                for name, values in zip(
                    ("rk2", "rk3", "rk4", "next"),
                    gmres_stage_diagnostics_host,
                    strict=True,
                )
            )
            print(
                f"[diagnostics] step={step} rejected phi inversion: "
                f"{stage_text}; state={state_diagnostics}",
                flush=True,
            )
            _print_rk_stage_diagnostics(field_names, rk_stage_diagnostics_host)
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
            _print_rk_stage_diagnostics(field_names, rk_stage_diagnostics_host)
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
            _print_rk_stage_diagnostics(field_names, rk_stage_diagnostics_host)
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
        metric_cache_path=np.asarray(
            "" if metric_cache_path is None else str(metric_cache_path)
        ),
        metric_cache_format_version=np.asarray(
            METRIC_CACHE_FORMAT_VERSION,
            dtype=np.int64,
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
        fci_parallel_leg_scheme=np.asarray(
            str((metadata or {}).get("fci_parallel_leg_scheme", "centered"))
        ),
        fci_trace_substeps=np.asarray(
            int((metadata or {}).get("fci_trace_substeps", 4)),
            dtype=np.int64,
        ),
        **{
            name: np.stack(values, axis=0)
            for name, values in history.items()
        },
        curvature_scale=np.asarray(float(curvature_scale), dtype=np.float64),
        run_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **({
            f"face_topology_{name}": np.asarray(value)
            for name, value in outgoing_face_topology_host.items()
        } if outgoing_face_topology_host is not None else {}),
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
        "[simulation] average RK4 GMRES iterations: "
        f"{accumulated_gmres_iterations / num_steps:.2f} "
        "(four solves per timestep)",
        flush=True,
    )
    return materialized_state(state)


def _validate_flux_framework(args: argparse.Namespace) -> None:
    """Validate native production/diagnostic selectors before compilation."""

    framework = str(args.flux_framework)
    if not np.isfinite(args.parallel_short_leg_cfl_limit) or (
        args.parallel_short_leg_cfl_limit <= 0.0
    ):
        raise ValueError("--parallel-short-leg-cfl-limit must be finite and positive")
    if (
        args.parallel_short_leg_treatment == "local-backward-euler"
        and framework != "production-split"
    ):
        raise ValueError(
            "--parallel-short-leg-treatment local-backward-euler requires "
            "--flux-framework production-split"
        )
    if args.curvature_evolution_component != "full" and framework != "production-split":
        raise ValueError(
            "--curvature-evolution-component diagnostic selectors require "
            "--flux-framework production-split"
        )
    if args.curvature_radial_ablation != "none" and framework != "production-split":
        raise ValueError(
            "--curvature-radial-ablation requires "
            "--flux-framework production-split"
        )
    if (
        args.curvature_wall_flux_closure == "bc-characteristic"
        and framework != "production-split"
    ):
        raise ValueError(
            "--curvature-wall-flux-closure bc-characteristic requires "
            "--flux-framework production-split"
        )
    if args.parallel_flux_pairing == "support-core":
        if args.parallel_velocity_layout != "cell-centered":
            raise ValueError("support-core requires cell-centered parallel velocities")
        if args.parallel_operator_scheme != "fci":
            raise ValueError("support-core requires --parallel-operator-scheme fci")
    if args.curvature_component_diagnostic_scheme != "directional" and (
        args.rhs_replay_history is None
    ):
        raise ValueError(
            "non-directional curvature component diagnostics require "
            "--rhs-replay-history"
        )
    if args.curvature_characteristic_axes == "radial-poloidal":
        if args.curvature_scheme != "conservative":
            raise ValueError(
                "radial-poloidal curvature characteristics require conservative curvature"
            )
        if args.curvature_rlp_face_scheme not in (
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            raise ValueError(
                "radial-poloidal curvature characteristics require a "
                "fine-glue-characteristic face scheme"
            )
    if args.curvature_radial_characteristic_scheme == "third-order-upwind":
        if args.curvature_scheme != "conservative":
            raise ValueError("third-order radial curvature requires conservative curvature")
        if args.topology != "toroidal":
            raise ValueError("third-order radial curvature requires toroidal topology")
        if args.curvature_characteristic_axes != "legacy":
            raise ValueError(
                "third-order radial curvature requires --curvature-characteristic-axes legacy"
            )
        if args.curvature_rlp_face_scheme != "projected-fine":
            raise ValueError(
                "third-order radial curvature requires --curvature-rlp-face-scheme projected-fine"
            )
    if (
        args.curvature_poloidal_characteristic_scheme == "third-order-upwind"
        and args.curvature_radial_characteristic_scheme != "third-order-upwind"
    ):
        raise ValueError(
            "third-order poloidal curvature requires the matching radial scheme"
        )
    radial_penalty = float(args.curvature_rlp_fine_glue_penalty)
    poloidal_penalty = (
        radial_penalty
        if args.poloidal_characteristic_penalty is None
        else float(args.poloidal_characteristic_penalty)
    )
    if not np.isfinite(radial_penalty) or radial_penalty < 0.0:
        raise ValueError("curvature characteristic penalty must be finite and nonnegative")
    if not np.isfinite(poloidal_penalty) or poloidal_penalty < 0.0:
        raise ValueError("poloidal characteristic penalty must be finite and nonnegative")
    if args.parallel_velocity_layout == "fci-staggered":
        if args.topology != "toroidal" or args.parallel_operator_scheme != "fci":
            raise ValueError("fci-staggered requires toroidal FCI operators")
        if args.time_integrator != "rk4" or args.fci_parallel_leg_scheme != "centered":
            raise ValueError("fci-staggered requires centered-leg RK4")
        if args.restart_from is not None:
            raise ValueError("fci-staggered restart is not face-basis aware")
    if args.rhs_term_history is not None:
        if args.parallel_velocity_layout != "fci-staggered":
            raise ValueError("--rhs-term-history requires fci-staggered velocities")
        if args.rhs_term_output is None:
            raise ValueError("--rhs-term-history requires --rhs-term-output")
        try:
            frames = tuple(
                int(item) for item in args.rhs_term_frames.split(",") if item
            )
        except ValueError as error:
            raise ValueError("--rhs-term-frames must be comma-separated integers") from error
        if not frames or any(frame < 0 for frame in frames):
            raise ValueError("--rhs-term-frames must contain nonnegative indices")

    if framework == "legacy":
        return
    if framework != "production-split":
        raise ValueError(f"unsupported flux framework {framework!r}")
    if args.time_integrator != "rk4":
        raise ValueError("production-split requires --time-integrator rk4")
    if args.parallel_operator_scheme != "fci":
        raise ValueError("production-split requires --parallel-operator-scheme fci")
    if args.parallel_velocity_layout != "cell-centered":
        raise ValueError("production-split requires cell-centered parallel velocities")
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
    if args.curvature_scheme != "conservative":
        raise ValueError("production-split requires conservative curvature")
    if args.curvature_rlp_face_scheme != "projected-fine":
        raise ValueError("production-split requires projected-fine owner geometry")
    if args.poisson_bracket_scheme != "compatible-flux":
        raise ValueError("production-split requires compatible-flux Poisson brackets")
    if frozenset(args.curvature_equations) != frozenset(
        ("density", "Te", "Ti", "vorticity")
    ):
        raise ValueError("production-split requires all four curvature equations")
    if args.curvature_characteristic_axes != "legacy":
        raise ValueError("production-split forbids legacy characteristic-axis selectors")
    if args.curvature_radial_characteristic_scheme != "legacy":
        raise ValueError("production-split forbids legacy radial characteristic selectors")
    if args.curvature_poloidal_characteristic_scheme != "legacy":
        raise ValueError("production-split forbids legacy poloidal characteristic selectors")
    if radial_penalty != 1.0 or args.poloidal_characteristic_penalty is not None:
        raise ValueError("production-split forbids legacy characteristic penalties")
    if args.fci_parallel_leg_scheme != "centered":
        raise ValueError("production-split forbids boundary-only parallel correction")
    if args.parallel_inflow_closure != "central":
        raise ValueError("production-split requires central parallel inflow")
    if args.vorticity_current_inflow_trace != "operator" and (
        args.rhs_replay_history is None
    ):
        raise ValueError("production-split forbids boundary-only current correction")
    if args.curvature_inflow_closure != "central":
        raise ValueError("production-split requires central curvature inflow")


def _configure_runtime_selectors(args: argparse.Namespace) -> None:
    """Export native CLI selectors consumed by LocalFciDrbEBRhs factories."""

    os.environ["DRBX_FLUX_FRAMEWORK"] = str(args.flux_framework)
    os.environ["DRBX_PARALLEL_VELOCITY_LAYOUT"] = str(args.parallel_velocity_layout)
    os.environ["DRBX_PARALLEL_FLUX_PAIRING"] = str(args.parallel_flux_pairing)
    os.environ["DRBX_PARALLEL_BOUNDARY_PAIRING"] = (
        str(args.parallel_boundary_pairing)
        if args.parallel_flux_pairing == "support-core"
        else "legacy"
    )
    os.environ["DRBX_PARALLEL_SHORT_LEG_TREATMENT"] = str(
        args.parallel_short_leg_treatment
    )
    os.environ["DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT"] = str(
        args.parallel_short_leg_cfl_limit
    )
    os.environ["DRBX_CURVATURE_EVOLUTION_COMPONENT"] = str(
        args.curvature_evolution_component
    )
    os.environ["DRBX_CURVATURE_RADIAL_ABLATION"] = str(
        args.curvature_radial_ablation
    )
    os.environ["DRBX_CURVATURE_CHARACTERISTIC_AXES"] = str(
        args.curvature_characteristic_axes
    )
    os.environ["DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME"] = str(
        args.curvature_radial_characteristic_scheme
    )
    os.environ["DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME"] = str(
        args.curvature_poloidal_characteristic_scheme
    )
    os.environ["DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME"] = str(
        args.curvature_component_diagnostic_scheme
    )
    if args.flux_framework == "production-split":
        os.environ["DRBX_CURVATURE_SPLIT_SCHEME"] = "production-path"
        os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] = "production-path"
        os.environ["DRBX_CURVATURE_WALL_FLUX_CLOSURE"] = {
            "equilibrium-exterior": (
                "equilibrium-exterior-canonical-face-state"
            ),
            "bc-characteristic": (
                "bc-characteristic-operator-trace-canonical-face-state"
            ),
        }[str(args.curvature_wall_flux_closure)]
        os.environ["DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"] = (
            "characteristic-projected-operator-trace-canonical-face-state"
        )
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY", None)
        os.environ.pop("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE", None)
    else:
        for name in (
            "DRBX_CURVATURE_SPLIT_SCHEME",
            "DRBX_PARALLEL_MATERIAL_SCHEME",
            "DRBX_CURVATURE_WALL_FLUX_CLOSURE",
            "DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE",
        ):
            os.environ.pop(name, None)
        penalty = (
            args.curvature_rlp_fine_glue_penalty
            if args.poloidal_characteristic_penalty is None
            else args.poloidal_characteristic_penalty
        )
        os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY"] = str(penalty)
        os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE"] = (
            "simulate_hsx_blob.py:--poloidal-characteristic-penalty"
            if args.poloidal_characteristic_penalty is not None
            else "inherited-from-curvature-rlp-fine-glue-penalty"
        )
    if args.rhs_term_history is None:
        for name in (
            "DRBX_RHS_TERM_HISTORY",
            "DRBX_RHS_TERM_FRAMES",
            "DRBX_RHS_TERM_OUTPUT",
        ):
            os.environ.pop(name, None)
    else:
        frames = tuple(int(item) for item in args.rhs_term_frames.split(",") if item)
        os.environ["DRBX_RHS_TERM_HISTORY"] = str(args.rhs_term_history)
        os.environ["DRBX_RHS_TERM_FRAMES"] = ",".join(str(frame) for frame in frames)
        os.environ["DRBX_RHS_TERM_OUTPUT"] = str(args.rhs_term_output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build wall-fitted HSX geometry and run the full local/sharding-"
            "compatible seven-field electrostatic Boussinesq model."
        )
    )
    parser.add_argument(
        "--resolution",
        nargs=3,
        type=int,
        metavar=("NU", "NSECOND", "NETA"),
        default=(8, 8, 8),
        help=(
            "Global PDE cell counts: (NU, NV, NETA) for square topology or "
            "(NU, NTHETA, NETA) for toroidal topology; NETA spans the full 2π."
        ),
    )
    parser.add_argument(
        "--topology",
        choices=("square", "toroidal"),
        default="square",
        help="Logical mesh topology used for global geometry construction.",
    )
    parser.add_argument(
        "--angular-group-profile",
        default="",
        metavar="Q0,Q1,...",
        help=(
            "Diagnostic toroidal override for the radius-dependent angular "
            "owner-group profile, with one comma-separated group size per "
            "radial ring. The explicit profile must retain all production "
            "nesting and metric-width safety checks."
        ),
    )
    parser.add_argument(
        "--square-agglomeration",
        choices=("none", "corner-edge"),
        default="none",
        help=(
            "Optional square-topology projected-owner agglomeration. "
            "'corner-edge' detects explicit-parallel CFL seeds from the "
            "metric and forms eta-plane-local aggregates."
        ),
    )
    parser.add_argument(
        "--agglomeration-volume-ratio",
        type=float,
        default=1.2,
        help=(
            "Corner-edge aggregate volumes must lie between median/ratio "
            "and ratio*median."
        ),
    )
    parser.add_argument(
        "--agglomeration-rate-threshold",
        type=float,
        default=None,
        help=(
            "Optional explicit cell-centred parallel coordinate-rate threshold. "
            "By default it is derived from the RK4 stability radius, the "
            "equilibrium characteristic speed, and the requested timestep."
        ),
    )
    parser.add_argument(
        "--agglomeration-rk4-safety",
        type=float,
        default=0.85,
        help=(
            "Positive fraction of the RK4 imaginary-axis stability radius "
            "used by the automatic centered parallel-flux corner-edge seed threshold."
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
        "--fci-parallel-leg-scheme",
        choices=("centered", "boundary-characteristic-upwind"),
        default="centered",
        help=(
            "FCI mapped-leg closure. 'boundary-characteristic-upwind' replaces "
            "the centered five-field principal operator only on target rows "
            "whose forward or backward leg terminates at the vessel wall."
        ),
    )
    parser.add_argument(
        "--fci-trace-substeps",
        type=int,
        default=4,
        help="RK4 substeps per toroidal plane used when generating FCI maps.",
    )
    parser.add_argument(
        "--parallel-velocity-layout",
        choices=("cell-centered", "fci-staggered"),
        default="cell-centered",
        help="Storage layout for the parallel ion/electron velocities.",
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
        "--parallel-short-leg-treatment",
        choices=("explicit", "local-backward-euler"),
        default="explicit",
        help=(
            "Treatment of selected short FCI wall legs. local-backward-euler "
            "applies the opt-in material-block split after each RK4 step."
        ),
    )
    parser.add_argument(
        "--parallel-short-leg-cfl-limit",
        type=float,
        default=2.5,
        help="CFL threshold selecting short wall legs for the local implicit split.",
    )
    parser.add_argument(
        "--curvature-evolution-component",
        choices=("full", "centered-only", "dissipation-only"),
        default="full",
        help="Select all or one component of the production curvature flux.",
    )
    parser.add_argument(
        "--curvature-wall-flux-closure",
        choices=("equilibrium-exterior", "bc-characteristic"),
        default="equilibrium-exterior",
        help=(
            "Physical-wall state for the production curvature flux.  "
            "bc-characteristic derives incoming data from the primitive "
            "operator boundary traces and requires no equilibrium reference."
        ),
    )
    parser.add_argument(
        "--curvature-radial-ablation",
        choices=(
            "none",
            "upper-physical-face",
            "rlp-transition-faces",
            "ordinary-interior-faces",
            "last-interior-face",
            "within-cell-path",
        ),
        default="none",
        help=(
            "Analysis-only removal of one radial production-curvature "
            "contribution; the default leaves the production RHS unchanged."
        ),
    )
    parser.add_argument(
        "--curvature-characteristic-axes",
        choices=("legacy", "radial", "radial-poloidal"),
        default="legacy",
        help="Diagnostic legacy curvature characteristic-correction axes.",
    )
    parser.add_argument(
        "--curvature-radial-characteristic-scheme",
        choices=("legacy", "third-order-upwind"),
        default="legacy",
        help="Radial coupled-curvature characteristic scheme.",
    )
    parser.add_argument(
        "--curvature-poloidal-characteristic-scheme",
        choices=("legacy", "third-order-upwind"),
        default="legacy",
        help="Poloidal coupled-curvature characteristic scheme.",
    )
    parser.add_argument(
        "--curvature-component-diagnostic-scheme",
        choices=("directional", "centered-dissipation", "radial-provenance"),
        default="directional",
        help="Lane layout used by frozen curvature RHS diagnostics.",
    )
    parser.add_argument(
        "--poloidal-characteristic-penalty",
        type=float,
        default=None,
        help=(
            "Nonnegative legacy poloidal characteristic multiplier; defaults "
            "to --curvature-rlp-fine-glue-penalty."
        ),
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
        "--curvature-scheme",
        choices=("direct", "conservative", "disabled"),
        default="conservative",
        help=(
            "Regular local curvature discretization. 'conservative' uses "
            "precomputed shared-face B-field flux coefficients; 'direct' "
            "uses the existing cell-centered coefficient stencil; 'disabled' "
            "sets every curvature contribution in the EB RHS to zero."
        ),
    )
    parser.add_argument(
        "--curvature-scale",
        type=float,
        default=1.0,
        help="Nonnegative multiplier applied to every assembled curvature contribution.",
    )
    parser.add_argument(
        "--curvature-rlp-face-scheme",
        choices=(
            "projected-fine",
            "moment-shared",
            "bounded-moment-shared",
            "constrained-flux-shared",
            "fine-glue-sat",
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ),
        default="projected-fine",
        help=(
            "Angular-RLP curvature treatment. 'projected-fine' uses R A_f P; "
            "'moment-shared' replaces radial group-size transitions by one "
            "moment-fitted shared flux per physical fine subface; "
            "'bounded-moment-shared' applies a conservative local-state "
            "bound to that fitted correction; 'constrained-flux-shared' "
            "instead constrains the metric-flux correction to zero coarse-"
            "face integral and non-positive discrete face power; "
            "'fine-glue-sat' uses the existing physical fine subfaces as a "
            "common glue grid with a scalar jump penalty; "
            "'fine-glue-characteristic' applies the coupled H-compatible "
            "curvature |M| jump flux on that same glue grid only at RLP "
            "transitions; 'fine-glue-characteristic-bulk' applies that "
            "same high-order trace-transpose flux at every interior radial "
            "face."
        ),
    )
    parser.add_argument(
        "--curvature-rlp-fine-glue-penalty",
        type=float,
        default=1.0,
        help=(
            "Nonnegative dimensionless jump-penalty multiplier for "
            "either fine-glue curvature face scheme. A value of one uses "
            "0.5*abs(Q^u) times the scalar or characteristic jump on each "
            "active radial subface: RLP transitions for the transition-only "
            "schemes and every interior radial face for the bulk "
            "characteristic scheme."
        ),
    )
    parser.add_argument(
        "--curvature-equations",
        nargs="+",
        choices=("density", "Te", "Ti", "vorticity"),
        default=("density", "Te", "Ti", "vorticity"),
        help=(
            "RHS equations receiving their assembled curvature contribution. "
            "The default enables density, Te, Ti, and vorticity; this setting "
            "has no effect when --curvature-scheme=disabled."
        ),
    )
    parser.add_argument(
        "--ion-temperature-curvature-self-form",
        choices=("product", "flux"),
        default="product",
        help=(
            "Conservative ion-temperature self-curvature form: 'product' "
            "uses Ti*C(Ti) (default), while 'flux' uses C(Ti**2)/2 and "
            "requires --curvature-scheme=conservative."
        ),
    )
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
        "--parallel-velocity-wall-bc",
        choices=("dirichlet-zero", "neumann", "bohm"),
        default="neumann",
        help=(
            "Primitive Vi/Ve condition on physical vessel faces. "
            "'dirichlet-zero' retains the reflecting compatibility mode; "
            "'neumann' extrapolates both parallel velocities; 'bohm' sets "
            "outward Vi=Ve=sign(B.n)*sqrt(Te+tau*Ti), a zero-current "
            "sheath-entry diagnostic without a magnetic-presheath model."
        ),
    )
    parser.add_argument(
        "--parallel-inflow-closure",
        choices=(
            "central",
            "local-characteristic",
            "equilibrium-characteristic",
        ),
        default="central",
        help=(
            "Parallel physical-wall inflow closure. 'local-characteristic' "
            "uses local five-field material characteristics for the "
            "(n, Te, Ti, Vi, Ve) subsystem, excludes phi and vorticity, "
            "and uses --parallel-velocity-wall-bc as candidate incoming "
            "data; 'equilibrium-characteristic' instead sets incoming "
            "perturbations to zero relative to (1,1,1,0,0) while retaining "
            "owner outgoing/stationary components; 'central' retains the "
            "centered closure."
        ),
    )
    parser.add_argument(
        "--vorticity-current-inflow-trace",
        choices=("operator", "parallel-characteristic"),
        default="operator",
        help=(
            "Trace used by the parallel-current divergence in the vorticity "
            "equation. 'operator' preserves the production primitive trace; "
            "'parallel-characteristic' is a closure-consistency ablation that "
            "reuses the first-order characteristic material current trace."
        ),
    )
    parser.add_argument(
        "--parallel-subsystem-only",
        action="store_true",
        help=(
            "RK4 diagnostic mode: advance only the production parallel "
            "subsystem while retaining normal algebraic phi reconstruction; "
            "exclude Poisson brackets, curvature, perpendicular diffusion, "
            "and sources."
        ),
    )
    parser.add_argument(
        "--curvature-inflow-closure",
        choices=("central", "upwind-equilibrium"),
        default="central",
        help=(
            "Wall closure for conservative curvature fluxes. "
            "'upwind-equilibrium' retains owner values in outgoing and "
            "stationary background-linearized characteristics and supplies "
            "the normalized equilibrium state (n, Te, Ti, omega)=(1,1,1,0) "
            "to incoming characteristics. Interior and shard faces remain "
            "centered. 'central' retains the centered compatibility mode."
        ),
    )
    parser.add_argument(
        "--poisson-bracket-scheme",
        choices=("direct", "compatible-flux"),
        default="compatible-flux",
        help=(
            "Poisson-bracket discretization. 'compatible-flux' is the "
            "production antisymmetrized shared-face flux form and includes "
            "the RHS 1/B factor."
        ),
    )
    parser.add_argument("--makegrid", type=Path, default=DEFAULT_MAKEGRID)
    parser.add_argument(
        "--makegrid-currents",
        type=lambda value: tuple(float(part) for part in value.split(",")),
        default=DEFAULT_HSX_QHS_MAKEGRID_CURRENTS,
        help=(
            "comma-separated MAKEGRID group multipliers; defaults to the "
            "1 T HSX QHS setting (10722 for MainCoil1-6, zero for AuxCoil1-6)"
        ),
    )
    parser.add_argument("--vessel", type=Path, default=DEFAULT_VESSEL)
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=DEFAULT_METRIC_CACHE_DIR,
        help="Directory for reusable evaluated HSX metric caches.",
    )
    parser.add_argument(
        "--no-metric-cache",
        action="store_true",
        help="Disable loading and writing evaluated metric caches.",
    )
    parser.add_argument(
        "--rebuild-metric-cache",
        action="store_true",
        help="Ignore a matching cache and replace it after evaluation.",
    )
    parser.add_argument(
        "--fit-sample-shape",
        nargs=3,
        type=int,
        metavar=("NR", "NPHI", "NZ"),
        default=(8, 9, 8),
        help=(
            "Scalar-fit sample counts (R, phi, Z). These also determine the "
            "one-period wall-fitted MMPDE node grid as (NR, NZ, NPHI) in "
            "(u, v, eta) for square topology; toroidal metric fitting uses "
            "--metric-mesh-shape instead. --resolution sets final PDE cells."
        ),
    )
    parser.add_argument("--radial-degree", type=int, default=3)
    parser.add_argument("--vertical-degree", type=int, default=3)
    parser.add_argument(
        "--toroidal-modes", type=int, default=2,
        help="Toroidal Fourier modes for the scalar-potential eta fit.",
    )
    parser.add_argument("--metric-spline-degree", type=int, default=1,
                        help="Square-topology metric spline degree.")
    parser.add_argument("--mmpde-iterations", type=int, default=0,
                        help="Square-topology MMPDE iterations.")
    parser.add_argument(
        "--metric-mesh-shape", nargs=3, type=int,
        metavar=("NU", "NTHETA", "NETA_PER_PERIOD"), default=None,
        help="Toroidal Fourier-Zernike metric nodes; NETA is per field period.",
    )
    parser.add_argument(
        "--metric-radial-degree", type=int, default=17,
        help="Toroidal Fourier-Zernike radial degree.",
    )
    parser.add_argument(
        "--metric-poloidal-modes", type=int, default=15,
        help="Toroidal Fourier-Zernike maximum poloidal mode.",
    )
    parser.add_argument(
        "--metric-toroidal-modes", type=int, default=16,
        help="Toroidal Fourier-Zernike maximum eta mode per field period.",
    )
    parser.add_argument(
        "--eta-projection-iterations", type=int, default=0,
        help="Toroidal metric interior eta projection iterations.",
    )
    parser.add_argument("--axis-core-radius", type=float, default=0.03)
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
        "--rhs-replay-electron-force-wall-audit",
        action="store_true",
        help=(
            "Include exact wall-face electron parallel-force traces, "
            "directional stencil contributions, masks, and leg lengths in "
            "an RHS replay archive."
        ),
    )
    parser.add_argument(
        "--rhs-term-history",
        type=Path,
        default=None,
        help=(
            "Evaluate the staggered Vi RHS decomposition at selected frames "
            "of an existing history."
        ),
    )
    parser.add_argument(
        "--rhs-term-frames",
        default="100,180,225",
        help="Comma-separated frames for --rhs-term-history.",
    )
    parser.add_argument(
        "--rhs-term-output",
        type=Path,
        default=None,
        help="JSON output for --rhs-term-history.",
    )
    parser.add_argument(
        "--curvature-manufactured-output",
        type=Path,
        default=None,
        help=(
            "Apply the conservative curvature operator to smooth analytic "
            "fields and export fine-grid versus R A_f P results without "
            "advancing time."
        ),
    )
    parser.add_argument(
        "--curvature-transition-audit-output",
        type=Path,
        default=None,
        help=(
            "Matrix-free mass-adjoint and frozen coupled-Jacobian audit of "
            "the angular-RLP curvature transitions, without advancing time."
        ),
    )
    parser.add_argument(
        "--curvature-transition-audit-face",
        type=int,
        default=None,
        metavar="I",
        help=(
            "Audit-only one-based radial face index selecting one fine-glue "
            "SAT angular-RLP transition. Requires "
            "--curvature-transition-audit-output and "
            "--curvature-rlp-face-scheme=fine-glue-sat."
        ),
    )
    parser.add_argument(
        "--blob-initialization",
        choices=("field-aligned", "logical"),
        default="field-aligned",
        help=(
            "Use a backtraced, finite field-aligned density filament by "
            "default. 'logical' restores the eta-copied Gaussian."
        ),
    )
    parser.add_argument("--density-amplitude", type=float, default=0.05)
    parser.add_argument(
        "--temperature-amplitude",
        type=float,
        default=0.0,
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
        default=(0.65, 0.50),
    )
    parser.add_argument("--blob-width", type=float, default=0.10)
    parser.add_argument(
        "--blob-reference-eta",
        type=float,
        default=np.pi,
        help="Toroidal reference plane of the field-aligned filament.",
    )
    parser.add_argument(
        "--blob-parallel-half-length",
        type=float,
        default=np.pi,
        help=(
            "Half-length in eta radians of the compact cos^2 filament "
            "envelope. It must be no larger than pi so the perturbation "
            "vanishes at or before the full-torus periodic seam."
        ),
    )
    parser.add_argument(
        "--fieldline-substeps-per-plane",
        type=int,
        default=4,
        help=(
            "RK4 tracing substeps per toroidal plane spacing, used only to "
            "construct the initial field-line labels."
        ),
    )
    parser.add_argument(
        "--toroidal-perturbation-amplitude",
        type=float,
        default=0.0,
        help=(
            "Relative cosine modulation used only by the legacy logical "
            "blob. The finite field-aligned filament already breaks "
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
    parser.add_argument(
        "--reference-magnetic-field",
        type=float,
        default=None,
        metavar="B0_TESLA",
        help=(
            "Normalize MAKEGRID B by this field strength. By default the "
            "median cell-centered |B| is used."
        ),
    )
    parser.add_argument("--mi-over-me", type=float, default=1836.0)
    parser.add_argument("--perp-diffusion", type=float, default=1.0e-5)
    parser.add_argument("--parallel-diffusion", type=float, default=1.0e-5)
    parser.add_argument("--electron-collision-frequency", type=float, default=0.0)
    parser.add_argument(
        "--time-integrator",
        choices=("rk4",),
        default="rk4",
        help="Classical four-stage Runge--Kutta integrator.",
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
        ),
        default="line-u",
        help=(
            "SOLVAX right preconditioner for the phi inversion. Line "
            "preconditioners use local complete u and/or v grid lines."
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
            "automatically for the RK4 integrator."
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_flux_framework(args)
    except ValueError as error:
        parser.error(str(error))
    _configure_runtime_selectors(args)
    print(
        "[simulation] flux_framework="
        f"{args.flux_framework}; "
        f"parallel_velocity_layout={args.parallel_velocity_layout}; "
        f"parallel_flux_pairing={args.parallel_flux_pairing}; "
        "parallel_boundary_pairing="
        f"{os.environ['DRBX_PARALLEL_BOUNDARY_PAIRING']}; "
        f"parallel_short_leg_treatment={args.parallel_short_leg_treatment}; "
        f"curvature_evolution_component={args.curvature_evolution_component}",
        flush=True,
    )
    try:
        angular_group_profile = tuple(
            int(value)
            for value in str(args.angular_group_profile).split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--angular-group-profile must be comma-separated integers")
    if angular_group_profile:
        if args.topology != "toroidal":
            parser.error("--angular-group-profile requires --topology=toroidal")
        if len(angular_group_profile) != int(args.resolution[0]):
            parser.error(
                "--angular-group-profile must contain one value per radial ring"
            )
    try:
        descriptor = topology_descriptor(args.topology)
    except ValueError as error:
        parser.error(str(error))
    resolution = tuple(int(value) for value in args.resolution)
    if args.topology == "toroidal" and resolution[1] % 2:
        parser.error("toroidal global NTHETA must be even")
    if args.topology == "toroidal" and args.metric_mesh_shape is None:
        parser.error(
            "--metric-mesh-shape NU NTHETA NETA_PER_PERIOD is required for "
            "--topology toroidal"
        )
    if args.square_agglomeration != "none" and args.topology != "square":
        parser.error("--square-agglomeration applies only to --topology=square")
    if args.agglomeration_volume_ratio <= 1.0:
        parser.error("--agglomeration-volume-ratio must be greater than one")
    if (
        args.agglomeration_rate_threshold is not None
        and args.agglomeration_rate_threshold <= 0.0
    ):
        parser.error("--agglomeration-rate-threshold must be positive")
    if not 0.0 < args.agglomeration_rk4_safety <= 1.0:
        parser.error("--agglomeration-rk4-safety must lie in (0, 1]")
    if args.parallel_operator_scheme == "fci" and args.topology != "toroidal":
        parser.error("--parallel-operator-scheme=fci requires --topology=toroidal")
    if (
        args.fci_parallel_leg_scheme != "centered"
        and args.parallel_operator_scheme != "fci"
    ):
        parser.error(
            "--fci-parallel-leg-scheme=boundary-characteristic-upwind requires "
            "--parallel-operator-scheme=fci"
        )
    if (
        args.fci_parallel_leg_scheme == "boundary-characteristic-upwind"
        and args.parallel_inflow_closure != "equilibrium-characteristic"
    ):
        parser.error(
            "--fci-parallel-leg-scheme=boundary-characteristic-upwind requires "
            "--parallel-inflow-closure=equilibrium-characteristic"
        )
    if args.fci_trace_substeps < 1:
        parser.error("--fci-trace-substeps must be positive")
    if (
        not np.isfinite(args.curvature_rlp_fine_glue_penalty)
        or args.curvature_rlp_fine_glue_penalty < 0.0
    ):
        parser.error("--curvature-rlp-fine-glue-penalty must be finite and nonnegative")
    if (
        args.curvature_rlp_face_scheme
        in ("fine-glue-characteristic", "fine-glue-characteristic-bulk")
        and frozenset(args.curvature_equations)
        != frozenset(("density", "Te", "Ti", "vorticity"))
    ):
        parser.error(
            "characteristic fine-glue curvature requires "
            "all four --curvature-equations"
        )
    if (
        args.curvature_rlp_face_scheme == "fine-glue-characteristic-bulk"
        and args.curvature_transition_audit_face is not None
    ):
        parser.error(
            "--curvature-transition-audit-face is incompatible with "
            "--curvature-rlp-face-scheme=fine-glue-characteristic-bulk"
        )
    shard_counts = tuple(int(value) for value in args.shard_counts)
    if any(value < 1 for value in shard_counts):
        parser.error("--shard-counts entries must be positive")
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        parser.error(
            "production sharding is eta-only; use --shard-counts 1 1 NETA_SHARDS"
        )
    if args.topology == "toroidal":
        if args.gmres_preconditioner not in ("none", "line-u"):
            parser.error("toroidal RLP supports only --gmres-preconditioner none or line-u")
        if args.poisson_bracket_scheme != "compatible-flux":
            parser.error("toroidal RLP requires --poisson-bracket-scheme=compatible-flux")
        if args.curvature_scheme != "conservative":
            parser.error("toroidal RLP requires --curvature-scheme=conservative")
        if (
            args.curvature_rlp_face_scheme in (
                "moment-shared",
                "bounded-moment-shared",
                "constrained-flux-shared",
            )
            and shard_counts != (1, 1, 1)
        ):
            parser.error(
                "moment-shared RLP curvature currently requires "
                "--shard-counts 1 1 1"
            )
    elif args.curvature_rlp_face_scheme != "projected-fine":
        parser.error(
            "moment-shared RLP curvature is only valid for "
            "toroidal angular RLP"
        )
    if args.square_agglomeration == "corner-edge":
        if args.time_integrator != "rk4":
            parser.error("square corner-edge agglomeration currently requires --time-integrator=rk4")
        if args.gmres_preconditioner != "line-u":
            parser.error("square corner-edge agglomeration currently requires --gmres-preconditioner=line-u")
        if args.poisson_bracket_scheme != "compatible-flux":
            parser.error("square corner-edge agglomeration requires --poisson-bracket-scheme=compatible-flux")
        if args.curvature_scheme != "conservative":
            parser.error("square corner-edge agglomeration requires --curvature-scheme=conservative")
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
    if args.curvature_scale < 0.0:
        parser.error("--curvature-scale must be nonnegative")
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
    if args.parallel_subsystem_only and args.time_integrator != "rk4":
        parser.error(
            "--parallel-subsystem-only is currently supported only with "
            "--time-integrator=rk4"
        )
    if args.halo_width < 1:
        parser.error("--halo-width must be positive")
    if args.density_amplitude < 0.0:
        parser.error("--density-amplitude must be nonnegative")
    if args.temperature_amplitude < 0.0:
        parser.error("--temperature-amplitude must be nonnegative")
    if args.blob_width <= 0.0:
        parser.error("--blob-width must be positive")
    if not 0.0 < args.blob_parallel_half_length <= np.pi:
        parser.error(
            "--blob-parallel-half-length must lie in (0, pi]"
        )
    if args.fieldline_substeps_per_plane < 1:
        parser.error("--fieldline-substeps-per-plane must be positive")
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
        args.rhs_replay_electron_force_wall_audit
        and args.rhs_replay_history is None
    ):
        parser.error(
            "--rhs-replay-electron-force-wall-audit requires --rhs-replay-history"
        )
    if (
        args.curvature_manufactured_output is not None
        and args.rhs_replay_history is not None
    ):
        parser.error(
            "--curvature-manufactured-output cannot be combined with RHS replay"
        )
    if (
        args.curvature_manufactured_output is not None
        and args.curvature_scheme != "conservative"
    ):
        parser.error(
            "--curvature-manufactured-output requires --curvature-scheme=conservative"
        )
    if (
        args.curvature_transition_audit_output is not None
        and (
            args.rhs_replay_history is not None
            or args.curvature_manufactured_output is not None
        )
    ):
        parser.error(
            "--curvature-transition-audit-output cannot be combined with "
            "RHS replay or the manufactured audit"
        )
    if (
        args.curvature_transition_audit_output is not None
        and args.curvature_scheme != "conservative"
    ):
        parser.error(
            "--curvature-transition-audit-output requires "
            "--curvature-scheme=conservative"
        )
    if args.curvature_transition_audit_face is not None:
        if args.curvature_transition_audit_output is None:
            parser.error(
                "--curvature-transition-audit-face requires "
                "--curvature-transition-audit-output"
            )
        if args.curvature_rlp_face_scheme not in (
            "fine-glue-sat",
            "fine-glue-characteristic",
            "fine-glue-characteristic-bulk",
        ):
            parser.error(
                "--curvature-transition-audit-face requires "
                "a fine-glue --curvature-rlp-face-scheme"
            )
    if args.curvature_scheme != "conservative" and args.curvature_inflow_closure != "central":
        parser.error(
            "--curvature-inflow-closure applies only to "
            "--curvature-scheme=conservative"
        )
    if (
        args.ion_temperature_curvature_self_form == "flux"
        and args.curvature_scheme != "conservative"
    ):
        parser.error(
            "--ion-temperature-curvature-self-form=flux requires "
            "--curvature-scheme=conservative"
        )
    if not 0.0 <= args.toroidal_perturbation_amplitude < 1.0:
        parser.error(
            "--toroidal-perturbation-amplitude must lie in [0, 1)"
        )
    if args.toroidal_perturbation_mode < 1:
        parser.error("--toroidal-perturbation-mode must be positive")
    if args.blob_initialization == "field-aligned":
        if args.temperature_amplitude != 0.0:
            parser.error(
                "--temperature-amplitude must be zero for the density-only "
                "field-aligned filament"
            )
        if args.toroidal_perturbation_amplitude != 0.0:
            parser.error(
                "--toroidal-perturbation-amplitude must be zero for the "
                "field-aligned filament; its finite parallel envelope "
                "already breaks field-period symmetry"
            )
    if not args.makegrid.is_file():
        parser.error(f"MAKEGRID file does not exist: {args.makegrid}")
    if len(args.makegrid_currents) != 12 or not np.all(
        np.isfinite(args.makegrid_currents)
    ):
        parser.error("--makegrid-currents must contain exactly 12 finite values")
    if not args.vessel.is_file():
        parser.error(f"vessel file does not exist: {args.vessel}")

    print(
        "building HSX geometry: "
        f"cells={resolution}, shards={shard_counts}, "
        f"devices={int(np.prod(shard_counts))}, "
        f"makegrid={args.makegrid}, currents={args.makegrid_currents}, "
        f"vessel={args.vessel}",
        flush=True,
    )
    geometry_start = time.perf_counter()
    geometry_result = (
        build_hsx_fci_geometry(
            makegrid_path=args.makegrid,
            makegrid_currents=args.makegrid_currents,
            vessel_path=args.vessel,
            resolution=resolution,
            fit_sample_shape=tuple(
                int(value) for value in args.fit_sample_shape
            ),
            radial_degree=args.radial_degree,
            vertical_degree=args.vertical_degree,
            toroidal_modes=args.toroidal_modes,
            metric_spline_degree=args.metric_spline_degree,
            mmpde_iterations=args.mmpde_iterations,
            axis_core_radius=args.axis_core_radius,
            reference_magnetic_field=args.reference_magnetic_field,
            topology=descriptor.name,
            metric_mesh_shape=(
                None
                if args.metric_mesh_shape is None
                else tuple(int(value) for value in args.metric_mesh_shape)
            ),
            metric_radial_degree=args.metric_radial_degree,
            metric_poloidal_modes=args.metric_poloidal_modes,
            metric_toroidal_modes=args.metric_toroidal_modes,
            eta_projection_iterations=args.eta_projection_iterations,
            construct_fci_maps=(args.parallel_operator_scheme == "fci"),
            fci_trace_substeps=int(args.fci_trace_substeps),
            metric_cache_dir=(
                None if args.no_metric_cache else args.metric_cache_dir
            ),
            rebuild_metric_cache=bool(args.rebuild_metric_cache),
            return_metric_evaluator=(args.topology == "toroidal"),
        )
    )
    if args.topology == "toroidal":
        (
            global_geometry,
            cell_positions,
            nfp,
            metric_cache_path,
            metric_evaluator,
        ) = geometry_result
    else:
        global_geometry, cell_positions, nfp, metric_cache_path = geometry_result
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
    owner_host_geometry = None
    control_volume_descriptor = None
    control_volume_fields = None
    control_volume_boundary_bc = None
    control_volume_assembler = None
    control_volume_field_count = RLP_PACKED_FIELD_COUNT
    angular_profile_safety_ratio = None
    if args.topology == "toroidal":
        try:
            (
                owner_host_geometry,
                angular_profile_safety_ratio,
            ) = build_metric_aware_polar_angular_agglomeration_geometry(
                global_geometry,
                metric_evaluator,
                # Diagnostic overrides are intentionally rebuilt instead of
                # entering the production angular-host cache namespace.
                metric_cache_path=(
                    None if angular_group_profile else metric_cache_path
                ),
                explicit_profile=(
                    angular_group_profile if angular_group_profile else None
                ),
            )
        except ValueError as error:
            parser.error(f"invalid --angular-group-profile: {error}")
        print(
            f"[angular-rlp-host] profile={owner_host_geometry.angular_group_size.tolist()} "
            f"minimum_width_ratio={angular_profile_safety_ratio:.6g}",
            flush=True,
        )
        if args.curvature_transition_audit_face is not None:
            transition_face = int(args.curvature_transition_audit_face)
            profile = tuple(
                int(value) for value in owner_host_geometry.angular_group_size
            )
            if not 0 < transition_face < len(profile):
                parser.error(
                    "--curvature-transition-audit-face must identify an "
                    f"interior radial face in [1, {len(profile) - 1}]"
                )
            if profile[transition_face - 1] == profile[transition_face]:
                parser.error(
                    "--curvature-transition-audit-face must identify an "
                    "angular-group transition"
                )
        (
            control_volume_descriptor,
            control_volume_fields,
        ) = build_sharded_polar_angular_agglomeration_payload(
            owner_host_geometry,
            sharded_geometry.domain,
            compile_compact_transition_faces=(
                args.curvature_rlp_face_scheme in (
                    "moment-shared",
                    "bounded-moment-shared",
                    "constrained-flux-shared",
                )
            ),
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
    staggered_face_provenance = None
    staggered_face_topology_host = None
    if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered":
        if (
            control_volume_descriptor is None
            or control_volume_assembler is None
            or control_volume_fields is None
            or sharded_geometry.map_fields is None
        ):
            raise ValueError("fci-staggered provenance requires angular RLP and FCI maps")

        def staggered_face_preflight(cell_fields_owned, map_fields_owned, cv_fields_owned):
            local_geometry = assemble_local_fci_geometry(
                sharded_geometry, cell_fields_owned, map_fields_owned
            )
            local_cv = control_volume_assembler(
                control_volume_descriptor, cv_fields_owned, local_geometry
            )
            topology = build_local_outgoing_fci_face_topology_from_geometry(
                local_cv.cells, local_geometry.maps
            )
            return (
                topology.edge_owner_i + jax.lax.axis_index("x") * topology.shape[0],
                topology.edge_owner_j + jax.lax.axis_index("y") * topology.shape[1],
                topology.edge_owner_k + jax.lax.axis_index("z") * topology.shape[2],
                topology.edge_active, topology.is_active_owner,
                topology.edge_measure, topology.aggregate_measure,
                topology.edge_destination_i + jax.lax.axis_index("x") * topology.shape[0],
                topology.edge_destination_j + jax.lax.axis_index("y") * topology.shape[1],
                topology.edge_destination_k + jax.lax.axis_index("z") * topology.shape[2],
                topology.edge_interpolation_provenance, topology.edge_destination_support,
            )

        staggered_face_preflight_compiled = jax.jit(jax.shard_map(
            staggered_face_preflight, mesh=mesh,
            in_specs=(P("x", "y", "z", None),) * 3,
            out_specs=(P("x", "y", "z"),) * 10 + (P("x", "y", "z", None),) * 2,
            check_vma=True,
        ))
        staggered_face_arrays = tuple(np.asarray(value) for value in (
            staggered_face_preflight_compiled(
                jax.device_put(jnp.asarray(sharded_geometry.cell_fields, dtype=jnp.float64),
                               NamedSharding(mesh, P("x", "y", "z", None))),
                jax.device_put(jnp.asarray(sharded_geometry.map_fields, dtype=jnp.float64),
                               NamedSharding(mesh, P("x", "y", "z", None))),
                jax.device_put(jnp.asarray(control_volume_fields, dtype=jnp.float64),
                               NamedSharding(mesh, P("x", "y", "z", None))),
            )
        ))
        (face_owner_i, face_owner_j, face_owner_k, face_active, face_owner_active,
         face_measure, face_aggregate_measure, face_destination_i, face_destination_j,
         face_destination_k, face_provenance, face_destination_support) = staggered_face_arrays
        fine_indices = np.indices(face_active.shape, dtype=np.int32)
        face_alias = face_active & (
            (face_owner_i != fine_indices[0]) | (face_owner_j != fine_indices[1])
            | (face_owner_k != fine_indices[2])
        )
        face_member_count = np.zeros(face_active.shape, dtype=np.int64)
        np.add.at(face_member_count,
                  (face_owner_i[face_active], face_owner_j[face_active], face_owner_k[face_active]), 1)

        def face_sha256(*arrays):
            digest = hashlib.sha256()
            for array in arrays:
                canonical = np.ascontiguousarray(array)
                digest.update(str(canonical.dtype).encode())
                digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
                digest.update(canonical.tobytes())
            return digest.hexdigest()

        staggered_face_provenance = {
            "face_basis_policy": OUTGOING_FCI_FACE_OWNERSHIP_POLICY,
            "face_basis_version": OUTGOING_FCI_FACE_OWNERSHIP_POLICY.rsplit("-v", 1)[-1],
            "fine_face_count": int(np.count_nonzero(face_active)),
            "face_owner_count": int(np.count_nonzero(face_owner_active)),
            "face_alias_count": int(np.count_nonzero(face_alias)),
            "face_max_fine_edges_per_owner": int(np.max(face_member_count, initial=0)),
            "face_owner_map_sha256": face_sha256(
                face_owner_i, face_owner_j, face_owner_k, face_active, face_owner_active
            ),
            "face_measure_sha256": face_sha256(face_measure, face_aggregate_measure),
            "face_provenance_sha256": face_sha256(
                face_destination_i, face_destination_j, face_destination_k, face_provenance,
                face_destination_support
            ),
        }
        staggered_face_topology_host = {
            "edge_owner_i": face_owner_i, "edge_owner_j": face_owner_j,
            "edge_owner_k": face_owner_k, "edge_active": face_active,
            "is_active_owner": face_owner_active, "edge_measure": face_measure,
            "aggregate_measure": face_aggregate_measure,
            "edge_destination_i": face_destination_i,
            "edge_destination_j": face_destination_j,
            "edge_destination_k": face_destination_k,
            "edge_interpolation_provenance": face_provenance,
            "edge_destination_support": face_destination_support,
        }
        print(
            "[staggered-face-preflight] "
            f"fine={staggered_face_provenance['fine_face_count']}, "
            f"owners={staggered_face_provenance['face_owner_count']}, "
            f"aliases={staggered_face_provenance['face_alias_count']}, "
            f"max_edges_per_owner={staggered_face_provenance['face_max_fine_edges_per_owner']}",
            flush=True,
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
    if args.square_agglomeration == "corner-edge":
        equilibrium_matrix = np.asarray(
            parallel_characteristic_matrix(
                jnp.asarray(1.0),
                jnp.asarray(1.0),
                jnp.asarray(1.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                float(args.tau),
                float(args.mi_over_me),
            ),
            dtype=np.float64,
        )
        corner_edge_characteristic_speed = float(
            np.max(np.abs(np.linalg.eigvals(equilibrium_matrix)))
        )
        corner_edge_rate_threshold = (
            float(args.agglomeration_rate_threshold)
            if args.agglomeration_rate_threshold is not None
            else (
                float(args.agglomeration_rk4_safety)
                * (2.0 * np.sqrt(2.0))
                / (corner_edge_characteristic_speed * timestep)
            )
        )
        owner_host_geometry = build_corner_edge_agglomeration(
            global_geometry,
            rate_threshold=corner_edge_rate_threshold,
            volume_ratio=float(args.agglomeration_volume_ratio),
        )
        owner_index = np.asarray(
            owner_host_geometry.topology.owner_index,
            dtype=np.int32,
        )
        (
            control_volume_descriptor,
            control_volume_fields,
        ) = build_sharded_plane_local_owner_map_payload(
            owner_index[..., 0],
            owner_index[..., 1],
            owner_host_geometry.raw_volume,
            owner_host_geometry.aggregate_chart_volume,
            sharded_geometry.domain,
        )
        control_volume_boundary_bc = empty_angular_agglomeration_boundary_bc()
        control_volume_assembler = assemble_local_plane_local_owner_map_geometry
        control_volume_field_count = CORNER_EDGE_PACKED_FIELD_COUNT
        topology = owner_host_geometry.topology
        aggregate_targets = np.asarray(topology.is_active_owner) & (
            np.asarray(topology.aggregate_volume)
            > np.asarray(owner_host_geometry.raw_volume) * (1.0 + 1.0e-14)
        )
        member_count = np.bincount(
            np.asarray(topology.aggregate_id, dtype=np.int64).ravel(),
            minlength=int(np.prod(resolution)),
        ).reshape(resolution)
        active_group_sizes = member_count[aggregate_targets]
        aggregate_volume_ratios = (
            np.asarray(topology.aggregate_volume)[aggregate_targets]
            / float(np.median(owner_host_geometry.raw_volume))
        )
        active_projected_rates = np.asarray(
            owner_host_geometry.projected_parallel_rate
        )[np.asarray(topology.is_active_owner)]
        print(
            "[corner-edge-rlp-host] "
            f"characteristic_speed={corner_edge_characteristic_speed:.6g}, "
            f"rate_threshold={corner_edge_rate_threshold:.6g}, "
            f"seeds={int(np.count_nonzero(owner_host_geometry.seed_mask))}, "
            f"aggregates={int(np.count_nonzero(aggregate_targets))}, "
            f"aliases={int(np.count_nonzero(topology.is_merge_source))}, "
            f"members={int(np.sum(active_group_sizes))}, "
            f"group_size=[{int(np.min(active_group_sizes))},"
            f"{int(np.max(active_group_sizes))}], "
            f"volume_band=[{owner_host_geometry.target_volume_lower:.6e},"
            f"{owner_host_geometry.target_volume_upper:.6e}], "
            f"achieved_volume_ratio=[{float(np.min(aggregate_volume_ratios)):.6g},"
            f"{float(np.max(aggregate_volume_ratios)):.6g}], "
            f"projected_rate_max={float(np.max(active_projected_rates)):.6g}, "
            f"preferred_upper_exceptions="
            f"{int(np.count_nonzero(aggregate_volume_ratios > float(args.agglomeration_volume_ratio)))}",
            flush=True,
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
                None if args.no_metric_cache else args.metric_cache_dir
            ),
            rebuild_filament_cache=bool(args.rebuild_metric_cache),
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
        initial_state = _aggregate_initial_owner_state(initial_state, owner_host_geometry)
        if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") != "fci-staggered":
            _assert_owner_sparse(initial_state, owner_host_geometry, staggered_face_topology_host)
        print(
            "[simulation] initial scalar state volume-aggregated into canonical owners; "
            "staggered Vi/Ve are projected through FCI faces before evolution",
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
        f"[simulation] curvature scheme: {str(args.curvature_scheme)}; "
        f"RLP faces: {str(args.curvature_rlp_face_scheme)}; wall closure: "
        f"{str(args.curvature_wall_flux_closure)}",
        flush=True,
    )
    print(
        "[simulation] parallel operator scheme: "
        f"{str(args.parallel_operator_scheme)}; FCI leg scheme: "
        f"{str(args.fci_parallel_leg_scheme)}; FCI trace substeps: "
        f"{int(args.fci_trace_substeps)}",
        flush=True,
    )
    print(
        "[simulation] Poisson bracket scheme: "
        f"{str(args.poisson_bracket_scheme)}",
        flush=True,
    )
    print(
        f"[simulation] curvature scale: {float(args.curvature_scale):.6e}",
        flush=True,
    )
    print(
        "[simulation] curvature equations: "
        f"{', '.join(str(equation) for equation in args.curvature_equations)}",
        flush=True,
    )
    print(
        "[simulation] ion-temperature curvature self form: "
        f"{str(args.ion_temperature_curvature_self_form)}",
        flush=True,
    )
    print(
        "[simulation] Neumann ghost scheme: "
        f"{str(args.neumann_ghost_scheme)}",
        flush=True,
    )
    print(
        "[simulation] parallel velocity wall BC: "
        f"{str(args.parallel_velocity_wall_bc)}",
        flush=True,
    )
    parallel_closure_description = {
        "central": "centered operator traces",
        "local-characteristic": (
            "local five-field material characteristics with primitive BC "
            "incoming data; phi/vorticity excluded"
        ),
        "equilibrium-characteristic": (
            "local five-field material characteristics with incoming "
            "perturbations zeroed around (1,1,1,0,0); phi/vorticity excluded"
        ),
    }[str(args.parallel_inflow_closure)]
    print(
        "[simulation] parallel inflow closure: "
        f"{str(args.parallel_inflow_closure)} ({parallel_closure_description})",
        flush=True,
    )
    print(
        "[simulation] vorticity-current inflow trace: "
        f"{str(args.vorticity_current_inflow_trace)}",
        flush=True,
    )
    print(
        "[simulation] parallel subsystem only: "
        f"{bool(args.parallel_subsystem_only)}",
        flush=True,
    )
    print(
        "[simulation] conservative curvature boundary flux closure: "
        f"{str(args.curvature_inflow_closure)}",
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
        global_geometry=global_geometry,
        cell_positions=cell_positions,
        nfp=nfp,
        sharded_geometry=sharded_geometry,
        mesh=mesh,
        parameters=parameters,
        metric_cache_path=metric_cache_path,
        gmres_target_tolerance=float(args.gmres_target_tolerance),
        gmres_acceptance_tolerance=float(args.gmres_acceptance_tolerance),
        gmres_max_iterations=int(args.gmres_max_iterations),
        gmres_restart=int(args.gmres_restart),
        gmres_preconditioner=str(args.gmres_preconditioner),
        gmres_residual_correction_steps=int(
            args.gmres_residual_correction_steps
        ),
        parallel_operator_scheme=str(args.parallel_operator_scheme),
        fci_parallel_leg_scheme=str(args.fci_parallel_leg_scheme),
        time_integrator=str(args.time_integrator),
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
        curvature_manufactured_output=args.curvature_manufactured_output,
        curvature_transition_audit_output=args.curvature_transition_audit_output,
        run_metadata={
            "command": " ".join(sys.argv),
            "drbx_source_root": str(DRBX_SRC),
            **_topology_metadata(descriptor),
            "restart_from": None if args.restart_from is None else str(args.restart_from),
            "restart_frame": int(args.restart_frame),
            "final_time": float(args.final_time),
            "num_steps": int(args.num_steps),
            "dt": float(timestep),
            "sharding_policy": "eta-only",
            "shard_counts": [int(value) for value in shard_counts],
            "halo_width": int(args.halo_width),
            "fit_sample_shape": [
                int(value) for value in args.fit_sample_shape
            ],
            "radial_degree": int(args.radial_degree),
            "vertical_degree": int(args.vertical_degree),
            "toroidal_modes": int(args.toroidal_modes),
            "metric_mesh_shape": (
                None
                if args.metric_mesh_shape is None
                else [int(value) for value in args.metric_mesh_shape]
            ),
            "metric_radial_degree": int(args.metric_radial_degree),
            "metric_poloidal_modes": int(args.metric_poloidal_modes),
            "metric_toroidal_modes": int(args.metric_toroidal_modes),
            "eta_projection_iterations": int(args.eta_projection_iterations),
            "parallel_operator_scheme": str(args.parallel_operator_scheme),
            "parallel_velocity_layout": os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT", "cell-centered"),
            "parallel_flux_pairing": os.environ.get("DRBX_PARALLEL_FLUX_PAIRING", "legacy"),
            "parallel_boundary_pairing": os.environ.get("DRBX_PARALLEL_BOUNDARY_PAIRING", "legacy"),
            "parallel_boundary_pairing_source": "simulate_hsx_blob.py:--parallel-boundary-pairing",
            "parallel_short_leg_treatment": os.environ.get("DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"),
            "parallel_short_leg_treatment_source": "simulate_hsx_blob.py:--parallel-short-leg-treatment",
            "parallel_short_leg_cfl_limit": float(os.environ.get("DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT", "2.5")),
            "parallel_short_leg_cfl_limit_source": "simulate_hsx_blob.py:--parallel-short-leg-cfl-limit",
            "curvature_evolution_component": os.environ.get("DRBX_CURVATURE_EVOLUTION_COMPONENT", "full"),
            "curvature_evolution_component_source": "simulate_hsx_blob.py:--curvature-evolution-component",
            "curvature_radial_ablation": os.environ.get("DRBX_CURVATURE_RADIAL_ABLATION", "none"),
            "curvature_radial_ablation_source": "simulate_hsx_blob.py:--curvature-radial-ablation",
            "curvature_wall_flux_closure": os.environ.get("DRBX_CURVATURE_WALL_FLUX_CLOSURE", str(args.curvature_inflow_closure)),
            "curvature_wall_flux_closure_source": "simulate_hsx_blob.py:--curvature-wall-flux-closure",
            "curvature_wall_characteristic_jump": (
                "direct-boundary-minus-interior"
                if args.curvature_wall_flux_closure == "bc-characteristic"
                else "equilibrium-minus-interior"
            ),
            "parallel_material_wall_flux_closure": os.environ.get("DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE", str(args.parallel_inflow_closure)),
            "parallel_material_wall_flux_closure_source": "DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE",
            "curvature_characteristic_axes": os.environ.get("DRBX_CURVATURE_CHARACTERISTIC_AXES", "legacy"),
            "curvature_characteristic_axes_source": "simulate_hsx_blob.py:--curvature-characteristic-axes",
            "curvature_radial_characteristic_scheme": os.environ.get("DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME", "legacy"),
            "curvature_radial_characteristic_scheme_source": "simulate_hsx_blob.py:--curvature-radial-characteristic-scheme",
            "curvature_poloidal_characteristic_scheme": os.environ.get("DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME", "legacy"),
            "curvature_poloidal_characteristic_scheme_source": "simulate_hsx_blob.py:--curvature-poloidal-characteristic-scheme",
            "curvature_component_diagnostic_scheme": os.environ.get("DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME", "directional"),
            "curvature_component_diagnostic_scheme_source": "simulate_hsx_blob.py:--curvature-component-diagnostic-scheme",
            "poloidal_characteristic_penalty": (None if os.environ.get("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY") is None else float(os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY"])),
            "poloidal_characteristic_penalty_source": os.environ.get("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE", "inherited-from-curvature-rlp-fine-glue-penalty"),
            "field_locations": {"Vi": "fci-outgoing-face/source-edge" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else "cell-center", "Ve": "fci-outgoing-face/source-edge" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else "cell-center"},
            "face_owner_layout": (None if staggered_face_provenance is None else staggered_face_provenance["face_basis_policy"]),
            "outgoing_edge_mass_convention": "raw-fluid-cell-volume" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,
            "cell_velocity_projection": "PcRc-after-face-to-center" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,
            "face_native_parallel_forces": "direct-Gc-and-compatible-Dc" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,
            "center_force_to_face_transfer": "Pe-L-Rc-mass-adjoint-f2c" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,
            "initial_velocity_projection": "center-to-outgoing-face-Re" if os.environ.get("DRBX_PARALLEL_VELOCITY_LAYOUT") == "fci-staggered" else None,
            **({} if staggered_face_provenance is None else staggered_face_provenance),
            "perpendicular_velocity_geometry": "face-to-center-perpendicular-center-to-face",
            "fci_parallel_leg_scheme": str(args.fci_parallel_leg_scheme),
            "fci_trace_substeps": int(args.fci_trace_substeps),
            "metric_spline_degree": int(args.metric_spline_degree),
            "mmpde_iterations": int(args.mmpde_iterations),
            "axis_core_radius": float(args.axis_core_radius),
            "reference_magnetic_field": (
                None
                if args.reference_magnetic_field is None
                else float(args.reference_magnetic_field)
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
            "curvature_split_scheme": os.environ.get("DRBX_CURVATURE_SPLIT_SCHEME"),
            "curvature_split_scheme_env": os.environ.get("DRBX_CURVATURE_SPLIT_SCHEME"),
            "curvature_split_scheme_source": (
                "DRBX_CURVATURE_SPLIT_SCHEME"
                if os.environ.get("DRBX_CURVATURE_SPLIT_SCHEME") is not None
                else None
            ),
            "parallel_material_scheme": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME"),
            "parallel_material_scheme_env": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME"),
            "parallel_material_scheme_source": (
                "DRBX_PARALLEL_MATERIAL_SCHEME"
                if os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME") is not None
                else None
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
            "phi_solver_space": (
                "owner-grid-RLP"
                if control_volume_descriptor is not None
                else "full-grid"
            ),
            "curvature_scheme": str(args.curvature_scheme),
            "curvature_scale": float(args.curvature_scale),
            "curvature_rlp_face_scheme": str(
                args.curvature_rlp_face_scheme
            ),
            "curvature_rlp_fine_glue_penalty": float(
                args.curvature_rlp_fine_glue_penalty
            ),
            "curvature_transition_audit_face": (
                None
                if args.curvature_transition_audit_face is None
                else int(args.curvature_transition_audit_face)
            ),
            "curvature_equations": [
                str(equation) for equation in args.curvature_equations
            ],
            "ion_temperature_curvature_self_form": str(
                args.ion_temperature_curvature_self_form
            ),
            "neumann_ghost_scheme": str(args.neumann_ghost_scheme),
            "parallel_velocity_wall_bc": str(
                args.parallel_velocity_wall_bc
            ),
            "parallel_inflow_closure": str(args.parallel_inflow_closure),
            "vorticity_current_inflow_trace": str(
                args.vorticity_current_inflow_trace
            ),
            "parallel_subsystem_only": bool(args.parallel_subsystem_only),
            "curvature_inflow_closure": str(
                args.curvature_inflow_closure
            ),
            "poisson_bracket_scheme": str(args.poisson_bracket_scheme),
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
        reconstruct_initial_phi=not restart_used,
        curvature_scheme=str(args.curvature_scheme),
        curvature_scale=float(args.curvature_scale),
        curvature_rlp_face_scheme=str(args.curvature_rlp_face_scheme),
        curvature_rlp_fine_glue_penalty=float(
            args.curvature_rlp_fine_glue_penalty
        ),
        curvature_rlp_fine_glue_transition_face=(
            args.curvature_transition_audit_face
        ),
        curvature_equations=tuple(str(equation) for equation in args.curvature_equations),
        ion_temperature_curvature_self_form=(
            str(args.ion_temperature_curvature_self_form)
        ),
        neumann_ghost_scheme=str(args.neumann_ghost_scheme),
        parallel_velocity_wall_bc=str(args.parallel_velocity_wall_bc),
        parallel_inflow_closure=str(args.parallel_inflow_closure),
        vorticity_current_inflow_trace=str(
            args.vorticity_current_inflow_trace
        ),
        parallel_subsystem_only=bool(args.parallel_subsystem_only),
        curvature_inflow_closure=str(args.curvature_inflow_closure),
        poisson_bracket_scheme=str(args.poisson_bracket_scheme),
        curvature_split_scheme=(
            "production-path"
            if str(args.flux_framework) == "production-split"
            else "legacy"
        ),
        parallel_material_scheme=(
            "production-path"
            if str(args.flux_framework) == "production-split"
            else "legacy"
        ),
        track_curvature_chain_rule_defect=bool(
            args.track_curvature_chain_rule_defect
        ),
        control_volume_descriptor=control_volume_descriptor,
        control_volume_fields_host=control_volume_fields,
        control_volume_boundary_bc=control_volume_boundary_bc,
        control_volume_assembler=control_volume_assembler,
        control_volume_field_count=control_volume_field_count,
        owner_host_geometry=owner_host_geometry,
        outgoing_face_topology_host=staggered_face_topology_host,
    )


if __name__ == "__main__":
    main()
