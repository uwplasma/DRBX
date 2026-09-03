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
from typing import Callable, Mapping, Sequence
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
    LocalCurvatureFaceCoefficients3D,
    LocalFciGeometry3D,
    MetricEvaluator,
    MetricGeometry,
    Spacing3D,
    WallEvaluator,
    bfield_evaluator_from_makegrid,
    build_fci_maps_from_callbacks,
    build_local_conservative_stencil_from_field,
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
# Bump only when the callback tracer numerics or serialized map contract
# changes.  Unrelated edits elsewhere in fci_geometry.py must not invalidate a
# multi-minute full-torus trace.
FCI_MAP_TRACER_REVISION = 2


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
    """Fingerprint the driver-visible FCI map schema and tracer revision.

    The metric cache deliberately does not include ``fci_geometry.py`` in its
    metric-cache key: changing the tracer should not force an expensive metric
    rebuild.  Cached maps carry this separate, explicitly versioned
    fingerprint instead.  A whole-file mtime/size fingerprint is intentionally
    avoided because curvature and other unrelated edits live in the same
    module as the callback tracer.
    """

    contract = {
        "cache_format": FCI_MAP_CACHE_FORMAT_VERSION,
        "fields": list(FCI_MAP_FIELDS),
        "endpoint_interpolation_order": 2,
        "tracer_revision": FCI_MAP_TRACER_REVISION,
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
    direction_checkpoint_path = (
        None
        if cache_path is None
        else cache_path.with_name(
            f".{cache_path.stem}.fci_trace_s{int(fci_trace_substeps)}_"
            f"{source_fingerprint[:16]}.npz"
        )
    )
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
            if (
                direction_checkpoint_path is not None
                and direction_checkpoint_path.exists()
            ):
                direction_checkpoint_path.unlink()
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
        direction_checkpoint_path=direction_checkpoint_path,
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
            if (
                direction_checkpoint_path is not None
                and direction_checkpoint_path.exists()
            ):
                direction_checkpoint_path.unlink()
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
    *,
    parallel_velocity_wall_bc: str = "neumann",
    physical_wall_model: str = "legacy-velocity-trace",
    conducting_sheath_wall_potential: float | None = None,
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
    return model(state, geometry, domain, parameters)


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
    neumann_ghost_scheme: str = "physical",
    parallel_velocity_wall_bc: str = "neumann",
    physical_wall_model: str = "legacy-velocity-trace",
    conducting_sheath_wall_potential: float | None = None,
    parallel_operator_scheme: str = "coordinate",
    poisson_bracket_scheme: str = "direct",
    parallel_material_scheme: str | None = None,
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
    ):
        raise ValueError(
            "poisson_bracket_scheme must be 'direct', 'compatible-flux', or "
            "'compatible-third-order-upwind', or "
            "'material-scalar-third-order-upwind', "
            f"got {poisson_bracket_scheme!r}"
        )
    if parallel_material_scheme is None:
        parallel_material_scheme = os.environ.get(
            "DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"
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
    def face_bc_builder(state, local_geometry, local_domain, local_parameters):
        return build_face_bc_bundle(
            state,
            local_geometry,
            local_domain,
            local_parameters,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            physical_wall_model=physical_wall_model,
            conducting_sheath_wall_potential=conducting_sheath_wall_potential,
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
        face_bc_builder=face_bc_builder,
        axis_regular_axes=domain.axis_regular_axes,
        curvature_face_coefficients=curvature_face_coefficients,
        poisson_bracket_scheme=poisson_bracket_scheme,
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
    parallel_material_scheme: str | None = None,
    track_curvature_chain_rule_defect: bool = False,
    control_volume_descriptor=None,
    control_volume_fields_host=None,
    control_volume_boundary_bc=None,
    control_volume_assembler=None,
    control_volume_field_count: int = RLP_PACKED_FIELD_COUNT,
    owner_host_geometry=None,
    source_evaluator: Callable[[float], FciDrbEBState] | None = None,
    history_dtype: str = "float32",
    frozen_diagnostic: FrozenEbDiagnosticRequest | None = None,
    staged_audit_cells: tuple[tuple[int, int, int], ...] = (),
    staged_audit_output: Path | None = None,
    staged_audit_explicit_ablation: str = "none",
) -> FciDrbEBState | FrozenEbDiagnosticResult:
    """Advance the global EB state or evaluate its sharded frozen diagnostic."""

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
    if short_leg_treatment == "local-backward-euler" and time_integrator != "imex-ssp222":
        raise ValueError(
            "local-backward-euler short legs require the stage-wise "
            "time_integrator='imex-ssp222'; post-step RK4 splitting is not "
            "a consistent handoff"
        )
    if time_integrator == "imex-ssp222" and short_leg_treatment != "local-backward-euler":
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
            map_fields_owned if parallel_operator_scheme == "fci" else None,
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
            gmres_preconditioner=str(gmres_preconditioner),
            gmres_residual_correction_steps=int(
                gmres_residual_correction_steps
            ),
            neumann_ghost_scheme=neumann_ghost_scheme,
            parallel_velocity_wall_bc=parallel_velocity_wall_bc,
            physical_wall_model=physical_wall_model,
            conducting_sheath_wall_potential=conducting_sheath_wall_potential,
            parallel_operator_scheme=parallel_operator_scheme,
            parallel_material_scheme=parallel_material_scheme,
            poisson_bracket_scheme=poisson_bracket_scheme,
            control_volume_geometry=local_control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            curvature_face_coefficients_override=(
                local_curvature_face_coefficients
            ),
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
            frozen_stage = jax.jit(frozen_stage_sharded)
            frozen_implicit = jax.jit(frozen_implicit_sharded)
            frozen_reconstruct_phi = jax.jit(frozen_reconstruct_phi_sharded)
        else:
            frozen_stage = frozen_stage_sharded
            frozen_implicit = frozen_implicit_sharded
            frozen_reconstruct_phi = frozen_reconstruct_phi_sharded

        def execute(callable_, *values):
            with jax.disable_jit(frozen_diagnostic.execution == "eager"):
                result = callable_(*values)
            jax.block_until_ready(result)
            return result

        common_geometry = (cell_fields, map_fields, control_volume_fields)
        exact_explicit, exact_terms = execute(
            frozen_stage,
            state,
            zero_source,
            *common_geometry,
        )
        sourced_explicit, sourced_terms = execute(
            frozen_stage,
            state,
            sharded_source,
            *common_geometry,
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
        reconstructed_explicit, reconstructed_terms = execute(
            frozen_stage,
            reconstructed_state,
            zero_source,
            *common_geometry,
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
        if track_curvature_chain_rule_defect:
            curvature_diagnostics = (
                model.ion_temperature_curvature_chain_rule_diagnostics(next_state)
            )
            for mesh_axis_name in ("x", "y", "z"):
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
            stage_phi, phi_info = reconstruct_stage_phi(updated, model)
            stage = updated.replace(phi=stage_phi)
            implicit_rate = increment.map_fields(
                lambda value: value / gamma_dt
            )
            return stage, implicit_rate, phi_info

        # The persisted current state already carries its consistent algebraic
        # potential.  Solve the complete selected-wall residual before the
        # first explicit evaluation, not after a finished timestep.
        stage_1, implicit_1, gmres_info_1 = implicit_stage(current)
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
        explicit_1 = evaluate_operators(
            stage_1, stage_1.phi, model, source_1
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
            stage_2, stage_2.phi, model, source_2
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

    full_advance = (
        full_rk4_advance if time_integrator == "rk4" else full_imex_advance
    )
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
        with jax.disable_jit(advance_execution == "eager"):
            return compiled_advance(*advance_args)

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
                integrator=time_integrator,
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
        args.parallel_short_leg_treatment == "local-backward-euler"
        and framework != "production-split"
    ):
        raise ValueError(
            "--parallel-short-leg-treatment local-backward-euler requires "
            "--flux-framework production-split"
        )
    if (
        args.parallel_short_leg_treatment == "local-backward-euler"
        and args.time_integrator != "imex-ssp222"
    ):
        raise ValueError(
            "--parallel-short-leg-treatment local-backward-euler requires "
            "--time-integrator imex-ssp222 so the complete selected residual "
            "is solved at every stage"
        )
    if (
        args.time_integrator == "imex-ssp222"
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
    ):
        raise ValueError("production-split requires compatible Poisson brackets")


def _configure_runtime_selectors(args: argparse.Namespace) -> None:
    """Export native CLI selectors consumed by LocalFciDrbEBRhs factories."""

    os.environ["DRBX_FLUX_FRAMEWORK"] = str(args.flux_framework)
    os.environ["DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"] = str(args.parallel_characteristic_wall_law)
    os.environ["DRBX_PARALLEL_FLUX_PAIRING"] = str(args.parallel_flux_pairing)
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
        "--fci-trace-substeps",
        type=int,
        default=4,
        help="RK4 substeps per toroidal plane used when generating FCI maps.",
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
            "trace with warm-ion Bohm outflow and electron saturation response. "
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
            "compatible bracket for vorticity."
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
        choices=("rk4", "imex-ssp222"),
        default="rk4",
        help=(
            "Time integrator. Classical RK4 is used for fully explicit "
            "configurations. 'imex-ssp222' is the stage-wise two-stage IMEX "
            "method required by local backward-Euler short wall legs."
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
        "parallel_velocities=cell-centered; "
        f"parallel_flux_pairing={args.parallel_flux_pairing}; "
        f"parallel_characteristic_wall_law={args.parallel_characteristic_wall_law}; "
        f"physical_wall_model={args.physical_wall_model}; "
        "parallel_boundary_pairing="
        f"{os.environ['DRBX_PARALLEL_BOUNDARY_PAIRING']}; "
        f"parallel_short_leg_treatment={args.parallel_short_leg_treatment}; "
        f"parallel_short_leg_selection={args.parallel_short_leg_selection}",
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
    if args.fci_trace_substeps < 1:
        parser.error("--fci-trace-substeps must be positive")
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
        if args.poisson_bracket_scheme not in (
            "compatible-flux",
            "compatible-third-order-upwind",
            "material-scalar-third-order-upwind",
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
        f"{int(args.fci_trace_substeps)}",
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
        f"{'production-rung2-simple-conducting-sheath' if args.physical_wall_model == 'simple-conducting-sheath' else 'no-flow' if args.physical_wall_model == 'no-flow' else 'legacy'}"
        ")",
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
        time_integrator=str(args.time_integrator),
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
            "parallel_flux_pairing": os.environ.get("DRBX_PARALLEL_FLUX_PAIRING", "legacy"),
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
                [
                    "selected-characteristic-material-action",
                    "selected-mu-tau-grad-parallel-Ti",
                ]
                if args.parallel_short_leg_treatment == "local-backward-euler"
                else []
            ),
            "parallel_short_leg_explicit_energy_pair": (
                "mu-grad-parallel-phi<->weighted-adjoint-current-divergence"
            ),
            "parallel_short_leg_time_handoff": (
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
                (
                    "implicit-short-leg-plus-phi",
                    "explicit-rhs",
                    "standalone-phi",
                    "stage-diagnostics",
                )
                if args.advance_execution == "staged-compiled"
                else ("monolithic-advance",)
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
            "neumann_ghost_scheme": str(args.neumann_ghost_scheme),
            "physical_wall_model": str(args.physical_wall_model),
            "conducting_sheath_wall_potential": (
                None
                if args.conducting_sheath_wall_potential is None
                else float(args.conducting_sheath_wall_potential)
            ),
            "physical_wall_model_provenance": (
                "production-rung2-simple-conducting-sheath"
                if args.physical_wall_model == "simple-conducting-sheath"
                else "no-flow-rung1"
                if args.physical_wall_model == "no-flow"
                else "legacy-compatibility"
            ),
            "parallel_velocity_wall_bc": str(
                args.parallel_velocity_wall_bc
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
        neumann_ghost_scheme=str(args.neumann_ghost_scheme),
        parallel_velocity_wall_bc=str(args.parallel_velocity_wall_bc),
        physical_wall_model=str(args.physical_wall_model),
        conducting_sheath_wall_potential=(
            None
            if args.conducting_sheath_wall_potential is None
            else float(args.conducting_sheath_wall_potential)
        ),
        poisson_bracket_scheme=str(args.poisson_bracket_scheme),
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
        staged_audit_cells=staged_audit_cells,
        staged_audit_output=args.staged_audit_output,
        staged_audit_explicit_ablation=str(
            args.staged_audit_explicit_ablation
        ),
    )


if __name__ == "__main__":
    main()
