"""Producer-side HSX physical geometry builder.

This module owns MAKEGRID/vessel loading, metric fitting, resolution-local sampling,
FCI map tracing, and producer checkpoints. The simulation consumer must receive
a completed geometry artifact and never imports or calls this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
from typing import Mapping
import zipfile

import jax.numpy as jnp
import numpy as np

from .Bfield_evaluator import bfield_evaluator_from_makegrid
from .MetricEvaluator import MetricEvaluator, build_metric_evaluator
from .ScalarPotential_evaluator import scalar_potential_evaluator_from_bfield
from .WallEvaluator import WallEvaluator
from .fci_geometry import (
    BFieldGeometry, CellCenteredGrid3D, CurvatureEdgeOneForm3D,
    FaceBFieldGeometry, FaceMetricGeometry, FciGeometry3D, FciMaps3D,
    Grid1D, MetricGeometry, Spacing3D,
    build_fci_maps_from_callbacks,
    build_metric_aware_polar_angular_agglomeration_geometry,
)
from .solve_MMPDE import MMPDEOptions

SCRIPT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE_DATA_DIR = SCRIPT_DIR.parent

DEFAULT_MAKEGRID = DEFAULT_WORKSPACE_DATA_DIR / "mgrid_res2p5cm_180pln.nc"
DEFAULT_VESSEL = DEFAULT_WORKSPACE_DATA_DIR / "vessel_hsx_flare.txt"
DEFAULT_METRIC_CACHE_DIR = DEFAULT_WORKSPACE_DATA_DIR / ".hsx_metric_cache"
HSX_QHS_MAIN_CURRENT_AMPERES = 10722.0
DEFAULT_HSX_QHS_MAKEGRID_CURRENTS = (
    (HSX_QHS_MAIN_CURRENT_AMPERES,) * 6 + (0.0,) * 6
)
METRIC_CACHE_FORMAT_VERSION = 7
# Cache validity follows this explicit numerical contract, rather than the
# checkout path or source-file mtimes. Bump this revision whenever the fitted
# metric/wall-coordinate algorithms change their numerical output.
METRIC_BUILDER_REVISION = 1
METRIC_CACHE_IDENTITY_VERSION = 2
METRIC_INPUT_FULL_HASH_LIMIT = 64 * 2**20
METRIC_INPUT_SAMPLE_BYTES = 2**20
METRIC_INPUT_SAMPLE_COUNT = 16
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
FCI_MAP_FLOAT_FIELDS = FCI_MAP_FIELDS[:-2]
FCI_MAP_BOOL_FIELDS = FCI_MAP_FIELDS[-2:]
FCI_MAP_CACHE_PREFIX = "fci_maps_"
FCI_MAP_CACHE_FORMAT_VERSION = 2
# Bump only when the callback tracer numerics or serialized map contract
# changes.  Unrelated edits elsewhere in fci_geometry.py must not invalidate a
# multi-minute full-torus trace.
FCI_MAP_TRACER_REVISION = 3


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


def _metric_input_content_identity(path: Path) -> dict[str, object]:
    """Return a path/mtime-independent identity for a metric input file.

    Small files are hashed completely. Multi-gigabyte MAKEGRID files use a
    deterministic set of evenly spaced 1 MiB samples, including both ends, so
    cache lookup does not require rereading the complete equilibrium file.
    """

    path = Path(path)
    size = int(path.stat().st_size)
    digest = hashlib.sha256()
    digest.update(b"drbx-metric-input-v1\0")
    digest.update(size.to_bytes(16, byteorder="little", signed=False))
    if size <= METRIC_INPUT_FULL_HASH_LIMIT:
        scheme = "sha256-full-v1"
        with path.open("rb") as stream:
            while chunk := stream.read(METRIC_INPUT_SAMPLE_BYTES):
                digest.update(chunk)
    else:
        sample_size = min(METRIC_INPUT_SAMPLE_BYTES, size)
        maximum_offset = size - sample_size
        offsets = np.unique(
            np.linspace(
                0,
                maximum_offset,
                num=METRIC_INPUT_SAMPLE_COUNT,
                dtype=np.int64,
            )
        )
        scheme = f"sha256-sampled-{offsets.size}x{sample_size}-v1"
        with path.open("rb") as stream:
            for offset_value in offsets:
                offset = int(offset_value)
                stream.seek(offset)
                chunk = stream.read(sample_size)
                if len(chunk) != sample_size:
                    raise OSError(
                        f"short read while fingerprinting {path} at {offset}"
                    )
                digest.update(offset.to_bytes(16, byteorder="little", signed=False))
                digest.update(chunk)
    return {"size": size, "hash_scheme": scheme, "sha256": digest.hexdigest()}


def _metric_cache_specs_compatible(
    cached_spec: Mapping[str, object],
    expected_spec: Mapping[str, object],
) -> bool:
    """Compare metric-cache contracts, including pre-v2 legacy contracts.

    Version-2 identities require content fingerprints. Legacy version-7
    caches did not store them, so they are admitted when all numerical options
    and input sizes match; paths, mtimes, and source checkout metadata are
    deliberately ignored. Payload shape/topology validation still follows.
    """

    cached = dict(cached_spec)
    expected = dict(expected_spec)
    cached_makegrid = cached.pop("makegrid", None)
    expected_makegrid = expected.pop("makegrid", None)
    cached_vessel = cached.pop("vessel", None)
    expected_vessel = expected.pop("vessel", None)
    cached_identity_version = cached.pop("identity_version", None)
    expected_identity_version = expected.pop("identity_version", None)
    cached_builder_revision = cached.pop("metric_builder_revision", None)
    expected_builder_revision = expected.pop("metric_builder_revision", None)
    cached_sources = cached.pop("geometry_sources", None)
    expected.pop("geometry_sources", None)

    if cached != expected:
        return False
    if cached_identity_version is not None and (
        cached_identity_version != expected_identity_version
    ):
        return False
    if cached_builder_revision is not None and (
        cached_builder_revision != expected_builder_revision
    ):
        return False

    def input_matches(cached_input: object, expected_input: object) -> bool:
        if not isinstance(cached_input, Mapping) or not isinstance(
            expected_input, Mapping
        ):
            return False
        if int(cached_input.get("size", -1)) != int(expected_input.get("size", -2)):
            return False
        cached_hash = cached_input.get("sha256")
        expected_hash = expected_input.get("sha256")
        if cached_hash is not None:
            return (
                cached_hash == expected_hash
                and cached_input.get("hash_scheme")
                == expected_input.get("hash_scheme")
            )
        return True

    if not input_matches(cached_makegrid, expected_makegrid):
        return False
    if not input_matches(cached_vessel, expected_vessel):
        return False

    # Old caches used source size/mtime metadata as an implicit algorithm
    # fingerprint. Only accept that legacy form for this payload format; new
    # caches use METRIC_BUILDER_REVISION instead.
    if cached_identity_version is None and cached_sources is not None:
        if int(cached_spec.get("format_version", -1)) != METRIC_CACHE_FORMAT_VERSION:
            return False
    return True


def _find_compatible_metric_cache(
    cache_dir: Path,
    expected_path: Path,
    expected_spec: Mapping[str, object],
) -> Path | None:
    """Find an exact-key or compatible legacy cache without loading arrays."""

    if expected_path.is_file():
        return expected_path
    try:
        candidates = sorted(
            cache_dir.glob("hsx_metric_*.npz"),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=False) as cached:
                cached_spec = json.loads(str(cached["cache_spec"].item()))
            if _metric_cache_specs_compatible(cached_spec, expected_spec):
                return candidate
        except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
            continue
    return None


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
    # A map-only cache is deliberately created empty on its first use.  That
    # is a normal cache miss, not an invalid cached map set, so only enter the
    # validation path once map metadata is actually present.
    if payload is not None and any(
        name in payload
        for name in (
            "fci_maps_trace_substeps",
            "fci_maps_source_fingerprint",
            "fci_maps_shape",
        )
    ):
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
                "[fci-map-cache] validated cached full-torus HSX maps from "
                f"{cache_path}",
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

    endpoint_field_ok = True
    for prefix, boundary in (
        ("forward", forward_boundary),
        ("backward", backward_boundary),
    ):
        endpoint_field_ok &= bool(
            np.all(arrays[f"{prefix}_endpoint_bmag"][boundary] > 0.0)
        )
        endpoint_field_ok &= all(
            bool(
                np.all(
                    np.isfinite(
                        arrays[f"{prefix}_endpoint_b_contra_{axis}"][boundary]
                    )
                )
            )
            for axis in ("x", "y", "z")
        )
    checks["physical_endpoint_magnetic_field"] = endpoint_field_ok
    if not endpoint_field_ok:
        errors.append(
            "physical FCI endpoints require a finite continuous magnetic-field "
            "evaluation and positive |B|"
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


def _build_hsx_curvature_edge_one_form(
    *,
    metric_evaluator: MetricEvaluator,
    bfield: object,
    u_faces: np.ndarray,
    v_faces: np.ndarray,
    eta0: float,
    eta_period: float,
    nfp: int,
    neta: int,
    reference_magnetic_field: float,
) -> CurvatureEdgeOneForm3D:
    """Sample the continuous ``(b/B)_alpha`` one-form on unique edges.

    Each edge is queried once at its logical midpoint.  The collapsed
    ``u=0`` rows are reconstructed from the first positive radial layer using
    the same half-turn trace used by the toroidal metric assembly; the
    singular evaluator is therefore never queried directly on the axis.
    """

    u_faces = np.asarray(u_faces, dtype=np.float64)
    v_faces = np.asarray(v_faces, dtype=np.float64)
    nfp = int(nfp)
    neta = int(neta)
    if nfp < 1 or neta < 1 or neta % nfp:
        raise ValueError("nfp must divide positive neta when sampling curvature edges")
    b0 = float(reference_magnetic_field)
    if not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("reference_magnetic_field must be positive and finite")
    nu = int(u_faces.size - 1)
    nv = int(v_faces.size - 1)
    neta_period = neta // nfp
    u_centers = 0.5 * (u_faces[:-1] + u_faces[1:])
    v_centers = 0.5 * (v_faces[:-1] + v_faces[1:])
    eta_faces_period = eta0 + np.arange(neta_period + 1, dtype=np.float64) * (
        float(eta_period) / float(neta_period)
    )
    eta_centers_period = 0.5 * (eta_faces_period[:-1] + eta_faces_period[1:])

    def sample(points: np.ndarray) -> np.ndarray:
        metric = metric_evaluator.evaluate(points)
        magnetic = metric_evaluator.evaluate_magnetic_field(points, bfield)
        bcontra = np.asarray(magnetic.B_contravariant, dtype=np.float64) / b0
        bmag = np.maximum(np.asarray(magnetic.magnitude, dtype=np.float64) / b0, 1.0e-30)
        result = np.einsum(
            "...ij,...j->...i",
            np.asarray(metric.g_cov, dtype=np.float64),
            bcontra / bmag[..., None],
        ) / bmag[..., None]
        if not np.all(np.isfinite(result)):
            raise ValueError("continuous curvature edge one-form evaluation is not finite")
        return result

    def points(u: np.ndarray, v: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return np.stack(np.meshgrid(u, v, eta, indexing="ij"), axis=-1)

    # Positive-u portions of the two edge families that touch the collapsed
    # axis.  The axis rows are supplied below from a finite proxy layer.
    az_positive = sample(points(u_faces[1:], v_faces, eta_centers_period))[..., 2]
    ay_positive = sample(points(u_faces[1:], v_centers, eta_faces_period))[..., 1]
    ax_period = sample(points(u_centers, v_faces, eta_faces_period))[..., 0]

    half_turn = nv // 2
    if nv % 2:
        raise ValueError("toroidal curvature edge sampling requires an even theta count")
    proxy_u = np.asarray([u_centers[0]], dtype=np.float64)
    az_proxy = sample(points(proxy_u, v_faces, eta_centers_period))[..., 2][0]
    ay_proxy = sample(points(proxy_u, v_centers, eta_faces_period))[..., 1][0]
    az_axis = 0.5 * (az_proxy + np.roll(az_proxy, half_turn, axis=0))
    ay_axis = 0.5 * (ay_proxy + np.roll(ay_proxy, half_turn, axis=0))
    az_period = np.concatenate((az_axis[None, ...], az_positive), axis=0)
    ay_period = np.concatenate((ay_axis[None, ...], ay_positive), axis=0)

    # The duplicate logical theta edge and the retained eta endpoint are
    # shared seams, not additional samples.  Normalize them explicitly so
    # every field-period copy enters the same incidence complex.
    az_period[:, -1, :] = az_period[:, 0, :]
    ay_period[:, :, -1] = ay_period[:, :, 0]
    ax_period[..., -1] = ax_period[..., 0]
    ax_period[:, -1, :] = ax_period[:, 0, :]
    if not (
        np.allclose(az_period[:, -1, :], az_period[:, 0, :], rtol=0.0, atol=0.0)
        and np.allclose(ay_period[:, :, -1], ay_period[:, :, 0], rtol=0.0, atol=0.0)
        and np.allclose(ax_period[..., -1], ax_period[..., 0], rtol=0.0, atol=0.0)
        and np.allclose(ax_period[:, -1, :], ax_period[:, 0, :], rtol=0.0, atol=0.0)
    ):
        raise ValueError("curvature edge one-form seam closure failed")

    result = CurvatureEdgeOneForm3D(
        Az_xy=_repeat_field_period_cells(az_period, nfp),
        Ay_xz=_repeat_field_period_eta_faces(ay_period, nfp),
        Ax_yz=_repeat_field_period_eta_faces(ax_period, nfp),
    )
    return result


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
    topology: str = "toroidal",
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
    topology: str = "toroidal",
    metric_mesh_shape: tuple[int, int, int] | None = None,
    metric_radial_degree: int = 17,
    metric_poloidal_modes: int = 15,
    metric_toroidal_modes: int = 16,
    eta_projection_iterations: int = 0,
    construct_fci_maps: bool = True,
    fci_trace_substeps: int = 64,
    metric_cache_dir: Path | None = DEFAULT_METRIC_CACHE_DIR,
    fci_map_cache_path: Path | None = None,
    rebuild_metric_cache: bool = False,
    metric_context: HSXMetricContext | None = None,
    return_metric_evaluator: bool = False,
    return_curvature_edge_one_form: bool = False,
) -> tuple[FciGeometry3D, np.ndarray, int, Path | None] | tuple[
    FciGeometry3D, np.ndarray, int, Path | None, MetricEvaluator
] | tuple[
    FciGeometry3D, np.ndarray, int, Path | None, CurvatureEdgeOneForm3D
] | tuple[
    FciGeometry3D,
    np.ndarray,
    int,
    Path | None,
    MetricEvaluator,
    CurvatureEdgeOneForm3D,
]:
    """Build the global HSX geometry and its Cartesian cell embedding."""

    descriptor = topology_descriptor(topology)
    topology = descriptor.name
    if not construct_fci_maps:
        raise ValueError(
            "build_hsx_fci_geometry requires mapped geometry; coordinate-only "
            "callers must use build_hsx_metric_evaluator explicitly"
        )
    if topology != "toroidal":
        raise ValueError(
            "mapped HSX geometry is currently supported only for "
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
        cache_spec = {
            "format_version": METRIC_CACHE_FORMAT_VERSION,
            "identity_version": METRIC_CACHE_IDENTITY_VERSION,
            "metric_builder_revision": METRIC_BUILDER_REVISION,
            "makegrid": _metric_input_content_identity(makegrid_path),
            "makegrid_currents": (
                None
                if makegrid_current_array is None
                else makegrid_current_array.tolist()
            ),
            "vessel": _metric_input_content_identity(vessel_path),
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
            if not rebuild_metric_cache:
                compatible_path = _find_compatible_metric_cache(
                    metric_cache_dir, cache_path, cache_spec
                )
                if compatible_path is not None and compatible_path != cache_path:
                    print(
                        "[metric-cache] compatible hit: "
                        f"{compatible_path.name} (stable contract matched; "
                        "paths/mtimes ignored)",
                        flush=True,
                    )
                    cache_path = compatible_path
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
            cached_spec = json.loads(str(cache_payload["cache_spec"].item()))
            if not _metric_cache_specs_compatible(cached_spec, cache_spec):
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

    curvature_edge_one_form = None
    if return_curvature_edge_one_form:
        if topology != "toroidal":
            raise ValueError(
                "return_curvature_edge_one_form is currently supported only "
                "for topology='toroidal'"
            )
        if metric_evaluator is None:
            raise RuntimeError(
                "a MetricEvaluator is required for curvature edge sampling"
            )
        if bfield is None:
            print(
                "[geometry] loading MAKEGRID magnetic field for curvature edges",
                flush=True,
            )
            bfield = bfield_evaluator_from_makegrid(
                makegrid_path,
                currents=makegrid_current_array,
                method="cubic",
            )
        curvature_edge_one_form = _build_hsx_curvature_edge_one_form(
            metric_evaluator=metric_evaluator,
            bfield=bfield,
            u_faces=u_faces,
            v_faces=v_faces,
            eta0=float(metric_evaluator.eta[0]),
            eta_period=float(metric_evaluator.period),
            nfp=nfp,
            neta=neta,
            reference_magnetic_field=reference_magnetic_field,
        )
        print(
            "[geometry] sampled exact shared curvature edge one-form "
            f"(Az_xy={curvature_edge_one_form.Az_xy.shape}, "
            f"Ay_xz={curvature_edge_one_form.Ay_xz.shape}, "
            f"Ax_yz={curvature_edge_one_form.Ax_yz.shape})",
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
    # A caller that deliberately reuses a fitted ``metric_context`` has no
    # resolution-local metric payload to serialize.  It can still provide a
    # separate map-only cache: FCI maps depend on the PDE grid while the
    # continuous metric representation remains shared.  Keep this distinct
    # from the metric cache so its metadata never claims that the evaluator
    # was refit on the target resolution.
    map_cache_path = cache_path
    map_cache_payload = cache_payload
    if fci_map_cache_path is not None:
        map_cache_path = Path(fci_map_cache_path).resolve()
        map_cache_payload = {}
        if map_cache_path.is_file():
            try:
                with np.load(map_cache_path, allow_pickle=False) as cached:
                    map_cache_payload = {
                        name: np.array(cached[name], copy=True)
                        for name in cached.files
                    }
            except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
                print(
                    f"[fci-map-cache] ignored unreadable map cache "
                    f"({error}); regenerating",
                    flush=True,
                )
                map_cache_payload = {}
    maps, cache_payload, bfield = _build_or_load_hsx_fci_maps(
        grid=grid,
        topology=topology,
        construct_fci_maps=bool(construct_fci_maps),
        fci_trace_substeps=int(fci_trace_substeps),
        cache_payload=map_cache_payload,
        cache_path=map_cache_path,
        metric_evaluator=metric_evaluator,
        bfield=bfield,
        makegrid_path=makegrid_path,
        makegrid_currents=makegrid_current_array,
    )
    if maps is None:  # pragma: no cover - guarded by construct_fci_maps above
        raise RuntimeError("mapped HSX geometry construction returned no FCI maps")
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
    if return_metric_evaluator and return_curvature_edge_one_form:
        if metric_evaluator is None or curvature_edge_one_form is None:
            raise RuntimeError("requested curvature edge payload was not built")
        return (
            geometry,
            cell_positions,
            nfp,
            usable_cache_path,
            metric_evaluator,
            curvature_edge_one_form,
        )
    if return_metric_evaluator:
        if metric_evaluator is None:
            raise RuntimeError(
                "return_metric_evaluator=True requires a continuous evaluator"
            )
        return geometry, cell_positions, nfp, usable_cache_path, metric_evaluator
    if return_curvature_edge_one_form:
        return geometry, cell_positions, nfp, usable_cache_path, curvature_edge_one_form
    return geometry, cell_positions, nfp, usable_cache_path
