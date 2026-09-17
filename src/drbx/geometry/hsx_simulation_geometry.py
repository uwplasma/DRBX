"""Independent HSX simulation-geometry producer.

This module is deliberately an orchestration layer.  The existing HSX metric
builder remains the numerical source of the global geometry and cell-centre
FCI maps; this producer adds the raw-vertex trace atlas and the frozen,
direct owner-boundary overlap graph needed by the simulation artifact.

The artifact module is intentionally imported lazily.  It is developed as a
separate, package-owned contract and this module can therefore be imported by
lightweight command-line/help tooling while that module is being installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .fci_geometry import (
    trace_fci_points_to_plane_from_callbacks,
)
from .fci_geometry import build_metric_aware_polar_angular_agglomeration_geometry
from .fci_owner_boundary_overlap import build_owner_boundary_overlap_geometry

TRACE_SUBSTEPS = 64
QUALIFICATION_SUBSTEPS = 128
DEFAULT_ENDPOINT_TOLERANCE_CELLS = 1.0e-3
OVERLAP_CLOSURE_TOLERANCE = 1.0e-10
PRODUCER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class HsxSimulationGeometryConfig:
    """Inputs for :func:`build_hsx_simulation_geometry`.

    The defaults match the established HSX driver defaults, but callers must
    always provide the computational ``resolution`` and input files.
    """

    makegrid_path: Path
    vessel_path: Path
    resolution: tuple[int, int, int]
    fit_sample_shape: tuple[int, int, int] = (8, 9, 8)
    radial_degree: int = 3
    vertical_degree: int = 3
    toroidal_modes: int = 2
    metric_spline_degree: int = 1
    mmpde_iterations: int = 0
    axis_core_radius: float = 0.03
    reference_magnetic_field: float | None = None
    makegrid_currents: Sequence[float] | None = None
    metric_mesh_shape: tuple[int, int, int] | None = None
    metric_radial_degree: int = 17
    metric_poloidal_modes: int = 15
    metric_toroidal_modes: int = 16
    eta_projection_iterations: int = 0
    output: Path | None = None
    metric_cache_dir: Path | None = None
    map_cache_path: Path | None = None
    rebuild_metric_cache: bool = False
    include_curvature_edge_one_form: bool = False
    trace_tolerance_cells: float = DEFAULT_ENDPOINT_TOLERANCE_CELLS

    def __post_init__(self) -> None:
        resolution = tuple(int(value) for value in self.resolution)
        if len(resolution) != 3 or any(value < 1 for value in resolution):
            raise ValueError("resolution must contain three positive integers")
        object.__setattr__(self, "resolution", resolution)
        if len(tuple(self.fit_sample_shape)) != 3:
            raise ValueError("fit_sample_shape must contain three integers")
        object.__setattr__(self, "fit_sample_shape", tuple(int(v) for v in self.fit_sample_shape))
        if resolution[0] < 3 or resolution[1] < 3 or resolution[2] < 4:
            raise ValueError("resolution must satisfy NU >= 3, NV >= 3, NETA >= 4")
        if resolution[1] % 2:
            raise ValueError("toroidal HSX resolution requires an even NTHETA")
        if not np.isfinite(self.trace_tolerance_cells) or self.trace_tolerance_cells <= 0.0:
            raise ValueError("trace_tolerance_cells must be positive and finite")


def _artifact_api() -> tuple[Any, Any, Callable[..., Any], Callable[..., Any]]:
    """Import the canonical artifact API at the point it is needed."""

    try:
        from .fci_simulation_geometry import (
            FciSimulationGeometry3D,
            FciVertexTraceAtlas,
            validate_fci_simulation_geometry,
            write_fci_simulation_geometry,
        )
    except ImportError as error:  # pragma: no cover - transitional checkout
        raise RuntimeError(
            "fci_simulation_geometry must provide FciSimulationGeometry3D, "
            "FciVertexTraceAtlas and the public validation/write functions"
        ) from error
    return (
        FciSimulationGeometry3D,
        FciVertexTraceAtlas,
        validate_fci_simulation_geometry,
        write_fci_simulation_geometry,
    )


def _producer_checkpoint_directory(output: Path) -> Path:
    return output.parent / f".{output.name}.producer-checkpoints"


def _status_path(output: Path, status_path: Path | None) -> Path:
    return (
        Path(status_path)
        if status_path is not None
        else _producer_checkpoint_directory(output) / "process_status.json"
    )


def _log_path(output: Path, log_path: Path | None) -> Path:
    return (
        Path(log_path)
        if log_path is not None
        else _producer_checkpoint_directory(output) / "run.log"
    )


def _record_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["updated_unix"] = time.time()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def _checkpoint_identity(config: HsxSimulationGeometryConfig) -> str:
    """Return the exact explicit producer-input identity for stage checkpoints."""

    payload = asdict(config)
    for name in ("makegrid_path", "vessel_path", "output", "metric_cache_dir", "map_cache_path"):
        value = payload.get(name)
        payload[name] = None if value is None else str(Path(value).expanduser().resolve())
    for name in ("makegrid_path", "vessel_path"):
        path = Path(payload[name])
        stat = path.stat()
        payload[f"{name}_stat"] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    payload["producer_format_version"] = PRODUCER_FORMAT_VERSION
    payload["trace_substeps"] = TRACE_SUBSTEPS
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _write_stage_checkpoint(
    path: Path, identity: str, payload: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(payload)
    arrays["producer_identity_json"] = np.asarray(identity)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_stage_checkpoint(
    path: Path, identity: str, loader: Callable[[Mapping[str, Any]], Any]
) -> Any | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            payload = {name: np.array(stored[name], copy=True) for name in stored.files}
        if str(payload.pop("producer_identity_json").item()) != identity:
            return None
        return loader(payload)
    except (EOFError, KeyError, OSError, TypeError, ValueError):
        return None


def _call_builder(
    config: HsxSimulationGeometryConfig,
) -> tuple[Any, np.ndarray, int, Path | None, Any, Any | None]:
    """Call the established HSX builder, fixing the producer trace policy."""

    # Geometry production is deliberately behind a package-owned boundary.
    # The simulation consumer never calls this function; it receives the
    # completed directory artifact produced here.
    from . import hsx_fci_builder as hsx

    map_cache_path = config.map_cache_path
    if map_cache_path is None and config.output is not None:
        map_cache_path = (
            _producer_checkpoint_directory(Path(config.output))
            / "cell_center_maps.npz"
        )
    kwargs = dict(
        makegrid_path=Path(config.makegrid_path),
        vessel_path=Path(config.vessel_path),
        resolution=config.resolution,
        fit_sample_shape=config.fit_sample_shape,
        radial_degree=int(config.radial_degree),
        vertical_degree=int(config.vertical_degree),
        toroidal_modes=int(config.toroidal_modes),
        metric_spline_degree=int(config.metric_spline_degree),
        mmpde_iterations=int(config.mmpde_iterations),
        axis_core_radius=float(config.axis_core_radius),
        reference_magnetic_field=config.reference_magnetic_field,
        makegrid_currents=config.makegrid_currents,
        topology="toroidal",
        metric_mesh_shape=config.metric_mesh_shape,
        metric_radial_degree=int(config.metric_radial_degree),
        metric_poloidal_modes=int(config.metric_poloidal_modes),
        metric_toroidal_modes=int(config.metric_toroidal_modes),
        eta_projection_iterations=int(config.eta_projection_iterations),
        construct_fci_maps=True,
        fci_trace_substeps=TRACE_SUBSTEPS,
        metric_cache_dir=config.metric_cache_dir,
        fci_map_cache_path=map_cache_path,
        rebuild_metric_cache=bool(config.rebuild_metric_cache),
        return_metric_evaluator=True,
        return_curvature_edge_one_form=bool(
            config.include_curvature_edge_one_form
        ),
    )
    result = hsx.build_hsx_fci_geometry(**kwargs)
    expected_length = 6 if config.include_curvature_edge_one_form else 5
    if len(result) != expected_length:
        raise ValueError(
            "HSX geometry builder returned an unexpected producer payload"
        )
    geometry, positions, nfp, metric_cache_path, metric_evaluator, *optional = result
    curvature_edge_one_form = optional[0] if optional else None
    if tuple(int(v) for v in geometry.shape) != config.resolution:
        raise ValueError(f"builder returned shape {geometry.shape}, expected {config.resolution}")
    return (
        geometry,
        np.asarray(positions),
        int(nfp),
        metric_cache_path,
        metric_evaluator,
        curvature_edge_one_form,
    )


def _continuous_field_callback(geometry: Any, metric_evaluator: Any, config: HsxSimulationGeometryConfig) -> Callable[[np.ndarray], Any]:
    """Return the continuous MAKEGRID/metric field callback used by tracing."""

    del geometry
    from .Bfield_evaluator import bfield_evaluator_from_makegrid

    if not hasattr(metric_evaluator, "evaluate_magnetic_field"):
        raise TypeError(
            "the HSX metric evaluator must provide evaluate_magnetic_field"
        )
    bfield = bfield_evaluator_from_makegrid(
        Path(config.makegrid_path),
        currents=config.makegrid_currents,
        method="cubic",
    )

    def evaluate(points: np.ndarray) -> Any:
        return metric_evaluator.evaluate_magnetic_field(points, bfield)

    return evaluate


def _trace_atlas(geometry: Any, field: Callable[[np.ndarray], Any], *, substeps: int = TRACE_SUBSTEPS, progress: Callable[[Mapping[str, Any]], None] | None = None) -> Any:
    """Trace the complete canonical raw transverse vertex lattice."""

    shape = tuple(int(v) for v in geometry.shape)
    nx, ny, nz = shape
    _, atlas_cls, _, _ = _artifact_api()
    x_faces = np.asarray(geometry.grid.x.faces, dtype=float)
    y_faces = np.asarray(geometry.grid.y.faces, dtype=float)
    z_centers = np.asarray(geometry.grid.z.centers, dtype=float)
    z_faces = np.asarray(geometry.grid.z.faces, dtype=float)
    eta_period = float(z_faces[-1] - z_faces[0])
    vertex_shape = (nx + 1, ny, nz)
    source = np.stack(np.meshgrid(x_faces, y_faces[:-1], z_centers, indexing="ij"), axis=-1)
    source = np.asarray(source, dtype=float)
    forward_endpoint = np.empty(vertex_shape + (3,), dtype=float)
    backward_endpoint = np.empty_like(forward_endpoint)
    forward_length = np.empty(vertex_shape, dtype=float)
    backward_length = np.empty_like(forward_length)
    forward_boundary = np.empty(vertex_shape, dtype=bool)
    backward_boundary = np.empty_like(forward_boundary)
    for direction, endpoint, lengths, boundaries in (
        (1, forward_endpoint, forward_length, forward_boundary),
        (-1, backward_endpoint, backward_length, backward_boundary),
    ):
        for k in range(nz):
            target = (k + direction) % nz
            step = float(z_centers[target] - z_centers[k])
            if direction > 0 and target == 0:
                step += eta_period
            if direction < 0 and target == nz - 1:
                step -= eta_period
            seeds = source[:, :, k, :].reshape(-1, 3)
            result = trace_fci_points_to_plane_from_callbacks(
                geometry.grid,
                field,
                seeds,
                step,
                substeps=int(substeps),
                periodic_axes=(False, True, True),
                axis_regular_axes=(True, False, False),
            )
            endpoint[:, :, k, :] = np.asarray(result["endpoint"]).reshape(nx + 1, ny, 3)
            lengths[:, :, k] = np.asarray(result["length"]).reshape(nx + 1, ny)
            boundaries[:, :, k] = np.asarray(result["boundary"]).reshape(nx + 1, ny)
            if progress is not None:
                progress({"stage": "vertex-trace", "direction": "forward" if direction > 0 else "backward", "completed_planes": k + 1, "total_planes": nz})
    metadata = {
        "format_version": PRODUCER_FORMAT_VERSION,
        "trace_substeps": int(substeps),
        "vertex_shape": list(vertex_shape),
        "canonical_lattice": "raw_transverse_vertices_u_faces_theta_vertices_eta_centers",
        "periodic_axes": [False, True, True],
        "axis_regular_axes": [True, False, False],
    }
    return atlas_cls(
        vertex_positions=source,
        forward_endpoint=forward_endpoint,
        backward_endpoint=backward_endpoint,
        forward_length=forward_length,
        backward_length=backward_length,
        forward_boundary=forward_boundary,
        backward_boundary=backward_boundary,
        metadata=metadata,
    )


def _trace_qualification(
    geometry: Any,
    field: Callable[[np.ndarray], Any],
    atlas: Any,
    *,
    tolerance_cells: float = DEFAULT_ENDPOINT_TOLERANCE_CELLS,
) -> Any:
    """Qualify 64-step atlas endpoints against deterministic 128-step traces.

    The comparison is made in transverse logical cell units: radial endpoint
    error is divided by the local radial cell width and angular error by the
    uniform angular cell width.  In addition to deterministic stratified
    samples, every path starting outside ``u=0.03`` whose 64-step endpoint
    enters that core is included, since those trajectories are the most
    sensitive to the axis-regular branch.
    """

    if tolerance_cells <= 0.0 or not np.isfinite(tolerance_cells):
        raise ValueError("trace qualification tolerance must be positive and finite")
    nx, ny, nz = (int(v) for v in geometry.shape)
    source = np.asarray(atlas.vertex_positions, dtype=float)
    vertex_count = int(np.prod(source.shape[:-1]))
    source_flat = source.reshape(-1, 3)
    x_faces = np.asarray(geometry.grid.x.faces, dtype=float)
    z_centers = np.asarray(geometry.grid.z.centers, dtype=float)
    eta_period = float(np.asarray(geometry.grid.z.faces)[-1] - np.asarray(geometry.grid.z.faces)[0])
    angular_cell = 2.0 * np.pi / float(ny)
    radial_widths = np.diff(x_faces)
    samples: set[int] = set()
    # Stratification is stable across machines and independent of hash order:
    # one sample per flattened interval, capped at 64 points per direction.
    for value in np.linspace(0, vertex_count - 1, min(vertex_count, 64), dtype=np.int64):
        samples.add(int(value))
    selected_by_direction: dict[str, list[int]] = {}
    max_discrepancy = 0.0
    discrepancies: list[float] = []
    for direction, endpoints in (("forward", np.asarray(atlas.forward_endpoint)), ("backward", np.asarray(atlas.backward_endpoint))):
        entering_core = np.flatnonzero((source_flat[:, 0] > 0.03) & (endpoints.reshape(-1, 3)[:, 0] < 0.03))
        selected = sorted(samples | {int(value) for value in entering_core})
        selected_by_direction[direction] = selected
        endpoint_128 = np.empty((len(selected), 3), dtype=float)
        selected_position = {index: position for position, index in enumerate(selected)}
        for k in range(nz):
            records = [index for index in selected if index % nz == k]
            if not records:
                continue
            # Flattening is C-order for (u-face, theta-vertex, eta-plane).
            seeds = source_flat[records]
            target = (k + (1 if direction == "forward" else -1)) % nz
            step = float(z_centers[target] - z_centers[k])
            if direction == "forward" and target == 0:
                step += eta_period
            if direction == "backward" and target == nz - 1:
                step -= eta_period
            result = trace_fci_points_to_plane_from_callbacks(
                geometry.grid,
                field,
                seeds,
                step,
                substeps=QUALIFICATION_SUBSTEPS,
                periodic_axes=(False, True, True),
                axis_regular_axes=(True, False, False),
            )
            endpoint_128[[selected_position[index] for index in records]] = np.asarray(result["endpoint"])
        endpoint_64 = endpoints.reshape(-1, 3)[selected]
        delta_u = endpoint_128[:, 0] - endpoint_64[:, 0]
        delta_theta = (endpoint_128[:, 1] - endpoint_64[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
        radial_index = np.clip(np.searchsorted(x_faces, endpoint_64[:, 0], side="right") - 1, 0, nx - 1)
        discrepancy = np.sqrt((delta_u / radial_widths[radial_index]) ** 2 + (delta_theta / angular_cell) ** 2)
        discrepancies.extend(np.asarray(discrepancy, dtype=float).tolist())
    if discrepancies:
        max_discrepancy = float(np.max(discrepancies))
    qualification = {
        "reference_substeps": TRACE_SUBSTEPS,
        "comparison_substeps": QUALIFICATION_SUBSTEPS,
        "sample_count": int(sum(len(values) for values in selected_by_direction.values())),
        "stratified_sample_count_per_direction": int(len(samples)),
        "core_entry_threshold_u": 0.03,
        "core_entry_paths": {direction: int(len(set(values) - samples)) for direction, values in selected_by_direction.items()},
        "max_endpoint_discrepancy_transverse_cells": max_discrepancy,
        "tolerance_transverse_cells": float(tolerance_cells),
        "passed": bool(max_discrepancy <= tolerance_cells),
    }
    if max_discrepancy > tolerance_cells:
        raise ValueError(
            "HSX FCI trace qualification failed: max endpoint discrepancy "
            f"{max_discrepancy:.6e} transverse cells exceeds "
            f"{tolerance_cells:.6e}"
        )
    return replace(atlas, metadata={**dict(getattr(atlas, "metadata", {}) or {}), "qualification": qualification})


def _write_artifact(
    path: Path,
    artifact: Any,
    identity: Mapping[str, Any],
    *,
    closure_tolerance: float,
) -> Any:
    _, _, validator, writer = _artifact_api()
    # Qualification is performed on the in-memory object.  The portable
    # artifact validator intentionally does not load paths (loading is a
    # consumer concern), while custom artifact APIs may return a path here.
    report = validator(
        artifact,
        require_vertex_traces=True,
        require_owner_overlap=True,
        closure_tolerance=closure_tolerance,
        require_trace_qualification=True,
    )
    try:
        qualified = replace(artifact, producer_validation_report=report)
    except TypeError:  # lightweight focused-test fixture
        qualified = artifact
        setattr(qualified, "producer_validation_report", report)
    writer(qualified, path)
    return qualified


def build_hsx_simulation_geometry(
    config: HsxSimulationGeometryConfig,
    *,
    status_path: Path | None = None,
    log_path: Path | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    """Build and write the independent HSX simulation geometry artifact."""

    _artifact_api()
    if config.output is None:
        raise ValueError("config.output is required")
    output = Path(config.output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output must be a directory artifact path, got file: {output}")
    status = _status_path(output, status_path)
    log = _log_path(output, log_path)
    checkpoint_directory = _producer_checkpoint_directory(output)
    checkpoint_identity = _checkpoint_identity(config)
    identity = {
        "producer": "hsx_simulation_geometry",
        "format_version": PRODUCER_FORMAT_VERSION,
        "resolution": list(config.resolution),
        "trace_substeps": TRACE_SUBSTEPS,
        "trace_tolerance_transverse_cells": float(config.trace_tolerance_cells),
        "vertex_trace_mode": "complete_canonical_transverse_raw_vertices",
        "owner_boundary_mode": "lean_second_order",
    }
    state = {"state": "running", "stage": "start", "output": str(output), "identity": identity, "config": asdict(config)}
    state["config"]["makegrid_path"] = str(config.makegrid_path)
    state["config"]["vessel_path"] = str(config.vessel_path)
    state["config"]["output"] = str(output)
    _record_status(status, state)
    started = time.monotonic()
    try:
        _log(log, f"start resolution={config.resolution}")
        state["stage"] = "metric-and-cell-maps"
        _record_status(status, state)
        (
            geometry,
            positions,
            nfp,
            metric_cache_path,
            metric_evaluator,
            curvature_edge_one_form,
        ) = _call_builder(config)
        field = _continuous_field_callback(geometry, metric_evaluator, config)
        state["stage"] = "vertex-trace-atlas"
        _record_status(status, state)
        def trace_progress(event: Mapping[str, Any]) -> None:
            state["progress"] = dict(event)
            _record_status(status, state)
            if progress_callback is not None:
                progress_callback(event)

        from . import fci_simulation_geometry as artifact_io

        atlas_checkpoint = checkpoint_directory / "vertex_traces.npz"
        atlas = _read_stage_checkpoint(
            atlas_checkpoint,
            checkpoint_identity,
            artifact_io._vertex_from_payload,
        )
        if atlas is None:
            atlas = _trace_atlas(geometry, field, progress=trace_progress)
            atlas = _trace_qualification(
                geometry,
                field,
                atlas,
                tolerance_cells=config.trace_tolerance_cells,
            )
            _write_stage_checkpoint(
                atlas_checkpoint,
                checkpoint_identity,
                artifact_io._vertex_payload(atlas),
            )
        else:
            _log(log, f"reused stage checkpoint {atlas_checkpoint}")
        state["qualification"] = dict(getattr(atlas, "metadata", {}).get("qualification", {}))
        _record_status(status, state)
        state["stage"] = "angular-owner-geometry"
        _record_status(status, state)
        topology_checkpoint = checkpoint_directory / "rlp_topology.npz"
        owner_host = _read_stage_checkpoint(
            topology_checkpoint,
            checkpoint_identity,
            artifact_io._polar_from_payload,
        )
        if owner_host is None:
            owner_host, safety_ratio = build_metric_aware_polar_angular_agglomeration_geometry(
                geometry, metric_evaluator, metric_cache_path=metric_cache_path
            )
            _write_stage_checkpoint(
                topology_checkpoint,
                checkpoint_identity,
                artifact_io._polar_payload(owner_host),
            )
        else:
            safety_ratio = float("nan")
            _log(log, f"reused stage checkpoint {topology_checkpoint}")
        state["stage"] = "owner-boundary-overlap"
        _record_status(status, state)
        closure_tolerance = OVERLAP_CLOSURE_TOLERANCE

        def overlap_metric(points: np.ndarray, **_: Any) -> np.ndarray:
            metric = metric_evaluator.evaluate(points, reject_nonpositive_J=False)
            magnetic = field(points)
            jacobian = np.asarray(metric.J, dtype=float)
            b_contra = np.asarray(
                magnetic["B_contravariant"]
                if isinstance(magnetic, Mapping)
                else magnetic.B_contravariant,
                dtype=float,
            )
            b_magnitude = np.maximum(
                np.asarray(
                    magnetic["magnitude"]
                    if isinstance(magnetic, Mapping)
                    else magnetic.magnitude,
                    dtype=float,
                ),
                1.0e-30,
            )
            return jacobian * np.abs(b_contra[..., 2]) / b_magnitude

        overlap_checkpoint = checkpoint_directory / "owner_overlap.npz"
        owner_boundary = _read_stage_checkpoint(
            overlap_checkpoint,
            checkpoint_identity,
            artifact_io._overlap_from_payload,
        )
        if owner_boundary is None:
            def overlap_progress(event: Mapping[str, Any]) -> None:
                state["progress"] = {"stage": "owner-boundary-overlap", **dict(event)}
                _record_status(status, state)
                if progress_callback is not None:
                    progress_callback(state["progress"])

            owner_boundary = build_owner_boundary_overlap_geometry(
                geometry,
                owner_host,
                atlas,
                cell_center_wall_masks=(
                    np.asarray(geometry.maps.forward_boundary, dtype=bool),
                    np.asarray(geometry.maps.backward_boundary, dtype=bool),
                ),
                metric_callback=overlap_metric,
                coverage_tolerance=closure_tolerance,
                progress_callback=overlap_progress,
                metadata={"producer": "hsx_simulation_geometry", "trace_substeps": TRACE_SUBSTEPS},
            )
            _write_stage_checkpoint(
                overlap_checkpoint,
                checkpoint_identity,
                artifact_io._overlap_payload(owner_boundary),
            )
        else:
            _log(log, f"reused stage checkpoint {overlap_checkpoint}")
        simulation_cls, _, _, _ = _artifact_api()
        state["stage"] = "artifact-write"
        _record_status(status, state)
        artifact = simulation_cls(
            geometry=geometry,
            cell_positions=positions,
            topology_name="toroidal",
            nfp=nfp,
            vertex_traces=atlas,
            polar_angular_geometry=owner_host,
            owner_overlap=owner_boundary,
            curvature_edge_one_form=curvature_edge_one_form,
            metadata={
                **identity,
                "makegrid_path": str(Path(config.makegrid_path).resolve()),
                "vessel_path": str(Path(config.vessel_path).resolve()),
                "makegrid_currents": (
                    None
                    if config.makegrid_currents is None
                    else [float(value) for value in config.makegrid_currents]
                ),
                "fit_sample_shape": list(config.fit_sample_shape),
                "radial_degree": int(config.radial_degree),
                "vertical_degree": int(config.vertical_degree),
                "toroidal_modes": int(config.toroidal_modes),
                "metric_spline_degree": int(config.metric_spline_degree),
                "mmpde_iterations": int(config.mmpde_iterations),
                "metric_mesh_shape": (
                    None
                    if config.metric_mesh_shape is None
                    else list(config.metric_mesh_shape)
                ),
                "metric_radial_degree": int(config.metric_radial_degree),
                "metric_poloidal_modes": int(config.metric_poloidal_modes),
                "metric_toroidal_modes": int(config.metric_toroidal_modes),
                "eta_projection_iterations": int(
                    config.eta_projection_iterations
                ),
                "axis_core_radius": float(config.axis_core_radius),
                "reference_magnetic_field": config.reference_magnetic_field,
                "include_curvature_edge_one_form": bool(
                    config.include_curvature_edge_one_form
                ),
                "producer_metric_checkpoint": (
                    None if metric_cache_path is None else str(metric_cache_path)
                ),
                "angular_profile_safety_ratio": (
                    None if not np.isfinite(safety_ratio) else float(safety_ratio)
                ),
                "producer_checkpoint_directory": str(checkpoint_directory),
            },
        )
        artifact = _write_artifact(
            output,
            artifact,
            identity,
            closure_tolerance=closure_tolerance,
        )
        state.update({"state": "completed", "stage": "done", "elapsed_seconds": time.monotonic() - started})
        _record_status(status, state)
        _log(log, f"completed output={output}")
        return artifact
    except Exception as error:
        state.update({"state": "failed", "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started})
        _record_status(status, state)
        _log(log, f"failed {type(error).__name__}: {error}")
        raise


def validate_hsx_simulation_geometry(geometry: Any) -> Mapping[str, Any]:
    """Apply the frozen HSX producer qualification contract."""

    _, _, validator, _ = _artifact_api()
    shape = tuple(int(value) for value in geometry.global_geometry.shape)
    closure_tolerance = OVERLAP_CLOSURE_TOLERANCE
    return validator(
        geometry,
        require_vertex_traces=True,
        require_owner_overlap=True,
        closure_tolerance=closure_tolerance,
        require_trace_qualification=True,
    )


__all__ = [
    "HsxSimulationGeometryConfig",
    "TRACE_SUBSTEPS",
    "OVERLAP_CLOSURE_TOLERANCE",
    "build_hsx_simulation_geometry",
    "validate_hsx_simulation_geometry",
]
