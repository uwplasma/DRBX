"""Portable, producer-qualified FCI simulation geometry artifacts.

This module is intentionally a small boundary around the existing FCI geometry
dataclasses.  It does not alter the native operator payloads and, in
particular, does not make loading an artifact a geometry/physics qualification
step.  Qualification is explicit through
:func:`validate_fci_simulation_geometry_producer`.

Artifacts are directories rather than one opaque pickle/NPZ file.  The
manifest is human-readable and checksums are informational; each component
contains only ordinary NumPy arrays and JSON metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np

from .fci_geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    MetricGeometry,
    Spacing3D,
    CurvatureEdgeOneForm3D,
)
from .fci_control_volumes import (
    GlobalControlVolumeTopology3D,
    PolarAngularAgglomerationGeometry3D,
)
from .fci_rlp_overlap import GlobalRlpParallelOverlapGeometry


SCHEMA_NAME = "drbx.fci_simulation_geometry"
SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert common NumPy values to deterministic JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _put_json(payload: dict[str, np.ndarray], name: str, value: Any) -> None:
    payload[name] = np.asarray(json.dumps(_json_value(value), sort_keys=True))


def _get_json(data: Mapping[str, Any], name: str, default: Any) -> Any:
    if name not in data:
        return default
    raw = data[name]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return json.loads(str(raw))


def _write_npz(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_component(data: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"component is missing array {name!r}")
    return data[name]


@dataclass(frozen=True)
class FciVertexTraceAtlas:
    """Traces evaluated at a common atlas of logical cell vertices.

    The first dimensions are deliberately unconstrained at construction time.
    The qualified HSX producer retains the canonical periodic transverse atlas
    with shape ``(nx + 1, ny, nz)``.
    Positions and endpoints use the producer's logical ``(u, theta, eta)``
    coordinates; lengths are physical arclengths.  The endpoint magnetic-field
    arrays are optional and are retained when available for wall-law consumers.
    """

    vertex_positions: np.ndarray
    forward_endpoint: np.ndarray
    backward_endpoint: np.ndarray
    forward_length: np.ndarray
    backward_length: np.ndarray
    forward_boundary: np.ndarray
    backward_boundary: np.ndarray
    forward_endpoint_b_contra: np.ndarray | None = None
    forward_endpoint_bmag: np.ndarray | None = None
    backward_endpoint_b_contra: np.ndarray | None = None
    backward_endpoint_bmag: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.vertex_positions, dtype=np.float64)
        if positions.ndim < 2 or positions.shape[-1] != 3:
            raise ValueError("vertex_positions must have shape (..., 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("vertex_positions must be finite")
        object.__setattr__(self, "vertex_positions", np.array(positions, copy=True))
        vertex_shape = positions.shape[:-1]
        for name in ("forward_endpoint", "backward_endpoint"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != positions.shape:
                raise ValueError(f"{name} must have shape {positions.shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, np.array(value, copy=True))
        for name in ("forward_length", "backward_length"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != vertex_shape:
                raise ValueError(f"{name} must have shape {vertex_shape}, got {value.shape}")
            if not np.all(np.isfinite(value)) or np.any(value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, np.array(value, copy=True))
        for name in ("forward_boundary", "backward_boundary"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != vertex_shape:
                raise ValueError(f"{name} must have shape {vertex_shape}, got {value.shape}")
            object.__setattr__(self, name, np.array(value, copy=True))
        for name, shape in (
            ("forward_endpoint_b_contra", positions.shape),
            ("backward_endpoint_b_contra", positions.shape),
            ("forward_endpoint_bmag", vertex_shape),
            ("backward_endpoint_bmag", vertex_shape),
        ):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = np.asarray(raw, dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, np.array(value, copy=True))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.vertex_positions.shape[:-1])

    @property
    def positions(self) -> np.ndarray:
        return self.vertex_positions

    @property
    def vertex_x(self) -> np.ndarray:
        return self.vertex_positions[..., 0]

    @property
    def vertex_y(self) -> np.ndarray:
        return self.vertex_positions[..., 1]

    @property
    def vertex_z(self) -> np.ndarray:
        return self.vertex_positions[..., 2]

    @property
    def forward_endpoint_x(self) -> np.ndarray:
        return self.forward_endpoint[..., 0]

    @property
    def forward_endpoint_y(self) -> np.ndarray:
        return self.forward_endpoint[..., 1]

    @property
    def forward_endpoint_z(self) -> np.ndarray:
        return self.forward_endpoint[..., 2]

    @property
    def backward_endpoint_x(self) -> np.ndarray:
        return self.backward_endpoint[..., 0]

    @property
    def backward_endpoint_y(self) -> np.ndarray:
        return self.backward_endpoint[..., 1]

    @property
    def backward_endpoint_z(self) -> np.ndarray:
        return self.backward_endpoint[..., 2]

    @property
    def forward_endpoint_b_contra_x(self) -> np.ndarray | None:
        return None if self.forward_endpoint_b_contra is None else self.forward_endpoint_b_contra[..., 0]

    @property
    def forward_endpoint_b_contra_y(self) -> np.ndarray | None:
        return None if self.forward_endpoint_b_contra is None else self.forward_endpoint_b_contra[..., 1]

    @property
    def forward_endpoint_b_contra_z(self) -> np.ndarray | None:
        return None if self.forward_endpoint_b_contra is None else self.forward_endpoint_b_contra[..., 2]

    @property
    def backward_endpoint_b_contra_x(self) -> np.ndarray | None:
        return None if self.backward_endpoint_b_contra is None else self.backward_endpoint_b_contra[..., 0]

    @property
    def backward_endpoint_b_contra_y(self) -> np.ndarray | None:
        return None if self.backward_endpoint_b_contra is None else self.backward_endpoint_b_contra[..., 1]

    @property
    def backward_endpoint_b_contra_z(self) -> np.ndarray | None:
        return None if self.backward_endpoint_b_contra is None else self.backward_endpoint_b_contra[..., 2]

    @property
    def forward_vertices(self) -> np.ndarray:
        return self.forward_endpoint

    @property
    def backward_vertices(self) -> np.ndarray:
        return self.backward_endpoint


@dataclass(frozen=True)
class FciSimulationGeometry3D:
    """Complete host-side geometry payload needed by an FCI simulation.

    ``curvature_edge_one_form`` accepts the existing
    :class:`CurvatureEdgeOneForm3D` payload.  For small producer fixtures a
    finite ``geometry.shape + (3,)`` NumPy array is also accepted and stored
    losslessly.
    """

    geometry: FciGeometry3D
    cell_positions: np.ndarray
    topology_name: str
    nfp: int
    vertex_traces: FciVertexTraceAtlas | None = None
    polar_angular_geometry: PolarAngularAgglomerationGeometry3D | None = None
    owner_overlap: GlobalRlpParallelOverlapGeometry | None = None
    curvature_edge_one_form: CurvatureEdgeOneForm3D | np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None
    producer_validation_report: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FciGeometry3D):
            raise TypeError("geometry must be an FciGeometry3D")
        positions = np.asarray(self.cell_positions, dtype=np.float64)
        expected = self.geometry.shape + (3,)
        if positions.shape != expected:
            raise ValueError(f"cell_positions must have shape {expected}, got {positions.shape}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("cell_positions must be finite")
        object.__setattr__(self, "cell_positions", np.array(positions, copy=True))
        name = str(self.topology_name)
        if not name:
            raise ValueError("topology_name must be nonempty")
        object.__setattr__(self, "topology_name", name)
        if isinstance(self.nfp, (bool, np.bool_)) or int(self.nfp) != self.nfp or int(self.nfp) < 1:
            raise ValueError("nfp must be a positive integer")
        object.__setattr__(self, "nfp", int(self.nfp))
        if self.owner_overlap is not None and tuple(self.owner_overlap.raw_shape) != self.geometry.shape:
            raise ValueError("owner_overlap.raw_shape must match geometry.shape")
        if self.polar_angular_geometry is not None and tuple(self.polar_angular_geometry.topology.shape) != self.geometry.shape:
            raise ValueError("polar_angular_geometry topology shape must match geometry.shape")
        if self.curvature_edge_one_form is not None:
            if not isinstance(self.curvature_edge_one_form, CurvatureEdgeOneForm3D):
                curvature = np.asarray(self.curvature_edge_one_form, dtype=np.float64)
                if curvature.shape[:3] != self.geometry.shape or curvature.shape[-1:] != (3,):
                    raise ValueError("curvature_edge_one_form must have shape geometry.shape + (3,)")
                if not np.all(np.isfinite(curvature)):
                    raise ValueError("curvature_edge_one_form must be finite")
                object.__setattr__(self, "curvature_edge_one_form", np.array(curvature, copy=True))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "producer_validation_report", None if self.producer_validation_report is None else dict(self.producer_validation_report))

    @property
    def fci_geometry(self) -> FciGeometry3D:
        return self.geometry

    @property
    def global_geometry(self) -> FciGeometry3D:
        """Global host geometry consumed by device-layout lowering."""

        return self.geometry

    @property
    def fci(self) -> FciGeometry3D:
        return self.geometry

    @property
    def topology(self) -> str:
        return self.topology_name

    @property
    def polar_angular_agglomeration(self) -> PolarAngularAgglomerationGeometry3D | None:
        return self.polar_angular_geometry

    @property
    def owner_geometry(self) -> PolarAngularAgglomerationGeometry3D | None:
        return self.polar_angular_geometry

    @property
    def owner_overlap_geometry(self) -> GlobalRlpParallelOverlapGeometry | None:
        return self.owner_overlap

    def with_producer_validation_report(self, report: Mapping[str, Any]) -> "FciSimulationGeometry3D":
        return replace(self, producer_validation_report=dict(report))

    def write(self, path: str | Path) -> Path:
        return write_fci_simulation_geometry(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FciSimulationGeometry3D":
        return load_fci_simulation_geometry(path)


def _metric_payload(payload: dict[str, np.ndarray], prefix: str, metric: MetricGeometry) -> None:
    for name in ("J", "g11", "g22", "g33", "g12", "g13", "g23", "g_11", "g_22", "g_33", "g_12", "g_13", "g_23"):
        payload[f"{prefix}.{name}"] = np.asarray(getattr(metric, name))


def _metric_from_payload(data: Mapping[str, Any], prefix: str) -> MetricGeometry:
    names = ("J", "g11", "g22", "g33", "g12", "g13", "g23", "g_11", "g_22", "g_33", "g_12", "g_13", "g_23")
    return MetricGeometry(**{name: _require_component(data, f"{prefix}.{name}") for name in names})


def _base_geometry_payload(geometry: FciGeometry3D) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for axis_name, axis in (("x", geometry.grid.x), ("y", geometry.grid.y), ("z", geometry.grid.z)):
        payload[f"grid.{axis_name}.centers"] = np.asarray(axis.centers)
        payload[f"grid.{axis_name}.faces"] = np.asarray(axis.faces)
    for name in ("dx", "dy", "dz"):
        payload[f"spacing.{name}"] = np.asarray(getattr(geometry.spacing, name))
    _metric_payload(payload, "cell_metric", geometry.cell_metric)
    _metric_payload(payload, "face_metric.x", geometry.face_metric.x)
    _metric_payload(payload, "face_metric.y", geometry.face_metric.y)
    _metric_payload(payload, "face_metric.z", geometry.face_metric.z)
    for prefix, bfield in (("cell_bfield", geometry.cell_bfield), ("face_bfield.x", geometry.face_bfield.x), ("face_bfield.y", geometry.face_bfield.y), ("face_bfield.z", geometry.face_bfield.z)):
        payload[f"{prefix}.B_contra"] = np.asarray(bfield.B_contra)
        payload[f"{prefix}.Bmag"] = np.asarray(bfield.Bmag)
    return payload


def _center_maps_payload(geometry: FciGeometry3D) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(geometry.maps, name))
        for name in (
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_length", "backward_length", "forward_boundary", "backward_boundary",
        "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y", "forward_endpoint_b_contra_z", "forward_endpoint_bmag",
        "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y", "backward_endpoint_b_contra_z", "backward_endpoint_bmag",
        )
    }


def _geometry_from_payload(
    data: Mapping[str, Any], maps_data: Mapping[str, Any]
) -> FciGeometry3D:
    grid = CellCenteredGrid3D(*[Grid1D(_require_component(data, f"grid.{axis}.centers"), _require_component(data, f"grid.{axis}.faces")) for axis in ("x", "y", "z")])
    map_names = (
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_length", "backward_length", "forward_boundary", "backward_boundary",
        "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y", "forward_endpoint_b_contra_z", "forward_endpoint_bmag",
        "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y", "backward_endpoint_b_contra_z", "backward_endpoint_bmag",
    )
    maps = FciMaps3D(**{name: _require_component(maps_data, name) for name in map_names})
    spacing = Spacing3D(**{name: _require_component(data, f"spacing.{name}") for name in ("dx", "dy", "dz")})
    cell_metric = _metric_from_payload(data, "cell_metric")
    face_metric = FaceMetricGeometry(*[_metric_from_payload(data, f"face_metric.{axis}") for axis in ("x", "y", "z")])
    def bfield(prefix: str) -> BFieldGeometry:
        return BFieldGeometry(_require_component(data, f"{prefix}.B_contra"), _require_component(data, f"{prefix}.Bmag"))
    return FciGeometry3D(grid, maps, spacing, cell_metric, face_metric, bfield("cell_bfield"), FaceBFieldGeometry(*[bfield(f"face_bfield.{axis}") for axis in ("x", "y", "z")]))


def _vertex_payload(value: FciVertexTraceAtlas) -> dict[str, np.ndarray]:
    payload = {
        "vertex_positions": value.vertex_positions,
        "forward_endpoint": value.forward_endpoint,
        "backward_endpoint": value.backward_endpoint,
        "forward_length": value.forward_length,
        "backward_length": value.backward_length,
        "forward_boundary": value.forward_boundary,
        "backward_boundary": value.backward_boundary,
    }
    for name in ("forward_endpoint_b_contra", "forward_endpoint_bmag", "backward_endpoint_b_contra", "backward_endpoint_bmag"):
        if getattr(value, name) is not None:
            payload[name] = getattr(value, name)
    _put_json(payload, "metadata_json", value.metadata)
    return payload


def _vertex_from_payload(data: Mapping[str, Any]) -> FciVertexTraceAtlas:
    kwargs = {name: _require_component(data, name) for name in ("vertex_positions", "forward_endpoint", "backward_endpoint", "forward_length", "backward_length", "forward_boundary", "backward_boundary")}
    for name in ("forward_endpoint_b_contra", "forward_endpoint_bmag", "backward_endpoint_b_contra", "backward_endpoint_bmag"):
        kwargs[name] = data.get(name)
    kwargs["metadata"] = _get_json(data, "metadata_json", {})
    return FciVertexTraceAtlas(**kwargs)


def _topology_payload(payload: dict[str, np.ndarray], topology: GlobalControlVolumeTopology3D) -> None:
    _put_json(payload, "shape_json", topology.shape)
    for name in ("aggregate_id", "owner_index", "is_merge_source", "is_active_owner", "retained_cut_cell", "aggregate_volume", "aggregate_centroid", "aggregate_second_moment", "aggregate_third_moment", "face_id", "face_axis", "face_storage_index", "face_minus_aggregate_id", "face_plus_aggregate_id", "face_measure"):
        payload[name] = np.asarray(getattr(topology, name))


def _topology_from_payload(data: Mapping[str, Any]) -> GlobalControlVolumeTopology3D:
    names = ("aggregate_id", "owner_index", "is_merge_source", "is_active_owner", "retained_cut_cell", "aggregate_volume", "aggregate_centroid", "aggregate_second_moment", "aggregate_third_moment", "face_id", "face_axis", "face_storage_index", "face_minus_aggregate_id", "face_plus_aggregate_id", "face_measure")
    return GlobalControlVolumeTopology3D(shape=tuple(int(v) for v in _get_json(data, "shape_json", ())), **{name: _require_component(data, name) for name in names})


def _polar_payload(value: PolarAngularAgglomerationGeometry3D) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    _topology_payload(payload, value.topology)
    for name in ("angular_group_size", "radial_centers", "radial_widths", "raw_volume", "raw_chart_centroid", "raw_chart_second_moment", "raw_chart_third_moment", "raw_radial_centroid", "raw_radial_second_moment", "raw_radial_third_moment", "aggregate_chart_volume", "aggregate_chart_centroid", "aggregate_chart_second_moment", "aggregate_chart_third_moment"):
        payload[name] = np.asarray(getattr(value, name))
    _put_json(payload, "scalars_json", {"theta_period": value.theta_period, "eta_period": value.eta_period, "quadrature_order": value.quadrature_order})
    return payload


def _polar_from_payload(data: Mapping[str, Any]) -> PolarAngularAgglomerationGeometry3D:
    names = ("angular_group_size", "radial_centers", "radial_widths", "raw_volume", "raw_chart_centroid", "raw_chart_second_moment", "raw_chart_third_moment", "raw_radial_centroid", "raw_radial_second_moment", "raw_radial_third_moment", "aggregate_chart_volume", "aggregate_chart_centroid", "aggregate_chart_second_moment", "aggregate_chart_third_moment")
    scalars = _get_json(data, "scalars_json", {})
    return PolarAngularAgglomerationGeometry3D(topology=_topology_from_payload(data), **{name: _require_component(data, name) for name in names}, theta_period=float(scalars["theta_period"]), eta_period=float(scalars["eta_period"]), quadrature_order=int(scalars["quadrature_order"]))


def _overlap_payload(value: GlobalRlpParallelOverlapGeometry) -> dict[str, np.ndarray]:
    payload = {name: np.asarray(getattr(value, name)) for name in ("owner_flat_ids", "owner_volumes", "link_owner_a", "link_owner_b", "link_interface", "overlap_measure", "transmissibility")}
    payload["raw_shape"] = np.asarray(value.raw_shape, dtype=np.int32)
    _put_json(payload, "metadata_json", value.metadata)
    _put_json(payload, "diagnostics_json", value.diagnostics)
    return payload


def _overlap_from_payload(data: Mapping[str, Any]) -> GlobalRlpParallelOverlapGeometry:
    names = ("owner_flat_ids", "owner_volumes", "link_owner_a", "link_owner_b", "link_interface", "overlap_measure", "transmissibility")
    return GlobalRlpParallelOverlapGeometry(raw_shape=tuple(int(v) for v in _require_component(data, "raw_shape")), **{name: _require_component(data, name) for name in names}, metadata=_get_json(data, "metadata_json", {}), diagnostics=_get_json(data, "diagnostics_json", {}))


def _publish_directory(staging: Path, target: Path) -> None:
    if not target.exists():
        os.replace(staging, target)
        return
    backup = target.with_name(f".{target.name}.previous-{uuid.uuid4().hex}")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup)


def write_fci_simulation_geometry(
    artifact: FciSimulationGeometry3D, path: str | Path
) -> Path:
    """Write an artifact using staged component files and atomic publication."""

    if not isinstance(artifact, FciSimulationGeometry3D):
        raise TypeError("artifact must be an FciSimulationGeometry3D")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    components: dict[str, str] = {
        "base_geometry": "base_geometry.npz",
        "center_maps": "center_maps.npz",
    }
    try:
        _write_npz(staging / components["base_geometry"], _base_geometry_payload(artifact.geometry))
        _write_npz(staging / components["center_maps"], _center_maps_payload(artifact.geometry))
        if artifact.vertex_traces is not None:
            components["vertex_traces"] = "vertex_traces.npz"
            _write_npz(staging / components["vertex_traces"], _vertex_payload(artifact.vertex_traces))
        if artifact.polar_angular_geometry is not None:
            components["rlp_topology"] = "rlp_topology.npz"
            _write_npz(
                staging / components["rlp_topology"],
                _polar_payload(artifact.polar_angular_geometry),
            )
        if artifact.owner_overlap is not None:
            components["owner_overlap"] = "owner_overlap.npz"
            _write_npz(staging / components["owner_overlap"], _overlap_payload(artifact.owner_overlap))
        if artifact.curvature_edge_one_form is not None:
            components["curvature_edge_one_form"] = "curvature_edge_one_form.npz"
            if isinstance(artifact.curvature_edge_one_form, CurvatureEdgeOneForm3D):
                curvature_payload = {
                    "Az_xy": np.asarray(artifact.curvature_edge_one_form.Az_xy),
                    "Ay_xz": np.asarray(artifact.curvature_edge_one_form.Ay_xz),
                    "Ax_yz": np.asarray(artifact.curvature_edge_one_form.Ax_yz),
                }
            else:
                curvature_payload = {"curvature_edge_one_form": artifact.curvature_edge_one_form}
            _write_npz(staging / components["curvature_edge_one_form"], curvature_payload)
        components["cell_positions"] = "cell_positions.npz"
        _write_npz(staging / components["cell_positions"], {"cell_positions": artifact.cell_positions})
        components["producer_validation_report"] = "producer_validation_report.json"
        (staging / components["producer_validation_report"]).write_text(
            json.dumps(
                _json_value(artifact.producer_validation_report),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "topology_name": artifact.topology_name,
            "nfp": artifact.nfp,
            "shape": list(artifact.geometry.shape),
            "components": components,
            "metadata": _json_value(artifact.metadata),
            "checksums": {name: {"file": filename, "sha256": _sha256(staging / filename)} for name, filename in components.items()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish_directory(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def load_fci_simulation_geometry(path: str | Path) -> FciSimulationGeometry3D:
    """Deserialize an artifact without producer qualification or cache lookup."""

    source = Path(path)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FCI simulation geometry manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_NAME or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported FCI simulation geometry artifact schema")
    components = manifest.get("components", {})
    geometry = _geometry_from_payload(
        _read_npz(source / components["base_geometry"]),
        _read_npz(source / components["center_maps"]),
    )
    vertex = None if "vertex_traces" not in components else _vertex_from_payload(_read_npz(source / components["vertex_traces"]))
    polar_component = components.get(
        "rlp_topology", components.get("polar_angular_geometry")
    )
    polar = (
        None
        if polar_component is None
        else _polar_from_payload(_read_npz(source / polar_component))
    )
    overlap = None if "owner_overlap" not in components else _overlap_from_payload(_read_npz(source / components["owner_overlap"]))
    curvature = None
    if "curvature_edge_one_form" in components:
        curvature_data = _read_npz(source / components["curvature_edge_one_form"])
        if "curvature_edge_one_form" in curvature_data:
            curvature = curvature_data["curvature_edge_one_form"]
        elif all(name in curvature_data for name in ("Az_xy", "Ay_xz", "Ax_yz")):
            curvature = CurvatureEdgeOneForm3D(
                Az_xy=curvature_data["Az_xy"], Ay_xz=curvature_data["Ay_xz"], Ax_yz=curvature_data["Ax_yz"]
            )
        else:
            raise ValueError("curvature edge one-form component is incomplete")
    positions = _require_component(_read_npz(source / components["cell_positions"]), "cell_positions")
    report_component = components.get("producer_validation_report")
    producer_report = (
        manifest.get("producer_validation_report")
        if report_component is None
        else json.loads((source / report_component).read_text(encoding="utf-8"))
    )
    return FciSimulationGeometry3D(geometry=geometry, cell_positions=positions, topology_name=str(manifest["topology_name"]), nfp=int(manifest["nfp"]), vertex_traces=vertex, polar_angular_geometry=polar, owner_overlap=overlap, curvature_edge_one_form=curvature, metadata=manifest.get("metadata", {}), producer_validation_report=producer_report)


def audit_fci_simulation_geometry_checksums(path: str | Path) -> dict[str, Any]:
    """Audit bundle file integrity without selecting or loading geometry.

    This API is intentionally separate from deserialization.  A checksum
    mismatch is reported to the caller but can never trigger discovery,
    regeneration, or substitution of another artifact.
    """

    source = Path(path)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"FCI simulation geometry manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for name, expected in manifest.get("checksums", {}).items():
        filename = str(expected["file"])
        component_path = source / filename
        exists = component_path.is_file()
        actual = _sha256(component_path) if exists else None
        digest = str(expected["sha256"])
        records[str(name)] = {
            "file": filename,
            "exists": exists,
            "expected_sha256": digest,
            "actual_sha256": actual,
            "matches": bool(exists and actual == digest),
        }
    return {
        "valid": bool(records) and all(record["matches"] for record in records.values()),
        "components": records,
    }


def validate_fci_simulation_geometry_producer(
    artifact: FciSimulationGeometry3D,
    *,
    require_vertex_traces: bool = True,
    require_owner_overlap: bool = True,
    closure_tolerance: float = 1.0e-10,
    trace_qualification_report: Mapping[str, Any] | None = None,
    trace_qualification: Mapping[str, Any] | None = None,
    require_trace_qualification: bool = False,
) -> dict[str, Any]:
    """Run explicit producer-side qualification and return a JSON report.

    This is deliberately separate from deserialization.  A producer report is
    accepted only when the complete raw-vertex atlas uses the frozen 64
    substep policy and owner-overlap arrays are structurally sane.  Closure is
    compared with ``closure_tolerance`` and reported as a diagnostic-only
    check; exceeding it does not reject an otherwise valid artifact.  A trace
    qualification report can be
    required for campaigns that have performed the 64-versus-128 comparison;
    loading never performs any of these checks.
    """

    if not isinstance(artifact, FciSimulationGeometry3D):
        raise TypeError("artifact must be an FciSimulationGeometry3D")
    tolerance = float(closure_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("closure_tolerance must be finite and positive")
    closure_diagnostic_summary: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "geometry_shape": artifact.geometry.shape == artifact.cell_positions.shape[:3],
        "cell_positions_finite": bool(np.all(np.isfinite(artifact.cell_positions))),
        "positive_nfp": artifact.nfp >= 1,
        "vertex_traces_present": artifact.vertex_traces is not None or not require_vertex_traces,
        "owner_overlap_present": artifact.owner_overlap is not None or not require_owner_overlap,
        "rlp_topology_present": (
            artifact.polar_angular_geometry is not None
            or not require_owner_overlap
        ),
    }
    finite_geometry_arrays: list[np.ndarray] = []
    for axis in (artifact.geometry.grid.x, artifact.geometry.grid.y, artifact.geometry.grid.z):
        finite_geometry_arrays.extend((np.asarray(axis.centers), np.asarray(axis.faces)))
    for name in ("dx", "dy", "dz"):
        finite_geometry_arrays.append(np.asarray(getattr(artifact.geometry.spacing, name)))
    metric_names = (
        "J", "g11", "g22", "g33", "g12", "g13", "g23",
        "g_11", "g_22", "g_33", "g_12", "g_13", "g_23",
    )
    for metric in (
        artifact.geometry.cell_metric,
        artifact.geometry.face_metric.x,
        artifact.geometry.face_metric.y,
        artifact.geometry.face_metric.z,
    ):
        finite_geometry_arrays.extend(
            np.asarray(getattr(metric, name)) for name in metric_names
        )
    for magnetic in (
        artifact.geometry.cell_bfield,
        artifact.geometry.face_bfield.x,
        artifact.geometry.face_bfield.y,
        artifact.geometry.face_bfield.z,
    ):
        finite_geometry_arrays.extend(
            (np.asarray(magnetic.B_contra), np.asarray(magnetic.Bmag))
        )
    checks["base_geometry_finite"] = all(
        np.all(np.isfinite(value)) for value in finite_geometry_arrays
    )
    checks["base_geometry_positive_jacobian_and_bmag"] = bool(
        np.all(np.asarray(artifact.geometry.cell_metric.J) > 0.0)
        and np.all(np.asarray(artifact.geometry.cell_bfield.Bmag) > 0.0)
    )
    center_map_names = (
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_length", "backward_length",
    )
    checks["center_maps_complete_finite"] = all(
        np.asarray(getattr(artifact.geometry.maps, name)).shape
        == artifact.geometry.shape
        and np.all(np.isfinite(np.asarray(getattr(artifact.geometry.maps, name))))
        for name in center_map_names
    )
    endpoint_field_names = (
        "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y",
        "forward_endpoint_b_contra_z", "forward_endpoint_bmag",
        "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y",
        "backward_endpoint_b_contra_z", "backward_endpoint_bmag",
    )
    checks["center_endpoint_fields_complete_finite"] = all(
        np.asarray(getattr(artifact.geometry.maps, name)).shape
        == artifact.geometry.shape
        and np.all(np.isfinite(np.asarray(getattr(artifact.geometry.maps, name))))
        for name in endpoint_field_names
    )
    checks["center_map_lengths_positive"] = bool(
        np.all(
            (np.asarray(artifact.geometry.maps.forward_length) > 0.0)
            | np.asarray(artifact.geometry.maps.forward_boundary, dtype=bool)
        )
        and np.all(
            (np.asarray(artifact.geometry.maps.backward_length) > 0.0)
            | np.asarray(artifact.geometry.maps.backward_boundary, dtype=bool)
        )
    )

    atlas = artifact.vertex_traces
    if atlas is not None:
        expected_vertex_shape = (artifact.geometry.shape[0] + 1, artifact.geometry.shape[1], artifact.geometry.shape[2])
        checks["vertex_atlas_shape"] = atlas.shape == expected_vertex_shape
        raw_substeps = atlas.metadata.get("trace_substeps", artifact.metadata.get("trace_substeps"))
        checks["vertex_trace_substeps_64"] = (
            not isinstance(raw_substeps, (bool, np.bool_))
            and raw_substeps is not None
            and int(raw_substeps) == raw_substeps == 64
        )
    else:
        checks["vertex_atlas_shape"] = not require_vertex_traces
        checks["vertex_trace_substeps_64"] = not require_vertex_traces

    overlap = artifact.owner_overlap
    if overlap is not None:
        owner_ids = np.asarray(overlap.owner_flat_ids)
        volumes = np.asarray(overlap.owner_volumes, dtype=float)
        checks["owner_overlap_shape"] = tuple(overlap.raw_shape) == artifact.geometry.shape
        checks["owner_volumes_positive_finite"] = bool(volumes.size and np.all(np.isfinite(volumes)) and np.all(volumes > 0.0))
        checks["owner_ids_valid"] = bool(
            owner_ids.ndim == 1
            and owner_ids.size == volumes.size
            and np.issubdtype(owner_ids.dtype, np.integer)
            and np.all(owner_ids >= 0)
            and np.all(owner_ids < int(np.prod(artifact.geometry.shape)))
            and np.unique(owner_ids).size == owner_ids.size
        )
        overlap_arrays = {
            "overlap_measure": np.asarray(overlap.overlap_measure, dtype=float),
            "transmissibility": np.asarray(overlap.transmissibility, dtype=float),
        }
        checks["overlap_nonnegative_finite"] = all(
            bool(values.size == 0 or (np.all(np.isfinite(values)) and np.all(values >= 0.0)))
            for values in overlap_arrays.values()
        )
        link_arrays = [
            np.asarray(overlap.link_owner_a), np.asarray(overlap.link_owner_b),
            np.asarray(overlap.link_interface), overlap_arrays["overlap_measure"],
            overlap_arrays["transmissibility"],
        ]
        same_length = len({values.reshape(-1).size for values in link_arrays}) == 1
        if same_length:
            owner_count = owner_ids.reshape(-1).size
            a = link_arrays[0].reshape(-1)
            b = link_arrays[1].reshape(-1)
            interface = link_arrays[2].reshape(-1)
            checks["link_indices_valid"] = bool(
                np.issubdtype(a.dtype, np.integer)
                and np.issubdtype(b.dtype, np.integer)
                and np.issubdtype(interface.dtype, np.integer)
                and np.all(a >= 0) and np.all(a < owner_count)
                and np.all(b >= 0) and np.all(b < owner_count)
                and np.all(a != b) and np.all(interface >= 0)
            )
        else:
            checks["link_indices_valid"] = False

        closure_values: list[float] = []
        for name, value in overlap.diagnostics.items():
            if "closure" in str(name).lower() and "error" in str(name).lower():
                try:
                    closure_values.extend(np.asarray(value, dtype=float).reshape(-1).tolist())
                except (TypeError, ValueError):
                    checks["closure_diagnostics_within_tolerance"] = False
        for record in overlap.diagnostics.get("interfaces", ()):
            if isinstance(record, Mapping):
                for name in ("max_closure_error", "volume_weighted_closure_error"):
                    if name in record:
                        try:
                            closure_values.append(float(record[name]))
                        except (TypeError, ValueError):
                            checks["closure_diagnostics_within_tolerance"] = False
        checks["closure_diagnostics_present"] = bool(closure_values)
        finite_closure_values = np.asarray(closure_values, dtype=float)
        reported_closure_status = overlap.diagnostics.get(
            "closure_within_reference_tolerance"
        )
        if isinstance(reported_closure_status, (bool, np.bool_)):
            checks["closure_diagnostics_within_tolerance"] = bool(
                reported_closure_status
                and closure_values
                and np.all(np.isfinite(finite_closure_values))
            )
        else:
            checks["closure_diagnostics_within_tolerance"] = bool(
                closure_values
                and np.all(np.isfinite(finite_closure_values))
                and np.all(np.abs(finite_closure_values) <= tolerance)
            )
        closure_diagnostic_summary = {
            "reference_tolerance": float(
                overlap.diagnostics.get("closure_reference_tolerance", tolerance)
            ),
            "within_reference_tolerance": checks[
                "closure_diagnostics_within_tolerance"
            ],
            "max_closure_error": float(
                overlap.diagnostics.get(
                    "max_closure_error",
                    np.max(np.abs(finite_closure_values))
                    if finite_closure_values.size
                    else 0.0,
                )
            ),
            "max_relative_closure_error": overlap.diagnostics.get(
                "max_relative_closure_error"
            ),
            "reference_exceedance_count": overlap.diagnostics.get(
                "closure_reference_exceedance_count"
            ),
        }
    else:
        for name in (
            "owner_overlap_shape", "owner_volumes_positive_finite", "owner_ids_valid",
            "overlap_nonnegative_finite", "link_indices_valid", "closure_diagnostics_present",
            "closure_diagnostics_within_tolerance",
        ):
            checks[name] = not require_owner_overlap

    if artifact.polar_angular_geometry is not None:
        polar = artifact.polar_angular_geometry
        checks["polar_topology_shape"] = tuple(polar.topology.shape) == artifact.geometry.shape
        active = np.asarray(polar.topology.is_active_owner, dtype=bool)
        polar_volumes = np.asarray(polar.aggregate_chart_volume, dtype=float)
        checks["polar_owner_volumes_positive_finite"] = bool(
            polar_volumes.shape == artifact.geometry.shape
            and np.all(np.isfinite(polar_volumes[active]))
            and np.all(polar_volumes[active] > 0.0)
        )
        raw_volumes = np.asarray(polar.raw_volume, dtype=float)
        checks["raw_physical_volumes_positive_finite"] = bool(
            raw_volumes.shape == artifact.geometry.shape
            and np.all(np.isfinite(raw_volumes))
            and np.all(raw_volumes > 0.0)
        )
        if overlap is not None:
            aggregate = np.asarray(
                polar.aggregate_chart_volume, dtype=float
            ).reshape(-1)
            owner_ids = np.asarray(overlap.owner_flat_ids, dtype=int).reshape(-1)
            checks["owner_overlap_volumes_match_rlp"] = bool(
                np.allclose(
                    np.asarray(overlap.owner_volumes, dtype=float),
                    aggregate[owner_ids],
                    rtol=1.0e-12,
                    atol=0.0,
                )
            )
            active_ids = np.flatnonzero(active.reshape(-1))
            checks["owner_overlap_ids_match_rlp"] = bool(
                np.array_equal(owner_ids, active_ids)
            )
    elif require_owner_overlap:
        checks["raw_physical_volumes_positive_finite"] = False
        checks["owner_overlap_volumes_match_rlp"] = False
        checks["owner_overlap_ids_match_rlp"] = False

    qualification = trace_qualification_report if trace_qualification_report is not None else trace_qualification
    if qualification is None and atlas is not None:
        candidate = atlas.metadata.get("qualification")
        if isinstance(candidate, Mapping):
            qualification = candidate
    if trace_qualification_report is not None and trace_qualification is not None and trace_qualification_report != trace_qualification:
        raise ValueError("trace_qualification_report and trace_qualification disagree")
    if require_trace_qualification or qualification is not None:
        checks["trace_qualification_64_vs_128_passed"] = _trace_qualification_passed(qualification)

    diagnostic_only = ["closure_diagnostics_within_tolerance"]
    required = [
        name
        for name in checks
        if name
        not in {
            "vertex_traces_present",
            "owner_overlap_present",
            *diagnostic_only,
        }
    ]
    valid = all(checks[name] for name in required)
    report = {
        "schema": SCHEMA_NAME,
        "valid": bool(valid),
        "checks": checks,
        "required": required,
        "diagnostic_only": diagnostic_only,
        "closure_tolerance": tolerance,
        "closure_diagnostics": closure_diagnostic_summary,
    }
    if not valid:
        raise ValueError(
            "FCI simulation geometry producer validation failed: "
            + ", ".join(name for name in required if not checks[name])
        )
    return report


def _trace_qualification_passed(report: Mapping[str, Any] | None) -> bool:
    """Recognize the small, intentionally flexible 64-vs-128 report shape."""

    if report is None:
        return False
    try:
        text = json.dumps(_json_value(report), sort_keys=True).lower()
    except (TypeError, ValueError):
        return False
    if "64" not in text or "128" not in text or "failed" in text:
        return False

    def walk(value: Any) -> int:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if "pass" in str(key).lower() and isinstance(item, (bool, np.bool_)):
                    if bool(item) is False:
                        return -1
                    if bool(item) is True:
                        return 1
                status = walk(item)
                if status:
                    return status
        elif isinstance(value, (tuple, list)):
            for item in value:
                status = walk(item)
                if status:
                    return status
        return 0

    status = walk(report)
    return status > 0 or (status == 0 and "passed" in text and "not passed" not in text)


# Descriptive aliases used by callers that name the directory an artifact.
write_fci_simulation_geometry_artifact = write_fci_simulation_geometry
load_fci_simulation_geometry_artifact = load_fci_simulation_geometry
validate_fci_simulation_geometry = validate_fci_simulation_geometry_producer
# Short names keep the producer orchestration layer independent of the
# artifact module's longer descriptive names during this migration.
write = write_fci_simulation_geometry
load = load_fci_simulation_geometry
validate = validate_fci_simulation_geometry_producer


__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "FciVertexTraceAtlas", "FciSimulationGeometry3D",
    "write_fci_simulation_geometry", "load_fci_simulation_geometry",
    "audit_fci_simulation_geometry_checksums",
    "write_fci_simulation_geometry_artifact", "load_fci_simulation_geometry_artifact",
    "validate_fci_simulation_geometry_producer", "validate_fci_simulation_geometry",
    "write", "load", "validate",
]
