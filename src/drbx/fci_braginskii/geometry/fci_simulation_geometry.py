"""Consumer for FCI simulation geometry artifacts (schema v1).

Only the components the native operators consume are read: base geometry,
center maps, RLP topology and cell positions.  Other artifact components are
ignored, and loading never generates, qualifies or searches for geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fci_control_volumes import (
    GlobalControlVolumeTopology3D,
    PolarAngularAgglomerationGeometry3D,
)
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
)

SCHEMA_NAME = "drbx.fci_simulation_geometry"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FciSimulationGeometry3D:
    geometry: FciGeometry3D
    cell_positions: np.ndarray
    topology_name: str
    nfp: int
    polar_angular_geometry: PolarAngularAgglomerationGeometry3D | None
    metadata: Mapping[str, Any]


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _require(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"component is missing array {name!r}")
    return data[name]


def _json(data: Mapping[str, np.ndarray], name: str) -> Any:
    return json.loads(str(_require(data, name).item()))


def _metric(data: Mapping[str, np.ndarray], prefix: str) -> MetricGeometry:
    return MetricGeometry(
        **{f.name: _require(data, f"{prefix}.{f.name}") for f in fields(MetricGeometry)}
    )


def _bfield(data: Mapping[str, np.ndarray], prefix: str) -> BFieldGeometry:
    return BFieldGeometry(
        _require(data, f"{prefix}.B_contra"), _require(data, f"{prefix}.Bmag")
    )


def _geometry(
    base: Mapping[str, np.ndarray], maps: Mapping[str, np.ndarray]
) -> FciGeometry3D:
    return FciGeometry3D(
        grid=CellCenteredGrid3D(
            *(
                Grid1D(_require(base, f"grid.{axis}.centers"), _require(base, f"grid.{axis}.faces"))
                for axis in "xyz"
            )
        ),
        maps=FciMaps3D(**{f.name: _require(maps, f.name) for f in fields(FciMaps3D)}),
        spacing=Spacing3D(**{name: _require(base, f"spacing.{name}") for name in ("dx", "dy", "dz")}),
        cell_metric=_metric(base, "cell_metric"),
        face_metric=FaceMetricGeometry(*(_metric(base, f"face_metric.{axis}") for axis in "xyz")),
        cell_bfield=_bfield(base, "cell_bfield"),
        face_bfield=FaceBFieldGeometry(*(_bfield(base, f"face_bfield.{axis}") for axis in "xyz")),
    )


def _polar_angular_geometry(
    data: Mapping[str, np.ndarray],
) -> PolarAngularAgglomerationGeometry3D:
    topology_arrays = {f.name for f in fields(GlobalControlVolumeTopology3D)} - {"shape"}
    topology = GlobalControlVolumeTopology3D(
        shape=tuple(_json(data, "shape_json")),
        **{name: _require(data, name) for name in topology_arrays},
    )
    scalars = _json(data, "scalars_json")
    arrays = {f.name for f in fields(PolarAngularAgglomerationGeometry3D)} - {"topology", *scalars}
    return PolarAngularAgglomerationGeometry3D(
        topology=topology, **scalars, **{name: _require(data, name) for name in arrays}
    )


def load_fci_simulation_geometry(path: str | Path) -> FciSimulationGeometry3D:
    """Deserialize an artifact without producer qualification or cache lookup."""

    source = Path(path)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FCI simulation geometry manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema"), manifest.get("schema_version")) != (SCHEMA_NAME, SCHEMA_VERSION):
        raise ValueError("unsupported FCI simulation geometry artifact schema")
    components = manifest.get("components", {})

    def component(name: str) -> dict[str, np.ndarray]:
        if name not in components:
            raise ValueError(f"artifact is missing component {name!r}")
        return _read_npz(source / components[name])

    geometry = _geometry(component("base_geometry"), component("center_maps"))
    polar = (
        _polar_angular_geometry(component("rlp_topology"))
        if "rlp_topology" in components
        else None
    )
    positions = _require(component("cell_positions"), "cell_positions")
    nfp = int(manifest["nfp"])
    if positions.shape != geometry.shape + (3,):
        raise ValueError(f"cell_positions must have shape {geometry.shape + (3,)}")
    if polar is not None and tuple(polar.topology.shape) != geometry.shape:
        raise ValueError("RLP topology shape must match the geometry shape")
    if nfp < 1:
        raise ValueError("nfp must be positive")
    return FciSimulationGeometry3D(
        geometry=geometry,
        cell_positions=positions,
        topology_name=str(manifest["topology_name"]),
        nfp=nfp,
        polar_angular_geometry=polar,
        metadata=dict(manifest.get("metadata", {})),
    )
