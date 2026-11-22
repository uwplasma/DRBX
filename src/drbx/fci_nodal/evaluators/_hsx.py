"""HSX metric-cache and fitting helpers from the shared source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

METRIC_CACHE_FORMAT_VERSION = 7

METRIC_BUILDER_REVISION = 1
METRIC_CACHE_IDENTITY_VERSION = 2
METRIC_INPUT_FULL_HASH_LIMIT = 64 * 2**20
METRIC_INPUT_SAMPLE_BYTES = 2**20
METRIC_INPUT_SAMPLE_COUNT = 16

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
    raise ValueError("Only toroidal topology is supported")

def _mode_cutoff(value: object) -> int:
    """Normalize scalar/array/tuple mode properties to max(abs(mode))."""

    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError("cached mode set is empty")
    return max(abs(int(mode)) for mode in values)

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
