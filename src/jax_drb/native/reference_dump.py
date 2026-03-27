from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class LocalReferenceSnapshot:
    mesh: StructuredMesh
    metrics: StructuredMetrics
    fields: dict[str, np.ndarray]


def load_local_reference_snapshot(
    dump_path: str | Path,
    *,
    field_names: tuple[str, ...],
    time_index: int = 0,
) -> LocalReferenceSnapshot:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - exercised only when validation extra is missing
        raise ImportError("Local reference snapshot loading requires netCDF4.") from exc

    path = Path(dump_path)
    with Dataset(path) as dataset:
        nx = int(dataset.dimensions["x"].size)
        local_ny = int(dataset.dimensions["y"].size)
        nz = int(dataset.dimensions["z"].size)
        mxg = _read_scalar(dataset, "MXG", default=2)
        myg = _read_scalar(dataset, "MYG", default=2)
        ny = local_ny - 2 * myg

        mesh = StructuredMesh(
            nx=nx,
            ny=ny,
            nz=nz,
            mxg=mxg,
            myg=myg,
            symmetric_global_x=False,
            symmetric_global_y=False,
            jyseps1_1=_read_scalar(dataset, "jyseps1_1", default=-1),
            jyseps2_1=_read_scalar(dataset, "jyseps2_1", default=max(ny - 1, 0)),
            jyseps1_2=_read_scalar(dataset, "jyseps1_2", default=max(ny - 1, 0)),
            jyseps2_2=_read_scalar(dataset, "jyseps2_2", default=max(ny - 1, 0)),
            ny_inner=_read_scalar(dataset, "ny_inner", default=max(ny, 0)),
            x=jnp.arange(nx, dtype=jnp.float64),
            y=jnp.arange(local_ny, dtype=jnp.float64) - float(myg),
            z=jnp.arange(nz, dtype=jnp.float64) / float(max(nz, 1)),
        )

        metrics = StructuredMetrics(
            dx=_read_metric(dataset, "dx", shape=(nx, local_ny, nz), default=1.0),
            dy=_read_metric(dataset, "dy", shape=(nx, local_ny, nz), default=1.0),
            dz=_read_metric(dataset, "dz", shape=(nx, local_ny, nz), default=1.0),
            J=_read_metric(dataset, "J", shape=(nx, local_ny, nz), default=1.0),
            g11=_read_metric(dataset, "g11", shape=(nx, local_ny, nz), default=1.0),
            g33=_read_metric(dataset, "g33", shape=(nx, local_ny, nz), default=1.0),
            g22=_read_metric(dataset, "g22", shape=(nx, local_ny, nz), default=1.0),
            g_22=_read_metric(
                dataset,
                "g_22",
                shape=(nx, local_ny, nz),
                default=None,
                fallback=_reciprocal_metric(dataset, "g22", shape=(nx, local_ny, nz)),
            ),
            g23=_read_metric(dataset, "g23", shape=(nx, local_ny, nz), default=0.0),
            Bxy=_read_metric(dataset, "Bxy", shape=(nx, local_ny, nz), default=1.0),
        )

        fields = {
            name: _read_field(dataset, name, time_index=time_index, shape=(nx, local_ny, nz))
            for name in field_names
        }

    return LocalReferenceSnapshot(mesh=mesh, metrics=metrics, fields=fields)


def _read_scalar(dataset: object, name: str, *, default: int) -> int:
    variable = getattr(dataset, "variables", {}).get(name)
    if variable is None:
        return default
    value = np.asarray(variable[:], dtype=np.int64)
    if value.shape == ():
        return int(value)
    return int(value.reshape(-1)[0])


def _read_metric(
    dataset: object,
    name: str,
    *,
    shape: tuple[int, int, int],
    default: float | None,
    fallback: np.ndarray | None = None,
) -> jnp.ndarray:
    variable = getattr(dataset, "variables", {}).get(name)
    if variable is None:
        if fallback is not None:
            return jnp.asarray(fallback, dtype=jnp.float64)
        if default is None:
            raise KeyError(f"Missing required metric field {name!r} in local reference dump.")
        return jnp.full(shape, default, dtype=jnp.float64)
    array = np.asarray(variable[:], dtype=np.float64)
    return jnp.asarray(_coerce_field_shape(array, shape=shape), dtype=jnp.float64)


def _reciprocal_metric(
    dataset: object,
    name: str,
    *,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    variable = getattr(dataset, "variables", {}).get(name)
    if variable is None:
        return None
    array = np.asarray(variable[:], dtype=np.float64)
    coerced = _coerce_field_shape(array, shape=shape)
    return 1.0 / np.maximum(coerced, 1.0e-30)


def _read_field(
    dataset: object,
    name: str,
    *,
    time_index: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    variable = getattr(dataset, "variables", {}).get(name)
    if variable is None:
        raise KeyError(f"Missing field {name!r} in local reference dump.")
    array = np.asarray(variable[:], dtype=np.float64)
    if array.ndim == 4:
        array = array[time_index]
    return np.asarray(_coerce_field_shape(array, shape=shape), dtype=np.float64)


def _coerce_field_shape(array: np.ndarray, *, shape: tuple[int, int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    if array.ndim == 2 and array.shape == shape[:2]:
        return np.broadcast_to(array[..., None], shape)
    if array.ndim == 1 and array.shape[0] == shape[0]:
        return np.broadcast_to(array[:, None, None], shape)
    if array.ndim == 1 and array.shape[0] == shape[1]:
        return np.broadcast_to(array[None, :, None], shape)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float64)
    raise ValueError(f"Unsupported field shape {array.shape!r}; expected compatible with {shape!r}.")
