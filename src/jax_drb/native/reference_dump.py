from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class LocalReferenceSnapshot:
    mesh: StructuredMesh
    metrics: StructuredMetrics
    fields: dict[str, np.ndarray]
    optional_fields: dict[str, np.ndarray]
    scalar_values: dict[str, float]


def save_local_reference_snapshot_cache(
    snapshot: LocalReferenceSnapshot,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "mesh:nx": np.asarray(snapshot.mesh.nx, dtype=np.int64),
        "mesh:ny": np.asarray(snapshot.mesh.ny, dtype=np.int64),
        "mesh:nz": np.asarray(snapshot.mesh.nz, dtype=np.int64),
        "mesh:mxg": np.asarray(snapshot.mesh.mxg, dtype=np.int64),
        "mesh:myg": np.asarray(snapshot.mesh.myg, dtype=np.int64),
        "mesh:symmetric_global_x": np.asarray(int(snapshot.mesh.symmetric_global_x), dtype=np.int64),
        "mesh:symmetric_global_y": np.asarray(int(snapshot.mesh.symmetric_global_y), dtype=np.int64),
        "mesh:jyseps1_1": np.asarray(snapshot.mesh.jyseps1_1, dtype=np.int64),
        "mesh:jyseps2_1": np.asarray(snapshot.mesh.jyseps2_1, dtype=np.int64),
        "mesh:jyseps1_2": np.asarray(snapshot.mesh.jyseps1_2, dtype=np.int64),
        "mesh:jyseps2_2": np.asarray(snapshot.mesh.jyseps2_2, dtype=np.int64),
        "mesh:ny_inner": np.asarray(snapshot.mesh.ny_inner, dtype=np.int64),
        "mesh:has_lower_y_target": np.asarray(int(snapshot.mesh.has_lower_y_target), dtype=np.int64),
        "mesh:has_upper_y_target": np.asarray(int(snapshot.mesh.has_upper_y_target), dtype=np.int64),
        "mesh:x": np.asarray(snapshot.mesh.x, dtype=np.float64),
        "mesh:y": np.asarray(snapshot.mesh.y, dtype=np.float64),
        "mesh:z": np.asarray(snapshot.mesh.z, dtype=np.float64),
        "metrics:dx": np.asarray(snapshot.metrics.dx, dtype=np.float64),
        "metrics:dy": np.asarray(snapshot.metrics.dy, dtype=np.float64),
        "metrics:dz": np.asarray(snapshot.metrics.dz, dtype=np.float64),
        "metrics:J": np.asarray(snapshot.metrics.J, dtype=np.float64),
        "metrics:g11": np.asarray(snapshot.metrics.g11, dtype=np.float64),
        "metrics:g33": np.asarray(snapshot.metrics.g33, dtype=np.float64),
        "metrics:g22": np.asarray(snapshot.metrics.g22, dtype=np.float64),
        "metrics:g_22": np.asarray(snapshot.metrics.g_22, dtype=np.float64),
        "metrics:g23": np.asarray(snapshot.metrics.g23, dtype=np.float64),
        "metrics:Bxy": np.asarray(snapshot.metrics.Bxy, dtype=np.float64),
        "metrics:g_23": np.asarray(
            snapshot.metrics.g_23 if snapshot.metrics.g_23 is not None else np.zeros_like(snapshot.metrics.g23),
            dtype=np.float64,
        ),
    }
    for name, value in snapshot.fields.items():
        payload[f"field:{name}"] = np.asarray(value, dtype=np.float64)
    for name, value in snapshot.optional_fields.items():
        payload[f"optional:{name}"] = np.asarray(value, dtype=np.float64)
    for name, value in snapshot.scalar_values.items():
        payload[f"scalar:{name}"] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(target, **payload)


def load_local_reference_snapshot_cache(
    path: str | Path,
    *,
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...] = (),
    scalar_names: tuple[str, ...] = (),
) -> LocalReferenceSnapshot:
    with np.load(Path(path)) as dataset:
        mesh = StructuredMesh(
            nx=int(dataset["mesh:nx"]),
            ny=int(dataset["mesh:ny"]),
            nz=int(dataset["mesh:nz"]),
            mxg=int(dataset["mesh:mxg"]),
            myg=int(dataset["mesh:myg"]),
            symmetric_global_x=bool(int(dataset["mesh:symmetric_global_x"])),
            symmetric_global_y=bool(int(dataset["mesh:symmetric_global_y"])),
            jyseps1_1=int(dataset["mesh:jyseps1_1"]),
            jyseps2_1=int(dataset["mesh:jyseps2_1"]),
            jyseps1_2=int(dataset["mesh:jyseps1_2"]),
            jyseps2_2=int(dataset["mesh:jyseps2_2"]),
            ny_inner=int(dataset["mesh:ny_inner"]),
            has_lower_y_target=bool(int(dataset["mesh:has_lower_y_target"])),
            has_upper_y_target=bool(int(dataset["mesh:has_upper_y_target"])),
            x=jnp.asarray(dataset["mesh:x"], dtype=jnp.float64),
            y=jnp.asarray(dataset["mesh:y"], dtype=jnp.float64),
            z=jnp.asarray(dataset["mesh:z"], dtype=jnp.float64),
        )
        metrics = StructuredMetrics(
            dx=jnp.asarray(dataset["metrics:dx"], dtype=jnp.float64),
            dy=jnp.asarray(dataset["metrics:dy"], dtype=jnp.float64),
            dz=jnp.asarray(dataset["metrics:dz"], dtype=jnp.float64),
            J=jnp.asarray(dataset["metrics:J"], dtype=jnp.float64),
            g11=jnp.asarray(dataset["metrics:g11"], dtype=jnp.float64),
            g33=jnp.asarray(dataset["metrics:g33"], dtype=jnp.float64),
            g22=jnp.asarray(dataset["metrics:g22"], dtype=jnp.float64),
            g_22=jnp.asarray(dataset["metrics:g_22"], dtype=jnp.float64),
            g23=jnp.asarray(dataset["metrics:g23"], dtype=jnp.float64),
            Bxy=jnp.asarray(dataset["metrics:Bxy"], dtype=jnp.float64),
            g_23=jnp.asarray(
                dataset["metrics:g_23"] if "metrics:g_23" in dataset else np.zeros_like(dataset["metrics:g23"]),
                dtype=jnp.float64,
            ),
        )
        fields = {
            name: np.asarray(dataset[f"field:{name}"], dtype=np.float64)
            for name in field_names
            if f"field:{name}" in dataset
        }
        optional_fields = {
            name: np.asarray(dataset[f"optional:{name}"], dtype=np.float64)
            for name in optional_field_names
            if f"optional:{name}" in dataset
        }
        scalar_values = {
            name: float(np.asarray(dataset[f"scalar:{name}"], dtype=np.float64))
            for name in scalar_names
            if f"scalar:{name}" in dataset
        }
    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=fields,
        optional_fields=optional_fields,
        scalar_values=scalar_values,
    )


def save_optional_field_history_cache(
    history: Mapping[str, np.ndarray],
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{name: np.asarray(value, dtype=np.float64) for name, value in history.items()})


def load_optional_field_history_cache(
    path: str | Path,
    *,
    field_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as dataset:
        return {
            name: np.asarray(dataset[name], dtype=np.float64)
            for name in field_names
            if name in dataset
        }


def synthesize_local_reference_snapshot_from_active_history(
    *,
    initial_snapshot: LocalReferenceSnapshot,
    array_history_path: str | Path,
    timestep: float,
    state_field_names: tuple[str, ...],
    rhs_field_names: tuple[str, ...] = (),
    optional_history_path: str | Path | None = None,
    optional_field_names: tuple[str, ...] = (),
) -> LocalReferenceSnapshot:
    active = (
        slice(initial_snapshot.mesh.xstart, initial_snapshot.mesh.xend + 1),
        slice(initial_snapshot.mesh.ystart, initial_snapshot.mesh.yend + 1),
        slice(None),
    )
    with np.load(Path(array_history_path), allow_pickle=True) as dataset:
        fields: dict[str, np.ndarray] = {}
        for name in state_field_names:
            history_name = f"var__{name}"
            if history_name not in dataset or name not in initial_snapshot.fields:
                continue
            final_active = np.asarray(dataset[history_name][-1], dtype=np.float64)
            full = np.asarray(initial_snapshot.fields[name], dtype=np.float64, copy=True)
            full[active] = final_active
            fields[name] = full
            rhs_name = f"ddt({name})"
            if rhs_name in rhs_field_names:
                rhs = np.zeros_like(full, dtype=np.float64)
                rhs[active] = (
                    final_active - np.asarray(initial_snapshot.fields[name][active], dtype=np.float64)
                ) / float(timestep)
                fields[rhs_name] = rhs
        for rhs_name in rhs_field_names:
            if rhs_name not in fields:
                base_name = rhs_name.removeprefix("ddt(").removesuffix(")")
                template = fields.get(base_name, initial_snapshot.fields.get(base_name))
                if template is not None:
                    fields[rhs_name] = np.zeros_like(np.asarray(template, dtype=np.float64), dtype=np.float64)

    optional_fields: dict[str, np.ndarray] = {}
    if optional_history_path is not None and optional_field_names:
        history = load_optional_field_history_cache(optional_history_path, field_names=optional_field_names)
        template = next(iter(fields.values()), next(iter(initial_snapshot.fields.values())))
        for name in optional_field_names:
            if name in history:
                history_final = np.asarray(history[name][-1], dtype=np.float64)
                if history_final.shape == np.asarray(template).shape:
                    full = history_final.copy()
                else:
                    full = np.asarray(
                        initial_snapshot.optional_fields.get(name, np.zeros_like(template, dtype=np.float64)),
                        dtype=np.float64,
                        copy=True,
                    )
                    full[active] = history_final
            elif name in initial_snapshot.optional_fields:
                full = np.asarray(initial_snapshot.optional_fields[name], dtype=np.float64, copy=True)
            else:
                continue
            optional_fields[name] = full

    return LocalReferenceSnapshot(
        mesh=initial_snapshot.mesh,
        metrics=initial_snapshot.metrics,
        fields=fields,
        optional_fields=optional_fields,
        scalar_values={},
    )


def load_local_reference_snapshot(
    dump_path: str | Path,
    *,
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...] = (),
    scalar_names: tuple[str, ...] = (),
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
            has_lower_y_target=_read_scalar(dataset, "PE_YIND", default=0) == 0,
            has_upper_y_target=_read_scalar(dataset, "PE_YIND", default=0) == max(_read_scalar(dataset, "NYPE", default=1) - 1, 0),
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
            g_23=_read_metric(dataset, "g_23", shape=(nx, local_ny, nz), default=0.0),
        )

        fields = {
            name: _read_field(dataset, name, time_index=time_index, shape=(nx, local_ny, nz))
            for name in field_names
        }
        optional_fields = {
            name: value
            for name in optional_field_names
            if (value := _read_optional_field(dataset, name, time_index=time_index, shape=(nx, local_ny, nz))) is not None
        }
        scalar_values = {
            name: _read_float_scalar(dataset, name)
            for name in scalar_names
            if _has_variable(dataset, name)
        }

    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=fields,
        optional_fields=optional_fields,
        scalar_values=scalar_values,
    )


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
    elif array.ndim == 3 and array.shape[1:] == shape[:2]:
        array = array[time_index]
    return np.asarray(_coerce_field_shape(array, shape=shape), dtype=np.float64)


def _read_optional_field(
    dataset: object,
    name: str,
    *,
    time_index: int,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    if not _has_variable(dataset, name):
        return None
    return _read_field(dataset, name, time_index=time_index, shape=shape)


def _read_float_scalar(dataset: object, name: str) -> float:
    variable = getattr(dataset, "variables", {}).get(name)
    if variable is None:
        raise KeyError(f"Missing scalar {name!r} in local reference dump.")
    value = np.asarray(variable[:], dtype=np.float64)
    if value.shape == ():
        return float(value)
    return float(value.reshape(-1)[0])


def _has_variable(dataset: object, name: str) -> bool:
    return name in getattr(dataset, "variables", {})


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
