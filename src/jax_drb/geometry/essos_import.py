from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ESSOS_LANDREMAN_QA_RELATIVE_JSON = Path("examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json")
_PRIVATE_DEFAULT_ESSOS_ROOT = Path.home() / "local" / "ESSOS"


@dataclass(frozen=True)
class EssosFieldLineBundle:
    """Field and field-line arrays exported from an ESSOS tracing run.

    The bundle deliberately stores arrays, not ESSOS objects. This keeps
    `jax_drb` independent of ESSOS at runtime after the import/export step.
    """

    trajectories_xyz: np.ndarray
    times: np.ndarray
    initial_xyz: np.ndarray
    poincare_r: np.ndarray
    poincare_z: np.ndarray
    poincare_time: np.ndarray
    poincare_section: np.ndarray
    poincare_line_index: np.ndarray
    field_sample_xyz: np.ndarray
    field_sample_b_xyz: np.ndarray
    coil_gamma_xyz: np.ndarray
    coil_currents: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_field_lines(self) -> int:
        return int(self.trajectories_xyz.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.trajectories_xyz.shape[1])

    @property
    def poincare_point_count(self) -> int:
        return int(self.poincare_r.size)


def resolve_essos_landreman_qa_json(path: str | Path | None = None, *, essos_root: str | Path | None = None) -> Path:
    """Resolve the Landreman-Paul QA coil JSON from an ESSOS checkout."""

    if path is not None:
        resolved = Path(path)
    else:
        root = Path(essos_root) if essos_root is not None else Path(os.environ.get("JAX_DRB_ESSOS_ROOT", _PRIVATE_DEFAULT_ESSOS_ROOT))
        resolved = root / ESSOS_LANDREMAN_QA_RELATIVE_JSON
    if not resolved.exists():
        raise FileNotFoundError(
            "ESSOS Landreman-Paul QA coil JSON was not found. Pass coil_json_path "
            "or set JAX_DRB_ESSOS_ROOT to an ESSOS checkout containing "
            f"{ESSOS_LANDREMAN_QA_RELATIVE_JSON}."
        )
    return resolved


def essos_runtime_available(*, essos_root: str | Path | None = None) -> bool:
    """Return whether ESSOS can be imported by the optional adapter."""

    try:
        _import_essos_modules(essos_root=essos_root)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def trace_essos_coil_field_lines(
    *,
    coil_json_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    r_min: float = 1.21,
    r_max: float = 1.40,
    n_field_lines: int = 8,
    maxtime: float = 1000.0,
    times_to_trace: int = 6000,
    trace_tolerance: float = 1.0e-8,
    poincare_shifts: tuple[float, ...] = (0.0, float(np.pi / 2.0)),
    field_sample_count: int = 256,
) -> EssosFieldLineBundle:
    """Trace coil-produced field lines with ESSOS and export arrays.

    ESSOS owns the magnetic-field object, adaptive field-line integration, and
    Poincare root extraction. `jax_drb` only normalizes the resulting arrays
    into a stable import bundle.
    """

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_coil_json.parents[2])

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    coils = modules["Coils_from_json"](str(resolved_coil_json))
    field = modules["BiotSavart"](coils)

    r0 = jnp.linspace(float(r_min), float(r_max), int(n_field_lines))
    z0 = jnp.zeros(int(n_field_lines))
    phi0 = jnp.zeros(int(n_field_lines))
    initial_xyz = jnp.array([r0 * jnp.cos(phi0), r0 * jnp.sin(phi0), z0]).T

    tracing = jax.block_until_ready(
        modules["Tracing"](
            field=field,
            model="FieldLineAdaptative",
            initial_conditions=initial_xyz,
            maxtime=float(maxtime),
            times_to_trace=int(times_to_trace),
            atol=float(trace_tolerance),
            rtol=float(trace_tolerance),
        )
    )
    trajectories_xyz = np.asarray(tracing.trajectories[:, :, :3], dtype=np.float64)
    times = np.asarray(tracing.times, dtype=np.float64)
    initial_xyz_np = np.asarray(initial_xyz, dtype=np.float64)

    fig, axis = plt.subplots(figsize=(2.0, 2.0))
    plotting_data = tracing.poincare_plot(
        shifts=[jnp.asarray(value) for value in poincare_shifts],
        ax=axis,
        show=False,
        s=0.5,
    )
    plt.close(fig)
    poincare = _flatten_essos_poincare_data(
        plotting_data,
        n_field_lines=int(n_field_lines),
        shifts=tuple(float(value) for value in poincare_shifts),
    )

    flat_trajectory = trajectories_xyz.reshape((-1, 3))
    sample_count = min(int(field_sample_count), int(flat_trajectory.shape[0]))
    sample_indices = np.linspace(0, flat_trajectory.shape[0] - 1, sample_count, dtype=int)
    field_sample_xyz = flat_trajectory[sample_indices]
    field_sample_b_xyz = _sample_essos_field(field, field_sample_xyz)

    metadata = {
        "source": "ESSOS",
        "coil_json_file": resolved_coil_json.name,
        "field_model": "essos.fields.BiotSavart",
        "tracing_model": "essos.dynamics.Tracing(FieldLineAdaptative)",
        "poincare_method": "essos.dynamics.Tracing.poincare_plot",
        "n_field_lines": int(n_field_lines),
        "times_to_trace": int(times_to_trace),
        "maxtime": float(maxtime),
        "trace_tolerance": float(trace_tolerance),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "poincare_shifts": [float(value) for value in poincare_shifts],
    }
    return EssosFieldLineBundle(
        trajectories_xyz=trajectories_xyz,
        times=times,
        initial_xyz=initial_xyz_np,
        poincare_r=poincare["r"],
        poincare_z=poincare["z"],
        poincare_time=poincare["time"],
        poincare_section=poincare["section"],
        poincare_line_index=poincare["line_index"],
        field_sample_xyz=field_sample_xyz,
        field_sample_b_xyz=field_sample_b_xyz,
        coil_gamma_xyz=np.asarray(coils.gamma, dtype=np.float64),
        coil_currents=np.asarray(coils.currents, dtype=np.float64),
        metadata=metadata,
    )


def save_essos_field_line_bundle_npz(bundle: EssosFieldLineBundle, path: str | Path) -> Path:
    """Write a portable ESSOS field-line import bundle."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        resolved,
        trajectories_xyz=bundle.trajectories_xyz.astype(np.float32),
        times=bundle.times.astype(np.float64),
        initial_xyz=bundle.initial_xyz.astype(np.float32),
        poincare_r=bundle.poincare_r.astype(np.float32),
        poincare_z=bundle.poincare_z.astype(np.float32),
        poincare_time=bundle.poincare_time.astype(np.float32),
        poincare_section=bundle.poincare_section.astype(np.float32),
        poincare_line_index=bundle.poincare_line_index.astype(np.int32),
        field_sample_xyz=bundle.field_sample_xyz.astype(np.float32),
        field_sample_b_xyz=bundle.field_sample_b_xyz.astype(np.float32),
        coil_gamma_xyz=bundle.coil_gamma_xyz.astype(np.float32),
        coil_currents=bundle.coil_currents.astype(np.float64),
    )
    return resolved


def load_essos_field_line_bundle_npz(path: str | Path, *, metadata: dict[str, Any] | None = None) -> EssosFieldLineBundle:
    """Load an ESSOS field-line import bundle produced by `jax_drb`."""

    with np.load(Path(path)) as data:
        return EssosFieldLineBundle(
            trajectories_xyz=np.asarray(data["trajectories_xyz"], dtype=np.float64),
            times=np.asarray(data["times"], dtype=np.float64),
            initial_xyz=np.asarray(data["initial_xyz"], dtype=np.float64),
            poincare_r=np.asarray(data["poincare_r"], dtype=np.float64),
            poincare_z=np.asarray(data["poincare_z"], dtype=np.float64),
            poincare_time=np.asarray(data["poincare_time"], dtype=np.float64),
            poincare_section=np.asarray(data["poincare_section"], dtype=np.float64),
            poincare_line_index=np.asarray(data["poincare_line_index"], dtype=np.int32),
            field_sample_xyz=np.asarray(data["field_sample_xyz"], dtype=np.float64),
            field_sample_b_xyz=np.asarray(data["field_sample_b_xyz"], dtype=np.float64),
            coil_gamma_xyz=np.asarray(data["coil_gamma_xyz"], dtype=np.float64),
            coil_currents=np.asarray(data["coil_currents"], dtype=np.float64),
            metadata={} if metadata is None else dict(metadata),
        )


def _import_essos_modules(*, essos_root: str | Path | None = None) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    if essos_root is not None:
        root = Path(essos_root)
    else:
        root = Path(os.environ.get("JAX_DRB_ESSOS_ROOT", _PRIVATE_DEFAULT_ESSOS_ROOT))
    if root.exists():
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    coils_module = importlib.import_module("essos.coils")
    fields_module = importlib.import_module("essos.fields")
    dynamics_module = importlib.import_module("essos.dynamics")
    return {
        "Coils_from_json": coils_module.Coils_from_json,
        "BiotSavart": fields_module.BiotSavart,
        "Tracing": dynamics_module.Tracing,
    }


def _flatten_essos_poincare_data(
    plotting_data: list[tuple[Any, Any, Any]],
    *,
    n_field_lines: int,
    shifts: tuple[float, ...],
) -> dict[str, np.ndarray]:
    r_values: list[np.ndarray] = []
    z_values: list[np.ndarray] = []
    time_values: list[np.ndarray] = []
    section_values: list[np.ndarray] = []
    line_values: list[np.ndarray] = []
    for index, (r_slice, z_slice, time_slice) in enumerate(plotting_data):
        r_array = np.asarray(r_slice, dtype=np.float64).reshape(-1)
        z_array = np.asarray(z_slice, dtype=np.float64).reshape(-1)
        time_array = np.asarray(time_slice, dtype=np.float64).reshape(-1)
        finite = np.isfinite(r_array) & np.isfinite(z_array) & np.isfinite(time_array)
        r_array = r_array[finite]
        z_array = z_array[finite]
        time_array = time_array[finite]
        if r_array.size == 0:
            continue
        shift_index = min(index // max(int(n_field_lines), 1), max(len(shifts) - 1, 0))
        line_index = index % max(int(n_field_lines), 1)
        r_values.append(r_array)
        z_values.append(z_array)
        time_values.append(time_array)
        section_values.append(np.full(r_array.shape, shifts[shift_index], dtype=np.float64))
        line_values.append(np.full(r_array.shape, line_index, dtype=np.int32))
    if not r_values:
        return {
            "r": np.empty(0, dtype=np.float64),
            "z": np.empty(0, dtype=np.float64),
            "time": np.empty(0, dtype=np.float64),
            "section": np.empty(0, dtype=np.float64),
            "line_index": np.empty(0, dtype=np.int32),
        }
    return {
        "r": np.concatenate(r_values),
        "z": np.concatenate(z_values),
        "time": np.concatenate(time_values),
        "section": np.concatenate(section_values),
        "line_index": np.concatenate(line_values),
    }


def _sample_essos_field(field: Any, points_xyz: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    points = jnp.asarray(points_xyz, dtype=jnp.float64)
    return np.asarray(jax.vmap(field.B)(points), dtype=np.float64)
