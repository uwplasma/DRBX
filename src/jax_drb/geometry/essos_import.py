from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp

from .fci_maps import FciMaps
from .metric_tensor import MetricTensor3D


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


@dataclass(frozen=True)
class EssosImportedFciGeometry:
    """Annular FCI geometry whose field-line maps are exported from ESSOS."""

    coordinates_x: jnp.ndarray
    coordinates_y: jnp.ndarray
    coordinates_z: jnp.ndarray
    minor_radius: jnp.ndarray
    toroidal_angle: jnp.ndarray
    poloidal_angle: jnp.ndarray
    magnetic_field_magnitude: jnp.ndarray
    connection_length: jnp.ndarray
    metric: MetricTensor3D
    maps: FciMaps
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.minor_radius.shape)


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


def build_essos_imported_fci_geometry(
    *,
    coil_json_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    nx: int = 6,
    ny: int = 8,
    nz: int = 16,
    rho_min: float = 0.10,
    rho_max: float = 0.46,
    maxtime: float = 140.0,
    times_to_trace: int = 768,
    trace_tolerance: float = 1.0e-8,
) -> EssosImportedFciGeometry:
    """Build FCI maps from an ESSOS-traced annular seed grid.

    The magnetic field and field-line integration remain external. `jax_drb`
    provides only the logical-grid conversion needed by the native FCI,
    sheath/recycling, neutral, and PyTree RHS kernels.
    """

    if nx < 2 or ny < 2 or nz < 4:
        raise ValueError("ESSOS imported FCI geometry requires nx >= 2, ny >= 2, and nz >= 4")
    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_coil_json.parents[2])

    import jax
    import jax.numpy as local_jnp

    coils = modules["Coils_from_json"](str(resolved_coil_json))
    field = modules["BiotSavart"](coils)
    axis_major_radius = float(field.r_axis)
    axis_vertical = float(field.z_axis)

    rho_1d = np.linspace(float(rho_min), float(rho_max), int(nx))
    phi_1d = np.linspace(0.0, 2.0 * np.pi, int(ny), endpoint=False)
    theta_1d = np.linspace(0.0, 2.0 * np.pi, int(nz), endpoint=False)
    rho, phi, theta = np.meshgrid(rho_1d, phi_1d, theta_1d, indexing="ij")
    major = axis_major_radius + rho * np.cos(theta)
    vertical = axis_vertical + rho * np.sin(theta)
    coordinates_x = major * np.cos(phi)
    coordinates_y = major * np.sin(phi)
    coordinates_z = vertical
    initial_xyz = np.stack([coordinates_x, coordinates_y, coordinates_z], axis=-1).reshape((-1, 3))
    start_phi = phi.reshape(-1)
    dphi = float(2.0 * np.pi / float(ny))

    forward_trajectories = _trace_essos_initial_conditions(
        modules=modules,
        resolved_coil_json=resolved_coil_json,
        initial_xyz=initial_xyz,
        current_sign=1.0,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
    )
    if _median_toroidal_advance(forward_trajectories) < 0.0:
        forward_current_sign = -1.0
        forward_trajectories = _trace_essos_initial_conditions(
            modules=modules,
            resolved_coil_json=resolved_coil_json,
            initial_xyz=initial_xyz,
            current_sign=forward_current_sign,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            trace_tolerance=trace_tolerance,
        )
    else:
        forward_current_sign = 1.0
    backward_trajectories = _trace_essos_initial_conditions(
        modules=modules,
        resolved_coil_json=resolved_coil_json,
        initial_xyz=initial_xyz,
        current_sign=-forward_current_sign,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
    )

    forward_endpoint, forward_length, forward_crossed = _interpolate_trajectories_at_toroidal_plane(
        forward_trajectories,
        target_phi=start_phi + dphi,
    )
    backward_endpoint, backward_length, backward_crossed = _interpolate_trajectories_at_toroidal_plane(
        backward_trajectories,
        target_phi=start_phi - dphi,
    )
    forward_x, forward_z, forward_boundary = _cartesian_to_annular_indices(
        forward_endpoint,
        crossed=forward_crossed,
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        nx=int(nx),
        nz=int(nz),
    )
    backward_x, backward_z, backward_boundary = _cartesian_to_annular_indices(
        backward_endpoint,
        crossed=backward_crossed,
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        nx=int(nx),
        nz=int(nz),
    )
    maps = FciMaps(
        forward_x=jnp.asarray(forward_x.reshape((nx, ny, nz)), dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z.reshape((nx, ny, nz)), dtype=jnp.float64),
        backward_x=jnp.asarray(backward_x.reshape((nx, ny, nz)), dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z.reshape((nx, ny, nz)), dtype=jnp.float64),
        forward_boundary=jnp.asarray(forward_boundary.reshape((nx, ny, nz))),
        backward_boundary=jnp.asarray(backward_boundary.reshape((nx, ny, nz))),
        dphi=dphi,
    )

    b_xyz = np.asarray(jax.vmap(field.B)(local_jnp.asarray(initial_xyz, dtype=local_jnp.float64)), dtype=np.float64)
    bmag = np.linalg.norm(b_xyz, axis=1).reshape((nx, ny, nz))
    exit_length = _annular_exit_length_from_trajectories(
        forward_trajectories,
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
        rho_min=float(rho_min),
        rho_max=float(rho_max),
    ).reshape((nx, ny, nz))
    adjacent_length = 0.5 * (forward_length + backward_length).reshape((nx, ny, nz))
    connection_length = np.where(np.isfinite(exit_length), exit_length, adjacent_length)
    metric = _annular_metric_tensor(
        rho=rho,
        major=major,
        bmag=bmag,
        drho=float(rho_1d[1] - rho_1d[0]) if nx > 1 else 1.0,
        dphi=dphi,
        dtheta=float(2.0 * np.pi / float(nz)),
    )
    return EssosImportedFciGeometry(
        coordinates_x=jnp.asarray(coordinates_x, dtype=jnp.float64),
        coordinates_y=jnp.asarray(coordinates_y, dtype=jnp.float64),
        coordinates_z=jnp.asarray(coordinates_z, dtype=jnp.float64),
        minor_radius=jnp.asarray(rho, dtype=jnp.float64),
        toroidal_angle=jnp.asarray(phi, dtype=jnp.float64),
        poloidal_angle=jnp.asarray(theta, dtype=jnp.float64),
        magnetic_field_magnitude=jnp.asarray(bmag, dtype=jnp.float64),
        connection_length=jnp.asarray(connection_length, dtype=jnp.float64),
        metric=metric,
        maps=maps,
        metadata={
            "geometry_family": "essos_imported_annular_fci",
            "source": "ESSOS",
            "coil_json_file": resolved_coil_json.name,
            "field_model": "essos.fields.BiotSavart",
            "tracing_model": "essos.dynamics.Tracing(FieldLineAdaptative)",
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
            "axis_major_radius": float(axis_major_radius),
            "axis_vertical": float(axis_vertical),
            "maxtime": float(maxtime),
            "times_to_trace": int(times_to_trace),
            "trace_tolerance": float(trace_tolerance),
        },
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
    if os.environ.get("JAX_DRB_ESSOS_PROGRESS") != "1" and hasattr(dynamics_module, "NoProgressMeter"):
        dynamics_module.TqdmProgressMeter = dynamics_module.NoProgressMeter
    return {
        "Coils_from_json": coils_module.Coils_from_json,
        "BiotSavart": fields_module.BiotSavart,
        "Tracing": dynamics_module.Tracing,
    }


def _trace_essos_initial_conditions(
    *,
    modules: dict[str, Any],
    resolved_coil_json: Path,
    initial_xyz: np.ndarray,
    current_sign: float,
    maxtime: float,
    times_to_trace: int,
    trace_tolerance: float,
) -> np.ndarray:
    import jax
    import jax.numpy as local_jnp

    coils = modules["Coils_from_json"](str(resolved_coil_json))
    if current_sign < 0.0:
        coils.dofs_currents = -coils.dofs_currents
    field = modules["BiotSavart"](coils)
    tracing = jax.block_until_ready(
        modules["Tracing"](
            field=field,
            model="FieldLineAdaptative",
            initial_conditions=local_jnp.asarray(initial_xyz, dtype=local_jnp.float64),
            maxtime=float(maxtime),
            times_to_trace=int(times_to_trace),
            atol=float(trace_tolerance),
            rtol=float(trace_tolerance),
        )
    )
    return np.asarray(tracing.trajectories[:, :, :3], dtype=np.float64)


def _median_toroidal_advance(trajectories_xyz: np.ndarray) -> float:
    phi = np.unwrap(np.arctan2(trajectories_xyz[:, :, 1], trajectories_xyz[:, :, 0]), axis=1)
    return float(np.nanmedian(phi[:, -1] - phi[:, 0]))


def _interpolate_trajectories_at_toroidal_plane(
    trajectories_xyz: np.ndarray,
    *,
    target_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints = np.full((trajectories_xyz.shape[0], 3), np.nan, dtype=np.float64)
    lengths = np.full(trajectories_xyz.shape[0], np.nan, dtype=np.float64)
    crossed = np.zeros(trajectories_xyz.shape[0], dtype=bool)
    for index, trajectory in enumerate(trajectories_xyz):
        phi = np.unwrap(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
        arc_length = np.concatenate(
            [
                np.zeros(1, dtype=np.float64),
                np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)),
            ]
        )
        if phi[-1] < phi[0]:
            phi = phi[::-1]
            trajectory = trajectory[::-1]
            arc_length = arc_length[-1] - arc_length[::-1]
        target = float(target_phi[index])
        if target < phi[0] or target > phi[-1] or not np.all(np.isfinite(phi)):
            continue
        endpoints[index, 0] = np.interp(target, phi, trajectory[:, 0])
        endpoints[index, 1] = np.interp(target, phi, trajectory[:, 1])
        endpoints[index, 2] = np.interp(target, phi, trajectory[:, 2])
        lengths[index] = np.interp(target, phi, arc_length)
        crossed[index] = True
    return endpoints, lengths, crossed


def _cartesian_to_annular_indices(
    points_xyz: np.ndarray,
    *,
    crossed: np.ndarray,
    axis_major_radius: float,
    axis_vertical: float,
    rho_min: float,
    rho_max: float,
    nx: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    major = np.sqrt(points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2)
    vertical = points_xyz[:, 2]
    radial_offset = major - float(axis_major_radius)
    vertical_offset = vertical - float(axis_vertical)
    rho = np.sqrt(radial_offset * radial_offset + vertical_offset * vertical_offset)
    theta = np.mod(np.arctan2(vertical_offset, radial_offset), 2.0 * np.pi)
    x_index = (rho - float(rho_min)) / max(float(rho_max - rho_min), 1.0e-30) * float(nx - 1)
    z_index = theta / (2.0 * np.pi) * float(nz)
    boundary = (~crossed) | (~np.isfinite(x_index)) | (x_index < 0.0) | (x_index > float(nx - 1))
    x_index = np.where(boundary, 0.0, x_index)
    z_index = np.where(boundary | (~np.isfinite(z_index)), 0.0, z_index)
    return x_index, z_index, boundary


def _annular_exit_length_from_trajectories(
    trajectories_xyz: np.ndarray,
    *,
    axis_major_radius: float,
    axis_vertical: float,
    rho_min: float,
    rho_max: float,
) -> np.ndarray:
    major = np.sqrt(trajectories_xyz[:, :, 0] ** 2 + trajectories_xyz[:, :, 1] ** 2)
    vertical = trajectories_xyz[:, :, 2]
    rho = np.sqrt((major - float(axis_major_radius)) ** 2 + (vertical - float(axis_vertical)) ** 2)
    tolerance = 1.0e-10 * max(float(rho_max - rho_min), 1.0)
    outside = (rho < float(rho_min) - tolerance) | (rho > float(rho_max) + tolerance) | (~np.isfinite(rho))
    arc_length = np.concatenate(
        [
            np.zeros((trajectories_xyz.shape[0], 1), dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(trajectories_xyz, axis=1), axis=2), axis=1),
        ],
        axis=1,
    )
    first_exit = np.argmax(outside, axis=1)
    has_exit = np.any(outside, axis=1)
    return np.where(has_exit, arc_length[np.arange(trajectories_xyz.shape[0]), first_exit], arc_length[:, -1])


def _annular_metric_tensor(
    *,
    rho: np.ndarray,
    major: np.ndarray,
    bmag: np.ndarray,
    drho: float,
    dphi: float,
    dtheta: float,
) -> MetricTensor3D:
    zeros = np.zeros_like(rho)
    safe_rho = np.maximum(rho, 1.0e-8)
    safe_major = np.maximum(major, 1.0e-8)
    jacobian = safe_major * safe_rho
    return MetricTensor3D(
        dx=jnp.asarray(np.full_like(rho, float(drho)), dtype=jnp.float64),
        dy=jnp.asarray(np.full_like(rho, float(dphi)), dtype=jnp.float64),
        dz=jnp.asarray(np.full_like(rho, float(dtheta)), dtype=jnp.float64),
        J=jnp.asarray(jacobian, dtype=jnp.float64),
        Bxy=jnp.asarray(bmag, dtype=jnp.float64),
        g11=jnp.asarray(np.ones_like(rho), dtype=jnp.float64),
        g22=jnp.asarray(1.0 / (safe_major * safe_major), dtype=jnp.float64),
        g33=jnp.asarray(1.0 / (safe_rho * safe_rho), dtype=jnp.float64),
        g12=jnp.asarray(zeros, dtype=jnp.float64),
        g13=jnp.asarray(zeros, dtype=jnp.float64),
        g23=jnp.asarray(zeros, dtype=jnp.float64),
        g_11=jnp.asarray(np.ones_like(rho), dtype=jnp.float64),
        g_22=jnp.asarray(safe_major * safe_major, dtype=jnp.float64),
        g_33=jnp.asarray(safe_rho * safe_rho, dtype=jnp.float64),
        g_12=jnp.asarray(zeros, dtype=jnp.float64),
        g_13=jnp.asarray(zeros, dtype=jnp.float64),
        g_23=jnp.asarray(zeros, dtype=jnp.float64),
    )


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
