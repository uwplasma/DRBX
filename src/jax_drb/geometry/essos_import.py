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
ESSOS_LANDREMAN_QA_RELATIVE_WOUT = Path("examples/input_files/wout_LandremanPaul2021_QA_reactorScale_lowres.nc")
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
    """VMEC-shaped FCI geometry whose field-line maps are exported from ESSOS."""

    coordinates_x: jnp.ndarray
    coordinates_y: jnp.ndarray
    coordinates_z: jnp.ndarray
    minor_radius: jnp.ndarray
    toroidal_angle: jnp.ndarray
    poloidal_angle: jnp.ndarray
    magnetic_field_magnitude: jnp.ndarray
    connection_length: jnp.ndarray
    adjacent_step_length: jnp.ndarray | None
    target_exit_length: jnp.ndarray | None
    forward_target_exit_length: jnp.ndarray | None
    backward_target_exit_length: jnp.ndarray | None
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


def resolve_essos_landreman_qa_wout(path: str | Path | None = None, *, essos_root: str | Path | None = None) -> Path:
    """Resolve the matching Landreman-Paul QA VMEC wout file."""

    if path is not None:
        resolved = Path(path)
    else:
        root = Path(essos_root) if essos_root is not None else Path(os.environ.get("JAX_DRB_ESSOS_ROOT", _PRIVATE_DEFAULT_ESSOS_ROOT))
        resolved = root / ESSOS_LANDREMAN_QA_RELATIVE_WOUT
    if not resolved.exists():
        raise FileNotFoundError(
            "ESSOS Landreman-Paul QA VMEC wout file was not found. Pass vmec_wout_path "
            "or set JAX_DRB_ESSOS_ROOT to an ESSOS checkout containing "
            f"{ESSOS_LANDREMAN_QA_RELATIVE_WOUT}."
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


def load_essos_coil_field_axis(
    *,
    coil_json_path: str | Path | None = None,
    essos_root: str | Path | None = None,
) -> tuple[float, float]:
    """Return the magnetic-axis location reported by the imported ESSOS coil field."""

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_coil_json.parents[2])
    coils = modules["Coils_from_json"](str(resolved_coil_json))
    field = modules["BiotSavart"](coils)
    return float(field.r_axis), float(field.z_axis)


def load_essos_vmec_field_axis(
    *,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
) -> tuple[float, float]:
    """Return the magnetic-axis location in an ESSOS VMEC field object."""

    resolved_wout = resolve_essos_landreman_qa_wout(vmec_wout_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_wout.parents[2])
    vmec = modules["Vmec"](str(resolved_wout))
    return float(vmec.r_axis), float(vmec.z_axis)


def trace_essos_coil_initial_conditions(
    initial_xyz: np.ndarray,
    *,
    coil_json_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    current_sign: float = 1.0,
    maxtime: float = 1000.0,
    times_to_trace: int = 6000,
    trace_tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Trace arbitrary Cartesian seed points through the optional ESSOS coil field."""

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_coil_json.parents[2])
    return _trace_essos_initial_conditions(
        modules=modules,
        resolved_coil_json=resolved_coil_json,
        initial_xyz=np.asarray(initial_xyz, dtype=np.float64),
        current_sign=float(current_sign),
        maxtime=float(maxtime),
        times_to_trace=int(times_to_trace),
        trace_tolerance=float(trace_tolerance),
    )


def trace_essos_vmec_initial_conditions(
    initial_stp: np.ndarray,
    *,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    maxtime: float = 1000.0,
    times_to_trace: int = 6000,
    trace_tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Trace VMEC-coordinate seed points through the optional ESSOS VMEC field."""

    resolved_wout = resolve_essos_landreman_qa_wout(vmec_wout_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_wout.parents[2])
    return _trace_essos_vmec_initial_conditions(
        modules=modules,
        resolved_wout=resolved_wout,
        initial_stp=np.asarray(initial_stp, dtype=np.float64),
        maxtime=float(maxtime),
        times_to_trace=int(times_to_trace),
        trace_tolerance=float(trace_tolerance),
    )


def build_essos_imported_fci_geometry(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 6,
    ny: int = 8,
    nz: int = 16,
    rho_min: float = 0.10,
    rho_max: float = 0.46,
    maxtime: float = 140.0,
    times_to_trace: int = 768,
    trace_tolerance: float = 1.0e-8,
) -> EssosImportedFciGeometry:
    """Build FCI maps from ESSOS tracing on a VMEC-shaped QA seed grid.

    ``map_source`` selects the field-line-map semantics:

    - ``"coil"`` traces the imported coil field and keeps its open-field masks;
    - ``"vmec"`` builds surface-preserving VMEC-coordinate maps with closed
      field-line masks;
    - ``"hybrid"`` uses VMEC-coordinate map coordinates with coil-derived
      boundary masks, connection lengths, and magnetic-field modulation.

    The external field implementation owns the field evaluation. `jax_drb`
    provides only the logical-grid conversion needed by the native FCI,
    sheath/recycling, neutral, and PyTree RHS kernels.
    """

    if nx < 2 or ny < 2 or nz < 4:
        raise ValueError("ESSOS imported FCI geometry requires nx >= 2, ny >= 2, and nz >= 4")
    map_source = _normalize_essos_map_source(map_source)
    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    modules = _import_essos_modules(essos_root=essos_root if essos_root is not None else resolved_coil_json.parents[2])

    coils = modules["Coils_from_json"](str(resolved_coil_json))
    field = modules["BiotSavart"](coils)
    axis_major_radius = float(field.r_axis)
    axis_vertical = float(field.z_axis)

    resolved_vmec_wout = resolve_essos_landreman_qa_wout(vmec_wout_path, essos_root=essos_root)
    coordinates = build_essos_vmec_scaled_qa_coordinates(
        resolved_vmec_wout,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
    )
    rho_1d = coordinates["rho_1d"]
    phi_1d = coordinates["phi_1d"]
    theta_1d = coordinates["theta_1d"]
    rho = coordinates["rho"]
    phi = coordinates["phi"]
    theta = coordinates["theta"]
    coordinates_x = coordinates["x"]
    coordinates_y = coordinates["y"]
    coordinates_z = coordinates["z"]
    initial_xyz = np.stack([coordinates_x, coordinates_y, coordinates_z], axis=-1).reshape((-1, 3))
    start_phi = phi.reshape(-1)
    dphi = float(2.0 * np.pi / float(ny))
    start_y_index = np.broadcast_to(np.arange(int(ny), dtype=int)[None, :, None], (int(nx), int(ny), int(nz))).reshape(-1)
    coil_data: dict[str, Any] | None = None
    if map_source in {"coil", "hybrid"}:
        coil_data = _build_essos_coil_fci_map_data(
            modules=modules,
            resolved_coil_json=resolved_coil_json,
            field=field,
            initial_xyz=initial_xyz,
            start_phi=start_phi,
            start_y_index=start_y_index,
            dphi=dphi,
            shape=(int(nx), int(ny), int(nz)),
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            coordinates_z=coordinates_z,
            maxtime=float(maxtime),
            times_to_trace=int(times_to_trace),
            trace_tolerance=float(trace_tolerance),
        )
    vmec_data: dict[str, Any] | None = None
    if map_source in {"vmec", "hybrid"}:
        vmec_data = _build_essos_vmec_fci_map_data(
            modules=modules,
            resolved_wout=resolved_vmec_wout,
            coordinates=coordinates,
            shape=(int(nx), int(ny), int(nz)),
            dphi=dphi,
        )

    if map_source == "coil":
        assert coil_data is not None
        maps = coil_data["maps"]
        bmag = coil_data["bmag"]
        connection_length = coil_data["connection_length"]
        adjacent_step_length = coil_data["adjacent_step_length"]
        target_exit_length = coil_data["target_exit_length"]
        forward_target_exit_length = coil_data["forward_target_exit_length"]
        backward_target_exit_length = coil_data["backward_target_exit_length"]
        field_model = "essos.fields.BiotSavart"
        tracing_model = "essos.dynamics.Tracing(FieldLineAdaptative)"
    elif map_source == "vmec":
        assert vmec_data is not None
        maps = vmec_data["maps"]
        bmag = vmec_data["bmag"]
        connection_length = vmec_data["connection_length"]
        adjacent_step_length = vmec_data["connection_length"]
        target_exit_length = np.full_like(connection_length, np.nan, dtype=np.float64)
        forward_target_exit_length = np.full_like(connection_length, np.nan, dtype=np.float64)
        backward_target_exit_length = np.full_like(connection_length, np.nan, dtype=np.float64)
        field_model = "essos.fields.Vmec"
        tracing_model = "vmec_coordinate_rk4_map"
    else:
        assert coil_data is not None
        assert vmec_data is not None
        maps = FciMaps(
            forward_x=vmec_data["maps"].forward_x,
            forward_z=vmec_data["maps"].forward_z,
            backward_x=vmec_data["maps"].backward_x,
            backward_z=vmec_data["maps"].backward_z,
            forward_boundary=coil_data["maps"].forward_boundary,
            backward_boundary=coil_data["maps"].backward_boundary,
            dphi=dphi,
        )
        bmag = coil_data["bmag"]
        connection_length = coil_data["connection_length"]
        adjacent_step_length = vmec_data["connection_length"]
        target_exit_length = coil_data["target_exit_length"]
        forward_target_exit_length = coil_data["forward_target_exit_length"]
        backward_target_exit_length = coil_data["backward_target_exit_length"]
        field_model = "hybrid: VMEC map coordinates with Biot-Savart |B| and target masks"
        tracing_model = "vmec_coordinate_rk4_map + essos.dynamics.Tracing(FieldLineAdaptative) masks"

    forward_boundary = np.asarray(maps.forward_boundary, dtype=bool)
    backward_boundary = np.asarray(maps.backward_boundary, dtype=bool)
    metric = _metric_from_coordinates(
        coordinates_x,
        coordinates_y,
        coordinates_z,
        s_1d=rho_1d,
        phi_1d=phi_1d,
        theta_1d=theta_1d,
        Bxy=bmag,
    )
    forward_boundary_fraction = float(np.mean(forward_boundary))
    backward_boundary_fraction = float(np.mean(backward_boundary))
    return EssosImportedFciGeometry(
        coordinates_x=jnp.asarray(coordinates_x, dtype=jnp.float64),
        coordinates_y=jnp.asarray(coordinates_y, dtype=jnp.float64),
        coordinates_z=jnp.asarray(coordinates_z, dtype=jnp.float64),
        minor_radius=jnp.asarray(rho, dtype=jnp.float64),
        toroidal_angle=jnp.asarray(phi, dtype=jnp.float64),
        poloidal_angle=jnp.asarray(theta, dtype=jnp.float64),
        magnetic_field_magnitude=jnp.asarray(bmag, dtype=jnp.float64),
        connection_length=jnp.asarray(connection_length, dtype=jnp.float64),
        adjacent_step_length=jnp.asarray(adjacent_step_length, dtype=jnp.float64),
        target_exit_length=jnp.asarray(target_exit_length, dtype=jnp.float64),
        forward_target_exit_length=jnp.asarray(forward_target_exit_length, dtype=jnp.float64),
        backward_target_exit_length=jnp.asarray(backward_target_exit_length, dtype=jnp.float64),
        metric=metric,
        maps=maps,
        metadata={
            "geometry_family": "essos_imported_vmec_qa_fci",
            "source": "ESSOS",
            "coil_json_file": resolved_coil_json.name,
            "vmec_wout_file": resolved_vmec_wout.name,
            "coordinate_model": "scaled_vmec_fourier_flux_surfaces",
            **coordinates["metadata"],
            "map_source": map_source,
            "field_model": field_model,
            "tracing_model": tracing_model,
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
            "coil_trace_current_sign": float(coil_data["current_sign"]) if coil_data is not None else 0.0,
            "vmec_map_theta_step_count": int(vmec_data["theta_step_count"]) if vmec_data is not None else 0,
            "forward_boundary_fraction": forward_boundary_fraction,
            "backward_boundary_fraction": backward_boundary_fraction,
        },
    )


def _normalize_essos_map_source(map_source: str) -> str:
    normalized = str(map_source).strip().lower().replace("-", "_")
    aliases = {
        "essos": "coil",
        "essos_coil": "coil",
        "coil_map": "coil",
        "vmec_map": "vmec",
        "hybrid_map": "hybrid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"coil", "vmec", "hybrid"}:
        raise ValueError("map_source must be one of 'coil', 'vmec', or 'hybrid'")
    return normalized


def _build_essos_coil_fci_map_data(
    *,
    modules: dict[str, Any],
    resolved_coil_json: Path,
    field: Any,
    initial_xyz: np.ndarray,
    start_phi: np.ndarray,
    start_y_index: np.ndarray,
    dphi: float,
    shape: tuple[int, int, int],
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    coordinates_z: np.ndarray,
    maxtime: float,
    times_to_trace: int,
    trace_tolerance: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as local_jnp

    nx, ny, nz = shape
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
    forward_x, forward_z, forward_boundary = _cartesian_to_structured_surface_indices(
        forward_endpoint,
        crossed=forward_crossed,
        target_y_index=(start_y_index + 1) % int(ny),
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        coordinates_z=coordinates_z,
    )
    backward_x, backward_z, backward_boundary = _cartesian_to_structured_surface_indices(
        backward_endpoint,
        crossed=backward_crossed,
        target_y_index=(start_y_index - 1) % int(ny),
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        coordinates_z=coordinates_z,
    )
    b_xyz = np.asarray(jax.vmap(field.B)(local_jnp.asarray(initial_xyz, dtype=local_jnp.float64)), dtype=np.float64)
    bmag = np.linalg.norm(b_xyz, axis=1).reshape(shape)
    forward_exit_length = _structured_exit_length_from_trajectories(
        forward_trajectories,
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        coordinates_z=coordinates_z,
    ).reshape(shape)
    backward_exit_length = _structured_exit_length_from_trajectories(
        backward_trajectories,
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        coordinates_z=coordinates_z,
    ).reshape(shape)
    exit_length = _combine_bidirectional_exit_lengths(
        forward_exit_length,
        backward_exit_length,
    )
    adjacent_length = 0.5 * (forward_length + backward_length).reshape(shape)
    connection_length = np.where(np.isfinite(exit_length), exit_length, adjacent_length)
    maps = FciMaps(
        forward_x=jnp.asarray(forward_x.reshape(shape), dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z.reshape(shape), dtype=jnp.float64),
        backward_x=jnp.asarray(backward_x.reshape(shape), dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z.reshape(shape), dtype=jnp.float64),
        forward_boundary=jnp.asarray(forward_boundary.reshape(shape)),
        backward_boundary=jnp.asarray(backward_boundary.reshape(shape)),
        dphi=dphi,
    )
    return {
        "maps": maps,
        "bmag": bmag,
        "connection_length": connection_length,
        "adjacent_step_length": adjacent_length,
        "target_exit_length": exit_length,
        "forward_target_exit_length": forward_exit_length,
        "backward_target_exit_length": backward_exit_length,
        "current_sign": float(forward_current_sign),
    }


def _build_essos_vmec_fci_map_data(
    *,
    modules: dict[str, Any],
    resolved_wout: Path,
    coordinates: dict[str, Any],
    shape: tuple[int, int, int],
    dphi: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as local_jnp

    nx, ny, nz = shape
    vmec = modules["Vmec"](str(resolved_wout))
    s = np.broadcast_to(np.asarray(coordinates["s_1d"], dtype=np.float64)[:, None, None], shape).reshape(-1)
    theta = np.asarray(coordinates["theta"], dtype=np.float64).reshape(-1)
    phi = np.asarray(coordinates["phi"], dtype=np.float64).reshape(-1)
    x_index = np.broadcast_to(np.arange(nx, dtype=np.float64)[:, None, None], shape).reshape(-1)
    step_count = max(12, min(48, int(2 * ny)))
    forward_theta = _integrate_vmec_theta_to_toroidal_offset(
        vmec,
        s=s,
        theta=theta,
        phi=phi,
        delta_phi=float(dphi),
        step_count=step_count,
    )
    backward_theta = _integrate_vmec_theta_to_toroidal_offset(
        vmec,
        s=s,
        theta=theta,
        phi=phi,
        delta_phi=-float(dphi),
        step_count=step_count,
    )
    forward_z = np.mod(forward_theta, 2.0 * np.pi) / (2.0 * np.pi) * float(nz)
    backward_z = np.mod(backward_theta, 2.0 * np.pi) / (2.0 * np.pi) * float(nz)
    finite = np.isfinite(forward_z) & np.isfinite(backward_z) & (s >= 0.0) & (s <= 1.0)
    forward_boundary = ~finite
    backward_boundary = ~finite
    initial_stp = np.stack([s, theta, phi], axis=-1)
    bmag = np.asarray(jax.vmap(vmec.AbsB)(local_jnp.asarray(initial_stp, dtype=local_jnp.float64)), dtype=np.float64).reshape(shape)
    forward_length = _vmec_map_step_length(
        coordinates=coordinates,
        x_index=x_index,
        y_index=(np.broadcast_to(np.arange(ny, dtype=int)[None, :, None], shape).reshape(-1) + 1) % ny,
        z_index=forward_z,
    )
    backward_length = _vmec_map_step_length(
        coordinates=coordinates,
        x_index=x_index,
        y_index=(np.broadcast_to(np.arange(ny, dtype=int)[None, :, None], shape).reshape(-1) - 1) % ny,
        z_index=backward_z,
    )
    connection_length = 0.5 * (forward_length + backward_length).reshape(shape)
    maps = FciMaps(
        forward_x=jnp.asarray(x_index.reshape(shape), dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z.reshape(shape), dtype=jnp.float64),
        backward_x=jnp.asarray(x_index.reshape(shape), dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z.reshape(shape), dtype=jnp.float64),
        forward_boundary=jnp.asarray(forward_boundary.reshape(shape)),
        backward_boundary=jnp.asarray(backward_boundary.reshape(shape)),
        dphi=float(dphi),
    )
    return {
        "maps": maps,
        "bmag": bmag,
        "connection_length": connection_length,
        "theta_step_count": int(step_count),
    }


def _integrate_vmec_theta_to_toroidal_offset(
    vmec: Any,
    *,
    s: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    delta_phi: float,
    step_count: int,
) -> np.ndarray:
    import jax
    import jax.numpy as local_jnp

    s_jax = local_jnp.asarray(s, dtype=local_jnp.float64)
    theta_jax = local_jnp.asarray(theta, dtype=local_jnp.float64)
    phi_jax = local_jnp.asarray(phi, dtype=local_jnp.float64)
    h = float(delta_phi) / float(step_count)

    def rhs(theta_value: jax.Array, phi_value: jax.Array) -> jax.Array:
        points = local_jnp.stack([s_jax, theta_value, phi_value], axis=-1)
        b_contra = jax.vmap(vmec.B_contravariant)(points)
        b_phi = local_jnp.where(local_jnp.abs(b_contra[:, 2]) > 1.0e-30, b_contra[:, 2], 1.0e-30)
        return b_contra[:, 1] / b_phi

    def body(_: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        theta_value, phi_value = carry
        k1 = rhs(theta_value, phi_value)
        k2 = rhs(theta_value + 0.5 * h * k1, phi_value + 0.5 * h)
        k3 = rhs(theta_value + 0.5 * h * k2, phi_value + 0.5 * h)
        k4 = rhs(theta_value + h * k3, phi_value + h)
        next_theta = theta_value + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_theta, phi_value + h

    integrate = jax.jit(lambda theta0, phi0: jax.lax.fori_loop(0, int(step_count), body, (theta0, phi0))[0])
    return np.asarray(integrate(theta_jax, phi_jax), dtype=np.float64)


def _vmec_map_step_length(
    *,
    coordinates: dict[str, Any],
    x_index: np.ndarray,
    y_index: np.ndarray,
    z_index: np.ndarray,
) -> np.ndarray:
    current = np.stack(
        [
            np.asarray(coordinates["x"], dtype=np.float64).reshape(-1),
            np.asarray(coordinates["y"], dtype=np.float64).reshape(-1),
            np.asarray(coordinates["z"], dtype=np.float64).reshape(-1),
        ],
        axis=-1,
    )
    endpoint = _sample_structured_coordinates(
        np.asarray(coordinates["x"], dtype=np.float64),
        np.asarray(coordinates["y"], dtype=np.float64),
        np.asarray(coordinates["z"], dtype=np.float64),
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )
    length = np.linalg.norm(endpoint - current, axis=1)
    return np.where(np.isfinite(length), length, 0.0)


def _sample_structured_coordinates(
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    coordinates_z: np.ndarray,
    *,
    x_index: np.ndarray,
    y_index: np.ndarray,
    z_index: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = coordinates_x.shape
    x0 = np.clip(np.floor(x_index).astype(int), 0, nx - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y = np.mod(y_index.astype(int), ny)
    z = np.mod(z_index, float(nz))
    z0 = np.floor(z).astype(int) % nz
    z1 = (z0 + 1) % nz
    wx = np.clip(x_index - x0.astype(np.float64), 0.0, 1.0)
    wz = z - np.floor(z)
    result = []
    for values in (coordinates_x, coordinates_y, coordinates_z):
        f00 = values[x0, y, z0]
        f10 = values[x1, y, z0]
        f01 = values[x0, y, z1]
        f11 = values[x1, y, z1]
        result.append((1.0 - wx) * (1.0 - wz) * f00 + wx * (1.0 - wz) * f10 + (1.0 - wx) * wz * f01 + wx * wz * f11)
    return np.stack(result, axis=-1)


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
        "Vmec": fields_module.Vmec,
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


def _trace_essos_vmec_initial_conditions(
    *,
    modules: dict[str, Any],
    resolved_wout: Path,
    initial_stp: np.ndarray,
    maxtime: float,
    times_to_trace: int,
    trace_tolerance: float,
) -> np.ndarray:
    import jax
    import jax.numpy as local_jnp

    vmec = modules["Vmec"](str(resolved_wout))
    tracing = jax.block_until_ready(
        modules["Tracing"](
            field=vmec,
            model="FieldLineAdaptative",
            initial_conditions=local_jnp.asarray(initial_stp, dtype=local_jnp.float64),
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


def build_essos_vmec_scaled_qa_coordinates(
    wout_path: Path,
    *,
    nx: int,
    ny: int,
    nz: int,
    rho_min: float,
    rho_max: float,
    axis_major_radius: float,
    axis_vertical: float,
) -> dict[str, Any]:
    """Evaluate scaled Landreman-Paul QA VMEC Fourier surfaces on a logical grid."""

    from netCDF4 import Dataset

    with Dataset(wout_path) as dataset:
        xm = np.asarray(dataset.variables["xm"][:], dtype=np.float64)
        xn = np.asarray(dataset.variables["xn"][:], dtype=np.float64)
        rmnc = np.asarray(dataset.variables["rmnc"][:], dtype=np.float64)
        zmns = np.asarray(dataset.variables["zmns"][:], dtype=np.float64)
        nfp = int(np.asarray(dataset.variables["nfp"][:]).reshape(()))

    if rmnc.ndim != 2 or zmns.shape != rmnc.shape:
        raise ValueError("VMEC wout Fourier arrays must have shape (ns, mnmax)")

    rho_1d = np.linspace(float(rho_min), float(rho_max), int(nx))
    phi_1d = np.linspace(0.0, 2.0 * np.pi, int(ny), endpoint=False)
    theta_1d = np.linspace(0.0, 2.0 * np.pi, int(nz), endpoint=False)
    rho, phi, theta = np.meshgrid(rho_1d, phi_1d, theta_1d, indexing="ij")

    ns = int(rmnc.shape[0])
    s_full = np.linspace(0.0, 1.0, ns)
    normalized_radius = (rho_1d - float(rho_min)) / max(float(rho_max - rho_min), 1.0e-30)
    normalized_radius = (float(rho_min) / max(float(rho_max), 1.0e-30)) + (
        1.0 - float(rho_min) / max(float(rho_max), 1.0e-30)
    ) * normalized_radius
    s_requested = np.clip(normalized_radius * normalized_radius, 0.0, 1.0)

    rmnc_shells = np.empty((int(nx), rmnc.shape[1]), dtype=np.float64)
    zmns_shells = np.empty_like(rmnc_shells)
    for mode_index in range(rmnc.shape[1]):
        rmnc_shells[:, mode_index] = np.interp(s_requested, s_full, rmnc[:, mode_index])
        zmns_shells[:, mode_index] = np.interp(s_requested, s_full, zmns[:, mode_index])

    raw_axis_major = float(rmnc[0, 0])
    raw_axis_vertical = float(zmns[0, 0])
    scale = float(axis_major_radius) / max(abs(raw_axis_major), 1.0e-30)
    phase = xm[None, None, None, :] * theta[..., None] - xn[None, None, None, :] * phi[..., None]
    raw_major = np.sum(rmnc_shells[:, None, None, :] * np.cos(phase), axis=-1)
    raw_vertical = np.sum(zmns_shells[:, None, None, :] * np.sin(phase), axis=-1)
    major = float(axis_major_radius) + scale * (raw_major - raw_axis_major)
    vertical = float(axis_vertical) + scale * (raw_vertical - raw_axis_vertical)
    x = major * np.cos(phi)
    y = major * np.sin(phi)
    z = vertical

    edge_major = major[-1]
    edge_vertical = vertical[-1]
    mean_edge_major_by_phi = np.mean(edge_major, axis=1)
    nonaxisymmetric_major_rms = float(np.std(mean_edge_major_by_phi) / max(abs(float(np.mean(mean_edge_major_by_phi))), 1.0e-30))
    edge_extent = np.sqrt((edge_major - np.mean(edge_major, axis=1, keepdims=True)) ** 2 + edge_vertical**2)
    poloidal_extent_rms = float(np.sqrt(np.mean(edge_extent * edge_extent)))

    return {
        "rho_1d": rho_1d,
        "s_1d": s_requested,
        "phi_1d": phi_1d,
        "theta_1d": theta_1d,
        "rho": rho,
        "phi": phi,
        "theta": theta,
        "major": major,
        "vertical": vertical,
        "x": x,
        "y": y,
        "z": z,
        "metadata": {
            "vmec_nfp": int(nfp),
            "vmec_ns": ns,
            "vmec_mnmax": int(rmnc.shape[1]),
            "vmec_raw_axis_major_radius": raw_axis_major,
            "vmec_raw_axis_vertical": raw_axis_vertical,
            "vmec_to_essos_length_scale": scale,
            "vmec_s_min": float(np.min(s_requested)),
            "vmec_s_max": float(np.max(s_requested)),
            "surface_nonaxisymmetric_major_rms": nonaxisymmetric_major_rms,
            "surface_poloidal_extent_rms": poloidal_extent_rms,
        },
    }


def _cartesian_to_structured_surface_indices(
    points_xyz: np.ndarray,
    *,
    crossed: np.ndarray,
    target_y_index: np.ndarray,
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    coordinates_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = coordinates_x.shape
    x_index = np.zeros(points_xyz.shape[0], dtype=np.float64)
    z_index = np.zeros(points_xyz.shape[0], dtype=np.float64)
    boundary = np.ones(points_xyz.shape[0], dtype=bool)

    plane_spacing = _structured_plane_spacing(coordinates_x, coordinates_y, coordinates_z)
    for index, point in enumerate(points_xyz):
        if not bool(crossed[index]) or not np.all(np.isfinite(point)):
            continue
        y_index = int(target_y_index[index]) % ny
        plane = np.column_stack(
            [
                coordinates_x[:, y_index, :].reshape(-1),
                coordinates_y[:, y_index, :].reshape(-1),
                coordinates_z[:, y_index, :].reshape(-1),
            ]
        )
        distance_squared = np.sum((plane - point[None, :]) ** 2, axis=1)
        nearest = int(np.argmin(distance_squared))
        nearest_distance = float(np.sqrt(distance_squared[nearest]))
        radial_index = nearest // nz
        poloidal_index = nearest % nz
        x_index[index] = float(radial_index)
        z_index[index] = float(poloidal_index)
        max_distance = 2.75 * float(plane_spacing[y_index])
        boundary[index] = (
            nearest_distance > max_distance
            or radial_index <= 0
            or radial_index >= nx - 1
        )
        if boundary[index]:
            x_index[index] = 0.0
            z_index[index] = 0.0
    return x_index, z_index, boundary


def _structured_plane_spacing(coordinates_x: np.ndarray, coordinates_y: np.ndarray, coordinates_z: np.ndarray) -> np.ndarray:
    coords = np.stack([coordinates_x, coordinates_y, coordinates_z], axis=-1)
    radial_spacing = np.linalg.norm(np.diff(coords, axis=0), axis=-1)
    poloidal_spacing = np.linalg.norm(np.diff(np.concatenate([coords, coords[:, :, :1, :]], axis=2), axis=2), axis=-1)
    spacing = []
    for y_index in range(coords.shape[1]):
        values = np.concatenate([radial_spacing[:, y_index, :].reshape(-1), poloidal_spacing[:, y_index, :].reshape(-1)])
        finite = values[np.isfinite(values) & (values > 0.0)]
        spacing.append(float(np.median(finite)) if finite.size else 1.0)
    return np.asarray(spacing, dtype=np.float64)


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


def _structured_exit_length_from_trajectories(
    trajectories_xyz: np.ndarray,
    *,
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    coordinates_z: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = coordinates_x.shape
    coords = np.stack([coordinates_x, coordinates_y, coordinates_z], axis=-1)
    radial_min = np.nanmin(np.sqrt(coordinates_x[0] * coordinates_x[0] + coordinates_y[0] * coordinates_y[0]))
    radial_max = np.nanmax(np.sqrt(coordinates_x[-1] * coordinates_x[-1] + coordinates_y[-1] * coordinates_y[-1]))
    vertical_min = np.nanmin(coordinates_z[-1])
    vertical_max = np.nanmax(coordinates_z[-1])
    padding = max(float(np.nanmedian(_structured_plane_spacing(coordinates_x, coordinates_y, coordinates_z))), 1.0e-6)
    arc_length = np.concatenate(
        [
            np.zeros((trajectories_xyz.shape[0], 1), dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(trajectories_xyz, axis=1), axis=2), axis=1),
        ],
        axis=1,
    )
    major = np.sqrt(trajectories_xyz[:, :, 0] ** 2 + trajectories_xyz[:, :, 1] ** 2)
    vertical = trajectories_xyz[:, :, 2]
    broad_exit = (
        (major < radial_min - padding)
        | (major > radial_max + padding)
        | (vertical < vertical_min - padding)
        | (vertical > vertical_max + padding)
        | (~np.isfinite(major))
        | (~np.isfinite(vertical))
    )
    first_exit = np.argmax(broad_exit, axis=1)
    has_exit = np.any(broad_exit, axis=1)
    if nx * ny * nz <= 4096:
        flat_coords = coords.reshape((-1, 3))
        max_distance = 3.0 * padding
        for line_index, trajectory in enumerate(trajectories_xyz):
            if has_exit[line_index]:
                continue
            stride = max(int(trajectory.shape[0] // 48), 1)
            sampled = trajectory[::stride]
            distances = np.sqrt(np.min(np.sum((sampled[:, None, :] - flat_coords[None, :, :]) ** 2, axis=-1), axis=1))
            exit_candidates = np.flatnonzero(distances > max_distance)
            if exit_candidates.size:
                first_exit[line_index] = min(int(exit_candidates[0]) * stride, trajectory.shape[0] - 1)
                has_exit[line_index] = True
    return np.where(has_exit, arc_length[np.arange(trajectories_xyz.shape[0]), first_exit], np.nan)


def _combine_bidirectional_exit_lengths(
    forward_exit_length: np.ndarray,
    backward_exit_length: np.ndarray,
) -> np.ndarray:
    """Return the shortest finite target-exit length from either traced direction."""

    forward = np.asarray(forward_exit_length, dtype=np.float64)
    backward = np.asarray(backward_exit_length, dtype=np.float64)
    if forward.shape != backward.shape:
        raise ValueError(
            "Forward and backward target-exit length arrays must have the same shape."
        )
    combined = np.full(forward.shape, np.nan, dtype=np.float64)
    forward_finite = np.isfinite(forward)
    backward_finite = np.isfinite(backward)
    both = forward_finite & backward_finite
    combined[both] = np.minimum(forward[both], backward[both])
    only_forward = forward_finite & ~backward_finite
    only_backward = backward_finite & ~forward_finite
    combined[only_forward] = forward[only_forward]
    combined[only_backward] = backward[only_backward]
    return combined


def _metric_from_coordinates(
    x_cart: np.ndarray,
    y_cart: np.ndarray,
    z_cart: np.ndarray,
    *,
    s_1d: np.ndarray,
    phi_1d: np.ndarray,
    theta_1d: np.ndarray,
    Bxy: np.ndarray,
) -> MetricTensor3D:
    ds = float(s_1d[1] - s_1d[0]) if s_1d.size > 1 else 1.0
    dphi = float(phi_1d[1] - phi_1d[0]) if phi_1d.size > 1 else 2.0 * np.pi
    dtheta = float(theta_1d[1] - theta_1d[0]) if theta_1d.size > 1 else 2.0 * np.pi
    edge_order = 2 if min(x_cart.shape) > 2 else 1

    derivs = []
    for coords in (x_cart, y_cart, z_cart):
        derivs.append(np.gradient(coords, ds, dphi, dtheta, edge_order=edge_order))

    r_s = np.stack([derivs[0][0], derivs[1][0], derivs[2][0]], axis=-1)
    r_phi = np.stack([derivs[0][1], derivs[1][1], derivs[2][1]], axis=-1)
    r_theta = np.stack([derivs[0][2], derivs[1][2], derivs[2][2]], axis=-1)
    cov = np.empty(x_cart.shape + (3, 3), dtype=np.float64)
    basis = (r_s, r_phi, r_theta)
    for i, left in enumerate(basis):
        for j, right in enumerate(basis):
            cov[..., i, j] = np.sum(left * right, axis=-1)
    determinant = np.linalg.det(cov)
    regularization = np.maximum(1.0e-12, 1.0e-11 * np.nanmax(np.abs(determinant)))
    bad = determinant <= regularization
    if np.any(bad):
        cov[bad] = cov[bad] + np.eye(3) * regularization
    contrav = np.linalg.inv(cov)
    jacobian = np.sqrt(np.maximum(np.linalg.det(cov), regularization))
    return MetricTensor3D(
        dx=jnp.asarray(np.full_like(x_cart, ds), dtype=jnp.float64),
        dy=jnp.asarray(np.full_like(x_cart, dphi), dtype=jnp.float64),
        dz=jnp.asarray(np.full_like(x_cart, dtheta), dtype=jnp.float64),
        J=jnp.asarray(jacobian, dtype=jnp.float64),
        Bxy=jnp.asarray(Bxy, dtype=jnp.float64),
        g11=jnp.asarray(contrav[..., 0, 0], dtype=jnp.float64),
        g22=jnp.asarray(contrav[..., 1, 1], dtype=jnp.float64),
        g33=jnp.asarray(contrav[..., 2, 2], dtype=jnp.float64),
        g12=jnp.asarray(contrav[..., 0, 1], dtype=jnp.float64),
        g13=jnp.asarray(contrav[..., 0, 2], dtype=jnp.float64),
        g23=jnp.asarray(contrav[..., 1, 2], dtype=jnp.float64),
        g_11=jnp.asarray(cov[..., 0, 0], dtype=jnp.float64),
        g_22=jnp.asarray(cov[..., 1, 1], dtype=jnp.float64),
        g_33=jnp.asarray(cov[..., 2, 2], dtype=jnp.float64),
        g_12=jnp.asarray(cov[..., 0, 1], dtype=jnp.float64),
        g_13=jnp.asarray(cov[..., 0, 2], dtype=jnp.float64),
        g_23=jnp.asarray(cov[..., 1, 2], dtype=jnp.float64),
    )


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
