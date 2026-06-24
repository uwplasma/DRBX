from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import FciGeometry3D, logical_grid_from_axis_vectors
from ..geometry.vmec_extender_import import (
    VmecExtenderGrid,
    build_vmec_extender_fci_geometry,
    load_vmec_extender_grid_netcdf,
)
from ..native.fci import conservative_parallel_diffusion_fci
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class VmecExtenderSolSmokeResult:
    history: np.ndarray
    time: np.ndarray
    source: np.ndarray
    loss_profile: np.ndarray
    endpoint_mask: np.ndarray
    fci_map_identity_max_abs_error: float
    parallel_mode_decay_relative_error: float | None


@dataclass(frozen=True)
class VmecExtenderSolSmokeArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_vmec_extender_sol_smoke_package(
    *,
    output_root: str | Path,
    field_grid_path: str | Path,
    case_label: str = "vmec_extender_sol_smoke",
    strict_metadata: bool = True,
) -> VmecExtenderSolSmokeArtifacts:
    """Run a compact imported-field SOL smoke campaign and write artifacts."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    grid = load_vmec_extender_grid_netcdf(field_grid_path, strict_metadata=strict_metadata)
    result = simulate_vmec_extender_scalar_sol_smoke(grid)
    report = build_vmec_extender_sol_smoke_report(grid, result)

    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        history=np.asarray(result.history, dtype=np.float32),
        time=np.asarray(result.time, dtype=np.float64),
        source=np.asarray(result.source, dtype=np.float32),
        loss_profile=np.asarray(result.loss_profile, dtype=np.float32),
        endpoint_mask=np.asarray(result.endpoint_mask, dtype=bool),
    )

    plot_png_path = save_vmec_extender_sol_smoke_plot(grid, result, report, images_dir / f"{case_label}.png")
    return VmecExtenderSolSmokeArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def simulate_vmec_extender_scalar_sol_smoke(
    grid: VmecExtenderGrid,
    *,
    frames: int = 14,
    substeps_per_frame: int = 4,
    dt: float = 0.006,
    parallel_diffusivity: float = 1.0e-2,
    perpendicular_diffusivity: float = 2.5e-4,
    source_strength: float = 3.0e-2,
    radial_loss_rate: float = 2.4e-2,
    endpoint_loss_rate: float = 8.0e-2,
) -> VmecExtenderSolSmokeResult:
    """Advance a small scalar SOL model on imported VMEC-extender FCI maps.

    The model is intentionally modest: field-aligned conservative diffusion on
    the imported FCI map, open-boundary perpendicular R-Z diffusion, a localized
    source, and edge/endpoint losses. It is a geometry-coupling gate, not a
    self-consistent turbulence claim.
    """

    maps = build_vmec_extender_fci_geometry(grid)
    R, Z, phi = _mesh(grid)
    Rn = _normalised(R)
    Zn = _normalised_symmetric(Z)
    phase = 2.0 * np.pi * (phi - grid.phi[0]) / float(grid.phi_period)
    source = source_strength * jnp.exp(-((Rn - 0.40) / 0.22) ** 2 - (Zn / 0.46) ** 2)
    source = source * (1.0 + 0.08 * jnp.cos(phase))
    radial_loss = radial_loss_rate * jnp.clip((Rn - 0.65) / 0.35, 0.0, 1.0) ** 2
    endpoint_mask = jnp.asarray(maps.forward_boundary | maps.backward_boundary, dtype=bool)
    loss_profile = radial_loss + endpoint_loss_rate * endpoint_mask.astype(jnp.float64)

    state = 1.0 + 0.045 * jnp.cos(phase) + 0.035 * jnp.sin(np.pi * Rn) * jnp.cos(np.pi * Zn)
    state = jnp.maximum(state, 1.0e-6)
    jacobian = jnp.broadcast_to(R, state.shape)
    geometry = _build_vmec_extender_geometry(grid, maps, jacobian)
    parallel_coefficient = jnp.ones_like(state) * float(parallel_diffusivity)
    dR = float(np.mean(np.diff(np.asarray(grid.R, dtype=np.float64))))
    dZ = float(np.mean(np.diff(np.asarray(grid.Z, dtype=np.float64))))

    history: list[np.ndarray] = []
    time: list[float] = []
    for frame in range(int(frames)):
        history.append(np.asarray(state, dtype=np.float64))
        time.append(float(frame * int(substeps_per_frame) * dt))
        for _ in range(int(substeps_per_frame)):
            parallel = conservative_parallel_diffusion_fci(
                state,
                parallel_coefficient,
                geometry,
            )
            perpendicular = float(perpendicular_diffusivity) * _laplace_open_RZ(state, dR=dR, dZ=dZ)
            rhs = parallel + perpendicular + source - loss_profile * state
            state = jnp.maximum(state + float(dt) * rhs, 1.0e-8)

    map_identity_error = _fci_map_identity_max_abs_error(maps)
    mode_error = _parallel_mode_decay_relative_error(
        grid,
        maps,
        diffusivity=float(parallel_diffusivity),
    )
    return VmecExtenderSolSmokeResult(
        history=np.asarray(history, dtype=np.float64),
        time=np.asarray(time, dtype=np.float64),
        source=np.asarray(source, dtype=np.float64),
        loss_profile=np.asarray(loss_profile, dtype=np.float64),
        endpoint_mask=np.asarray(endpoint_mask, dtype=bool),
        fci_map_identity_max_abs_error=float(map_identity_error),
        parallel_mode_decay_relative_error=mode_error,
    )


def build_vmec_extender_sol_smoke_report(
    grid: VmecExtenderGrid,
    result: VmecExtenderSolSmokeResult,
) -> dict[str, object]:
    """Summarise scalar SOL smoke diagnostics from an imported field grid."""

    history = np.asarray(result.history, dtype=np.float64)
    final = history[-1]
    mass = np.mean(history, axis=(1, 2, 3))
    rms = np.sqrt(np.mean((history - mass[:, None, None, None]) ** 2, axis=(1, 2, 3)))
    source_integral = float(np.mean(result.source))
    loss_integral = float(np.mean(result.loss_profile * final))
    source_loss_mismatch = abs(source_integral - loss_integral) / max(abs(source_integral), 1.0e-30)
    mode_error = result.parallel_mode_decay_relative_error
    passed = (
        bool(np.all(np.isfinite(history)))
        and float(np.min(final)) > 0.0
        and float(np.max(final)) < 2.0
        and float(rms[-1]) > 1.0e-4
        and abs(float(mass[-1] - mass[0]) / max(abs(float(mass[0])), 1.0e-30)) < 0.10
        and (mode_error is None or float(mode_error) <= 1.0e-10)
    )
    return {
        "family": "vmec_extender_sol_smoke",
        "case": "imported_field_scalar_sol_smoke",
        "source": str(grid.metadata.get("source", "unknown")),
        "grid_shape": [int(value) for value in grid.shape],
        "nfp": int(grid.nfp),
        "phi_period": float(grid.phi_period),
        "frame_count": int(history.shape[0]),
        "time_start": float(result.time[0]),
        "time_end": float(result.time[-1]),
        "final_min": float(np.min(final)),
        "final_max": float(np.max(final)),
        "final_mean": float(np.mean(final)),
        "final_rms_fluctuation": float(rms[-1]),
        "mass_relative_change": float((mass[-1] - mass[0]) / max(abs(mass[0]), 1.0e-30)),
        "source_integral": source_integral,
        "loss_integral_final": loss_integral,
        "source_loss_relative_mismatch_final": float(source_loss_mismatch),
        "endpoint_fraction": float(np.mean(result.endpoint_mask)),
        "fci_map_identity_max_abs_error": float(result.fci_map_identity_max_abs_error),
        "parallel_mode_decay_relative_error": None if mode_error is None else float(mode_error),
        "passed": bool(passed),
        "notes": (
            "Compact imported-field SOL smoke gate with conservative FCI parallel "
            "diffusion, open R-Z perpendicular diffusion, localized source, and "
            "edge/endpoint losses. It validates geometry coupling only."
        ),
    }


def save_vmec_extender_sol_smoke_plot(
    grid: VmecExtenderGrid,
    result: VmecExtenderSolSmokeResult,
    report: dict[str, object],
    path: str | Path,
) -> Path:
    """Save a compact summary figure for the imported-field SOL smoke gate."""

    history = np.asarray(result.history, dtype=np.float64)
    time = np.asarray(result.time, dtype=np.float64)
    final = history[-1]
    mass = np.mean(history, axis=(1, 2, 3))
    rms = np.sqrt(np.mean((history - mass[:, None, None, None]) ** 2, axis=(1, 2, 3)))
    R = np.asarray(grid.R, dtype=np.float64)
    Z = np.asarray(grid.Z, dtype=np.float64)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    toroidal_index = 0

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)
    image = axes[0].pcolormesh(RR, ZZ, final[:, :, toroidal_index], shading="auto", cmap="viridis")
    axes[0].contour(
        RR,
        ZZ,
        result.source[:, :, toroidal_index],
        levels=4,
        colors="white",
        linewidths=0.7,
        alpha=0.82,
    )
    axes[0].set_title(rf"final scalar, $\phi={float(grid.phi[toroidal_index]):.2f}$", fontsize=10.5)
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("Z")
    figure.colorbar(image, ax=axes[0], shrink=0.86, label="density proxy")

    axes[1].plot(time, mass, color="#005f73", lw=2.0, label="mean")
    axes[1].plot(time, rms, color="#bb3e03", lw=2.0, label="RMS")
    axes[1].plot(time, np.min(history, axis=(1, 2, 3)), color="#6a4c93", lw=1.7, label="min")
    style_axis(axes[1], title="global traces", xlabel="time", ylabel="scalar metric")
    axes[1].legend(frameon=False, fontsize=8.0)

    metric_names = ["mode\ndecay", "map\nidentity", "mass\nchange", "endpoint\nfraction"]
    mode_error = report["parallel_mode_decay_relative_error"]
    metric_values = np.asarray(
        [
            np.nan if mode_error is None else float(mode_error),
            report["fci_map_identity_max_abs_error"],
            abs(float(report["mass_relative_change"])),
            max(report["endpoint_fraction"], 1.0e-16),
        ],
        dtype=np.float64,
    )
    finite_metrics = np.where(np.isfinite(metric_values), np.maximum(metric_values, 1.0e-16), np.nan)
    axes[2].bar(np.arange(len(metric_names)), finite_metrics, color="#0a9396", alpha=0.92)
    axes[2].set_xticks(np.arange(len(metric_names)), metric_names)
    style_axis(axes[2], title="validation metrics", ylabel="value", yscale="log")
    axes[2].tick_params(axis="x", labelsize=8.0)
    axes[2].text(
        0.96,
        0.94,
        "passed" if report["passed"] else "failed",
        transform=axes[2].transAxes,
        va="top",
        ha="right",
        fontsize=9.0,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d9d9d9", "linewidth": 0.6},
    )
    figure.suptitle("VMEC-extender imported-field SOL smoke gate", fontsize=12.5, fontweight="semibold")
    save_publication_figure(figure, path)
    return Path(path)


def _mesh(grid: VmecExtenderGrid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jnp.meshgrid(grid.R, grid.Z, grid.phi, indexing="ij")


def _normalised(values: jnp.ndarray) -> jnp.ndarray:
    return (values - jnp.min(values)) / jnp.maximum(jnp.max(values) - jnp.min(values), 1.0e-30)


def _normalised_symmetric(values: jnp.ndarray) -> jnp.ndarray:
    centred = values - 0.5 * (jnp.max(values) + jnp.min(values))
    return centred / jnp.maximum(0.5 * (jnp.max(values) - jnp.min(values)), 1.0e-30)


def _laplace_open_RZ(values: jnp.ndarray, *, dR: float, dZ: float) -> jnp.ndarray:
    return _second_derivative_zero_flux(values, spacing=float(dR), axis=0) + _second_derivative_zero_flux(
        values,
        spacing=float(dZ),
        axis=1,
    )


def _second_derivative_zero_flux(values: jnp.ndarray, *, spacing: float, axis: int) -> jnp.ndarray:
    h2 = float(spacing) * float(spacing)
    centred = (jnp.roll(values, -1, axis=axis) - 2.0 * values + jnp.roll(values, 1, axis=axis)) / h2
    first = _axis_index(axis, 0)
    second = _axis_index(axis, 1)
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    first_value = 2.0 * (values[second] - values[first]) / h2
    last_value = 2.0 * (values[penultimate] - values[last]) / h2
    return centred.at[first].set(first_value).at[last].set(last_value)


def _axis_index(axis: int, index: int) -> tuple[object, object, object]:
    slices: list[object] = [slice(None), slice(None), slice(None)]
    slices[int(axis)] = index
    return tuple(slices)


def _fci_map_identity_max_abs_error(maps: FciGeometry3D) -> float:
    nx, ny, nz = maps.shape
    x = jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float64)[:, None, None], (nx, ny, nz))
    y = jnp.broadcast_to(jnp.arange(ny, dtype=jnp.float64)[None, :, None], (nx, ny, nz))
    error = jnp.max(
        jnp.asarray(
            [
                jnp.max(jnp.abs(maps.forward_x - x)),
                jnp.max(jnp.abs(maps.backward_x - x)),
                jnp.max(jnp.abs(maps.forward_y - y)),
                jnp.max(jnp.abs(maps.backward_y - y)),
            ]
        )
    )
    return float(error)


def _parallel_mode_decay_relative_error(
    grid: VmecExtenderGrid,
    maps: FciGeometry3D,
    *,
    diffusivity: float,
    mode: int = 1,
    dt: float = 0.004,
    steps: int = 12,
) -> float | None:
    if _fci_map_identity_max_abs_error(maps) > 1.0e-10:
        return None
    if bool(np.any(np.asarray(maps.forward_boundary | maps.backward_boundary, dtype=bool))):
        return None
    phi = np.asarray(grid.phi, dtype=np.float64)
    if phi.size < 4:
        return None
    dphi_values = np.diff(phi)
    if float(np.max(np.abs(dphi_values - np.mean(dphi_values)))) > 1.0e-12:
        return None
    _, _, P = _mesh(grid)
    phase = 2.0 * np.pi * int(mode) * (P - grid.phi[0]) / float(grid.phi_period)
    basis = jnp.cos(phase)
    state = basis
    coefficient = jnp.ones_like(state) * float(diffusivity)
    R, _, _ = _mesh(grid)
    jacobian = jnp.broadcast_to(R, state.shape)
    for _ in range(int(steps)):
        state = state + float(dt) * conservative_parallel_diffusion_fci(
            state,
            coefficient,
            _build_vmec_extender_geometry(grid, maps, jacobian),
        )
    amplitude = jnp.sum(state * basis) / jnp.maximum(jnp.sum(basis * basis), 1.0e-30)
    eigenvalue = -4.0 * np.sin(np.pi * int(mode) / int(phi.size)) ** 2 / (float(maps.dz) ** 2)
    expected = (1.0 + float(dt) * float(diffusivity) * eigenvalue) ** int(steps)
    return float(jnp.abs(amplitude - expected) / max(abs(expected), 1.0e-30))


def _build_vmec_extender_geometry(
    grid: VmecExtenderGrid,
    maps: FciGeometry3D,
    jacobian: jnp.ndarray,
) -> FciGeometry3D:
    return maps
