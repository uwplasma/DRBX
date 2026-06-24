from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import FciGeometry3D, build_synthetic_stellarator_geometry, logical_grid_from_axis_vectors
from ..native.fci import metric_weighted_scalar_laplacian_3d


@dataclass(frozen=True)
class StellaratorMetricMmsCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_metric_mms_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_metric_mms_campaign",
    resolutions: tuple[int, ...] = (16, 24, 32, 48),
) -> StellaratorMetricMmsCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_stellarator_metric_mms_campaign(resolutions=resolutions)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_metric_mms_plot(report, arrays, plot_png_path)
    return StellaratorMetricMmsCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_metric_mms_campaign(
    *,
    resolutions: tuple[int, ...] = (16, 24, 32, 48),
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rms_errors = []
    max_errors = []
    final_numeric = np.empty((1, 1), dtype=np.float64)
    final_exact = np.empty((1, 1), dtype=np.float64)
    final_error = np.empty((1, 1), dtype=np.float64)
    for resolution in resolutions:
        result = _identity_metric_mms_result(resolution)
        rms_errors.append(result["rms_error"])
        max_errors.append(result["max_error"])
        final_numeric = result["numeric_slice"]
        final_exact = result["exact_slice"]
        final_error = result["error_slice"]

    resolution_array = np.asarray(resolutions, dtype=np.int64)
    mesh_spacing = 1.0 / resolution_array.astype(np.float64)
    observed_order = _fit_convergence_slope(mesh_spacing, np.asarray(rms_errors, dtype=np.float64))

    synthetic_report, synthetic_arrays = _run_synthetic_metric_probe()
    report: dict[str, object] = {
        "case": "stellarator_metric_weighted_scalar_laplacian_mms",
        "operator": "J^-1 partial_i(J K g^ij partial_j f)",
        "resolutions": resolution_array.tolist(),
        "identity_mms_rms_error": rms_errors,
        "identity_mms_max_error": max_errors,
        "identity_mms_observed_order": float(observed_order),
        **synthetic_report,
    }
    report["passed"] = (
        float(report["identity_mms_observed_order"]) > 1.75
        and float(report["synthetic_constant_residual_linf"]) < 1.0e-12
        and float(report["synthetic_energy_rate"]) < 0.0
        and float(report["synthetic_energy_monotone_fraction"]) > 0.95
        and float(report["synthetic_cross_term_fraction"]) > 1.0e-3
    )
    arrays: dict[str, np.ndarray] = {
        "resolution": resolution_array,
        "mesh_spacing": mesh_spacing,
        "identity_mms_rms_error": np.asarray(rms_errors, dtype=np.float64),
        "identity_mms_max_error": np.asarray(max_errors, dtype=np.float64),
        "identity_numeric_slice": final_numeric.astype(np.float32),
        "identity_exact_slice": final_exact.astype(np.float32),
        "identity_error_slice": final_error.astype(np.float32),
        **synthetic_arrays,
    }
    return report, arrays


def save_stellarator_metric_mms_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True)

    axes[0, 0].loglog(
        arrays["mesh_spacing"],
        arrays["identity_mms_rms_error"],
        "o-",
        color="#005f73",
        linewidth=2.0,
        label="RMS error",
    )
    reference = arrays["identity_mms_rms_error"][-1] * (arrays["mesh_spacing"] / arrays["mesh_spacing"][-1]) ** 2
    axes[0, 0].loglog(arrays["mesh_spacing"], reference, "--", color="#6c757d", linewidth=1.6, label="$h^2$")
    axes[0, 0].invert_xaxis()
    axes[0, 0].set_xlabel("logical mesh spacing")
    axes[0, 0].set_ylabel("error")
    axes[0, 0].set_title("Manufactured solution: full metric scalar operator")
    axes[0, 0].legend(frameon=False)

    image0 = axes[0, 1].imshow(arrays["identity_exact_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Exact $\\nabla^2 f$ slice")
    axes[0, 1].set_xlabel("toroidal index")
    axes[0, 1].set_ylabel("radial index")
    fig.colorbar(image0, ax=axes[0, 1])

    image1 = axes[0, 2].imshow(arrays["identity_error_slice"], origin="lower", aspect="auto", cmap="coolwarm")
    axes[0, 2].set_title("MMS error slice")
    axes[0, 2].set_xlabel("toroidal index")
    axes[0, 2].set_ylabel("radial index")
    fig.colorbar(image1, ax=axes[0, 2])

    image2 = axes[1, 0].imshow(arrays["synthetic_jacobian_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 0].set_title("Synthetic non-axisymmetric Jacobian")
    axes[1, 0].set_xlabel("poloidal index")
    axes[1, 0].set_ylabel("radial index")
    fig.colorbar(image2, ax=axes[1, 0])

    image3 = axes[1, 1].imshow(arrays["synthetic_cross_term_slice"], origin="lower", aspect="auto", cmap="coolwarm")
    axes[1, 1].set_title("Full minus diagonal metric operator")
    axes[1, 1].set_xlabel("poloidal index")
    axes[1, 1].set_ylabel("radial index")
    fig.colorbar(image3, ax=axes[1, 1])

    normalized_energy = arrays["synthetic_energy_history"] / arrays["synthetic_energy_history"][0]
    axes[1, 2].plot(1.0 - normalized_energy, color="#0a9396", linewidth=2.0)
    axes[1, 2].set_xlabel("explicit metric-diffusion step")
    axes[1, 2].set_ylabel("fractional energy dissipated")
    axes[1, 2].set_title("Dissipation on synthetic stellarator metric")

    fig.suptitle(
        "Full 3D metric operator gate: "
        f"MMS order {float(report['identity_mms_observed_order']):.2f}, "
        f"cross-term fraction {100.0 * float(report['synthetic_cross_term_fraction']):.1f}%",
        fontsize=12,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _identity_metric_mms_result(resolution: int) -> dict[str, object]:
    nx = int(resolution)
    ny = int(resolution)
    nz = 2 * int(resolution)
    geometry = _identity_geometry(nx=nx, ny=ny, nz=nz)
    x = jnp.arange(nx, dtype=jnp.float64) / float(nx)
    y = 2.0 * jnp.pi * jnp.arange(ny, dtype=jnp.float64) / float(ny)
    z = 2.0 * jnp.pi * jnp.arange(nz, dtype=jnp.float64) / float(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    field = jnp.sin(2.0 * jnp.pi * X) * jnp.cos(3.0 * Y) * jnp.sin(2.0 * Z)
    exact = -((2.0 * jnp.pi) ** 2 + 3.0**2 + 2.0**2) * field
    numeric = metric_weighted_scalar_laplacian_3d(field, geometry, coefficient=1.0, periodic_axes=(True, True, True))
    error = numeric - exact
    slice_index = max(1, ny // 6)
    return {
        "rms_error": float(jnp.sqrt(jnp.mean(jnp.square(error)))),
        "max_error": float(jnp.max(jnp.abs(error))),
        "numeric_slice": np.asarray(numeric[:, slice_index, :], dtype=np.float64),
        "exact_slice": np.asarray(exact[:, slice_index, :], dtype=np.float64),
        "error_slice": np.asarray(error[:, slice_index, :], dtype=np.float64),
    }


def _run_synthetic_metric_probe() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    geometry = build_synthetic_stellarator_geometry(nx=22, ny=18, nz=36)
    field = (
        jnp.sin(jnp.pi * geometry.radial) * jnp.cos(2.0 * geometry.poloidal_angle - 5.0 * geometry.toroidal_angle)
        + 0.18 * jnp.sin(3.0 * geometry.poloidal_angle + 2.0 * geometry.toroidal_angle)
    )
    coefficient = 8.0e-4 * (1.0 + 0.25 * geometry.radial)
    constant = jnp.ones_like(field, dtype=jnp.float64)
    rhs = metric_weighted_scalar_laplacian_3d(field, geometry, coefficient)
    diagonal_rhs = metric_weighted_scalar_laplacian_3d(field, geometry, coefficient)
    cross_term = rhs - diagonal_rhs
    constant_residual = metric_weighted_scalar_laplacian_3d(constant, geometry, coefficient)
    energy_rate = jnp.mean(geometry.J * field * rhs)

    dt = 1.0e-3
    energy_history = []
    evolved = field
    for _ in range(60):
        energy_history.append(float(jnp.mean(geometry.J * jnp.square(evolved))))
        evolved = evolved + dt * metric_weighted_scalar_laplacian_3d(evolved, geometry, coefficient)
    energy_history.append(float(jnp.mean(geometry.J * jnp.square(evolved))))
    energy_history_array = np.asarray(energy_history, dtype=np.float64)
    cross_fraction = float(
        jnp.sqrt(jnp.mean(jnp.square(cross_term))) / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(rhs))), 1.0e-30)
    )
    report = {
        "synthetic_shape": list(geometry.shape),
        "synthetic_constant_residual_linf": float(jnp.max(jnp.abs(constant_residual))),
        "synthetic_energy_rate": float(energy_rate),
        "synthetic_energy_initial": float(energy_history_array[0]),
        "synthetic_energy_final": float(energy_history_array[-1]),
        "synthetic_energy_monotone_fraction": float(np.mean(np.diff(energy_history_array) <= 1.0e-14)),
        "synthetic_energy_drop_fraction": float(1.0 - energy_history_array[-1] / energy_history_array[0]),
        "synthetic_cross_term_fraction": cross_fraction,
    }
    arrays = {
        "synthetic_energy_history": energy_history_array,
        "synthetic_jacobian_slice": np.asarray(geometry.J[:, 0, :], dtype=np.float32),
        "synthetic_rhs_slice": np.asarray(rhs[:, 0, :], dtype=np.float32),
        "synthetic_cross_term_slice": np.asarray(cross_term[:, 0, :], dtype=np.float32),
        "synthetic_constant_residual_slice": np.asarray(constant_residual[:, 0, :], dtype=np.float32),
    }
    return report, arrays


def _identity_geometry(*, nx: int, ny: int, nz: int) -> FciGeometry3D:
    shape = (nx, ny, nz)
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    logical_grid = logical_grid_from_axis_vectors(
        jnp.arange(nx, dtype=jnp.float64),
        jnp.arange(ny, dtype=jnp.float64),
        jnp.arange(nz, dtype=jnp.float64),
    )
    return FciGeometry3D(
        logical_grid=logical_grid,
        forward_x=jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float64)[:, None, None], shape),
        forward_y=jnp.broadcast_to(jnp.arange(ny, dtype=jnp.float64)[None, :, None], shape),
        backward_x=jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float64)[:, None, None], shape),
        backward_y=jnp.broadcast_to(jnp.arange(ny, dtype=jnp.float64)[None, :, None], shape),
        forward_length=ones,
        backward_length=ones,
        forward_boundary=zeros.astype(bool),
        backward_boundary=zeros.astype(bool),
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        B_contravariant=jnp.zeros(shape + (3,), dtype=jnp.float64).at[..., 2].set(1.0),
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )


def _fit_convergence_slope(h: np.ndarray, error: np.ndarray) -> float:
    safe = np.maximum(error, 1.0e-16)
    slope, _ = np.polyfit(np.log(h), np.log(safe), 1)
    return float(slope)
