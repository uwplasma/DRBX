from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_synthetic_stellarator_geometry
from ..native.fci import (
    conservative_parallel_diffusion_fci,
    conservative_perp_diffusion_xz,
    fci_ydown,
    fci_yup,
    grad_parallel_fci,
    laplace_parallel_fci,
    laplace_perp_xz,
)


@dataclass(frozen=True)
class StellaratorFciOperatorCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_fci_operator_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_fci_operator_campaign",
) -> StellaratorFciOperatorCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_stellarator_fci_operator_campaign()
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_fci_operator_plot(report, arrays, plot_png_path)
    return StellaratorFciOperatorCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_fci_operator_campaign() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    resolutions = np.asarray([16, 24, 32, 48], dtype=np.int64)
    rms_errors = []
    max_errors = []
    grad_rms_errors = []
    for resolution in resolutions:
        geometry = build_synthetic_stellarator_geometry(nx=resolution, ny=resolution, nz=2 * resolution)
        field = _analytic_field_on_grid(geometry)
        expected_up = _analytic_field_at_maps(geometry, direction=1)
        expected_down = _analytic_field_at_maps(geometry, direction=-1)
        actual_up = np.asarray(fci_yup(jnp.asarray(field), geometry.maps))
        actual_down = np.asarray(fci_ydown(jnp.asarray(field), geometry.maps))
        expected_grad = (expected_up - expected_down) / (2.0 * geometry.maps.dphi)
        actual_grad = np.asarray(grad_parallel_fci(jnp.asarray(field), geometry.maps))
        valid = np.isfinite(expected_up) & np.isfinite(expected_down)
        interp_error = np.concatenate([(actual_up - expected_up)[valid], (actual_down - expected_down)[valid]])
        grad_error = (actual_grad - expected_grad)[valid]
        rms_errors.append(float(np.sqrt(np.mean(interp_error * interp_error))))
        max_errors.append(float(np.max(np.abs(interp_error))))
        grad_rms_errors.append(float(np.sqrt(np.mean(grad_error * grad_error))))

    h = 1.0 / resolutions.astype(np.float64)
    interp_slope = _fit_convergence_slope(h, np.asarray(rms_errors, dtype=np.float64))
    grad_slope = _fit_convergence_slope(h, np.asarray(grad_rms_errors, dtype=np.float64))

    diffusion_geometry = build_synthetic_stellarator_geometry(nx=30, ny=28, nz=56)
    diffusion_history, energy_history = _run_parallel_diffusion_probe(diffusion_geometry)
    conservative_history, conservative_energy_history, constant_residual = _run_conservative_diffusion_probe(diffusion_geometry)
    final_geometry = build_synthetic_stellarator_geometry(nx=32, ny=32, nz=64)
    final_field = _analytic_field_on_grid(final_geometry)
    final_up_error = np.asarray(
        fci_yup(jnp.asarray(final_field), final_geometry.maps)
    ) - _analytic_field_at_maps(final_geometry, direction=1)
    report = {
        "case": "non_axisymmetric_fci_operator_validation",
        "resolution": resolutions.tolist(),
        "interpolation_rms_error": rms_errors,
        "interpolation_max_error": max_errors,
        "gradient_rms_error": grad_rms_errors,
        "interpolation_convergence_slope": float(interp_slope),
        "gradient_convergence_slope": float(grad_slope),
        "diffusion_energy_initial": float(energy_history[0]),
        "diffusion_energy_final": float(energy_history[-1]),
        "diffusion_energy_monotone_fraction": float(np.mean(np.diff(energy_history) <= 1.0e-12)),
        "diffusion_energy_drop_fraction": float(1.0 - energy_history[-1] / energy_history[0]),
        "conservative_diffusion_energy_initial": float(conservative_energy_history[0]),
        "conservative_diffusion_energy_final": float(conservative_energy_history[-1]),
        "conservative_diffusion_energy_monotone_fraction": float(np.mean(np.diff(conservative_energy_history) <= 1.0e-12)),
        "conservative_diffusion_energy_drop_fraction": float(
            1.0 - conservative_energy_history[-1] / conservative_energy_history[0]
        ),
        "conservative_constant_residual_linf": float(constant_residual),
    }
    report["passed"] = (
        report["interpolation_convergence_slope"] > 1.55
        and report["gradient_convergence_slope"] > 1.35
        and report["diffusion_energy_final"] < report["diffusion_energy_initial"]
        and report["diffusion_energy_monotone_fraction"] > 0.95
        and report["conservative_diffusion_energy_final"] < report["conservative_diffusion_energy_initial"]
        and report["conservative_diffusion_energy_monotone_fraction"] > 0.95
        and report["conservative_constant_residual_linf"] < 1.0e-10
    )
    arrays = {
        "resolution": resolutions,
        "mesh_spacing": h,
        "interpolation_rms_error": np.asarray(rms_errors, dtype=np.float64),
        "interpolation_max_error": np.asarray(max_errors, dtype=np.float64),
        "gradient_rms_error": np.asarray(grad_rms_errors, dtype=np.float64),
        "diffusion_final_slice": diffusion_history[-1, :, 0, :].astype(np.float32),
        "diffusion_energy_history": energy_history,
        "conservative_diffusion_final_slice": conservative_history[-1, :, 0, :].astype(np.float32),
        "conservative_diffusion_energy_history": conservative_energy_history,
        "operator_error_slice": final_up_error[:, 0, :].astype(np.float32),
        "operator_field_slice": final_field[:, 0, :].astype(np.float32),
    }
    return report, arrays


def save_stellarator_fci_operator_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True)
    axes[0, 0].loglog(
        arrays["mesh_spacing"],
        arrays["interpolation_rms_error"],
        "o-",
        color="#005f73",
        linewidth=2.0,
        label="interpolation",
    )
    axes[0, 0].loglog(
        arrays["mesh_spacing"],
        arrays["gradient_rms_error"],
        "s-",
        color="#ca6702",
        linewidth=2.0,
        label="parallel gradient",
    )
    axes[0, 0].invert_xaxis()
    axes[0, 0].set_xlabel("logical mesh spacing")
    axes[0, 0].set_ylabel("RMS error")
    axes[0, 0].set_title("FCI operator convergence on analytic field-line maps")
    axes[0, 0].legend(frameon=False)

    time_index = np.arange(arrays["diffusion_energy_history"].size)
    axes[0, 1].plot(time_index, arrays["diffusion_energy_history"], color="#9b2226", linewidth=2.0)
    axes[0, 1].set_xlabel("explicit diffusion step")
    axes[0, 1].set_ylabel("mean fluctuation energy")
    axes[0, 1].set_title("compact parallel diffusion energy")

    conservative_time_index = np.arange(arrays["conservative_diffusion_energy_history"].size)
    axes[0, 2].plot(
        conservative_time_index,
        arrays["conservative_diffusion_energy_history"],
        color="#0a9396",
        linewidth=2.0,
    )
    axes[0, 2].set_xlabel("explicit diffusion step")
    axes[0, 2].set_ylabel("mean fluctuation energy")
    axes[0, 2].set_title("metric-weighted conservative diffusion")

    image0 = axes[1, 0].imshow(arrays["operator_field_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[1, 0].set_title("Analytic validation field at one toroidal plane")
    axes[1, 0].set_xlabel("poloidal index")
    axes[1, 0].set_ylabel("radial index")
    fig.colorbar(image0, ax=axes[1, 0])

    image1 = axes[1, 1].imshow(arrays["operator_error_slice"], origin="lower", aspect="auto", cmap="coolwarm")
    axes[1, 1].set_title("Forward-map interpolation error")
    axes[1, 1].set_xlabel("poloidal index")
    axes[1, 1].set_ylabel("radial index")
    fig.colorbar(image1, ax=axes[1, 1])

    image2 = axes[1, 2].imshow(arrays["conservative_diffusion_final_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 2].set_title("Final conservative-diffusion field")
    axes[1, 2].set_xlabel("poloidal index")
    axes[1, 2].set_ylabel("radial index")
    fig.colorbar(image2, ax=axes[1, 2])
    fig.suptitle(
        "3D FCI operator gate: "
        f"slope {float(report['interpolation_convergence_slope']):.2f}, "
        f"conservative drop {100.0 * float(report['conservative_diffusion_energy_drop_fraction']):.1f}%",
        fontsize=12,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _analytic_field_on_grid(geometry: object) -> np.ndarray:
    s = np.asarray(geometry.radial, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    return _analytic_field(s=s, phi=phi, theta=theta)


def _analytic_field_at_maps(geometry: object, *, direction: int) -> np.ndarray:
    maps = geometry.maps
    nx, ny, nz = maps.shape
    if direction > 0:
        x_index = np.asarray(maps.forward_x, dtype=np.float64)
        z_index = np.asarray(maps.forward_z, dtype=np.float64)
        y_index = (np.arange(ny)[None, :, None] + 1) % ny
    else:
        x_index = np.asarray(maps.backward_x, dtype=np.float64)
        z_index = np.asarray(maps.backward_z, dtype=np.float64)
        y_index = (np.arange(ny)[None, :, None] - 1) % ny
    s = 0.08 + np.clip(x_index, 0.0, nx - 1.0) / float(nx - 1) * (1.0 - 0.08)
    phi = 2.0 * np.pi * y_index / float(ny)
    theta = 2.0 * np.pi * np.mod(z_index, nz) / float(nz)
    values = _analytic_field(s=s, phi=phi, theta=theta)
    valid = (x_index >= 0.0) & (x_index <= nx - 1.0)
    return np.where(valid, values, 0.0)


def _analytic_field(*, s: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (
        np.sin(np.pi * s) * np.cos(2.0 * theta - 5.0 * phi)
        + 0.28 * s**2 * np.sin(3.0 * theta + 2.0 * phi)
        + 0.11 * np.cos(theta - phi) * (1.0 - s)
    )


def _run_parallel_diffusion_probe(geometry: object) -> tuple[np.ndarray, np.ndarray]:
    field = jnp.asarray(_analytic_field_on_grid(geometry), dtype=jnp.float64)
    dx = float(1.0 / (geometry.maps.shape[0] - 1))
    dz = float(2.0 * np.pi / geometry.maps.shape[2])
    dt = 3.0e-4
    chi_parallel = 0.018
    chi_perp = 4.0e-5
    snapshots = []
    energy = []
    for step in range(80):
        if step % 4 == 0:
            snapshots.append(np.asarray(field))
            energy.append(float(jnp.mean(jnp.square(field))))
        rhs = chi_parallel * laplace_parallel_fci(field, geometry.maps) + chi_perp * laplace_perp_xz(
            field,
            dx=dx,
            dz=dz,
        )
        field = field + dt * rhs
    snapshots.append(np.asarray(field))
    energy.append(float(jnp.mean(jnp.square(field))))
    return np.asarray(snapshots, dtype=np.float64), np.asarray(energy, dtype=np.float64)


def _run_conservative_diffusion_probe(geometry: object) -> tuple[np.ndarray, np.ndarray, float]:
    field = jnp.asarray(_analytic_field_on_grid(geometry), dtype=jnp.float64)
    radial = jnp.asarray(geometry.radial, dtype=jnp.float64)
    coefficient = 0.015 + 0.010 * radial
    constant = jnp.ones_like(field, dtype=jnp.float64)
    constant_residual = jnp.max(
        jnp.abs(
            conservative_parallel_diffusion_fci(
                constant,
                coefficient,
                geometry.maps,
                jacobian=geometry.metric.J,
            )
            + conservative_perp_diffusion_xz(constant, 2.5e-4 * coefficient, geometry.metric)
        )
    )
    dt = 1.0e-4
    snapshots = []
    energy = []
    for step in range(80):
        if step % 4 == 0:
            snapshots.append(np.asarray(field))
            energy.append(float(jnp.mean(jnp.square(field))))
        rhs = conservative_parallel_diffusion_fci(
            field,
            coefficient,
            geometry.maps,
            jacobian=geometry.metric.J,
        ) + conservative_perp_diffusion_xz(field, 2.5e-4 * coefficient, geometry.metric)
        field = field + dt * rhs
    snapshots.append(np.asarray(field))
    energy.append(float(jnp.mean(jnp.square(field))))
    return np.asarray(snapshots, dtype=np.float64), np.asarray(energy, dtype=np.float64), float(constant_residual)


def _fit_convergence_slope(h: np.ndarray, error: np.ndarray) -> float:
    safe = np.maximum(error, 1.0e-16)
    slope, _ = np.polyfit(np.log(h), np.log(safe), 1)
    return float(slope)
