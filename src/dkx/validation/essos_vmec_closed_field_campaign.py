from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_essos_imported_fci_geometry
from ..native.fci import (
    conservative_parallel_diffusion_fci,
    grad_parallel_fci,
    laplace_parallel_fci,
)
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class EssosVmecClosedFieldArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosVmecClosedFieldDryRunArtifacts:
    contract_json_path: Path


def create_essos_vmec_closed_field_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_vmec_closed_field_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
) -> EssosVmecClosedFieldArtifacts:
    """Write the VMEC closed-field validation campaign package."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    geometry = build_essos_imported_fci_geometry(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source="vmec",
        nx=nx,
        ny=ny,
        nz=nz,
        rho_min=rho_min,
        rho_max=rho_max,
    )
    report, arrays = build_essos_vmec_closed_field_report(geometry)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_vmec_closed_field_plot(report, arrays, plot_png_path)
    return EssosVmecClosedFieldArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def create_essos_vmec_closed_field_dry_run_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_vmec_closed_field_campaign",
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
) -> EssosVmecClosedFieldDryRunArtifacts:
    """Write a self-contained contract for the live VMEC closed-field gate."""

    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "case": "essos_vmec_closed_field_dry_run_contract",
        "schema_version": 1,
        "self_contained": True,
        "execution_mode": "dry_run",
        "requires_essos_runtime": False,
        "live_run_requires_essos_runtime": True,
        "map_source": "vmec",
        "case_label": str(case_label),
        "output_root": str(root),
        "planned_artifacts": {
            "report_json": str(data_dir / f"{case_label}.json"),
            "arrays_npz": str(data_dir / f"{case_label}.npz"),
            "plot_png": str(root / "images" / f"{case_label}.png"),
            "dry_run_contract_json": str(data_dir / f"{case_label}_dry_run_contract.json"),
        },
        "grid": {
            "shape": [int(nx), int(ny), int(nz)],
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
        },
        "claim_scope": (
            "VMEC closed-field validation: periodic FCI maps, zero endpoint "
            "masks, constant-state operator checks, and no target/sheath/"
            "recycling/neutral-loss semantics."
        ),
        "required_live_gates": [
            "forward_boundary_fraction == 0",
            "backward_boundary_fraction == 0",
            "endpoint_fraction == 0",
            "constant_grad_parallel_linf < tolerance",
            "constant_laplace_parallel_linf < tolerance",
            "constant_conservative_parallel_diffusion_linf < tolerance",
            "magnetic_field_modulation > 1.01",
        ],
        "passed": True,
    }
    contract_json_path = data_dir / f"{case_label}_dry_run_contract.json"
    contract_json_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EssosVmecClosedFieldDryRunArtifacts(contract_json_path=contract_json_path)


def build_essos_vmec_closed_field_report(geometry: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build closed-field invariants from a VMEC-map geometry object."""

    shape = tuple(int(value) for value in geometry.shape)
    bmag = np.asarray(geometry.magnetic_field_magnitude, dtype=np.float64)
    connection = np.asarray(geometry.connection_length, dtype=np.float64)
    forward_boundary = np.asarray(geometry.maps.forward_boundary, dtype=bool)
    backward_boundary = np.asarray(geometry.maps.backward_boundary, dtype=bool)
    forward_x = np.asarray(geometry.maps.forward_x, dtype=np.float64)
    forward_z = np.asarray(geometry.maps.forward_z, dtype=np.float64)
    backward_x = np.asarray(geometry.maps.backward_x, dtype=np.float64)
    backward_z = np.asarray(geometry.maps.backward_z, dtype=np.float64)
    endpoint = forward_boundary | backward_boundary
    ones = jnp.ones(shape, dtype=jnp.float64)
    coefficient = jnp.ones(shape, dtype=jnp.float64)
    constant_grad = np.asarray(grad_parallel_fci(ones, geometry.maps), dtype=np.float64)
    constant_laplace = np.asarray(laplace_parallel_fci(ones, geometry.maps), dtype=np.float64)
    constant_diffusion = np.asarray(
        conservative_parallel_diffusion_fci(
            ones,
            coefficient,
            geometry.maps,
            jacobian=geometry.metric.J,
        ),
        dtype=np.float64,
    )

    forward_shift = _periodic_cell_delta(forward_z - np.arange(shape[2], dtype=np.float64)[None, None, :], shape[2])
    backward_shift = _periodic_cell_delta(backward_z - np.arange(shape[2], dtype=np.float64)[None, None, :], shape[2])
    report: dict[str, Any] = {
        "case": "essos_vmec_closed_field_campaign",
        "source": "ESSOS VMEC-coordinate closed-field maps with DKX FCI operator checks",
        "map_source": "vmec",
        "geometry": dict(geometry.metadata),
        "shape": [int(value) for value in shape],
        "claim_scope": (
            "Closed-field VMEC control: endpoint masks, target losses, sheath "
            "sources, recycling, and neutral-loss semantics are disabled."
        ),
        "target_semantics_applied": False,
        "sheath_recycling_semantics_applied": False,
        "neutral_loss_semantics_applied": False,
        "forward_boundary_fraction": float(np.mean(forward_boundary)),
        "backward_boundary_fraction": float(np.mean(backward_boundary)),
        "endpoint_fraction": float(np.mean(endpoint)),
        "map_coordinate_finite_fraction": float(
            np.mean(
                np.isfinite(forward_x)
                & np.isfinite(forward_z)
                & np.isfinite(backward_x)
                & np.isfinite(backward_z)
            )
        ),
        "magnetic_field_modulation": float(np.max(bmag) / max(float(np.min(bmag)), 1.0e-30)),
        "connection_length_min": float(np.min(connection)),
        "connection_length_mean": float(np.mean(connection)),
        "connection_length_max": float(np.max(connection)),
        "forward_abs_radial_shift_max": float(np.max(np.abs(forward_x - np.arange(shape[0], dtype=np.float64)[:, None, None]))),
        "backward_abs_radial_shift_max": float(np.max(np.abs(backward_x - np.arange(shape[0], dtype=np.float64)[:, None, None]))),
        "forward_abs_poloidal_shift_p95": float(np.percentile(np.abs(forward_shift), 95.0)),
        "backward_abs_poloidal_shift_p95": float(np.percentile(np.abs(backward_shift), 95.0)),
        "constant_grad_parallel_linf": float(np.max(np.abs(constant_grad))),
        "constant_laplace_parallel_linf": float(np.max(np.abs(constant_laplace))),
        "constant_conservative_parallel_diffusion_linf": float(np.max(np.abs(constant_diffusion))),
    }
    report["closed_field_semantics_passed"] = bool(
        report["forward_boundary_fraction"] < 1.0e-12
        and report["backward_boundary_fraction"] < 1.0e-12
        and report["endpoint_fraction"] < 1.0e-12
        and report["map_coordinate_finite_fraction"] == 1.0
        and report["constant_grad_parallel_linf"] < 1.0e-10
        and report["constant_laplace_parallel_linf"] < 1.0e-10
        and report["constant_conservative_parallel_diffusion_linf"] < 1.0e-10
        and report["magnetic_field_modulation"] > 1.01
        and report["connection_length_min"] > 0.0
    )
    report["passed"] = bool(report["closed_field_semantics_passed"])
    major_radius = np.sqrt(
        np.asarray(geometry.coordinates_x, dtype=np.float64) ** 2
        + np.asarray(geometry.coordinates_y, dtype=np.float64) ** 2
    )
    arrays = {
        "major_radius_section": major_radius[:, 0, :].astype(np.float32),
        "vertical_section": np.asarray(geometry.coordinates_z, dtype=np.float64)[:, 0, :].astype(np.float32),
        "magnetic_field_section": bmag[:, 0, :].astype(np.float32),
        "connection_toroidal": np.mean(connection, axis=0).astype(np.float32),
        "forward_poloidal_shift_toroidal": np.mean(forward_shift, axis=0).astype(np.float32),
        "backward_poloidal_shift_toroidal": np.mean(backward_shift, axis=0).astype(np.float32),
        "endpoint_count_toroidal": np.sum(endpoint.astype(np.float64), axis=0).astype(np.float32),
        "constant_operator_summary": np.asarray(
            [
                report["constant_grad_parallel_linf"],
                report["constant_laplace_parallel_linf"],
                report["constant_conservative_parallel_diffusion_linf"],
            ],
            dtype=np.float64,
        ),
        "summary": np.asarray(
            [
                report["magnetic_field_modulation"],
                report["connection_length_mean"],
                report["endpoint_fraction"],
                float(report["closed_field_semantics_passed"]),
            ],
            dtype=np.float64,
        ),
    }
    return report, arrays


def save_essos_vmec_closed_field_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save a compact VMEC closed-field validation figure."""

    major = np.asarray(arrays["major_radius_section"], dtype=np.float64)
    vertical = np.asarray(arrays["vertical_section"], dtype=np.float64)
    b_section = np.asarray(arrays["magnetic_field_section"], dtype=np.float64)
    connection = np.asarray(arrays["connection_toroidal"], dtype=np.float64)
    forward_shift = np.asarray(arrays["forward_poloidal_shift_toroidal"], dtype=np.float64)
    endpoint = np.asarray(arrays["endpoint_count_toroidal"], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(major.T, vertical.T, color="0.35", lw=0.7, alpha=0.72)
    axes[0].set_aspect("equal", adjustable="box")
    style_axis(axes[0], title="VMEC section geometry", xlabel="R", ylabel="Z", grid="both")

    image = axes[1].imshow(b_section.T, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=axes[1], label="|B|")
    style_axis(axes[1], title="magnetic-field section", xlabel="radial index", ylabel="poloidal index", grid="both")

    image = axes[2].imshow(connection.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(image, ax=axes[2], label="step length")
    style_axis(axes[2], title="closed-map step length", xlabel="toroidal index", ylabel="poloidal index", grid="both")

    image = axes[3].imshow(forward_shift.T, origin="lower", aspect="auto", cmap="coolwarm")
    fig.colorbar(image, ax=axes[3], label="cells")
    style_axis(axes[3], title="forward poloidal shift", xlabel="toroidal index", ylabel="poloidal index", grid="both")

    image = axes[4].imshow(endpoint.T, origin="lower", aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axes[4], label="endpoint count")
    style_axis(axes[4], title="endpoint mask must stay zero", xlabel="toroidal index", ylabel="poloidal index", grid="both")

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.96,
        "\n".join(
            [
                "VMEC closed-field control",
                f"shape: {tuple(report['shape'])}",
                f"|B| modulation: {report['magnetic_field_modulation']:.3f}",
                f"endpoint fraction: {report['endpoint_fraction']:.1e}",
                f"connection mean: {report['connection_length_mean']:.3e}",
                f"grad(1) linf: {report['constant_grad_parallel_linf']:.1e}",
                f"laplace(1) linf: {report['constant_laplace_parallel_linf']:.1e}",
                f"diffusion(1) linf: {report['constant_conservative_parallel_diffusion_linf']:.1e}",
                f"passed: {report['passed']}",
                "No target, sheath, recycling, or neutral-loss semantics.",
            ]
        ),
        transform=axes[5].transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    fig.suptitle("ESSOS VMEC closed-field FCI validation", fontsize=15, fontweight="semibold")
    save_publication_figure(fig, path)
    return Path(path)


def _periodic_cell_delta(delta: np.ndarray, period: int) -> np.ndarray:
    half = 0.5 * float(period)
    return (np.asarray(delta, dtype=np.float64) + half) % float(period) - half
