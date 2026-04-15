from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset

from .geometry_adapter import build_geometry_adapter_contract, build_geometry_adapter_manifest
from .geometry_observables import build_geometry_observable_report, profile_group_from_report, write_geometry_observable_report


@dataclass(frozen=True)
class StellaratorVmecScaffoldArtifacts:
    manifest_json_path: Path
    input_report_json_path: Path
    validation_contract_json_path: Path
    profile_report_json_path: Path
    profile_arrays_npz_path: Path
    profile_plot_png_path: Path
    surface_report_json_path: Path
    surface_arrays_npz_path: Path
    surface_plot_png_path: Path
    surface_gif_path: Path
    observable_report_json_path: Path


@dataclass(frozen=True)
class StellaratorVmecSource:
    payload: dict[str, object]
    source_format: str


def create_stellarator_vmec_scaffold_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_vmec_scaffold",
    equilibrium_path: str | Path | None = None,
) -> StellaratorVmecScaffoldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_equilibrium = Path(equilibrium_path) if equilibrium_path is not None else None
    preview_mode = resolved_equilibrium is None
    with tempfile.TemporaryDirectory(prefix="jax_drb_stellarator_vmec_") as temp_dir:
        temp_root = Path(temp_dir)
        source_path = resolved_equilibrium
        source_format = "vmec_wout_netcdf"
        if source_path is None:
            source_path = temp_root / "synthetic_vmec_wout.nc"
            _write_synthetic_vmec_wout(source_path)
            source_format = "synthetic_vmec_wout"
        source = _load_vmec_source(source_path, preview_mode=preview_mode, source_format=source_format)
        payload = source.payload

    input_report = _build_input_report(payload=payload, preview_mode=preview_mode, source_format=source.source_format)
    input_report_json_path = data_dir / f"{case_label}_input_report.json"
    input_report_json_path.write_text(json.dumps(input_report, indent=2, sort_keys=True), encoding="utf-8")

    validation_contract = _build_validation_contract()
    validation_contract_json_path = data_dir / f"{case_label}_validation_contract.json"
    validation_contract_json_path.write_text(json.dumps(validation_contract, indent=2, sort_keys=True), encoding="utf-8")

    profile_report = _build_profile_report(payload)
    profile_report_json_path = data_dir / f"{case_label}_profile_report.json"
    profile_report_json_path.write_text(json.dumps(profile_report, indent=2, sort_keys=True), encoding="utf-8")
    profile_arrays_npz_path = _write_profile_arrays(profile_report, data_dir / f"{case_label}_profile_arrays.npz")
    profile_plot_png_path = _save_profile_plot(profile_report, images_dir / f"{case_label}_profiles.png")

    surface_report, surface_arrays = _build_surface_report(payload)
    surface_report_json_path = data_dir / f"{case_label}_surface_report.json"
    surface_report_json_path.write_text(json.dumps(surface_report, indent=2, sort_keys=True), encoding="utf-8")
    surface_arrays_npz_path = data_dir / f"{case_label}_surface_arrays.npz"
    np.savez_compressed(surface_arrays_npz_path, **surface_arrays)
    surface_plot_png_path = _save_surface_plot(surface_report, surface_arrays, images_dir / f"{case_label}_surface_summary.png")
    surface_gif_path = _save_surface_gif(surface_arrays, images_dir / f"{case_label}_surface_movie.gif")

    observable_report = build_geometry_observable_report(
        geometry_family="stellarator_vmec_3d",
        benchmark_adapter="stellarator_vmec_scaffold",
        observable_groups=(
            profile_group_from_report(
                profile_report,
                name="equilibrium_profiles",
                description="Normalized-flux equilibrium profiles for rotational transform, pressure, and toroidal flux.",
            ),
            {
                "name": "flux_surface_cross_sections",
                "description": "Toroidal-angle family of sampled R-Z flux-surface cross-sections.",
                "families": [
                    {
                        "name": "sampled_flux_surfaces",
                        "kind": "surface_cross_section",
                        "coordinate_name": "toroidal_angle",
                        "field_names": ["R_surface", "Z_surface"],
                    }
                ],
            },
        ),
        metadata={
            "source_format": source.source_format,
            "nfp": int(payload["nfp"]),
            "surface_count": int(np.asarray(payload["surface_indices"]).size),
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )

    manifest = build_geometry_adapter_manifest(
        case_label=case_label,
        geometry_family="stellarator_vmec_3d",
        benchmark_adapter="stellarator_vmec_scaffold",
        preview_mode=preview_mode,
        artifacts={
            "input_report_json": str(input_report_json_path.relative_to(root)),
            "validation_contract_json": str(validation_contract_json_path.relative_to(root)),
            "profile_report_json": str(profile_report_json_path.relative_to(root)),
            "profile_arrays_npz": str(profile_arrays_npz_path.relative_to(root)),
            "profile_plot_png": str(profile_plot_png_path.relative_to(root)),
            "surface_report_json": str(surface_report_json_path.relative_to(root)),
            "surface_arrays_npz": str(surface_arrays_npz_path.relative_to(root)),
            "surface_plot_png": str(surface_plot_png_path.relative_to(root)),
            "surface_gif": str(surface_gif_path.relative_to(root)),
            "observable_report_json": str(observable_report_json_path.relative_to(root)),
        },
        metadata={"source_format": source.source_format},
    )
    manifest_json_path = data_dir / f"{case_label}_manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return StellaratorVmecScaffoldArtifacts(
        manifest_json_path=manifest_json_path,
        input_report_json_path=input_report_json_path,
        validation_contract_json_path=validation_contract_json_path,
        profile_report_json_path=profile_report_json_path,
        profile_arrays_npz_path=profile_arrays_npz_path,
        profile_plot_png_path=profile_plot_png_path,
        surface_report_json_path=surface_report_json_path,
        surface_arrays_npz_path=surface_arrays_npz_path,
        surface_plot_png_path=surface_plot_png_path,
        surface_gif_path=surface_gif_path,
        observable_report_json_path=observable_report_json_path,
    )


def _write_synthetic_vmec_wout(path: Path) -> None:
    ns = 12
    mn_mode = 4
    s = np.linspace(0.0, 1.0, ns)
    xm = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
    xn = np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    rmnc = np.zeros((ns, mn_mode), dtype=np.float64)
    zmns = np.zeros((ns, mn_mode), dtype=np.float64)
    rmnc[:, 0] = 3.8 + 0.05 * np.cos(2.0 * np.pi * s)
    rmnc[:, 1] = 0.55 * s
    rmnc[:, 2] = 0.08 * s**2
    rmnc[:, 3] = 0.03 * s**2
    zmns[:, 1] = 0.7 * s
    zmns[:, 2] = 0.06 * s**2
    zmns[:, 3] = 0.025 * s**2
    iotaf = 0.38 + 0.18 * s - 0.03 * s**2
    presf = 1.2e3 * (1.0 - s**1.6)
    phi = 2.1 * s**2
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ns", ns)
        dataset.createDimension("mn_mode", mn_mode)
        dataset.createVariable("iotaf", "f8", ("ns",))[:] = iotaf
        dataset.createVariable("presf", "f8", ("ns",))[:] = presf
        dataset.createVariable("phi", "f8", ("ns",))[:] = phi
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = xm
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = xn
        dataset.createVariable("rmnc", "f8", ("ns", "mn_mode"))[:] = rmnc
        dataset.createVariable("zmns", "f8", ("ns", "mn_mode"))[:] = zmns
        dataset.createVariable("nfp", "i4")[:] = 5


def _load_vmec_source(path: Path, *, preview_mode: bool, source_format: str) -> StellaratorVmecSource:
    if path.suffix.lower() != ".nc":
        raise ValueError(f"Unsupported stellarator equilibrium source: {path}")
    return StellaratorVmecSource(
        payload=_load_vmec_wout(path),
        source_format=source_format if preview_mode else "vmec_wout_netcdf",
    )


def _load_vmec_wout(path: Path) -> dict[str, object]:
    with Dataset(path) as dataset:
        iotaf = np.asarray(dataset.variables["iotaf"][:], dtype=np.float64)
        presf = np.asarray(dataset.variables["presf"][:], dtype=np.float64)
        phi = np.asarray(dataset.variables["phi"][:], dtype=np.float64)
        xm = np.asarray(dataset.variables["xm"][:], dtype=np.float64)
        xn = np.asarray(dataset.variables["xn"][:], dtype=np.float64)
        rmnc = np.asarray(dataset.variables["rmnc"][:], dtype=np.float64)
        zmns = np.asarray(dataset.variables["zmns"][:], dtype=np.float64)
        nfp = int(np.asarray(dataset.variables["nfp"][:]).reshape(-1)[0])
        theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        toroidal_angle = np.linspace(0.0, 2.0 * np.pi / max(nfp, 1), 24, endpoint=False)
        surface_indices = np.asarray(sorted({max(1, iotaf.size // 4), max(1, iotaf.size // 2), iotaf.size - 1}), dtype=np.int64)
        sampled_R = np.zeros((surface_indices.size, toroidal_angle.size, theta.size), dtype=np.float64)
        sampled_Z = np.zeros_like(sampled_R)
        theta_grid = theta[None, :]
        phi_grid = toroidal_angle[:, None]
        for surface_offset, surface_index in enumerate(surface_indices):
            phase = xm[:, None, None] * theta_grid - xn[:, None, None] * phi_grid
            sampled_R[surface_offset] = np.sum(rmnc[surface_index, :, None, None] * np.cos(phase), axis=0)
            sampled_Z[surface_offset] = np.sum(zmns[surface_index, :, None, None] * np.sin(phase), axis=0)
        normalized_flux = np.linspace(0.0, 1.0, iotaf.size)
        return {
            "geometry_name": path.stem,
            "coordinate_system": "vmec_flux_coordinates",
            "nfp": nfp,
            "normalized_flux": normalized_flux,
            "iota": iotaf,
            "pressure": presf,
            "toroidal_flux": phi,
            "theta": theta,
            "toroidal_angle": toroidal_angle,
            "surface_indices": surface_indices,
            "surface_R": sampled_R,
            "surface_Z": sampled_Z,
        }


def _build_input_report(*, payload: dict[str, object], preview_mode: bool, source_format: str) -> dict[str, object]:
    normalized_flux = np.asarray(payload["normalized_flux"], dtype=np.float64)
    theta = np.asarray(payload["theta"], dtype=np.float64)
    toroidal_angle = np.asarray(payload["toroidal_angle"], dtype=np.float64)
    return {
        "available": True,
        "parse_status": "ok",
        "preview_mode": preview_mode,
        "source_format": source_format,
        "geometry_name": str(payload["geometry_name"]),
        "geometry_family": "stellarator_vmec_3d",
        "coordinate_system": str(payload["coordinate_system"]),
        "nfp": int(payload["nfp"]),
        "dimensions": {
            "ns": int(normalized_flux.size),
            "ntheta": int(theta.size),
            "nphi": int(toroidal_angle.size),
            "selected_surfaces": int(np.asarray(payload["surface_indices"]).size),
        },
        "declared_profiles": ["iota", "pressure", "toroidal_flux"],
    }


def _build_validation_contract() -> dict[str, object]:
    return build_geometry_adapter_contract(
        geometry_family="stellarator_vmec_3d",
        benchmark_adapter="stellarator_vmec_scaffold",
        diagnostic_layer="shared_geometry_adapter",
        references=[
            {
                "label": "VMEC free-boundary and fixed-boundary equilibria",
                "kind": "equilibrium_source",
            },
            {
                "label": "Flux-surface geometry and profile diagnostics",
                "kind": "geometry_validation",
            },
        ],
        promotion_gates=[
            "input_report",
            "profile_bundle",
            "surface_cross_section_bundle",
            "observable_report",
            "native_execution_bundle",
        ],
        metadata={
            "profile_checks": ["monotonic_flux_coordinate", "finite_iota", "finite_pressure"],
            "surface_checks": ["finite_surface_coordinates", "positive_radial_extent", "nonzero_vertical_extent"],
        },
    )


def _build_profile_report(payload: dict[str, object]) -> dict[str, object]:
    s = np.asarray(payload["normalized_flux"], dtype=np.float64)
    iota = np.asarray(payload["iota"], dtype=np.float64)
    pressure = np.asarray(payload["pressure"], dtype=np.float64)
    toroidal_flux = np.asarray(payload["toroidal_flux"], dtype=np.float64)
    return {
        "available": True,
        "parse_status": "ok",
        "diagnostics": {
            "radial_profiles": {
                "iota": {
                    "units": "1",
                    "coordinate_name": "normalized_toroidal_flux",
                    "positions": s.tolist(),
                    "mean": iota.tolist(),
                    "std": np.zeros_like(iota).tolist(),
                },
                "pressure": {
                    "units": "Pa",
                    "coordinate_name": "normalized_toroidal_flux",
                    "positions": s.tolist(),
                    "mean": pressure.tolist(),
                    "std": np.zeros_like(pressure).tolist(),
                },
                "toroidal_flux": {
                    "units": "Wb",
                    "coordinate_name": "normalized_toroidal_flux",
                    "positions": s.tolist(),
                    "mean": toroidal_flux.tolist(),
                    "std": np.zeros_like(toroidal_flux).tolist(),
                },
            }
        },
    }


def _write_profile_arrays(report: dict[str, object], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    radial_profiles = report["diagnostics"]["radial_profiles"]
    payload = {
        "normalized_toroidal_flux": np.asarray(radial_profiles["iota"]["positions"], dtype=np.float64),
        "iota": np.asarray(radial_profiles["iota"]["mean"], dtype=np.float64),
        "pressure": np.asarray(radial_profiles["pressure"]["mean"], dtype=np.float64),
        "toroidal_flux": np.asarray(radial_profiles["toroidal_flux"]["mean"], dtype=np.float64),
    }
    np.savez_compressed(target, **payload)
    return target


def _save_profile_plot(report: dict[str, object], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    radial_profiles = report["diagnostics"]["radial_profiles"]
    s = np.asarray(radial_profiles["iota"]["positions"], dtype=np.float64)
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), constrained_layout=True, sharex=True)
    for axis, field_name, color in (
        (axes[0], "iota", "#005f73"),
        (axes[1], "pressure", "#ca6702"),
        (axes[2], "toroidal_flux", "#3a86ff"),
    ):
        values = np.asarray(radial_profiles[field_name]["mean"], dtype=np.float64)
        axis.plot(s, values, linewidth=2.2, color=color)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.set_ylabel(str(radial_profiles[field_name]["units"]))
        axis.set_title(field_name.replace("_", " ").title())
    axes[-1].set_xlabel("Normalized toroidal flux")
    figure.suptitle("Stellarator VMEC scaffold profiles", fontsize=16, fontweight="bold")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _build_surface_report(payload: dict[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    toroidal_angle = np.asarray(payload["toroidal_angle"], dtype=np.float64)
    theta = np.asarray(payload["theta"], dtype=np.float64)
    surface_indices = np.asarray(payload["surface_indices"], dtype=np.int64)
    normalized_flux = np.asarray(payload["normalized_flux"], dtype=np.float64)
    surface_r = np.asarray(payload["surface_R"], dtype=np.float64)
    surface_z = np.asarray(payload["surface_Z"], dtype=np.float64)
    frames = []
    for phi_index, angle in enumerate(toroidal_angle):
        surfaces = []
        for surface_offset, surface_index in enumerate(surface_indices):
            r_line = surface_r[surface_offset, phi_index]
            z_line = surface_z[surface_offset, phi_index]
            radial_extent = float(np.max(r_line) - np.min(r_line))
            vertical_extent = float(np.max(z_line) - np.min(z_line))
            surfaces.append(
                {
                    "surface_index": int(surface_index),
                    "normalized_flux": float(normalized_flux[surface_index]),
                    "radial_extent": radial_extent,
                    "vertical_extent": vertical_extent,
                    "elongation": float(vertical_extent / max(radial_extent, np.finfo(np.float64).tiny)),
                }
            )
        frames.append(
            {
                "toroidal_index": int(phi_index),
                "toroidal_angle": float(angle),
                "surface_summaries": surfaces,
            }
        )
    report = {
        "available": True,
        "parse_status": "ok",
        "coordinate_name": "toroidal_angle",
        "surface_indices": surface_indices.tolist(),
        "normalized_flux_values": normalized_flux[surface_indices].tolist(),
        "frames": frames,
    }
    arrays = {
        "theta": theta,
        "toroidal_angle": toroidal_angle,
        "surface_indices": surface_indices,
        "normalized_flux_values": normalized_flux[surface_indices],
        "R_surface": surface_r,
        "Z_surface": surface_z,
    }
    return report, arrays


def _save_surface_plot(report: dict[str, object], arrays: dict[str, np.ndarray], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    toroidal_angle = np.asarray(arrays["toroidal_angle"], dtype=np.float64)
    surface_r = np.asarray(arrays["R_surface"], dtype=np.float64)
    surface_z = np.asarray(arrays["Z_surface"], dtype=np.float64)
    normalized_flux_values = np.asarray(arrays["normalized_flux_values"], dtype=np.float64)
    frame_indices = np.linspace(0, toroidal_angle.size - 1, 3, dtype=int)
    figure, axes = plt.subplots(1, frame_indices.size, figsize=(13.5, 4.5), constrained_layout=True, sharex=True, sharey=True)
    colors = ("#005f73", "#ca6702", "#3a86ff", "#bb3e03")
    for axis, frame_index in zip(np.atleast_1d(axes), frame_indices, strict=False):
        for surface_offset, s_value in enumerate(normalized_flux_values):
            axis.plot(
                surface_r[surface_offset, frame_index],
                surface_z[surface_offset, frame_index],
                linewidth=2.0,
                color=colors[surface_offset % len(colors)],
                label=f"s={s_value:.2f}",
            )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.set_title(f"phi={toroidal_angle[frame_index]:.2f} rad")
        axis.set_xlabel("R [m]")
    np.atleast_1d(axes)[0].set_ylabel("Z [m]")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Stellarator VMEC sampled flux surfaces", fontsize=16, fontweight="bold")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _save_surface_gif(arrays: dict[str, np.ndarray], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    toroidal_angle = np.asarray(arrays["toroidal_angle"], dtype=np.float64)
    surface_r = np.asarray(arrays["R_surface"], dtype=np.float64)
    surface_z = np.asarray(arrays["Z_surface"], dtype=np.float64)
    normalized_flux_values = np.asarray(arrays["normalized_flux_values"], dtype=np.float64)
    colors = ("#005f73", "#ca6702", "#3a86ff", "#bb3e03")
    r_min = float(np.min(surface_r))
    r_max = float(np.max(surface_r))
    z_min = float(np.min(surface_z))
    z_max = float(np.max(surface_z))
    figure, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    lines = []
    for surface_offset, s_value in enumerate(normalized_flux_values):
        (line,) = axis.plot([], [], linewidth=2.0, color=colors[surface_offset % len(colors)], label=f"s={s_value:.2f}")
        lines.append(line)
    title = axis.set_title("")
    axis.set_xlim(r_min - 0.05, r_max + 0.05)
    axis.set_ylim(z_min - 0.05, z_max + 0.05)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.legend(frameon=False, fontsize=8, loc="upper right")

    def _update(phi_index: int):
        for surface_offset, line in enumerate(lines):
            line.set_data(surface_r[surface_offset, phi_index], surface_z[surface_offset, phi_index])
        title.set_text(f"Toroidal angle {float(toroidal_angle[phi_index]):.2f} rad")
        return tuple(lines) + (title,)

    anim = animation.FuncAnimation(
        figure,
        _update,
        frames=toroidal_angle.size,
        interval=150,
        blit=False,
    )
    anim.save(target, writer=animation.PillowWriter(fps=6))
    plt.close(figure)
    return target
