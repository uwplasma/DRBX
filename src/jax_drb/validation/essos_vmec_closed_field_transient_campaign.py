from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from ..geometry import EssosImportedFciGeometry, build_essos_imported_fci_geometry
from ..native.fci import (
    conservative_parallel_diffusion_fci,
    conservative_perp_diffusion_xz,
    grad_parallel_fci,
    logical_exb_bracket_xz,
)
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class EssosVmecClosedFieldTransientArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path
    movie_gif_path: Path | None


@dataclass(frozen=True)
class EssosVmecClosedFieldTransientDryRunArtifacts:
    contract_json_path: Path


def create_essos_vmec_closed_field_transient_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_vmec_closed_field_transient",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    nx: int = 6,
    ny: int = 10,
    nz: int = 24,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    frames: int = 14,
    substeps_per_frame: int = 3,
    dt: float = 2.0e-3,
    write_movie: bool = True,
) -> EssosVmecClosedFieldTransientArtifacts:
    """Write a live VMEC closed-field reduced-transient package."""

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
    return create_essos_vmec_closed_field_transient_package_from_geometry(
        geometry,
        output_root=output_root,
        case_label=case_label,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
        write_movie=write_movie,
    )


def create_essos_vmec_closed_field_transient_package_from_geometry(
    geometry: EssosImportedFciGeometry | Any,
    *,
    output_root: str | Path,
    case_label: str = "essos_vmec_closed_field_transient",
    frames: int = 14,
    substeps_per_frame: int = 3,
    dt: float = 2.0e-3,
    write_movie: bool = True,
) -> EssosVmecClosedFieldTransientArtifacts:
    """Write a closed-field transient package from an already-built geometry."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if write_movie:
        movies_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_essos_vmec_closed_field_transient_campaign(
        geometry,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(
        json.dumps(_strict_json_payload(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_vmec_closed_field_transient_plot(report, arrays, plot_png_path)
    movie_gif_path: Path | None = None
    if write_movie:
        movie_gif_path = movies_dir / f"{case_label}.gif"
        save_essos_vmec_closed_field_transient_movie(report, arrays, movie_gif_path)
    return EssosVmecClosedFieldTransientArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
        movie_gif_path=movie_gif_path,
    )


def create_essos_vmec_closed_field_transient_dry_run_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_vmec_closed_field_transient",
    nx: int = 6,
    ny: int = 10,
    nz: int = 24,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    frames: int = 14,
    substeps_per_frame: int = 3,
    dt: float = 2.0e-3,
    write_movie: bool = True,
) -> EssosVmecClosedFieldTransientDryRunArtifacts:
    """Write a self-contained contract for the live VMEC transient gate."""

    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "case": "essos_vmec_closed_field_transient_dry_run_contract",
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
            "movie_gif": str(root / "movies" / f"{case_label}.gif") if write_movie else None,
            "dry_run_contract_json": str(data_dir / f"{case_label}_dry_run_contract.json"),
        },
        "grid": {
            "shape": [int(nx), int(ny), int(nz)],
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
            "frames": int(frames),
            "substeps_per_frame": int(substeps_per_frame),
            "dt": float(dt),
        },
        "claim_scope": (
            "VMEC closed-field reduced transient: periodic FCI maps, profile "
            "and spectrum diagnostics, no target endpoints, no sheath losses, "
            "no recycling, and no neutral-loss semantics."
        ),
        "required_live_gates": [
            "endpoint_fraction == 0",
            "target_semantics_applied is false",
            "sheath_recycling_semantics_applied is false",
            "neutral_loss_semantics_applied is false",
            "mass_relative_drift < tolerance",
            "final_fluctuation_rms > 0",
            "spectrum_finite is true",
        ],
        "passed": True,
    }
    contract_json_path = data_dir / f"{case_label}_dry_run_contract.json"
    contract_json_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EssosVmecClosedFieldTransientDryRunArtifacts(contract_json_path=contract_json_path)


def build_essos_vmec_closed_field_transient_campaign(
    geometry: EssosImportedFciGeometry | Any,
    *,
    frames: int = 14,
    substeps_per_frame: int = 3,
    dt: float = 2.0e-3,
    parallel_diffusivity: float = 2.0e-2,
    perpendicular_diffusivity: float = 3.0e-4,
    advection_strength: float = 0.055,
    drive_strength: float = 0.025,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run a compact closed-field density transient on periodic VMEC FCI maps."""

    shape = tuple(int(value) for value in geometry.shape)
    if len(shape) != 3:
        raise ValueError(f"VMEC closed-field transient requires a 3D geometry, got {shape!r}.")
    endpoint = np.asarray(geometry.maps.forward_boundary, dtype=bool) | np.asarray(
        geometry.maps.backward_boundary,
        dtype=bool,
    )
    endpoint_fraction = float(np.mean(endpoint))
    if endpoint_fraction > 0.0:
        raise ValueError(
            "VMEC closed-field transient requires zero endpoint masks; use an open/hybrid "
            "SOL campaign for target, sheath, recycling, or neutral-loss semantics."
        )

    density = _initial_closed_field_density(geometry)
    jacobian = jnp.asarray(geometry.metric.J, dtype=jnp.float64)
    initial_mass = _weighted_sum(density, jacobian)
    history: list[np.ndarray] = []
    profile_history: list[np.ndarray] = []
    rms_history: list[float] = []
    mass_history: list[float] = []
    grad_history: list[float] = []

    def record(state: jnp.ndarray) -> None:
        profile = _radial_profile(state, jacobian)
        fluctuation = state - profile[:, None, None]
        grad = grad_parallel_fci(state, geometry.maps)
        history.append(np.asarray(fluctuation, dtype=np.float32))
        profile_history.append(np.asarray(profile, dtype=np.float64))
        rms_history.append(float(jnp.sqrt(jnp.mean(jnp.square(fluctuation)))))
        mass_history.append(float(_weighted_sum(state, jacobian)))
        grad_history.append(float(jnp.sqrt(jnp.mean(jnp.square(grad)))))

    record(density)
    for frame_index in range(int(frames)):
        for local_index in range(int(substeps_per_frame)):
            step_index = frame_index * int(substeps_per_frame) + local_index
            scalar_time = float(step_index) * float(dt)
            density = _advance_closed_field_density(
                geometry,
                density,
                scalar_time=scalar_time,
                dt=dt,
                parallel_diffusivity=parallel_diffusivity,
                perpendicular_diffusivity=perpendicular_diffusivity,
                advection_strength=advection_strength,
                drive_strength=drive_strength,
            )
        record(density)

    density_history = np.asarray(history, dtype=np.float32)
    profile_array = np.asarray(profile_history, dtype=np.float64)
    rms_array = np.asarray(rms_history, dtype=np.float64)
    mass_array = np.asarray(mass_history, dtype=np.float64)
    grad_array = np.asarray(grad_history, dtype=np.float64)
    time = np.arange(int(frames) + 1, dtype=np.float64) * float(substeps_per_frame) * float(dt)
    final_density = np.asarray(density, dtype=np.float64)
    final_profile = profile_array[-1]
    initial_profile = profile_array[0]
    final_fluctuation = final_density - final_profile[:, None, None]
    spectrum = np.abs(np.fft.rfft2(final_fluctuation[shape[0] // 2], axes=(0, 1))) ** 2
    if spectrum.size:
        spectrum[0, 0] = 0.0
    total_spectrum_power = float(np.sum(spectrum))
    low_mode_power = float(np.sum(spectrum[: min(4, spectrum.shape[0]), : min(6, spectrum.shape[1])]))
    peak_mode = (
        tuple(int(value) for value in np.unravel_index(int(np.argmax(spectrum)), spectrum.shape))
        if spectrum.size
        else (0, 0)
    )
    mass_relative_drift = float(
        np.max(np.abs(mass_array - float(initial_mass))) / max(abs(float(initial_mass)), 1.0e-30)
    )
    profile_l2_change = float(
        np.linalg.norm(final_profile - initial_profile)
        / max(float(np.linalg.norm(initial_profile)), 1.0e-30)
    )
    finite = bool(
        np.all(np.isfinite(density_history))
        and np.all(np.isfinite(profile_array))
        and np.all(np.isfinite(spectrum))
        and np.all(np.isfinite(mass_array))
    )
    bmag = np.asarray(geometry.magnetic_field_magnitude, dtype=np.float64)
    report: dict[str, Any] = {
        "case": "essos_vmec_closed_field_transient",
        "source": "JAXDRB reduced closed-field transient on ESSOS/VMEC periodic FCI maps",
        "map_source": "vmec",
        "geometry": dict(getattr(geometry, "metadata", {})),
        "shape": [int(value) for value in shape],
        "claim_scope": (
            "Closed-field VMEC control: periodic FCI coupling, profile and "
            "spectrum diagnostics, and no target/sheath/recycling/neutral-loss semantics."
        ),
        "target_semantics_applied": False,
        "sheath_recycling_semantics_applied": False,
        "neutral_loss_semantics_applied": False,
        "frames": int(frames),
        "substeps_per_frame": int(substeps_per_frame),
        "dt": float(dt),
        "parallel_diffusivity": float(parallel_diffusivity),
        "perpendicular_diffusivity": float(perpendicular_diffusivity),
        "advection_strength": float(advection_strength),
        "drive_strength": float(drive_strength),
        "endpoint_fraction": endpoint_fraction,
        "magnetic_field_modulation": float(np.max(bmag) / max(float(np.min(bmag)), 1.0e-30)),
        "initial_fluctuation_rms": float(rms_array[0]),
        "final_fluctuation_rms": float(rms_array[-1]),
        "max_fluctuation_rms": float(np.max(rms_array)),
        "mass_relative_drift": mass_relative_drift,
        "profile_l2_change": profile_l2_change,
        "final_parallel_gradient_rms": float(grad_array[-1]),
        "final_min_density": float(np.min(final_density)),
        "final_max_density": float(np.max(final_density)),
        "spectrum_finite": bool(np.all(np.isfinite(spectrum))),
        "low_mode_spectral_power_fraction": float(low_mode_power / max(total_spectrum_power, 1.0e-30)),
        "dominant_toroidal_mode_index": int(peak_mode[0]),
        "dominant_poloidal_mode_index": int(peak_mode[1]),
        "closed_field_control_ready": False,
        "open_sol_publication_ready": False,
        "open_sol_rejection_reason": "closed_vmec_map_has_no_target_endpoint_sheath_or_recycling_semantics",
    }
    report["closed_field_control_ready"] = bool(
        finite
        and endpoint_fraction < 1.0e-12
        and report["final_min_density"] > 0.0
        and report["final_fluctuation_rms"] > 1.0e-5
        and mass_relative_drift < 2.0e-2
        and report["spectrum_finite"]
        and 0.0 <= report["low_mode_spectral_power_fraction"] <= 1.0
    )
    report["passed"] = bool(report["closed_field_control_ready"])

    major_radius = np.sqrt(
        np.asarray(geometry.coordinates_x, dtype=np.float64) ** 2
        + np.asarray(geometry.coordinates_y, dtype=np.float64) ** 2
    )
    movie_vmax = float(np.nanpercentile(np.abs(density_history), 97.0))
    if not np.isfinite(movie_vmax) or movie_vmax <= 0.0:
        movie_vmax = 1.0
    arrays = {
        "time": time.astype(np.float64),
        "density_fluctuation_history": density_history,
        "profile_history": profile_array.astype(np.float32),
        "fluctuation_rms_history": rms_array.astype(np.float64),
        "mass_history": mass_array.astype(np.float64),
        "parallel_gradient_rms_history": grad_array.astype(np.float64),
        "radial_coordinate": np.mean(np.asarray(geometry.minor_radius, dtype=np.float64), axis=(1, 2)).astype(np.float64),
        "major_radius_section": major_radius[:, 0, :].astype(np.float32),
        "vertical_section": np.asarray(geometry.coordinates_z, dtype=np.float64)[:, 0, :].astype(np.float32),
        "magnetic_field_section": bmag[:, 0, :].astype(np.float32),
        "initial_density_section": np.asarray(_initial_closed_field_density(geometry), dtype=np.float64)[:, 0, :].astype(np.float32),
        "final_density_section": final_density[:, 0, :].astype(np.float32),
        "final_fluctuation_section": final_fluctuation[:, 0, :].astype(np.float32),
        "final_spectrum_log10": np.log10(spectrum + max(total_spectrum_power, 1.0e-30) * 1.0e-16).astype(np.float32),
        "movie_vmax": np.asarray([movie_vmax], dtype=np.float64),
        "summary": np.asarray(
            [
                report["final_fluctuation_rms"],
                report["mass_relative_drift"],
                report["profile_l2_change"],
                float(report["closed_field_control_ready"]),
            ],
            dtype=np.float64,
        ),
    }
    return report, arrays


def save_essos_vmec_closed_field_transient_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the closed-field transient profile/spectrum diagnostic figure."""

    major = np.asarray(arrays["major_radius_section"], dtype=np.float64)
    vertical = np.asarray(arrays["vertical_section"], dtype=np.float64)
    initial = np.asarray(arrays["initial_density_section"], dtype=np.float64)
    final = np.asarray(arrays["final_density_section"], dtype=np.float64)
    radial = np.asarray(arrays["radial_coordinate"], dtype=np.float64)
    profiles = np.asarray(arrays["profile_history"], dtype=np.float64)
    time = np.asarray(arrays["time"], dtype=np.float64)
    rms = np.asarray(arrays["fluctuation_rms_history"], dtype=np.float64)
    mass = np.asarray(arrays["mass_history"], dtype=np.float64)
    spectrum = np.asarray(arrays["final_spectrum_log10"], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.7), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(major.T, vertical.T, color="0.36", lw=0.7, alpha=0.70)
    axes[0].pcolormesh(major, vertical, final - initial, shading="gouraud", cmap="coolwarm")
    axes[0].set_aspect("equal", adjustable="box")
    style_axis(axes[0], title="closed VMEC section fluctuation", xlabel="R", ylabel="Z", grid="both")

    image = axes[1].pcolormesh(major, vertical, final, shading="gouraud", cmap="turbo")
    fig.colorbar(image, ax=axes[1], label="density")
    axes[1].set_aspect("equal", adjustable="box")
    style_axis(axes[1], title="final density on closed map", xlabel="R", ylabel="Z", grid="both")

    axes[2].plot(radial, profiles[0], lw=2.0, label="initial")
    axes[2].plot(radial, profiles[-1], lw=2.0, label="final")
    axes[2].legend(frameon=False, fontsize=9)
    style_axis(axes[2], title="radial profile", xlabel=r"$\rho$", ylabel=r"$\langle n\rangle$")

    axes[3].plot(time, rms, lw=2.0, color="#005f73", label="fluctuation RMS")
    mass_drift = (mass - mass[0]) / max(abs(float(mass[0])), 1.0e-30)
    axes[3].plot(time, mass_drift, lw=1.8, color="#bb3e03", label="relative mass drift")
    axes[3].legend(frameon=False, fontsize=9)
    style_axis(axes[3], title="closed-field scalar controls", xlabel="time")

    image = axes[4].imshow(spectrum.T, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=axes[4], label=r"$\log_{10}$ power")
    style_axis(axes[4], title="final toroidal-poloidal spectrum", xlabel="toroidal mode", ylabel="poloidal mode", grid="both")

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.96,
        "\n".join(
            [
                "VMEC closed-field reduced transient",
                f"shape: {tuple(report['shape'])}",
                f"frames: {report['frames']}",
                f"endpoint fraction: {report['endpoint_fraction']:.1e}",
                f"final RMS: {report['final_fluctuation_rms']:.2e}",
                f"mass drift: {report['mass_relative_drift']:.2e}",
                f"profile change: {report['profile_l2_change']:.2e}",
                f"closed control ready: {report['closed_field_control_ready']}",
                "No target, sheath, recycling, or neutral-loss terms.",
            ]
        ),
        transform=axes[5].transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    fig.suptitle("ESSOS VMEC closed-field transient control", fontsize=15, fontweight="semibold")
    save_publication_figure(fig, path)
    return Path(path)


def save_essos_vmec_closed_field_transient_movie(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save a fixed-camera GIF for the VMEC closed-field transient."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(arrays["density_fluctuation_history"], dtype=np.float64)
    major = np.asarray(arrays["major_radius_section"], dtype=np.float64)
    vertical = np.asarray(arrays["vertical_section"], dtype=np.float64)
    time = np.asarray(arrays["time"], dtype=np.float64)
    vmax = float(np.asarray(arrays["movie_vmax"], dtype=np.float64)[0])
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    frame_indices = np.linspace(0, history.shape[0] - 1, min(18, history.shape[0]), dtype=int)
    with tempfile.TemporaryDirectory(prefix="jax_drb_vmec_closed_movie_") as temp_dir:
        frame_paths: list[Path] = []
        for local_index, frame_index in enumerate(frame_indices):
            frame_path = Path(temp_dir) / f"frame_{local_index:03d}.png"
            fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
            image = axis.pcolormesh(
                major,
                vertical,
                history[frame_index, :, 0, :],
                shading="gouraud",
                cmap="coolwarm",
                norm=norm,
            )
            axis.plot(major[0, :], vertical[0, :], color="white", lw=1.2)
            axis.plot(major[-1, :], vertical[-1, :], color="0.20", lw=1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("R")
            axis.set_ylabel("Z")
            axis.set_title(
                "Closed VMEC field: periodic density fluctuation\n"
                f"t={time[frame_index]:.3f}, no target/sheath/recycling losses",
                fontsize=11,
            )
            fig.colorbar(image, ax=axis, label=r"$\tilde{n}$")
            fig.savefig(frame_path, dpi=150, facecolor="white")
            plt.close(fig)
            frame_paths.append(frame_path)
        first = Image.open(frame_paths[0]).convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        images = [first]
        for frame_path in frame_paths[1:]:
            images.append(Image.open(frame_path).convert("RGB").quantize(palette=first, dither=Image.Dither.NONE))
        images[0].save(resolved, save_all=True, append_images=images[1:], duration=130, loop=0)
        for image in images:
            image.close()
    return resolved


def _initial_closed_field_density(geometry: EssosImportedFciGeometry | Any) -> jnp.ndarray:
    rho = jnp.asarray(geometry.minor_radius, dtype=jnp.float64)
    theta = jnp.asarray(geometry.poloidal_angle, dtype=jnp.float64)
    phi = jnp.asarray(geometry.toroidal_angle, dtype=jnp.float64)
    radial = _normalized_radius(rho)
    envelope = jnp.exp(-jnp.square((radial - 0.55) / 0.30))
    bnorm = _normalized_field(jnp.asarray(geometry.magnetic_field_magnitude, dtype=jnp.float64))
    fluctuation = envelope * (
        jnp.sin(2.0 * theta - phi)
        + 0.32 * jnp.cos(3.0 * theta + 2.0 * phi)
        + 0.20 * bnorm
    )
    fluctuation = fluctuation / jnp.maximum(jnp.std(fluctuation), 1.0e-12)
    return 1.0 + 0.07 * (1.0 - radial) + 0.035 * fluctuation


def _advance_closed_field_density(
    geometry: EssosImportedFciGeometry | Any,
    density: jnp.ndarray,
    *,
    scalar_time: float,
    dt: float,
    parallel_diffusivity: float,
    perpendicular_diffusivity: float,
    advection_strength: float,
    drive_strength: float,
) -> jnp.ndarray:
    jacobian = jnp.asarray(geometry.metric.J, dtype=jnp.float64)
    potential = _closed_field_potential(geometry, scalar_time)
    parallel = conservative_parallel_diffusion_fci(
        density,
        jnp.ones_like(density) * float(parallel_diffusivity),
        geometry.maps,
        jacobian=jacobian,
    )
    perpendicular = conservative_perp_diffusion_xz(
        density,
        jnp.ones_like(density) * float(perpendicular_diffusivity),
        geometry.metric,
    )
    advection = -float(advection_strength) * logical_exb_bracket_xz(
        potential,
        density,
        geometry.metric,
    )
    drive = _closed_field_drive(geometry, scalar_time)
    rhs = advection + parallel + perpendicular + float(drive_strength) * drive
    rhs = _remove_weighted_mean(rhs, jacobian)
    return jnp.maximum(density + float(dt) * rhs, 1.0e-8)


def _closed_field_potential(geometry: EssosImportedFciGeometry | Any, scalar_time: float) -> jnp.ndarray:
    theta = jnp.asarray(geometry.poloidal_angle, dtype=jnp.float64)
    phi = jnp.asarray(geometry.toroidal_angle, dtype=jnp.float64)
    rho = jnp.asarray(geometry.minor_radius, dtype=jnp.float64)
    radial = _normalized_radius(rho)
    envelope = jnp.exp(-jnp.square((radial - 0.55) / 0.35))
    bnorm = _normalized_field(jnp.asarray(geometry.magnetic_field_magnitude, dtype=jnp.float64))
    return 0.060 * envelope * (
        jnp.sin(2.0 * theta - phi + 4.0 * float(scalar_time))
        + 0.28 * jnp.cos(3.0 * theta + 2.0 * phi - 2.5 * float(scalar_time))
        + 0.18 * bnorm
    )


def _closed_field_drive(geometry: EssosImportedFciGeometry | Any, scalar_time: float) -> jnp.ndarray:
    theta = jnp.asarray(geometry.poloidal_angle, dtype=jnp.float64)
    phi = jnp.asarray(geometry.toroidal_angle, dtype=jnp.float64)
    rho = jnp.asarray(geometry.minor_radius, dtype=jnp.float64)
    radial = _normalized_radius(rho)
    envelope = jnp.exp(-jnp.square((radial - 0.48) / 0.22))
    pattern = envelope * (
        jnp.sin(4.0 * theta - 3.0 * phi + 3.0 * float(scalar_time))
        + 0.35 * jnp.cos(5.0 * theta + phi - 1.5 * float(scalar_time))
    )
    return _remove_weighted_mean(pattern, jnp.asarray(geometry.metric.J, dtype=jnp.float64))


def _normalized_radius(rho: jnp.ndarray) -> jnp.ndarray:
    rho_min = jnp.min(rho)
    rho_max = jnp.max(rho)
    return (rho - rho_min) / jnp.maximum(rho_max - rho_min, 1.0e-12)


def _normalized_field(field: jnp.ndarray) -> jnp.ndarray:
    return (field - jnp.mean(field)) / jnp.maximum(jnp.std(field), 1.0e-12)


def _weighted_sum(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.asarray(values, dtype=jnp.float64) * jnp.asarray(weights, dtype=jnp.float64))


def _remove_weighted_mean(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    weights = jnp.asarray(weights, dtype=jnp.float64)
    mean = _weighted_sum(values, weights) / jnp.maximum(jnp.sum(weights), 1.0e-30)
    return jnp.asarray(values, dtype=jnp.float64) - mean


def _radial_profile(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    numerator = jnp.sum(values * weights, axis=(1, 2))
    denominator = jnp.maximum(jnp.sum(weights, axis=(1, 2)), 1.0e-30)
    return numerator / denominator


def _strict_json_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_payload(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value
