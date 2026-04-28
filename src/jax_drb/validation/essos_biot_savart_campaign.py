from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from ..geometry import FciMaps, FourierCoilSet, biot_savart_field, coil_axis_guess, load_essos_biot_savart_json
from ..native.fci import laplace_parallel_fci, laplace_perp_xz


ESSOS_LANDREMAN_QA_RELATIVE_JSON = Path("examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json")
_PRIVATE_DEFAULT_ESSOS_ROOT = Path.home() / "local" / "ESSOS"


@dataclass(frozen=True)
class BiotSavartAnnulusGeometry:
    """Annular FCI grid built from a coil-produced Cartesian magnetic field."""

    coordinates_x: jnp.ndarray
    coordinates_y: jnp.ndarray
    coordinates_z: jnp.ndarray
    minor_radius: jnp.ndarray
    toroidal_angle: jnp.ndarray
    poloidal_angle: jnp.ndarray
    magnetic_field: jnp.ndarray
    magnetic_field_magnitude: jnp.ndarray
    radial_field_fraction: jnp.ndarray
    curvature: jnp.ndarray
    connection_length: jnp.ndarray
    maps: FciMaps
    metadata: dict[str, float | int | str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.minor_radius.shape)


@dataclass(frozen=True)
class EssosBiotSavartCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path
    movie_gif_path: Path


def create_essos_biot_savart_campaign_package(
    *,
    output_root: str | Path,
    coil_json_path: str | Path | None = None,
    case_label: str = "essos_biot_savart_landreman_paul_qa_campaign",
    nx: int = 14,
    ny: int = 18,
    nz: int = 28,
) -> EssosBiotSavartCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path)
    coils = load_essos_biot_savart_json(resolved_coil_json)
    report, arrays, closed_geometry, open_geometry, closed_history, open_history, time = build_essos_biot_savart_campaign(
        coils=coils,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    report["coil_json_file"] = resolved_coil_json.name
    report["coil_json_format"] = "ESSOS Fourier-coil JSON"
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_biot_savart_campaign_plot(report, arrays, coils, closed_geometry, open_geometry, plot_png_path)
    movie_gif_path = movies_dir / f"{case_label}.gif"
    save_essos_biot_savart_campaign_movie(closed_geometry, open_geometry, closed_history, open_history, time, movie_gif_path)
    return EssosBiotSavartCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
        movie_gif_path=movie_gif_path,
    )


def resolve_essos_landreman_qa_json(path: str | Path | None = None) -> Path:
    """Resolve the ESSOS Landreman-Paul QA coil JSON used by the demo."""

    if path is not None:
        resolved = Path(path)
    else:
        essos_root = Path(os.environ.get("JAX_DRB_ESSOS_ROOT", _PRIVATE_DEFAULT_ESSOS_ROOT))
        resolved = essos_root / ESSOS_LANDREMAN_QA_RELATIVE_JSON
    if not resolved.exists():
        raise FileNotFoundError(
            "ESSOS Landreman-Paul QA coil JSON was not found. Pass coil_json_path "
            "or set JAX_DRB_ESSOS_ROOT to a checkout containing "
            f"{ESSOS_LANDREMAN_QA_RELATIVE_JSON}."
        )
    return resolved


def build_essos_biot_savart_campaign(
    *,
    coils: FourierCoilSet,
    nx: int = 14,
    ny: int = 18,
    nz: int = 28,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    BiotSavartAnnulusGeometry,
    BiotSavartAnnulusGeometry,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    major_radius, vertical_axis = coil_axis_guess(coils)
    closed_geometry = build_biot_savart_annulus_geometry(
        coils,
        region_label="closed_like_inner_annulus",
        radius_range=(0.18, 0.72),
        nx=nx,
        ny=ny,
        nz=nz,
        major_radius=major_radius,
        vertical_axis=vertical_axis,
    )
    open_geometry = build_biot_savart_annulus_geometry(
        coils,
        region_label="open_sol_like_outer_annulus",
        radius_range=(0.55, 1.00),
        nx=nx,
        ny=ny,
        nz=nz,
        major_radius=major_radius,
        vertical_axis=vertical_axis,
    )
    closed_history, time = simulate_biot_savart_annulus_turbulence(closed_geometry, region_kind="closed")
    open_history, _ = simulate_biot_savart_annulus_turbulence(open_geometry, region_kind="open")
    closed_metrics = _region_report(closed_geometry, closed_history, time, "closed_like_inner_annulus")
    open_metrics = _region_report(open_geometry, open_history, time, "open_sol_like_outer_annulus")
    report: dict[str, object] = {
        "case": "essos_biot_savart_landreman_paul_qa_closed_open_turbulence",
        "coil_metadata": coils.metadata,
        "axis_guess_major_radius": float(major_radius),
        "axis_guess_vertical_axis": float(vertical_axis),
        "regions": {
            "closed_like_inner_annulus": closed_metrics,
            "open_sol_like_outer_annulus": open_metrics,
        },
    }
    report["passed"] = bool(
        closed_metrics["passed"]
        and open_metrics["passed"]
        and closed_metrics["boundary_fraction"] < open_metrics["boundary_fraction"]
        and open_metrics["boundary_fraction"] > 0.0
    )
    arrays = {
        "closed_history_final": closed_history[-1].astype(np.float32),
        "open_history_final": open_history[-1].astype(np.float32),
        "closed_energy_history": _energy_history(closed_history),
        "open_energy_history": _energy_history(open_history),
        "time": time,
        "closed_B_magnitude_plane": np.asarray(closed_geometry.magnetic_field_magnitude[:, 0, :], dtype=np.float32),
        "open_B_magnitude_plane": np.asarray(open_geometry.magnetic_field_magnitude[:, 0, :], dtype=np.float32),
        "closed_radial_field_fraction_plane": np.asarray(
            closed_geometry.radial_field_fraction[:, 0, :],
            dtype=np.float32,
        ),
        "open_radial_field_fraction_plane": np.asarray(open_geometry.radial_field_fraction[:, 0, :], dtype=np.float32),
        "coil_gamma": np.asarray(coils.gamma, dtype=np.float32),
        "coil_currents": np.asarray(coils.currents, dtype=np.float64),
    }
    return report, arrays, closed_geometry, open_geometry, closed_history, open_history, time


def build_biot_savart_annulus_geometry(
    coils: FourierCoilSet,
    *,
    region_label: str,
    radius_range: tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    major_radius: float,
    vertical_axis: float,
) -> BiotSavartAnnulusGeometry:
    rho_1d = np.linspace(float(radius_range[0]), float(radius_range[1]), int(nx))
    phi_1d = np.linspace(0.0, 2.0 * np.pi, int(ny), endpoint=False)
    theta_1d = np.linspace(0.0, 2.0 * np.pi, int(nz), endpoint=False)
    rho, phi, theta = np.meshgrid(rho_1d, phi_1d, theta_1d, indexing="ij")
    major = major_radius + rho * np.cos(theta)
    vertical = vertical_axis + rho * np.sin(theta)
    x = major * np.cos(phi)
    y = major * np.sin(phi)
    points = jnp.asarray(np.stack([x, y, vertical], axis=-1), dtype=jnp.float64)
    field = biot_savart_field(coils, points)
    field_np = np.asarray(field, dtype=np.float64)
    bmag = np.linalg.norm(field_np, axis=-1)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    b_radial_cyl = cos_phi * field_np[..., 0] + sin_phi * field_np[..., 1]
    b_phi = -sin_phi * field_np[..., 0] + cos_phi * field_np[..., 1]
    b_vertical = field_np[..., 2]
    radial_field_fraction = b_radial_cyl / np.maximum(bmag, 1.0e-30)
    dphi = 2.0 * np.pi / float(ny)
    forward_x, forward_z, forward_boundary = _annular_fci_step(
        rho=rho,
        theta=theta,
        major=major,
        b_radial=b_radial_cyl,
        b_vertical=b_vertical,
        b_phi=b_phi,
        radius_range=radius_range,
        nx=nx,
        nz=nz,
        dphi=dphi,
        sign=1.0,
    )
    backward_x, backward_z, backward_boundary = _annular_fci_step(
        rho=rho,
        theta=theta,
        major=major,
        b_radial=b_radial_cyl,
        b_vertical=b_vertical,
        b_phi=b_phi,
        radius_range=radius_range,
        nx=nx,
        nz=nz,
        dphi=dphi,
        sign=-1.0,
    )
    connection_length = 2.0 * np.pi * major * bmag / np.maximum(np.abs(b_phi), 1.0e-30)
    maps = FciMaps(
        forward_x=jnp.asarray(forward_x, dtype=jnp.float64),
        forward_z=jnp.asarray(forward_z, dtype=jnp.float64),
        backward_x=jnp.asarray(backward_x, dtype=jnp.float64),
        backward_z=jnp.asarray(backward_z, dtype=jnp.float64),
        forward_boundary=jnp.asarray(forward_boundary),
        backward_boundary=jnp.asarray(backward_boundary),
        dphi=float(dphi),
    )
    return BiotSavartAnnulusGeometry(
        coordinates_x=jnp.asarray(x, dtype=jnp.float64),
        coordinates_y=jnp.asarray(y, dtype=jnp.float64),
        coordinates_z=jnp.asarray(vertical, dtype=jnp.float64),
        minor_radius=jnp.asarray(rho, dtype=jnp.float64),
        toroidal_angle=jnp.asarray(phi, dtype=jnp.float64),
        poloidal_angle=jnp.asarray(theta, dtype=jnp.float64),
        magnetic_field=field,
        magnetic_field_magnitude=jnp.asarray(bmag, dtype=jnp.float64),
        radial_field_fraction=jnp.asarray(radial_field_fraction, dtype=jnp.float64),
        curvature=jnp.asarray(radial_field_fraction, dtype=jnp.float64),
        connection_length=jnp.asarray(connection_length, dtype=jnp.float64),
        maps=maps,
        metadata={
            "geometry_family": "essos_biot_savart_annular_fci",
            "region_label": region_label,
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "radius_min": float(radius_range[0]),
            "radius_max": float(radius_range[1]),
            "major_radius": float(major_radius),
            "vertical_axis": float(vertical_axis),
        },
    )


def simulate_biot_savart_annulus_turbulence(
    geometry: BiotSavartAnnulusGeometry,
    *,
    region_kind: str,
    frames: int = 26,
    substeps_per_frame: int = 3,
    dt: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Run compact reduced turbulence dynamics on a coil-field annular FCI grid."""

    rho = np.asarray(geometry.minor_radius, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
    radial_min = float(geometry.metadata["radius_min"])
    radial_max = float(geometry.metadata["radius_max"])
    radial_width = max(radial_max - radial_min, 1.0e-12)
    radial_unit = (rho - radial_min) / radial_width
    curvature = np.asarray(geometry.curvature, dtype=np.float64)
    curvature = curvature / max(float(np.max(np.abs(curvature))), 1.0e-12)
    source_center = 0.42 if region_kind == "closed" else 0.22
    source_width = 0.20 if region_kind == "closed" else 0.16
    envelope = np.exp(-((radial_unit - source_center) / source_width) ** 2)
    seed = (
        0.15 * np.cos(2.0 * theta - 2.0 * phi)
        + 0.11 * np.sin(3.0 * theta - 4.0 * phi + 0.4)
        + 0.06 * np.cos(6.0 * theta + phi)
    )
    state = jnp.asarray(envelope * seed, dtype=jnp.float64)
    dx = radial_width / max(geometry.shape[0] - 1, 1)
    dz = 2.0 * np.pi / float(geometry.shape[2])
    curvature_jax = jnp.asarray(curvature, dtype=jnp.float64)
    envelope_jax = jnp.asarray(envelope, dtype=jnp.float64)
    endpoint_mask = jnp.asarray(np.asarray(geometry.maps.forward_boundary | geometry.maps.backward_boundary), dtype=jnp.float64)
    source_pattern = jnp.asarray(
        envelope
        * (
            0.12 * np.cos(4.0 * theta - 2.0 * phi)
            + 0.08 * np.sin(theta + 3.0 * phi)
            + 0.05 * np.cos(5.0 * theta - 5.0 * phi + 0.7)
        ),
        dtype=jnp.float64,
    )
    target_loss = 0.025 if region_kind == "closed" else 0.12
    history = []
    time = []
    for frame in range(frames):
        history.append(np.asarray(state, dtype=np.float64))
        time.append(frame * substeps_per_frame * dt)
        for substep in range(substeps_per_frame):
            current_time = (frame * substeps_per_frame + substep) * dt
            radial_gradient = (jnp.roll(state, -1, axis=0) - jnp.roll(state, 1, axis=0)) / (2.0 * dx)
            dz_adv = (jnp.roll(state, -1, axis=2) - jnp.roll(state, 1, axis=2)) / (2.0 * dz)
            source_drive = source_pattern * (1.0 + 0.28 * jnp.sin(5.0 * current_time))
            interchange = 0.18 * curvature_jax * envelope_jax * radial_gradient
            nonlinear_transfer = -0.34 * state * dz_adv
            damping = -0.045 * state - target_loss * endpoint_mask * state
            saturation = -0.36 * envelope_jax * state**3
            diffusion = 0.009 * laplace_parallel_fci(state, geometry.maps) + 2.0e-5 * laplace_perp_xz(
                state,
                dx=dx,
                dz=dz,
            )
            state = state + dt * (diffusion + source_drive + interchange + nonlinear_transfer + damping + saturation)
            state = jnp.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)
            state = jnp.clip(state, -1.25, 1.25)
    return np.asarray(history, dtype=np.float64), np.asarray(time, dtype=np.float64)


def save_essos_biot_savart_campaign_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    coils: FourierCoilSet,
    closed_geometry: BiotSavartAnnulusGeometry,
    open_geometry: BiotSavartAnnulusGeometry,
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.0, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    gamma = np.asarray(coils.gamma)
    for coil_index, curve in enumerate(gamma):
        color = "#7f5539" if np.asarray(coils.currents)[coil_index] > 0.0 else "#005f73"
        ax0.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, lw=1.2, alpha=0.86)
    _set_equal_3d(ax0, gamma.reshape((-1, 3)))
    ax0.set_title("ESSOS Fourier coils")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")

    ax1 = fig.add_subplot(grid[0, 1])
    _plot_plane(ax1, closed_geometry, arrays["closed_B_magnitude_plane"], title="Closed-like annulus |B|", cmap="magma")
    ax2 = fig.add_subplot(grid[0, 2])
    _plot_plane(ax2, open_geometry, arrays["open_B_magnitude_plane"], title="Open/SOL-like annulus |B|", cmap="magma")

    vmax = float(
        np.percentile(
            np.abs(np.concatenate([arrays["closed_history_final"].ravel(), arrays["open_history_final"].ravel()])),
            99.0,
        )
    )
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax3 = fig.add_subplot(grid[1, 0])
    _plot_plane(ax3, closed_geometry, arrays["closed_history_final"][:, 0, :], title="Closed-like final fluctuation", cmap="coolwarm", norm=norm)
    ax4 = fig.add_subplot(grid[1, 1])
    _plot_plane(ax4, open_geometry, arrays["open_history_final"][:, 0, :], title="Open/SOL-like final fluctuation", cmap="coolwarm", norm=norm)
    ax5 = fig.add_subplot(grid[1, 2])
    ax5.plot(arrays["time"], arrays["closed_energy_history"], color="#005f73", lw=2.0, label="closed-like")
    ax5.plot(arrays["time"], arrays["open_energy_history"], color="#ca6702", lw=2.0, label="open/SOL-like")
    ax5.set_xlabel("time")
    ax5.set_ylabel("mean fluctuation energy")
    ax5.set_title("Reduced turbulence response")
    ax5.legend(frameon=False)
    closed = report["regions"]["closed_like_inner_annulus"]
    open_region = report["regions"]["open_sol_like_outer_annulus"]
    fig.suptitle(
        "Coil-produced field gate: "
        f"closed boundary {100.0 * closed['boundary_fraction']:.1f}%, "
        f"open boundary {100.0 * open_region['boundary_fraction']:.1f}%",
        fontsize=12,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def save_essos_biot_savart_campaign_movie(
    closed_geometry: BiotSavartAnnulusGeometry,
    open_geometry: BiotSavartAnnulusGeometry,
    closed_history: np.ndarray,
    open_history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.percentile(np.abs(np.concatenate([closed_history.ravel(), open_history.ravel()])), 99.0))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    frames = []
    with plt.ioff():
        for index in range(0, closed_history.shape[0], 2):
            fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
            _plot_plane(
                axes[0],
                closed_geometry,
                closed_history[index, :, 0, :],
                title=f"closed-like, t={time[index]:.2f}",
                cmap="coolwarm",
                norm=norm,
            )
            _plot_plane(
                axes[1],
                open_geometry,
                open_history[index, :, 0, :],
                title=f"open/SOL-like, t={time[index]:.2f}",
                cmap="coolwarm",
                norm=norm,
            )
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("P", palette=Image.ADAPTIVE))
            plt.close(fig)
    frames[0].save(resolved, save_all=True, append_images=frames[1:], duration=125, loop=0)
    return resolved


def _annular_fci_step(
    *,
    rho: np.ndarray,
    theta: np.ndarray,
    major: np.ndarray,
    b_radial: np.ndarray,
    b_vertical: np.ndarray,
    b_phi: np.ndarray,
    radius_range: tuple[float, float],
    nx: int,
    nz: int,
    dphi: float,
    sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_bphi = np.where(np.abs(b_phi) > 1.0e-9 * np.nanmax(np.abs(b_phi)), b_phi, np.sign(b_phi + 1.0e-30) * 1.0e-9)
    dmajor = sign * major * b_radial / safe_bphi * dphi
    dvertical = sign * major * b_vertical / safe_bphi * dphi
    radial_offset = rho * np.cos(theta) + dmajor
    vertical_offset = rho * np.sin(theta) + dvertical
    rho_next = np.sqrt(radial_offset**2 + vertical_offset**2)
    theta_next = np.mod(np.arctan2(vertical_offset, radial_offset), 2.0 * np.pi)
    x_index = (rho_next - radius_range[0]) / max(radius_range[1] - radius_range[0], 1.0e-12) * float(nx - 1)
    z_index = theta_next / (2.0 * np.pi) * float(nz)
    boundary = (x_index < 0.0) | (x_index > float(nx - 1))
    return x_index, z_index, boundary


def _region_report(
    geometry: BiotSavartAnnulusGeometry,
    history: np.ndarray,
    time: np.ndarray,
    label: str,
) -> dict[str, object]:
    energy = _energy_history(history)
    final = history[-1]
    endpoint = np.asarray(geometry.maps.forward_boundary | geometry.maps.backward_boundary)
    radial = np.asarray(geometry.minor_radius)
    positive = np.maximum(final, 0.0)
    radial_center = float(np.sum(radial * positive) / max(np.sum(positive), 1.0e-12))
    radial_flux_proxy = float(np.mean(final * np.asarray(geometry.radial_field_fraction)))
    report = {
        "label": label,
        "metadata": geometry.metadata,
        "frame_count": int(history.shape[0]),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "boundary_fraction": float(np.mean(endpoint)),
        "mean_B": float(np.mean(np.asarray(geometry.magnetic_field_magnitude))),
        "min_B": float(np.min(np.asarray(geometry.magnetic_field_magnitude))),
        "max_B": float(np.max(np.asarray(geometry.magnetic_field_magnitude))),
        "radial_field_fraction_rms": float(np.sqrt(np.mean(np.asarray(geometry.radial_field_fraction) ** 2))),
        "connection_length_mean": float(np.mean(np.asarray(geometry.connection_length))),
        "energy_initial": float(energy[0]),
        "energy_final": float(energy[-1]),
        "energy_growth_factor": float(energy[-1] / max(energy[0], 1.0e-12)),
        "final_rms_fluctuation": float(np.sqrt(np.mean(final * final))),
        "positive_fluctuation_minor_radius_center": radial_center,
        "radial_flux_proxy": radial_flux_proxy,
    }
    report["passed"] = bool(
        np.isfinite(report["energy_final"])
        and report["energy_final"] > 0.0
        and report["final_rms_fluctuation"] > 1.0e-3
        and report["min_B"] > 0.0
        and np.isfinite(report["connection_length_mean"])
    )
    return report


def _energy_history(history: np.ndarray) -> np.ndarray:
    return np.mean(history * history, axis=(1, 2, 3))


def _plot_plane(axis: plt.Axes, geometry: BiotSavartAnnulusGeometry, values: np.ndarray, *, title: str, cmap: str, norm=None) -> None:
    x = np.asarray(geometry.minor_radius[:, 0, :] * np.cos(np.asarray(geometry.poloidal_angle[:, 0, :])))
    z = np.asarray(geometry.minor_radius[:, 0, :] * np.sin(np.asarray(geometry.poloidal_angle[:, 0, :])))
    image = axis.pcolormesh(x, z, values, shading="gouraud", cmap=cmap, norm=norm)
    axis.set_aspect("equal")
    axis.set_xlabel("R - R0")
    axis.set_ylabel("Z - Z0")
    axis.set_title(title)
    axis.figure.colorbar(image, ax=axis)


def _set_equal_3d(axis: plt.Axes, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)
