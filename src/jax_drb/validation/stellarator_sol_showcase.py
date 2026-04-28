from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from ..geometry import SyntheticStellaratorGeometry, build_synthetic_stellarator_geometry
from ..native.fci import laplace_parallel_fci, laplace_perp_xz


@dataclass(frozen=True)
class StellaratorSolShowcaseArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    snapshot_png_path: Path
    diagnostics_png_path: Path
    poster_png_path: Path
    movie_gif_path: Path


def create_stellarator_sol_showcase_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_sol_showcase",
    nx: int = 30,
    ny: int = 30,
    nz: int = 56,
) -> StellaratorSolShowcaseArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    history, time = simulate_reduced_stellarator_sol_dynamics(geometry)
    report = build_stellarator_sol_showcase_report(geometry, history, time)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    _write_showcase_arrays(geometry, history, time, arrays_npz_path)
    snapshot_png_path = images_dir / f"{case_label}_snapshots.png"
    save_stellarator_sol_snapshot_panel(geometry, history, time, snapshot_png_path)
    diagnostics_png_path = images_dir / f"{case_label}_diagnostics.png"
    save_stellarator_sol_diagnostics_panel(geometry, history, time, diagnostics_png_path)
    poster_png_path = images_dir / f"{case_label}_poster.png"
    save_stellarator_sol_3d_frame(geometry, history[-1], float(time[-1]), poster_png_path)
    movie_gif_path = movies_dir / f"{case_label}.gif"
    save_stellarator_sol_3d_movie(geometry, history, time, movie_gif_path)
    return StellaratorSolShowcaseArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        snapshot_png_path=snapshot_png_path,
        diagnostics_png_path=diagnostics_png_path,
        poster_png_path=poster_png_path,
        movie_gif_path=movie_gif_path,
    )


def simulate_reduced_stellarator_sol_dynamics(
    geometry: SyntheticStellaratorGeometry,
    *,
    frames: int = 34,
    substeps_per_frame: int = 3,
    dt: float = 0.010,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a compact reduced 3D SOL dynamics benchmark on the FCI geometry."""

    radial = np.asarray(geometry.radial, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    curvature = np.asarray(geometry.curvature, dtype=np.float64)
    curvature = curvature / max(float(np.max(np.abs(curvature))), 1.0e-12)
    envelope = np.exp(-((radial - 0.73) / 0.17) ** 2)
    source_envelope = np.exp(-((radial - 0.60) / 0.12) ** 2)
    seed_modes = (
        0.18 * np.cos(2.0 * theta - 5.0 * phi)
        + 0.13 * np.sin(4.0 * theta - 5.0 * phi + 0.7)
        + 0.08 * np.cos(7.0 * theta - 10.0 * phi + 1.4)
        + 0.05 * np.sin(11.0 * theta + 3.0 * phi)
    )
    state = jnp.asarray(envelope * seed_modes, dtype=jnp.float64)
    dx = float(1.0 / (geometry.maps.shape[0] - 1))
    dz = float(2.0 * np.pi / geometry.maps.shape[2])
    curvature_jax = jnp.asarray(curvature, dtype=jnp.float64)
    envelope_jax = jnp.asarray(envelope, dtype=jnp.float64)
    source_pattern = jnp.asarray(
        source_envelope
        * (
            0.160 * np.cos(4.0 * theta - 5.0 * phi)
            + 0.105 * np.sin(2.0 * theta + 5.0 * phi)
            + 0.070 * np.cos(6.0 * theta - 10.0 * phi + 0.3)
        ),
        dtype=jnp.float64,
    )
    history = []
    time = []
    for frame in range(frames):
        history.append(np.asarray(state, dtype=np.float64))
        time.append(frame * substeps_per_frame * dt)
        for substep in range(substeps_per_frame):
            current_time = (frame * substeps_per_frame + substep) * dt
            dz_adv = (jnp.roll(state, -1, axis=2) - jnp.roll(state, 1, axis=2)) / (2.0 * dz)
            dy_adv = (jnp.roll(state, -1, axis=1) - jnp.roll(state, 1, axis=1)) / (2.0 * geometry.maps.dphi)
            radial_gradient = (jnp.roll(state, -1, axis=0) - jnp.roll(state, 1, axis=0)) / (2.0 * dx)
            nonlinear_transfer = -0.42 * state * dz_adv + 0.24 * jnp.roll(state, 2, axis=2) * dy_adv
            interchange_drive = 0.28 * curvature_jax * envelope_jax * radial_gradient
            source_drive = source_pattern * (1.0 + 0.35 * jnp.sin(7.5 * current_time))
            coherent_drive = 0.18 * curvature_jax * envelope_jax * (state + source_drive)
            saturation = -0.48 * envelope_jax * state * state * state
            damping = -0.035 * (1.0 - 0.25 * envelope_jax) * state
            diffusion = 0.010 * laplace_parallel_fci(state, geometry.maps) + 1.8e-5 * laplace_perp_xz(
                state,
                dx=dx,
                dz=dz,
            )
            state = state + dt * (
                diffusion
                + interchange_drive
                + coherent_drive
                + nonlinear_transfer
                + saturation
                + damping
                + source_drive
            )
            state = jnp.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)
            state = jnp.clip(state, -1.5, 1.5)
    return np.asarray(history, dtype=np.float64), np.asarray(time, dtype=np.float64)


def build_stellarator_sol_showcase_report(
    geometry: SyntheticStellaratorGeometry,
    history: np.ndarray,
    time: np.ndarray,
) -> dict[str, object]:
    energy = np.mean(history * history, axis=(1, 2, 3))
    final = history[-1]
    radial = np.asarray(geometry.radial, dtype=np.float64)
    curvature = np.asarray(geometry.curvature, dtype=np.float64)
    positive = np.maximum(final, 0.0)
    radial_center = float(np.sum(radial * positive) / max(np.sum(positive), 1.0e-12))
    potential_proxy = np.roll(final, 2, axis=2)
    dtheta_phi = -(np.roll(potential_proxy, -1, axis=2) - np.roll(potential_proxy, 1, axis=2)) / (
        2.0 * 2.0 * np.pi / geometry.maps.shape[2]
    )
    radial_flux_proxy = float(np.mean(final * dtheta_phi * curvature))
    spectrum = np.abs(np.fft.rfftn(final, axes=(1, 2))) ** 2
    total_spectral_power = float(np.sum(spectrum))
    low_mode_power = float(np.sum(spectrum[:, :4, :6]) / max(total_spectral_power, 1.0e-12))
    mode_power = np.mean(np.abs(np.fft.rfftn(final, axes=(1, 2))) ** 2, axis=0)
    mode_power[0, 0] = 0.0
    peak_mode = np.unravel_index(int(np.argmax(mode_power)), mode_power.shape)
    cell_rms = np.std(history, axis=0)
    cell_skewness = _cell_moment(history, 3)
    report = {
        "case": "reduced_non_axisymmetric_stellarator_sol_dynamics",
        "geometry": geometry.metadata,
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "frame_count": int(history.shape[0]),
        "energy_initial": float(energy[0]),
        "energy_final": float(energy[-1]),
        "energy_peak": float(np.max(energy)),
        "energy_growth_factor": float(energy[-1] / max(energy[0], 1.0e-12)),
        "final_rms_fluctuation": float(np.sqrt(np.mean(final * final))),
        "final_skewness": _moment(final, 3),
        "final_kurtosis": _moment(final, 4),
        "positive_fluctuation_radial_center": radial_center,
        "radial_flux_proxy": radial_flux_proxy,
        "low_mode_spectral_power_fraction": low_mode_power,
        "dominant_poloidal_mode_index": int(peak_mode[1]),
        "dominant_toroidal_mode_index": int(peak_mode[0]),
        "mean_cell_rms_fluctuation": float(np.mean(cell_rms)),
        "max_cell_rms_fluctuation": float(np.max(cell_rms)),
        "skewness_linf": float(np.max(np.abs(cell_skewness))),
        "connection_length_weighted_rms": float(
            np.sqrt(
                np.mean(
                    final
                    * final
                    * np.asarray(geometry.connection_length)
                    / np.mean(np.asarray(geometry.connection_length))
                )
            )
        ),
    }
    report["passed"] = (
        np.isfinite(report["energy_final"])
        and report["energy_final"] > 0.0
        and report["final_rms_fluctuation"] > 1.0e-3
        and abs(report["radial_flux_proxy"]) > 1.0e-5
        and 0.0 < report["low_mode_spectral_power_fraction"] < 1.0
    )
    return report


def save_stellarator_sol_snapshot_panel(
    geometry: SyntheticStellaratorGeometry,
    history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    time_indices = np.asarray([0, history.shape[0] // 2, history.shape[0] - 1], dtype=int)
    toroidal_indices = np.asarray(
        [0, geometry.shape[1] // 4, geometry.shape[1] // 2, 3 * geometry.shape[1] // 4],
        dtype=int,
    )
    x = np.asarray(geometry.coordinates_x, dtype=np.float64)
    y = np.asarray(geometry.coordinates_y, dtype=np.float64)
    major_radius = np.sqrt(x * x + y * y)
    z = np.asarray(geometry.coordinates_z, dtype=np.float64)
    vmax = float(np.percentile(np.abs(history), 99.0))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.2), constrained_layout=True)
    image = None
    for row, time_index in enumerate(time_indices):
        for col, toroidal_index in enumerate(toroidal_indices):
            axis = axes[row, col]
            image = axis.pcolormesh(
                major_radius[:, toroidal_index, :],
                z[:, toroidal_index, :],
                history[time_index, :, toroidal_index, :],
                shading="gouraud",
                cmap="coolwarm",
                norm=norm,
            )
            axis.plot(major_radius[0, toroidal_index, :], z[0, toroidal_index, :], color="white", lw=2.0)
            axis.plot(major_radius[-1, toroidal_index, :], z[-1, toroidal_index, :], color="0.35", lw=1.4)
            axis.set_aspect("equal", adjustable="box")
            phi_value = 2.0 * np.pi * toroidal_index / geometry.shape[1]
            axis.set_title(rf"$t={time[time_index]:.2f}$, $\phi={phi_value:.2f}$")
            axis.set_xlabel("R")
            axis.set_ylabel("Z")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.72, label="density fluctuation proxy")
    fig.suptitle(
        "Reduced 3D stellarator SOL dynamics: R-Z panels at four toroidal angles",
        fontsize=15,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def save_stellarator_sol_diagnostics_panel(
    geometry: SyntheticStellaratorGeometry,
    history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    toroidal_index = 0
    x = np.asarray(geometry.coordinates_x, dtype=np.float64)
    y = np.asarray(geometry.coordinates_y, dtype=np.float64)
    major_radius = np.sqrt(x * x + y * y)
    z = np.asarray(geometry.coordinates_z, dtype=np.float64)
    final = history[-1]
    rms = np.std(history, axis=0)
    skewness = _cell_moment(history, 3)
    potential_proxy = np.roll(history, 2, axis=3)
    dtheta = 2.0 * np.pi / geometry.shape[2]
    radial_velocity_proxy = -(
        np.roll(potential_proxy, -1, axis=3) - np.roll(potential_proxy, 1, axis=3)
    ) / (2.0 * dtheta)
    flux_proxy = np.mean(history * radial_velocity_proxy * np.asarray(geometry.curvature)[None, :, :, :], axis=0)
    spectrum = np.mean(np.abs(np.fft.rfftn(final, axes=(1, 2))) ** 2, axis=0)
    spectrum[0, 0] = 0.0
    energy = np.mean(history * history, axis=(1, 2, 3))

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    density_proxy = 1.0 + 0.35 * final
    image0 = axes[0, 0].pcolormesh(
        major_radius[:, toroidal_index, :],
        z[:, toroidal_index, :],
        density_proxy[:, toroidal_index, :],
        shading="gouraud",
        cmap="turbo",
    )
    axes[0, 0].set_title("final density proxy")
    fig.colorbar(image0, ax=axes[0, 0], label="n / n0")

    image1 = axes[0, 1].pcolormesh(
        major_radius[:, toroidal_index, :],
        z[:, toroidal_index, :],
        rms[:, toroidal_index, :],
        shading="gouraud",
        cmap="magma",
    )
    axes[0, 1].set_title("fluctuation RMS")
    fig.colorbar(image1, ax=axes[0, 1], label="std(n~)")

    skew_vmax = float(np.nanpercentile(np.abs(skewness), 98.0))
    image2 = axes[0, 2].pcolormesh(
        major_radius[:, toroidal_index, :],
        z[:, toroidal_index, :],
        skewness[:, toroidal_index, :],
        shading="gouraud",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-skew_vmax, vcenter=0.0, vmax=skew_vmax),
    )
    axes[0, 2].set_title("density skewness")
    fig.colorbar(image2, ax=axes[0, 2], label="skew(n~)")

    flux_vmax = float(np.nanpercentile(np.abs(flux_proxy), 98.0))
    image3 = axes[1, 0].pcolormesh(
        major_radius[:, toroidal_index, :],
        z[:, toroidal_index, :],
        flux_proxy[:, toroidal_index, :],
        shading="gouraud",
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-flux_vmax, vcenter=0.0, vmax=flux_vmax),
    )
    axes[1, 0].set_title("radial flux proxy")
    fig.colorbar(image3, ax=axes[1, 0], label="<n~ vR~>")

    mode_image = axes[1, 1].imshow(
        np.log10(np.maximum(spectrum.T, 1.0e-16)),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 1].set_title("final toroidal-poloidal spectrum")
    axes[1, 1].set_xlabel("toroidal mode index")
    axes[1, 1].set_ylabel("poloidal mode index")
    fig.colorbar(mode_image, ax=axes[1, 1], label="log10 power")

    axes[1, 2].plot(time, energy, color="#005f73", lw=2.4, label="energy")
    axes[1, 2].plot(time, np.sqrt(energy), color="#9b2226", lw=2.0, label="RMS")
    axes[1, 2].set_title("time traces")
    axes[1, 2].set_xlabel("normalized time")
    axes[1, 2].set_ylabel("global metric")
    axes[1, 2].legend(frameon=False)
    for axis in axes[:, :].flat:
        if axis is not axes[1, 1] and axis is not axes[1, 2]:
            axis.plot(major_radius[0, toroidal_index, :], z[0, toroidal_index, :], color="white", lw=1.6)
            axis.plot(major_radius[-1, toroidal_index, :], z[-1, toroidal_index, :], color="0.25", lw=1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("R")
            axis.set_ylabel("Z")
    fig.suptitle("Reduced stellarator SOL diagnostics: fluctuations, flux proxy, and mode content", fontsize=15)
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def save_stellarator_sol_3d_movie(
    geometry: SyntheticStellaratorGeometry,
    history: np.ndarray,
    time: np.ndarray,
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = np.linspace(0, history.shape[0] - 1, min(24, history.shape[0]), dtype=int)
    with tempfile.TemporaryDirectory(prefix="jax_drb_stellarator_movie_") as temp_dir:
        frame_paths = []
        for local_index, frame_index in enumerate(frame_indices):
            frame_path = Path(temp_dir) / f"frame_{local_index:03d}.png"
            save_stellarator_sol_3d_frame(geometry, history[frame_index], float(time[frame_index]), frame_path)
            frame_paths.append(frame_path)
        images = [Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE) for frame_path in frame_paths]
        images[0].save(resolved, save_all=True, append_images=images[1:], duration=120, loop=0)
        for image in images:
            image.close()
    return resolved


def save_stellarator_sol_3d_frame(
    geometry: SyntheticStellaratorGeometry,
    field: np.ndarray,
    time_value: float,
    path: str | Path,
) -> Path:
    if max(geometry.shape) >= 24:
        try:
            return _save_stellarator_sol_3d_frame_pyvista(geometry, field, time_value, path)
        except Exception:
            pass
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(geometry.coordinates_x)
    y = np.asarray(geometry.coordinates_y)
    z = np.asarray(geometry.coordinates_z)
    values = np.asarray(field, dtype=np.float64)
    vmax = float(np.percentile(np.abs(values), 99.0))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    phi_indices = np.arange(2, geometry.shape[1] - 5, 2)
    theta_indices = np.arange(0, geometry.shape[2], 2)
    radial_indices = np.arange(2, geometry.shape[0] - 1, 2)
    cut_j = 3
    outer_i = max(geometry.shape[0] - 4, 1)

    fig = plt.figure(figsize=(8.0, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface_values = values[np.ix_([outer_i], phi_indices, theta_indices)][0]
    ax.plot_surface(
        x[np.ix_([outer_i], phi_indices, theta_indices)][0],
        y[np.ix_([outer_i], phi_indices, theta_indices)][0],
        z[np.ix_([outer_i], phi_indices, theta_indices)][0],
        facecolors=cmap(norm(surface_values)),
        linewidth=0,
        antialiased=False,
        alpha=0.86,
        shade=False,
    )
    cut_values = values[np.ix_(radial_indices, [cut_j], theta_indices)][:, 0, :]
    ax.plot_surface(
        x[np.ix_(radial_indices, [cut_j], theta_indices)][:, 0, :],
        y[np.ix_(radial_indices, [cut_j], theta_indices)][:, 0, :],
        z[np.ix_(radial_indices, [cut_j], theta_indices)][:, 0, :],
        facecolors=cmap(norm(cut_values)),
        linewidth=0,
        antialiased=False,
        alpha=0.96,
        shade=False,
    )
    inner_i = max(geometry.shape[0] // 3, 1)
    ax.plot_wireframe(
        x[np.ix_([inner_i], phi_indices[::2], theta_indices[::3])][0],
        y[np.ix_([inner_i], phi_indices[::2], theta_indices[::3])][0],
        z[np.ix_([inner_i], phi_indices[::2], theta_indices[::3])][0],
        color="0.25",
        linewidth=0.35,
        alpha=0.28,
    )
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, shrink=0.72, pad=0.02, label="density fluctuation proxy")
    ax.set_title(
        "Native reduced FCI stellarator SOL dynamics\n"
        f"opened toroidal/radial view, t = {time_value:.2f}",
        fontsize=11,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22.0, azim=-48.0 + 18.0 * np.sin(0.7 * time_value))
    extent = float(np.max(np.sqrt(x * x + y * y)))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(float(np.min(z)) * 1.1, float(np.max(z)) * 1.1)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    fig.savefig(resolved, dpi=160)
    plt.close(fig)
    return resolved


def _save_stellarator_sol_3d_frame_pyvista(
    geometry: SyntheticStellaratorGeometry,
    field: np.ndarray,
    time_value: float,
    path: str | Path,
) -> Path:
    import pyvista as pv

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(geometry.coordinates_x)
    y = np.asarray(geometry.coordinates_y)
    z = np.asarray(geometry.coordinates_z)
    values = np.asarray(field, dtype=np.float64)
    vmax = float(np.nanpercentile(np.abs(values), 99.2))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    scalar_name = "density fluctuation proxy"
    ny = geometry.shape[1]
    phi_window = np.arange(max(2, ny // 10), max(3, ny - ny // 5), dtype=int)
    theta_window = np.arange(0, geometry.shape[2], 1, dtype=int)
    radial_window = np.arange(1, geometry.shape[0] - 1, 1, dtype=int)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")

    def add_surface(
        x_surface: np.ndarray,
        y_surface: np.ndarray,
        z_surface: np.ndarray,
        scalar_values: np.ndarray,
        *,
        opacity: float,
        show_scalar_bar: bool,
    ) -> None:
        mesh = pv.StructuredGrid(x_surface, y_surface, z_surface)
        mesh[scalar_name] = np.asarray(scalar_values, dtype=np.float64).ravel(order="F")
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="coolwarm",
            clim=(-vmax, vmax),
            opacity=opacity,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": scalar_name,
                "title_font_size": 18,
                "label_font_size": 14,
                "shadow": False,
                "fmt": "%.2f",
            },
        )

    outer_i = max(geometry.shape[0] - 3, 1)
    middle_i = max(int(0.72 * geometry.shape[0]), 1)
    for radial_index, opacity, show_bar in ((outer_i, 0.78, True), (middle_i, 0.52, False)):
        add_surface(
            x[np.ix_([radial_index], phi_window, theta_window)][0],
            y[np.ix_([radial_index], phi_window, theta_window)][0],
            z[np.ix_([radial_index], phi_window, theta_window)][0],
            values[np.ix_([radial_index], phi_window, theta_window)][0],
            opacity=opacity,
            show_scalar_bar=show_bar,
        )

    for cut_j in (max(1, ny // 12), max(2, 7 * ny // 12)):
        add_surface(
            x[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            y[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            z[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            values[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            opacity=0.94,
            show_scalar_bar=False,
        )

    for theta_seed in (0.15, 1.55, 3.15):
        line_points = _field_line_points(geometry, radial_index=outer_i, theta_seed=theta_seed)
        line = pv.PolyData(line_points)
        line.lines = np.hstack([[line_points.shape[0]], np.arange(line_points.shape[0])])
        plotter.add_mesh(line, color="black", line_width=2.2, opacity=0.72)

    plotter.add_text(
        "Reduced 3D stellarator SOL dynamics\n"
        f"opened toroidal/radial view, t = {time_value:.2f}",
        position=(32, 835),
        font_size=14,
        color="black",
    )
    plotter.add_text(
        "Outer and mid-radius traced surfaces plus two radial cuts show interior fluctuations",
        position="lower_left",
        font_size=11,
        color="black",
    )
    center = (float(np.nanmean(x)), float(np.nanmean(y)), float(np.nanmean(z)))
    radius = 1.75 * max(float(np.nanmax(x) - np.nanmin(x)), float(np.nanmax(y) - np.nanmin(y)))
    angle = np.deg2rad(-38.0 + 26.0 * np.sin(1.35 * time_value))
    camera = (
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        center[2] + 0.55 * radius,
    )
    plotter.camera_position = [camera, center, (0.0, 0.0, 1.0)]
    plotter.screenshot(str(resolved))
    plotter.close()
    return resolved


def _field_line_points(
    geometry: SyntheticStellaratorGeometry,
    *,
    radial_index: int,
    theta_seed: float,
) -> np.ndarray:
    x = np.asarray(geometry.coordinates_x)
    y = np.asarray(geometry.coordinates_y)
    z = np.asarray(geometry.coordinates_z)
    phi = np.linspace(0.0, 2.0 * np.pi, geometry.shape[1], endpoint=False)
    iota = np.asarray(geometry.iota[radial_index, :, 0], dtype=np.float64)
    theta = np.mod(theta_seed + np.cumsum(iota) * geometry.maps.dphi, 2.0 * np.pi)
    theta_indices = np.mod(np.rint(theta * geometry.shape[2] / (2.0 * np.pi)).astype(int), geometry.shape[2])
    return np.column_stack(
        [
            x[radial_index, np.arange(geometry.shape[1]), theta_indices],
            y[radial_index, np.arange(geometry.shape[1]), theta_indices],
            z[radial_index, np.arange(geometry.shape[1]), theta_indices],
        ]
    )


def _write_showcase_arrays(
    geometry: SyntheticStellaratorGeometry,
    history: np.ndarray,
    time: np.ndarray,
    path: Path,
) -> Path:
    np.savez_compressed(
        path,
        history=np.asarray(history, dtype=np.float16),
        time=np.asarray(time, dtype=np.float32),
        x=np.asarray(geometry.coordinates_x, dtype=np.float32),
        y=np.asarray(geometry.coordinates_y, dtype=np.float32),
        z=np.asarray(geometry.coordinates_z, dtype=np.float32),
        curvature=np.asarray(geometry.curvature, dtype=np.float32),
        connection_length=np.asarray(geometry.connection_length, dtype=np.float32),
        Bxy=np.asarray(geometry.metric.Bxy, dtype=np.float32),
    )
    return path


def _moment(values: np.ndarray, order: int) -> float:
    centered = values - np.mean(values)
    sigma = float(np.std(centered))
    if sigma <= 0.0:
        return 0.0
    return float(np.mean((centered / sigma) ** order))


def _cell_moment(history: np.ndarray, order: int) -> np.ndarray:
    centered = history - np.mean(history, axis=0, keepdims=True)
    sigma = np.std(centered, axis=0)
    safe_sigma = np.where(sigma > 1.0e-12, sigma, 1.0)
    moment = np.mean((centered / safe_sigma[None, :, :, :]) ** order, axis=0)
    return np.where(sigma > 1.0e-12, moment, 0.0)
