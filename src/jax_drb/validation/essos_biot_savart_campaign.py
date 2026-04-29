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
    field_line_png_path: Path
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
    field_line_png_path = images_dir / f"{case_label}_field_lines.png"
    save_essos_biot_savart_field_line_plot(report, arrays, field_line_png_path)
    movie_gif_path = movies_dir / f"{case_label}.gif"
    save_essos_biot_savart_campaign_movie(closed_geometry, open_geometry, closed_history, open_history, time, movie_gif_path)
    return EssosBiotSavartCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
        field_line_png_path=field_line_png_path,
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
    field_line_report, field_line_arrays = build_essos_biot_savart_field_line_diagnostics(
        coils=coils,
        major_radius=major_radius,
        vertical_axis=vertical_axis,
        closed_radius_range=(0.18, 0.72),
        open_radius_range=(0.55, 1.00),
    )
    report: dict[str, object] = {
        "case": "essos_biot_savart_landreman_paul_qa_closed_open_turbulence",
        "coil_metadata": coils.metadata,
        "axis_guess_major_radius": float(major_radius),
        "axis_guess_vertical_axis": float(vertical_axis),
        "regions": {
            "closed_like_inner_annulus": closed_metrics,
            "open_sol_like_outer_annulus": open_metrics,
        },
        "field_line_diagnostics": field_line_report,
    }
    report["passed"] = bool(
        closed_metrics["passed"]
        and open_metrics["passed"]
        and field_line_report["passed"]
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
    arrays.update(field_line_arrays)
    return report, arrays, closed_geometry, open_geometry, closed_history, open_history, time


def build_essos_biot_savart_field_line_diagnostics(
    *,
    coils: FourierCoilSet,
    major_radius: float,
    vertical_axis: float,
    closed_radius_range: tuple[float, float] = (0.18, 0.72),
    open_radius_range: tuple[float, float] = (0.55, 1.00),
    max_turns: float = 5.0,
    steps_per_turn: int = 96,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Trace annular field lines and classify closed-like/open-like regions."""

    closed_report, closed_arrays = trace_biot_savart_annular_field_lines(
        coils=coils,
        region_label="closed_like_inner_annulus",
        radius_range=closed_radius_range,
        major_radius=major_radius,
        vertical_axis=vertical_axis,
        max_turns=max_turns,
        steps_per_turn=steps_per_turn,
    )
    open_report, open_arrays = trace_biot_savart_annular_field_lines(
        coils=coils,
        region_label="open_sol_like_outer_annulus",
        radius_range=open_radius_range,
        major_radius=major_radius,
        vertical_axis=vertical_axis,
        max_turns=max_turns,
        steps_per_turn=steps_per_turn,
    )
    report: dict[str, object] = {
        "method": "annular_biot_savart_field_line_trace",
        "max_turns": float(max_turns),
        "steps_per_turn": int(steps_per_turn),
        "closed_like_inner_annulus": closed_report,
        "open_sol_like_outer_annulus": open_report,
    }
    report["passed"] = bool(
        closed_report["passed"]
        and open_report["passed"]
        and closed_report["mean_exit_turns"] > 1.8 * open_report["mean_exit_turns"]
        and closed_report["mean_exit_connection_length"] > open_report["mean_exit_connection_length"]
        and open_report["escaped_fraction"] > 0.5
    )
    arrays: dict[str, np.ndarray] = {}
    arrays.update({f"closed_field_line_{key}": value for key, value in closed_arrays.items()})
    arrays.update({f"open_field_line_{key}": value for key, value in open_arrays.items()})
    return report, arrays


def trace_biot_savart_annular_field_lines(
    *,
    coils: FourierCoilSet,
    region_label: str,
    radius_range: tuple[float, float],
    major_radius: float,
    vertical_axis: float,
    max_turns: float = 5.0,
    steps_per_turn: int = 96,
    seed_radii: int = 5,
    seed_angles: int = 12,
    trace_sample_count: int = 8,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Trace a seed grid in an annulus using toroidal angle as the independent variable."""

    rho_min, rho_max = float(radius_range[0]), float(radius_range[1])
    radial_width = max(rho_max - rho_min, 1.0e-12)
    seed_rho_1d = np.linspace(rho_min + 0.12 * radial_width, rho_max - 0.12 * radial_width, int(seed_radii))
    seed_theta_1d = np.linspace(0.0, 2.0 * np.pi, int(seed_angles), endpoint=False)
    seed_rho, seed_theta = np.meshgrid(seed_rho_1d, seed_theta_1d, indexing="ij")
    seed_rho = seed_rho.ravel()
    seed_theta = seed_theta.ravel()
    seed_count = int(seed_rho.size)
    radial_offset = seed_rho * np.cos(seed_theta)
    vertical_offset = seed_rho * np.sin(seed_theta)
    major = float(major_radius) + radial_offset
    vertical = float(vertical_axis) + vertical_offset
    active = np.ones(seed_count, dtype=bool)
    exit_length = np.full(seed_count, np.nan, dtype=np.float64)
    exit_turns = np.full(seed_count, float(max_turns), dtype=np.float64)
    path_length = np.zeros(seed_count, dtype=np.float64)
    radial_min_seen = seed_rho.copy()
    radial_max_seen = seed_rho.copy()
    total_steps = int(round(float(max_turns) * int(steps_per_turn)))
    dphi = 2.0 * np.pi / float(steps_per_turn)
    field_period_stride = max(1, int(round(float(steps_per_turn) / max(int(coils.nfp), 1))))
    sample_indices = np.linspace(0, seed_count - 1, min(int(trace_sample_count), seed_count), dtype=int)
    sample_major = [major[sample_indices].copy()]
    sample_vertical = [vertical[sample_indices].copy()]
    sample_phi = [0.0]
    poincare_major = [major.copy()]
    poincare_vertical = [vertical.copy()]
    poincare_turn = [np.zeros(seed_count, dtype=np.float64)]

    for step in range(1, total_steps + 1):
        phi = (step - 1) * dphi
        k1_major, k1_vertical, k1_length = _biot_savart_cylindrical_rhs(
            coils,
            major,
            vertical,
            phi,
        )
        k2_major, k2_vertical, k2_length = _biot_savart_cylindrical_rhs(
            coils,
            major + 0.5 * dphi * k1_major,
            vertical + 0.5 * dphi * k1_vertical,
            phi + 0.5 * dphi,
        )
        k3_major, k3_vertical, k3_length = _biot_savart_cylindrical_rhs(
            coils,
            major + 0.5 * dphi * k2_major,
            vertical + 0.5 * dphi * k2_vertical,
            phi + 0.5 * dphi,
        )
        k4_major, k4_vertical, k4_length = _biot_savart_cylindrical_rhs(
            coils,
            major + dphi * k3_major,
            vertical + dphi * k3_vertical,
            phi + dphi,
        )
        next_major = major + dphi * (k1_major + 2.0 * k2_major + 2.0 * k3_major + k4_major) / 6.0
        next_vertical = vertical + dphi * (k1_vertical + 2.0 * k2_vertical + 2.0 * k3_vertical + k4_vertical) / 6.0
        step_length = dphi * (k1_length + 2.0 * k2_length + 2.0 * k3_length + k4_length) / 6.0
        major = np.where(active, next_major, major)
        vertical = np.where(active, next_vertical, vertical)
        path_length = np.where(active, path_length + step_length, path_length)
        rho = np.sqrt((major - float(major_radius)) ** 2 + (vertical - float(vertical_axis)) ** 2)
        radial_min_seen = np.where(active, np.minimum(radial_min_seen, rho), radial_min_seen)
        radial_max_seen = np.where(active, np.maximum(radial_max_seen, rho), radial_max_seen)
        finite = np.isfinite(major) & np.isfinite(vertical) & np.isfinite(path_length)
        exited_now = active & ((rho < rho_min) | (rho > rho_max) | ~finite)
        exit_length = np.where(exited_now, path_length, exit_length)
        exit_turns = np.where(exited_now, step / float(steps_per_turn), exit_turns)
        active = active & ~exited_now
        if step % max(1, steps_per_turn // 24) == 0:
            sample_major.append(major[sample_indices].copy())
            sample_vertical.append(vertical[sample_indices].copy())
            sample_phi.append(step * dphi)
        if step % field_period_stride == 0:
            poincare_major.append(major.copy())
            poincare_vertical.append(vertical.copy())
            poincare_turn.append(np.full(seed_count, step / float(steps_per_turn), dtype=np.float64))

    escaped = ~active
    observed_exit_length = np.where(np.isfinite(exit_length), exit_length, path_length)
    survival_turns = np.linspace(0.0, float(max_turns), 64)
    survival_fraction = np.asarray([float(np.mean(exit_turns >= turn)) for turn in survival_turns], dtype=np.float64)
    radial_excursion = radial_max_seen - radial_min_seen
    report = {
        "label": region_label,
        "radius_min": rho_min,
        "radius_max": rho_max,
        "seed_count": seed_count,
        "escaped_fraction": float(np.mean(escaped)),
        "confined_fraction": float(np.mean(~escaped)),
        "mean_exit_connection_length": float(np.mean(observed_exit_length)),
        "median_exit_connection_length": float(np.median(observed_exit_length)),
        "max_observed_connection_length": float(np.max(observed_exit_length)),
        "mean_exit_turns": float(np.mean(exit_turns)),
        "median_exit_turns": float(np.median(exit_turns)),
        "radial_excursion_mean": float(np.mean(radial_excursion)),
        "radial_excursion_max": float(np.max(radial_excursion)),
        "poincare_points": int(np.asarray(poincare_major).size),
    }
    report["passed"] = bool(
        np.isfinite(report["mean_exit_connection_length"])
        and report["mean_exit_connection_length"] > 0.0
        and np.isfinite(report["radial_excursion_mean"])
    )
    arrays = {
        "seed_rho": seed_rho.astype(np.float32),
        "seed_theta": seed_theta.astype(np.float32),
        "exit_length": observed_exit_length.astype(np.float32),
        "exit_turns": exit_turns.astype(np.float32),
        "escaped": escaped.astype(np.int8),
        "radial_excursion": radial_excursion.astype(np.float32),
        "survival_turns": survival_turns.astype(np.float32),
        "survival_fraction": survival_fraction.astype(np.float32),
        "poincare_major": np.asarray(poincare_major, dtype=np.float32),
        "poincare_vertical": np.asarray(poincare_vertical, dtype=np.float32),
        "poincare_turn": np.asarray(poincare_turn, dtype=np.float32),
        "sample_major": np.asarray(sample_major, dtype=np.float32),
        "sample_vertical": np.asarray(sample_vertical, dtype=np.float32),
        "sample_phi": np.asarray(sample_phi, dtype=np.float32),
    }
    return report, arrays


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


def save_essos_biot_savart_field_line_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the Poincare/connection-length figure for the coil-field gate."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    major_radius = float(report["axis_guess_major_radius"])
    vertical_axis = float(report["axis_guess_vertical_axis"])
    trace_report = report["field_line_diagnostics"]
    closed = trace_report["closed_like_inner_annulus"]
    open_region = trace_report["open_sol_like_outer_annulus"]
    fig = plt.figure(figsize=(14.8, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(grid[0, 0])
    plot_radius = 1.15 * max(float(open_region["radius_max"]), float(closed["radius_max"]))
    for prefix, color, label, alpha in (
        ("closed", "#005f73", "closed-like annulus", 0.42),
        ("open", "#ca6702", "open/SOL-like annulus", 0.34),
    ):
        poincare_major = arrays[f"{prefix}_field_line_poincare_major"].reshape(-1)
        poincare_vertical = arrays[f"{prefix}_field_line_poincare_vertical"].reshape(-1)
        plot_x = poincare_major - major_radius
        plot_z = poincare_vertical - vertical_axis
        display = np.sqrt(plot_x * plot_x + plot_z * plot_z) <= plot_radius
        ax0.scatter(
            plot_x[display],
            plot_z[display],
            s=5.0,
            color=color,
            alpha=alpha,
            label=label,
            linewidths=0.0,
        )
    _draw_annulus(ax0, closed["radius_min"], closed["radius_max"], color="#005f73", lw=1.3)
    _draw_annulus(ax0, open_region["radius_min"], open_region["radius_max"], color="#ca6702", lw=1.3)
    ax0.set_aspect("equal")
    ax0.set_xlim(-plot_radius, plot_radius)
    ax0.set_ylim(-plot_radius, plot_radius)
    ax0.set_xlabel("R - R0")
    ax0.set_ylabel("Z - Z0")
    ax0.set_title("Near-annulus Poincare samples from coil-produced field")
    ax0.legend(frameon=False, loc="upper right")

    ax1 = fig.add_subplot(grid[0, 1])
    for prefix, color, label in (
        ("closed", "#005f73", "closed-like"),
        ("open", "#ca6702", "open/SOL-like"),
    ):
        major = arrays[f"{prefix}_field_line_sample_major"]
        vertical = arrays[f"{prefix}_field_line_sample_vertical"]
        for trace_index in range(major.shape[1]):
            ax1.plot(
                major[:, trace_index] - major_radius,
                vertical[:, trace_index] - vertical_axis,
                color=color,
                alpha=0.55,
                lw=1.0,
            )
        ax1.plot([], [], color=color, lw=2.0, label=label)
    _draw_annulus(ax1, closed["radius_min"], closed["radius_max"], color="#005f73", lw=1.1)
    _draw_annulus(ax1, open_region["radius_min"], open_region["radius_max"], color="#ca6702", lw=1.1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("R - R0")
    ax1.set_ylabel("Z - Z0")
    ax1.set_title("Sample annular field-line traces")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(grid[1, 0])
    bins = np.linspace(
        0.0,
        max(
            float(np.max(arrays["closed_field_line_exit_length"])),
            float(np.max(arrays["open_field_line_exit_length"])),
        ),
        28,
    )
    ax2.hist(arrays["closed_field_line_exit_length"], bins=bins, color="#005f73", alpha=0.62, label="closed-like")
    ax2.hist(arrays["open_field_line_exit_length"], bins=bins, color="#ca6702", alpha=0.62, label="open/SOL-like")
    ax2.set_xlabel("annular exit connection-length proxy")
    ax2.set_ylabel("seed count")
    ax2.set_title("Connection-length proxy to annular exit")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(
        arrays["closed_field_line_survival_turns"],
        arrays["closed_field_line_survival_fraction"],
        color="#005f73",
        lw=2.2,
        label=f"closed-like mean exit {closed['mean_exit_turns']:.2f} turns",
    )
    ax3.plot(
        arrays["open_field_line_survival_turns"],
        arrays["open_field_line_survival_fraction"],
        color="#ca6702",
        lw=2.2,
        label=f"open/SOL-like mean exit {open_region['mean_exit_turns']:.2f} turns",
    )
    ax3.set_xlabel("toroidal turns")
    ax3.set_ylabel("seed survival fraction")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("Annular residence classification")
    ax3.legend(frameon=False)

    fig.suptitle(
        "ESSOS Landreman-Paul QA coil field: field-line classification "
        f"(closed mean exit {closed['mean_exit_turns']:.2f} turns, "
        f"open mean exit {open_region['mean_exit_turns']:.2f} turns)",
        fontsize=12,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _biot_savart_cylindrical_rhs(
    coils: FourierCoilSet,
    major: np.ndarray,
    vertical: np.ndarray,
    phi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_phi = np.cos(float(phi))
    sin_phi = np.sin(float(phi))
    points = np.stack([major * cos_phi, major * sin_phi, vertical], axis=-1)
    field = np.asarray(biot_savart_field(coils, jnp.asarray(points, dtype=jnp.float64)), dtype=np.float64)
    b_radial = cos_phi * field[:, 0] + sin_phi * field[:, 1]
    b_phi = -sin_phi * field[:, 0] + cos_phi * field[:, 1]
    b_vertical = field[:, 2]
    b_magnitude = np.linalg.norm(field, axis=-1)
    bphi_floor = 1.0e-9 * max(float(np.nanmax(np.abs(b_phi))), 1.0e-30)
    safe_bphi = np.where(np.abs(b_phi) > bphi_floor, b_phi, np.sign(b_phi + 1.0e-30) * bphi_floor)
    dmajor_dphi = major * b_radial / safe_bphi
    dvertical_dphi = major * b_vertical / safe_bphi
    dlength_dphi = np.abs(major) * b_magnitude / np.maximum(np.abs(safe_bphi), bphi_floor)
    return dmajor_dphi, dvertical_dphi, dlength_dphi


def _draw_annulus(axis: plt.Axes, radius_min: float, radius_max: float, *, color: str, lw: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for radius, linestyle in ((float(radius_min), "--"), (float(radius_max), "-")):
        axis.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, lw=lw, ls=linestyle, alpha=0.8)


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
