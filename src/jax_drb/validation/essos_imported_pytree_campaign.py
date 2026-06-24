from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import EssosImportedFciGeometry, build_essos_imported_fci_geometry
from ..native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs


@dataclass(frozen=True)
class EssosImportedPytreeCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_essos_imported_pytree_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_pytree_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 4,
    ny: int = 6,
    nz: int = 12,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 60.0,
    times_to_trace: int = 280,
    steps: int = 5,
) -> EssosImportedPytreeCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_essos_imported_pytree_campaign(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source=map_source,
        nx=nx,
        ny=ny,
        nz=nz,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        steps=steps,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_imported_pytree_campaign_plot(report, arrays, plot_png_path)
    return EssosImportedPytreeCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_essos_imported_pytree_campaign(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 4,
    ny: int = 6,
    nz: int = 12,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 60.0,
    times_to_trace: int = 280,
    steps: int = 5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = build_essos_imported_fci_geometry(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source=map_source,
        nx=nx,
        ny=ny,
        nz=nz,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
    )
    parameters = FciDrbRhsParameters(
        recycling_fraction=0.965,
        recycled_neutral_energy=0.026,
        potential_iterations=64,
        potential_regularization=1.0,
    )
    dt = 1.0e-5
    run_transient = _build_imported_transient(geometry, parameters=parameters, dt=dt, steps=steps)
    initial = initial_essos_imported_drb_state(geometry, drive_scale=1.0)

    t0 = time.perf_counter()
    final_state, history = run_transient(initial)
    _block_until_ready((final_state, history))
    compile_and_first_execute_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    final_state, history = run_transient(initial)
    _block_until_ready((final_state, history))
    warm_execute_seconds = time.perf_counter() - t1

    objective = _build_imported_objective(geometry, parameters=parameters, dt=dt, steps=steps)
    value, jvp_derivative = jax.jvp(
        objective,
        (jnp.asarray(1.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    _block_until_ready((value, jvp_derivative))
    eps = 1.0e-3
    finite_difference = (objective(1.0 + eps) - objective(1.0 - eps)) / (2.0 * eps)
    _block_until_ready(finite_difference)
    jvp_relative_error = float(
        jnp.abs(jvp_derivative - finite_difference) / jnp.maximum(jnp.abs(finite_difference), 1.0e-14)
    )

    batched_objective = jax.jit(jax.vmap(objective))
    serial_scales = np.linspace(0.90, 1.10, 4)
    serial_values = np.asarray([float(objective(scale)) for scale in serial_scales], dtype=np.float64)
    vmap_values = np.asarray(batched_objective(jnp.asarray(serial_scales, dtype=jnp.float64)), dtype=np.float64)
    _block_until_ready(vmap_values)
    vmap_serial_linf = float(np.max(np.abs(serial_values - vmap_values)))

    batch_sizes = np.asarray([1, 2, 4], dtype=np.int64)
    batch_times = []
    for batch_size in batch_sizes:
        scales = jnp.linspace(0.90, 1.10, int(batch_size), dtype=jnp.float64)
        batched_objective(scales).block_until_ready()
        t_batch = time.perf_counter()
        batched_objective(scales).block_until_ready()
        batch_times.append(time.perf_counter() - t_batch)

    history_np = np.asarray(history, dtype=np.float64)
    final_ion = np.asarray(final_state.ion_density, dtype=np.float64)
    final_neutral = np.asarray(final_state.neutral_density, dtype=np.float64)
    final_vorticity = np.asarray(final_state.vorticity, dtype=np.float64)
    bmag = np.asarray(geometry.magnetic_field_magnitude, dtype=np.float64)
    endpoint_fraction = float(
        np.mean(np.asarray(geometry.forward_boundary, dtype=bool) | np.asarray(geometry.backward_boundary, dtype=bool))
    )
    actual_map_source = str(geometry.metadata.get("map_source", "coil"))
    endpoint_gate = endpoint_fraction < 1.0e-12 if actual_map_source == "vmec" else 0.05 < endpoint_fraction <= 1.0
    b_modulation_gate = 1.01 if actual_map_source == "vmec" else 1.05
    min_density = float(min(np.min(final_ion), np.min(final_neutral)))
    report: dict[str, Any] = {
        "case": "essos_imported_fci_drb_pytree_transient",
        "source": "ESSOS-imported field-line maps with JAXDRB fixed-layout PyTree RHS",
        "map_source": actual_map_source,
        "geometry": geometry.metadata,
        "steps": int(steps),
        "dt": float(dt),
        "compile_and_first_execute_seconds": float(compile_and_first_execute_seconds),
        "warm_execute_seconds": float(warm_execute_seconds),
        "jvp_objective_value": float(value),
        "jvp_derivative": float(jvp_derivative),
        "finite_difference_derivative": float(finite_difference),
        "jvp_relative_error": jvp_relative_error,
        "vmap_serial_linf": vmap_serial_linf,
        "batch_sizes": batch_sizes.tolist(),
        "batch_execute_seconds": [float(item) for item in batch_times],
        "batch_throughput_cases_per_second": [
            float(size / max(seconds, 1.0e-30)) for size, seconds in zip(batch_sizes, batch_times, strict=True)
        ],
        "endpoint_fraction": endpoint_fraction,
        "magnetic_field_modulation": float(np.max(bmag) / max(float(np.min(bmag)), 1.0e-30)),
        "final_min_density": min_density,
        "final_ion_density_mean": float(np.mean(final_ion)),
        "final_neutral_density_mean": float(np.mean(final_neutral)),
        "final_vorticity_rms": float(np.sqrt(np.mean(final_vorticity * final_vorticity))),
        "final_potential_residual_l2": float(history_np[-1, 4]),
    }
    report["passed"] = (
        np.all(np.isfinite(history_np))
        and min_density > 0.0
        and endpoint_gate
        and report["magnetic_field_modulation"] > b_modulation_gate
        and report["final_vorticity_rms"] > 0.0
        and report["final_potential_residual_l2"] < 2.5
        and jvp_relative_error < 1.0e-2
        and vmap_serial_linf < 1.0e-6
        and report["warm_execute_seconds"] > 0.0
    )
    arrays = {
        "time_index": np.arange(history_np.shape[0], dtype=np.int64),
        "history": history_np.astype(np.float32),
        "final_ion_density_section": final_ion[:, 0, :].astype(np.float32),
        "final_neutral_density_section": final_neutral[:, 0, :].astype(np.float32),
        "final_vorticity_section": final_vorticity[:, 0, :].astype(np.float32),
        "endpoint_count_toroidal": (
            np.asarray(geometry.forward_boundary, dtype=np.float64)
            + np.asarray(geometry.backward_boundary, dtype=np.float64)
        ).sum(axis=0).astype(np.float32),
        "magnetic_field_section": bmag[:, 0, :].astype(np.float32),
        "serial_scales_4": serial_scales.astype(np.float32),
        "serial_values_4": serial_values.astype(np.float32),
        "vmap_values_4": vmap_values.astype(np.float32),
        "batch_sizes": batch_sizes.astype(np.float32),
        "batch_execute_seconds": np.asarray(batch_times, dtype=np.float32),
        "jvp_summary": np.asarray([float(jvp_derivative), float(finite_difference), jvp_relative_error], dtype=np.float32),
    }
    return report, arrays


def initial_essos_imported_drb_state(
    geometry: EssosImportedFciGeometry,
    *,
    drive_scale: jax.Array | float = 1.0,
) -> FciDrbState:
    rho = geometry.minor_radius
    rho_min = jnp.min(rho)
    rho_span = jnp.maximum(jnp.max(rho) - rho_min, 1.0e-12)
    radial = (rho - rho_min) / rho_span
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    bnorm = geometry.magnetic_field_magnitude / jnp.maximum(jnp.mean(geometry.magnetic_field_magnitude), 1.0e-12)
    helical = jnp.cos(2.0 * theta - phi)
    sideband = jnp.sin(theta + 2.0 * phi)
    scale = jnp.asarray(drive_scale, dtype=jnp.float64)
    ion_density = 0.55 + scale * 0.70 * jnp.exp(-jnp.square((radial - 0.46) / 0.26)) * (1.0 + 0.05 * helical)
    neutral_density = 0.11 + scale * 0.25 * jnp.exp(-jnp.square((radial - 0.88) / 0.15)) * (1.0 + 0.10 * helical)
    electron_density = ion_density * (1.0 + 0.005 * sideband)
    ion_temperature = (0.058 + 0.10 * (1.0 - radial)) / jnp.maximum(bnorm, 0.15) ** 0.20
    electron_temperature = (0.075 + 0.14 * (1.0 - radial)) / jnp.maximum(bnorm, 0.15) ** 0.35
    neutral_temperature = 0.020 + 0.018 * radial
    return FciDrbState(
        ion_density=jnp.maximum(ion_density, 1.0e-6),
        electron_density=jnp.maximum(electron_density, 1.0e-6),
        neutral_density=jnp.maximum(neutral_density, 1.0e-8),
        ion_pressure=ion_density * ion_temperature,
        electron_pressure=electron_density * electron_temperature,
        neutral_pressure=neutral_density * neutral_temperature,
        ion_momentum=0.018 * ion_density * helical,
        neutral_momentum=0.010 * neutral_density * jnp.sin(theta - phi),
        vorticity=0.030 * jnp.sin(2.0 * theta - phi) + 0.010 * sideband,
    )


def save_essos_imported_pytree_campaign_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), constrained_layout=True)
    history = arrays["history"]
    for column, label in ((0, "mean ion density"), (1, "mean neutral density"), (3, "vorticity RMS")):
        reference = max(abs(float(history[0, column])), 1.0e-30)
        axes[0, 0].plot(
            arrays["time_index"],
            1.0e6 * (history[:, column] - history[0, column]) / reference,
            lw=2.0,
            label=label,
        )
    axes[0, 0].set_title("imported-map PyTree transient")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("change from step 0 [ppm]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    image1 = axes[0, 1].imshow(arrays["endpoint_count_toroidal"].T, origin="lower", aspect="auto", cmap="magma")
    axes[0, 1].set_title("imported endpoint count")
    axes[0, 1].set_xlabel("toroidal angle")
    axes[0, 1].set_ylabel("poloidal angle")
    fig.colorbar(image1, ax=axes[0, 1])

    image2 = axes[0, 2].imshow(arrays["final_ion_density_section"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 2].set_title("final ion density section")
    fig.colorbar(image2, ax=axes[0, 2])

    image3 = axes[1, 0].imshow(arrays["final_neutral_density_section"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 0].set_title("final neutral density section")
    fig.colorbar(image3, ax=axes[1, 0])

    axes[1, 1].plot(arrays["serial_scales_4"], arrays["serial_values_4"], "o-", lw=2.0, label="serial")
    axes[1, 1].plot(arrays["serial_scales_4"], arrays["vmap_values_4"], "s--", lw=2.0, label="vmap")
    axes[1, 1].set_title("batched objective parity")
    axes[1, 1].set_xlabel("drive scale")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    labels = ["JVP", "finite diff", "rel. error"]
    axes[1, 2].bar(np.arange(3), np.abs(arrays["jvp_summary"]), color=["#005f73", "#9b2226", "#ee9b00"])
    axes[1, 2].set_xticks(np.arange(3), labels, rotation=18, ha="right")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].set_title("JVP derivative gate")
    axes[1, 2].text(
        0.03,
        0.96,
        "\n".join(
            [
                f"warm run = {report['warm_execute_seconds']:.3f} s",
                f"JVP rel. err = {report['jvp_relative_error']:.1e}",
                f"endpoint fraction = {report['endpoint_fraction']:.2f}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    for axis in (axes[0, 2], axes[1, 0]):
        axis.set_xlabel("poloidal index")
        axis.set_ylabel("radial index")
    fig.suptitle(
        "Imported field-line PyTree/JVP gate: external FCI maps with JAXDRB transformable RHS",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _build_imported_transient(
    geometry: EssosImportedFciGeometry,
    *,
    parameters: FciDrbRhsParameters,
    dt: float,
    steps: int,
):
    def run(initial_state: FciDrbState) -> tuple[FciDrbState, jax.Array]:
        def step(state: FciDrbState, _unused: None) -> tuple[FciDrbState, jax.Array]:
            result = compute_fci_drb_rhs(state, geometry=geometry, parameters=parameters)
            next_state = _clip_state(_add_scaled_state(state, result.rhs, dt))
            diagnostics = jnp.asarray(
                [
                    jnp.mean(next_state.ion_density),
                    jnp.mean(next_state.neutral_density),
                    jnp.mean(next_state.electron_pressure),
                    jnp.sqrt(jnp.mean(jnp.square(next_state.vorticity))),
                    result.potential_residual_l2,
                ],
                dtype=jnp.float64,
            )
            return next_state, diagnostics

        return jax.lax.scan(step, initial_state, None, length=int(steps))

    return jax.jit(run)


def _build_imported_objective(
    geometry: EssosImportedFciGeometry,
    *,
    parameters: FciDrbRhsParameters,
    dt: float,
    steps: int,
):
    run_transient = _build_imported_transient(geometry, parameters=parameters, dt=dt, steps=steps)

    def objective(drive_scale: jax.Array) -> jax.Array:
        final_state, history = run_transient(initial_essos_imported_drb_state(geometry, drive_scale=drive_scale))
        return (
            jnp.mean(final_state.ion_density)
            + 0.25 * jnp.mean(final_state.neutral_density)
            + 0.05 * jnp.sqrt(jnp.mean(jnp.square(final_state.vorticity)))
            + 0.0 * history[-1, 4]
        )

    return objective


def _add_scaled_state(state: FciDrbState, rhs: FciDrbState, scale: float) -> FciDrbState:
    return jax.tree_util.tree_map(lambda value, increment: value + float(scale) * increment, state, rhs)


def _clip_state(state: FciDrbState) -> FciDrbState:
    return FciDrbState(
        ion_density=jnp.maximum(state.ion_density, 1.0e-6),
        electron_density=jnp.maximum(state.electron_density, 1.0e-6),
        neutral_density=jnp.maximum(state.neutral_density, 1.0e-8),
        ion_pressure=jnp.maximum(state.ion_pressure, 1.0e-8),
        electron_pressure=jnp.maximum(state.electron_pressure, 1.0e-8),
        neutral_pressure=jnp.maximum(state.neutral_pressure, 1.0e-10),
        ion_momentum=state.ion_momentum,
        neutral_momentum=state.neutral_momentum,
        vorticity=state.vorticity,
    )


def _block_until_ready(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
