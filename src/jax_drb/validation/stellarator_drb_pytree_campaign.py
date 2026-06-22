from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import SyntheticStellaratorGeometry, build_synthetic_stellarator_geometry
from ..native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs


@dataclass(frozen=True)
class StellaratorDrbPytreeCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_drb_pytree_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_drb_pytree_campaign",
    nx: int = 18,
    ny: int = 16,
    nz: int = 32,
    steps: int = 8,
) -> StellaratorDrbPytreeCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = build_stellarator_drb_pytree_campaign(nx=nx, ny=ny, nz=nz, steps=steps)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_drb_pytree_plot(report, arrays, plot_png_path)
    return StellaratorDrbPytreeCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_drb_pytree_campaign(
    *,
    nx: int = 18,
    ny: int = 16,
    nz: int = 32,
    steps: int = 8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    parameters = FciDrbRhsParameters(potential_iterations=32, potential_boussinesq=True)
    non_boussinesq_parameters = replace(parameters, potential_boussinesq=False)
    dt = 1.5e-5
    run_transient = _build_jitted_transient(geometry, parameters=parameters, dt=dt, steps=steps)
    initial = initial_fci_drb_state(geometry, drive_scale=1.0)
    model_switch_report, model_switch_arrays = _build_boussinesq_model_switch_gate(
        geometry,
        initial=initial,
        boussinesq_parameters=parameters,
        non_boussinesq_parameters=non_boussinesq_parameters,
        dt=dt,
        steps=steps,
    )

    t0 = time.perf_counter()
    final_state, history = run_transient(initial)
    _block_until_ready((final_state, history))
    compile_and_first_execute_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    final_state, history = run_transient(initial)
    _block_until_ready((final_state, history))
    warm_execute_seconds = time.perf_counter() - t1

    objective = _build_objective(geometry, parameters=parameters, dt=dt, steps=steps)
    value, jvp_derivative = jax.jvp(objective, (jnp.asarray(1.0, dtype=jnp.float64),), (jnp.asarray(1.0, dtype=jnp.float64),))
    _block_until_ready((value, jvp_derivative))
    eps = 1.0e-3
    finite_difference = (objective(1.0 + eps) - objective(1.0 - eps)) / (2.0 * eps)
    _block_until_ready(finite_difference)
    jvp_relative_error = float(
        jnp.abs(jvp_derivative - finite_difference) / jnp.maximum(jnp.abs(finite_difference), 1.0e-14)
    )

    batch_sizes = np.asarray([1, 2, 4, 8], dtype=np.int64)
    batch_times = []
    batch_values: list[np.ndarray] = []
    batched_objective = jax.jit(jax.vmap(objective))
    for batch_size in batch_sizes:
        scales = jnp.linspace(0.85, 1.15, int(batch_size), dtype=jnp.float64)
        values = batched_objective(scales)
        _block_until_ready(values)
        t_batch = time.perf_counter()
        values = batched_objective(scales)
        _block_until_ready(values)
        batch_times.append(time.perf_counter() - t_batch)
        batch_values.append(np.asarray(values, dtype=np.float64))

    serial_scales = np.linspace(0.85, 1.15, 4)
    serial_values = np.asarray([float(objective(scale)) for scale in serial_scales], dtype=np.float64)
    vmap_values = np.asarray(batched_objective(jnp.asarray(serial_scales, dtype=jnp.float64)), dtype=np.float64)
    vmap_serial_linf = float(np.max(np.abs(serial_values - vmap_values)))

    pmap_seconds = None
    pmap_values = None
    local_device_count = len(jax.local_devices())
    if local_device_count > 1:
        pmap_objective = jax.pmap(objective)
        device_scales = jnp.linspace(0.90, 1.10, local_device_count, dtype=jnp.float64)
        pmap_objective(device_scales).block_until_ready()
        t_pmap = time.perf_counter()
        pmap_values_array = pmap_objective(device_scales)
        pmap_values_array.block_until_ready()
        pmap_seconds = time.perf_counter() - t_pmap
        pmap_values = np.asarray(pmap_values_array, dtype=np.float64)

    history_np = np.asarray(history, dtype=np.float64)
    final_ion = np.asarray(final_state.ion_density, dtype=np.float64)
    final_neutral = np.asarray(final_state.neutral_density, dtype=np.float64)
    final_vorticity = np.asarray(final_state.vorticity, dtype=np.float64)
    min_density = float(min(np.min(final_ion), np.min(final_neutral)))
    report: dict[str, Any] = {
        "case": "non_axisymmetric_fci_drb_pytree_transient",
        "geometry": geometry.metadata,
        "steps": int(steps),
        "dt": float(dt),
        "potential_boussinesq": bool(parameters.potential_boussinesq),
        "compile_and_first_execute_seconds": float(compile_and_first_execute_seconds),
        "warm_execute_seconds": float(warm_execute_seconds),
        "jvp_objective_value": float(value),
        "jvp_derivative": float(jvp_derivative),
        "finite_difference_derivative": float(finite_difference),
        "jvp_relative_error": jvp_relative_error,
        "vmap_serial_linf": vmap_serial_linf,
        "batch_sizes": batch_sizes.tolist(),
        "batch_execute_seconds": [float(item) for item in batch_times],
        "batch_throughput_cases_per_second": [float(size / max(seconds, 1.0e-30)) for size, seconds in zip(batch_sizes, batch_times, strict=True)],
        "local_devices": [str(device) for device in jax.local_devices()],
        "local_device_count": int(local_device_count),
        "pmap_execute_seconds": None if pmap_seconds is None else float(pmap_seconds),
        "final_min_density": min_density,
        "final_ion_density_mean": float(np.mean(final_ion)),
        "final_neutral_density_mean": float(np.mean(final_neutral)),
        "final_vorticity_rms": float(np.sqrt(np.mean(final_vorticity * final_vorticity))),
        "final_potential_residual_l2": float(history_np[-1, 4]),
        **model_switch_report,
    }
    report["passed"] = (
        np.all(np.isfinite(history_np))
        and min_density > 0.0
        and report["final_vorticity_rms"] > 0.0
        and report["final_potential_residual_l2"] < 2.0
        and jvp_relative_error < 5.0e-3
        and report["non_boussinesq_jvp_relative_error"] < 5.0e-3
        and report["boussinesq_non_boussinesq_potential_relative_l2"] > 1.0e-4
        and report["boussinesq_non_boussinesq_rhs_state_linf"] < 1.0e-12
        and vmap_serial_linf < 1.0e-8
        and report["warm_execute_seconds"] > 0.0
    )
    arrays = {
        "time_index": np.arange(history_np.shape[0], dtype=np.int64),
        "history": history_np.astype(np.float32),
        "final_ion_density_slice": final_ion[:, 0, :].astype(np.float32),
        "final_neutral_density_slice": final_neutral[:, 0, :].astype(np.float32),
        "final_vorticity_slice": final_vorticity[:, 0, :].astype(np.float32),
        "batch_sizes": batch_sizes.astype(np.float32),
        "batch_execute_seconds": np.asarray(batch_times, dtype=np.float32),
        "batch_values_4": vmap_values.astype(np.float32),
        "serial_values_4": serial_values.astype(np.float32),
        "serial_scales_4": serial_scales.astype(np.float32),
        "pmap_values": np.asarray([] if pmap_values is None else pmap_values, dtype=np.float32),
        "jvp_summary": np.asarray([float(jvp_derivative), float(finite_difference), jvp_relative_error], dtype=np.float32),
        **model_switch_arrays,
    }
    return report, arrays


def initial_fci_drb_state(geometry: SyntheticStellaratorGeometry, *, drive_scale: float = 1.0) -> FciDrbState:
    radial = geometry.radial
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    helical = jnp.cos(2.0 * theta - 5.0 * phi)
    sideband = jnp.sin(3.0 * theta + 2.0 * phi)
    ion_density = 0.60 + drive_scale * 0.78 * jnp.exp(-jnp.square((radial - 0.67) / 0.18)) * (1.0 + 0.06 * helical)
    neutral_density = 0.10 + drive_scale * 0.28 * jnp.exp(-jnp.square((radial - 0.92) / 0.08)) * (1.0 + 0.12 * helical)
    electron_density = ion_density * (1.0 + 0.006 * sideband)
    ion_temperature = 0.065 + 0.10 * (1.0 - radial)
    electron_temperature = 0.080 + 0.15 * (1.0 - radial)
    neutral_temperature = 0.018 + 0.018 * radial
    return FciDrbState(
        ion_density=ion_density,
        electron_density=electron_density,
        neutral_density=neutral_density,
        ion_pressure=ion_density * ion_temperature,
        electron_pressure=electron_density * electron_temperature,
        neutral_pressure=neutral_density * neutral_temperature,
        ion_momentum=0.020 * ion_density * helical,
        neutral_momentum=0.012 * neutral_density * jnp.sin(theta - 5.0 * phi),
        vorticity=0.035 * jnp.sin(2.0 * theta - 5.0 * phi) + 0.012 * sideband,
    )


def _build_boussinesq_model_switch_gate(
    geometry: SyntheticStellaratorGeometry,
    *,
    initial: FciDrbState,
    boussinesq_parameters: FciDrbRhsParameters,
    non_boussinesq_parameters: FciDrbRhsParameters,
    dt: float,
    steps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    boussinesq_result = compute_fci_drb_rhs(
        initial,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=boussinesq_parameters,
    )
    non_boussinesq_result = compute_fci_drb_rhs(
        initial,
        maps=geometry.maps,
        metric=geometry.metric,
        parameters=non_boussinesq_parameters,
    )
    _block_until_ready((boussinesq_result, non_boussinesq_result))
    boussinesq_potential = np.asarray(boussinesq_result.potential, dtype=np.float64)
    non_boussinesq_potential = np.asarray(non_boussinesq_result.potential, dtype=np.float64)
    potential_difference = non_boussinesq_potential - boussinesq_potential
    potential_relative_l2 = float(
        np.sqrt(np.mean(potential_difference * potential_difference))
        / max(np.sqrt(np.mean(boussinesq_potential * boussinesq_potential)), 1.0e-30)
    )
    rhs_state_linf = _state_linf_difference(
        boussinesq_result.rhs,
        non_boussinesq_result.rhs,
    )
    coefficient = np.asarray(
        initial.ion_density / jnp.maximum(jnp.square(geometry.metric.Bxy), 1.0e-30),
        dtype=np.float64,
    )
    coefficient_contrast = float(np.max(coefficient) / max(float(np.min(coefficient)), 1.0e-30))
    non_boussinesq_objective = _build_objective(
        geometry,
        parameters=non_boussinesq_parameters,
        dt=dt,
        steps=steps,
    )
    value, derivative = jax.jvp(
        non_boussinesq_objective,
        (jnp.asarray(1.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    _block_until_ready((value, derivative))
    eps = 1.0e-3
    finite_difference = (
        non_boussinesq_objective(1.0 + eps)
        - non_boussinesq_objective(1.0 - eps)
    ) / (2.0 * eps)
    _block_until_ready(finite_difference)
    jvp_relative_error = float(
        jnp.abs(derivative - finite_difference)
        / jnp.maximum(jnp.abs(finite_difference), 1.0e-14)
    )
    report = {
        "boussinesq_gate_potential_boussinesq": bool(boussinesq_parameters.potential_boussinesq),
        "non_boussinesq_gate_potential_boussinesq": bool(
            non_boussinesq_parameters.potential_boussinesq
        ),
        "boussinesq_potential_residual_l2": float(np.asarray(boussinesq_result.potential_residual_l2)),
        "non_boussinesq_potential_residual_l2": float(np.asarray(non_boussinesq_result.potential_residual_l2)),
        "boussinesq_non_boussinesq_potential_relative_l2": potential_relative_l2,
        "boussinesq_non_boussinesq_rhs_state_linf": rhs_state_linf,
        "density_over_b_squared_contrast": coefficient_contrast,
        "non_boussinesq_jvp_objective_value": float(value),
        "non_boussinesq_jvp_derivative": float(derivative),
        "non_boussinesq_finite_difference_derivative": float(finite_difference),
        "non_boussinesq_jvp_relative_error": jvp_relative_error,
    }
    arrays = {
        "boussinesq_potential_slice": boussinesq_potential[:, 0, :].astype(np.float32),
        "non_boussinesq_potential_slice": non_boussinesq_potential[:, 0, :].astype(np.float32),
        "potential_model_difference_slice": potential_difference[:, 0, :].astype(np.float32),
        "density_over_b_squared_slice": coefficient[:, 0, :].astype(np.float32),
        "model_switch_summary": np.asarray(
            [
                potential_relative_l2,
                report["boussinesq_potential_residual_l2"],
                report["non_boussinesq_potential_residual_l2"],
                rhs_state_linf,
                jvp_relative_error,
                coefficient_contrast,
            ],
            dtype=np.float32,
        ),
    }
    return report, arrays


def save_stellarator_drb_pytree_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16.2, 12.6), constrained_layout=True)
    history = arrays["history"]
    for column, label in ((0, "mean ion density"), (1, "mean neutral density"), (3, "vorticity RMS")):
        reference = max(abs(float(history[0, column])), 1.0e-30)
        axes[0, 0].plot(
            arrays["time_index"],
            1.0e6 * (history[:, column] - history[0, column]) / reference,
            lw=2.0,
            label=label,
        )
    axes[0, 0].set_title("short 3D PyTree transient")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("change from step 0 [ppm]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    image1 = axes[0, 1].imshow(arrays["final_ion_density_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 1].set_title("final ion density")
    fig.colorbar(image1, ax=axes[0, 1])

    image2 = axes[0, 2].imshow(arrays["final_neutral_density_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[0, 2].set_title("final neutral density")
    fig.colorbar(image2, ax=axes[0, 2])

    axes[1, 0].plot(arrays["serial_scales_4"], arrays["serial_values_4"], "o-", lw=2.0, label="serial")
    axes[1, 0].plot(arrays["serial_scales_4"], arrays["batch_values_4"], "s--", lw=2.0, label="vmap")
    axes[1, 0].set_title("batched objective matches serial")
    axes[1, 0].set_xlabel("drive scale")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    throughput = arrays["batch_sizes"] / np.maximum(arrays["batch_execute_seconds"], 1.0e-30)
    axes[1, 1].plot(arrays["batch_sizes"], throughput, "o-", lw=2.2, color="#0a9396")
    axes[1, 1].set_title("single-device batched throughput")
    axes[1, 1].set_xlabel("batch size")
    axes[1, 1].set_ylabel("cases / second")
    axes[1, 1].grid(alpha=0.25)

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
                f"devices = {report['local_device_count']}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    image6 = axes[2, 0].imshow(
        arrays["boussinesq_potential_slice"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[2, 0].set_title("Boussinesq potential solve")
    fig.colorbar(image6, ax=axes[2, 0])

    image7 = axes[2, 1].imshow(
        arrays["non_boussinesq_potential_slice"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[2, 1].set_title("non-Boussinesq potential solve")
    fig.colorbar(image7, ax=axes[2, 1])

    vmax = float(np.max(np.abs(arrays["potential_model_difference_slice"])))
    image8 = axes[2, 2].imshow(
        arrays["potential_model_difference_slice"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[2, 2].set_title("non-Boussinesq minus Boussinesq")
    fig.colorbar(image8, ax=axes[2, 2])
    axes[2, 2].text(
        0.03,
        0.96,
        "\n".join(
            [
                f"rel. L2 = {report['boussinesq_non_boussinesq_potential_relative_l2']:.2e}",
                f"non-Bq JVP err = {report['non_boussinesq_jvp_relative_error']:.1e}",
                f"n/B^2 contrast = {report['density_over_b_squared_contrast']:.2f}",
            ]
        ),
        transform=axes[2, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    for axis in (axes[0, 1], axes[0, 2], axes[2, 0], axes[2, 1], axes[2, 2]):
        axis.set_xlabel("poloidal index")
        axis.set_ylabel("radial index")
    fig.suptitle(
        "True 3D non-axisymmetric PyTree lane: DRB RHS, model switch, JVP, and batched execution",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _state_linf_difference(left: FciDrbState, right: FciDrbState) -> float:
    differences = [
        float(np.max(np.abs(np.asarray(left_leaf) - np.asarray(right_leaf))))
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    ]
    return max(differences, default=0.0)


def _build_jitted_transient(
    geometry: SyntheticStellaratorGeometry,
    *,
    parameters: FciDrbRhsParameters,
    dt: float,
    steps: int,
):
    def run(initial_state: FciDrbState) -> tuple[FciDrbState, jax.Array]:
        def step(state: FciDrbState, _unused: None) -> tuple[FciDrbState, jax.Array]:
            result = compute_fci_drb_rhs(state, maps=geometry.maps, metric=geometry.metric, parameters=parameters)
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


def _build_objective(
    geometry: SyntheticStellaratorGeometry,
    *,
    parameters: FciDrbRhsParameters,
    dt: float,
    steps: int,
):
    run_transient = _build_jitted_transient(geometry, parameters=parameters, dt=dt, steps=steps)

    def objective(drive_scale: jax.Array) -> jax.Array:
        final_state, history = run_transient(initial_fci_drb_state(geometry, drive_scale=drive_scale))
        return (
            jnp.mean(final_state.ion_density)
            + 0.25 * jnp.mean(final_state.neutral_density)
            + 0.05 * jnp.sqrt(jnp.mean(jnp.square(final_state.vorticity)))
            + 0.01 * history[-1, 4]
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
