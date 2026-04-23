from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..native.mesh import StructuredMesh
from ..native.open_field import (
    build_target_boundary_geometry,
    compute_electron_force_balance,
    compute_target_recycling_sources,
    grad_par_y,
)
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class OpenFieldOperatorCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_open_field_operator_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "open_field_operator_campaign",
    resolutions: tuple[int, ...] = (32, 64, 128, 256),
    length: float = 2.0 * np.pi,
) -> OpenFieldOperatorCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_open_field_operator_campaign_report(resolutions=resolutions, length=length)

    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        resolutions=np.asarray(report["resolutions"], dtype=np.int64),
        gradient_l2=np.asarray([entry["gradient_l2"] for entry in report["convergence_runs"]], dtype=np.float64),
        force_balance_l2=np.asarray(
            [entry["force_balance_l2"] for entry in report["convergence_runs"]],
            dtype=np.float64,
        ),
        gradient_order=np.asarray([entry["gradient_order"] for entry in report["observed_orders"]], dtype=np.float64),
        force_balance_order=np.asarray(
            [entry["force_balance_order"] for entry in report["observed_orders"]],
            dtype=np.float64,
        ),
        recycling_velocity=np.asarray(report["target_recycling_identity"]["velocity"], dtype=np.float64),
        recycling_density_source=np.asarray(
            report["target_recycling_identity"]["density_source"],
            dtype=np.float64,
        ),
        recycling_expected_density_source=np.asarray(
            report["target_recycling_identity"]["expected_density_source"],
            dtype=np.float64,
        ),
        recycling_lower_density_source=np.asarray(
            report["target_recycling_identity"]["lower_density_source"],
            dtype=np.float64,
        ),
        recycling_expected_lower_density_source=np.asarray(
            report["target_recycling_identity"]["expected_lower_density_source"],
            dtype=np.float64,
        ),
    )

    plot_png_path = save_open_field_operator_campaign_plot(report, images_dir / f"{case_label}.png")
    return OpenFieldOperatorCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_open_field_operator_campaign_report(
    *,
    resolutions: tuple[int, ...] = (32, 64, 128, 256),
    length: float = 2.0 * np.pi,
) -> dict[str, object]:
    runs = [_run_parallel_operator_resolution(ny=ny, length=length) for ny in resolutions]
    orders: list[dict[str, float | int]] = []
    for coarse, fine in zip(runs[:-1], runs[1:], strict=False):
        orders.append(
            {
                "from_ny": int(coarse["ny"]),
                "to_ny": int(fine["ny"]),
                "gradient_order": _observed_order(float(coarse["gradient_l2"]), float(fine["gradient_l2"])),
                "force_balance_order": _observed_order(
                    float(coarse["force_balance_l2"]),
                    float(fine["force_balance_l2"]),
                ),
            }
        )
    recycling_identity = _target_recycling_identity()
    autodiff_check = _force_balance_autodiff_check(ny=max(resolutions), length=length)
    return {
        "family": "open_field_operator_campaign",
        "case": "open_field_operator_campaign",
        "operator_family": "parallel_gradient_force_balance_target_recycling",
        "claim": (
            "Operator-level verification of the open-field parallel-gradient, electron-force-balance, "
            "and finite-volume target-recycling kernels used by promoted recycling and sheath-connected cases."
        ),
        "literature_anchors": [
            {
                "label": "Dudson et al. MMS verification",
                "url": "https://arxiv.org/abs/1602.06747",
                "role": "Refinement and observed-order tests separate implementation verification from validation.",
            },
            {
                "label": "Hermes-3 equations and boundary documentation",
                "url": "https://hermes3.readthedocs.io/en/latest/boundary_conditions.html",
                "role": "Sheath particle flux, target recycling, and heat/particle source formulas.",
            },
            {
                "label": "Hermes-3 multi-component edge/SOL model paper",
                "url": "https://arxiv.org/abs/2303.12131",
                "role": "Reference code family and edge/SOL multi-species context.",
            },
        ],
        "equations": {
            "parallel_gradient": "D_y f_j = (f_{j+1} - f_{j-1}) / (2 dy)",
            "electron_force_balance": "E_parallel = (-D_y p_e + S_{mom,e}) / max(n_e, n_floor)",
            "target_recycling_density_source": (
                "S_N = R max(0, s 0.25 (n_i+n_g)(v_i+v_g)) A_parallel / V"
            ),
            "target_recycling_energy_source": "S_E = S_N (1 - R_fast) E_recycle",
        },
        "resolutions": [int(ny) for ny in resolutions],
        "length": float(length),
        "convergence_runs": runs,
        "observed_orders": orders,
        "min_observed_order": {
            "gradient": float(min(entry["gradient_order"] for entry in orders)),
            "force_balance": float(min(entry["force_balance_order"] for entry in orders)),
        },
        "target_recycling_identity": recycling_identity,
        "autodiff_check": autodiff_check,
    }


def save_open_field_operator_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    runs = list(report["convergence_runs"])
    orders = list(report["observed_orders"])
    resolutions = np.asarray([entry["ny"] for entry in runs], dtype=np.float64)
    gradient_l2 = np.asarray([entry["gradient_l2"] for entry in runs], dtype=np.float64)
    force_balance_l2 = np.asarray([entry["force_balance_l2"] for entry in runs], dtype=np.float64)

    recycling = dict(report["target_recycling_identity"])
    velocity = np.asarray(recycling["velocity"], dtype=np.float64)
    density_source = np.asarray(recycling["density_source"], dtype=np.float64)
    expected_density_source = np.asarray(recycling["expected_density_source"], dtype=np.float64)
    lower_density_source = np.asarray(recycling["lower_density_source"], dtype=np.float64)
    expected_lower_density_source = np.asarray(recycling["expected_lower_density_source"], dtype=np.float64)
    energy_source = np.asarray(recycling["energy_source"], dtype=np.float64)
    source_ratio = np.asarray(recycling["energy_to_density_ratio"], dtype=np.float64)

    autodiff = dict(report["autodiff_check"])
    gradient_values = np.asarray(
        [autodiff["autodiff_gradient"], autodiff["finite_difference_gradient"]],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)

    axes[0, 0].loglog(resolutions, gradient_l2, marker="o", linewidth=2.0, color="#005f73", label="parallel gradient")
    axes[0, 0].loglog(
        resolutions,
        force_balance_l2,
        marker="s",
        linewidth=2.0,
        color="#ca6702",
        label="force balance",
    )
    style_axis(
        axes[0, 0],
        title="Open-field operator refinement",
        xlabel="interior Ny resolution",
        ylabel="L2 error",
        xscale="log",
        yscale="log",
        grid="both",
    )
    axes[0, 0].legend(frameon=False)

    order_labels = [f"{entry['from_ny']}->{entry['to_ny']}" for entry in orders]
    x = np.arange(len(order_labels))
    width = 0.34
    gradient_order = np.asarray([entry["gradient_order"] for entry in orders], dtype=np.float64)
    force_balance_order = np.asarray([entry["force_balance_order"] for entry in orders], dtype=np.float64)
    axes[0, 1].bar(x - width / 2.0, gradient_order, width=width, color="#005f73", label="parallel gradient")
    axes[0, 1].bar(x + width / 2.0, force_balance_order, width=width, color="#ca6702", label="force balance")
    axes[0, 1].axhline(2.0, color="#9b2226", linestyle="--", linewidth=1.5, label="second order")
    axes[0, 1].set_xticks(x, order_labels)
    style_axis(axes[0, 1], title="Observed refinement order", ylabel="observed order")
    axes[0, 1].legend(frameon=False)
    annotate_bars(axes[0, 1], x - width / 2.0, gradient_order, fmt="{:.2f}", fontsize=8.5)
    annotate_bars(axes[0, 1], x + width / 2.0, force_balance_order, fmt="{:.2f}", fontsize=8.5)

    axes[1, 0].plot(velocity, expected_density_source, color="#001219", linewidth=2.2, label="analytic max(0, R n v)")
    axes[1, 0].scatter(velocity, density_source, color="#ee9b00", s=38, label="JAXDRB source")
    axes[1, 0].plot(
        velocity,
        expected_lower_density_source,
        color="#5f0f40",
        linestyle="--",
        linewidth=2.0,
        label="lower-target analytic",
    )
    axes[1, 0].scatter(velocity, lower_density_source, color="#9b5de5", s=32, label="lower-target source")
    axes[1, 0].plot(velocity, energy_source, color="#0a9396", linewidth=2.0, label="energy source")
    style_axis(
        axes[1, 0],
        title="Target recycling source identity",
        xlabel="target-normalized velocity",
        ylabel="source per cell volume",
    )
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar([0, 1], gradient_values, color=["#005f73", "#ca6702"], width=0.6)
    axes[1, 1].set_xticks([0, 1], ["autodiff", "finite diff"])
    style_axis(
        axes[1, 1],
        title="Force-balance sensitivity check",
        ylabel="d objective / d pressure amplitude",
    )
    annotate_bars(axes[1, 1], np.asarray([0, 1]), gradient_values, fmt="{:.5e}", fontsize=9.0, rotation=8.0)
    axes[1, 1].text(
        0.02,
        0.05,
        f"relative error = {autodiff['relative_error']:.2e}\n"
        f"energy/source ratio = {np.nanmean(source_ratio[source_ratio > 0.0]):.2f}",
        transform=axes[1, 1].transAxes,
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c9c9c9"},
    )

    figure.suptitle(
        "Open-field operator verification and differentiability audit",
        fontsize=15.0,
        fontweight="semibold",
    )
    save_publication_figure(figure, target)
    return target


def _parallel_mesh(*, ny: int, length: float, nx: int = 1) -> StructuredMesh:
    myg = 1
    return StructuredMesh(
        nx=int(nx),
        ny=int(ny),
        nz=1,
        mxg=0,
        myg=myg,
        symmetric_global_x=True,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=int(ny) // 2,
        jyseps1_2=int(ny) // 2,
        jyseps2_2=int(ny) - 1,
        ny_inner=int(ny) // 2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(int(nx), dtype=jnp.float64),
        y=jnp.arange(int(ny) + 2 * myg, dtype=jnp.float64),
        z=jnp.asarray([0.0], dtype=jnp.float64),
    )


def _cell_center_y(*, mesh: StructuredMesh, length: float) -> np.ndarray:
    dy = float(length) / float(mesh.ny)
    indices = np.arange(mesh.local_ny, dtype=np.float64) - float(mesh.myg) + 0.5
    return indices * dy


def _field_from_profile(values: np.ndarray, *, nx: int = 1) -> jnp.ndarray:
    field = np.broadcast_to(np.asarray(values, dtype=np.float64)[None, :, None], (int(nx), values.size, 1))
    return jnp.asarray(field, dtype=jnp.float64)


def _broadcast_metric(value: float, *, mesh: StructuredMesh) -> jnp.ndarray:
    return jnp.full((mesh.nx, mesh.local_ny, mesh.nz), float(value), dtype=jnp.float64)


def _pressure_profile(y: np.ndarray) -> np.ndarray:
    return 1.2 + 0.17 * np.sin(1.7 * y) + 0.05 * np.cos(2.3 * y)


def _pressure_gradient(y: np.ndarray) -> np.ndarray:
    return 0.17 * 1.7 * np.cos(1.7 * y) - 0.05 * 2.3 * np.sin(2.3 * y)


def _density_profile(y: np.ndarray) -> np.ndarray:
    return 1.4 + 0.08 * np.cos(0.9 * y)


def _momentum_source_profile(y: np.ndarray) -> np.ndarray:
    return 0.03 * np.cos(1.1 * y)


def _run_parallel_operator_resolution(*, ny: int, length: float) -> dict[str, float | int]:
    mesh = _parallel_mesh(ny=ny, length=length)
    y = _cell_center_y(mesh=mesh, length=length)
    dy_value = float(length) / float(ny)
    dy = _broadcast_metric(dy_value, mesh=mesh)

    pressure = _field_from_profile(_pressure_profile(y))
    density = _field_from_profile(_density_profile(y))
    momentum_source = _field_from_profile(_momentum_source_profile(y))

    numerical_gradient = grad_par_y(pressure, mesh=mesh, dy=dy)
    force_balance = compute_electron_force_balance(
        pressure,
        density,
        mesh=mesh,
        dy=dy,
        electron_momentum_source=momentum_source,
    )

    exact_gradient = _field_from_profile(_pressure_gradient(y))
    exact_force_density = _field_from_profile(-_pressure_gradient(y) + _momentum_source_profile(y))
    exact_epar = exact_force_density / density

    y_slice = slice(mesh.ystart, mesh.yend + 1)
    gradient_l2 = _l2(np.asarray(numerical_gradient[:, y_slice, :]), np.asarray(exact_gradient[:, y_slice, :]))
    force_balance_l2 = _l2(np.asarray(force_balance.epar[:, y_slice, :]), np.asarray(exact_epar[:, y_slice, :]))
    force_density_l2 = _l2(
        np.asarray(force_balance.force_density[:, y_slice, :]),
        np.asarray(exact_force_density[:, y_slice, :]),
    )
    return {
        "ny": int(ny),
        "dy": dy_value,
        "gradient_l2": gradient_l2,
        "force_balance_l2": force_balance_l2,
        "force_density_l2": force_density_l2,
    }


def _target_recycling_identity() -> dict[str, object]:
    velocity = np.linspace(-1.5, 1.5, 13, dtype=np.float64)
    mesh = _parallel_mesh(ny=1, length=1.0, nx=velocity.size)
    density_value = 1.35
    target_multiplier = 0.92
    target_energy = 3.5

    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density = jnp.full(shape, density_value, dtype=jnp.float64)
    temperature = jnp.full(shape, 4.0, dtype=jnp.float64)
    velocity_field = jnp.broadcast_to(jnp.asarray(velocity, dtype=jnp.float64)[:, None, None], shape)
    unit_metric = jnp.ones(shape, dtype=jnp.float64)
    upper_geometry = build_target_boundary_geometry(
        J=unit_metric,
        dy=unit_metric,
        dx=unit_metric,
        dz=unit_metric,
        g_22=unit_metric,
        y_index=mesh.yend,
        guard_index=mesh.yend + 1,
    )
    lower_geometry = build_target_boundary_geometry(
        J=unit_metric,
        dy=unit_metric,
        dx=unit_metric,
        dz=unit_metric,
        g_22=unit_metric,
        y_index=mesh.ystart,
        guard_index=mesh.ystart - 1,
    )

    upper_result = compute_target_recycling_sources(
        density,
        velocity_field,
        temperature,
        mesh=mesh,
        J=unit_metric,
        dy=unit_metric,
        dx=unit_metric,
        dz=unit_metric,
        g_22=unit_metric,
        target_multiplier=target_multiplier,
        target_energy=target_energy,
        gamma_i=3.5,
        lower_y=False,
        upper_y=True,
        upper_geometry=upper_geometry,
    )
    lower_result = compute_target_recycling_sources(
        density,
        velocity_field,
        temperature,
        mesh=mesh,
        J=unit_metric,
        dy=unit_metric,
        dx=unit_metric,
        dz=unit_metric,
        g_22=unit_metric,
        target_multiplier=target_multiplier,
        target_energy=target_energy,
        gamma_i=3.5,
        lower_y=True,
        upper_y=False,
        lower_geometry=lower_geometry,
    )
    density_source = np.asarray(upper_result.density_source[:, mesh.yend, 0], dtype=np.float64)
    energy_source = np.asarray(upper_result.energy_source[:, mesh.yend, 0], dtype=np.float64)
    lower_density_source = np.asarray(lower_result.density_source[:, mesh.ystart, 0], dtype=np.float64)
    lower_energy_source = np.asarray(lower_result.energy_source[:, mesh.ystart, 0], dtype=np.float64)
    expected_density_source = target_multiplier * np.maximum(0.0, density_value * velocity)
    expected_lower_density_source = target_multiplier * np.maximum(0.0, -density_value * velocity)
    expected_energy_source = expected_density_source * target_energy
    expected_lower_energy_source = expected_lower_density_source * target_energy
    active = expected_density_source > 0.0
    ratio = np.divide(
        energy_source,
        density_source,
        out=np.zeros_like(energy_source),
        where=density_source > 0.0,
    )
    return {
        "velocity": velocity.tolist(),
        "density_source": density_source.tolist(),
        "expected_density_source": expected_density_source.tolist(),
        "lower_density_source": lower_density_source.tolist(),
        "expected_lower_density_source": expected_lower_density_source.tolist(),
        "energy_source": energy_source.tolist(),
        "expected_energy_source": expected_energy_source.tolist(),
        "lower_energy_source": lower_energy_source.tolist(),
        "expected_lower_energy_source": expected_lower_energy_source.tolist(),
        "energy_to_density_ratio": ratio.tolist(),
        "max_density_source_abs_error": float(np.max(np.abs(density_source - expected_density_source))),
        "max_energy_source_abs_error": float(np.max(np.abs(energy_source - expected_energy_source))),
        "max_lower_density_source_abs_error": float(
            np.max(np.abs(lower_density_source - expected_lower_density_source))
        ),
        "max_lower_energy_source_abs_error": float(np.max(np.abs(lower_energy_source - expected_lower_energy_source))),
        "mean_active_energy_to_density_ratio": float(np.mean(ratio[active])),
    }


def _force_balance_autodiff_check(*, ny: int, length: float) -> dict[str, float]:
    mesh = _parallel_mesh(ny=ny, length=length)
    y = _cell_center_y(mesh=mesh, length=length)
    dy = _broadcast_metric(float(length) / float(ny), mesh=mesh)
    base_pressure = _field_from_profile(_pressure_profile(y))
    density = _field_from_profile(_density_profile(y))
    momentum_source = _field_from_profile(_momentum_source_profile(y))
    y_slice = slice(mesh.ystart, mesh.yend + 1)

    def objective(amplitude: jnp.ndarray) -> jnp.ndarray:
        result = compute_electron_force_balance(
            amplitude * base_pressure,
            density,
            mesh=mesh,
            dy=dy,
            electron_momentum_source=momentum_source,
        )
        active = result.epar[:, y_slice, :]
        return jnp.mean(jnp.square(active))

    amplitude = jnp.asarray(1.0, dtype=jnp.float64)
    autodiff_gradient = float(jax.grad(objective)(amplitude))
    eps = 1.0e-5
    finite_difference_gradient = float(
        (objective(amplitude + eps) - objective(amplitude - eps)) / (2.0 * eps)
    )
    abs_error = abs(autodiff_gradient - finite_difference_gradient)
    relative_error = abs_error / max(abs(finite_difference_gradient), 1.0e-30)
    return {
        "amplitude": 1.0,
        "autodiff_gradient": autodiff_gradient,
        "finite_difference_gradient": finite_difference_gradient,
        "absolute_error": float(abs_error),
        "relative_error": float(relative_error),
    }


def _l2(numerical: np.ndarray, exact: np.ndarray) -> float:
    residual = np.asarray(numerical, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))


def _observed_order(coarse_error: float, fine_error: float) -> float:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("inf")
    return float(np.log(coarse_error / fine_error) / np.log(2.0))
