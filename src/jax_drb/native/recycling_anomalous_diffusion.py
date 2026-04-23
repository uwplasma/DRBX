from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.boutinp import BoutConfig
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .neutral_mixed import _div_a_grad_perp_flows
from .recycling_setup import OpenFieldSpecies, resolve_species_numeric_option
from .recycling_state import axisymmetric_profile, raw_species_velocity, safe_temperature


@dataclass(frozen=True)
class AnomalousDiffusionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def apply_anomalous_diffusion(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> AnomalousDiffusionTerms:
    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}

    supports_nz1_nonorthogonal = mesh.nz == 1 and getattr(metrics, "g_23", None) is not None
    if (
        not np.allclose(np.asarray(metrics.g23, dtype=np.float64), 0.0, rtol=1.0e-12, atol=1.0e-12)
        and not supports_nz1_nonorthogonal
    ):
        return AnomalousDiffusionTerms(
            density_source=density_source,
            energy_source=energy_source,
            momentum_source=momentum_source,
            diagnostics=diagnostics,
        )

    diffusion_norm = float(dataset_scalars["rho_s0"]) ** 2 * float(dataset_scalars["Omega_ci"])
    anomalous_operator = (
        div_a_grad_perp_upwind_flows_nz1
        if supports_nz1_nonorthogonal
        else _div_a_grad_perp_flows
    )

    for name, sp in species.items():
        if not config.has_section(name):
            continue

        include_d = config.has_option(name, "anomalous_D") and abs(resolve_species_numeric_option(config, name, "anomalous_D")) > 0.0
        include_chi = config.has_option(name, "anomalous_chi") and abs(resolve_species_numeric_option(config, name, "anomalous_chi")) > 0.0
        include_nu = config.has_option(name, "anomalous_nu") and abs(resolve_species_numeric_option(config, name, "anomalous_nu")) > 0.0
        if not (include_d or include_chi or include_nu):
            continue

        anomalous_sheath_flux = (
            bool(config.parsed(name, "anomalous_sheath_flux"))
            if config.has_option(name, "anomalous_sheath_flux")
            else False
        )
        density_2d = axisymmetric_profile(sp.density)
        temperature_2d = axisymmetric_profile(safe_temperature(sp.pressure, sp.density, sp.density_floor))
        if sp.has_momentum:
            velocity_2d = axisymmetric_profile(raw_species_velocity(sp))
        else:
            velocity_2d = np.zeros_like(sp.density, dtype=np.float64)

        if not anomalous_sheath_flux:
            density_2d[:, mesh.ystart - 1, :] = density_2d[:, mesh.ystart, :]
            density_2d[:, mesh.yend + 1, :] = density_2d[:, mesh.yend, :]
            temperature_2d[:, mesh.ystart - 1, :] = temperature_2d[:, mesh.ystart, :]
            temperature_2d[:, mesh.yend + 1, :] = temperature_2d[:, mesh.yend, :]
            velocity_2d[:, mesh.ystart - 1, :] = velocity_2d[:, mesh.ystart, :]
            velocity_2d[:, mesh.yend + 1, :] = velocity_2d[:, mesh.yend, :]

        anomalous_d = (
            np.full_like(sp.density, resolve_species_numeric_option(config, name, "anomalous_D") / diffusion_norm, dtype=np.float64)
            if include_d
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        anomalous_chi = (
            np.full_like(sp.density, resolve_species_numeric_option(config, name, "anomalous_chi") / diffusion_norm, dtype=np.float64)
            if include_chi
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        anomalous_nu = (
            np.full_like(sp.density, resolve_species_numeric_option(config, name, "anomalous_nu") / diffusion_norm, dtype=np.float64)
            if include_nu
            else np.zeros_like(sp.density, dtype=np.float64)
        )

        if include_d:
            density_source[name] = density_source[name] + anomalous_operator(
                anomalous_d,
                density_2d,
                mesh=mesh,
                metrics=metrics,
            )
            if sp.has_momentum:
                momentum_source[name] = momentum_source[name] + anomalous_operator(
                    sp.atomic_mass * velocity_2d * anomalous_d,
                    density_2d,
                    mesh=mesh,
                    metrics=metrics,
                )
            if sp.has_pressure:
                energy_source[name] = energy_source[name] + anomalous_operator(
                    1.5 * temperature_2d * anomalous_d,
                    density_2d,
                    mesh=mesh,
                    metrics=metrics,
                )
        if include_chi and sp.has_pressure:
            energy_source[name] = energy_source[name] + anomalous_operator(
                anomalous_chi * density_2d,
                temperature_2d,
                mesh=mesh,
                metrics=metrics,
            )
        if include_nu and sp.has_momentum:
            momentum_source[name] = momentum_source[name] + anomalous_operator(
                anomalous_nu * sp.atomic_mass * density_2d,
                velocity_2d,
                mesh=mesh,
                metrics=metrics,
            )
        if include_d:
            diagnostics[f"anomalous_D_{name}"] = anomalous_d
        if include_chi:
            diagnostics[f"anomalous_Chi_{name}"] = anomalous_chi
        if include_nu:
            diagnostics[f"anomalous_nu_{name}"] = anomalous_nu

    return AnomalousDiffusionTerms(
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )


def div_a_grad_perp_upwind_flows_nz1(
    coefficient: np.ndarray,
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    """Subset of Hermes Div_a_Grad_perp_upwind_flows for nz=1 tokamak meshes."""
    result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g23 = np.asarray(metrics.g23, dtype=np.float64)
    g_23 = np.asarray(
        metrics.g_23 if metrics.g_23 is not None else np.zeros_like(g23),
        dtype=np.float64,
    )
    Bxy = np.asarray(metrics.Bxy, dtype=np.float64)

    ix_edge = slice(mesh.xstart - 1, mesh.xend + 1)
    ix_next = slice(mesh.xstart, mesh.xend + 2)
    jy = slice(mesh.ystart, mesh.yend + 1)

    gradient = (
        (J[ix_edge, jy, :] * g11[ix_edge, jy, :] + J[ix_next, jy, :] * g11[ix_next, jy, :])
        * (field[ix_next, jy, :] - field[ix_edge, jy, :])
        / (dx[ix_edge, jy, :] + dx[ix_next, jy, :])
    )
    flux = gradient * np.where(gradient > 0.0, coefficient[ix_next, jy, :], coefficient[ix_edge, jy, :])
    result[ix_edge, jy, :] += flux / (dx[ix_edge, jy, :] * J[ix_edge, jy, :])
    result[ix_next, jy, :] -= flux / (dx[ix_next, jy, :] * J[ix_next, jy, :])

    if np.allclose(g23, 0.0, rtol=1.0e-12, atol=1.0e-12):
        return result

    ix = slice(mesh.xstart, mesh.xend + 1)
    jminus = slice(mesh.ystart - 1, mesh.yend)
    jplus = slice(mesh.ystart + 1, mesh.yend + 2)

    coef_up = 0.5 * (
        g_23[ix, jy, :] / np.square(J[ix, jy, :] * Bxy[ix, jy, :])
        + g_23[ix, jplus, :] / np.square(J[ix, jplus, :] * Bxy[ix, jplus, :])
    )
    dfdy_up = 2.0 * (field[ix, jplus, :] - field[ix, jy, :]) / (dy[ix, jplus, :] + dy[ix, jy, :])
    flux_up = (
        0.25
        * (coefficient[ix, jy, :] + coefficient[ix, jplus, :])
        * (J[ix, jy, :] * g23[ix, jy, :] + J[ix, jplus, :] * g23[ix, jplus, :])
        * (-coef_up * dfdy_up)
    )
    result[ix, jy, :] += flux_up / (dy[ix, jy, :] * J[ix, jy, :])

    coef_down = 0.5 * (
        g_23[ix, jy, :] / np.square(J[ix, jy, :] * Bxy[ix, jy, :])
        + g_23[ix, jminus, :] / np.square(J[ix, jminus, :] * Bxy[ix, jminus, :])
    )
    dfdy_down = 2.0 * (field[ix, jy, :] - field[ix, jminus, :]) / (dy[ix, jy, :] + dy[ix, jminus, :])
    flux_down = (
        0.25
        * (coefficient[ix, jy, :] + coefficient[ix, jminus, :])
        * (J[ix, jy, :] * g23[ix, jy, :] + J[ix, jminus, :] * g23[ix, jminus, :])
        * (-coef_down * dfdy_down)
    )
    result[ix, jy, :] -= flux_down / (dy[ix, jy, :] * J[ix, jy, :])

    return result
