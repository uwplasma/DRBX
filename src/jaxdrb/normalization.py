from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192369e-27


@dataclass(frozen=True)
class NormalizationInfo:
    """Reference scales for mapping physical inputs to normalized units."""

    length: float
    time: float
    density: float
    temperature: float
    potential: float
    velocity: float
    rho_s: float
    cs: float
    omega_ci: float
    B0: float
    n0: float
    Te0_eV: float
    Ti0_eV: float
    m_i: float
    Z_i: float


def _as_float(val: Any, default: float) -> float:
    if val is None:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def compute_normalization(cfg: dict[str, Any]) -> NormalizationInfo:
    mode = str(cfg.get("mode", "physics")).lower()

    if mode == "manual":
        length = _as_float(cfg.get("length"), 1.0)
        time = _as_float(cfg.get("time"), 1.0)
        density = _as_float(cfg.get("density"), 1.0)
        temperature = _as_float(cfg.get("temperature"), 1.0)
        potential = _as_float(cfg.get("potential"), temperature)
        velocity = _as_float(cfg.get("velocity"), length / max(time, 1e-30))
        rho_s = _as_float(cfg.get("rho_s"), length)
        cs = _as_float(cfg.get("cs"), velocity)
        omega_ci = _as_float(cfg.get("omega_ci"), 1.0 / max(time, 1e-30))
        B0 = _as_float(cfg.get("B0"), 1.0)
        n0 = _as_float(cfg.get("n0"), density)
        Te0_eV = _as_float(cfg.get("Te0_eV"), temperature)
        Ti0_eV = _as_float(cfg.get("Ti0_eV"), Te0_eV)
        m_i = _as_float(cfg.get("m_i"), 2.0 * M_PROTON)
        Z_i = _as_float(cfg.get("Z_i"), 1.0)
        return NormalizationInfo(
            length=length,
            time=time,
            density=density,
            temperature=temperature,
            potential=potential,
            velocity=velocity,
            rho_s=rho_s,
            cs=cs,
            omega_ci=omega_ci,
            B0=B0,
            n0=n0,
            Te0_eV=Te0_eV,
            Ti0_eV=Ti0_eV,
            m_i=m_i,
            Z_i=Z_i,
        )

    Te0_eV = _as_float(cfg.get("Te0_eV", cfg.get("Te0")), 1.0)
    Ti0_eV = _as_float(cfg.get("Ti0_eV", cfg.get("Ti0", Te0_eV)), Te0_eV)
    n0 = _as_float(cfg.get("n0", cfg.get("n0_m3")), 1.0)
    B0 = _as_float(cfg.get("B0", cfg.get("B0_T")), 1.0)
    m_i_amu = _as_float(cfg.get("m_i_amu", cfg.get("ion_amu")), 2.0)
    Z_i = _as_float(cfg.get("Z_i", cfg.get("ion_Z")), 1.0)

    m_i = m_i_amu * M_PROTON
    Te0_J = Te0_eV * E_CHARGE
    Ti0_J = Ti0_eV * E_CHARGE
    cs_include_Ti = bool(cfg.get("cs_include_Ti", False))
    Te_for_cs = Te0_J + (Ti0_J if cs_include_Ti else 0.0)
    cs = math.sqrt(max(Te_for_cs, 1e-30) / max(m_i, 1e-30))
    omega_ci = (Z_i * E_CHARGE * B0 / max(m_i, 1e-30)) if B0 != 0.0 else 1.0
    rho_s = cs / max(omega_ci, 1e-30)

    length_unit = str(cfg.get("length_unit", "rho_s")).lower()
    if length_unit in ("lref", "l_ref", "lref_m"):
        length = _as_float(cfg.get("Lref_m", cfg.get("Lref")), rho_s)
    else:
        length = rho_s

    time = length / max(cs, 1e-30)
    velocity = cs
    temperature = Te0_eV
    potential = Te0_eV

    return NormalizationInfo(
        length=length,
        time=time,
        density=n0,
        temperature=temperature,
        potential=potential,
        velocity=velocity,
        rho_s=rho_s,
        cs=cs,
        omega_ci=omega_ci,
        B0=B0,
        n0=n0,
        Te0_eV=Te0_eV,
        Ti0_eV=Ti0_eV,
        m_i=m_i,
        Z_i=Z_i,
    )


def _scale_value(value: Any, scale: float) -> Any:
    if isinstance(value, (int, float)):
        return float(value) * scale
    return value


def _scale_section(section: dict[str, Any], keys: dict[str, float]) -> dict[str, Any]:
    out = dict(section)
    for key, scale in keys.items():
        if key in out:
            out[key] = _scale_value(out[key], scale)
    return out


def apply_normalization(cfg: dict[str, Any]) -> tuple[dict[str, Any], NormalizationInfo | None]:
    norm_cfg = cfg.get("normalization", {})
    if not isinstance(norm_cfg, dict) or not bool(norm_cfg.get("enabled", False)):
        return cfg, None

    info = compute_normalization(norm_cfg)

    length_scale = max(info.length, 1e-30)
    time_scale = max(info.time, 1e-30)
    density_scale = max(info.density, 1e-30)
    temperature_scale = max(info.temperature, 1e-30)
    potential_scale = max(info.potential, 1e-30)
    velocity_scale = max(info.velocity, 1e-30)
    B_scale = max(info.B0, 1e-30)

    length = 1.0 / length_scale
    inv_length = length_scale
    density = 1.0 / density_scale
    temperature = 1.0 / temperature_scale
    potential = 1.0 / potential_scale
    velocity = 1.0 / velocity_scale
    rate = time_scale
    diffusivity = time_scale / (length_scale**2)
    diffusivity4 = time_scale / (length_scale**4)
    magnetic = 1.0 / B_scale

    out = dict(cfg)

    # Geometry physical inputs (lengths and fields).
    geom_phys = cfg.get("geometry_physical", None)
    if isinstance(geom_phys, dict):
        geom = dict(cfg.get("geometry", {}))
        converted = _scale_section(
            geom_phys,
            {
                "Lx": length,
                "Ly": length,
                "Lz": length,
                "R0": length,
                "r0": length,
                "r_minor": length,
                "a": length,
                "sigma0": length,
                "R1": length,
                "Z1": length,
                "Z2": length,
                "R_start": length,
                "Z_start": length,
                "rho_s0": length,
                "x_min": length,
                "x_max": length,
                "y_min": length,
                "y_max": length,
                "B0": magnetic,
                "B0_T": magnetic,
            },
        )
        geom.update(converted)
        out["geometry"] = geom
        out.pop("geometry_physical", None)

    # Physics physical inputs.
    phys_phys = cfg.get("physics_physical", None)
    if isinstance(phys_phys, dict):
        physics = dict(cfg.get("physics", {}))
        source_x_mode = phys_phys.get("source_x_mode", physics.get("source_x_mode", "grid"))
        source_x_scale = 1.0 if str(source_x_mode).lower() == "bout" else length
        converted = _scale_section(
            phys_phys,
            {
                "omega_n": inv_length,
                "omega_Te": inv_length,
                "omega_Ti": inv_length,
                "n0": density,
                "n0_min": density,
                "n0_max": density,
                "source_n0": time_scale * density,
                "source_Te0": time_scale * temperature,
                "source_x0": source_x_scale,
                "source_y0": length,
                "source_width_x": source_x_scale,
                "source_width_y": length,
            },
        )
        physics.update(converted)
        if "tau_i" not in physics and info.Te0_eV > 0.0:
            physics["tau_i"] = float(info.Ti0_eV / info.Te0_eV)
        out["physics"] = physics
        out.pop("physics_physical", None)

    # Transport physical inputs (diffusivity/rates).
    trans_phys = cfg.get("transport_physical", None)
    if isinstance(trans_phys, dict):
        transport = dict(cfg.get("transport", {}))
        converted = _scale_section(
            trans_phys,
            {
                "Dn": diffusivity,
                "DOmega": diffusivity,
                "Dvpar": diffusivity,
                "DTe": diffusivity,
                "DTi": diffusivity,
                "Dpsi": diffusivity,
                "Dn4": diffusivity4,
                "DOmega4": diffusivity4,
                "DTe4": diffusivity4,
                "DTi4": diffusivity4,
                "Dpsi4": diffusivity4,
                "nu_par_e": rate,
                "nu_par_i": rate,
                "nu_sink_n": rate,
                "nu_sink_Te": rate,
                "nu_sink_vpar": rate,
                "mu_zonal_omega": rate,
                "mu_lin_n": rate,
                "mu_lin_omega": rate,
                "mu_lin_vpar_e": rate,
                "mu_lin_vpar_i": rate,
                "mu_lin_Te": rate,
            },
        )
        transport.update(converted)
        out["transport"] = transport
        out.pop("transport_physical", None)

    # Closure physical inputs (SOL lengths/rates).
    closures_phys = cfg.get("closures_physical", None)
    if isinstance(closures_phys, dict):
        closures = dict(cfg.get("closures", {}))
        sol_phys = closures_phys.get("sol", None)
        if isinstance(sol_phys, dict):
            sol_base = dict(closures.get("sol", {}))
            sol_converted = _scale_section(
                sol_phys,
                {
                    "sol_n_core": density,
                    "sol_n_sol": density,
                    "sol_Te_core": temperature,
                    "sol_Te_sol": temperature,
                    "sol_xs": length,
                    "sol_width": length,
                    "sol_source_xs": length,
                    "sol_source_width": length,
                    "sol_source2_xs": length,
                    "sol_source2_width": length,
                    "sol_source_y_taper": length,
                    "sol_mask_y_taper": length,
                    "sol_source_n0": time_scale * density,
                    "sol_source_Te0": time_scale * temperature,
                    "sol_source2_n0": time_scale * density,
                    "sol_source2_Te0": time_scale * temperature,
                    "sol_relax_core": rate,
                    "sol_relax_open": rate,
                    "sol_sink_open_n": rate,
                    "sol_sink_open_Te": rate,
                    "sol_sink_open_omega": rate,
                    "sol_sink_open_vpar": rate,
                    "sol_edge_relax_nu": rate,
                    "sol_omega_bc_nu": rate,
                    "sol_vpar_bc_nu": rate,
                },
            )
            sol_base.update(sol_converted)
            closures["sol"] = sol_base
        out["closures"] = closures
        out.pop("closures_physical", None)

    # Initial physical inputs (density/temperature/velocity scales).
    init_phys = cfg.get("initial_physical", None)
    if isinstance(init_phys, dict):
        init = dict(cfg.get("initial", {}))
        converted = _scale_section(
            init_phys,
            {
                "n0": density,
                "Te0": temperature,
                "Ti0": temperature,
                "phi0": potential,
                "vpar_e0": velocity,
                "vpar_i0": velocity,
            },
        )
        init.update(converted)
        out["initial"] = init
        out.pop("initial_physical", None)

    # BC physical inputs (timescales -> normalized rates).
    bc_phys = cfg.get("bc_physical", None)
    if isinstance(bc_phys, dict):
        bc = dict(cfg.get("bc", {}))

        def _set_rate(timescale_key: str, rate_key: str) -> None:
            if rate_key in bc:
                return
            if timescale_key not in bc_phys:
                return
            tau = _as_float(bc_phys.get(timescale_key), 0.0)
            if tau <= 0.0:
                return
            bc[rate_key] = time_scale / tau

        # Global and phi-specific relaxation timescales.
        _set_rate("bc_enforce_timescale", "bc_enforce_nu")
        _set_rate("bc_enforce_timescale_phi", "bc_enforce_nu_phi")
        _set_rate("phi_boundary_timescale", "bc_enforce_nu_phi")

        out["bc"] = bc
        out.pop("bc_physical", None)

    # Numerics defaults (Poisson scaling).
    # If lengths are normalized by Lref, the normalized vorticity satisfies
    #   omega ~ (rho_s / Lref)^2 * ∇⊥^2 phi
    # so we set poisson_scale accordingly unless the user overrides it.
    numerics = dict(cfg.get("numerics", {}))
    if "poisson_scale" not in numerics:
        poisson_scale = (info.rho_s / max(info.length, 1e-30)) ** 2
        numerics["poisson_scale"] = float(poisson_scale)
    out["numerics"] = numerics

    return out, info
