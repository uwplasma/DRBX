from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxdrb.core.closures.sheath import sheath_gamma_e as _sheath_gamma_auto
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState, _state_zeros_like
from .fields import log_rhs, phys_Te, phys_n


def _psi_from_current(geom: GeometryAdapter, jpar: jnp.ndarray) -> jnp.ndarray:
    rhs = -jpar
    return geom.inv_laplacian(rhs)


def _gamma_e_value(params: DRBSystemParams) -> float:
    # Keep legacy behavior (explicit gamma_e, default 0) unless sheath heat
    # transmission is explicitly enabled.
    if bool(params.sheath_gamma_auto) and bool(params.sheath_heat_on):
        return float(_sheath_gamma_auto(params))
    return float(params.sheath_gamma_e)


def _limit_free(fm: jnp.ndarray, fc: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(
        jnp.logical_or(fm < fc, fm < 1e-10),
        fc,
        (fc * fc) / jnp.maximum(fm, 1e-10),
    )


def _face_to_volume_factor(geom: GeometryAdapter) -> tuple[jnp.ndarray, jnp.ndarray]:
    gpar = getattr(geom, "gpar", None)
    if gpar is not None:
        grid = getattr(geom, "grid", None)
        dz = abs(float(getattr(grid, "dz", 1.0))) if grid is not None else 1.0
        dz = max(dz, 1e-30)
        base = 1.0 / (dz * jnp.sqrt(jnp.maximum(jnp.asarray(gpar, dtype=jnp.float64), 1e-30)))
        if base.ndim == 1:
            base = base[:, None, None]
        elif base.ndim == 2:
            base = base[None, :, :]
    else:
        dpar = getattr(geom, "dpar_factor", None)
        if dpar is None:
            base = jnp.asarray(1.0, dtype=jnp.float64)
        else:
            # Fallback when ``gpar`` is unavailable.
            base = jnp.asarray(dpar, dtype=jnp.float64)
            if base.ndim == 1:
                base = base[:, None, None]
            elif base.ndim == 2:
                base = base[None, :, :]
    return base[0], base[-1]


def _hermes_electron_energy_rhs(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, phi: jnp.ndarray
) -> jnp.ndarray:
    if y.Te.shape[0] < 2:
        return jnp.zeros_like(y.Te)

    n_phys = phys_n(params, y.n)
    Te_phys = phys_Te(params, y.Te)

    n_l = n_phys[0]
    n_lp = n_phys[1]
    n_lg = _limit_free(n_lp, n_l)
    n_u = n_phys[-1]
    n_um = n_phys[-2]
    n_ug = _limit_free(n_um, n_u)

    Te_l = Te_phys[0]
    Te_lp = Te_phys[1]
    Te_lg = _limit_free(Te_lp, Te_l)
    Te_u = Te_phys[-1]
    Te_um = Te_phys[-2]
    Te_ug = _limit_free(Te_um, Te_u)

    phi_l = phi[0]
    phi_lp = phi[1]
    phi_lg = 2.0 * phi_l - phi_lp
    phi_u = phi[-1]
    phi_um = phi[-2]
    phi_ug = 2.0 * phi_u - phi_um

    Me = jnp.maximum(float(params.me_hat), 1e-12)
    Ge = jnp.clip(float(params.sheath_secondary_electron_coef), 0.0, 1.0)
    phi_wall = float(params.sheath_wall_potential)
    e_adi = max(float(params.sheath_electron_adiabatic), 1.0 + 1e-8)

    ne_sh_l = 0.5 * (n_l + n_lg)
    ne_sh_u = 0.5 * (n_u + n_ug)
    Te_sh_l = jnp.maximum(0.5 * (Te_l + Te_lg), 1e-10)
    Te_sh_u = jnp.maximum(0.5 * (Te_u + Te_ug), 1e-10)
    phi_sh_l = 0.5 * (phi_l + phi_lg)
    phi_sh_u = 0.5 * (phi_u + phi_ug)
    if bool(params.sheath_floor_potential):
        phi_sh_l = jnp.maximum(phi_sh_l, phi_wall)
        phi_sh_u = jnp.maximum(phi_sh_u, phi_wall)

    gamma_l = jnp.maximum(
        2.0 / (1.0 - Ge) + (phi_sh_l - phi_wall) / jnp.maximum(Te_sh_l, 1e-5),
        0.0,
    )
    gamma_u = jnp.maximum(
        2.0 / (1.0 - Ge) + (phi_sh_u - phi_wall) / jnp.maximum(Te_sh_u, 1e-5),
        0.0,
    )

    pref_l = jnp.sqrt(Te_sh_l / (2.0 * jnp.pi * Me))
    pref_u = jnp.sqrt(Te_sh_u / (2.0 * jnp.pi * Me))
    ve_l = -(1.0 - Ge) * pref_l * jnp.exp(-(phi_sh_l - phi_wall) / jnp.maximum(Te_sh_l, 1e-5))
    ve_u = (1.0 - Ge) * pref_u * jnp.exp(-(phi_sh_u - phi_wall) / jnp.maximum(Te_sh_u, 1e-5))

    q_l = (
        ((gamma_l - 1.0 - 1.0 / (e_adi - 1.0)) * Te_sh_l - 0.5 * Me * ve_l * ve_l) * ne_sh_l * ve_l
    )
    q_u = (
        ((gamma_u - 1.0 - 1.0 / (e_adi - 1.0)) * Te_sh_u - 0.5 * Me * ve_u * ve_u) * ne_sh_u * ve_u
    )
    q_l = jnp.minimum(q_l, 0.0)
    q_u = jnp.maximum(q_u, 0.0)

    fac_l, fac_u = _face_to_volume_factor(geom)
    scale = float(params.sheath_energy_flux_scale)
    # Hermes stores sheath power as an energy source and then converts it to a
    # pressure RHS in evolve_pressure with (gamma - 1) * energy_source.
    pressure_factor = max(e_adi - 1.0, 1e-12)
    dPe = jnp.zeros_like(Te_phys)
    dPe = dPe.at[0].add(scale * pressure_factor * q_l * fac_l)
    dPe = dPe.at[-1].add(-scale * pressure_factor * q_u * fac_u)

    dTe_phys = dPe / jnp.maximum(n_phys, 1e-12)
    return log_rhs(
        params,
        dTe_phys,
        Te_phys,
        max(float(params.temperature_floor), 1e-12),
        bool(params.log_Te),
    )


def _sheath_nu_base(params: DRBSystemParams, geom: GeometryAdapter) -> float:
    grid = getattr(geom, "grid", None)
    geom_line = getattr(geom, "geom", None)
    Lpar = 1.0
    if grid is not None and hasattr(grid, "dz"):
        nz = int(getattr(grid, "nz", len(np.asarray(getattr(grid, "z")))))
        dz = abs(float(grid.dz))
        if bool(getattr(grid, "open_field_line", False)):
            Lpar = dz * float(max(nz - 1, 1))
        else:
            Lpar = dz * float(max(nz, 1))
    elif grid is not None and hasattr(grid, "l"):
        l = np.asarray(grid.l, dtype=float)
        if l.size >= 2:
            Lpar = abs(float(l[-1] - l[0]))
    elif grid is not None and hasattr(grid, "z"):
        z = np.asarray(grid.z, dtype=float)
        if z.size >= 2:
            Lpar = abs(float(z[-1] - z[0]))
    elif geom_line is not None and hasattr(geom_line, "l"):
        l = np.asarray(geom_line.l, dtype=float)
        if l.size >= 2:
            Lpar = abs(float(l[-1] - l[0]))
    elif hasattr(geom, "l"):
        l = np.asarray(getattr(geom, "l"), dtype=float)
        if l.size >= 2:
            Lpar = abs(float(l[-1] - l[0]))
    factor = float(params.sheath_nu_factor)
    if not bool(params.sheath_on) and bool(params.sheath_bc_on):
        factor = float(params.sheath_bc_nu_factor)
    return float(factor) * (2.0 / (float(Lpar) + 1e-30))


def _sheath_nu(params: DRBSystemParams, geom: GeometryAdapter) -> tuple[float, float, float]:
    nu_base = _sheath_nu_base(params, geom)
    nu_m = float(params.sheath_nu_mom)
    nu_p = float(params.sheath_nu_particle)
    nu_e = float(params.sheath_nu_energy)
    if nu_m == 0.0 and (params.sheath_on or params.sheath_bc_on):
        nu_m = nu_base
    if nu_p == 0.0 and params.sheath_bc_on:
        nu_p = nu_base
    if nu_e == 0.0 and params.sheath_bc_on:
        nu_e = nu_base
    return nu_m, nu_p, nu_e


def sheath_terms(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, phi: jnp.ndarray
) -> DRBSystemState:
    if not (params.sheath_on or params.sheath_bc_on):
        return _state_zeros_like(y)
    model = params.sheath_bc_model_fci
    if isinstance(params.sheath_bc_model, str):
        model = params.sheath_bc_model
    if isinstance(model, int):
        model = "loizu_linear" if int(model) == 1 else "simple"
    if model in {"loizu_linear", "loizu2012", "loizu"}:
        return _sheath_loizu_linear(params, geom, y, phi)
    if model in {"bohm_current", "bohm", "stangeby"}:
        return _sheath_bohm_current(params, geom, y, phi)
    return _sheath_simple(params, geom, y, phi)


def _sheath_simple(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, phi: jnp.ndarray
) -> DRBSystemState:
    mask, sign = geom.sheath_mask_sign()
    dve = jnp.zeros_like(y.vpar_e)
    dvi = jnp.zeros_like(y.vpar_i)
    dn = jnp.zeros_like(y.n)
    domega = jnp.zeros_like(y.omega)
    dTe = jnp.zeros_like(y.Te)
    dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
    dpsi = None if y.psi is None else jnp.zeros_like(y.psi)
    gamma_e = _gamma_e_value(params)

    nu_m, nu_p, nu_e = _sheath_nu(params, geom)
    if nu_m != 0.0:
        hot_on = bool(params.hot_ion_on) and (y.Ti is not None)
        tau_i = float(params.tau_i) if hot_on else 0.0
        cs0 = jnp.sqrt(1.0 + tau_i)
        dcs = (
            0.5 * (y.Te + (y.Ti if hot_on and y.Ti is not None else 0.0)) / jnp.maximum(cs0, 1e-12)
        )
        vpar_i_target = sign * (1.0 - float(params.sheath_delta)) * dcs
        vpar_e_target = sign * (dcs - phi)
        dvi = dvi - nu_m * mask * (y.vpar_i - vpar_i_target)
        dve = dve - nu_m * mask * (y.vpar_e - vpar_e_target)

    if nu_p != 0.0:
        dn = dn - nu_p * mask * y.n
        domega = domega - nu_p * mask * y.omega

    if nu_e != 0.0:
        dTe = dTe - nu_e * gamma_e * mask * y.Te
        if dTi is not None:
            dTi = dTi - nu_e * params.sheath_gamma_i * mask * y.Ti

    if dpsi is not None and params.em_on:
        dj_sh = dvi - dve
        dpsi = _psi_from_current(geom, dj_sh)

    return DRBSystemState(
        n=dn,
        omega=domega,
        vpar_e=dve,
        vpar_i=dvi,
        Te=dTe,
        Ti=dTi,
        psi=dpsi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _sheath_bohm_current(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, phi: jnp.ndarray
) -> DRBSystemState:
    """Bohm + current-balance sheath without direct particle damping."""

    mask, sign = geom.sheath_mask_sign()
    dve = jnp.zeros_like(y.vpar_e)
    dvi = jnp.zeros_like(y.vpar_i)
    dn = jnp.zeros_like(y.n)
    domega = jnp.zeros_like(y.omega)
    dTe = jnp.zeros_like(y.Te)
    dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
    dpsi = None if y.psi is None else jnp.zeros_like(y.psi)
    nu_m, nu_p, nu_e = _sheath_nu(params, geom)
    if nu_m != 0.0:
        hot_on = bool(params.hot_ion_on) and (y.Ti is not None)
        tau_i = float(params.tau_i) if hot_on else 0.0
        cs0 = jnp.sqrt(1.0 + tau_i)
        dcs = (
            0.5 * (y.Te + (y.Ti if hot_on and y.Ti is not None else 0.0)) / jnp.maximum(cs0, 1e-12)
        )
        vpar_i_target = sign * (1.0 - float(params.sheath_delta)) * dcs
        vpar_e_target = sign * (dcs - phi)
        dvi = dvi - nu_m * mask * (y.vpar_i - vpar_i_target)
        dve = dve - nu_m * mask * (y.vpar_e - vpar_e_target)

    if nu_p != 0.0 and bool(params.sheath_loss_on):
        dn = dn - nu_p * mask * y.n
        domega = domega - nu_p * mask * y.omega

    energy_model = str(getattr(params, "sheath_energy_model", "relaxation")).lower()
    if energy_model == "hermes_flux":
        dTe = dTe + _hermes_electron_energy_rhs(params, geom, y, phi)
    elif nu_e != 0.0:
        gamma_e = _gamma_e_value(params)
        dTe = dTe - nu_e * gamma_e * mask * y.Te
        if dTi is not None:
            dTi = dTi - nu_e * params.sheath_gamma_i * mask * y.Ti

    if dpsi is not None and params.em_on:
        dj_sh = dvi - dve
        dpsi = _psi_from_current(geom, dj_sh)

    return DRBSystemState(
        n=dn,
        omega=domega,
        vpar_e=dve,
        vpar_i=dvi,
        Te=dTe,
        Ti=dTi,
        psi=dpsi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


def _sheath_loizu_linear(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, phi: jnp.ndarray
) -> DRBSystemState:
    grid = getattr(geom, "grid", None)
    geom_line = getattr(geom, "geom", None)
    line = None
    if grid is not None and hasattr(grid, "l"):
        line = jnp.asarray(grid.l)
    elif grid is not None and hasattr(grid, "z"):
        line = jnp.asarray(grid.z)
    elif geom_line is not None and hasattr(geom_line, "l"):
        line = jnp.asarray(geom_line.l)
    elif hasattr(geom, "l"):
        line = jnp.asarray(getattr(geom, "l"))
    if line is None or line.size < 5:
        return _sheath_simple(params, geom, y, phi)
    nz = int(getattr(grid, "nz", int(line.size)))

    mask, sign = geom.sheath_mask_sign()
    nu_m, nu_p, nu_e = _sheath_nu(params, geom)
    gamma_e = _gamma_e_value(params)

    dn = jnp.zeros_like(y.n)
    domega = jnp.zeros_like(y.omega)
    dve = jnp.zeros_like(y.vpar_e)
    dvi = jnp.zeros_like(y.vpar_i)
    dTe = jnp.zeros_like(y.Te)
    dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
    dpsi = None if y.psi is None else jnp.zeros_like(y.psi)

    hot_on = bool(params.hot_ion_on) and (y.Ti is not None)
    tau_i = float(params.tau_i) if hot_on else 0.0
    cs0 = jnp.sqrt(1.0 + tau_i)
    inv_cs0 = 1.0 / jnp.maximum(cs0, 1e-12)
    delta = float(params.sheath_delta)
    cos2 = jnp.maximum(float(params.sheath_cos2), 1e-8)

    left = 0
    right = nz - 1
    mask_l = jnp.asarray(mask[left], dtype=y.n.dtype)
    mask_r = jnp.asarray(mask[right], dtype=y.n.dtype)
    sign_l = jnp.where(sign[left] != 0.0, sign[left], -1.0)
    sign_r = jnp.where(sign[right] != 0.0, sign[right], +1.0)

    Ti_arr = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
    dcs_l = 0.5 * inv_cs0 * (y.Te[left] + Ti_arr[left])
    dcs_r = 0.5 * inv_cs0 * (y.Te[right] + Ti_arr[right])

    vi_bc_l = sign_l * (1.0 - delta) * dcs_l
    vi_bc_r = sign_r * (1.0 - delta) * dcs_r

    phi_bc_l = phi[1] + cs0 * (y.vpar_i[1] - vi_bc_l)
    phi_bc_r = phi[-2] + cs0 * (y.vpar_i[-2] - vi_bc_r)

    n_bc_l = y.n[1] + inv_cs0 * (y.vpar_i[1] - vi_bc_l)
    n_bc_r = y.n[-2] + inv_cs0 * (y.vpar_i[-2] - vi_bc_r)

    dcs_bc_l = 0.5 * inv_cs0 * (y.Te[1] + Ti_arr[1])
    dcs_bc_r = 0.5 * inv_cs0 * (y.Te[-2] + Ti_arr[-2])
    ve_bc_l = sign_l * (dcs_bc_l - phi_bc_l)
    ve_bc_r = sign_r * (dcs_bc_r - phi_bc_r)

    phi_target = phi
    phi_target = phi_target.at[left].set(phi_bc_l)
    phi_target = phi_target.at[right].set(phi_bc_r)
    omega_from_phi = geom.laplacian(phi_target)
    omega_bc_l = omega_from_phi[left]
    omega_bc_r = omega_from_phi[right]

    dl = jnp.maximum(jnp.asarray(jnp.mean(jnp.diff(line)), dtype=y.n.dtype), 1e-8)
    dl2 = dl * dl
    v2_target_l = -omega_bc_l / (cos2 * cs0)
    v2_target_r = -omega_bc_r / (cos2 * cs0)
    vi_adj_l = (2.0 * vi_bc_l + 4.0 * y.vpar_i[2] - y.vpar_i[3] - dl2 * v2_target_l) / 5.0
    vi_adj_r = (2.0 * vi_bc_r + 4.0 * y.vpar_i[-3] - y.vpar_i[-4] - dl2 * v2_target_r) / 5.0

    if nu_m != 0.0:
        dvi = dvi.at[left].add(-nu_m * mask_l * (y.vpar_i[left] - vi_bc_l))
        dvi = dvi.at[right].add(-nu_m * mask_r * (y.vpar_i[right] - vi_bc_r))
        dvi = dvi.at[1].add(-nu_m * mask_l * (y.vpar_i[1] - vi_adj_l))
        dvi = dvi.at[-2].add(-nu_m * mask_r * (y.vpar_i[-2] - vi_adj_r))

        dve = dve.at[left].add(-nu_m * mask_l * (y.vpar_e[left] - ve_bc_l))
        dve = dve.at[right].add(-nu_m * mask_r * (y.vpar_e[right] - ve_bc_r))

    if nu_p != 0.0:
        dn = dn.at[left].add(-nu_p * mask_l * (y.n[left] - n_bc_l))
        dn = dn.at[right].add(-nu_p * mask_r * (y.n[right] - n_bc_r))
        domega = domega.at[left].add(-nu_p * mask_l * (y.omega[left] - omega_bc_l))
        domega = domega.at[right].add(-nu_p * mask_r * (y.omega[right] - omega_bc_r))

    if nu_e != 0.0:
        Te_bc_l = y.Te[1]
        Te_bc_r = y.Te[-2]
        dTe = dTe.at[left].add(
            -nu_e * mask_l * (y.Te[left] - Te_bc_l) - nu_e * gamma_e * mask_l * y.Te[left]
        )
        dTe = dTe.at[right].add(
            -nu_e * mask_r * (y.Te[right] - Te_bc_r) - nu_e * gamma_e * mask_r * y.Te[right]
        )
        if dTi is not None and y.Ti is not None:
            Ti_bc_l = y.Ti[1]
            Ti_bc_r = y.Ti[-2]
            dTi = dTi.at[left].add(
                -nu_e * mask_l * (y.Ti[left] - Ti_bc_l)
                - nu_e * params.sheath_gamma_i * mask_l * y.Ti[left]
            )
            dTi = dTi.at[right].add(
                -nu_e * mask_r * (y.Ti[right] - Ti_bc_r)
                - nu_e * params.sheath_gamma_i * mask_r * y.Ti[right]
            )

    if dpsi is not None and params.em_on:
        dj_sh = dvi - dve
        dpsi = _psi_from_current(geom, dj_sh)

    return DRBSystemState(
        n=dn,
        omega=domega,
        vpar_e=dve,
        vpar_i=dvi,
        Te=dTe,
        Ti=dTi,
        psi=dpsi if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
