from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState, _state_zeros_like


def _psi_from_current(geom: GeometryAdapter, jpar: jnp.ndarray) -> jnp.ndarray:
    rhs = -jpar
    return geom.inv_laplacian(rhs)


def _sheath_nu_base(params: DRBSystemParams, geom: GeometryAdapter) -> float:
    grid = getattr(geom, "grid", None)
    geom_line = getattr(geom, "geom", None)
    l = None
    if grid is not None and hasattr(grid, "l"):
        l = jnp.asarray(grid.l)
    elif geom_line is not None and hasattr(geom_line, "l"):
        l = jnp.asarray(geom_line.l)
    elif hasattr(geom, "l"):
        l = jnp.asarray(getattr(geom, "l"))
    if l is None or l.size < 2:
        Lpar = 1.0
    else:
        Lpar = jnp.abs(l[-1] - l[0])
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

    nu_m, nu_p, nu_e = _sheath_nu(params, geom)
    if nu_m != 0.0:
        hot_on = bool(params.hot_ion_on) and (y.Ti is not None)
        tau_i = float(params.tau_i) if hot_on else 0.0
        cs0 = jnp.sqrt(1.0 + tau_i)
        dcs = (
            0.5
            * (y.Te + (y.Ti if hot_on and y.Ti is not None else 0.0))
            / jnp.maximum(cs0, 1e-12)
        )
        vpar_i_target = sign * (1.0 - float(params.sheath_delta)) * dcs
        vpar_e_target = sign * (dcs - phi)
        dvi = dvi - nu_m * mask * (y.vpar_i - vpar_i_target)
        dve = dve - nu_m * mask * (y.vpar_e - vpar_e_target)

    if nu_p != 0.0:
        dn = dn - nu_p * mask * y.n
        domega = domega - nu_p * mask * y.omega

    if nu_e != 0.0:
        dTe = dTe - nu_e * params.sheath_gamma_e * mask * y.Te
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
    elif geom_line is not None and hasattr(geom_line, "l"):
        line = jnp.asarray(geom_line.l)
    elif hasattr(geom, "l"):
        line = jnp.asarray(getattr(geom, "l"))
    if line is None or line.size < 5:
        return _sheath_simple(params, geom, y, phi)
    nz = int(getattr(grid, "nz", int(line.size)))

    mask, sign = geom.sheath_mask_sign()
    nu_m, nu_p, nu_e = _sheath_nu(params, geom)

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
    vi_adj_r = (
        2.0 * vi_bc_r + 4.0 * y.vpar_i[-3] - y.vpar_i[-4] - dl2 * v2_target_r
    ) / 5.0

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
            -nu_e * mask_l * (y.Te[left] - Te_bc_l)
            - nu_e * params.sheath_gamma_e * mask_l * y.Te[left]
        )
        dTe = dTe.at[right].add(
            -nu_e * mask_r * (y.Te[right] - Te_bc_r)
            - nu_e * params.sheath_gamma_e * mask_r * y.Te[right]
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
