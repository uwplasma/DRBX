from __future__ import annotations

import jax.numpy as jnp


def sheath_boundary_tendencies(
    n: jnp.ndarray,
    Te: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    *,
    dz: float,
    n_floor: float,
    Te_floor: float,
    particle_on: bool,
    momentum_on: bool,
    energy_on: bool,
    relax_coeff: float,
    electron_target_coeff: float,
    gamma_e: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Open-field sheath boundary tendencies for n, v_parallel, Te.

    Boundary planes are index 0 (low-z target) and -1 (high-z target).
    """

    inv_dz = 1.0 / max(float(dz), 1e-30)
    n_eff = jnp.maximum(n, float(n_floor))
    Te_eff = jnp.maximum(Te, float(Te_floor))
    cs = jnp.sqrt(Te_eff)

    dn = jnp.zeros_like(n)
    dvpar_e = jnp.zeros_like(vpar_e)
    dvpar_i = jnp.zeros_like(vpar_i)
    dTe = jnp.zeros_like(Te)

    if particle_on:
        gamma_low = n_eff[0] * cs[0]
        gamma_high = n_eff[-1] * cs[-1]
        dn = dn.at[0].add(-gamma_low * inv_dz)
        dn = dn.at[-1].add(-gamma_high * inv_dz)

    if momentum_on:
        cs_low = cs[0]
        cs_high = cs[-1]
        vi_t_low = -cs_low
        vi_t_high = cs_high
        ve_t_low = -float(electron_target_coeff) * cs_low
        ve_t_high = float(electron_target_coeff) * cs_high

        nu_low = float(relax_coeff) * cs_low * inv_dz
        nu_high = float(relax_coeff) * cs_high * inv_dz

        dvpar_i = dvpar_i.at[0].add(nu_low * (vi_t_low - vpar_i[0]))
        dvpar_i = dvpar_i.at[-1].add(nu_high * (vi_t_high - vpar_i[-1]))
        dvpar_e = dvpar_e.at[0].add(nu_low * (ve_t_low - vpar_e[0]))
        dvpar_e = dvpar_e.at[-1].add(nu_high * (ve_t_high - vpar_e[-1]))

    if energy_on:
        q_low = float(gamma_e) * n_eff[0] * Te_eff[0] * cs[0]
        q_high = float(gamma_e) * n_eff[-1] * Te_eff[-1] * cs[-1]
        dTe = dTe.at[0].add(-(q_low / jnp.maximum(n_eff[0], float(n_floor))) * inv_dz)
        dTe = dTe.at[-1].add(-(q_high / jnp.maximum(n_eff[-1], float(n_floor))) * inv_dz)

    return dn, dvpar_e, dvpar_i, dTe
