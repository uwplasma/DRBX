from __future__ import annotations

import jax.numpy as jnp


def _loss_rate_from_Lpar(Lpar: jnp.ndarray, *, nu_factor: float) -> jnp.ndarray:
    # Common reduced SOL estimate: nu_sh ~ 2 c_s / Lpar. In our normalization c_s~O(1),
    # so we use nu_sh ~ 2/Lpar with a multiplier.
    return float(nu_factor) * (2.0 / (Lpar + 1e-30))


def sheath_bc_rate(params, geom) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Return (nu_bc, mask) for MPSE boundary enforcement, or None if unavailable."""
    on = bool(getattr(params, "sheath_bc_on", False))
    if not on:
        return None
    if not hasattr(geom, "sheath_mask"):
        return None
    mask = getattr(geom, "sheath_mask", None)
    if mask is None:
        return None
    Lpar = jnp.abs(jnp.asarray(geom.l[-1] - geom.l[0], dtype=jnp.float64))
    nu = _loss_rate_from_Lpar(Lpar, nu_factor=float(getattr(params, "sheath_bc_nu_factor", 1.0)))
    return nu, mask


def sheath_loss_rate(params, geom) -> jnp.ndarray:
    """Return nu_sh for the optional volumetric sheath-loss proxy (or 0 if disabled)."""
    on = bool(getattr(params, "sheath_loss_on", False) or getattr(params, "sheath_on", False))
    if not on:
        return jnp.asarray(0.0, dtype=jnp.float64)

    nu_factor = float(getattr(params, "sheath_loss_nu_factor", 1.0))
    if bool(getattr(params, "sheath_on", False)):
        nu_factor = float(getattr(params, "sheath_nu_factor", nu_factor))

    Lpar = jnp.abs(jnp.asarray(geom.l[-1] - geom.l[0], dtype=jnp.float64))
    return _loss_rate_from_Lpar(Lpar, nu_factor=nu_factor)


def sheath_lambda_effective(params) -> jnp.ndarray:
    """Effective sheath parameter Λ including optional secondary electron emission (SEE)."""

    lam = jnp.asarray(getattr(params, "sheath_lambda", 3.28), dtype=jnp.float64)
    if not bool(getattr(params, "sheath_see_on", False)):
        return lam
    delta = jnp.asarray(getattr(params, "sheath_see_yield", 0.0), dtype=jnp.float64)
    delta = jnp.clip(delta, 0.0, 0.999999)
    return lam + jnp.log1p(-delta)


def sheath_gamma_e(params) -> jnp.ndarray:
    """Electron heat transmission factor γ_e."""

    if bool(getattr(params, "sheath_gamma_auto", True)):
        # Common fluid-sheath estimate: γ_e ≈ 2 + Λ_eff.
        return 2.0 + sheath_lambda_effective(params)
    return jnp.asarray(getattr(params, "sheath_gamma_e", 0.0), dtype=jnp.float64)


def sheath_energy_losses(
    *,
    params,
    geom,
    Te: jnp.ndarray,
    Ti: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Return (dTe_sheath, dTi_sheath) from sheath heat transmission closures."""

    if not bool(getattr(params, "sheath_heat_on", False)):
        return jnp.zeros_like(Te), None if Ti is None else jnp.zeros_like(Ti)
    bc = sheath_bc_rate(params, geom)
    if bc is None:
        return jnp.zeros_like(Te), None if Ti is None else jnp.zeros_like(Ti)
    nu, mask = bc
    mask = jnp.asarray(mask, dtype=jnp.float64)

    ge = sheath_gamma_e(params)
    dTe = -nu * mask * ge * Te

    if Ti is None:
        return dTe, None
    gi = jnp.asarray(getattr(params, "sheath_gamma_i", 3.5), dtype=jnp.float64)
    dTi = -nu * mask * gi * Ti
    return dTe, dTi


def apply_loizu_mpse_boundary_conditions(
    *,
    params,
    geom,
    eq,
    phi: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    Te: jnp.ndarray,
    Ti: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Loizu-type MPSE boundary conditions (linearized or nonlinear)."""
    on = bool(getattr(params, "sheath_bc_on", False))
    if not on:
        return jnp.zeros_like(vpar_e), jnp.zeros_like(vpar_i)
    if not (hasattr(geom, "sheath_mask") and hasattr(geom, "sheath_sign")):
        return jnp.zeros_like(vpar_e), jnp.zeros_like(vpar_i)

    mask = getattr(geom, "sheath_mask", None)
    sign = getattr(geom, "sheath_sign", None)
    if mask is None or sign is None:
        return jnp.zeros_like(vpar_e), jnp.zeros_like(vpar_i)

    Lpar = jnp.abs(jnp.asarray(geom.l[-1] - geom.l[0], dtype=jnp.float64))
    nu = _loss_rate_from_Lpar(Lpar, nu_factor=float(getattr(params, "sheath_bc_nu_factor", 1.0)))

    delta = float(getattr(params, "sheath_delta", 0.0))

    if bool(getattr(params, "sheath_bc_linearized", True)):
        tau_i = float(getattr(params, "tau_i", 0.0))
        cs0 = jnp.sqrt(jnp.asarray(eq.Te0, dtype=jnp.float64) * (1.0 + tau_i))
        dcs = (0.5 / jnp.maximum(cs0, 1e-12)) * (Te if Ti is None else (Te + Ti))
        vpar_i_bc = sign * (1.0 - delta) * dcs
        vpar_e_bc = sign * (dcs - phi)
    else:
        Te0 = jnp.asarray(eq.Te0, dtype=jnp.float64)
        Te_tot = Te0 + Te
        Te_floor = float(getattr(params, "sheath_Te_floor", 1e-6))
        Te_tot = jnp.where(jnp.real(Te_tot) > Te_floor, Te_tot, Te_floor + 0j)

        tau_i = float(getattr(params, "tau_i", 0.0))
        Ti0 = tau_i * Te0
        Ti_tot = Ti0 + (jnp.zeros_like(Te_tot) if Ti is None else Ti)

        cs0 = jnp.sqrt(Te0 + Ti0)
        cs = jnp.sqrt(Te_tot + Ti_tot)

        lam = sheath_lambda_effective(params)
        phi_float = lam * Te0
        exp_arg = lam - (phi_float + phi) / Te_tot
        exp_arg = jnp.clip(exp_arg, a_min=-80.0, a_max=80.0)

        v_i_abs = sign * (1.0 - delta) * cs
        v_e_abs = sign * cs * jnp.exp(exp_arg)
        v_i0 = sign * (1.0 - delta) * cs0
        v_e0 = sign * cs0

        vpar_i_bc = v_i_abs - v_i0
        vpar_e_bc = v_e_abs - v_e0

    dvpar_i = -nu * mask * (vpar_i - vpar_i_bc)
    dvpar_e = -nu * mask * (vpar_e - vpar_e_bc)
    return dvpar_e, dvpar_i


def _one_sided_d2_target(
    f0: jnp.ndarray,
    f2: jnp.ndarray,
    f3: jnp.ndarray,
    *,
    dl: float,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Return the f1 value that satisfies a 2nd-order one-sided d2/dl2 stencil at the boundary."""

    # (2 f0 - 5 f1 + 4 f2 - f3) / dl^2 = target
    return (2.0 * f0 + 4.0 * f2 - f3 - target * (dl**2)) / 5.0


def apply_loizu2012_mpse_full_linear_bc(
    *,
    params,
    geom,
    eq,
    kperp2: jnp.ndarray,
    phi: jnp.ndarray,
    n: jnp.ndarray,
    omega: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    Te: jnp.ndarray,
    dpar=None,
    d2par=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linearized Loizu (2012) MPSE full-set BCs for cold-ion models.

    Returns (dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc).
    """

    phi = jnp.asarray(phi)
    n = jnp.asarray(n)
    omega = jnp.asarray(omega)
    vpar_e = jnp.asarray(vpar_e)
    vpar_i = jnp.asarray(vpar_i)
    Te = jnp.asarray(Te)

    _ = (dpar, d2par)
    on = bool(getattr(params, "sheath_bc_on", False))
    if not on:
        z = jnp.zeros_like(n)
        return z, z, z, z, z

    bc = sheath_bc_rate(params, geom)
    if bc is None:
        z = jnp.zeros_like(n)
        return z, z, z, z, z
    nu, mask = bc
    mask = jnp.asarray(mask, dtype=jnp.float64)

    sign = getattr(geom, "sheath_sign", None)
    if sign is None:
        sign = jnp.zeros_like(mask)
    sign = jnp.asarray(sign, dtype=jnp.float64)

    Te0 = jnp.asarray(eq.Te0, dtype=jnp.float64)
    cs0 = jnp.sqrt(jnp.maximum(Te0, 1e-12))
    cos2 = float(getattr(params, "sheath_cos2", 1.0))
    delta = float(getattr(params, "sheath_delta", 0.0))

    k2 = jnp.maximum(
        jnp.asarray(kperp2, dtype=jnp.float64), float(getattr(params, "kperp2_min", 1e-6))
    )

    # Velocity BC targets (linearized Bohm/MPSE).
    vpar_i_target = sign * (1.0 - delta) * 0.5 * Te
    vpar_e_target = sign * (0.5 * Te - phi)

    dvpar_i_bc = -nu * mask * (vpar_i - vpar_i_target)
    dvpar_e_bc = -nu * mask * (vpar_e - vpar_e_target)

    # Density gradient constraint.
    n_target = n
    n_target = n_target.at[0].set(n[1] + (vpar_i[1] - vpar_i[0]) / cs0[0])
    n_target = n_target.at[-1].set(n[-2] + (vpar_i[-2] - vpar_i[-1]) / cs0[-1])
    dn_bc = -nu * mask * (n - n_target)

    # Potential-gradient constraint mapped to omega via polarization.
    phi_target = phi
    phi_target = phi_target.at[0].set(phi[1] + cs0[0] * (vpar_i[1] - vpar_i[0]))
    phi_target = phi_target.at[-1].set(phi[-2] + cs0[-1] * (vpar_i[-2] - vpar_i[-1]))
    omega_target = -k2 * phi_target
    domega_bc = -nu * mask * (omega - omega_target)

    # Te Neumann constraint.
    Te_target = Te
    Te_target = Te_target.at[0].set(Te[1])
    Te_target = Te_target.at[-1].set(Te[-2])
    dTe_bc = -nu * mask * (Te - Te_target)

    # Additional vpar_i adjacent-point constraint (Loizu2012 Eq. 24).
    nl = int(n.size)
    if nl >= 4:
        dl = float(getattr(geom, "dl", float(geom.l[1] - geom.l[0])))
        target_left = -omega[0] / (cos2 * cs0[0] + 1e-12)
        target_right = -omega[-1] / (cos2 * cs0[-1] + 1e-12)
        v1_target = _one_sided_d2_target(vpar_i[0], vpar_i[2], vpar_i[3], dl=dl, target=target_left)
        vNm1_target = _one_sided_d2_target(
            vpar_i[-1], vpar_i[-3], vpar_i[-4], dl=dl, target=target_right
        )
        adj = jnp.zeros_like(vpar_i)
        adj = adj.at[1].set(-nu * mask[0] * (vpar_i[1] - v1_target))
        adj = adj.at[-2].set(-nu * mask[-1] * (vpar_i[-2] - vNm1_target))
        dvpar_i_bc = dvpar_i_bc + adj

    return dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc


def apply_loizu2012_mpse_full_linear_bc_hot_ion(
    *,
    params,
    geom,
    eq,
    kperp2: jnp.ndarray,
    phi: jnp.ndarray,
    n: jnp.ndarray,
    omega: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    Te: jnp.ndarray,
    Ti: jnp.ndarray,
    dpar=None,
    d2par=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linearized Loizu (2012) MPSE full-set BCs for hot-ion models."""

    Ti = jnp.asarray(Ti)
    dn, domega, dvpar_e, dvpar_i, dTe = apply_loizu2012_mpse_full_linear_bc(
        params=params,
        geom=geom,
        eq=eq,
        kperp2=kperp2,
        phi=phi,
        n=n,
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
        dpar=dpar,
        d2par=d2par,
    )

    bc = sheath_bc_rate(params, geom)
    if bc is None:
        return dn, domega, dvpar_e, dvpar_i, dTe, jnp.zeros_like(Ti)
    nu, mask = bc
    mask = jnp.asarray(mask, dtype=jnp.float64)

    Ti_target = Ti
    Ti_target = Ti_target.at[0].set(Ti[1])
    Ti_target = Ti_target.at[-1].set(Ti[-2])
    dTi = -nu * mask * (Ti - Ti_target)

    return dn, domega, dvpar_e, dvpar_i, dTe, dTi
