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
    if bool(getattr(params, "sheath_on", False)):  # deprecated knob
        nu_factor = float(getattr(params, "sheath_nu_factor", nu_factor))

    Lpar = jnp.abs(jnp.asarray(geom.l[-1] - geom.l[0], dtype=jnp.float64))
    return _loss_rate_from_Lpar(Lpar, nu_factor=nu_factor)


def sheath_lambda_effective(params) -> jnp.ndarray:
    """Effective sheath parameter Λ including optional secondary electron emission (SEE).

    We use a simple ambipolar correction with a constant SEE yield δ:

      Λ_eff = Λ + ln(1 - δ)

    where 0 <= δ < 1. This reduces Λ_eff for δ>0, consistent with a reduced floating potential drop.

    Notes
    -----
    `jaxdrb` treats `phi` in sheath BCs as a floating-potential-shifted perturbation potential, so
    SEE mostly changes the equilibrium floating shift, not the linear response coefficient.
    """

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
    """Apply Loizu-type magnetic-pre-sheath entrance (MPSE) boundary conditions.

    We enforce the (simplified) Bohm-sheath entrance relations at the two ends of an open field line:

      v_||i = ± (1-δ) c_s
      v_||e = ± c_s exp(Λ - eφ/T_e)

    where:
      c_s = sqrt(T_e) (cold-ion limit),
      Λ ≈ 0.5 ln(mi/(2π me)),
      δ is an optional transmission correction (δ=0 for cold ions).

    Implementation details:
      - `jaxdrb` evolves perturbations about an equilibrium with Bohm-matched end flows. We therefore
        apply the BCs to perturbations by subtracting the equilibrium boundary values.
      - BCs are enforced weakly via a relaxation (penalty) term at the boundary points. The rate is
        scaled as nu_bc ~ 2/L_parallel with a user factor.
      - This requires an *open* geometry providing `sheath_mask` and `sheath_sign`.

    Notes:
      - We neglect additional terms from ExB drift along the wall and magnetic-incidence angle that
        appear in full Loizu derivations; those can be added later as geometry-dependent corrections.
    """
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
        # Linearized MPSE BCs for *perturbations* about a Bohm-matched equilibrium (Te0=1):
        #
        #   δv_||i = ± (1-δ) (δTe / 2)
        #   δv_||e = ± (δTe / 2 - δφ)
        #
        # where phi is the floating-potential-shifted perturbation potential.
        tau_i = float(getattr(params, "tau_i", 0.0))
        cs0 = jnp.sqrt(jnp.asarray(eq.Te0, dtype=jnp.float64) * (1.0 + tau_i))
        dcs = (0.5 / jnp.maximum(cs0, 1e-12)) * (Te if Ti is None else (Te + Ti))
        vpar_i_bc = sign * (1.0 - delta) * dcs
        vpar_e_bc = sign * (dcs - phi)
    else:
        # Nonlinear MPSE (kept for completeness; in linear studies prefer the linearized form).
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

    # Relaxation at boundary points only.
    dvpar_i = -nu * mask * (vpar_i - vpar_i_bc)
    dvpar_e = -nu * mask * (vpar_e - vpar_e_bc)
    return dvpar_e, dvpar_i


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
    Ti: jnp.ndarray | None = None,
    dpar,
    d2par,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply a linearized 'full set' of MPSE boundary conditions from Loizu et al. (2012).

    This implements a *model-aligned* subset of the boundary relations derived in:

      J. Loizu et al., Phys. Plasmas 19, 122307 (2012).

    Specifically, it enforces (in linearized form) boundary constraints for:

      - v_||i  (Bohm-Chodura / MPSE ion flow)
      - v_||e  (magnetized-electron response)
      - ∂_|| n and ∂_|| phi  (density/potential gradient relations)
      - omega  (vorticity boundary relation, linearized form of Eq. (24))
      - ∂_|| Te = 0 (isothermal-electron assumption at MPSE, Eq. (23))

    Notes / mapping to `jaxdrb`
    ---------------------------
    - Loizu uses coordinates (s,x) with incidence angle `a`. Here, we treat `l` as the parallel
      coordinate and approximate cos^2(a) by `params.sheath_cos2` (default 1).
    - Terms involving transverse gradients (∂_x) and ExB/diamagnetic corrections are omitted in
      this initial implementation to stay consistent with `jaxdrb`'s 1D field-line + Fourier-perp
      closure.
    - These constraints are enforced weakly via SAT/penalty relaxation terms at the boundary
      nodes, with rate nu ~ 2/L_parallel.
    """

    # Ensure JAX arrays (tests may pass NumPy arrays).
    kperp2 = jnp.asarray(kperp2)
    phi = jnp.asarray(phi)
    n = jnp.asarray(n)
    omega = jnp.asarray(omega)
    vpar_e = jnp.asarray(vpar_e)
    vpar_i = jnp.asarray(vpar_i)
    Te = jnp.asarray(Te)
    if Ti is not None:
        Ti = jnp.asarray(Ti)

    if not bool(getattr(params, "sheath_bc_on", False)):
        z = jnp.zeros_like(n)
        return z, z, z, z, z
    if not (hasattr(geom, "sheath_mask") and hasattr(geom, "sheath_sign")):
        z = jnp.zeros_like(n)
        return z, z, z, z, z

    mask = getattr(geom, "sheath_mask", None)
    sign = getattr(geom, "sheath_sign", None)
    if mask is None or sign is None:
        z = jnp.zeros_like(n)
        return z, z, z, z, z

    bc = sheath_bc_rate(params, geom)
    if bc is None:
        z = jnp.zeros_like(n)
        return z, z, z, z, z
    nu, _mask = bc
    # Trust geometry-provided mask.
    mask = jnp.asarray(mask, dtype=jnp.float64)
    sign = jnp.asarray(sign, dtype=jnp.float64)

    # Ion sound speed in our normalization (hot-ion extension uses cs^2 ~ Te0*(1+tau_i)).
    tau_i = float(getattr(params, "tau_i", 0.0))
    Te0 = jnp.asarray(eq.Te0, dtype=jnp.float64)
    cs0 = jnp.sqrt(Te0 * (1.0 + tau_i))

    delta = float(getattr(params, "sheath_delta", 0.0))
    cos2 = float(getattr(params, "sheath_cos2", 1.0))

    # In Loizu 2012, several MPSE relations are expressed as constraints on *parallel derivatives*.
    # To keep the enforcement stable with a generic open-grid dpar operator, we convert these into
    # equivalent constraints on boundary *values* using one-sided finite differences.

    # We implement the following MPSE relations (Loizu et al., Phys. Plasmas 19, 122307 (2012))
    # in a form consistent with jaxdrb's 1D field-line + Fourier-perp closure:
    #
    #   (17) v_||i = ± (1-δ) c_s   (Bohm/Chodura ion flow)
    #   (21) ∂_s φ = -c_s (1+...) ∂_s v_||i   (linear: ∂_s φ = -c_s0 ∂_s v_||i)
    #   (22) ∂_s n = -(n/c_s)(1+...) ∂_s v_||i (linear about n0=1: ∂_s n = -(1/c_s0) ∂_s v_||i)
    #   (23) ∂_s T_e = 0
    #   (24) ω = -cos^2(a)[ (1+...) (∂_s v)^2 + c_s (1+...) ∂_s^2 v ]
    #        (linear: ω = -cos^2(a) c_s0 ∂_s^2 v_||i)
    #
    # Since jaxdrb evolves perturbations about an equilibrium (n0=Te0=1 by default), we apply the
    # linearized forms, i.e. we drop products of perturbations.

    nl = int(n.size)
    if nl < 5:
        raise ValueError("Loizu2012 full MPSE BC requires nl>=5 for 2nd-order boundary stencils.")

    # One-sided (2nd order) second derivative stencil at endpoints:
    # Left:  f''(0) ≈ (2 f0 - 5 f1 + 4 f2 - f3) / dl^2
    # Right: f''(N) ≈ (2 fN - 5 f_{N-1} + 4 f_{N-2} - f_{N-3}) / dl^2
    dl = jnp.asarray(geom.dl, dtype=jnp.float64)
    dl2 = jnp.maximum(dl * dl, 1e-30)

    # Linearized velocity BCs for perturbations about Bohm-matched equilibrium:
    #   δv_i = ±(1-δ) δcs, with δcs = (δTe + δTi)/(2 cs0).
    dcs = (0.5 / jnp.maximum(cs0, 1e-12)) * (Te if Ti is None else (Te + Ti))
    vpar_i_target = sign * (1.0 - delta) * dcs

    # Te-gradient constraint at the ends: ∂_|| Te = 0 -> Te boundary equals neighbor (Neumann).
    Te_target = Te
    Te_target = Te_target.at[0].set(Te[1])
    Te_target = Te_target.at[-1].set(Te[-2])

    # Potential-gradient constraint (Loizu 2012 Eq. (21), linearized):
    #   ∂_s φ = -c_s0 ∂_s v_||i  ->  boundary φ value implied by neighbor and v_||i.
    #
    # Using a 1st-order one-sided derivative at the boundary:
    #   (φ_1 - φ_0)/dl = -c_s0 (v_1 - v_0)/dl
    # -> φ_0 = φ_1 + c_s0 (v_1 - v_0)
    # and similarly at the right end.
    phi_target = phi
    phi_target = phi_target.at[0].set(phi[1] + cs0[0] * (vpar_i[1] - vpar_i_target[0]))
    phi_target = phi_target.at[-1].set(phi[-2] + cs0[-1] * (vpar_i[-2] - vpar_i_target[-1]))

    # Density-gradient constraint (simplified Loizu 2012 Eq. (22), with n0=1):
    #   ∂_|| n + (1/c_s) ∂_|| v_||i = 0  ->  n boundary value from neighbor + vpar_i.
    invcs = 1.0 / jnp.maximum(cs0, 1e-12)
    n_target = n
    n_target = n_target.at[0].set(n[1] + invcs[0] * (vpar_i[1] - vpar_i_target[0]))
    n_target = n_target.at[-1].set(n[-2] + invcs[-1] * (vpar_i[-2] - vpar_i_target[-1]))

    # vpar_e target (linearized magnetized-electron response):
    # use the *target* boundary phi and (Te,Ti) through δcs.
    dcs_target = (0.5 / jnp.maximum(cs0, 1e-12)) * (Te_target if Ti is None else (Te_target + Ti))
    vpar_e_target = sign * (dcs_target - phi_target)

    # Convert phi_target -> omega_target via omega = -k_perp^2 phi at the boundary.
    k2_safe = jnp.maximum(
        jnp.asarray(kperp2, dtype=jnp.float64), float(getattr(params, "kperp2_min", 1e-6))
    )
    omega_target = omega
    omega_target = omega_target.at[0].set((-k2_safe[0]) * phi_target[0])
    omega_target = omega_target.at[-1].set((-k2_safe[-1]) * phi_target[-1])

    # Vorticity relation (Loizu 2012 Eq. (24), linearized) enforced by adjusting the
    # *adjacent* ion flow values (v_1 and v_{N-1}) rather than the boundary Bohm value.
    #
    # With ω = -k⊥² φ (Fourier-perp), Eq. (24) provides a target for ∂_s² v at the boundary:
    #   ω = -cos²(a) c_s0 ∂_s² v  ->  ∂_s² v_target = -ω_target / (cos² c_s0).
    #
    # Using the one-sided stencil with v0 fixed by Bohm:
    #   v''(0) ≈ (2 v0 - 5 v1 + 4 v2 - v3)/dl²  -> solve for v1_target.
    # and similarly at the right end solving for v_{N-1} (index -2).
    vort_coef = jnp.maximum(cos2 * cs0, 1e-12)
    v2_target_left = -omega_target[0] / vort_coef[0]
    v2_target_right = -omega_target[-1] / vort_coef[-1]

    v1_target_left = (
        2.0 * vpar_i_target[0] + 4.0 * vpar_i[2] - vpar_i[3] - dl2 * v2_target_left
    ) / 5.0
    v1_target_right = (
        2.0 * vpar_i_target[-1] + 4.0 * vpar_i[-3] - vpar_i[-4] - dl2 * v2_target_right
    ) / 5.0

    # Weak enforcement via relaxation on boundary *values* only.
    dvpar_i = -nu * mask * (vpar_i - vpar_i_target)
    dvpar_e = -nu * mask * (vpar_e - vpar_e_target)
    dn = -nu * mask * (n - n_target)
    dTe = -nu * mask * (Te - Te_target)
    domega = -nu * mask * (omega - omega_target)

    # Additional weak enforcement at the adjacent points for the vorticity relation.
    mask_adj = jnp.zeros((nl,), dtype=jnp.float64).at[1].set(1.0).at[-2].set(1.0)
    v_adj_target = jnp.zeros_like(vpar_i)
    v_adj_target = v_adj_target.at[1].set(v1_target_left)
    v_adj_target = v_adj_target.at[-2].set(v1_target_right)
    dvpar_i = dvpar_i - nu * mask_adj * (vpar_i - v_adj_target)

    return dn, domega, dvpar_e, dvpar_i, dTe


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
    dpar,
    d2par,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Hot-ion extension of the Loizu (2012) full-set MPSE BCs.

    This function wraps :func:`apply_loizu2012_mpse_full_linear_bc` and adds a simple, robust
    ion-temperature endpoint constraint:

      ∂_|| T_i = 0  (Neumann at the two MPSE nodes)

    This mirrors the common electron-temperature entrance constraint (Eq. (23) in Loizu 2012),
    and provides a reasonable baseline for hot-ion model development/benchmarking in open
    field-line geometries.
    """

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
        Ti=Ti,
        dpar=dpar,
        d2par=d2par,
    )

    Ti = jnp.asarray(Ti)
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
