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


def apply_loizu_mpse_boundary_conditions(
    *,
    params,
    geom,
    eq,
    phi: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    Te: jnp.ndarray,
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
        vpar_i_bc = sign * (1.0 - delta) * (0.5 * Te)
        vpar_e_bc = sign * (0.5 * Te - phi)
    else:
        # Nonlinear MPSE (kept for completeness; in linear studies prefer the linearized form).
        Te0 = jnp.asarray(eq.Te0, dtype=jnp.float64)
        Te_tot = Te0 + Te
        Te_floor = float(getattr(params, "sheath_Te_floor", 1e-6))
        Te_tot = jnp.where(jnp.real(Te_tot) > Te_floor, Te_tot, Te_floor + 0j)

        cs0 = jnp.sqrt(Te0)
        cs = jnp.sqrt(Te_tot)

        lam = float(getattr(params, "sheath_lambda", 3.28))
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
