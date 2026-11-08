"""Fluid neutral atoms on the FCI grid: evolved ``n_n`` and ``T_n`` (stage N1).

Advanced fluid neutral (AFN) closure of Horsten et al. (Nucl. Fusion 57, 2017) as used by
Van Uytven et al. (Nucl. Fusion 62, 086023, 2022), hydrogen, ``m_n = m_i`` (backend units, so ``m = 1``):

- ``tau_n^-1 = n_i K_cx + n_e K_i`` (``collision_frequencies``), pressure diffusion
  ``Gamma = -(tau_n/m) grad p_n``, viscosity ``eta = p_n tau_n`` and conductivity
  ``kappa = 5 p_n tau_n / (2 m)`` (``transport_coefficients``; ``eta`` is for the N2 momentum equation).
- Flux limiter on the conductive parts, ``Gamma_fl = Gamma_c [1 + (|Gamma_c| / (alpha Gamma_lim))^gamma]^(-1/gamma)``
  with ``alpha = 1, gamma = 2``; ``Gamma_lim = n_n sqrt(T_n/(2 pi m))`` (particles, ``(1/4) n_n sqrt(8 T_n/(pi m))``)
  and ``p_n sqrt(2 T_n/(pi m))`` (heat).  The limiter is a smooth cell factor that scales the coefficient.
- Conservative face-flux form on the full metric of ``fci_neutrals.NeutralGrid`` (nine-point cross terms kept):
  ``d_t n = div(D grad p)``; ``d_t (3/2 p) = -div(5/2 T Gamma - kappa grad T) + S_E`` (kinetic energy of the
  pressure-diffusive drift neglected; parallel momentum is N2).  ``n`` is advanced by backward Euler in ``p``
  (lagged ``T``, ``D``), ``T`` by backward Euler with the upwind enthalpy flux and conduction; both end with a
  flux-form update, so nuclei and the neutral energy change exactly by the fluxes and sources.
- Exchange with the plasma: ``fci_neutrals.reaction_source`` (AMJUEL rates, one source hook), with the evolved
  ``T_n`` in the charge-exchange temperature and the energy handed to the neutrals as thermal energy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fci_neutrals import (
    NeutralGrid,
    NeutralParameters,
    _diffusion_diagonal,
    _face_fluxes,
    _fine,
    apply_volumetric_source,
    collision_frequencies,
    grad_norm,
    owner_inflow,
    rate_validity_counts,
    reaction_source,
)

__all__ = [
    "FLUID_LEDGER_COLUMNS",
    "flux_limiter",
    "limited_coefficients",
    "neutral_fluid_step",
    "neutral_transport",
    "transport_coefficients",
]

FLUID_LEDGER_COLUMNS = (
    "ionisation",  # sum V I dt
    "recombination",
    "charge_exchange",
    "electron_energy_loss",  # -sum V S_Ee dt
    "neutral_momentum",  # sum V S_Mn dt (to the N2 momentum equation; dropped until then)
    "particle_source",  # sum V S dt of the volumetric neutral source (recycling)
    "energy_source",  # sum V S_E dt carried by that source
    "density_relres",
    "energy_relres",
    "min_particle_limiter",  # min of the particle limiter factor (1 = diffusive)
    "min_heat_limiter",
    "max_dt_nu",
    "out_of_range_Te",
    "out_of_range_ne",
    "out_of_range_Tcx",
)


def flux_limiter(gamma_c, gamma_lim, alpha=1.0, power=2.0):
    """``Gamma_fl / Gamma_c = [1 + (|Gamma_c| / (alpha Gamma_lim))^power]^(-1/power)`` (smooth, in (0, 1])."""

    return (1.0 + (jnp.abs(gamma_c) / (alpha * gamma_lim)) ** power) ** (-1.0 / power)


def transport_coefficients(density, Te, Ti, neutral_density, neutral_temperature, params: NeutralParameters):
    """``(D_p, kappa, eta)``: ``tau_n/m``, ``5 p_n tau_n/(2m)``, ``p_n tau_n`` with ``tau_n = 1/(nu_cx + nu_iz)``."""

    nu_cx, nu_iz = collision_frequencies(density, Te, Ti, params, neutral_temperature)
    tau = 1.0 / jnp.maximum(nu_cx + nu_iz, 1.0e-300)
    p = neutral_density * neutral_temperature
    return tau, 2.5 * p * tau, p * tau


def limited_coefficients(density, Te, Ti, neutral_density, neutral_temperature, grid: NeutralGrid,
                         params: NeutralParameters):
    """Limited ``(D_p, kappa)`` and the two limiter factors, on owner cells (fine-cell gradients)."""

    dp, kappa, _ = transport_coefficients(density, Te, Ti, neutral_density, neutral_temperature, params)
    p, tn = neutral_density * neutral_temperature, neutral_temperature
    fp = flux_limiter(dp * _own(grad_norm(_fine(p, grid), grid), grid),
                      neutral_density * jnp.sqrt(tn / (2.0 * jnp.pi)))
    fh = flux_limiter(kappa * _own(grad_norm(_fine(tn, grid), grid), grid), p * jnp.sqrt(2.0 * tn / jnp.pi))
    owner = grid.owner
    return jnp.where(owner, dp * fp, 0.0), jnp.where(owner, kappa * fh, 0.0), fp, fh


def _own(fine, grid: NeutralGrid):
    """Fine-cell field restricted to the owner cells (aliases of an aggregate share the owner's value)."""

    return jnp.where(grid.owner, fine, 0.0)


def _gmres(operator, rhs, x0, diag, params: NeutralParameters):
    solution, _ = jax.scipy.sparse.linalg.gmres(
        operator, rhs, x0=x0, tol=params.gmres_tolerance, atol=0.0, restart=params.gmres_restart,
        maxiter=params.gmres_maxiter, M=lambda v: v / diag, solve_method="batched",
    )
    relres = jnp.linalg.norm(operator(solution) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1.0e-300)
    return solution, relres


def neutral_transport(neutral_density, neutral_temperature, dp, kappa, dt, grid: NeutralGrid, params: NeutralParameters,
                      volume_source=None, volume_energy=None, tn_floor=1.0e-12):
    """One implicit conservative transport step of ``(n_n, T_n)`` with the (limited) ``D_p`` and ``kappa``.

    ``volume_source`` / ``volume_energy``: owner particle and thermal-energy sources per unit volume and time.
    Returns ``(n_new, T_new, density_relres, energy_relres)``.
    """

    owner = grid.owner
    sn = jnp.zeros_like(neutral_density) if volume_source is None else jnp.where(owner, volume_source, 0.0)
    se = (1.5 * neutral_temperature * sn) if volume_energy is None else jnp.where(owner, volume_energy, 0.0)
    tl = jnp.maximum(neutral_temperature, tn_floor)

    # particles: (p/T - dt L[p]) = n + dt S, then n_new = n + dt (L[p*] + S)
    def particle_operator(u):
        return u / tl - dt * owner_inflow(*_face_fluxes(_fine(u, grid), _fine(dp, grid), grid), grid)

    rhs = neutral_density + dt * sn
    p_star, rel_n = _gmres(particle_operator, rhs, rhs * tl, 1.0 / tl + dt * _diffusion_diagonal(dp, grid), params)
    fx, fy, fz = _face_fluxes(_fine(p_star, grid), _fine(dp, grid), grid)  # particle flux -D grad p
    n_new = jnp.where(owner, neutral_density + dt * (owner_inflow(fx, fy, fz, grid) + sn), 0.0)

    # energy: upwind enthalpy 5/2 T Gamma plus conduction, implicit in T
    scale = jnp.max(jnp.abs(fx)) + jnp.max(jnp.abs(fy)) + jnp.max(jnp.abs(fz))
    kf = _fine(kappa, grid)
    n_safe = jnp.maximum(n_new, 1.0e-300)

    def energy_flux(t):
        tf = _fine(t, grid)
        out = []
        for ax, flux, cond in zip((0, 1, 2), (fx, fy, fz), _face_fluxes(tf, kf, grid)):
            s = flux / jnp.sqrt(flux**2 + (1.0e-8 * scale + 1.0e-30) ** 2)
            tn_ = jnp.roll(tf, -1, axis=ax)
            out.append(2.5 * (0.5 * (tf + tn_) - 0.5 * s * (tn_ - tf)) * flux + cond)
        return out

    def energy_operator(t):
        return 1.5 * n_safe * t - dt * owner_inflow(*energy_flux(t), grid)

    w_old = 1.5 * neutral_density * neutral_temperature
    rhs_e = w_old + dt * se
    diag = 1.5 * n_safe + dt * _diffusion_diagonal(kappa, grid)
    t_star, rel_e = _gmres(energy_operator, rhs_e, neutral_temperature, diag, params)
    w_new = rhs_e + dt * owner_inflow(*energy_flux(t_star), grid)
    t_new = jnp.where(owner, w_new / (1.5 * n_safe), 0.0)
    return n_new, t_new, rel_n, rel_e


def neutral_fluid_step(density, Vi, Te, Ti, neutral_density, neutral_temperature, dt, grid: NeutralGrid,
                       params: NeutralParameters, volume_source=None, volume_energy=None):
    """Lie step: reactions (explicit, exact exchange), then limited fluid transport of ``(n_n, T_n)``.

    Returns ``(n, Vi, Te, Ti, n_n, T_n, ledger, ionisation_rate)`` with ``ledger`` ordered as ``FLUID_LEDGER_COLUMNS``.
    ``volume_energy`` defaults to atoms born at the local ``T_n`` (no energy added beyond ``3/2 T_n`` per atom).
    """

    owner, v = grid.owner, grid.volume
    h = dt / int(params.reaction_substeps)
    zero = jnp.zeros((), dtype=jnp.float64)

    def body(_, carry):
        n, vi, te, ti, nn, tn, acc, max_nu = carry
        plasma, out = reaction_source(n, vi, te, ti, nn, params, owner, tn)
        n2, vi2, te2, ti2 = apply_volumetric_source(n, vi, te, ti, plasma, h, tau=params.tau, owner=owner)
        nn2 = jnp.where(owner, nn + h * out.neutral_density, 0.0)
        w2 = 1.5 * nn * tn + h * out.neutral_energy
        tn2 = jnp.where(owner, w2 / (1.5 * jnp.maximum(nn2, 1.0e-300)), tn)
        inc = h * jnp.stack([jnp.sum(v * x) for x in (
            out.ionization_rate, out.recombination_rate, out.charge_exchange_rate, -out.electron_energy,
            out.neutral_momentum)])
        safe_n, safe_nn = jnp.where(owner & (n > 0), n, 1.0), jnp.where(owner & (nn > 0), nn, 1.0)
        nu = jnp.maximum((out.ionization_rate + out.charge_exchange_rate) / safe_nn,
                         (out.recombination_rate + out.charge_exchange_rate) / safe_n)
        return n2, vi2, te2, ti2, nn2, tn2, acc + inc, jnp.maximum(max_nu, h * jnp.max(jnp.where(owner, nu, 0.0)))

    n, vi, te, ti, nn, tn, acc, max_nu = jax.lax.fori_loop(
        0, int(params.reaction_substeps), body,
        (density, Vi, Te, Ti, neutral_density, neutral_temperature, jnp.zeros(5), zero),
    )
    _, out = reaction_source(density, Vi, Te, Ti, neutral_density, params, owner, neutral_temperature)
    dp, kappa, fp, fh = limited_coefficients(n, te, ti, nn, tn, grid, params)
    nn_new, tn_new, rel_n, rel_e = neutral_transport(nn, tn, dp, kappa, dt, grid, params, volume_source, volume_energy)
    s_n = zero if volume_source is None else dt * jnp.sum(v * jnp.where(owner, volume_source, 0.0))
    s_e = (1.5 * jnp.sum(v * tn * jnp.where(owner, 0.0 if volume_source is None else volume_source, 0.0)) * dt
           if volume_energy is None else dt * jnp.sum(v * jnp.where(owner, volume_energy, 0.0)))
    ledger = jnp.concatenate((
        acc,
        jnp.stack((s_n, s_e, rel_n, rel_e, jnp.min(jnp.where(owner, fp, 1.0)), jnp.min(jnp.where(owner, fh, 1.0)),
                   max_nu)),
        jnp.stack(rate_validity_counts(n, te, ti, params, owner)),
    ))
    return n, vi, te, ti, nn_new, tn_new, ledger, out.ionization_rate
