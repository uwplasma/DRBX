"""Fluid neutral atoms on the FCI grid: evolved ``n_n``, parallel velocity ``u_n`` and ``T_n`` (stages N1-N2).

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
- Parallel momentum (N2, Horsten model 2): ``Gamma = n u b - D_p (I - b b) grad p`` (perpendicular pressure diffusion,
  parallel flow carried by ``M = n u``), ``d_t M + div(u Gamma - eta grad u) = -b.grad p + S_M`` with the limited viscosity
  ``eta`` (flux limit ``p_n``).  ``(n, M)`` are advanced together implicitly (acoustic coupling through ``p = n T``,
  so no sound-speed or diffusion CFL), then ``u`` by an implicit upwind-advection and viscosity solve in flux form.  The
  internal energy gains the parallel pressure work ``u b.grad p`` and the viscous heating ``eta |grad u|^2``.  (The work of
  the perpendicular pressure-diffusive drift is cancelled by its friction heating, so neither appears.)  Curvature of
  ``b`` (``S_curv``) is neglected.  Friction with the ions is the CX/ionisation momentum exchange of the reaction step
  (explicit, ``max_dt_nu`` is its stability number), so momentum, nuclei and total energy are exchanged exactly.
- Exchange with the plasma: ``fci_neutrals.reaction_source`` (AMJUEL rates, one source hook), with the evolved
  ``T_n`` in the charge-exchange temperature and the energy handed to the neutrals as thermal energy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fci_neutrals import (
    NeutralGrid,
    NeutralParameters,
    _cell_gradients,
    _diffusion_diagonal,
    _face_fluxes,
    _fine,
    _owner_sum,
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
    "neutral_reactions",
    "neutral_transport",
    "transport_coefficients",
]

FLUID_LEDGER_COLUMNS = (
    "ionisation",  # sum V I dt
    "recombination",
    "charge_exchange",
    "electron_energy_loss",  # -sum V S_Ee dt
    "wall_energy_loss",  # energy the wall takes from the neutrals (absorbed and reflected-with-loss atoms)
    "neutral_momentum",  # sum V S_Mn dt (exact exchange with the ions)
    "transport_particle_residual",  # sum V dn_n - (sources - wall absorption) dt over the transport step (round-off)
    "particle_source",  # sum V S dt of the volumetric neutral source (recycling)
    "max_relres",  # largest implicit-solve relative residual
    "max_dt_nu",  # reaction stability number (raise reaction_substeps if > ~0.5)
    "out_of_range_Te",
    "out_of_range_ne",
    "out_of_range_Tcx",
    "energy_source",  # sum V S_E dt carried by that source
    "wall_absorbed",  # nuclei absorbed by the wall
    "min_particle_limiter",  # min of the particle limiter factor (1 = unlimited)
    "min_heat_limiter",
    "velocity_relres",
)


def flux_limiter(gamma_c, gamma_lim, alpha=1.0, power=2.0):
    """``Gamma_fl / Gamma_c = [1 + (|Gamma_c| / (alpha Gamma_lim))^power]^(-1/power)`` (smooth, in (0, 1])."""

    return (1.0 + (jnp.abs(gamma_c) / jnp.maximum(alpha * gamma_lim, 1.0e-300)) ** power) ** (-1.0 / power)


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


def _upwind(field, fluxes, grid: NeutralGrid):
    """Face values of ``field`` upwinded on the (smoothly signed) ``fluxes``: (x, y, z)."""

    scale = sum(jnp.max(jnp.abs(x)) for x in fluxes)
    f = _fine(field, grid)
    out = []
    for ax, flux in enumerate(fluxes):
        s = flux / jnp.sqrt(flux**2 + (1.0e-8 * scale + 1.0e-30) ** 2)
        nxt = jnp.roll(f, -1, axis=ax)
        out.append(0.5 * (f + nxt) - 0.5 * s * (nxt - f))
    return out


def _par_grad(field, grid: NeutralGrid):
    """``b . grad`` of an owner field, on the fine cells."""

    return sum(b * g for b, g in zip(grid.b_cell, _cell_gradients(_fine(field, grid), grid)))


def _parallel_flux(cell_flux, grid: NeutralGrid):
    """Integrated face flux ``J b^i dA`` of a cell-centred parallel flux density (fine cells)."""

    avg = lambda v, ax: 0.5 * (v + jnp.roll(v, -1, axis=ax))
    return [c * avg(cell_flux, ax) for ax, c in enumerate((grid.pb_x, grid.pb_y, grid.pb_z))]


def neutral_transport(neutral_density, neutral_temperature, neutral_velocity, dp, kappa, eta, dt, grid: NeutralGrid,
                      params: NeutralParameters, volume_source=None, volume_energy=None, tn_floor=1.0e-12):
    """One implicit conservative transport step of ``(n_n, u_n, T_n)`` with the limited ``D_p``, ``kappa``, ``eta``.

    ``volume_source`` / ``volume_energy``: owner particle and thermal-energy sources per unit volume and time (atoms
    are born at rest relative to the grid: the parallel momentum source is zero).
    The x wall is a reflecting wall with absorption: the Maxwellian wall flux ``n v/4`` (``v = sqrt(8 T/pi)``) is absorbed
    with probability ``params.wall_absorption``; the atoms that return keep the fraction ``wall_energy_reflection`` of
    their energy, so the wall takes ``(a + (1 - a)(1 - R_E)) 2 T`` per incident atom.  Both are implicit sinks.
    Returns ``(n_new, u_new, T_new, relres, wall)``: solver residuals ``(density, velocity, energy)`` and the wall's
    ``(absorbed nuclei, energy loss)`` of the step.
    """

    owner = grid.owner
    sn = jnp.zeros_like(neutral_density) if volume_source is None else jnp.where(owner, volume_source, 0.0)
    se = (1.5 * neutral_temperature * sn) if volume_energy is None else jnp.where(owner, volume_energy, 0.0)
    tl = jnp.maximum(neutral_temperature, tn_floor)
    m0 = neutral_density * neutral_velocity
    sel = lambda x: jnp.where(owner, x, 0.0)
    a, r_e = params.wall_absorption, params.wall_energy_reflection
    area = jnp.zeros(grid.shape).at[-1].set(grid.wall_area)
    geom = sel(_owner_sum(area, grid) / jnp.where(owner, grid.volume, 1.0))  # wall area per owner volume
    vbar = jnp.sqrt(8.0 * tl / jnp.pi)
    rate_n, rate_e = a * vbar / 4.0 * geom, (a + (1.0 - a) * (1.0 - r_e)) * vbar / 2.0 * geom  # 1/time; energy: x T n

    # (n, M): M_new = M - dt b.grad p_new; the parallel part of the limited isotropic diffusion is replaced by it, so the
    # flux is  -D g.grad p + b (D b.grad p + M_new) = perpendicular diffusion + the inertial parallel flow.
    def particle_flux(p):
        coefficient = (_fine(dp, grid) - dt) * _par_grad(p, grid)
        return [a + b for a, b in zip(_face_fluxes(_fine(p, grid), _fine(dp, grid), grid),
                                      _parallel_flux(coefficient, grid))]

    base = _parallel_flux(_fine(m0, grid), grid)
    rhs = neutral_density + dt * (sn + owner_inflow(*base, grid))
    p_star, rel_n = _gmres(lambda p: (1.0 + dt * rate_n) * p / tl - dt * owner_inflow(*particle_flux(p), grid), rhs,
                           rhs * tl, (1.0 + dt * rate_n) / tl + dt * _diffusion_diagonal(dp, grid), params)
    gp = _par_grad(p_star, grid)
    fx, fy, fz = [a + b for a, b in zip(particle_flux(p_star), base)]
    absorbed = rate_n * p_star / tl
    n_new = sel(neutral_density + dt * (owner_inflow(fx, fy, fz, grid) + sn - absorbed))
    m_a = sel(m0 - dt * gp)
    # a vacuum floor relative to the largest density keeps the implicit solves O(1) on empty and inactive cells
    n_safe = jnp.where(owner, jnp.maximum(n_new, 1.0e-12 * jnp.max(n_new)), 1.0)

    # u: implicit upwind advection by the particle flux and limited viscosity, conservative in M
    u_a = m_a / n_safe
    gu = sel(grad_norm(_fine(u_a, grid), grid))
    p_n = neutral_density * neutral_temperature
    eta_l = sel(eta * flux_limiter(eta * gu, p_n))

    def momentum_flux(u):
        up = _upwind(u, (fx, fy, fz), grid)
        return [f * x + v for f, x, v in zip(up, (fx, fy, fz), _face_fluxes(_fine(u, grid), _fine(eta_l, grid), grid))]

    def momentum_operator(u):
        return n_safe * u - dt * owner_inflow(*momentum_flux(u), grid)

    diag = n_safe + dt * _diffusion_diagonal(eta_l, grid)
    u_star, rel_u = _gmres(momentum_operator, m_a, u_a, diag, params)
    m_new = m_a + dt * owner_inflow(*momentum_flux(u_star), grid)
    u_new = sel(m_new / n_safe)

    # T: upwind enthalpy 5/2 T Gamma plus conduction, implicit in T; pressure work and viscous heating as sources
    kf = _fine(kappa, grid)
    se = se + sel(u_a * gp + eta_l * gu**2)

    def energy_flux(t):
        up = _upwind(t, (fx, fy, fz), grid)
        return [2.5 * f * x + c for f, x, c in zip(up, (fx, fy, fz), _face_fluxes(_fine(t, grid), kf, grid))]

    def energy_operator(t):
        return (1.5 + dt * rate_e) * n_safe * t - dt * owner_inflow(*energy_flux(t), grid)

    w_old = 1.5 * neutral_density * neutral_temperature
    rhs_e = w_old + dt * se
    diag = (1.5 + dt * rate_e) * n_safe + dt * _diffusion_diagonal(kappa, grid)
    t_star, rel_e = _gmres(energy_operator, rhs_e, sel(neutral_temperature), diag, params)
    wall_energy = rate_e * n_safe * t_star
    w_new = rhs_e + dt * (owner_inflow(*energy_flux(t_star), grid) - wall_energy)
    t_new = sel(w_new / (1.5 * n_safe))
    v = grid.volume
    return n_new, u_new, t_new, jnp.stack((rel_n, rel_u, rel_e)), dt * jnp.stack(
        (jnp.sum(v * absorbed), jnp.sum(v * wall_energy)))


def neutral_reactions(density, Vi, Te, Ti, neutral_density, neutral_velocity, neutral_temperature, dt, grid: NeutralGrid,
                      params: NeutralParameters):
    """Explicit reaction substeps in conserved variables: exact exchange of nuclei, momentum and energy.

    The plasma takes ``apply_volumetric_source``; the neutrals take ``(n_n, n_n u_n, 3/2 n_n T_n + n_n u_n^2/2)`` changed
    by the same reaction source (kinetic work ``u S_M - u^2 S_n/2``), so the only energy sink is the electron loss.
    Returns ``(n, Vi, Te, Ti, n_n, u_n, T_n, increments, max_dt_nu)``; ``increments``: ionisation, recombination,
    charge exchange, electron energy loss, neutral momentum (all ``sum V ... dt``).
    """

    owner, v = grid.owner, grid.volume
    h = dt / int(params.reaction_substeps)

    def body(_, carry):
        n, vi, te, ti, nn, un, tn, acc, max_nu = carry
        plasma, out = reaction_source(n, vi, te, ti, nn, params, owner, tn, un)
        n2, vi2, te2, ti2 = apply_volumetric_source(n, vi, te, ti, plasma, h, tau=params.tau, owner=owner)
        nn2 = jnp.where(owner, nn + h * out.neutral_density, 0.0)
        m2 = nn * un + h * out.neutral_momentum
        w2 = 1.5 * nn * tn + 0.5 * nn * un**2 + h * (out.neutral_energy + un * out.neutral_momentum
                                                      - 0.5 * un**2 * out.neutral_density)
        safe = jnp.where(owner, jnp.maximum(nn2, 1.0e-300), 1.0)
        un2 = jnp.where(owner, m2 / safe, un)
        tn2 = jnp.where(owner, (w2 - 0.5 * nn2 * un2**2) / (1.5 * safe), tn)
        inc = h * jnp.stack([jnp.sum(v * x) for x in (
            out.ionization_rate, out.recombination_rate, out.charge_exchange_rate, -out.electron_energy,
            out.neutral_momentum)])
        safe_n, safe_nn = jnp.where(owner & (n > 0), n, 1.0), jnp.where(owner & (nn > 0), nn, 1.0)
        nu = jnp.maximum((out.ionization_rate + out.charge_exchange_rate) / safe_nn,
                         (out.recombination_rate + out.charge_exchange_rate) / safe_n)
        return n2, vi2, te2, ti2, nn2, un2, tn2, acc + inc, jnp.maximum(max_nu, h * jnp.max(jnp.where(owner, nu, 0.0)))

    carry = (density, Vi, Te, Ti, neutral_density, neutral_velocity, neutral_temperature, jnp.zeros(5),
             jnp.zeros((), dtype=jnp.float64))
    return jax.lax.fori_loop(0, int(params.reaction_substeps), body, carry)


def neutral_fluid_step(density, Vi, Te, Ti, neutral_density, neutral_velocity, neutral_temperature, dt,
                       grid: NeutralGrid, params: NeutralParameters, volume_source=None, volume_energy=None):
    """Lie step: reactions (explicit, exact exchange), then limited fluid transport of ``(n_n, u_n, T_n)``.

    Returns ``(n, Vi, Te, Ti, n_n, u_n, T_n, ledger, ionisation_rate)`` with ``ledger`` ordered as
    ``FLUID_LEDGER_COLUMNS``.  ``volume_energy`` defaults to atoms born at the local ``T_n``.
    """

    owner, v = grid.owner, grid.volume
    n, vi, te, ti, nn, un, tn, acc, max_nu = neutral_reactions(
        density, Vi, Te, Ti, neutral_density, neutral_velocity, neutral_temperature, dt, grid, params)
    _, out = reaction_source(density, Vi, Te, Ti, neutral_density, params, owner, neutral_temperature, neutral_velocity)
    dp, kappa, fp, fh = limited_coefficients(n, te, ti, nn, tn, grid, params)
    eta = transport_coefficients(n, te, ti, nn, tn, params)[2]
    nn_new, un_new, tn_new, relres, wall = neutral_transport(
        nn, tn, un, dp, kappa, eta, dt, grid, params, volume_source, volume_energy)
    sn = jnp.where(owner, 0.0 if volume_source is None else volume_source, 0.0)
    s_n = dt * jnp.sum(v * sn)
    s_e = dt * jnp.sum(v * (1.5 * tn * sn if volume_energy is None else jnp.where(owner, volume_energy, 0.0)))
    residual = jnp.sum(v * (nn_new - nn)) - s_n + wall[0]
    ledger = jnp.concatenate((
        acc[:4], wall[1:], acc[4:], jnp.stack((residual, s_n, jnp.max(relres), max_nu)),
        jnp.stack(rate_validity_counts(n, te, ti, params, owner)),
        jnp.stack((s_e, wall[0], jnp.min(jnp.where(owner, fp, 1.0)), jnp.min(jnp.where(owner, fh, 1.0)), relres[1])),
    ))
    return n, vi, te, ti, nn_new, un_new, tn_new, ledger, out.ionization_rate
