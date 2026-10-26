"""Spitzer-Harm parallel heat conduction: coefficients and the implicit step.

The thermal-energy rows ``W = 3/2 n T`` conduct ``q = -kappa G(T)`` with
``kappa_e = 3.16 n Te tau_e / m_e`` and ``kappa_i = 3.9 n Ti tau_i / m_i``.  Both
are independent of ``n`` (``tau`` is inversely proportional to it) and
proportional to ``T^(5/2)``, so one normalized prefactor per species fixes them.
The step is backward Euler with ``kappa`` frozen at the stage start,

    ``(3/2 n - dt D kappa G) dT = dt D(kappa G T)``,

in the flux form of the weighted support pair ``(G, D)``: ``sum V D(.) = 0``, so
the domain sum of ``W`` is unchanged and the wall legs stay the only loss.
"""

from __future__ import annotations

import math
from typing import Callable

import jax.numpy as jnp

ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
VACUUM_PERMITTIVITY = 8.8541878128e-12
B_REF = 1.0  # tesla
L_REF = 1.0  # metre
TEMPERATURE_FLOOR = 1.0e-6


def spitzer_prefactors(
    rho_star: float,
    mi_over_me: float,
    tau: float,
    reference_density_m3: float,
    coulomb_logarithm: float,
) -> tuple[float, float, float]:
    """Return ``(kappa0_e, kappa0_i, Te0_eV)``: ``kappa = kappa0 T^(5/2)``.

    The reference temperature follows from ``rho_star = sqrt(Te0 m_i)/(e B_ref
    L_ref)``; ``kappa0`` is the SI conductivity at ``T0`` divided by ``n0 c_s0
    L_ref``, so ``kappa0_e`` is the Spitzer ``kappa/n`` at ``T0`` in ``c_s0 L_ref`` (about 75 at
    24 eV, 1e19 m^-3, ln Lambda = 15).  The ion value uses ``Ti0 = tau Te0`` and
    ``tau_i/tau_e = sqrt(2 m_i/m_e)`` at equal temperature.
    """

    ion_mass = mi_over_me * ELECTRON_MASS
    te0 = (rho_star * ELEMENTARY_CHARGE * B_REF * L_REF) ** 2 / ion_mass
    cs0 = math.sqrt(te0 / ion_mass)
    kappa_e = (
        3.16 * 6.0 * math.sqrt(2.0) * math.pi**1.5 * VACUUM_PERMITTIVITY**2 * te0**2.5
        / (math.sqrt(ELECTRON_MASS) * ELEMENTARY_CHARGE**4 * coulomb_logarithm)
    )
    kappa0_e = kappa_e / (reference_density_m3 * cs0 * L_REF)
    kappa0_i = kappa0_e * (3.9 / 3.16) * math.sqrt(2.0 / mi_over_me) * tau**2.5
    return kappa0_e, kappa0_i, te0 / ELEMENTARY_CHARGE


def conductivity(
    temperature: jnp.ndarray,
    density: jnp.ndarray,
    kappa0: float,
    thermal_speed_squared: float,
    flux_limiter: float,
    gradient: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """``kappa0 T^(5/2)``, limited by ``q_fs = alpha n T v_th`` when ``alpha > 0``.

    ``thermal_speed_squared`` is ``v_th^2 / T`` in ``c_s0^2`` (``m_i/m_e`` for
    electrons, ``tau`` for ions).  The limiter is ``kappa / (1 + |q_SH|/q_fs)``
    with ``q_SH = kappa |G T|``, evaluated at the frozen state.
    """

    safe = jnp.maximum(temperature, TEMPERATURE_FLOOR)
    kappa = kappa0 * safe**2.5
    if flux_limiter > 0.0:
        free_streaming = flux_limiter * density * safe * jnp.sqrt(thermal_speed_squared * safe)
        kappa = kappa / (1.0 + kappa * jnp.abs(gradient(safe)) / free_streaming)
    return kappa


def backward_euler_conduction(
    flux_divergence: Callable[[jnp.ndarray], jnp.ndarray],
    capacity: jnp.ndarray,
    temperature: jnp.ndarray,
    dt: jnp.ndarray,
    solve: Callable[[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray], tuple],
) -> tuple[jnp.ndarray, object]:
    """Solve ``(capacity - dt L) dT = dt L(T)`` with ``L = D kappa G``; return ``(dT, info)``."""

    return solve(lambda x: capacity * x - dt * flux_divergence(x), dt * flux_divergence(temperature))
