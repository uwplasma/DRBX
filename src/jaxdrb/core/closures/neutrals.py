from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class NeutralParams(eqx.Module):
    """Minimal neutral interaction model (toggable).

    Inspired by fluid plasma–neutral models used in SOL turbulence codes.
    Neutral density is advected by ExB and diffused, while ionization/recombination
    transfer particles between neutrals and plasma.
    """

    enabled: bool = False
    Dn0: float = 0.0  # neutral diffusion

    # Background density for HW-like perturbation models.
    n_background: float = 1.0
    n_floor: float = 1e-6
    N_floor: float = 1e-6

    # Ionization / recombination rates.
    nu_ion: float = 0.0
    nu_rec: float = 0.0

    # Optional uniform source/sink terms.
    S0: float = 0.0
    nu_sink: float = 0.0

    # Optional charge-exchange-like momentum drag proxy in vorticity equation.
    nu_cx_omega: float = 0.0


def rhs_neutral(
    *,
    N: jnp.ndarray,
    n: jnp.ndarray,
    omega: jnp.ndarray,
    dn0: NeutralParams,
    adv_N: jnp.ndarray,
    lap_N: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (dN/dt, dn/dt contribution, domega/dt contribution) from neutral physics."""

    if not dn0.enabled:
        z = jnp.zeros_like(N)
        return z, z, z

    diff = dn0.Dn0 * lap_N
    src = dn0.S0 - dn0.nu_sink * N

    n_abs = jnp.maximum(float(dn0.n_background) + n, float(dn0.n_floor))
    N_abs = jnp.maximum(N, float(dn0.N_floor))

    ion = dn0.nu_ion * n_abs * N_abs
    rec = dn0.nu_rec * n_abs

    dN = -adv_N + diff + src - ion + rec
    dn_contrib = ion - rec
    domega_contrib = -dn0.nu_cx_omega * N_abs * omega
    return dN, dn_contrib, domega_contrib
