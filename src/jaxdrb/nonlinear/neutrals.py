from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class NeutralParams(eqx.Module):
    """Minimal neutral interaction model (toggable).

    This is a first physically-motivated step inspired by fluid plasma–neutral
    models used in SOL turbulence codes: neutrals are advected by E×B and
    diffuse, while ionization transfers particles from neutrals to the plasma.

    The goal is to provide a clean hook for more realistic models:
      - energy exchange (ionization/radiation),
      - charge exchange momentum sinks,
      - recycling sources tied to sheath fluxes and geometry.
    """

    enabled: bool = False
    Dn0: float = 0.0  # neutral diffusion

    # In HW2D, `n` is typically interpreted as a fluctuation about a background.
    # Ionization and recombination rates should depend on the *absolute* density.
    n_background: float = 1.0
    n_floor: float = 1e-6
    N_floor: float = 1e-6

    # Ionization / recombination rates.
    #
    # Minimal particle exchange (conserves ∫(n+N) if Sn=0 and D=0):
    #   S_ion = nu_ion * n * N
    #   dn += +S_ion
    #   dN += -S_ion
    nu_ion: float = 0.0
    nu_rec: float = 0.0  # optional recombination: dn -= nu_rec*n, dN += nu_rec*n

    # Optional uniform source/sink terms.
    S0: float = 0.0
    nu_sink: float = 0.0


def rhs_neutral(
    *,
    N: jnp.ndarray,
    n: jnp.ndarray,
    dn0: NeutralParams,
    adv_N: jnp.ndarray,
    lap_N: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (dN/dt, dn/dt contribution) from neutral physics."""

    if not dn0.enabled:
        z = jnp.zeros_like(N)
        return z, z

    diff = dn0.Dn0 * lap_N
    src = dn0.S0 - dn0.nu_sink * N

    n_abs = jnp.maximum(float(dn0.n_background) + n, float(dn0.n_floor))
    N_abs = jnp.maximum(N, float(dn0.N_floor))

    ion = dn0.nu_ion * n_abs * N_abs
    rec = dn0.nu_rec * n_abs

    dN = -adv_N + diff + src - ion + rec
    dn_contrib = ion - rec
    return dN, dn_contrib
