"""Gate for the plasma<->neutral reaction-source assembly.

Pins the conservation laws the coupling must satisfy (particle and momentum
transfer between the ion and neutral fluids cancel), the Galilean invariance of
the charge-exchange frictional heating, and the physical ionization/recombination
balance (recombination wins at a cold target, ionization wins in a hot upstream).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.native.neutrals import (
    PlasmaNormalization,
    compute_hydrogen_reaction_sources,
)

jax.config.update("jax_enable_x64", True)

NORM = PlasmaNormalization()


def _sources(te_norm, *, ion_v=0.3, neutral_v=-0.1, ni=1.0, nn=0.8):
    shape = (5,)
    return compute_hydrogen_reaction_sources(
        ion_density=jnp.full(shape, ni),
        ion_velocity=jnp.full(shape, ion_v),
        ion_temperature=jnp.full(shape, te_norm),
        electron_temperature=jnp.full(shape, te_norm),
        neutral_density=jnp.full(shape, nn),
        neutral_velocity=jnp.full(shape, neutral_v),
        neutral_temperature=jnp.full(shape, 0.03),
        normalization=NORM,
    )


def test_particle_and_momentum_transfer_conserve() -> None:
    s = _sources(0.1)
    # Every ion created is a neutral destroyed and vice versa.
    assert np.allclose(np.asarray(s.ion_density + s.neutral_density), 0.0, atol=1e-30)
    # Momentum exchanged between the fluids cancels exactly.
    assert np.allclose(np.asarray(s.ion_momentum + s.neutral_momentum), 0.0, atol=1e-30)
    # Rates are non-negative.
    assert float(jnp.min(s.ionization_rate)) >= 0.0
    assert float(jnp.min(s.recombination_rate)) >= 0.0
    assert float(jnp.min(s.charge_exchange_rate)) >= 0.0


def test_charge_exchange_friction_is_galilean_invariant() -> None:
    # Shifting every velocity by a constant leaves the momentum-conservation
    # identity intact and the CX frictional heating unchanged (it depends only on
    # the ion-neutral velocity difference).
    base = _sources(0.1, ion_v=0.4, neutral_v=-0.2)
    shifted = _sources(0.1, ion_v=0.4 + 0.7, neutral_v=-0.2 + 0.7)
    assert np.allclose(np.asarray(shifted.ion_momentum + shifted.neutral_momentum), 0.0, atol=1e-30)
    # The CX friction contributes equally to both fluids; the shared piece
    # (neutral_energy has only the CX friction with equal sign) is invariant.
    cx_friction_base = base.ion_energy + base.neutral_energy  # thermal parts cancel -> 2 * friction
    cx_friction_shift = shifted.ion_energy + shifted.neutral_energy
    assert np.allclose(np.asarray(cx_friction_base), np.asarray(cx_friction_shift), rtol=1e-10)


def test_recombination_dominates_cold_ionization_dominates_hot() -> None:
    # Cold target (~1 eV): recombination wins, the ion fluid loses particles.
    cold = _sources(0.01)  # Te = 1 eV
    assert float(cold.ion_density[0]) < 0.0
    assert float(cold.recombination_rate[0]) > float(cold.ionization_rate[0])
    # Hot upstream (~30 eV): ionization wins, the ion fluid gains particles.
    hot = _sources(0.3)  # Te = 30 eV
    assert float(hot.ion_density[0]) > 0.0
    assert float(hot.ionization_rate[0]) > float(hot.recombination_rate[0])


def test_reaction_sources_are_jit_and_grad_transparent() -> None:
    def total_ionization(te):
        s = compute_hydrogen_reaction_sources(
            ion_density=jnp.ones(()),
            ion_velocity=jnp.zeros(()),
            ion_temperature=te,
            electron_temperature=te,
            neutral_density=jnp.ones(()),
            neutral_velocity=jnp.zeros(()),
            neutral_temperature=jnp.full((), 0.03),
            normalization=NORM,
        )
        return s.ionization_rate

    jitted = jax.jit(total_ionization)
    assert np.isfinite(float(jitted(jnp.asarray(0.1))))
    gradient = float(jax.grad(total_ionization)(jnp.asarray(0.1)))
    assert np.isfinite(gradient) and gradient > 0.0
