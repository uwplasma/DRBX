from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jaxdrb.hermes_literal import LiteralSpeciesState, compute_fastest_wave


def test_compute_fastest_wave_matches_collective_sound_speed_identity() -> None:
    ne = LiteralSpeciesState(
        density=jnp.full((3, 4, 5), 2.0, dtype=jnp.float64),
        pressure=jnp.full((3, 4, 5), 6.0, dtype=jnp.float64),
        temperature=jnp.full((3, 4, 5), 3.0, dtype=jnp.float64),
        AA=5.446623093681916e-4,
        charge=-1.0,
    )
    ion = LiteralSpeciesState(
        density=jnp.full((3, 4, 5), 2.0, dtype=jnp.float64),
        pressure=jnp.full((3, 4, 5), 8.0, dtype=jnp.float64),
        temperature=jnp.full((3, 4, 5), 4.0, dtype=jnp.float64),
        AA=2.0,
        charge=1.0,
    )
    result = compute_fastest_wave([ne, ion], electron_dynamics=True)
    total_pressure = 14.0
    total_density = 2.0 * 5.446623093681916e-4 + 2.0 * 2.0
    sound = np.sqrt(total_pressure / total_density)
    species_max = max(np.sqrt(3.0 / 5.446623093681916e-4), np.sqrt(4.0 / 2.0))
    expect = max(sound, species_max)
    np.testing.assert_allclose(np.asarray(result.sound_speed), sound)
    np.testing.assert_allclose(np.asarray(result.fastest_wave), expect)


def test_compute_fastest_wave_is_differentiable() -> None:
    def objective(t):
        ne = LiteralSpeciesState(
            density=jnp.ones((2, 2, 2), dtype=jnp.float64),
            pressure=t * jnp.ones((2, 2, 2), dtype=jnp.float64),
            temperature=t * jnp.ones((2, 2, 2), dtype=jnp.float64),
            AA=1.0,
            charge=-1.0,
        )
        ion = LiteralSpeciesState(
            density=jnp.ones((2, 2, 2), dtype=jnp.float64),
            pressure=2.0 * jnp.ones((2, 2, 2), dtype=jnp.float64),
            temperature=2.0 * jnp.ones((2, 2, 2), dtype=jnp.float64),
            AA=2.0,
            charge=1.0,
        )
        result = compute_fastest_wave([ne, ion], electron_dynamics=True)
        return jnp.sum(result.fastest_wave)

    grad = jax.grad(objective)(3.0)
    assert np.isfinite(np.asarray(grad)).all()
