from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_setup import OpenFieldSpecies
from jax_drb.native.recycling_state import PreparedSpeciesState
from jax_drb.native.safe_math import sqrt_nonnegative


def _species(name: str, *, charge: float, atomic_mass: float) -> OpenFieldSpecies:
    field = np.ones((1, 3, 1), dtype=np.float64)
    return OpenFieldSpecies(
        name=name,
        density=field,
        pressure=field,
        momentum=np.zeros_like(field),
        charge=charge,
        atomic_mass=atomic_mass,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=False,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=0.0,
    )


def test_sqrt_nonnegative_has_finite_zero_tangent_at_clip() -> None:
    value = jnp.asarray([-1.0, 0.0, 4.0], dtype=jnp.float64)
    tangent = jnp.zeros_like(value)

    primal, derivative = jax.jvp(sqrt_nonnegative, (value,), (tangent,))

    np.testing.assert_allclose(np.asarray(primal), np.asarray([0.0, 0.0, 2.0]))
    assert bool(jnp.all(jnp.isfinite(derivative)))
    np.testing.assert_allclose(np.asarray(derivative), np.zeros(3))


def test_sqrt_nonnegative_preserves_numpy_backend() -> None:
    value = np.asarray([-1.0, 0.0, 9.0], dtype=np.float64)

    result = sqrt_nonnegative(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.asarray([0.0, 0.0, 3.0]))


def test_collision_frequencies_have_finite_zero_tangent_at_zero_relative_temperature() -> None:
    config = load_bout_input(
        Path("tests/fixtures/reference-root/tests/integrated/1D-recycling-dthe/data/BOUT.inp")
    )
    species = {
        "e": _species("e", charge=-1.0, atomic_mass=1.0 / 1836.0),
        "d": _species("d", charge=0.0, atomic_mass=2.0),
        "d+": _species("d+", charge=1.0, atomic_mass=2.0),
    }
    density = jnp.ones((1, 3, 1), dtype=jnp.float64)
    zero_temperature = jnp.zeros_like(density)
    dataset_scalars = {
        "Nnorm": 1.0e19,
        "Tnorm": 100.0,
        "rho_s0": 1.0,
        "Omega_ci": 1.0,
    }

    def packed_rates(temperature):
        prepared = {
            name: PreparedSpeciesState(
                density=density,
                pressure=temperature * density,
                temperature=temperature,
                velocity=jnp.zeros_like(density),
                momentum=jnp.zeros_like(density),
                momentum_error=jnp.zeros_like(density),
            )
            for name in species
        }
        rates = compute_collision_frequencies(
            config,
            species,
            prepared,
            dataset_scalars=dataset_scalars,
        )
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(rates[key], dtype=jnp.float64)) for key in sorted(rates)]
        )

    primal, derivative = jax.jvp(
        packed_rates,
        (zero_temperature,),
        (jnp.zeros_like(zero_temperature),),
    )

    assert bool(jnp.all(jnp.isfinite(primal)))
    assert bool(jnp.all(jnp.isfinite(derivative)))
