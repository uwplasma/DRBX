from __future__ import annotations

import numpy as np
import pytest

from jax_drb.native.array_backend import use_jax_backend
from jax_drb.native.recycling_setup import OpenFieldSpecies
from jax_drb.native.recycling_source_accumulation import (
    add_species_sources,
    apply_species_source_overrides,
    zero_species_sources,
)


def _species(field) -> dict[str, OpenFieldSpecies]:
    zero = field * 0.0
    return {
        "e": OpenFieldSpecies(
            name="e",
            density=field,
            pressure=field,
            momentum=zero,
            charge=-1.0,
            atomic_mass=1.0 / 1836.0,
            density_floor=1.0e-8,
            has_pressure=True,
            has_momentum=False,
            noflow_lower_y=False,
            noflow_upper_y=False,
            target_recycle=False,
            recycle_as=None,
            target_recycle_multiplier=0.0,
            target_recycle_energy=0.0,
            target_fast_recycle_fraction=0.0,
            target_fast_recycle_energy_factor=0.0,
        ),
        "d+": OpenFieldSpecies(
            name="d+",
            density=2.0 * field,
            pressure=3.0 * field,
            momentum=zero,
            charge=1.0,
            atomic_mass=2.0,
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
        ),
    }


def test_source_accumulator_adds_updates_and_ignores_unknown_species() -> None:
    field = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    accumulator = zero_species_sources(_species(field))

    add_species_sources(
        accumulator,
        {
            "d+": np.full_like(field, 2.5),
            "unknown": np.full_like(field, 9.0),
        },
    )
    add_species_sources(accumulator, {"d+": np.full_like(field, -0.75)})
    apply_species_source_overrides(accumulator, {"e": np.full_like(field, 4.0), "missing": np.full_like(field, 6.0)})

    np.testing.assert_allclose(accumulator["d+"], 1.75)
    np.testing.assert_allclose(accumulator["e"], 4.0)
    assert "unknown" not in accumulator


def test_source_accumulator_preserves_jax_backend_and_jvp() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    base = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float64)
    weights = jnp.asarray([[[0.2], [0.5], [1.1]]], dtype=jnp.float64)

    def qoi(scale):
        field = scale * jnp.asarray(base)
        accumulator = zero_species_sources(_species(field))
        add_species_sources(accumulator, {"d+": 2.0 * field, "e": -0.5 * field})
        add_species_sources(accumulator, {"d+": field * field})
        apply_species_source_overrides(accumulator, {"e": 3.0 * field})
        assert use_jax_backend(accumulator["d+"])
        assert use_jax_backend(accumulator["e"])
        return jnp.sum(accumulator["d+"] * weights) + 0.25 * jnp.sum(accumulator["e"] * weights)

    _, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    eps = 1.0e-5
    finite_difference = (qoi(1.0 + eps) - qoi(1.0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-6, atol=2.0e-8)
