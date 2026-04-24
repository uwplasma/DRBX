from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np

from .array_backend import asarray, use_jax_backend, zeros_like
from .recycling_setup import OpenFieldSpecies


def zero_species_sources(species: Mapping[str, OpenFieldSpecies]) -> dict[str, np.ndarray]:
    """Return float source arrays on the dynamic-state backend.

    The recycling RHS accumulates several physics closures into density,
    pressure/energy, and momentum source dictionaries.  This helper keeps the
    dictionary-compatible public path while preventing the initial zeros from
    forcing JAX tracer inputs back through NumPy.
    """

    use_jax = use_jax_backend(*(sp.density for sp in species.values()))
    return {name: zeros_like(sp.density, use_jax=use_jax) for name, sp in species.items()}


def add_species_sources(
    accumulator: MutableMapping[str, np.ndarray],
    updates: Mapping[str, np.ndarray],
) -> None:
    """Add per-species source updates while preserving the active backend."""

    for name, value in updates.items():
        if name not in accumulator:
            continue
        use_jax = use_jax_backend(accumulator[name], value)
        accumulator[name] = asarray(accumulator[name], use_jax=use_jax) + asarray(value, use_jax=use_jax)


def apply_species_source_overrides(
    accumulator: MutableMapping[str, np.ndarray],
    overrides: Mapping[str, np.ndarray] | None,
) -> None:
    """Replace selected source entries without forcing JAX overrides to NumPy."""

    if not overrides:
        return
    for name, value in overrides.items():
        if name not in accumulator:
            continue
        accumulator[name] = asarray(value, use_jax=use_jax_backend(accumulator[name], value))
