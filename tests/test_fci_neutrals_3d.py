"""Neutrals coupled to the 3D FCI geometries -- closed and open field lines.

The hydrogenic atomic reactions couple a plasma fluid to a neutral fluid on any
field shape, so they act directly on the 3D FCI fields; on open field lines the
target sheath recycles the ion flux into the neutral source. This gate checks
that the coupling is coherent on both a genuinely non-axisymmetric closed
geometry (the rotating ellipse) and an open one (the SOL slab):

- on the closed rotating ellipse the reaction particle and momentum sources
  cancel between the ion and neutral fluids cell-by-cell and integrate to zero
  over the non-axisymmetric volume (nothing is created or destroyed by the
  coupling); and
- on the open slab the FCI Bohm-sheath recycling closure turns the target ion
  flux into a neutral source that exactly matches the recycled particle
  accounting, and the volumetric reactions conserve particles on top of it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import build_open_slab_geometry, build_rotating_ellipse_geometry
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling
from drbx.native.neutrals import (
    PlasmaNormalization,
    compute_hydrogen_reaction_sources,
)

jax.config.update("jax_enable_x64", True)

NORM = PlasmaNormalization()


def _plasma_neutral_fields(shape, seed=0):
    rng = np.random.default_rng(seed)
    ion_density = jnp.asarray(1.0 + 0.3 * rng.random(shape))
    ion_velocity = jnp.asarray(0.2 * (rng.random(shape) - 0.5))
    ion_temperature = jnp.asarray(0.05 + 0.25 * rng.random(shape))  # ~2.5 - 15 eV
    neutral_density = jnp.asarray(0.2 + 0.5 * rng.random(shape))
    neutral_velocity = jnp.asarray(0.05 * (rng.random(shape) - 0.5))
    neutral_temperature = jnp.full(shape, 0.03)
    return ion_density, ion_velocity, ion_temperature, neutral_density, neutral_velocity, neutral_temperature


def test_reactions_conserve_on_closed_rotating_ellipse() -> None:
    geometry = build_rotating_ellipse_geometry((8, 10, 6), elongation=0.35, n_field_periods=1)
    shape = geometry.shape
    ni, vi, ti, nn, vn, tn = _plasma_neutral_fields(shape)
    sources = compute_hydrogen_reaction_sources(ni, vi, ti, ti, nn, vn, tn, normalization=NORM)

    # Cell-by-cell conservation on the genuinely non-axisymmetric grid.
    assert np.allclose(np.asarray(sources.ion_density + sources.neutral_density), 0.0, atol=1e-30)
    assert np.allclose(np.asarray(sources.ion_momentum + sources.neutral_momentum), 0.0, atol=1e-30)

    # Integrated over the non-axisymmetric volume (metric Jacobian weight) the net
    # particle and momentum sources vanish -- the coupling moves particles between
    # the fluids, it does not create them.
    jacobian = np.asarray(geometry.cell_metric.J)
    net_particles = float(np.sum((np.asarray(sources.ion_density) + np.asarray(sources.neutral_density)) * jacobian))
    net_momentum = float(np.sum((np.asarray(sources.ion_momentum) + np.asarray(sources.neutral_momentum)) * jacobian))
    assert abs(net_particles) < 1e-20
    assert abs(net_momentum) < 1e-20

    # The reactions genuinely act (non-trivial ionization) and vary across the
    # rotating geometry.
    ionization = np.asarray(sources.ionization_rate)
    assert float(ionization.max()) > 0.0
    assert float(ionization.std()) > 0.0


def test_neutrals_and_sheath_recycling_close_on_open_slab() -> None:
    geometry = build_open_slab_geometry((4, 4, 20), parallel_length=30.0)
    shape = geometry.shape
    ni, vi, ti, nn, vn, tn = _plasma_neutral_fields(shape, seed=1)
    electron_temperature = ti

    # Volumetric reactions still conserve particles between the fluids.
    sources = compute_hydrogen_reaction_sources(ni, vi, ti, electron_temperature, nn, vn, tn, normalization=NORM)
    assert np.allclose(np.asarray(sources.ion_density + sources.neutral_density), 0.0, atol=1e-30)

    # Open field lines: the Bohm sheath recycles the target ion flux into a
    # neutral source. The recycled source matches the recycling fraction times the
    # target ion loss, and the accounting identities close to machine precision.
    recycling_fraction = 0.95
    sheath = compute_fci_sheath_recycling(
        ni, electron_temperature, ti, geometry.maps, recycling_fraction=recycling_fraction
    )
    assert float(sheath.total_recycled_particle_source) > 0.0
    assert float(sheath.total_recycled_particle_source) == \
        __import__("pytest").approx(recycling_fraction * float(sheath.total_ion_particle_loss))
    assert abs(float(sheath.particle_recycling_residual)) < 1e-12
    assert abs(float(sheath.current_balance_residual)) < 1e-12
    # The recycled neutral source lands on the target planes (the open endpoints).
    recycled = np.asarray(sheath.recycled_particle_source)
    assert float(recycled[:, :, -1].sum() + recycled[:, :, 0].sum()) > 0.0
    assert float(recycled[:, :, 1:-1].sum()) == 0.0
