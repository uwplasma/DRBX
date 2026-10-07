"""Energy ledgers for the plasma-neutral reaction sources and 1D conduction.

These are analytical identities, independent of the implementation:

- reaction sources conserve thermal plus bulk-kinetic energy,
  ``sum_s [3/2 S_p,s + u_s S_M,s - m u_s^2 S_n,s / 2] = 0`` (the compact FCI
  closure evolves pressures, the hermes-style closure internal energies);
- isolated charge exchange of a drifting ion fluid through stationary neutrals
  (the closed-cell counterexample): kinetic energy lost equals heat gained;
- SD1D-model conduction carries no heat at a uniform temperature and is
  conservative on the stretched, expanding grid;
- the SD1D CX velocity-damping rate of ions on stationary neutrals is
  independent of the ion density.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import build_synthetic_stellarator_geometry
from drbx.native.fci_neutral import compute_fci_neutral_reaction_diffusion
from drbx.native.neutrals import PlasmaNormalization, compute_hydrogen_reaction_sources
from drbx.native.neutrals import detachment_sol_model as dsm

jax.config.update("jax_enable_x64", True)


def _energy_rate(s_n, s_m, s_heat, u, mass):
    return s_heat + u * s_m - 0.5 * mass * u**2 * s_n


def _compact(ni, ui, ti, nn, un, tn, *, mi=1.0, mn=1.0, **rates):
    geometry = build_synthetic_stellarator_geometry(nx=6, ny=6, nz=6)
    shape = geometry.shape
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), shape)  # noqa: E731
    kwargs = dict(neutral_parallel_diffusivity=0.0, neutral_perp_diffusivity=0.0, neutral_mass=mn, ion_mass=mi)
    kwargs.update(rates)
    out = compute_fci_neutral_reaction_diffusion(
        neutral_density=full(nn), neutral_pressure=full(nn) * full(tn), neutral_momentum=mn * full(nn) * full(un),
        ion_density=full(ni), ion_pressure=full(ni) * full(ti), ion_momentum=mi * full(ni) * full(ui),
        electron_density=full(ni), electron_pressure=full(ni) * full(ti),
        maps=geometry.maps, metric=geometry.metric, **kwargs,
    )
    return out, full


def test_compact_charge_exchange_closed_cell_counterexample() -> None:
    out, _ = _compact(1.0, 1.0, 0.1, 1.0, 0.0, 0.1, ionisation_coefficient=0.0, recombination_coefficient=0.0)
    c = np.asarray(out.charge_exchange_rate)
    kinetic = 1.0 * np.asarray(out.ion_momentum_source)  # u_i S_Mi, u_n = 0
    heat = 1.5 * np.asarray(out.ion_pressure_source + out.neutral_pressure_source)
    np.testing.assert_allclose(kinetic, -c, rtol=1e-13)
    np.testing.assert_allclose(heat + kinetic, 0.0, atol=1e-13 * np.max(c))


def test_compact_reaction_sources_conserve_energy_random_states() -> None:
    rng = np.random.default_rng(3)
    for mi, mn in ((1.0, 1.0), (2.0, 2.0), (2.0, 1.0)):
        vals = [0.5 + rng.random(), rng.normal(), 0.05 + rng.random(), 0.2 + rng.random(), rng.normal(), 0.02 + rng.random()]
        out, _ = _compact(*vals, mi=mi, mn=mn)
        ui, un = vals[1], vals[4]
        total = _energy_rate(out.ion_density_source, out.ion_momentum_source, 1.5 * out.ion_pressure_source, ui, mi)
        neutral_reaction = out.neutral_density_source - out.neutral_diffusion_source
        neutral_heat = 1.5 * (out.neutral_pressure_source - out.neutral_pressure_diffusion_source)
        total = total + _energy_rate(neutral_reaction, out.neutral_momentum_source, neutral_heat, un, mn)
        scale = float(np.max(np.asarray(out.charge_exchange_rate + out.ionisation_rate + out.recombination_rate)))
        np.testing.assert_allclose(np.asarray(total), 0.0, atol=1e-12 * scale * (1.0 + ui**2 + un**2))


def test_hydrogen_reaction_sources_conserve_energy() -> None:
    rng = np.random.default_rng(7)
    shape = (32,)
    ni = jnp.asarray(0.5 + rng.random(shape))
    ui = jnp.asarray(rng.normal(size=shape))
    ti = jnp.asarray(0.05 + 0.3 * rng.random(shape))
    nn = jnp.asarray(0.2 + rng.random(shape))
    un = jnp.asarray(0.3 * rng.normal(size=shape))
    tn = jnp.asarray(0.01 + 0.05 * rng.random(shape))
    norm = PlasmaNormalization()
    s = compute_hydrogen_reaction_sources(ni, ui, ti, ti, nn, un, tn, normalization=norm)
    m = norm.ion_mass
    total = _energy_rate(s.ion_density, s.ion_momentum, s.ion_energy, ui, m) + _energy_rate(
        s.neutral_density, s.neutral_momentum, s.neutral_energy, un, m
    )
    scale = np.max(np.abs(np.asarray(s.ion_energy))) + np.max(np.abs(np.asarray(s.neutral_energy)))
    np.testing.assert_allclose(np.asarray(total), 0.0, atol=1e-12 * scale)


def _sd1d_grid(ny=64):
    params = dsm.DetachmentSolParameters(ny=ny)
    return params, dsm._grid(params)


def _ext(x):
    return jnp.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]])


def test_detachment_conduction_uniform_temperature_null() -> None:
    params, grid = _sd1d_grid()
    z = jnp.linspace(0.0, 1.0, params.ny)
    kappa = 2.0 + jnp.sin(3.0 * z) ** 2
    out = dsm._div_diffusion(_ext(kappa), _ext(jnp.full(params.ny, 0.7)), grid, upwind=True)
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-15)


def test_detachment_conduction_conserves_thermal_energy() -> None:
    # The area-weighted finite-volume conduction moves heat between cells without
    # creating any: sum(div * A * dl) = 0 for any profile, on the stretched,
    # expanding SD1D grid.
    params, grid = _sd1d_grid()
    z = jnp.linspace(0.0, 1.0, params.ny)
    temperature = 0.2 + jnp.exp(-((z - 0.3) ** 2) / 0.01)
    out = dsm._div_diffusion(_ext(temperature**2.5), _ext(temperature), grid, upwind=True)
    weights = grid.dl[2:-2] * grid.area[2:-2]
    total = float(jnp.sum(out * weights))
    assert abs(total) < 1e-13 * float(jnp.sum(jnp.abs(out) * weights))


def test_detachment_cx_damping_independent_of_ion_density() -> None:
    # SD1D friction F_cx = N Nn <sigma v>_cx (V - Vn): the momentum-damping rate
    # per ion, F_cx / (N V), depends on Nn and T only.
    params = dsm.DetachmentSolParameters(ny=16, recombination=False)
    grid = dsm._grid(params)
    nn, temperature, velocity = 0.5, 0.1, 0.3
    rates = []
    for n_i in (0.5, 2.0):
        u = jnp.stack([jnp.full(16, n_i), jnp.full(16, n_i * velocity), jnp.full(16, 2 * n_i * temperature),
                       jnp.full(16, nn), jnp.zeros(16), jnp.full(16, nn * temperature)], axis=1)
        terms = dsm._terms(u, 0.0, dsm._theta(params), params, grid)
        ionisation_part = 0.0  # Vn = 0, so ionisation carries no friction
        rates.append(float((terms.F[8] - ionisation_part) / (n_i * velocity)))
    expected = nn * params.Nnorm * float(dsm.sd1d_charge_exchange_rate(temperature * params.Tnorm)) / params.omega_ci
    np.testing.assert_allclose(rates, expected, rtol=1e-12)
