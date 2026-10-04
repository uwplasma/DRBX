"""Energy ledgers for the plasma-neutral reaction sources and 1D conduction.

These are analytical identities, independent of the implementation:

- reaction sources conserve thermal plus bulk-kinetic energy,
  ``sum_s [3/2 S_p,s + u_s S_M,s - m u_s^2 S_n,s / 2] = 0`` (the compact FCI
  closure evolves pressures, the hermes-style closure internal energies);
- isolated charge exchange of a drifting ion fluid through stationary neutrals
  (the closed-cell counterexample): kinetic energy lost equals heat gained;
- detachment-model conduction with a uniform temperature carries no heat at a
  nonuniform density, and conserves ``sum(3 n T)``;
- the CX velocity-damping rate of ions on stationary neutrals is independent of
  the ion density.
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


def test_detachment_conduction_uniform_temperature_null() -> None:
    nz = 64
    z = (jnp.arange(nz) + 0.5) / nz
    density = 1.0 + 0.8 * jnp.sin(3.0 * z) ** 2
    temperature = jnp.full(nz, 0.7)
    kappa = 2.0 * 0.7**2.5 * jnp.ones(nz - 1)
    out = dsm._implicit_conduction(density, temperature, kappa, 0.05)
    np.testing.assert_allclose(np.asarray(out), 0.7, rtol=1e-13)


def test_detachment_conduction_conserves_thermal_energy() -> None:
    nz = 64
    z = (jnp.arange(nz) + 0.5) / nz
    density = 1.0 + 0.8 * jnp.sin(3.0 * z) ** 2
    temperature = 0.2 + jnp.exp(-((z - 0.3) ** 2) / 0.01)
    face = 0.5 * (temperature[:-1] + temperature[1:])
    out = dsm._implicit_conduction(density, temperature, 2.0 * face**2.5, 0.01)
    np.testing.assert_allclose(float(jnp.sum(3 * density * out)), float(jnp.sum(3 * density * temperature)), rtol=1e-13)
    assert float(jnp.max(out)) < float(jnp.max(temperature))


def test_detachment_cx_damping_independent_of_ion_density() -> None:
    nz, dt, nn, temperature = 16, 1.0e-7, 0.5, 0.1
    params = dsm.DetachmentSolParameters(upstream_power=0.0, recycling_fraction=0.0, neutral_diffusion=0.0)
    norm = params.normalization
    rate_scale = norm.Nnorm * (params.parallel_length / norm.sound_speed)
    t_eff = jnp.clip(2.0 * temperature * norm.Tnorm / params.ion_mass, 0.01, 1.0e4)
    expected_cx = float(dsm.charge_exchange_rate_coefficient(t_eff)) * nn * rate_scale
    for n_i in (0.5, 2.0):
        density = jnp.full(nz, n_i)
        momentum = params.ion_mass * density * 0.3
        state = dsm.DetachmentSolState(density, momentum, 2.0 * density * temperature, jnp.full(nz, nn))
        new = dsm.detachment_sol_step(state, params, dt)
        mid = nz // 2
        damping = (float(momentum[mid] / new.ion_momentum[mid]) - 1.0) / dt
        rec = float(dsm.rate_coefficient("d", "rec", temperature * norm.Tnorm, n_i * norm.Nnorm)) * rate_scale * n_i
        np.testing.assert_allclose(damping - rec, expected_cx, rtol=2e-3)


def _isolated_target_state(nz, *, temperature, mach):
    params = dsm.DetachmentSolParameters(
        upstream_density=1.0, upstream_power=0.0, conduction_coefficient=0.0, neutral_diffusion=0.0,
    )
    density = jnp.ones(nz)
    pressure = 2.0 * density * temperature
    cs = float(jnp.sqrt(pressure[0] / (params.ion_mass * density[0])))
    momentum = params.ion_mass * density * mach * cs
    return params, dsm.DetachmentSolState(density, momentum, pressure, jnp.zeros(nz)), cs


def test_detachment_target_heat_loss_counts_enthalpy_and_kinetic_once() -> None:
    # Uniform supersonic outflow, no sources: interior fluxes cancel, so the
    # thermal-energy loss over one small step is the advected enthalpy
    # 5 n T v_t plus the sheath excess (gamma - 6) n T v_t. With the ion
    # kinetic power M^2 n T v_t carried by the momentum flux the total is
    # (gamma - 1 + M^2) n T v_t, i.e. gamma n T c_s at M = 1 (Stangeby 2.95, 14.5).
    nz, dt, temperature, mach = 32, 1.0e-6, 1.0, 1.3
    params, state, cs = _isolated_target_state(nz, temperature=temperature, mach=mach)
    new = dsm.detachment_sol_step(state, params, dt)
    dz = 1.0 / nz
    loss = -float(jnp.sum(1.5 * (new.plasma_pressure - state.plasma_pressure)) * dz / dt)
    expected = (params.sheath_transmission - 1.0) * 1.0 * temperature * mach * cs
    np.testing.assert_allclose(loss, expected, rtol=1e-3)

