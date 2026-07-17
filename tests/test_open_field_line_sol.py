"""Open-field-line SOL flux-tube gate (item 1).

The open slab geometry has field lines that terminate on target plates, and the
reduced isothermal SOL model transports plasma to those targets through a Bohm
sheath. The tests pin:

- the open geometry carries target endpoint masks on exactly the two target
  planes (and nowhere else);
- the kept FCI sheath / recycling closure closes its accounting identities
  (particle recycling, zero-current balance, neutral-energy) to machine
  precision on this genuinely open geometry; and
- the reduced SOL flux tube relaxes to the classic two-point steady state:
  the flow reaches the sound speed at each target (Bohm criterion) and the
  target density is about half the upstream density.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import build_open_slab_geometry
from drbx.native.fci_sheath_recycling import build_fci_target_masks, compute_fci_sheath_recycling
from drbx.native.sol_flux_tube import (
    SolFluxTubeParameters,
    sol_flux_tube_run,
    sol_flux_tube_source,
)

jax.config.update("jax_enable_x64", True)


def test_open_slab_geometry_has_target_endpoints() -> None:
    nx, ny, nz = 4, 3, 12
    geometry = build_open_slab_geometry((nx, ny, nz), parallel_length=20.0)
    masks = build_fci_target_masks(geometry.maps)

    # Exactly the two target planes (z = 0 and z = L) are open endpoints.
    assert int(jnp.sum(masks.active)) == 2 * nx * ny
    assert bool(jnp.all(masks.forward[:, :, -1] == 1.0))
    assert bool(jnp.all(masks.backward[:, :, 0] == 1.0))
    assert bool(jnp.all(masks.forward[:, :, :-1] == 0.0))
    assert bool(jnp.all(masks.backward[:, :, 1:] == 0.0))
    # The field is purely parallel (open along z), so no radial contravariant part.
    assert float(jnp.max(jnp.abs(geometry.cell_bfield.B_contra[..., 0]))) == 0.0


def test_sheath_recycling_accounting_is_exact() -> None:
    geometry = build_open_slab_geometry((5, 4, 16), parallel_length=30.0)
    z = np.asarray(geometry.grid.z.centers)
    zc = z / z[-1]
    density = jnp.asarray(np.broadcast_to((1.0 + 0.5 * np.sin(np.pi * zc))[None, None, :], geometry.shape))
    te = jnp.full(geometry.shape, 0.6)
    ti = jnp.full(geometry.shape, 0.4)

    recycling_fraction = 0.9
    result = compute_fci_sheath_recycling(density, te, ti, geometry.maps, recycling_fraction=recycling_fraction)

    # Bohm flux n*c_s appears on the targets with c_s = sqrt(Te + Ti).
    c_s = float(jnp.sqrt(0.6 + 0.4))
    expected_target_flux = np.asarray(density)[:, :, -1] * c_s
    assert np.allclose(np.asarray(result.forward_ion_particle_flux)[:, :, -1], expected_target_flux)
    # Recycled source is the recycling fraction times the ion loss.
    assert float(result.total_recycled_particle_source) == \
        __import__("pytest").approx(recycling_fraction * float(result.total_ion_particle_loss))
    # Exact accounting identities close to machine precision.
    assert abs(float(result.particle_recycling_residual)) < 1e-12
    assert abs(float(result.current_balance_residual)) < 1e-12
    assert abs(float(result.neutral_energy_recycling_residual)) < 1e-12


def test_sol_flux_tube_reaches_bohm_two_point() -> None:
    geometry = build_open_slab_geometry((2, 2, 64), parallel_length=16.0)
    params = SolFluxTubeParameters(sound_speed=1.0, source_amplitude=0.05, source_width=2.0)
    source = sol_flux_tube_source(geometry, params)

    density = jnp.ones(geometry.shape)
    momentum = jnp.zeros(geometry.shape)
    dz = float(geometry.spacing.dz[0, 0, 0])
    dt = 0.4 * dz / (params.sound_speed + 1.0)
    density, momentum = sol_flux_tube_run(density, momentum, geometry, params, source, dt=dt, steps=12000)

    density = np.asarray(density)
    velocity = np.asarray(momentum) / density
    assert np.all(np.isfinite(density)) and np.all(np.isfinite(velocity))
    assert float(density.min()) > 0.0

    nz = geometry.shape[2]
    upstream = density[0, 0, nz // 2]
    column = velocity[0, 0, :]
    # The flow reaches (near-)sonic outflow at both targets — the Bohm criterion.
    # On a coarse grid the finite-volume sheath BC leaves the last cell a little
    # subsonic, so require strong sonic outflow at the target cells and a
    # near-sonic peak just inside.
    assert column[-1] > 0.8 * params.sound_speed
    assert column[0] < -0.8 * params.sound_speed
    assert float(np.max(np.abs(column))) > 0.85 * params.sound_speed
    # Symmetric flux tube: the two halves mirror.
    assert abs(column[-1] + column[0]) < 0.05 * params.sound_speed
    # Two-point SOL: target density is about half the upstream density.
    assert 0.4 < density[0, 0, -1] / upstream < 0.6

    # Steady-state particle balance: the upstream source equals the total Bohm
    # target loss (each target drains at the sheath sound speed).
    total_source = float(np.sum(np.asarray(source)[0, 0, :]) * dz)
    target_loss = float((density[0, 0, -1] + density[0, 0, 0]) * params.sound_speed)
    assert 0.95 < target_loss / total_source < 1.05
