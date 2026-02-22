from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config, run_simulation


def test_energy_conservation_advection_only():
    cfg = {
        "geometry": {
            "kind": "plane",
            "nx": 32,
            "ny": 32,
            "Lx": 8.0,
            "Ly": 8.0,
            "bc_x": "periodic",
            "bc_y": "periodic",
        },
        "physics": {
            "nonlinear_on": True,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
            "curvature_on": False,
            "omega_n": 0.0,
            "omega_Te": 0.0,
            "source_on": False,
            "sol_on": False,
        },
        "transport": {
            "Dn": 0.0,
            "DOmega": 0.0,
            "DTe": 0.0,
            "Dn4": 0.0,
            "DOmega4": 0.0,
            "DTe4": 0.0,
            "mu_lin_n": 0.0,
            "mu_lin_omega": 0.0,
            "mu_lin_Te": 0.0,
        },
        "numerics": {"poisson": "spectral", "bracket": "arakawa"},
        "terms": {"term_schedule": ["advection"]},
        "initial": {
            "amplitude": 2.0e-3,
            "seed": 0,
            "noise_mode": "state",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_scan",
            "dt": 2.0e-3,
            "nsteps": 300,
            "save_every": 10,
            "diag_mode": "basic",
            "poisson_warm_start": True,
            "return_numpy": True,
            "save_fields": True,
            "snapshot_fields": ["n", "omega", "Te"],
        },
    }

    result = run_simulation(cfg, as_numpy=True)
    payload = dict(result.diagnostics)
    snaps_n = jnp.asarray(payload["snapshots_n"])
    snaps_omega = jnp.asarray(payload["snapshots_omega"])
    snaps_Te = jnp.asarray(payload["snapshots_Te"])
    zeros = jnp.zeros_like(snaps_n)

    system = build_system_from_config(cfg).system

    def _energy(n, omega, Te):
        state = DRBSystemState(
            n=n,
            omega=omega,
            vpar_e=zeros[0],
            vpar_i=zeros[0],
            Te=Te,
            Ti=None,
            psi=None,
            N=None,
        )
        return system.energy(state)

    energy = jax.vmap(_energy)(snaps_n, snaps_omega, snaps_Te)
    energy = jnp.asarray(energy)
    rel_err = (energy - energy[0]) / energy[0]
    max_err = float(jnp.max(jnp.abs(rel_err)))
    assert max_err < 5.0e-3

    mass = jnp.mean(snaps_n, axis=(1, 2))
    mass_err = (mass - mass[0]) / (jnp.abs(mass[0]) + 1e-12)
    max_mass_err = float(jnp.max(jnp.abs(mass_err)))
    assert max_mass_err < 5.0e-3
