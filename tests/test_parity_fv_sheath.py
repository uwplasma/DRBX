from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config


def _cfg(*, open_field_line: bool, sheath_on: bool) -> dict:
    return {
        "engine": "parity_fv",
        "geometry": {
            "kind": "slab",
            "nx": 10,
            "ny": 8,
            "nz": 12,
            "Lx": 1.0,
            "Ly": 1.0,
            "Lz": 1.2,
            "open_field_line": open_field_line,
        },
        "terms": {
            "parallel_on": False,
            "curvature_on": False,
            "sheath_on": sheath_on,
        },
        "closures": {
            "sheath": {
                "sheath_bc_on": sheath_on,
                "sheath_loss_on": True,
                "sheath_bohm_velocity_on": True,
                "sheath_energy_on": True,
                "sheath_gamma_e": 3.5,
                "sheath_current_closure_coeff": 1.0,
            }
        },
        "initial": {
            "n0": 1.0,
            "Te0": 1.0,
            "vpar_e0": 0.0,
            "vpar_i0": 0.0,
        },
    }


def test_sheath_term_zero_when_disabled_or_closed() -> None:
    y = None
    for open_field_line, sheath_on in ((False, True), (True, False), (False, False)):
        built = build_system_from_config(_cfg(open_field_line=open_field_line, sheath_on=sheath_on))
        y = built.state
        _, term_map, _, _ = built.system.rhs_terms(0.0, y)
        sheath = term_map["sheath"]
        assert np.allclose(np.asarray(sheath.n), 0.0, atol=1e-14)
        assert np.allclose(np.asarray(sheath.vpar_e), 0.0, atol=1e-14)
        assert np.allclose(np.asarray(sheath.vpar_i), 0.0, atol=1e-14)
        assert np.allclose(np.asarray(sheath.Te), 0.0, atol=1e-14)
    assert y is not None


def test_sheath_boundary_particle_momentum_energy_channels() -> None:
    built = build_system_from_config(_cfg(open_field_line=True, sheath_on=True))
    y = DRBSystemState(
        n=jnp.ones_like(built.state.n),
        omega=jnp.zeros_like(built.state.omega),
        vpar_e=jnp.zeros_like(built.state.vpar_e),
        vpar_i=jnp.zeros_like(built.state.vpar_i),
        Te=jnp.ones_like(built.state.Te),
        Ti=None,
        psi=None,
        N=None,
    )
    _, term_map, _, _ = built.system.rhs_terms(0.0, y)
    sheath = term_map["sheath"]

    dn = np.asarray(sheath.n)
    dve = np.asarray(sheath.vpar_e)
    dvi = np.asarray(sheath.vpar_i)
    dte = np.asarray(sheath.Te)

    assert float(np.max(np.abs(dn[1:-1]))) == 0.0
    assert float(np.max(np.abs(dte[1:-1]))) == 0.0
    assert float(np.max(np.abs(dve[1:-1]))) == 0.0
    assert float(np.max(np.abs(dvi[1:-1]))) == 0.0

    assert np.all(dn[0] < 0.0)
    assert np.all(dn[-1] < 0.0)
    assert np.all(dte[0] < 0.0)
    assert np.all(dte[-1] < 0.0)
    assert np.all(dvi[0] < 0.0)
    assert np.all(dvi[-1] > 0.0)
    assert np.all(dve[0] < 0.0)
    assert np.all(dve[-1] > 0.0)


def test_sheath_term_exposed_in_scheduler_map() -> None:
    built = build_system_from_config(_cfg(open_field_line=True, sheath_on=True))
    split, term_map = built.system.scheduler.run_with_terms(None, built.state)
    assert "sheath" in term_map
    total = split.total()
    assert total.n.shape == built.state.n.shape
