from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config
from jaxdrb.hermes_literal.bcs import resolve_bcs
from jaxdrb.hermes_literal.field import interior_view
from jaxdrb.hermes_literal.state import build_literal_stage1_state


def _strict_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    return tomllib.loads(cfg_path.read_text(encoding="utf-8"))


def test_build_literal_stage1_state_preserves_prepared_interiors() -> None:
    built = build_system_from_config(_strict_cfg())
    params = built.system.params
    geom = built.system.geom
    y = built.state
    bcs = resolve_bcs(params, geom)

    density = built.system._phys_n(y.n)
    Te = built.system._phys_Te(y.Te)
    Ti = y.Ti if y.Ti is not None else np.zeros_like(np.asarray(Te))
    pe = density * Te
    pi = density * Ti
    phi = built.system._phi_from_omega(y.omega, n=density, Ti=y.Ti, Te=Te)

    state = build_literal_stage1_state(
        params=params,
        geom=geom,
        bcs=bcs,
        density=density,
        electron_temperature=Te,
        ion_temperature=Ti,
        electron_pressure=pe,
        ion_pressure=pi,
        phi=phi,
        vpar_e=y.vpar_e,
        vpar_i=y.vpar_i,
    )

    assert state.electrons.layout is not None
    np.testing.assert_allclose(
        np.asarray(interior_view(state.electrons.density_guarded, state.electrons.layout)),
        np.asarray(density),
    )
    np.testing.assert_allclose(
        np.asarray(interior_view(state.fields.phi_guarded, state.fields.layout)),
        np.asarray(phi),
    )
    assert np.min(np.asarray(state.fields.fastest_wave)) >= 0.0
