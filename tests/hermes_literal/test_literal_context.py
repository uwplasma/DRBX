from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.core.terms import build_context as build_core_context
from jaxdrb.driver import build_system_from_config
from jaxdrb.hermes_literal.context import build_context as build_literal_context


def _strict_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    return tomllib.loads(cfg_path.read_text(encoding="utf-8"))


def test_literal_context_matches_frozen_runtime_contract() -> None:
    built = build_system_from_config(_strict_cfg())

    literal = build_literal_context(built.system.params, built.system.geom, built.state)
    core = build_core_context(built.system.params, built.system.geom, built.state)

    assert np.allclose(np.asarray(literal.n_phys), np.asarray(core.n_phys))
    assert np.allclose(np.asarray(literal.Te_phys), np.asarray(core.Te_phys))
    assert np.allclose(np.asarray(literal.phi), np.asarray(core.phi))
    assert np.allclose(np.asarray(literal.n_prepared), np.asarray(core.n_prepared))
    assert np.allclose(np.asarray(literal.Te_prepared), np.asarray(core.Te_prepared))
    assert np.allclose(np.asarray(literal.pe_prepared), np.asarray(core.pe_prepared))
    assert np.allclose(np.asarray(literal.pi_prepared), np.asarray(core.pi_prepared))
    assert literal.hot_on == core.hot_on
    assert literal.em_on == core.em_on
    assert literal.neut_on == core.neut_on
