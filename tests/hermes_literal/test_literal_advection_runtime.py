from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.core.terms import build_context
from jaxdrb.core.terms.advection import exb_advection_terms as unified_exb_advection_terms
from jaxdrb.driver import build_system_from_config
from jaxdrb.hermes_literal.advection import exb_advection_terms as literal_exb_advection_terms


def _strict_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    return tomllib.loads(cfg_path.read_text(encoding="utf-8"))


def test_literal_advection_runtime_matches_frozen_runtime_contract() -> None:
    built = build_system_from_config(_strict_cfg())
    ctx = build_context(built.system.params, built.system.geom, built.state)

    literal = literal_exb_advection_terms(ctx, built.state)
    unified = unified_exb_advection_terms(ctx, built.state)

    assert np.allclose(np.asarray(literal.n), np.asarray(unified.n), atol=1e-6)
    assert np.allclose(np.asarray(literal.omega), np.asarray(unified.omega), atol=1e-6)
    assert np.allclose(np.asarray(literal.vpar_e), np.asarray(unified.vpar_e), atol=1e-6)
    assert np.allclose(np.asarray(literal.vpar_i), np.asarray(unified.vpar_i), atol=1e-6)
    assert np.allclose(np.asarray(literal.Te), np.asarray(unified.Te), atol=1e-6)
