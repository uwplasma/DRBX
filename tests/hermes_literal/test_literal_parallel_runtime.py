from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.core.terms import build_context
from jaxdrb.core.terms.parallel import (
    parallel_conservative_terms as unified_parallel_conservative_terms,
    parallel_vars as unified_parallel_vars,
)
from jaxdrb.driver import build_system_from_config
from jaxdrb.hermes_literal.parallel import (
    parallel_conservative_terms as literal_parallel_conservative_terms,
    parallel_vars as literal_parallel_vars,
)


def _strict_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    return tomllib.loads(cfg_path.read_text(encoding="utf-8"))


def test_literal_parallel_runtime_matches_frozen_runtime_contract() -> None:
    cfg = copy.deepcopy(_strict_cfg())
    cfg["numerics"]["hermes_mirror_parallel_subdomain_size"] = 0
    built = build_system_from_config(cfg)
    ctx = build_context(built.system.params, built.system.geom, built.state)

    literal = literal_parallel_vars(ctx, built.state)
    unified = unified_parallel_vars(ctx, built.state)

    assert np.allclose(np.asarray(literal.vpar_e_flux), np.asarray(unified.vpar_e_flux))
    assert np.allclose(np.asarray(literal.vpar_i_flux), np.asarray(unified.vpar_i_flux))
    assert np.allclose(np.asarray(literal.dpar_j), np.asarray(unified.dpar_j))
    assert np.allclose(np.asarray(literal.jpar_total), np.asarray(unified.jpar_total))

    assert literal.sheath_data is not None
    assert unified.sheath_data is not None
    assert np.allclose(
        np.asarray(literal.sheath_data.n_ghost_low), np.asarray(unified.sheath_data.n_ghost_low)
    )
    assert np.allclose(
        np.asarray(literal.sheath_data.n_ghost_high), np.asarray(unified.sheath_data.n_ghost_high)
    )

    literal_terms = literal_parallel_conservative_terms(ctx, built.state, literal)
    unified_terms = unified_parallel_conservative_terms(ctx, built.state, unified)
    assert np.allclose(np.asarray(literal_terms.n), np.asarray(unified_terms.n), atol=1e-6)
    assert np.allclose(np.asarray(literal_terms.omega), np.asarray(unified_terms.omega), atol=1e-12)
    assert np.allclose(np.asarray(literal_terms.Te), np.asarray(unified_terms.Te), atol=1e-6)
