"""Lightweight checks for the frozen selective stage replay."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "work" / "boundary_load_audit"))
from analyze_selective_implicit_subsystems import FIELD_NAMES  # noqa: E402
from analyze_selective_implicit_subsystems import _ssp222_amplification  # noqa: E402
from replay_selective_implicit_subsystems import (  # noqa: E402
    CASES,
    _selection_mask,
    _ssp222_replay,
)


def _blocks(seed: int = 20260910, block_size: int = 2, algebraic_size: int = 3):
    rng = np.random.default_rng(seed)
    n = len(FIELD_NAMES) * block_size
    M = 1.3 * np.eye(n) + 0.01 * rng.standard_normal((n, n))
    A = 1.8 * np.eye(algebraic_size) + 0.01 * rng.standard_normal((algebraic_size, algebraic_size))
    B = 0.03 * rng.standard_normal((n, algebraic_size))
    C = 0.03 * rng.standard_normal((algebraic_size, n))
    H = M - B @ np.linalg.solve(A, C)
    return M, A, C, np.eye(n) - H


def test_selection_mask_has_exact_complement_and_field_block_layout():
    mask = _selection_mask(CASES["minimal"], 2)
    assert mask.shape == (12,)
    assert mask.reshape((len(FIELD_NAMES), 2)).tolist() == [
        [False, False], [True, True], [False, False],
        [True, True], [True, True], [True, True],
    ]
    assert int(mask.sum()) == 8
    assert int((~mask).sum()) == 4


def test_stage_equations_and_algebraic_reconstruction_are_satisfied():
    M, A, C, K = _blocks()
    initial = np.linspace(-0.4, 0.7, M.shape[0])
    result = _ssp222_replay(
        K, A, C, initial, CASES["minimal"], newton_tol=1.0e-11
    )
    assert result["excluded_rows_are_identity_unknowns"] is True
    assert result["excluded_rows_fixed_stage_data"] is False
    for stage in result["stages"].values():
        assert stage["converged"] is True
        assert stage["iterations"] == 1
        assert stage["final_residual_l2"] < 1.0e-11
        assert stage["algebraic_residual_l2"] < 1.0e-11
        assert stage["admissibility"]["available"] is False
        assert stage["bohm"]["available"] is False
    assert result["final_stage_equation_residual_l2"] < 1.0e-11


def test_full_case_is_reference_and_complement_is_explicit():
    M, A, C, K = _blocks()
    initial = np.ones(M.shape[0])
    minimal = _ssp222_replay(K, A, C, initial, CASES["minimal"], newton_tol=1.0e-10)
    conservative = _ssp222_replay(K, A, C, initial, CASES["conservative"], newton_tol=1.0e-10)
    full = _ssp222_replay(K, A, C, initial, CASES["full"], newton_tol=1.0e-10)
    assert minimal["explicit_material_fields"] == ["density", "Ti"]
    assert conservative["explicit_material_fields"] == ["density"]
    assert full["explicit_material_fields"] == []
    assert full["candidate_dof_count"] == M.shape[0] + A.shape[0]
    assert np.isfinite(minimal["final_stage_equation_residual_l2"])


def test_frozen_replay_matches_direct_ssp222_amplification_identity():
    _M, A, C, K = _blocks(seed=20260911, block_size=3, algebraic_size=2)
    rng = np.random.default_rng(20260912)
    initial = rng.standard_normal(K.shape[0])
    fields = CASES["minimal"]
    replay = _ssp222_replay(K, A, C, initial, fields, newton_tol=1.0e-11)
    selected = np.asarray(
        [field in set(fields) for field in FIELD_NAMES for _ in range(K.shape[0] // len(FIELD_NAMES))],
        dtype=bool,
    )
    direct = _ssp222_amplification(K, selected) @ initial
    np.testing.assert_allclose(replay["final_state"], direct, rtol=2.0e-12, atol=2.0e-12)
