"""Focused checks for the frozen selective implicit subsystem analyzer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work" / "boundary_load_audit"))
from analyze_selective_implicit_subsystems import (  # noqa: E402
    FIELD_NAMES,
    SSP222_GAMMA,
    _ssp222_amplification,
    analyze_subsystems,
    enumerate_field_subsets,
    reduced_stage_matrix,
    validate_blocks,
)


def _synthetic_blocks(seed: int = 20260910, block_size: int = 2, algebraic_size: int = 2):
    rng = np.random.default_rng(seed)
    material_size = len(FIELD_NAMES) * block_size
    M = 1.4 * np.eye(material_size) + 0.03 * rng.standard_normal((material_size, material_size))
    A = 1.7 * np.eye(algebraic_size) + 0.03 * rng.standard_normal((algebraic_size, algebraic_size))
    B = 0.04 * rng.standard_normal((material_size, algebraic_size))
    C = 0.04 * rng.standard_normal((algebraic_size, material_size))
    J = np.block([[M, B], [C, A]])
    return {"J": J, "M": M, "B": B, "C": C, "A": A}


def test_algebraic_reduction_matches_dense_schur_complement():
    blocks = _synthetic_blocks()
    expected = blocks["M"] - blocks["B"] @ np.linalg.solve(blocks["A"], blocks["C"])
    np.testing.assert_allclose(reduced_stage_matrix(blocks), expected)


def test_empty_and_all_limits_are_exact():
    result = analyze_subsystems(_synthetic_blocks())
    by_fields = {tuple(candidate["implicit_material_fields"]): candidate for candidate in result["candidates"]}
    assert () in by_fields
    assert tuple(FIELD_NAMES) in by_fields
    H = reduced_stage_matrix(_synthetic_blocks())
    K = np.eye(H.shape[0]) - H
    full_be = np.linalg.solve(H, np.eye(H.shape[0]))
    # Reconstruct the two limiting amplifications from the reported metrics.
    assert by_fields[()]["relative_frobenius_difference_from_full_BE"] == pytest.approx(
        np.linalg.norm((2.0 * np.eye(H.shape[0]) - H) - full_be, ord="fro")
        / np.linalg.norm(full_be, ord="fro")
    )
    assert by_fields[tuple(FIELD_NAMES)]["relative_frobenius_difference_from_full_BE"] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isfinite(K))


def test_default_and_all_subset_enumerations_have_expected_coverage_and_order():
    default = enumerate_field_subsets()
    all_subsets = enumerate_field_subsets(all_subsets=True)
    assert len(default) == 23
    assert len(all_subsets) == 64
    assert default == sorted(default, key=lambda subset: (len(subset), tuple(FIELD_NAMES.index(field) for field in subset)))
    assert all(set(("Ve", "vorticity")) <= set(subset) for subset in default if len(subset) >= 2 and subset not in [tuple([field]) for field in FIELD_NAMES])
    assert set(default).issubset(set(all_subsets))


def test_validation_rejects_bad_shapes_and_inconsistent_full_matrix():
    blocks = _synthetic_blocks()
    malformed = dict(blocks)
    malformed["B"] = blocks["B"][:-1]
    with pytest.raises(ValueError, match="B has"):
        validate_blocks(malformed)
    malformed = dict(blocks)
    malformed["J"] = blocks["J"].copy()
    malformed["J"][0, 0] += 1.0
    with pytest.raises(ValueError, match="J does not match"):
        validate_blocks(malformed)


def test_report_contains_requested_cost_and_caveat_fields():
    result = analyze_subsystems(_synthetic_blocks())
    assert result["candidate_count"] == 23
    assert set(result["field_block_frobenius_norms_of_K"]) == set(FIELD_NAMES)
    for candidate in result["candidates"]:
        assert "candidate_dof_count" in candidate
        assert "condition_D" in candidate
        assert "spectral_radius_G" in candidate
        assert "max_singular_value_G" in candidate
        assert "relative_frobenius_difference_from_full_BE" in candidate
        assert "spectral_radius_G_ssp222" in candidate
        assert "max_singular_value_G_ssp222" in candidate
        assert "relative_frobenius_difference_from_full_ssp222" in candidate
        assert "most_damped_mode_gain_max_ssp222" in candidate
    assert any("48^3" in caveat for caveat in result["caveats"])


def test_ssp222_empty_and_all_limits_match_their_closed_forms():
    rng = np.random.default_rng(20260911)
    K = 0.04 * rng.standard_normal((6, 6)) - 0.2 * np.eye(6)
    identity = np.eye(6)
    empty = _ssp222_amplification(K, np.zeros(6, dtype=bool))
    scaled = K / SSP222_GAMMA
    np.testing.assert_allclose(empty, identity + scaled + 0.5 * scaled @ scaled)

    all_fields = _ssp222_amplification(K, np.ones(6, dtype=bool))
    D = identity - K
    S1 = np.linalg.solve(D, identity)
    S2 = np.linalg.solve(D, identity + (1.0 - 2.0 * SSP222_GAMMA) * scaled @ S1)
    expected_all = identity + 0.5 * scaled @ (S1 + S2)
    np.testing.assert_allclose(all_fields, expected_all)

    blocks = _synthetic_blocks(block_size=1, algebraic_size=6)
    result = analyze_subsystems(blocks)
    candidates = {
        tuple(candidate["implicit_material_fields"]): candidate
        for candidate in result["candidates"]
    }
    assert candidates[()]["relative_frobenius_difference_from_full_ssp222"] > 0.0
    assert candidates[tuple(FIELD_NAMES)]["relative_frobenius_difference_from_full_ssp222"] == pytest.approx(0.0, abs=1e-12)


def test_nonnormal_stiff_modes_distinguish_selective_candidates():
    # A triangular K has the same real eigenvalues as its diagonal but a
    # large nonnormal coupling from vorticity into Ve.  The selective
    # amplification sees that coupling even when an eigenvalue-only summary
    # does not.
    K = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    K[4, 5] = 10.0
    M = np.eye(6) - K
    A = np.eye(6)
    B = np.zeros((6, 6))
    C = np.zeros((6, 6))
    blocks = {"J": np.block([[M, B], [C, A]]), "M": M, "B": B, "C": C, "A": A}
    result = analyze_subsystems(blocks)
    candidates = {
        tuple(candidate["implicit_material_fields"]): candidate
        for candidate in result["candidates"]
    }
    selective = candidates[("Ve", "vorticity")]
    all_fields = candidates[tuple(FIELD_NAMES)]
    assert selective["most_damped_mode_gain_max"] > all_fields["most_damped_mode_gain_max"]
    assert selective["most_damped_mode_gain_max"] > 1.0
    assert len(selective["most_damped_mode_gains"]) == 4
    assert all("full_BE_gain" in mode for mode in selective["most_damped_mode_gains"])
