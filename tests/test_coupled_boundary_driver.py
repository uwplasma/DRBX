"""CLI contract tests for the guarded coupled-boundary IMEX selector."""

import pytest

from simulate_hsx_blob import _build_parser, _coupled_phi_diagnostics_accepted
from drbx.native.fci_boundary_imex import advance_ssp222_coupled
import jax.numpy as jnp
import numpy as np


def test_coupled_boundary_selector_is_opt_in_and_historical_remains_default():
    parser = _build_parser()
    assert parser.parse_args([]).imex_split == "historical"
    args = parser.parse_args(["--imex-split", "coupled-boundary"])
    assert args.imex_split == "coupled-boundary"


def test_unknown_split_is_rejected():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--imex-split", "not-a-split"])


def test_final_nonfinite_diagnostic_rolls_back_transaction():
    def solve_stage(base, stage_dt):
        return base, jnp.ones_like(base), {"converged": True}
    def reconstruct(value):
        return value, {"converged": False, "reason": "phi_diagnostics_nonfinite"}
    result, info = advance_ssp222_coupled(
        jnp.asarray([3.0]), 0.1, solve_stage=solve_stage,
        explicit=lambda state, source: jnp.zeros_like(state),
        reconstruct_final=reconstruct,
    )
    np.testing.assert_array_equal(result, jnp.asarray([3.0]))
    assert not info["converged"]
    assert info["reason"] == "final_reconstruction_failed:phi_diagnostics_nonfinite"


def test_coupled_phi_acceptance_requires_all_strict_guards():
    baseline = np.zeros(19, dtype=float)
    baseline[3] = 1.0
    baseline[[11, 12, 13, 15, 16]] = 1.0
    assert _coupled_phi_diagnostics_accepted(baseline)
    for index in (3, 11, 12, 13, 15, 16):
        candidate = baseline.copy()
        candidate[index] = 0.0
        assert not _coupled_phi_diagnostics_accepted(candidate)
    candidate = baseline.copy(); candidate[6] = 1.0e-9
    assert not _coupled_phi_diagnostics_accepted(candidate)
    candidate = baseline.copy(); candidate[8] = np.nan
    assert not _coupled_phi_diagnostics_accepted(candidate)
