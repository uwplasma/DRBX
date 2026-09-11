"""Contracts for the fixed-size phi solver diagnostic vector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_hsx_blob as driver  # noqa: E402


def test_phi_solver_diagnostics_include_augmented_solve_fields() -> None:
    info = SimpleNamespace(
        num_steps=7,
        final_residual_rel_l2=1.25e-12,
        failed=False,
        converged=True,
        compatibility_multiplier=0.125,
        raw_compatibility_defect=0.75,
        final_gauge_residual=2.5e-13,
    )
    diagnostics = np.asarray(driver._format_phi_solver_diagnostics(info))
    np.testing.assert_allclose(
        diagnostics,
        np.asarray(
            (
                7.0,
                1.25e-12,
                0.0,
                1.0,
                0.125,
                0.75,
                2.5e-13,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
            )
        ),
    )


def test_staged_phi_diagnostics_accept_only_finite_accepted_payload() -> None:
    diagnostics = np.zeros(driver._PHI_DIAGNOSTIC_WIDTH, dtype=np.float64)
    diagnostics[0] = 12.0
    diagnostics[1] = 2.0e-6
    diagnostics[3] = 1.0
    diagnostics[list(driver._PHI_DIAGNOSTIC_FINITE_SLOTS)] = 1.0
    accepted = driver._validate_staged_phi_solver_diagnostics(
        diagnostics, "imex1"
    )
    np.testing.assert_array_equal(accepted, diagnostics)

    for index in (2, 3, 11):
        rejected = diagnostics.copy()
        if index == 11:
            rejected[index] = 0.0
        elif index == 3:
            rejected[index] = 0.0
        else:
            rejected[index] = 1.0
        with pytest.raises(FloatingPointError, match="imex1 phi inversion rejected"):
            driver._validate_staged_phi_solver_diagnostics(rejected, "imex1")

    nonfinite = diagnostics.copy()
    nonfinite[1] = np.nan
    with pytest.raises(FloatingPointError, match="imex1 phi inversion rejected"):
        driver._validate_staged_phi_solver_diagnostics(nonfinite, "imex1")

    malformed = diagnostics[:-1]
    with pytest.raises(FloatingPointError, match="malformed diagnostics"):
        driver._validate_staged_phi_solver_diagnostics(malformed, "next")


def test_staged_phi_checks_precede_dependent_explicit_kernels() -> None:
    source = (ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert source.index(
        '_validate_staged_phi_solver_diagnostics(gmres_info_1, "imex1")'
    ) < source.index("explicit_1 = staged_explicit(")
    assert source.index(
        '_validate_staged_phi_solver_diagnostics(gmres_info_2, "imex2")'
    ) < source.index("explicit_2 = staged_explicit(")
