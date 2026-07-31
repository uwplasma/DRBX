"""Focused parser coverage for the HSX ARK2 IMEX driver options."""

from __future__ import annotations

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulate_hsx_blob import _build_parser  # noqa: E402


def test_ark2_imex_cli_options_are_exposed_with_independent_tolerances() -> None:
    args = _build_parser().parse_args(
        (
            "--time-integrator",
            "ark2-imex",
            "--newton-rtol",
            "2e-7",
            "--newton-atol",
            "3e-10",
            "--newton-acceptance-rtol",
            "4e-7",
            "--newton-acceptance-atol",
            "5e-10",
            "--newton-max-steps",
            "9",
            "--newton-linear-restart",
            "13",
            "--newton-linear-rtol",
            "6e-3",
            "--newton-linear-atol",
            "7e-11",
            "--newton-linear-max-restarts",
            "8",
            "--newton-preconditioner",
            "none",
        )
    )
    assert args.time_integrator == "ark2-imex"
    assert args.newton_rtol == 2.0e-7
    assert args.newton_atol == 3.0e-10
    assert args.newton_acceptance_rtol == 4.0e-7
    assert args.newton_acceptance_atol == 5.0e-10
    assert args.newton_max_steps == 9
    assert args.newton_linear_restart == 13
    assert args.newton_linear_rtol == 6.0e-3
    assert args.newton_linear_atol == 7.0e-11
    assert args.newton_linear_max_restarts == 8
    assert args.newton_preconditioner == "none"


def test_rk4_remains_the_default_integrator() -> None:
    args = _build_parser().parse_args(())
    assert args.time_integrator == "rk4"


def test_ark_stage_positivity_guard_includes_both_temperatures() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert "stage_Ti_min" in source
    assert "or stage_Te_min <= 0.0" in source
    assert "or stage_Ti_min <= 0.0" in source
