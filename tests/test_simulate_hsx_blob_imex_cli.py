"""Focused parser coverage for the HSX IMEX driver options."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulate_hsx_blob import (  # noqa: E402
    _build_parser,
    _print_imex_stage_diagnostics,
)


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


def test_imex_bdf2_cli_choice_and_fixed_step_startup_contract() -> None:
    args = _build_parser().parse_args(("--time-integrator", "imex-bdf2"))
    assert args.time_integrator == "imex-bdf2"
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert "fixed-step SBDF2 with one classical RK4 startup step" in source
    assert "one complete shard_map RK4 startup" in source
    assert "full_rk4_advance" in source
    assert "ImexBdf2Stepper" in source
    assert "bdf2_state_nm1" in source
    assert "bdf2_explicit_rhs_nm1" in source
    assert "result.explicit_rhs" in source
    assert "one Newton solve per BDF2 step after classical RK4 startup" in source
    assert "startup RK4 phi-GMRES work is reported separately" in source


def test_bdf2_uses_predictor_for_known_explicit_fields_and_extrapolation_guess() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    residual_start = source.index("def full_imex_bdf2_advance")
    residual_source = source[residual_start:]
    assert "implicit_state_from_eb_state(predictor)" in residual_source
    assert "implicit_state_from_eb_state(extrapolated_state)" in residual_source
    assert "dt_gamma=alpha" in residual_source
    assert "raw.replace(phi=alpha * raw.phi)" in residual_source
    assert "model.polarization_residual" in residual_source
    provisional = residual_source.index("provisional_state =")
    reconstruct = residual_source.index(
        "reconstructed_phi, final_phi_info = model.reconstruct_phi"
    )
    constrained = residual_source.index(
        "accepted_state = provisional_state.replace(phi=reconstructed_phi)"
    )
    explicit_history = residual_source.index("result = stepper(")
    assert provisional < reconstruct < constrained < explicit_history
    assert "result.explicit_rhs" in residual_source


def test_six_field_implicit_rhs_embedding_keeps_only_vi_explicit() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    embedding_start = source.index("def _implicit_rhs_as_full_state")
    embedding_end = source.index("def _newton_diagnostics", embedding_start)
    embedding = source[embedding_start:embedding_end]

    assert '"""Embed the six implicit derivatives' in embedding
    assert "Ti=implicit_rhs.Ti" in embedding
    assert "Vi=jnp.zeros_like(reference.Vi)" in embedding
    assert "Ti=jnp.zeros_like(reference.Ti)" not in embedding
    assert "only ``Vi`` remains" in embedding

    ark2_start = source.index("def full_ark2_imex_advance")
    assert "six unknowns ``(n, Te, Ti, Ve,\n        omega, phi)``" in source[ark2_start:]


def test_bdf2_projects_ti_and_uses_constrained_six_field_state() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    bdf2_start = source.index("def full_imex_bdf2_advance")
    bdf2_source = source[bdf2_start:]

    assert "six-field Newton state" in bdf2_source
    assert "only Vi remains known outside the implicit solve" in bdf2_source
    assert "implicit_predictor = implicit_state_from_eb_state(predictor)" in bdf2_source
    assert "initial_guess = implicit_state_from_eb_state(extrapolated_state)" in bdf2_source
    assert "accepted_state = provisional_state.replace(phi=reconstructed_phi)" in bdf2_source
    assert "return (\n                accepted_state," in bdf2_source
    assert "result.explicit_rhs" in bdf2_source


def test_bdf2_exposes_newton_and_final_phi_rows_and_host_accepts_both() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert 'imex_stage_names = ("bdf2-newton", "final-phi")' in source
    assert "_newton_diagnostics(info, provisional_phi_residual)" in source
    assert "final_phi_info.final_residual_rel_l2" in source
    assert "imex_solver_diagnostics_host[1, 2]" in source
    assert "imex_solver_diagnostics_host[1, 4]" in source
    assert "imex_solver_diagnostics_host[0, 4]" in source
    assert "imex_solver_diagnostics_host[0, 0]" in source


def test_bdf2_disables_phase_timing_and_curvature_defect_restriction_is_shared() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert 'time_integrator in ("ark2-imex", "imex-bdf2")' in source
    assert 'only by --time-integrator=rk4' in source


def test_ark_stage_positivity_guard_includes_both_temperatures() -> None:
    source = (_ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    assert "stage_Ti_min" in source
    assert "or stage_Te_min <= 0.0" in source
    assert "or stage_Ti_min <= 0.0" in source


def test_bdf2_rejection_diagnostics_accept_actual_three_states_without_ark_zip_error(
    capsys,
) -> None:
    diagnostics = np.zeros((3, 7, 3), dtype=np.float64)
    _print_imex_stage_diagnostics(
        ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"),
        diagnostics,
    )
    output = capsys.readouterr().out
    assert "[diagnostics] bdf2-current:" in output
    assert "[diagnostics] bdf2-predictor:" in output
    assert "[diagnostics] bdf2-solved/constrained:" in output
