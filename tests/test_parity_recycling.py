from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native import runner as native_runner
from jax_drb.native import run_curated_case
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import compute_recycling_1d_rhs
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    build_dataset_array_payload,
    load_portable_array_payload,
)
from jax_drb.parity.diff import (
    build_array_diff_report,
    build_scaled_array_diff_entries,
    compare_recycling_artifacts,
    format_array_diff_report,
    format_recycling_diff_report,
)
from jax_drb.parity.reference import run_reference_case
from jax_drb.parity.arrays import write_portable_array_payload
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.reference.cases import load_reference_cases
from jax_drb.runtime.run_config import RunConfiguration


_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")
_STAGED_REFERENCE_1D = Path("/private/tmp/jax_drb_recycling_1d_one_step_inspect")
_STAGED_REFERENCE_DTHE = Path("/private/tmp/jax_drb_recycling_dthe_one_step_inspect")
_STAGED_REFERENCE_DTHE_DIAG = Path("/private/tmp/jax_drb_recycling_dthe_one_step_diag2")
_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_INPUT_DTHE = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


def _has_staged_reference_files(path: Path) -> bool:
    return path.exists() and (path / "BOUT.dmp.0.nc").exists() and (path / "BOUT.restart.0.nc").exists()


def test_staged_reference_controller_snapshot_extracts_expected_1d_values() -> None:
    if not _has_staged_reference_files(_STAGED_REFERENCE_1D):
        pytest.skip("staged 1D recycling reference artifacts are unavailable")

    snapshot = extract_recycling_controller_snapshot(
        _STAGED_REFERENCE_1D / "BOUT.dmp.0.nc",
        _STAGED_REFERENCE_1D / "BOUT.restart.0.nc",
        controller_species=("d+",),
    )

    assert snapshot.controller_multipliers["d+"] == pytest.approx(3.1024529348950325)
    assert snapshot.controller_proportional_terms["d+"] == pytest.approx(3.099049102856133)
    assert snapshot.controller_integral_terms["d+"] == pytest.approx(0.003403832038899506)
    assert snapshot.controller_sources["d+"].reshape(-1)[0] == pytest.approx(3.238863060600471e-07)
    assert snapshot.restart_integrals["d+"] == pytest.approx(6.807664077799012)


def test_staged_reference_controller_snapshot_extracts_expected_dthe_values() -> None:
    if not _has_staged_reference_files(_STAGED_REFERENCE_DTHE):
        pytest.skip("staged multispecies recycling reference artifacts are unavailable")

    snapshot = extract_recycling_controller_snapshot(
        _STAGED_REFERENCE_DTHE / "BOUT.dmp.0.nc",
        _STAGED_REFERENCE_DTHE / "BOUT.restart.0.nc",
        controller_species=("d+", "t+", "he+"),
    )

    assert snapshot.controller_multipliers["he+"] == pytest.approx(496.2819811969886)
    assert snapshot.controller_proportional_terms["he+"] == pytest.approx(494.9937667193512)
    assert snapshot.controller_integral_terms["he+"] == pytest.approx(1.2882144776374491)
    assert snapshot.controller_sources["he+"].reshape(-1)[0] == pytest.approx(0.0)
    assert snapshot.restart_integrals["he+"] == pytest.approx(2576.4289552748983)


def test_staged_recycling_1d_evolved_rhs_stays_within_locked_tolerances() -> None:
    if not _has_staged_reference_files(_STAGED_REFERENCE_1D):
        pytest.skip("staged 1D recycling reference artifacts are unavailable")

    diffs = _staged_rhs_differences(
        _INPUT_1D,
        _STAGED_REFERENCE_1D,
        controller_species=("d+",),
        state_fields=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        compare_fields=("ddt(Nd+)", "ddt(Nd)", "ddt(Pe)", "ddt(Pd+)", "ddt(Pd)", "ddt(NVd+)"),
    )

    assert diffs["ddt(Nd+)"] <= 2.0e-5
    assert diffs["ddt(Nd)"] <= 1.0e-3
    assert diffs["ddt(Pe)"] <= 4.0e-4
    assert diffs["ddt(Pd+)"] <= 4.5e-4
    assert diffs["ddt(Pd)"] <= 2.0e-4
    assert diffs["ddt(NVd+)"] <= 1.0e-3


def test_staged_recycling_dthe_evolved_rhs_stays_within_locked_tolerances() -> None:
    if not _has_staged_reference_files(_STAGED_REFERENCE_DTHE):
        pytest.skip("staged multispecies recycling reference artifacts are unavailable")

    diffs = _staged_rhs_differences(
        _INPUT_DTHE,
        _STAGED_REFERENCE_DTHE,
        controller_species=("d+", "t+", "he+"),
        state_fields=(
            "Nd+",
            "Pd+",
            "NVd+",
            "Nt+",
            "Pt+",
            "NVt+",
            "Nhe+",
            "Phe+",
            "NVhe+",
            "Nd",
            "Pd",
            "NVd",
            "Nt",
            "Pt",
            "NVt",
            "Nhe",
            "Phe",
            "NVhe",
            "Pe",
        ),
        compare_fields=(
            "ddt(Nd+)",
            "ddt(Nt+)",
            "ddt(Nhe+)",
            "ddt(Pe)",
            "ddt(Pd+)",
            "ddt(Pt+)",
            "ddt(Phe+)",
            "ddt(NVd+)",
            "ddt(NVt+)",
            "ddt(NVhe+)",
        ),
    )

    assert diffs["ddt(Nd+)"] <= 2.0e-5
    assert diffs["ddt(Nt+)"] <= 2.0e-5
    assert diffs["ddt(Nhe+)"] <= 1.0e-7
    assert diffs["ddt(Pe)"] <= 5.0e-4
    assert diffs["ddt(Pd+)"] <= 4.0e-4
    assert diffs["ddt(Pt+)"] <= 1.0e-3
    assert diffs["ddt(Phe+)"] <= 2.0e-7
    assert diffs["ddt(NVd+)"] <= 2.0e-3
    assert diffs["ddt(NVt+)"] <= 2.0e-3
    assert diffs["ddt(NVhe+)"] <= 2.0e-5


def test_staged_recycling_dthe_evolved_diagnostics_stay_within_locked_tolerances() -> None:
    if not _STAGED_REFERENCE_DTHE_DIAG.exists():
        pytest.skip("staged multispecies recycling diagnostic artifacts are unavailable")

    diffs = _staged_rhs_differences(
        _INPUT_DTHE,
        _STAGED_REFERENCE_DTHE_DIAG,
        controller_species=("d+", "t+", "he+"),
        state_fields=(
            "Nd+",
            "Pd+",
            "NVd+",
            "Nt+",
            "Pt+",
            "NVt+",
            "Nhe+",
            "Phe+",
            "NVhe+",
            "Nd",
            "Pd",
            "NVd",
            "Nt",
            "Pt",
            "NVt",
            "Nhe",
            "Phe",
            "NVhe",
            "Pe",
        ),
        compare_fields=(
            "Ve",
            "Epar",
            "SNVd+",
            "SNVt+",
            "SNVhe+",
            "SNVd",
            "SNVt",
            "SNVhe",
            "Fd+d_coll",
            "Fd+e_coll",
            "Ft+t_coll",
            "Ft+e_coll",
            "DivPiPar_d+",
            "DivPiPar_t+",
            "DivPiPar_he+",
            "Fd+_iz",
            "Ft+_iz",
            "Fd+_rec",
            "Ft+_rec",
            "Fdt+_cx",
            "Ft+d_cx",
            "Ftd+_cx",
            "Fd+t_cx",
            "Fdd+_cx",
            "Ftt+_cx",
        ),
    )

    assert diffs["Ve"] <= 1.0e-12
    assert diffs["Epar"] <= 8.0e-5
    assert diffs["SNVd+"] <= 1.0e-3
    assert diffs["SNVt+"] <= 2.5e-3
    assert diffs["SNVhe+"] <= 3.5e-5
    assert diffs["SNVd"] <= 2.5e-3
    assert diffs["SNVt"] <= 1.0e-3
    assert diffs["SNVhe"] <= 3.0e-5
    assert diffs["Fd+d_coll"] <= 1.0e-9
    assert diffs["Fd+e_coll"] <= 1.0e-9
    assert diffs["Ft+t_coll"] <= 1.0e-9
    assert diffs["Ft+e_coll"] <= 1.0e-9
    assert diffs["DivPiPar_d+"] <= 1.3e-3
    assert diffs["DivPiPar_t+"] <= 1.7e-3
    assert diffs["DivPiPar_he+"] <= 3.5e-5
    assert diffs["Fd+_iz"] <= 2.0e-6
    assert diffs["Ft+_iz"] <= 2.0e-6
    assert diffs["Fd+_rec"] <= 1.0e-9
    assert diffs["Ft+_rec"] <= 1.0e-9
    assert diffs["Fdt+_cx"] <= 1.0e-9
    assert diffs["Ft+d_cx"] <= 1.0e-9
    assert diffs["Ftd+_cx"] <= 1.0e-9
    assert diffs["Fd+t_cx"] <= 1.0e-9
    assert diffs["Fdd+_cx"] <= 1.0e-9
    assert diffs["Ftt+_cx"] <= 1.0e-9


def test_recycling_1d_one_step_native_parity_stays_within_exact_relative_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY=1 to run the bounded recycling one-step parity gate")

    expected = load_portable_array_payload(_BASELINE_DIR / "recycling_1d_one_step.npz")
    result = run_curated_case("recycling_1d_one_step", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=tuple(expected["compare_variables"]),
    )

    assert entries
    worst_relative = max(
        entry.relative_to_expected_max or 0.0
        for entry in entries
    )
    assert worst_relative < 5.0e-2, entries


def test_recycling_dthe_one_step_native_parity_stays_within_exact_relative_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY=1 to run the bounded multispecies recycling one-step parity gate")

    expected = load_portable_array_payload(_BASELINE_DIR / "recycling_dthe_one_step.npz")
    result = run_curated_case("recycling_dthe_one_step", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=tuple(expected["compare_variables"]),
    )

    assert entries
    worst_relative = max(
        entry.relative_to_expected_max or 0.0
        for entry in entries
    )
    assert worst_relative < 5.0e-2, entries


def _assert_mixed_exact_band(
    *,
    case_name: str,
    env_flag: str,
    near_zero_atol: float,
    max_relative: float,
    max_near_zero_abs_diff: float,
) -> None:
    if os.environ.get(env_flag) != "1":
        pytest.skip(f"set {env_flag}=1 to run the bounded parity gate for {case_name}")

    expected = load_portable_array_payload(_BASELINE_DIR / f"{case_name}.npz")
    result = run_curated_case(case_name, reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=tuple(expected["compare_variables"]),
        near_zero_atol=near_zero_atol,
    )

    assert entries
    non_near_zero_entries = tuple(entry for entry in entries if not entry.near_zero_expected)
    near_zero_entries = tuple(entry for entry in entries if entry.near_zero_expected)

    worst_relative = max((entry.relative_to_expected_max or 0.0) for entry in non_near_zero_entries)
    worst_near_zero_abs_diff = max((entry.max_abs_diff for entry in near_zero_entries), default=0.0)

    assert worst_relative < max_relative, entries
    assert worst_near_zero_abs_diff < max_near_zero_abs_diff, near_zero_entries


def _assert_live_tokamak_recycling_dthe_two_output_window_operational_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_2D_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_2D_PARITY=1 to run the bounded parity gate for the richer D/T tokamak recycling window")

    case = next(reference_case for reference_case in load_reference_cases() if reference_case.name == "tokamak_recycling_dthe_one_step")
    probe_case = replace(
        case,
        name="tokamak_recycling_dthe_short_window",
        parity_mode="short_window",
        capability_tier="native_operational",
        extra_overrides=tuple(
            override for override in case.extra_overrides if not override.startswith("nout=")
        )
        + ("nout=2",),
    )
    input_path = _REFERENCE_ROOT / case.reference_path

    with tempfile.TemporaryDirectory(prefix="jaxdrb-dthe-two-output-window-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=_REFERENCE_ROOT,
            extra_overrides=("nout=2",),
            workdir=workdir,
            keep_workdir=True,
        )
        summary = execution.summary
        expected = build_dataset_array_payload(
            Path(summary.artifacts["BOUT.dmp.0.nc"]),
            case_name=probe_case.name,
            parity_mode=probe_case.parity_mode,
            capability_tier=probe_case.capability_tier,
            compare_variables=probe_case.compare_variables,
            component_labels=tuple(summary.component_labels),
            overrides=tuple(summary.overrides),
            trim_x_guards=probe_case.trim_x_guards,
            x_guards=2,
            trim_y_guards=probe_case.trim_y_guards,
            y_guards=2,
            configured_nout=2,
            configured_timestep=summary.timestep,
        )

    result = native_runner._run_integrated_2d_recycling_transient_case(
        probe_case,
        input_path=input_path,
        reference_root=_REFERENCE_ROOT,
        steps=2,
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=probe_case.compare_variables,
        near_zero_atol=5.0e-5,
    )

    assert entries
    non_near_zero_entries = tuple(entry for entry in entries if not entry.near_zero_expected)
    near_zero_entries = tuple(entry for entry in entries if entry.near_zero_expected)

    worst_relative = max((entry.relative_to_expected_max or 0.0) for entry in non_near_zero_entries)
    worst_near_zero_abs_diff = max((entry.max_abs_diff for entry in near_zero_entries), default=0.0)

    assert worst_relative < 3.0e-2, entries
    assert worst_near_zero_abs_diff < 3.0e-5, near_zero_entries


def _assert_live_tokamak_recycling_dthe_drifts_two_output_window_operational_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_2D_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_2D_PARITY=1 to run the bounded parity gate for the richer drift-enabled D/T tokamak recycling window")

    case = next(
        reference_case
        for reference_case in load_reference_cases()
        if reference_case.name == "tokamak_recycling_dthe_drifts_one_step"
    )
    probe_case = replace(
        case,
        name="tokamak_recycling_dthe_drifts_short_window",
        parity_mode="short_window",
        capability_tier="native_operational",
        extra_overrides=tuple(
            override for override in case.extra_overrides if not override.startswith("nout=")
        )
        + ("nout=2",),
    )
    input_path = _REFERENCE_ROOT / case.reference_path

    with tempfile.TemporaryDirectory(prefix="jaxdrb-dthe-drifts-two-output-window-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=_REFERENCE_ROOT,
            extra_overrides=("nout=2",),
            workdir=workdir,
            keep_workdir=True,
        )
        summary = execution.summary
        expected = build_dataset_array_payload(
            Path(summary.artifacts["BOUT.dmp.0.nc"]),
            case_name=probe_case.name,
            parity_mode=probe_case.parity_mode,
            capability_tier=probe_case.capability_tier,
            compare_variables=probe_case.compare_variables,
            component_labels=tuple(summary.component_labels),
            overrides=tuple(summary.overrides),
            trim_x_guards=probe_case.trim_x_guards,
            x_guards=2,
            trim_y_guards=probe_case.trim_y_guards,
            y_guards=2,
            configured_nout=2,
            configured_timestep=summary.timestep,
        )

    result = native_runner._run_integrated_2d_recycling_transient_case(
        probe_case,
        input_path=input_path,
        reference_root=_REFERENCE_ROOT,
        steps=2,
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=probe_case.compare_variables,
        near_zero_atol=5.0e-5,
    )

    assert entries
    non_near_zero_entries = tuple(entry for entry in entries if not entry.near_zero_expected)
    near_zero_entries = tuple(entry for entry in entries if entry.near_zero_expected)

    worst_relative = max((entry.relative_to_expected_max or 0.0) for entry in non_near_zero_entries)
    worst_near_zero_abs_diff = max((entry.max_abs_diff for entry in near_zero_entries), default=0.0)

    assert worst_relative < 3.0e-2, entries
    assert worst_near_zero_abs_diff < 3.0e-5, near_zero_entries


def _assert_live_tokamak_recycling_dthene_one_step_operational_mixed_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_2D_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_2D_PARITY=1 to run the bounded parity gate for the neon-enabled tokamak recycling one-step lane")

    case = next(
        reference_case
        for reference_case in load_reference_cases()
        if reference_case.name == "tokamak_recycling_dthene_one_step"
    )

    with tempfile.TemporaryDirectory(prefix="jaxdrb-dthene-one-step-live-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=_REFERENCE_ROOT,
            workdir=workdir,
            keep_workdir=True,
        )
        summary = execution.summary
        expected = build_dataset_array_payload(
            Path(summary.artifacts["BOUT.dmp.0.nc"]),
            case_name=case.name,
            parity_mode=case.parity_mode,
            capability_tier="native_operational",
            compare_variables=case.compare_variables,
            component_labels=tuple(summary.component_labels),
            overrides=tuple(summary.overrides),
            trim_x_guards=case.trim_x_guards,
            x_guards=2,
            trim_y_guards=case.trim_y_guards,
            y_guards=2,
            configured_nout=summary.nout,
            configured_timestep=summary.timestep,
        )

    result = run_curated_case(case.name, reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=case.compare_variables,
        near_zero_atol=1.0e-4,
    )

    assert entries
    non_near_zero_entries = tuple(entry for entry in entries if not entry.near_zero_expected)
    near_zero_entries = tuple(entry for entry in entries if entry.near_zero_expected)

    worst_relative = max((entry.relative_to_expected_max or 0.0) for entry in non_near_zero_entries)
    worst_near_zero_abs_diff = max((entry.max_abs_diff for entry in near_zero_entries), default=0.0)

    assert worst_relative < 1.0e-3, entries
    assert worst_near_zero_abs_diff < 1.0e-4, near_zero_entries


def _assert_live_tokamak_recycling_dthene_three_output_window_operational_mixed_band() -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_2D_PARITY") != "1":
        pytest.skip("set JAX_DRB_RUN_RECYCLING_2D_PARITY=1 to run the bounded parity gate for the neon-enabled tokamak recycling short window")

    case = next(
        reference_case
        for reference_case in load_reference_cases()
        if reference_case.name == "tokamak_recycling_dthene_one_step"
    )
    probe_case = replace(
        case,
        name="tokamak_recycling_dthene_short_window",
        parity_mode="short_window",
        capability_tier="native_operational",
        extra_overrides=tuple(
            override for override in case.extra_overrides if not override.startswith("nout=")
        )
        + ("nout=3",),
    )
    input_path = _REFERENCE_ROOT / case.reference_path

    with tempfile.TemporaryDirectory(prefix="jaxdrb-dthene-three-output-window-") as workdir:
        execution = run_reference_case(
            case.name,
            reference_root=_REFERENCE_ROOT,
            extra_overrides=("nout=3",),
            workdir=workdir,
            keep_workdir=True,
        )
        summary = execution.summary
        expected = build_dataset_array_payload(
            Path(summary.artifacts["BOUT.dmp.0.nc"]),
            case_name=probe_case.name,
            parity_mode=probe_case.parity_mode,
            capability_tier=probe_case.capability_tier,
            compare_variables=probe_case.compare_variables,
            component_labels=tuple(summary.component_labels),
            overrides=tuple(summary.overrides),
            trim_x_guards=probe_case.trim_x_guards,
            x_guards=2,
            trim_y_guards=probe_case.trim_y_guards,
            y_guards=2,
            configured_nout=3,
            configured_timestep=summary.timestep,
        )

    result = native_runner._run_integrated_2d_recycling_transient_case(
        probe_case,
        input_path=input_path,
        reference_root=_REFERENCE_ROOT,
        steps=3,
    )
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=probe_case.compare_variables,
        near_zero_atol=5.0e-4,
    )

    assert entries
    non_near_zero_entries = tuple(entry for entry in entries if not entry.near_zero_expected)
    near_zero_entries = tuple(entry for entry in entries if entry.near_zero_expected)

    worst_relative = max((entry.relative_to_expected_max or 0.0) for entry in non_near_zero_entries)
    worst_near_zero_abs_diff = max((entry.max_abs_diff for entry in near_zero_entries), default=0.0)

    assert worst_relative < 5.0e-3, entries
    assert worst_near_zero_abs_diff < 3.0e-4, near_zero_entries


def test_integrated_2d_recycling_one_step_native_parity_stays_within_exact_mixed_band() -> None:
    _assert_mixed_exact_band(
        case_name="integrated_2d_recycling_one_step",
        env_flag="JAX_DRB_RUN_RECYCLING_2D_PARITY",
        near_zero_atol=1.0e-6,
        max_relative=2.0e-2,
        max_near_zero_abs_diff=1.0e-8,
    )


def test_integrated_2d_recycling_short_window_native_parity_stays_within_exact_mixed_band() -> None:
    _assert_mixed_exact_band(
        case_name="integrated_2d_recycling_short_window",
        env_flag="JAX_DRB_RUN_RECYCLING_2D_PARITY",
        near_zero_atol=1.0e-6,
        max_relative=2.0e-2,
        max_near_zero_abs_diff=1.0e-8,
    )


def test_integrated_2d_recycling_medium_window_native_parity_stays_within_exact_mixed_band() -> None:
    _assert_mixed_exact_band(
        case_name="integrated_2d_recycling_medium_window",
        env_flag="JAX_DRB_RUN_RECYCLING_2D_PARITY",
        near_zero_atol=1.0e-6,
        max_relative=2.0e-2,
        max_near_zero_abs_diff=1.0e-8,
    )


def test_tokamak_recycling_dthe_one_step_native_parity_stays_within_exact_mixed_band() -> None:
    _assert_mixed_exact_band(
        case_name="tokamak_recycling_dthe_one_step",
        env_flag="JAX_DRB_RUN_RECYCLING_2D_PARITY",
        near_zero_atol=2.0e-5,
        max_relative=5.0e-2,
        max_near_zero_abs_diff=2.0e-5,
    )


def test_tokamak_recycling_dthe_two_output_window_native_parity_stays_within_operational_mixed_band() -> None:
    _assert_live_tokamak_recycling_dthe_two_output_window_operational_band()


def test_tokamak_recycling_dthe_drifts_two_output_window_native_parity_stays_within_operational_mixed_band() -> None:
    _assert_live_tokamak_recycling_dthe_drifts_two_output_window_operational_band()


def test_tokamak_recycling_dthene_one_step_native_parity_stays_within_operational_mixed_band() -> None:
    _assert_live_tokamak_recycling_dthene_one_step_operational_mixed_band()


def test_tokamak_recycling_dthene_three_output_window_native_parity_stays_within_operational_mixed_band() -> None:
    _assert_live_tokamak_recycling_dthene_three_output_window_operational_mixed_band()


def test_recycling_array_diff_report_localizes_worst_cell(tmp_path: Path) -> None:
    expected_path = _BASELINE_DIR / "recycling_dthe_one_step.npz"
    actual_path = tmp_path / "recycling_dthe_one_step.npz"
    payload = load_portable_array_payload(expected_path)
    payload["variables"] = {name: np.array(value, copy=True) for name, value in payload["variables"].items()}
    target_name = "NVhe+"
    if target_name not in payload["variables"]:
        target_name = next(iter(payload["variables"]))
    target = payload["variables"][target_name]
    flat_index = target.size // 2 if target.size else 0
    target.reshape(-1)[flat_index] += 1.25
    write_portable_array_payload(payload, actual_path)

    report = compare_recycling_artifacts(expected_path, actual_path, artifact_kind="arrays")

    assert not report.ok
    assert report.worst_variable == target_name
    assert report.worst_location == np.unravel_index(flat_index, target.shape) if target.shape else ()
    assert report.max_abs_diff == pytest.approx(1.25)

    text = format_recycling_diff_report(report)
    assert "arrays:" in text
    assert "worst_location:" in text


def _staged_rhs_differences(
    input_path: Path,
    stage_dir: Path,
    *,
    controller_species: tuple[str, ...],
    state_fields: tuple[str, ...],
    compare_fields: tuple[str, ...],
) -> dict[str, float]:
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    active = (slice(None), slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))

    snapshot = extract_recycling_controller_snapshot(
        stage_dir / "BOUT.dmp.0.nc",
        stage_dir / "BOUT.restart.0.nc",
        controller_species=controller_species,
    )
    field_overrides: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    with Dataset(stage_dir / "BOUT.dmp.0.nc") as dataset:
        for name in state_fields:
            if name in dataset.variables:
                field_overrides[name] = np.asarray(dataset.variables[name][-1], dtype=np.float64)
        for name in compare_fields:
            if name in dataset.variables:
                expected[name] = np.asarray(dataset.variables[name][-1:], dtype=np.float64)

    actual = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        field_overrides=field_overrides,
        feedback_integrals=snapshot.restart_integrals,
    ).variables

    return {
        name: float(np.nanmax(np.abs(np.asarray(actual[name], dtype=np.float64)[active] - expected[name][active])))
        for name in compare_fields
    }
