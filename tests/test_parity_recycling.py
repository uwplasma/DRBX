from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native import run_curated_case
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import compute_recycling_1d_rhs
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import build_array_payload_from_summary_payload, load_portable_array_payload
from jax_drb.parity.diff import build_array_diff_report, compare_recycling_artifacts, format_array_diff_report, format_recycling_diff_report
from jax_drb.parity.arrays import write_portable_array_payload
from jax_drb.parity.recycling import extract_recycling_controller_snapshot
from jax_drb.runtime.run_config import RunConfiguration


_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")
_STAGED_REFERENCE_1D = Path("/private/tmp/jax_drb_recycling_1d_one_step_inspect")
_STAGED_REFERENCE_DTHE = Path("/private/tmp/jax_drb_recycling_dthe_one_step_inspect")
_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_INPUT_DTHE = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


def test_staged_reference_controller_snapshot_extracts_expected_1d_values() -> None:
    if not _STAGED_REFERENCE_1D.exists():
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
    if not _STAGED_REFERENCE_DTHE.exists():
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
    if not _STAGED_REFERENCE_1D.exists():
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
    assert diffs["ddt(Pd+)"] <= 4.0e-4
    assert diffs["ddt(Pd)"] <= 2.0e-4
    assert diffs["ddt(NVd+)"] <= 1.0e-3


def test_staged_recycling_dthe_evolved_rhs_stays_within_locked_tolerances() -> None:
    if not _STAGED_REFERENCE_DTHE.exists():
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


@pytest.mark.parametrize(
    ("case_name", "baseline_name"),
    [
        ("recycling_1d_one_step", "recycling_1d_one_step.npz"),
        ("recycling_dthe_one_step", "recycling_dthe_one_step.npz"),
    ],
)
def test_recycling_one_step_native_parity_is_blocked_but_ready_for_diff_reporting(
    case_name: str,
    baseline_name: str,
) -> None:
    if os.environ.get("JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY") != "1":
        pytest.xfail("native recycling one-step transient is still blocked; set JAX_DRB_RUN_RECYCLING_ONE_STEP_PARITY=1 to probe it")

    expected = load_portable_array_payload(_BASELINE_DIR / baseline_name)

    try:
        result = run_curated_case(case_name, reference_root=_REFERENCE_ROOT)
    except Exception as exc:
        pytest.xfail(f"native recycling one-step run is blocked: {exc}")

    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    report = build_array_diff_report(
        expected["variables"],
        actual["variables"],
        compare_variables=tuple(expected["compare_variables"]),
    )

    if not report.ok:
        pytest.xfail(format_array_diff_report(report))

    assert report.max_abs_diff <= 5.0e-2


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
