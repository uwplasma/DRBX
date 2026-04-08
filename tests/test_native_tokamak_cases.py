from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native import runner as native_runner
from jax_drb.native import run_curated_case
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.reference_dump import LocalReferenceSnapshot
from jax_drb.native.reference_dump import save_local_reference_snapshot_cache, save_optional_field_history_cache
from jax_drb.parity.arrays import build_array_payload_from_summary_payload, compare_array_payloads, load_portable_array_payload
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.reference.cases import ReferenceCase

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-flow-evolveT/BOUT.inp")
_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference")
_ARRAY_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]
    time_points: tuple[float, ...]
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def _tokamak_snapshot(*, time_index: int) -> LocalReferenceSnapshot:
    mesh = StructuredMesh(
        nx=6,
        ny=8,
        nz=1,
        mxg=2,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=-1,
        jyseps2_1=7,
        jyseps1_2=7,
        jyseps2_2=7,
        ny_inner=8,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(6, dtype=jnp.float64),
        y=jnp.arange(12, dtype=jnp.float64) - 2.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((6, 12, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    scale = float(time_index + 1)
    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={
            "Nh": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64),
            "Ph": np.full((6, 12, 1), 2.0 * scale, dtype=np.float64),
            "NVh": np.full((6, 12, 1), 3.0 * scale, dtype=np.float64),
        },
        optional_fields={},
        scalar_values={"Nnorm": 1.0e19, "Tnorm": 10.0, "Bnorm": 1.0, "Cs0": 2.0, "Omega_ci": 3.0, "rho_s0": 4.0},
    )


def test_tokamak_diffusion_flow_one_step_stacks_initial_and_final_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("tokamak diffusion-flow reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion-flow.nc"},
                time_points=(0.0, 50.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        return _tokamak_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_flow_one_step",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nh", "Ph", "NVh"),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_flow_one_step_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 50.0)
    assert result.variables["Nh"].shape == (2, 2, 8, 1)
    assert result.variables["Ph"].shape == (2, 2, 8, 1)
    assert result.variables["NVh"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh"][1], 2.0)
    np.testing.assert_allclose(result.variables["Ph"][0], 2.0)
    np.testing.assert_allclose(result.variables["Ph"][1], 4.0)
    np.testing.assert_allclose(result.variables["NVh"][0], 3.0)
    np.testing.assert_allclose(result.variables["NVh"][1], 6.0)
    assert result.payload["compare_variables"] == ["Nh", "Ph", "NVh"]


def test_tokamak_diffusion_one_step_stacks_initial_and_final_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion/BOUT.inp")
    if not diffusion_input.exists():
        pytest.skip("tokamak diffusion reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion.nc"},
                time_points=(0.0, 50.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields={"Nh": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64)},
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/diffusion/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nh",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_one_step_case(
        case,
        input_path=diffusion_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 50.0)
    assert result.variables["Nh"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh"][1], 2.0)
    assert result.payload["compare_variables"] == ["Nh"]


def test_tokamak_diffusion_transport_one_step_stacks_initial_and_final_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-transport/BOUT.inp")
    if not transport_input.exists():
        pytest.skip("tokamak diffusion-transport reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion-transport.nc"},
                time_points=(0.0, 50.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        fields = dict(snapshot.fields)
        scale = float(time_index + 1)
        fields = {
            "Nh+": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64),
            "Ph+": np.full((6, 12, 1), 2.0 * scale, dtype=np.float64),
            "NVh+": np.full((6, 12, 1), 3.0 * scale, dtype=np.float64),
            "Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64),
        }
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_transport_one_step",
        stage="stage7",
        reference_path=str(transport_input),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nh+", "Ph+", "NVh+", "Pe"),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_transport_one_step_case(
        case,
        input_path=transport_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 50.0)
    assert result.variables["Nh+"].shape == (2, 2, 8, 1)
    assert result.variables["Ph+"].shape == (2, 2, 8, 1)
    assert result.variables["NVh+"].shape == (2, 2, 8, 1)
    assert result.variables["Pe"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh+"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh+"][1], 2.0)
    np.testing.assert_allclose(result.variables["Ph+"][0], 2.0)
    np.testing.assert_allclose(result.variables["Ph+"][1], 4.0)
    np.testing.assert_allclose(result.variables["NVh+"][0], 3.0)
    np.testing.assert_allclose(result.variables["NVh+"][1], 6.0)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][1], 8.0)
    assert result.payload["compare_variables"] == ["Nh+", "Ph+", "NVh+", "Pe"]


def test_tokamak_diffusion_transport_one_step_matches_committed_baselines() -> None:
    transport_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-transport/BOUT.inp")
    if not transport_input.exists():
        pytest.skip("tokamak diffusion-transport reference input is unavailable")

    expected_summary = load_summary_json(_BASELINE_DIR / "tokamak_diffusion_transport_one_step.json")
    expected_arrays = load_portable_array_payload(_ARRAY_BASELINE_DIR / "tokamak_diffusion_transport_one_step.npz")

    result = run_curated_case("tokamak_diffusion_transport_one_step", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues

def test_tokamak_diffusion_one_step_matches_committed_baselines() -> None:
    diffusion_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion/BOUT.inp")
    if not diffusion_input.exists():
        pytest.skip("tokamak diffusion reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_diffusion_one_step.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_diffusion_one_step.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak diffusion one-step baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_diffusion_one_step", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_diffusion_transport_short_window_stacks_full_snapshot_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-transport/BOUT.inp")
    if not transport_input.exists():
        pytest.skip("tokamak diffusion-transport reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion-transport.nc"},
                time_points=(0.0, 50.0, 100.0, 150.0, 200.0, 250.0),
                overrides=("nout=5",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        fields = {
            "Nh+": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64),
            "Ph+": np.full((6, 12, 1), 2.0 * scale, dtype=np.float64),
            "NVh+": np.full((6, 12, 1), 3.0 * scale, dtype=np.float64),
            "Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64),
        }
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_transport_short_window",
        stage="stage7",
        reference_path="examples/tokamak-2D/diffusion-transport/BOUT.inp",
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Nh+", "Ph+", "NVh+", "Pe"),
        extra_overrides=("nout=5",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_transport_short_window_case(
        case,
        input_path=transport_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1, 2, 3, 4, 5]
    assert result.time_points == (0.0, 50.0, 100.0, 150.0, 200.0, 250.0)
    assert result.variables["Nh+"].shape == (6, 2, 8, 1)
    assert result.variables["Ph+"].shape == (6, 2, 8, 1)
    assert result.variables["NVh+"].shape == (6, 2, 8, 1)
    assert result.variables["Pe"].shape == (6, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh+"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh+"][5], 6.0)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][5], 24.0)
    assert result.payload["compare_variables"] == ["Nh+", "Ph+", "NVh+", "Pe"]


def test_tokamak_diffusion_transport_short_window_matches_committed_baselines() -> None:
    transport_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-transport/BOUT.inp")
    if not transport_input.exists():
        pytest.skip("tokamak diffusion-transport reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_diffusion_transport_short_window.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_diffusion_transport_short_window.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak diffusion-transport short-window baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_diffusion_transport_short_window", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_heat_transport_one_step_stacks_initial_and_final_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heat_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/heat-transport/BOUT.inp")
    if not heat_input.exists():
        pytest.skip("tokamak heat-transport reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-heat-transport.nc"},
                time_points=(0.0, 4000.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        fields = {"Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64)}
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_heat_transport_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/heat-transport/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Pe",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_heat_transport_one_step_case(
        case,
        input_path=heat_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 4000.0)
    assert result.variables["Pe"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][1], 8.0)
    assert result.payload["compare_variables"] == ["Pe"]


def test_tokamak_heat_transport_one_step_matches_committed_baselines() -> None:
    heat_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/heat-transport/BOUT.inp")
    if not heat_input.exists():
        pytest.skip("tokamak heat-transport reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_heat_transport_one_step.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_heat_transport_one_step.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak heat-transport one-step baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_heat_transport_one_step", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_heat_transport_short_window_stacks_full_snapshot_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heat_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/heat-transport/BOUT.inp")
    if not heat_input.exists():
        pytest.skip("tokamak heat-transport reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-heat-transport.nc"},
                time_points=(0.0, 4000.0, 8000.0),
                overrides=("nout=2",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        fields = {"Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64)}
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_heat_transport_short_window",
        stage="stage7",
        reference_path="examples/tokamak-2D/heat-transport/BOUT.inp",
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Pe",),
        extra_overrides=("nout=2",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_heat_transport_short_window_case(
        case,
        input_path=heat_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1, 2]
    assert result.time_points == (0.0, 4000.0, 8000.0)
    assert result.variables["Pe"].shape == (3, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][2], 12.0)
    assert result.payload["compare_variables"] == ["Pe"]


def test_tokamak_heat_transport_short_window_matches_committed_baselines() -> None:
    heat_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/heat-transport/BOUT.inp")
    if not heat_input.exists():
        pytest.skip("tokamak heat-transport reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_heat_transport_short_window.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_heat_transport_short_window.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak heat-transport short-window baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_heat_transport_short_window", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_heat_transport_short_window_uses_committed_snapshot_and_history_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    heat_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/heat-transport/BOUT.inp")
    if not heat_input.exists():
        pytest.skip("tokamak heat-transport reference input is unavailable")

    snapshot = _tokamak_snapshot(time_index=0)
    snapshot_cache = tmp_path / "tokamak_heat_transport_short_window_snapshot.npz"
    save_local_reference_snapshot_cache(
        LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields={},
            optional_fields={},
            scalar_values=snapshot.scalar_values,
        ),
        snapshot_cache,
    )
    history_cache = tmp_path / "tokamak_heat_transport_short_window_field_history.npz"
    save_optional_field_history_cache(
        {
            "Pe": np.stack(
                [
                    np.full((6, 12, 1), 4.0, dtype=np.float64),
                    np.full((6, 12, 1), 5.0, dtype=np.float64),
                    np.full((6, 12, 1), 6.0, dtype=np.float64),
                ],
                axis=0,
            )
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_tokamak_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_heat_transport_short_window" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_tokamak_field_history_cache_path",
        lambda case_name: history_cache if case_name == "tokamak_heat_transport_short_window" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when tokamak caches are present")),
    )

    case = ReferenceCase(
        name="tokamak_heat_transport_short_window",
        stage="stage7",
        reference_path="examples/tokamak-2D/heat-transport/BOUT.inp",
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Pe",),
        extra_overrides=("nout=2", "e:diagnose=false"),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_heat_transport_short_window_case(
        case,
        input_path=heat_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert result.time_points == (0.0, 4000.0, 8000.0)
    assert result.payload["overrides"] == ["nout=2", "e:diagnose=false"]
    assert np.asarray(result.variables["Pe"]).shape == (3, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][2], 6.0)


def test_tokamak_diffusion_conduction_one_step_stacks_initial_and_final_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conduction_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-conduction/BOUT.inp")
    if not conduction_input.exists():
        pytest.skip("tokamak diffusion-conduction reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-diffusion-conduction.nc"},
                time_points=(0.0, 50.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        fields = {
            "Nh+": np.full((6, 12, 1), 1.0 * scale, dtype=np.float64),
            "Ph+": np.full((6, 12, 1), 2.0 * scale, dtype=np.float64),
            "Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64),
        }
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_diffusion_conduction_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/diffusion-conduction/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Nh+", "Ph+", "Pe"),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_diffusion_conduction_one_step_case(
        case,
        input_path=conduction_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 50.0)
    assert result.variables["Nh+"].shape == (2, 2, 8, 1)
    assert result.variables["Ph+"].shape == (2, 2, 8, 1)
    assert result.variables["Pe"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Nh+"][0], 1.0)
    np.testing.assert_allclose(result.variables["Nh+"][1], 2.0)
    np.testing.assert_allclose(result.variables["Ph+"][0], 2.0)
    np.testing.assert_allclose(result.variables["Ph+"][1], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][1], 8.0)
    assert result.payload["compare_variables"] == ["Nh+", "Ph+", "Pe"]


def test_tokamak_diffusion_conduction_one_step_matches_committed_baselines() -> None:
    conduction_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/diffusion-conduction/BOUT.inp")
    if not conduction_input.exists():
        pytest.skip("tokamak diffusion-conduction reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_diffusion_conduction_one_step.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_diffusion_conduction_one_step.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak diffusion-conduction one-step baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_diffusion_conduction_one_step", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_linear_transport_one_step_stacks_initial_and_final_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/linear-transport/BOUT.inp")
    if not linear_input.exists():
        pytest.skip("tokamak linear-transport reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-tokamak-linear-transport.nc"},
                time_points=(0.0, 10000.0),
                overrides=("nout=1",),
            )
        ),
    )

    def fake_load_snapshot(*args, **kwargs):
        time_index = kwargs["time_index"]
        captured_time_indices.append(time_index)
        snapshot = _tokamak_snapshot(time_index=time_index)
        scale = float(time_index + 1)
        fields = {"Pe": np.full((6, 12, 1), 4.0 * scale, dtype=np.float64)}
        return LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields=fields,
            optional_fields=snapshot.optional_fields,
            scalar_values=snapshot.scalar_values,
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="tokamak_linear_transport_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/linear-transport/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Pe",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_linear_transport_one_step_case(
        case,
        input_path=linear_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 10000.0)
    assert result.variables["Pe"].shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Pe"][0], 4.0)
    np.testing.assert_allclose(result.variables["Pe"][1], 8.0)
    assert result.payload["compare_variables"] == ["Pe"]


def test_tokamak_linear_transport_one_step_matches_committed_baselines() -> None:
    linear_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/linear-transport/BOUT.inp")
    if not linear_input.exists():
        pytest.skip("tokamak linear-transport reference input is unavailable")
    summary_path = _BASELINE_DIR / "tokamak_linear_transport_one_step.json"
    arrays_path = _ARRAY_BASELINE_DIR / "tokamak_linear_transport_one_step.npz"
    if not summary_path.exists() or not arrays_path.exists():
        pytest.skip("tokamak linear-transport one-step baselines are unavailable")

    expected_summary = load_summary_json(summary_path)
    expected_arrays = load_portable_array_payload(arrays_path)

    result = run_curated_case("tokamak_linear_transport_one_step", reference_root=_REFERENCE_ROOT)
    summary_comparison = compare_summary_payloads(expected_summary, result.payload, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)
    actual_arrays = build_array_payload_from_summary_payload(result.payload, result.variables)
    array_comparison = compare_array_payloads(expected_arrays, actual_arrays, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert summary_comparison.ok, summary_comparison.issues
    assert array_comparison.ok, array_comparison.issues


def test_tokamak_linear_transport_one_step_uses_committed_snapshot_and_history_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    linear_input = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/linear-transport/BOUT.inp")
    if not linear_input.exists():
        pytest.skip("tokamak linear-transport reference input is unavailable")

    snapshot = _tokamak_snapshot(time_index=0)
    snapshot_cache = tmp_path / "tokamak_linear_transport_one_step_snapshot.npz"
    save_local_reference_snapshot_cache(
        LocalReferenceSnapshot(
            mesh=snapshot.mesh,
            metrics=snapshot.metrics,
            fields={},
            optional_fields={},
            scalar_values=snapshot.scalar_values,
        ),
        snapshot_cache,
    )
    history_cache = tmp_path / "tokamak_linear_transport_one_step_field_history.npz"
    save_optional_field_history_cache(
        {
            "Pe": np.stack(
                [
                    np.full((6, 12, 1), 7.0, dtype=np.float64),
                    np.full((6, 12, 1), 9.0, dtype=np.float64),
                ],
                axis=0,
            )
        },
        history_cache,
    )

    monkeypatch.setattr(
        native_runner,
        "_tokamak_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "tokamak_linear_transport_one_step" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "_tokamak_field_history_cache_path",
        lambda case_name: history_cache if case_name == "tokamak_linear_transport_one_step" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference run should not be used when tokamak caches are present")
        ),
    )

    case = ReferenceCase(
        name="tokamak_linear_transport_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/linear-transport/BOUT.inp",
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Pe",),
        extra_overrides=("e:diagnose=false",),
        trim_x_guards=True,
        trim_y_guards=True,
        process_count=6,
    )

    result = native_runner._run_tokamak_linear_transport_one_step_case(
        case,
        input_path=linear_input,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert result.time_points == (0.0, 10000.0)
    assert result.payload["overrides"] == ["nout=1", "e:diagnose=false"]
    assert np.asarray(result.variables["Pe"]).shape == (2, 2, 8, 1)
    np.testing.assert_allclose(result.variables["Pe"][0], 7.0)
    np.testing.assert_allclose(result.variables["Pe"][1], 9.0)
