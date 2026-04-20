from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native import runner as native_runner
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.reference_dump import LocalReferenceSnapshot
from jax_drb.reference.cases import ReferenceCase

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]
    time_points: tuple[float, ...]


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def _alfven_snapshot(*, time_index: int) -> LocalReferenceSnapshot:
    mesh = StructuredMesh(
        nx=5,
        ny=32,
        nz=27,
        mxg=2,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=True,
        jyseps1_1=-1,
        jyseps2_1=31,
        jyseps1_2=31,
        jyseps2_2=31,
        ny_inner=32,
        has_lower_y_target=False,
        has_upper_y_target=False,
        x=jnp.arange(5, dtype=jnp.float64),
        y=jnp.arange(36, dtype=jnp.float64) - 2.0,
        z=jnp.arange(27, dtype=jnp.float64),
    )
    ones = jnp.ones((5, 36, 27), dtype=jnp.float64)
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
    fields = {
        "Apar": np.full((5, 36, 27), 1.0e-3 * scale, dtype=np.float64),
        "phi": np.full((5, 36, 27), 3.0 * scale, dtype=np.float64),
        "Vort": np.full((5, 36, 27), 4.0 * scale, dtype=np.float64),
        "NVe": np.full((5, 36, 27), -(2.0 / 1836.0) * scale, dtype=np.float64),
        "Ne": np.ones((5, 36, 27), dtype=np.float64),
        "Ni": np.ones((5, 36, 27), dtype=np.float64),
    }
    optional_fields = {
        "ddt(NVe)": np.full((5, 36, 27), 6.0 * scale, dtype=np.float64),
        "ddt(Vort)": np.full((5, 36, 27), 7.0 * scale, dtype=np.float64),
    }
    return LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=fields,
        optional_fields=optional_fields,
        scalar_values={"Nnorm": 1.0e19, "Tnorm": 100.0, "Bnorm": 0.2, "Cs0": 1.0, "Omega_ci": 2.0, "rho_s0": 0.5},
    )


def test_alfven_wave_rhs_uses_dump_backed_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("alfven-wave reference input is unavailable")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-alfven.nc"}, time_points=(0.0,))
        ),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        captured["field_names"] = tuple(field_names)
        captured["optional_field_names"] = tuple(optional_field_names)
        captured["scalar_names"] = tuple(scalar_names)
        captured["time_index"] = time_index
        return _alfven_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)
    monkeypatch.setattr(
        native_runner,
        "solve_slab_neumann_apar",
        lambda *args, **kwargs: np.full((5, 36, 27), 9.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        native_runner,
        "invert_slab_neumann_apar_to_current_density",
        lambda *args, **kwargs: np.full((5, 36, 27), 2.0, dtype=np.float64),
    )

    case = ReferenceCase(
        name="alfven_wave_rhs",
        stage="stage8",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Apar", "Ajpar", "phi", "Vort", "NVe", "ddt(NVe)", "ddt(Vort)"),
    )

    result = native_runner._run_alfven_wave_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["field_names"] == ("Apar", "phi", "Vort", "NVe", "Ne", "Ni")
    assert captured["optional_field_names"] == ("ddt(NVe)", "ddt(Vort)")
    assert captured["time_index"] == 0
    assert result.payload["dimensions"] == {"t": 1, "x": 5, "y": 36, "z": 27}
    assert result.variables["ddt(NVe)"].shape == (1, 5, 36, 27)
    assert float(result.variables["Apar"][0, 0, 0, 0]) == 9.0
    assert float(result.variables["Ajpar"][0, 0, 0, 0]) == 0.0
    assert float(result.variables["Ajpar"][0, 2, 0, 0]) == pytest.approx(2.0)
    assert float(result.variables["ddt(NVe)"][0, 2, 0, 0]) == 6.0
    assert float(result.variables["ddt(NVe)"][0, 2, 2, 0]) == 0.0
    assert float(result.variables["ddt(Vort)"][0, 1, 0, 0]) == 7.0
    assert float(result.variables["ddt(Vort)"][0, 1, 2, 0]) != 7.0


def test_alfven_wave_one_step_stacks_initial_and_final_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("alfven-wave reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-alfven.nc"}, time_points=(0.0, 10.0))
        ),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        captured_time_indices.append(time_index)
        return _alfven_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)
    monkeypatch.setattr(
        native_runner,
        "solve_slab_neumann_apar",
        lambda current_density, **kwargs: np.full_like(np.asarray(current_density, dtype=np.float64), np.max(np.abs(current_density))),
    )
    monkeypatch.setattr(
        native_runner,
        "invert_slab_neumann_apar_to_current_density",
        lambda apar, **kwargs: np.full_like(np.asarray(apar, dtype=np.float64), 8.0),
    )

    case = ReferenceCase(
        name="alfven_wave_one_step",
        stage="stage8",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_step",
        rationale="test",
        compare_variables=("Apar", "Ajpar", "phi", "Vort", "NVe"),
    )

    result = native_runner._run_alfven_wave_one_step_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1]
    assert result.time_points == (0.0, 10.0)
    assert result.variables["Apar"].shape == (2, 5, 36, 27)
    assert float(result.variables["Apar"][0, 0, 0, 0]) == 2.0
    assert float(result.variables["Apar"][1, 0, 0, 0]) == 4.0
    assert float(result.variables["Ajpar"][0, 0, 0, 0]) == 0.0
    assert float(result.variables["Ajpar"][1, 0, 0, 0]) == 0.0
    assert float(result.variables["NVe"][0, 2, 0, 0]) == pytest.approx(-(2.0 / 1836.0))
    assert float(result.variables["NVe"][1, 2, 0, 0]) == pytest.approx(-(4.0 / 1836.0))
    assert float(result.variables["NVe"][0, 2, 2, 0]) == pytest.approx(-8.0 / 1836.0)
    assert float(result.variables["NVe"][1, 2, 2, 0]) == pytest.approx(-8.0 / 1836.0)


def test_alfven_wave_short_window_uses_all_reference_time_points(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("alfven-wave reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-alfven.nc"},
                time_points=(0.0, 10.0, 20.0, 30.0),
            )
        ),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        captured_time_indices.append(time_index)
        return _alfven_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)
    monkeypatch.setattr(
        native_runner,
        "solve_slab_neumann_apar",
        lambda current_density, **kwargs: np.full_like(
            np.asarray(current_density, dtype=np.float64),
            np.max(np.abs(current_density)),
        ),
    )
    monkeypatch.setattr(
        native_runner,
        "invert_slab_neumann_apar_to_current_density",
        lambda apar, **kwargs: np.full_like(np.asarray(apar, dtype=np.float64), 8.0),
    )

    case = ReferenceCase(
        name="alfven_wave_short_window",
        stage="stage8",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Apar", "Ajpar", "phi", "Vort", "NVe"),
        extra_overrides=("nout=20",),
    )

    result = native_runner._run_alfven_wave_short_window_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1, 2, 3]
    assert result.time_points == (0.0, 10.0, 20.0, 30.0)
    assert result.variables["Apar"].shape == (4, 5, 36, 27)


def test_alfven_wave_medium_window_uses_all_reference_time_points(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("alfven-wave reference input is unavailable")

    captured_time_indices: list[int] = []

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(
            summary=_FakeSummary(
                artifacts={"BOUT.dmp.0.nc": "/tmp/fake-alfven.nc"},
                time_points=(0.0, 10.0, 20.0),
            )
        ),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        captured_time_indices.append(time_index)
        return _alfven_snapshot(time_index=time_index)

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)
    monkeypatch.setattr(
        native_runner,
        "solve_slab_neumann_apar",
        lambda current_density, **kwargs: np.full_like(
            np.asarray(current_density, dtype=np.float64),
            np.max(np.abs(current_density)),
        ),
    )
    monkeypatch.setattr(
        native_runner,
        "invert_slab_neumann_apar_to_current_density",
        lambda apar, **kwargs: np.full_like(np.asarray(apar, dtype=np.float64), 8.0),
    )

    case = ReferenceCase(
        name="alfven_wave_medium_window",
        stage="stage8",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="short_window",
        rationale="test",
        compare_variables=("Apar", "Ajpar", "phi", "Vort", "NVe"),
    )

    result = native_runner._run_alfven_wave_medium_window_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured_time_indices == [0, 1, 2]
    assert result.time_points == (0.0, 10.0, 20.0)
    assert result.variables["Apar"].shape == (3, 5, 36, 27)
