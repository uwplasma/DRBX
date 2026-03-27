from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native import runner as native_runner
from jax_drb.native.reference_dump import LocalReferenceSnapshot
from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.reference.cases import ReferenceCase

_REFERENCE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp")


@dataclass(frozen=True)
class _FakeSummary:
    artifacts: dict[str, str]


@dataclass(frozen=True)
class _FakeExecution:
    summary: _FakeSummary


def test_integrated_2d_recycling_rhs_uses_local_reference_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    netcdf4 = pytest.importorskip("netCDF4")
    dump_path = tmp_path / "BOUT.dmp.0.nc"
    with netcdf4.Dataset(dump_path, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 5)
        dataset.createDimension("z", 1)
        dataset.createDimension("t", 1)

        for name, value in {
            "MXG": 1,
            "MYG": 1,
            "jyseps1_1": 0,
            "jyseps2_1": 2,
            "jyseps1_2": 2,
            "jyseps2_2": 2,
            "ny_inner": 3,
            "PE_YIND": 0,
            "NYPE": 2,
        }.items():
            variable = dataset.createVariable(name, "i4")
            variable.assignValue(value)

        field2d = np.ones((4, 5), dtype=np.float64)
        for name in ("dx", "dy", "J", "g11", "g22", "g_22", "g33", "g23", "Bxy"):
            variable = dataset.createVariable(name, "f8", ("x", "y"))
            variable[:] = field2d

        state_fields = {
            "Nd+": 2.0 * field2d,
            "Pd+": 3.0 * field2d,
            "NVd+": np.zeros_like(field2d),
            "Nd": np.zeros_like(field2d),
            "Pd": np.zeros_like(field2d),
            "NVd": np.zeros_like(field2d),
            "Pe": 3.0 * field2d,
        }
        for name, value in state_fields.items():
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[:] = value.reshape(1, 4, 5, 1)

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": str(dump_path)})),
    )

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)", "ddt(Pe)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    result = native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert result.payload["dimensions"] == {"t": 1, "x": 4, "y": 5, "z": 1}
    assert "ddt(Nd+)" in result.payload["variable_summaries"]
    assert result.variables["Nd+"].shape == (1, 2, 3, 1)


def test_integrated_2d_recycling_rhs_requests_auxiliary_dump_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    captured: dict[str, tuple[str, ...]] = {}

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
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
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )

    def fake_load_snapshot(dump_path, *, field_names, optional_field_names=(), scalar_names=(), time_index=0):
        del dump_path, time_index
        captured["field_names"] = tuple(field_names)
        captured["optional_field_names"] = tuple(optional_field_names)
        captured["scalar_names"] = tuple(scalar_names)
        return LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={"is_pump": np.zeros((4, 5, 1), dtype=np.float64)},
            scalar_values={"Nnorm": 1.0e17},
        )

    monkeypatch.setattr(native_runner, "load_local_reference_snapshot", fake_load_snapshot)

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe", "ddt(Nd+)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["field_names"] == ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe")
    assert captured["optional_field_names"] == ("Ne", "Sd_target_recycle", "Ed_target_recycle", "is_pump")
    assert captured["scalar_names"] == ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")


def test_integrated_2d_recycling_rhs_preserves_dump_sheath_state(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _REFERENCE_INPUT.exists():
        pytest.skip("integrated 2D recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=4,
        ny=3,
        nz=1,
        mxg=1,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(4, dtype=jnp.float64),
        y=jnp.arange(5, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((4, 5, 1), dtype=jnp.float64)
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
    fields = {
        "Nd+": np.ones((4, 5, 1), dtype=np.float64),
        "Pd+": np.ones((4, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((4, 5, 1), dtype=np.float64),
        "Nd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pd": np.zeros((4, 5, 1), dtype=np.float64),
        "NVd": np.zeros((4, 5, 1), dtype=np.float64),
        "Pe": np.ones((4, 5, 1), dtype=np.float64),
    }

    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: _FakeExecution(summary=_FakeSummary(artifacts={"BOUT.dmp.0.nc": "/tmp/fake-dump.nc"})),
    )
    monkeypatch.setattr(
        native_runner,
        "load_local_reference_snapshot",
        lambda *args, **kwargs: LocalReferenceSnapshot(
            mesh=mesh,
            metrics=metrics,
            fields=fields,
            optional_fields={},
            scalar_values={"Nnorm": 1.0e17},
        ),
    )

    captured: dict[str, bool] = {}
    original = native_runner.compute_recycling_1d_rhs

    def wrapper(*args, **kwargs):
        captured["apply_sheath_boundaries"] = kwargs["apply_sheath_boundaries"]
        return original(*args, **kwargs)

    monkeypatch.setattr(native_runner, "compute_recycling_1d_rhs", wrapper)

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path=str(_REFERENCE_INPUT),
        parity_mode="one_rhs",
        rationale="test",
        compare_variables=("Nd+", "ddt(Nd+)"),
        trim_x_guards=True,
        trim_y_guards=True,
    )

    native_runner._run_integrated_2d_recycling_rhs_case(
        case,
        input_path=_REFERENCE_INPUT,
        reference_root=Path("/Users/rogerio/local/hermes-3"),
    )

    assert captured["apply_sheath_boundaries"] is False
