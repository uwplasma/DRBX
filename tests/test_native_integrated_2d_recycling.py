from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from jax_drb.native import runner as native_runner
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
