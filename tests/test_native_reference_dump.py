from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.native.reference_dump import load_local_reference_snapshot


def test_load_local_reference_snapshot_reads_mesh_metrics_and_fields(tmp_path: Path) -> None:
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
            "NYPE": 4,
            "Nnorm": 1.0e17,
        }.items():
            variable = dataset.createVariable(name, "f8" if name == "Nnorm" else "i4")
            variable.assignValue(value)

        field2d = np.arange(20, dtype=np.float64).reshape(4, 5)
        for name in ("dx", "dy", "J", "g11", "g22", "g_22", "g33", "g23", "Bxy"):
            variable = dataset.createVariable(name, "f8", ("x", "y"))
            variable[:] = field2d + (0.0 if name != "g_22" else 100.0)

        for name in ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"):
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[:] = np.arange(20, dtype=np.float64).reshape(1, 4, 5, 1)

        optional = dataset.createVariable("is_pump", "f8", ("x", "y"))
        optional[:] = np.eye(4, 5, dtype=np.float64)

    snapshot = load_local_reference_snapshot(
        dump_path,
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        optional_field_names=("is_pump", "missing_field"),
        scalar_names=("Nnorm", "missing_scalar"),
    )

    assert snapshot.mesh.nx == 4
    assert snapshot.mesh.ny == 3
    assert snapshot.mesh.local_ny == 5
    assert snapshot.mesh.mxg == 1
    assert snapshot.mesh.myg == 1
    assert snapshot.mesh.has_lower_y_target is True
    assert snapshot.mesh.has_upper_y_target is False
    assert snapshot.metrics.dx.shape == (4, 5, 1)
    assert snapshot.metrics.g_22.shape == (4, 5, 1)
    np.testing.assert_allclose(np.asarray(snapshot.metrics.dx)[..., 0], field2d)
    np.testing.assert_allclose(np.asarray(snapshot.fields["Nd+"])[..., 0], np.arange(20, dtype=np.float64).reshape(4, 5))
    np.testing.assert_allclose(np.asarray(snapshot.optional_fields["is_pump"])[..., 0], np.eye(4, 5, dtype=np.float64))
    assert "missing_field" not in snapshot.optional_fields
    assert snapshot.scalar_values["Nnorm"] == pytest.approx(1.0e17)
    assert "missing_scalar" not in snapshot.scalar_values
