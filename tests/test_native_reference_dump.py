from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native.mesh import StructuredMesh
from jax_drb.native.metrics import StructuredMetrics
from jax_drb.native.reference_dump import (
    LocalReferenceSnapshot,
    load_local_reference_snapshot,
    load_local_reference_snapshot_cache,
    load_optional_field_history_cache,
    save_local_reference_snapshot_cache,
    save_optional_field_history_cache,
    synthesize_local_reference_snapshot_from_active_history,
)


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
        for name in ("dx", "dy", "J", "g11", "g22", "g_22", "g33", "g23", "g_23", "Bxy"):
            variable = dataset.createVariable(name, "f8", ("x", "y"))
            variable[:] = field2d + (0.0 if name != "g_22" else 100.0)

        for name in ("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"):
            variable = dataset.createVariable(name, "f8", ("t", "x", "y", "z"))
            variable[:] = np.arange(20, dtype=np.float64).reshape(1, 4, 5, 1)

        optional = dataset.createVariable("is_pump", "f8", ("x", "y"))
        optional[:] = np.eye(4, 5, dtype=np.float64)
        anomalous = dataset.createVariable("anomalous_nu_e", "f8", ("t", "x", "y"))
        anomalous[:] = np.arange(20, dtype=np.float64).reshape(1, 4, 5) + 100.0

    snapshot = load_local_reference_snapshot(
        dump_path,
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        optional_field_names=("is_pump", "anomalous_nu_e", "missing_field"),
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
    assert snapshot.metrics.g_23 is not None
    np.testing.assert_allclose(np.asarray(snapshot.metrics.dx)[..., 0], field2d)
    np.testing.assert_allclose(np.asarray(snapshot.fields["Nd+"])[..., 0], np.arange(20, dtype=np.float64).reshape(4, 5))
    np.testing.assert_allclose(np.asarray(snapshot.optional_fields["is_pump"])[..., 0], np.eye(4, 5, dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(snapshot.optional_fields["anomalous_nu_e"])[..., 0],
        np.arange(20, dtype=np.float64).reshape(4, 5) + 100.0,
    )
    assert "missing_field" not in snapshot.optional_fields
    assert snapshot.scalar_values["Nnorm"] == pytest.approx(1.0e17)
    assert "missing_scalar" not in snapshot.scalar_values


def test_local_reference_snapshot_cache_roundtrip(tmp_path: Path) -> None:
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
        dy=ones * 2.0,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones * 3.0,
        g_23=ones * 4.0,
    )
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={"Nd+": np.full((4, 5, 1), 2.0, dtype=np.float64)},
        optional_fields={"SNd+": np.full((4, 5, 1), 4.0, dtype=np.float64)},
        scalar_values={"Nnorm": 1.0e17},
    )
    cache_path = tmp_path / "snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, cache_path)

    loaded = load_local_reference_snapshot_cache(
        cache_path,
        field_names=("Nd+",),
        optional_field_names=("SNd+",),
        scalar_names=("Nnorm",),
    )

    assert loaded.mesh.nx == 4
    assert loaded.mesh.local_ny == 5
    np.testing.assert_allclose(np.asarray(loaded.metrics.dy), np.asarray(metrics.dy))
    np.testing.assert_allclose(np.asarray(loaded.metrics.g_23), np.asarray(metrics.g_23))
    np.testing.assert_allclose(loaded.fields["Nd+"], snapshot.fields["Nd+"])
    np.testing.assert_allclose(loaded.optional_fields["SNd+"], snapshot.optional_fields["SNd+"])
    assert loaded.scalar_values["Nnorm"] == pytest.approx(1.0e17)


def test_optional_field_history_cache_roundtrip(tmp_path: Path) -> None:
    history = {
        "Vd+": np.arange(8, dtype=np.float64).reshape(2, 4, 1, 1),
        "Sd_target_recycle": np.arange(8, dtype=np.float64).reshape(2, 4, 1, 1) + 10.0,
    }
    cache_path = tmp_path / "history.npz"
    save_optional_field_history_cache(history, cache_path)

    loaded = load_optional_field_history_cache(
        cache_path,
        field_names=("Vd+", "Sd_target_recycle", "missing"),
    )

    assert tuple(loaded) == ("Vd+", "Sd_target_recycle")
    np.testing.assert_allclose(loaded["Vd+"], history["Vd+"])
    np.testing.assert_allclose(loaded["Sd_target_recycle"], history["Sd_target_recycle"])


def test_synthesize_local_reference_snapshot_from_active_history(tmp_path: Path) -> None:
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
        g_23=ones * 0.5,
    )
    initial = np.zeros((4, 5, 1), dtype=np.float64)
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields={"Pe": initial.copy()},
        optional_fields={"Vd+": initial.copy()},
        scalar_values={},
    )
    arrays_path = tmp_path / "case.npz"
    np.savez_compressed(
        arrays_path,
        __metadata__="{}",
        var__Pe=np.stack(
            [
                np.zeros((mesh.nx - 2 * mesh.mxg, mesh.local_ny - 2 * mesh.myg, 1), dtype=np.float64),
                np.full((mesh.nx - 2 * mesh.mxg, mesh.local_ny - 2 * mesh.myg, 1), 2.0, dtype=np.float64),
            ],
            axis=0,
        ),
    )
    history_path = tmp_path / "history.npz"
    np.savez_compressed(
        history_path,
        **{"Vd+": np.stack([np.zeros((2, 3, 1), dtype=np.float64), np.ones((2, 3, 1), dtype=np.float64)], axis=0)}
    )

    synthesized = synthesize_local_reference_snapshot_from_active_history(
        initial_snapshot=snapshot,
        array_history_path=arrays_path,
        optional_history_path=history_path,
        timestep=0.5,
        state_field_names=("Pe",),
        rhs_field_names=("ddt(Pe)",),
        optional_field_names=("Vd+",),
    )

    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    np.testing.assert_allclose(synthesized.fields["Pe"][active], 2.0)
    np.testing.assert_allclose(synthesized.fields["ddt(Pe)"][active], 4.0)
    np.testing.assert_allclose(synthesized.optional_fields["Vd+"][active], 1.0)
