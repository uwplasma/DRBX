from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation.diverted_tokamak_movie import (
    assemble_tokamak_rank_history,
    create_diverted_tokamak_movie_package_from_arrays,
    load_diverted_tokamak_arrays_npz,
    load_diverted_tokamak_geometry,
    toroidal_mean_fluctuation,
    write_diverted_tokamak_arrays_npz,
)


def _write_tokamak_mesh(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 6)
        dataset.createDimension("y", 6)
        rxy = dataset.createVariable("Rxy", "f8", ("x", "y"))
        zxy = dataset.createVariable("Zxy", "f8", ("x", "y"))
        psixy = dataset.createVariable("psixy", "f8", ("x", "y"))
        xx = np.linspace(1.0, 2.0, 6)[:, None]
        yy = np.linspace(-0.5, 0.5, 6)[None, :]
        rxy[:] = xx + 0.01 * yy
        zxy[:] = yy + 0.02 * xx
        psixy[:] = xx - 1.5


def _write_dump(path: Path, *, pe_yind: int) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 6)
        dataset.createDimension("y", 5)
        dataset.createDimension("z", 2)
        dataset.createDimension("t", 2)
        for name, value in {
            "MYPE": pe_yind,
            "PE_XIND": 0,
            "PE_YIND": pe_yind,
            "NXPE": 1,
            "NYPE": 2,
            "MXSUB": 4,
            "MYSUB": 3,
            "MXG": 1,
            "MYG": 1,
        }.items():
            variable = dataset.createVariable(name, "i4")
            variable.assignValue(value)
        t_array = dataset.createVariable("t_array", "f8", ("t",))
        t_array[:] = np.array([0.0, 0.1], dtype=np.float64)
        field = dataset.createVariable("Nd+", "f8", ("t", "x", "y", "z"))
        values = np.zeros((2, 6, 5, 2), dtype=np.float64)
        for time_index in range(2):
            for x_index in range(4):
                for y_index in range(3):
                    values[time_index, 1 + x_index, 1 + y_index, :] = (
                        100.0 * time_index + 10.0 * pe_yind + 2.0 * x_index + y_index
                    )
        field[:] = values


def test_assemble_tokamak_rank_history_stitches_global_y_domain(tmp_path: Path) -> None:
    _write_dump(tmp_path / "BOUT.dmp.0.nc", pe_yind=0)
    _write_dump(tmp_path / "BOUT.dmp.1.nc", pe_yind=1)

    history = assemble_tokamak_rank_history(tmp_path, field_name="Nd+")

    assert history.field_name == "Nd+"
    np.testing.assert_allclose(history.time_points, np.array([0.0, 0.1]))
    assert history.history_4d.shape == (2, 4, 6, 2)
    np.testing.assert_allclose(history.history_4d[0, 0, 0, :], 0.0)
    np.testing.assert_allclose(history.history_4d[1, 3, 5, :], 100.0 + 10.0 + 2.0 * 3 + 2.0)


def test_load_diverted_tokamak_geometry_trims_x_guards_and_builds_curves(tmp_path: Path) -> None:
    mesh_path = tmp_path / "tokamak.nc"
    _write_tokamak_mesh(mesh_path)

    geometry = load_diverted_tokamak_geometry(mesh_path, active_nx=4)

    assert geometry.rxy.shape == (4, 6)
    assert geometry.zxy.shape == (4, 6)
    assert geometry.psixy.shape == (4, 6)
    np.testing.assert_allclose(geometry.wall_r, geometry.rxy[-1, :])
    np.testing.assert_allclose(geometry.lower_target_z, geometry.zxy[:, 0])
    np.testing.assert_allclose(geometry.upper_target_z, geometry.zxy[:, -1])


def test_toroidal_mean_fluctuation_is_zero_at_initial_time(tmp_path: Path) -> None:
    _write_dump(tmp_path / "BOUT.dmp.0.nc", pe_yind=0)
    _write_dump(tmp_path / "BOUT.dmp.1.nc", pe_yind=1)

    history = assemble_tokamak_rank_history(tmp_path, field_name="Nd+")
    fluctuation = toroidal_mean_fluctuation(history)

    assert fluctuation.shape == (2, 4, 6)
    np.testing.assert_allclose(fluctuation[0], 0.0)


def test_diverted_tokamak_package_can_be_regenerated_from_saved_arrays(
    monkeypatch, tmp_path: Path
) -> None:
    geometry = load_diverted_tokamak_geometry(_make_mesh(tmp_path), active_nx=4)
    history = assemble_tokamak_rank_history(_make_dumps(tmp_path), field_name="Nd+")
    field_history_2d = toroidal_mean_fluctuation(history)
    arrays_path = write_diverted_tokamak_arrays_npz(
        geometry,
        history,
        field_history_2d=field_history_2d,
        path=tmp_path / "arrays.npz",
    )

    loaded_geometry, field_name, time_points, loaded_field = load_diverted_tokamak_arrays_npz(arrays_path)

    assert field_name == "Nd+"
    np.testing.assert_allclose(loaded_geometry.rxy, geometry.rxy)
    np.testing.assert_allclose(time_points, history.time_points)
    np.testing.assert_allclose(loaded_field, field_history_2d)

    def fake_writer(*args, path, **kwargs):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub", encoding="utf-8")
        return target

    monkeypatch.setattr(
        "jax_drb.validation.diverted_tokamak_movie.save_diverted_tokamak_snapshot_panel",
        fake_writer,
    )
    monkeypatch.setattr(
        "jax_drb.validation.diverted_tokamak_movie.save_diverted_tokamak_poster_frame",
        fake_writer,
    )
    monkeypatch.setattr(
        "jax_drb.validation.diverted_tokamak_movie.save_diverted_tokamak_gif",
        fake_writer,
    )

    artifacts = create_diverted_tokamak_movie_package_from_arrays(
        arrays_npz_path=arrays_path,
        output_root=tmp_path / "release_replay",
        case_label="case",
    )

    assert artifacts.arrays_npz_path.exists()
    assert artifacts.analysis_json_path.exists()
    assert artifacts.snapshots_png_path.read_text(encoding="utf-8") == "stub"
    assert artifacts.poster_png_path.read_text(encoding="utf-8") == "stub"
    assert artifacts.movie_gif_path.read_text(encoding="utf-8") == "stub"


def _make_mesh(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "tokamak.nc"
    _write_tokamak_mesh(mesh_path)
    return mesh_path


def _make_dumps(tmp_path: Path) -> Path:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    _write_dump(workdir / "BOUT.dmp.0.nc", pe_yind=0)
    _write_dump(workdir / "BOUT.dmp.1.nc", pe_yind=1)
    return workdir
