from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset
import pytest


_REPO = Path("/Users/rogerio/local/jax_drb")


def _load_script_module(relative_path: str, module_name: str):
    path = _REPO / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_target_cell_indices_pick_upper_target_adjacent_row() -> None:
    module = _load_script_module(
        "scripts/diagnose_recycling_target_cell_history.py",
        "recycling_target_history_diag",
    )
    mesh = SimpleNamespace(xstart=2, xend=4, ystart=10, yend=14)

    trimmed, global_index = module.target_cell_indices(
        mesh,
        x_index=1,
        y_offset=0,
        z_index=3,
        target_edge="upper",
    )

    assert trimmed == (1, 4, 3)
    assert global_index == (3, 14, 3)


def test_extract_active_cell_series_reads_expected_history() -> None:
    module = _load_script_module(
        "scripts/diagnose_recycling_target_cell_history.py",
        "recycling_target_history_diag_extract",
    )
    mesh = SimpleNamespace(xstart=1, xend=2, ystart=4, yend=6)
    values = np.zeros((3, 4, 8, 2), dtype=np.float64)
    values[:, 1, 6, 0] = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)

    series = module.extract_active_cell_series(
        values,
        mesh,
        x_index=0,
        y_offset=0,
        z_index=0,
        target_edge="upper",
    )

    assert np.array_equal(series, np.asarray([1.0, 2.0, 4.0], dtype=np.float64))


def test_controller_integral_series_from_term_divides_by_gain() -> None:
    module = _load_script_module(
        "scripts/diagnose_recycling_controller_history.py",
        "recycling_controller_history_diag",
    )

    values = module.controller_integral_series_from_term(
        np.asarray([0.0, 0.25, 0.5], dtype=np.float64),
        controller_gain=0.5,
    )

    assert np.array_equal(values, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))


def test_extract_scalar_series_reads_time_history(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/diagnose_recycling_controller_history.py",
        "recycling_controller_history_diag_scalar",
    )
    path = tmp_path / "diag.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("t", 3)
        variable = dataset.createVariable("density_feedback_src_mult_d+", "f8", ("t",))
        variable[:] = np.asarray([1.0, 1.5, 2.5], dtype=np.float64)

    with Dataset(path) as dataset:
        values = module.extract_scalar_series(dataset, "density_feedback_src_mult_d+")

    assert np.array_equal(values, np.asarray([1.0, 1.5, 2.5], dtype=np.float64))


def test_relative_error_metrics_separates_small_denominator_artifacts() -> None:
    module = _load_script_module(
        "scripts/diagnose_recycling_neutral_transient.py",
        "recycling_neutral_transient_diag",
    )

    actual = np.asarray([1.2, 0.05, 2.0e-4], dtype=np.float64)
    reference = np.asarray([1.0, 0.0, 1.0e-6], dtype=np.float64)
    metrics = module.relative_error_metrics(
        actual,
        reference,
        magnitude_floor_ratio=1.0e-2,
        absolute_floor=1.0e-5,
    )

    assert metrics["max_abs"] == np.max(np.abs(actual - reference))
    assert metrics["max_rel"] > metrics["max_rel_significant"]
    assert metrics["significant_count"] == 1
    assert metrics["max_rel_significant"] == pytest.approx(0.2)


def test_production_target_band_cells_selects_lower_active_row() -> None:
    module = _load_script_module(
        "scripts/diagnose_integrated_2d_production_ion_terms.py",
        "integrated_2d_production_ion_terms_diag",
    )
    mesh = SimpleNamespace(ystart=7)

    cells = module.production_target_band_cells(mesh, x_indices=(14, 15), z_index=2)

    assert cells == ((14, 7, 2), (15, 7, 2))


def test_strip_anomalous_diffusion_from_boutinp_text_removes_only_target_component() -> None:
    module = _load_script_module(
        "scripts/diagnose_integrated_2d_production_anomalous_diffusion.py",
        "integrated_2d_production_anom_diag",
    )
    text = (
        "[d+]\n"
        "type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion\n"
        "\n"
        "[e]\n"
        "type = quasineutral, evolve_pressure, zero_current, anomalous_diffusion\n"
        "\n"
        "[d]\n"
        "type = neutral_mixed, neutral_boundary\n"
    )

    rewritten = module.strip_anomalous_diffusion_from_boutinp_text(text)

    assert "evolve_density, evolve_momentum, evolve_pressure" in rewritten
    assert "quasineutral, evolve_pressure, zero_current" in rewritten
    assert rewritten.count("anomalous_diffusion") == 0
    assert "[d]\n" in rewritten


def test_default_tokamak_recycling_cases_include_three_direct_one_step_lanes() -> None:
    module = _load_script_module(
        "scripts/diagnose_tokamak_recycling_one_step.py",
        "tokamak_recycling_one_step_diag",
    )

    assert module.default_tokamak_recycling_cases() == (
        "tokamak_recycling_dthe_one_step",
        "tokamak_recycling_dthe_drifts_one_step",
        "tokamak_recycling_dthene_one_step",
    )


def test_default_tokamak_recycling_blocker_cells_pick_lower_target_corner_pair() -> None:
    module = _load_script_module(
        "scripts/diagnose_tokamak_recycling_ion_viscosity.py",
        "tokamak_recycling_ion_viscosity_diag",
    )
    mesh = SimpleNamespace(xstart=2, ystart=5)

    cells = module.default_tokamak_recycling_blocker_cells(mesh)

    assert cells == ((2, 5, 0), (3, 5, 0))


def test_read_last_time_field_uses_last_time_plane(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/diagnose_tokamak_recycling_ion_viscosity.py",
        "tokamak_recycling_ion_viscosity_diag_field",
    )
    path = tmp_path / "diag.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("x", 1)
        dataset.createDimension("y", 2)
        dataset.createDimension("z", 1)
        variable = dataset.createVariable("DivPiPar_d+", "f8", ("t", "x", "y", "z"))
        values = np.zeros((2, 1, 2, 1), dtype=np.float64)
        values[1, 0, 1, 0] = 3.5
        variable[:] = values

    extracted = module._read_last_time_field(path, "DivPiPar_d+")

    assert extracted.shape == (1, 2, 1)
    assert extracted[0, 1, 0] == pytest.approx(3.5)


def test_hermes_collision_field_name_matches_bout_diagnostic_convention() -> None:
    module = _load_script_module(
        "scripts/diagnose_tokamak_recycling_ion_viscosity.py",
        "tokamak_recycling_ion_viscosity_diag_collision_name",
    )

    assert module._hermes_collision_field_name("d+", "t+") == "Kd+t+_coll"
    assert module._hermes_collision_field_name("t+", "e") == "Kt+e_coll"
