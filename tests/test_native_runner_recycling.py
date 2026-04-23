from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.runner_recycling import (
    apply_species_velocity_overrides,
    direct_recycling_optional_field_names,
    direct_recycling_species_names,
    direct_recycling_state_field_names,
    integrated_2d_initial_rhs_case_name,
    open_field_initial_rhs_case_name,
    restrict_field_template_overrides_to_non_owned_y_guards,
    snapshot_density_source_overrides,
    snapshot_momentum_source_overrides,
    snapshot_pressure_source_overrides,
    snapshot_velocity_overrides,
)


_RUNNER_RECYCLING_INPUT = """
[model]
components = e, d+, d

[e]
type = quasineutral

[d+]
type = evolve_density, evolve_pressure, evolve_momentum
AA = 2.0

[d]
type = neutral_mixed
AA = 2.0
"""


def test_direct_recycling_field_metadata_detects_species_and_outputs() -> None:
    config = parse_bout_input(_RUNNER_RECYCLING_INPUT)

    assert direct_recycling_species_names(config) == ("e", "d+", "d")
    assert direct_recycling_state_field_names(config) == (
        "Pe",
        "Nd+",
        "Pd+",
        "NVd+",
        "Nd",
        "Pd",
        "NVd",
    )

    optional_names = direct_recycling_optional_field_names(config)
    assert optional_names[0] == "Ne"
    assert "Vd+" in optional_names
    assert "Vd" in optional_names
    assert "SNd+" in optional_names
    assert "SPd" in optional_names
    assert "Sd_target_recycle" in optional_names


def test_snapshot_source_and_velocity_overrides_extract_species_payloads() -> None:
    config = parse_bout_input(_RUNNER_RECYCLING_INPUT)
    field = np.ones((1, 4, 1), dtype=np.float64)
    optional_fields = {
        "SNd+": 2.0 * field,
        "SPd+": 3.0 * field,
        "SNVd+": 4.0 * field,
        "Vd+": 5.0 * field,
    }

    density = snapshot_density_source_overrides(config, optional_fields)
    pressure = snapshot_pressure_source_overrides(config, optional_fields)
    momentum = snapshot_momentum_source_overrides(config, optional_fields)
    velocity = snapshot_velocity_overrides(config, optional_fields)

    assert np.allclose(density["d+"], 2.0)
    assert np.allclose(pressure["d+"], 3.0)
    assert np.allclose(momentum["d+"], 4.0)
    assert np.allclose(velocity["d+"], 5.0)


def test_apply_species_velocity_overrides_reconstructs_momentum_from_density() -> None:
    config = parse_bout_input(_RUNNER_RECYCLING_INPUT)
    density = np.full((1, 3, 1), 2.5, dtype=np.float64)
    field_overrides = {
        "Nd+": density,
        "NVd+": np.zeros_like(density),
    }
    updated = apply_species_velocity_overrides(
        config,
        field_overrides=field_overrides,
        velocity_field_overrides={"d+": np.full_like(density, 4.0)},
    )

    assert np.allclose(updated["NVd+"], 2.0 * density * 4.0)


def test_restrict_field_template_overrides_to_non_owned_y_guards() -> None:
    base = {"Nd+": np.zeros((1, 6, 1), dtype=np.float64)}
    override = {"Nd+": np.arange(6, dtype=np.float64).reshape(1, 6, 1)}
    mesh = SimpleNamespace(myg=1, ystart=1, yend=4, has_lower_y_target=False, has_upper_y_target=True)

    restricted = restrict_field_template_overrides_to_non_owned_y_guards(base, override, mesh=mesh)

    assert np.allclose(restricted["Nd+"][:, :1, :], override["Nd+"][:, :1, :])
    assert np.allclose(restricted["Nd+"][:, 1:5, :], 0.0)
    assert np.allclose(restricted["Nd+"][:, 5:, :], 0.0)


def test_runner_recycling_initial_rhs_case_name_helpers_cover_transient_rungs() -> None:
    assert integrated_2d_initial_rhs_case_name("integrated_2d_recycling_one_step") == "integrated_2d_recycling_rhs"
    assert integrated_2d_initial_rhs_case_name("integrated_2d_recycling_short_window") == "integrated_2d_recycling_rhs"
    assert integrated_2d_initial_rhs_case_name("integrated_2d_production_medium_window") == "integrated_2d_production_rhs"
    assert integrated_2d_initial_rhs_case_name("tokamak_recycling_one_step") == "tokamak_recycling_rhs"
    assert integrated_2d_initial_rhs_case_name("tokamak_recycling_dthe_one_step") == "tokamak_recycling_dthe_rhs"
    assert integrated_2d_initial_rhs_case_name("tokamak_recycling_dthe_drifts_one_step") == "tokamak_recycling_dthe_drifts_rhs"
    assert integrated_2d_initial_rhs_case_name("tokamak_recycling_dthene_one_step") == "tokamak_recycling_dthene_rhs"
    assert open_field_initial_rhs_case_name("recycling_dthe_one_step") == "recycling_dthe_rhs"
    assert open_field_initial_rhs_case_name("recycling_1d_short_window") == "recycling_1d_rhs"
