from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.recycling_setup import (
    build_recycling_runtime_model,
    explicit_pressure_source,
    initialize_species,
    load_density_feedback_controllers,
    resolve_species_numeric_option,
    try_literal_reference,
)
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.reference.paths import default_reference_root


_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")
_INPUT_1D = _REFERENCE_BASE / "tests/integrated/1D-recycling/data/BOUT.inp"
_TOKAMAK_RECYCLING_INPUT = _REFERENCE_BASE / "examples/tokamak-2D/recycling/BOUT.inp"


def _simple_mesh() -> StructuredMesh:
    return StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=0,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=2,
        has_lower_y_target=False,
        has_upper_y_target=False,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([0.0, 1.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )


def test_try_literal_reference_recognizes_existing_option() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_REFERENCE_BASE / "examples/tokamak-2D/recycling-dthe/BOUT.inp")

    assert try_literal_reference(config, "`d+:anomalous_D`") == ("d+", "anomalous_D")
    assert try_literal_reference(config, "d+:anomalous_D") is None


def test_resolve_species_numeric_option_handles_literal_reference() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_REFERENCE_BASE / "examples/tokamak-2D/recycling-dthe/BOUT.inp")

    assert resolve_species_numeric_option(config, "e", "anomalous_D") == pytest.approx(
        resolve_species_numeric_option(config, "d+", "anomalous_D")
    )


def test_initialize_species_keeps_neutral_mixed_species_from_string_type() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_TOKAMAK_RECYCLING_INPUT)
    mesh = _simple_mesh()
    field_overrides = {
        "Nd+": np.ones((1, 2, 1), dtype=np.float64),
        "Pd+": np.ones((1, 2, 1), dtype=np.float64),
        "NVd+": np.zeros((1, 2, 1), dtype=np.float64),
        "Nd": np.ones((1, 2, 1), dtype=np.float64),
        "Pd": np.ones((1, 2, 1), dtype=np.float64),
        "NVd": np.zeros((1, 2, 1), dtype=np.float64),
        "Pe": np.ones((1, 2, 1), dtype=np.float64),
    }

    species = initialize_species(config, mesh=mesh, field_overrides=field_overrides)

    assert "d+" in species
    assert "d" in species
    assert species["d"].has_pressure
    assert species["d"].has_momentum


def test_explicit_pressure_source_normalizes_scalar_expression() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = apply_bout_overrides(load_bout_input(_INPUT_1D), ("Pd+:source=2.0",))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    scalars = resolved_dataset_scalars(run_config)

    source = explicit_pressure_source(config, "d+", mesh=mesh, dataset_scalars=scalars)
    expected = 2.0 / (1.60218e-19 * scalars["Nnorm"] * scalars["Tnorm"] * scalars["Omega_ci"])

    np.testing.assert_allclose(source, expected, rtol=1.0e-12, atol=1.0e-12)


def test_load_density_feedback_controllers_normalizes_source_shape() -> None:
    config_path = Path("/tmp/jax_drb_recycling_setup_controller.inp")
    config_path.write_text(
        "[e]\n"
        "type = quasineutral\n"
        "charge = -1\n"
        "\n"
        "[Pe]\n"
        "function = 1.0\n"
        "\n"
        "[d+]\n"
        "type = evolve_density, evolve_pressure, evolve_momentum, upstream_density_feedback\n"
        "charge = 1.0\n"
        "AA = 2.0\n"
        "density_upstream = 5.0\n"
        "density_controller_p = 0.2\n"
        "density_controller_i = 0.03\n"
        "diagnose = true\n"
        "\n"
        "[Nd+]\n"
        "function = 2.0\n"
        "source_shape = `feedback_shape:function`\n"
        "\n"
        "[Pd+]\n"
        "function = 3.0\n"
        "\n"
        "[NVd+]\n"
        "function = 0.0\n"
        "\n"
        "[feedback_shape]\n"
        "function = 6.0\n",
        encoding="utf-8",
    )
    config = load_bout_input(config_path)
    mesh = _simple_mesh()
    scalars = {"Nnorm": 2.0, "Tnorm": 4.0, "Omega_ci": 8.0}
    species = initialize_species(config, mesh=mesh, dataset_scalars=scalars)

    controllers = load_density_feedback_controllers(
        config,
        species=species,
        mesh=mesh,
        dataset_scalars=scalars,
    )

    assert tuple(sorted(controllers)) == ("d+",)
    controller = controllers["d+"]
    np.testing.assert_allclose(controller.density_source_shape, 6.0 / (2.0 * 8.0), rtol=1.0e-12, atol=1.0e-12)
    assert controller.density_upstream == pytest.approx(2.5)
    assert controller.diagnose is True


def test_build_recycling_runtime_model_collects_feedback_names_and_overrides() -> None:
    config_path = Path("/tmp/jax_drb_recycling_setup_runtime.inp")
    config_path.write_text(
        "[e]\n"
        "type = quasineutral\n"
        "charge = -1\n"
        "\n"
        "[Pe]\n"
        "function = 1.0\n"
        "\n"
        "[d+]\n"
        "type = evolve_density, evolve_pressure, evolve_momentum, upstream_density_feedback\n"
        "charge = 1.0\n"
        "AA = 2.0\n"
        "density_upstream = 4.0\n"
        "\n"
        "[Nd+]\n"
        "function = 2.0\n"
        "\n"
        "[Pd+]\n"
        "function = 3.0\n"
        "source = 5.0\n"
        "\n"
        "[NVd+]\n"
        "function = 0.0\n",
        encoding="utf-8",
    )
    config = load_bout_input(config_path)
    mesh = _simple_mesh()
    scalars = {"Nnorm": 2.0, "Tnorm": 4.0, "Omega_ci": 8.0}
    density_override = {"d+": np.full((1, 2, 1), 0.25, dtype=np.float64)}
    pressure_override = {"d+": np.full((1, 2, 1), 0.5, dtype=np.float64)}

    runtime_model = build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
        density_source_overrides=density_override,
        pressure_source_overrides=pressure_override,
    )

    assert runtime_model.field_names == ("Pe", "Nd+", "Pd+", "NVd+")
    assert runtime_model.feedback_names == ("d+",)
    np.testing.assert_allclose(runtime_model.density_source_overrides["d+"], 0.25)
    np.testing.assert_allclose(runtime_model.pressure_source_overrides["d+"], 0.5)
    np.testing.assert_allclose(
        runtime_model.explicit_pressure_sources["d+"],
        5.0 / (1.60218e-19 * scalars["Nnorm"] * scalars["Tnorm"] * scalars["Omega_ci"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
