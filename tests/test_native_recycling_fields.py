from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_drb.native.recycling_fields import (
    build_recycling_state_fields,
    recycling_evolving_variable_names,
    recycling_field_templates,
)


def _sample_species():
    return {
        "e": SimpleNamespace(
            pressure=np.array([[[1.0]], [[2.0]]], dtype=np.float64),
        ),
        "d": SimpleNamespace(
            density_name="Nd",
            pressure_name="Pd",
            momentum_name="NVd",
            density=np.array([[[3.0]], [[4.0]]], dtype=np.float64),
            pressure=np.array([[[5.0]], [[6.0]]], dtype=np.float64),
            momentum=np.array([[[7.0]], [[8.0]]], dtype=np.float64),
        ),
        "h": SimpleNamespace(
            density_name="Nh",
            pressure_name="Ph",
            momentum_name="NVh",
            density=np.array([[[9.0]], [[10.0]]], dtype=np.float64),
            pressure=np.array([[[11.0]], [[12.0]]], dtype=np.float64),
            momentum=np.array([[[13.0]], [[14.0]]], dtype=np.float64),
        ),
    }


def test_recycling_evolving_variable_names_orders_electron_then_species_triplets() -> None:
    names = recycling_evolving_variable_names(_sample_species())

    assert names == ("Pe", "Nd", "Pd", "NVd", "Nh", "Ph", "NVh")


def test_recycling_field_templates_selects_expected_arrays() -> None:
    species = _sample_species()
    field_names = ("Pe", "Nd", "Pd", "NVd")

    templates = recycling_field_templates(species, field_names=field_names)

    np.testing.assert_allclose(templates["Pe"], species["e"].pressure)
    np.testing.assert_allclose(templates["Nd"], species["d"].density)
    np.testing.assert_allclose(templates["Pd"], species["d"].pressure)
    np.testing.assert_allclose(templates["NVd"], species["d"].momentum)


def test_build_recycling_state_fields_applies_overrides_without_mutating_templates() -> None:
    species = _sample_species()
    runtime_model = SimpleNamespace(
        species_templates=species,
        field_names=("Pe", "Nd", "Pd", "NVd"),
    )
    override = np.array([[[21.0]], [[22.0]]], dtype=np.float64)

    fields = build_recycling_state_fields(runtime_model, field_overrides={"Pd": override})

    np.testing.assert_allclose(fields["Pd"], override)
    np.testing.assert_allclose(runtime_model.species_templates["d"].pressure, np.array([[[5.0]], [[6.0]]]))
    assert fields["Pd"] is not override
