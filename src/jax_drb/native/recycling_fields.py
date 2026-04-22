from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def recycling_evolving_variable_names(species: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ordered evolving variable names for recycling-style species.

    The ordering is part of the implicit-state contract. Electrons contribute
    only `Pe`, while ion and neutral species contribute density, pressure, and
    momentum in that order.
    """

    names: list[str] = []
    for name, sp in species.items():
        if name == "e":
            names.append("Pe")
            continue
        names.extend((sp.density_name, sp.pressure_name, sp.momentum_name))
    return tuple(names)


def recycling_field_templates(
    species: Mapping[str, Any],
    *,
    field_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Build dense field templates keyed by the evolving recycling field names."""

    templates: dict[str, np.ndarray] = {}
    for name in field_names:
        if name == "Pe":
            templates[name] = np.asarray(species["e"].pressure, dtype=np.float64)
            continue
        species_name = name[1:] if name.startswith("N") else name[1:]
        if name.startswith("NV"):
            species_name = name[2:]
        sp = species[species_name]
        if name.startswith("N") and not name.startswith("NV"):
            templates[name] = np.asarray(sp.density, dtype=np.float64)
        elif name.startswith("P"):
            templates[name] = np.asarray(sp.pressure, dtype=np.float64)
        else:
            templates[name] = np.asarray(sp.momentum, dtype=np.float64)
    return templates


def build_recycling_state_fields(
    runtime_model: Any,
    *,
    field_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Construct mutable field arrays from runtime-model templates and overrides."""

    overrides = field_overrides or {}
    fields = recycling_field_templates(runtime_model.species_templates, field_names=runtime_model.field_names)
    for name, value in overrides.items():
        if name in fields:
            fields[name] = np.asarray(value, dtype=np.float64, copy=True)
    return fields
