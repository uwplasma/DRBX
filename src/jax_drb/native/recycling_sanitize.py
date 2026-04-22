from __future__ import annotations

from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver


def sanitize_recycling_fields(
    config: BoutConfig,
    fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sanitized = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    resolver = NumericResolver(config)
    ion_density_names = sorted(name for name in sanitized if name.startswith("N") and not name.startswith("NV") and name != "Ne")
    electron_density = np.zeros_like(sanitized[ion_density_names[0]], dtype=np.float64) if ion_density_names else None
    if electron_density is not None:
        for density_name in ion_density_names:
            species_name = density_name[1:]
            if species_name == "e":
                continue
            charge = float(resolver.resolve(species_name, "charge")) if config.has_option(species_name, "charge") else 0.0
            if charge > 0.0:
                electron_density = electron_density + charge * np.maximum(sanitized[density_name], 1.0e-12)
    for name in list(sanitized):
        if name.startswith("N") and not name.startswith("NV") and name != "Ne":
            species_name = name[1:]
            density_floor = float(resolver.resolve(species_name, "density_floor")) if config.has_option(species_name, "density_floor") else 1.0e-7
            sanitized[name] = np.maximum(sanitized[name], density_floor)
        elif name.startswith("P"):
            species_name = name[1:]
            if config.has_option(species_name, "temperature_floor"):
                temperature_floor = float(resolver.resolve(species_name, "temperature_floor"))
            elif species_name == "e":
                temperature_floor = 0.1
            else:
                charge = float(resolver.resolve(species_name, "charge")) if config.has_option(species_name, "charge") else 0.0
                temperature_floor = 0.1 if charge != 0.0 else 0.0
            if species_name == "e" and electron_density is not None:
                sanitized[name] = np.maximum(sanitized[name], temperature_floor * np.maximum(electron_density, 1.0e-7))
            else:
                density_name = f"N{species_name}"
                density = sanitized.get(density_name)
                floor_density = np.maximum(density, 1.0e-7) if density is not None else 1.0e-7
                sanitized[name] = np.maximum(sanitized[name], temperature_floor * floor_density)
    return sanitized
