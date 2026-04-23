from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from .mesh import StructuredMesh


def direct_recycling_species_names(config: BoutConfig) -> tuple[str, ...]:
    names: list[str] = []
    for section_name in config.section_names():
        if section_name == "e":
            names.append(section_name)
            continue
        if not config.has_option(section_name, "type"):
            continue
        type_values = config.parsed(section_name, "type")
        type_items = type_values if isinstance(type_values, tuple) else (type_values,)
        if any(
            str(item).startswith("evolve_") or str(item) in {"quasineutral", "neutral_mixed"}
            for item in type_items
        ):
            names.append(section_name)
    return tuple(names)


def direct_recycling_state_field_names(config: BoutConfig) -> tuple[str, ...]:
    names: list[str] = []
    for species_name in direct_recycling_species_names(config):
        if species_name == "e":
            names.append("Pe")
        else:
            names.extend((f"N{species_name}", f"P{species_name}", f"NV{species_name}"))
    return tuple(names)


def species_optional_velocity_field_map(config: BoutConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (species_name, f"V{species_name}")
        for species_name in direct_recycling_species_names(config)
        if species_name != "e"
    )


def direct_recycling_velocity_optional_field_names(config: BoutConfig) -> tuple[str, ...]:
    return tuple(field_name for _, field_name in species_optional_velocity_field_map(config))


def direct_recycling_optional_field_names(config: BoutConfig) -> tuple[str, ...]:
    names: list[str] = list(direct_recycling_velocity_optional_field_names(config))
    for species_name in direct_recycling_species_names(config):
        if species_name == "e":
            continue
        names.extend((f"SN{species_name}", f"SNV{species_name}", f"SP{species_name}"))
    names.extend(
        (
            "SPe",
            "Sd_target_recycle",
            "Ed_target_recycle",
            "Sd_wall_recycle",
            "Ed_wall_recycle",
            "Sd_pump",
            "Ed_pump",
            "Ed_target_refl",
            "Ed_wall_refl",
            "is_pump",
            "anomalous_D_d+",
            "anomalous_Chi_d+",
            "anomalous_nu_d+",
            "anomalous_D_e",
            "anomalous_Chi_e",
            "anomalous_nu_e",
        )
    )
    return tuple(dict.fromkeys(names))


def restrict_field_template_overrides_to_non_owned_y_guards(
    base_fields: Mapping[str, np.ndarray],
    override_fields: Mapping[str, np.ndarray] | None,
    *,
    mesh: StructuredMesh,
) -> dict[str, np.ndarray] | None:
    if override_fields is None:
        return None
    restricted = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in base_fields.items()
    }
    if mesh.myg <= 0:
        return restricted
    for name, override in override_fields.items():
        if name not in restricted:
            continue
        override_array = np.asarray(override, dtype=np.float64)
        if not mesh.has_lower_y_target:
            restricted[name][:, : mesh.ystart, :] = override_array[:, : mesh.ystart, :]
        if not mesh.has_upper_y_target:
            restricted[name][:, mesh.yend + 1 :, :] = override_array[:, mesh.yend + 1 :, :]
    return restricted


def snapshot_density_source_overrides(
    config: BoutConfig,
    optional_fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    overrides = {
        species_name: np.asarray(optional_fields[field_name], dtype=np.float64)
        for species_name in direct_recycling_species_names(config)
        if species_name != "e"
        for field_name in (f"SN{species_name}",)
        if field_name in optional_fields
    }
    return overrides or None


def snapshot_pressure_source_overrides(
    config: BoutConfig,
    optional_fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    overrides = {
        species_name: np.asarray(optional_fields[field_name], dtype=np.float64)
        for species_name in direct_recycling_species_names(config)
        if species_name != "e"
        for field_name in (f"SP{species_name}",)
        if field_name in optional_fields
    }
    return overrides or None


def snapshot_momentum_source_overrides(
    config: BoutConfig,
    optional_fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    overrides = {
        species_name: np.asarray(optional_fields[field_name], dtype=np.float64)
        for species_name in direct_recycling_species_names(config)
        if species_name != "e"
        for field_name in (f"SNV{species_name}",)
        if field_name in optional_fields
    }
    return overrides or None


def snapshot_velocity_overrides(
    config: BoutConfig,
    optional_fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    overrides = {
        species_name: np.asarray(optional_fields[field_name], dtype=np.float64)
        for species_name, field_name in species_optional_velocity_field_map(config)
        if field_name in optional_fields
    }
    return overrides or None


def apply_species_velocity_overrides(
    config: BoutConfig,
    *,
    field_overrides: Mapping[str, np.ndarray],
    velocity_field_overrides: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not velocity_field_overrides:
        return {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in field_overrides.items()}
    resolver = NumericResolver(config)
    updated = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in field_overrides.items()}
    for species_name, velocity in velocity_field_overrides.items():
        density_name = f"N{species_name}"
        momentum_name = f"NV{species_name}"
        if density_name not in updated or momentum_name not in updated:
            continue
        if config.has_option(species_name, "AA"):
            atomic_mass = float(resolver.resolve(species_name, "AA"))
        else:
            atomic_mass = 1.0 / 1836.0 if species_name == "e" else 1.0
        updated[momentum_name] = atomic_mass * np.asarray(updated[density_name], dtype=np.float64) * np.asarray(
            velocity,
            dtype=np.float64,
        )
    return updated


def integrated_2d_initial_rhs_case_name(case_name: str) -> str:
    if case_name.startswith("tokamak_recycling_dthene"):
        return "tokamak_recycling_dthene_rhs"
    if case_name.startswith("tokamak_recycling_dthe_drifts"):
        return "tokamak_recycling_dthe_drifts_rhs"
    if case_name.startswith("tokamak_recycling_dthe"):
        return "tokamak_recycling_dthe_rhs"
    if case_name.startswith("tokamak_recycling"):
        return "tokamak_recycling_rhs"
    if case_name.startswith("integrated_2d_production"):
        return "integrated_2d_production_rhs"
    return "integrated_2d_recycling_rhs"


def open_field_initial_rhs_case_name(case_name: str) -> str:
    if case_name == "recycling_dthe_one_step":
        return "recycling_dthe_rhs"
    return "recycling_1d_rhs"
