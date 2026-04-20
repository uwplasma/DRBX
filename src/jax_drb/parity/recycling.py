from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


@dataclass(frozen=True)
class RecyclingControllerSnapshot:
    controller_multipliers: dict[str, float]
    controller_proportional_terms: dict[str, float]
    controller_integral_terms: dict[str, float]
    controller_sources: dict[str, np.ndarray]
    restart_integrals: dict[str, float]


def extract_recycling_controller_snapshot(
    dmp_path: str | Path,
    restart_path: str | Path,
    *,
    controller_species: tuple[str, ...],
) -> RecyclingControllerSnapshot:
    with Dataset(Path(dmp_path)) as dmp, Dataset(Path(restart_path)) as restart:
        controller_multipliers = {
            species: _last_scalar(dmp, f"density_feedback_src_mult_{species}")
            for species in controller_species
            if f"density_feedback_src_mult_{species}" in dmp.variables
        }
        controller_proportional_terms = {
            species: _last_scalar(dmp, f"density_feedback_src_p_{species}")
            for species in controller_species
            if f"density_feedback_src_p_{species}" in dmp.variables
        }
        controller_integral_terms = {
            species: _last_scalar(dmp, f"density_feedback_src_i_{species}")
            for species in controller_species
            if f"density_feedback_src_i_{species}" in dmp.variables
        }
        controller_sources = {
            species: np.asarray(dmp.variables[f"S{species}_feedback"][-1], dtype=np.float64)
            for species in controller_species
            if f"S{species}_feedback" in dmp.variables
        }
        restart_integrals = {
            name.removesuffix("_density_error_integral"): _scalar_value(restart.variables[name])
            for name in restart.variables
            if name.endswith("_density_error_integral")
        }
    return RecyclingControllerSnapshot(
        controller_multipliers=controller_multipliers,
        controller_proportional_terms=controller_proportional_terms,
        controller_integral_terms=controller_integral_terms,
        controller_sources=controller_sources,
        restart_integrals=restart_integrals,
    )


def compare_snapshot_mappings(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            continue
        expected_value = np.asarray(expected[key], dtype=np.float64)
        actual_value = np.asarray(actual[key], dtype=np.float64)
        report[key] = {
            "max_abs_diff": float(np.max(np.abs(actual_value - expected_value))) if actual_value.size else 0.0,
            "expected": float(expected_value.reshape(-1)[0]) if expected_value.size else 0.0,
            "actual": float(actual_value.reshape(-1)[0]) if actual_value.size else 0.0,
        }
    return report


def _last_scalar(dataset: Dataset, variable_name: str) -> float:
    values = np.asarray(dataset.variables[variable_name][:], dtype=np.float64)
    return float(values.reshape(-1)[-1])


def _scalar_value(variable: Any) -> float:
    values = np.asarray(variable[...], dtype=np.float64)
    return float(values.reshape(-1)[0])
