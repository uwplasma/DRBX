from __future__ import annotations

from pathlib import Path
from typing import Any


def resolved_capability_tier(reference_case: Any | None) -> str:
    if reference_case is None:
        return "native_exact"
    return reference_case.capability_tier


def integrated_2d_snapshot_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name}_snapshot.npz"


def integrated_2d_optional_history_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name}_optional_history.npz"


def open_field_snapshot_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name}_snapshot.npz"


def tokamak_snapshot_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name}_snapshot.npz"


def tokamak_field_history_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name}_field_history.npz"


def uses_open_field_snapshot_cache(case_name: str) -> bool:
    return case_name in {
        "recycling_1d_rhs",
        "recycling_dthe_rhs",
    }


def uses_tokamak_snapshot_cache(case_name: str) -> bool:
    return case_name in {
        "tokamak_diffusion_flow_one_step",
        "tokamak_diffusion_one_step",
        "tokamak_diffusion_transport_one_step",
        "tokamak_diffusion_transport_short_window",
        "tokamak_heat_transport_one_step",
        "tokamak_heat_transport_short_window",
        "tokamak_diffusion_conduction_one_step",
        "tokamak_diffusion_conduction_short_window",
        "tokamak_linear_transport_one_step",
        "tokamak_linear_transport_short_window",
        "tokamak_isothermal_rhs",
        "tokamak_isothermal_one_step",
        "tokamak_isothermal_short_window",
        "tokamak_isothermal_medium_window",
        "tokamak_turbulence_rhs",
        "tokamak_turbulence_one_step",
        "tokamak_turbulence_short_window",
    }


def uses_tokamak_field_history_cache(case_name: str) -> bool:
    return uses_tokamak_snapshot_cache(case_name)


def uses_snapshot_cache(case_name: str) -> bool:
    return case_name.startswith("integrated_2d_production") or case_name in {
        "tokamak_recycling_rhs",
        "tokamak_recycling_dthe_rhs",
        "tokamak_recycling_dthe_drifts_rhs",
        "tokamak_recycling_dthene_rhs",
    }


def uses_optional_history_cache(case_name: str) -> bool:
    return case_name.startswith("integrated_2d_production") or case_name in {
        "tokamak_recycling_one_step",
        "tokamak_recycling_dthe_one_step",
        "tokamak_recycling_dthe_drifts_one_step",
        "tokamak_recycling_dthene_one_step",
    }
