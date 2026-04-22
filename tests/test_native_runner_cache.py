from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from jax_drb.native.runner_cache import (
    integrated_2d_optional_history_cache_path,
    integrated_2d_snapshot_cache_path,
    open_field_snapshot_cache_path,
    resolved_capability_tier,
    tokamak_field_history_cache_path,
    tokamak_snapshot_cache_path,
    uses_open_field_snapshot_cache,
    uses_optional_history_cache,
    uses_snapshot_cache,
    uses_tokamak_field_history_cache,
    uses_tokamak_snapshot_cache,
)


def test_resolved_capability_tier_defaults_to_native_exact() -> None:
    assert resolved_capability_tier(None) == "native_exact"


def test_resolved_capability_tier_uses_reference_case_value() -> None:
    reference_case = SimpleNamespace(capability_tier="native_operational")
    assert resolved_capability_tier(reference_case) == "native_operational"


def test_runner_cache_path_builders_use_expected_suffixes(tmp_path: Path) -> None:
    assert integrated_2d_snapshot_cache_path(tmp_path, "integrated_2d_production_rhs") == tmp_path / "integrated_2d_production_rhs_snapshot.npz"
    assert integrated_2d_optional_history_cache_path(tmp_path, "integrated_2d_production_short_window") == tmp_path / "integrated_2d_production_short_window_optional_history.npz"
    assert open_field_snapshot_cache_path(tmp_path, "recycling_1d_rhs") == tmp_path / "recycling_1d_rhs_snapshot.npz"
    assert tokamak_snapshot_cache_path(tmp_path, "tokamak_diffusion_flow_one_step") == tmp_path / "tokamak_diffusion_flow_one_step_snapshot.npz"
    assert tokamak_field_history_cache_path(tmp_path, "tokamak_diffusion_flow_one_step") == tmp_path / "tokamak_diffusion_flow_one_step_field_history.npz"


def test_runner_cache_usage_flags_cover_promoted_case_families() -> None:
    assert uses_open_field_snapshot_cache("recycling_1d_rhs")
    assert not uses_open_field_snapshot_cache("recycling_1d_one_step")

    assert uses_tokamak_snapshot_cache("tokamak_diffusion_flow_one_step")
    assert uses_tokamak_field_history_cache("tokamak_turbulence_short_window")
    assert not uses_tokamak_snapshot_cache("tokamak_recycling_dthene_one_step")

    assert uses_snapshot_cache("integrated_2d_production_rhs")
    assert uses_snapshot_cache("tokamak_recycling_dthene_rhs")
    assert not uses_snapshot_cache("neutral_mixed_short_window")

    assert uses_optional_history_cache("integrated_2d_production_short_window")
    assert uses_optional_history_cache("tokamak_recycling_dthene_one_step")
    assert not uses_optional_history_cache("tokamak_recycling_dthene_rhs")
