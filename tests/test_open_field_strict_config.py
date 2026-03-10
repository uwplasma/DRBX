from __future__ import annotations

import tomllib
from pathlib import Path


def test_open_field_strict_config_uses_flux_form_numerics() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_benchmark_hermes_strict.toml"
    )
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    numerics = cfg["numerics"]

    assert numerics["exb_advection_form"] == "flux"
    assert numerics["exb_advect_conservative"] is True
    assert numerics["exb_flux_scheme"] == "hermes_xppm"
    assert numerics["parallel_flux_scheme"] == "hermes_mirror"
    assert numerics["parallel_fixflux"] is True
    assert numerics["parallel_sheath_flux_mode"] == "boundary_flux"
    assert numerics["parallel_transform"] == "shifted"
    assert numerics["parallel_shift_interp"] == "spectral"
    assert numerics["parallel_current_limiter"] == "none"


def test_open_field_strict_early_config_splits_advective_and_current_limiters() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["engine"] == "hermes_literal"
    numerics = cfg["numerics"]

    assert numerics["exb_flux_scheme"] == "hermes_mirror"
    assert numerics["hermes_mirror_parallel_edge_block"] == 8
    assert numerics["hermes_mirror_parallel_subdomain_size"] == 8
    assert numerics["exb_poloidal_y_scale"] == 1.0
    assert numerics["parallel_limiter"] == "mc"
    assert numerics["parallel_current_limiter"] == "none"
    assert numerics["parallel_flux_scheme"] == "hermes_mirror"
