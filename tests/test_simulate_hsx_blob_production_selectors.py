"""Native canonical-driver contracts for production flux selectors."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


DRIVER = Path(__file__).resolve().parents[1] / "simulate_hsx_blob.py"


def _driver_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_hsx_blob_production_selectors", DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _production_args(driver, *extra: str):
    return driver._build_parser().parse_args(
        (
            "--flux-framework",
            "production-split",
            "--topology",
            "toroidal",
            "--parallel-operator-scheme",
            "fci",
            "--parallel-flux-pairing",
            "support-core",
            "--curvature-rlp-face-scheme",
            "projected-fine",
            *extra,
        )
    )


def test_canonical_driver_is_tracked_at_repository_root():
    assert DRIVER.is_file()
    assert DRIVER.parent.name == "DRBX"
    assert not (DRIVER.parent / "run_staggered_hsx_blob.py").exists()


def test_parser_owns_production_and_sat_selectors():
    driver = _driver_module()
    args = driver._build_parser().parse_args(())
    assert args.parallel_velocity_layout == "cell-centered"
    assert args.parallel_flux_pairing == "legacy"
    assert args.parallel_boundary_pairing == "current-phi"
    assert args.parallel_short_leg_treatment == "explicit"
    boundary_action = next(
        action
        for action in driver._build_parser()._actions
        if action.dest == "parallel_boundary_pairing"
    )
    assert tuple(boundary_action.choices) == (
        "legacy",
        "current-phi",
        "characteristic-sat",
    )
    curvature_wall_action = next(
        action
        for action in driver._build_parser()._actions
        if action.dest == "curvature_wall_flux_closure"
    )
    assert curvature_wall_action.default == "equilibrium-exterior"
    assert tuple(curvature_wall_action.choices) == (
        "equilibrium-exterior",
        "bc-characteristic",
    )


def test_fresh_production_trajectory_accepts_characteristic_sat():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_BOUNDARY_PAIRING"] == "characteristic-sat"


def test_fresh_production_trajectory_rejects_legacy_boundary_pairing():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "legacy",
    )
    with pytest.raises(ValueError, match="current-phi or characteristic-sat"):
        driver._validate_flux_framework(args)


def test_support_core_validation_uses_native_arguments():
    driver = _driver_module()
    args = driver._build_parser().parse_args(
        ("--parallel-flux-pairing", "support-core")
    )
    with pytest.raises(ValueError, match="parallel-operator-scheme fci"):
        driver._validate_flux_framework(args)


def test_native_configuration_exports_short_leg_and_curvature_selectors():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--parallel-short-leg-treatment",
        "local-backward-euler",
        "--parallel-short-leg-cfl-limit",
        "2.25",
        "--curvature-radial-ablation",
        "upper-physical-face",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_TREATMENT"] == (
        "local-backward-euler"
    )
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT"] == "2.25"
    assert driver.os.environ["DRBX_CURVATURE_SPLIT_SCHEME"] == "production-path"
    assert driver.os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] == "production-path"
    assert driver.os.environ["DRBX_CURVATURE_RADIAL_ABLATION"] == (
        "upper-physical-face"
    )


def test_bc_characteristic_curvature_wall_closure_is_exported(monkeypatch):
    driver = _driver_module()
    monkeypatch.delenv("DRBX_CURVATURE_WALL_FLUX_CLOSURE", raising=False)
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--curvature-wall-flux-closure",
        "bc-characteristic",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_CURVATURE_WALL_FLUX_CLOSURE"] == (
        "bc-characteristic-operator-trace-canonical-face-state"
    )


def test_bc_characteristic_curvature_wall_closure_requires_production_split():
    driver = _driver_module()
    args = driver._build_parser().parse_args(
        ("--curvature-wall-flux-closure", "bc-characteristic")
    )
    with pytest.raises(ValueError, match="requires --flux-framework production-split"):
        driver._validate_flux_framework(args)


def test_short_leg_split_is_native_to_compiled_rk4_source():
    source = DRIVER.read_text(encoding="utf-8")
    assert "short_leg_selection_dt=(" in source
    assert "model.apply_short_leg_implicit_material_step(" in source
    assert "solve_dt=dt" in source
    assert "selection_dt=dt" in source


def test_run_metadata_attributes_selectors_to_canonical_driver():
    source = DRIVER.read_text(encoding="utf-8")
    for option in (
        "parallel-boundary-pairing",
        "parallel-short-leg-treatment",
        "parallel-short-leg-cfl-limit",
        "curvature-evolution-component",
        "curvature-radial-ablation",
        "curvature-wall-flux-closure",
        "curvature-characteristic-axes",
        "curvature-radial-characteristic-scheme",
        "curvature-poloidal-characteristic-scheme",
        "curvature-component-diagnostic-scheme",
    ):
        assert f'simulate_hsx_blob.py:--{option}' in source
    assert "run_staggered_hsx_blob.py" not in source


def test_canonical_driver_contains_materialized_face_provenance_path():
    source = DRIVER.read_text(encoding="utf-8")
    assert "build_local_outgoing_fci_face_topology_from_geometry(" in source
    assert '"face_provenance_sha256"' in source
    assert "project_initial_staggered_velocities" in source
    assert "prolong_local_outgoing_fci_face_owner_field" in source
