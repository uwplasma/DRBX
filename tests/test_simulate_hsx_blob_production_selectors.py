"""Native canonical-driver contracts for production flux selectors."""

from __future__ import annotations

import importlib.util
import ast
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import HaloLayout3D, LocalCurvatureFaceCoefficients3D
from drbx.native.fci_drb_EB_rhs import (
    _select_characteristic_sat_current_divergence,
)


DRIVER = Path(__file__).resolve().parents[1] / "simulate_hsx_blob.py"


@pytest.fixture(autouse=True)
def _restore_drbx_environment():
    """Keep runtime-selector exports local to each driver contract test."""

    original = {
        key: value for key, value in os.environ.items() if key.startswith("DRBX_")
    }
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith("DRBX_"):
                os.environ.pop(key, None)
        os.environ.update(original)


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
    assert args.parallel_characteristic_wall_law == "primitive-least-residual"
    assert args.parallel_boundary_pairing == "current-phi"
    assert args.parallel_short_leg_treatment == "explicit"
    assert args.parallel_short_leg_selection == "cfl"
    assert args.characteristic_sat_affine_current_lift == "enabled"
    assert args.parallel_current_phi_pair == "enabled"
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
    wall_law_action = next(
        action
        for action in driver._build_parser()._actions
        if action.dest == "parallel_characteristic_wall_law"
    )
    assert tuple(wall_law_action.choices) == (
        "primitive-least-residual",
        "energy-absorbing",
    )
    assert wall_law_action.default == "primitive-least-residual"
    selection_action = next(
        action for action in driver._build_parser()._actions
        if action.dest == "parallel_short_leg_selection"
    )
    assert tuple(selection_action.choices) == ("cfl", "all-physical-walls")
    assert selection_action.default == "cfl"
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
    poisson_action = next(
        action
        for action in driver._build_parser()._actions
        if action.dest == "poisson_bracket_scheme"
    )
    assert "compatible-third-order-upwind" in tuple(poisson_action.choices)


def test_production_accepts_characteristic_poisson_bracket():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--poisson-bracket-scheme",
        "compatible-third-order-upwind",
    )
    driver._validate_flux_framework(args)


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


def test_characteristic_sat_affine_current_lift_ablation_is_exact_and_exported(
    monkeypatch,
):
    homogeneous = jnp.asarray((1.0, -2.0), dtype=jnp.float64)
    affine = jnp.asarray((0.25, 3.0), dtype=jnp.float64)
    np.testing.assert_array_equal(
        _select_characteristic_sat_current_divergence(
            homogeneous + affine, homogeneous, affine_lift="enabled"
        ),
        homogeneous + affine,
    )
    np.testing.assert_array_equal(
        _select_characteristic_sat_current_divergence(
            homogeneous + affine, homogeneous, affine_lift="suppressed"
        ),
        homogeneous,
    )

    driver = _driver_module()
    monkeypatch.delenv(
        "DRBX_CHARACTERISTIC_SAT_AFFINE_CURRENT_LIFT", raising=False
    )
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--characteristic-sat-affine-current-lift",
        "suppressed",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ[
        "DRBX_CHARACTERISTIC_SAT_AFFINE_CURRENT_LIFT"
    ] == "suppressed"


def test_characteristic_sat_affine_current_lift_rejects_non_sat_path():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--characteristic-sat-affine-current-lift",
        "suppressed",
    )
    with pytest.raises(ValueError, match="characteristic-sat"):
        driver._validate_flux_framework(args)


def test_parallel_current_phi_pair_suppression_is_exported_and_paired(
    monkeypatch,
):
    driver = _driver_module()
    monkeypatch.delenv("DRBX_PARALLEL_CURRENT_PHI_PAIR", raising=False)
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--parallel-current-phi-pair",
        "suppressed",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_CURRENT_PHI_PAIR"] == "suppressed"

    source = (DRIVER.parent / "src/drbx/native/fci_drb_EB_rhs.py").read_text(
        encoding="utf-8"
    )
    assert 'self.parallel_current_phi_pair == "suppressed"' in source
    assert "Ve_phi_force_term = jnp.zeros_like(Ve_phi_force_term)" in source
    assert "vorticity_current_term = jnp.zeros_like(vorticity_current_term)" in source


def test_parallel_current_phi_pair_suppression_rejects_nonproduction_path():
    driver = _driver_module()
    args = driver._build_parser().parse_args(
        ("--parallel-current-phi-pair", "suppressed")
    )
    with pytest.raises(ValueError, match="production support-core"):
        driver._validate_flux_framework(args)


def test_fresh_production_trajectory_rejects_legacy_boundary_pairing():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "legacy",
    )
    with pytest.raises(ValueError, match="current-phi or characteristic-sat"):
        driver._validate_flux_framework(args)


def test_energy_absorbing_wall_law_is_exported_for_compatible_production_path(
    monkeypatch,
):
    driver = _driver_module()
    monkeypatch.delenv("DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW", raising=False)
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--parallel-characteristic-wall-law",
        "energy-absorbing",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"] == (
        "energy-absorbing"
    )
    assert driver.os.environ["DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"] == (
        "maximally-dissipative-energy-absorbing-normalized-equilibrium"
    )


def test_wall_law_metadata_is_conditional_and_provenance_is_explicit():
    driver = _driver_module()
    primitive = driver._parallel_characteristic_wall_metadata(
        "primitive-least-residual"
    )
    assert primitive["parallel_material_wall_flux_closure"] == (
        "characteristic-projected-operator-trace-canonical-face-state"
    )
    assert primitive["parallel_characteristic_wall_equilibrium_reference"] is None
    assert primitive["parallel_characteristic_wall_energy_normalizer"] is None

    absorbing = driver._parallel_characteristic_wall_metadata("energy-absorbing")
    assert absorbing["parallel_material_wall_flux_closure"] == (
        "maximally-dissipative-energy-absorbing-normalized-equilibrium"
    )
    assert absorbing["parallel_characteristic_wall_equilibrium_reference"] == [
        1.0, 1.0, 1.0, 0.0, 0.0
    ]
    assert absorbing["parallel_characteristic_wall_provenance"] == (
        "experimental-normalized-equilibrium-absorber"
    )
    assert absorbing["parallel_characteristic_wall_energy_normalizer"] == (
        "unit-modal-mathematical"
    )
@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (
            ("--parallel-velocity-layout", "fci-staggered"),
            "cell-centered",
        ),
        (
            ("--flux-framework", "legacy"),
            "production-path",
        ),
        (
            (),
            "characteristic-sat",
        ),
        (
            (
                "--parallel-boundary-pairing",
                "characteristic-sat",
                "--parallel-inflow-closure",
                "local-characteristic",
            ),
            "central",
        ),
    ),
)
def test_energy_absorbing_wall_law_rejects_incompatible_selectors(
    extra, message
):
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-characteristic-wall-law",
        "energy-absorbing",
        *extra,
    )
    with pytest.raises(ValueError, match=message):
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
        "--time-integrator",
        "imex-ssp222",
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
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_SELECTION"] == "cfl"
    assert driver.os.environ["DRBX_CURVATURE_SPLIT_SCHEME"] == "production-path"
    assert driver.os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] == "production-path"
    assert driver.os.environ["DRBX_CURVATURE_RADIAL_ABLATION"] == (
        "upper-physical-face"
    )


@pytest.mark.parametrize(
    "override",
    (
        ("--parallel-short-leg-treatment", "explicit"),
        ("--parallel-characteristic-wall-law", "primitive-least-residual"),
        ("--parallel-boundary-pairing", "current-phi"),
    ),
)
def test_all_physical_walls_requires_absorbing_be_configuration(override):
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-short-leg-selection", "all-physical-walls",
        "--parallel-short-leg-treatment", "local-backward-euler",
        "--parallel-characteristic-wall-law", "energy-absorbing",
        "--parallel-boundary-pairing", "characteristic-sat",
        "--time-integrator", "imex-ssp222",
        *override,
    )
    with pytest.raises(ValueError):
        driver._validate_flux_framework(args)


def test_all_physical_walls_exports_selection_without_inf_sentinel():
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-short-leg-selection", "all-physical-walls",
        "--parallel-short-leg-treatment", "local-backward-euler",
        "--parallel-characteristic-wall-law", "energy-absorbing",
        "--parallel-boundary-pairing", "characteristic-sat",
        "--time-integrator", "imex-ssp222",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_SELECTION"] == (
        "all-physical-walls"
    )
    source = DRIVER.read_text(encoding="utf-8")
    assert "parallel_short_leg_selection" in source
    assert "selection_dt=jnp.inf" not in source


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


def test_short_leg_split_is_native_to_compiled_imex_source():
    source = DRIVER.read_text(encoding="utf-8")
    assert "short_leg_selection_dt=(" in source
    assert "model.apply_short_leg_implicit_material_step(" in source
    assert "solve_dt=gamma_dt" in source
    assert "selection_dt=dt" in source
    assert "full_imex_advance" in source
    assert "IMEX_SSP222_GAMMA" in source


def test_short_leg_handoff_rejects_poststep_rk4_and_requires_imex():
    driver = _driver_module()
    rk4 = _production_args(
        driver,
        "--parallel-boundary-pairing", "characteristic-sat",
        "--parallel-short-leg-treatment", "local-backward-euler",
    )
    with pytest.raises(ValueError, match="imex-ssp222"):
        driver._validate_flux_framework(rk4)

    imex_without_split = _production_args(
        driver,
        "--parallel-boundary-pairing", "characteristic-sat",
        "--time-integrator", "imex-ssp222",
    )
    with pytest.raises(ValueError, match="local-backward-euler"):
        driver._validate_flux_framework(imex_without_split)


def test_imex_ssp222_scalar_split_is_second_order_without_jit():
    driver = _driver_module()
    explicit_rate = -0.5
    implicit_rate = -4.0

    def advance(step_count: int) -> float:
        value = jnp.asarray(1.0, dtype=jnp.float64)
        dt = 0.5 / step_count

        def explicit(y):
            return explicit_rate * y

        def implicit(base, stage_dt):
            stage = base / (1.0 - stage_dt * implicit_rate)
            return stage, implicit_rate * stage

        with jax.disable_jit():
            for _ in range(step_count):
                value, _stages, _rates = driver._imex_ssp222_step(
                    value, dt, explicit, implicit
                )
        return float(value)

    exact = np.exp((explicit_rate + implicit_rate) * 0.5)
    error_20 = abs(advance(20) - exact)
    error_40 = abs(advance(40) - exact)
    assert error_20 / error_40 > 3.5


def test_frozen_rhs_replay_exposes_eager_no_outer_compile_mode():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    assert args.rhs_replay_execution == "auto"
    action = next(
        action for action in parser._actions
        if action.dest == "rhs_replay_execution"
    )
    assert tuple(action.choices) == ("auto", "compiled", "eager")
    source = DRIVER.read_text(encoding="utf-8")
    assert "with jax.disable_jit(rhs_replay_execution == \"eager\")" in source
    assert "else replay_sharded" in source


def test_short_work_auto_selects_eager_and_production_batches_compile():
    driver = _driver_module()
    assert driver._resolve_execution_mode("auto", work_items=1) == "eager"
    assert driver._resolve_execution_mode("auto", work_items=20) == "eager"
    assert driver._resolve_execution_mode("auto", work_items=99) == "eager"
    assert driver._resolve_execution_mode("auto", work_items=100) == "compiled"
    assert driver._resolve_execution_mode("auto", work_items=600) == "compiled"
    assert driver._resolve_execution_mode("eager", work_items=20) == "eager"
    assert driver._resolve_execution_mode("compiled", work_items=1) == "compiled"
    with pytest.raises(ValueError, match="positive work_items"):
        driver._resolve_execution_mode("auto", work_items=0)


def test_time_advance_exposes_true_eager_mode_and_auto_default():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    assert args.advance_execution == "auto"
    action = next(
        action for action in parser._actions if action.dest == "advance_execution"
    )
    assert tuple(action.choices) == ("auto", "compiled", "eager")
    source = DRIVER.read_text(encoding="utf-8")
    assert "with jax.disable_jit(advance_execution == \"eager\")" in source
    assert "compiled_advance = sharded_advance" in source


def test_eager_advance_keeps_cell_centered_setup_kernels_compiled():
    source = DRIVER.read_text(encoding="utf-8")
    assert "curvature_face_setup = jax.jit(" in source
    assert "reconstruct_phi = jax.jit(" in source
    assert "jax.device_put(\n            np.asarray(value" in source


def test_restart_phi_reuse_flag_is_not_shadowed_by_setup_kernel():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    run_full_eb = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_eb"
    )
    argument_names = {
        argument.arg
        for argument in (*run_full_eb.args.args, *run_full_eb.args.kwonlyargs)
    }
    nested_function_names = {
        node.name
        for node in run_full_eb.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "reconstruct_initial_phi" in argument_names
    assert "reconstruct_initial_phi" not in nested_function_names
    assert "reconstruct_initial_phi_kernel" in nested_function_names


def test_invariant_curvature_faces_round_trip_through_cell_channels():
    driver = _driver_module()
    layout = HaloLayout3D((3, 4, 5), halo_width=2)
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=jnp.arange(4 * 4 * 5, dtype=jnp.float64).reshape(4, 4, 5),
        y=(1000.0 + jnp.arange(3 * 5 * 5, dtype=jnp.float64)).reshape(3, 5, 5),
        z=(2000.0 + jnp.arange(3 * 4 * 6, dtype=jnp.float64)).reshape(3, 4, 6),
    )
    packed = driver._pack_curvature_face_coefficients(coefficients)
    assert packed.shape == (3, 4, 5, 6)
    recovered = driver._unpack_curvature_face_coefficients(packed, layout)
    for actual, expected in zip(recovered.axes, coefficients.axes, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_run_full_eb_reconstructs_phi_after_short_leg_implicit_step():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    run_full_eb = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_eb"
    )
    implicit = [
        node.lineno for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_short_leg_implicit_material_step"
    ]
    reconstruct = [
        node.lineno for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reconstruct_stage_phi"
    ]
    assert implicit and reconstruct
    assert max(implicit) < max(reconstruct)


def test_run_metadata_attributes_selectors_to_canonical_driver():
    source = DRIVER.read_text(encoding="utf-8")
    for option in (
        "parallel-boundary-pairing",
        "parallel-short-leg-treatment",
        "parallel-short-leg-cfl-limit",
        "parallel-short-leg-selection",
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
    assert '"parallel_characteristic_wall_law": str(args.parallel_characteristic_wall_law)' in source
    assert '"parallel_short_leg_selection": str(args.parallel_short_leg_selection)' in source
    assert (
        '"parallel_characteristic_wall_law_source": '
        '"simulate_hsx_blob.py:--parallel-characteristic-wall-law"'
    ) in source


def test_startup_announces_parallel_characteristic_wall_law():
    source = DRIVER.read_text(encoding="utf-8")
    assert "[simulation] parallel characteristic wall law:" in source
    assert "source=simulate_hsx_blob.py:--parallel-characteristic-wall-law" in source
    assert "mathematical" in source
    assert "unit modal" in source


def test_canonical_driver_contains_materialized_face_provenance_path():
    source = DRIVER.read_text(encoding="utf-8")
    assert "build_local_outgoing_fci_face_topology_from_geometry(" in source
    assert '"face_provenance_sha256"' in source
    assert "project_initial_staggered_velocities" in source
    assert "prolong_local_outgoing_fci_face_owner_field" in source
