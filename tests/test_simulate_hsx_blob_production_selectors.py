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
            *extra,
        )
    )


def test_canonical_driver_is_tracked_at_repository_root():
    assert DRIVER.is_file()
    assert DRIVER.parent.name == "DRBX"


def test_source_stage_times_match_integrator_and_cache_rk4_midpoint():
    driver = _driver_module()
    assert driver._explicit_source_stage_times("rk4", 0.25, 0.1) == (
        0.25,
        0.3,
        0.3,
        0.35,
    )
    assert driver._explicit_source_stage_times("imex-ssp222", 0.25, 0.1) == (
        0.25,
        0.35,
    )


def test_run_full_eb_source_hook_is_optional_and_stage_sharded():
    source = DRIVER.read_text()
    run_start = source.index("def run_full_eb(")
    run_end = source.index("def _validate_flux_framework", run_start)
    run_source = source[run_start:run_end]
    assert "source_evaluator: Callable[[float], FciDrbEBState] | None = None" in run_source
    assert "source_spec = P(None, \"x\", \"y\", \"z\")" in run_source
    assert "source_evaluator(float(stage_time))" not in run_source
    assert "source = source_evaluator(stage_key)" in run_source
    assert "source_owned=source_owned" in run_source
    assert "source_1" in run_source and "source_2" in run_source
    assert "stage_1, stage_1.phi, model, source_1" in run_source
    assert "stage_2, stage_2.phi, model, source_2" in run_source


def test_parser_production_selector_contract():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    driver._validate_flux_framework(args)
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
        "physical-boundary-state",
    )
    assert wall_law_action.default == "primitive-least-residual"
    selection_action = next(
        action for action in driver._build_parser()._actions
        if action.dest == "parallel_short_leg_selection"
    )
    assert tuple(selection_action.choices) == ("cfl", "all-physical-walls")
    assert selection_action.default == "cfl"
    assert all(
        action.dest != "curvature_wall_flux_closure"
        for action in driver._build_parser()._actions
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


def test_physical_boundary_state_wall_law_is_exported_for_no_flow_model(
    monkeypatch,
):
    driver = _driver_module()
    monkeypatch.delenv("DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW", raising=False)
    args = _production_args(
        driver,
        "--parallel-boundary-pairing",
        "characteristic-sat",
        "--parallel-characteristic-wall-law",
        "physical-boundary-state",
        "--parallel-velocity-wall-bc",
        "dirichlet-zero",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"] == (
        "physical-boundary-state"
    )
    assert driver.os.environ["DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"] == (
        "live-characteristic-physical-boundary-state"
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

    physical = driver._parallel_characteristic_wall_metadata(
        "physical-boundary-state"
    )
    assert physical["parallel_material_wall_flux_closure"] == (
        "live-characteristic-physical-boundary-state"
    )
    assert physical["parallel_characteristic_wall_equilibrium_reference"] is None
    assert physical["parallel_characteristic_wall_provenance"] == (
        "physical-face-trace-live-characteristic-split"
    )

@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (
            ("--flux-framework", "legacy"),
            "production-path",
        ),
        (
            (),
            "characteristic-sat",
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


def test_native_configuration_exports_only_live_short_leg_selectors():
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
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_TREATMENT"] == (
        "local-backward-euler"
    )
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT"] == "2.25"
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_SELECTION"] == "cfl"
    assert "DRBX_CURVATURE_SPLIT_SCHEME" not in driver.os.environ
    assert driver.os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] == "production-path"
    for name in (
        "DRBX_CURVATURE_RADIAL_ABLATION",
        "DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME",
    ):
        assert name not in driver.os.environ


@pytest.mark.parametrize(
    "override",
    (
        ("--parallel-short-leg-treatment", "explicit"),
        ("--parallel-boundary-pairing", "current-phi"),
    ),
)
def test_all_physical_walls_requires_production_be_configuration(override):
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


@pytest.mark.parametrize(
    "wall_law", ("primitive-least-residual", "energy-absorbing")
)
def test_all_physical_walls_exports_selection_without_inf_sentinel(wall_law):
    driver = _driver_module()
    args = _production_args(
        driver,
        "--parallel-short-leg-selection", "all-physical-walls",
        "--parallel-short-leg-treatment", "local-backward-euler",
        "--parallel-characteristic-wall-law", wall_law,
        "--parallel-boundary-pairing", "characteristic-sat",
        "--time-integrator", "imex-ssp222",
    )
    driver._validate_flux_framework(args)
    driver._configure_runtime_selectors(args)
    assert driver.os.environ["DRBX_PARALLEL_SHORT_LEG_SELECTION"] == (
        "all-physical-walls"
    )
    assert driver.os.environ["DRBX_PARALLEL_CHARACTERISTIC_WALL_LAW"] == wall_law
    source = DRIVER.read_text(encoding="utf-8")
    assert "parallel_short_leg_selection" in source
    assert "selection_dt=jnp.inf" not in source


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


def test_imex_explicit_source_times_are_second_order_for_nonautonomous_ode():
    """The explicit source partition uses the SSP222 abscissas ``t,t+dt``.

    This is a source-only ODE check, so no geometry or production model is
    involved.  The implicit callback is identically zero while the source
    callback consumes the two stage times supplied by the canonical helper.
    Evaluating the source at the implicit SDIRK abscissas would exercise a
    different ARK partition, even though their symmetry can also give a
    second-order quadrature in this degenerate source-only problem.
    """

    driver = _driver_module()
    final_time = 0.8

    def advance(step_count: int) -> float:
        value = jnp.asarray(0.0, dtype=jnp.float64)
        dt = final_time / step_count
        for step in range(step_count):
            start = step * dt
            source_times = iter(
                driver._explicit_source_stage_times("imex-ssp222", start, dt)
            )

            def explicit_rate(_state):
                stage_time = next(source_times)
                return jnp.asarray(stage_time * stage_time, dtype=jnp.float64)

            def implicit_stage(base, _stage_dt):
                return base, jnp.asarray(0.0, dtype=jnp.float64)

            value, _stages, _rates = driver._imex_ssp222_step(
                value, dt, explicit_rate, implicit_stage
            )
        return float(value)

    exact = final_time**3 / 3.0
    error_20 = abs(advance(20) - exact)
    error_40 = abs(advance(40) - exact)
    assert error_20 / error_40 > 3.5


def test_imex_nonautonomous_source_remains_second_order_with_implicit_split():
    """Exercise the production ARK helper with both split partitions active.

    For ``y=exp(t)``, split ``y' = lambda*y + (1-lambda)*exp(t)`` with the
    linear autonomous term implicit and the manufactured time-dependent
    source explicit.  This covers the cross-partition order conditions that
    the source-only regression above cannot exercise.
    """

    driver = _driver_module()
    final_time = 0.8
    implicit_rate = -4.0

    def advance(step_count: int, *, record_times: bool = False):
        value = jnp.asarray(1.0, dtype=jnp.float64)
        dt = final_time / step_count
        called_times: list[float] = []
        for step in range(step_count):
            start = step * dt
            source_times = iter(
                driver._explicit_source_stage_times("imex-ssp222", start, dt)
            )

            def explicit_source(_state):
                stage_time = next(source_times)
                if record_times:
                    called_times.append(stage_time)
                return jnp.asarray(
                    (1.0 - implicit_rate) * np.exp(stage_time),
                    dtype=jnp.float64,
                )

            def implicit_stage(base, stage_dt):
                stage = base / (1.0 - stage_dt * implicit_rate)
                return stage, implicit_rate * stage

            value, _stages, _rates = driver._imex_ssp222_step(
                value, dt, explicit_source, implicit_stage
            )
            with pytest.raises(StopIteration):
                next(source_times)
        return float(value), called_times

    _value, called_times = advance(4, record_times=True)
    dt = final_time / 4
    assert called_times == pytest.approx([
        time
        for step in range(4)
        for time in (step * dt, (step + 1) * dt)
    ])

    value_80, _ = advance(80)
    value_160, _ = advance(160)
    exact = np.exp(final_time)
    error_80 = abs(value_80 - exact)
    error_160 = abs(value_160 - exact)
    assert error_80 / error_160 > 3.5


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


def test_execution_mode_auto_supports_staged_short_imex_and_compiled_batches():
    driver = _driver_module()
    assert driver._resolve_execution_mode("auto", work_items=1) == "eager"
    assert driver._resolve_execution_mode("auto", work_items=20) == "eager"
    assert driver._resolve_execution_mode("auto", work_items=99) == "eager"
    assert driver._resolve_execution_mode(
        "auto", work_items=20, auto_short_mode="staged-compiled"
    ) == "staged-compiled"
    assert driver._resolve_execution_mode("auto", work_items=100) == "compiled"
    assert driver._resolve_execution_mode("auto", work_items=600) == "compiled"
    assert driver._resolve_execution_mode("eager", work_items=20) == "eager"
    assert driver._resolve_execution_mode("compiled", work_items=1) == "compiled"
    assert driver._resolve_execution_mode(
        "staged-compiled", work_items=1
    ) == "staged-compiled"
    with pytest.raises(ValueError, match="positive work_items"):
        driver._resolve_execution_mode("auto", work_items=0)
    with pytest.raises(ValueError, match="auto_short_mode"):
        driver._resolve_execution_mode(
            "auto", work_items=20, auto_short_mode="unknown"
        )


def test_time_advance_exposes_true_eager_mode_and_auto_default():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    assert args.advance_execution == "auto"
    action = next(
        action for action in parser._actions if action.dest == "advance_execution"
    )
    assert tuple(action.choices) == (
        "auto", "compiled", "staged-compiled", "eager"
    )
    source = DRIVER.read_text(encoding="utf-8")
    assert "with jax.disable_jit(advance_execution == \"eager\")" in source
    assert "compiled_advance = sharded_advance" in source
    assert "staged_implicit_sharded = jax.shard_map(" in source
    assert "staged_explicit_sharded = jax.shard_map(" in source
    assert "staged_phi_sharded = jax.shard_map(" in source
    assert "staged_finalize_sharded = jax.shard_map(" in source
    assert "def compile_staged_kernel(label: str, sharded_kernel" in source
    assert '"implicit+phi"' in source
    assert '"explicit-rhs"' in source
    assert '"standalone-phi"' in source
    assert '"stage-diagnostics"' in source
    # The staged device-side operations must retain the same SSP222 algebra
    # as _imex_ssp222_step; the canonical helper has numerical order tests
    # above, while these checks protect the production wiring.
    assert "stage_2_base_before_phi = current.axpy(" in source
    assert "weighted_rate = explicit_1.axpy(explicit_2, scale=1.0).axpy(" in source
    assert "next_state = current.axpy(weighted_rate, scale=dt_dynamic)" in source


def test_staged_selected_cell_audit_is_explicit_and_machine_readable():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(
        (
            "--staged-audit-cell", "45", "14", "17",
            "--staged-audit-cell", "46", "14", "17",
            "--staged-audit-output", "audit.npz",
            "--staged-audit-explicit-ablation", "curvature-parallel-material",
        )
    )
    assert args.staged_audit_cell == [[45, 14, 17], [46, 14, 17]]
    assert args.staged_audit_output == Path("audit.npz")
    assert args.staged_audit_explicit_ablation == "curvature-parallel-material"
    source = DRIVER.read_text(encoding="utf-8")
    assert "staged_explicit_term_audit_kernel" in source
    assert '"audit-explicit-term-lanes"' in source
    for closure_name in (
        "implicit_1_closure",
        "explicit_probe_closure",
        "explicit_ablation_closure",
        "explicit_term_closure",
        "curvature_component_closure",
        "parallel_material_component_closure",
        "stage_2_base_closure",
        "implicit_2_closure",
        "weighted_rate_closure",
        "final_closure",
    ):
        assert closure_name in source


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
    ):
        assert f'simulate_hsx_blob.py:--{option}' in source
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


def test_canonical_driver_uses_cell_centered_velocity_basis():
    source = DRIVER.read_text(encoding="utf-8")
    assert '"field_locations": {"Vi": "cell-center", "Ve": "cell-center"}' in source


def test_initial_owner_sparse_check_uses_current_two_argument_api():
    source = DRIVER.read_text(encoding="utf-8")
    assert "_assert_owner_sparse(initial_state, owner_host_geometry)" in source
    assert "_assert_owner_sparse(initial_state, owner_host_geometry, None)" not in source
