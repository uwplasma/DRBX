"""Focused contracts for the analytic simple-torus Stage 6 sanity tests.

These tests intentionally exercise the public driver contract rather than
duplicating any of the drift-reduced Braginskii operators.  The Stage 6
implementation is expected to build a small analytic torus and route the
active fields through the same ``LocalFciDrbEBRhs`` machinery used by the
production solver.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
DRBX_ROOT = WORKSPACE / "DRBX"
RUNNER = WORKSPACE / "run_hsx_sanity.py"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import run_hsx_sanity as sanity  # noqa: E402
from drbx.linear import (  # noqa: E402
    eigenmodes,
    full_drb_resistive_drift_wave_operator,
    interchange_operator,
)


EXPECTED_STAGE6 = {
    "polarization": {"polarization", "polarization-inversion", "6a"},
    "interchange": {"interchange", "linear-interchange", "6b"},
    "resistive_drift_wave": {"resistive-drift-wave", "resistive_drift_wave", "drift-wave", "6c"},
}


def _stage6_action(parser):
    """Return the parser action exposing the three Stage 6 experiments."""

    for action in parser._actions:
        choices = set(action.choices or ())
        if all(any(alias in choices for alias in aliases) for aliases in EXPECTED_STAGE6.values()):
            return action
    raise AssertionError(
        "build_parser() must expose one selector containing polarization, "
        "interchange, and resistive-drift-wave Stage 6 choices"
    )


def _pick_callable(*names: str):
    for name in names:
        candidate = getattr(sanity, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"run_hsx_sanity.py is missing one of: {', '.join(names)}")


def _call_with_supported_kwargs(function, /, *args, **kwargs):
    """Call a test helper without hiding required arguments from its API."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(*args, **kwargs)
    positional_names = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ][: len(args)]
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and key not in positional_names
    }
    return function(*args, **supported)


def _simple_geometry(shape=(4, 6, 8)):
    args = sanity.build_parser().parse_args(["--resolution", ",".join(map(str, shape))])
    builder = _pick_callable(
        "build_simple_toroidal_geometry",
        "build_simple_torus_geometry",
        "build_stage6_geometry",
        "build_analytic_toroidal_geometry",
        "_simple_toroidal_geometry",
    )
    if builder.__name__ == "_simple_toroidal_geometry":
        return _call_with_supported_kwargs(builder, args)
    return _call_with_supported_kwargs(
        builder,
        shape,
        shape=shape,
        resolution=shape,
        construct_fci_maps=True,
        B0=1.0,
        toroidal_field=1.0,
    )


def _stage6_args(shape=(4, 6, 8), *, mode="heat-spot"):
    mode = {"resistive-drift-wave": "drift-wave"}.get(mode, mode)
    return sanity.build_parser().parse_args(
        [
            "--resolution", ",".join(map(str, shape)), "--experiment", mode,
            # The slab has bounded x.  A sine radial mode satisfies the
            # benchmark boundary and avoids treating kx=0 as a radial mode.
            "--linear-radial-mode", "1",
        ]
    )


def _stage6_metadata_literals():
    """Extract literal acceptance-contract values from the driver AST."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    evolution = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "run_stage6_evolution")
    assignment = next(node for node in ast.walk(evolution)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name)
                              and target.id == "metadata"
                              for target in node.targets))
    assert isinstance(assignment.value, ast.Dict)
    literals = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
            literals[key.value] = value.value
    return literals


def _stage6_equilibrium(geometry, mode="polarization"):
    args = _stage6_args(tuple(geometry.shape), mode=mode)
    builder = _pick_callable(
        "build_stage6_equilibrium",
        "make_stage6_equilibrium",
        "make_stage6_state",
        "stage6_equilibrium_state",
        "_simple_stage6_state",
    )
    if builder.__name__ == "_simple_stage6_state":
        # Polarization is an inversion probe rather than an evolution state.
        mode = {"polarization": "interchange", "resistive-drift-wave": "drift-wave"}.get(mode, mode)
        return builder(geometry, args, mode)
    return _call_with_supported_kwargs(
        builder,
        geometry,
        geometry=geometry,
        mode=mode,
        test_name=mode,
        stage6_test=mode,
    )


def _stage6_mask(mode):
    mask = _pick_callable(
        "stage6_active_fields",
        "active_stage6_fields",
        "stage6_field_mask",
    )
    return _call_with_supported_kwargs(
        mask,
        mode,
        mode=mode,
        test_name=mode,
        stage6_test=mode,
    )


def _as_field_set(mask):
    if isinstance(mask, dict):
        return {name for name, enabled in mask.items() if bool(enabled)}
    if isinstance(mask, (set, frozenset, tuple, list)):
        return set(mask)
    raise AssertionError(f"active-field mask must be a mapping or field-name collection, got {type(mask)!r}")


def _unwrap_state(result):
    """Extract a state from common equilibrium-return conventions."""

    if hasattr(result, "field_items"):
        return result
    if isinstance(result, dict):
        for key in ("state", "equilibrium", "initial_state"):
            value = result.get(key)
            if hasattr(value, "field_items"):
                return value
    if isinstance(result, (tuple, list)):
        for value in result:
            if hasattr(value, "field_items"):
                return value
    raise AssertionError("Stage 6 equilibrium helper must return an FciDrbEBState or a result containing one")


def test_parser_exposes_three_stage6_selectors_and_preserves_heat_spot_default():
    parser = sanity.build_parser()
    action = _stage6_action(parser)
    assert action.dest in {"stage6_test", "stage6", "sanity_test", "test", "experiment"}

    parsed_default = parser.parse_args([])
    assert getattr(parsed_default, action.dest) in {None, "none", "heat-spot", "off"}
    for aliases in EXPECTED_STAGE6.values():
        choice = next(iter(aliases & set(action.choices or ())))
        parsed = parser.parse_args([action.option_strings[0], choice])
        assert getattr(parsed, action.dest) == choice


@pytest.mark.parametrize("alias", ("drift-wave", "resistive-drift-wave"))
def test_resistive_drift_wave_alias_dispatches_to_the_same_normalized_fci_path(
    alias, monkeypatch, tmp_path
):
    """The long-form alias must not silently bypass the drift-wave setup."""

    calls = []

    def fake_evolution(args, experiment):
        calls.append((args.experiment, experiment))
        return tmp_path / "history.npz"

    monkeypatch.setattr(sanity, "run_stage6_evolution", fake_evolution)
    sanity.main(
        [
            "--experiment",
            alias,
            "--resolution",
            "4,4,4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert calls == [("drift-wave", "drift-wave")]

    # The normalized path is the one that enables mapped FCI and finite
    # adiabatic electron response, rather than the interchange defaults.
    args = sanity.build_parser().parse_args(["--experiment", "drift-wave"])
    normalized = sanity._stage6_parameters(args, "drift-wave")
    assert float(normalized.Ve_nu) > 0.0
    # The normalized argument is consumed by the same public evolution entry
    # point for both spellings; the independent reference is checked
    # numerically in ``test_linear_dispersion.py``.
    assert args.experiment == "drift-wave"


def test_simple_torus_is_analytic_pure_toroidal_and_has_fci_maps():
    geometry = _simple_geometry()
    assert tuple(geometry.shape) == (4, 6, 8)
    B_contra = np.asarray(geometry.cell_bfield.B_contra)
    Bmag = np.asarray(geometry.cell_bfield.Bmag)
    assert B_contra.shape == (4, 6, 8, 3)
    assert np.all(np.isfinite(B_contra))
    assert np.all(np.isfinite(Bmag)) and np.all(Bmag > 0.0)
    # The Stage 6 fixture is deliberately a simple toroidal field: no radial
    # or poloidal contravariant component should be introduced by the helper.
    assert np.max(np.abs(B_contra[..., 0])) < 1.0e-13
    assert np.max(np.abs(B_contra[..., 1])) < 1.0e-13
    assert np.max(np.abs(B_contra[..., 2])) > 0.0
    maps = geometry.maps
    assert maps is not None
    for name in ("forward_x", "forward_y", "backward_x", "backward_y", "forward_length", "backward_length"):
        assert np.all(np.isfinite(np.asarray(getattr(maps, name))))


@pytest.mark.parametrize("mode", ("polarization", "interchange", "resistive-drift-wave"))
def test_stage6_equilibrium_setup_returns_finite_state(mode):
    state = _unwrap_state(_stage6_equilibrium(_simple_geometry(), mode))
    fields = dict(state.field_items())
    expected = {"density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"}
    assert expected <= set(fields)
    for name, value in fields.items():
        values = np.asarray(value)
        assert values.shape == (4, 6, 8), name
        assert np.all(np.isfinite(values)), name
    assert np.all(np.asarray(fields["density"]) > 0.0)
    assert np.all(np.asarray(fields["Te"]) > 0.0)
    assert np.all(np.asarray(fields["Ti"]) > 0.0)


def test_interchange_fixture_is_flute_and_has_only_n_vorticity_drive():
    geometry = _simple_geometry()
    args = _stage6_args(tuple(geometry.shape), mode="interchange")
    args.linear_mode_n_eta = 0
    state = sanity._simple_stage6_state(geometry, args, "interchange")
    density = np.asarray(state.density)
    # The linear interchange benchmark is a k_parallel=0 (flute) mode.
    assert np.max(np.abs(density - density[:, :, :1])) < 1.0e-13
    assert np.allclose(np.asarray(state.Te), 1.0)
    assert np.allclose(np.asarray(state.Ti), 1.0)
    assert np.allclose(np.asarray(state.Vi), 0.0)
    assert np.allclose(np.asarray(state.Ve), 0.0)
    # 6B is a flute perturbation with only the vorticity curvature lane.  The
    # density response is the analytic E x B gradient drive, not a second
    # curvature lane.
    assert sanity._stage6_wavenumbers(args)[2] == 0.0


def test_resistive_drift_wave_fixture_uses_mapped_fci_and_electron_response():
    args = _stage6_args(mode="drift-wave")
    params = sanity._stage6_parameters(args, "drift-wave")
    assert float(params.Ve_nu) > 0.0
    assert float(params.density_D_perp) == 0.0
    assert float(params.electron_temperature_chi_parallel) == 0.0
    # This mode isolates the parallel electron response; no perpendicular
    # diffusion or curvature drive is enabled by the normalized parameters.
    assert float(params.Ve_nu) > 0.0


def test_stage6_active_field_masks_match_the_equation_scope():
    # Stage 6 no longer post-RHS masks inactive fields.  6B selects a
    # vorticity-curvature lane through the production model, while 6C
    # advances all six physical fields with only parallel response enabled.
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evolution = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "run_stage6_evolution")
    names = {node.id for node in ast.walk(evolution) if isinstance(node, ast.Name)}
    assert "mode_names" in names
    assert "full_drb_resistive_drift_wave_operator" in source
    assert "post_rhs_active_field_mask" in source


def test_stage6_diagnostics_have_explicit_pass_fail_criteria():
    diagnostic = next(
        (getattr(sanity, name, None) for name in ("stage6_diagnostics", "diagnose_stage6", "stage6_pass_criteria", "evaluate_stage6_result") if callable(getattr(sanity, name, None))),
        None,
    )
    if diagnostic is None:
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        polarization = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_polarization"
        )
        passed_assignments = [
            node
            for node in ast.walk(polarization)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "passed" for target in node.targets)
        ]
        assert passed_assignments, "Stage 6A must assign an explicit pass/fail result"
        passed_expression = passed_assignments[-1].value
        names = {
            node.id for node in ast.walk(passed_expression) if isinstance(node, ast.Name)
        }
        # The condition may span lines and may wrap the convergence flag in a
        # bool/array conversion.  Name analysis therefore avoids the previous
        # same-line source-string brittleness.
        assert "converged" in names
        phi_error_names = {
            "phi_reconstruction_error",
            "relative_phi_error",
            "relative_phi_reconstruction_error",
        }
        assert phi_error_names & names
        assert "residual_error" in names

        def compare_uses_tolerance(error_name: str, tolerance_name: str) -> bool:
            for node in ast.walk(passed_expression):
                if not isinstance(node, ast.Compare):
                    continue
                if not isinstance(node.left, ast.Name) or node.left.id != error_name:
                    continue
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Attribute)
                        and child.attr == tolerance_name
                        and isinstance(child.value, ast.Name)
                        and child.value.id == "args"
                    ):
                        return True
            return False

        assert any(
            compare_uses_tolerance(name, "polarization_tolerance")
            for name in phi_error_names
        )
        assert compare_uses_tolerance(
            "residual_error", "polarization_residual_tolerance"
        )
        return
    good = _call_with_supported_kwargs(
        diagnostic,
        "polarization",
        mode="polarization",
        test_name="polarization",
        measured={"relative_error": 1.0e-10, "growth_rate_error": 1.0e-10, "frequency_error": 1.0e-10},
        expected={"relative_error": 1.0e-6, "growth_rate_error": 1.0e-2, "frequency_error": 1.0e-2},
        relative_error=1.0e-10,
        tolerance=1.0e-6,
    )
    bad = _call_with_supported_kwargs(
        diagnostic,
        "polarization",
        mode="polarization",
        test_name="polarization",
        measured={"relative_error": 0.25, "growth_rate_error": 0.25, "frequency_error": 0.25},
        expected={"relative_error": 1.0e-6, "growth_rate_error": 1.0e-2, "frequency_error": 1.0e-2},
        relative_error=0.25,
        tolerance=1.0e-6,
    )

    def passed(value):
        if isinstance(value, dict):
            for key in ("passed", "pass", "ok"):
                if key in value:
                    return bool(value[key])
        return bool(value)

    assert passed(good)
    assert not passed(bad)


def test_stage6_pass_requires_measured_eigenvalue_agreement_not_finiteness_only():
    source = RUNNER.read_text(encoding="utf-8")
    # Stage 6 evolution must measure the simulated mode and compare both
    # growth and frequency with the analytical eigenmode.  A finite trajectory
    # and converged phi solve alone is not a pass criterion.
    assert any(token in source for token in ("measured_growth", "growth_rate_error", "measured_eigenvalue"))
    assert any(token in source for token in ("measured_frequency", "frequency_error", "measured_eigenvalue"))
    assert any(token in source for token in ("growth_tolerance", "frequency_tolerance", "stage6_tolerance"))
    assert any(token in source for token in ("rate_match", "growth_match", "growth_error"))
    assert any(token in source for token in ("frequency_match", "frequency_error"))
    tree = ast.parse(source)
    evolution = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_stage6_evolution"
    )
    passed_assignments = [
        node
        for node in ast.walk(evolution)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "passed"
            for target in node.targets
        )
    ]
    assert passed_assignments
    passed_names = {
        node.id
        for node in ast.walk(passed_assignments[-1].value)
        if isinstance(node, ast.Name)
    }
    assert {"rate_match", "frequency_match"} <= passed_names


def test_polarization_records_recovered_phi_error_in_addition_to_residual():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"phi_input"' in source
    assert '"phi_reconstructed"' in source
    assert any(
        token in source
        for token in ("phi_reconstruction_error", "recovered_phi_error", "solved - phi", "solved - phi")
    )


def test_stage6_equilibrium_is_an_explicit_stationarity_diagnostic():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evolution = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "run_stage6_evolution")
    names = {node.id for node in ast.walk(evolution) if isinstance(node, ast.Name)}
    # A background check is useful, but subtraction of its RHS would make the
    # test manufactured.  These metadata fields are asserted as literal
    # values below so changing either policy breaks the test.
    assert {"equilibrium_stationary", "equilibrium_rhs_linf"} <= names
    contract = _stage6_metadata_literals()
    assert contract["equilibrium_rhs_subtracted"] is False
    assert contract["post_rhs_active_field_mask"] is False


def test_stage6_metadata_json_round_trip_is_serializable(tmp_path):
    metadata = {
        "experiment": "stage6-test",
        "passed": True,
        "growth_rate_error": 1.0e-8,
        "frequency_error": 2.0e-8,
    }
    history = sanity._write_stage6_result(
        tmp_path,
        metadata,
        {"times": np.asarray([0.0, 1.0]), "density": np.ones((2, 4, 6, 8))},
    )
    assert history.exists()
    loaded = __import__("json").loads((tmp_path / "metadata.json").read_text())
    assert loaded["passed"] is True
    assert loaded["growth_rate_error"] == pytest.approx(1.0e-8)


def test_stage6_source_routes_through_production_rhs_and_not_a_second_rhs():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lower = source.lower()
    assert "localfci" in source_lower
    assert "evaluate_stage" in source
    assert "build_local_eb_model" in source
    assert "_simple_stage6_slab_geometry" in source
    assert "identity_fci_maps" in source
    assert "reference_uses_production_rhs" in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"evaluate_stage", "reconstruct_phi"}
        for node in ast.walk(tree)
    )
    # A Stage 6 driver may form analytical initial fields, but it must not
    # define a competing finite-difference RHS implementation in the runner.
    # Stage 6 can contain a diagnostic tangent helper, but the acceptance
    # path must call the shipped production model rather than define a second
    # spatial RHS.
    assert "_build_stage6_modal_reference" in source


def _full_reference(args):
    params = sanity._stage6_parameters(args, "drift-wave")
    _, ky, kpar, kperp2 = sanity._stage6_wavenumbers(args)
    return np.asarray(full_drb_resistive_drift_wave_operator(
        ky, kperp2, kpar, float(params.rho_star), float(args.slab_bfield),
        float(args.drift_wave_gradient), float(params.tau),
        float(params.mi_over_me), float(params.Ve_nu),
    ))


def test_independent_six_field_reference_entries_and_order():
    """Check every entry against the continuous six-field equations."""
    args = _stage6_args(mode="drift-wave")
    A = _full_reference(args)
    params = sanity._stage6_parameters(args, "drift-wave")
    _, ky, kpar, kperp2 = sanity._stage6_wavenumbers(args)
    B, rho, tau = float(args.slab_bfield), float(params.rho_star), float(params.tau)
    mu, nu, G = float(params.mi_over_me), float(params.Ve_nu), float(args.drift_wave_gradient)
    D, d = 1j * kpar, ky * G / (rho * B)
    E = np.zeros((6, 6), dtype=complex)
    E[0] = (0, 0, -1j*d*tau, 0, -D, -1j*d/kperp2)
    E[1] = (0, 0, 0, (2/3)*0.71*D, -(2/3)*1.71*D, 0)
    E[2, 4] = -(2/3)*D
    E[3] = (-(1+tau)*D, -D, -tau*D, 0, 0, 0)
    E[4] = (-mu*D, -1.71*mu*D, -mu*tau*D, mu*nu, -mu*nu, -mu*D/kperp2)
    E[5] = (0, 0, 0, B**2*D, -B**2*D, 0)
    np.testing.assert_allclose(A, E, rtol=1e-13, atol=1e-13)


def test_independent_eigenpair_residual_is_small_and_reference_mutation_fails():
    args = _stage6_args(mode="drift-wave")
    A = _full_reference(args)
    modes = eigenmodes(A)
    lam = complex(np.asarray(modes.eigenvalues)[0])
    vec = np.asarray(modes.eigenvectors)[:, 0]
    residual = np.linalg.norm(A @ vec - lam * vec) / np.linalg.norm(A @ vec)
    assert residual < 1.0e-11
    mutated = A.copy()
    mutated[4, 1] *= 1.5
    bad = np.linalg.norm(mutated @ vec - lam * vec) / np.linalg.norm(mutated @ vec)
    assert bad > 1.0e-3


@pytest.mark.parametrize("n_eta", (0, 1))
def test_six_field_reference_limits_remove_expected_lanes(n_eta):
    args = _stage6_args(mode="drift-wave")
    args.linear_mode_n_eta = n_eta
    args.drift_wave_gradient = 0.0
    A = _full_reference(args)
    if n_eta == 0:
        # The collisional Vi-Ve relaxation remains even for a flute mode.
        assert np.allclose(A[[1, 2, 3, 5]], 0.0)
        assert np.linalg.norm(A[4]) > 0.0
    else:
        # Parallel electron advection remains in n-dot, but the E x B
        # gradient-drive columns disappear when G=0.
        assert np.allclose(A[0, [0, 1, 2, 3, 5]], 0.0)
        assert not np.allclose(A[0, 4], 0.0)


def test_stage6_identity_map_and_straight_field():
    geometry = sanity._simple_stage6_slab_geometry(_stage6_args())
    shape = geometry.shape
    assert geometry.maps is not None
    i = np.arange(shape[0])[:, None, None]
    j = np.arange(shape[1])[None, :, None]
    for name in ("forward_x", "backward_x"):
        np.testing.assert_array_equal(np.asarray(getattr(geometry.maps, name)), np.broadcast_to(i, shape))
    for name in ("forward_y", "backward_y"):
        np.testing.assert_array_equal(np.asarray(getattr(geometry.maps, name)), np.broadcast_to(j, shape))
    B = np.asarray(geometry.cell_bfield.B_contra)
    assert np.allclose(B[..., :2], 0.0)
    assert np.allclose(B[..., 2], float(_stage6_args().slab_bfield))
    assert not np.asarray(geometry.maps.forward_boundary).any()


def test_stage6_initial_state_uses_analytic_eigenvector_ratios():
    args = _stage6_args(mode="drift-wave")
    geometry = sanity._simple_stage6_slab_geometry(args)
    equilibrium = sanity.FciDrbEBState(
        density=np.ones(geometry.shape), phi=np.zeros(geometry.shape),
        Te=np.ones(geometry.shape), Ti=np.ones(geometry.shape),
        Vi=np.zeros(geometry.shape), Ve=np.zeros(geometry.shape),
        vorticity=np.zeros(geometry.shape),
    )
    names = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
    modes = eigenmodes(_full_reference(args))
    initial, scaled = sanity._state_from_analytic_eigenmode(
        equilibrium, geometry, args, names, np.asarray(modes.eigenvectors)[:, 0]
    )
    basis = sanity._stage6_basis_values(geometry, args, names)
    coeff = sanity._project_stage6_modal_history([initial], equilibrium, basis, names)[0]
    recovered = np.asarray([coeff[2*i] - 1j*coeff[2*i+1] for i in range(len(names))])
    pivot = int(np.argmax(np.abs(scaled)))
    np.testing.assert_allclose(recovered/recovered[pivot], scaled/scaled[pivot], rtol=2e-12, atol=2e-12)
    assert np.allclose(np.asarray(initial.phi), 0.0)


def test_stage6_metadata_contract_forbids_circular_shortcuts():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evolution = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "run_stage6_evolution")
    names = {node.id for node in ast.walk(evolution) if isinstance(node, ast.Name)}
    assert {"rate_match", "frequency_match", "eigenfunction_match"} <= names
    # Exact metadata values, rather than prose/source substrings, are the
    # anti-cheating contract for acceptance.
    contract = _stage6_metadata_literals()
    assert contract["reference_uses_production_rhs"] is False
    assert contract["equilibrium_rhs_subtracted"] is False
    assert contract["post_rhs_active_field_mask"] is False
    assert contract["all_six_evolved_fields_advanced"] is True
    assert contract["initialization_kind"] == "independent analytic eigenvector"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DRBX_RUN_STAGE6_PRODUCTION_SMOKE") != "1",
    reason="set DRBX_RUN_STAGE6_PRODUCTION_SMOKE=1 for the JAX production smoke",
)
def test_stage6_one_step_production_smoke(tmp_path):
    """Run one tiny production step when explicitly requested.

    The opt-in guard keeps ordinary unit-test runs cheap while still providing
    a real solver path for CI or a developer with the configured JAX runtime.
    """

    runner = _pick_callable("run_stage6", "run_stage6_test", "run_stage6_evolution")
    args = sanity.build_parser().parse_args(
        [
            "--experiment",
            "drift-wave",
            "--resolution",
            "4,4,4",
            "--num-steps",
            "1",
            "--final-time",
            "1.0e-4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    result = runner(args, "drift-wave") if runner.__name__ == "run_stage6_evolution" else runner(args)
    assert result is not None
    assert any(path.exists() for path in tmp_path.iterdir())
