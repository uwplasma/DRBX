"""Low-cost tests for the standalone scalar HSX curvature driver."""

import ast
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "simulate_hsx_linearized_ti_curvature.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("hsx_linearized_ti_test", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_linearized_rhs_formula_and_zero_curvature_input():
    driver = _load_driver()
    theta = jnp.asarray([[1.0, -2.0]])
    curvature = jnp.asarray([[3.0, -6.0]])
    B = jnp.asarray([[2.0, 4.0]])
    result = driver.linearized_ti_rhs(theta, curvature, B, tau=1.5)
    np.testing.assert_allclose(result, -(10.0 * 1.5 / 3.0) * curvature / B)
    np.testing.assert_allclose(
        driver.linearized_ti_rhs(theta, jnp.zeros_like(theta), B, tau=1.0),
        0.0,
    )


def test_localized_initialization_is_deterministic_and_resolution_aware():
    driver = _load_driver()
    shape = (32, 32, 32)
    first = driver.initialize_perturbation(shape, initialization="corner", random_seed=7)
    second = driver.initialize_perturbation(shape, initialization="corner", random_seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.unravel_index(np.argmax(first), shape) == (30, 30, 13)
    coarse = driver.initialize_perturbation((16, 16, 16), initialization="corner")
    assert np.unravel_index(np.argmax(coarse), coarse.shape) == (15, 15, 6)
    random_a = driver.initialize_perturbation(shape, initialization="random", random_seed=3)
    random_b = driver.initialize_perturbation(shape, initialization="random", random_seed=3)
    np.testing.assert_array_equal(random_a, random_b)


def test_cli_has_runtime_time_controls_and_no_gmres_dependency():
    tree = ast.parse(DRIVER.read_text())
    parser_text = DRIVER.read_text()
    assert "--final-time" in parser_text
    assert "--num-steps" in parser_text
    assert "dt_runtime" in parser_text
    assert "jax.shard_map" in parser_text
    assert "GMRES" not in parser_text.split("def run", 1)[1].split("def main", 1)[0]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shard_map"
        for node in ast.walk(tree)
    )


def test_curvature_closure_parser_and_metadata_wiring():
    driver = _load_driver()
    parser = driver.build_parser()
    for mode in ("central-neumann", "upwind-neumann", "upwind-equilibrium"):
        assert parser.parse_args(["--curvature-flux-closure", mode]).curvature_flux_closure == mode
    assert parser.parse_args([]).curvature_flux_closure == "central-neumann"
    source = DRIVER.read_text()
    assert "curvature_flux_closure=args.curvature_flux_closure" in source
    assert '"curvature_flux_closure": args.curvature_flux_closure' in source


def test_upwind_axis_face_states_have_n_plus_one_faces_and_correct_orientation():
    from drbx.geometry import HaloLayout3D
    from drbx.native.fci_boundaries import BC_NEUMANN, LocalStencil1D
    from drbx.native.fci_operators import _local_axis_upwind_face_values_from_stencil

    layout = HaloLayout3D((3, 1, 1), 1)
    shape = (3, 1, 1)
    stencil = LocalStencil1D(
        center=jnp.asarray([[[10.0]], [[20.0]], [[30.0]]]),
        minus=jnp.asarray([[[1.0]], [[2.0]], [[3.0]]]),
        plus=jnp.asarray([[[11.0]], [[21.0]], [[31.0]]]),
        dx_min=jnp.ones(shape), dx_plus=jnp.ones(shape),
    )
    kind = jnp.zeros(layout.face_control_shape(0), dtype=jnp.int32).at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN)
    value = jnp.zeros(layout.face_control_shape(0))
    mask = jnp.zeros(layout.face_control_shape(0), dtype=bool).at[0].set(True).at[-1].set(True)

    positive = _local_axis_upwind_face_values_from_stencil(
        stencil, jnp.ones(layout.face_control_shape(0)), axis=0,
        axis_kind=kind, axis_value=value, axis_mask=mask,
        axis_regular_axes=(False, False, False), equilibrium_inflow=False,
    )
    negative = _local_axis_upwind_face_values_from_stencil(
        stencil, -jnp.ones(layout.face_control_shape(0)), axis=0,
        axis_kind=kind, axis_value=value, axis_mask=mask,
        axis_regular_axes=(False, False, False), equilibrium_inflow=False,
    )
    np.testing.assert_allclose(positive[:, 0, 0], [5.5, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(negative[:, 0, 0], [10.0, 20.0, 30.0, 30.5])

    equilibrium = _local_axis_upwind_face_values_from_stencil(
        stencil, jnp.ones(layout.face_control_shape(0)), axis=0,
        axis_kind=kind, axis_value=value, axis_mask=mask,
        axis_regular_axes=(False, False, False), equilibrium_inflow=True,
    )
    np.testing.assert_allclose(equilibrium[:, 0, 0], [0.0, 10.0, 20.0, 30.0])
