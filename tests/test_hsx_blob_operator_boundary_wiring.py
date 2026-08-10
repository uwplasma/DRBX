from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


DRIVER = Path(__file__).parents[2] / "simulate_hsx_blob.py"


def _source():
    return DRIVER.read_text()


def _load_driver():
    name = "_test_simulate_hsx_blob_operator_boundaries"
    spec = importlib.util.spec_from_file_location(name, DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_driver_builds_layer_paired_two_halo_logical_neumann_weights():
    source = _source()
    tree = ast.parse(source)
    assert "paired_neumann_weights" in source
    assert "owned_weights=jnp.eye(h, dtype=jnp.float64)" in source
    assert "neumann_lower_weights" in source
    assert "neumann_upper_weights" in source
    assert "ghost - owner" in source

    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "dirichlet_ghost_weights"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    owned_weights = next(
        keyword.value for keyword in assignment.value.keywords
        if keyword.arg == "owned_weights"
    )
    assert isinstance(owned_weights, ast.UnaryOp)
    assert isinstance(owned_weights.op, ast.USub)
    assert isinstance(owned_weights.operand, ast.Call)
    assert isinstance(owned_weights.operand.func, ast.Attribute)
    assert owned_weights.operand.func.attr == "eye"
    compile(tree, str(DRIVER), "exec")


def test_build_local_model_materializes_layer_paired_h2_weights(monkeypatch):
    driver = _load_driver()
    h = 2
    n = 4
    centers = np.asarray((-1.7, -0.6, 0.5, 1.7, 3.1, 4.8, 6.8, 9.1))
    axis_grid = lambda: SimpleNamespace(centers_halo=centers)
    geometry = SimpleNamespace(
        layout=SimpleNamespace(halo_width=h),
        owned_shape=(n, n, n),
        grid=SimpleNamespace(x=axis_grid(), y=axis_grid(), z=axis_grid()),
    )
    domain = SimpleNamespace(
        mesh_axis_names=(None, None, None),
        periodic_axes=(False, True, True),
        axis_regular_axes=(False, False, False),
    )
    monkeypatch.setattr(
        driver, "build_local_perp_laplacian_face_projectors",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        driver, "LocalFciDrbEBRhs", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    model = driver.build_local_eb_model(
        geometry,
        domain,
        SimpleNamespace(phi_inversion_regularization=0.0),
        gmres_target_tolerance=1.0e-8,
        gmres_acceptance_tolerance=1.0e-8,
        gmres_max_iterations=4,
        curvature_scheme="disabled",
        neumann_ghost_scheme="logical",
    )
    filler = model.physical_ghost_filler
    dirichlet = filler.dirichlet[0]
    np.testing.assert_array_equal(dirichlet.owned_weights, -np.eye(h))
    owners = np.asarray((10.0, 20.0))
    np.testing.assert_allclose(
        np.asarray(dirichlet.owned_weights) @ owners
        + np.asarray(dirichlet.bc_weights) * 3.0,
        np.asarray((-4.0, -14.0)),
    )

    lower = filler.neumann_lower[0]
    upper = filler.neumann_upper[0]
    np.testing.assert_array_equal(lower.owned_weights, np.eye(h))
    np.testing.assert_array_equal(upper.owned_weights, np.eye(h))
    np.testing.assert_allclose(lower.bc_weights, (-1.1, -3.4))
    np.testing.assert_allclose(upper.bc_weights, (2.0, 6.0))


def test_driver_defaults_to_central_operator_aware_curvature_and_preserves_toroidal_axis():
    source = _source()
    assert 'curvature_inflow_closure: str = "central"' in source
    assert 'default="central"' in source
    assert "axis_regular_axes[0]" in source
    assert "radial_axis_lower_regular=True" in source
    assert "fill_periodic_axes=domain.periodic_axes" in source
