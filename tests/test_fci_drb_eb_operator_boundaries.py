from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from drbx.geometry import HaloLayout3D, LocalDomain3D, LocalFciGeometry3D, ShardSpec3D
from drbx.native import (
    LocalFciDrbEBOperatorBoundaryBundle,
    build_local_fci_drb_eb_operator_boundary_bundle as public_builder,
)
from drbx.geometry.fci_geometry import SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC
from drbx.native.fci_boundaries import BC_DIRICHLET, LocalBoundaryFaceBC3D
from drbx.native.fci_drb_EB_rhs import (
    FciDrbEBState,
    LocalFciDrbEBFaceBCBundle,
    build_local_fci_drb_eb_operator_boundary_bundle,
)


RHS = Path(__file__).parents[1] / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"


def test_operator_boundary_bundle_api_is_publicly_importable():
    assert LocalFciDrbEBOperatorBoundaryBundle.__name__ == "LocalFciDrbEBOperatorBoundaryBundle"
    assert public_builder.__name__ == "build_local_fci_drb_eb_operator_boundary_bundle"


def _domain_and_geometry(layout):
    h = layout.halo_width
    n = layout.owned_shape
    centers = jnp.arange(-h, n[0] + h, dtype=jnp.float64) + 0.5
    grid = lambda c: SimpleNamespace(
        centers_halo=jnp.asarray(c, dtype=jnp.float64),
        faces_halo=jnp.arange(len(c) + 1, dtype=jnp.float64) - h,
    )
    geometry = object.__new__(LocalFciGeometry3D)
    object.__setattr__(geometry, "layout", layout)
    object.__setattr__(geometry, "grid", SimpleNamespace(
        x=grid(centers),
        y=grid(jnp.arange(-h, n[1] + h, dtype=jnp.float64) + 0.5),
        z=grid(jnp.arange(-h, n[2] + h, dtype=jnp.float64) + 0.5),
    ))
    sides = (SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC)
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=n, owned_start=(0, 0, 0), owned_stop=n,
            shard_index=(0, 0, 0), shard_counts=(1, 1, 1),
            periodic_axes=(False, True, True), axis_regular_axes=(False, False, False),
            side_kind_lower=sides, side_kind_upper=sides, halo_width=h,
        ),
        layout=layout, mesh_axis_names=(None, None, None),
    )
    return domain, geometry


def test_derived_operator_traces_are_products_of_primitive_wall_traces():
    layout = HaloLayout3D((4, 2, 2), 2)
    domain, geometry = _domain_and_geometry(layout)
    x = jnp.arange(layout.cell_halo_shape[0], dtype=jnp.float64)[:, None, None]
    fields = [
        2.0 + x, 0.1 * x, 3.0 + x, 4.0 + x,
        5.0 + x, 0.25 * x, 7.0 + x,
    ]
    state = FciDrbEBState(*(jnp.broadcast_to(f, layout.cell_halo_shape) for f in fields))
    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=bc.value_x.at[0].set(11.0).at[-1].set(13.0),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    face_bc = LocalFciDrbEBFaceBCBundle(
        density=bc, phi=bc, Te=bc, Ti=bc, Vi=bc, Ve=bc, vorticity=bc
    )
    bundle = build_local_fci_drb_eb_operator_boundary_bundle(
        state, geometry, domain, face_bc, tau=2.0
    )
    np.testing.assert_allclose(
        bundle.density_flux.value_x[0],
        bundle.density.value_x[0] * bundle.Ve.value_x[0],
    )
    np.testing.assert_allclose(
        bundle.pressure.value_x[0],
        bundle.density.value_x[0]
        * (bundle.Te.value_x[0] + 2.0 * bundle.Ti.value_x[0]),
    )
    ghost_product_average = 0.5 * (
        state.density[1, 1, 1] * state.Ve[1, 1, 1]
        + state.density[2, 1, 1] * state.Ve[2, 1, 1]
    )
    assert not np.isclose(float(bundle.density_flux.value_x[0, 0, 0]), float(ghost_product_average))


def test_production_conservative_calls_carry_operand_traces_and_div_b_does_not():
    tree = ast.parse(RHS.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"local_parallel_flux_div_op", "local_grad_parallel_op_conservative"}
    ]
    assert calls
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        if call.func.id == "local_parallel_flux_div_op" and any(
            isinstance(arg, ast.Name) and arg.id == "unit_stencil" for arg in call.args
        ):
            assert "boundary_trace" not in keywords
        else:
            assert "boundary_trace" in keywords


def test_production_poisson_bracket_calls_carry_matching_operand_traces():
    tree = ast.parse(RHS.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_poisson_bracket_over_B"
    ]
    assert len(calls) == 6
    expected_g_trace_by_stencil = {
        "density_conservative_stencil": "density",
        "Te_conservative_stencil": "Te",
        "Ti_conservative_stencil": "Ti",
        "Vi_conservative_stencil": "Vi",
        "Ve_conservative_stencil": "Ve",
        "Ve_stencil": "Ve",
        "vorticity_conservative_stencil": "vorticity",
    }
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert _operator_boundary_attribute(keywords["f_boundary_trace"]) == "phi"
        assert isinstance(call.args[3], ast.Name)
        expected = expected_g_trace_by_stencil[call.args[3].id]
        assert _operator_boundary_attribute(keywords["g_boundary_trace"]) == expected


def _operator_boundary_attribute(node: ast.AST) -> str:
    assert isinstance(node, ast.Attribute)
    assert isinstance(node.value, ast.Name)
    assert node.value.id in {"operator_boundary", "perpendicular_operator_boundary"}
    return node.attr


def test_curvature_uses_operator_boundary_traces_without_upwind_ablation():
    source = RHS.read_text()
    for field in ("density", "Te", "Ti", "vorticity"):
        assert f"operator_boundary.{field}" in source
    assert "operator_boundary.Ti_squared" in source
    assert "_upwind_equilibrium_boundary_face_bcs" not in source
