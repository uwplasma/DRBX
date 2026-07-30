"""Regression tests for the static HSX curvature experiment wiring."""

import ast
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RHS_PATH = WORKSPACE / "DRBX" / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"
DRIVER_PATH = WORKSPACE / "simulate_hsx_blob.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_ti_curvature_contribution_is_whole_equation_gated():
    tree = _tree(RHS_PATH)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stage"
    )

    gated_assignments = []

    def visit(node, enclosing_ifs=()):
        if isinstance(node, ast.If):
            enclosing_ifs = (*enclosing_ifs, node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "curvature_Ti_contribution"
            for target in node.targets
        ):
            gated_assignments.append(enclosing_ifs)
        for child in ast.iter_child_nodes(node):
            visit(child, enclosing_ifs)

    visit(method)
    assert len(gated_assignments) == 3
    for enclosing_ifs in gated_assignments:
        assert any(
            isinstance(test, ast.Compare)
            and isinstance(test.ops[0], ast.In)
            and isinstance(test.left, ast.Constant)
            and test.left.value == "Ti"
            for test in (if_node.test for if_node in enclosing_ifs)
        )


def test_tracked_rk4_uses_three_output_specs_and_scalar_diagnostic_halo():
    rhs_tree = _tree(RHS_PATH)
    diagnostic = _function(
        rhs_tree,
        "ion_temperature_curvature_chain_rule_diagnostics",
    )
    diagnostic_source = ast.get_source_segment(RHS_PATH.read_text(), diagnostic)
    assert diagnostic_source is not None
    assert "self._prepare_scalar_halo" in diagnostic_source
    assert "prepare_local_fci_drb_eb_state" not in diagnostic_source

    driver_tree = _tree(DRIVER_PATH)
    run_full_eb = _function(driver_tree, "run_full_eb")
    out_spec_assignment = next(
        node
        for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "rk4_out_specs"
            for target in node.targets
        )
    )
    assert isinstance(out_spec_assignment.value, ast.IfExp)
    assert isinstance(out_spec_assignment.value.test, ast.Name)
    assert out_spec_assignment.value.test.id == (
        "track_curvature_chain_rule_defect"
    )
    shard_map_calls = [
        node
        for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shard_map"
    ]
    rk4_call = next(
        call
        for call in shard_map_calls
        if any(
            keyword.arg == "out_specs"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "rk4_out_specs"
            for keyword in call.keywords
        )
    )
    assert rk4_call is not None


def test_hsx_gmres_uses_tight_target_and_preserves_looser_acceptance():
    source = DRIVER_PATH.read_text()
    assert "GMRES_TARGET_TOLERANCE = 1.0e-8" in source
    build = _function(_tree(DRIVER_PATH), "build_local_eb_model")
    build_source = ast.get_source_segment(source, build)
    assert build_source is not None
    assert "tol=float(gmres_target_tolerance)" in build_source
    assert "atol=float(gmres_target_tolerance)" in build_source
    assert "acceptance_tol=float(gmres_acceptance_tolerance)" in build_source
    assert "acceptance_atol=float(gmres_acceptance_tolerance)" in build_source


def test_hsx_gmres_cli_separates_target_and_acceptance_with_legacy_alias():
    source = DRIVER_PATH.read_text()
    assert '"--gmres-target-tolerance"' in source
    assert '"--gmres-acceptance-tolerance"' in source
    assert '"--gmres-tolerance"' in source
    assert 'dest="gmres_acceptance_tolerance"' in source
    assert 'default=GMRES_TARGET_TOLERANCE' in source
    assert 'default=5.0e-5' in source
    assert 'gmres_target_tolerance=float(args.gmres_target_tolerance)' in source
    assert (
        'gmres_acceptance_tolerance=float(args.gmres_acceptance_tolerance)'
        in source
    )


def test_hsx_rk4_returns_replicated_solvax_diagnostics():
    source = DRIVER_PATH.read_text()
    run_source = ast.get_source_segment(
        source,
        _function(_tree(DRIVER_PATH), "run_full_eb"),
    )
    assert run_source is not None
    assert "info.num_steps" in run_source
    assert "info.final_residual_rel_l2" in run_source
    assert "info.failed" in run_source
    assert "gmres_info_2" in run_source
    assert "gmres_info_3" in run_source
    assert "gmres_info_4" in run_source
    assert "gmres_info_next" in run_source
    assert "gmres_stage_diagnostics" in run_source
    assert "gmres_iterations =" in run_source
    assert "replicated_spec" in run_source
    assert "gmres_iterations=gmres_iterations_host" in run_source
    assert "gmres_relative_residual=gmres_relative_residual_host" in run_source
    assert "rejected phi inversion" in run_source
    assert "gmres-iters(avg4)=" in source
    assert "gmres-relres(max4)=" in source


def test_hsx_phi_reconstruction_exposes_solvax_gmres_info():
    rhs_source = RHS_PATH.read_text()
    rhs_tree = _tree(RHS_PATH)
    reconstruct = _function(rhs_tree, "reconstruct_phi")
    reconstruct_source = ast.get_source_segment(rhs_source, reconstruct)
    assert reconstruct_source is not None
    assert "return_diagnostics" in reconstruct_source
    assert "return_diagnostics=return_diagnostics" in reconstruct_source
    assert "SolvaxGmresInfo" in rhs_source


def test_full_eb_conservative_curvature_exposes_upwind_equilibrium_mode():
    driver_tree = _tree(DRIVER_PATH)
    build = _function(driver_tree, "build_local_eb_model")
    defaults = {
        arg.arg: default.value
        for arg, default in zip(build.args.kwonlyargs, build.args.kw_defaults)
        if isinstance(default, ast.Constant)
    }
    assert defaults["curvature_inflow_closure"] == "central"
    source = DRIVER_PATH.read_text()
    assert '"--curvature-inflow-closure"' in source
    assert 'default="central"' in source
    assert "curvature_inflow_closure=str(args.curvature_inflow_closure)" in source

    rhs_tree = _tree(RHS_PATH)
    rhs_source = RHS_PATH.read_text()
    assert "_upwind_equilibrium_boundary_face_bcs" in rhs_source
    assert '"upwind-equilibrium"' in rhs_source
    assert '"neumann"' not in rhs_source
    assert "_characteristic_projectors_background" in rhs_source
    evaluate = _function(rhs_tree, "evaluate_stage")
    evaluate_source = ast.get_source_segment(rhs_source, evaluate)
    assert evaluate_source is not None
    assert evaluate_source.count("self._conservative_curvature(") >= 2
    for field in (
        "curvature_Pe = self._conservative_curvature(",
        "curvature_pressure = curvature(",
        "curvature_phi = curvature(phi_conservative_stencil, face_bc.phi if phi_wall_bc is None else phi_wall_bc)",
        "curvature_Te = curvature(Te_conservative_stencil, face_bc.Te if Te_wall_bc is None else Te_wall_bc)",
        "curvature_Ti = curvature(Ti_conservative_stencil, face_bc.Ti if Ti_wall_bc is None else Ti_wall_bc)",
    ):
        assert field in evaluate_source


def test_full_eb_keeps_centered_conservative_compatibility_mode():
    rhs_source = RHS_PATH.read_text()
    helper = _function(_tree(RHS_PATH), "_conservative_curvature")
    helper_source = ast.get_source_segment(rhs_source, helper)
    assert helper_source is not None
    assert "local_curvature_conservative_op(" in helper_source
    assert 'inflow_closure="equilibrium"' not in helper_source


def test_upwind_equilibrium_sets_incoming_perturbations_to_equilibrium_for_both_signs():
    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from drbx.native.fci_drb_EB_rhs import (
        _characteristic_outgoing_incoming_projectors,
        _characteristic_projectors_background,
        _upwind_equilibrium_characteristic_state,
    )

    equilibrium = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    owner = jnp.asarray([1.25, 0.82, 1.31, -0.44])
    pe, pi, p0 = _characteristic_projectors_background(
        jnp.asarray([0.9, 0.9]), 0.7
    )
    out, incoming, stationary = _characteristic_outgoing_incoming_projectors(
        jnp.asarray([1.0, -1.0]), pe, pi, p0
    )
    states = _upwind_equilibrium_characteristic_state(
        jnp.broadcast_to(owner, (2, 4)),
        equilibrium,
        out + stationary,
    )
    for index in range(2):
        delta_owner = owner - equilibrium
        delta_state = np.asarray(states[index] - equilibrium)
        np.testing.assert_allclose(
            np.asarray(incoming[index]) @ delta_state,
            0.0,
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            np.asarray((out[index] + stationary[index]) @ delta_state),
            np.asarray((out[index] + stationary[index]) @ delta_owner),
            rtol=1e-11,
            atol=1e-11,
        )


def test_upwind_equilibrium_is_the_only_noncentral_inflow_closure_name():
    rhs_source = RHS_PATH.read_text()
    assert '"neumann"' not in rhs_source
    assert '"central", "upwind-equilibrium"' in rhs_source


def test_upwind_equilibrium_wall_projectors_are_precomputed_payload():
    import numpy as np
    import jax
    import jax.numpy as jnp

    from drbx.native.fci_drb_EB_rhs import UpwindEquilibriumWallProjectors

    leaf = jnp.arange(32, dtype=jnp.float64).reshape(2, 4, 4)
    payload = UpwindEquilibriumWallProjectors(
        axes=tuple(
            tuple(leaf + 3 * axis + side for side in range(2))
            for axis in range(3)
        )
    )
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    assert len(leaves) == 6
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    for original, actual in zip(
        jax.tree_util.tree_leaves(payload), jax.tree_util.tree_leaves(restored)
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))


def test_upwind_boundary_stage_path_consumes_precomputed_projectors():
    import ast

    rhs_source = RHS_PATH.read_text()
    tree = _tree(RHS_PATH)
    helper = _function(tree, "_upwind_equilibrium_boundary_face_bcs")
    helper_source = ast.get_source_segment(rhs_source, helper)
    assert helper_source is not None
    assert "self.upwind_equilibrium_wall_projectors" in helper_source
    assert "_characteristic_projectors_background" not in helper_source
    assert "_characteristic_outgoing_incoming_projectors" not in helper_source

    driver_source = DRIVER_PATH.read_text()
    driver_tree = ast.parse(driver_source)
    build = _function(driver_tree, "build_local_eb_model")
    build_source = ast.get_source_segment(driver_source, build)
    assert build_source is not None
    assert "upwind_equilibrium_wall_projectors=upwind_equilibrium_wall_projectors" in build_source
    assert "build_upwind_equilibrium_wall_projectors(" not in build_source
    assert "precompute_wall_projectors" in driver_source
    assert "out_specs=wall_projector_specs" in driver_source
    assert "tuple(projector_axis_specs[axis] for _ in range(3))" not in driver_source
    advance = _function(driver_tree, "full_rk4_advance")
    advance_source = ast.get_source_segment(driver_source, advance)
    assert advance_source is not None
    assert "build_upwind_equilibrium_wall_projectors(" not in advance_source
    assert "local_wall_projectors" in advance_source


def test_compact_wall_projectors_runtime_vma_for_periodic_and_normal_sharding():
    """Exercise the real projector builder under compact shard_map outputs."""

    import math
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from jax.sharding import PartitionSpec as P

    from drbx.geometry import LocalCurvatureFaceCoefficients3D
    from drbx.native import build_local_fci_geometries, make_shard_mesh
    from drbx.native.fci_drb_EB_rhs import (
        UpwindEquilibriumWallProjectors,
        build_upwind_equilibrium_wall_projectors,
    )
    from tests.test_fci_operators_domain_decomp import _build_domain

    shape = (4, 4, 4)
    shard_layouts = ((1, 1, 4), (2, 1, 1))
    if len(jax.devices()) < 4:
        shard_layouts = ((1, 1, 1),)
    for shard_counts in shard_layouts:
        required_devices = math.prod(shard_counts)
        if len(jax.devices()) < required_devices:
            pytest.skip(f"requires {required_devices} devices")
        # The helper only needs the local face B magnitudes and curvature
        # coefficients, so a compact synthetic bundle is sufficient here.
        domain = _build_domain(shape, 1, shard_counts)
        owned = tuple(size // count for size, count in zip(shape, shard_counts))

        def face(axis):
            face_shape = list(owned)
            face_shape[axis] += 1
            return SimpleNamespace(
                Bmag_owned=jnp.ones(tuple(face_shape), dtype=jnp.float64)
            )

        geometry = SimpleNamespace(
            face_bfield=SimpleNamespace(x=face(0), y=face(1), z=face(2))
        )
        coefficients = LocalCurvatureFaceCoefficients3D(
            layout=domain.layout,
            x=jnp.ones(domain.layout.face_control_shape(0), dtype=jnp.float64),
            y=jnp.ones(domain.layout.face_control_shape(1), dtype=jnp.float64),
            z=jnp.ones(domain.layout.face_control_shape(2), dtype=jnp.float64),
        )
        projector_specs = UpwindEquilibriumWallProjectors(
            axes=tuple(
                tuple(
                    P(None, "y", "z")
                    if axis == 0
                    else P("x", None, "z")
                    if axis == 1
                    else P("x", "y", None)
                    for _side in range(2)
                )
                for axis in range(3)
            )
        )
        mesh = make_shard_mesh(shard_counts)
        dummy = jax.device_put(
            jnp.zeros(shape, dtype=jnp.float64),
            jax.sharding.NamedSharding(mesh, P("x", "y", "z")),
        )

        def kernel(_dummy):
            return build_upwind_equilibrium_wall_projectors(
                geometry,
                domain,
                coefficients,
                1.0,
            )

        with mesh:
            payload = jax.jit(
                jax.shard_map(
                    kernel,
                    mesh=mesh,
                    in_specs=P("x", "y", "z"),
                    out_specs=projector_specs,
                    check_vma=True,
                )
            )(dummy)
        leaves = jax.tree_util.tree_leaves(payload)
        assert len(leaves) == 6
        x_lower_out = np.asarray(leaves[0])
        if shard_counts == (1, 1, 4):
            assert np.max(np.abs(x_lower_out)) > 0.0
            # z is periodic, so both z-side payloads are identically zero.
            assert np.max(np.abs(np.asarray(leaves[4]))) == 0.0
            assert np.max(np.abs(np.asarray(leaves[5]))) == 0.0
        else:
            # The x wall plane is replicated after the x-axis psum.
            assert np.max(np.abs(x_lower_out)) > 0.0


def test_background_characteristic_projectors_are_complete_and_invariant():
    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from drbx.native.fci_drb_EB_rhs import (
        _characteristic_outgoing_incoming_projectors,
        _characteristic_projectors_background,
    )

    pe, pi, p0 = _characteristic_projectors_background(
        jnp.asarray([0.8, 1.1]), 0.7
    )
    eye = np.eye(4)
    for projector in (pe, pi, p0):
        array = np.asarray(projector)
        np.testing.assert_allclose(array @ array, array, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(pe + pi + p0), np.broadcast_to(eye, (2, 4, 4)), rtol=1e-11, atol=1e-11
    )

    # The background principal matrix commutes with each spectral projector.
    bmag = np.asarray([0.8, 1.1])
    tau = 0.7
    matrices = []
    for b in bmag:
        matrices.append(
            np.array(
                [
                    [2.0, 2.0, 0.0, 0.0],
                    [4.0 / 3.0, 14.0 / 3.0, 0.0, 0.0],
                    [4.0 / 3.0, 4.0 / 3.0, -10.0 * tau / 3.0, 0.0],
                    [2.0 * b * b * (1.0 + tau), 2.0 * b * b, 2.0 * tau * b * b, 0.0],
                ]
            )

        )
    for index, matrix in enumerate(matrices):
        for projector in (pe, pi, p0):
            np.testing.assert_allclose(
                matrix @ np.asarray(projector)[index],
                np.asarray(projector)[index] @ matrix,
                rtol=1e-11,
                atol=1e-11,
            )

    # For A_n=-q_n M, lower/upper outward signs are handled through q_n.
    out_pos, in_pos, stationary = _characteristic_outgoing_incoming_projectors(
        jnp.asarray([1.0, -1.0]), pe, pi, p0
    )
    np.testing.assert_allclose(np.asarray(out_pos[0]), np.asarray(pi[0]))
    np.testing.assert_allclose(np.asarray(in_pos[0]), np.asarray(pe[0]))
    np.testing.assert_allclose(np.asarray(out_pos[1]), np.asarray(pe[1]))
    np.testing.assert_allclose(np.asarray(in_pos[1]), np.asarray(pi[1]))
    out_zero, in_zero, stationary_zero = (
        _characteristic_outgoing_incoming_projectors(
            jnp.asarray([0.0]), pe[:1], pi[:1], p0[:1]
        )
    )
    np.testing.assert_allclose(np.asarray(out_zero), 0.0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(in_zero), 0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(stationary_zero), eye[None, ...], rtol=1e-11, atol=1e-11
    )
