"""Source-level contracts for the opt-in production parallel material path."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    build_local_fci_drb_eb_operator_boundary_bundle,
    curvature_component_diagnostic_names,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
from drbx.geometry import build_local_curvature_face_coefficients  # noqa: E402
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
)


RHS = Path(__file__).resolve().parents[1] / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"
SOURCE = RHS.read_text()
SHARED_DRIVER = RHS.parents[3] / "simulate_hsx_blob.py"


def _mapped_fixture():
    """Small real mapped fixture used by the runtime integration contracts."""

    context, mesh, local, partition, fields, _ = _context_and_sharded_inputs()
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    sharded = build_local_fci_geometries(
        mapped_geometry,
        (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    return (
        context,
        mesh,
        local,
        partition,
        fields,
        jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition)),
        jax.device_put(sharded.map_fields, NamedSharding(mesh, partition)),
        sharded,
    )


def test_parallel_material_selector_is_environment_backed_and_legacy_default():
    assert 'DRBX_PARALLEL_MATERIAL_SCHEME", "legacy"' in SOURCE
    assert 'self.parallel_material_scheme not in ("legacy", "production-path")' in SOURCE
    assert "DRBX_PRODUCTION_CHARACTERISTIC_SOLVER" not in SOURCE
    assert "production_characteristic_solver" not in SOURCE


def test_production_kernels_have_no_runtime_characteristic_selector():
    curvature_start = SOURCE.index("material_result = local_curvature_production_path_op(")
    curvature_end = SOURCE.index("        if return_directional_components:", curvature_start)
    curvature_call = SOURCE[curvature_start:curvature_end]
    parallel_start = SOURCE.index("parallel_target_row_material_residual(")
    parallel_end = SOURCE.index("                )", parallel_start)
    parallel_call = SOURCE[parallel_start:parallel_end]
    assert "characteristic_solver=" not in curvature_call
    assert "characteristic_solver=" not in parallel_call
    assert "production_characteristic_solver" not in SOURCE


def test_shared_builder_threads_production_selectors_explicitly():
    source = SHARED_DRIVER.read_text()
    signature = source[source.index("def build_local_eb_model("):source.index(") -> LocalFciDrbEBRhs:", source.index("def build_local_eb_model("))]
    assert "curvature_split_scheme: str | None = None" in signature
    assert "parallel_material_scheme: str | None = None" in signature
    assert "curvature_split_scheme=str(curvature_split_scheme)" in source
    assert "parallel_material_scheme=str(parallel_material_scheme)" in source
    run_signature = source[source.index("def run_full_eb("):source.index(") -> FciDrbEBState:", source.index("def run_full_eb("))]
    assert "curvature_split_scheme: str | None = None" in run_signature
    assert "parallel_material_scheme: str | None = None" in run_signature


def test_rhs_production_curvature_requires_all_four_equations():
    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        base = _build_rhs(context, local, geometry)
        return replace(
            base,
            curvature_scheme="conservative",
            curvature_face_coefficients=build_local_curvature_face_coefficients(
                geometry, base.domain
            ),
            curvature_split_scheme="production-path",
            curvature_equations=("density", "Te", "Ti"),
        )

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    with pytest.raises(ValueError, match="requires all four curvature equations"):
        compiled(*fields, cell_fields, map_fields)


def test_production_curvature_centered_dissipation_diagnostics_close_to_rhs():
    """Six analysis lanes sum to the unchanged four-field curvature RHS."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        base = _build_rhs(context, local, geometry)
        rhs = replace(
            base,
            curvature_scheme="conservative",
            curvature_face_coefficients=build_local_curvature_face_coefficients(
                geometry, base.domain
            ),
            curvature_split_scheme="production-path",
            curvature_component_diagnostic_scheme="centered-dissipation",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        baseline = rhs.evaluate_stage(state, phi_owned=phi)
        stage, terms, components = rhs.evaluate_stage(
            state,
            phi_owned=phi,
            return_rhs_term_fields=True,
            return_curvature_component_fields=True,
        )
        curvature_lanes = jnp.stack(
            tuple(
                terms[field_index, RHS_TERM_NAMES[field_index].index("curvature")]
                for field_index in (0, 1, 2, 5)
            ),
            axis=0,
        )
        baseline_fields = jnp.stack(
            tuple(getattr(baseline, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        diagnostic_fields = jnp.stack(
            tuple(getattr(stage, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        return components, curvature_lanes, baseline_fields, diagnostic_fields

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=(
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            ),
            check_vma=False,
        )
    )
    components, curvature_lanes, baseline_fields, diagnostic_fields = tuple(
        np.asarray(value) for value in compiled(*fields, cell_fields, map_fields)
    )
    assert components.shape[:2] == (4, 6)
    np.testing.assert_allclose(
        np.sum(components, axis=1),
        curvature_lanes,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        diagnostic_fields,
        baseline_fields,
        atol=3.0e-12,
        rtol=3.0e-12,
    )


def test_production_curvature_radial_provenance_diagnostics_close_to_rhs():
    """Eight replay lanes resolve radial face classes without changing RHS."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        base = _build_rhs(context, local, geometry)
        rhs = replace(
            base,
            curvature_scheme="conservative",
            curvature_face_coefficients=build_local_curvature_face_coefficients(
                geometry, base.domain
            ),
            curvature_split_scheme="production-path",
            curvature_component_diagnostic_scheme="radial-provenance",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        stage, terms, components = rhs.evaluate_stage(
            state,
            phi_owned=phi,
            return_rhs_term_fields=True,
            return_curvature_component_fields=True,
        )
        curvature_lanes = jnp.stack(
            tuple(
                terms[field_index, RHS_TERM_NAMES[field_index].index("curvature")]
                for field_index in (0, 1, 2, 5)
            ),
            axis=0,
        )
        stage_fields = jnp.stack(
            tuple(getattr(stage, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        return components, curvature_lanes, stage_fields

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=(
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            ),
            check_vma=False,
        )
    )
    components, curvature_lanes, stage_fields = tuple(
        np.asarray(value) for value in compiled(*fields, cell_fields, map_fields)
    )
    names = curvature_component_diagnostic_names("radial-provenance")
    assert names == (
        "radial_lower_axis_face",
        "radial_upper_physical_face",
        "radial_rlp_transition_faces",
        "radial_ordinary_interior_faces",
        "radial_within_cell_path",
        "radial_nonlocal_psi_remainder",
        "theta_total",
        "eta_total",
    )
    assert components.shape[:2] == (4, len(names))
    np.testing.assert_allclose(
        np.sum(components, axis=1), curvature_lanes, atol=4.0e-12, rtol=4.0e-12
    )
    assert np.all(np.isfinite(stage_fields))
    # The vorticity equation has no nonlocal psi remainder by construction.
    np.testing.assert_allclose(components[3, names.index("radial_nonlocal_psi_remainder")], 0.0)
    # The mapped/RLP fixture must actually exercise the provenance classes
    # that the frozen HSX audit relies on.
    for name in (
        "radial_upper_physical_face",
        "radial_ordinary_interior_faces",
        "radial_within_cell_path",
    ):
        assert np.max(np.abs(components[:, names.index(name)])) > 0.0
    # This compact mapped fixture has no angular-RLP transition; that lane is
    # deliberately present and zero so a real RLP replay remains self-describing.
    np.testing.assert_allclose(
        components[:, names.index("radial_rlp_transition_faces")], 0.0
    )


def test_production_guards_prevent_incompatible_legacy_paths():
    assert "parallel_material_scheme == \"production-path\"" in SOURCE
    assert "parallel_operator_scheme != \"fci\"" in SOURCE
    assert "parallel_flux_pairing != \"support-core\"" in SOURCE
    assert "parallel_material_scheme='production-path' currently requires" in SOURCE
    assert "parallel_material_scheme='production-path' cannot be combined" in SOURCE
    assert "parallel_inflow_closure != \"central\"" in SOURCE
    assert "parallel_inflow_closure='central'" in SOURCE


def test_production_path_builds_all_five_mapped_rows_and_aggregates_lanes():
    start = SOURCE.index('if self.parallel_material_scheme == "production-path":')
    end = SOURCE.index('        result = {', start)
    block = SOURCE[start:end]
    assert "parallel_target_row_material_residual(" in block
    assert "backward_wall=backward_wall" in block
    assert "forward_wall=forward_wall" in block
    assert "backward_wall_state=minus" in block
    assert "forward_wall_state=plus" in block
    assert "div_b=div_b_fine if self._uses_compact_face_operators else div_b" in block
    assert "parallel_material_residual = jnp.moveaxis" in block
    assert '"parallel_material_diagnostics"' in SOURCE


def test_evaluate_stage_replaces_material_package_and_uses_psi_force():
    assert "production_material_residual = stage_parallel_terms.get(" in SOURCE
    for lane in range(5):
        assert f"production_material_residual[..., {lane}]" in SOURCE
    assert "Ve_phi_force_term = mi_over_me * grad_parallel_phi" in SOURCE
    assert "mi_over_me * tau * grad_parallel_Ti" in SOURCE
    assert "Ve_electrostatic_term = Ve_phi_force_term + Ve_Ti_force_term" in SOURCE
    assert "jnp.where(selected_short_wall, 0.0, Ve_Ti_force_complete_term)" in SOURCE

    # The full RHS uses the coupled residual as the material contribution and
    # does not add the old characteristic correction in production mode.
    rhs_start = SOURCE.index("density_rhs = (")
    rhs_end = SOURCE.index("        vorticity_rhs = (", rhs_start)
    rhs_block = SOURCE[rhs_start:rhs_end]
    assert "production_material_residual[..., 4]" in rhs_block
    assert "0.0 if production_parallel else Ve_characteristic_upwind_term" in rhs_block


def test_div_b_source_is_not_reapplied_by_old_material_terms_in_production():
    # The replacement block still computes the old diagnostics for compatibility,
    # but each equation selects either the coupled residual or the old package.
    assert "if production_parallel else -parallel_density_flux_divergence" in SOURCE
    assert "if production_parallel else Te_parallel_advection" in SOURCE
    assert "if production_parallel else Ti_parallel_advection" in SOURCE
    assert "if production_parallel else Vi_self_advection_term + Vi_pressure_term" in SOURCE


def test_production_material_terms_run_on_every_mapped_row_under_jit():
    """Exercise the actual mapped stencil path, rather than only source text."""

    (
        context,
        mesh,
        local,
        partition,
        fields,
        cell_fields,
        map_fields,
        sharded,
    ) = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = rhs._face_bcs(state)
        state_halo = rhs._prepare_state_halo(state, face_bc)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, geometry, rhs.domain, face_bc, tau=rhs.parameters.tau
        )
        parallel_boundary = rhs._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary
        )
        terms = rhs._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=rhs._stencil_builder_context(),
        )
        residual = terms["parallel_material_residual"]
        diagnostics = terms["parallel_material_diagnostics"]
        covered = diagnostics["ordinary_row"] | diagnostics["wall_row"]
        return jnp.asarray(
            (
                residual.shape[-1],
                jnp.all(jnp.isfinite(residual)),
                jnp.all(covered),
                jnp.max(jnp.abs(residual)),
            )
        )

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    lanes, finite, covered, amplitude = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert lanes == 5
    assert finite
    assert covered
    assert amplitude > 0.0


def test_production_parallel_subsystem_packs_coupled_material_and_preserved_force():
    """The returned parallel-only RHS is exactly its diagnostic term sum.

    In production mode the material slot is the one coupled five-field
    residual.  The electron-velocity collision and electrostatic slots remain
    active, while the old pressure/thermal/characteristic slots are zero.
    """

    (
        context,
        mesh,
        local,
        partition,
        fields,
        cell_fields,
        map_fields,
        sharded,
    ) = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_subsystem_only=True,
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        stage, terms = rhs.evaluate_stage(
            state, phi_owned=phi, return_rhs_term_fields=True
        )
        stage_fields = jnp.stack(
            tuple(getattr(stage, name) for name in
                  ("density", "Te", "Ti", "Vi", "Ve", "vorticity")), axis=0
        )
        error = jnp.max(jnp.abs(jnp.sum(terms, axis=1) - stage_fields))
        # Te/Ti slot 3 is the old compression package.  Ve slots 4, 5, and 8
        # are old pressure, thermal-force, and characteristic correction.
        old_package = jnp.max(jnp.abs(jnp.concatenate((
            terms[1, 3].reshape(-1), terms[2, 3].reshape(-1),
            terms[4, 4].reshape(-1), terms[4, 5].reshape(-1),
            terms[4, 8].reshape(-1),
        ))))
        preserved = jnp.max(jnp.abs(jnp.concatenate((
            terms[4, 2].reshape(-1), terms[4, 3].reshape(-1),
        ))))
        material = jnp.max(jnp.abs(terms[:5, 1]))
        return jnp.asarray((error, old_package, preserved, material))

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    error, old_package, preserved, material = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert error < 2.0e-11
    assert old_package == 0.0
    assert preserved > 0.0
    assert material > 0.0


def test_production_model_rejects_characteristic_wall_traces_at_construction():
    """Prevent applying the production wall projection twice."""

    (
        context,
        mesh,
        local,
        partition,
        fields,
        cell_fields,
        map_fields,
        sharded,
    ) = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_inflow_closure="equilibrium-characteristic",
        )
        return jnp.asarray(0.0)

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    with pytest.raises(ValueError, match="parallel_inflow_closure='central'"):
        compiled(*fields, cell_fields, map_fields)


def test_production_ve_diagnostic_uses_coupled_material_lane():
    """The legacy Ve diagnostic must describe the production RHS it reports."""

    (
        context,
        mesh,
        local,
        partition,
        fields,
        cell_fields,
        map_fields,
        sharded,
    ) = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_subsystem_only=True,
        )
        state, terms = rhs.evaluate_stage(
            FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity),
            phi_owned=phi,
            return_term_fields=True,
        )
        sum_error = jnp.max(jnp.abs(jnp.sum(terms, axis=0) - state.Ve))
        old_terms = jnp.max(jnp.abs(jnp.concatenate((
            terms[4].reshape(-1), terms[5].reshape(-1), terms[9].reshape(-1),
        ))))
        coupled_material = jnp.max(jnp.abs(terms[1]))
        preserved_force = jnp.max(jnp.abs(jnp.concatenate((
            terms[2].reshape(-1), terms[3].reshape(-1),
        ))))
        return jnp.asarray((sum_error, old_terms, coupled_material, preserved_force))

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    sum_error, old_terms, coupled_material, preserved_force = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert sum_error < 2.0e-11
    assert old_terms == 0.0
    assert coupled_material > 0.0
    assert preserved_force > 0.0
