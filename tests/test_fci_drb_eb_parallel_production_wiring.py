"""Source and runtime contracts for the production parallel material path."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
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
    curvature_end = SOURCE.index("        if request_split_diagnostics:", curvature_start)
    curvature_call = SOURCE[curvature_start:curvature_end]
    parallel_start = SOURCE.index("parallel_target_row_material_residual(")
    parallel_end = SOURCE.index("                )", parallel_start)
    parallel_call = SOURCE[parallel_start:parallel_end]
    assert "characteristic_solver=" not in curvature_call
    assert "characteristic_solver=" not in parallel_call
    assert "production_characteristic_solver" not in SOURCE


def test_shared_builder_hardwires_production_curvature():
    source = SHARED_DRIVER.read_text()
    signature = source[source.index("def build_local_eb_model("):source.index(
        ") -> LocalFciDrbEBRhs:", source.index("def build_local_eb_model(")
    )]
    assert "parallel_material_scheme: str | None = None" in signature
    assert "curvature_scheme" not in signature
    assert "curvature_split_scheme" not in signature
    assert "build_local_curvature_face_coefficients(geometry, domain)" in source
    assert "curvature_face_coefficients=curvature_face_coefficients" in source
    assert "parallel_material_scheme=str(parallel_material_scheme)" in source
    run_signature = source[source.index("def run_full_eb("):source.index(
        ") -> FciDrbEBState:", source.index("def run_full_eb(")
    )]
    assert "curvature_scheme" not in run_signature
    assert "curvature_split_scheme" not in run_signature
    assert "parallel_material_scheme: str | None = None" in run_signature


def test_production_guards_prevent_incompatible_legacy_paths():
    assert "parallel_material_scheme == \"production-path\"" in SOURCE
    assert "parallel_operator_scheme != \"fci\"" in SOURCE
    assert "parallel_flux_pairing != \"support-core\"" in SOURCE
    assert "parallel_material_scheme='production-path' requires" in SOURCE


def test_production_path_builds_all_five_dense_mapped_rows():
    start = SOURCE.index('if self.parallel_material_scheme == "production-path":')
    end = SOURCE.index('        result = {', start)
    block = SOURCE[start:end]
    assert "parallel_target_row_material_residual(" in block
    assert "backward_wall=backward_wall" in block
    assert "forward_wall=forward_wall" in block
    assert "backward_wall_state=minus" in block
    assert "forward_wall_state=plus" in block
    assert "div_b=div_b," in block
    assert '"parallel_material_diagnostics"' in SOURCE


def test_evaluate_stage_replaces_material_package_and_uses_psi_force():
    assert "production_material_residual = stage_parallel_terms.get(" in SOURCE
    for lane in range(5):
        assert f"production_material_residual[..., {lane}]" in SOURCE
    assert "Ve_phi_force_term = mi_over_me * grad_parallel_phi" in SOURCE
    assert "mi_over_me * tau * grad_parallel_Ti" in SOURCE
    assert "Ve_electrostatic_term = Ve_phi_force_term + Ve_Ti_force_term" in SOURCE
    assert "jnp.where(selected_short_wall, 0.0, Ve_Ti_force_complete_term)" in SOURCE
    rhs_start = SOURCE.index("density_rhs = (")
    rhs_end = SOURCE.index("        vorticity_rhs = (", rhs_start)
    rhs_block = SOURCE[rhs_start:rhs_end]
    assert "production_material_residual[..., 4]" in rhs_block
    assert "0.0 if production_parallel else Ve_characteristic_upwind_term" in rhs_block


def test_div_b_source_is_not_reapplied_by_old_material_terms_in_production():
    assert "if production_parallel else -parallel_density_flux_divergence" in SOURCE
    assert "if production_parallel else Te_parallel_advection" in SOURCE
    assert "if production_parallel else Ti_parallel_advection" in SOURCE
    assert "if production_parallel else Vi_self_advection_term + Vi_pressure_term" in SOURCE


def test_production_material_terms_run_on_every_mapped_row_under_jit():
    """Exercise the actual mapped stencil path, rather than only source text."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = _mapped_fixture()

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
    lanes, finite, covered, amplitude = np.asarray(compiled(*fields, cell_fields, map_fields))
    assert lanes == 5
    assert finite
    assert covered
    assert amplitude > 0.0


def test_production_ve_diagnostic_uses_coupled_material_lane():
    """The full RHS Ve diagnostic must include the coupled material lane."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
        )
        state, terms = rhs.evaluate_stage(
            FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity),
            phi_owned=phi,
            return_term_fields=True,
        )
        sum_error = jnp.max(jnp.abs(jnp.sum(terms, axis=0) - state.Ve))
        # Full production Ve has nine lanes: Poisson bracket, coupled
        # material, collision, electrostatic, two retired legacy lanes,
        # perpendicular and parallel diffusion, and a retired characteristic
        # correction lane.  The retired lanes must remain zero.
        old_terms = jnp.max(jnp.abs(jnp.concatenate((
            terms[4].reshape(-1), terms[5].reshape(-1), terms[8].reshape(-1),
        ))))
        coupled_material = jnp.max(jnp.abs(terms[1]))
        preserved_force = jnp.max(jnp.abs(jnp.concatenate((
            terms[2].reshape(-1), terms[3].reshape(-1),
        ))) )
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
