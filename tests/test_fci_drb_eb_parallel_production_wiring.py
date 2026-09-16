"""Source and runtime contracts for the production parallel material path."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

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
    assert (
        'DRBX_PARALLEL_VORTICITY_ADVECTION_SCHEME", "first-order"'
        in SOURCE
    )
    assert (
        'DRBX_PARALLEL_MATERIAL_FALLBACK_REPRESENTATION", "legacy-p"'
        in SOURCE
    )
    assert (
        'DRBX_PARALLEL_MATERIAL_DIV_B_FALLBACK_SCHEME", "legacy-p"'
        in SOURCE
    )
    assert '"raw-metric"' in SOURCE
    assert '"h-mf-consistent"' in SOURCE
    assert '"h-mf-second-order"' in SOURCE
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
    assert "parallel_vorticity_advection_scheme: str | None = None" in signature
    assert (
        "parallel_material_fallback_representation: str | None = None"
        in signature
    )
    assert (
        "parallel_material_div_b_fallback_scheme: str | None = None"
        in signature
    )
    assert "curvature_scheme" not in signature
    assert "curvature_split_scheme" not in signature
    assert "build_local_curvature_face_coefficients(geometry, domain)" in source
    assert "curvature_face_coefficients=curvature_face_coefficients" in source
    assert "parallel_material_scheme=str(parallel_material_scheme)" in source
    assert "parallel_vorticity_advection_scheme=str(" in source
    assert "parallel_material_fallback_representation=str(" in source
    assert "parallel_material_div_b_fallback_scheme=str(" in source
    run_signature = source[source.index("def run_full_eb("):source.index(
        ") -> FciDrbEBState:", source.index("def run_full_eb(")
    )]
    assert "curvature_scheme" not in run_signature
    assert "curvature_split_scheme" not in run_signature
    assert "parallel_material_scheme: str | None = None" in run_signature
    assert (
        "parallel_vorticity_advection_scheme: str | None = None"
        in run_signature
    )
    assert (
        "parallel_material_fallback_representation: str | None = None"
        in run_signature
    )
    assert (
        "parallel_material_div_b_fallback_scheme: str | None = None"
        in run_signature
    )


def test_production_guards_prevent_incompatible_legacy_paths():
    assert "parallel_material_scheme == \"production-path\"" in SOURCE
    assert "parallel_operator_scheme != \"fci\"" in SOURCE
    assert "parallel_flux_pairing != \"support-core\"" in SOURCE
    assert "parallel_material_scheme='production-path' requires" in SOURCE
    assert "and self.geometry.material_maps is None" in SOURCE
    assert '"requires material_maps"' in SOURCE


def test_h_mf_fallback_guard_rejects_incompatible_models():
    context, mesh, local, partition, _fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def construct(cells, representation):
        geometry = assemble_local_fci_geometry(local, cells)
        rhs = _build_rhs(context, local, geometry)
        replace(rhs, parallel_material_fallback_representation=representation)
        return jnp.asarray(0.0)

    with pytest.raises(
        ValueError,
        match="parallel_material_fallback_representation must be",
    ):
        jax.shard_map(
            lambda cells: construct(cells, "unknown"),
            mesh=mesh,
            in_specs=partition,
            out_specs=P(),
            check_vma=False,
        )(cell_fields)
    with pytest.raises(
        ValueError,
        match="requires parallel_material_scheme='production-path'",
    ):
        jax.shard_map(
            lambda cells: construct(cells, "h-mf-consistent"),
            mesh=mesh,
            in_specs=partition,
            out_specs=P(),
            check_vma=False,
        )(cell_fields)


def test_raw_metric_div_b_fallback_guard_rejects_incompatible_models():
    context, mesh, local, partition, _fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def construct(cells, scheme):
        geometry = assemble_local_fci_geometry(local, cells)
        rhs = _build_rhs(context, local, geometry)
        replace(rhs, parallel_material_div_b_fallback_scheme=scheme)
        return jnp.asarray(0.0)

    with pytest.raises(
        ValueError,
        match="parallel_material_div_b_fallback_scheme must be",
    ):
        jax.shard_map(
            lambda cells: construct(cells, "unknown"),
            mesh=mesh,
            in_specs=partition,
            out_specs=P(),
            check_vma=False,
        )(cell_fields)
    with pytest.raises(
        ValueError,
        match="requires parallel_material_scheme='production-path'",
    ):
        jax.shard_map(
            lambda cells: construct(cells, "raw-metric"),
            mesh=mesh,
            in_specs=partition,
            out_specs=P(),
            check_vma=False,
        )(cell_fields)


def test_production_path_builds_all_five_dense_mapped_rows():
    start = SOURCE.index('if self.parallel_material_scheme == "production-path":')
    end = SOURCE.index('        result = {', start)
    helper_start = SOURCE.index("def _fci_parallel_characteristic_wall_data")
    helper_end = SOURCE.index("    def _fci_parallel_terms", helper_start)
    block = SOURCE[start:end] + SOURCE[helper_start:helper_end]
    assert "parallel_target_row_material_residual(" in block
    assert "backward_wall=backward_wall" in block
    assert "forward_wall=forward_wall" in block
    # Wall endpoint construction is factored into the shared helper used by
    # both the production RHS and the Rung-3 startup initializer.
    assert "_fci_parallel_characteristic_wall_data(" in SOURCE
    assert (
        'for name in ("density", "Te", "Ti", "Vi", "Ve", "phi")'
        in block
    )
    assert 'for name in ("density", "Te", "Ti", "Vi", "Ve")' in block
    assert "backward_wall_state=backward_wall_state" in block
    assert "forward_wall_state=forward_wall_state" in block
    assert "endpoint_b_contra_x" in block
    assert "endpoint_bmag" in block
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


def _curvature_contribution_probe(
    monkeypatch, *, with_direct_faces, omit_phi_owner=False, lean=False
):
    """Exercise curvature face payload selection without constructing a mesh."""

    import drbx.native.fci_drb_EB_rhs as rhs_module

    shape = (1, 1, 1)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    calls = []
    direct_calls = []
    patch_calls = []
    halo_calls = []
    stencil_calls = []

    def fake_direct(**kwargs):
        direct_calls.append(kwargs)
        return f"direct-{len(direct_calls)}"

    def fake_patch(*args, **kwargs):
        patch_calls.append((args, kwargs))
        return "patched-psi-stencil"

    def fake_lean_direct(**kwargs):
        direct_calls.append(kwargs)
        return "lean-direct"

    def fake_stencil(*args, **kwargs):
        stencil_calls.append(args[0])
        return f"stencil-{len(stencil_calls)}"

    def fake_curvature(*args, **kwargs):
        return zeros

    def fake_production(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return jnp.zeros(shape + (4,), dtype=jnp.float64)

    def fake_periodic_endpoints(*args, **kwargs):
        return "periodic-endpoint-states"

    monkeypatch.setattr(rhs_module, "build_local_control_volume_direct_face_states", fake_direct)
    monkeypatch.setattr(rhs_module, "build_local_radial_curvature_face_states", fake_lean_direct)
    monkeypatch.setattr(rhs_module, "patch_local_control_volume_direct_face_values", fake_patch)
    monkeypatch.setattr(rhs_module, "patch_local_radial_curvature_face_values", fake_patch)
    monkeypatch.setattr(rhs_module, "build_local_conservative_stencil_from_field", fake_stencil)
    monkeypatch.setattr(rhs_module, "local_curvature_production_path_op", fake_production)
    monkeypatch.setattr(
        rhs_module,
        "build_periodic_curvature_endpoint_face_states",
        fake_periodic_endpoints,
    )
    monkeypatch.setattr(rhs_module, "_combine_operator_traces", lambda *a, **k: "psi-trace")

    control_volume = (
        SimpleNamespace(
            has_angular_agglomeration=True,
            face_functionals=None if lean else SimpleNamespace(max_rows=1),
            radial_curvature_faces=(
                SimpleNamespace(max_rows=1) if lean else None
            ),
        )
        if with_direct_faces else None
    )
    rhs = SimpleNamespace(
        control_volume_geometry=control_volume,
        geometry=SimpleNamespace(layout=SimpleNamespace(halo_width=2)),
        domain=SimpleNamespace(periodic_axes=(False, True, True)),
        halo_exchange="exchange",
        topology_filler="topology",
        curvature_face_coefficients="coefficients",
        parameters=SimpleNamespace(n0=1.0, Te0=1.0, Ti0=1.0),
        _conservative_curvature=fake_curvature,
        _conservative_curvature_components=fake_curvature,
    )
    if with_direct_faces:
        def fake_halo(values, boundary):
            halo_calls.append((values, boundary))
            return jnp.full(shape, float(len(halo_calls)))

        rhs._prepare_rlp_reconstructed_halo = fake_halo
    state_halo = SimpleNamespace(phi=zeros, Ti=zeros)
    boundary = SimpleNamespace(density=None, Te=None, Ti=None, phi=None, vorticity=None)
    owners = dict(
        density_owner=jnp.ones(shape),
        Te_owner=jnp.ones(shape),
        Ti_owner=jnp.ones(shape),
        vorticity_owner=zeros,
        phi_owner=zeros,
    )
    if omit_phi_owner:
        owners.pop("phi_owner")
    result = rhs_module.LocalFciDrbEBRhs._curvature_rhs_contributions(
        rhs,
        state_halo=state_halo,
        context="context",
        density=jnp.ones(shape),
        Te=jnp.ones(shape),
        Ti=jnp.ones(shape),
        bmag=jnp.ones(shape),
        tau=0.7,
        density_conservative_stencil="density-stencil",
        Te_conservative_stencil="Te-stencil",
        Ti_conservative_stencil="Ti-stencil",
        vorticity_conservative_stencil="vorticity-stencil",
        operator_boundary=boundary,
        face_bc=SimpleNamespace(
            density="density-bc", Te="Te-bc", Ti="Ti-bc",
            vorticity="vorticity-bc", phi="phi-bc",
        ),
        **owners if with_direct_faces else {},
    )
    return result, calls, direct_calls, patch_calls, halo_calls, stencil_calls


def test_curvature_rhs_wires_direct_quadrature_and_patches_psi(monkeypatch):
    _result, calls, direct_calls, patch_calls, halo_calls, stencil_calls = _curvature_contribution_probe(
        monkeypatch, with_direct_faces=True
    )
    assert len(direct_calls) == 5
    assert [call["positivity_floor"] for call in direct_calls] == [
        1.0e-12, 1.0e-12, 1.0e-12, None, None
    ]
    assert all(call["halo_exchange"] == "exchange" for call in direct_calls)
    assert all(call["topology_filler"] == "topology" for call in direct_calls)
    assert len(halo_calls) == 5
    assert [entry[1] for entry in halo_calls] == [
        "density-bc", "Te-bc", "Ti-bc", "vorticity-bc", "phi-bc"
    ]
    np.testing.assert_array_equal(halo_calls[0][0], jnp.ones((1, 1, 1)))
    np.testing.assert_array_equal(halo_calls[1][0], jnp.ones((1, 1, 1)))
    np.testing.assert_array_equal(halo_calls[2][0], jnp.ones((1, 1, 1)))
    np.testing.assert_array_equal(halo_calls[3][0], jnp.zeros((1, 1, 1)))
    np.testing.assert_array_equal(halo_calls[4][0], jnp.zeros((1, 1, 1)))
    assert calls[0]["args"][0] == (
        "stencil-1", "stencil-2", "stencil-3", "stencil-4"
    )
    assert calls[0]["direct_face_quadrature_states"] == (
        "direct-1", "direct-2", "direct-3", "direct-4"
    )
    assert calls[0]["periodic_endpoint_face_states"] == "periodic-endpoint-states"
    assert patch_calls[0][0][1] == "direct-5"
    assert patch_calls[0][1] == {"transition_only": False}


def test_curvature_rhs_prefers_one_batched_lean_radial_payload(monkeypatch):
    _result, calls, direct_calls, patch_calls, halo_calls, _stencil_calls = (
        _curvature_contribution_probe(
            monkeypatch, with_direct_faces=True, lean=True
        )
    )
    assert len(direct_calls) == 1
    assert direct_calls[0]["owner_values_owned"].shape == (1, 1, 1, 5)
    assert direct_calls[0]["positive_field_count"] == 3
    assert calls[0]["direct_face_quadrature_states"] is None
    assert calls[0]["radial_curvature_face_states"] == "lean-direct"
    assert calls[0]["periodic_endpoint_face_states"] == "periodic-endpoint-states"
    assert patch_calls[0][0][1] == "lean-direct"
    assert patch_calls[0][1] == {"field_index": 4}
    assert len(halo_calls) == 5


def test_curvature_rhs_preserves_legacy_path_without_direct_payload(monkeypatch):
    _result, calls, direct_calls, patch_calls, halo_calls, stencil_calls = _curvature_contribution_probe(
        monkeypatch, with_direct_faces=False
    )
    assert len(calls) == 1
    assert calls[0]["direct_face_quadrature_states"] is None
    assert direct_calls == []
    assert patch_calls == []
    assert halo_calls == []


def test_curvature_rhs_rejects_missing_phi_owner_before_direct_build(monkeypatch):
    with pytest.raises(ValueError, match="phi owner"):
        _curvature_contribution_probe(
            monkeypatch, with_direct_faces=True, omit_phi_owner=True
        )


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
            parallel_vorticity_advection_scheme="h-mf-second-order",
            parallel_material_fallback_representation="h-mf-consistent",
            parallel_material_div_b_fallback_scheme="raw-metric",
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
        vorticity_advection = terms["vorticity_parallel_advection"]
        return jnp.asarray(
            (
                residual.shape[-1],
                jnp.all(jnp.isfinite(residual)),
                jnp.all(covered),
                jnp.max(jnp.abs(residual)),
                jnp.all(jnp.isfinite(vorticity_advection)),
                jnp.all(diagnostics["vorticity_h_second_order_used"]),
                jnp.max(jnp.abs(vorticity_advection)),
                jnp.all(diagnostics["material_h_mf_fallback_used"]),
                jnp.any(diagnostics["wall_center_reconstruction_mismatch"]),
                jnp.max(jnp.abs(
                    terms["parallel_characteristic_wall_data"]["center"]
                    - terms["parallel_material_implicit_data"]["center"]
                )),
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
    (
        lanes,
        finite,
        covered,
        amplitude,
        vorticity_finite,
        vorticity_h_second_order,
        vorticity_amplitude,
        material_h_mf_fallback,
        wall_center_mismatch,
        implicit_center_error,
    ) = np.asarray(compiled(*fields, cell_fields, map_fields))
    assert lanes == 5
    assert finite
    assert covered
    assert amplitude > 0.0
    assert vorticity_finite
    assert vorticity_h_second_order
    assert vorticity_amplitude > 0.0
    assert material_h_mf_fallback
    assert not wall_center_mismatch
    assert implicit_center_error == 0.0


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
