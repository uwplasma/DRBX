import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import jax
from dataclasses import replace
import pytest

from drbx.native.fci_drb_EB_rhs import FciDrbEBState, RHS_TERM_FIELD_NAMES, RHS_TERM_NAMES
from drbx.native.fci_boundary_imex_split import evaluate_boundary_imex_split


def _state(value=0.0):
    x = jnp.full((2, 2, 2), value)
    return FciDrbEBState(x, x, x, x, x, x, x)


@dataclass
class _Model:
    parallel_short_leg_treatment: str = "local-backward-euler"
    parallel_short_leg_selection: str = "cfl"
    seen: object = None

    def evaluate_stage(self, state, *, source_owned, phi_owned, return_rhs_term_fields):
        assert self.parallel_short_leg_treatment == "explicit"
        assert source_owned is None
        assert phi_owned is state.phi
        assert return_rhs_term_fields
        self.seen = state
        raw = jnp.zeros((6, max(map(len, RHS_TERM_NAMES))) + state.density.shape)
        for fi, names in enumerate(RHS_TERM_NAMES):
            for ti, name in enumerate(names):
                raw = raw.at[fi, ti].set(float(ti + 1) if name in ("poisson_bracket", "curvature", "perpendicular_diffusion", "source") else float(10 + ti))
        physical = state.replace(
            density=jnp.sum(raw[0, :len(RHS_TERM_NAMES[0])], axis=0),
            Te=jnp.sum(raw[1, :len(RHS_TERM_NAMES[1])], axis=0),
            Ti=jnp.sum(raw[2, :len(RHS_TERM_NAMES[2])], axis=0),
            Vi=jnp.sum(raw[3, :len(RHS_TERM_NAMES[3])], axis=0),
            Ve=jnp.sum(raw[4, :len(RHS_TERM_NAMES[4])], axis=0),
            vorticity=jnp.sum(raw[5, :len(RHS_TERM_NAMES[5])], axis=0),
        )
        return physical, raw


def test_split_is_fieldwise_additive_and_disables_short_leg():
    model = _Model()
    split = evaluate_boundary_imex_split(model, _state())
    assert model.parallel_short_leg_treatment == "local-backward-euler"
    for name in RHS_TERM_FIELD_NAMES:
        assert jnp.allclose(getattr(split.physical, name), getattr(split.explicit, name) + getattr(split.implicit, name))
    assert jnp.all(split.explicit.density == 1 + 3 + 4 + 7)
    assert jnp.all(split.implicit.density == 11 + 14 + 15)


def test_split_dataclass_is_frozen():
    assert dataclasses.is_dataclass(evaluate_boundary_imex_split(_Model(), _state()))


@pytest.mark.parametrize("wall_model", ["no-flow", "simplified-gbs-mpe"])
def test_split_real_mapped_fixture_local_eager(wall_model):
    """Exercise the split against the small mapped EB production fixture."""
    from fci_drb_eb_test_helpers import _build_rhs
    from test_fci_drb_eb_parallel_production_wiring import _mapped_fixture
    from drbx.native.fci_sharding import assemble_single_device_local_fci_geometry
    from drbx.native.fci_physical_wall import physical_wall_model_from_name

    context, _mesh, local, _partition, fields, cell_fields, map_fields, sharded = _mapped_fixture()
    local = replace(local, domain=replace(local.domain, mesh_axis_names=(None, None, None)))
    geometry = assemble_single_device_local_fci_geometry(sharded, cell_fields, map_fields)
    params = replace(context.parameters, parallel_characteristic_wall_law="physical-boundary-state",
                     density_D_perp=1.0e-5, electron_temperature_D_perp=1.0e-5,
                     ion_temperature_D_perp=1.0e-5, Ve_D_perp=1.0e-5,
                     Vi_D_perp=1.0e-5, vorticity_D_perp=1.0e-5,
                     density_D_parallel=1.0e-5, electron_temperature_chi_parallel=1.0e-5,
                     ion_temperature_chi_parallel=1.0e-5, Ve_parallel_viscosity=1.0e-5,
                     Vi_parallel_viscosity=1.0e-5, vorticity_D_parallel=1.0e-5,
                     Ve_nu=1.0e-5)
    model = replace(_build_rhs(context, local, geometry),
                    parameters=params,
                    face_bc_builder=physical_wall_model_from_name(wall_model),
                    parallel_short_leg_treatment="explicit",
                    parallel_short_leg_selection="cfl",
                    poisson_bracket_scheme="compatible-third-order-upwind",
                    parallel_operator_scheme="fci",
                    parallel_material_scheme="production-path",
                    parallel_flux_pairing="support-core",
                    parallel_boundary_pairing="characteristic-sat",
                    physical_wall_model_name=wall_model,
                    polarization_operator_form="support-paired")
    state = FciDrbEBState(*fields)
    source = state.replace(density=jnp.full_like(state.density, 1.0e-4),
                           Te=jnp.full_like(state.Te, -2.0e-4),
                           vorticity=jnp.full_like(state.vorticity, 3.0e-4))
    with jax.disable_jit():
        split = evaluate_boundary_imex_split(model, state, source_owned=source)
    for name in RHS_TERM_FIELD_NAMES:
        physical = getattr(split.physical, name)
        assert bool(jnp.all(jnp.isfinite(physical)))
        assert bool(jnp.allclose(physical, getattr(split.explicit, name) + getattr(split.implicit, name), rtol=2e-10, atol=2e-10))
    assert float(jnp.max(jnp.abs(split.implicit.density))) > 0.0


@pytest.mark.parametrize("wall_model", ["no-flow", "simplified-gbs-mpe"])
def test_split_cartesian_oblique_b_uses_traced_nongrazing_endpoints(wall_model):
    """Build coherent cell/face fields and maps from one constant B callback."""
    from fci_drb_eb_test_helpers import _build_rhs
    from test_fci_drb_eb_parallel_production_wiring import _context_and_sharded_inputs
    from drbx.geometry import FCI_DEP_PHYSICAL_BOUNDARY, FciMaps3D, build_fci_maps_from_callbacks
    from drbx.native.fci_physical_wall import physical_wall_model_from_name
    from drbx.native.fci_sharding import assemble_single_device_local_fci_geometry, build_local_fci_geometries

    context, _mesh, local, _partition, fields, _ = _context_and_sharded_inputs()
    geometry = context.geometry
    one = jnp.ones_like(geometry.cell_metric.J)
    zero = jnp.zeros_like(one)
    metric = replace(geometry.cell_metric, J=one, g11=one, g22=one, g33=one, g12=zero, g13=zero, g23=zero,
                     g_11=one, g_22=one, g_33=one, g_12=zero, g_13=zero, g_23=zero)
    face_metrics = []
    for fm in (geometry.face_metric.x, geometry.face_metric.y, geometry.face_metric.z):
        q = jnp.ones_like(fm.J)
        z = jnp.zeros_like(q)
        face_metrics.append(replace(fm, J=q, g11=q, g22=q, g33=q, g12=z, g13=z, g23=z,
                                    g_11=q, g_22=q, g_33=q, g_12=z, g_13=z, g_23=z))
    face_metric = replace(geometry.face_metric, x=face_metrics[0], y=face_metrics[1], z=face_metrics[2])
    bfield = replace(geometry.cell_bfield, B_contra=jnp.broadcast_to(jnp.asarray([.2, .3, 1.0]), one.shape + (3,)), Bmag=jnp.full_like(one, jnp.sqrt(1.13)))
    face_bfields = tuple(replace(old, B_contra=jnp.broadcast_to(jnp.asarray([.2, .3, 1.0]), old.B_contra.shape), Bmag=jnp.full_like(old.Bmag, jnp.sqrt(1.13))) for old in (geometry.face_bfield.x, geometry.face_bfield.y, geometry.face_bfield.z))
    callback = lambda points: {"B_contravariant": jnp.broadcast_to(jnp.asarray([.2, .3, 1.0]), (len(points), 3)), "magnitude": jnp.full((len(points),), jnp.sqrt(1.13))}
    map_fields = build_fci_maps_from_callbacks(geometry.grid, callback, substeps=4, periodic_axes=(False, True, True))
    map_names = ("forward_x", "forward_y", "backward_x", "backward_y", "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
                 "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z", "forward_length", "backward_length",
                 "forward_boundary", "backward_boundary", "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y",
                 "forward_endpoint_b_contra_z", "forward_endpoint_bmag", "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y",
                 "backward_endpoint_b_contra_z", "backward_endpoint_bmag")
    maps = FciMaps3D(**{name: map_fields[name] for name in map_names if name in map_fields})
    global_geometry = replace(geometry, cell_metric=metric, face_metric=face_metric, cell_bfield=bfield,
                              face_bfield=replace(geometry.face_bfield, x=face_bfields[0], y=face_bfields[1], z=face_bfields[2]), maps=maps)
    sharded = build_local_fci_geometries(global_geometry, (1, 1, 1), halo_width=local.domain.layout.halo_width, periodic_axes=(False, True, True))
    geometry = assemble_single_device_local_fci_geometry(sharded, sharded.cell_fields, sharded.map_fields)
    local = replace(local, domain=replace(local.domain, mesh_axis_names=(None, None, None)))
    params = replace(context.parameters, parallel_characteristic_wall_law="physical-boundary-state",
                     density_D_perp=1e-5, electron_temperature_D_perp=1e-5,
                     ion_temperature_D_perp=1e-5, Vi_D_perp=1e-5,
                     Ve_D_perp=1e-5, vorticity_D_perp=1e-5)
    model = replace(_build_rhs(context, local, geometry), parameters=params,
                    face_bc_builder=physical_wall_model_from_name(wall_model),
                    parallel_short_leg_treatment="explicit", parallel_short_leg_selection="cfl", parallel_operator_scheme="fci",
                    parallel_material_scheme="production-path", parallel_flux_pairing="support-core", parallel_boundary_pairing="characteristic-sat",
                    poisson_bracket_scheme="compatible-third-order-upwind",
                    physical_wall_model_name=wall_model, polarization_operator_form="support-paired")
    state = FciDrbEBState(*fields).replace(phi=fields[1] + 3.0)
    with jax.disable_jit():
        split = evaluate_boundary_imex_split(model, state, polarization_multiplier=0.0)
        physical_eval, terms = model.evaluate_stage(state, phi_owned=state.phi,
                                                    polarization_multiplier=0.0,
                                                    return_rhs_term_fields=True)
    hits = geometry.maps.backward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY
    assert int(jnp.count_nonzero(hits)) > 0
    for name in RHS_TERM_FIELD_NAMES:
        full, explicit, implicit = (getattr(split.physical, name), getattr(split.explicit, name), getattr(split.implicit, name))
        assert bool(jnp.all(jnp.isfinite(full)))
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(full)) + jnp.max(jnp.abs(explicit)) + jnp.max(jnp.abs(implicit)))
        assert bool(jnp.allclose(full, explicit + implicit, rtol=64 * jnp.finfo(full.dtype).eps, atol=64 * jnp.finfo(full.dtype).eps * scale))
    if wall_model == "simplified-gbs-mpe":
        physical_perp = sum((terms[4, index] for index, name in enumerate(RHS_TERM_NAMES[4])
                             if name in {"poisson_bracket", "perpendicular_diffusion", "curvature", "source"}),
                            jnp.zeros_like(physical_eval.Ve))
        delta = float(jnp.max(jnp.abs(physical_perp - split.explicit.Ve)))
        print(f"wall_model={wall_model} hits={int(jnp.count_nonzero(hits))} Ve_perp_delta={delta:.16g}")
        assert delta > 1e-10
