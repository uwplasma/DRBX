"""Focused tests for stage-local physical wall bundles."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import HaloLayout3D, LocalDomain3D, LocalFciGeometry3D, ShardSpec3D
from drbx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN
from drbx.native import FciDrbEBState
from drbx.native import fci_drb_EB_rhs as rhs_mod
from drbx.native.fci_physical_wall import (
    LegacyParallelVelocityPhysicalWallModel,
    NoFlowPhysicalWallModel,
    PHYSICAL_WALL_MODEL_NAMES,
    SimpleConductingSheathPhysicalWallModel,
    SimplifiedGbsMpePhysicalWallModel,
    equilibrium_warm_ion_floating_sheath_drop,
    physical_wall_model_from_name,
    resolve_fci_material_wall_endpoint_state,
    warm_ion_floating_sheath_face_potential_target,
)

jax.config.update("jax_enable_x64", True)


def _inputs(*, phi=0.0, vi=0.0, ve=0.0, b_sign=1.0):
    shape = (3, 3, 3)
    layout = HaloLayout3D(shape, 1)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    coords = jnp.asarray([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=jnp.float64)
    faces = jnp.asarray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    def face_metric(matrix):
        values = jnp.broadcast_to(
            jnp.asarray(matrix, dtype=jnp.float64), layout.cell_halo_shape + (3, 3)
        )
        return SimpleNamespace(g_contra=values)

    state = SimpleNamespace(
        density=jnp.ones(shape, dtype=jnp.float64),
        phi=jnp.full(shape, phi, dtype=jnp.float64),
        Te=jnp.full(shape, 4.0, dtype=jnp.float64),
        Ti=jnp.ones(shape, dtype=jnp.float64),
        Vi=jnp.full(shape, vi, dtype=jnp.float64),
        Ve=jnp.full(shape, ve, dtype=jnp.float64),
        vorticity=zeros,
    )

    def face_bfield(axis):
        face_shape = list(shape)
        face_shape[axis] += 1
        values = jnp.zeros(tuple(face_shape) + (3,), dtype=jnp.float64)
        values = values.at[..., axis].set(b_sign)
        return SimpleNamespace(B_contra_owned=values)

    geometry = SimpleNamespace(
        layout=layout,
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=coords, faces_halo=faces),
            y=SimpleNamespace(centers_halo=coords, faces_halo=faces),
            z=SimpleNamespace(centers_halo=coords, faces_halo=faces),
        ),
        face_metric=SimpleNamespace(
            axes=tuple(
                face_metric(np.eye(3, dtype=np.float64))
                for _ in range(3)
            )
        ),
        face_bfield=SimpleNamespace(
            axes=tuple(face_bfield(axis) for axis in range(3))
        ),
    )
    domain = SimpleNamespace(
        layout=layout,
        runtime_has_physical_lower=lambda axis: True,
        runtime_has_physical_upper=lambda axis: True,
    )
    parameters = SimpleNamespace(
        Te0=4.0,
        Ti0=1.0,
        tau=2.0,
        mi_over_me=10.0,
    )
    return state, geometry, domain, parameters


def _real_geometry_domain(geometry, domain):
    """Adapt the compact fixture namespaces for halo trace inversion tests."""
    real_geometry = object.__new__(LocalFciGeometry3D)
    for name, value in geometry.__dict__.items():
        object.__setattr__(real_geometry, name, value)
    shape = tuple(int(v) for v in domain.layout.owned_shape)
    shard_spec = ShardSpec3D(
        global_shape=shape, owned_start=(0, 0, 0), owned_stop=shape,
        shard_index=(0, 0, 0), shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False), halo_width=domain.layout.halo_width,
    )
    real_domain = LocalDomain3D(
        shard_spec=shard_spec, layout=domain.layout,
        mesh_axis_names=(None, None, None),
    )
    return real_geometry, real_domain


def _halo_prefilled_inputs():
    shape = (3, 3, 3)
    layout = HaloLayout3D(shape, 1)
    shard_spec = ShardSpec3D(
        global_shape=shape,
        owned_start=(0, 0, 0),
        owned_stop=shape,
        shard_index=(0, 0, 0),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, True),
        halo_width=1,
    )
    domain = LocalDomain3D(
        shard_spec=shard_spec,
        layout=layout,
        mesh_axis_names=(None, None, None),
    )
    coords = jnp.asarray([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=jnp.float64)
    faces = jnp.asarray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    skew = np.asarray(
        [
            [1.00, 0.0, 0.50],
            [0.0, 1.00, 0.0],
            [0.50, 0.0, 1.00],
        ],
        dtype=np.float64,
    )

    def face_metric(matrix):
        values = jnp.broadcast_to(
            jnp.asarray(matrix, dtype=jnp.float64), layout.cell_halo_shape + (3, 3)
        )
        return SimpleNamespace(g_contra=values)

    geometry = SimpleNamespace(
        layout=layout,
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=coords, faces_halo=faces),
            y=SimpleNamespace(centers_halo=coords, faces_halo=faces),
            z=SimpleNamespace(centers_halo=coords, faces_halo=faces),
        ),
        face_metric=SimpleNamespace(
            axes=tuple(face_metric(skew) for _ in range(3))
        ),
    )

    field_halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    z_pattern = jnp.asarray([0.0, 2.0, 0.0], dtype=jnp.float64)
    owned = jnp.indices(shape, dtype=jnp.float64)[0] + z_pattern[None, None, :]
    field_halo = field_halo.at[1:4, 1:4, 1:4].set(owned)
    field_halo = field_halo.at[1:4, 1:4, 0].set(owned[:, :, -1])
    field_halo = field_halo.at[1:4, 1:4, -1].set(owned[:, :, 0])
    # The helper only reads the owned slice plus the tangential halos, but fill
    # the remaining halo faces with the nearest owned planes for completeness.
    field_halo = field_halo.at[0, 1:4, 1:4].set(owned[0, :, :])
    field_halo = field_halo.at[-1, 1:4, 1:4].set(owned[-1, :, :])
    field_halo = field_halo.at[:, 0, 1:4].set(field_halo[:, 1, 1:4])
    field_halo = field_halo.at[:, -1, 1:4].set(field_halo[:, -2, 1:4])
    return geometry, domain, field_halo


def _wall_values(bundle, axis=0, side="upper"):
    return getattr(bundle, "Vi").value_x[-1 if side == "upper" else 0]


def _rung3_topology_halos(state, domain):
    """Construct the complete prefilled halo payload required by rung 3."""
    return {
        name: rhs_mod.inject_owned_field_to_halo(getattr(state, name), domain.layout)
        for name in ("density", "phi", "Vi", "Te", "Ti")
    }


def test_model_names_are_the_four_rung_stage_local_choices():
    assert PHYSICAL_WALL_MODEL_NAMES == (
        "legacy-velocity-trace",
        "no-flow",
        "simple-conducting-sheath",
        "simplified-gbs-mpe",
    )
    assert isinstance(
        physical_wall_model_from_name("legacy-velocity-trace"),
        LegacyParallelVelocityPhysicalWallModel,
    )
    assert isinstance(physical_wall_model_from_name("no-flow"), NoFlowPhysicalWallModel)
    assert isinstance(
        physical_wall_model_from_name("simple-conducting-sheath"),
        SimpleConductingSheathPhysicalWallModel,
    )
    assert isinstance(
        physical_wall_model_from_name("simplified-gbs-mpe"),
        SimplifiedGbsMpePhysicalWallModel,
    )
    with pytest.raises(ValueError):
        physical_wall_model_from_name("ion-bohm-chodura")


def test_no_flow_preserves_scalar_bc_kinds_and_zero_values():
    state, geometry, domain, parameters = _inputs(vi=0.4, ve=-0.7)
    bundle = NoFlowPhysicalWallModel()(state, geometry, domain, parameters)
    for field in (bundle.Vi, bundle.Ve, bundle.phi, bundle.vorticity):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.Vi.value_x[0], bundle.Vi.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.Ve.value_x[0], bundle.Ve.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.phi.value_x[0], bundle.phi.value_x[-1])), 0.0)
    np.testing.assert_allclose(jnp.stack((bundle.vorticity.value_x[0], bundle.vorticity.value_x[-1])), 0.0)
    for field in (bundle.density, bundle.Te, bundle.Ti):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_NEUMANN)


def test_legacy_model_keeps_neumann_or_zero_dirichlet_velocity_trace():
    state, geometry, domain, parameters = _inputs()
    neumann = LegacyParallelVelocityPhysicalWallModel("neumann")(
        state, geometry, domain, parameters
    )
    zero = LegacyParallelVelocityPhysicalWallModel("dirichlet-zero")(
        state, geometry, domain, parameters
    )
    np.testing.assert_array_equal(jnp.stack((neumann.Vi.kind_x[0], neumann.Vi.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(jnp.stack((neumann.Ve.kind_x[0], neumann.Ve.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(jnp.stack((zero.Vi.kind_x[0], zero.Vi.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_array_equal(jnp.stack((zero.Ve.kind_x[0], zero.Ve.kind_x[-1])), BC_DIRICHLET)


def test_conducting_sheath_default_is_equilibrium_compatible_both_orientations():
    for b_sign in (1.0, -1.0):
        state, geometry, domain, parameters = _inputs(b_sign=b_sign, phi=0.0)
        bundle = SimpleConductingSheathPhysicalWallModel()(
            state, geometry, domain, parameters
        )
        c_b = np.sqrt(4.0 + 2.0)
        expected_upper = b_sign * c_b
        expected_lower = -b_sign * c_b
        np.testing.assert_allclose(bundle.Vi.value_x[-1], expected_upper)
        np.testing.assert_allclose(bundle.Ve.value_x[-1], expected_upper)
        np.testing.assert_allclose(bundle.Vi.value_x[0], expected_lower)
        np.testing.assert_allclose(bundle.Ve.value_x[0], expected_lower)
        np.testing.assert_allclose(
            bundle.Vi.value_x[-1] - bundle.Ve.value_x[-1], 0.0, atol=1e-13
        )


def test_warm_ion_reference_drop_equalizes_maxwellian_electron_and_bohm_speeds():
    te, ti, tau, mass_ratio = 4.0, 1.0, 2.0, 10.0
    drop = equilibrium_warm_ion_floating_sheath_drop(
        te, ti, tau=tau, mi_over_me=mass_ratio
    )
    electron_loss_speed = np.sqrt(mass_ratio * te / (2.0 * np.pi)) * np.exp(
        -float(drop) / te
    )
    warm_bohm_speed = np.sqrt(te + tau * ti)
    np.testing.assert_allclose(electron_loss_speed, warm_bohm_speed, rtol=1e-13)


def test_warm_ion_reference_drop_has_standard_cold_ion_lambda_limit():
    mass_ratio = 1836.0
    drop = equilibrium_warm_ion_floating_sheath_drop(
        1.0, 3.0, tau=0.0, mi_over_me=mass_ratio
    )
    lambda0 = 0.5 * np.log(mass_ratio / (2.0 * np.pi))
    np.testing.assert_allclose(drop, lambda0, rtol=1e-13)


def test_wall_reference_target_preserves_gauge_shift_invariance():
    te, ti, tau, mass_ratio = 4.0, 1.0, 2.0, 10.0
    drop = equilibrium_warm_ion_floating_sheath_drop(
        te, ti, tau=tau, mi_over_me=mass_ratio
    )
    wall = -drop
    face = warm_ion_floating_sheath_face_potential_target(
        wall, te, ti, tau=tau, mi_over_me=mass_ratio
    )
    shift = 0.37
    shifted_face = warm_ion_floating_sheath_face_potential_target(
        wall + shift, te, ti, tau=tau, mi_over_me=mass_ratio
    )
    np.testing.assert_allclose(face, 0.0, atol=1e-13)
    np.testing.assert_allclose(shifted_face - shift, face, atol=1e-13)
    # The electron response depends only on the potential difference, so a
    # common shift of phi_face and phi_wall leaves the loss speed unchanged.
    base_speed = np.sqrt(mass_ratio * te / (2.0 * np.pi)) * np.exp(
        -(face - wall) / te
    )
    shifted_speed = np.sqrt(mass_ratio * te / (2.0 * np.pi)) * np.exp(
        -(shifted_face - (wall + shift)) / te
    )
    np.testing.assert_allclose(shifted_speed, base_speed, rtol=1e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electron_temperature": 0.0, "ion_temperature": 1.0},
        {"electron_temperature": np.nan, "ion_temperature": 1.0},
        {"electron_temperature": 1.0, "ion_temperature": 0.0},
        {"electron_temperature": 1.0, "ion_temperature": np.inf},
        {"electron_temperature": 1.0, "ion_temperature": 1.0, "tau": -1.0},
        {"electron_temperature": 1.0, "ion_temperature": 1.0, "mi_over_me": 0.0},
    ],
)
def test_warm_ion_reference_rejects_invalid_thermodynamics(kwargs):
    with pytest.raises(ValueError, match="finite"):
        equilibrium_warm_ion_floating_sheath_drop(**kwargs)


def test_conducting_sheath_passes_supersonic_owner_flow_and_rejects_subsonic():
    # At the upper wall B.n>0.  Outward owner flow passes through; inward or
    # subsonic owner flow is replaced by the sonic target.
    state, geometry, domain, parameters = _inputs(vi=3.0, ve=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[-1], 3.0)

    state, geometry, domain, parameters = _inputs(vi=0.2, ve=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[-1], np.sqrt(6.0))
    np.testing.assert_allclose(bundle.Vi.value_x[0], -np.sqrt(6.0))


def test_conducting_sheath_nonzero_current_for_perturbed_plasma_potential():
    state, geometry, domain, parameters = _inputs(phi=0.25)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    c_b = np.sqrt(6.0)
    np.testing.assert_allclose(bundle.Vi.value_x[-1], c_b)
    assert not np.allclose(bundle.Ve.value_x[-1], c_b)
    current = np.asarray(state.density[-1]) * (
        np.asarray(bundle.Vi.value_x[-1]) - np.asarray(bundle.Ve.value_x[-1])
    )
    assert np.max(np.abs(current)) > 1e-8


def test_conducting_sheath_fixed_wall_potential_override_changes_electron_loss():
    state, geometry, domain, parameters = _inputs(phi=0.0)
    default = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    override = physical_wall_model_from_name(
        "simple-conducting-sheath",
        conducting_sheath_wall_potential=0.0,
    )(state, geometry, domain, parameters)
    assert not np.allclose(default.Ve.value_x[-1], override.Ve.value_x[-1])
    np.testing.assert_allclose(override.phi.value_x, 0.0)


def test_conducting_sheath_preserves_neumann_thermodynamics_and_provisional_vorticity():
    state, geometry, domain, parameters = _inputs()
    bundle = physical_wall_model_from_name("simple-conducting-sheath")(
        state, geometry, domain, parameters
    )
    for field in (bundle.density, bundle.Te, bundle.Ti):
        np.testing.assert_array_equal(jnp.stack((field.kind_x[0], field.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(
        jnp.stack((bundle.Vi.kind_x[0], bundle.Vi.kind_x[-1])), BC_DIRICHLET
    )
    np.testing.assert_array_equal(
        jnp.stack((bundle.Ve.kind_x[0], bundle.Ve.kind_x[-1])), BC_DIRICHLET
    )
    np.testing.assert_array_equal(jnp.stack((bundle.vorticity.kind_x[0], bundle.vorticity.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.vorticity.value_x[0], bundle.vorticity.value_x[-1])), 0.0)
    np.testing.assert_array_equal(jnp.stack((bundle.phi.kind_x[0], bundle.phi.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_allclose(jnp.stack((bundle.phi.value_x[0], bundle.phi.value_x[-1])), 0.0)


def test_simplified_gbs_mpe_uses_metric_aware_neumann_density_and_phi():
    state, geometry, domain, parameters = _inputs()
    geometry, domain = _real_geometry_domain(geometry, domain)
    bundle = physical_wall_model_from_name("simplified-gbs-mpe")(
        state, geometry, domain, parameters,
        topology_halos=_rung3_topology_halos(state, domain),
    )
    np.testing.assert_array_equal(jnp.stack((bundle.density.kind_x[0], bundle.density.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_array_equal(jnp.stack((bundle.phi.kind_x[0], bundle.phi.kind_x[-1])), BC_NEUMANN)
    assert np.all(np.isfinite(np.asarray(bundle.density.value_x)))
    assert np.all(np.isfinite(np.asarray(bundle.phi.value_x)))
    # The topology-halo affine map, rather than the legacy owner-only jump,
    # is the contract exercised here.
    assert np.all(np.asarray(bundle.density.value_x[0]) < 0.0)
    np.testing.assert_array_equal(jnp.stack((bundle.Te.kind_x[0], bundle.Te.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_allclose(jnp.stack((bundle.Te.value_x[0], bundle.Te.value_x[-1])), 0.0)
    np.testing.assert_array_equal(jnp.stack((bundle.Ti.kind_x[0], bundle.Ti.kind_x[-1])), BC_NEUMANN)
    np.testing.assert_allclose(jnp.stack((bundle.Ti.value_x[0], bundle.Ti.value_x[-1])), 0.0)
    np.testing.assert_array_equal(jnp.stack((bundle.Vi.kind_x[0], bundle.Vi.kind_x[-1])), BC_DIRICHLET)
    np.testing.assert_array_equal(jnp.stack((bundle.Ve.kind_x[0], bundle.Ve.kind_x[-1])), BC_DIRICHLET)
    assert not np.any(np.asarray(bundle.vorticity.mask_x))
    np.testing.assert_array_equal(bundle.vorticity.kind_x, 0)
    np.testing.assert_array_equal(bundle.vorticity.value_x, 0.0)
    assert physical_wall_model_from_name("simplified-gbs-mpe").vorticity_from_polarization is True


def test_simplified_gbs_mpe_face_bcs_passes_all_topology_prefilled_halos():
    state, geometry, domain, parameters = _inputs()
    geometry, domain = _real_geometry_domain(geometry, domain)
    recorded: dict[str, object] = {}
    model = physical_wall_model_from_name("simplified-gbs-mpe")

    def builder(wall_state, geometry_, domain_, parameters_, *, topology_halos=None):
        recorded["topology_halos"] = topology_halos
        return model(wall_state, geometry_, domain_, parameters_, topology_halos=topology_halos)

    halo_calls: list[tuple[str, tuple[int, ...]]] = []
    topology_count = {"value": 0}

    def halo_exchange(field, _domain):
        halo_calls.append(("halo", tuple(int(v) for v in field.shape)))
        return field

    def topology_filler(field, _domain):
        halo_calls.append(("topology", tuple(int(v) for v in field.shape)))
        offset = 0.125 * (topology_count["value"] + 1)
        topology_count["value"] += 1
        return field + offset

    rhs_stub = SimpleNamespace(
        face_bc_builder=builder,
        geometry=geometry,
        domain=domain,
        parameters=parameters,
        physical_wall_model_name="simplified-gbs-mpe",
        control_volume_geometry=None,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        _owner_field=lambda values: values,
    )

    face_bc = rhs_mod.LocalFciDrbEBRhs._face_bcs(rhs_stub, state)
    halos = recorded["topology_halos"]
    assert set(halos) == {"density", "phi", "Vi", "Te", "Ti"}
    for name, value in halos.items():
        assert tuple(value.shape) == tuple(domain.layout.cell_halo_shape)
        assert np.all(np.isfinite(np.asarray(value)))
    assert sum(kind == "topology" for kind, _shape in halo_calls) == 5
    assert all(shape == tuple(domain.layout.cell_halo_shape) for _kind, shape in halo_calls)
    assert np.all(np.isfinite(np.asarray(face_bc.density.value_x)))
    with pytest.raises(ValueError, match="topology_halos"):
        model(state, geometry, domain, parameters)


def test_simplified_gbs_mpe_rlp_wall_bcs_ignore_noncanonical_alias_values():
    """RLP wall laws must see owner-expanded values, never sparse aliases."""

    state, geometry, domain, parameters = _inputs(vi=0.25)
    geometry, domain = _real_geometry_domain(geometry, domain)
    shape = state.Vi.shape
    owner_i, owner_j, owner_k = jnp.indices(shape, dtype=jnp.int32)
    # Make the upper-x plane a merged source of the lower-x owner.  Its raw
    # state is deliberately non-finite; a direct wall-model call would read
    # this plane and poison the rung-3 face values.
    owner_i = owner_i.at[-1].set(0)
    active_owner = jnp.ones(shape, dtype=bool).at[-1].set(False)
    cells = SimpleNamespace(
        layout=domain.layout,
        shape=shape,
        owner_i=owner_i,
        owner_j=owner_j,
        owner_k=owner_k,
        is_active_owner=active_owner,
        owner_is_remote=jnp.zeros(shape, dtype=bool),
        remote_owner_halo_i=jnp.zeros(shape, dtype=jnp.int32),
        remote_owner_halo_j=jnp.zeros(shape, dtype=jnp.int32),
        remote_owner_halo_k=jnp.zeros(shape, dtype=jnp.int32),
        raw_volume=jnp.ones(shape, dtype=jnp.float64),
    )
    state = FciDrbEBState(
        density=state.density.at[-1].set(jnp.nan),
        phi=state.phi.at[-1].set(jnp.nan),
        Te=state.Te.at[-1].set(jnp.nan),
        Ti=state.Ti.at[-1].set(jnp.nan),
        Vi=state.Vi.at[-1].set(jnp.nan),
        Ve=state.Ve.at[-1].set(jnp.nan),
        vorticity=state.vorticity.at[-1].set(jnp.nan),
    )
    from drbx.native import fci_drb_EB_rhs as rhs_mod

    model = physical_wall_model_from_name("simplified-gbs-mpe")
    rhs_stub = SimpleNamespace(
        face_bc_builder=model,
        geometry=geometry,
        domain=domain,
        parameters=parameters,
        physical_wall_model_name="simplified-gbs-mpe",
        control_volume_geometry=SimpleNamespace(cells=cells),
        halo_exchange=lambda field, _domain: field,
        topology_filler=lambda field, _domain: field,
    )
    rhs_stub._owner_field = lambda values: jnp.where(
        cells.is_active_owner, jnp.asarray(values, dtype=jnp.float64), 0.0
    )
    rhs_stub._owner_state = lambda current: rhs_mod.LocalFciDrbEBRhs._owner_state(
        rhs_stub, current
    )

    face_bc = rhs_mod.LocalFciDrbEBRhs._face_bcs(rhs_stub, state)
    wall_state = rhs_mod.LocalFciDrbEBRhs._materialized_wall_state(rhs_stub, state)
    # Every merged-source wall value is supplied by its canonical owner.
    for value in (
        wall_state.density,
        wall_state.phi,
        wall_state.Te,
        wall_state.Ti,
        wall_state.Vi,
        wall_state.Ve,
        wall_state.vorticity,
    ):
        np.testing.assert_allclose(value[-1], value[0])
    assert np.all(np.isfinite(np.asarray(face_bc.density.value_x)))
    assert np.all(np.isfinite(np.asarray(face_bc.phi.value_x)))

    expected = model(
        wall_state, geometry, domain, parameters,
        topology_halos=_rung3_topology_halos(wall_state, domain),
    )
    np.testing.assert_allclose(face_bc.density.value_x, expected.density.value_x)
    np.testing.assert_allclose(face_bc.phi.value_x, expected.phi.value_x)


@pytest.mark.parametrize("b_sign", (1.0, -1.0))
def test_simplified_gbs_mpe_respects_b_orientation_for_velocity_targets(b_sign):
    state, geometry, domain, parameters = _inputs(b_sign=b_sign)
    geometry, domain = _real_geometry_domain(geometry, domain)
    bundle = physical_wall_model_from_name("simplified-gbs-mpe")(
        state, geometry, domain, parameters,
        topology_halos=_rung3_topology_halos(state, domain),
    )
    c_b = np.sqrt(6.0)
    expected = b_sign * c_b
    np.testing.assert_allclose(bundle.Vi.value_x[-1], expected)
    np.testing.assert_allclose(bundle.Vi.value_x[0], -expected)
    assert np.all(np.asarray(bundle.Ve.value_x[-1]) * b_sign > 0.0)
    assert np.all(np.asarray(bundle.Ve.value_x[0]) * b_sign < 0.0)


def test_conducting_sheath_uses_owner_values_on_grazing_faces():
    state, geometry, domain, parameters = _inputs(vi=0.31, ve=-0.27, b_sign=0.0)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    np.testing.assert_allclose(bundle.Vi.value_x[0], 0.31)
    np.testing.assert_allclose(bundle.Vi.value_x[-1], 0.31)
    np.testing.assert_allclose(bundle.Ve.value_x[0], -0.27)
    np.testing.assert_allclose(bundle.Ve.value_x[-1], -0.27)


def test_nonpositive_thermodynamics_are_rejected_without_clamping():
    state, geometry, domain, parameters = _inputs()
    state.Te = state.Te.at[0, 0, 0].set(0.0)
    with pytest.raises(ValueError, match="finite positive"):
        SimpleConductingSheathPhysicalWallModel()(state, geometry, domain, parameters)


def test_inverse_sheath_exponential_is_not_clipped():
    state, geometry, domain, parameters = _inputs(phi=-0.1)
    bundle = SimpleConductingSheathPhysicalWallModel()(
        state, geometry, domain, parameters
    )
    # A negative sheath drop is not clipped to zero; the electron speed is
    # correspondingly larger than the equilibrium value.
    assert float(bundle.Ve.value_x[-1][0, 0]) > np.sqrt(6.0)


def test_endpoint_native_sheath_selects_branch_after_interpolating_b_normal():
    """Regression for the HSX theta=34 wrong-sign wall target."""

    plasma = jnp.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]])
    parameters = SimpleNamespace(Te0=1.0, Ti0=1.0, tau=1.0, mi_over_me=1836.0)
    resolved = resolve_fci_material_wall_endpoint_state(
        "simple-conducting-sheath",
        plasma,
        jnp.asarray([0.0]),
        # Continuous MetricEvaluator result at the traced backward endpoint.
        # It is not recoverable accurately by interpolating the coarse
        # coordinate-face samples below.
        jnp.asarray([-0.28590450084598473]),
        jnp.asarray([0.8775673487952317]),
        parameters,
    )
    np.testing.assert_allclose(resolved[0, 3], -np.sqrt(2.0), rtol=1e-13)
    np.testing.assert_allclose(resolved[0, 4], -np.sqrt(2.0), rtol=1e-13)
    assert float((-0.28590450084598473) * resolved[0, 3]) > 0.0

    # Evaluating sign(B.n)*cs on the four regular wall nodes first and then
    # interpolating gives +1.30463: a convex mixture dominated by the wrong
    # wall branch.  Even interpolating B itself gives +0.05123, also the wrong
    # sign compared with the continuous endpoint evaluation above.
    weights = np.asarray(
        [0.25268600122069357, 0.6944722707003752,
         0.014097290137977177, 0.038744437940954106]
    )
    nodal_b = np.asarray(
        [0.10622818693644548, 0.046367322079636265,
         0.1359753339941373, -0.25108642634927064]
    )
    old_target = np.sum(weights * np.sign(nodal_b) * np.sqrt(2.0))
    np.testing.assert_allclose(old_target, 1.3046277431678552, rtol=1e-13)
    assert float(np.sum(weights * nodal_b)) > 0.0
    assert old_target * (-0.28590450084598473) < 0.0


def test_endpoint_native_no_flow_commutes_with_interpolation():
    plasma = jnp.asarray([[1.2, 0.9, 1.1, 3.0, -4.0]])
    resolved = resolve_fci_material_wall_endpoint_state(
        "no-flow",
        plasma,
        jnp.asarray([0.2]),
        jnp.asarray([-0.4]),
        jnp.asarray([1.3]),
        SimpleNamespace(tau=1.0, mi_over_me=1836.0),
    )
    np.testing.assert_allclose(resolved[0, :3], plasma[0, :3])
    np.testing.assert_allclose(resolved[0, 3:], 0.0)
