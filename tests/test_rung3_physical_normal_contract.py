"""Physics-level checks for the topology-halo GBS-MPE wall contract."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native import FciDrbEBState
from drbx.native.fci_boundaries import build_local_boundary_face_trace_from_halo
from drbx.native.fci_halo import (
    neumann_face_trace_physical_affine,
    physical_normal_derivative_for_face_trace,
)
from drbx.native.fci_physical_wall import physical_wall_model_from_name
from test_neumann_face_trace_affine import _fixture, _halo

jax.config.update("jax_enable_x64", True)


def _case():
    layout, domain, geometry, filler = _fixture(2)
    b_axes = tuple(SimpleNamespace(
        B_contra_owned=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64)[0], layout.face_control_shape(axis) + (3,)
        )
    ) for axis in range(3))
    object.__setattr__(geometry, "face_bfield", SimpleNamespace(axes=b_axes))
    shape = layout.owned_shape
    ijk = jnp.indices(shape, dtype=jnp.float64)
    x = ijk[0]
    y = 2.0 * jnp.pi * ijk[1] / shape[1]
    state = FciDrbEBState(
        1.0 + 0.02 * x + 0.03 * jnp.sin(y),
        0.1 * x + 0.02 * jnp.cos(y),
        4.0 + 0.05 * x + 0.1 * jnp.sin(y),
        1.0 + 0.02 * x + 0.03 * jnp.cos(y),
        0.2 + 0.01 * x + 0.02 * jnp.sin(y),
        jnp.full(shape, 0.2),
        jnp.zeros(shape),
    )
    params = SimpleNamespace(Te0=4.0, Ti0=1.0, tau=2.0, mi_over_me=10.0)
    halos = {name: _halo(getattr(state, name), layout) for name in ("density", "phi", "Vi", "Te", "Ti")}
    return layout, domain, geometry, filler, state, params, halos


def _traces(bundle, state, filler, domain, geometry, halos):
    result = {}
    for name in ("density", "phi", "Te", "Ti"):
        bc = getattr(bundle, name)
        result[name] = build_local_boundary_face_trace_from_halo(
            filler(halos[name], domain, bc), geometry, domain, bc
        )
    return result


def test_simplified_gbs_mpe_physical_normal_face_contract_and_jit():
    layout, domain, geometry, filler, state, params, halos = _case()
    model = physical_wall_model_from_name("simplified-gbs-mpe")
    model = replace(model, conducting_sheath_wall_potential=0.0)
    bundle = model(state, geometry, domain, params, topology_halos=halos)
    traces = _traces(bundle, state, filler, domain, geometry, halos)
    mu = params.mi_over_me
    for side, index, sigma in (("lower", 0, -1.0), ("upper", -1, 1.0)):
        te = traces["Te"].value_x[index]
        ti = traces["Ti"].value_x[index]
        nface = traces["density"].value_x[index]
        phiface = traces["phi"].value_x[index]
        g_n = physical_normal_derivative_for_face_trace(halos["density"], nface, geometry, domain, 0, side)
        g_phi = physical_normal_derivative_for_face_trace(halos["phi"], phiface, geometry, domain, 0, side)
        g_te = physical_normal_derivative_for_face_trace(halos["Te"], te, geometry, domain, 0, side)
        g_ti = physical_normal_derivative_for_face_trace(halos["Ti"], ti, geometry, domain, 0, side)
        cb = jnp.sqrt(te + params.tau * ti)
        vi_owner = state.Vi[index]
        vi_face = bundle.Vi.value_x[index]
        vi_base, vi_response = neumann_face_trace_physical_affine(halos["Vi"], geometry, domain, 0, side)
        g_vi = (vi_face - vi_base) / vi_response
        np.testing.assert_allclose(g_te, 0.0, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(g_ti, 0.0, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(g_n + sigma * nface * g_vi / cb, 0.0, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(g_phi + sigma * te * g_vi / cb, 0.0, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(vi_face, sigma * jnp.maximum(cb, sigma * vi_owner), rtol=0.0, atol=1e-11)
        expected_ve = sigma * jnp.sqrt(mu * te / (2.0 * jnp.pi)) * jnp.exp(-phiface / te)
        np.testing.assert_allclose(bundle.Ve.value_x[index], expected_ve, rtol=0.0, atol=1e-11)
        old_ve = sigma * jnp.sqrt(mu * state.Te[index] / (2.0 * jnp.pi)) * jnp.exp(-state.phi[index] / state.Te[index])
        assert np.max(np.abs(np.asarray(bundle.Ve.value_x[index] - old_ve))) > 1.0e-4

    def fields():
        out = model(state, geometry, domain, params, topology_halos=halos)
        return out.phi.value_x, out.Ve.value_x
    compiled = jax.jit(fields)()
    np.testing.assert_allclose(compiled[0], bundle.phi.value_x, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(compiled[1], bundle.Ve.value_x, rtol=0.0, atol=1e-11)


def test_simplified_gbs_mpe_invalid_density_eager_and_jit():
    layout, domain, geometry, _filler, state, params, halos = _case()
    bad = replace(state, density=-jnp.ones(layout.owned_shape))
    bad_halos = dict(halos, density=_halo(bad.density, layout))
    model = replace(
        physical_wall_model_from_name("simplified-gbs-mpe"),
        conducting_sheath_wall_potential=0.0,
    )
    with pytest.raises(ValueError, match="nonpositive"):
        model(bad, geometry, domain, params, topology_halos=bad_halos)
    compiled = jax.jit(lambda: (
        model(bad, geometry, domain, params, topology_halos=bad_halos).density.value_x,
        model(bad, geometry, domain, params, topology_halos=bad_halos).phi.value_x,
    ))()
    assert np.all(np.isnan(np.asarray(compiled[0])[[0, -1]]))
    assert np.all(np.isnan(np.asarray(compiled[1])[[0, -1]]))


def test_simplified_gbs_mpe_spmd_wall_masks_are_shard_local():
    if jax.device_count() < 2:
        pytest.skip("requires two CPU devices")
    layout, domain, geometry, filler, state, params, halos = _case()
    b_axes = tuple(SimpleNamespace(
        B_contra_owned=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64)[0], layout.face_control_shape(axis) + (3,)
        )
    ) for axis in range(3))
    object.__setattr__(geometry, "face_bfield", SimpleNamespace(axes=b_axes))
    # One static domain object uses the SPMD axis index to decide which
    # physical endpoint is runtime-owned on each shard.
    from drbx.geometry import ShardSpec3D, LocalDomain3D
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=(8, 5, 3), owned_start=(0, 0, 0), owned_stop=(4, 5, 3),
            shard_index=(0, 0, 0), shard_counts=(2, 1, 1),
            periodic_axes=(False, True, True), halo_width=2,
        ),
        layout=layout, mesh_axis_names=("s", None, None),
    )
    model = replace(physical_wall_model_from_name("simplified-gbs-mpe"), conducting_sheath_wall_potential=0.0)

    def one(s, h):
        out = model(s, geometry, domain, params, topology_halos=h)
        return (out.density.mask_x, out.phi.mask_x, out.Vi.mask_x, out.Ve.mask_x,
                out.density.value_x, out.phi.value_x, out.Vi.value_x, out.Ve.value_x)

    def poison_pair(value):
        return jnp.stack((value.at[2:].set(jnp.nan), value.at[:2].set(jnp.nan)))
    states = jax.tree_util.tree_map(poison_pair, state)
    halo_stacked = {
        name: jnp.stack((_halo(getattr(states, name)[0], layout),
                         _halo(getattr(states, name)[1], layout)))
        for name in halos
    }
    result = jax.pmap(one, axis_name="s", in_axes=(0, 0))(states, halo_stacked)
    masks = np.asarray(result[0])
    assert np.all(masks[0, 0]) and not np.any(masks[0, -1])
    assert not np.any(masks[1, 0]) and np.all(masks[1, -1])
    for values in result[4:]:
        values = np.asarray(values)
        assert np.all(np.isfinite(values[:, 0]))
        assert np.all(np.isfinite(values[:, -1]))
    for values in result[4:]:
        values = np.asarray(values)
        np.testing.assert_allclose(values[0, -1], 0.0)
        np.testing.assert_allclose(values[1, 0], 0.0)
