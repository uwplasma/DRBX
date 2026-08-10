"""Distributed polar half-turn tests, including split-shard exchanges."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry import (
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
)
from drbx.native.fci_halo import (
    PolarAxisRegularScalarRule3D,
    PolarAxisRegularVectorRule3D,
)

from test_fci_halo import _domain


def _axis_theta_domain(*, halo_width: int):
    base = _domain(
        owned_shape=(3, 2, 2),
        shard_counts=(1, 3, 1),
        periodic_axes=(False, True, False),
        halo_width=halo_width,
        mesh_axis_names=(None, "theta", None),
    )
    spec = replace(
        base.shard_spec,
        side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
    )
    return replace(base, shard_spec=spec)


def _scalar_fields(domain, *, vector: bool):
    h = domain.layout.halo_width
    nx, local_nt, nz = domain.owned_shape
    fields = []
    for shard in range(3):
        field_shape = domain.layout.cell_halo_shape + ((3,) if vector else ())
        field = jnp.full(field_shape, -999.0)
        global_theta = shard * local_nt + jnp.arange(local_nt)
        i = jnp.arange(nx)[:, None, None]
        theta = global_theta[None, :, None]
        k = jnp.arange(nz)[None, None, :]
        base = 100.0 * i + 10.0 * theta + k
        if vector:
            owned = jnp.stack((base + 1.0, base + 2.0, base + 3.0), axis=-1)
        else:
            owned = base
        fields.append(field.at[domain.layout.owned_slices_cell].set(owned))
    return jax.device_put_sharded(fields, jax.local_devices()[:3])


def _expected_source(*, shard: int, domain, vector: bool, transform):
    nx, local_nt, nz = domain.owned_shape
    global_nt = 3 * local_nt
    half_turn = global_nt // 2
    target_theta = shard * local_nt + jnp.arange(local_nt)
    source_theta = (target_theta + half_turn) % global_nt
    i = jnp.arange(nx)[:, None, None]
    theta = source_theta[None, :, None]
    k = jnp.arange(nz)[None, None, :]
    base = 100.0 * i + 10.0 * theta + k
    if vector:
        source = jnp.stack((base + 1.0, base + 2.0, base + 3.0), axis=-1)
        return jnp.einsum("ij,...j->...i", transform, source[: domain.layout.halo_width][::-1])
    return base[: domain.layout.halo_width][::-1]


@pytest.mark.skipif(
    jax.local_device_count() < 3,
    reason="requires three local devices for split-shard topology exchange",
)
@pytest.mark.parametrize("halo_width", [1, 2])
def test_scalar_three_shard_half_turn_matches_global_reference(halo_width):
    domain = _axis_theta_domain(halo_width=halo_width)
    rule = PolarAxisRegularScalarRule3D(angle_axis_name="theta")
    filled = jax.pmap(lambda field: rule(field, domain), axis_name="theta")(
        _scalar_fields(domain, vector=False)
    )
    h = domain.layout.halo_width
    for shard in range(3):
        actual = filled[shard, :h, h : h + 2, h : h + 2]
        expected = _expected_source(shard=shard, domain=domain, vector=False, transform=None)
        assert jnp.array_equal(actual, expected)


@pytest.mark.skipif(
    jax.local_device_count() < 3,
    reason="requires three local devices for split-shard topology exchange",
)
@pytest.mark.parametrize("transform", [
    jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))),
    jnp.diag(jnp.asarray((1.0, -1.0, -1.0))),
])
def test_vector_three_shard_split_half_turn_applies_vector_or_density_transform(transform):
    domain = _axis_theta_domain(halo_width=2)
    # The legacy constructor stores the ppermute destination offset.  An
    # offset of -1 plus a local shift of +1 represents source shard t+1 and
    # then the first cell of that source shard, i.e. a global half-turn for
    # six angle cells split over three shards.
    rule = PolarAxisRegularVectorRule3D(
        axis=0,
        side="lower",
        angular_axis=1,
        mesh_axis_name="theta",
        source_shard_offset=-1,
        local_shift_cells=1,
        component_transform=transform,
    )
    filled = jax.pmap(lambda field: rule(field, domain), axis_name="theta")(
        _scalar_fields(domain, vector=True)
    )
    h = domain.layout.halo_width
    for shard in range(3):
        actual = filled[shard, :h, h : h + 2, h : h + 2, :]
        expected = _expected_source(shard=shard, domain=domain, vector=True, transform=transform)
        assert jnp.array_equal(actual, expected)


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two local devices for existing distributed topology coverage",
)
def test_two_shard_half_turn_regression():
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 2, 1),
        periodic_axes=(False, True, False),
        mesh_axis_names=(None, "theta", None),
    )
    domain = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec,
            side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
            side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
        ),
    )
    rule = PolarAxisRegularScalarRule3D(angle_axis_name="theta")
    fields = []
    for shard in range(2):
        field = jnp.full(domain.layout.cell_halo_shape, -999.0)
        owned = jnp.full(domain.owned_shape, 10.0 * shard)
        fields.append(field.at[domain.layout.owned_slices_cell].set(owned))
    filled = jax.pmap(lambda field: rule(field, domain), axis_name="theta")(
        jax.device_put_sharded(fields, jax.local_devices()[:2])
    )
    h = domain.layout.halo_width
    assert jnp.all(filled[0, :h, h : h + 2, h : h + 2] == 10.0)
    assert jnp.all(filled[1, :h, h : h + 2, h : h + 2] == 0.0)
