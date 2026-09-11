"""Bounded eta-shard regression for the factorized coarse action."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_polarization_coarse import build_coarse_data


class _EtaDomain:
    mesh_axis_names = (None, None, "z")


def test_eta_shards_match_full_action():
    if jax.local_device_count() < 2:
        pytest.skip("requires two virtual CPU devices")
    nx, ny, nz = 10, 3, 8
    active = np.ones((nx, ny, nz), dtype=bool)
    active[2, 1, 3] = False
    weights = .5 + np.arange(nx * ny * nz, dtype=float).reshape(active.shape) / 100
    data = build_coarse_data(lambda x: x, active, weights)
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.normal(size=active.shape))
    full = data.coarse(x, jnp.asarray(active), jnp.asarray(weights))
    ls = nz // 2
    xs = x.reshape(nx, ny, 2, ls).transpose(2, 0, 1, 3)
    aa = jnp.asarray(active).reshape(nx, ny, 2, ls).transpose(2, 0, 1, 3)
    ww = jnp.asarray(weights).reshape(nx, ny, 2, ls).transpose(2, 0, 1, 3)
    def shard(v, a, w):
        return data.coarse(v, a, w, _EtaDomain())
    got = jax.pmap(shard, axis_name="z")(xs, aa, ww)
    got = np.asarray(got).transpose(1, 2, 0, 3).reshape(nx, ny, nz)
    assert np.allclose(got, np.asarray(full), rtol=2e-8, atol=2e-9)
