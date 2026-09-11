import jax
import jax.numpy as jnp
import pytest

from drbx.geometry import HaloLayout3D, LocalControlVolumeCellGeometry3D
from drbx.native.fci_operators import expand_local_control_volume_owner_field


def _cells(remote=False):
    layout = HaloLayout3D((2, 2, 2), 1)
    base = LocalControlVolumeCellGeometry3D.identity(layout, volume=jnp.ones((2, 2, 2)), centroid=jnp.zeros((2, 2, 2, 3)))
    if remote:
        source = (0, 0, 0)
        return LocalControlVolumeCellGeometry3D(
            **{**base.__dict__,
               "is_merged_source": base.is_merged_source.at[source].set(True),
               "is_active_owner": base.is_active_owner.at[source].set(False),
               "aggregate_volume": base.aggregate_volume.at[source].set(0.0),
               "owner_is_remote": jnp.zeros((2, 2, 2), dtype=bool).at[source].set(True),
               "remote_owner_halo_i": jnp.zeros((2, 2, 2), dtype=jnp.int32).at[source].set(3),
               "remote_owner_halo_j": jnp.zeros((2, 2, 2), dtype=jnp.int32).at[source].set(1),
               "remote_owner_halo_k": jnp.zeros((2, 2, 2), dtype=jnp.int32).at[source].set(1)})
    return base


def test_owner_halo_is_required_for_eager_remote_geometry():
    cells = _cells(True)
    with pytest.raises(ValueError, match="owner_values_halo"):
        expand_local_control_volume_owner_field(jnp.ones((2, 2, 2)), cells)


def test_owner_expansion_jit_path_has_no_tracer_exception_probe(monkeypatch):
    cells = _cells(False)
    calls = []
    error_type = jax.errors.TracerBoolConversionError
    original_init = error_type.__init__

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(error_type, "__init__", spy)

    @jax.jit
    def expand(values):
        return expand_local_control_volume_owner_field(values, cells)

    with jax.disable_jit(False):
        result = expand(jnp.ones((2, 2, 2)))
    assert jnp.all(result == 1.0)
    assert calls == []
