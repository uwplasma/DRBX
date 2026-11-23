from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from drbx.fci_nodal.drb import Switches, owner_operator
from drbx.fci_nodal.operators.parallel import owner_runtime


@pytest.mark.parametrize("override, width", [(None, 4), (6, 6)])
def test_face_record_chunk_reaches_scan(monkeypatch, override, width):
    q = SimpleNamespace(mesh=object(), params=(1., 1836.), rhs_chunk=4, arrays=lambda: ())
    ctx = SimpleNamespace(nodal=SimpleNamespace(structure=SimpleNamespace(P=1)))
    resolution = SimpleNamespace(
        plan=SimpleNamespace(ctx=ctx, opts=None, wall_ops=None), q=q, slabs=SimpleNamespace(capacity=1),
        wall_grid=SimpleNamespace(wall_per_plane=1), node_active=None, hit_active=None,
    )
    monkeypatch.setattr(owner_operator, "perpendicular_partition_specs", lambda _: ())
    monkeypatch.setattr(jax, "shard_map", lambda fn, **_: fn)
    monkeypatch.setattr(jax, "jit", lambda fn: fn)
    monkeypatch.setattr(owner_runtime.rr, "reduce_status", lambda status: status)
    widths = []

    def scan(body, status, records):
        faces, jumps, ids = records
        assert faces.shape == jumps.shape == (2, width, 6)
        assert ids.shape == (2, width)
        widths.append(ids.shape[1])
        return status, faces

    def coupled_action(mesh, params, chunk):
        face = jnp.ones((7, 6))
        records = owner_runtime.coupled_records(mesh, params, chunk)
        records(face, face, jnp.arange(7))

    monkeypatch.setattr(jax.lax, "scan", scan)
    monkeypatch.setattr(owner_operator, "coupled_action", coupled_action)
    owner_operator.owner_actions(resolution, switches=Switches(), face_record_chunk=override)
    assert widths == [width]
