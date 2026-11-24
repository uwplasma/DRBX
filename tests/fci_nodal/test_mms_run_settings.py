from dataclasses import dataclass
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_nodal.case import case as driver, _mms_stages
from drbx.fci_nodal.drb import Switches, owner_operator
from drbx.fci_nodal.drb.wall_plan import build_owner_wall_plan
from drbx.fci_nodal.operators.parallel import owner_products, owner_runtime
from drbx.fci_nodal.options import RuntimeOptions


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

    def coupled_action(mesh, params, chunk, *, robust):
        assert robust
        face = jnp.ones((7, 6))
        records = owner_runtime.coupled_records(mesh, params, chunk)
        records(face, face, jnp.arange(7))

    monkeypatch.setattr(jax.lax, "scan", scan)
    monkeypatch.setattr(owner_operator, "coupled_action", coupled_action)
    owner_operator.owner_actions(resolution, switches=Switches(), face_record_chunk=override)
    assert widths == [width]


@dataclass(frozen=True)
class Device:
    platform: str
    id: int = 0


@pytest.mark.parametrize("devices, message", [
    ([Device("gpu")], "mesh construction reached"),
    ([Device("tpu")], "same CPU or GPU platform"),
    ([Device("cpu"), Device("gpu")], "same CPU or GPU platform"),
    ([Device("gpu"), Device("gpu")], "distinct device"),
])
def test_owner_products_platform_guard(tmp_path, monkeypatch, devices, message):
    (tmp_path / "metadata.json").write_text(json.dumps(dict(
        version=owner_products.VERSION, complete=True, owner_count=len(devices), N=16, symmetry="none",
        field_periods=1, params=[1., 1836.],
        face_quadrature_degree=owner_products.MOMENT_REDUCED_FACE_QUADRATURE_DEGREE,
    )))

    class MeshReached(Exception):
        pass

    def mesh(selected, axes):
        assert selected.tolist() == devices
        assert axes == ("records",)
        raise MeshReached("mesh construction reached")

    monkeypatch.setattr(owner_products, "Mesh", mesh)
    error = MeshReached if devices == [Device("gpu")] else ValueError
    with pytest.raises(error, match=message):
        owner_products.load_owner_parallel(tmp_path, RuntimeOptions(owner_count=len(devices)), devices=devices)


@pytest.mark.parametrize("devices, axes, message", [
    ([Device("gpu")], ("records",), "At least four eta planes"),
    ([Device("tpu")], ("records",), "requires the Q records mesh"),
    ([Device("cpu"), Device("gpu")], ("records",), "requires the Q records mesh"),
    ([Device("gpu")], ("wrong",), "requires the Q records mesh"),
])
def test_wall_mesh_platform_guard(devices, axes, message):
    mesh = SimpleNamespace(size=len(devices), axis_names=axes, devices=np.asarray(devices))
    with pytest.raises(ValueError, match=message):
        build_owner_wall_plan(mesh, None, None, None, None, eta_planes=1, nodes_per_plane=1,
                              wall_per_plane=1, deta=1., Om=None, Bdiv=None, Wm=None, gi=None)


@pytest.mark.parametrize("missing_backend", [False, True])
def test_gpu_run_refuses_missing_device(monkeypatch, missing_backend):
    def devices(platform):
        assert platform == "gpu"
        if missing_backend:
            raise RuntimeError("CUDA backend unavailable")
        return []

    monkeypatch.setattr(jax, "devices", devices)
    with pytest.raises(ValueError, match="No GPU is visible to JAX.*GPU build matching the installed JAX"):
        _mms_stages.run({"device": "gpu"}, None, {})


def test_gpu_selection_uses_first_visible_device(monkeypatch):
    devices = [Device("gpu", 3), Device("gpu", 1)]
    monkeypatch.setattr(jax, "devices", lambda _: devices)
    assert _mms_stages._first_gpu(jax) is devices[0]


@pytest.mark.parametrize("launch, message", [
    ("per_owner", "require local launch mode"),
    ("scheduler", "require local launch mode"),
    ("local", "require exactly one owner"),
])
def test_run_mms_refuses_unsupported_gpu_layout(tmp_path, monkeypatch, launch, message):
    initial = (dict(identity=dict(options=dict(owner_count=2))), {})
    monkeypatch.setattr(driver, "verify_chain", lambda *_: {"owner_inputs": initial})
    with pytest.raises(ValueError, match=message):
        driver.run_mms(tmp_path, dict(runtime=RuntimeOptions(), device="gpu", launch_mode=launch))
