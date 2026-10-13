"""eta-plane sharding of the ``core_schur`` apply over the devices of one node (``shard_map``; the apply is block-Jacobi over planes,
so no communication is needed)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

from drbx.native.fci_perpendicular_plane_preconditioner import CoreSchurPreconditioner, apply_core_schur_preconditioner

import pgb_precond as PC
import pgb_timing as T


def _shard_map(f, mesh, in_specs, out_specs):
    sm = getattr(jax, "shard_map", None)
    if sm is None:                                                               # older jax
        from jax.experimental.shard_map import shard_map as sm
    for kw in ({"check_vma": False}, {"check_rep": False}, {}):
        try:
            return sm(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kw)
        except TypeError:
            continue
    raise RuntimeError("shard_map signature not recognised")


def n_shards(n_planes: int, n_devices: int) -> int:
    """Largest divisor of ``n_planes`` not above ``n_devices``."""
    return max(d for d in range(1, min(n_planes, n_devices) + 1) if n_planes % d == 0)


def sharded_apply(prec: CoreSchurPreconditioner, n_dev: int, devices=None):
    """``(fn, prec_sharded, shard_vec)``: ``fn(prec_sharded, r_sharded)`` applies ``M^-1`` with the planes split over ``n_dev`` devices."""
    devices = list(devices or jax.devices())[:n_dev]
    mesh = Mesh(np.array(devices), ("e",))
    leaves, treedef = jax.tree_util.tree_flatten(prec)
    specs = [PS(None, "e"), PS(None, "e"), PS("e"), PS("e")]
    spec_tree = jax.tree_util.tree_unflatten(treedef, specs)
    prec_s = jax.tree_util.tree_unflatten(
        treedef, [jax.device_put(x, NamedSharding(mesh, s)) for x, s in zip(leaves, specs)])
    fn = _shard_map(apply_core_schur_preconditioner, mesh, (spec_tree, PS("e")), PS("e"))
    vec_sharding = NamedSharding(mesh, PS("e"))
    return fn, prec_s, (lambda r: jax.device_put(r, vec_sharding))


def run_sharding(prec: CoreSchurPreconditioner, rhs: np.ndarray, *, reps: int, devices=None) -> dict:
    """Single-device versus sharded apply: times (median / min), speed-up, relative difference of the results."""
    devices = list(devices or jax.devices())
    E = prec.z.shape[0]
    nd = n_shards(E, len(devices))
    r1 = jnp.asarray(rhs)
    s1, y1 = T.time_jit(apply_core_schur_preconditioner, prec, r1, reps=reps)
    fn, prec_s, put = sharded_apply(prec, nd, devices)
    rs = put(jnp.asarray(rhs))
    s2, y2 = T.time_jit(fn, prec_s, rs, reps=reps)
    y1n, y2n = np.asarray(y1), np.asarray(y2)
    return dict(n_devices_visible=len(devices), n_devices_used=nd, planes=E, planes_per_device=E // nd,
                single_device=s1, sharded=s2, speedup=s1["median_s"] / s2["median_s"],
                rel_diff_vs_single=float(np.linalg.norm(y2n - y1n) / np.linalg.norm(y1n)),
                storage_bytes_total=int(prec.nbytes), storage_bytes_per_device=int(prec.nbytes // nd),
                device_kind=getattr(devices[0], "device_kind", None))
