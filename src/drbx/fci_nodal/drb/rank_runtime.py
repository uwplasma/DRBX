"""Rank-local P input loading, halo setup, and replicated metadata agreement."""
import hashlib

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import numpy as np

from ..operators.nodal.sbp_laplacian_plan import LaplacianMetric
from ..operators.perp import sbp_owner as sbp
from ..operators.nodal.nodal_plan import NodalMetric
from .owner_route_store import addressable


METRICS = ('points','h','jac','B','K','A','ginv_u')
SETUP = ('aee_h','ge_diag','tau','tau_w','rings_per_block')


def read_owned_p(bundle, owner):
    """Load only this owner's metric and setup files, dropping saved legacy halos."""
    bundle.verify(owner,leaves=METRICS+SETUP)
    metrics = bundle.read(owner,leaves=METRICS)
    specs = bundle.manifest['sbp_leaves']
    setup = bundle._load(owner,SETUP,specs)
    halo = specs['aee_h']['halo']
    if halo not in (0,sbp.PLAN_HALO) or specs['ge_diag']['halo']!=0:
        raise ValueError('Unexpected layout of owner SBP setup leaves')
    if halo:
        setup['aee_h'] = setup['aee_h'][halo:-halo]
    return metrics,setup


def rank_sbp(mesh, slabs, owner, local_data):
    """Exchange two metric planes each way; other ranks' files stay closed."""
    metrics,setup = local_data
    count = slabs.counts[owner]
    local = addressable(mesh)
    if len(local)!=1 or local[0][0]!=owner:
        raise ValueError('Expected one addressable CPU device at the owner rank')
    spec = P(sbp.AXIS)
    sharding = NamedSharding(mesh,spec)
    device = local[0][1]
    leaves = (metrics['A'],metrics['jac'],metrics['ginv_u'],setup['aee_h'])
    def put(value):
        pad = [(0,slabs.capacity-count)]+[(0,0)]*(value.ndim-1)
        data = jax.device_put(np.pad(value,pad,mode='edge')[None],device)
        return jax.make_array_from_single_device_arrays((mesh.size,*data.shape[1:]),sharding,[data])
    def exchange(*xs):
        return tuple(sbp.exchange_owner_halo(x[0],sbp.PLAN_HALO,slabs)[None] for x in xs)
    mapped = jax.jit(jax.shard_map(exchange,mesh=mesh,in_specs=(spec,)*4,
                                  out_specs=(spec,)*4,check_vma=False))
    windows = tuple(np.asarray(v.addressable_data(0))[0,:count+2*sbp.PLAN_HALO]
                    for v in mapped(*map(put,leaves)))
    nodal = NodalMetric(*(metrics[name] for name in ('h','jac','B','K')))
    return sbp.OwnerSBPSlice(nodal,LaplacianMetric(*windows[:3]),windows[3],
        float(setup['tau']),float(setup['tau_w']),setup['ge_diag'],int(setup['rings_per_block']))


def verify_replicas(mesh, tree, *, metadata):
    """Compare one 32-byte digest per rank, with scalar success on each rank."""
    h = hashlib.sha256(repr(metadata).encode())
    for value in jax.tree.leaves(tree):
        if isinstance(value,jax.Array) and value.sharding.is_fully_replicated:
            a = np.asarray(value.addressable_data(0))
            h.update(repr((a.shape,a.dtype.str)).encode())
            h.update(a.tobytes())
    _,device = addressable(mesh)[0]
    local = jax.device_put(np.frombuffer(h.digest(),dtype=np.uint8)[None],device)
    spec = P(sbp.AXIS)
    placed = jax.make_array_from_single_device_arrays(
        (mesh.size,32),NamedSharding(mesh,spec),[local])
    def same(x):
        return jnp.all(jax.lax.pmin(x,sbp.AXIS)==jax.lax.pmax(x,sbp.AXIS))
    check = jax.jit(jax.shard_map(same,mesh=mesh,in_specs=spec,out_specs=P(),check_vma=False))
    if not bool(check(placed).addressable_data(0)):
        raise ValueError('Replicated owner templates, scalars or settings differ across ranks')
