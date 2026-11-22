"""Data-free checks for owner layout, setup, and limiter transport."""

import json

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest
from scipy import sparse

from drbx.fci_nodal.operator_build.perp.owner_bundle import (
    OwnerPerpendicularBundle, _publish_owner_bundle, _record_leaf,
)
from drbx.fci_nodal.drb import afc
from drbx.fci_nodal.mms.cache import OwnerPrepared, _OWNER_SCHEMA, _manufactured_field_members, _save_owner_members
from drbx.fci_nodal.mms.evolved import StageParameters, load_owner_stage, manufactured_inputs
from drbx.fci_nodal.mms.evolved import ExactJet
from drbx.fci_nodal.operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout
from drbx.fci_nodal.operators.nodal.sbp_laplacian_plan import LaplacianMetric, build_laplacian_plan
from drbx.fci_nodal.operators.nodal.sbp_laplacian_assembly import global_assembly_data
from drbx.fci_nodal.operators.perp.sbp_owner import SlabLayout, exchange_owner_halo
from drbx.fci_nodal.operators.parallel.owner_products import LocalLayout
from drbx.fci_nodal.operators.parallel.owner_routes import owner_array


def cpu_mesh(parts):
    devices = jax.devices('cpu')
    if len(devices) < parts:
        pytest.skip(f'Requires {parts} CPU devices on the records mesh')
    return Mesh(np.asarray(devices[:parts]), ('records',))


@pytest.mark.parametrize('planes,parts,edges', [(8,1,(0,8)), (8,2,(0,4,8)), (16,3,(0,6,11,16))])
def test_ceiling_slabs(planes, parts, edges):
    layout = SlabLayout(planes,parts)
    assert layout.edges == edges
    assert layout.capacity == max(np.diff(edges))
    np.testing.assert_array_equal(layout.plane_ids(parts-1,2),
                                  np.arange(edges[-2]-2,edges[-1]+2)%planes)


def test_short_owner_slab_is_rejected():
    with pytest.raises(ValueError, match='four'):
        SlabLayout(10,3)


@pytest.mark.parametrize('parts,planes', [(1,8),(2,8),(3,16)])
def test_periodic_halo_wrap(parts, planes):
    mesh = cpu_mesh(parts)
    layout = SlabLayout(planes,parts)
    pieces = {i:np.pad(layout.plane_ids(i),(0,layout.capacity-count),constant_values=-99)
              for i,count in enumerate(layout.counts)}
    placed = owner_array(mesh,pieces,packed=True)
    for width in (2,3,4):
        exchange = jax.shard_map(lambda a:exchange_owner_halo(a[0],width,layout)[None],
                                 mesh=mesh,in_specs=P('records'),out_specs=P('records'),check_vma=False)
        result = np.asarray(exchange(placed))
        for i,count in enumerate(layout.counts):
            np.testing.assert_array_equal(result[i,:count+2*width],layout.plane_ids(i,width))


def test_owner_perpendicular_bundle_round_trip(tmp_path):
    layout = build_zernike_core_ring_layout(16,n_eta=8)
    E, width = layout.n_eta, layout.P
    planes = np.arange(E)[:,None]
    jac = np.broadcast_to(1.+planes/16,(E,width)).copy()
    A = np.eye(3)*jac[...,None,None]
    ginv = np.broadcast_to(np.array([1.,0.,0.]),(E,width,3)).copy()
    identity = dict(n=16,n_eta=E,P=width,created=dict(inputs=dict(symmetry='none',field_periods=1)),
                    preconditioner=dict(info=dict(rings_per_block=2)))
    leaves = dict(A=A,jac=jac,ginv_u=ginv,h=ginv,B=jac,K=ginv)
    root = tmp_path/'owners'
    edges = (0,4,8)
    dirs = [root/f'owner{i:04d}' for i in range(2)]
    owners = [dict(begin=a,end=b,files={}) for a,b in zip(edges[:-1],edges[1:])]
    for folder in dirs:
        folder.mkdir(parents=True)
    schema = {}
    for name,data in leaves.items():
        paths = [folder/f'{name}.npy' for folder in dirs]
        for path,(a,b) in zip(paths,zip(edges[:-1],edges[1:]),strict=True):
            np.save(path,data[a:b],allow_pickle=False)
        schema[name] = _record_leaf(root,owners,paths,name,data.shape,data.dtype)
    identity_path = root/'identity.json'
    identity_path.write_text(json.dumps(identity))
    _publish_owner_bundle(root,identity_path,identity,edges,dirs,owners,schema)
    assert not list(root.rglob('*.npz'))
    reader = OwnerPerpendicularBundle(root)
    assert reader.manifest['sbp_leaves']['aee_h']['halo'] == 0
    full = build_laplacian_plan(layout,LaplacianMetric(A,jac,ginv))
    diagonal = global_assembly_data(full)['Ge_diag']
    for owner,(begin,end) in enumerate(zip(reader.edges[:-1],reader.edges[1:])):
        reader.verify(owner)
        for name, rows in reader.read(owner).items():
            np.testing.assert_array_equal(rows,leaves[name][begin:end])
        halo = np.arange(begin-2,end+2)%E
        sbp = reader.read_owner_sbp(owner,edges=reader.edges)
        np.testing.assert_array_equal(sbp.laplacian.A_log,A[halo])
        np.testing.assert_array_equal(sbp.aee_h,full.aee_h[halo])
        np.testing.assert_array_equal(sbp.ge_diag,diagonal[begin:end])
        assert (sbp.tau,sbp.tau_w,sbp.rings_per_block)==(full.tau,full.tau_w,2)


def test_cross_owner_limiter_correction():
    mesh = cpu_mesh(2)
    ids = {0:np.array([0,1],np.int32),1:np.array([2,3],np.int32)}
    nodes = LocalLayout(4,2,2,ids)
    K = sparse.csr_matrix([[1.,-2.,1.,0.],[-2.,2.,0.,0.],[1.,0.,1.,-2.],[0.,0.,-2.,2.]])
    empty = np.empty(0,np.int32)
    edges = {0:(np.array([0]),np.array([0]),np.array([2]),np.array([1.])),
             1:(empty,empty,empty,np.empty(0))}
    u = np.array([1.,1.,3.,3.])
    mass = np.ones(4)
    diagonal = np.full(4,2.)
    sharded = lambda a:nodes.put({p:a[idx] for p,idx in ids.items()},mesh)
    owner = afc.build_owner_limiter_graph(mesh,nodes,{p:K[idx] for p,idx in ids.items()},edges,
                                  {p:diagonal[idx] for p,idx in ids.items()})
    single_mesh = cpu_mesh(1)
    single = LocalLayout(4, 4, 1, {0: np.arange(4, dtype=np.int32)})
    single_graph = afc.build_owner_limiter_graph(single_mesh, single, {0: K}, {0: edges[0]},
                                         {0: diagonal})
    result = afc.owner_action(owner, sharded(u), 1., sharded(mass))
    expected = afc.owner_action(single_graph, single.put({0: u}, single_mesh), 1.,
                                single.put({0: mass}, single_mesh))
    assert int(result[2]['wrong_sign_pairs'])==1
    np.testing.assert_array_equal(result[1].ravel(),[2.,0.,-2.,0.])
    for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(expected),strict=True):
        np.testing.assert_array_equal(a,b)


def test_owner_mms_cache_and_stage(tmp_path):
    def jet(value):
        value = np.asarray(value)
        return ExactJet(value,np.zeros(value.shape+(3,)),np.zeros(value.shape+(3,3)),np.zeros_like(value))

    state = np.broadcast_to([1.,1.,1.,0.,0.,0.,0.],(4,7)).copy()
    normal = np.broadcast_to([1.,0.,0.],(4,1,3))
    data = dict(nodal=jet(state),pwall=jet(state),A=jet(np.zeros(4)),g=state[:,:6],
                h=normal[:,0],jac=np.ones(4),K=np.zeros((4,3)),B=np.ones(4),
                metric=np.broadcast_to(np.eye(3),(4,3,3)),divA=np.zeros((4,3)),
                ds=np.zeros((4,7)),divb=np.zeros(4),lap=np.zeros((4,6)),wall_ginv=normal)
    files = _save_owner_members(tmp_path,0,dict(manufactured_field_data_floating_wall=_manufactured_field_members(data)))
    manifest = dict(schema=_OWNER_SCHEMA,owner_count=1,eta_planes=4,edges=[0,4],wall_per_plane=1,
                    owners=[dict(begin=0,end=4,files=files)])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    params = StageParameters(.00045,jnp.ones(6),jnp.ones(6),1.,1836.,.0027,19.29)
    expected = manufactured_inputs(jnp.array(.001),params,data,wall_shape=(4,1),psi_mode='phi_plus_tau_pi')
    result = load_owner_stage(OwnerPrepared(tmp_path),0,jnp.array(.001),params,
                              wall_law='floating_wall',psi_mode='phi_plus_tau_pi')
    for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(expected),strict=True):
        np.testing.assert_array_equal(a,b)
