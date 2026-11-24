"""Addressable-file and route checks without a distributed backend."""
import builtins
import io
import json
from pathlib import Path
from types import SimpleNamespace

import jax
from jax.sharding import Mesh
import numpy as np
import pytest
from scipy import sparse

from drbx.fci_nodal.operator_build.perp.owner_bundle import OwnerPerpendicularBundle, _leaf_entry
from drbx.fci_nodal.case._owner_inputs import OwnerParallelInputs
from drbx.fci_nodal.drb.afc import build_owner_limiter_graph
from drbx.fci_nodal.drb.rank_runtime import METRICS, SETUP, read_owned_p
from drbx.fci_nodal.drb.owner_route_store import (
    HIT_FIELDS, MATRICES, WALL_FIELDS, RouteStore, publish_routes, route_plans,
)
from drbx.fci_nodal.drb.wall_plan import build_owner_wall_plan
from drbx.fci_nodal.mms.cache import OwnerPrepared, _OWNER_SCHEMA, _save_owner_members, read_mms_rows
from drbx.fci_nodal.mms.evolved import ExactJet
from drbx.fci_nodal.operators.parallel.owner_products import LocalLayout
from drbx.fci_nodal._hashing import sha256_file as digest


@pytest.fixture
def opened(monkeypatch):
    paths = set()
    def guard(original):
        def tracked(path, *args, **kwargs):
            if isinstance(path,(str,Path)):
                paths.add(Path(path).resolve())
            return original(path,*args,**kwargs)
        return tracked
    monkeypatch.setattr(io,'open',guard(io.open))
    monkeypatch.setattr(builtins,'open',guard(builtins.open))
    return paths


@pytest.mark.parametrize('rank',[0,1])
def test_rank_reads_only_its_owned_sources(tmp_path, opened, rank):
    p = tmp_path/'p'
    p.mkdir()
    descriptors = {key:dict(shape=(8,1),dtype='<f8') for key in METRICS+SETUP[:2]}
    descriptors.update({key:dict(shape=(),dtype='<f8') for key in SETUP[2:]})
    q_root = tmp_path/'q/owners'
    m = tmp_path/'mms'
    m.mkdir()
    owners,mms_owner,q_owner,checksums = [],[],[],{}
    for i in range(2):
        owner = p/f'owner{i:04d}'
        owner.mkdir()
        files = {}
        for key,desc in descriptors.items():
            shape = (4,1) if desc['shape'] else ()
            path = owner/f'{key}.npy'
            value = np.full(shape,i+1.,dtype=desc['dtype'])
            np.save(path,value)
            files[key] = _leaf_entry(p,path,shape,value.dtype)
        a,b = i*4,(i+1)*4
        owners.append(dict(begin=a,end=b,files=files))
        construction = f'owners/owner_{i:05d}'
        q_owner.append(dict(directory=construction))
        for plane in range(a,b):
            folder = q_root.parent/construction/f'plane_{plane:05d}'
            folder.mkdir(parents=True)
            for name in ('wall_credited_parallel_mass.npy','MB_V.npy'):
                path = folder/name
                np.save(path,np.array([i+1.]))
                checksums[str(path)] = digest(path)
        state = np.full((4,7),i+1.)
        members = dict(wall_ginv=np.ones((4,1,3)))
        for name in ('nodal','pwall','A'):
            jet = ExactJet(state,np.zeros(state.shape+(3,)),np.zeros(state.shape+(3,3)),np.zeros_like(state))
            members.update({name+'__'+field:v for field,v in jet._asdict().items()})
        md = _save_owner_members(m,i,dict(manufactured_field_data_floating_wall=members,hit_layout=dict(ids=np.arange(a,b))))
        mms_owner.append(dict(begin=a,end=b,files=md))
    manifest = dict(schema='drbx.perpendicular.owner_bundle.v1',N=16,n_eta=8,P=1,owner_count=2,edges=[0,4,8],
                    owners=owners,leaves={k:descriptors[k] for k in METRICS},
                    sbp_leaves={k:dict(descriptors[k],halo=0) for k in SETUP})
    (p/'manifest.json').write_text(json.dumps(manifest))
    (m/'manifest.json').write_text(json.dumps(dict(schema=_OWNER_SCHEMA,eta_planes=8,owner_count=2,
                    edges=[0,4,8],owners=mms_owner)))
    opened.clear()
    bundle = OwnerPerpendicularBundle(p)
    metrics,setup = read_owned_p(bundle,rank)
    ids = np.arange(rank*4,(rank+1)*4)
    mass = OwnerParallelInputs(q_root,{},dict(owners=q_owner),bundle,checksums=checksums).mass(rank,node_ids=ids)
    manufactured_field_data = read_mms_rows(OwnerPrepared(m),rank,'floating_wall',ids,jax.devices('cpu')[0])
    assert (metrics['B']==rank+1.).all() and (setup['aee_h']==rank+1.).all()
    assert all((leaf==rank+1.).all() for leaf in mass)
    assert (np.asarray(manufactured_field_data['nodal'].value)==rank+1.).all()
    selected = {path for path in opened if path.is_relative_to(tmp_path) and path.name!='manifest.json'}
    wanted = set((p/f'owner{rank:04d}').glob('*.npy'))
    wanted.update((m/f'owner{rank:04d}').glob('*.npz'))
    wanted.update((q_root/f'owner_{rank:05d}').rglob('*.npy'))
    assert selected==wanted


def test_streamed_routes_match_existing_packing(tmp_path, opened):
    wall,q_root = tmp_path/'wall',tmp_path/'q'
    q_root.mkdir()
    ids = {i:np.arange(4*i,4*(i+1),dtype=np.int32) for i in range(2)}
    layout = LocalLayout(8,4,2,ids)
    identity = sparse.eye(8,format='csr')
    K = identity.tolil()
    K[3,4] = K[4,3] = 1.
    K = K.tocsr()
    rows,edges,diag,files = {},{},{},{}
    for i,j in ids.items():
        folder = wall/f'owner{i:04d}'
        folder.mkdir(parents=True)
        path = q_root/f'nodes_{i}.npy'
        np.save(path,j)
        files[str(i)] = dict(path=path.name,sha256=digest(path))
        rows[i] = {name:(K if name=='K' else identity)[j] for name in MATRICES}
        data = {}
        for name,A in rows[i].items():
            data.update({name+'_data':A.data,name+'_indices':A.indices,
                         name+'_indptr':A.indptr,name+'_shape':A.shape})
        for name in ('H',*WALL_FIELDS,*HIT_FIELDS):
            a = np.ones(4)
            rows[i][name] = data[name] = a
        np.savez(folder/'wall_plan.npz',**data)
        take = np.arange(1-i,dtype=np.int32)
        edges[i] = (take,take+3,take+4,np.ones(1-i))
        diag[i] = np.array([2. if k in (3,4) else 1. for k in j])
        np.savez(folder/'limiter_graph.npz',**dict(zip(('id','i','j','v'),edges[i])),diag=diag[i])
    meta = dict(owner_count=2,layouts=dict(nodes=dict(files=files,n=8,cap=4)))
    (q_root/'metadata.json').write_text(json.dumps(meta))
    q = SimpleNamespace(meta=meta,edges=(0,4,8),width=1,K=lambda p,*,node_ids:K[node_ids])
    root = publish_routes(tmp_path/'routes',wall,q_root,q,wall_per_plane=1)
    store = RouteStore.open(root,wall)
    for rank in ids:
        opened.clear()
        store.wall_rows(rank)
        for name in (*MATRICES,'afc_K','afc'):
            store.read(rank,name)
        reference = set((root/f'owner{rank:04d}').glob('*.npz'))
        reference.add(wall/f'owner{rank:04d}'/'wall_plan.npz')
        assert {p for p in opened if p.is_relative_to(tmp_path)}==reference
    devices = jax.devices('cpu')
    if len(devices)<2:
        pytest.skip('Requires two local CPU devices for route-array comparison')
    mesh = Mesh(np.asarray(devices[:2]),('records',))
    constants = dict(Om=1.,gi=1.,Bdiv=np.ones((5,5)),Wm=np.ones(5))
    w,grid,graph = route_plans(store,mesh,layout,layout,layout,edges=q.edges,deta=1.,
                              q_products=q_root,constants=constants)
    ref,ref_grid = build_owner_wall_plan(mesh,layout,layout,layout,rows,
        eta_planes=8,nodes_per_plane=1,wall_per_plane=1,deta=1.,**constants)
    ref_graph = build_owner_limiter_graph(mesh,layout,{p:K[j] for p,j in ids.items()},edges,diag)
    assert grid==ref_grid
    for actual,expected in zip(jax.tree.leaves((w,graph)),jax.tree.leaves((ref,ref_graph)),strict=True):
        assert actual.dtype==expected.dtype
        assert actual.shape==expected.shape
        assert np.asarray(actual).tobytes()==np.asarray(expected).tobytes()
