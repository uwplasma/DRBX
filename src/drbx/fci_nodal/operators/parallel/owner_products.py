"""Read the addressable owners of a bounded indexed parallel assembly version-3 application export."""
import json
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy import sparse

from ..._hashing import sha256_file as digest
from . import MOMENT_REDUCED_FACE_QUADRATURE_DEGREE
from .owner_runtime import OwnedRuntime
from .owner_sparse import HaloCSR, OwnedSolve


VERSION = 3


class LocalLayout:
    """Accept already owner-ordered rows, with or without neutral padding."""
    def __init__(self,n,cap,parts,ids):
        self.n,self.cap,self.parts,self.ids_by_owner=n,cap,parts,ids

    def put(self,x,mesh,neutral=0):
        if not isinstance(x,dict):
            raise ValueError('Pass a sharded JAX array or a mapping from owner to local rows')
        arrays=[]
        for owner,device in enumerate(mesh.devices.flat):
            if device.process_index!=jax.process_index():
                continue
            v=np.asarray(x[owner])
            tail=v.shape[1:]
            if len(v)!=self.cap:
                count=int(np.sum(self.ids_by_owner[owner]>=0))
                if len(v)!=count:
                    raise ValueError('Owner input row count differs from the product')
                padded=np.empty((self.cap,)+tail,v.dtype);padded[...] = neutral;padded[:count]=v;v=padded
            arrays.append(jax.device_put(v,device))
        return jax.make_array_from_single_device_arrays((self.parts*self.cap,)+tail,
                   NamedSharding(mesh,P('records')),arrays)


def load_owner_parallel(path,options,*,devices=None):
    path=Path(path).resolve()
    md=json.loads((path/'metadata.json').read_text())
    if md['version']!=VERSION or not md['complete']:
        raise ValueError('Incomplete or incompatible owner products')
    if md['owner_count']!=options.owner_count:
        raise ValueError('Owner count differs from the requested owner_count')
    expected = dict(resolution=md['N'],symmetry=md['symmetry'],field_periods=md['field_periods'])
    if any(getattr(options,k) not in (None,v) for k,v in expected.items()):
        raise ValueError('Owner product layout differs from runtime options')
    if tuple(md['params'])!=(options.tau,options.mu) or md['face_quadrature_degree']!=MOMENT_REDUCED_FACE_QUADRATURE_DEGREE:
        raise ValueError('Owner product parameters differ from supported operator settings')
    devices=jax.devices('cpu') if devices is None else devices
    options.check_cpu_device_capacity(len(devices))
    selected=tuple(devices[:md['owner_count']])
    platforms = {d.platform for d in selected}
    if len(set(selected))!=len(selected) or platforms not in ({'cpu'}, {'gpu'}):
        raise ValueError('One distinct device from the same CPU or GPU platform is required per owner')
    mesh=Mesh(np.asarray(selected),('records',))
    sh=NamedSharding(mesh,P('records'))
    local=[(p,d) for p,d in enumerate(mesh.devices.flat) if d.process_index==jax.process_index()]
    if not local:
        raise ValueError('The owner mesh has no addressable device on this process')

    def read(file):
        dest=(path/file['path']).resolve()
        if not dest.is_relative_to(path) or digest(dest)!=file['sha256']:
            raise ValueError(f'Changed or out-of-root owner product {dest}')
        return np.load(dest,mmap_mode='r',allow_pickle=False)

    def arr(desc):
        if desc['kind']=='scalar':
            return jax.make_array_from_process_local_data(NamedSharding(mesh,P()),
                       np.asarray(desc['value'],dtype=desc['dtype']))
        d=md['arrays'][desc['name']]
        shape=tuple(d['shape'])
        expected=(shape[0]//mesh.size,*shape[1:])
        values=[]
        for p,device in local:
            data=read(d['files'][str(p)])
            if data.shape!=expected or data.dtype!=np.dtype(d['dtype']):
                raise ValueError(f'Owner array schema differs for {desc["name"]}')
            values.append(jax.device_put(data,device))
        return jax.make_array_from_single_device_arrays(shape,sh,values)

    def from_parts(parts):
        return [jax.make_array_from_single_device_arrays((mesh.size,)+values[0].shape[1:],sh,values)
                for values in parts]

    def devices_for(arrays):
        return [{v.device:v.data for v in a.addressable_shards} for a in arrays]

    def transpose(data,index,ptr,shape,nnz):
        A=sparse.csr_matrix((data,index,ptr),shape=shape)
        T=A.T.tocsr();T.sort_indices()
        td=np.zeros(nnz);ti=np.zeros(nnz,np.int32)
        td[:T.nnz]=T.data;ti[:T.nnz]=T.indices
        return td,ti,T.indptr.astype(np.int32)

    def wall_transpose(R):
        if R.delta:
            raise ValueError('Wall transpose requires plane-local observations')
        bydev=devices_for(R.arrays)
        parts=[[] for _ in range(10)]
        nnz=R.arrays[0].shape[1]
        for _,device in local:
            data,index,ptr=[np.asarray(v[device])[0] for v in bydev[:3]]
            order=np.asarray(bydev[8][device])[0]
            td,ti,tp=transpose(data,order[index],ptr,(R.rows,R.cols),nnz)
            arrays=[td,ti,tp,np.zeros(1),np.zeros(1,np.int32),np.zeros(1,np.int32),
                    np.zeros((0,0),np.int32),np.zeros((0,0),bool),
                    np.arange(R.rows,dtype=np.int32),np.arange(R.rows,dtype=np.int32)]
            for values,value in zip(parts,arrays):
                values.append(jax.device_put(value[None],device))
        return HaloCSR(tuple(from_parts(parts)),mesh,R.cols,R.rows,(),0)

    def decode(z):
        kind=z['kind']
        if kind in ('array','scalar'):
            return arr(z)
        if kind=='dict':
            result={}
            for k,v in z['items'].items():
                if v['kind']!='wall_transpose':
                    result[k]=decode(v)
            for k,v in z['items'].items():
                if v['kind']=='wall_transpose':
                    result[k]=wall_transpose(result[v['source']])
            return result
        if kind=='solve':
            return OwnedSolve(tuple(arr(v) for v in z['arrays']),mesh,z['n'])
        if kind=='csr':
            aa=[arr(v) for v in z['arrays']]
            ext=z['cols']+len(z['delta'])*z['width']
            nnz=aa[0].shape[1]
            bydev=devices_for(aa[:3])
            back=[[],[],[]]
            for _,device in local:
                if z['transpose']:
                    data,index,ptr=[np.asarray(v[device])[0] for v in bydev]
                    values=transpose(data,index,ptr,(z['rows'],ext),nnz)
                else:
                    values=(np.zeros(1),np.zeros(1,np.int32),np.zeros(1,np.int32))
                for dest,value in zip(back,values):
                    dest.append(jax.device_put(value[None],device))
            return HaloCSR(tuple(aa[:3]+from_parts(back)+aa[3:]),mesh,z['rows'],z['cols'],
                           tuple(z['delta']),z['width'])
        raise ValueError(f'Unknown owner product kind {kind}')

    rt=OwnedRuntime()
    rt.mesh=mesh
    rt.params=tuple(md['params'])
    rt.rhs_chunk=md['rhs_chunk']
    rt.cache_chunk=md['cache_chunk']
    rt.structure=md['structure']
    rt.__dict__.update(decode(md['tree']))
    for key,desc in md['layouts'].items():
        ids={p:np.asarray(read(desc['files'][str(p)])) for p,_ in local}
        for rows in ids.values():
            if rows.shape!=(desc['cap'],) or rows.dtype!=np.int32:
                raise ValueError(f'Invalid {key} owner layout')
        setattr(rt,key,LocalLayout(desc['n'],desc['cap'],mesh.size,ids))
    rt.configure()
    jax.block_until_ready(rt.arrays())
    return rt
