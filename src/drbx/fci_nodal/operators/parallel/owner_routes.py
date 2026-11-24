"""Arrange existing HaloCSR requests from addressable owner-row inputs."""
from dataclasses import dataclass
from functools import lru_cache

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from .owner_sparse import HaloCSR, _expand


def owner_array(mesh, arrays, *, packed=False):
    """Place equally shaped local pieces without constructing a host gather."""
    devices=tuple(mesh.devices.flat)
    values=[jax.device_put(np.asarray(arrays[p])[None] if packed else arrays[p],d)
            for p,d in enumerate(devices)]
    shape=(len(devices)*values[0].shape[0],*values[0].shape[1:])
    return jax.make_array_from_single_device_arrays(shape,NamedSharding(mesh,P('records')),values)


def locate(layout, ids):
    """Return source owner and local slot for each canonical ID."""
    ids=np.asarray(ids)
    owner=np.full(len(ids),-1,np.int32);slot=np.empty(len(ids),np.int32)
    for p,keys in layout.ids_by_owner.items():
        keys=keys[keys>=0]
        if len(keys) and np.any(keys[1:]<=keys[:-1]):
            raise ValueError('Canonical owner IDs must be strictly ascending')
        pos=np.searchsorted(keys,ids)
        at=np.flatnonzero(pos<len(keys))
        at=at[keys[pos[at]]==ids[at]]
        if np.any(owner[at]>=0):
            raise ValueError('Canonical IDs overlap owners')
        owner[at]=p;slot[at]=pos[at]
    if np.any(owner<0):
        raise ValueError('A sparse request is missing its source owner')
    return owner,slot


def request_layout(mesh, source, requests):
    """Resolve indexed source requests, preserving an ordered list per receiver."""
    edges={};deltas=set();width=0;placements={}
    for p,ids in requests.items():
        owners,slots=locate(source,ids)
        placements[p]=(owners,slots)
        for peer in np.unique(owners[owners!=p]):
            take=owners==peer
            _,idx=np.unique(ids[take],return_index=True)
            subset=slots[take][idx]
            edges[p,int(peer)]=(ids[take][idx],subset)
            deltas.add((p-int(peer))%mesh.size)
            width=max(width,len(subset))
    delta=tuple(sorted(deltas))
    send={p:np.zeros((len(delta),width),np.int32) for p in requests}
    valid={p:np.zeros((len(delta),width),bool) for p in requests}
    slots={}
    for p,ids in requests.items():
        owners,where=placements[p]
        result=where.copy()
        for k,d in enumerate(delta):
            peer=(p-d)%mesh.size
            item=edges.get((p,peer))
            if item is None:
                continue
            keys,indices=item
            send[peer][k,:len(indices)]=indices
            valid[peer][k,:len(indices)]=True
            take=owners==peer
            result[take]=source.cap+k*width+np.searchsorted(keys,ids[take])
        slots[p]=result
    return slots,send,valid,delta,width


def csr_piece(p,A,source,target,requests,slots,send,valid,delta,width,nnz):
    """Pack one sorted CSR row slab into the shared route capacities."""
    ext=source.cap+len(delta)*width
    if A.shape!=(int(np.sum(target.ids_by_owner[p]>=0)),source.n):
        raise ValueError('Owner CSR layout does not match canonical rows')
    keys=np.arange(ext,dtype=np.int64)+source.n
    padded=source.ids_by_owner[p]
    keys[:source.cap]=np.where(padded>=0,padded,keys[:source.cap])
    keys[slots[p]]=requests[p]
    order=np.argsort(keys,kind='stable').astype(np.int32)
    inverse=np.empty_like(order);inverse[order]=np.arange(ext,dtype=np.int32)
    data=np.zeros(nnz);col=np.zeros(nnz,np.int32)
    data[:A.nnz]=A.data;col[:A.nnz]=np.searchsorted(keys[order],A.indices)
    ptr=np.full(target.cap+1,A.nnz,np.int32);ptr[:len(A.indptr)]=A.indptr
    return (data,col,ptr,np.zeros(1),np.zeros(1,np.int32),np.zeros(1,np.int32),
               send[p],valid[p],order,inverse)


def pack_owner_csr(mesh, source, target, rows):
    """Pack addressable CSR rows whose column indices are canonical source IDs."""
    rows={p:A.tocsr(copy=True) for p,A in rows.items()}
    for A in rows.values():
        A.sum_duplicates();A.sort_indices()
    requests={p:np.unique(A.indices) for p,A in rows.items()}
    slots,send,valid,delta,width=request_layout(mesh,source,requests)
    nnz=max(1,max(A.nnz for A in rows.values()))
    pieces={p:csr_piece(p,A,source,target,requests,slots,send,valid,delta,width,nnz)
            for p,A in rows.items()}
    arrays=tuple(owner_array(mesh,{p:v[k] for p,v in pieces.items()},packed=True) for k in range(10))
    return HaloCSR(arrays,mesh,target.cap,source.cap,delta,width)


@dataclass(frozen=True)
class Requests:
    slots: object
    send: object
    delta: tuple
    width: int


jax.tree_util.register_dataclass(Requests,data_fields=['slots','send'],meta_fields=['delta','width'])


def pack_requests(mesh,source,requests,cap):
    slots,send,_,delta,width=request_layout(mesh,source,requests)
    idx={p:np.pad(value,(0,cap-len(value))) for p,value in slots.items()}
    return Requests(owner_array(mesh,idx,packed=True),owner_array(mesh,send,packed=True),delta,width)


def take_requests(mesh,request,x):
    return _take(mesh,request.delta,request.width)(request.slots,request.send,x)


@lru_cache(None)
def _take(mesh,delta,width):
    def local(slots,send,x):
        return _expand(x,send[0],delta,mesh)[slots[0]]
    return jax.jit(jax.shard_map(local,mesh=mesh,in_specs=(P('records'),)*3,
                                out_specs=P('records'),check_vma=False))
