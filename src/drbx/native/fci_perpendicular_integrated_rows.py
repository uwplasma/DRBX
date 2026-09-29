"""Bucketed, array-only JAX application of frozen P07 integrated face rows."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays


class IntegratedFaceBatch(NamedTuple):
    face_ids: np.ndarray
    donor_ids: np.ndarray
    weights: np.ndarray
    boundary_donor_ids: np.ndarray
    tangential_ids: np.ndarray
    tangential_weights: np.ndarray
    conditioned: np.ndarray


class IntegratedFacePayload(NamedTuple):
    batches: tuple[IntegratedFaceBatch, ...]
    lower_owner: np.ndarray
    upper_owner: np.ndarray
    owner_volume: np.ndarray
    boundary_query_count: int
    face_count: int


class IntegratedFacePlan(NamedTuple):
    payload: IntegratedFacePayload
    boundary_points: np.ndarray
    donor_eta_offsets: tuple[tuple[tuple[int, ...], ...], ...]
    bucket_summary: tuple[tuple[int, bool, int, int], ...]


def save_integrated_face_plan(path: str | Path, plan: IntegratedFacePlan, identity: dict) -> None:
    """Atomically cache prepared arrays and an exact caller-supplied identity."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    metadata={'schema':'drbx.p07-integrated-face-plan.v2','identity':identity,
              'donor_eta_offsets':plan.donor_eta_offsets,'bucket_summary':plan.bucket_summary,
              'batch_count':len(plan.payload.batches)}
    arrays={'metadata_json':np.asarray(json.dumps(metadata,sort_keys=True)),
            'boundary_points':plan.boundary_points,'lower_owner':plan.payload.lower_owner,
            'upper_owner':plan.payload.upper_owner,'owner_volume':plan.payload.owner_volume,
            'boundary_query_count':np.asarray(plan.payload.boundary_query_count),
            'face_count':np.asarray(plan.payload.face_count)}
    for q,batch in enumerate(plan.payload.batches):
        for name,value in zip(IntegratedFaceBatch._fields,batch,strict=True):
            arrays[f'batch_{q}_{name}']=np.asarray(value)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)


def load_integrated_face_plan(path: str | Path, identity: dict) -> IntegratedFacePlan:
    """Reject a stale geometry, coefficient, face, or BC policy cache."""
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(np.asarray(z['metadata_json']).item()))
        if metadata.get('schema')!='drbx.p07-integrated-face-plan.v2' or metadata.get('identity')!=identity:
            raise ValueError('integrated face plan identity mismatch')
        batches=tuple(IntegratedFaceBatch(*(np.asarray(z[f'batch_{q}_{name}'])
                      for name in IntegratedFaceBatch._fields))
                      for q in range(int(metadata['batch_count'])))
        payload=IntegratedFacePayload(batches,np.asarray(z['lower_owner']),
            np.asarray(z['upper_owner']),np.asarray(z['owner_volume']),
            int(np.asarray(z['boundary_query_count']).item()),int(np.asarray(z['face_count']).item()))
        points=np.asarray(z['boundary_points'])
    offsets=tuple(tuple(tuple(int(v) for v in donor) for donor in row)
                  for row in metadata['donor_eta_offsets'])
    summary=tuple(tuple(item) for item in metadata['bucket_summary'])
    return IntegratedFacePlan(payload,points,offsets,summary)


def lower_integrated_face_rows(context,rows: tuple[IntegratedFaceRow,...],endpoints) -> IntegratedFacePlan:
    """Bucket by family, conditioning, and donor count; preserve face order."""
    endpoints=np.asarray(endpoints,dtype=np.int64)
    if endpoints.shape!=(len(rows),2):raise ValueError('one lower/upper owner pair is required per face')
    if np.any((endpoints< -1)|(endpoints>=len(context.vol))):raise ValueError('invalid face owner endpoint')
    queries=[];query_map={}
    def query_id(point):
        key=tuple(map(float,point))
        if key not in query_map:
            query_map[key]=len(queries);queries.append(np.asarray(point,dtype=np.float64))
        return query_map[key]
    groups={};eta_offsets=[]
    for face,row in enumerate(rows):
        donor=np.asarray(row.donor_ids,dtype=np.int64);count=len(donor)
        if np.any((donor<0)|(donor>=len(context.vol))) or row.weights.shape!=(count,):
            raise ValueError('invalid integrated donor or weight')
        dq=np.zeros(count,dtype=np.int64);tq=np.zeros(9,dtype=np.int64);tw=np.zeros((9,2))
        if row.boundary_conditioned:
            if row.trace_donor_points.shape!=(count,3) or row.trace_target_points.shape!=(9,3):
                raise ValueError('conditioned integrated row requires complete Dirichlet queries')
            if not np.array_equal(row.value_loading,-row.weights):
                raise ValueError('unsupported integrated boundary value loading')
            dq=np.asarray([query_id(p) for p in row.trace_donor_points],dtype=np.int64)
            tq=np.asarray([query_id(p) for p in row.trace_target_points],dtype=np.int64)
            tw=np.asarray(row.tangential_loading,dtype=np.float64)
        width=16*((count+15)//16) if count else 0
        groups.setdefault((int(row.family),bool(row.boundary_conditioned),width),[]).append(
            (face,donor,np.asarray(row.weights),dq,tq,tw))
        if count:
            point=row.trace_target_points[4]
            target_plane=int(np.argmin(abs(context.centers[2]-point[2])))
            offsets=[]
            for oid in donor:
                planes=np.unique(context.members(int(oid))%context.n)
                offsets.append(tuple(int((int(k)-target_plane+context.n//2)%context.n-context.n//2)
                                     for k in planes))
            eta_offsets.append(tuple(offsets))
        else:eta_offsets.append(())
    batches=[];summary=[]
    for (family,conditioned,width),items in sorted(groups.items()):
        count=len(items);face_ids=np.empty(count,dtype=np.int64)
        donor_ids=np.zeros((count,width),dtype=np.int64)
        weights=np.zeros((count,width));donor_query=np.zeros((count,width),dtype=np.int64)
        tangent_ids=np.zeros((count,9),dtype=np.int64);tangent_weights=np.zeros((count,9,2))
        for q,(face,donor,w,dq,tq,tw) in enumerate(items):
            d=len(donor);face_ids[q]=face
            donor_ids[q,:d]=donor;weights[q,:d]=w;donor_query[q,:d]=dq
            tangent_ids[q]=tq;tangent_weights[q]=tw
        batches.append(IntegratedFaceBatch(face_ids,donor_ids,weights,donor_query,
                                           tangent_ids,tangent_weights,np.full(count,conditioned,dtype=bool)))
        summary.append((family,conditioned,width,sum(len(item[1]) for item in items)))
    payload=IntegratedFacePayload(tuple(batches),endpoints[:,0],endpoints[:,1],
                                  np.asarray(context.vol),len(queries),len(rows))
    return IntegratedFacePlan(payload,np.asarray(queries).reshape(-1,3),tuple(eta_offsets),tuple(summary))


def apply_integrated_face_rows(payload: IntegratedFacePayload,owner_fields,
                               boundary: BoundaryArrays | None=None):
    """Return one shared, oriented integrated flux per selected physical face."""
    field=jnp.asarray(owner_fields)
    if field.ndim!=2:raise ValueError('owner_fields must have shape (owners, fields)')
    if payload.boundary_query_count:
        if boundary is None:raise ValueError('conditioned integrated rows require Dirichlet data')
        if (boundary.values.shape!=(payload.boundary_query_count,field.shape[1]) or
            boundary.tangential_gradients.shape!=(payload.boundary_query_count,2,field.shape[1])):
            raise ValueError('Dirichlet array shape mismatch')
        bv=jnp.asarray(boundary.values);bg=jnp.asarray(boundary.tangential_gradients)
    else:
        bv=jnp.zeros((1,field.shape[1]),dtype=field.dtype)
        bg=jnp.zeros((1,2,field.shape[1]),dtype=field.dtype)
    flux=jnp.zeros((payload.face_count,field.shape[1]),dtype=field.dtype)
    for batch in payload.batches:
        donors=field[batch.donor_ids]
        donors=donors-jnp.where(batch.conditioned[:,None,None],bv[batch.boundary_donor_ids],0)
        values=jnp.einsum('fd,fdk->fk',batch.weights,donors)
        values+=jnp.einsum('fqa,fqak->fk',batch.tangential_weights,bg[batch.tangential_ids])
        flux=flux.at[batch.face_ids].set(values)
    return flux


def scatter_integrated_face_flux(payload: IntegratedFacePayload,face_flux):
    """P07 residual: lower minus, upper plus, divided by stored owner volume."""
    flux=jnp.asarray(face_flux)
    out=jnp.zeros((len(payload.owner_volume),flux.shape[1]),dtype=flux.dtype)
    lower=payload.lower_owner;upper=payload.upper_owner
    out=out.at[jnp.maximum(lower,0)].add(jnp.where((lower>=0)[:,None],-flux,0))
    out=out.at[jnp.maximum(upper,0)].add(jnp.where((upper>=0)[:,None],flux,0))
    return out/jnp.asarray(payload.owner_volume)[:,None]
