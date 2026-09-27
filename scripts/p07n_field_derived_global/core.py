"""Portable P07N static actions on canonical HSX owners and q3 faces."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts/p07_combined_global'))
sys.path.insert(0,str(REPO/'scripts'))
import kernels as k,topology
from p07_diffusion_global import numerics
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext
from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
from drbx.geometry.fci_perpendicular_integrated_rows import prepare_integrated_face_rows
from fields import NAMES,evaluate,normal,normal_data


def context(t):
    return PointRowContext.from_arrays(faces=t.faces,centers=t.centers,raw_to_owner=t.ro,
        raw_volume=t.rv,owner_volume=t.vol,owner_centroid_xy=t.g.owner_centroid_xy,
        eta_period=t.g.eta_period,dr=t.g.dr,dtheta=t.g.dtheta,deta=t.g.deta)


def load(input_root,sidecar,n):
    k.configure(input_root);t=k.load(n)
    ref=numerics.reference(sidecar,verify_hashes=False)
    return t,ref


def observation_chunk(t,ref,raw_ids):
    raw_ids=np.asarray(raw_ids,dtype=np.int64)
    q=t.pts[raw_ids];values=np.column_stack([evaluate(ref,q,name,t.g.eta_period,derivatives=False)[0] for name in NAMES])
    return {'ids':raw_ids,'owner_ids':t.ro[raw_ids],
            'numerator':t.rv[raw_ids,None]*values}


def selected_observations(t,ref,donors):
    donors=np.unique(np.asarray(donors,dtype=np.int64))
    raw=np.concatenate([k.members(t,int(o)) for o in donors]) if len(donors) else np.empty(0,dtype=np.int64)
    data=observation_chunk(t,ref,raw)
    sums=np.zeros((len(t.vol),len(NAMES)))
    np.add.at(sums,data['owner_ids'],data['numerator'])
    return sums/t.vol[:,None]


def reference_chunk(t,ref,raw_ids,step=1e-4):
    ids=np.asarray(raw_ids,dtype=np.int64);q=t.pts[ids]
    metric=ref._metric(q);J=metric['J'];T,div=ref._perpendicular_geometry(q)
    source=[]
    for name in NAMES:
        _,G,H=evaluate(ref,q,name,t.g.eta_period,step=step)
        point=-(np.einsum('qj,qj->q',div,G)+np.einsum('qij,qij->q',T,H))/J
        source.append(point)
    return {'ids':ids,'owner_ids':t.ro[ids],'numerator':t.rv[ids,None]*np.column_stack(source)}


def normal_samples(ref,wall,period):
    """Column j = normal_data(ref,wall,NAMES[j],period): field-derived physical-normal data."""
    return np.column_stack([normal_data(ref,wall,name,period) for name in NAMES])


def oracle_faces(t,ref,face_ids,order=5):
    ids=np.asarray(face_ids,dtype=np.int64)
    keys=topology.decode(t.n,ids)
    points,weights=numerics.quadrature(t.faces,keys,order,face=True)
    tensor=ref._perpendicular_flux_tensor(points.reshape(-1,3)).reshape(len(ids),order*order,3,3)
    q=np.arange(order*order)
    integ=weights[:,:,None]*tensor[np.arange(len(ids))[:,None],q[None,:],keys[:,0,None],:]
    out=np.empty((len(ids),len(NAMES)))
    for col,name in enumerate(NAMES):
        grad=evaluate(ref,points.reshape(-1,3),name,t.g.eta_period)[1].reshape(len(ids),order*order,3)
        out[:,col]=np.einsum('fqa,fqa->f',integ,grad)
    return {'ids':ids,'O_q5':out}


def face_chunk(t,ref,face_ids,family,endpoints,owner_values=None,patch_limit=256,linearity=False):
    """Return every requested face once; D uses only extra boundary loadings.

    When `linearity` is True, also return, for each non-regular (boundary
    family 1/2/4) face, N_zero_data (owner values kept, boundary/wall data
    replaced by zeros) and N_zero_owner (owner values replaced by zeros,
    boundary/wall data kept). The row application is linear in (owner
    values, boundary data), so N == N_zero_data + N_zero_owner exactly."""
    ids=np.asarray(face_ids,dtype=np.int64);fam=np.asarray(family,dtype=np.uint8);ep=np.asarray(endpoints,dtype=np.int32)
    keys=topology.decode(t.n,ids);p,w=numerics.quadrature(t.faces,keys,3,face=True)
    integ=np.zeros((len(ids),9,3));active=np.flatnonzero(fam!=0)
    if len(active):
        tensor=ref._perpendicular_flux_tensor(p[active].reshape(-1,3)).reshape(len(active),9,3,3)
        integ[active]=w[active,:,None]*tensor[np.arange(len(active))[:,None],np.arange(9)[None,:],keys[active,0,None],:]
    c=context(t);ordinary=np.flatnonzero(~np.isin(fam,(1,2,4)))
    regular_rows=prepare_integrated_face_rows(c,keys[ordinary],fam[ordinary],p[ordinary],integ[ordinary])
    regular=dict(zip(ordinary,regular_rows,strict=True))
    patch_cache={};bc_cache={};neumann={};dirichlet={}
    normal_coefficients=lambda q:normal(ref,q)
    for j in range(len(ids)):
        if j in regular:continue
        degree=3 if fam[j]==4 else 4
        neumann[j]=prepare_neumann_point_rows(c,p[j],normal_coefficients=normal_coefficients,
                       radial_degree=degree,patch_cache=patch_cache)
        dirichlet[j]=prepare_integrated_face_rows(c,keys[j:j+1],fam[j:j+1],p[j:j+1],integ[j:j+1])[0]
        if len(patch_cache)>patch_limit:patch_cache.clear()
        if len(bc_cache)>patch_limit:bc_cache.clear()
    if owner_values is None:
        donors=[row.donor_ids for row in regular.values()]
        donors += [row.donor_ids for row in dirichlet.values()]
        donors += [row.donor_ids for rows in neumann.values() for row in rows]
        owner_values=selected_observations(t,ref,np.concatenate(donors) if donors else [])
    if owner_values.shape!=(len(t.vol),len(NAMES)):
        raise ValueError('owner observation shape mismatch')
    N=np.zeros((len(ids),len(NAMES)));D=N.copy();O=N.copy()
    N_zero_data=np.zeros_like(N) if linearity else None
    N_zero_owner=np.zeros_like(N) if linearity else None
    if len(active):
        grads=np.stack([evaluate(ref,p[active].reshape(-1,3),name,t.g.eta_period)[1] for name in NAMES],axis=1)
        grads=grads.reshape(len(active),9,len(NAMES),3)
        O[active]=np.einsum('fqa,fqka->fk',integ[active],grads)
    wall_area=np.zeros(len(ids));wall_signed=np.zeros_like(N);wall_square=np.zeros_like(N)
    wall_grad_square=np.zeros_like(N);wall_max=np.zeros_like(N);wall_normalized_max=np.zeros_like(N)
    for j in range(len(ids)):
        if j in regular:
            row=regular[j];N[j]=D[j]=row.weights@owner_values[row.donor_ids]
            continue
        restored=np.empty((9,3,len(NAMES)))
        if linearity:
            restored_zero_data=np.empty((9,3,len(NAMES)))
            restored_zero_owner=np.empty((9,3,len(NAMES)))
        for q,row in enumerate(neumann[j]):
            key=row.boundary_points.tobytes()
            if key not in bc_cache:bc_cache[key]=normal_samples(ref,row.boundary_points,t.g.eta_period)
            bc=bc_cache[key]
            owner_term=row.gradient@owner_values[row.donor_ids];bc_term=row.boundary_gradient@bc
            restored[q]=owner_term+bc_term
            if linearity:
                restored_zero_data[q]=owner_term;restored_zero_owner[q]=bc_term
        N[j]=np.einsum('qa,qak->k',integ[j],restored)
        if linearity:
            N_zero_data[j]=np.einsum('qa,qak->k',integ[j],restored_zero_data)
            N_zero_owner[j]=np.einsum('qa,qak->k',integ[j],restored_zero_owner)
        row=dirichlet[j]
        trace=np.column_stack([evaluate(ref,row.trace_donor_points,name,t.g.eta_period,derivatives=False)[0] for name in NAMES])
        tangent=np.stack([evaluate(ref,row.trace_target_points,name,t.g.eta_period)[1][:,1:] for name in NAMES],axis=1)
        D[j]=row.weights@owner_values[row.donor_ids]+row.value_loading@trace+np.einsum('qa,qka->k',row.tangential_loading,tangent)
        if fam[j]==1:
            metric=ref._metric(p[j]);a=normal(ref,p[j]);area=w[j]*abs(metric['J'])*np.sqrt(metric['gcontra'][:,0,0])
            exact=normal_samples(ref,p[j],t.g.eta_period)
            residual=np.einsum('qa,qak->qk',a,restored)-exact
            exact_grad=np.stack([evaluate(ref,p[j],name,t.g.eta_period)[1] for name in NAMES],axis=1)
            norm=np.sqrt(np.maximum(np.einsum('qka,qab,qkb->qk',exact_grad,metric['gcontra'],exact_grad),0))
            wall_area[j]=sum(area)
            wall_signed[j]=area@residual;wall_square[j]=area@(residual**2)
            wall_grad_square[j]=area@(norm**2);wall_max[j]=np.max(abs(residual),axis=0)
            wall_normalized_max[j]=np.max(abs(residual)/np.maximum(norm,1e-12),axis=0)
    out={'ids':ids,'family':fam,'endpoints':ep,'N':N,'D':D,'O_q3':O,
       'wall_area':wall_area,'wall_signed':wall_signed,'wall_square':wall_square,
       'wall_grad_square':wall_grad_square,'wall_max':wall_max,'wall_normalized_max':wall_normalized_max,
       'patch_count':np.array(len(patch_cache),dtype=np.int32)}
    if linearity:
        out['N_zero_data']=N_zero_data;out['N_zero_owner']=N_zero_owner
    return out
