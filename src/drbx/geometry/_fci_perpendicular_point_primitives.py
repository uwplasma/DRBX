"""Compatibility exports for frozen perpendicular point reconstruction algebra."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ._reconstruction_primitives import (
    EXP4,
    wrap,
    nearest,
    _leave_one_out,
    _loo_products,
    theta_rows,
    eta_rows,
    members,
    cardinal,
    pinv,
    resid,
    basis,
    planar,
    fit,
    eta_plane_rows,
    rows,
    residual,
)


def family(n,key):
    axis,i=map(int,key[:2])
    if axis==0:return 'quartic_wall' if i>=n-2 else 'centered_radial'
    return 'boundary_transverse' if i>=n-2 else 'interior_transverse'


def boundary_map(n,case,key,face_index,points,weights,integrand,top,period):
    qcount=len(points);h=1/n;axis,i=map(int,key[:2]);kind=family(n,key)
    grid=top[6];th=grid['grid.y.centers'];et=grid['grid.z.centers']
    ti=np.array([nearest(th,q[1],7,2*np.pi) for q in points]);ei=np.array([nearest(et,q[2],4,period) for q in points])
    tv,td=np.array([theta_rows(th[v],q[1]) for v,q in zip(ti,points)]).transpose(1,0,2)
    ev,ed=np.array([eta_rows(et[v],q[2],period,period/n) for v,q in zip(ei,points)]).transpose(1,0,2)
    bc=kind in ('quartic_wall','boundary_transverse')
    if kind=='quartic_wall':nodes=np.array([0.,-.5,-1.5,-2.5,-3.5]);layers=np.arange(n-1,n-5,-1);s=(points[:,0]-1)/h
    elif kind=='boundary_transverse':nodes=np.array([0.,-.5,-1.5,-2.5]);layers=np.arange(n-1,n-4,-1);s=(points[:,0]-1)/h
    elif kind=='centered_radial':nodes=np.array([1.5,.5,-.5,-1.5]);layers=np.array([i+1,i,i-1,i-2]);s=(points[:,0]-i*h)/h
    else:nodes=np.array([-1.,0.,1.,2.]);layers=np.array([i-1,i,i+1,i+2]);s=(points[:,0]-(i+.5)*h)/h
    if not np.all((layers>=0)&(layers<n)):
        raise ValueError(f"unsupported boundary radial layers for {key}")
    L,D=rows(nodes,s);owners=np.empty((qcount,len(layers),4,7),int);raws=np.empty_like(owners)
    for q in range(qcount):
        for l,r in enumerate(layers):
            for ee,eta in enumerate(ei[q]):
                for tt,theta in enumerate(ti[q]):
                    raw=np.ravel_multi_index((r,int(theta),int(eta)),(n,n,n));oid=int(top[0][raw]);members=top[4][top[5][oid]:top[5][oid+1]]
                    if len(members)!=1 or members[0]!=raw:
                        raise ValueError(f"unsupported aggregated boundary donor: {case} {key} owner={oid} raw={raw}")
                    owners[q,l,ee,tt]=oid;raws[q,l,ee,tt]=raw
    ids=np.unique(owners);idx=np.searchsorted(ids,owners);raw_ids=np.array([top[4][top[5][oid]] for oid in ids])
    angular=np.stack((ev[:,:,None]*tv[:,None,:],ev[:,:,None]*td[:,None,:],ed[:,:,None]*tv[:,None,:]))
    layer=np.zeros((3,qcount,len(layers),len(ids)))
    for a in range(3):
        for q in range(qcount):
            for l in range(len(layers)):np.add.at(layer[a,q,l],idx[q,l].ravel(),angular[a,q].ravel())
    off=int(bc);grad=np.empty((qcount,3,len(ids)))
    grad[:,0]=np.einsum('ql,qld->qd',D[:,off:]/h,layer[0]);grad[:,1]=np.einsum('ql,qld->qd',L[:,off:],layer[1]);grad[:,2]=np.einsum('ql,qld->qd',L[:,off:],layer[2])
    comp=np.einsum('qa,qad->ad',integrand,grad);W=comp.sum(axis=0);csr=csr_matrix(W[None])
    trace_donor=top[3][raw_ids].copy();trace_donor[:,0]=1;trace_target=points.copy();trace_target[:,0]=1
    return dict(case=case,n=n,order=int(round(np.sqrt(qcount))),kind=kind,face_key=key,face_index=face_index,points=points,weights=weights,integrand=integrand,
                radial_nodes=nodes,radial_targets=s,radial_layers=layers,radial_value=L,radial_derivative=D,h=h,period=period,
                theta_indices=ti,eta_indices=ei,theta_value=tv,theta_derivative=td,eta_value=ev,eta_derivative=ed,
                donor_ids=ids,raw_ids=raw_ids,raw_coordinates=top[3][raw_ids],requested_raw_ids=raws,owner_indices=idx,
                layer_maps=layer,gradient_map=grad,component_rows=comp,W=W,boundary_conditioned=bc,
                trace_donor_points=trace_donor,trace_target_points=trace_target,value_loading=-W if bc else np.zeros_like(W),
                tangential_loading=integrand[:,1:] if bc else np.zeros((qcount,2)),
                W_csr_data=csr.data,W_csr_indices=csr.indices,W_csr_indptr=csr.indptr,W_csr_shape=csr.shape)
