"""Frozen host algebra for the qualified perpendicular structured point rows.

Extracted without campaign imports or module-global input paths. Numerical policy
comes from scripts/p07_combined_global/kernels.py at the accepted P05 revision.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

EXP4 = np.array([(a, d-a) for d in range(5) for a in range(d+1)], int)

def wrap(x,period):return x-period*np.floor(x/period+.5)


def nearest(nodes,target,count,period):
 distance=np.round(abs(wrap(nodes-target,period))/(period/len(nodes)),12)
 return np.lexsort((np.arange(len(nodes)),distance))[:count]


# Vectorized 28 September 2026: same elementwise operations and reduction order as the frozen
# formulas (bitwise-identical outputs, verified against the loop versions); only Python loops removed.
def _leave_one_out(m):
    others = np.array([[k for k in range(m) if k != j] for j in range(m)], dtype=np.intp)
    drop = np.array([[i for i in range(m - 1) if i != k] for k in range(m - 1)], dtype=np.intp)
    return others, drop


def _loo_products(f, drop):
    # f[..., k] over the m-1 "others"; returns P[..., k] = prod of f without entry k, in original order.
    return np.prod(f[..., drop], axis=-1)


def theta_rows(nodes, target):
    nn = np.asarray(nodes, dtype=np.longdouble); t = np.longdouble(target); m = len(nn)
    others_idx, drop = _leave_one_out(m)
    others = nn[others_idx]
    den = np.sin((nn[:, None] - others) / 2); f = np.sin((t - others) / 2) / den; df = .5 * np.cos((t - others) / 2) / den
    v = np.prod(f, axis=1); P = _loo_products(f, drop)
    d = 0
    for k in range(m - 1): d = d + df[:, k] * P[:, k]
    hit = np.flatnonzero(abs(wrap(target - nodes, 2 * np.pi)) < 32 * np.finfo(float).eps * 2 * np.pi)
    if len(hit): v = np.eye(m)[hit[0]]
    return np.array(v, float), np.array(d, float)


def eta_rows(nodes, target, period, step):
    x = np.asarray(wrap(nodes - target, period) / step, dtype=np.longdouble); m = len(x)
    others_idx, drop = _leave_one_out(m)
    xo = x[others_idx]
    den = x[:, None] - xo; f = -xo / den; df = 1 / den
    v = np.prod(f, axis=1); P = _loo_products(f, drop)
    d = 0
    for k in range(m - 1): d = d + df[:, k] * P[:, k]
    d = d / step
    hit = np.flatnonzero(abs(x) * step < 32 * np.finfo(float).eps * period)
    if len(hit): v = np.eye(len(x))[hit[0]]
    return np.array(v, float), np.array(d, float)


def members(t,o):return t.order[t.starts[o]:t.starts[o+1]]


def cardinal(nodes, targets):
    """Vectorized literal replay of frozen trigonometric cardinal formula."""
    nodes = np.asarray(nodes, np.longdouble); targets = np.asarray(targets, np.longdouble)
    others_idx, drop = _leave_one_out(7)
    others = nodes[others_idx]                                   # (7, 6)
    den = np.sin((nodes[:, None] - others) / 2)                  # (7, 6)
    arg = (targets[:, None, None] - others[None]) / 2            # (Q, 7, 6)
    f = np.sin(arg) / den; df = .5 * np.cos(arg) / den
    v = np.prod(f, axis=2)
    P = _loo_products(f, drop)
    d = 0
    for k in range(6): d = d + df[:, :, k] * P[:, :, k]
    hit = np.abs(wrap(targets[:, None] - nodes[None], 2 * np.pi)) < 32 * np.finfo(float).eps * 2 * np.pi
    for i in np.flatnonzero(hit.any(axis=1)): v[i] = hit[i].astype(float)
    return np.asarray(v, float), np.asarray(d, float)


def pinv(A,root):
    u,s,vt=np.linalg.svd(root[:,None]*A,full_matrices=False);keep=s>=1e-4*s[0]
    C=(vt[keep].T/s[keep])@u[:,keep].T*root[None]
    return C,int(sum(keep)),float(s[0]/s[keep][-1])


def resid(target,C,A):
    return float(np.linalg.norm(target@C@A-target)/max(np.linalg.norm(target),1e-300))


def basis(xy,center,scale,exp):return np.prod(((xy-center)/scale)[:,None,:]**exp[None],axis=2)


def planar(points,center,scale,exp):
 xy=np.column_stack((points[:,0]*np.cos(points[:,1]),points[:,0]*np.sin(points[:,1])));z=(xy-center)/scale
 B=np.prod(z[:,None,:]**exp[None],axis=2);der=np.zeros((len(points),len(exp),2))
 for axis in (0,1):
  mask=exp[:,axis]>0;pow=exp[mask].copy();pow[:,axis]-=1
  der[:,mask,axis]=np.prod(z[:,None,:]**pow[None],axis=2)*exp[mask,axis]/scale
 dx,dy=der[:,:,0],der[:,:,1]
 Dr=np.cos(points[:,1,None])*dx+np.sin(points[:,1,None])*dy
 Dt=points[:,0,None]*(-np.sin(points[:,1,None])*dx+np.cos(points[:,1,None])*dy)
 return B,Dr,Dt


def fit(t,don,center,scale,target):
 A=[];U=[]
 for oid in don:
  mm=members(t,int(oid));B=basis(t.xy[mm],center,scale,EXP4)
  A.append((t.rv[mm]/t.vol[oid])@B);U.append(B.mean(axis=0))
 A=np.asarray(A);U=np.asarray(U)
 root=1/(1+np.linalg.norm((t.g.owner_centroid_xy[don]-center)/scale,axis=1)**2)
 C,rank,cond=pinv(A,root);CU,urank,ucond=pinv(U,root)
 return A,U,root,C,rank,urank,cond,ucond,max(residual(target,C,A),residual(target,CU,U))


def eta_plane_rows(t,p):
 ei=[];ev=[];ed=[]
 for et in p[:,2]:
  ids=nearest(t.centers[2],et,4,t.g.eta_period);v,d=eta_rows(t.centers[2][ids],et,t.g.eta_period,t.g.deta)
  ei.append(ids);ev.append(v);ed.append(d)
 return np.asarray(ei),np.asarray(ev),np.asarray(ed)


def rows(nodes, targets):
    x = np.asarray(nodes, float); s = np.asarray(targets, float); m = len(x)
    others_idx, drop = _leave_one_out(m)
    other = x[others_idx]                                        # (m, m-1)
    den = np.prod(x[:, None] - other, axis=1)                    # (m,)
    diff = s[:, None, None] - other[None]                        # (Q, m, m-1)
    v = np.prod(diff, axis=2) / den
    P = _loo_products(diff, drop)
    d = 0
    for j in range(m - 1): d = d + P[:, :, j]
    return v, d / den


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

residual = resid
