"""Frozen P07 combined candidate numerical kernels, curated from bounded studies.

Do not change donor selection, SVD cutoff, polynomial degree, or BC channels
without a new scientific review. See README.md for source attribution.
"""
from __future__ import annotations
import os,sys,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from p07_diffusion_global import numerics as num
INPUT_ROOT=None
EXP=np.array([(a,d-a) for d in range(4) for a in range(d+1)])
EXP4=np.array([(a,d-a) for d in range(5) for a in range(d+1)],int)
def configure(input_root):
 global INPUT_ROOT
 INPUT_ROOT=Path(input_root).resolve()
def rss():
 v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return v/(2**30 if sys.platform=='darwin' else 2**20)


# Frozen from work/p07_matched_radial_layers_20260924/run.py
def wrap(x,period):return x-period*np.floor(x/period+.5)


# Frozen from work/p07_matched_radial_layers_20260924/run.py
def nearest(nodes,target,count,period):
 distance=np.round(abs(wrap(nodes-target,period))/(period/len(nodes)),12)
 return np.lexsort((np.arange(len(nodes)),distance))[:count]


# Frozen from work/p07_matched_radial_layers_20260924/run.py
def theta_rows(nodes,target):
 nn=np.asarray(nodes,dtype=np.longdouble);t=np.longdouble(target);v=[];d=[]
 for j in range(len(nn)):
  kk=[k for k in range(len(nn)) if k!=j];den=np.sin((nn[j]-nn[kk])/2);f=np.sin((t-nn[kk])/2)/den;df=.5*np.cos((t-nn[kk])/2)/den
  v.append(np.prod(f));d.append(sum(df[k]*np.prod(np.delete(f,k)) for k in range(len(kk))))
 hit=np.flatnonzero(abs(wrap(target-nodes,2*np.pi))<32*np.finfo(float).eps*2*np.pi)
 if len(hit):v=np.eye(len(nn))[hit[0]]
 return np.array(v,float),np.array(d,float)


# Frozen from work/p07_matched_radial_layers_20260924/run.py
def eta_rows(nodes,target,period,step):
 x=np.asarray(wrap(nodes-target,period)/step,dtype=np.longdouble);v=[];d=[]
 for j in range(len(x)):
  kk=[k for k in range(len(x)) if k!=j];den=x[j]-x[kk];f=-x[kk]/den;df=1/den
  v.append(np.prod(f));d.append(sum(df[k]*np.prod(np.delete(f,k)) for k in range(len(kk)))/step)
 hit=np.flatnonzero(abs(x)*step<32*np.finfo(float).eps*period)
 if len(hit):v=np.eye(len(x))[hit[0]]
 return np.array(v,float),np.array(d,float)


# Frozen from work/p07_aggregated_structured_extension_20260924/run.py
def load(n):
    faces,centers,ro,rv,vol,g=num.build_context(n,INPUT_ROOT/'geometry_artifacts/rlp_convergence_32_48_64_20260917')
    ijk=np.array(np.unravel_index(np.arange(n**3),(n,n,n))).T
    pts=np.column_stack([centers[a][ijk[:,a]] for a in range(3)])
    xy=np.column_stack((pts[:,0]*np.cos(pts[:,1]),pts[:,0]*np.sin(pts[:,1])))
    order=np.argsort(ro,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(ro))]
    return SimpleNamespace(n=n,faces=faces,centers=centers,ro=ro,rv=rv,vol=vol,g=g,pts=pts,xy=xy,order=order,starts=starts)


# Frozen from work/p07_aggregated_structured_extension_20260924/run.py
def members(t,o):return t.order[t.starts[o]:t.starts[o+1]]


# Frozen from work/p07_aggregated_structured_extension_20260924/run.py
def radial(nodes,target):
    v=[];d=[]
    for j,x in enumerate(nodes):
        other=np.delete(nodes,j);den=np.prod(x-other)
        v.append(np.prod(target[:,None]-other,axis=1)/den)
        d.append(sum(np.prod(target[:,None]-np.delete(other,k),axis=1) for k in range(3))/den)
    return np.array(v).T,np.array(d).T


# Frozen from work/p07_direct_structured_comparison_20260924/compare.py
def cardinal(nodes,targets):
    """Vectorized literal replay of frozen trigonometric cardinal formula."""
    nodes=np.asarray(nodes,np.longdouble);targets=np.asarray(targets,np.longdouble)
    v=np.empty((len(targets),7),np.longdouble);d=v.copy()
    for j in range(7):
        others=np.delete(nodes,j);den=np.sin((nodes[j]-others)/2)
        f=np.sin((targets[:,None]-others)/2)/den
        df=.5*np.cos((targets[:,None]-others)/2)/den
        v[:,j]=np.prod(f,axis=1)
        d[:,j]=sum(df[:,k]*np.prod(np.delete(f,k,axis=1),axis=1) for k in range(6))
    hit=np.abs(wrap(targets[:,None]-nodes[None],2*np.pi))<32*np.finfo(float).eps*2*np.pi
    for i in np.flatnonzero(hit.any(axis=1)):v[i]=hit[i].astype(float)
    return np.asarray(v,float),np.asarray(d,float)


# Frozen from work/p07_direct_structured_comparison_20260924/compare.py
def pinv(A,root):
    u,s,vt=np.linalg.svd(root[:,None]*A,full_matrices=False);keep=s>=1e-4*s[0]
    C=(vt[keep].T/s[keep])@u[:,keep].T*root[None]
    return C,int(sum(keep)),float(s[0]/s[keep][-1])


# Frozen from work/p07_direct_structured_comparison_20260924/compare.py
def resid(target,C,A):
    return float(np.linalg.norm(target@C@A-target)/max(np.linalg.norm(target),1e-300))

residual = resid


# Frozen from work/p07_axis_quartic_functional_20260924/prepare.py
def basis(xy,center,scale,exp):return np.prod(((xy-center)/scale)[:,None,:]**exp[None],axis=2)


# Frozen from work/p07_axis_quartic_functional_20260924/prepare.py
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


# Frozen from work/p07_axis_quartic_functional_20260924/prepare.py
def target_for_plane(p,integ,ei,ev,ed,kk,center,scale,exp):
 B,Dr,Dt=planar(p,center,scale,exp);v=np.sum(ev*(ei==kk),axis=1);d=np.sum(ed*(ei==kk),axis=1)
 return np.sum(integ[:,0,None]*Dr*v[:,None]+integ[:,1,None]*Dt*v[:,None]+integ[:,2,None]*B*d[:,None],axis=0)


# Frozen from work/p07_axis_quartic_functional_20260924/prepare.py
def fit(t,don,center,scale,target):
 A=[];U=[]
 for oid in don:
  mm=members(t,int(oid));B=basis(t.xy[mm],center,scale,EXP4)
  A.append((t.rv[mm]/t.vol[oid])@B);U.append(B.mean(axis=0))
 A=np.asarray(A);U=np.asarray(U)
 root=1/(1+np.linalg.norm((t.g.owner_centroid_xy[don]-center)/scale,axis=1)**2)
 C,rank,cond=pinv(A,root);CU,urank,ucond=pinv(U,root)
 return A,U,root,C,rank,urank,cond,ucond,max(residual(target,C,A),residual(target,CU,U))


# Frozen from work/p07_axis_quartic_functional_20260924/prepare.py
def eta_plane_rows(t,p):
 ei=[];ev=[];ed=[]
 for et in p[:,2]:
  ids=nearest(t.centers[2],et,4,t.g.eta_period);v,d=eta_rows(t.centers[2][ids],et,t.g.eta_period,t.g.deta)
  ei.append(ids);ev.append(v);ed.append(d)
 return np.asarray(ei),np.asarray(ev),np.asarray(ed)


# Frozen from work/p07_wall_quartic_closure_20260924/prepare.py
def rows(nodes,targets):
    x=np.asarray(nodes,float);s=np.asarray(targets,float);v=np.empty((len(s),len(x)));d=np.empty_like(v)
    for k,here in enumerate(x):
        other=np.delete(x,k);den=np.prod(here-other);v[:,k]=np.prod(s[:,None]-other[None],axis=1)/den
        d[:,k]=sum(np.prod(s[:,None]-np.delete(other,j)[None],axis=1) for j in range(len(other)))/den
    return v,d

def family(n,key):
    axis,i=map(int,key[:2])
    if axis==0:return 'quartic_wall' if i>=n-2 else 'centered_radial'
    return 'boundary_transverse' if i>=n-2 else 'interior_transverse'


# Frozen from work/p07_boundary_quadrature_20260924/prepare.py
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
    assert np.all((layers>=0)&(layers<n))
    L,D=rows(nodes,s);owners=np.empty((qcount,len(layers),4,7),int);raws=np.empty_like(owners)
    for q in range(qcount):
        for l,r in enumerate(layers):
            for ee,eta in enumerate(ei[q]):
                for tt,theta in enumerate(ti[q]):
                    raw=np.ravel_multi_index((r,int(theta),int(eta)),(n,n,n));oid=int(top[0][raw]);members=top[4][top[5][oid]:top[5][oid+1]]
                    assert len(members)==1 and members[0]==raw,(case,key,'aggregated donor',oid,raw)
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
