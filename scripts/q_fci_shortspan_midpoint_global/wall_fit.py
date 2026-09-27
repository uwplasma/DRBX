"""Frozen 60-coefficient Q wall fit, extracted from 2026-09-27 pilot."""
"""Research-only joint quartic-transverse/cubic-eta wall reconstruction."""
import numpy as np
from drbx.geometry.fci_boundary_functional_reconstruction import BoundaryRelation,prepare_boundary_reconstruction

XY=tuple((a,d-a) for d in range(5) for a in range(d+1))
FAMILIES=('K5','K6','K7','K8','rect_2_3','rect_3_3','rect_4_4','nearest_count_4_4')

def basis(q,anchor,scale,H):
    q=np.asarray(q,float).reshape(-1,3);u,t,e=q.T
    x=(u*np.cos(t)-anchor[0])/scale;y=(u*np.sin(t)-anchor[1])/scale
    z=((e-anchor[2]+np.pi)%(2*np.pi)-np.pi)/H
    out=np.empty((len(q),60));dx=np.empty_like(out);dy=np.empty_like(out);dz=np.empty_like(out)
    for i,(a,b) in enumerate(XY):
        for m in range(4):
            j=4*i+m;out[:,j]=x**a*y**b*z**m
            dx[:,j]=(a*x**(a-1)*y**b*z**m/scale) if a else 0
            dy[:,j]=(b*x**a*y**(b-1)*z**m/scale) if b else 0
            dz[:,j]=(m*x**a*y**b*z**(m-1)/H) if m else 0
    return out,dx,dy,dz

def relation(ctx,anchor,scale,H,kind):
    n=ctx['N'];theta=anchor[3];eta=anchor[2]
    gauss,_=np.polynomial.legendre.leggauss(4)
    q=np.array([[1.,theta+v*H,eta+2*g*H] for v in (-4,-2,0,2,4) for g in gauss])
    val,dx,dy,dz=basis(q,anchor,scale,H)
    if kind=='D':C=val
    else:
        A=ctx['evaluator']._position_and_jacobian(q)[1]
        inv=np.linalg.inv(A);g=np.einsum('nik,njk->nij',inv,inv)
        a=g[:,0,:]/np.sqrt(g[:,0,0])[:,None]
        u,t=q[:,0],q[:,1]
        du=dx*np.cos(t)[:,None]+dy*np.sin(t)[:,None]
        dt=dx*(-u*np.sin(t))[:,None]+dy*(u*np.cos(t))[:,None]
        C=a[:,0,None]*du+a[:,1,None]*dt+a[:,2,None]*dz
    rel=BoundaryRelation(constraint_rows=C,rhs_map=np.eye(len(q)),labels=tuple(f'wall_{i}' for i in range(len(q))),functional='value' if kind=='D' else 'normal_derivative')
    return q,C,rel

def relation_cached(ctx,index,anchor,scale,H,kind):
    """Reuse wall nodes and basis across families and D/N, with distinct constraints."""
    key=(np.asarray(anchor,float).tobytes(),float(scale),float(H))
    cached=index._wall_relations.get(key)
    if cached is None:
        n=ctx['N'];theta=anchor[3];eta=anchor[2]
        gauss,_=np.polynomial.legendre.leggauss(4)
        q=np.array([[1.,theta+v*H,eta+2*g*H] for v in (-4,-2,0,2,4) for g in gauss])
        val,dx,dy,dz=basis(q,anchor,scale,H)
        cached=(q,val,dx,dy,dz,None);index._wall_relations[key]=cached
    q,val,dx,dy,dz,normals=cached
    if kind=='D':C=val
    else:
        if normals is None:
            A=ctx['evaluator']._position_and_jacobian(q)[1]
            inv=np.linalg.inv(A);g=np.einsum('nik,njk->nij',inv,inv)
            normals=g[:,0,:]/np.sqrt(g[:,0,0])[:,None]
            index._wall_relations[key]=(q,val,dx,dy,dz,normals)
        u,t=q[:,0],q[:,1]
        du=dx*np.cos(t)[:,None]+dy*np.sin(t)[:,None]
        dt=dx*(-u*np.sin(t))[:,None]+dy*(u*np.cos(t))[:,None]
        C=normals[:,0,None]*du+normals[:,1,None]*dt+normals[:,2,None]*dz
    rel=BoundaryRelation(constraint_rows=C,rhs_map=np.eye(len(q)),labels=tuple(f'wall_{i}' for i in range(len(q))),functional='value' if kind=='D' else 'normal_derivative')
    return q,C,rel

def _rectangle(index,i,j,plane,hr,ht):
    n=index.shape[0];rr=np.arange(max(0,i-hr),min(n-1,i+hr)+1);tt=(j+np.arange(-ht,ht+1))%n
    return np.unique(index.labels.reshape(index.shape)[np.ix_(rr,tt,[plane])].ravel())

def donors(index,midpoint,ijk,plane,family):
    n=index.shape[0];i,j=ijk[:2]
    if family.startswith('K'):
        return np.unique(index.labels.reshape(index.shape)[:int(family[1:]),:,plane].ravel())
    if family.startswith('rect_'):
        hr,ht=map(int,family.split('_')[1:]);return _rectangle(index,i,j,plane,hr,ht)
    pool=_rectangle(index,i,j,plane,6,6);count=len(_rectangle(index,i,j,plane,4,4))
    xy=np.array((midpoint[0]*np.cos(midpoint[1]),midpoint[0]*np.sin(midpoint[1])))
    centers=np.asarray(index.ctx['model'].centroid)[pool]
    order=np.lexsort((pool,np.linalg.norm(centers-xy,axis=1)))
    return np.sort(pool[order[:count]])

def prepare(ctx,index,midpoint,ijk,targets,family,kind,*,normalize_neumann=False):
    n=ctx['N'];H=2*np.pi/n;grid=ctx['artifact'].geometry.grid
    scale=max(float(grid.x.centers[1]-grid.x.centers[0]),float(midpoint[0]*H))
    anchor=np.array((midpoint[0]*np.cos(midpoint[1]),midpoint[0]*np.sin(midpoint[1]),midpoint[2],midpoint[1]))
    xy=anchor[:2];k=int(ijk[2]);M=[];W=[];donor_ids=[];plane_counts=[];fail=[]
    for off in (-2,-1,0,1,2):
        pl=(k+off)%n;d=donors(index,midpoint,ijk,pl,family);plane_counts.append(len(d))
        if len(d)>81:fail.append(f'owner_cap_plane_{pl}_{len(d)}')
        if len(d)<8:fail.append(f'few_owners_plane_{pl}_{len(d)}')
        if np.any(index.plane[d]!=pl):fail.append(f'cross_plane_owner_{pl}')
        if fail:continue
        xyrows=index.local(d,xy,np.array((scale,scale)),XY)
        Z=off;rows=np.empty((len(d),60))
        for m in range(4):rows[:,m::4]=xyrows*Z**m
        centroid=np.asarray(ctx['model'].centroid)[d]
        dist=np.linalg.norm((centroid-xy)/scale,axis=1);w=1/(1+dist*dist)**2
        M.append(rows);W.append(w);donor_ids.extend(d.tolist())
    if fail:return None,dict(failure=fail,plane_counts=plane_counts)
    M=np.vstack(M);W=np.concatenate(W)
    q,C,rel=relation_cached(ctx,index,anchor,scale,H,kind)
    constraint_scale=1.
    if kind=='N' and normalize_neumann:
        A=ctx['evaluator']._position_and_jacobian(np.array([[1.,midpoint[1],midpoint[2]]]))[1][0]
        constraint_scale=(1/n)/np.linalg.norm(np.linalg.inv(A)[0])
        rel=BoundaryRelation(constraint_rows=constraint_scale*C,rhs_map=constraint_scale*np.eye(len(q)),labels=rel.labels,functional='normal_derivative')
    target,_,_,_=basis(targets,anchor,scale,H)
    # The compiler accepts weights under a square root; the interior W is the row multiplier.
    try:prepared=prepare_boundary_reconstruction(M,W*W,rel,rcond=1e-12)
    except Exception as exc:return None,dict(failure=[type(exc).__name__+': '+str(exc)],plane_counts=plane_counts)
    Eowner=target@prepared.owner_map;Ebc=target@prepared.boundary_map
    ident=prepared.owner_map@M+prepared.boundary_map@C
    repro=float(np.max(abs(ident-np.eye(60))))
    cond=max(prepared.observation_condition,prepared.constraint_condition,prepared.reduced_condition)
    if repro>3e-10:fail.append(f'reproduction_{repro}')
    if not np.isfinite(cond) or cond>1e8:fail.append(f'condition_{cond}')
    if not np.all(np.isfinite(Eowner)) or not np.all(np.isfinite(Ebc)):fail.append('nonfinite_map')
    metadata=dict(failure=fail,plane_counts=plane_counts,donor_count=len(donor_ids),constraint_rank=prepared.constraint_rank,reduced_rank=prepared.reduced_rank,
                  observation_rank=prepared.observation_rank,observation_condition=prepared.observation_condition,constraint_condition=prepared.constraint_condition,
                  reduced_condition=prepared.reduced_condition,max_condition=cond,basis_reproduction=repro,constraint_scale=float(constraint_scale),
                  boundary_residual=float(np.max(abs(C@prepared.owner_map))),
                  boundary_rhs_residual=float(np.max(abs(C@prepared.boundary_map-np.eye(20)))),
                  endpoint_owner_l1=float(np.max(np.sum(abs(Eowner),axis=1))),endpoint_bc_l1=float(np.max(np.sum(abs(Ebc),axis=1))))
    return dict(M=M,C=C,wall_nodes=q,donors=np.array(donor_ids),target=target,Eowner=Eowner,Ebc=Ebc,prepared=prepared,anchor=anchor,scale=scale,H=H),metadata
