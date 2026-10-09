"""Portable canonical topology and frozen physical evaluators."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from .fields import FIELDS,all_fields
HERE=Path(__file__).resolve().parent
ALPHAS=np.array([1/16,1/32])

def context(n,root,physical=True,magnetic=True):
    cfg=json.loads((HERE/'inputs.json').read_text());root=Path(root);folder=root/cfg['geometry']/f'{n}x{n}x{n}'
    with np.load(folder/'base_geometry.npz') as z:centers=tuple(z[f'grid.{a}.centers'].copy() for a in 'xyz')
    with np.load(folder/'rlp_topology.npz') as z:
        active=np.flatnonzero(z['is_active_owner'].ravel());lookup=np.full(n**3,-1,int);lookup[active]=np.arange(len(active));ro=lookup[z['aggregate_id'].ravel()];rv=z['raw_volume'].ravel().copy()
    assert np.all(ro>=0) and np.all(rv>0)
    vol=np.bincount(ro,weights=rv);pts=np.stack(np.meshgrid(*centers,indexing='ij'),axis=-1).reshape(-1,3);xy=np.column_stack((pts[:,0]*np.cos(pts[:,1]),pts[:,0]*np.sin(pts[:,1])))
    centroid=np.column_stack([np.bincount(ro,weights=rv*xy[:,a])/vol for a in (0,1)])
    counts=np.bincount(ro);t=SimpleNamespace(n=n,centers=centers,ro=ro,rv=rv,vol=vol,pts=pts,xy=xy,order=np.argsort(ro,kind='stable'),starts=np.r_[0,np.cumsum(counts)],g=SimpleNamespace(dr=1/n,dtheta=2*np.pi/n,deta=2*np.pi/n,eta_period=2*np.pi,owner_centroid_xy=centroid))
    ctx={}
    if physical:
        from .vendor.geometry.MetricEvaluator import MetricEvaluator
        with np.load(root/cfg['metric_cache']) as z:ctx['evaluator']=MetricEvaluator.from_cache_payload(z,prefix='metric_evaluator_')
    if magnetic:
        from .vendor.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
        from .vendor.geometry.hsx_jax_field import JaxHsxMagneticField
        ctx['bfield']=bfield_evaluator_from_makegrid(root/cfg['makegrid'],currents=np.array(cfg['currents']),method='cubic',toroidal_method='compact_c3')
        ctx['tracer']=JaxHsxMagneticField.from_evaluators(ctx['evaluator'],ctx['bfield'])
    return ctx,t

def states(t):
    values=all_fields(t.pts)[0];out=np.empty((len(t.vol),len(FIELDS)),complex)
    for j in range(len(FIELDS)):out[:,j]=(np.bincount(t.ro,weights=t.rv*values[:,j].real)+1j*np.bincount(t.ro,weights=t.rv*values[:,j].imag))/t.vol
    return out

def geom(ctx,points):
    p=np.asarray(points);out=[np.empty(len(p)),np.empty((len(p),3)),np.empty(len(p))];inside=(p[:,0]>=0)&(p[:,0]<=1)
    for flag in (True,False):
        ids=np.flatnonzero(inside==flag)
        for start in range(0,len(ids),512 if flag else 128):
            ii=ids[start:start+(512 if flag else 128)];q=p[ii]
            if flag:
                position,J=ctx['evaluator']._position_and_jacobian(q);Bc=ctx['bfield'].evaluate_cartesian(position);B=np.linalg.norm(Bc,axis=-1);b=np.linalg.solve(J,Bc[...,None])[...,0]/B[:,None]
            else:
                import jax.numpy as jnp
                q=q.copy();q[:,1:]%=2*np.pi;_,J=ctx['tracer'].metric.position_and_jacobian(jnp.asarray(q));Bc,B=ctx['tracer'](jnp.asarray(q));J=np.asarray(J);B=np.asarray(B);b=np.asarray(Bc)/B[:,None]
            out[0][ii]=abs(np.linalg.det(J));out[1][ii]=b;out[2][ii]=B
    if not all(np.all(np.isfinite(v)) for v in out) or np.any(out[0]<=0) or np.any(out[2]<=0) or np.any(out[1][:,2]<=1e-10):raise ValueError('invalid physical geometry')
    return tuple(out)

def references(ctx,points):
    J0=geom(ctx,points)[0];outputs=[]
    for step in (1e-4,5e-5):
        pp=np.broadcast_to(points[:,None,None,:],(len(points),3,4,3)).copy()
        for a in range(3):pp[:,a,:,a]+=np.array([-2,-1,1,2])*step
        flat=pp.reshape(-1,3);J,b,B=geom(ctx,flat);J=J.reshape(-1,3,4);b=b.reshape(-1,3,4,3);B=B.reshape(-1,3,4);wt=np.array([1,-8,8,-1])/(12*step)
        div=sum(np.einsum('pq,q->p',J[:,a]*b[:,a,:,a]*B[:,a],wt) for a in range(3))/J0
        grad=all_fields(flat)[1].reshape(len(points),3,4,3,len(FIELDS));flux=np.einsum('paqi,paqif->paqf',b,grad)
        ref=sum(np.einsum('pqf,q->pf',J[:,a,:,None]*b[:,a,:,a,None]*flux[:,a],wt) for a in range(3))/J0[:,None]
        outputs.append((ref,div))
    return outputs

def slots_and_action(ctx,t,raw,ends):
    points=t.pts[raw];slots=np.stack((ends[:,0],ends[:,1],points,ends[:,2],ends[:,3],points),axis=1)
    _,b,B=geom(ctx,slots.reshape(-1,3));b=b.reshape(-1,6,3);B=B.reshape(-1,6);(R,db),(Rh,dbh)=references(ctx,points)
    L=np.zeros((len(raw),2,6));scale=B[:,2]*b[:,2,2]
    for ai,a in enumerate(ALPHAS):
        s=ai*3;L[:,ai,s]=-scale/(a*t.g.deta*B[:,s]);L[:,ai,s+1]=scale/(a*t.g.deta*B[:,s+1]);L[:,ai,s+2]=db/B[:,2]
    exact=all_fields(slots.reshape(-1,3))[1].reshape(len(raw),6,3,len(FIELDS));O=np.einsum('ras,rsi,rsif->raf',L,b,exact)
    return slots,b,L,O,R,Rh
