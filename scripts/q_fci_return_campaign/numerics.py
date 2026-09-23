"""Host-side traced FCI preparation and independent analytical reference kernels.

No historical workspace imports. All physical geometry comes from the frozen
HSX metric/MAKEGRID inputs. Runtime state is never replaced by an exact field.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
from .endpoints import OwnerMoments
from ..q03_direct_campaign.frozen_mms import ManufacturedField

FIELDS = ('radial_eta', 'angular_x', 'mixed_y_eta', 'constant')
EXPS = tuple((a,b,t-a-b) for t in range(4) for a in range(t+1) for b in range(t-a+1))


def regular(p, eta):
    p=np.asarray(p)
    return np.column_stack((p[:,0]*np.cos(p[:,1]),p[:,0]*np.sin(p[:,1]),
                            eta+(p[:,2]-eta+np.pi)%(2*np.pi)-np.pi))


def basis(p, center, scale):
    z=(regular(p,float(center[2]))-center)/scale
    return np.column_stack([np.prod(z**e,axis=1) for e in EXPS])


def gradient_basis(p,b,center,scale):
    u,theta,_=p.T;c,s=np.cos(theta),np.sin(theta)
    bc=np.column_stack((b[:,0]*c-b[:,1]*u*s,b[:,0]*s+b[:,1]*u*c,b[:,2]))/scale
    z=(regular(p,float(center[2]))-center)/scale
    out=[]
    for powers in EXPS[1:]:
        col=np.zeros(len(p))
        for axis,power in enumerate(powers):
            if power:
                e=list(powers);e[axis]-=1
                col+=power*bc[:,axis]*np.prod(z**e,axis=1)
        out.append(col)
    return np.column_stack(out)


def base(ctx,p):
    # The frozen producer exposes this primitive; no metric inverse/diagnostic is needed.
    position,A=ctx['evaluator']._position_and_jacobian(p)
    B=np.asarray(ctx['bfield'].evaluate_cartesian(position))
    Bmag=np.linalg.norm(B,axis=-1)
    Bcontra=np.linalg.solve(A,B[...,None])[...,0]
    return np.abs(np.linalg.det(A)),Bcontra/Bmag[:,None],Bmag


def fields(ctx,p):
    out=[]
    for name in FIELDS:
        f=ctx['fields'][name]['field'];v,g,h=f.value_gradient_hessian(p);a=f.amplitude
        out.append((1+a*v,a*g,a*h))
    return tuple(np.stack([x[k] for x in out],axis=1) for k in range(3))


def context(n,input_root,config):
    from drbx.geometry.MetricEvaluator import MetricEvaluator
    from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
    root=Path(input_root);directory=root/config['geometry']/f'{n}x{n}x{n}'
    with np.load(directory/'base_geometry.npz') as z:
        grid=SimpleNamespace(**{a:SimpleNamespace(centers=z[f'grid.{a}.centers'].copy(),faces=z[f'grid.{a}.faces'].copy()) for a in 'xyz'})
    with np.load(directory/'rlp_topology.npz') as z:
        topology={k:z[k].copy() for k in ('aggregate_id','is_active_owner','raw_volume','face_id','face_axis','face_storage_index','face_minus_aggregate_id','face_plus_aggregate_id')}
    owner_ids=np.flatnonzero(topology['is_active_owner'].ravel());lookup=np.full(n**3,-1,dtype=int);lookup[owner_ids]=np.arange(len(owner_ids))
    labels=lookup[topology['aggregate_id']];assert np.all(labels>=0)
    rawvol=topology['raw_volume'].reshape(n,n,n)
    vol=np.bincount(labels.ravel(),weights=rawvol.ravel(),minlength=len(owner_ids))
    keys=np.column_stack((topology['face_axis'],topology['face_storage_index']))
    lower=np.array(topology['face_minus_aggregate_id'],dtype=int);upper=np.array(topology['face_plus_aggregate_id'],dtype=int)
    for a in (lower,upper):
        ok=a>=0;a[ok]=lookup[a[ok]]
    # Stored periodic endpoints are canonicalized and validated, not double counted.
    for axis in (1,2):keys[keys[:,0]==axis,axis+1]%=n
    if len(np.unique(keys,axis=0))!=len(keys):raise ValueError('duplicate canonical topology faces')
    with np.load(root/config['metric_cache']) as z:evaluator=MetricEvaluator.from_cache_payload(z,prefix='metric_evaluator_')
    bfield=bfield_evaluator_from_makegrid(root/config['makegrid'],currents=np.array(config['currents']),method='cubic')
    fs={}
    for name,data in config['fields'].items():
        fs[name]={'field':ManufacturedField(m=data['m'],amplitude=data['amplitude'],lam=data['lambda'],chi=data['chi'],eta_period=data['eta_period'],radial_profile=data['radial_profile'],field_kind=name)}
    ctx={'N':n,'artifact':SimpleNamespace(geometry=SimpleNamespace(shape=(n,n,n),grid=grid),polar_angular_geometry=SimpleNamespace(raw_volume=rawvol)),
         'topology':{'compact_raw_owner':labels,'owner_eta':owner_ids%n},'volume':vol,'owner_ids':owner_ids,
         'evaluator':evaluator,'bfield':bfield,'fields':fs,'keys':keys,'lower':lower,'upper':upper,'config':config}
    state=np.zeros((len(vol),4));centers=(grid.x.centers,grid.y.centers,grid.z.centers)
    for lo in range(0,n**3,4096):
        ids=np.arange(lo,min(lo+4096,n**3));ij=np.array(np.unravel_index(ids,(n,n,n))).T
        p=np.column_stack([centers[a][ij[:,a]] for a in range(3)])
        np.add.at(state,labels.ravel()[ids],rawvol.ravel()[ids,None]*fields(ctx,p)[0])
    state/=vol[:,None];ctx['state']=state
    for fi,name in enumerate(FIELDS):fs[name]['state']=state[:,fi]
    ctx['model']=OwnerMoments(ctx)
    return ctx


def face_center_scale(ctx,key):
    axis,i,j,k=map(int,key);s=(i,j,k);grid=ctx['artifact'].geometry.grid;ff=(grid.x.faces,grid.y.faces,grid.z.faces)
    p=np.array([ff[a][s[a]] if a==axis else .5*(ff[a][s[a]]+ff[a][s[a]+1]) for a in range(3)])
    center=regular(p[None],p[2])[0];dr=ff[0][1]-ff[0][0];dtheta=ff[1][1]-ff[1][0]
    return center,np.array([max(dr,abs(center[0])*dtheta),max(dr,abs(center[1])*dtheta),ff[2][1]-ff[2][0]])


def quadrature(ctx,keys,order,face=True):
    grid=ctx['artifact'].geometry.grid;ff=(grid.x.faces,grid.y.faces,grid.z.faces)
    nodes,weights=np.polynomial.legendre.leggauss(order);points=[];factors=[]
    for key in keys:
        axis=int(key[0]) if face else -1;cell=key[1:] if face else key
        values=[];ww=[]
        for a,idx in enumerate(cell):
            if a==axis:values.append(np.array([ff[a][idx]]));ww.append(np.ones(1))
            else:
                lo,hi=ff[a][idx:idx+2];values.append(.5*(lo+hi)+.5*(hi-lo)*nodes);ww.append(.5*(hi-lo)*weights)
        points.append(np.stack(np.meshgrid(*values,indexing='ij'),axis=-1).reshape(-1,3))
        factors.append(np.einsum('i,j,k->ijk',*ww).ravel())
    return np.array(points),np.array(factors)


def exact_flux(ctx,keys,order):
    p,w=quadrature(ctx,keys,order);shape=w.shape
    J,b,_=base(ctx,p.reshape(-1,3));g=fields(ctx,p.reshape(-1,3))[1]
    b=b.reshape(*shape,3);grad=np.einsum('nqi,nqfi->nqf',b,g.reshape(*shape,4,3))
    normal=b[np.arange(len(keys))[:,None],np.arange(shape[1])[None,:],np.asarray(keys)[:,0,None]]
    return np.sum((w*J.reshape(shape)*normal)[:,:,None]*grad,axis=1)


def prescribed_boundary_flux(keys,loads):
    """Dynamic integrated coordinate-positive flux, supplied by boundary data.

    Frozen scalar MMS catalogue has zero gradient at u=1. This is not a claim
    that normal-Neumann data alone imply zero parallel flux for general fields.
    """
    loads=np.asarray(loads,dtype=float)
    if loads.shape!=(len(keys),4) or not np.all(np.isfinite(loads)):raise ValueError('invalid boundary loads')
    return loads.copy()


def row_keys(n,ids):
    ids=np.asarray(ids);raw=ids//2;ijk=np.array(np.unravel_index(raw,(n,n,n))).T
    return np.column_stack((ijk,2*(ids%2)-1))


def row_ids(n,keys):
    q=np.asarray(keys,dtype=int)
    return 2*np.ravel_multi_index(q[:,:3].T,(n,n,n))+(q[:,3]>0)


def footprint(ctx,keys):
    grid=ctx['artifact'].geometry.grid;keys=np.asarray(keys);i,j,k=keys[:,:3].T
    frac=np.array([.25,.75]);u=grid.x.faces[i,None]+np.diff(grid.x.faces)[i,None]*frac
    th=grid.y.faces[j,None]+np.diff(grid.y.faces)[j,None]*frac
    p=np.stack(np.broadcast_arrays(u[:,:,None],th[:,None,:],grid.z.centers[k,None,None]+np.zeros((len(keys),2,2))),axis=-1).reshape(-1,3)
    area=np.repeat(np.diff(grid.x.faces)[i]*np.diff(grid.y.faces)[j]/4,4)
    return p,area


def trace(ctx,seeds,direction,steps=None):
    from .rk4 import advance
    from drbx.geometry.hsx_jax_field import JaxHsxMagneticField
    if 'rk_field' not in ctx:
        ctx['rk_field']=JaxHsxMagneticField.from_evaluators(ctx['evaluator'],ctx['bfield'])
    steps=int(steps or ctx['config']['policy']['trace_substeps'])
    if steps<1:raise ValueError('RK4 substeps must be positive')
    # Pad only the final short batch to a fixed per-stage shape, avoiding recompilation.
    count=len(seeds);capacity=max(count,int(ctx.get('trace_capacity',count)))
    padded=np.concatenate((seeds,np.repeat(seeds[:1],capacity-count,axis=0)))
    result=advance(ctx['rk_field'],padded,float(direction*2*np.pi/ctx['N']),steps=steps)
    end,ell,valid,bad=(np.asarray(x)[:count] for x in result)
    if np.any(bad):raise ValueError('nonpositive or nonfinite b^eta: frozen orientation invalid')
    return end,ell,valid,{'rhs_batches':4*steps,'seed_stage_evaluations':capacity*4*steps}


def trace_rows(ctx,ids):
    """Trace a source-plane/direction batch; exclude any footprint with an invalid seed."""
    t=time.monotonic();ids=np.asarray(ids,dtype=int);keys=row_keys(ctx['N'],ids)
    if len(set(keys[:,2]))!=1 or len(set(keys[:,3]))!=1:raise ValueError('mixed trace batch')
    source,area=footprint(ctx,keys);ns=len(ids)
    endpoint,length,seed_valid,stats=trace(ctx,source,int(keys[0,3]))
    valid=seed_valid.reshape(ns,4).all(axis=1)
    end=endpoint.reshape(ns,4,3).copy();ell=length.reshape(ns,4).copy()
    end[~valid]=np.nan;ell[~valid]=np.nan
    J,b,B=base(ctx,source);F=(area*J*B*np.abs(b[:,2])).reshape(ns,4)
    out={'ids':ids,'valid':valid,'source':source.reshape(ns,4,3),'endpoint':end,'ell':ell,'F':F,
         'Z':np.full(ns,np.nan),'g_sec':np.full((ns,4),np.nan),'numerical':np.full((ns,4),np.nan),
         'fit_l1':np.zeros(ns),'fit_defect':np.zeros(ns)}
    fit_start=time.monotonic()
    for ri in np.flatnonzero(valid):
        pp=out['source'][ri];ee=end[ri];out['Z'][ri]=F[ri]@ell[ri]
        exact=fields(ctx,np.vstack((pp,ee)))[0];vals=[]
        for point in np.vstack((pp,ee)):
            donors,coef,_,meta=ctx['model'].endpoint_pair(point,include_control=False)
            vals.append(coef@ctx['state'][donors]);out['fit_l1'][ri]=max(out['fit_l1'][ri],meta['chosen']['coefficient_l1'])
            out['fit_defect'][ri]=max(out['fit_defect'][ri],meta['chosen']['residual'])
        vals=np.array(vals);d=keys[ri,3]
        out['g_sec'][ri]=d*F[ri]@(exact[4:]-exact[:4])/out['Z'][ri]
        out['numerical'][ri]=d*F[ri]@(vals[4:]-vals[:4])/out['Z'][ri]
    out['seconds']=np.array(time.monotonic()-t);out['fit_seconds']=np.array(time.monotonic()-fit_start);out['nfev']=np.array(stats['rhs_batches']);out['seed_stage_evaluations']=np.array(stats['seed_stage_evaluations'])
    return out


def candidate_ids(n,key,halo):
    axis,i,j,k=map(int,key);ii=min(i,n-1)
    rows=[(a,b%n,c%n,d) for a in range(max(0,ii-halo),min(n,ii+halo+1))
          for b in range(j-halo,j+halo+1) for c in (k-1,k,k+1) for d in (-1,1)]
    return np.unique(row_ids(n,rows))


def interval_counts(n,ids):
    k=row_keys(n,ids);inter=np.minimum(k[:,2],(k[:,2]+k[:,3])%n)
    # A seam interval is identified by its lower directed plane, N-1.
    inter=np.where(np.abs(k[:,2]-(k[:,2]+k[:,3])%n)>1,n-1,inter)
    return inter


def return_map(ctx,key,rows,halo=2,frozen_ids=None):
    """One field-independent shared-face endpoint-moment map.

    `rows` is a dense indexable memory-map catalogue. Caller ensures that the
    requested candidate rows have been produced before requesting a map.
    """
    n=ctx['N'];fc,sc=face_center_scale(ctx,key)
    ids=candidate_ids(n,key,halo) if frozen_ids is None else np.asarray(frozen_ids)
    ids=ids[np.asarray(rows['valid'][ids],bool)]
    if not len(ids):raise ValueError('no valid interior observations')
    src=rows['source'][ids];F=rows['F'][ids]
    chart=regular(src.reshape(-1,3),float(fc[2])).reshape(src.shape)
    centers=np.sum(F[:,:,None]*chart,axis=1)/F.sum(axis=1)[:,None]
    d=centers-fc;d[:,2]=(d[:,2]+np.pi)%(2*np.pi)-np.pi;distance=np.sum((d/sc)**2,axis=1)
    order=np.lexsort((ids,distance));inter=interval_counts(n,ids)
    if frozen_ids is None:
        chosen=list(order[:min(96,len(order))])
        for interval in sorted(set(inter)):
            chosen.extend(int(x) for x in order[inter[order]==interval][:3] if int(x) not in chosen)
        chosen=np.array(sorted(set(chosen),key=lambda x:(distance[x],ids[x])))
    else:chosen=np.arange(len(ids))
    selected=ids[chosen];dist=distance[chosen];weights=(1+dist)**-1.5
    src=rows['source'][selected];end=rows['endpoint'][selected];F=rows['F'][selected];Z=rows['Z'][selected]
    delta=basis(end.reshape(-1,3),fc,sc).reshape(len(selected),4,20)[:,:,1:]-basis(src.reshape(-1,3),fc,sc).reshape(len(selected),4,20)[:,:,1:]
    direction=np.where(selected%2,1,-1)
    A=np.matmul((direction[:,None]*F)[:,None,:],delta)[:,0,:]/Z[:,None]
    p,lw=quadrature(ctx,np.asarray(key)[None],5);p=p[0];lw=lw[0];J,b,_=base(ctx,p)
    target=(lw*J*b[:,int(key[0])])@gradient_basis(p,b,fc,sc)
    sw=np.sqrt(weights);U,s,Vt=np.linalg.svd(A*sw[:,None],full_matrices=False)
    keep=s>s[0]*1e-11;K=(Vt[keep].T/s[keep])@(U[:,keep].T*sw[None])
    functional=target@K;defect=float(np.linalg.norm(functional@A-target)/max(np.linalg.norm(target),1e-300))
    intervals=interval_counts(n,selected);counts=[int(np.sum(intervals==i)) for i in sorted(set(intervals))]
    meta={'rank':int(keep.sum()),'condition':float(s[0]/s[keep][-1]),'target_relative_defect':defect,
          'l1':float(np.sum(abs(functional))),'rows':len(selected),'interval_counts':counts,'halo':halo,
          'extent':float(np.sqrt(dist.max())),'singular_values':s.tolist()}
    # This only detects an unresolved target functional, not a scientific order gate.
    if len(counts)<4 or min(counts)<3 or defect>1e-8:raise ValueError(f'unresolved face functional: {meta}')
    return functional@rows['numerical'][selected],functional@rows['g_sec'][selected],selected,functional,meta


def assemble(ctx,face_ids,flux):
    """Divergence of coordinate-positive flux: + at lower, - at upper owner."""
    out=np.zeros((len(ctx['volume']),flux.shape[1]))
    for sign,side in ((1,ctx['lower']),(-1,ctx['upper'])):
        ids=side[face_ids];good=ids>=0;np.add.at(out,ids[good],sign*flux[good])
    return out/ctx['volume'][:,None]


def strong_reference(ctx,raw_ids,order=9,step=1e-5):
    """Independent volume integral of div(b (b.grad T)); no face reconstruction."""
    ijk=np.array(np.unravel_index(np.asarray(raw_ids),(ctx['N'],)*3)).T
    total=np.zeros(4);volume=0.
    for cell in ijk:
        p,lw=quadrature(ctx,cell[None],order,face=False);p=p[0];lw=lw[0];J,b,_=base(ctx,p)
        _,g,h=fields(ctx,p);db=np.empty((len(p),3,3));div=np.zeros(len(p))
        # Radial Gauss nodes remain positive under a locally bounded FD step.
        for axis in range(3):
            eps=min(step,float(p[:,0].min())/4) if axis==0 else step
            vals=[]
            for multiple in (-2,-1,1,2):
                pp=p.copy();pp[:,axis]+=multiple*eps;vals.append(base(ctx,pp)[:2])
            jm2,bm2=vals[0];jm1,bm1=vals[1];jp1,bp1=vals[2];jp2,bp2=vals[3]
            db[:,:,axis]=(-bp2+8*bp1-8*bm1+bm2)/(12*eps)
            div+=(-jp2*bp2[:,axis]+8*jp1*bp1[:,axis]-8*jm1*bm1[:,axis]+jm2*bm2[:,axis])/(12*eps*J)
        bg=np.einsum('ni,nfi->nf',b,g)
        derivative=np.einsum('nai,nfa->nfi',db,g)+np.einsum('na,nfai->nfi',b,h)
        L=np.einsum('ni,nfi->nf',b,derivative)+div[:,None]*bg
        total+=np.sum((lw*J)[:,None]*L,axis=0);volume+=float(lw@J)
    return total/volume,volume
