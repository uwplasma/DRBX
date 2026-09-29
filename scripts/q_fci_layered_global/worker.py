"""Node-local CPU tasks, complete-owner projection and bounded fit caches."""
import os,time,resource,sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
from . import storage as io,model
from .fields import FIELDS,all_fields
from .support import BalancedHybrid as Hybrid
from .wall import Wall,normal
CTX=T=H=STATE=ROOT=IDENT=MODE=None

def init(root,input_root,n,mode):
    global CTX,T,H,STATE,ROOT,IDENT,MODE
    ROOT=Path(root);IDENT,_=io.identity(ROOT);MODE=mode
    if hasattr(os,'sched_getaffinity'):
        import multiprocessing
        cpus=sorted(os.sched_getaffinity(0));slot=multiprocessing.current_process()._identity
        os.sched_setaffinity(0,{cpus[((slot[0] if slot else 1)-1)%len(cpus)]})
    os.environ['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'cache/cpu');os.environ['DRBX_CACHE_DIR']=str(ROOT/'cache/cpu')
    CTX,T=model.context(n,input_root,physical=True,magnetic=mode=='score');H=Hybrid(T);STATE=model.states(T) if mode=='score' else None
    import jax
    if jax.default_backend()!='cpu':raise RuntimeError('CPU worker backend required')

def max_rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)

def raw_members(owners):return np.concatenate([T.order[T.starts[o]:T.starts[o+1]] for o in owners])

def boundary_grad(w,slots,d):
    W,I,GW,GI=w.basis(slots);V,B,D,E=w.maps(slots);wv,wg=all_fields(w.wall);data=np.einsum('si,sif->sf',w.a,wg)
    normal_gradient=np.einsum('sid,df->sif',D,d)+np.einsum('sib,bf->sif',E,data)
    wallq=slots.copy();wallq[:,0]=1;_,gq=all_fields(wallq)
    direct=np.einsum('sid,df->sif',GI,d-np.tile(wv,(4,1)));direct[:,1:]+=gq[:,1:]
    return np.stack((direct,normal_gradient))

def task(job):
    stage,block,owners=job;dest=ROOT/stage/f'N{T.n}'/f'{MODE}_{block:06d}.npz'
    if io.complete(dest,IDENT):return dict(skipped=True,block=block)
    start=time.monotonic();raw=raw_members(owners);meta=dict(block=block,owners=len(owners),raw=len(raw),max_condition=0.,max_reproduction=0.,max_donors=0,repair_planes=0,outside_caps=0,wall_holdout_normal_max=0.,constant_max=0.)
    if MODE=='score':
        tracepath=ROOT/stage/f'N{T.n}'/f'trace_{block:06d}.npz'
        if not io.complete(tracepath,IDENT):raise RuntimeError('missing trace chunk')
        g=np.load(tracepath);assert np.array_equal(g['raw'],raw);slots,b,L,O,R,Rh=model.slots_and_action(CTX,T,raw,g['ends']);N=np.empty((len(raw),2,2,len(FIELDS)),complex)
    else:
        slots=np.repeat(T.pts[raw,None,:],3,axis=1);slots[:,0,2]-=T.g.deta/32;slots[:,2,2]+=T.g.deta/32
    for pos,rr in enumerate(raw):
        ijk=np.array(np.unravel_index(rr,(T.n,)*3));points=slots[pos]
        if ijk[0]>=T.n-2:
            w=Wall(CTX,T,*ijk);_,_,D,E=w.maps(w.wall);a=w.a
            residual=max(abs(np.einsum('si,sid->sd',a,D)).max(),abs(np.einsum('si,sib->sb',a,E)-np.eye(35)).max())
            if residual>1e-8:raise RuntimeError(('Neumann constraint',int(rr),residual))
            meta['max_condition']=max(meta['max_condition'],w.condition);meta['max_reproduction']=max(meta['max_reproduction'],float(residual));meta['max_donors']=max(meta['max_donors'],len(w.ids))
            if MODE=='score':
                grad=boundary_grad(w,points,STATE[w.ids]);N[pos]=np.einsum('as,si,bsif->baf',L[pos],b[pos],grad)
                hold=np.array([[1.,w.anchor[1]+v*T.g.dtheta,w.anchor[2]+e*T.g.deta] for v,e in ((-.4,-.3),(.2,.35),(.45,-.15),(0.,0.))]);_,_,D,E=w.maps(hold);gn=np.einsum('si,sif->sf',a,all_fields(w.wall)[1]);gg=np.einsum('sid,df->sif',D,STATE[w.ids])+np.einsum('sib,bf->sif',E,gn)
                err=np.einsum('si,sif->sf',normal(CTX,hold),gg-all_fields(hold)[1]);meta['wall_holdout_normal_max']=max(meta['wall_holdout_normal_max'],float(abs(err).max()))
        else:
            if np.any(points[:,0]>1):raise RuntimeError(('non-wall support exits wall',int(rr)))
            ids,V,D,m=H.rows(ijk,points);meta['max_condition']=max(meta['max_condition'],m['condition']);meta['max_reproduction']=max(meta['max_reproduction'],m['reproduction']);meta['max_donors']=max(meta['max_donors'],len(ids))
            if not np.all(np.isfinite(D)) or abs(D.sum(axis=-1)*np.array([T.g.dr,T.g.dtheta,T.g.deta])).max()>1e-8:raise RuntimeError(('constant row',int(rr)))
            if MODE=='score':
                act=L[pos]@np.einsum('si,sid->sd',b[pos],D)@STATE[ids];N[pos]=act[None]
        meta['outside_caps']+=int(np.sum(points[:,0]>1))
    # Inspect every cached plane fit, including geometry-only preflight. A larger
    # runtime stencil is forbidden: pool expansion is setup-only.
    plane_info=[fit[3] for fit in H.inner.fits.values()]
    counts=[len(p['donors']) for p in plane_info]
    if any(c != 28 for c in counts) or any(p['rank'] != 15 for p in plane_info):
        raise RuntimeError(('balanced28 plane invariant',block,counts))
    meta['repair_planes']=0
    meta['balanced_planes']=len(plane_info)
    meta['inner_plane_donor_counts']=sorted(set(counts))
    meta['inner_max_pool']=max((p['selection']['pool_size'] for p in plane_info),default=0)
    meta['inner_max_scaled_radius']=max((p['max_radius'] for p in plane_info),default=0.)
    meta['inner_expanded_planes']=sum(p['expansion']>0 for p in plane_info)
    meta['inner_policy']='balanced28'
    if MODE=='score':
        # Project every complete raw owner, retaining signed complex fields.
        Ni=[];Oi=[];Ri=[];Rhi=[];rad=[];vol=[]
        for o in owners:
            hit=np.flatnonzero(T.ro[raw]==o);weight=T.rv[raw[hit]]/T.vol[o];assert abs(weight.sum()-1)<1e-12
            Ni.append(np.einsum('r,rbaf->baf',weight,N[hit]));Oi.append(np.einsum('r,raf->af',weight,O[hit]));Ri.append(weight@R[hit]);Rhi.append(weight@Rh[hit]);rad.append(int(np.unravel_index(raw[hit[0]],(T.n,)*3)[0]));vol.append(T.vol[o])
        Ni=np.array(Ni);meta['constant_max']=float(abs(Ni[:,:,:,FIELDS.index('constant')]).max())
        if meta['constant_max']>1e-7:raise RuntimeError(('constant action',meta))
        if not all(np.all(np.isfinite(z)) for z in (Ni,Oi,Ri,Rhi)):raise RuntimeError('nonfinite action')
        io.arrays(dest,owners=owners,raw=raw,N=Ni,O=Oi,R=Ri,R_half=Rhi,radial=rad,volume=vol)
    else:io.arrays(dest,owners=owners,raw=raw)
    H.clear();meta.update(wall_s=time.monotonic()-start,peak_rss_gib=max_rss(),backend='cpu');io.record(dest,IDENT,**meta);return meta
