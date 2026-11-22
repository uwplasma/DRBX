"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
import numpy as np
import time


def bind(context):
    N = context.N
    TAU = context.TAU
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    relative = context.relative
    metric_chunks = context.metric_chunks
    aggregate = context.aggregate
    c = context.merge_fraction

    def stats(w,ring,hop,mask):
     z=w[mask];ratio=z[:,8]/hop[ring[mask]];mu=z[:,6];coef=mu/z[:,8]
     return [dict(cut=t,row_share=float(np.mean(ratio<t)),mu_share=float(mu[ratio<t].sum()/mu.sum()),mu_over_ell_share=float(coef[ratio<t].sum()/coef.sum())) for t in [.01,.05,.1,.25]]


    def main():
     dest=OUT/f'merge_N{N}.json'
     if dest.exists():return
     start=time.time();ev=EVALUATOR;cells=gu.build()[1];P=gu.P
     geo=load(OUT/f'arrays/polygon_geometry_N{N}.npz');a=load(OUT/f'arrays/wall_rows_N{N}.npz');w=a['wall'];psi=a['psi'];side=a['side'];flags=a['grazing_flag'];faces=geo['faces']
     ring=np.rint(cells[w[:,0].astype(int)%P,0]*N).astype(int);physical=w[:,9]==0;hops={}
     fr=np.rint(cells[faces[:,:2].astype(int)%P,0]*N).astype(int)
     for r in np.unique(ring[physical]):
      ell=faces[np.any(fr==r,axis=1),9];assert len(ell);hops[int(r)]=float(np.median(ell))
     hop=np.ones(N+1);hop[list(hops)]=list(hops.values());base=aggregate(w,psi,side,P);results=[]
     ident=physical & (np.arange(len(w))%max(1,len(w)//128)==0);m=metric_chunks(ev,w[ident,1:4]);identerr=relative(w[ident,6]*m['M'],psi[ident]);assert identerr<1e-10
     before={name:stats(w,ring,hop,physical&mask) for name,mask in [('all',np.ones(len(w),bool)),('wall_ring',ring==N-1)]}
     groups={}
     for i in np.flatnonzero(physical):groups.setdefault((int(w[i,0]),int(side[i])),[]).append(i)
     targets=np.arange(len(w));fallback=[]
     for (owner,sg),ids in groups.items():
      ids=np.asarray(ids);short=w[ids,8]<c*hop[ring[ids]];ss=ids[short];ll=ids[~short]
      if not len(ss):continue
      if len(ll):
       # Brute small-group distances with deterministic minimum-index tie break.
       for i in ss:targets[i]=ll[np.argmin(np.sum((w[ll,4:6]-w[i,4:6])**2,axis=1))]
      else:
       targets[ss]=ss[0];fallback.append(dict(owner=owner,side=sg,pieces=len(ss),flux=float(psi[ss].sum())))
     roots,inv=np.unique(targets,return_inverse=True);mw=w[roots].copy();mp=np.zeros(len(roots));np.add.at(mp,inv,psi);ms=side[roots];mf=np.zeros(len(roots),bool);np.logical_or.at(mf,inv,flags);counts=np.bincount(inv);changed=np.flatnonzero(counts>1)
     for j in changed:
      ids=np.flatnonzero(inv==j);weights=psi[ids]/mp[j];ref=w[roots[j],2:4];ang=ref+(w[ids,2:4]-ref+np.pi)%TAU-np.pi
      mw[j,1]=1.;mw[j,2:4]=(weights@ang)%TAU;mw[j,4:6]=weights@w[ids,4:6];mw[j,8]=weights@w[ids,8]
     for lo in range(0,len(changed),1024):
      ix=changed[lo:lo+1024];q=mw[ix,1:4];met=metric_chunks(ev,q);mw[ix,6]=mp[ix]/met['M'];mw[ix,7]=np.sign(met['B'][:,0])
     after=aggregate(mw,mp,ms,P);absolute=float(abs(after-base).max());norm=float(np.max(abs(after-base)/np.maximum(abs(base),1e-12)));assert norm<=1e-12 and absolute<=1e-12
     rr=np.rint(cells[mw[:,0].astype(int)%P,0]*N).astype(int);phys=mw[:,9]==0
     save(OUT/f'arrays/merged_N{N}_c{c:g}.npz',wall=mw,psi=mp,side=ms,grazing_flag=mf,old_to_new=inv,source_count=counts,hop=hop)
     results.append(dict(c=c,physical_before=int(physical.sum()),physical_after=int(phys.sum()),merged_targets=len(changed),all_short_owner_sides=fallback,balance_absolute_max=absolute,balance_relative_max_floor1e12=norm,coverage_relative=abs(mp.sum()-psi.sum())/psi.sum(),short_leg={name:stats(mw,rr,hop,phys&mask) for name,mask in [('all',np.ones(len(mw),bool)),('wall_ring',rr==N-1)]},sign_changed_targets=int(np.sum(mw[changed,7]!=w[roots[changed],7]))))
     dump(dest,dict(complete=True,N=N,eta_planes=N,measurement='geometry, no PDE BC',h_hop_by_ring=hops,h_hop_definition='Median full face ell among faces incident to each ring, each face counted once per ring; full torus.',before=before,variants=results,unchanged_row_flux_identity_relative=identerr,new_rays=0,seconds=time.time()-start));log(f'N{N} merge complete; flux balance preserved; new rays 0')


    return {'stats': stats, 'main': main}
