"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
import numpy as np
import jax
import time
from scipy.special import roots_legendre


def bind(context, midpoint_mesh):
    TAU = context.TAU
    HOP = context.HOP
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    relative = context.relative
    sample = context.sample
    seeds = context.seeds
    metric_chunks = context.metric_chunks

    class Flux:
     def __init__(self,ev):self.fn=jax.jit(ev.table.flux_density)
     def __call__(self,q):
      out=[]
      for lo in range(0,len(q),4096):
       qq=q[lo:lo+4096];n=len(qq);pad=np.vstack([qq,np.repeat(qq[-1:],4096-n,axis=0)]);out.append(np.asarray(self.fn(pad))[:n])
      return np.concatenate(out)


    def phi(flux,k,order):
     cells=gu.build()[1];x,w=roots_legendre(order);x=(x+1)/2;w/=2;out=[]
     for lo in range(0,len(cells),8):
      c=cells[lo:lo+8];r=c[:,0,None,None]+(c[:,1]-c[:,0])[:,None,None]*x[None,:,None];t=c[:,2,None,None]+(c[:,3]-c[:,2])[:,None,None]*x[None,None,:];r,t=np.broadcast_arrays(r,t);q=np.c_[r.ravel(),t.ravel()%TAU,np.full(r.size,(k+.5)*HOP)];b=flux(q)[:,2].reshape(r.shape);out.extend(np.sum(b*w[None,:,None]*w[None,None,:],axis=(1,2))*(c[:,1]-c[:,0])*(c[:,3]-c[:,2]))
     return np.array(out)


    def main(k):
     dest=OUT/f'checkpoints/quadrature_{k}.json'
     if dest.exists():return
     start=time.time();ev=EVALUATOR;fl=Flux(ev);q=seeds(midpoint_mesh()['xy'][::19],(k+.5)*HOP);met=metric_chunks(ev,q);err=relative(fl(q),met['J'][:,None]*met['B']);assert err<=1e-10,err
     ladder=[];prior=None
     for order in [8,16,32,64,96]:
      p=phi(fl,k,order);change=None if prior is None else float(np.max(abs(p-prior)/np.maximum(abs(p),1e-12)));ladder.append(dict(order=order,max_cell_relative_change=change,total=float(p.sum())));prior=p
      if change is not None and change<1e-10:break
     save(OUT/f'arrays/phi_{k}.npz',phi=p,cells=gu.build()[1])
     z=load(OUT/f'checkpoints/wall_quad_{k}.npz');q=z['seed'].copy();q[:,0]=1.;b=fl(q);signed=z['weight']*b[:,0];incoming=float(-signed[signed<0].sum());outgoing=float(signed[signed>0].sum());canonical=z['hit']&(z['sign']>0);ids=np.flatnonzero(canonical);sf=np.abs(z['weight']*fl(z['seed'])[:,0]);ratio=np.ones(len(z['seed']));x,w=roots_legendre(8);x=(x+1)/2;w/=2
     if len(ids):
      zz={k:v[ids] for k,v in z.items()};extent=abs((zz['wall'][:,2]-zz['seed'][:,2]+np.pi)%TAU-np.pi)/abs(zz['delta']);exponent=np.zeros(len(ids))
      for t,wt in zip(x,w):
       pt=sample(zz,extent*t);q=np.c_[np.linalg.norm(pt[:,:2],axis=1),np.arctan2(pt[:,1],pt[:,0])%TAU,(zz['seed'][:,2]+zz['delta']*extent*t)%TAU];met=gu.metric(ev,q);exponent+=wt*met['divB']/met['B'][:,2]
      ratio[ids]=np.exp(zz['delta']*extent*exponent)
     slentry=float(sf[ids].sum());slexit=float((sf[ids]*ratio[ids]).sum());qref=z['endpoint'].copy();qref[:,2]%=TAU
     save(OUT/f'arrays/wall_reference_{k}.npz',seed=z['seed'],endpoint=qref,wall=z['wall'],hit=z['hit'],source_flux=sf,transport_ratio=ratio,canonical=canonical,sign=z['sign'],delta=z['delta'])
     rec=dict(complete=True,plane=k,seconds=time.time()-start,phi_ladder=ladder,flux_density_identity=err,surface_influx=incoming,surface_outflux=outgoing,surface_absolute_flux=incoming+outgoing,signed_inward_flux=incoming-outgoing,sliver_entry_flux=slentry,sliver_exit_flux=slexit,physical_reference=incoming+outgoing-slentry-slexit,plane_reaching_surface_flux=float(sf[~z['hit']].sum()),canonical_slivers=len(ids),surface_rule='q3 on 3N/2 x 6 strip patches; canonical sliver exit uses unchanged 8-point divergence transport; legacy gathered geometry products-anchor completion unavailable');dump(dest,rec);log(f'plane {k} quadrature checkpoint {rec["seconds"]:.1f}s; no traces')


    return {'Flux': Flux, 'phi': phi, 'main': main}
