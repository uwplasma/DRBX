"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import jax


def bind(context):
    N = context.N
    NP = context.NP
    HOP = context.HOP
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    load = context.load
    dump = context.dump
    log = context.log
    sample = context.sample
    seeds = context.seeds
    moment = context.moment

    class Flux:
     def __init__(self,ev):self.fn=jax.jit(ev.table.flux_density)
     def __call__(self,q):
      if not len(q):return np.empty(0)
      out=[]
      for lo in range(0,len(q),4096):
       x=q[lo:lo+4096];n=len(x);pad=np.vstack([x,np.repeat(x[-1:],4096-n,axis=0)]);out.extend(np.asarray(self.fn(pad))[:n,2])
      return np.abs(np.asarray(out))/q[:,0]


    def main():
     ev=EVALUATOR;fl=Flux(ev);rows=[];detail=[]
     for k in range(NP):
      p=load(ROOT/(f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz'));tri=p['xy'][p['tri']];area=[];point=[]
      for t in tri:
       a,b,pt=moment(t,np.eye(3),'half');area.append(a);point.append(pt)
      area=np.array(area);represented=np.zeros_like(area)
      for name in REACH_CLASSES:
       z=load(ROOT/(f'base_polygon/arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz'))
       for t,owner in zip(z['xy'],z['parent']):represented[owner]+=moment(t,np.eye(3),'half')[0]
      gap=area-represented;classes=np.array([len(np.unique(p['classes'][ii])) for ii in p['tri']]);q=seeds(np.asarray(point),(k+1)*HOP);density=fl(q);rad=np.linalg.norm(np.asarray(point),axis=1);sample=np.where(rad>=1-.016/N,'outer_0016h',np.where(rad>=1-.16/N,'0016h_to_016h','below_016h'))
      for nc in np.unique(classes):
       for r in np.unique(sample):
        m=(classes==nc)&(sample==r)
        if m.any():rows.append(dict(strip=k,vertex_classes=int(nc),radial_layer=r,triangles=int(m.sum()),source_area=float(area[m].sum()),missing_area=float(np.maximum(gap[m],0).sum()),overlap_area=float(np.maximum(-gap[m],0).sum()),net_gap_area=float(gap[m].sum()),midpoint_gap_flux_proxy=float((gap[m]*density[m]).sum())))
      for j in np.flatnonzero(gap>1e-10):detail.append(dict(strip=k,parent=int(j),vertex_classes=int(classes[j]),theta=float(q[j,1]),rho=float(q[j,0]),gap_area=float(gap[j]),area_fraction=float(gap[j]/area[j])))
     dump(OUT/f'partition_N{N}.json',dict(complete=True,N=N,scope='full source band',rows=rows,gap_parents=detail,new_rays=0,note='Area deficit is measured from saved class polygons; midpoint gap flux is only a proxy, not an allocated landing flux or causal delta decomposition.'))
     log(f'N{N} saved source partition audit complete; no rays')


    return {'Flux': Flux, 'main': main}
