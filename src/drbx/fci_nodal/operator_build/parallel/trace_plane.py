"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import PLANE_WALL_STEM, REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import time
from scipy.special import roots_legendre


def bind(context, midpoint_mesh, plane_mesh, midpoint_observe, clip_class, clip_plane_seeded_wall, plane_seeds):
    N = context.N
    NP = context.NP
    TAU = context.TAU
    HOP = context.HOP
    DISK_ANGLES = context.disk_angles
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    book = context.book
    trace_cached = context.trace_cached
    plane_charged = context.plane_charged
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log

    def main(k):
     assert 0<=k<NP
     dest=OUT/f'checkpoints/traced_{k}.json'
     if dest.exists():print('Validated trace checkpoint',k);return
     start=time.time();n0=book()['new_rays'];ev=EVALUATOR;mesh=midpoint_mesh();q=OUT/f'checkpoints/midpoint_trace_vertices_{k}.npz'
     if q.exists():p=load(q)
     else:p={**mesh,**midpoint_observe(ev,mesh['xy'],k,f'base polygon and wall classification midpoint{k} vertices')};save(q,**p)
     assert np.all(p['classes'][:DISK_ANGLES]==0),('Artificial boundary hits wall; expand band',k)
     for c,name in enumerate(REACH_CLASSES):
      q=OUT/f'arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz'
      if not q.exists():save(q,**clip_class(ev,p,k,c))
     mesh=plane_mesh();m={'plane_xy':mesh['xy'],'plane_tri':mesh['tri'],'plane_owner':mesh['owner']}
     def side(sg):
      path=OUT/f'checkpoints/plane_vertices_{k}_{sg}.npz'
      if path.exists():p=load(path)
      else:
       z=trace_cached(ev,plane_seeds(mesh,k),sg*.5*HOP,f'base polygon and wall classification plane_seeded_wall_rows plane{k} side{sg} vertices');p={f:z[f] for f in ['hit','wall','wall_length']};save(path,**p)
      q=OUT/f'arrays/{PLANE_WALL_STEM}_{k}_{sg}.npz'
      if not q.exists():save(q,**clip_plane_seeded_wall(ev,m,k,sg,p))
     for sg in [-1,1]:side(sg)
     path=OUT/f'checkpoints/wall_quad_{k}.npz'
     if not path.exists():
      q,ww,sg,delta=wall_quad_requests(k,ev);z=trace_cached(ev,q,delta,f'base polygon and wall classification wall quad strip{k}');z['weight']=np.array(ww);z['sign']=sg;save(path,**z)
     rec=dict(complete=True,plane=k,new_rays=plane_charged(k,'base polygon and wall classification'),seconds=time.time()-start,N=N,midpoint_class_counts=np.bincount(load(OUT/f'checkpoints/midpoint_trace_vertices_{k}.npz')['classes'],minlength=4).tolist());dump(dest,rec);log(f'plane {k} trace checkpoint complete: {rec}')


    def wall_quad_requests(k,ev):
     x,w=roots_legendre(3);x=(x+1)/2;w/=2;nt=3*N//2;q=[];ww=[]
     for i in range(nt):
      for j in range(6):
       for a,wa in zip(x,w):
        for b,wb in zip(x,w):q.append([1-1e-10,(i+a)*TAU/nt,(k+.5+(j+b)/6)*HOP]);ww.append(TAU/nt*HOP/6*wa*wb)
     q=np.asarray(q);met=gu.metric(ev,q,False);sg=-np.sign(met['B'][:,0]/met['B'][:,2]);delta=(k+.5+(sg>0))*HOP-q[:,2]
     return q,ww,sg,delta

    return {'main': main, 'wall_quad_requests': wall_quad_requests}
