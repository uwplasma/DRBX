"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import time
from scipy.spatial import ConvexHull


def bind(context, remainder):
    N = context.N
    TAU = context.TAU
    HOP = context.HOP
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    book = context.book
    trace_sides = context.trace_sides
    plane_charged = context.plane_charged
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    sample = context.sample
    seeds = context.seeds
    edges = context.edges
    transform = context.transform
    moment = context.moment
    classify = context.classify

    def main(k):
     dest=OUT/f'checkpoints/children_N{N}_{k}.json'
     if dest.exists():return
     start=time.time();ev=EVALUATOR;p=load(ROOT/(f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz'));g=load(ROOT/(f'base_polygon/arrays/{REACH_OUTPUT_STEMS[REACH_CLASSES[0]][0]}_{k}.npz'));gap_centroid_trace_audit=load(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz');ids=gap_centroid_trace_audit['parent'][gap_centroid_trace_audit['subdivide']];edges=np.unique(np.sort(np.concatenate([p['tri'][ids][:,[0,1]],p['tri'][ids][:,[1,2]],p['tri'][ids][:,[2,0]]]),axis=1),axis=0);ref=p['xy'][edges].mean(1);xy,_=transform(ref,'half');q=seeds(xy,(k+1)*HOP);before=book()['new_rays'];zz=trace_sides(ev,q,[-HOP,HOP],[f'repaired polygon and merged wall rows child midpoints plane{k} side{sg}' for sg in [-1,1]]);tau=np.stack([np.where(z['hit'],abs((z['wall'][:,2]-q[:,2]+np.pi)%TAU-np.pi)/HOP,1.) for z in zz],1);cls=classify(tau>=.5)
     save(OUT/f'arrays/midpoints_N{N}_{k}.npz',edges=edges,xy=ref,seed=q,classes=cls,maps=np.stack([sample(z,.5)[:,:2] for z in zz],1),arcs=np.stack([sample(z,.5)[:,2] for z in zz],1),wallhit=np.stack([z['wall'] for z in zz],1),hitell=np.stack([z['wall_length'] for z in zz],1),hit=np.stack([z['hit'] for z in zz],1))
     errors=[];rows=[];rpolys=[];rparent=[];offset=[0]
     for ii,parent in enumerate(gap_centroid_trace_audit['parent']):
      t=p['xy'][p['tri'][parent]];gt=g['xy'][g['parent']==parent];gp=np.unique(np.concatenate(gt),axis=0) if len(gt) else np.empty((0,2));gp=gp[ConvexHull(gp).vertices] if len(gp)>2 else np.empty((0,2));ab,bc,ca=(t+np.roll(t,-1,axis=0))/2;children=[np.array(v) for v in [[t[0],ab,ca],[ab,t[1],bc],[ca,bc,t[2]],[ab,bc,ca]]] if gap_centroid_trace_audit['subdivide'][ii] else [t];total=moment(t,np.eye(3),'half')[0];garea=sum(moment(v,np.eye(3),'half')[0] for v in gt);r_area=0
      for child in children:
       for poly in remainder(child,gp):
        r_area+=moment(poly[:3],np.eye(3),'half')[0] if len(poly)==3 else sum(moment(poly[[0,j,j+1]],np.eye(3),'half')[0] for j in range(1,len(poly)-1));rpolys.extend(poly);offset.append(len(rpolys));rparent.append(int(parent))
      error=abs(garea+r_area-total)/total;errors.append(error);rows.append(dict(parent=int(parent),subdivided=bool(gap_centroid_trace_audit['subdivide'][ii]),both_planes_reached_area=garea,remainder_area=r_area,parent_area=total,relative_reach_plus_repair_error=error))
     assert max(errors)<1e-12,max(errors)
     save(OUT/f'arrays/remainders_N{N}_{k}.npz',xy=np.array(rpolys),offset=np.array(offset),parent=np.array(rparent));rec=dict(complete=True,N=N,plane=k,subdivided_parents=len(ids),unique_midpoints=len(edges),new_rays=plane_charged(k,'repaired polygon and merged wall rows child midpoints'),G_plus_R_tiling_max=max(errors),G_preserved=True,remainder_classified=False,parents=rows,seconds=time.time()-start);dump(dest,rec);log(f'N{N} child midpoint plane{k} complete; G+R tiling max {max(errors):.3g}; ready for authorized four-class joint partition')


    return main
