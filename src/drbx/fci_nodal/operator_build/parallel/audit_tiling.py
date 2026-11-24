"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np


def bind(context):
    N = context.N
    NP = context.NP
    HOP = context.HOP
    COPIES = context.copies
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    load = context.load
    dump = context.dump
    log = context.log
    relative = context.relative
    seeds = context.seeds
    moment = context.moment
    metric_chunks = context.metric_chunks

    def main(discover=False):
     ev=EVALUATOR;rows=[];worst=0.;remaining=[];classes={name:0. for name in REACH_CLASSES}
     for k in range(NP):
      p=load(ROOT/(f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz'));parts=np.zeros((len(p['tri']),4));areas=[];points=[]
      for t in p['xy'][p['tri']]:a,b,pt=moment(t,np.eye(3),'half');areas.append(a);points.append(pt)
      areas=np.array(areas);q=seeds(np.array(points),(k+1)*HOP);m=metric_chunks(ev,q);density=abs(m['J']*m['B'][:,2])/q[:,0]
      for c,name in enumerate(REACH_CLASSES):
       z=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz');aa=[];qq=[]
       for t,parent in zip(z['xy'],z['parent']):a,b,pt=moment(t,np.eye(3),'half');parts[parent,c]+=a;aa.append(a);qq.append(pt)
       if qq:
        q=seeds(np.array(qq),(k+1)*HOP);m=metric_chunks(ev,q);classes[name]+=COPIES*float(np.sum(np.array(aa)*abs(m['J']*m['B'][:,2])/q[:,0]))
      gaps=areas-parts.sum(1);relative=abs(gaps)/areas;worst=max(worst,float(relative.max()));gap_centroid_trace_audit=load(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz');changed=np.isin(np.arange(len(areas)),gap_centroid_trace_audit['parent'])
      for j in np.flatnonzero(relative>1e-12):remaining.append(dict(plane=k,parent=int(j),changed=bool(changed[j]),gap_area=float(gaps[j]),relative=float(relative[j]),midpoint_flux_proxy=float(gaps[j]*density[j])))
      rows.append(dict(plane=k,parents=len(areas),changed_parents=int(changed.sum()),max_relative=float(relative.max()),changed_max_relative=float(relative[changed].max()),above_relative_1e12=int((relative>1e-12).sum()),net_gap_area=float(gaps.sum()),absolute_gap_area=float(abs(gaps).sum()),net_midpoint_gap_flux_proxy_full=COPIES*float(gaps@density)))
     dump(OUT/f'tiling_N{N}.json',dict(complete=True,N=N,scope='base polygon and wall classification full source band',rows=rows,parents=sum(r['parents'] for r in rows),changed_parents=sum(r['changed_parents'] for r in rows),max_relative=worst,above_relative_1e12=len(remaining),remaining=remaining,class_source_flux_full=classes,new_rays=0));assert discover or worst<=1e-12,('Untreated tiny parent gap; return evidence, no unapproved repair',worst,remaining);log(f'N{N} all-source-parent tiling audit: maximum {worst:.3g}, {len(remaining)} above 1e-12; no rays')


    return main
