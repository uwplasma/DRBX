"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json


def bind(context):
    N = context.N
    NP = context.NP
    TAU = context.TAU
    COPIES = context.copies
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    metric_chunks = context.metric_chunks
    contains = context.contains
    BITS = context.BITS

    def stats(x):
     x=np.asarray(x);x=x[np.isfinite(x)]
     return dict(count=len(x),rms=float(np.sqrt(np.mean(x*x))) if len(x) else None,quantile_levels=[0,.01,.05,.25,.5,.75,.95,.99,1],quantiles=np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist() if len(x) else [])


    def main():
     result=[];all_before=[];all_after=[];all_parent_before=[];all_parent_after=[];ev=EVALUATOR
     for k in range(NP):
      p=load(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz');ids=p['parent'];xy=p['prediction_reference_coordinates'];pred=np.full(len(ids),-1,int);maps=np.full((len(ids),2,2),np.nan)
      for c,name in enumerate(REACH_CLASSES):
       z=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz')
       for j,parent in enumerate(ids):
        for t in np.flatnonzero(z['parent']==parent):
         b=contains(xy[j],z['xy'][t])
         if b is not None:pred[j]=c;maps[j]=np.einsum('i,isk->sk',b,z['maps'][t]);break
      assert (pred>=0).all();err=np.full((len(ids),2),np.nan)
      for s in range(2):
       good=BITS[p['actual_class'],s]&BITS[pred,s];land=p['truth_maps'][good,s];r=np.linalg.norm(land,axis=1);unit=land/r[:,None];d=maps[good,s]-land;err[good,s]=np.sqrt((np.sum(d*unit,axis=1)*N)**2+(gu.cross(unit,d)/(r*TAU/N))**2)
      before=p['landing_error_cells'];mx=np.max(np.where(np.isfinite(err),err,0),axis=1);oldmx=np.max(np.where(np.isfinite(before),before,0),axis=1);mismatch=pred!=p['actual_class'];area=np.array([r['area'] for r in json.loads((OUT/f'checkpoints/joint_partition_N{N}_{k}.json').read_text())['parents']]);met=metric_chunks(ev,p['seed']);flux=area*abs(met['J']*met['B'][:,2])/p['seed'][:,0]
      save(OUT/f'arrays/repaired_tiling_audit_N{N}_{k}.npz',parent=ids,before=before,after=err,actual_class=p['actual_class'],predicted_class=pred,parent_flux=flux,above=mx>.005,class_disagreement=mismatch)
      for j,parent in enumerate(ids):result.append(dict(plane=k,parent=int(parent),subdivided=bool(p['subdivide'][j]),actual_class=int(p['actual_class'][j]),predicted_class=int(pred[j]),class_disagrees=bool(mismatch[j]),before_max=float(oldmx[j]),after_max_comparable_reaching=float(mx[j]),above_after=bool(mx[j]>.005),parent_source_flux=float(flux[j])))
      all_before.extend(before.ravel());all_after.extend(err.ravel());all_parent_before.extend(oldmx);all_parent_after.extend(mx)
     remaining=[r for r in result if r['above_after']];mismatch=[r for r in result if r['class_disagrees']]
     dump(OUT/f'repaired_tiling_audit_N{N}.json',dict(complete=True,N=N,metric='Pointwise landing error in radial/tangential cell widths; before/after at the same independently traced physical parent centroid. After errors use arms reaching in both measured and predicted classes; class disagreements are separately retained, never treated as zero error success.',parents=len(result),before=stats(all_before),after=stats(all_after),before_parent_max=stats(all_parent_before),after_parent_max=stats(all_parent_after),above_after=len(remaining),above_after_full_torus_parent_flux=COPIES*sum(r['parent_source_flux'] for r in remaining),class_disagreements_after=len(mismatch),class_disagreement_full_torus_parent_flux=COPIES*sum(r['parent_source_flux'] for r in mismatch),remaining_subdivided_above=sum(r['subdivided'] for r in remaining),rows=result,new_rays=0))
     log(f'N{N} post-subdivision gap_centroid_trace_audit: {len(remaining)} parents above criterion, {len(mismatch)} class disagreements; no new rays')


    return {'stats': stats, 'main': main}
