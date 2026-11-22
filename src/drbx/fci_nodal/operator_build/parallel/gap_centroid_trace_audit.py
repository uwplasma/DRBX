"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json
import time


def bind(context):
    N = context.N
    TAU = context.TAU
    HOP = context.HOP
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    book = context.book
    trace_sides = context.trace_sides
    plane_charged = context.plane_charged
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    key = context.key
    sample = context.sample
    seeds = context.seeds
    moment = context.moment
    classify = context.classify
    contains = context.contains
    inverse = context.inverse

    def main(k):
     dest=OUT/f'checkpoints/gap_centroid_trace_audit_N{N}_{k}.json'
     if dest.exists():return
     start=time.time();ev=EVALUATOR;diag=json.loads((ROOT/f'partition_audit/partition_N{N}.json').read_text());ids=np.array([r['parent'] for r in diag['gap_parents'] if r['strip']==k],int);p=load(ROOT/(f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz'));tri=p['xy'][p['tri'][ids]];barys=[];points=[]
     for t in tri:
      area,bary,point=moment(t,np.eye(3),'half');barys.append(bary);points.append(point)
     bary=np.asarray(barys);q=seeds(np.asarray(points),(k+1)*HOP);before=book()['new_rays'];zz=trace_sides(ev,q,[-HOP,HOP],[f'repaired polygon and merged wall rows gap_centroid_trace_audit plane{k} side{sg}' for sg in [-1,1]]);tau=np.stack([np.where(z['hit'],abs((z['wall'][:,2]-q[:,2]+np.pi)%TAU-np.pi)/HOP,1.) for z in zz],1);actual=classify(tau>=.5);truth=np.stack([sample(z,.5)[:,:2] for z in zz],1)
     ref=inverse(q);prediction_bary=[]
     for xx,tt in zip(ref,tri):
      v=np.linalg.solve(np.column_stack([tt[1]-tt[0],tt[2]-tt[0]]),xx-tt[0]);prediction_bary.append(np.r_[1-v.sum(),v])
     prediction_bary=np.asarray(prediction_bary);assert prediction_bary.min()>-1e-10
     wh=p['wallhit'];vt=np.where(np.isfinite(wh[:,:,2]),abs((wh[:,:,2]-(k+1)*HOP+np.pi)%TAU-np.pi)/HOP,1.);predtau=np.einsum('ij,ijk->ik',prediction_bary,vt[p['tri'][ids]]);predcls=classify(predtau>=.5);predmap=np.einsum('ij,ijks->iks',prediction_bary,p['maps'][p['tri'][ids]]);original_piece=np.zeros(len(ids),bool);pred_from_class=np.full(len(ids),-1,int)
     # Use the existing class-piece affine map wherever a polygon contains the inverse-mapped physical barycentre.
     # A gap has no existing class prediction; retain the explicitly reported parent time interpolant there.
     for c,name in enumerate(REACH_CLASSES):
      z=load(ROOT/(f'base_polygon/arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz'))
      for j,owner in enumerate(ids):
       x=ref[j]
       for ti in np.flatnonzero(z['parent']==owner):
        bc=contains(x,z['xy'][ti])
        if bc is not None:
         original_piece[j]=True;predcls[j]=c;pred_from_class[j]=c;predmap[j]=np.einsum('j,jks->ks',bc,z['maps'][ti]);break
     err=np.full((len(ids),2),np.nan)
     for reach in [0,1]:
      good=tau[:,reach]>=.5;land=truth[good,reach];r=np.linalg.norm(land,axis=1);unit=land/r[:,None];d=predmap[good,reach]-land;err[good,reach]=np.sqrt((np.sum(d*unit,axis=1)*N)**2+(gu.cross(unit,d)/(r*TAU/N))**2)
     mx=np.max(np.where(np.isfinite(err),err,0),axis=1);mismatch=predcls!=actual;refine=mismatch|(mx>.005)
     save(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz',parent=ids,seed=q,bary=bary,prediction_reference_coordinates=ref,prediction_bary=prediction_bary,actual_class=actual,predicted_class=predcls,original_piece=original_piece,landing_error_cells=err,subdivide=refine,truth_maps=truth,**{f'{side}_{key}':z[key] for side,z in enumerate(zz) for key in ['wall','wall_length','hit','delta']})
     rows=[dict(parent=int(i),actual_class=int(ac),predicted_class=int(pc),prediction_in_existing_piece=bool(op),class_disagrees=bool(mi),max_reaching_landing_error_cells=float(e),subdivide=bool(rr)) for i,ac,pc,op,mi,e,rr in zip(ids,actual,predcls,original_piece,mismatch,mx,refine)]
     rec=dict(complete=True,actual_inverse_coordinates_used=True,N=N,plane=k,parents=len(ids),gap_centroid_rays_before_reuse=2*len(ids),new_rays=plane_charged(k,'repaired polygon and merged wall rows gap_centroid_trace_audit'),cumulative_ledger=book()['new_rays'],class_disagreements=int(mismatch.sum()),above_landing_criterion=int((mx>.005).sum()),subdivide_count=int(refine.sum()),predictions_in_gap_using_parent_time_interpolant=int((~original_piece).sum()),rows=rows,seconds=time.time()-start,prediction='Saved class-piece affine map when containing the inverse-mapped physical area-centroid; in old gaps use parent interpolated reach times/maps. Seed is the exact physical area centroid.');dump(dest,rec);log(f'N{N} gap_centroid_trace_audit plane{k}: {len(ids)} parents, {refine.sum()} subdivision flags; ledger {book()["new_rays"]}/{book()["cap"]}')


    return main
