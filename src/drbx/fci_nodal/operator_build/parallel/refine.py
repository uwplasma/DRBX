"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import PLANE_WALL_STEM
import numpy as np
import json
import time


def bind(context, plane_mesh, clip_plane_seeded_wall):
    NP = context.NP
    HOP = context.HOP
    OUT = context.out
    HISTORY = context.history
    EVALUATOR = context.evaluator
    book = context.book
    trace_cached = context.trace_cached
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    seeds = context.seeds
    transform = context.transform

    def main(plane=None):
     rec=json.loads((OUT/'balance.json').read_text());requested=rec['unresolved_first_period'];done=OUT/(f'refinement_{plane}.json' if plane is not None else 'refinement.json')
     if done.exists():return
     rows=[];planes=[]
     if requested:
      HISTORY.mkdir(parents=True,exist_ok=True)
      ev=EVALUATOR;base=plane_mesh();meshbase={'plane_xy':base['xy'],'plane_tri':base['tri'],'plane_owner':base['owner']}
      for k in ([plane] if plane is not None else range(NP)):
       changed=False
       for sg in [-1,1]:
        cells={r['cell'] for r in requested if r['plane']==k and r['side']==sg}
        if not cells:continue
        receipt=OUT/f'checkpoints/refine_{k}_{sg}.json'
        if receipt.exists():
         saved=json.loads(receipt.read_text());assert saved['complete'];rows.append(saved);changed=True;continue
        ts=base['xy'][base['tri'][np.isin(base['owner'],list(cells))]];ow=base['owner'][np.isin(base['owner'],list(cells))];tri=[];owners=[]
        for t,j in zip(ts,ow):
         a,b,c=t;ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2;tri.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]);owners.extend([j]*4)
        if not tri:raise RuntimeError(('refinement owner missing from all-band mesh',k,sg,cells))
        tri=np.asarray(tri);xy,iv=np.unique(tri.reshape(-1,2),axis=0,return_inverse=True);m=dict(plane_xy=xy,plane_tri=iv.reshape(-1,3),plane_owner=np.array(owners));actual,_=transform(xy,'plane');calls0=len(book()['calls']);z=trace_cached(ev,seeds(actual,(k+.5)*HOP),sg*.5*HOP,f'base polygon and wall classification refine plane{k} side{sg} vertices');p=clip_plane_seeded_wall(ev,m,k,sg,z);old=load(OUT/f'arrays/{PLANE_WALL_STEM}_{k}_{sg}.npz');keep=~np.isin(old['owner'],list(cells));fan_keep=keep[old['fanrow']];row_map=np.cumsum(keep)-1;out={}
        for f in ['fans','wallhit','hitell']:out[f]=np.concatenate([old[f][fan_keep],p[f]])
        for f in ['owner','parent']:out[f]=np.concatenate([old[f][keep],p[f]])
        out['fanrow']=np.r_[row_map[old['fanrow'][fan_keep]],p['fanrow']+int(keep.sum())];out['boundary_edges']=int(old['boundary_edges'])+int(p['boundary_edges']);archive=HISTORY/f'base_polygon_before_refine_{k}_{sg}.npz';(OUT/f'arrays/{PLANE_WALL_STEM}_{k}_{sg}.npz').rename(archive);save(OUT/f'arrays/{PLANE_WALL_STEM}_{k}_{sg}.npz',**out);unit=dict(complete=True,plane=k,side=sg,cells=sorted(cells),found_rows=len(p['owner']),new_rays=sum(c['rays'] for c in book()['calls'][calls0:] if c['label']==f'base polygon and wall classification refine plane{k} side{sg} vertices' or c['label'].startswith(f'base polygon and wall classification plane_seeded_wall_rows plane{k} side{sg} bisect')));dump(receipt,unit);rows.append(unit);changed=True
       if changed:
        planes.append(k)
        for stem,folder,ext in [('geometry','checkpoints','json'),('geometry','arrays','npz'),('interpolation','checkpoints','json')]:
         p=OUT/f'{folder}/{stem}_{k}.{ext}'
         if p.exists():p.rename(HISTORY/f'base_polygon_pre_refine_{stem}_{k}_{time.time_ns()}.{ext}')
     dump(done,dict(complete=True,criterion='positive residual/Phi>1% and no plane_seeded_wall_rows; one level only',units=rows,rebuild_planes=planes,ledger=book()['new_rays']));log(f'one-level refinement complete; {len(requested)} first-period cellsides; ledger {book()["new_rays"]}/{book()["cap"]}')


    return main
