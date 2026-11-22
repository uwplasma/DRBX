"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json


def bind(context):
    N = context.N
    NP = context.NP
    TAU = context.TAU
    HOP = context.HOP
    BISECT = context.BISECT
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    Lookup = context.Lookup
    book = context.book
    plane_charged = context.plane_charged
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    key = context.key
    edges = context.edges
    moment = context.moment
    classify = context.classify
    observe = context.observe
    cut = context.cut
    field_at = context.field_at
    make_port = context.make_port
    BITS = context.BITS
    FIELDS = context.FIELDS

    def inputs(k,projection=False):
     p=load(ROOT/(f'base_polygon/checkpoints/midpoint_trace_vertices_{k}.npz'))
     gap_centroid_trace_audit=load(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz');mp=OUT/f'arrays/midpoints_N{N}_{k}.npz'
     if mp.exists():m=load(mp)
     else:
      assert projection;tt=p['tri'][gap_centroid_trace_audit['parent'][gap_centroid_trace_audit['subdivide']]];ee=edges(tt);xy=p['xy'][ee].mean(1);cache=Lookup();wh=p['wallhit'];vt=np.where(np.isfinite(wh[:,:,2]),abs((wh[:,:,2]-(k+1)*HOP+np.pi)%TAU-np.pi)/HOP,1.);tau=vt[ee].mean(1)
      for i,x in enumerate(xy):
       _,_,observed=cache.observation(x,k)
       for s in range(2):
        if observed[s] is not None:tau[i,s]=observed[s]
      m=dict(edges=ee,xy=xy,classes=classify(tau>=.5),**{f:np.zeros((len(ee),)+p[f].shape[1:]) for f in FIELDS})
     nv=len(p['xy']);mid={tuple(e):nv+i for i,e in enumerate(m['edges'])}
     v={f:np.concatenate([p[f],m[f]]) for f in ['xy','classes']+FIELDS};tri=[];parents=[]
     for par,sub in zip(gap_centroid_trace_audit['parent'],gap_centroid_trace_audit['subdivide']):
      a,b,c=p['tri'][par]
      if sub:
       ab,bc,ca=[mid[tuple(sorted(e))] for e in [(a,b),(b,c),(c,a)]];tt=[[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]
      else:tt=[[a,b,c]]
      tri.extend(tt);parents.extend([par]*len(tt))
     tri=np.array(tri);ee=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
     jobs=[]
     for e in ee:
      for s in range(2):
       if BITS[v['classes'][e[0]],s]!=BITS[v['classes'][e[1]],s]:jobs.append([*e,s])
     return p,gap_centroid_trace_audit,v,tri,np.array(parents),np.array(jobs,dtype=int).reshape(-1,3)


    def project():
     cache=Lookup();rows=[];missing_first=set();tails=0;midpoint_missing=set();conditional=False
     for k in range(NP):
      p,gap_centroid_trace_audit,v,tri,parents,jobs=inputs(k,True);known=0;tail=0;first=set();conditional|=not (OUT/f'arrays/midpoints_N{N}_{k}.npz').exists()
      for x in v['xy'][len(p['xy']):]:
       q,z,_=cache.observation(x,k)
       for sg,zz in zip([-1,1],z):
        if zz is None:midpoint_missing.add(key(q,sg*HOP))
      for a,b,s in jobs:
       left=v['xy'][a].copy();right=v['xy'][b].copy();bit=BITS[v['classes'][a],s]
       for it in range(BISECT):
        x=(left+right)/2;q,z,tau=cache.observation(x,k)
        if any(t is None for t in tau):
         for d,zz in zip([-1,1],z):
          if zz is None:first.add(key(q,d*HOP))
         tail+=2*(BISECT-it-1);break
        known+=1
        if (tau[s]>=.5)==bit:left=x
        else:right=x
      # Junctions use intersections and class-side affine limits, not new rays.
      missing_first|=first;tails+=tail
      rows.append(dict(plane=k,parents=len(gap_centroid_trace_audit['parent']),children=len(tri),mixed_indicator_edges=len(jobs),cached_iterations=known,first_missing_rays=len(first),remaining_tail_upper_bound=tail))
     ledger=book();extra=len(missing_first|midpoint_missing)+tails;total=ledger['new_rays']+extra
     r=dict(complete=True,N=N,decision='polygon_gap_decisions.md',ledger_before=ledger['new_rays'],cap=ledger['cap'],new_ray_upper_bound=extra,projected_cumulative=total,continue_authorized=total<=ledger['cap'],units=rows,method='Two shared reaching-indicator cut segments per child. Cache-exact known binary-bisection prefix, unique first misses, two rays per remaining iteration per mixed indicator edge (conservative duplicate allowance). Shared junction coordinates and geometry use class-side affine limits of traced vertices/bisection endpoints; no junction rays. No further parent refinement.')
     r.update(midpoints_projected=conditional,midpoint_cache_misses=len(midpoint_missing),limitation='Before midpoint measurement, missing midpoint classes are predicted from parent reaching times and the bound is conditional on those labels. Re-project with measured midpoint classes before any bisections.' if conditional else 'Bound on remaining bisection rays for the measured child-vertex labels; includes conservative duplicate tails.')
     dump(OUT/f'joint_projection_N{N}.json',r)
     if not r['continue_authorized']:dump(OUT/f'budget_stop_N{N}.json',dict(reason='joint partition projected cumulative exceeds cap',projection=r))
     log(f'N{N} joint projection {total}/{ledger["cap"]}; new ray upper bound {extra}; no rays')


    def bisect(k):
     dest=OUT/f'checkpoints/joint_bisection_N{N}_{k}.json'
     if dest.exists():return
     projection=json.loads((OUT/f'joint_projection_N{N}.json').read_text());assert projection['continue_authorized']
     p,gap_centroid_trace_audit,v,tri,parents,jobs=inputs(k);ev=EVALUATOR;a,b,s=jobs.T;left=v['xy'][a].copy();right=v['xy'][b].copy();lv={f:v[f][a].copy() for f in ['classes']+FIELDS};rv={f:v[f][b].copy() for f in ['classes']+FIELDS};bit=BITS[v['classes'][a],s];start=book()['new_rays']
     for it in range(BISECT):
      cp=OUT/f'checkpoints/joint_iteration_N{N}_{k}_{it}.npz'
      if cp.exists():
       z=load(cp);left=z['left'];right=z['right'];lv={f:z['left_'+f] for f in lv};rv={f:z['right_'+f] for f in rv};continue
      if len(jobs):
       mid=(left+right)/2;obs=observe(ev,mid,k,f'repaired polygon and merged wall rows joint plane{k} iteration{it}');same=BITS[obs['classes'],s]==bit
       left[same]=mid[same];right[~same]=mid[~same]
       for f in lv:lv[f][same]=obs[f][same];rv[f][~same]=obs[f][~same]
      save(cp,left=left,right=right,**{'left_'+f:v for f,v in lv.items()},**{'right_'+f:v for f,v in rv.items()})
     save(OUT/f'arrays/joint_edges_N{N}_{k}.npz',jobs=jobs,left=left,right=right,**{'left_'+f:v for f,v in lv.items()},**{'right_'+f:v for f,v in rv.items()})
     dump(dest,dict(complete=True,N=N,plane=k,jobs=len(jobs),iterations=BISECT,new_rays=plane_charged(k,'repaired polygon and merged wall rows joint'),cumulative_ledger=book()['new_rays']));log(f'N{N} joint plane{k} bisections complete; ledger {book()["new_rays"]}/{book()["cap"]}')


    def partition(k):
     dest=OUT/f'checkpoints/joint_partition_N{N}_{k}.json'
     if dest.exists():return
     p,gap_centroid_trace_audit,v,tri,parents,jobs=inputs(k);e=load(OUT/f'arrays/joint_edges_N{N}_{k}.npz');assert np.array_equal(jobs,e['jobs']);index={tuple(j):i for i,j in enumerate(jobs)};out=[{f:[] for f in ['xy']+FIELDS+['parent','child']} for c in range(4)];checks=[];junctions=[]
     for ti,(ids,parent) in enumerate(zip(tri,parents)):
      xx=v['xy'][ids];local=np.array([[0.,0.],[1.,0.],[0.,1.]]);local_id={int(a):j for j,a in enumerate(ids)};bits=BITS[v['classes'][ids]];lines={};direction={}
      port=make_port(jobs,v,e,local,local_id)
      for s in range(2):
       ports=[]
       for a,b in zip(ids,np.roll(ids,-1)):
        if BITS[v['classes'][a],s]!=BITS[v['classes'][b],s]:
         j=index[(*sorted([a,b]),s)];ports.append((j,port(j)))
       if ports:
        assert len(ports)==2;aa,bb=[q[1] for q in ports];orient=np.sign(gu.cross(bb-aa,local[np.flatnonzero(bits[:,s])[0]]-aa));assert orient!=0;lines[s]=(aa,bb,orient)
       for bit in [False,True]:
        poly=[];fv=[]
        for a,b in zip(ids,np.roll(ids,-1)):
         if BITS[v['classes'][a],s]==bit:
          poly.append(local[local_id[a]]);fv.append((v['maps'][a,s],v['arcs'][a,s]) if bit else (v['wallhit'][a,s],v['hitell'][a,s]))
         if BITS[v['classes'][a],s]!=BITS[v['classes'][b],s]:
          j=index[(*sorted([a,b]),s)];which='left' if BITS[e['left_classes'][j],s]==bit else 'right';poly.append(port(j));fv.append((e[which+'_maps'][j,s],e[which+'_arcs'][j,s]) if bit else (e[which+'_wallhit'][j,s],e[which+'_hitell'][j,s]))
        if len(poly)>=3:
         val=np.array([np.r_[a,b] for a,b in fv]);assert np.isfinite(val).all()
         if not bit:val[:,1:3]=val[0,1:3]+(val[:,1:3]-val[0,1:3]+np.pi)%TAU-np.pi
         direction[s,bit]=(np.array(poly),val)
      if len(lines)==2:
       a,b,_=lines[0];c,d,_=lines[1];mat=np.column_stack([b-a,-(d-c)])
       if abs(np.linalg.det(mat))>1e-20:
        tt=np.linalg.solve(mat,c-a)
        if np.all(tt>0) and np.all(tt<1):junctions.append(dict(parent=int(parent),child=ti,xy=(np.r_[1-(a+tt[0]*(b-a)).sum(),a+tt[0]*(b-a)]@xx).tolist(),degree=4))
      source_area=moment(xx,np.eye(3),'half')[0];total=0.;classareas=[]
      for c,cb in enumerate(BITS):
       poly=local.copy()
       for s in range(2):
        if s in lines:poly=cut(poly,lines[s],cb[s])
        elif bits[0,s]!=cb[s]:poly=np.empty((0,2));break
       area=0.
       for j in range(1,len(poly)-1):
        local_t=poly[[0,j,j+1]];t=np.c_[1-local_t.sum(1),local_t]@xx
        if gu.cross(t[1]-t[0],t[2]-t[0])<=0:continue
        a=moment(t,np.eye(3),'half')[0];area+=a;maps=np.full((3,2,2),np.nan);arcs=np.full((3,2),np.nan);wh=np.full((3,2,3),np.nan);hl=np.full((3,2),np.nan)
        for s in range(2):
         dp,val=direction[s,bool(cb[s])];vv=np.array([field_at(x,dp,val) for x in local_t])
         if cb[s]:maps[:,s]=vv[:,:2];arcs[:,s]=vv[:,2]
         else:wh[:,s]=vv[:,:3];wh[:,s,1:]%=TAU;hl[:,s]=vv[:,3]
        for f,value in dict(xy=t,maps=maps,arcs=arcs,wallhit=wh,hitell=hl,parent=parent,child=ti).items():out[c][f].append(value)
       classareas.append(area);total+=area
      error=abs(total-source_area)/source_area;assert error<1e-11,(k,ti,error);checks.append(dict(parent=int(parent),child=ti,area=source_area,class_area=classareas,relative_error=error))
     identity=[]
     shapes={'xy':(3,2),'maps':(3,2,2),'arcs':(3,2),'wallhit':(3,2,3),'hitell':(3,2),'parent':(),'child':()}
     for c,name in enumerate(REACH_CLASSES):
      old=load(ROOT/(f'base_polygon/arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz'));keep=~np.isin(old['parent'],gap_centroid_trace_audit['parent']);new={f:np.asarray(a).reshape((-1,)+shapes[f]) for f,a in out[c].items()};result={f:np.concatenate([old[f][keep],new[f]]) for f in ['xy']+FIELDS+['parent']};result['child']=np.r_[np.full(keep.sum(),-1),new['child']].astype(int);result['parent']=result['parent'].astype(int)
      assert all(np.array_equal(result[f][:keep.sum()],old[f][keep],equal_nan=True) for f in ['xy']+FIELDS+['parent']);save(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz',**result)
      identity.append(dict(cls=name,untouched_triangles=int(keep.sum()),removed_triangles=int((~keep).sum()),new_triangles=len(new['xy']),untouched_bitwise=True))
     parentchecks=[]
     for par in gap_centroid_trace_audit['parent']:
      rr=[r for r in checks if r['parent']==par];area=moment(p['xy'][p['tri'][par]],np.eye(3),'half')[0];sums=np.array([r['class_area'] for r in rr]).sum(0);err=abs(sums.sum()-area)/area;assert err<1e-12
      parentchecks.append(dict(parent=int(par),area=area,class_area=sums.tolist(),relative_error=err))
     dump(dest,dict(complete=True,N=N,plane=k,parents=parentchecks,children=checks,parent_max_relative=max(r['relative_error'] for r in parentchecks),child_max_relative=max(r['relative_error'] for r in checks),identity=identity,junctions=junctions,three_way_junctions=0,four_way_junctions=len(junctions),geometry_convention='One common midpoint of each final production bisection bracket; one-sided traced limits for each reaching/stopping direction, interpolated on that direction polygon. Interior junction is exact intersection of both straight cut segments. No independent class erosion or new junction rays.'))
     log(f'N{N} joint partition plane{k}: {len(checks)} children tile, max relative {max(r["relative_error"] for r in checks):.3g}; untouched polygons bitwise; no rays')


    return {'inputs': inputs, 'project': project, 'bisect': bisect, 'partition': partition}
