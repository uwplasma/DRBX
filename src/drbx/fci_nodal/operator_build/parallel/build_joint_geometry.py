"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import MIDPOINT_WALL_CLASSES, REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import json


def bind(context):
    N = context.N
    TAU = context.TAU
    HOP = context.HOP
    ROOT = context.root
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    Lookup = context.Lookup
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    arc_tiles = context.arc_tiles
    metric_chunks = context.metric_chunks
    overlaps = context.overlaps

    def classpath(name,k):return ROOT/(f'base_polygon/arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz')


    def main(k):
     dest=OUT/f'checkpoints/geometry_N{N}_{k}.json'
     if dest.exists():return
     ev=EVALUATOR;tiles=arc_tiles();cache=Lookup();cells=gu.build()[1];gap_centroid_trace_audit=load(OUT/f'arrays/gap_centroid_trace_audit_N{N}_{k}.npz');ids=gap_centroid_trace_audit['parent'];validation=[];faceout=[];faceflux=[];walls=[];wp=[];ws=[];flags=[];overrides=[];changed=[]
     old=load(ROOT/f'base_polygon/arrays/geometry_{k}.npz');oldface=old['faces'];oldw=old;oldvr=json.loads((ROOT/f'base_polygon/checkpoints/interpolation_{k}.json').read_text());oldkind=old['wall_kind'];oldside=old['wall_side'];oldwp=old['wall_flux']
     facekeep=np.ones(len(oldface),bool);wallkeep=np.ones(len(oldw['wall']),bool);maxold=0.
     for c,name in enumerate(REACH_CLASSES[:3]):
      changed_ids=ids
      extra=OUT/f'supplemental_tiny_gaps_N{N}.json'
      if c>0 and extra.exists():changed_ids=np.r_[ids,[r['parent'] for r in json.loads(extra.read_text())['parents'] if r['plane']==k]]
      oldp=load(classpath(name,k));records=overlaps(oldp,c,tiles);parent=np.array([oldp['parent'][r[0]] for r in records]);remove=np.isin(parent,changed_ids)
      if c==0:
       # The legacy polygon repair emits its repaired outer-ring rows last; base polygon and wall classification emits all both_planes_reached rows in order.
       offset=len(oldface)-len(records);assert offset>=0
       for j,(i,owners,area,bary,q,poly) in enumerate(records):
        q[2]=(k+1)*HOP;expected=np.r_[k*gu.P+owners[0],(k+1)*gu.P+owners[1],q,bary@oldp['maps'][i,:,0],bary@oldp['maps'][i,:,1],bary@(oldp['arcs'][i,:,0]+oldp['arcs'][i,:,1])];maxold=max(maxold,float(abs(oldface[offset+j,:10]-expected).max()))
       assert maxold<1e-12,maxold;facekeep[offset+np.flatnonzero(remove)]=False
      else:
       group=np.array([i for i,r in enumerate(oldvr) if r['cls']==MIDPOINT_WALL_CLASSES[name]]);assert len(group)>=len(records);where=group[-len(records):] if len(records) else np.array([],int)
       for jj,(i,owners,area,bary,q,poly) in enumerate(records):
        assert int(oldw['wall'][where[jj],0])==(k+(c==2))*gu.P+owners[0]
        assert np.max(abs(oldw['wall'][where[jj],4:6]-bary@oldp['maps'][i,:,0 if c==1 else 1]))<1e-12
       wallkeep[where[remove]]=False
      new=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][1]}_N{N}_{k}.npz');rr=overlaps(new,c,tiles,np.isin(new['parent'],changed_ids))
      if not rr:continue
      q=np.array([r[4] for r in rr]);q[:,2]=(k+1)*HOP;met=metric_chunks(ev,q);flux=np.array([r[2] for r in rr])*abs(met['J']*met['B'][:,2])/q[:,0]
      for j,(i,owners,area,bary,_,poly) in enumerate(rr):
       if c==0:
        faceout.append(np.r_[k*gu.P+owners[0],(k+1)*gu.P+owners[1],q[j],bary@new['maps'][i,:,0],bary@new['maps'][i,:,1],bary@(new['arcs'][i,:,0]+new['arcs'][i,:,1]),flux[j]/met['M'][j]]);faceflux.append(flux[j]);changed.append(int(new['parent'][i]))
       else:
        reach=0 if c==1 else 1;hit=1-reach;sg=1 if c==1 else -1;owner=owners[0];hits=new['wallhit'][i,:,hit];angles=hits[:,1:].copy();angles=angles[0]+(angles-angles[0]+np.pi)%TAU-np.pi;wallpoint=np.r_[1.,(bary@angles)%TAU];ra=float(bary@new['arcs'][i,:,reach]);ell=ra+float(bary@new['hitell'][i,:,hit]);span=np.ptp(poly@angles,axis=0)/np.array([cells[owner,3]-cells[owner,2],HOP]);trace=cache.ray(tuple(q[j]),sg*.5*HOP);valid=trace is not None and trace['hit'];dist=None;er=None
        if valid:dist=float(np.linalg.norm(((wallpoint[1:]-trace['wall'][1:]+np.pi)%TAU-np.pi)/np.array([cells[owner,3]-cells[owner,2],HOP])));er=abs(ell-ra-trace['wall_length'])/max(ra+trace['wall_length'],1e-12)
        flag=bool((span>1).any() or (valid and dist>.25));override=flag and valid
        if override:wallpoint=trace['wall'].copy();wallpoint[0]=1.;ell=ra+trace['wall_length']
        assert np.isfinite(wallpoint).all() and ell>0
        walls.append(np.r_[(k+(c==2))*gu.P+owner,wallpoint,bary@new['maps'][i,:,reach],flux[j],0.,ell,0.]);wp.append(flux[j]);ws.append(sg);flags.append(flag);overrides.append(override);validation.append(dict(cls=MIDPOINT_WALL_CLASSES[name],strip=k,parent=int(new['parent'][i]),owner=int(owner),side=sg,flux=float(flux[j]),vertex_span_theta_cells=float(span[0]),vertex_span_eta_cells=float(span[1]),barycentre_cached=trace is not None,barycentre_hit=bool(valid),wall_distance_cells=dist,relative_ell_error=er,grazing_flag=flag,used_traced_barycentre=override))
     nw=np.asarray(walls).reshape(-1,10)
     if len(nw):m=metric_chunks(ev,nw[:,1:4]);nw[:,6]/=m['M'];nw[:,7]=np.sign(m['B'][:,0])
     nf=np.asarray(faceout).reshape(-1,11);fullf=np.vstack([oldface[facekeep],nf]);fullw=np.vstack([oldw['wall'][wallkeep],nw]);allpsi=np.r_[oldwp[wallkeep],wp];alls=np.r_[oldside[wallkeep],ws].astype(int);allk=np.r_[oldkind[wallkeep],np.ones(len(nw),int)];allflags=np.r_[oldw['grazing_flag'][wallkeep],flags];allover=np.r_[oldw['used_traced_barycentre'][wallkeep],overrides]
     oldmet=metric_chunks(ev,oldface[:,2:5]);oldfp=oldface[:,10]*oldmet['M'];fullfp=np.r_[oldfp[facekeep],faceflux]
     save(OUT/f'arrays/geometry_N{N}_{k}.npz',faces=fullf,face_flux=fullfp,wall=fullw,wall_flux=allpsi,wall_side=alls,wall_kind=allk,grazing_flag=allflags,used_traced_barycentre=allover,old_face_index=np.r_[np.flatnonzero(facekeep),np.full(len(nf),-1)].astype(int),old_wall_index=np.r_[np.flatnonzero(wallkeep),np.full(len(nw),-1)].astype(int))
     save(OUT/f'arrays/changed_faces_N{N}_{k}.npz',old=oldface[~facekeep],new=nf,old_flux=oldfp[~facekeep],new_flux=np.array(faceflux),new_parent=np.array(changed))
     rings=np.rint(cells[:,0]*N).astype(int);byring=[]
     for side in [0,1]:
      for ring in np.unique(rings):
       a=rings[oldface[~facekeep,side].astype(int)%gu.P]==ring;b=rings[nf[:,side].astype(int)%gu.P]==ring
       if a.any() or b.any():byring.append(dict(owner_side=side,ring=int(ring),old_rows=int(a.sum()),new_rows=int(b.sum()),old_flux=float(oldfp[~facekeep][a].sum()),new_flux=float(np.array(faceflux)[b].sum()),flux_change=float(np.array(faceflux)[b].sum()-oldfp[~facekeep][a].sum())))
     dump(dest,dict(complete=True,N=N,plane=k,untouched_both_planes_reached_rows=int(facekeep.sum()),removed_both_planes_reached_rows=int((~facekeep).sum()),new_both_planes_reached_rows=len(nf),untouched_both_planes_reached_bitwise=bool(np.array_equal(fullf[:facekeep.sum()],oldface[facekeep])),old_geometry_mapping_absolute_max=maxold,changed_by_ring=byring,changed_old_owners=np.unique(oldface[~facekeep,:2].astype(int),axis=0).tolist(),changed_new_owners=np.unique(nf[:,:2].astype(int),axis=0).tolist(),new_midpoint_wall_row_count=len(nw),plane_seeded_wall_rows_unchanged=True,wall_interpolation=validation,new_rays=0))
     log(f'N{N} geometry plane{k}: {facekeep.sum()} both_planes_reached rows retained bitwise, {(~facekeep).sum()} replaced by {len(nf)}; no rays')


    return {'classpath': classpath, 'main': main}
