"""base polygon and wall classification/repaired polygon and merged wall rows bundle numerics bound to explicit local inputs.

"""
from ._case import MIDPOINT_WALL_CLASSES, PLANE_WALL_STEM, REACH_CLASSES, REACH_OUTPUT_STEMS
import numpy as np
import time


def bind(context):
    N = context.N
    TAU = context.TAU
    HOP = context.HOP
    OUT = context.out
    EVALUATOR = context.evaluator
    gu = context.gu
    Lookup = context.Lookup
    load = context.load
    save = context.save
    dump = context.dump
    log = context.log
    relative = context.relative
    seeds = context.seeds
    transform = context.transform
    moment = context.moment
    arc_tiles = context.arc_tiles
    intersections = context.intersections
    metric_chunks = context.metric_chunks
    BQ = context.BQ
    WQ = context.WQ
    BQ8 = context.BQ8
    WQ8 = context.WQ8

    def main(k):
     path=OUT/f'checkpoints/geometry_{k}.json'
     if path.exists():return
     assert (OUT/f'checkpoints/traced_{k}.json').exists()
     start=time.time();ev=EVALUATOR;tiles=arc_tiles();cells=gu.build()[1];cache=Lookup();walls=[];wp=[];kinds=[];sides=[];validation=[];classes={};faces=[];fpsi=[]
     def wallgeom(cls,sg,owner,flux,foot,ell,hits,bary,poly,qref,reacharc=0.):
      angles=hits[:,1:].copy();angles=angles[0]+(angles-angles[0]+np.pi)%TAU-np.pi;wi=np.r_[1.,(bary@angles)%TAU];width=np.array([cells[owner,3]-cells[owner,2],HOP]);span=np.ptp(poly@angles,axis=0)/width;trace=cache.ray(qref,sg*.5*HOP);valid=trace is not None and bool(trace['hit']);dist=er=None
      if valid:
       trlen=reacharc+float(trace['wall_length']);dist=float(np.linalg.norm(((wi[1:]-trace['wall'][1:]+np.pi)%TAU-np.pi)/width));er=abs(ell-trlen)/max(trlen,1e-12)
      flag=bool((span>1).any() or (valid and dist>.25));override=flag and valid
      if override:wi=trace['wall'].copy();wi[0]=1.;ell=trlen
      plane=k if cls!=MIDPOINT_WALL_CLASSES[REACH_CLASSES[2]] else k+1
      rec=dict(cls=cls,strip=k,plane=plane,side=sg,owner=int(owner),ring=int(round(cells[owner,0]*N)),flux=float(flux),vertex_span_theta_cells=float(span[0]),vertex_span_eta_cells=float(span[1]),barycentre_cached=trace is not None,barycentre_hit=valid,wall_distance_cells=dist,relative_ell_error=er,grazing_flag=flag,used_traced_barycentre=override,flagged_without_hit=flag and not valid)
      # Store flux temporarily; convert to mu in one vectorized field evaluation.
      walls.append(np.r_[plane*gu.P+owner,wi,foot,flux,0.,ell,0.]);wp.append(flux);sides.append(sg);kinds.append(2 if cls==PLANE_WALL_STEM else 1);validation.append(rec)
     for c,name in enumerate(REACH_CLASSES):
      p=load(OUT/f'arrays/{REACH_OUTPUT_STEMS[name][0]}_{k}.npz');records=[];areas=[];points=[];barys=[]
      for i,xx in enumerate(p['xy']):
       kind='half' if np.max(np.linalg.norm(xx,axis=1))>(N-1)/N+1e-12 else None
       a,b,pt=moment(xx,np.eye(3),kind);areas.append(a);points.append(pt);barys.append(b)
       if c==3:continue
       for owners,poly in intersections(xx,p['maps'][i],c,tiles):
        area,bary,point=moment(xx,poly,kind);q=seeds(point[None],(k+1)*HOP)[0];records.append((i,owners,area,bary,q,poly))
      if points:
       q=seeds(np.asarray(points),(k+1)*HOP);met=metric_chunks(ev,q);flux=np.asarray(areas)*abs(met['J']*met['B'][:,2])/q[:,0];classes[name]=dict(area=float(sum(areas)),flux=float(sum(flux)),triangles=len(areas))
      else:classes[name]=dict(area=0.,flux=0.,triangles=0)
      if records:
       q=np.array([r[4] for r in records]);met=metric_chunks(ev,q);flux=np.array([r[2] for r in records])*abs(met['J']*met['B'][:,2])/q[:,0]
       for j,(i,owners,area,bary,q,poly) in enumerate(records):
        if c==0:
         faces.append(np.r_[k*gu.P+owners[0],(k+1)*gu.P+owners[1],q,bary@p['maps'][i,:,0],bary@p['maps'][i,:,1],bary@(p['arcs'][i,:,0]+p['arcs'][i,:,1]),flux[j]/met['M'][j]]);fpsi.append(flux[j])
        else:
         sg=1 if c==1 else -1;reach=0 if c==1 else 1;hit=1-reach;ra=float(bary@p['arcs'][i,:,reach]);wallgeom(MIDPOINT_WALL_CLASSES[name],sg,int(owners[0]),float(flux[j]),bary@p['maps'][i,:,reach],ra+float(bary@p['hitell'][i,:,hit]),p['wallhit'][i,:,hit],bary,poly,q,ra)
     quadchecks=[]
     for sg in [-1,1]:
      p=load(OUT/f'arrays/{PLANE_WALL_STEM}_{k}_{sg}.npz');nf=len(p['fans']);fluxes=[]
      for bq,wq in [(BQ,WQ),(BQ8,WQ8)]:
       ff=[]
       for lo in range(0,nf,24):
        t=p['fans'][lo:lo+24];ref=np.einsum('qv,tvd->tqd',bq,t);xy,jac=transform(ref,'plane');rr=np.linalg.norm(xy,axis=2);q=seeds(xy.reshape(-1,2),(k+.5)*HOP);met=metric_chunks(ev,q);f=(abs(met['J']*met['B'][:,2])/rr.ravel()).reshape(rr.shape);det=gu.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);ff.extend(det*np.sum(f*jac*wq,axis=1))
       fluxes.append(np.asarray(ff))
      quadchecks.append(dict(side=sg,order8_16_relative=relative(*fluxes)))
      det=gu.cross(p['fans'][:,1]-p['fans'][:,0],p['fans'][:,2]-p['fans'][:,0])
      for row,owner in enumerate(p['owner']):
       ids=np.flatnonzero((p['fanrow']==row)&(det>2e-18));area=[];barys=[];points=[]
       for i in ids:
        a,b,pt=moment(p['fans'][i],np.eye(3),'plane');area.append(a);barys.append(b);points.append(pt)
       if not area or sum(area)<=1e-16:continue
       weights=np.array(area)/sum(area);foot=weights@np.asarray(points);coeff=(weights[:,None]*np.asarray(barys)).ravel();hits=p['wallhit'][ids].reshape(-1,3);ell=float(coeff@p['hitell'][ids].ravel());flux=float(fluxes[0][ids].sum());q=seeds(foot[None],(k+.5)*HOP)[0]
       wallgeom(PLANE_WALL_STEM,sg,int(owner),flux,foot,ell,hits,coeff,np.eye(len(coeff)),q)
     wall=np.array(walls).reshape(-1,10);met=metric_chunks(ev,wall[:,1:4]);wall[:,6]/=met['M'];wall[:,7]=np.sign(met['B'][:,0])
     for r,w in zip(validation,wall):r['sg_side_mismatch']=bool(w[7]!=r['side'])
     save(OUT/f'arrays/geometry_{k}.npz',faces=np.asarray(faces).reshape(-1,11),face_flux=np.asarray(fpsi),wall=wall,wall_flux=np.asarray(wp),wall_kind=np.asarray(kinds),wall_side=np.asarray(sides),grazing_flag=np.array([r['grazing_flag'] for r in validation]),used_traced_barycentre=np.array([r['used_traced_barycentre'] for r in validation]))
     dump(OUT/f'checkpoints/interpolation_{k}.json',validation);rec=dict(complete=True,plane=k,seconds=time.time()-start,classes=classes,faces=len(faces),wall_rows=len(wall),plane_seeded_wall_rows_quadrature=quadchecks);dump(path,rec);log(f'plane {k} geometry checkpoint complete; {rec["seconds"]:.1f}s')


    return main
