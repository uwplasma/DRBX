"""Local clipping and footprint planning."""
import numpy as np
from .clipping_primitives import area_centroid, cross, disk, inpoly

def candidates(data,low,high):
 bb0=data['bb0'];bb1=data['bb1'];index=data.get('candidate_index')
 if data.get('candidate_mode')=='indexed' and index is None:
  from .clip_index import CandidateIndex
  index=data['candidate_index']=CandidateIndex(bb0,bb1)
 out=index(low,high) if index is not None else np.flatnonzero(np.all(bb1>=low-1e-13,axis=1)&np.all(bb0<=high+1e-13,axis=1))
 return out

def triangle_range_numpy(d,bounds):
 lo,hi=bounds;xy=d['xy'];maps=d['maps'];arcs=d['arcs'];good=d['good'];boundary=d['boundary'];quads=d['quads'];owner=d['owner'];bb0=d['bb0'];bb1=d['bb1'];records=[];areas=[];source=[];record_keys=[];poly_diag=[];subcount=0;nleft=0;validleft=0;nright=0;validright=0
 for tri_index in range(lo,hi):
  ids=d['tris'][tri_index];poly=[];land=[];arc=[]
  for a,b in zip(ids,np.roll(ids,-1)):
   if good[a]:poly.append(xy[a]);land.append(maps[a]);arc.append(arcs[a])
   if good[a]!=good[b]:
    x,m,ell=boundary[tuple(sorted([a,b]))];poly.append(x);land.append(m);arc.append(ell)
  if len(poly)<3:continue
  poly=np.array(poly);land=np.array(land);arc=np.array(arc)
  for j in range(1,len(poly)-1):
   inds=[0,j,j+1];xx=poly[inds];lands=[land[inds,d] for d in [0,1]];ell=arc[inds];base=np.eye(3);a,_=area_centroid(base,xx)
   if not a:continue
   subcount+=1
   low=lands[0].min(0);high=lands[0].max(0);left=candidates(d,low,high)
   for l in left:
    lp=inpoly(base,lands[0],quads[l]);nleft+=1;validleft+=len(lp)>=3
    if d.get('clip_topology_capture'):poly_diag.append((tri_index,j,int(l),-1,len(lp)))
    if len(lp)<3:continue
    low=(lp@lands[1]).min(0);high=(lp@lands[1]).max(0);right=candidates(d,low,high)
    for r in right:
     overlap=inpoly(lp,lands[1],quads[r]);nright+=1;validright+=len(overlap)>=3
     if d.get('clip_topology_capture'):poly_diag.append((tri_index,j,int(l),int(r),len(overlap)))
     area,bary=area_centroid(overlap,xx)
     if area<=1e-16:continue
     pt=bary@xx;records.append([owner[l],owner[r],*pt,*(bary@lands[0]),*(bary@lands[1]),bary@ell]);areas.append(area);source.append(tri_index);record_keys.append((tri_index,j,int(l),int(r)))
 return dict(records=np.asarray(records).reshape(-1,9),areas=np.asarray(areas),source_triangle=np.asarray(source,dtype=np.int64),record_key=np.asarray(record_keys,np.int64).reshape(-1,4),polygon_topology=np.asarray(poly_diag,np.int64).reshape(-1,5),clip_counts=np.array([[subcount,nleft,validleft,nright,validright,len(records)]],np.int64))

def tiles(cells,N,angular_subdivisions):
 tau=2*np.pi
 angles=np.unique(np.round(np.r_[np.arange(angular_subdivisions*N)*tau/(angular_subdivisions*N),cells[:,2:4].ravel()%tau],13))
 out=[];owners=[]
 for j,(r0,r1,t0,t1) in enumerate(cells):
  local=np.r_[angles-tau,angles,angles+tau];ts=np.r_[t0,local[(local>t0+1e-12)&(local<t1-1e-12)],t1];ts.sort()
  for a,b in zip(ts[:-1],ts[1:]):
   q=np.array([[r0*np.cos(a),r0*np.sin(a)],[r1*np.cos(a),r1*np.sin(a)],[r1*np.cos(b),r1*np.sin(b)],[r0*np.cos(b),r0*np.sin(b)]])
   if r0==0:q=q[:3]
   out.append(q);owners.append(j)
 return out,np.array(owners),np.array([q.min(0) for q in out]),np.array([q.max(0) for q in out])
