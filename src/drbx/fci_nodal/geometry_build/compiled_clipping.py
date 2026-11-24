"""Canonical batched clip traversal."""
import numpy as np
from .compiled_polygons import CAP

def padded(poly):
 out=np.zeros((CAP,3));out[:len(poly)]=poly;return out

def triangle_range(d,bounds):
 from . import clipping as c
 xy=d['xy'];maps=d['maps'];arcs=d['arcs'];good=d['good'];boundary=d['boundary'];quads=d['quads'];owners=d['owner'];records=[];areas=[];source=[];record_keys=[];poly_diag=[];fan_keys=[];fan_lands=[];fan_arcs=[];fan_xxs=[];left_groups=[];subcount=0
 # Canonical source triangle / fan / left tile order, as the frozen loop.
 for tri_index in range(*bounds):
  ids=d['tris'][tri_index];poly=[];land=[];arc=[]
  for a,b in zip(ids,np.roll(ids,-1)):
   if good[a]:poly.append(xy[a]);land.append(maps[a]);arc.append(arcs[a])
   if good[a]!=good[b]:
    x,m,ell=boundary[tuple(sorted([a,b]))];poly.append(x);land.append(m);arc.append(ell)
  if len(poly)<3:continue
  poly=np.array(poly);land=np.array(land);arc=np.array(arc)
  for j in range(1,len(poly)-1):
   inds=[0,j,j+1];xx=poly[inds];ll=land[inds];ell=arc[inds]
   if c.cross(xx[1]-xx[0],xx[2]-xx[0])/2<=1e-18:continue
   subcount+=1;left=c.candidates(d,ll[:,0].min(0),ll[:,0].max(0))
   if len(left):fan_keys.append((tri_index,j));fan_lands.append(ll);fan_arcs.append(ell);fan_xxs.append(xx);left_groups.append(left)
 nleft=sum(map(len,left_groups))
 if nleft:
  fan=np.repeat(np.arange(len(left_groups)),[len(v) for v in left_groups]);qleft=np.concatenate(left_groups)
  bases=np.broadcast_to(padded(np.eye(3)),(nleft,CAP,3)).copy();counts=np.full(nleft,3,np.int64)
  lands=np.asarray(fan_lands)[fan];arclist=np.asarray(fan_arcs)[fan];xxs=np.asarray(fan_xxs)[fan];keys=np.c_[np.asarray(fan_keys,np.int64)[fan],qleft]
  if hasattr(quads,'padded'):quad=quads.padded[qleft];qn=quads.counts[qleft]
  else:
   quad=np.zeros((nleft,4,2));qn=np.array([len(quads[l]) for l in qleft],np.int64)
   for i,l in enumerate(qleft):quad[i,:qn[i]]=quads[l]
  args=(bases,counts,lands[:,:,0],quad,qn,xxs);lp,ln,la,lb=d['compiled_polygons'].run(*args);right_groups=[];left_parents=[]
  for i,key in enumerate(keys):
   if d.get('clip_topology_capture'):poly_diag.append((*key,-1,int(ln[i])))
   if ln[i]<3:continue
   footprint=lp[i,:ln[i]]@lands[i,:,1];right=c.candidates(d,footprint.min(0),footprint.max(0))
   if len(right):right_groups.append(right);left_parents.append(i)
  nright=sum(map(len,right_groups))
  if nright:
   parent=np.repeat(np.asarray(left_parents,np.int64),[len(v) for v in right_groups]);right_ids=np.concatenate(right_groups);right_keys=np.c_[keys[parent],right_ids]
   if hasattr(quads,'padded'):right_quads=quads.padded[right_ids];right_qn=quads.counts[right_ids]
   else:
    right_quads=np.zeros((nright,4,2));right_qn=np.array([len(quads[r]) for r in right_ids],np.int64)
    for i,r in enumerate(right_ids):right_quads[i,:right_qn[i]]=quads[r]
   args=tuple(np.ascontiguousarray(a) for a in [lp[parent],ln[parent],lands[parent,:,1],right_quads,right_qn,xxs[parent]]);rp,rn,ra,rb=d['compiled_polygons'].run(*args)
   for i,(key,p) in enumerate(zip(right_keys,parent)):
    if d.get('clip_topology_capture'):poly_diag.append((*key,int(rn[i])))
    if ra[i]<=1e-16:continue
    bary=rb[i];pt=bary@xxs[p];l=key[2];r=key[3];records.append([owners[l],owners[r],*pt,*(bary@lands[p,:,0]),*(bary@lands[p,:,1]),bary@arclist[p]]);areas.append(ra[i]);source.append(key[0]);record_keys.append(key)
  else:rn=np.empty(0,int)
 else:nright=0;ln=np.empty(0,int);rn=np.empty(0,int)
 # Right records above are already in the canonical left/right traversal order.
 # Diagnostics are keyed, not used to choose numerical output.
 return dict(records=np.asarray(records).reshape(-1,9),areas=np.asarray(areas),source_triangle=np.asarray(source,np.int64),record_key=np.asarray(record_keys,np.int64).reshape(-1,4),polygon_topology=np.asarray(poly_diag,np.int64).reshape(-1,5),clip_counts=np.array([[subcount,nleft,int((ln>=3).sum()),nright,int((rn>=3).sum()),len(records)]],np.int64))
