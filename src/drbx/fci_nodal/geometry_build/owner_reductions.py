"""Canonical source census, owner routing and ordered reductions."""
import numpy as np
from .clipping_primitives import disk,cross
from .production_splice import polygon_keys, retained_keys, source_band
from .storage import arrays

def volume_parts(base,sign):
 cut=arrays(base/f'q8_{sign}/plan',['cuts'])['cuts']
 for order in [3,8]:
  for leaf in sorted((base/f'q{order}_{sign}').glob('leaf_*')):
   z=arrays(leaf);take=~np.isin(z['owner'],cut) if order==3 else np.ones(len(z['owner']),bool)
   yield {key:value[take] if key not in ['slab_volume','divergence_exponent_max'] and value.ndim and len(value)==len(take) else value for key,value in z.items()},leaf

def census(m,ns,k,base):
 N=m['N'];nt=m['angular_subdivisions']*N;xy,_,tri=disk(N,nt)
 mask=source_band(xy,tri,N,m['polygon_rings'],nt)
 got=np.zeros(len(tri));bounds=[];counts=[];allrecords=0
 for leaf in sorted((base/'metric').glob('leaf_*')):
  z=arrays(leaf,['area','source_triangle']);np.add.at(got,z['source_triangle'],z['area']);counts.append(int((~mask[z['source_triangle']]).sum()));bounds.append(int(leaf.name.split('_')[-1]));allrecords+=len(z['area'])
 area=cross(xy[tri][:,1]-xy[tri][:,0],xy[tri][:,2]-xy[tri][:,0])/2;bulk=~mask;diff=abs(got[bulk]-area[bulk]);relative=float(np.max(diff/area[bulk],initial=0));absolute=float(np.max(diff,initial=0))
 bad=(diff/area[bulk]>1e-10)&(diff>1e-14)
 if np.any(bad):raise ArithmeticError(('Original per-triangle amended source-coverage guard',k,np.flatnonzero(bulk)[bad].tolist(),relative,absolute))
 return dict(mask=mask,task_lo=np.asarray(bounds,np.int64),retained_counts=np.asarray(counts,np.int64)),dict(stage='source_census',plane=k,plane_nodes=ns['P'],ad_per_period=sum(counts),source_records=allrecords,source_area_relative=relative,source_area_absolute=absolute)

def physical_count(lo,hi):
 count=hi-lo
 return dict(count=np.asarray(count,np.int64)),dict(stage='physical_wall_count',lo=lo,hi=hi)

def plane(m,plan,w,base,k,c,production):
 N=m['N'];p=m['replication_periods'];K=N//p;I=plan['plane_nodes'];mask=c['mask'];face0=plan['face_prefix'][k];nfp=m['field_periods'];period_angle=2*np.pi/nfp
 H=np.zeros(I);mom=np.zeros((I,6));cent=np.zeros((I,3))
 for sign in [1,-1]:
  for z,leaf in volume_parts(base,sign):
   np.add.at(H,z['owner'],z['volume']);np.add.at(mom,z['owner'],z['label_moments']);np.add.at(cent,z['owner'],z['centroid_integral'])
 if np.any(H<=0) or not np.isfinite(H).all():raise ArithmeticError('Original physical H positivity/finite guard')
 for per in range(p):
  actual=k+per*K;ids=actual*I+np.arange(I);w.emit(f'physical_p{actual:03d}',dict(H=H,moments=mom/H[:,None],centroid=cent/H[:,None]),ids,w.owner(actual))
  ordinal=0
  original=plan['original_face_prefix'][k]+per*plan['coverage'][k]['source_records']
  for leaf in sorted((base/'metric').glob('leaf_*')):
   z=arrays(leaf);keep=~mask[z['source_triangle']];q=np.array(z['point'][keep]);q[:,2]=(q[:,2]+0.)%(2*np.pi);q[:,2]=(q[:,2]+per*period_angle)%(2*np.pi)
   f=np.c_[actual*I+z['owners'][keep,0],((actual+1)%N)*I+z['owners'][keep,1],q,z['foot_left'][keep],z['foot_right'][keep],z['ell'][keep],z['mu'][keep]];ids=face0+per*plan['ad_per_period'][k]+ordinal+np.arange(len(f))
   if len(f):
    data=dict(faces=f,face_splice_provenance=np.ones(len(f),np.uint8),face_source_plane_triangle=np.c_[np.full(len(f),actual),z['source_triangle'][keep]])
    data['face_parent_row']=original+np.flatnonzero(keep)
    extra=production.metadata('face',ids,retained_keys(actual,z['source_quad_pair'][keep]))
    w.emit(f'ad_k{k:03d}_per{per}_{leaf.name}',data,ids,w.owner(actual),extra)
   original+=len(keep)
   ordinal+=len(f)
  if ordinal!=plan['ad_per_period'][k]:raise RuntimeError('Count/prefix census changed')

def polygon(m,plan,w,lo,hi,production):
 a=production.faces
 f=a[lo:hi];ids=np.arange(lo,hi,dtype=np.int64);dest=(f[:,0].astype(np.int64)//plan['plane_nodes'])*plan['storage_bins']//m['N']
 data=dict(faces=f,face_splice_provenance=np.zeros(len(f),np.uint8),face_source_plane_triangle=np.full((len(f),2),-1,np.int64))
 data['face_parent_row']=ids
 extra=production.metadata('face',ids,polygon_keys(ids,production.face_stream))
 w.emit(f'polygon_{lo:012d}',data,ids,dest,extra)

def physical_wall(m,plan,w,lo,hi,prefix,production):
 data={name:rows[lo:hi] for name,rows in production.wall.items()}
 ids=prefix+np.arange(hi-lo,dtype=np.int64);parent=np.arange(lo,hi,dtype=np.int64)
 data.update(wall_splice_provenance=np.zeros(len(ids),np.uint8),wall_parent_row=parent)
 dest=(data['wall'][:,0].astype(np.int64)//plan['plane_nodes'])*plan['storage_bins']//m['N']
 w.emit(f'physical_wall_{lo:012d}',data,ids,dest,production.metadata('wall',ids,polygon_keys(parent,production.wall_stream)))

