"""Vectorized canonical full-grid topology and disjoint face-family census."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys,time,json,resource
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;START=time.process_time()
FAMILY=('collapsed_r0','physical_wall_quartic_BC','adjacent_radial_quartic_BC','centered_radial_cubic',
        'boundary_transverse_cubic_BC','direct_ordinary_singleton','direct_ringwise','direct_coupled_quartic')
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform=='darwin' else 2**20)
def raw_owner(n,input_root):
 d=Path(input_root)/f'geometry_artifacts/rlp_convergence_32_48_64_20260917/{n}x{n}x{n}'
 with np.load(d/'rlp_topology.npz') as z:active=z['is_active_owner'].ravel();aggregate=z['aggregate_id'].ravel();rv=z['raw_volume'].ravel()
 lookup=np.full(n**3,-1,np.int32);lookup[np.flatnonzero(active)]=np.arange(active.sum(),dtype=np.int32)
 return lookup[aggregate].reshape((n,n,n)),rv
def decode(n,ids):
 ids=np.asarray(ids);R=(n+1)*n*n;T=n**3;axis=np.where(ids<R,0,np.where(ids<R+T,1,2));local=np.where(axis==0,ids,np.where(axis==1,ids-R,ids-R-T))
 i=local//(n*n);j=(local//n)%n;k=local%n
 return np.column_stack((axis,i,j,k))
def support_rings(n,axis,i):
 layers=np.arange(i-2,i+2) if axis==0 else np.arange(i-1,i+3)
 return np.where(layers<0,-layers-1,layers)
def census(n,input_root,output_root):
 started=time.process_time();ro,rv=raw_owner(n,input_root);profile=np.empty(n,int)
 for i in range(n):
  owners=ro[i,:,0];counts=np.bincount(owners-min(owners));profile[i]=int(np.max(counts));assert np.all(np.unique(counts[counts>0])==profile[i])
 assert np.all(n%profile==0)
 radial_lo=np.full((n+1,n,n),-1,np.int32);radial_hi=radial_lo.copy()
 # Face i separates radial cells i-1 and i; i=0 and i=n have exterior ends.
 radial_lo[1:n]=ro[:-1];radial_lo[n]=ro[-1];radial_hi[1:n]=ro[1:];radial_hi[0]=ro[0];radial_hi[n]=-1
 theta_lo=np.roll(ro,1,axis=1);theta_hi=ro
 eta_lo=np.roll(ro,1,axis=2);eta_hi=ro
 off=(n+1)*n*n;off2=off+n**3
 arrays=[];internal={};seams={}
 for axis,(lo,hi,start) in enumerate(((radial_lo,radial_hi,0),(theta_lo,theta_hi,off),(eta_lo,eta_hi,off2))):
  valid=(lo!=hi).ravel();fid=start+np.flatnonzero(valid);ep=np.column_stack((lo.ravel()[valid],hi.ravel()[valid])).astype(np.int32)
  arrays.append((fid.astype(np.int32),ep));internal[str(axis)]=int(valid.size-valid.sum())
  if axis:
   key=decode(n,fid);seams['theta' if axis==1 else 'eta']=int(np.sum((key[:,2] if axis==1 else key[:,3])==0))
 face_ids=np.concatenate([x[0] for x in arrays]);endpoints=np.concatenate([x[1] for x in arrays]);assert np.all(np.diff(face_ids)>0)
 keys=decode(n,face_ids);axis=keys[:,0];i=keys[:,1];fam=np.full(len(keys),255,np.uint8)
 radial=axis==0;trans=~radial
 fam[radial&(i==0)]=0;fam[radial&(i==n)]=1;fam[radial&np.isin(i,(n-1,n-2))]=2
 fam[radial&(i>=n-6)&(i<=n-3)]=3;fam[trans&(i>=n-2)]=4
 direct=fam==255;angular=n//profile
 for aa in (0,1,2):
  pick=np.flatnonzero(direct&(axis==aa))
  if not len(pick):continue
  rings=np.stack([support_rings(n,aa,int(q)) for q in np.unique(i[pick])])
  levels=np.unique(i[pick]);idx=np.searchsorted(levels,i[pick]);layers=rings[idx]
  assert np.all((layers>=0)&(layers<n))
  plain=np.all(profile[layers]==1,axis=1);ring=np.all(angular[layers]>=7,axis=1)
  fam[pick[plain]]=5;fam[pick[~plain&ring]]=6;fam[pick[~ring]]=7
 assert np.all(fam<8)
 # Check every declared fixed radial donor layer, including BC closures.
 assert np.all(profile[n-4:n]==1)
 assert n>=7 and n>=4
 for layer in range(n-6,n-2):
  fixed=np.arange(layer-2,layer+2)
  assert fixed.min()>=0 and fixed.max()<n and np.all(profile[fixed]==1)
 ring_ids=np.flatnonzero(fam==6);coupled_ids=np.flatnonzero(fam==7)
 if len(ring_ids):
  for ax,layer in np.unique(keys[ring_ids,:2],axis=0):assert np.all(angular[support_rings(n,int(ax),int(layer))]>=7)
 capacity=[]
 for ax,layer in np.unique(keys[coupled_ids,:2],axis=0):
  base=support_rings(n,int(ax),int(layer));lo=max(0,int(base.min())-5);hi=min(n-1,int(base.max())+5)
  capacity.append(int(np.minimum(7,angular[lo:hi+1]).sum()))
 assert not capacity or min(capacity)>=15
 no=int(ro.max())+1;incidence_counts=np.bincount(endpoints[endpoints>=0],minlength=no);assert np.all(incidence_counts>0)
 noncollapsed=(fam!=0);shared=(endpoints[:,0]>=0)&(endpoints[:,1]>=0)
 assert np.all(endpoints[shared,0]!=endpoints[shared,1]);assert np.all(shared[noncollapsed&(fam!=1)])
 axis_internal_theta=int(np.sum(theta_lo[0]==theta_hi[0]));assert axis_internal_theta==n*n
 assert int(np.sum(fam==0))==n*n and int(np.sum(fam==1))==n*n
 # Every active face has exactly one face ID and one flux object, with at most two recipients.
 np.savez_compressed(Path(output_root)/f'N{n}.topology.npz',face_ids=face_ids,endpoints=endpoints,family=fam,profile=profile,incidence_counts=incidence_counts)
 counts={FAMILY[j]:int(np.sum(fam==j)) for j in range(8)}
 result=dict(n=n,raw_cells=n**3,canonical_owners=no,distinct_faces=len(face_ids),families=counts,
  internal_raw_faces=internal,axis_internal_theta_faces=axis_internal_theta,theta_periodic_seam_faces=seams['theta'],eta_periodic_seam_faces=seams['eta'],
  physical_wall_faces=int(np.sum(fam==1)),collapsed_r0_faces=int(np.sum(fam==0)),shared_faces=int(np.sum(shared)),exterior_faces=int(sum(~shared)),
  donor_index_ranges_valid=True,wall_and_centered_donors_singleton=True,ringwise_min_angular_donors=7,
  coupled_min_bounded_capacity=min(capacity) if capacity else None,all_owners_incident=True,profile=profile.tolist(),profile_transitions=(np.flatnonzero(profile[1:]!=profile[:-1])+1).tolist(),
  cpu_seconds=time.process_time()-started,peak_rss_gib=rss())
 (Path(output_root)/f'N{n}.topology.json').write_text(json.dumps(result,indent=2)+'\n')
 print('TOPOLOGY',n,len(face_ids),counts,'CPU',result['cpu_seconds'],'RSS',rss(),flush=True)
 return result
