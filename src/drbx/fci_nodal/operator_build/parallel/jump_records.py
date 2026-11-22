"""bounded indexed parallel assembly moment-reduced face quadrature recipe."""

import numpy as np

from .pair_cubature import reduce_pair


def group_keys(f,prov,N):
 interface=np.rint(f[:,4]*N/(2*np.pi)).astype(int)%N
 return np.c_[f[:,:2].astype(np.int64),interface,prov]


def select(f,keys,ids,degree):
 unique,inverse=np.unique(keys,axis=0,return_inverse=True)
 order=np.argsort(inverse,kind='stable');edges=np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1,len(order)] if len(order) else np.array([0],np.int64)
 counts=np.diff(edges);schedule=np.lexsort((np.arange(len(counts)),-counts))
 xy=np.c_[f[:,2]*np.cos(f[:,3]),f[:,2]*np.sin(f[:,3])]
 selected=[];weights=[]
 for group_id in schedule:
  group=order[edges[group_id]:edges[group_id+1]]
  sel,w,_=reduce_pair(xy[group],f[group,10],degree);selected.extend(ids[group[sel]].tolist());weights.extend(w.tolist())
 selected=np.array(selected,np.int64);weights=np.array(weights,dtype=np.float64);perm=np.argsort(selected)
 return selected[perm],weights[perm],len(unique)


