import numpy as np

from ...geometry_build.clipping_primitives import cross

def in_triangles(points,tri):
 points=np.asarray(points).reshape(-1,2);out=np.zeros(len(points),bool)
 for t in tri:
  det=cross(t[1]-t[0],t[2]-t[0])
  if abs(det)<=1e-18:continue
  if det<0:t=t[[0,2,1]]
  lo=t.min(0)-2e-12;hi=t.max(0)+2e-12;ii=np.flatnonzero(~out&np.all(points>=lo,axis=1)&np.all(points<=hi,axis=1))
  if len(ii):out[ii]=np.all(np.stack([cross(b-a,points[ii]-a)>=-2e-13 for a,b in zip(t,np.roll(t,-1,axis=0))]),axis=0)
 return out

def stats(v):
 v=np.asarray(v);v=v[np.isfinite(v)]
 if not len(v):return dict(count=0)
 return dict(count=len(v),minimum=float(v.min()),maximum=float(v.max()),rms=float(np.sqrt(np.mean(v*v))),quantiles=np.quantile(v,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())

def relative(a,b):return float(np.max(abs(np.asarray(a)-b),initial=0)/max(np.max(abs(b),initial=0),1e-12))

