"""Conservative broad phase only; original predicate and ascending IDs retained."""
import numpy as np
from scipy.spatial import cKDTree

class CandidateIndex:
 def __init__(self,bb0,bb1):
  self.bb0=bb0;self.bb1=bb1
  self.center=(bb0+bb1)/2
  self.radius=float(np.max((bb1-bb0)/2))
  self.tree=cKDTree(self.center)
 def __call__(self,low,high):
  # Logical disk coordinates are bounded by one. 1e-12 conservatively
  # covers the original 1e-13 tolerance plus center/radius rounding.
  center=(low+high)/2;radius=float(np.max((high-low)/2))+self.radius+1e-12
  ids=np.asarray(self.tree.query_ball_point(center,radius,p=np.inf),dtype=np.int64)
  ids.sort()
  return ids[np.all(self.bb1[ids]>=low-1e-13,axis=1)&np.all(self.bb0[ids]<=high+1e-13,axis=1)]
