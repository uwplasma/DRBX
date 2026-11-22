"""Geometry/layout contracts and header-only planning; no tracing at import."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Layout:
 N:int
 eta_planes:int
 nodes:int
 plane_offsets:np.ndarray
 symmetry_period_planes:int
 node_u:np.ndarray
 node_theta:np.ndarray
 node_ring:np.ndarray
 @classmethod
 def from_options(cls,options,geo):
  from ..nodal.nodal_layout_specs import build_zernike_core_ring_layout
  n=options.resolution;l=build_zernike_core_ring_layout(n,n_eta=options.eta_planes);P=int(l.P);nn=P*l.n_eta
  if len(geo['H']) != nn:raise ValueError('Q geometry H must match the nodal layout')
  return cls(n,l.n_eta,nn,np.arange(l.n_eta+1,dtype=np.int64)*P,options.symmetry_period_planes,np.asarray(l.node_u),np.asarray(l.node_theta),np.asarray(l.node_ring))
 @property
 def P(self):
  sizes=np.diff(self.plane_offsets)
  if not np.all(sizes==sizes[0]):raise ValueError('Frozen PointCubic family requires repeated plane topology')
  return int(sizes[0])
 @property
 def period_nodes(self):return int(self.plane_offsets[self.symmetry_period_planes])
 def plane(self,owner):return np.searchsorted(self.plane_offsets,np.asarray(owner),side='right')-1
 def landing(self,owner,feet):
  plane=self.plane(owner);eta=(plane+.5)*2*np.pi/self.eta_planes
  return np.c_[np.linalg.norm(feet,axis=1),np.arctan2(feet[:,1],feet[:,0])%(2*np.pi),eta]
 def points(self):
  ids=np.arange(self.nodes);plane=self.plane(ids);local=ids-self.plane_offsets[plane]
  return np.c_[self.node_u[local],self.node_theta[local],(plane+.5)*2*np.pi/self.eta_planes]


