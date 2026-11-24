"""Deterministic expanded PointCubic, variable supports and owner bank."""
import numpy as np
from ...operators.parallel.weights import POWERS, PointCubic, basis


class BlockRuleStop(ValueError):
 """Stop rule: a 5x5 block contains a node outside the allowed fit-node set."""

class BlockCubic(PointCubic):
 """5x5 logical-block owner fits (the QF undershoot study's BlockCubic.fit ported onto this PointCubic).
 1. Fit as now: PointCubic.fit with rule 'expand_fit_support' (unchanged).
 2. If that support has > 25 nodes and the owner is on the ring block (node_ring >= 0, not the core), replace it by rings
    lo..lo+4 x theta columns j-2..j+2 (periodic in theta), lo = min(max(r-2, r0), r1-4) (shifted inward at the wall);
    owner first, the rest by (distance rounded to 1e-11 h, id). Basis, weights 1/max(|delta|,1), pinv, rank test and
    exactness |C V - I| <= 1e-12 are PointCubic.coefficients unchanged; the owner-preserving row is FitTable unchanged.
 3. Everything else is unchanged. A block node outside the allowed set raises BlockRuleStop (count reported)."""
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.rule='expand_fit_support';self.ring=np.asarray(self.layout.node_ring);self.theta=np.asarray(self.layout.node_theta)
  rings=sorted(set(self.ring[self.ring>=0].tolist()));self.r0,self.r1=rings[0],rings[-1];self.by_ring={}
  for r in rings:ids=np.flatnonzero(self.ring==r);self.by_ring[r]=ids[np.argsort(self.theta[ids],kind='stable')]
  self.replaced={};self.kept_core_long=[];self.sealed={}
 def fit(self,owner):
  plane=int(self.layout.plane(owner))%self.layout.symmetry_period_planes;i=int(owner)-int(self.layout.plane_offsets[int(self.layout.plane(owner))]);key=(plane,i)
  if key in self.fits:return self.fits[key]
  ids,C=super().fit(owner);self.sealed[key]=(len(ids),self.conditions[-1])
  if len(ids)<=25 or key in getattr(self,'trace_owners',set()):return ids,C
  r=int(self.ring[i])
  if r<0:self.kept_core_long.append(key);return ids,C
  col=self.by_ring[r];j=int(np.flatnonzero(col==i)[0]);n=len(col);lo=min(max(r-2,self.r0),self.r1-4)
  if any(len(self.by_ring[rr])!=n for rr in range(lo,lo+5)):raise ValueError(('ring theta counts differ',key))
  sel=np.array([int(self.by_ring[rr][(j+dj)%n]) for rr in range(lo,lo+5) for dj in range(-2,3)])
  if i not in sel or len(set(sel.tolist()))!=25:raise ValueError(('block selection',key))
  bad=int((~self.allowed[plane,sel]).sum())
  if bad:raise BlockRuleStop(('block node outside the allowed set',key,bad))
  others=sel[sel!=i];dd=np.round(np.linalg.norm(self.xy[others]-self.xy[i],axis=1)*self.N,11);ids_b=np.r_[i,others[np.lexsort((others,dd))]]
  res=self.coefficients(ids_b,i)
  if res is None:raise ValueError(('block fit rank or exactness failure',key))
  Cb,cond=res;delta=(self.xy[ids_b]-self.xy[i])*self.N;exact=float(np.max(abs(Cb@basis(delta[1:])-np.eye(len(POWERS)))))
  u=np.hypot(*self.xy[ids_b].T);arc=float(np.max(abs(((self.theta[ids_b]-self.theta[i]+np.pi)%(2*np.pi)-np.pi)*u[0]))*self.N)
  self.replaced[key]=dict(ring=r,sealed_support=len(ids),sealed_condition=self.conditions[-1],condition=cond,radial_reach_h=float(np.max(abs(u-u[0]))*self.N),arc_reach_h=arc,exactness=exact,wall_shifted=bool(r+2>self.r1))
  self.conditions[-1]=cond;self.fits[key]=(ids_b,Cb);return ids_b,Cb


class WallExpandedBlockFit(BlockCubic):
 """One expanded fit per geometric physical-wall owner; block all others."""
 def __init__(self,*args,trace_owners=(),**kw):
  super().__init__(*args,**kw);self.trace_owners=frozenset(trace_owners)
