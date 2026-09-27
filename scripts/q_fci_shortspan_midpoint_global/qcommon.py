"""Minimal frozen Q owner-moment and eta-plane helpers.

Extracted from work/q_fci_shared_quartic_nonseam_20260925/common.py,
source identities recorded in the 2026-09-27 campaign receipt.
"""
import math
import numpy as np
from scripts.q_fci_return_campaign.endpoints import EXPS4
def eta_planes(ctx, point):
    centers=np.asarray(ctx["artifact"].geometry.grid.z.centers); H=2*np.pi/ctx["N"]
    rr=(point[2]-centers[0])/H; near=round(rr)
    if abs(rr-near)*H<2e-10: return np.asarray([near],int),np.asarray([1.0])
    planes=math.floor(rr)+np.arange(-1,3)
    weights=np.asarray([np.prod([(rr-b)/(a-b) for b in planes if b!=a]) for a in planes])
    return planes,weights


class LocalOwnerMoments:
    """Owner-indexed physical raw members; members never become fit unknowns."""

    def __init__(self, ctx):
        self.begin_target()
        self.ctx=ctx; grid=ctx["artifact"].geometry.grid; shape=tuple(ctx["artifact"].geometry.shape);self.shape=shape
        labels=np.asarray(ctx["topology"]["compact_raw_owner"],int).ravel()
        self.labels=labels; self.nowner=len(ctx["volume"])
        self.order=np.argsort(labels,kind="stable")
        counts=np.bincount(labels,minlength=self.nowner)
        self.starts=np.r_[0,np.cumsum(counts)]
        u,theta,_=np.meshgrid(grid.x.centers,grid.y.centers,grid.z.centers,indexing="ij")
        self.x=(u*np.cos(theta)).ravel(); self.y=(u*np.sin(theta)).ravel()
        self.theta=theta.ravel(); self.volume=np.asarray(ctx["artifact"].polar_angular_geometry.raw_volume).ravel()
        self.rcol=np.broadcast_to(np.arange(shape[0])[:,None,None],shape).ravel()
        self.tcol=np.broadcast_to(np.arange(shape[1])[None,:,None],shape).ravel()
        self.counts=counts; self.plane=np.asarray(ctx["model"].eta,int)
        vcos=np.bincount(labels,weights=self.volume*np.cos(self.theta),minlength=self.nowner)
        vsin=np.bincount(labels,weights=self.volume*np.sin(self.theta),minlength=self.nowner)
        angle=np.mod(np.arctan2(vsin,vcos),2*np.pi)
        first=float(grid.y.centers[0]); dtheta=2*np.pi/shape[1]
        self.owner_tcol=np.rint(((angle-first)%(2*np.pi))/dtheta).astype(int)%shape[1]
        self.ntheta=shape[1]

    def members(self,owner):
        return self.order[self.starts[owner]:self.starts[owner+1]]

    def begin_target(self):
        """Bound all reusable rows and geometry to one raw target."""
        self._moment_cache={};self._direct_cache={};self._plane_cache={};self._donor_sets={};self._wall_relations={}

    def local(self,donors,xy,scale,exps=EXPS4):
        out=np.empty((len(donors),len(exps)))
        for row,owner in enumerate(donors):
            key=(int(owner),np.asarray(xy,float).tobytes(),np.asarray(scale,float).tobytes(),tuple(exps))
            cached=self._moment_cache.get(key)
            if cached is None:
                m=self.members(int(owner)); w=self.volume[m]; xx=(self.x[m]-xy[0])/scale[0]; yy=(self.y[m]-xy[1])/scale[1]
                cached=np.asarray([float(np.dot(w,xx**p*yy**q)/np.sum(w)) for p,q in exps])
                self._moment_cache[key]=cached
            out[row]=cached
        return out

    def footprint(self,donors):
        cells=np.concatenate([self.members(int(owner)) for owner in donors])
        return {"angular_columns":sorted(np.unique(self.tcol[cells]).astype(int).tolist()),
                "radial_layers":sorted(np.unique(self.rcol[cells]).astype(int).tolist()),
                "aggregate_owners":int(np.sum(self.counts[np.asarray(donors,int)]>1))}

    def expansion_priority(self,seed,plane,old,xy,scale):
        grid=self.ctx["artifact"].geometry.grid
        j0=int((np.searchsorted(grid.y.faces,seed[1]%(2*np.pi),side="right")-1)%self.ntheta)
        offsets=(0,4,-4,8,-8,2,-2,6,-6,1,-1,3,-3,5,-5,7,-7)
        columns=[(j0+offset)%self.ntheta for offset in offsets]
        pool=np.flatnonzero(self.plane==int(plane))
        pool=pool[np.isin(self.owner_tcol[pool],columns)]
        delta=(np.asarray(self.ctx["model"].centroid)[pool]-xy)/scale
        dist=np.linalg.norm(delta,axis=1)
        grouped={col:[] for col in columns}
        for owner,d in zip(pool,dist): grouped[int(self.owner_tcol[owner])].append((float(d),int(owner)))
        for col in columns:grouped[col].sort()
        oldset=set(map(int,old)); result=[]; seen=set(oldset)
        for depth in range(8):
            for col in columns:
                if depth<len(grouped[col]):
                    owner=grouped[col][depth][1]
                    if owner not in seen:result.append(owner);seen.add(owner)
        remaining=sorted((d,owner) for entries in grouped.values() for d,owner in entries if owner not in seen)
        result.extend(owner for _,owner in remaining)
        return np.asarray(result,int)


def hull_metrics(index, donors, seed, points):
    from scipy.spatial import ConvexHull
    donorxy=np.asarray(index.ctx["model"].centroid)[donors]
    pts=np.asarray(points,float);targetxy=np.column_stack((pts[:,0]*np.cos(pts[:,1]),pts[:,0]*np.sin(pts[:,1])))
    hull=ConvexHull(donorxy)
    equations=hull.equations
    distance=(equations[:,:2]@targetxy.T+equations[:,2,None])/np.linalg.norm(equations[:,:2],axis=1)[:,None]
    clearance=-np.max(distance,axis=0)
    anchor=np.array((seed[0]*np.cos(seed[1]),seed[0]*np.sin(seed[1])))
    er=np.array((np.cos(seed[1]),np.sin(seed[1])));et=np.array((-np.sin(seed[1]),np.cos(seed[1])))
    tang_scale=max(seed[0]*2*np.pi/index.ctx["N"],1/index.ctx["N"])
    def extent(xy):
        delta=xy-anchor
        return {"radial_min":float(np.min(delta@er)),"radial_max":float(np.max(delta@er)),
                "tangential_min":float(np.min(delta@et)),"tangential_max":float(np.max(delta@et)),
                "scaled_radial_min":float(np.min(delta@er)*index.ctx["N"]),"scaled_radial_max":float(np.max(delta@er)*index.ctx["N"]),
                "scaled_tangential_min":float(np.min(delta@et)/tang_scale),"scaled_tangential_max":float(np.max(delta@et)/tang_scale),
                "max_regular_xy_distance":float(np.max(np.linalg.norm(delta,axis=1)))}
    cells=np.concatenate([index.members(int(owner)) for owner in donors])
    memberxy=np.column_stack((index.x[cells],index.y[cells]))
    return clearance,extent(donorxy),extent(targetxy),extent(memberxy)
