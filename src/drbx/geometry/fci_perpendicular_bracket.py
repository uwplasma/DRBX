"""Host preparation for the qualified cubic perpendicular bracket.

The builder is deliberately field independent.  It compiles owner-observation
functionals into polynomial coefficient maps and evaluates continuous geometry
at q3 nodes; changing state values never repeats donor search or factorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "BRACKET_SIDECAR_VERSION", "CUBIC_EXPONENTS", "SelectionV3Policy",
    "CubicOwnerGeometry", "FaceBlock", "CellBlock", "select_cubic_support_v3",
    "build_face_block", "build_cell_block", "write_sidecar_block", "load_sidecar_block",
]

BRACKET_SIDECAR_VERSION = "drbx.perpendicular-bracket-sidecar-v1"
CUBIC_EXPONENTS = (
    (0,0,0),(1,0,0),(0,1,0),(0,0,1),
    (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2),
    (3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
    (0,3,0),(0,2,1),(0,1,2),(0,0,3),
)
BIAS = 0.75
EPS_FACTOR = 128.0
SVD_CUTOFF = 1.0e-12
REPRODUCTION_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class SelectionV3Policy:
    expansion: tuple[tuple[int, int], ...] = ((24, 32), (32, 32), (40, 48), (48, 64))
    minimum_angular_columns: int = 3
    bias: float = BIAS
    version: str = "hsx-cubic-selection-v3"


@dataclass(frozen=True)
class CubicOwnerGeometry:
    owner_flat_ids: np.ndarray
    owner_plane: np.ndarray
    owner_angular_column: np.ndarray
    owner_centroid_xy: np.ndarray
    owner_eta: np.ndarray
    owner_moments_xy: np.ndarray  # [owner, ten moments ordered by total degree]
    owner_volume: np.ndarray
    raw_owner: np.ndarray
    raw_volume: np.ndarray
    resolution: int
    dr: float
    dtheta: float
    deta: float
    eta_period: float
    identity: str

    def __post_init__(self):
        n = len(self.owner_flat_ids)
        for name in ("owner_plane", "owner_angular_column", "owner_eta", "owner_volume"):
            if len(np.asarray(getattr(self, name))) != n:
                raise ValueError(f"{name} length differs from owner count")
        if np.asarray(self.owner_centroid_xy).shape != (n, 2):
            raise ValueError("owner_centroid_xy must have shape (owner,2)")
        if np.asarray(self.owner_moments_xy).shape != (n, 10):
            raise ValueError("owner_moments_xy must contain ten degree<=3 xy moments")
        if np.any(np.asarray(self.raw_owner) < 0) or np.any(np.asarray(self.raw_owner) >= n):
            raise ValueError("raw_owner contains an invalid compact owner")


class FaceBlock(NamedTuple):
    indices: np.ndarray; keys: np.ndarray; valid: np.ndarray
    collapsed: np.ndarray; wall: np.ndarray
    donors: np.ndarray; donor_mask: np.ndarray; donor_count: np.ndarray
    scale: np.ndarray; expansion_level: np.ndarray
    central_map: np.ndarray; left_map: np.ndarray; right_map: np.ndarray
    boundary_gradient_map: np.ndarray
    basis: np.ndarray; derivative_basis: np.ndarray
    weights: np.ndarray; h: np.ndarray
    lower_raw: np.ndarray; lower_valid: np.ndarray
    upper_raw: np.ndarray; upper_valid: np.ndarray


class CellBlock(NamedTuple):
    indices: np.ndarray; valid: np.ndarray
    donors: np.ndarray; donor_mask: np.ndarray; donor_count: np.ndarray
    scale: np.ndarray; expansion_level: np.ndarray; central_map: np.ndarray
    basis: np.ndarray; derivative_basis: np.ndarray
    weights: np.ndarray; curl_h: np.ndarray


_MOMENTS = tuple((a, b) for total in range(4) for a in range(total + 1) for b in (total - a,))
_MOMENT_SLOT = {value: index for index, value in enumerate(_MOMENTS)}
_TREE_CACHE: dict[int, tuple[tuple[np.ndarray, cKDTree], ...]] = {}


def _plane_trees(geometry):
    key=id(geometry)
    cached=_TREE_CACHE.get(key)
    if cached is None:
        cached=tuple((indices:=np.flatnonzero(geometry.owner_plane==plane),cKDTree(geometry.owner_centroid_xy[indices])) for plane in range(geometry.resolution))
        _TREE_CACHE[key]=cached
    return cached


def _unwrap(value, center, period):
    return center + np.mod(np.asarray(value) - center + 0.5 * period, period) - 0.5 * period


def _tied_order(radius2, ids, tolerance):
    order = np.lexsort((ids, radius2))
    cursor = 0
    for start in np.flatnonzero(np.diff(radius2[order]) <= tolerance):
        if start < cursor:
            continue
        stop = int(start) + 1; anchor = radius2[order[start]]
        while stop < len(order) and radius2[order[stop]] - anchor <= tolerance:
            stop += 1
        order[start:stop] = order[start:stop][np.argsort(ids[order[start:stop]], kind="stable")]
        cursor = stop
    return order


def _sectors(displacement):
    octant = np.mod(np.arctan2(displacement[..., 1], displacement[..., 0]), 2*np.pi)/(np.pi/4)
    nearest = np.rint(octant)
    snapped = np.where(np.abs(octant-nearest) <= EPS_FACTOR*np.finfo(float).eps, nearest, octant)
    return np.floor(snapped).astype(np.int64) % 8


def _observation(geometry, donors, center, scale):
    result = np.empty((len(donors), 20))
    eta = _unwrap(geometry.owner_eta[donors], center[2], geometry.eta_period)
    zeta = (eta-center[2])/scale[2]
    for slot, (a,b,c) in enumerate(CUBIC_EXPONENTS):
        value = np.zeros(len(donors))
        for i in range(a+1):
            for j in range(b+1):
                value += (
                    math.comb(a,i)*math.comb(b,j)
                    *(-center[0])**(a-i)*(-center[1])**(b-j)
                    /(scale[0]**a*scale[1]**b)
                    * geometry.owner_moments_xy[donors, _MOMENT_SLOT[(i,j)]]
                )
        result[:,slot] = value*zeta**c
    return result


def _planes(axis, eta_index, n):
    offsets = (-2,-1,0,1,2) if axis in (0,1) else (-3,-2,-1,0,1,2)
    return tuple((eta_index+offset) % n for offset in offsets)


def select_cubic_support_v3(geometry, point, axis, eta_index, policy=SelectionV3Policy()):
    point=np.asarray(point,float); center=np.array((point[0]*np.cos(point[1]),point[0]*np.sin(point[1]),point[2]))
    planar=max(geometry.dr,max(abs(point[0]),.5*geometry.dr)*geometry.dtheta)
    scale=np.array((planar,planar,geometry.deta)); extent=max(np.max(abs(geometry.owner_centroid_xy)),np.max(abs(center[:2])),geometry.dr)
    tolerance=EPS_FACTOR*np.finfo(float).eps*extent**2
    plane_ids=_planes(axis,eta_index,geometry.resolution)
    for level,(count,pool_count) in enumerate(policy.expansion):
        parts=[]
        for plane in plane_ids:
            compact,tree=_plane_trees(geometry)[plane]
            take=min(pool_count,len(compact));distance,_=tree.query(center[:2],k=take)
            radius=np.sqrt(float(np.atleast_1d(distance)[-1])**2+4*tolerance)
            local=tree.query_ball_point(center[:2],np.nextafter(radius,np.inf))
            candidate=compact[np.asarray(local,dtype=np.int64)]
            if len(candidate)<take:raise RuntimeError("selection-v3 candidate query omitted neighbors")
            r2=np.sum((geometry.owner_centroid_xy[candidate]-center[:2])**2,axis=1)
            order=_tied_order(r2,geometry.owner_flat_ids[candidate],tolerance)
            pool=candidate[order[:take]]
            displacement=(geometry.owner_centroid_xy[pool]-center[:2])/planar
            sectors=_sectors(displacement); rr=np.sum(displacement**2,axis=1)
            groups=[]
            for sector in range(8):
                sel=np.flatnonzero(sectors==sector)
                local=_tied_order(rr[sel],geometry.owner_flat_ids[pool[sel]],tolerance/planar**2)
                groups.append(sel[local])
            sequence=[groups[s][rank] for rank in range(max(map(len,groups))) for s in range(8) if rank<len(groups[s])]
            parts.append(pool[np.asarray(sequence[:count])])
        donors=np.concatenate(parts)
        obs=_observation(geometry,donors,center,scale)
        eta=_unwrap(geometry.owner_eta[donors],center[2],geometry.eta_period)
        d=np.column_stack((geometry.owner_centroid_xy[donors]-center[:2],eta-center[2]))
        distance=np.sqrt(np.sum((d/scale)**2,axis=1)); root=1/(1+distance**2)
        # This is intentionally the same Gram-eigenspectrum and reproduction
        # test used by the frozen campaign.  A rectangular SVD is algebraically
        # equivalent in exact arithmetic, but it can accept a different
        # expansion level for nearly deficient rows at N64.
        weighted=obs*root[:,None]
        gram=weighted.T@weighted
        eigenvalues=np.linalg.eigvalsh(gram)
        maximum=max(float(eigenvalues[-1]),np.finfo(float).tiny)
        singular=np.sqrt(np.maximum(eigenvalues,0.0))
        rank=int(np.sum(singular>SVD_CUTOFF*np.sqrt(maximum)))
        target=np.zeros((20,3));target[1,0]=1/scale[0];target[2,1]=1/scale[1];target[3,2]=1/scale[2]
        solution=np.linalg.solve(gram,target)
        derivative_weights=root[:,None]**2*obs@solution
        residual=float(np.max(np.abs(obs.T@derivative_weights-target)))
        columns=geometry.owner_angular_column[donors].reshape(len(plane_ids),count)
        coverage=min(len(np.unique(row)) for row in columns)
        if rank==20 and residual<=REPRODUCTION_TOLERANCE and coverage>=policy.minimum_angular_columns:
            return donors,center,scale,obs,d,distance,level
    raise RuntimeError("selection-v3 cubic support remains deficient")


def _maps(obs, displacement, distance, scale, axis, theta, bias):
    weight2=(1/(1+distance**2))**2
    gram=obs.T@(weight2[:,None]*obs)
    central=np.linalg.solve(gram,obs.T*weight2[None,:])
    normal=(np.array((np.cos(theta),np.sin(theta),0.)) if axis==0 else
            np.array((-np.sin(theta),np.cos(theta),0.)) if axis==1 else np.array((0.,0.,1.)))
    s=displacement@normal/np.sqrt(np.sum((normal*scale)**2))
    side=[]
    for sign in (-1.,1.):
        w=np.sqrt(weight2*(1+sign*bias*np.tanh(s))); design=obs*w[:,None]
        side.append(np.linalg.lstsq(design,np.diag(w),rcond=None)[0])
    return central,side[0],side[1]


def _boundary_derivative_map(obs,distance,scale,point):
    kappa=1/(1+distance**2);weighted=obs*kappa[:,None];gram=weighted.T@weighted
    target=np.zeros((20,3));target[1,0]=1/scale[0];target[2,1]=1/scale[1];target[3,2]=1/scale[2]
    solution=np.linalg.solve(gram,target)
    regular=(kappa[:,None]**2*obs@solution).T
    c=np.cos(point[1]);s=np.sin(point[1]);logical=regular.copy()
    logical[0]=c*regular[0]+s*regular[1]
    logical[1]=-point[0]*s*regular[0]+point[0]*c*regular[1]
    return logical


def _basis(points,center,scale,eta_period):
    p=np.asarray(points); regular=np.column_stack((p[:,0]*np.cos(p[:,1]),p[:,0]*np.sin(p[:,1]),_unwrap(p[:,2],center[2],eta_period)))
    z=(regular-center)/scale; basis=np.empty((len(p),20)); derivative=np.zeros((len(p),20,3))
    for slot,exponent in enumerate(CUBIC_EXPONENTS):
        basis[:,slot]=np.prod([z[:,a]**power for a,power in enumerate(exponent)],axis=0)
        for a,power in enumerate(exponent):
            if power:
                factors=[z[:,b]**(q-(1 if b==a else 0)) for b,q in enumerate(exponent)]
                derivative[:,slot,a]=power/scale[a]*np.prod(factors,axis=0)
    cosine=np.cos(p[:,1]);sine=np.sin(p[:,1]);logical=derivative.copy()
    logical[...,0]=cosine[:,None]*derivative[...,0]+sine[:,None]*derivative[...,1]
    logical[...,1]=-p[:,0,None]*sine[:,None]*derivative[...,0]+p[:,0,None]*cosine[:,None]*derivative[...,1]
    return basis,logical


def _pad_maps(rows,max_entities,max_donors):
    donors=np.zeros((max_entities,max_donors),np.int32);mask=np.zeros_like(donors,bool)
    maps=[np.zeros((max_entities,20,max_donors)) for _ in range(3)]
    for r,(d,values) in enumerate(rows):
        donors[r,:len(d)]=d;mask[r,:len(d)]=True
        for out,value in zip(maps,values):out[r,:,:len(d)]=value
    return donors,mask,maps


def build_face_block(geometry,indices,keys,points,weights,h_evaluator,*,block_size=256,policy=SelectionV3Policy()):
    indices=np.asarray(indices,np.int64);keys=np.asarray(keys,np.int32);points=np.asarray(points,float);weights=np.asarray(weights,float)
    if len(indices)>block_size:raise ValueError("face block exceeds fixed block size")
    max_donors=max(c*(5 if a in (0,1) else 6) for c,_ in policy.expansion for a in range(3))
    rows=[];basis=np.zeros((block_size,9,20));derivative=np.zeros((block_size,9,20,3));boundary_gradient=np.zeros((block_size,3,max_donors));scale=np.ones((block_size,3));level=np.full(block_size,255,np.uint8);count=np.zeros(block_size,np.int16)
    collapsed=np.zeros(block_size,bool);wall=np.zeros(block_size,bool);valid=np.zeros(block_size,bool);valid[:len(indices)]=True
    for r,(axis,i,j,k) in enumerate(keys):
        collapsed[r]=axis==0 and i==0;wall[r]=axis==0 and i==geometry.resolution
        if collapsed[r]:rows.append((np.array([],int),(np.empty((20,0)),)*3));continue
        d,c,s,o,disp,dist,l=select_cubic_support_v3(geometry,points[r,4],int(axis),int(k),policy)
        m=_maps(o,disp,dist,s,int(axis),points[r,4,1],policy.bias);rows.append((d,m));scale[r]=s;level[r]=l;count[r]=len(d)
        if wall[r]:boundary_gradient[r,:,:len(d)]=_boundary_derivative_map(o,dist,s,points[r,4])
        basis[r],derivative[r]=_basis(points[r],c,s,geometry.eta_period)
    donors,mask,maps=_pad_maps(rows,block_size,max_donors)
    h=np.zeros((block_size,9,3));active=np.flatnonzero(valid&~collapsed)
    if len(active):h[active]=np.asarray(h_evaluator(points[active].reshape(-1,3))).reshape(len(active),9,3)
    lower=np.zeros(block_size,np.int32);upper=np.zeros(block_size,np.int32);lv=np.zeros(block_size,bool);uv=np.zeros(block_size,bool)
    n=geometry.resolution
    for r,(axis,i,j,k) in enumerate(keys):
        cell=[i,j,k]
        if cell[axis]>0:cell[axis]-=1;lower[r]=(cell[0]*n+cell[1])*n+cell[2];lv[r]=True
        cell=[i,j,k]
        if cell[axis]<n:upper[r]=(cell[0]*n+cell[1])*n+cell[2];uv[r]=True
    def pad(a,shape):out=np.zeros(shape,dtype=a.dtype);out[:len(a)]=a;return out
    return FaceBlock(pad(indices,(block_size,)),pad(keys,(block_size,4)),valid,collapsed,wall,donors,mask,count,scale,level,*maps,boundary_gradient,basis,derivative,pad(weights,(block_size,9)),h,lower,lv,upper,uv)


def build_cell_block(geometry,indices,points,weights,curl_h_evaluator,*,block_size=64,policy=SelectionV3Policy()):
    indices=np.asarray(indices,np.int64);points=np.asarray(points,float);weights=np.asarray(weights,float)
    if len(indices)>block_size:raise ValueError("cell block exceeds fixed block size")
    max_donors=max(c*5 for c,_ in policy.expansion);rows=[];basis=np.zeros((block_size,27,20));derivative=np.zeros((block_size,27,20,3));scale=np.ones((block_size,3));level=np.full(block_size,255,np.uint8);count=np.zeros(block_size,np.int16);valid=np.zeros(block_size,bool);valid[:len(indices)]=True
    n=geometry.resolution
    for r,index in enumerate(indices):
        i=index//(n*n);j=(index//n)%n;k=index%n; point=points[r,13]
        d,c,s,o,disp,dist,l=select_cubic_support_v3(geometry,point,0,int(k),policy);m=_maps(o,disp,dist,s,0,point[1],0.0)[0]
        rows.append((d,(m,m,m)));scale[r]=s;level[r]=l;count[r]=len(d);basis[r],derivative[r]=_basis(points[r],c,s,geometry.eta_period)
    donors,mask,maps=_pad_maps(rows,block_size,max_donors)
    curl=np.zeros((block_size,27,3));curl[:len(indices)]=np.asarray(curl_h_evaluator(points.reshape(-1,3))).reshape(len(indices),27,3)
    def pad(a,shape):out=np.zeros(shape,dtype=a.dtype);out[:len(a)]=a;return out
    return CellBlock(pad(indices,(block_size,)),valid,donors,mask,count,scale,level,maps[0],basis,derivative,pad(weights,(block_size,27)),curl)


def _identity_payload(block,identity,policy):
    # Normalize tuples here so the in-memory payload compares identically with
    # its JSON round trip during a sidecar load.
    policy_payload=json.loads(json.dumps(policy.__dict__,sort_keys=True))
    return {"schema":BRACKET_SIDECAR_VERSION,"identity":identity,"policy":policy_payload,"kind":"face" if isinstance(block,FaceBlock) else "cell"}


def write_sidecar_block(path,block,*,identity,policy=SelectionV3Policy()):
    path=Path(path);payload=_identity_payload(block,identity,policy);arrays={name:np.asarray(value) for name,value in zip(block._fields,block)}
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as stream:np.savez_compressed(stream,**arrays,metadata_json=np.asarray(json.dumps(payload,sort_keys=True)))
    temp.replace(path)


def load_sidecar_block(path,*,identity,policy=SelectionV3Policy()):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata_json"].item()));kind=meta["kind"];cls=FaceBlock if kind=="face" else CellBlock
        if meta!=_identity_payload(cls(*[z[name] for name in cls._fields]),identity,policy):raise ValueError("sidecar geometry/policy identity mismatch")
        return cls(*[np.asarray(z[name]) for name in cls._fields])
