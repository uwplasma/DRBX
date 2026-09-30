"""Geometry-only point functionals extending the P07 structured reconstruction.

Host-only package preparation. Owner observations are stored-physical-volume
weighted raw member-midpoint values. Boundary arrays enter at runtime.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from . import _fci_perpendicular_point_primitives as r


@dataclass(frozen=True)
class PointGeometry:
    eta_period: float
    dr: float
    dtheta: float
    deta: float
    owner_centroid_xy: np.ndarray


@dataclass(frozen=True)
class PointRowContext:
    """Explicit, field-independent topology and geometry for host preparation.

    Compact owner IDs are the evolved unknowns. Raw cells remain observation
    locations only. The stable member ordering matches the research builder.
    """

    n: int
    faces: tuple[np.ndarray, np.ndarray, np.ndarray]
    centers: tuple[np.ndarray, np.ndarray, np.ndarray]
    ro: np.ndarray
    rv: np.ndarray
    vol: np.ndarray
    pts: np.ndarray
    xy: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    g: PointGeometry

    @classmethod
    def from_arrays(cls, *, faces, centers, raw_to_owner, raw_volume,
                    owner_volume, owner_centroid_xy, eta_period, dr, dtheta, deta):
        centers = tuple(np.asarray(x, dtype=np.float64) for x in centers)
        faces = tuple(np.asarray(x, dtype=np.float64) for x in faces)
        n = len(centers[0])
        if len(centers) != 3 or len(faces) != 3 or any(len(x) != n for x in centers):
            raise ValueError("three equally sized center axes are required")
        if any(len(x) != n + 1 for x in faces):
            raise ValueError("face axes must have n+1 entries")
        ro = np.asarray(raw_to_owner, dtype=np.int64).reshape(-1)
        rv = np.asarray(raw_volume, dtype=np.float64).reshape(-1)
        vol = np.asarray(owner_volume, dtype=np.float64).reshape(-1)
        centroid = np.asarray(owner_centroid_xy, dtype=np.float64)
        if ro.shape != (n**3,) or rv.shape != (n**3,) or not len(vol):
            raise ValueError("raw topology or volumes have incompatible sizes")
        if np.any(ro < 0) or np.any(ro >= len(vol)) or np.any(rv <= 0) or np.any(vol <= 0):
            raise ValueError("owners and stored physical volumes must be positive and complete")
        if centroid.shape != (len(vol), 2):
            raise ValueError("owner centroids must have shape (owners, 2)")
        if not all(np.isfinite(x) and x > 0 for x in (eta_period, dr, dtheta, deta)):
            raise ValueError("stored coordinate periods and increments must be positive")
        summed = np.bincount(ro, weights=rv, minlength=len(vol))
        if not np.allclose(summed, vol, rtol=1e-12, atol=0):
            raise ValueError("owner volumes differ from complete stored raw-volume sums")
        order = np.argsort(ro, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(ro, minlength=len(vol)))]
        if np.any(np.diff(starts) == 0):
            raise ValueError("owner without raw members")
        ijk = np.array(np.unravel_index(np.arange(n**3), (n, n, n))).T
        pts = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
        xy = np.column_stack((pts[:, 0]*np.cos(pts[:, 1]), pts[:, 0]*np.sin(pts[:, 1])))
        # Preserve the producer's stored coordinate increments. Recomputing
        # them from faces changes low bits in coupled-fit scaling at N48/N64.
        g = PointGeometry(float(eta_period), float(dr), float(dtheta), float(deta), centroid)
        return cls(n, faces, centers, ro, rv, vol, pts, xy, order, starts, g)

    def members(self, owner):
        oid = int(owner)
        return self.order[self.starts[oid]:self.starts[oid + 1]]

    def observe_owners(self, raw_values, owners):
        """Preserve the stored raw-midpoint/physical-volume owner functional."""
        raw_values = np.asarray(raw_values, dtype=np.float64)
        if raw_values.shape[0] != self.n**3:
            raise ValueError("raw_values must contain every raw midpoint")
        scalar = raw_values.ndim == 1
        values = raw_values[:, None] if scalar else raw_values
        if values.ndim != 2:
            raise ValueError("raw_values must have shape (raw, fields)")
        out = np.empty((len(owners), values.shape[1]))
        for q, oid in enumerate(owners):
            members = self.members(oid)
            out[q] = (self.rv[members] @ values[members]) / self.vol[int(oid)]
        return out[:, 0] if scalar else out


@dataclass
class PointRows:
    donor_ids: np.ndarray
    value: np.ndarray
    gradient: np.ndarray
    boundary_conditioned: bool
    trace_donor_points: np.ndarray
    trace_target_points: np.ndarray
    diagnostics: dict


@dataclass
class PointFactors:
    """The 1-D factors a source's weights were actually built from (opt-in capture).

    Returned by ``StructuredReconstruction.rows_with_factors`` beside the unchanged
    ``PointRows``; only for the unconditioned ``singleton``, ``ringwise`` and
    ``centered_radial`` families. Arrays run over the source's ``q`` targets; the
    donor block of one target is (layer l, plane e, slot j) = 4 x 4 x 7.
    ``radial[:, 0]``/``radial[:, 1]`` are ``L``/``D`` of ``r.rows`` (``D`` undivided by ``h``).
    ``owner`` are the donor owners of singleton/centered_radial (theta_index, eta_index and
    ``layers`` locate their raw cells); ringwise instead carries per (l, e) ring entries
    ``ring_owner``/``ring_value``/``ring_derivative`` (owner ids and the ``vv``/``dd`` vectors).
    """
    family: str
    n: int
    layers: np.ndarray
    eta_index: np.ndarray
    eta_value: np.ndarray
    eta_derivative: np.ndarray
    radial: np.ndarray
    theta_index: np.ndarray = None
    theta_value: np.ndarray = None
    theta_derivative: np.ndarray = None
    owner: np.ndarray = None
    ring_owner: np.ndarray = None
    ring_value: np.ndarray = None
    ring_derivative: np.ndarray = None


#: inner donor-support rules of :class:`StructuredReconstruction`: the current ``"profile7"`` (coupled quartic
#: only while some ring of the four-layer stencil has fewer than seven owners) and ``"last_aggregate"`` (C1: also
#: coupled quartic through the last agglomerated ring)
INNER_SUPPORTS = ("profile7", "last_aggregate", "any_aggregate", "fixed_radius", "last_aggregate_nearest28")
#: C2 (``"last_aggregate_nearest28"``): Q's layout (C1: coupled through the last agglomerated ring), with Q's isotropic donor choice for the coupled quartic (the
#: 28 nearest owner centroids per eta plane, Q's tie order and rank repair over the nearest 40)
C2_DONORS = 28
C2_POOL = 40
#: C3 (``"fixed_radius"``): coupled quartic for stencils whose anchor-ring centre lies below this logical radius
#: (the N48/N64 C1-vs-C0 per-ring crossing, work/p08_donor_support_c1_20260930)
FIXED_SWITCH_U = 0.21


class StructuredReconstruction:
    def __init__(self, t, *, inner_support="profile7"):
        if inner_support not in INNER_SUPPORTS:
            raise ValueError(f"inner_support must be one of {INNER_SUPPORTS}, got {inner_support!r}")
        self.t = t
        self.inner_support = inner_support
        self.rings = {}
        self.fits = {}
        self.profile = np.array([len(np.unique(t.ro.reshape((t.n,)*3)[i,:,0])) for i in range(t.n)])
        # the last agglomerated ring: the first full ring (profile == n) minus one (n - 1 if no ring is full)
        full = np.flatnonzero(self.profile == t.n)
        self.last = int(full[0]) - 1 if len(full) else int(t.n) - 1
        self.top = (t.ro,t.rv,t.vol,t.pts,t.order,t.starts,
                    {'grid.y.centers':t.centers[1], 'grid.z.centers':t.centers[2]})
        # Per-instance bounded caches, with exact float keys (no quantization).
        self._nearest = lru_cache(maxsize=4096)(self._nearest_uncached)
        self._basis = lru_cache(maxsize=8192)(self._basis_uncached)

    def _nearest_uncached(self, axis, target):
        return r.nearest(self.t.centers[axis],target,7 if axis==1 else 4,
                         2*np.pi if axis==1 else self.t.g.eta_period)

    def _basis_uncached(self, axis, indices, target):
        nodes=self.t.centers[axis][list(indices)]
        return (r.theta_rows(nodes,target) if axis==1 else
                r.eta_rows(nodes,target,self.t.g.eta_period,self.t.g.deta))

    def ring(self, i, k):
        tag = int(i),int(k)
        if tag not in self.rings:
            t=self.t
            ids=np.unique(t.ro.reshape((t.n,)*3)[i,:,k]); angles=[]
            for oid in ids:
                mm=r.members(t,oid); a=t.pts[mm,1]
                angles.append(float(a[0]) if len(mm)==1 else float(np.angle(np.mean(np.exp(1j*a)))%(2*np.pi)))
            self.rings[tag]=ids,np.asarray(angles)
        return self.rings[tag]

    def anchor(self, key, location):
        t=self.t
        if location=='cell':
            return np.array([t.centers[a][int(key[a])%t.n] for a in range(3)])
        axis,*ijk=map(int,key)
        return np.array([(t.faces[a] if a==axis else t.centers[a])[ijk[a]] for a in range(3)])

    def rows(self, key, points, location='face', *, fixed_anchor=False):
        return self._rows(key,points,location,fixed_anchor,None)

    def rows_with_factors(self, key, points, location='face', *, fixed_anchor=False):
        """``(rows(...), PointFactors or None)``: the same ``PointRows`` bit for bit, plus the factors
        actually used (``None`` outside the unconditioned singleton/ringwise/centered_radial families)."""
        cap=[]
        row=self._rows(key,points,location,fixed_anchor,cap)
        return row,(cap[0] if cap else None)

    def _rows(self, key, points, location, fixed_anchor, cap):
        t=self.t; n=t.n; p=np.asarray(points,float)
        if len(self.fits)>4096:self.fits.clear()
        if location not in ('face','cell'):raise ValueError(location)
        anchor=self.anchor(key,location)
        axis,i=(int(key[0]),int(key[1])) if location=='face' else (1,int(key[0]))
        if location=='face' and axis==0 and i==0:
            return PointRows(np.array([],int),np.zeros((len(p),0)),np.zeros((len(p),3,0)),False,np.empty((0,3)),p.copy(),{'family':'collapsed_r0','max_residual':0.})
        if (axis==0 and i>=n-6) or (axis!=0 and i>=n-2):
            # The frozen boundary tensor kernel returns independent point gradients.
            facekey=key if location=='face' else (1,*key)
            z=r.boundary_map(n,'point',facekey,0,p,np.zeros(len(p)),np.zeros((len(p),3)),self.top,t.g.eta_period)
            off=int(z['boundary_conditioned'])
            v=np.einsum('ql,qld->qd',z['radial_value'][:,off:],z['layer_maps'][0])
            if cap is not None and z['kind']=='centered_radial' and not z['boundary_conditioned']:
                cap.append(PointFactors('centered_radial',n,np.asarray(z['radial_layers']),z['eta_indices'],z['eta_value'],z['eta_derivative'],
                    np.stack((z['radial_value'],z['radial_derivative']),axis=1),z['theta_indices'],z['theta_value'],z['theta_derivative'],t.ro[z['requested_raw_ids']]))
            return PointRows(z['donor_ids'],v,z['gradient_map'],z['boundary_conditioned'],z['trace_donor_points'],z['trace_target_points'],{'family':z['kind'],'max_residual':0.})
        layers=np.arange(i-2,i+2) if axis==0 else np.arange(i-1,i+3)
        rid=np.where(layers<0,-layers-1,layers)
        if (np.min(self.profile[rid])<7 or (self.inner_support in ("last_aggregate","last_aggregate_nearest28") and self.anchor_ring(axis,i,location)<=self.last)
                or (self.inner_support=="any_aggregate" and np.min(self.profile[rid])<n)
                or (self.inner_support=="fixed_radius" and self.below_fixed_switch(axis,i,location))):
            return self._coupled(key,p,anchor,layers,rid,fixed_anchor)
        if np.all(self.profile[rid]==n):
            return self._singleton(p,anchor,layers,rid,fixed_anchor,cap)
        return self._tensor(p,anchor,layers,rid,fixed_anchor,cap)

    def stencil_min_profile(self, axis, i, location='face'):
        """Smallest ring owner count over the four radial layers of the stencil ``_rows`` would use (layers
        mirrored through the axis as in ``_rows``, clipped at the wall)."""
        n=self.t.n; axis,i=int(axis),int(i)
        layers=np.arange(i-2,i+2) if (location=='face' and axis==0) else np.arange(i-1,i+3)
        rid=np.clip(np.where(layers<0,-layers-1,layers),0,n-1)
        return int(np.min(self.profile[rid]))

    def below_fixed_switch(self, axis, i, location='face'):
        """C3: the anchor ring's centre ``(anchor_ring + 1/2)/n`` lies below ``FIXED_SWITCH_U``."""
        return (self.anchor_ring(axis, i, location) + 0.5) / self.t.n < FIXED_SWITCH_U

    @staticmethod
    def anchor_ring(axis, i, location='face'):
        """The ring the stencil is anchored on: ``i`` for cells and theta/eta faces, ``i - 1`` for radial faces
        (axis 0, face ``i`` lies between rings ``i - 1`` and ``i``)."""
        return int(i)-1 if (location=='face' and int(axis)==0) else int(i)

    def _pack(self,p,columns,diagnostics):
        ids=np.array(sorted(columns),int)
        a=np.stack([columns[k] for k in ids],axis=-1)
        return PointRows(ids,a[:,0],a[:,1:],False,np.empty((0,3)),p.copy(),diagnostics)

    def _singleton(self,p,anchor,layers,rid,fixed_anchor,cap=None):
        t=self.t;n=t.n;L,D=r.rows((layers+.5)/n,p[:,0]);allids=[];blocks=[];fac=[]
        for q,point in enumerate(p):
            ta=anchor[1] if fixed_anchor else point[1]
            ea=anchor[2] if fixed_anchor else point[2]
            ti=self._nearest(1,float(ta));tv,td=self._basis(1,tuple(ti),float(point[1]))
            ei=self._nearest(2,float(ea));ev,ed=self._basis(2,tuple(ei),float(point[2]))
            theta=(ti[None,:]+np.where(layers<0,n//2,0)[:,None])%n
            raw=(rid[:,None,None]*n+theta[:,None,:])*n+ei[None,:,None]
            allids.append(t.ro[raw].ravel())
            if cap is not None:fac.append((ti,tv,td,ei,ev,ed,t.ro[raw]))
            blocks.append(np.array([L[q,:,None,None]*ev[None,:,None]*tv[None,None,:],
                D[q,:,None,None]*ev[None,:,None]*tv[None,None,:],
                L[q,:,None,None]*ev[None,:,None]*td[None,None,:],
                L[q,:,None,None]*ed[None,:,None]*tv[None,None,:]]).reshape(4,-1))
        ids=np.unique(allids);a=np.zeros((len(p),4,len(ids)))
        for q,don in enumerate(allids):
            for k in range(4):np.add.at(a[q,k],np.searchsorted(ids,don),blocks[q][k])
        if cap is not None:
            f=[np.array(x) for x in zip(*fac)]
            cap.append(PointFactors('singleton',n,np.array(layers),f[3],f[4],f[5],np.stack((L,D),axis=1),f[0],f[1],f[2],f[6]))
        return PointRows(ids,a[:,0],a[:,1:],False,np.empty((0,3)),p.copy(),{'family':'singleton','max_residual':0.})

    def _tensor(self,p,anchor,layers,rid,fixed_anchor,cap=None):
        t=self.t; n=t.n
        L,D=r.rows((layers+.5)/n,p[:,0]); columns={}; worst=0.; rank=7
        if cap is not None:fac=dict(ei=[],ev=[],ed=[],own=np.zeros((len(p),4,4,7),int),rv=np.zeros((len(p),4,4,7)),rd=np.zeros((len(p),4,4,7)))
        for q,point in enumerate(p):
            ea=anchor[2] if fixed_anchor else point[2]
            ei=r.nearest(t.centers[2],ea,4,t.g.eta_period)
            ev,ed=r.eta_rows(t.centers[2][ei],point[2],t.g.eta_period,t.g.deta)
            if cap is not None:fac['ei'].append(ei);fac['ev'].append(ev);fac['ed'].append(ed)
            for l,rr in enumerate(rid):
                theta=(point[1]+(np.pi if layers[l]<0 else 0))%(2*np.pi)
                ta=((anchor[1] if fixed_anchor else point[1])+(np.pi if layers[l]<0 else 0))%(2*np.pi)
                for e,kk in enumerate(ei):
                    ids,angles=self.ring(rr,kk); pick=r.nearest(angles,ta,7,2*np.pi); don=ids[pick]; nodes=angles[pick]
                    tag=('ring',int(rr),int(kk),*map(int,don))
                    if tag not in self.fits:
                        A=[]; U=[]
                        for oid in don:
                            mm=r.members(t,oid); b=r.cardinal(nodes,t.pts[mm,1])[0]
                            A.append((t.rv[mm]/t.vol[oid])@b); U.append(b.mean(axis=0))
                        A=np.asarray(A); U=np.asarray(U)
                        C,ra,_=r.pinv(A,np.ones(7)); CU,ru,_=r.pinv(U,np.ones(7))
                        self.fits[tag]=A,U,C,CU,ra,ru
                    A,U,C,CU,ra,ru=self.fits[tag]
                    v,d=r.cardinal(nodes,[theta]); target=np.vstack((v,d))
                    residual=max(r.resid(target,C,A),r.resid(target,CU,U)); worst=max(worst,residual);rank=min(rank,ra,ru)
                    if residual>1e-9:raise RuntimeError(('unsupported angular target',residual))
                    vv=v[0]@C; dd=d[0]@C
                    if cap is not None:fac['own'][q,l,e]=don;fac['rv'][q,l,e]=vv;fac['rd'][q,l,e]=dd
                    block=np.array([L[q,l]*ev[e]*vv,D[q,l]*ev[e]*vv,L[q,l]*ev[e]*dd,L[q,l]*ed[e]*vv])
                    for j,oid in enumerate(don):
                        if int(oid) not in columns:columns[int(oid)]=np.zeros((len(p),4))
                        columns[int(oid)][q]+=block[:,j]
        row=self._pack(p,columns,{'family':'singleton' if np.all(self.profile[rid]==n) else 'ringwise','max_residual':worst,'min_rank':rank})
        if cap is not None:
            cap.append(PointFactors('ringwise',n,np.array(layers),np.array(fac['ei']),np.array(fac['ev']),np.array(fac['ed']),np.stack((L,D),axis=1),
                ring_owner=fac['own'],ring_value=fac['rv'],ring_derivative=fac['rd']))
        return row

    def _coupled(self,key,p,anchor,layers,rid,fixed_anchor):
        t=self.t;n=t.n;center=anchor[0]*np.array([np.cos(anchor[1]),np.sin(anchor[1])]);scale=max(t.g.dr,anchor[0]*t.g.dtheta)
        if fixed_anchor:
            ids=r.nearest(t.centers[2],anchor[2],4,t.g.eta_period)
            ei=np.tile(ids,(len(p),1)); ev,ed=np.array([r.eta_rows(t.centers[2][ids],q[2],t.g.eta_period,t.g.deta) for q in p]).transpose(1,0,2)
        else:ei,ev,ed=r.eta_plane_rows(t,p)
        columns={};worst=0.;minrank=15;maxlevel=0;maxextra=0;fallbacks=0;repairs=0
        B,Dr,Dt=r.planar(p,center,scale,r.EXP4)
        for kk in np.unique(ei):
            v=np.sum(ev*(ei==kk),axis=1);d=np.sum(ed*(ei==kk),axis=1)
            target=np.stack((B*v[:,None],Dr*v[:,None],Dt*v[:,None],B*d[:,None]),axis=1).reshape(-1,15)
            def fit(lo,hi):
                don=[]
                for rr in range(lo,hi+1):
                    owners,angles=self.ring(rr,kk);don.extend(owners[r.nearest(angles,anchor[1],min(7,len(owners)),2*np.pi)])
                don=np.asarray(don,int)
                A,U,root,C,ra,ru,_,_,res=r.fit(t,don,center,scale,target)
                return don,A,U,root,C,ra,ru,res
            def ring_fit():
                for level in range(4):
                    lo=max(0,int(rid.min())-level);hi=min(n-1,int(rid.max())+level)
                    don,A,U,root,C,ra,ru,res=fit(lo,hi)
                    C3,_,_=r.pinv(A[:,:10],root);CU3,_,_=r.pinv(U[:,:10],root)
                    res3=max(r.resid(target[:,:10],C3,A[:,:10]),r.resid(target[:,:10],CU3,U[:,:10]))
                    if res3<=1e-9:break
                else:raise RuntimeError(('unsupported point cubic',key,int(kk),res3))
                for extra in range(3):
                    don,A,U,root,C,ra,ru,res=fit(max(0,lo-extra),min(n-1,hi+extra))
                    if res<=1e-9:break
                else:raise RuntimeError(('unsupported point quartic',key,int(kk),res))
                return don,A,U,root,C,ra,ru,res,level,extra
            got=self._nearest28(kk,center,scale,target) if self.inner_support=="last_aggregate_nearest28" else None
            if got is None:
                if self.inner_support=="last_aggregate_nearest28":fallbacks+=1
                don,A,U,root,C,ra,ru,res,level,extra=ring_fit()
            else:
                don,A,U,root,C,ra,ru,res,repaired=got;level=extra=0;repairs+=repaired
            block=(target@C).reshape(len(p),4,-1)
            for j,oid in enumerate(don):
                if int(oid) not in columns:columns[int(oid)]=np.zeros((len(p),4))
                columns[int(oid)]+=block[:,:,j]
            worst=max(worst,res);minrank=min(minrank,ra,ru);maxlevel=max(maxlevel,level);maxextra=max(maxextra,extra)
        meta={'family':'coupled_quartic','max_residual':worst,'min_rank':minrank,'cubic_expansion':maxlevel,'quartic_expansion':maxextra}
        if self.inner_support=="last_aggregate_nearest28":meta.update(donor_policy='nearest28',nearest28_repairs=repairs,nearest28_fallbacks=fallbacks)
        return self._pack(p,columns,meta)

    def _plane_pool(self,kk):
        cache=self.__dict__.setdefault('_plane_pools',{})
        if int(kk) not in cache:
            n=self.t.n;cache[int(kk)]=np.unique(self.t.ro.reshape((n,)*3)[:,:,int(kk)])
        return cache[int(kk)]

    def _nearest28(self,kk,center,scale,target):
        """C2 donors on eta plane ``kk``: the ``C2_DONORS`` owners whose centroids are nearest ``center`` (distance in
        units of ``scale`` rounded to 12 digits, ties by owner id: Q's order). If that fit is not full rank or does not
        reproduce the quartic, Q's rank repair (up to two best single swaps from the nearest ``C2_POOL`` improving the
        weighted conditioning ratio). Returns ``None`` if still unsupported (the caller falls back to P's ring fit)."""
        t=self.t;pool=self._plane_pool(kk)
        delta=t.g.owner_centroid_xy[pool]-center
        order=np.lexsort((pool,np.round(np.linalg.norm(delta/scale,axis=1),12)))
        if len(order)<C2_DONORS:return None
        def good(f):return f[4]==15 and f[5]==15 and f[8]<=1e-9
        don=pool[order[:C2_DONORS]];f=r.fit(t,don,center,scale,target)
        if good(f):return don,f[0],f[1],f[2],f[3],f[4],f[5],f[8],0
        choices=order[:min(C2_POOL,len(order))];ids=pool[choices]
        A,U,root=r.fit(t,ids,center,scale,np.eye(15))[:3]
        def quality(sel):
            sa=np.linalg.svd(root[sel,None]*A[sel],compute_uv=False);su=np.linalg.svd(root[sel,None]*U[sel],compute_uv=False)
            return float(min(sa[-1]/sa[0],su[-1]/su[0]))
        selected=list(range(C2_DONORS))
        for _ in range(2):
            best=quality(selected);bestset=None
            for incoming in range(len(ids)):
                if incoming in selected:continue
                for outgoing in selected:
                    cand=sorted([z for z in selected if z!=outgoing]+[incoming]);q=quality(cand)
                    if round(q,14)>round(best,14):best=q;bestset=cand
            if bestset is None:break
            selected=bestset;don=ids[selected];f=r.fit(t,don,center,scale,target)
            if good(f):return don,f[0],f[1],f[2],f[3],f[4],f[5],f[8],1
        return None

    def side_rows(self,key,points):
        return tuple(row for row,_ in self._sides(key,points,False))

    def side_rows_with_factors(self,key,points):
        """``side_rows`` with each side as ``(PointRows, PointFactors or None)``; ``(None, None)`` for a missing side."""
        return self._sides(key,points,True)

    def _sides(self,key,points,capture):
        """Adjacent-cell anchored structured states; exterior radial side is None.

        Common rows are unchanged. Distinct angular/eta supports are selected about
        the adjacent cell centers, not about the target. Wall-reaching side states
        use the same prescribed trace lift as the common reconstruction.
        """
        axis,*ijk=map(int,key);n=self.t.n
        left=ijk.copy();left[axis]-=1;right=ijk.copy()
        result=[]
        for cell in (left,right):
            if axis==0 and not 0<=cell[0]<n:result.append((None,None))
            else:
                cell[1]%=n;cell[2]%=n
                cell=tuple(cell)
                result.append(self.rows_with_factors(cell,points,'cell',fixed_anchor=True) if capture
                              else (self.rows(cell,points,'cell',fixed_anchor=True),None))
        return tuple(result)
