"""Geometry-only point functionals extending the P07 structured reconstruction.

Campaign infrastructure, not a production operator. Owner observations are physical
volume weighted raw member-center values. All boundary data enter at application.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from p07_combined_global import kernels as r


def load_context(n, input_root):
    r.configure(input_root)
    return r.load(n)


@dataclass
class PointRows:
    donor_ids: np.ndarray
    value: np.ndarray
    gradient: np.ndarray
    boundary_conditioned: bool
    trace_donor_points: np.ndarray
    trace_target_points: np.ndarray
    diagnostics: dict

    def apply(self, owner_values, trace=None):
        data = np.asarray(owner_values)[self.donor_ids]
        if data.ndim != 2:
            raise ValueError('owner_values must have shape (owners, fields)')
        if self.boundary_conditioned:
            if trace is None:
                raise ValueError('a prescribed Dirichlet trace callback is required')
            gd, _ = trace(self.trace_donor_points)
            gt, dgt = trace(self.trace_target_points)
            data = data - gd
        value = self.value @ data
        gradient = np.einsum('qad,df->qaf', self.gradient, data)
        if self.boundary_conditioned:
            value += gt
            gradient[:, 1:] += dgt[:, 1:]
        return value, gradient

    def apply_value(self, owner_values, trace=None):
        """The same value action without an unused gradient contraction."""
        data = np.asarray(owner_values)[self.donor_ids]
        if data.ndim != 2:
            raise ValueError('owner_values must have shape (owners, fields)')
        if self.boundary_conditioned:
            if trace is None:
                raise ValueError('a prescribed Dirichlet trace callback is required')
            gd, _ = trace(self.trace_donor_points)
            gt, _ = trace(self.trace_target_points)
            data = data - gd
        value = self.value @ data
        if self.boundary_conditioned:
            value += gt
        return value


class StructuredReconstruction:
    def __init__(self, t):
        self.t = t
        self.rings = {}
        self.fits = {}
        self.profile = np.array([len(np.unique(t.ro.reshape((t.n,)*3)[i,:,0])) for i in range(t.n)])
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
            return PointRows(z['donor_ids'],v,z['gradient_map'],z['boundary_conditioned'],z['trace_donor_points'],z['trace_target_points'],{'family':z['kind'],'max_residual':0.})
        layers=np.arange(i-2,i+2) if axis==0 else np.arange(i-1,i+3)
        rid=np.where(layers<0,-layers-1,layers)
        if np.min(self.profile[rid])<7:
            return self._coupled(key,p,anchor,layers,rid,fixed_anchor)
        if np.all(self.profile[rid]==n):
            return self._singleton(p,anchor,layers,rid,fixed_anchor)
        return self._tensor(p,anchor,layers,rid,fixed_anchor)

    def _pack(self,p,columns,diagnostics):
        ids=np.array(sorted(columns),int)
        a=np.stack([columns[k] for k in ids],axis=-1)
        return PointRows(ids,a[:,0],a[:,1:],False,np.empty((0,3)),p.copy(),diagnostics)

    def _singleton(self,p,anchor,layers,rid,fixed_anchor):
        t=self.t;n=t.n;L,D=r.rows((layers+.5)/n,p[:,0]);allids=[];blocks=[]
        for q,point in enumerate(p):
            ta=anchor[1] if fixed_anchor else point[1]
            ea=anchor[2] if fixed_anchor else point[2]
            ti=self._nearest(1,float(ta));tv,td=self._basis(1,tuple(ti),float(point[1]))
            ei=self._nearest(2,float(ea));ev,ed=self._basis(2,tuple(ei),float(point[2]))
            theta=(ti[None,:]+np.where(layers<0,n//2,0)[:,None])%n
            raw=(rid[:,None,None]*n+theta[:,None,:])*n+ei[None,:,None]
            allids.append(t.ro[raw].ravel())
            blocks.append(np.array([L[q,:,None,None]*ev[None,:,None]*tv[None,None,:],
                D[q,:,None,None]*ev[None,:,None]*tv[None,None,:],
                L[q,:,None,None]*ev[None,:,None]*td[None,None,:],
                L[q,:,None,None]*ed[None,:,None]*tv[None,None,:]]).reshape(4,-1))
        ids=np.unique(allids);a=np.zeros((len(p),4,len(ids)))
        for q,don in enumerate(allids):
            for k in range(4):np.add.at(a[q,k],np.searchsorted(ids,don),blocks[q][k])
        return PointRows(ids,a[:,0],a[:,1:],False,np.empty((0,3)),p.copy(),{'family':'singleton','max_residual':0.})

    def _tensor(self,p,anchor,layers,rid,fixed_anchor):
        t=self.t; n=t.n
        L,D=r.rows((layers+.5)/n,p[:,0]); columns={}; worst=0.; rank=7
        for q,point in enumerate(p):
            ea=anchor[2] if fixed_anchor else point[2]
            ei=r.nearest(t.centers[2],ea,4,t.g.eta_period)
            ev,ed=r.eta_rows(t.centers[2][ei],point[2],t.g.eta_period,t.g.deta)
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
                    block=np.array([L[q,l]*ev[e]*vv,D[q,l]*ev[e]*vv,L[q,l]*ev[e]*dd,L[q,l]*ed[e]*vv])
                    for j,oid in enumerate(don):
                        if int(oid) not in columns:columns[int(oid)]=np.zeros((len(p),4))
                        columns[int(oid)][q]+=block[:,j]
        return self._pack(p,columns,{'family':'singleton' if np.all(self.profile[rid]==n) else 'ringwise','max_residual':worst,'min_rank':rank})

    def _coupled(self,key,p,anchor,layers,rid,fixed_anchor):
        t=self.t;n=t.n;center=anchor[0]*np.array([np.cos(anchor[1]),np.sin(anchor[1])]);scale=max(t.g.dr,anchor[0]*t.g.dtheta)
        if fixed_anchor:
            ids=r.nearest(t.centers[2],anchor[2],4,t.g.eta_period)
            ei=np.tile(ids,(len(p),1)); ev,ed=np.array([r.eta_rows(t.centers[2][ids],q[2],t.g.eta_period,t.g.deta) for q in p]).transpose(1,0,2)
        else:ei,ev,ed=r.eta_plane_rows(t,p)
        columns={};worst=0.;minrank=15;maxlevel=0;maxextra=0
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
            block=(target@C).reshape(len(p),4,-1)
            for j,oid in enumerate(don):
                if int(oid) not in columns:columns[int(oid)]=np.zeros((len(p),4))
                columns[int(oid)]+=block[:,:,j]
            worst=max(worst,res);minrank=min(minrank,ra,ru);maxlevel=max(maxlevel,level);maxextra=max(maxextra,extra)
        return self._pack(p,columns,{'family':'coupled_quartic','max_residual':worst,'min_rank':minrank,'cubic_expansion':maxlevel,'quartic_expansion':maxextra})

    def side_rows(self,key,points):
        """Adjacent-cell anchored structured states; exterior radial side is None.

        Common rows are unchanged. Distinct angular/eta supports are selected about
        the adjacent cell centers, not about the target. Wall-reaching side states
        use the same prescribed trace lift as the common reconstruction.
        """
        axis,*ijk=map(int,key);n=self.t.n
        left=ijk.copy();left[axis]-=1;right=ijk.copy()
        result=[]
        for cell in (left,right):
            if axis==0 and not 0<=cell[0]<n:result.append(None)
            else:
                cell[1]%=n;cell[2]%=n
                result.append(self.rows(tuple(cell),points,'cell',fixed_anchor=True))
        return tuple(result)
