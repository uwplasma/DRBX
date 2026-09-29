"""Frozen bounded support policies; portable host-only setup algebra."""
import numpy as np
from scipy.linalg import qr
from . import primitives as r
class Layers:
    def __init__(self,t):
        self.t=t;self.rings={};self.cache={};self.profile=np.array([len(np.unique(t.ro.reshape((t.n,)*3)[i,:,0])) for i in range(t.n)])
    def ring(self,i,k):
        tag=int(i),int(k)
        if tag not in self.rings:
            ids=np.unique(self.t.ro.reshape((self.t.n,)*3)[i,:,k]);angles=[]
            for oid in ids:
                mm=r.members(self.t,oid);a=self.t.pts[mm,1]
                angles.append(a[0] if len(mm)==1 else np.angle(np.mean(np.exp(1j*a)))%(2*np.pi))
            self.rings[tag]=ids,np.array(angles)
        return self.rings[tag]
    def pack(self,p,col,meta):
        ids=np.array(sorted(col),int);a=np.stack([col[o] for o in ids],axis=-1)
        return ids,a[:,0],a[:,1:],meta
    def rows(self,ijk,p):
        n=self.t.n;i,j,k=map(int,ijk);center=self.t.pts[(i*n+j)*n+k]
        layers=np.arange(i-1,i+3);rid=np.where(layers<0,-layers-1,layers)
        ei=(k+np.arange(-2,3))%n
        ev,ed=np.array([r.eta_rows(self.t.centers[2][ei],z[2],2*np.pi,2*np.pi/n) for z in p]).transpose(1,0,2)
        if self.profile[rid].min()<7:return self.coupled(center,layers,rid,ei,ev,ed,p)
        return self.tensor(center,layers,rid,ei,ev,ed,p)
    def tensor(self,center,layers,rid,ei,ev,ed,p):
        t=self.t;n=t.n;L,D=r.rows((layers+.5)/n,p[:,0]);col={};worst=0.;condition=0.;minrank=7
        for l,rr in enumerate(rid):
            for e,k in enumerate(ei):
                owners,angles=self.ring(rr,k);pick=r.nearest(angles,center[1],7,2*np.pi);don=owners[pick];nodes=angles[pick]
                tag=('ring',int(rr),int(k),*map(int,don))
                if tag not in self.cache:
                    A=[];U=[]
                    for o in don:
                        mm=r.members(t,o);b=r.cardinal(nodes,t.pts[mm,1])[0]
                        A.append((t.rv[mm]/t.vol[o])@b);U.append(b.mean(axis=0))
                    A=np.array(A);U=np.array(U);C,ra,ca=r.pinv(A,np.ones(7));CU,ru,cu=r.pinv(U,np.ones(7))
                    self.cache[tag]=A,U,C,CU,ra,ru,max(ca,cu)
                A,U,C,CU,ra,ru,cond=self.cache[tag]
                v,d=r.cardinal(nodes,p[:,1]);target=np.vstack((v,d))
                residual=max(r.resid(target,C,A),r.resid(target,CU,U));worst=max(worst,residual);condition=max(condition,cond);minrank=min(minrank,ra,ru)
                assert residual<1e-9 and min(ra,ru)==7,('ring reproduction',rr,k,residual)
                vv=v@C;dd=d@C
                block=np.stack((L[:,l,None]*ev[:,e,None]*vv,D[:,l,None]*ev[:,e,None]*vv,L[:,l,None]*ev[:,e,None]*dd,L[:,l,None]*ed[:,e,None]*vv),axis=1)
                for z,o in enumerate(don):
                    if int(o) not in col:col[int(o)]=np.zeros((len(p),4))
                    col[int(o)]+=block[:,:,z]
        return self.pack(p,col,dict(family='singleton' if np.all(self.profile[rid]==n) else 'ringwise',reproduction=worst,condition=condition,minrank=minrank,cubic_expansion=0,quartic_expansion=0))
    def coupled(self,center,layers,rid,ei,ev,ed,p):
        t=self.t;n=t.n;xy=center[0]*np.array([np.cos(center[1]),np.sin(center[1])]);scale=max(1/n,center[0]*2*np.pi/n)
        B,Dr,Dt=r.planar(p,xy,scale,r.EXP4);col={};worst=0.;condition=0.;minrank=15;levels=[]
        for e,k in enumerate(ei):
            target=np.stack((B*ev[:,e,None],Dr*ev[:,e,None],Dt*ev[:,e,None],B*ed[:,e,None]),axis=1).reshape(-1,15)
            def fit(lo,hi):
                don=[]
                for rr in range(lo,hi+1):
                    ids,angles=self.ring(rr,k);don.extend(ids[r.nearest(angles,center[1],min(7,len(ids)),2*np.pi)])
                don=np.array(don,int)
                A,U,root,C,ra,ru,ca,cu,res=r.fit(t,don,xy,scale,target)
                return don,A,U,root,C,ra,ru,max(ca,cu),res
            for level in range(4):
                lo=max(0,int(rid.min())-level);hi=min(n-1,int(rid.max())+level)
                don,A,U,root,C,ra,ru,cond,res=fit(lo,hi)
                C3,_,_=r.pinv(A[:,:10],root);CU3,_,_=r.pinv(U[:,:10],root)
                res3=max(r.resid(target[:,:10],C3,A[:,:10]),r.resid(target[:,:10],CU3,U[:,:10]))
                if res3<1e-9:break
            else:raise RuntimeError(('cubic reproduction',center,k,res3))
            for extra in range(3):
                don,A,U,root,C,ra,ru,cond,res=fit(max(0,lo-extra),min(n-1,hi+extra))
                if res<1e-9:break
            else:raise RuntimeError(('quartic reproduction',center,k,res))
            block=(target@C).reshape(len(p),4,-1)
            for z,o in enumerate(don):
                if int(o) not in col:col[int(o)]=np.zeros((len(p),4))
                col[int(o)]+=block[:,:,z]
            worst=max(worst,res);condition=max(condition,cond);minrank=min(minrank,ra,ru);levels.append([level,extra,len(don)])
        return self.pack(p,col,dict(family='coupled_quartic',reproduction=worst,condition=condition,minrank=minrank,cubic_expansion=max(z[0] for z in levels),quartic_expansion=max(z[1] for z in levels),plane_support=levels))
class CompactBase(Layers):

    def __init__(self, t, policy):
        super().__init__(t)
        self.policy = policy
        self.planes = {}
        self.fits = {}
        self.centroid = np.column_stack([np.bincount(t.ro, weights=t.rv * t.xy[:, a], minlength=len(t.vol)) / t.vol for a in (0, 1)])

    def plane(self, k):
        if int(k) not in self.planes:
            self.planes[int(k)] = np.unique(self.t.ro.reshape((self.t.n,) * 3)[:, :, k])
        return self.planes[int(k)]
class Compact(CompactBase):

    def repair(self, pool, order, xy, scale, picks):
        choices = order[:40]
        ids = pool[choices]
        A, U, root, _, _, _, _, _, _ = r.fit(self.t, ids, xy, scale, np.eye(15))

        def quality(selected):
            wa = root[selected, None] * A[selected]
            wu = root[selected, None] * U[selected]
            sa = np.linalg.svd(wa, compute_uv=False)
            su = np.linalg.svd(wu, compute_uv=False)
            return float(min(sa[-1] / sa[0], su[-1] / su[0]))
        selected = list(range(28))
        changes = []
        for step in range(2):
            initial = quality(selected)
            best = initial
            bestset = None
            change = None
            for incoming in range(40):
                if incoming in selected:
                    continue
                for position, outgoing in enumerate(selected):
                    candidate = sorted([z for z in selected if z != outgoing] + [incoming])
                    q = quality(candidate)
                    if round(q, 14) > round(best, 14):
                        best = q
                        bestset = candidate
                        change = dict(removed=int(ids[outgoing]), added=int(ids[incoming]), before_score=initial, after_score=q)
            if bestset is None:
                break
            selected = bestset
            changes.append(change)
            chosen = choices[selected]
            fit = r.fit(self.t, pool[chosen], xy, scale, np.eye(15))
            if fit[4] == 15 and fit[5] == 15 and (fit[8] < 1e-09):
                return (chosen, changes)
        return (choices[selected], changes)

    def rows(self, ijk, points):
        t = self.t
        n = t.n
        i, j, k = map(int, ijk)
        center = t.pts[(i * n + j) * n + k]
        xy = center[0] * np.array([np.cos(center[1]), np.sin(center[1])])
        scale = max(1 / n, center[0] * 2 * np.pi / n)
        ei = (k + np.arange(-2, 3)) % n
        ev, ed = np.array([r.eta_rows(t.centers[2][ei], z[2], 2 * np.pi, 2 * np.pi / n) for z in points]).transpose(1, 0, 2)
        B, Dr, Dt = r.planar(points, xy, scale, r.EXP4)
        col = {}
        infos = []
        for e, kk in enumerate(ei):
            tag = (i, j, int(kk))
            if tag not in self.fits:
                pool = self.plane(kk)
                delta = self.centroid[pool] - xy
                distance = np.round(np.linalg.norm(delta / scale, axis=1), 12)
                order = np.lexsort((pool, distance))
                sector = np.floor(np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi) / (np.pi / 4)).astype(int)
                levels = [28 if self.policy == 'repair28' else 40]
                exchanges = []
                initial_rank = None
                for level, amount in enumerate(levels):
                    if self.policy == 'sector32':
                        picks = np.concatenate([order[sector[order] == s][:amount] for s in range(8)])
                        if any((np.sum(sector[picks] == s) != amount for s in range(8))):
                            continue
                    else:
                        picks = order[:amount]
                    don = pool[picks]
                    A, U, root, C, ra, ru, ca, cu, res = r.fit(t, don, xy, scale, np.eye(15))
                    initial_rank = [ra, ru]
                    if self.policy == 'repair28' and (not (ra == 15 and ru == 15 and (res < 1e-09))):
                        picks, exchanges = self.repair(pool, order, xy, scale, picks)
                        don = pool[picks]
                        A, U, root, C, ra, ru, ca, cu, res = r.fit(t, don, xy, scale, np.eye(15))
                    if ra == 15 and ru == 15 and (res < 1e-09):
                        break
                else:
                    raise RuntimeError(('fixed-count failure', self.policy, ijk.tolist(), int(kk), 'rank', ra, ru, 'residual', res, 'condition', ca, cu))
                cov = delta[picks].T @ delta[picks] / len(picks)
                eigen = np.linalg.eigvalsh(cov)
                info = dict(exchanges=exchanges, initial_rank=initial_rank, donors=don.tolist(), rank=min(ra, ru), condition=max(ca, cu), reproduction=res, expansion=level, aspect=float(np.sqrt(eigen[-1] / eigen[0])), max_radius=float(distance[picks].max()), sector_counts=np.bincount(sector[picks], minlength=8).tolist())
                self.fits[tag] = (don, A, C, info)
            don, A, C, info = self.fits[tag]
            target = np.stack((B * ev[:, e, None], Dr * ev[:, e, None], Dt * ev[:, e, None], B * ed[:, e, None]), axis=1).reshape(-1, 15)
            assert r.resid(target, C, A) < 1e-09
            block = (target @ C).reshape(len(points), 4, -1)
            for z, oid in enumerate(don):
                if int(oid) not in col:
                    col[int(oid)] = np.zeros((len(points), 4))
                col[int(oid)] += block[:, :, z]
            infos.append(info)
        return self.pack(points, col, dict(family=self.policy, reproduction=max((z['reproduction'] for z in infos)), condition=max((z['condition'] for z in infos)), minrank=min((z['rank'] for z in infos)), cubic_expansion=0, quartic_expansion=max((z['expansion'] for z in infos)), planes=infos))
def good(ft):return ft[4]==ft[5]==15 and ft[8]<1e-9
def quality(A,U,root):
    a=np.linalg.svd(root[:,None]*A,compute_uv=False);u=np.linalg.svd(root[:,None]*U,compute_uv=False)
    return float(min(a[-1]/a[0],u[-1]/u[0]))
class Repaired(Compact):
    def __init__(self,t):super().__init__(t,'repair28');self.events=[]
    def repair(self,pool,order,xy,scale,picks):
        chosen,changes=super().repair(pool,order,xy,scale,picks)
        if good(r.fit(self.t,pool[chosen],xy,scale,np.eye(15))):return chosen,changes
        t=self.t;n=t.n;u=np.linalg.norm(xy);i=int(np.argmin(abs(t.centers[0]-u)));theta=float(np.arctan2(xy[1],xy[0])%(2*np.pi));k=int(round((t.pts[r.members(t,int(pool[0]))[0],2]-t.centers[2][0])/t.g.deta))%n
        poolpos={int(o):z for z,o in enumerate(pool)}
        for radius,angular in ((2,7),(3,9),(4,11)):
            ids=set(map(int,pool[order[:40]]))
            for rr in range(max(0,i-radius),min(n-1,i+radius)+1):
                oo,aa=self.ring(rr,k);ids.update(map(int,oo[r.nearest(aa,theta,min(angular,len(oo)),2*np.pi)]))
            ids=np.array(sorted(ids));dist=np.round(np.linalg.norm((self.centroid[ids]-xy)/scale,axis=1),12);near=np.lexsort((ids,dist));ids=ids[near]
            ft=r.fit(t,ids,xy,scale,np.eye(15));A,U,root=ft[:3]
            pivots=[qr((root[:,None]*M).T,mode='economic',pivoting=True)[2][:15] for M in (A,U)]
            candidates=[]
            for name,pv in zip(('weighted','uniform'),pivots):
                take=list(map(int,pv));take+= [z for z in range(len(ids)) if z not in take][:28-len(take)];take=np.array(sorted(take))
                sub=r.fit(t,ids[take],xy,scale,np.eye(15))
                if good(sub):candidates.append((quality(sub[0],sub[1],sub[2]),name,take,sub))
            if candidates:
                _,name,take,sub=max(candidates,key=lambda x:x[0]);don=ids[take]
            else:
                take=sorted(set(map(int,np.concatenate(pivots))));take+= [z for z in range(len(ids)) if z not in take][:max(0,28-len(take))];don=ids[take];sub=r.fit(t,don,xy,scale,np.eye(15));name='pivot_union'
                if not good(sub):don=ids;sub=ft;name='full_pool'
            if good(sub):
                event=dict(policy='balanced_QR',ring=i,eta=k,pool_size=len(ids),radius=radius,angular=angular,selection=name,donor_count=len(don),fallback=len(don)>28,condition=max(sub[6:8]),reproduction=sub[8],donors=don.tolist())
                self.events.append(event);return np.array([poolpos[int(o)] for o in don]),changes+[event]
        raise RuntimeError(('balanced inner pool failure',i,k))

class Hybrid:
    def __init__(self,t):
        self.inner=Repaired(t);self.outer=Layers(t);self.last=int(np.flatnonzero(self.outer.profile==t.n)[0])-1
    def rows(self,ijk,points):
        return (self.inner if ijk[0]<=self.last else self.outer).rows(ijk,points)
    def clear(self):
        self.inner.fits.clear();self.inner.cache.clear();self.inner.events.clear();self.outer.cache.clear()
