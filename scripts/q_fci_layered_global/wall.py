"""Prescribed D and physical-normal N wall rows, fixed five-plane eta."""
import numpy as np
from . import primitives as r
from .fields import field
def check(key,value):
    if not np.all(np.isfinite(value)) or np.max(np.abs(value))>1e-8:raise ValueError((key,value))
def normal(ctx,pts):
    J=np.asarray(ctx['evaluator']._position_and_jacobian(pts)[1]);inv=np.linalg.inv(J);g=np.einsum('qik,qjk->qij',inv,inv);a=g[:,0,:]/np.sqrt(g[:,0,0,None])
    physical=np.einsum('qij,qj->qi',J,a);check('normal_unit',np.linalg.norm(physical,axis=1)-1);check('normal_outward',np.minimum(np.einsum('qi,qi->q',physical,J[:,:,0]),0))
    return a
class Wall:
    def __init__(self,ctx,t,i,j,k):
        self.ctx=ctx;self.t=t;self.n=t.n;self.anchor=t.pts[(i*t.n+j)*t.n+k];n=t.n
        self.ti=r.nearest(t.centers[1],self.anchor[1],7,2*np.pi);self.ei=(k+np.arange(-2,3))%n
        self.theta=t.centers[1][self.ti];self.eta=t.centers[2][self.ei];self.radial=np.r_[0.,-.5,-1.5,-2.5,-3.5]
        self.raw=((np.arange(n-1,n-5,-1)[:,None,None]*n+self.ti[None,:,None])*n+self.ei[None,None,:]).ravel();self.ids=t.ro[self.raw]
        assert len(set(self.ids))==140 and all(len(r.members(t,o))==1 for o in self.ids)
        self.donorpts=t.pts[self.raw];self.wall=np.column_stack((np.ones(35),np.repeat(self.theta,5),np.tile(self.eta,7)))
        self.a=normal(ctx,self.wall);_,d0=r.rows(self.radial,[0.]);td=np.array([r.theta_rows(self.theta,x)[1] for x in self.theta]);ed=np.array([r.eta_rows(self.eta,x,2*np.pi,t.g.deta)[1] for x in self.eta])
        M=np.diag(self.a[:,0]*d0[0,0]*n)+self.a[:,1,None]*np.kron(td,np.eye(5))+self.a[:,2,None]*np.kron(np.eye(7),ed)
        self.condition=float(np.linalg.cond(M));assert self.condition<1e8
        C=np.hstack([np.diag(self.a[:,0]*dl*n) for dl in d0[0,1:]])
        self.response=np.linalg.solve(M,-C);self.inverse=np.linalg.solve(M,np.eye(35));self.M=M;self.C=C
        self.solve_residual=float(max(abs(M@self.response+C).max(),abs(M@self.inverse-np.eye(35)).max()))
    def basis(self,points):
        lv,ld=r.rows(self.radial,(points[:,0]-1)*self.n);ld*=self.n
        tv,td=np.array([r.theta_rows(self.theta,x) for x in points[:,1]]).transpose(1,0,2)
        ev,ed=np.array([r.eta_rows(self.eta,x,2*np.pi,self.t.g.deta) for x in points[:,2]]).transpose(1,0,2)
        def prod(a,b,c):return (a[:,:,None,None]*b[:,None,:,None]*c[:,None,None,:]).reshape(len(points),-1)
        F=prod(lv,tv,ev);G=np.stack((prod(ld,tv,ev),prod(lv,td,ev),prod(lv,tv,ed)),axis=1)
        return F[:,:35],F[:,35:],G[:,:,:35],G[:,:,35:]
    def maps(self,points):
        W,I,GW,GI=self.basis(points)
        return I+W@self.response,W@self.inverse,GI+GW@self.response,GW@self.inverse
    def apply(self,points,name,kind):
        d=field(name,self.donorpts)[0];wv,wg=field(name,self.wall);W,I,GW,GI=self.basis(points)
        if kind=='N':
            g=np.einsum('si,si->s',self.a,wg);V,B,D,E=self.maps(points);return V@d+B@g,np.einsum('sid,d->si',D,d)+np.einsum('sib,b->si',E,g)
        if kind=='Dnodes':return I@d+W@wv,np.einsum('sid,d->si',GI,d)+np.einsum('sib,b->si',GW,wv)
        wallq=points.copy();wallq[:,0]=1;v,g=field(name,wallq);res=d-np.tile(wv,4)
        grad=np.einsum('sid,d->si',GI,res);grad[:,1:]+=g[:,1:]
        return I@res+v,grad
