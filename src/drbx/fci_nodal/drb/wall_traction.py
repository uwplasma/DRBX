"""Common-admittance total-traction wall law, local rows only."""
import numpy as np
from scipy import linalg as la
W=np.array([2.,1.5,1.5,1.,1/1836.]);Z=np.sqrt(10/3)
p=np.array([2.,1.,1.,0.,0.]);a=np.array([2.,1.71,1.,0.,0.])
Ev=np.array([[0.,0.,0.,1.,0.],[0.,0.,0.,1.,-1.]])
L=np.array([[1.,0.],[1.,-1.]])
Q=np.zeros((5,5));Q[0,4]=-1;Q[1,3]=2*.71/3;Q[1,4]=-2*1.71/3;Q[2,4]=-2/3;Q[3]=-p;Q[4]=-1836*a

def admittance(s,alpha):
    ev,V=la.eig(s*Q);V=V.real;Li=la.inv(V);C0=Li[ev.real>1e-10]
    N=np.eye(5)[:3];K0=la.solve(np.vstack([C0,N]),np.vstack([np.zeros((2,5)),N]))
    F=s*np.stack([p,-a]);Y0=(Ev@K0)@F.T@la.inv(F@F.T)
    Y=(1-alpha)*Y0+alpha*np.diag([1/Z,0.])
    return Y,F

def hit_map(sg,ell,alpha,chi):
    """N U=N r; v=Y_alpha(e-t), with physical ion and weighted electron stress."""
    sg=np.asarray(sg);ell=np.asarray(ell);chi=np.asarray(chi)
    assert 0<=alpha<=1 and np.all(ell>0)
    K=np.broadcast_to(np.eye(5),(len(sg),5,5)).copy()
    for s in [-1,1]:
        ids=sg==s;Y,F=admittance(s,alpha)
        wd=chi[3:5]*W[3:5]/ell[ids,None]
        G=np.einsum('ai,ha,aj->hij',L,wd,L)
        load=np.einsum('ai,ha,aj->hij',L,wd,np.eye(5)[3:])
        v=np.linalg.solve(np.eye(2)[None]+Y@G,Y@(F[None]+load))
        K[ids,3:]=L@v
    return K

