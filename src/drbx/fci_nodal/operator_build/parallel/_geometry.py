import numpy as np

TAU = 2*np.pi


def metric(ev,q,derivatives=True):
    B,M,A,_=ev(q);J=np.linalg.det(A);b=B/M[:,None]
    ans=dict(B=B,M=M,J=J,b=b)
    if not derivatives:return ans
    db=np.zeros((len(q),3,3));dm=np.zeros((len(q),3));div=np.zeros(len(q));eps=2.5e-5
    for axis in range(3):
        for shift,w in zip([-2,-1,1,2],np.array([1,-8,8,-1])/(12*eps)):
            qq=q.copy();qq[:,axis]+=shift*eps;Bp,Mp,Ap,_=ev(qq)
            db[:,axis]+=w*Bp/Mp[:,None];dm[:,axis]+=w*Mp
            div+=w*np.linalg.det(Ap)*Bp[:,axis]
    ans.update(along_db=np.einsum('pa,pai->pi',b,db),along_logmag=np.einsum('pa,pa->p',b,dm)/M,divB=div/J)
    return ans
