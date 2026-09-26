"""Frozen P07N axis-regular MMS fields; independent of numerical row assembly."""
import numpy as np
from scipy.special import expit

NAMES=('smooth_nonzero_control','zero_normal_m1','prescribed_nonzero_m1','heldout_zero_normal_m2m3','constant')
C_STEP=1e-4

def normal(ref,q):
    m=ref._metric(np.asarray(q))['gcontra']
    return m[:,0,:]/np.sqrt(m[:,0,0])[:,None]

def cutoff(u):
    u=np.asarray(u);r=np.zeros_like(u);rp=np.zeros_like(u);rpp=np.zeros_like(u)
    inside=(u>.25)&(u<.75)
    t=2*(u[inside]-.25)
    z=-1/t+1/(1-t)
    w=expit(z)
    zp=1/t**2+1/(1-t)**2
    zpp=-2/t**3+2/(1-t)**3
    r[inside]=w
    rp[inside]=2*w*(1-w)*zp
    rpp[inside]=4*w*(1-w)*((1-2*w)*zp*zp+zpp)
    r[u>=.75]=1
    return r,rp,rpp

def base(q,kind,period):
    q=np.asarray(q);u,th,eta=q.T;om=2*np.pi/period;z=om*eta
    v=np.empty(len(q));g=np.zeros((len(q),3));H=np.zeros((len(q),3,3))
    if kind==1:
        ct=np.cos(th);st=np.sin(th);cz=np.cos(z);sz=np.sin(z)
        v[:]=1+.12*u*ct*cz+.04*u*u
        g[:,0]=.12*ct*cz+.08*u;g[:,1]=-.12*u*st*cz;g[:,2]=-.12*u*ct*sz*om
        H[:,0,0]=.08;H[:,0,1]=H[:,1,0]=-.12*st*cz
        H[:,0,2]=H[:,2,0]=-.12*ct*sz*om
        H[:,1,1]=-.12*u*ct*cz
        H[:,1,2]=H[:,2,1]=.12*u*st*sz*om
        H[:,2,2]=-.12*u*ct*cz*om*om
    else:
        s2=np.sin(2*th);c2=np.cos(2*th);s3=np.sin(3*th);c3=np.cos(3*th);s2z=np.sin(2*z);c2z=np.cos(2*z);sz=np.sin(z);cz=np.cos(z)
        A=.08*u*u*s2*s2z;B=.03*u**3*c3*cz;v[:]=1+A+B
        g[:,0]=.16*u*s2*s2z+.09*u*u*c3*cz
        g[:,1]=.16*u*u*c2*s2z-.09*u**3*s3*cz
        g[:,2]=.16*om*u*u*s2*c2z-.03*om*u**3*c3*sz
        H[:,0,0]=.16*s2*s2z+.18*u*c3*cz
        H[:,0,1]=H[:,1,0]=.32*u*c2*s2z-.27*u*u*s3*cz
        H[:,0,2]=H[:,2,0]=.32*om*u*s2*c2z-.09*om*u*u*c3*sz
        H[:,1,1]=-.32*u*u*s2*s2z-.27*u**3*c3*cz
        H[:,1,2]=H[:,2,1]=.32*om*u*u*c2*c2z+.09*om*u**3*s3*sz
        H[:,2,2]=-.32*om*om*u*u*s2*s2z-.03*om*om*u**3*c3*cz
    return v,g,H

def datum(q,name,period):
    q=np.asarray(q)
    if name!='prescribed_nonzero_m1':return np.zeros(len(q))
    th=q[:,1];z=2*np.pi*q[:,2]/period
    # Physical derivative in field units per metre (L_ref = 1 m).
    return .10*(1+.3*np.cos(th)*np.cos(z)+.2*np.sin(2*th)*np.sin(z))

def correction(ref,q,name,period):
    wall=np.asarray(q).copy();wall[:,0]=1
    kind=2 if name=='heldout_zero_normal_m2m3' else 1
    a=normal(ref,wall);_,g,_=base(wall,kind,period)
    return (datum(wall,name,period)-np.einsum('qa,qa->q',a,g))/a[:,0]

def shifted(q,axis,h,period):
    p=np.asarray(q).copy();p[:,axis]+=h
    if axis==1:p[:,axis]%=2*np.pi
    elif axis==2:p[:,axis]%=period
    return p

def evaluate(ref,q,name,period,step=C_STEP,derivatives=True):
    q=np.atleast_2d(np.asarray(q,dtype=float));u=q[:,0]
    if name=='constant':
        return np.ones(len(q)),np.zeros((len(q),3)),np.zeros((len(q),3,3))
    kind=2 if name=='heldout_zero_normal_m2m3' else 1
    v,g,H=base(q,kind,period)
    if name=='smooth_nonzero_control':return v,g,H
    r,rp,rpp=cutoff(u);active=u>.25
    if not np.any(active):return v,g,H
    qa=q[active];ua=u[active]
    C=correction(ref,qa,name,period)
    K=(ua-1)*r[active];Kp=r[active]+(ua-1)*rp[active];Kpp=2*rp[active]+(ua-1)*rpp[active]
    v[active]+=K*C
    if not derivatives:return v,g,H
    Ct=[];Ctt=[]
    for axis in (1,2):
        cp=correction(ref,shifted(qa,axis,step,period),name,period)
        cm=correction(ref,shifted(qa,axis,-step,period),name,period)
        Ct.append((cp-cm)/(2*step))
        Ctt.append((cp-2*C+cm)/step**2)
    cpp=correction(ref,shifted(shifted(qa,1,step,period),2,step,period),name,period)
    cpm=correction(ref,shifted(shifted(qa,1,step,period),2,-step,period),name,period)
    cmp=correction(ref,shifted(shifted(qa,1,-step,period),2,step,period),name,period)
    cmm=correction(ref,shifted(shifted(qa,1,-step,period),2,-step,period),name,period)
    Ccross=(cpp-cpm-cmp+cmm)/(4*step*step)
    g[active,0]+=Kp*C
    H[active,0,0]+=Kpp*C
    for i,axis in enumerate((1,2)):
        g[active,axis]+=K*Ct[i]
        H[active,0,axis]+=Kp*Ct[i];H[active,axis,0]+=Kp*Ct[i]
        H[active,axis,axis]+=K*Ctt[i]
    H[active,1,2]+=K*Ccross;H[active,2,1]+=K*Ccross
    return v,g,H
