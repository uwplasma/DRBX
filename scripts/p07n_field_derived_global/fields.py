"""P07N field-derived axis-regular fields; independent of numerical row assembly.

Nonzero physical-normal Neumann fields whose wall data come from the field
itself: g_N = a.grad_x f at the wall, with a = normal(ref,q) the same
outward physical-normal-coefficient formula as the frozen
scripts/p07_neumann_global/fields.py. There is no finite-difference wall
correction: every field and its gradient/Hessian are fully analytic, so
`normal_data` is exact by construction rather than reconstructed from a
zero-normal manufactured solution.

field_b1 and heldout_field_b2 are the frozen `base(q,1,period)` and
`base(q,2,period)` formulas, copied verbatim. field_e3 and field_e12 are the
`_trace_eta` wall-trace family from
work/p07n_compatible_fields_20260927/fields_v2.py (design screen, not
tracked), copied verbatim, at theta-gradient strengths 0.03 and 0.12.
constant is the field 1 with zero derivatives.
"""
import numpy as np

NAMES=('field_b1','field_e3','field_e12','heldout_field_b2','constant')


def normal(ref,q):
    m=ref._metric(np.asarray(q))['gcontra']
    return m[:,0,:]/np.sqrt(m[:,0,0])[:,None]


def base(q,kind,period):
    """Frozen P07N base(q,kind,period); kind=1 -> field_b1, kind=2 -> heldout_field_b2."""
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


def _trace_eta(u,th,z,om,theta_amp):
    """1 + 0.10 u^2 cos z + 0.12 u(1-u^2) cos(th) cos z + theta_amp*u*cos(th)*cos(z+0.3).

    Copied verbatim from work/p07n_compatible_fields_20260927/fields_v2.py.
    The first two terms give a wall trace varying only in eta (u(1-u^2)
    vanishes at u=1) while keeping interior theta structure; theta_amp sets
    the wall theta-gradient. All terms are polynomials in (x,y)=u(cos,sin)th
    times eta modes: axis-regular.
    """
    n=len(u);v=np.empty(n);g=np.zeros((n,3));H=np.zeros((n,3,3))
    c,s=np.cos(th),np.sin(th);cz,sz=np.cos(z),np.sin(z)
    cz3,sz3=np.cos(z+.3),np.sin(z+.3)
    A,B,T=.10,.12,theta_amp
    p=u-u**3;p1=1-3*u*u;p2=-6*u          # u(1-u^2) and derivatives
    v[:]=1+A*u*u*cz+B*p*c*cz+T*u*c*cz3
    g[:,0]=2*A*u*cz+B*p1*c*cz+T*c*cz3
    g[:,1]=-B*p*s*cz-T*u*s*cz3
    g[:,2]=om*(-A*u*u*sz-B*p*c*sz-T*u*c*sz3)
    H[:,0,0]=2*A*cz+B*p2*c*cz
    H[:,0,1]=H[:,1,0]=-B*p1*s*cz-T*s*cz3
    H[:,0,2]=H[:,2,0]=om*(-2*A*u*sz-B*p1*c*sz-T*c*sz3)
    H[:,1,1]=-B*p*c*cz-T*u*c*cz3
    H[:,1,2]=H[:,2,1]=om*(B*p*s*sz+T*u*s*sz3)
    H[:,2,2]=-om*om*(A*u*u*cz+B*p*c*cz+T*u*c*cz3)
    return v,g,H


def _base_e(q,theta_amp,period):
    q=np.asarray(q);u,th,eta=q.T;om=2*np.pi/period;z=om*eta
    return _trace_eta(u,th,z,om,theta_amp)


def evaluate(ref,q,name,period,step=None,derivatives=True):
    """(v,g,H) for `name`; `ref` and `step` are unused, kept for API compatibility."""
    q=np.atleast_2d(np.asarray(q,dtype=float));n=len(q)
    if name=='constant':
        return np.ones(n),np.zeros((n,3)),np.zeros((n,3,3))
    if name=='field_b1':
        return base(q,1,period)
    if name=='heldout_field_b2':
        return base(q,2,period)
    if name=='field_e3':
        return _base_e(q,.03,period)
    if name=='field_e12':
        return _base_e(q,.12,period)
    raise ValueError(f'unknown field {name}')


def normal_data(ref,q,name,period):
    """Physical-normal wall datum g_N = a.grad_x f, a = normal(ref,q); zero for constant."""
    q=np.asarray(q)
    a=normal(ref,q)
    g=evaluate(ref,q,name,period)[1]
    return np.einsum('qa,qa->q',a,g)
