"""Finite differences of the frozen continuous magnetic field."""
import numpy as np

FD_STEPS=(1e-4,5e-5,2.5e-5)

def magnetic_derivatives(evaluator,points,step,base=None):
    """Fourth-order logical derivatives of b^i and |B|, with identical evaluators."""
    points=np.asarray(points,float).reshape(-1,3)
    if base is None:
        B,mag,_,_=evaluator(points)
        b=B/mag[:,None]
    else:b,mag=base
    db=np.zeros((len(points),3,3));dmag=np.zeros((len(points),3))
    weights=np.array([1,-8,8,-1])/(12*step)
    for axis in range(3):
        for shift,weight in zip((-2,-1,1,2),weights):
            probe=points.copy();probe[:,axis]+=shift*step
            Bp,Mp,_,_=evaluator(probe)
            db[:,axis,:]+=weight*(Bp/Mp[:,None])
            dmag[:,axis]+=weight*Mp
    along_db=np.einsum('pa,pai->pi',b,db)
    along_logmag=np.einsum('pa,pa->p',b,dmag)/mag
    return dict(b=b,mag=mag,db=db,dmag=dmag,along_db=along_db,along_logmag=along_logmag)

