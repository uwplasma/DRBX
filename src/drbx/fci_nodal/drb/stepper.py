"""Nonautonomous SSPRK3 and the positivity bound for the backend."""


def ssprk3_step(rhs,U,t,h,*,prepare,project,observe,m0):
    """Advance with an already prepared initial stage and caller-supplied stage data.

    The RHS returns a tendency and auxiliary data. The final stage data is
    returned with the new state so a caller can reuse it at the next step.
    """
    observe(U)
    f0,b0=rhs(U,m0)
    m1=prepare(t+h);y1=project(U+h*f0,m1);observe(y1)
    f1,b1=rhs(y1,m1)
    m2=prepare(t+h/2);y2=project(.75*U+.25*(y1+h*f1),m2);observe(y2)
    f2,b2=rhs(y2,m2)
    result=project(U/3+2*(y2+h*f2)/3,m1);observe(result)
    return result,(b0,b1,b2),m1


