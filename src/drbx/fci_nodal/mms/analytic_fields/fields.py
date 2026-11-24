"""Catalogue with exact value, gradient and Hessian in (u,theta,eta).

Second-order analytic jets apply the ordinary product and unary chain rules;
no numerical differentiation or geometry enters the catalogue derivatives.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

@dataclass
class Jet:
    v: jnp.ndarray
    g: jnp.ndarray
    h: jnp.ndarray

    def __add__(self,other):
        if not isinstance(other,Jet):return Jet(self.v+other,self.g,self.h)
        return Jet(self.v+other.v,self.g+other.g,self.h+other.h)
    __radd__=__add__
    def __neg__(self):return self*(-1)
    def __sub__(self,other):return self+-other
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        if not isinstance(other,Jet):return Jet(self.v*other,self.g*other,self.h*other)
        return Jet(self.v*other.v,
                   self.g*other.v[:,None]+other.g*self.v[:,None],
                   self.h*other.v[:,None,None]+other.h*self.v[:,None,None]+
                   jnp.einsum('pi,pj->pij',self.g,other.g)+jnp.einsum('pi,pj->pij',other.g,self.g))
    __rmul__=__mul__
    def unary(self,value,first,second):
        return Jet(value,first[:,None]*self.g,
                   first[:,None,None]*self.h+second[:,None,None]*jnp.einsum('pi,pj->pij',self.g,self.g))
    def sin(self):return self.unary(jnp.sin(self.v),jnp.cos(self.v),-jnp.sin(self.v))
    def cos(self):return self.unary(jnp.cos(self.v),-jnp.sin(self.v),-jnp.cos(self.v))

NAMES=('constant','base_multiharmonic_field','zero_wall_value_profile','zero_wall_derivative_profile')+tuple(
    f'wave_{axis}_l{wavelength:g}_{part}' for wavelength in (2.,4.,.5,.25)
    for axis in ('x','y') for part in ('cos','sin'))

@jax.jit
def analytic_fields_and_derivatives(points):
    """Vectorised arbitrary points (...,3) -> values (...,F), df (...,F,3), ddf (...,F,3,3)."""
    points=jnp.asarray(points,dtype=jnp.float64)
    shape=points.shape[:-1];q=points.reshape(-1,3);N=len(q)
    variables=[]
    for i in range(3):
        gradient=jnp.zeros((N,3));gradient=gradient.at[:,i].set(1)
        variables.append(Jet(q[:,i],gradient,jnp.zeros((N,3,3))))
    u,theta,eta=variables
    x=u*theta.cos();y=u*theta.sin()
    s=1+.1*x*eta.cos()+.05*y*eta.sin()+.02*(x*x-y*y)*(2*eta).cos()
    factor=(1-u*u)*(1-u*u)
    constant=Jet(jnp.ones(N),jnp.zeros((N,3)),jnp.zeros((N,3,3)))
    catalogue=[constant,s,factor*s,1+factor*s]
    for wavelength in (2.,4.,.5,.25):
        for coord in (x,y):
            phase=(2*jnp.pi/wavelength)*coord+eta
            catalogue.extend((phase.cos(),phase.sin()))
    values=jnp.stack([f.v for f in catalogue],axis=1)
    first=jnp.stack([f.g for f in catalogue],axis=1)
    second=jnp.stack([f.h for f in catalogue],axis=1)
    return (values.reshape(*shape,len(NAMES)),first.reshape(*shape,len(NAMES),3),
            second.reshape(*shape,len(NAMES),3,3))


# Consumer-specific projections let XLA remove unused Hessian/value arithmetic.
analytic_fields_and_gradients_device=jax.jit(lambda q:analytic_fields_and_derivatives(q)[:2])
def _points(points):
 q=np.asarray(points,float)
 if q.ndim<1 or q.shape[-1]!=3 or not np.isfinite(q).all():raise ValueError('points')
 return q
def first(points):return tuple(np.asarray(a) for a in analytic_fields_and_gradients_device(_points(points)))
