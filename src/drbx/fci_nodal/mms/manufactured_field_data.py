"""Manufactured field family, analytic product/chain jets; no source fitting."""
from ..options import CONDUCTING_WALL, FLOATING_WALL, PRESCRIBED_WALL_HEAT_FLUX
import json
import numpy as np
import jax,jax.numpy as jnp
from functools import partial
from .analytic_fields.fields import analytic_fields_and_derivatives,Jet
from .evolved import ExactJet, REST
C_E = C_I = 0.05  # manufactured check amplitudes; final values remain the user's choice.
C_J = 0.1  # Sole normal-field current coefficient; zero is only the named resolution 16 control.

FIELD_PROFILES=('wall_flat_temperature','prescribed_wall_heat_flux')

def flow_inputs(path):
    flow=json.loads(path.read_text())
    return jnp.asarray(flow['rounded_coefficients']),tuple(map(tuple,flow['modes']))

@partial(jax.jit,static_argnames=('modes','field_profile','wall_law'))
def base_jets(points,coeff,aw,c_j=C_J,*,modes,field_profile='wall_flat_temperature',wall_law='conducting_wall'):
    points=jnp.asarray(points).reshape(-1,3);v,g,h=analytic_fields_and_derivatives(points);nn=len(points);q=[]
    for i in range(3):q.append(Jet(points[:,i],jnp.broadcast_to(jnp.eye(3)[i],(nn,3)),jnp.zeros((nn,3,3))))
    u,theta,eta=q;s=Jet(v[:,1],g[:,1],h[:,1]);one=Jet(jnp.ones(nn),jnp.zeros((nn,3)),jnp.zeros((nn,3,3)))
    P=1-u*u;P2=P*P;P3=P2*P;H=P2*s;R=1-P2;sig=one*0
    for c,(m,k) in zip(coeff,modes):
        power=one if m==0 else u if m==1 else u*u
        sig=sig+power*c*(m*theta+k*eta).sin()
    shapes=[.2449*H+aw*R*(s-1),.1*P3,.2449*H*H,1.34*sig*R,.1*H,H,one*0]
    if field_profile==PRESCRIBED_WALL_HEAT_FLUX:
        shapes[1]=shapes[1]+C_E*P
        shapes[2]=shapes[2]+C_I*P
    if wall_law==CONDUCTING_WALL:shapes[3]=shapes[3]*P
    shapes[4]=shapes[3]-c_j*s*P
    values=REST+jnp.stack([x.v for x in shapes],1);grad=jnp.stack([x.g for x in shapes],1);hess=jnp.stack([x.h for x in shapes],1)
    A=.01*P3*s
    return ExactJet(values,grad,hess,jnp.zeros_like(values)),ExactJet(A.v,A.g,A.h,jnp.zeros(nn))

def _manufactured_field_rows(ref,hits,parallel,divb,coeff,modes,chi,D,*,wall_law,field_profile,c_j):
    nn=np.prod(ref.points.shape[:-1])
    kw=dict(modes=modes,field_profile=field_profile,wall_law=wall_law)
    aw=.5 if wall_law==FLOATING_WALL else 0.;jets,A=base_jets(ref.points,coeff,aw,c_j,**kw);pjets,_=base_jets(ref.wall_points,coeff,aw,c_j,**kw)
    b=jnp.asarray(parallel['b']);c=jnp.asarray(parallel['c']);hit_b=jnp.asarray(parallel['hit_b'])
    ds=jnp.einsum('pi,pfi->pf',b,jets.gradient)
    lap=jnp.einsum('pi,pj,pfij->pf',b,b,jets.hessian[:,:6])+jnp.einsum('pi,pfi->pf',c,jets.gradient[:,:6])
    hjets,_=base_jets(hits,coeff,aw,c_j,**kw);g=hjets.value[:,:6]
    flux=chi[None,:3]*jnp.einsum('pi,pfi->pf',hit_b,hjets.gradient[:,:3])
    # Legacy fields have flat thermal gradients; prescribed_wall_heat_flux retains their exact heat flux.
    if field_profile==PRESCRIBED_WALL_HEAT_FLUX:
        if wall_law==CONDUCTING_WALL:flux=flux.at[:,0].set(0.)  # Density stays as wall_flat_temperature.
    else:
        flux=flux.at[:,1:].set(0.)
        if wall_law==CONDUCTING_WALL:flux=jnp.zeros_like(flux)
    Js=ref.wall_J*jnp.sqrt(ref.wall_ginv_u[...,0])
    pflux=D[:3]*jnp.einsum('pi,pfi->pf',ref.wall_A_row.reshape(-1,3),pjets.gradient[:,:3])/Js.reshape(-1,1)
    if field_profile==PRESCRIBED_WALL_HEAT_FLUX:
        if wall_law==CONDUCTING_WALL:pflux=pflux.at[:,0].set(0.)
    else:
        pflux=pflux.at[:,1:].set(0.)
        if wall_law==CONDUCTING_WALL:pflux=jnp.zeros_like(pflux)
    result=dict(current_coefficient=jnp.asarray(c_j),nodal=jets,pwall=pjets,A=A,g=g,ds=ds,lap=lap,parallel_wall_flux_base=flux,perpendicular_wall_flux_base=pflux,uses_nonlinear_manufactured_potential=jnp.asarray(True),
        divb=jnp.asarray(divb),h=ref.h.reshape(nn,3),jac=ref.jac.reshape(nn),K=ref.K.reshape(nn,3),B=ref.B.reshape(nn),metric=ref.A.reshape(nn,3,3),divA=ref.divA.reshape(nn,3),wall_ginv=ref.wall_ginv_u)
    if field_profile==PRESCRIBED_WALL_HEAT_FLUX and wall_law==CONDUCTING_WALL:result['uses_grounded_wall_potential']=jnp.asarray(True)
    return result,hjets


def _hit_parallel_jets(jets,parallel,*,wall_law,field_profile):
    from .evolved import scaled
    full=scaled(jets,jnp.asarray(1.),jnp.asarray(.2*2*jnp.pi/.02),nonlinear_phi=True,grounded_phi=field_profile==PRESCRIBED_WALL_HEAT_FLUX and wall_law==CONDUCTING_WALL)
    g=np.asarray(full.gradient);hh=np.asarray(full.hessian);b=parallel['hit_b'];c=parallel['hit_c']
    return (np.einsum('pi,pfi->pf',b,g),
            np.einsum('pi,pj,pfij->pf',b,b,hh)+np.einsum('pi,pfi->pf',c,g))
