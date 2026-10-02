"""Five-field signs, DAE cancellation, BC wiring and AD on actual HSX rows."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from test_q_parallel_hsx_portable import patch
from test_q_parallel_transport import explicit
from drbx.native.q_parallel import QBoundaryData
from drbx.native.q_parallel_material import (material_from_slots, stage_material_transport,
    apply_material_transport, reconstruct_material_slots)
from drbx.stencils.q_parallel_material import prepare_material_transport
from drbx.native.fci_parallel_production_flux import parallel_matrix_from_state
from drbx.native.q_parallel_transport import apply_density_transport,stage_density_transport
from drbx.stencils.q_parallel_transport import prepare_density_transport


@pytest.mark.parametrize('beta',[.7,-.7])
def test_linear_principal_symbol_geometry_and_dae_cancellation(beta):
    center=np.array([1.2,1.1,.9,.12,.08]);slope=np.array([.2,-.1,.15,.03,-.04])
    step=.02;kappa=.13;tau=.8;mu=1836.
    slots=center+np.arange(-2,3)[:,None]*step*slope
    phi=.04+.06*step*np.array([-1,1,0])
    L=np.array([-beta/(2*step),beta/(2*step),kappa])
    got=material_from_slots(slots,phi,L,beta,step,tau=tau,mu=mu)
    n,te,ti,vi,ve=center
    geometry=np.array([-n*ve,2*te/3*(.71*(vi-ve)-ve),-2*ti*ve/3,0,0])*kappa
    expected=-np.asarray(parallel_matrix_from_state(center,tau,mu))@(beta*slope)+geometry
    np.testing.assert_allclose(got.material,expected,atol=2e-11)
    np.testing.assert_allclose(got.generalized_force,[0,0,0,0,mu*beta*(.06+tau*slope[2])],atol=2e-11)
    np.testing.assert_allclose(got.correction,0,atol=2e-10)
    # Ti change cancels in the centered electron row at fixed phi, not twice.
    shifted=slots.copy();shifted[:,2]+=np.arange(-2,3)*step*.7
    other=material_from_slots(shifted,phi,L,beta,step,tau=tau,mu=mu)
    np.testing.assert_allclose(got.centered[4],other.centered[4],atol=2e-10)
    # Constants have physical tube compression, not an identically zero RHS.
    const=material_from_slots(np.broadcast_to(center,(5,5)),np.zeros(3),L,beta,step,tau=tau,mu=mu)
    np.testing.assert_allclose(const.centered,geometry,atol=2e-12)


def test_slot_jvp_including_force_and_combined_no_double_count():
    step=.03;beta=.8;L=np.array([-beta/(2*step),beta/(2*step),.02])
    base=np.array([1.1,1.2,.9,.1,-.08]);slope=np.array([.05,.03,.02,.02,.01])
    z=np.arange(-2,3)[:,None]*step
    q=base+z*slope+z**3*.04;p=np.array([.02,.05,.03])
    # Hold the center/eigenbasis fixed for the inherited spectral AD contract.
    dq=np.ones_like(q)*z*.03;dp=np.array([.03,-.04,.01])
    fn=lambda q,p:material_from_slots(q,p,L,beta,step,tau=1.,mu=1836.).combined
    jit=jax.jit(fn);res=material_from_slots(q,p,L,beta,step,tau=1.,mu=1836.)
    np.testing.assert_allclose(jit(q,p),res.centered+res.correction,atol=1e-10)
    tangent=jax.jvp(fn,(jnp.array(q),jnp.array(p)),(jnp.array(dq),jnp.array(dp)))[1]
    eps=1e-5;fd=(fn(q+eps*dq,p+eps*dp)-fn(q-eps*dq,p-eps*dp))/(2*eps)
    np.testing.assert_allclose(tangent,fd,atol=2e-6,rtol=2e-7)
    invalid=material_from_slots(np.where(np.indices(q.shape)[1]==0,-q,q),p,L,beta,step,tau=1.,mu=1836.)
    assert not bool(invalid.thermodynamic_states_positive)


@pytest.mark.parametrize('kinds,pk',[(('D',)*5,'D'),(('N',)*5,'N'),(('D','N','D','N','D'),'N')])
def test_actual_hsx_mixed_bc_density_replay_and_projection(patch,kinds,pk):
    _,qs,x,bcs=patch;outer,inner=qs
    view=stage_material_transport(prepare_material_transport(inner,outer))
    # Positive thermodynamics with real perturbations and consistent BC offsets.
    x=np.real(x[:5])*.01+np.array([1.,1.1,.9,.1,.08])[:,None]
    def transform(b):
        a=[np.real(v[:5])*.01 for v in b]
        for i in (0,1):a[i]+=np.array([1.,1.1,.9,.1,.08])[:,None,None]
        return QBoundaryData(*a)
    bo,bi=map(transform,bcs);p=x[0]*.03
    pb=QBoundaryData(*(a[0]*.03 for a in bi))
    fn=lambda xx,ii,oo,pp,bb:apply_material_transport(view,xx,ii,oo,pp,bb,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.)
    result=fn(x,bi,bo,p,pb)
    assert np.all(result.inputs_finite) and np.all(result.thermodynamic_states_positive)
    for scalar,q,b in ((view.inner,inner,bi),(view.outer,outer,bo)):
        values=reconstruct_material_slots(scalar,x,b,kinds=kinds)
        expected=np.stack([explicit(q,x[f:f+1],QBoundaryData(*(a[f:f+1] for a in b)),kind)[0] for f,kind in enumerate(kinds)],axis=-1)
        np.testing.assert_allclose(values,expected,atol=2e-12)
    density=stage_density_transport(prepare_density_transport(inner))
    ref=apply_density_transport(density,x[0],x[4],QBoundaryData(*(a[0] for a in bi)),QBoundaryData(*(a[4] for a in bi)),density_kind=kinds[0],velocity_kind=kinds[4])
    np.testing.assert_allclose(result.centered[...,0],ref,atol=1e-11)
    np.testing.assert_allclose(result.combined,result.centered+result.correction,atol=1e-10)
    compiled=jax.jit(fn)(x,bi,bo,p,pb)
    np.testing.assert_allclose(compiled.combined,result.combined,atol=3e-8,rtol=2e-10)
    # A changed live phi affects only the electron force, exactly once.
    from drbx.stencils.q_traced_gradient import prepare_traced_gradient
    from drbx.native.q_traced_gradient import stage_traced_gradient,apply_traced_gradient
    gradient=stage_traced_gradient(prepare_traced_gradient(inner))
    forced=fn(x,bi,bo,2*p,QBoundaryData(*(2*a for a in pb)))
    np.testing.assert_allclose(forced.combined[...,:4],result.combined[...,:4],atol=1e-12)
    expected_force=1836.*apply_traced_gradient(gradient,p,pb,kind=pk)
    np.testing.assert_allclose(forced.combined[...,4]-result.combined[...,4],expected_force,atol=2e-8,rtol=1e-9)
    # Centered action has ordinary live state/BC AD (no frozen eigenbasis issue).
    centered=lambda xx,bb:fn(xx,bb,bo,p,pb).centered
    dx=x*.02;db=QBoundaryData(*(v*.02 for v in bi))
    tangent=jax.jvp(centered,(jnp.asarray(x),bi),(jnp.asarray(dx),db))[1]
    eps=1e-4
    fd=(centered(x+eps*dx,QBoundaryData(*(v+eps*d for v,d in zip(bi,db))))-
        centered(x-eps*dx,QBoundaryData(*(v-eps*d for v,d in zip(bi,db)))))/(2*eps)
    np.testing.assert_allclose(tangent,fd,atol=5e-6,rtol=5e-5)
    # Independent complete-owner projection of a slot-level action.
    from drbx.native.q_parallel_material import apply_raw_material_transport
    raw=apply_raw_material_transport(view,x,bi,bo,p,pb,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.)
    expected=np.sum(np.asarray(raw.combined)[inner.owner_raw]*inner.owner_weight[...,None],axis=1)
    np.testing.assert_allclose(result.combined,expected,atol=1e-11)


def test_preparation_and_input_failures(patch):
    _,qs,x,bcs=patch;a,b=qs[1],qs[0]
    with pytest.raises(ValueError,match='h/32'):prepare_material_transport(b,a)
    changed=b.choice.copy();changed[0]+=1
    with pytest.raises(ValueError):prepare_material_transport(a,replace(b,choice=changed))
    view=prepare_material_transport(a,b)
    with pytest.raises(ValueError,match='five D/N'):
        reconstruct_material_slots(view.inner,np.real(x[:5]),bcs[1],kinds=('D',))
    changed=a.magnetic_L.copy();changed[0,0]+=1e-8
    assert view.metadata['identity']!=prepare_material_transport(replace(a,magnetic_L=changed),b).metadata['identity']
