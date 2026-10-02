"""Remaining Q07 advection/channels on portable actual-HSX prepared rows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from test_q_parallel_hsx_portable import patch
from drbx.native.q_parallel import QBoundaryData,stage_q,apply_q
from drbx.stencils.q_parallel_material import prepare_material_transport
from drbx.native.q_parallel_material import stage_material_transport
from drbx.native.q_parallel_vorticity import vorticity_from_slots,apply_vorticity_advection
from drbx.stencils.q_parallel_channels import prepare_coefficient_diffusion
from drbx.native.q_parallel_channels import (apply_diffusion_channels,
    stage_coefficient_diffusion,apply_coefficient_diffusion)
from drbx.stencils.q_traced_gradient import prepare_traced_gradient
from drbx.native.q_traced_gradient import stage_traced_gradient,apply_traced_gradient


@pytest.mark.parametrize('speed',[-.8,0.,.8])
def test_vorticity_point_action_sign_and_orientation(speed):
    step=.03;z=np.arange(-2,3)*step;w=.2+z+.7*z**3
    x=vorticity_from_slots(w,speed,1.,step)
    back=(3*w[2]-4*w[1]+w[0])/(2*step)
    forward=(-3*w[2]+4*w[3]-w[4])/(2*step)
    np.testing.assert_allclose(x.combined,-max(speed,0)*back-min(speed,0)*forward,atol=1e-14)
    reversed=vorticity_from_slots(w[::-1],speed,-1.,step)
    np.testing.assert_allclose(x.combined,reversed.combined,atol=1e-14)
    constant=vorticity_from_slots(np.ones(5),speed,1.,step)
    np.testing.assert_array_equal(constant.combined,0)
    if speed==0:np.testing.assert_array_equal(x.combined,0)
    fine=vorticity_from_slots(.2+z/2+.7*(z/2)**3,speed,1.,step/2)
    np.testing.assert_allclose(x.correction,4*fine.correction,atol=1e-14)


@pytest.mark.parametrize('kind',['D','N'])
def test_vorticity_actual_hsx_constant_velocity_traced_gradient_and_ad(patch,kind):
    _,qs,state,bcs=patch;outer,inner=qs
    view=stage_material_transport(prepare_material_transport(inner,outer))
    w=np.real(state[3]);bi=QBoundaryData(*(np.real(a[3]) for a in bcs[1]));bo=QBoundaryData(*(np.real(a[3]) for a in bcs[0]))
    nr=len(inner.raw);vi=np.ones_like(w)*.12
    vb=QBoundaryData(np.full((nr,35),.12),np.full((nr,3),.12),np.zeros((nr,3,2)),np.zeros((nr,35)))
    fn=lambda a,b,c:apply_vorticity_advection(view,a,vi,b,c,vb,omega_kind=kind,velocity_kind=kind)
    result=fn(w,bi,bo);assert np.all(result.inputs_finite)
    G=stage_traced_gradient(prepare_traced_gradient(inner))
    expected=-.12*apply_traced_gradient(G,w,bi,kind=kind)
    np.testing.assert_allclose(result.centered,expected,atol=1e-10)
    jit=jax.jit(fn);np.testing.assert_allclose(jit(w,bi,bo).combined,result.combined,atol=1e-10)
    dw=.2*w;dbi=QBoundaryData(*(a*.2 for a in bi));dbo=QBoundaryData(*(a*.2 for a in bo))
    tangent=jax.jvp(lambda a,b,c:fn(a,b,c).combined,(jnp.array(w),bi,bo),(jnp.array(dw),dbi,dbo))[1]
    np.testing.assert_allclose(tangent,fn(dw,dbi,dbo).combined,atol=1e-10)


@pytest.mark.parametrize('ai',[0,1])
def test_six_constant_channels_preserve_normalization_and_live_parameters(patch,ai):
    _,qs,state,bcs=patch;q=qs[ai];runtime=stage_q(q.runtime_view())
    x=np.real(state[:6]);bc=QBoundaryData(*(np.real(v[:6]) for v in bcs[ai]))
    coeff=np.array([0.,.1,.2,.3,.4,.5]);kinds=('D','N','D','N','D','N')
    fn=lambda a,b,c:apply_diffusion_channels(runtime,a,b,c,kinds=kinds)
    result=fn(x,bc,coeff)
    assert np.all(result.coefficients_valid) and np.all(result.inputs_finite)
    expected=np.stack([coeff[f]*np.asarray(apply_q(runtime,x[f],QBoundaryData(*(v[f] for v in bc)),kind=kinds[f])) for f in range(6)])
    np.testing.assert_array_equal(result.action,expected)
    np.testing.assert_array_equal(result.action[0],0.)
    np.testing.assert_allclose(jax.jit(fn)(x,bc,coeff).action,expected,atol=1e-9)
    # Coefficient JVP is the unscaled diffusion, no hidden density/mass factors.
    tangent=jax.jvp(lambda a:fn(x,bc,a).action,(jnp.array(coeff),),(jnp.ones(6),))[1]
    np.testing.assert_allclose(tangent,fn(x,bc,np.ones(6)).action,atol=1e-9)
    assert not bool(fn(x,bc,-np.ones(6)).coefficients_valid)
    with pytest.raises(ValueError,match='six constant'):fn(x,bc,np.ones(5))


@pytest.mark.parametrize('ai',[0,1])
@pytest.mark.parametrize('kind',['D','N'])
def test_coefficient_at_caps_replays_frozen_diffusion_and_explicit_action(patch,ai,kind):
    _,qs,state,bcs=patch;q=qs[ai]
    runtime=stage_coefficient_diffusion(prepare_coefficient_diffusion(q));base=stage_q(q.runtime_view())
    x=np.real(state[3]);bc=QBoundaryData(*(np.real(v[3]) for v in bcs[ai]));chi=1+.1*q.slot_points[:,:,0]
    fn=lambda a,b,k:apply_coefficient_diffusion(runtime,a,b,k,kind=kind)
    result=fn(x,bc,chi);assert np.all(result.coefficients_valid)
    unit=fn(x,bc,np.ones_like(chi));np.testing.assert_allclose(unit.action,apply_q(base,x,bc,kind=kind),atol=1e-8)
    rows=q.row_gradient_D if kind=='D' else q.row_gradient_N
    logical=np.einsum('rd,rsad->rsa',x[q.donor],rows)
    if kind=='D':
        logical+=np.einsum('rj,rsaj->rsa',bc.dirichlet_trace,q.boundary_gradient_D_trace)
        wall=q.raw//32**2>=30
        logical[:,:,1:]+=bc.dirichlet_tangent*wall[:,None,None]
    else:logical+=np.einsum('rj,rsaj->rsa',bc.neumann_normal,q.boundary_gradient_N_normal)
    grad=np.einsum('rsa,rsa->rs',q.magnetic_b,logical)
    raw=np.sum(chi*grad*q.magnetic_L,axis=-1)
    expected=np.sum(raw[q.owner_raw]*q.owner_weight,axis=-1)
    np.testing.assert_allclose(result.action,expected,atol=1e-8)
    np.testing.assert_allclose(jax.jit(fn)(x,bc,chi).action,expected,atol=1e-8)
    # Bilinear state/BC/coefficient JVP, including live Neumann derivatives.
    dx=.03*x;db=QBoundaryData(*(a*.03 for a in bc));dk=.04*chi
    tangent=jax.jvp(lambda a,b,k:fn(a,b,k).action,(jnp.array(x),bc,jnp.array(chi)),(jnp.array(dx),db,jnp.array(dk)))[1]
    expected=fn(dx,db,chi).action+fn(x,bc,dk).action
    np.testing.assert_allclose(tangent,expected,atol=1e-8)
    assert not bool(fn(x,bc,-chi).coefficients_valid)
    with pytest.raises(ValueError,match='slots'):fn(x,bc,chi[:,0])
