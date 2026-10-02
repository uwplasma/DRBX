"""Compact-plan replay against the unchanged six-field actual-HSX API."""
from dataclasses import replace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from tests.test_q_parallel_hsx_portable import patch
from tests.test_q_parallel_rhs import assembled
from drbx.stencils.q_bank import build_q_bank
from drbx.stencils.q_plan import lower_q_plan
from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
from drbx.native.q_plan import (stage_q_plan, apply_q_plan, reconstruct_q_state,
                                apply_traced_gradient, apply_tube_divergence)
from drbx.native.q_parallel_rhs import apply_six_field_rhs
from drbx.native.q_parallel import QBoundaryData


@pytest.fixture
def pair(assembled):
    inner,outer,_,x,bi,bo,phi,pb=assembled
    bank=build_q_bank(outer,inner)
    return bank,inner,outer,x,bi,bo,phi,pb


def _assert_tree(a,b,atol=2e-8):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        if np.asarray(x).dtype.kind in ('V','b'):
            np.testing.assert_array_equal(x,y)
        else:
            np.testing.assert_allclose(x,y,atol=atol,rtol=1e-10)


@pytest.mark.parametrize('span',(1/16,1/32))
@pytest.mark.parametrize('kinds,pk',[(('D',)*6,'D'),(('N',)*6,'N'),(('D','N','D','N','D','N'),'N')])
def test_compact_six_field_legacy_replay(pair,span,kinds,pk):
    bank,inner,outer,x,bi,bo,phi,pb=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=span)
    plan=stage_q_plan(lower_q_plan(bank,diffusion_span=span,material_runtime=rt))
    coeff=np.arange(1,7)*.01
    fn=lambda p,x,bi,bo,phi,pb,c: apply_q_plan(p,x,bi,bo,phi,pb,c,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.)
    old=apply_six_field_rhs(rt,x,bi,bo,phi,pb,coeff,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.)
    actual=fn(plan,x,bi,bo,phi,pb,coeff)
    _assert_tree(actual,old)
    compiled=jax.jit(fn)
    _assert_tree(compiled(plan,x,bi,bo,phi,pb,coeff),old)
    assert all(isinstance(a,jax.Array) for a in jax.tree.leaves(plan))
    cache=compiled._cache_size()
    changed=replace(plan,bmag=plan.bmag*1.01,magnetic_L=plan.magnetic_L*1.001)
    compiled(changed,x,bi,bo,phi,pb,coeff*1.1).combined.block_until_ready()
    assert compiled._cache_size()==cache
    assert plan.row_value_D.shape[1]==5
    assert plan.nbytes < rt.material.nbytes+rt.diffusion.nbytes+rt.bmag.nbytes


def test_live_ad_vmap_coefficient_updates_and_flags(pair):
    bank,inner,outer,x,bi,bo,phi,pb=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    plan=stage_q_plan(lower_q_plan(bank,diffusion_span=1/32,material_runtime=rt))
    kinds=('D','N','D','N','D','N');coeff=np.arange(1,7)*.01
    def new(*args):
        return apply_q_plan(plan,*args,kinds=kinds,phi_kind='D',tau=1.,mu=1836.)
    def old(*args):
        return apply_six_field_rhs(rt,*args,kinds=kinds,phi_kind='D',tau=1.,mu=1836.)
    args=jax.tree.map(jnp.asarray,(x,bi,bo,phi,pb,coeff))
    directions=jax.tree.map(lambda a:.02*a,args)
    # Preserve inherited frozen spectral-projector AD; compare its implemented
    # JVP to the legacy API, with finite differences on centered+diffusion.
    _assert_tree(jax.jvp(new,args,directions)[1],jax.jvp(old,args,directions)[1],atol=2e-7)
    target=lambda *a:new(*a).centered+new(*a).diffusion
    ad=jax.jvp(target,args,directions)[1];eps=1e-4
    plus=jax.tree.map(lambda a,d:a+eps*d,args,directions)
    minus=jax.tree.map(lambda a,d:a-eps*d,args,directions)
    np.testing.assert_allclose(ad,(target(*plus)-target(*minus))/(2*eps),atol=5e-6,rtol=5e-5)
    # Balanced geometry coefficients remain ordinary differentiable leaves.
    coefficient_action=lambda L:apply_q_plan(replace(plan,magnetic_L=L),*args,
        kinds=kinds,phi_kind='D',tau=1.,mu=1836.).centered
    delta=.001*plan.magnetic_L
    coefficient_jvp=jax.jvp(coefficient_action,(plan.magnetic_L,),(delta,))[1]
    coefficient_fd=(coefficient_action(plan.magnetic_L+eps*delta)-
                    coefficient_action(plan.magnetic_L-eps*delta))/(2*eps)
    np.testing.assert_allclose(coefficient_jvp,coefficient_fd,atol=5e-6,rtol=5e-5)
    _,vjp=jax.vjp(lambda c:new(*args[:-1],c).combined,args[-1])
    _,old_vjp=jax.vjp(lambda c:old(*args[:-1],c).combined,args[-1])
    _assert_tree(vjp(jnp.ones_like(new(*args).combined)),old_vjp(jnp.ones_like(old(*args).combined)))
    # Full state/BC adjoints retain the legacy frozen eigensystem contract.
    cotangent=jnp.sin(jnp.arange(new(*args).combined.size)).reshape(new(*args).combined.shape)*.02
    _,pullback=jax.vjp(lambda *a:new(*a).combined,*args)
    _,old_pullback=jax.vjp(lambda *a:old(*a).combined,*args)
    _assert_tree(pullback(cotangent),old_pullback(cotangent),atol=2e-7)
    batched=jax.tree.map(lambda a:jnp.stack((a,a*1.01)),args)
    got=new(*batched)
    mapped=jax.vmap(new)(*batched)
    _assert_tree(got,mapped)
    assert not np.all(new(*args[:-1],-coeff).inputs_valid)
    bad=x.copy();bad[0]=-1
    assert not np.all(new(bad,*args[1:]).inputs_valid)


def test_scalar_complex_constant_normal_and_adapters(pair):
    from drbx.native.q_parallel_divergence import apply_raw_scalar_slots
    from drbx.native.q_parallel_rhs import field_boundary
    bank,inner,outer,x,bi,bo,phi,pb=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    plan=stage_q_plan(lower_q_plan(bank,diffusion_span=1/32,material_runtime=rt))
    for kind in ('D','N'):
        slots=reconstruct_q_state(plan,x*(1+2j),bi,bo,kinds=(kind,)*6)
        for f in range(6):
            expected=apply_raw_scalar_slots(rt.material.inner,x[f]*(1+2j),field_boundary(bi,f),kind=kind)
            np.testing.assert_allclose(slots.value[f,:,jnp.array([1,3,2])].T,expected,atol=2e-12,rtol=1e-12)
        actual=apply_tube_divergence(plan,slots.value)
        raw=jnp.sum(slots.value[...,jnp.array([1,3,2])]*plan.magnetic_L,axis=-1)
        expected=jnp.sum(jnp.take(raw,plan.owner_raw,axis=-1)*plan.owner_weight,axis=-1)
        np.testing.assert_allclose(actual,expected,atol=2e-12)
    nr=plan.donor.shape[0]
    bc=QBoundaryData(np.ones((1,nr,35)),np.ones((1,nr,3)),np.zeros((1,nr,3,2)),np.zeros((1,nr,35)))
    constants=np.ones((1,plan.n_owner))
    for kind in ('D','N'):
        slots=reconstruct_q_state(plan,constants,bc,bc,kinds=(kind,))
        np.testing.assert_allclose(slots.value,1.,atol=2e-10,rtol=0)
        np.testing.assert_allclose(apply_traced_gradient(plan,slots.value),0.,atol=1e-7,rtol=0)


def test_host_and_runtime_input_guards(pair):
    bank,inner,outer,x,bi,bo,phi,pb=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    plan=lower_q_plan(bank,diffusion_span=1/32,material_runtime=rt)
    kwargs=dict(kinds=('D',)*6,phi_kind='D',tau=1.,mu=1836.)
    with pytest.raises(ValueError,match='diffusion span'):
        lower_q_plan(bank,diffusion_span=1/64,material_runtime=rt)
    with pytest.raises(ValueError,match='span mismatch'):
        lower_q_plan(bank,diffusion_span=1/16,material_runtime=rt)
    with pytest.raises(ValueError,match='magnetic_L'):
        lower_q_plan(bank,diffusion_span=1/32,magnetic_L=np.zeros((1,3)),b_eta=plan.b_eta,eta_step=plan.eta_step,bmag=plan.bmag)
    with pytest.raises(ValueError,match='real'):
        apply_q_plan(plan,x*(1+1j),bi,bo,phi,pb,np.ones(6),**kwargs)
    with pytest.raises(ValueError,match='boundary shape'):
        reconstruct_q_state(plan,x,QBoundaryData(*(a[...,:1] for a in bi)),bo,kinds=('D',)*6)
    with pytest.raises(ValueError,match='n_owner'):
        reconstruct_q_state(plan,x[:,:-1],bi,bo,kinds=('D',)*6)
    with pytest.raises(ValueError,match='D/N'):
        apply_q_plan(plan,x,bi,bo,phi,pb,np.ones(6),**{**kwargs,'kinds':('D',)})
    with pytest.raises(ValueError,match='D/N'):
        apply_q_plan(plan,x,bi,bo,phi,pb,np.ones(6),**{**kwargs,'kinds':('bad',)*6})
    with pytest.raises(ValueError,match='phi and state'):
        apply_q_plan(plan,x,bi,bo,phi[:-1],pb,np.ones(6),**kwargs)
    with pytest.raises(ValueError,match='phi_kind'):
        apply_q_plan(plan,x,bi,bo,phi,pb,np.ones(6),**{**kwargs,'phi_kind':'bad'})
    with pytest.raises(TypeError,match='floating'):
        reconstruct_q_state(plan,x.astype(int),bi,bo,kinds=('D',)*6)


@pytest.mark.parametrize('dtype',(np.float32,np.float64,np.complex64,np.complex128))
def test_scalar_dtype_preserved(pair,dtype):
    bank,inner,outer,x,bi,bo,_,_=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    plan=stage_q_plan(lower_q_plan(bank,diffusion_span=1/32,material_runtime=rt))
    slots=reconstruct_q_state(plan,x.astype(dtype),bi,bo,kinds=('D',)*6)
    assert slots.value.dtype==dtype
    assert slots.homogeneous.dtype==dtype
    assert apply_traced_gradient(plan,slots.value).dtype==dtype
    assert apply_tube_divergence(plan,slots.value).dtype==dtype
