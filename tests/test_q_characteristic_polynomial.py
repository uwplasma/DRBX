"""Device-native characteristic candidate: algebra, fallback, AD and HSX replay."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from drbx.native.q_characteristic_polynomial import polynomial_characteristic_split, polynomial_basis
from drbx.native.fci_parallel_production_flux import parallel_matrix_from_state, parallel_characteristic_split
from drbx.native.q_parallel_characteristic import eta_characteristic_correction
from tests.test_q_parallel_hsx_portable import patch
from tests.test_q_parallel_rhs import assembled
from tests.test_q_plan import pair


def states(count=256, seed=20261003):
    r=np.random.default_rng(seed)
    return np.stack((10**r.uniform(-2,2,count),10**r.uniform(-1,1,count),
                     10**r.uniform(-1,1,count),r.uniform(-2,2,count),r.uniform(-2,2,count)),axis=-1)


def legacy(state,tau,mu,normal):
    return parallel_characteristic_split(parallel_matrix_from_state(state,tau,mu),normal)


def compare(got, expected, atol=1e-8, rtol=1e-11):
    for a,b in zip(jax.tree.leaves(got),jax.tree.leaves(expected),strict=True):
        if np.asarray(a).dtype.kind=='b':np.testing.assert_array_equal(a,b)
        else:np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)


@pytest.mark.parametrize('normal',(1.,-.7,0.,1e-12))
def test_full_split_matches_existing_eigensolver(normal):
    s=states(1024)
    compare(jax.jit(polynomial_characteristic_split)(s,1.,1836.,normal),
            jax.jit(legacy)(s,1.,1836.,normal))


def test_spectrum_residual_and_complex_fallback():
    s=states(512,11);mu=10**np.random.default_rng(12).uniform(1,3.5,len(s))
    values,right,left,valid=jax.jit(polynomial_basis)(s,1.,mu)
    matrix=np.asarray(parallel_matrix_from_state(s,1.,mu))
    err=np.linalg.norm(matrix@right-right*values[...,None,:],axis=(-2,-1))
    den=np.linalg.norm(matrix,axis=(-2,-1))*np.linalg.norm(right,axis=(-2,-1))
    assert np.max((err/den)[valid])<1e-12
    compare(polynomial_characteristic_split(s,1.,mu,-.4),legacy(s,1.,mu,-.4))


@pytest.mark.parametrize('tau',(0.,1.))
def test_equal_velocities_zero_ti_and_crossing_speed(tau):
    s=states(64);s[:,3:]=0.;s[:32,2]=0.
    compare(polynomial_characteristic_split(s,tau,1836.),legacy(s,tau,1836.,1.))
    assert np.all(polynomial_basis(s,tau,1836.)[-1])


def test_preserves_stopped_projector_jvp_and_vjp():
    s=jnp.asarray(states(12));d=.013*s
    new=lambda q:polynomial_characteristic_split(q,1.,1836.,-.6)[:2]
    old=lambda q:legacy(q,1.,1836.,-.6)[:2]
    compare(jax.jvp(new,(s,),(d,)),jax.jvp(old,(s,),(d,)),atol=2e-8)
    weights=jnp.arange(12*25).reshape(12,5,5)/300
    compare(jax.grad(lambda q:jnp.sum(new(q)[0]*weights))(s),
            jax.grad(lambda q:jnp.sum(old(q)[0]*weights))(s),atol=2e-8)


def test_compiled_candidate_targets():
    import re
    compiled=jax.jit(polynomial_characteristic_split).lower(jnp.asarray(states(4)),1.,1836.,.7).compile()
    hlo='\n'.join(re.findall(r'custom_call_target="([^"]+)"',compiled.as_text().lower()))
    for forbidden in ('geev','syevd','host_callback','pure_callback','python_cpu_callback'):
        assert forbidden not in hlo


@pytest.mark.parametrize('degree',(0,1,2,3,4))
def test_complete_correction_replay(degree):
    offset=np.arange(-2,3)*.02
    q=np.array([1.,1.1,.9,.13,.08])+offset[:,None]**degree*np.array([.03,.04,-.02,.015,-.025])
    old=eta_characteristic_correction(q,-.7,.02,tau=1.,mu=1836.)
    new=eta_characteristic_correction(q,-.7,.02,tau=1.,mu=1836.,characteristic_method='polynomial')
    compare(new,old)
    with pytest.raises(ValueError,match='characteristic_method'):
        eta_characteristic_correction(q,1.,.02,tau=1.,mu=1836.,characteristic_method='bad')


def test_nonphysical_candidate_inputs_are_flagged_without_host_fallback():
    s=states(4);s[0,0]=-1.;s[1,1]=-1.;s[2,2]=-1.
    result=polynomial_characteristic_split(s,1.,1836.)
    np.testing.assert_array_equal(result[-1],[False,False,False,True])
    assert all(np.isfinite(np.asarray(a)).all() for a in result)


def test_nonhyperbolic_states_use_existing_device_fallback():
    s=jnp.array([[1.,1.,1.,1.,0.],[1.,1.,1.,4.,0.],[1.,1.,1.,40.,0.]])
    mu=jnp.array([1.,10.,100.])
    new=polynomial_characteristic_split(s,1.,mu,-.4)
    assert not np.any(new[-1])
    compare(new,legacy(s,1.,mu,-.4))


def test_simple_root_at_alternative_eigenvector_denominator_zero():
    # A formula dividing by x*(x-d)+71Te/150 would be singular here.
    te,ti,tau=1.1,.9,1.
    g=107/50*te+5/3*tau*ti
    d=g/np.sqrt(g-71/150*te)
    s=jnp.array([1.,te,ti,d,0.])
    compare(polynomial_characteristic_split(s,tau,1836.),legacy(s,tau,1836.,1.))
    assert polynomial_basis(s,tau,1836.)[-1]


@pytest.mark.parametrize('span',(1/16,1/32))
@pytest.mark.parametrize('kinds,pk',[(('D',)*6,'D'),(('N',)*6,'N'),
                                   (('D','N','D','N','D','N'),'N')])
def test_actual_hsx_complete_rhs(pair,span,kinds,pk):
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.stencils.q_parallel_rhs import prepare_six_field_rhs
    from drbx.native.q_plan import apply_q_plan
    bank,inner,outer,x,bi,bo,phi,pb=pair
    rt,_=prepare_six_field_rhs(inner,outer,None,diffusion_span=span)
    plan=lower_q_plan(bank,diffusion_span=span,material_runtime=rt)
    args=jax.tree.map(jnp.asarray,(plan,x,bi,bo,phi,pb,np.arange(1,7)*.01))
    old=jax.jit(lambda *a:apply_q_plan(*a,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.))
    new=jax.jit(lambda *a:apply_q_plan(*a,kinds=kinds,phi_kind=pk,tau=1.,mu=1836.,
                                     characteristic_method='polynomial'))
    compare(new(*args),old(*args))
    assert np.all(new(*args).eigensystem_admissible)


def test_candidate_one_and_four_device_mechanics():
    import os,sys,subprocess
    code='''
import jax
import numpy as np
from jax.sharding import Mesh
from tests.test_q_sharding import synthetic_case,bounded_synthetic_case
from drbx.native.q_sharding import shard_q_plan,sharded_q_rhs
from drbx.native.q_plan import apply_q_plan
for fixture in (synthetic_case,bounded_synthetic_case):
    bank,top,plan,x,bc,phi,pb=fixture()
    kw=dict(kinds=('D','N','D','N','D','N'),phi_kind='N',tau=1.,mu=1836.)
    coef=np.arange(1,7)*.01
    ref=jax.jit(lambda *a:apply_q_plan(*a,**kw))(plan,x,bc,bc,phi,pb,coef)
    for count in (1,4):
        mesh=Mesh(np.array(jax.devices()[:count],object),('z',))
        sh=shard_q_plan(plan,bank,top,count)
        fn=jax.jit(lambda *a:sharded_q_rhs(*a,**kw,mesh=mesh,characteristic_method='polynomial'))
        got=fn(sh,x,bc,bc,phi,pb,coef)
        for a,b in zip(jax.tree.leaves(got),jax.tree.leaves(ref),strict=True):
            if a.dtype.kind=='b':np.testing.assert_array_equal(a,b)
            else:np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-11)
print('one/four device mechanics, including empty shards, passed')
'''
    env=dict(os.environ,JAX_ENABLE_X64='true',JAX_PLATFORMS='cpu',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             XLA_FLAGS='--xla_force_host_platform_device_count=4')
    run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=180)
    assert run.returncode==0,run.stdout+run.stderr[-5000:]
