"""Assembly controls; real-HSX C3 evidence is in the bounded audit report."""
import numpy as np
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from drbx.native.q_parallel_current_phi import current_phi_from_raw


def data():
    return tuple(jnp.array(x,dtype=jnp.float64) for x in
                 ([.2,-.3],[.04,.05],[.7,-.2],[.1,.3],[1.,2.],[1.3,.9]))


def test_lift_once_matched_force_and_projection_placement():
    x=data();a=current_phi_from_raw(*x,tau=1.,mu=1836.)
    assert np.asarray(a.inputs_valid).all()
    np.testing.assert_allclose(a.vorticity_current,a.vorticity_homogeneous+a.vorticity_lift,atol=1e-14)
    np.testing.assert_allclose(a.electron_generalized_force+a.electron_ti_compensation,a.electron_phi,atol=3e-13)
    np.testing.assert_allclose(a.vorticity_current,x[5]**2/x[4]*(x[0]+x[1]),atol=1e-14)
    w=np.array([.3,.7]);correct=w@a.vorticity_current
    wrong=(w@(x[5]**2/x[4]))*(w@a.divergence_physical)
    assert abs(correct-wrong)>.05


def test_jit_and_joint_live_data_jvp():
    x=data();dx=tuple(.03*a for a in x)
    f=lambda *a:current_phi_from_raw(*a,tau=.8,mu=1836.)
    eager=f(*x);compiled=jax.jit(f)(*x)
    for a,b in zip(eager,compiled):np.testing.assert_allclose(a,b,atol=1e-12)
    _,dy=jax.jvp(lambda *a:f(*a).vorticity_current,x,dx)
    np.testing.assert_allclose(dy,.06*eager.vorticity_current,atol=1e-13)
    _,dy=jax.jvp(lambda *a:f(*a).electron_phi,x,dx)
    np.testing.assert_allclose(dy,.03*eager.electron_phi,atol=1e-12)


def test_no_silent_clipping_or_broadcast():
    x=list(data());x[4]=jnp.array([0.,-1.])
    assert not np.asarray(current_phi_from_raw(*x,tau=1.,mu=1.).inputs_valid).any()
    x=list(data());x[0]=jnp.ones((1,2))
    with pytest.raises(ValueError,match='raw shape'):current_phi_from_raw(*x,tau=1.,mu=1.)
    x=list(data());x[0]=jnp.array([1,2])
    with pytest.raises(TypeError):current_phi_from_raw(*x,tau=1.,mu=1.)


def test_cancelling_current_does_not_hide_nonfinite_split():
    x=list(data());x[0]=jnp.full(2,1e308);x[1]=-x[0]
    x[4]=jnp.ones(2);x[5]=jnp.full(2,2.)
    a=current_phi_from_raw(*x,tau=1.,mu=1.)
    np.testing.assert_array_equal(a.vorticity_current,0.)
    assert not np.asarray(a.inputs_valid).any()
