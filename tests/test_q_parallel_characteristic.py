"""Algebra controls for the bounded eta-traced material correction."""
import numpy as np
import pytest
jax=pytest.importorskip('jax')
from drbx.native.q_parallel_characteristic import eta_characteristic_correction
from drbx.native.fci_parallel_production_flux import parallel_matrix_from_state


def states(step,degree):
    x=np.arange(-2,3)*step
    base=np.array([1.,1.1,.9,.13,.08])
    amplitude=np.array([.03,.04,-.02,.015,-.025])
    return base+x[:,None]**degree*amplitude


@pytest.mark.parametrize('degree',[0,1,2])
@pytest.mark.parametrize('normal',[.7,-.7])
def test_constant_linear_quadratic_have_no_correction(degree,normal):
    q=states(.02,degree)
    result=eta_characteristic_correction(q,normal,.02,tau=1.,mu=1836.)
    assert result.eigensystem_admissible and result.thermodynamic_states_positive and result.inputs_finite
    np.testing.assert_allclose(result.correction,0,atol=3e-11,rtol=0)


def test_independent_characteristic_action_and_orientation():
    step=.02;q=states(step,3)+.2*(states(step,4)-np.array([1.,1.1,.9,.13,.08]))
    A=np.asarray(parallel_matrix_from_state(q[2],1.,1836.))
    values,R=np.linalg.eig(A);assert abs(values.imag).max()<1e-10
    inverse=np.linalg.inv(R);b=-.7
    plus=(R@np.diag(np.maximum(b*values,0))@inverse).real
    minus=(R@np.diag(np.minimum(b*values,0))@inverse).real
    dm=(3*q[2]-4*q[1]+q[0])/(2*step);dp=(-3*q[2]+4*q[3]-q[4])/(2*step);dc=(q[3]-q[1])/(2*step)
    expected=-plus@(dm-dc)-minus@(dp-dc)
    got=eta_characteristic_correction(q,b,step,tau=1.,mu=1836.)
    np.testing.assert_allclose(got.correction,expected,atol=1e-10,rtol=1e-10)
    reverse=eta_characteristic_correction(q[::-1],-b,step,tau=1.,mu=1836.)
    np.testing.assert_allclose(reverse.correction,got.correction,atol=1e-12,rtol=1e-10)
    # Cubic correction scales like step^2; no full RHS is added by this kernel.
    a=eta_characteristic_correction(states(step,3),.7,step,tau=1.,mu=1836.).correction
    small=eta_characteristic_correction(states(step/2,3),.7,step/2,tau=1.,mu=1836.).correction
    np.testing.assert_allclose(a,4*small,atol=5e-11,rtol=1e-7)


def test_jit_jvp_with_center_and_eigenbasis_fixed():
    q=states(.02,3);dq=np.arange(25).reshape(5,5)*.001;dq[2]=0
    fn=jax.jit(lambda x:eta_characteristic_correction(x,.7,.02,tau=1.,mu=1836.).correction)
    result=fn(q);tangent=jax.jvp(fn,(q,),(dq,))[1];eps=1e-5
    np.testing.assert_allclose(tangent,(fn(q+eps*dq)-fn(q-eps*dq))/(2*eps),atol=1e-6,rtol=1e-6)
    assert np.isfinite(result).all()
    assert not eta_characteristic_correction(q,.7,-.02,tau=1.,mu=1836.).inputs_finite
    bad=q.copy();bad[0,0]=-1
    assert not eta_characteristic_correction(bad,.7,.02,tau=1.,mu=1836.).thermodynamic_states_positive
    with pytest.raises(ValueError,match='5 traced slots'):eta_characteristic_correction(q[:3],.7,.02,tau=1.,mu=1836.)
