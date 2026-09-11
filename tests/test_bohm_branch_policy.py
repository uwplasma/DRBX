"""A fixed Bohm derivative is linear and never changes the physical maximum."""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'work/boundary_load_audit'))
from bohm_branch_policy import selected_bohm_maximum,choose_branches


def test_primal_unchanged_and_mask_is_dynamic_under_jit():
    f=jax.jit(lambda x,d,m:jax.jvp(lambda y:selected_bohm_maximum(y[0],y[1],m),(x,),(d,)))
    x=jnp.array([[2.,2.,2.],[1.,2.,3.]])
    d=jnp.array([[1.,3.,5.],[2.,4.,6.]])
    for mask,expected in [(jnp.ones(3,dtype=bool),d[0]),(jnp.zeros(3,dtype=bool),d[1])]:
        value,tangent=f(x,d,mask)
        np.testing.assert_array_equal(value,jnp.maximum(x[0],x[1]))
        np.testing.assert_array_equal(tangent,expected)


def test_fixed_mask_action_is_linear_across_opposite_directions():
    x=jnp.array([[2.,2.],[2.,2.]])
    m=jnp.array([True,False])
    def action(d):return jax.jvp(lambda y:selected_bohm_maximum(y[0],y[1],m),(x,),(d,))[1]
    d=jnp.array([[1.,2.],[3.,4.]]);e=jnp.array([[5.,6.],[7.,8.]])
    np.testing.assert_allclose(action(2*d-3*e),2*action(d)-3*action(e))


def test_branches_follow_state_away_from_ties_and_direction_at_ties():
    values=(np.array([[2.,2.,2.,2.,2.],[1.,3.,2.,2.,2.],[1.,1.,1.,1.,0.]]),)
    derivatives=(np.array([[1.,1.,1.,1.,1.],[10.,-10.,3.,-3.,10.],[0.,0.,0.,0.,0.]]),)
    previous=(np.array([False,True,True,False,True]),)
    choice,rows=choose_branches(values,derivatives,previous)
    np.testing.assert_array_equal(choice[0],[True,False,False,True,True])
    assert rows[0]['changed']==4


def test_zero_direction_at_tie_preserves_previous_branch():
    values=(np.array([[2.,2.],[2.,2.],[1.,1.]]),)
    previous=(np.array([True,False]),)
    choice,_=choose_branches(values,(np.zeros((3,2)),),previous)
    np.testing.assert_array_equal(choice[0],previous[0])


def test_damped_newton_accepts_branch_crossing_and_refreshes_at_new_state():
    from reduced_implicit_stage import (
        InnerSolveResult, ReducedImplicitCallbacks, ReducedImplicitNewton,
        ReducedJacobianBlocks, ReducedNewtonConfig,
    )
    states=[];branches=[];events=[]

    def residual(u,z):
        return u+np.maximum(u,0.)+.1*u*u-1.

    def blocks(u,z):
        # At the initial tie deliberately select the branch opposite to the
        # entered positive direction. Its half step is still valid progress.
        branch=bool(u[0]>0.)
        states.append(float(u[0]));branches.append(branch)
        return ReducedJacobianBlocks(np.array([[1.+float(branch)+.2*u[0]]]),
            np.zeros((1,1)),-np.ones((1,1)),lambda rhs:rhs,np.ones((1,1)))

    cb=ReducedImplicitCallbacks(residual,lambda u,z:z-u,
        lambda u,z:InnerSolveResult(np.array(u),residual_norm=0.),blocks,
        progress=lambda e:events.append(e))
    result=ReducedImplicitNewton(cb,ReducedNewtonConfig(max_newton=3)).solve(np.zeros(1))
    accepted=[e for e in events if e['status']=='accepted']
    assert accepted[0]['alpha']==.5
    assert states[0]==0. and states[1]==.5
    assert branches[:2]==[False,True]
    assert result.final_full_norm < 1.e-9
    assert all(b['full_norm']<a['full_norm'] for a,b in zip(accepted[:-1],accepted[1:]))
