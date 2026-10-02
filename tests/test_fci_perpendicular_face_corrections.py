"""Focused algebra and JAX transformation checks for extracted P05/P06 faces."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from drbx.native.fci_perpendicular_face_corrections import (
    p05_scalar_face_jump, scatter_p05_jump,
    p06_midpoint_material_remainder, p06_characteristic_face_correction,
    scatter_p06_characteristic, _p06_absolute_action,
)

DATA=Path(__file__).parent/'data/p_shared_face_rows'


@pytest.mark.parametrize('n',(32,48,64))
def test_p06_real_hsx_complete_owner_face_and_midpoint(n):
    with np.load(DATA/f'P06_N{n}.npz',allow_pickle=False) as z:
        a={name:z[name] for name in z.files}
    for state in range(4):
        lo,hi,spectral,positivity,_=p06_characteristic_face_correction(
            a['central'][state],a['lower'][state],a['upper'][state],a['bmag'],a['normal'],
            a['quadrature_weight'],a['wall_mask'],a['collapsed_mask'])
        np.testing.assert_allclose(lo,a['expected_lower'][state],rtol=0,atol=2e-9)
        np.testing.assert_allclose(hi,a['expected_upper'][state],rtol=0,atol=2e-9)
        assert int(spectral)==int(positivity)==0
        action=scatter_p06_characteristic(lo,hi,a['lower_owner'],a['upper_owner'],a['evolution_volume'])
        np.testing.assert_allclose(np.asarray(action)*a['evolution_volume'][:,None],
                                   a['expected_owner_face'][state],rtol=0,atol=2e-9)
        material,remainder=p06_midpoint_material_remainder(a['cell_values'][state],
            a['cell_gradients'][state],a['cell_bmag'],a['cell_curvature'],a['cell_evolution_weight'])
        numerator=np.zeros((2,4));other=np.zeros_like(numerator)
        np.add.at(numerator,a['raw_owner_position'],np.asarray(material))
        np.add.at(other,a['raw_owner_position'],np.asarray(remainder))
        np.testing.assert_allclose(numerator/a['evolution_volume'][:,None],
            a['expected_centered'][state,0],rtol=0,atol=2e-9)
        np.testing.assert_allclose(other/a['evolution_volume'][:,None],
            a['expected_centered'][state,1],rtol=0,atol=2e-9)
        np.testing.assert_allclose((numerator+other)/a['evolution_volume'][:,None],
            a['expected_centered'][state,2],rtol=0,atol=2e-9)


@pytest.mark.parametrize('n',(32,48,64))
def test_p05_real_hsx_complete_owner_jump_and_decomposition(n):
    with np.load(DATA/f'P05_N{n}.npz',allow_pickle=False) as z:
        a={name:z[name] for name in z.files}
    fn=jax.jit(lambda side:p05_scalar_face_jump(a['common_gradient'],a['lower_value'],side,
        a['h_covariant_over_b'],a['quadrature_weight'],a['face_axis'],a['pairs']))
    face=np.asarray(fn(a['upper_value']))
    np.testing.assert_allclose(face,a['expected_active_jump'],rtol=0,atol=2e-9)
    full=np.zeros((len(a['lower_owner']),len(a['pairs'])))
    full[a['active_positions']]=face
    owner=np.asarray(scatter_p05_jump(full,a['lower_owner'],a['upper_owner'],a['owner_volume']))
    selection=a['selected_owner_positions']
    np.testing.assert_allclose(owner[selection],a['expected_owner_jump'],rtol=0,atol=2e-9)
    np.testing.assert_allclose(a['centered']+owner[selection],a['accepted'],rtol=0,atol=2e-9)


def test_p05_scalar_jump_orientation_and_changed_state():
    gradient=np.zeros((1,9,3,2));gradient[:,:,1,0]=1.
    h=np.zeros((1,9,3));h[:,:,2]=1.
    lower=np.zeros((1,9,2));upper=lower.copy();upper[:,:,1]=2.
    weight=np.ones((1,9))/9
    fn=jax.jit(lambda side:p05_scalar_face_jump(gradient,lower,side,h,weight,np.array([0]),np.array([[0,1]])))
    flux=np.asarray(fn(upper))
    np.testing.assert_allclose(flux,[[-1.]],rtol=0,atol=1e-14)
    owners=np.asarray(scatter_p05_jump(flux,np.array([0]),np.array([1]),np.array([2.,4.])))
    np.testing.assert_allclose(owners,[[-.5],[.25]],rtol=0,atol=1e-14)
    tangent=np.ones_like(upper);tangent[:,:,1]=3.
    _,derivative=jax.jvp(fn,(jnp.asarray(upper),),(jnp.asarray(tangent),))
    np.testing.assert_allclose(np.asarray(derivative),[[-1.5]],rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.asarray(fn(upper+0.1*tangent)),flux+0.1*np.asarray(derivative),rtol=0,atol=1e-14)
    move=jax.jit(lambda g:p05_scalar_face_jump(g,lower,upper,h,weight,np.array([0]),np.array([[0,1]])))
    direction=np.zeros_like(gradient);direction[:,:,1,0]=.1
    _,derivative=jax.jvp(move,(jnp.asarray(gradient),),(jnp.asarray(direction),))
    eps=1e-5
    finite=(move(gradient+eps*direction)-move(gradient-eps*direction))/(2*eps)
    np.testing.assert_allclose(derivative,finite,rtol=1e-7,atol=1e-10)


def test_p06_midpoint_material_remainder_closure():
    values=np.array([[[1.2,1.1,.9,.02,.03]]])
    gradients=np.array([[[[.1,.2,.3],[.2,-.1,.1],[.03,.05,-.02],
                          [.04,-.02,.01],[.07,.01,-.03]]]])
    B=np.array([[1.4]]);K=np.array([[[.3,-.2,.1]]]);weight=np.array([[.4]])
    fn=jax.jit(lambda v:p06_midpoint_material_remainder(v,gradients,B,K,weight))
    material,remainder=fn(values)
    assert material.shape==(1,4) and remainder.shape==(1,4)
    assert np.isfinite(np.asarray(material+remainder)).all()
    _,linear=jax.jvp(lambda v:fn(v)[0]+fn(v)[1],(jnp.asarray(values),),(jnp.ones_like(jnp.asarray(values))*0.01,))
    eps=1e-5
    finite=(fn(values+eps*.01)[0]+fn(values+eps*.01)[1]-fn(values-eps*.01)[0]-fn(values-eps*.01)[1])/(2*eps)
    np.testing.assert_allclose(linear,finite,rtol=1e-7,atol=1e-10)


def test_p06_distinct_sides_collapsed_and_finite_positive_face():
    central=np.array([[[1.2,1.1,.9,.03]]*9,[[1.,1.,1.,0.]]*9])
    lower=central.copy();upper=central.copy();upper[0,:,0]+=.02
    B=np.ones((2,9))*1.4;normal=np.ones((2,9))*.1
    normal[1]=0;weight=np.ones((2,9))/9
    fn=jax.jit(lambda u:p06_characteristic_face_correction(
        central,lower,u,B,normal,weight,np.array([False,False]),np.array([False,True])))
    lo,hi,spectral,positivity,wall=fn(upper)
    assert np.isfinite(np.asarray(lo)).all() and np.isfinite(np.asarray(hi)).all()
    np.testing.assert_array_equal(np.asarray(lo)[1],0)
    np.testing.assert_array_equal(np.asarray(hi)[1],0)
    assert int(spectral)==int(positivity)==int(wall)==0
    owners=scatter_p06_characteristic(lo,hi,np.array([0,0]),np.array([1,1]),np.array([2.,4.]))
    np.testing.assert_allclose(np.asarray(owners)[0],np.asarray(lo[0])/2,rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.asarray(owners)[1],np.asarray(hi[0])/4,rtol=0,atol=1e-14)
    tangent=np.zeros_like(upper);tangent[0,:,0]=.01
    _,derivative=jax.jvp(lambda u:fn(u)[0][0],(jnp.asarray(upper),),(jnp.asarray(tangent),))
    eps=1e-5
    finite=(fn(upper+eps*tangent)[0][0]-fn(upper-eps*tangent)[0][0])/(2*eps)
    np.testing.assert_allclose(derivative,finite,rtol=5e-5,atol=1e-8)
    moving=jax.jit(lambda c:p06_characteristic_face_correction(
        c,lower,upper,B,normal,weight,np.array([False,False]),np.array([False,True]))[0][0])
    _,derivative=jax.jvp(moving,(jnp.asarray(central),),(jnp.asarray(tangent),))
    finite=(moving(central+eps*tangent)-moving(central-eps*tangent))/(2*eps)
    np.testing.assert_allclose(derivative,finite,rtol=5e-5,atol=1e-8)


def test_p06_spectral_fallback_count_and_action():
    matrix=np.zeros((1,1,4,4));matrix[0,0,0,1]=-1;matrix[0,0,1,0]=1
    jump=np.array([[[.3,-.2,.1,.4]]])
    action,fallback=_p06_absolute_action(jnp.asarray(matrix),jnp.asarray(jump))
    assert int(np.sum(np.asarray(fallback)))==1
    np.testing.assert_allclose(action,np.linalg.norm(matrix[0,0])*jump,rtol=0,atol=1e-14)


def _wall_case(seed=0, faces=12, nodes=9):
    """Random q3 states with wall, non-wall and collapsed faces (one of each overlapping)."""
    rng = np.random.default_rng(seed)
    state = lambda: np.concatenate([1.+.3*rng.random((faces,nodes,3)),.1*rng.normal(size=(faces,nodes,1))],axis=-1)
    central,lower,upper=state(),state(),state()
    B=1.+.4*rng.random((faces,nodes));normal=.2*rng.normal(size=(faces,nodes));weight=rng.random((faces,nodes))/nodes
    wall=np.zeros(faces,bool);wall[[1,4,5,9]]=True
    collapsed=np.zeros(faces,bool);collapsed[[0,5]]=True       # face 5 is both wall and collapsed
    return central,lower,upper,B,normal,weight,wall,collapsed


def _bit_equal(a,b):
    return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)


@pytest.mark.parametrize('padding',(0,3))
def test_p06_wall_face_restricted_solve_is_bitwise_the_full_solve(padding):
    central,lower,upper,B,normal,weight,wall,collapsed=_wall_case()
    faces=len(wall)
    # a wall face with an inadmissible thermodynamic trace exercises the wall-fallback counter
    central[4,:,0]=-.5
    wall_faces=np.concatenate([np.flatnonzero(wall)[::-1],np.full(padding,faces)]).astype(np.int32)
    run=lambda **kw:jax.jit(lambda c,lo,up:p06_characteristic_face_correction(
        c,lo,up,B,normal,weight,wall,collapsed,**kw))(central,lower,upper)
    full,restricted=run(),run(wall_faces=wall_faces)
    assert int(full[4])==int(restricted[4])>0                              # counter semantics: wall faces only
    for a,b in zip(full,restricted):
        assert _bit_equal(a,b)
    # no wall face at all: the empty index set is a no-op
    none=np.zeros(faces,bool)
    empty=jax.jit(lambda c,lo,up:p06_characteristic_face_correction(
        c,lo,up,B,normal,weight,none,collapsed,wall_faces=np.zeros(padding,np.int32)+faces))(central,lower,upper)
    base=jax.jit(lambda c,lo,up:p06_characteristic_face_correction(
        c,lo,up,B,normal,weight,none,collapsed))(central,lower,upper)
    for a,b in zip(base,empty):
        assert _bit_equal(a,b)


def test_p06_wall_face_restricted_solve_jvp_is_bitwise_the_full_jvp():
    central,lower,upper,B,normal,weight,wall,collapsed=_wall_case(seed=2)
    wall_faces=np.flatnonzero(wall).astype(np.int32)
    tangent=np.random.default_rng(5).normal(size=central.shape)*.01
    def jvp(wf):
        fn=lambda c:p06_characteristic_face_correction(c,lower,upper,B,normal,weight,wall,collapsed,wall_faces=wf)[:2]
        return jax.jit(lambda c,t:jax.jvp(fn,(c,),(t,)))(jnp.asarray(central),jnp.asarray(tangent))
    for a,b in zip(jax.tree_util.tree_leaves(jvp(None)),jax.tree_util.tree_leaves(jvp(wall_faces))):
        assert _bit_equal(a,b)
