"""Direct/tube algebra and affine wall data on the portable actual-HSX patch."""
import numpy as np
import pytest
from pathlib import Path
import sys

jax=pytest.importorskip('jax')
sys.path.insert(0,str(Path(__file__).parent))
from test_q_parallel_hsx_portable import patch
from drbx.native.q_parallel import QBoundaryData
from drbx.stencils.q_parallel import topology_from_arrays
from drbx.native.q_parallel_gradient import apply_raw_slot_gradients
from drbx.native.q_parallel_divergence import (apply_parallel_divergence,
    apply_raw_parallel_divergence,apply_raw_scalar_slots,stage_parallel_divergence)
from drbx.stencils.q_parallel_divergence import prepare_parallel_divergence,prepare_kappa
from drbx.stencils.q_parallel_gradient import prepare_parallel_gradient


def controlled_geometry(q):
    """Known logical derivative at actual HSX raw midpoints, without new fixture I/O."""
    centers=q.slot_points[:,2]
    b0=q.magnetic_b[:,2]
    def geom(points):
        points=np.asarray(points)
        near=np.argmin(np.sum((points[:,None,:]-centers[None,:,:])**2,axis=-1),axis=1)
        delta=points-centers[near]
        b=b0[near]+delta*np.array([.07,-.04,.03])
        return np.ones(len(points)),b,np.full(len(points),2.)
    return geom


@pytest.mark.parametrize('ai',[0,1])
@pytest.mark.parametrize('kind',['D','N'])
def test_divergence_precontraction_and_wall_lifts(patch,ai,kind):
    _,prepared,state,boundaries=patch
    q=prepared[ai];bc=boundaries[ai]
    d=prepare_parallel_divergence(q,controlled_geometry(q))
    np.testing.assert_allclose(d.kappa_audit.kappa,.06,atol=2e-11,rtol=0)
    assert np.max(d.kappa_audit.step_sensitivity)<1e-10
    direct=stage_parallel_divergence(d.runtime_view('direct'))
    tube=stage_parallel_divergence(d.runtime_view('tube'))
    scalar=stage_parallel_divergence(d.scalar_view())
    gradient=prepare_parallel_gradient(q).runtime_view(include_caps=True)
    values=np.asarray(apply_raw_scalar_slots(scalar,state,bc,kind=kind))
    gradients=np.asarray(apply_raw_slot_gradients(gradient,state,bc,kind=kind))
    direct_exp=gradients[:,:,2]+d.kappa_audit.kappa[None,:]*values[:,:,2]
    tube_exp=np.einsum('frs,rs->fr',values,q.magnetic_L)
    np.testing.assert_allclose(apply_raw_parallel_divergence(direct,state,bc,kind=kind),
                               direct_exp,atol=2e-11,rtol=0)
    np.testing.assert_allclose(apply_raw_parallel_divergence(tube,state,bc,kind=kind),
                               tube_exp,atol=2e-11,rtol=0)
    for view,expected in ((direct,direct_exp),(tube,tube_exp)):
        projected=np.einsum('frm,rm->fr',np.take(expected,q.owner_raw,axis=-1),q.owner_weight)
        np.testing.assert_allclose(apply_parallel_divergence(view,state,bc,kind=kind),
                                   projected,atol=2e-11,rtol=0)
        fn=jax.jit(lambda x,b:apply_parallel_divergence(view,x,b,kind=kind))
        np.testing.assert_allclose(fn(state,bc),projected,atol=2e-11,rtol=0)
        changed=QBoundaryData(*(x+0.125 for x in bc))
        cache=fn._cache_size()
        np.testing.assert_allclose(fn(state,changed),
            apply_parallel_divergence(view,state,changed,kind=kind),atol=2e-11)
        assert fn._cache_size()==cache
        zero=QBoundaryData(*(np.zeros_like(x) for x in bc))
        np.testing.assert_allclose(jax.jvp(lambda x:fn(x,bc),(state,),
            (np.ones_like(state),))[1],fn(np.ones_like(state),zero),atol=2e-11)
        ones=QBoundaryData(*(np.ones_like(x) for x in bc))
        np.testing.assert_allclose(jax.jvp(lambda b:fn(state,b),(bc,),(ones,))[1],
                                   fn(np.zeros_like(state),ones),atol=2e-11)
        nonwall=q.raw//32**2<30
        poisoned=[np.array(x,copy=True) for x in bc]
        for x in poisoned:x[:,nonwall]=124.5
        np.testing.assert_array_equal(fn(state,QBoundaryData(*poisoned)),fn(state,bc))
        if kind=='D':
            query=[np.array(x,copy=True) for x in bc]
            query[1][:,~nonwall]+=0.25
            assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*query))-fn(state,bc))))>0
            tangent=[np.array(x,copy=True) for x in bc]
            tangent[2][:,~nonwall]+=0.25
            if view is direct:
                assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*tangent))-fn(state,bc))))>0
            else:
                np.testing.assert_array_equal(fn(state,QBoundaryData(*tangent)),fn(state,bc))
        else:
            normal=[np.array(x,copy=True) for x in bc]
            normal[3][:,~nonwall]+=0.25
            assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*normal))-fn(state,bc))))>0


def test_center_replay_tolerance_only_controls_acceptance(patch):
    _,prepared,_,_=patch
    q=prepared[0];geom=controlled_geometry(q)
    strict=prepare_kappa(q,geom)
    relaxed=prepare_kappa(q,geom,center_b_atol=1e-10)
    for name in ('kappa','step_values','identity_value','magnetic_B'):
        np.testing.assert_array_equal(getattr(strict,name),getattr(relaxed,name))
    assert strict.response_hashes==relaxed.response_hashes
    def shifted(points):
        J,b,B=geom(points);return J,b+2e-12,B
    with pytest.raises(ValueError,match='center magnetic b differs'):
        prepare_kappa(q,shifted)
    assert np.isfinite(prepare_kappa(q,shifted,center_b_atol=1e-10).kappa).all()
    def wrong(points):
        J,b,B=geom(points);return J,b+1e-8,B
    with pytest.raises(ValueError,match='center magnetic b differs'):
        prepare_kappa(q,wrong,center_b_atol=1e-10)
    for tolerance in (-1,np.inf,np.nan):
        with pytest.raises(ValueError,match='center_b_atol'):
            prepare_kappa(q,geom,center_b_atol=tolerance)


def test_divergence_constant_and_geometry_audit(patch):
    _,prepared,_,_=patch
    q=prepared[0];geom=controlled_geometry(q)
    audit=prepare_kappa(q,geom)
    assert audit.step_values.shape==(3,len(q.raw))
    d=prepare_parallel_divergence(q,geom)
    nr=len(q.raw);ones=np.ones(q.metadata['n_owner'])
    D=QBoundaryData(np.ones((nr,35)),np.ones((nr,3)),
                    np.zeros((nr,3,2)),np.zeros((nr,35)))
    N=QBoundaryData(np.zeros((nr,35)),np.zeros((nr,3)),
                    np.zeros((nr,3,2)),np.zeros((nr,35)))
    expected_kappa=np.einsum('rm,rm->r',
        np.take(audit.kappa,q.owner_raw,axis=-1),q.owner_weight)
    expected_tube=np.einsum('rm,rm->r',
        np.take(np.sum(q.magnetic_L,axis=-1),q.owner_raw,axis=-1),q.owner_weight)
    for kind,bc in (('D',D),('N',N)):
        direct=stage_parallel_divergence(d.runtime_view('direct'))
        tube=stage_parallel_divergence(d.runtime_view('tube'))
        np.testing.assert_allclose(apply_parallel_divergence(direct,ones,bc,kind=kind),
                                   expected_kappa,atol=1e-8,rtol=0)
        np.testing.assert_allclose(apply_parallel_divergence(tube,ones,bc,kind=kind),
                                   expected_tube,atol=1e-8,rtol=0)
    bad=controlled_geometry(q)
    def nonfinite(points):
        J,b,B=bad(points);b[0,0]=np.nan
        return J,b,B
    with pytest.raises(ValueError,match='invalid center geometry'):
        prepare_kappa(q,nonfinite)


@pytest.mark.parametrize('ai',[0,1])
@pytest.mark.parametrize('kind',['D','N'])
def test_independent_radial_manufacture_at_actual_hsx_points(patch,ai,kind):
    """Linear f=r uses independent analytic values and derivatives, including wall BC."""
    arrays,prepared,_,_=patch
    q=prepared[ai]
    topology=topology_from_arrays(tuple(arrays[f'centers_{i}'] for i in range(3)),
        arrays['active'],arrays['aggregate'],arrays['volume'])
    owner_r=np.bincount(topology.ro,weights=topology.rv*topology.pts[:,0])/topology.vol
    d=prepare_parallel_divergence(q,controlled_geometry(q))
    nr=len(q.raw);wall=q.raw//32**2>=30
    trace=np.zeros((nr,35));trace[wall]=1.
    query=np.zeros((nr,3));query[wall]=1.
    normal=np.zeros((nr,35));normal[wall]=q.boundary_wall_normal[wall,:,0]
    bc=QBoundaryData(trace,query,np.zeros((nr,3,2)),normal)
    midpoint=q.slot_points[:,2]
    direct_exact=q.magnetic_b[:,2,0]+d.kappa_audit.kappa*midpoint[:,0]
    tube_exact=np.sum(q.magnetic_L*q.slot_points[:,:,0],axis=1)
    for candidate,raw_exact in (('direct',direct_exact),('tube',tube_exact)):
        expected=np.einsum('rm,rm->r',np.take(raw_exact,q.owner_raw,axis=-1),q.owner_weight)
        actual=np.asarray(apply_parallel_divergence(
            stage_parallel_divergence(d.runtime_view(candidate)),owner_r,bc,kind=kind))
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
