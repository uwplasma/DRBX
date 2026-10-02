"""Captured Q outer factors: unchanged builder, exact decode and dense fallback."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from drbx.stencils.q_parallel import topology_from_arrays, load_chunk
from drbx.stencils.q_parallel_support import Layers, Hybrid
from drbx.stencils.q_tensor_rows import (build_q_tensor_rows, expand_q_tensor_rows,
    verified_q_tensor_rows, apply_q_tensor_values_numpy)


@pytest.fixture(scope='module')
def control():
    n=32
    centers=((np.arange(n)+.5)/n,(np.arange(n)+.5)*2*np.pi/n,(np.arange(n)+.5)*2*np.pi/n)
    topology=topology_from_arrays(centers,np.ones((n,)*3,bool),np.arange(n**3).reshape((n,)*3),
                                  np.ones((n,)*3))
    builder=Layers(topology); captures=[]; expected=[]
    for ijk in (np.array([20,0,0]),np.array([21,31,31])):
        center=topology.pts[np.ravel_multi_index(ijk,(n,)*3)]
        points=center+np.array([[-.001,.002,-.003],[-.0005,.001,-.0015],
                               [0,0,0],[.0005,-.001,.0015],[.001,-.002,.003]])
        plain=builder.rows(ijk,points)
        donor,value,gradient,meta,capture=builder.rows_with_factors(ijk,points)
        assert meta==plain[3]
        for a,b in zip((donor,value,gradient),plain[:3]):
            assert a.tobytes()==b.tobytes()
        captures.append(capture); expected.append((donor,value,gradient))
    return captures,expected


@pytest.mark.parametrize('gradients',[False,True])
def test_exact_capture_expansion_and_small_payload(control,gradients):
    captures,expected=control
    rows,report=verified_q_tensor_rows(captures,expected,include_gradients=gradients)
    assert rows is not None and report['exact'] and report['selected']
    assert rows.nbytes<report['dense_bytes']
    for got,want in zip(expand_q_tensor_rows(rows),expected):
        np.testing.assert_array_equal(got[0],want[0])
        assert got[1].dtype==want[1].dtype and got[1].shape==want[1].shape
        assert got[1].tobytes()==want[1].tobytes()
        if gradients:
            assert got[2].tobytes()==want[2].tobytes()
        else:
            assert got[2] is None
    assert rows.eta.shape[1]==5


def test_dense_fallback_for_nonexact_and_unsupported(control):
    captures,expected=control
    bad=[(d,v.copy(),g) for d,v,g in expected];bad[0][1][0,0]+=1
    rows,report=verified_q_tensor_rows(captures,bad)
    assert rows is None and not report['exact']
    rows,report=verified_q_tensor_rows([None],expected[:1])
    assert rows is None and 'unsupported' in report['reason']


def test_exact_but_nonbeneficial_payload_keeps_dense(control):
    # Algebra control for a one-target ringwise capture whose 20 angular
    # entries all differ. Their tables cost more than one dense scalar row.
    c=control[0][0].select_targets([0])
    theta=np.random.default_rng(998).normal(size=c.theta.shape)
    c=replace(c,theta=theta,family='ringwise')
    candidate=build_q_tensor_rows([c])
    expected=expand_q_tensor_rows(candidate)
    rows,report=verified_q_tensor_rows([c],expected)
    assert report['exact'] and not report['selected'] and rows is None
    assert report['shared_donor_net_saving_bytes'] < 0
    with pytest.raises(ValueError,match='source count'):
        verified_q_tensor_rows([c],[])


def test_complex_batch_direct_action_has_separate_replay_budget(control):
    captures,expected=control; rows=build_q_tensor_rows(captures)
    rng=np.random.default_rng(123);x=rng.normal(size=(3,6,32**3))*(1+2j)
    actual=apply_q_tensor_values_numpy(rows,x)
    expected_action=np.stack([np.einsum('...d,td->...t',x[...,d],v) for d,v,_ in expected],axis=-2)
    np.testing.assert_allclose(actual,expected_action,rtol=0,atol=2e-13)


def test_invalid_table_shapes_and_owner_bounds(control):
    rows=build_q_tensor_rows(control[0])
    with pytest.raises(ValueError,match='t_eta'):
        replace(rows,t_eta=np.zeros((1,),int)).validate()
    with pytest.raises(ValueError,match='owner identity'):
        replace(rows,owner=-np.ones_like(rows.owner)).validate()


@pytest.mark.parametrize('n',[32,48,64])
def test_bounded_c3_outer_rows_against_saved_prepared_if_available(n):
    root=Path(__file__).resolve().parents[2]
    fixture=root/'work/q08_implementation_20261002/c3_fixtures'
    directory=root/f'geometry_artifacts/rlp_convergence_32_48_64_20260917/{n}x{n}x{n}'
    if not (fixture/f'N{n}_h16.npz').exists() or not directory.exists():
        pytest.skip('local parent-produced stratified C3/topology fixtures unavailable')
    with np.load(directory/'base_geometry.npz') as z:
        centers=tuple(z[f'grid.{a}.centers'].copy() for a in 'xyz')
    with np.load(directory/'rlp_topology.npz') as z:
        topology=topology_from_arrays(centers,z['is_active_owner'],z['aggregate_id'],z['raw_volume'])
    outer,inner=(load_chunk(fixture/f'N{n}_h{denom}.npz') for denom in (16,32))
    hybrid=Hybrid(topology); captures=[]; expected=[]
    for pos,rr in enumerate(outer.raw):
        ijk=np.array(np.unravel_index(rr,(n,)*3))
        if not hybrid.last<ijk[0]<n-2:
            continue
        points=np.stack((outer.slot_points[pos,0],outer.slot_points[pos,1],outer.slot_points[pos,2],
                         inner.slot_points[pos,0],inner.slot_points[pos,1],inner.slot_points[pos,2]))
        d,v,g,meta,c=hybrid.outer.rows_with_factors(ijk,points)
        if c is None:
            continue
        for ai,q in enumerate((outer,inner)):
            target=np.array([0,1,2]) if ai==0 else np.array([3,4,5])
            count=q.row_count[pos]
            assert np.array_equal(d,q.donor[pos,:count])
            assert v[target].tobytes()==q.row_value_D[pos,:,:count].tobytes()
            assert g[target].tobytes()==q.row_gradient_D[pos,:,:,:count].tobytes()
        target=np.array([0,3,2,4,1])
        captures.append(c.select_targets(target));expected.append((d,v[target],g[target]))
    assert captures
    rows,report=verified_q_tensor_rows(captures,expected,include_gradients=True)
    assert rows is not None and report['exact']
