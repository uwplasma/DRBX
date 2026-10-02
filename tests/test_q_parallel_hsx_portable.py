"""Actual HSX preparation/action replay with no external workspace dependency."""
from dataclasses import replace, fields
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
from drbx.native.q_parallel import QBoundaryData, apply_q, stage_q
from drbx.native.q_parallel_gradient import (apply_parallel_gradient,
    apply_raw_slot_gradients, stage_parallel_gradient)
from drbx.stencils.q_parallel import prepare_chunk, topology_from_arrays, save_chunk, load_chunk
from drbx.stencils.q_parallel_gradient import prepare_parallel_gradient


@pytest.fixture(scope='module')
def patch():
    path=Path(__file__).parent/'data/q05_actual_hsx/patch.npz'
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    meta=json.loads(str(arrays['metadata']))
    topology=topology_from_arrays(tuple(arrays[f'centers_{i}'] for i in range(3)),
                                  arrays['active'],arrays['aggregate'],arrays['volume'])
    def geom(points):
        for i in range(meta['geom_calls']):
            if np.array_equal(points,arrays[f'geom_points_{i}']):
                return tuple(arrays[f'geom_{k}_{i}'] for k in ('J','b','B'))
        raise AssertionError('new geometry query outside actual HSX fixture')
    def jacobian(points):
        for i in range(meta['jac_calls']):
            if np.array_equal(points,arrays[f'jac_points_{i}']):
                return arrays[f'jac_J_{i}']
        raise AssertionError('new wall query outside actual HSX fixture')
    prepared=[prepare_chunk(topology,arrays['owners'],arrays['ends'],geom,jacobian,
                            span=span,source_identity='fixture-traces',
                            geometry_identity='fixture-HSX',raw=arrays['raw'])
              for span in (1/16,1/32)]
    state=np.zeros((len(meta['fields']),len(topology.vol)),complex)
    state[:,arrays['state_ids']]=arrays['state_values']
    bc=[QBoundaryData(*(arrays[f'bc_{ai}_{key}'] for key in ('trace','query','tangent','normal')))
        for ai in range(2)]
    return arrays,prepared,state,bc


@pytest.mark.parametrize('ai', [0,1])
@pytest.mark.parametrize('kind', ['D','N'])
def test_prepare_and_staged_action_match_saved_global(patch,ai,kind):
    arrays,prepared,state,boundaries=patch
    q=prepared[ai]
    view=stage_q(q.runtime_view())
    assert all(isinstance(getattr(view,f.name),jax.Array)
               for f in fields(view) if f.name!='metadata')
    result=np.asarray(apply_q(view,state,boundaries[ai],kind=kind))
    expected=arrays['expected'][:,0 if kind=='D' else 1,ai,:].T
    np.testing.assert_allclose(result,expected,atol=1e-8,rtol=0)
    jit=jax.jit(lambda x,b:apply_q(view,x,b,kind=kind))
    np.testing.assert_allclose(jit(state,boundaries[ai]),expected,atol=1e-8,rtol=0)
    # All non-wall prescribed-data entries are finite but arbitrary padding.
    nonwall=q.raw//32**2 < 30
    changed=[np.array(v,copy=True) for v in boundaries[ai]]
    for a in changed:a[:,nonwall]=1.2345
    np.testing.assert_array_equal(jit(state,QBoundaryData(*changed)),jit(state,boundaries[ai]))
    assert np.count_nonzero(q.boundary_D_tangent[nonwall])==0
    assert np.count_nonzero(q.boundary_D_tangent[~nonwall])>0


def test_artifact_integrity_and_old_schema_rejection(patch,tmp_path):
    _,prepared,_,_=patch
    q=prepared[0]
    path=tmp_path/'q.npz'
    save_chunk(q,path)
    loaded=load_chunk(path,source_identity='fixture-traces',geometry_identity='fixture-HSX')
    np.testing.assert_array_equal(loaded.diffusion_D,q.diffusion_D)
    with pytest.raises(ValueError,match='identity'):
        load_chunk(path,source_identity='different-traces')
    leaked=q.boundary_D_tangent.copy();leaked[0]=1.
    with pytest.raises(ValueError,match='non-wall'):
        replace(q,boundary_D_tangent=leaked).validate()
    old=dict(q.metadata,schema='drbx.q-traced-diffusion.v1')
    with pytest.raises(ValueError,match='schema'):
        replace(q,metadata=old).validate()
    with np.load(path,allow_pickle=False) as z:
        payload={k:z[k] for k in z.files}
    metadata=json.loads(str(payload['metadata']))
    metadata['array_hashes'].pop('diffusion_D')
    payload['metadata']=np.array(json.dumps(metadata))
    np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='checksum manifest'):
        load_chunk(path)


@pytest.mark.parametrize('ai', [0, 1])
@pytest.mark.parametrize('kind', ['D', 'N'])
def test_direct_gradient_contraction_and_diffusion_replay(patch, ai, kind):
    arrays, prepared, state, boundaries = patch
    q = prepared[ai]
    g = prepare_parallel_gradient(q)
    center = stage_parallel_gradient(g.runtime_view())
    slots = stage_parallel_gradient(g.runtime_view(include_caps=True))
    assert center.nbytes < slots.nbytes < q.nbytes
    bc = boundaries[ai]
    actual = np.asarray(apply_raw_slot_gradients(slots, state, bc, kind=kind))
    rows = q.row_gradient_D if kind == 'D' else q.row_gradient_N
    # Independent Cartesian contraction of saved derivatives and frozen b.
    fetched = np.take(state, q.donor, axis=-1)
    cartesian = np.einsum('frd,rsad->frsa', fetched, rows)
    expected = np.einsum('frsa,rsa->frs', cartesian, q.magnetic_b)
    if kind == 'D':
        expected += np.einsum('frj,rsaj,rsa->frs', bc.dirichlet_trace,
                              q.boundary_gradient_D_trace, q.magnetic_b)
        expected += np.einsum('frsa,rsa->frs',
                              np.pad(bc.dirichlet_tangent, ((0,0),(0,0),(0,0),(1,0))),
                              np.where((q.raw//32**2>=30)[:,None,None],q.magnetic_b,0.))
    else:
        expected += np.einsum('frj,rsaj,rsa->frs', bc.neumann_normal,
                              q.boundary_gradient_N_normal, q.magnetic_b)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    projected = np.einsum('frm,rm->fr',
                          np.take(expected[:,:,2], q.owner_raw, axis=-1), q.owner_weight)
    np.testing.assert_allclose(apply_parallel_gradient(center,state,bc,kind=kind),
                               projected,atol=2e-12,rtol=0)
    # Frozen diffusion recontracts cap and center gradients, including lifts.
    diffusion_raw = np.einsum('frs,rs->fr', actual, q.magnetic_L)
    frozen_coeff = q.diffusion_D if kind == 'D' else q.diffusion_N
    frozen_raw = np.einsum('frd,rd->fr', fetched, frozen_coeff)
    if kind == 'D':
        frozen_raw += np.einsum('frj,rj->fr',bc.dirichlet_trace,q.boundary_D_node)
        frozen_raw += np.einsum('frsa,rsa->fr',bc.dirichlet_tangent,q.boundary_D_tangent)
    else:
        frozen_raw += np.einsum('frj,rj->fr',bc.neumann_normal,q.boundary_N_normal)
    np.testing.assert_allclose(diffusion_raw,frozen_raw,atol=1e-8,rtol=0)
    np.testing.assert_allclose(np.einsum('frm,rm->fr',
        np.take(diffusion_raw,q.owner_raw,axis=-1),q.owner_weight),
        apply_q(stage_q(q.runtime_view()),state,bc,kind=kind),atol=1e-8,rtol=0)


@pytest.mark.parametrize('kind', ['D','N'])
def test_direct_gradient_jit_batch_affine_and_boundary(patch, kind):
    _, prepared, state, boundaries = patch
    q=prepared[0]; bc=boundaries[0]
    view=stage_parallel_gradient(prepare_parallel_gradient(q).runtime_view())
    fn=jax.jit(lambda x,b:apply_parallel_gradient(view,x,b,kind=kind))
    np.testing.assert_allclose(fn(state,bc),apply_parallel_gradient(view,state,bc,kind=kind),atol=2e-12)
    for i in (0,1):
        single=QBoundaryData(*(v[i] for v in bc))
        np.testing.assert_allclose(fn(state[i],single),fn(state,bc)[i],atol=2e-12)
    zero=QBoundaryData(*(np.zeros_like(v) for v in bc))
    state_jvp=jax.jvp(lambda x:fn(x,bc),(state,),(np.ones_like(state),))[1]
    np.testing.assert_allclose(state_jvp,fn(np.ones_like(state),zero),atol=2e-12)
    bc_jvp=jax.jvp(lambda b:fn(state,b),(bc,),
                   (QBoundaryData(*(np.ones_like(v) for v in bc)),))[1]
    ones_bc=QBoundaryData(*(np.ones_like(v) for v in bc))
    np.testing.assert_allclose(bc_jvp,fn(np.zeros_like(state),ones_bc),atol=2e-12)
    changed=QBoundaryData(*(v+0.125 for v in bc))
    cache=fn._cache_size(); np.testing.assert_allclose(fn(state,changed),
        apply_parallel_gradient(view,state,changed,kind=kind),atol=2e-12)
    assert fn._cache_size()==cache
    nonwall=q.raw//32**2<30
    poisoned=[np.array(v,copy=True) for v in bc]
    for v in poisoned:v[:,nonwall]=987.25
    np.testing.assert_array_equal(fn(state,QBoundaryData(*poisoned)),fn(state,bc))
    assert np.max(np.abs(np.asarray(fn(state,changed)-fn(state,bc))))>0
    if kind=='D':
        tangent=[np.array(v,copy=True) for v in bc]
        tangent[2][:]=0; tangent[2][:,~nonwall]=0.25
        assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*tangent))-fn(state,bc))))>0
        value=[np.array(v,copy=True) for v in bc];value[1][:]=42
        np.testing.assert_array_equal(fn(state,QBoundaryData(*value)),fn(state,bc))
    else:
        normal=[np.array(v,copy=True) for v in bc];normal[3][:,~nonwall]=0.25
        assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*normal))-fn(state,bc))))>0


def test_direct_gradient_constant_real_and_input_contract(patch):
    _,prepared,state,_=patch
    q=prepared[0]
    g=prepare_parallel_gradient(q)
    view=stage_parallel_gradient(g.runtime_view())
    nr=len(q.raw)
    compatible=QBoundaryData(np.ones((nr,35)),np.ones((nr,3)),
                             np.zeros((nr,3,2)),np.zeros((nr,35)))
    homogeneous=QBoundaryData(np.zeros((nr,35)),np.zeros((nr,3)),
                              np.zeros((nr,3,2)),np.zeros((nr,35)))
    result=np.asarray(apply_parallel_gradient(view,np.ones(q.metadata['n_owner']),
                                              compatible,kind='D'))
    assert result.dtype==np.float64
    assert np.max(np.abs(result))<1e-7
    assert np.max(np.abs(np.asarray(apply_parallel_gradient(view,
        np.ones(q.metadata['n_owner']),homogeneous,kind='N'))))<1e-7
    with pytest.raises(ValueError,match='shape'):
        apply_parallel_gradient(view,state,compatible,kind='D')
    with pytest.raises(ValueError,match='center-only'):
        apply_parallel_gradient(g.runtime_view(include_caps=True),state,
                                compatible,kind='D')
