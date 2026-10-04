"""Compact N32/N48/N64 real-HSX P07 integrated face replay."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor

from drbx.native.fci_perpendicular_integrated_rows import (
    IntegratedFaceBatch, IntegratedFacePayload, IntegratedFacePlan,
    apply_integrated_face_rows, scatter_integrated_face_flux,
    save_integrated_face_plan, load_integrated_face_plan,
)
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays

DATA=Path(__file__).parent/'data/p_shared_face_rows'


def fixture_payload(arrays):
    batches=tuple(IntegratedFaceBatch(*(arrays[f'batch_{q}_{name}']
                   for name in IntegratedFaceBatch._fields))
                   for q in range(int(arrays['batch_count'])))
    return IntegratedFacePayload(batches,arrays['lower_owner'],arrays['upper_owner'],
        arrays['owner_volume'],int(arrays['boundary_query_count']),int(arrays['face_count']))


def test_tensor_normal_row_convention():
    tensor=np.arange(2*9*3*3,dtype=float).reshape(2,9,3,3)
    weights=np.ones((2,9))*.25
    result=contract_face_tensor(weights,tensor,np.array([0,2]))
    np.testing.assert_array_equal(result[0],tensor[0,:,0,:]*.25)
    np.testing.assert_array_equal(result[1],tensor[1,:,2,:]*.25)


@pytest.mark.parametrize('n',(32,48,64))
def test_integrated_real_hsx_saved_and_heldout(n):
    with np.load(DATA/f'P07_N{n}.npz',allow_pickle=False) as z:
        arrays={name:z[name] for name in z.files}
    payload=fixture_payload(arrays)
    fn=jax.jit(lambda fields,boundary:apply_integrated_face_rows(payload,fields,boundary))
    for prefix,expected in (('saved','expected_saved_flux'),('heldout','expected_heldout_flux')):
        fields=jnp.asarray(arrays[f'{prefix}_fields'])
        boundary=BoundaryArrays(arrays[f'{prefix}_boundary_values'],arrays[f'{prefix}_boundary_tangential'])
        result=np.asarray(fn(fields,boundary))
        np.testing.assert_allclose(result,arrays[expected],rtol=0,atol=2e-9)
        if prefix=='heldout':
            tangent=jnp.ones_like(fields)*0.01
            _,linear=jax.jvp(lambda f:fn(f,boundary),(fields,),(tangent,))
            eps=1e-5
            finite=(fn(fields+eps*tangent,boundary)-fn(fields-eps*tangent,boundary))/(2*eps)
            np.testing.assert_allclose(linear,finite,rtol=1e-7,atol=1e-9)
    result=np.asarray(scatter_integrated_face_flux(payload,jnp.asarray(arrays['expected_saved_flux'])))
    for owner in arrays['complete_owners']:
        position=int(np.searchsorted(arrays['owner_ids'],owner))
        expected=np.zeros(result.shape[1])
        for face,flux in enumerate(arrays['expected_saved_flux']):
            if payload.lower_owner[face]==position:expected-=flux
            if payload.upper_owner[face]==position:expected+=flux
        expected/=payload.owner_volume[position]
        np.testing.assert_allclose(result[position],expected,rtol=0,atol=2e-9)
    assert np.all(arrays['expected_saved_flux'][arrays['family']==0]==0)
    assert np.any(arrays['family']==1) and np.any(arrays['family']==7)
    keys=arrays['face_keys']
    assert np.any((keys[:,0]==1)&(keys[:,2]==0))
    assert np.any((keys[:,0]==2)&(keys[:,3]==0))
    constant=BoundaryArrays(np.ones_like(arrays['saved_boundary_values']),
                            np.zeros_like(arrays['saved_boundary_tangential']))
    zero=np.asarray(fn(jnp.ones((len(arrays['saved_fields']),arrays['saved_fields'].shape[1])),constant))
    np.testing.assert_allclose(zero,0,rtol=0,atol=2e-9)


def test_integrated_cache_identity(tmp_path):
    with np.load(DATA/'P07_N32.npz',allow_pickle=False) as z:
        arrays={name:z[name] for name in z.files}
    payload=fixture_payload(arrays)
    plan=IntegratedFacePlan(payload,np.zeros((int(payload.boundary_query_count),3)),(),())
    path=tmp_path/'rows.npz'
    save_integrated_face_plan(path,plan,{'tensor':'fixed','bc':'dirichlet'})
    loaded=load_integrated_face_plan(path,{'tensor':'fixed','bc':'dirichlet'})
    for actual,expected in zip(loaded.payload.batches,plan.payload.batches,strict=True):
        np.testing.assert_array_equal(actual.weights,expected.weights)
    with pytest.raises(ValueError,match='identity'):
        load_integrated_face_plan(path,{'tensor':'changed','bc':'dirichlet'})
