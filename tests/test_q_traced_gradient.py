"""Traced-value gradient algebra, boundaries and JAX on actual HSX patches."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
jax = pytest.importorskip('jax')
sys.path.insert(0, str(Path(__file__).parent))
from test_q_parallel_hsx_portable import patch
from drbx.stencils.q_parallel import topology_from_arrays
from drbx.stencils.q_traced_gradient import prepare_traced_gradient
from drbx.native.q_traced_gradient import (apply_traced_gradient,
    apply_raw_traced_gradient, stage_traced_gradient)
from drbx.native.q_parallel import QBoundaryData


@pytest.mark.parametrize('ai', [0, 1])
@pytest.mark.parametrize('kind', ['D', 'N'])
def test_values_projection_jit_and_affine_bc(patch, ai, kind):
    _, prepared, state, boundaries = patch
    q = prepared[ai]; bc = boundaries[ai]
    view = stage_traced_gradient(prepare_traced_gradient(q))
    rows = q.row_value_D if kind == 'D' else q.row_value_N
    values = np.einsum('frd,rsd->frs', state[:, q.donor], rows)
    wall = q.raw//32**2 >= 30
    if kind == 'D':
        values += np.einsum('frj,rsj->frs', bc.dirichlet_trace, q.boundary_value_D_trace)
        values += bc.dirichlet_query_value*wall[None, :, None]
    else:
        values += np.einsum('frj,rsj->frs', bc.neumann_normal, q.boundary_value_N_normal)
    expected = (values[:, :, 1]-values[:, :, 0])*q.magnetic_b[None, :, 2, 2]/(q.metadata['span']*2*np.pi/32)
    np.testing.assert_allclose(apply_raw_traced_gradient(view, state, bc, kind=kind), expected, atol=2e-11, rtol=0)
    projected = np.einsum('frm,rm->fr', np.take(expected, q.owner_raw, axis=-1), q.owner_weight)
    fn = jax.jit(lambda x,b: apply_traced_gradient(view, x,b,kind=kind))
    np.testing.assert_allclose(fn(state,bc), projected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(jax.vmap(lambda x,b:apply_traced_gradient(view,x,b,kind=kind))(state,bc), projected,atol=2e-11)
    changed = QBoundaryData(*(x+.125 for x in bc)); cache=fn._cache_size()
    np.testing.assert_allclose(fn(state,changed),apply_traced_gradient(view,state,changed,kind=kind),atol=2e-11)
    assert cache==fn._cache_size()
    zero=QBoundaryData(*(np.zeros_like(x) for x in bc))
    np.testing.assert_allclose(jax.jvp(lambda x:fn(x,bc),(state,),(np.ones_like(state),))[1],fn(np.ones_like(state),zero),atol=2e-11)
    ones=QBoundaryData(*(np.ones_like(x) for x in bc))
    np.testing.assert_allclose(jax.jvp(lambda b:fn(state,b),(bc,),(ones,))[1],fn(np.zeros_like(state),ones),atol=2e-11)
    poison=[np.array(x,copy=True) for x in bc]
    for x in poison:x[:,~wall]=172.5
    np.testing.assert_array_equal(fn(state,QBoundaryData(*poison)),fn(state,bc))
    tangent=list(bc);tangent[2]=np.full_like(tangent[2],np.nan)
    np.testing.assert_array_equal(fn(state,QBoundaryData(*tangent)),fn(state,bc))
    # Individual nonzero wall BC contributions must survive (not merely shape checks).
    for index in ([0,1] if kind=='D' else [3]):
        changed=[np.array(x,copy=True) for x in bc]
        changed[index][:,wall] += np.arange(changed[index].shape[-1])*.1
        assert np.max(np.abs(np.asarray(fn(state,QBoundaryData(*changed))-fn(state,bc)))) > 1e-10


@pytest.mark.parametrize('ai', [0,1])
@pytest.mark.parametrize('kind', ['D','N'])
def test_constant_and_independent_radial_field(patch,ai,kind):
    arrays,prepared,_,_=patch;q=prepared[ai];view=prepare_traced_gradient(q)
    t=topology_from_arrays(tuple(arrays[f'centers_{i}'] for i in range(3)),arrays['active'],arrays['aggregate'],arrays['volume'])
    radial=np.bincount(t.ro,weights=t.rv*t.pts[:,0])/t.vol
    nr=len(q.raw);wall=q.raw//32**2>=30
    trace=np.zeros((nr,35));trace[wall]=1
    query=np.zeros((nr,3));query[wall]=1
    normal=np.zeros((nr,35));normal[wall]=q.boundary_wall_normal[wall,:,0]
    bc=QBoundaryData(trace,query,np.zeros((nr,3,2)),normal)
    exact=q.magnetic_b[:,2,2]*(q.slot_points[:,1,0]-q.slot_points[:,0,0])/(q.metadata['span']*2*np.pi/32)
    expected=np.sum(exact[q.owner_raw]*q.owner_weight,axis=-1)
    np.testing.assert_allclose(apply_traced_gradient(view,radial,bc,kind=kind),expected,atol=1e-10,rtol=0)
    constant=bc._replace(neumann_normal=np.zeros_like(normal))
    np.testing.assert_allclose(apply_traced_gradient(view,np.ones(t.vol.size),constant,kind=kind),0,atol=1e-10,rtol=0)


def test_geometry_identity_and_no_field_gradient_contraction(patch):
    _,prepared,_,_=patch;q=prepared[0];ref=prepare_traced_gradient(q)
    other=replace(q,row_gradient_D=np.zeros_like(q.row_gradient_D),
        row_gradient_N=np.zeros_like(q.row_gradient_N),magnetic_L=np.zeros_like(q.magnetic_L))
    got=prepare_traced_gradient(other)
    assert got.metadata['identity']==ref.metadata['identity']
    np.testing.assert_array_equal(got.coefficient_D,ref.coefficient_D)
    rev=prepare_traced_gradient(replace(q,magnetic_b=-q.magnetic_b))
    np.testing.assert_array_equal(rev.coefficient_D,-ref.coefficient_D)
    assert rev.metadata['identity']!=ref.metadata['identity']
    p=q.slot_points.copy();p[:,0,2]+=.0001;p[:,1,2]+=.0001
    with pytest.raises(ValueError,match='symmetric'):prepare_traced_gradient(replace(q,slot_points=p))
    b=q.magnetic_b.copy();b[:,2,2]=0
    with pytest.raises(ValueError,match='parameterize'):prepare_traced_gradient(replace(q,magnetic_b=b))
    b=q.magnetic_b.copy();b[:,0,2]*=-1
    with pytest.raises(ValueError,match='parameterize'):prepare_traced_gradient(replace(q,magnetic_b=b))
    a=q.boundary_value_D_trace.copy();a[0]=1
    with pytest.raises(ValueError,match='non-wall'):prepare_traced_gradient(replace(q,boundary_value_D_trace=a))


def test_runtime_rejects_wrong_shapes_and_kind(patch):
    _,prepared,state,boundaries=patch;v=prepare_traced_gradient(prepared[0]);bc=boundaries[0]
    with pytest.raises(ValueError,match='kind'):apply_traced_gradient(v,state,bc,kind='bad')
    with pytest.raises(ValueError,match='length'):apply_traced_gradient(v,state[:,:-1],bc,kind='D')
    with pytest.raises(ValueError,match='shape'):apply_traced_gradient(v,state,bc._replace(dirichlet_query_value=bc.dirichlet_query_value[:,:,:2]),kind='D')
    with pytest.raises(TypeError,match='floating'):apply_traced_gradient(v,state.real.astype(int),bc,kind='D')
