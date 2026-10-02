"""Fixed-row runtime invariants and local HSX artifact checks for traced Q."""
from pathlib import Path
import numpy as np
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from drbx.native.q_parallel import QBoundaryData,apply_q
from drbx.stencils.q_parallel import QRuntime,load_chunk,topology_from_arrays,FROZEN_IDENTITY
from drbx.stencils import q_parallel_primitives as primitives


def fixture():
    q=QRuntime(dict(n_owner=3),np.array([4,7]),np.array([[0,1,0],[1,2,0]],np.int32),
               np.array([[1.,2.,-1.],[3.,-2.,0.]]),np.array([[2.,-1.,1.],[1.,4.,0.]]),
               np.array([[.25]+[0.]*34,[.5]+[0.]*34]),np.ones((2,3,2))*.125,
               np.array([[.75]+[0.]*34,[-.25]+[0.]*34]),
               np.array([[0,1]],np.int32),np.array([[.25,.75]]))
    b=QBoundaryData(np.ones((2,35)),np.ones((2,3)),np.ones((2,3,2)),np.ones((2,35)))
    return q,b


def zeros(b):return QBoundaryData(*(np.zeros_like(v) for v in b))


def test_duplicate_projection_affine_jit_jvp_complex_batch():
    q,b=fixture();x=jnp.array([2.,-1.,3.]);d=np.asarray(apply_q(q,x,b,kind='D'));n=np.asarray(apply_q(q,x,b,kind='N'))
    assert d[0]==pytest.approx(.25*(-2.+1.)+.75*(-9.+1.25))
    assert n[0]==pytest.approx(.25*(2*2.+1.+2.+.75)+.75*(-1.+4*3.-.25))
    interior=np.asarray(apply_q(q,x,zeros(b),kind='D'))
    lift=np.asarray(apply_q(q,jnp.zeros_like(x),b,kind='D'))
    np.testing.assert_allclose(d,interior+lift,rtol=0,atol=1e-14)
    f=jax.jit(lambda values,boundary:apply_q(q,values,boundary,kind='N'))
    np.testing.assert_allclose(f(x,b),n,rtol=0,atol=1e-14)
    tangent=jax.jvp(lambda values:f(values,b),(x,),(jnp.ones_like(x),))[1]
    np.testing.assert_allclose(tangent,apply_q(q,jnp.ones_like(x),zeros(b),kind='N'))
    z=x.astype(jnp.complex128)*(1+2j)
    np.testing.assert_allclose(apply_q(q,z,b,kind='D'),interior*(1+2j)+lift)
    xb=jnp.stack((x,x*2));bb=QBoundaryData(*(np.stack((v,v)) for v in b))
    np.testing.assert_allclose(apply_q(q,xb,bb,kind='N')[0],n)


def test_real_hsx_roundtrip_identity_if_available():
    root=Path(__file__).resolve().parents[2]
    path=root/'work/q05_traced_extraction_20260929/review_v2/chunks/N32/q_000063_0_owners0_1.npz'
    if not path.exists():pytest.skip('local HSX frozen replay fixture unavailable')
    prepared=load_chunk(path,expected_identity=FROZEN_IDENTITY)
    assert prepared.metadata['families']=={'wall':1}
    with pytest.raises(ValueError,match='identity'):load_chunk(path,expected_identity='wrong')
    with pytest.raises(ValueError,match='span'):load_chunk(path,span=1/32)
    with pytest.raises(ValueError,match='geometry_identity'):load_chunk(path,geometry_identity='wrong')
    assert prepared.runtime_view().nbytes<prepared.nbytes


def test_real_hsx_owner_volumes_if_available():
    root=Path(__file__).resolve().parents[2]
    directory=root/'geometry_artifacts/rlp_convergence_32_48_64_20260917/32x32x32'
    result=root/'work/q_fci_selective_global_20260929/results_N32.npz'
    if not directory.exists() or not result.exists():pytest.skip('local HSX topology unavailable')
    with np.load(directory/'base_geometry.npz') as z:
        centers=tuple(z[f'grid.{a}.centers'].copy() for a in 'xyz')
    with np.load(directory/'rlp_topology.npz') as z:
        t=topology_from_arrays(centers,z['is_active_owner'],z['aggregate_id'],z['raw_volume'])
    with np.load(result) as z:
        np.testing.assert_array_equal(np.arange(len(t.vol)),z['owners'])
        np.testing.assert_allclose(t.vol,z['volume'],rtol=0,atol=0)
    assert len(t.order)==32**3
    assert np.array_equal(np.sort(t.order),np.arange(32**3))


def test_real_hsx_rank_repair_retains_28_if_available():
    root=Path(__file__).resolve().parents[2]
    path=root/'work/q05_traced_extraction_20260929/review_v2/chunks/N64/q_000122_0_owners88_1.npz'
    if not path.exists():pytest.skip('local N64 rank-repair fixture unavailable')
    prepared=load_chunk(path)
    assert len(prepared.raw)==2
    assert np.array_equal(prepared.choice,[0,0])
    for diagnostics in prepared.metadata['row_diagnostics']:
        assert all(min(rank)<15 for rank in diagnostics['plane_initial_rank'])
        assert diagnostics['plane_rank']==[15]*5
        assert all(len(donors)==28 for donors in diagnostics['plane_donors'])
        assert all(diagnostics['plane_exchanges'])


def test_real_hsx_rank_repair_polynomial_targets_if_available():
    root=Path(__file__).resolve().parents[2]
    path=root/'work/q05_traced_extraction_20260929/review_v2/chunks/N64/q_000122_0_owners88_1.npz'
    directory=root/'geometry_artifacts/rlp_convergence_32_48_64_20260917/64x64x64'
    if not path.exists() or not directory.exists():pytest.skip('local N64 rank fixture unavailable')
    prepared=load_chunk(path)
    with np.load(directory/'base_geometry.npz') as z:
        centers=tuple(z[f'grid.{a}.centers'].copy() for a in 'xyz')
    with np.load(directory/'rlp_topology.npz') as z:
        t=topology_from_arrays(centers,z['is_active_owner'],z['aggregate_id'],z['raw_volume'])
    for pos,rr in enumerate(prepared.raw):
        center=t.xy[rr];point=t.pts[rr];scale=max(1/t.n,point[0]*2*np.pi/t.n)
        raw_values=primitives.basis(t.xy,center,scale,primitives.EXP4)
        observed=np.column_stack([np.bincount(t.ro,weights=t.rv*raw_values[:,j],minlength=len(t.vol))/t.vol
                                  for j in range(len(primitives.EXP4))])
        ids=prepared.donor[pos,:prepared.row_count[pos]]
        expected,dr,dt=primitives.planar(prepared.slot_points[pos],center,scale,primitives.EXP4)
        actual=prepared.row_value_D[pos,:,:len(ids)]@observed[ids]
        gradient=np.einsum('sid,df->sif',prepared.row_gradient_D[pos,:,:,:len(ids)],observed[ids])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
        np.testing.assert_allclose(gradient,np.stack((dr,dt,np.zeros_like(dr)),axis=1),rtol=0,atol=1e-8)


def test_real_hsx_wall_ghost_query_matches_frozen_if_available():
    import sys
    root=Path(__file__).resolve().parents[2]
    source=root/'work/q_fci_diffusion_freeze_20260929/source'
    if not source.exists():pytest.skip('frozen HSX wall oracle unavailable')
    sys.path.insert(0,str(source))
    from frozen.model import context
    from frozen.wall import Wall as FrozenWall
    from drbx.stencils.q_parallel_wall import Wall as PreparedWall
    ctx,t=context(32,root,physical=True,magnetic=False)
    i,j,k=31,0,0
    frozen=FrozenWall(ctx,t,i,j,k)
    new=PreparedWall(lambda p:ctx['evaluator']._position_and_jacobian(p)[1],t,i,j,k)
    p=t.pts[(i*32+j)*32+k][None].copy();p[0,0]=1.0001
    for actual,expected in zip(new.maps(p),frozen.maps(p)):
        np.testing.assert_array_equal(actual,expected)
    assert np.isfinite(new.maps(p)[2]).all()


def test_real_hsx_complete_unit_and_subset_rows_if_available():
    root=Path(__file__).resolve().parents[2]
    directory=root/'work/q05_traced_extraction_20260929/review_v2/chunks/N64'
    full=directory/'q_000511_0.npz';subset=directory/'q_000511_0_owners0_1.npz'
    if not full.exists() or not subset.exists():pytest.skip('local complete-unit artifact unavailable')
    a=load_chunk(full);b=load_chunk(subset)
    assert a.raw[0]==b.raw[0] and a.owners[0]==b.owners[0]
    np.testing.assert_array_equal(a.donor[0],b.donor[0])
    np.testing.assert_array_equal(a.row_gradient_D[0],b.row_gradient_D[0])
    np.testing.assert_allclose(a.diffusion_D[0],b.diffusion_D[0],rtol=0,atol=1e-8)
    np.testing.assert_allclose(a.boundary_N_normal[0],b.boundary_N_normal[0],rtol=0,atol=1e-8)


def test_real_hsx_magnetic_action_coefficients_if_available():
    import sys
    root=Path(__file__).resolve().parents[2]
    source=root/'work/q_fci_diffusion_freeze_20260929/source'
    artifact=root/'work/q05_traced_extraction_20260929/review_v2/chunks/N32/q_000063_0_owners0_1.npz'
    trace=root/'work/q-fci-balanced28-dacf15f1-20260929T171025Z/global/N32/trace_000063.npz'
    if not source.exists() or not artifact.exists() or not trace.exists():
        pytest.skip('local frozen magnetic fixture unavailable')
    sys.path.insert(0,str(source))
    from frozen.model import context,slots_and_action
    prepared=load_chunk(artifact)
    ctx,t=context(32,root)
    with np.load(trace) as z:
        ix=np.flatnonzero(z['raw']==prepared.raw[0]);ends=z['ends'][ix]
    slots,b,L,*_=slots_and_action(ctx,t,prepared.raw,ends)
    np.testing.assert_allclose(prepared.slot_points,slots[:,:3],rtol=0,atol=0)
    np.testing.assert_allclose(prepared.magnetic_b,b[:,:3],rtol=0,atol=0)
    np.testing.assert_allclose(prepared.magnetic_L,L[:,0,:3],rtol=0,atol=1e-10)
