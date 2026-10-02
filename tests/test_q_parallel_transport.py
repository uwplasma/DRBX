"""Q07 nonlinear transport checks on portable actual-HSX rows."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
jax=pytest.importorskip('jax')
sys.path.insert(0,str(Path(__file__).parent))
from test_q_parallel_hsx_portable import patch
from drbx.native.q_parallel import QBoundaryData
from drbx.stencils.q_parallel_transport import prepare_density_transport
from drbx.stencils.q_parallel_divergence import prepare_tube_divergence
from drbx.native.q_parallel_divergence import apply_parallel_divergence,stage_parallel_divergence
from drbx.native.q_parallel_transport import (stage_density_transport,apply_density_transport,
    apply_raw_density_flux,apply_raw_density_transport)


def explicit(q,x,bc,kind):
    rows=q.row_value_D if kind=='D' else q.row_value_N
    value=np.einsum('frd,rsd->frs',x[:,q.donor],rows)
    if kind=='D':
        value+=np.einsum('frj,rsj->frs',bc.dirichlet_trace,q.boundary_value_D_trace)
        value+=bc.dirichlet_query_value*(q.raw//32**2>=30)[None,:,None]
    else:value+=np.einsum('frj,rsj->frs',bc.neumann_normal,q.boundary_value_N_normal)
    return value


@pytest.mark.parametrize('nk,vk',[('D','D'),('N','N'),('D','N'),('N','D')])
def test_density_product_bc_projection_jit_and_bilinear_jvp(patch,nk,vk):
    _,qq,state,bcs=patch;q=qq[1];bc=bcs[1]
    view=stage_density_transport(prepare_density_transport(q))
    x=state[:2];bx=QBoundaryData(*(a[:2] for a in bc))
    y=state[2:4];by=QBoundaryData(*(a[2:4] for a in bc))
    ns=explicit(q,x,bx,nk);vs=explicit(q,y,by,vk);flux=ns*vs
    raw=-np.einsum('frs,rs->fr',flux,q.magnetic_L)
    expected=np.sum(raw[:,q.owner_raw]*q.owner_weight[None,:,:],axis=-1)
    fn=lambda a,b,c,d:apply_density_transport(view,a,b,c,d,density_kind=nk,velocity_kind=vk)
    np.testing.assert_allclose(apply_raw_density_flux(view,x,y,bx,by,density_kind=nk,velocity_kind=vk),flux,atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(apply_raw_density_transport(view,x,y,bx,by,density_kind=nk,velocity_kind=vk),raw,atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(fn(x,y,bx,by),expected,atol=1e-9,rtol=1e-12)
    jit=jax.jit(fn);np.testing.assert_allclose(jit(x,y,bx,by),expected,atol=1e-9,rtol=1e-12)
    dx=.3*x+.1;dy=-.2*y+.07
    dbx=QBoundaryData(*(a*.3+.05 for a in bx));dby=QBoundaryData(*(a*-.2+.03 for a in by))
    tangent=jax.jvp(fn,(x,y,bx,by),(dx,dy,dbx,dby))[1]
    product_rule=fn(dx,y,dbx,by)+fn(x,dy,bx,dby)
    np.testing.assert_allclose(tangent,product_rule,atol=2e-9,rtol=1e-11)
    eps=1e-5
    fd=(fn(x+eps*dx,y+eps*dy,QBoundaryData(*(a+eps*b for a,b in zip(bx,dbx))),QBoundaryData(*(a+eps*b for a,b in zip(by,dby))))-
        fn(x-eps*dx,y-eps*dy,QBoundaryData(*(a-eps*b for a,b in zip(bx,dbx))),QBoundaryData(*(a-eps*b for a,b in zip(by,dby)))))/(2*eps)
    np.testing.assert_allclose(tangent,fd,atol=2e-7,rtol=1e-7)
    poisoned=[np.array(a,copy=True) for a in bx]
    for a in poisoned:a[:,q.raw//32**2<30]=7
    np.testing.assert_array_equal(fn(x,y,QBoundaryData(*poisoned),by),fn(x,y,bx,by))
    # Physical-normal data are live inputs, not substituted as radial derivatives.
    if nk=='N':
        changed=bx._replace(neumann_normal=bx.neumann_normal+.3)
        assert np.max(abs(np.asarray(fn(x,y,changed,by)-fn(x,y,bx,by))))>1e-8
        ignored=bx._replace(dirichlet_trace=bx.dirichlet_trace+19,dirichlet_query_value=bx.dirichlet_query_value-12)
        np.testing.assert_array_equal(fn(x,y,ignored,by),fn(x,y,bx,by))
    else:
        ignored=bx._replace(dirichlet_tangent=bx.dirichlet_tangent+19)
        np.testing.assert_array_equal(fn(x,y,ignored,by),fn(x,y,bx,by))


@pytest.mark.parametrize('kind',['D','N'])
def test_density_constant_factor_and_zero_velocity(patch,kind):
    _,qq,state,bcs=patch;q=qq[1];view=stage_density_transport(prepare_density_transport(q));D=stage_parallel_divergence(prepare_tube_divergence(q))
    nr=len(q.raw);x=state[0];bc=QBoundaryData(*(a[0] for a in bcs[1]))
    zero=QBoundaryData(np.zeros((nr,35)),np.zeros((nr,3)),np.zeros((nr,3,2)),np.zeros((nr,35)))
    one=zero._replace(dirichlet_trace=np.ones((nr,35)),dirichlet_query_value=np.ones((nr,3)))
    apply=lambda n,v,nb,vb:apply_density_transport(view,n,v,nb,vb,density_kind=kind,velocity_kind=kind)
    expected=-np.asarray(apply_parallel_divergence(D,x,bc,kind=kind))
    np.testing.assert_allclose(apply(np.ones_like(x),x,one,bc),expected,atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(apply(x,np.ones_like(x),bc,one),expected,atol=1e-8,rtol=1e-12)
    np.testing.assert_array_equal(apply(x,np.zeros_like(x),bc,zero),0)
    # Constant flux generally has div(b), so it must not be forced to zero.
    k=-np.sum(q.magnetic_L,axis=-1)
    projected=np.sum(k[q.owner_raw]*q.owner_weight,axis=-1)
    np.testing.assert_allclose(apply(np.ones_like(x),np.ones_like(x),one,one),projected,atol=1e-8,rtol=0)


def test_contract_and_shape_failures(patch):
    _,qq,state,bcs=patch
    with pytest.raises(ValueError,match='h/32'):prepare_density_transport(qq[0])
    q=qq[1];view=stage_density_transport(prepare_density_transport(q));x=state[0];bc=QBoundaryData(*(a[0] for a in bcs[1]))
    with pytest.raises(ValueError,match='shapes must match'):
        apply_density_transport(view,x,x[:-1],bc,bc,density_kind='D',velocity_kind='D')
    with pytest.raises(ValueError,match='kind must'):
        apply_density_transport(view,x,x,bc,bc,density_kind='invalid',velocity_kind='D')
    assert view.nbytes<q.nbytes
    before=prepare_density_transport(q).metadata['identity'];changed=q.magnetic_L.copy();changed[0,0]+=1e-8
    # In-memory preparation hashes live arrays; persisted Q loads separately
    # enforce their saved checksum manifest. Changed weights cannot reuse a key.
    assert prepare_density_transport(replace(q,magnetic_L=changed)).metadata['identity']!=before
    assert prepare_density_transport(q).metadata['identity']==before
