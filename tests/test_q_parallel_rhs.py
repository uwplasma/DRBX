"""Assembly regressions on portable actual-HSX rows; no new geometry claim."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pytest
jax = pytest.importorskip('jax')
import jax.numpy as jnp
from test_q_parallel_hsx_portable import patch
from drbx.native.q_parallel import QBoundaryData
from drbx.native.q_parallel_rhs import apply_six_field_rhs, stage_six_field_rhs, field_boundary
from drbx.native.q_parallel_material import apply_raw_material_transport, reconstruct_material_slots
from drbx.native.q_parallel_vorticity import apply_raw_vorticity_advection
from drbx.native.q_parallel_channels import apply_diffusion_channels
from drbx.stencils.q_parallel_divergence import KappaAudit
import drbx.stencils.q_parallel_rhs as prep


@pytest.fixture
def assembled(patch, monkeypatch):
    _, qs, values, bcs = patch
    outer, inner = qs
    # Algebra fixture only: actual C3 div(b) and B replay are in the bounded audit.
    nr = len(inner.raw)
    audit = KappaAudit(np.linspace(.01,.04,nr), np.zeros((3,nr)),
                       np.zeros(nr), np.linspace(.8,1.2,nr), ('fixture',))
    monkeypatch.setattr(prep, 'prepare_kappa', lambda *a, **kw: audit)
    base = np.array([1.,1.1,.9,.13,.08,.2])
    x = np.real(values[:6])*.001+base[:,None]
    def convert(b):
        z = [np.real(v[:6])*.001 for v in b]
        for i in (0,1): z[i] += base[:,None,None]
        return QBoundaryData(*z)
    bo, bi = map(convert,bcs)
    phi = .03*x[0]
    pb = QBoundaryData(*(a*.03 for a in field_boundary(bi,0)))
    return inner, outer, audit, x, bi, bo, phi, pb


@pytest.mark.parametrize('span',[1/16,1/32])
@pytest.mark.parametrize('kinds,pk',[(('D',)*6,'D'),(('N',)*6,'N'),(('D','N','D','N','D','N'),'N')])
def test_components_projection_and_prescribed_current(assembled,span,kinds,pk):
    inner,outer,audit,x,bi,bo,phi,pb=assembled
    before=inner.magnetic_L.copy()
    rt,_=prep.prepare_six_field_rhs(inner,outer,None,diffusion_span=span)
    np.testing.assert_array_equal(inner.magnetic_L,before)
    np.testing.assert_allclose(rt.material.magnetic_L.sum(-1),audit.kappa,atol=1e-12)
    np.testing.assert_array_equal(rt.diffusion.diffusion_D,
                                  (inner if span==1/32 else outer).diffusion_D)
    rt=stage_six_field_rhs(rt);coeff=np.arange(1,7)*.01
    call=lambda x,bi,bo,p,pb,c:apply_six_field_rhs(rt,x,bi,bo,p,pb,c,
        kinds=kinds,phi_kind=pk,tau=1.,mu=1836.)
    got=call(x,bi,bo,phi,pb,coeff)
    assert np.all(got.inputs_valid) and np.all(got.eigensystem_admissible)
    mat=apply_raw_material_transport(rt.material,x[:5],field_boundary(bi,slice(0,5)),
        field_boundary(bo,slice(0,5)),phi,pb,kinds=kinds[:5],phi_kind=pk,tau=1.,mu=1836.)
    om=apply_raw_vorticity_advection(rt.material,x[5],x[3],field_boundary(bi,5),
        field_boundary(bo,5),field_boundary(bi,3),omega_kind=kinds[5],velocity_kind=kinds[3])
    slots=reconstruct_material_slots(rt.material.inner,x[:5],field_boundary(bi,slice(0,5)),kinds=kinds[:5])
    dj=np.sum(rt.material.magnetic_L*slots[...,0]*(slots[...,3]-slots[...,4]),axis=-1)
    project=lambda a:np.sum(np.asarray(a)[inner.owner_raw]*inner.owner_weight[...,None],axis=-2)
    center=np.concatenate((mat.centered,(om.centered+rt.bmag**2/slots[:,2,0]*dj)[:,None]),axis=-1)
    corr=np.concatenate((mat.correction,om.correction[:,None]),axis=-1)
    diff=apply_diffusion_channels(rt.diffusion,x,bi if span==1/32 else bo,coeff,kinds=kinds).action.T
    np.testing.assert_allclose(got.centered,project(center),atol=1e-9)
    np.testing.assert_allclose(got.correction,project(corr),atol=1e-9)
    np.testing.assert_array_equal(got.diffusion,diff)
    np.testing.assert_allclose(got.combined,project(center)+project(corr)+diff,atol=1e-9)
    np.testing.assert_allclose(got.raw_current.divergence_physical,dj,atol=1e-10)
    np.testing.assert_allclose(got.raw_current.electron_ti_compensation+
        got.raw_current.electron_generalized_force,got.raw_current.electron_phi,atol=1e-10)
    compiled=jax.jit(call)(x,bi,bo,phi,pb,coeff)
    np.testing.assert_allclose(compiled.combined,got.combined,atol=2e-8,rtol=1e-10)
    forced=call(x,bi,bo,2*phi,QBoundaryData(*(2*a for a in pb)),coeff)
    np.testing.assert_array_equal(forced.combined[:,:4],got.combined[:,:4])
    np.testing.assert_array_equal(forced.combined[:,5],got.combined[:,5])
    expected=project(np.asarray(got.raw_current.electron_phi)[:,None])[:,0]
    np.testing.assert_allclose(forced.combined[:,4]-got.combined[:,4],expected,atol=2e-8)


def test_live_state_boundary_coefficient_ad_and_input_guards(assembled):
    inner,outer,_,x,bi,bo,phi,pb=assembled
    rt,_=prep.prepare_six_field_rhs(inner,outer,None,diffusion_span=1/32)
    rt=stage_six_field_rhs(rt);kinds=('D','N','D','N','D','N');coeff=np.arange(1,7)*.01
    fn=lambda x,bi,bo,p,pb,c:apply_six_field_rhs(rt,x,bi,bo,p,pb,c,kinds=kinds,
                                              phi_kind='D',tau=1.,mu=1836.)
    args=jax.tree.map(jnp.asarray,(x,bi,bo,phi,pb,coeff))
    tangent=jax.tree.map(lambda a:.02*a,args)
    # The inherited eigensystem freezes its spectral projectors for AD. Test
    # ordinary live state/BC derivatives on centered+diffusion, separately.
    target=lambda *a:fn(*a).centered+fn(*a).diffusion
    ad=jax.jvp(target,args,tangent)[1];eps=1e-4
    plus=jax.tree.map(lambda a,d:a+eps*d,args,tangent)
    minus=jax.tree.map(lambda a,d:a-eps*d,args,tangent)
    fd=(target(*plus)-target(*minus))/(2*eps)
    np.testing.assert_allclose(ad,fd,atol=5e-6,rtol=5e-5)
    # Phi and coefficients do not enter the frozen spectral basis; their full
    # combined-action JVP has an exact independent linear response.
    da=jax.tree.map(jnp.zeros_like,args)
    da=(*da[:3],args[3]*.02,jax.tree.map(lambda a:a*.02,args[4]),args[5]*.03)
    tangent=jax.jvp(lambda *a:fn(*a).combined,args,da)[1]
    base=fn(*args)
    extra=jnp.zeros_like(base.combined).at[...,4].set(jnp.sum(
        jnp.take(base.raw_current.electron_phi,rt.material.inner.owner_raw,axis=-1)*
        rt.material.inner.owner_weight,axis=-1)*.02)
    np.testing.assert_allclose(tangent,extra+.03*base.diffusion,atol=2e-8)
    assert not np.all(fn(x,bi,bo,phi,pb,-coeff).inputs_valid)
    assert not np.all(fn(x.at[0].set(-1) if hasattr(x,'at') else np.concatenate((-np.ones_like(x[:1]),x[1:])),bi,bo,phi,pb,coeff).inputs_valid)
    with pytest.raises(ValueError,match='six D/N'):
        apply_six_field_rhs(rt,x,bi,bo,phi,pb,coeff,kinds=('D',),phi_kind='D',tau=1.,mu=1836.)
    with pytest.raises(ValueError,match='diffusion span'):
        prep.prepare_six_field_rhs(inner,outer,None,diffusion_span=1/64)
    bad=replace(rt,material=replace(rt.material,metadata={'schema':'drbx.q-material.v1'}))
    with pytest.raises(ValueError,match='geometry-consistent'):
        apply_six_field_rhs(bad,x,bi,bo,phi,pb,coeff,kinds=kinds,phi_kind='D',tau=1.,mu=1836.)
