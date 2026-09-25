"""Midpoint reference contracts; actual-HSX checks are recorded separately."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT),str(ROOT/'src')]
from p05_structured_global import numerics as p05
from p06_structured_global import numerics as p06

def p07_campaign():
    directory=ROOT/'scripts/p07_combined_global';sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('p07_midpoint_campaign',directory/'campaign.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_midpoint_cell_rule_is_one_center_with_logical_volume():
    ctx=SimpleNamespace(x_faces=np.array([0.,2.]),y_faces=np.array([0.,3.]),z_faces=np.array([0.,4.]))
    p,w=p06._cell_quadrature(ctx,np.array([[0,0,0]]),1)
    np.testing.assert_array_equal(p,[[[1.,1.5,2.]]]);np.testing.assert_array_equal(w,[[24.]])

def test_p05_reference_uses_raw_mass_not_midpoint_metric_volume(monkeypatch):
    t=SimpleNamespace(n=2,faces=[np.array([0.,.5,1.])]*3,rv=np.array([2.,5.,1.,1.,1.,1.,1.,1.]),ro=np.array([0,0,1,1,2,2,3,3]))
    class Reference:
        def _metric(self,p):return {'J':np.full(len(p),3.),'B':np.ones(len(p)),'bcov':np.tile([0.,0.,1.],(len(p),1))}
    seen=[]
    def fields(ref,p,**kw):
        seen.append(kw);g=np.zeros((len(p),3,len(p05.FIELDS)));g[:,0,0]=1;g[:,1,1]=2
        return np.zeros((len(p),len(p05.FIELDS))),g
    monkeypatch.setattr(p05,'fields',fields)
    out=p05.reference_cells(t,Reference(),np.array([0,1]))
    np.testing.assert_array_equal(out['volume'],[2.,5.])
    np.testing.assert_allclose(out['numerator'][:,0],np.array([2.,5.])*(-2/3))
    assert seen==[{'full_omega_gradient':True}]

def test_p05_omega_gradient_includes_normal_and_clips_radial_queries(monkeypatch):
    queries=[]
    def omega(ref,p):
        queries.extend(p[:,0]);return p[:,0]**4+2*p[:,1]**3+3*p[:,2]**2
    monkeypatch.setattr(p05,'omega',omega)
    p=np.array([[.4,.2,.3],[1e-5,.3,.4],[1-1e-5,.4,.5]])
    g=p05.omega_derivative(SimpleNamespace(finite_difference_step=2e-4),p)
    np.testing.assert_allclose(g,np.column_stack([4*p[:,0]**3,6*p[:,1]**2,6*p[:,2]]),atol=2e-10,rtol=1e-8)
    assert min(queries)>0 and max(queries)<1

def test_p07_midpoint_uses_raw_volume_and_no_integrated_stage(monkeypatch):
    c=p07_campaign();t=SimpleNamespace(g=None,faces=None,rv=np.array([2.,5.]))
    def reference(ctx,ids,order):
        assert order==1
        return {'numerator':np.array([[6.]*4,[12.]*4]),'continuous_volume':np.array([3.,4.])}
    monkeypatch.setattr(c.k.num,'cell_chunk',reference)
    result=c.reference_cells(t,None,np.array([0,1]))
    np.testing.assert_array_equal(result['numerator'],[[4.]*4,[15.]*4])
    np.testing.assert_array_equal(result['continuous_volume'],[2.,5.])
    assert c.reference_orders('reference')==(1,)
    assert c.reference_orders('control')==(1,'1_halfstep')
    assert 'face_control' not in c.STAGES

def test_p06_halfstep_controls_restore_reference_and_do_not_integrate(monkeypatch):
    calls=[];ref=SimpleNamespace(finite_difference_step=2e-4)
    def reference(ctx,r,ids,order,time):
        calls.append((order,r.finite_difference_step))
        return {'volume_physical':np.ones(len(ids)),'volume_evolution':np.ones(len(ids)),
                'numerator_physical':np.zeros((4,3,len(ids),4)),
                'numerator_evolution':np.zeros((4,3,len(ids),4))}
    monkeypatch.setattr(p06,'_reference_on_raw_cells',reference)
    out,_=p06._compute_reference_control(None,ref,np.array([0,1]),time_value=.37)
    assert calls==[(1,2e-4),(1,1e-4)] and ref.finite_difference_step==2e-4
    assert 'numerator_evolution_q1_halfstep' in out
    assert not any('q3' in name or 'q5' in name or 'q7' in name for name in out)

@pytest.mark.parametrize('name',['p05_structured_global','p06_structured_global','p07_combined_global'])
def test_frozen_midpoint_configuration_disables_integrated_controls(name):
    cfg=json.loads((ROOT/'scripts'/name/'configuration.json').read_text())
    assert cfg['integrated_reference_controls'] is False
    assert cfg.get('candidate_face_quadrature',cfg.get('candidate_face_order'))==3
    assert cfg.get('reference_quadrature',cfg.get('reference_order',cfg.get('reference_volume_quadrature')))==1
