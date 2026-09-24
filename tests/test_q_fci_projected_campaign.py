"""Arithmetic, domain policy and resumability of the standalone campaign."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.q_fci_projected_campaign import kernel as k
from scripts.q_fci_projected_campaign import campaign as c


def test_fast_fields_match_frozen_mms():
    conf=c.config();ctx={'fields':{}}
    for name,d in conf['fields'].items():
        ctx['fields'][name]={'field':k.q.ManufacturedField(m=d['m'],amplitude=d['amplitude'],lam=d['lambda'],chi=d['chi'],eta_period=d['eta_period'],radial_profile=d['radial_profile'],field_kind=name)}
    rng=np.random.default_rng(4);p=rng.uniform(size=(47,3))*[1,2*np.pi,2*np.pi]
    v,g=k.field_values_gradients(p);expected=k.q.fields(ctx,p)
    np.testing.assert_allclose(v,expected[0],rtol=0,atol=4e-16)
    np.testing.assert_allclose(g,expected[1],rtol=5e-14,atol=4e-16)


def artificial_tracer(seeds,delta,steps):
    end=seeds.copy();end[:,0]+=delta;end[:,2]+=delta
    alive=(end[:,0]>0)&(end[:,0]<1)
    return [end,abs(delta),alive,np.zeros(len(seeds),bool),np.ones(len(seeds)),np.ones(len(seeds))],len(seeds)


def test_domain_halving_reuses_half_and_keeps_ordinary_legs():
    points=np.array([[.5,.2,.6],[.99,.3,.6]]);seeds=np.tile(points,(4,1));delta=np.repeat([-.1,.1,-.05,.05],2)
    calls=[]
    def tracer(s,d,steps):calls.append(len(s));return artificial_tracer(s,d,steps)
    ctx={'config':{'policy':{'trace_substeps':64}}}
    strict,*_=k.resolve_domain_legs(ctx,seeds,delta,tracer,0)
    a,work,real,levels,bad=k.resolve_domain_legs(ctx,seeds,delta,tracer,8)
    assert bad==2 and np.array_equal(levels,[0,4]) and a[2].all()
    assert real==16 and work==16  # eight original + two per reduction
    np.testing.assert_array_equal(a[0][::2],strict[0][::2])
    span=.2*2.**(-levels);eta=a[0][:,2].reshape(4,2)
    full=(eta[1]-eta[0])/span;half=(eta[3]-eta[2])/(span/2)
    np.testing.assert_allclose((4*half-full)/3,1,atol=3e-14)
    # Wrong original denominator cannot pass this control.
    wrong=(4*(eta[3]-eta[2])/.1-(eta[1]-eta[0])/.2)/3
    assert abs(wrong[1]-1)>.9


def test_bad_field_is_not_repaired_by_shortening():
    def tracer(s,d,steps):
        a,n=artificial_tracer(s,d,steps);a[3][0]=True;return a,n
    with pytest.raises(ValueError,match='shortening forbidden'):
        k.resolve_domain_legs({'config':{'policy':{'trace_substeps':64}}},np.tile([[.5,0.,0.]],(4,1)),np.array([-.1,.1,-.05,.05]),tracer,8)


def test_richardson_cancels_cubic_error_at_different_spans():
    span=np.array([.2,.05,.0125]);a=.7
    for power in range(5):
        val=lambda s:(a+s)**power
        full=(val(span/2)-val(-span/2))/span
        half=(val(span/4)-val(-span/4))/(span/2)
        np.testing.assert_allclose((4*half-full)/3,power*a**(power-1) if power else 0,rtol=1e-12,atol=3e-14)


def test_periodic_four_plane_transfer():
    centers=(np.arange(16)+.5)*2*np.pi/16
    class Model:
        grid=type('Grid',(),{'z':type('Axis',(),{'centers':centers})()})()
        def endpoint_pair(self,p,include_control):
            assert not include_control
            plane=int(np.argmin(abs(centers-p[2])))
            return np.array([plane]),np.ones(1),None,{'chosen':{'max_scaled_distance':0.,'residual':0.,'condition':1.}}
    ctx={'N':16,'model':Model(),'volume':np.ones(16)}
    p=np.array([[.5,.4,centers[6]+.23*(2*np.pi/16)]])
    G,meta=k.endpointmap(ctx,p);shift=p.copy();shift[:,2]+=2*np.pi;G2,_=k.endpointmap(ctx,shift)
    np.testing.assert_allclose(G.toarray(),G2.toarray(),atol=2e-14)
    for power in range(4):np.testing.assert_allclose(G@(centers**power),p[:,2]**power,rtol=1e-13)
    p[:,2]=centers[6];_,m=k.endpointmap(ctx,p);assert m[0]['plane_count']==1


def test_checkpoint_rejects_corruption_and_wrong_identity(tmp_path):
    p=tmp_path/'batch.npz';c.save(p,ids=np.array([3,7]),v=np.arange(6));c.receipt(p,'frozen',[3,7])
    assert c.completed(p,'frozen',[3,7])
    with pytest.raises(RuntimeError):c.completed(p,'changed',[3,7])
    with p.open('ab') as f:f.write(b'x')
    with pytest.raises(RuntimeError):c.completed(p,'frozen',[3,7])


def test_csr_and_shared_face_assembly():
    F=sparse.csr_matrix([[1.,-1.,0.],[0.,2.,-2.]])
    np.testing.assert_array_equal(k.unpack_csr('m',k.pack_csr('m',F)).toarray(),F.toarray())
    ctx={'lower':np.array([0,1]),'upper':np.array([1,2])};out=np.zeros((3,1));flux=np.array([[3.],[5.]])
    c.accumulate(ctx,np.array([0,1]),flux,out)
    np.testing.assert_array_equal(out[:,0],[3.,2.,-5.]);assert sum(out[:,0])==0
