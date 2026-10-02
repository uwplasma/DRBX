"""Owner observation data receipt and reuse, without geometry or new tracing."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.q08_extraction_global import compute
from scripts.q08_extraction_global import common as c


def topology():
    pts=np.array([[.2,.3,.7],[.4,.5,.9],[.7,1.,1.2],[.8,1.5,1.7]])
    ro=np.array([0,0,1,1]);rv=np.array([1.,2.,3.,4.])
    return SimpleNamespace(pts=pts,ro=ro,rv=rv,vol=np.array([3.,7.]))


def test_observations_use_volumeweighted_primitives_then_accepted_omega(tmp_path):
    t=topology();result=compute._data(32,t,tmp_path,'test-input-identity')
    values,_=c.fields(t.pts);phi,_=c.phi_fields(t.pts)
    expected=np.zeros((c.NF,5,2));p=np.zeros((c.NF,2))
    for r in range(4):
        expected[:,:,t.ro[r]]+=t.rv[r]*values[r];p[:,t.ro[r]]+=t.rv[r]*phi[r]
    expected/=t.vol;p/=t.vol
    np.testing.assert_array_equal(result['state'][:,:5],expected)
    np.testing.assert_array_equal(result['state'][:,5],c.WOFF+np.einsum('cfo,f->co',expected,c.WC))
    np.testing.assert_array_equal(result['phi'],p)
    np.testing.assert_array_equal(result['raw_to_owner'],t.ro)
    again=compute._data(32,t,tmp_path,'test-input-identity')
    assert isinstance(again['state'],np.memmap)
    with pytest.raises(ValueError,match='identity/content'):
        compute._data(32,t,tmp_path,'stale-identity')


def test_tampered_owner_state_rejected_before_worker_apply(tmp_path):
    t=topology();compute._data(32,t,tmp_path,'identity')
    path=tmp_path/'data/N32/state.npy'
    state=np.load(path);state[0,0,0]+=1
    with path.open('wb') as stream:np.save(stream,state)
    with pytest.raises(ValueError,match='identity/content'):
        compute._data(32,t,tmp_path,'identity')
