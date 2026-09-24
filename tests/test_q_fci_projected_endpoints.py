"""Equivalent endpoint optimization on compact, actual-HSX owner-plane fixtures."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.q_fci_return_campaign.endpoints import OwnerMoments, EXPS4
from scripts.q_fci_projected_campaign.endpoints import CachedOwnerMoments

DATA=Path(__file__).parent/'data/q_projected_endpoints'


def model(n):
    manifest=json.loads((DATA/'manifest.json').read_text())['records'][str(n)]
    path=DATA/manifest['fixture']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    with np.load(path,allow_pickle=False) as z:
        a=OwnerMoments.__new__(OwnerMoments)
        a.N=n;a.labels=z['labels'];a.eta=z['eta'];a.centroid=z['centroid']
        a.absolute={pq:z['absolute'][i] for i,pq in enumerate(EXPS4)}
        a.grid=SimpleNamespace(**{axis:SimpleNamespace(centers=z[f'{axis}_centers'],faces=z[f'{axis}_faces']) for axis in 'xyz'})
        ids=np.arange(len(a.eta));a.by_plane={0:(ids,cKDTree(a.centroid))}
        return a,z['points']


def same_pair(a,b):
    for i in (0,1,2):
        if a[i] is None:assert b[i] is None
        else:np.testing.assert_array_equal(a[i],b[i])
    assert a[3]==b[3]


@pytest.mark.parametrize('n',[32,48,64])
def test_hsx_stencils_coefficients_metadata_and_controls_are_identical(n):
    base,points=model(n);fast=CachedOwnerMoments.from_model(base)
    assert fast.absolute is base.absolute and fast.labels is base.labels
    for p in points:
        for control in (False,True):
            same_pair(base.endpoint_pair(p,include_control=control),fast.endpoint_pair(p,include_control=control))
        assert fast._endpoint_cache is None and not fast._in_endpoint
    # Helper calls outside endpoint_pair must not retain stale moment translations.
    _,_,_,xy,scale,ids,_=fast.geometry(points[0])
    assert fast._endpoint_cache is None
    for shift in (0.,.002):
        np.testing.assert_array_equal(base.local(ids,xy+shift,scale,EXPS4),fast.local(ids,xy+shift,scale,EXPS4))


@pytest.mark.parametrize('n',[32,48,64])
def test_fallback_control_and_failed_call_cleanup(n):
    class ForcedBaseline(OwnerMoments):
        def fit(self,*a,**kw):
            r=super().fit(*a,**kw)
            if r['degree']==3 and not r['label'].endswith('fallback'):r['feasible']=False
            return r
    class ForcedCached(CachedOwnerMoments):
        def fit(self,*a,**kw):
            r=super().fit(*a,**kw)
            if r['degree']==3 and not r['label'].endswith('fallback'):r['feasible']=False
            return r
    base,points=model(n);base.__class__=ForcedBaseline;fast=ForcedCached.from_model(base)
    for p in points[::4]:
        same_pair(base.endpoint_pair(p),fast.endpoint_pair(p))
    p=points[0].copy();p[2]+=.1*(2*np.pi/n)
    with pytest.raises(RuntimeError,match='off stored center planes'):fast.endpoint_pair(p)
    assert fast._endpoint_cache is None and not fast._in_endpoint
    same_pair(base.endpoint_pair(points[-1]),fast.endpoint_pair(points[-1]))
