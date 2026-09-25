"""Equivalent reference geometry with fewer exact metric queries."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perpendicular_structured.reference_geometry import metric_reuse, curvature_geometry
from hsx_mms_continuum_reference import ContinuumMmsReference


def analytic_reference():
    ref=ContinuumMmsReference.__new__(ContinuumMmsReference)
    ref.finite_difference_step=2e-4
    ref.eta_period=1.
    ref.enable_generalized_potential=False
    ref.calls=0
    def metric(q):
        ref.calls+=1
        u,t,e=q.T
        b=np.column_stack((.1+u,.2+np.sin(t),1+e))
        return dict(J=1+u*u,B=2+.1*np.cos(t),b=b,bcov=b,
                    gcontra=np.broadcast_to(np.eye(3),(len(q),3,3)).copy())
    ref._metric=metric
    return ref


def test_curvature_only_matches_full_geometry_and_reduces_queries():
    ref=analytic_reference()
    # Includes clipped radial derivative steps and periodic seams.
    p=np.array([[1e-5,0,0],[.5,1.1,.31],[1-1e-5,2*np.pi,1.]])
    old=ref.prepare(p);calls=ref.calls;ref.calls=0
    new=curvature_geometry(ref,p)
    for key in ('J','B','K'):
        np.testing.assert_array_equal(getattr(old,key),getattr(new,key))
    assert calls==40 and ref.calls==13


def test_exact_query_cache_is_scoped_and_restored_on_failure():
    ref=analytic_reference();original=ref._metric
    p=np.array([[.2,.3,.4]])
    with pytest.raises(RuntimeError):
        with metric_reuse(ref):
            ref._metric(p);ref._metric(p.copy())
            q=p.copy();q[0,0]=np.nextafter(q[0,0],1)
            ref._metric(q)
            assert ref.calls==2
            raise RuntimeError('restore')
    assert ref._metric is original
    with metric_reuse(ref):ref._metric(p)
    assert ref.calls==3
