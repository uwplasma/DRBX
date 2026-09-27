"""Analytic-correctness safeguards for the P07N field-derived Neumann catalogue.

Fast, no geometry inputs: exercises scripts/p07n_field_derived_global/fields.py
directly (loaded by file path, following the importlib.util pattern used by
test_p07_diffusion_global_campaign.py) and checks a stub-metric normal(), in
the style of test_fci_perpendicular_neumann_trace.py's constant-coefficient
oblique-metric check, rather than the real HSX geometry.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'scripts/p07n_field_derived_global'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


fields=module('p07n_field_derived_fields_test',PKG/'fields.py')
CONFIG=json.loads((PKG/'configuration.json').read_text())
PERIOD=1.7  # arbitrary eta physical period, unrelated to any winding number
RNG=np.random.default_rng(20260927)


def _random_points(n,u_range=(1e-3,1.0)):
    u=RNG.uniform(*u_range,n);th=RNG.uniform(-15,15,n);eta=RNG.uniform(-15,15,n)
    return np.column_stack([u,th,eta])


def _fd4(func,q,axis,h):
    """4th-order central first-derivative stencil, component `axis` of q."""
    e=np.zeros(3);e[axis]=h
    return (-func(q+2*e)+8*func(q+e)-8*func(q-e)+func(q-2*e))/(12*h)


def _assert_close(actual,expected,tol=1e-8):
    scale=np.maximum(np.abs(expected),1.0)
    np.testing.assert_array_less(np.abs(actual-expected),tol*scale+1e-12)


@pytest.mark.parametrize('name',fields.NAMES)
def test_gradient_matches_central_finite_difference(name):
    q=_random_points(40);h=2e-4
    v,g,H=fields.evaluate(None,q,name,PERIOD)
    def value(qq):return fields.evaluate(None,qq,name,PERIOD)[0]
    for axis in range(3):
        _assert_close(g[:,axis],_fd4(value,q,axis,h))


@pytest.mark.parametrize('name',fields.NAMES)
def test_hessian_matches_central_finite_difference_of_gradient(name):
    """Differentiate the analytic gradient (not the value) for a tight, independent Hessian check."""
    q=_random_points(40);h=2e-4
    v,g,H=fields.evaluate(None,q,name,PERIOD)
    def grad_component(qq,i):return fields.evaluate(None,qq,name,PERIOD)[1][:,i]
    for i in range(3):
        for j in range(3):
            _assert_close(H[:,i,j],_fd4(lambda qq,i=i:grad_component(qq,i),q,j,h))


@pytest.mark.parametrize('name',fields.NAMES)
def test_theta_periodicity(name):
    q=_random_points(20)
    v0,g0,H0=fields.evaluate(None,q,name,PERIOD)
    shifted=q.copy();shifted[:,1]+=2*np.pi
    v1,g1,H1=fields.evaluate(None,shifted,name,PERIOD)
    np.testing.assert_allclose(v1,v0,atol=1e-10)
    np.testing.assert_allclose(g1,g0,atol=1e-8)
    np.testing.assert_allclose(H1,H0,atol=1e-6)


@pytest.mark.parametrize('name',fields.NAMES)
def test_eta_periodicity(name):
    q=_random_points(20)
    v0,g0,H0=fields.evaluate(None,q,name,PERIOD)
    shifted=q.copy();shifted[:,2]+=PERIOD
    v1,g1,H1=fields.evaluate(None,shifted,name,PERIOD)
    np.testing.assert_allclose(v1,v0,atol=1e-10)
    np.testing.assert_allclose(g1,g0,atol=1e-8)
    np.testing.assert_allclose(H1,H0,atol=1e-6)


@pytest.mark.parametrize('name',fields.NAMES)
def test_axis_regularity_independent_of_theta(name):
    eta=RNG.uniform(-15,15,10);th_a=RNG.uniform(-10,10,10);th_b=RNG.uniform(-10,10,10)
    qa=np.column_stack([np.zeros(10),th_a,eta]);qb=np.column_stack([np.zeros(10),th_b,eta])
    va,_,_=fields.evaluate(None,qa,name,PERIOD);vb,_,_=fields.evaluate(None,qb,name,PERIOD)
    np.testing.assert_allclose(va,vb,atol=1e-12)


class _StubRef:
    """A metric with constant gcontra, so normal(ref,q) is known analytically."""
    def __init__(self,gcontra):self.gcontra=np.asarray(gcontra,dtype=float)
    def _metric(self,q):
        q=np.atleast_2d(q)
        return {'gcontra':np.broadcast_to(self.gcontra,(len(q),3,3))}


@pytest.mark.parametrize('name',fields.NAMES)
def test_normal_data_equals_a_dot_grad_f(name):
    gcontra=np.array([[2.0,.3,.1],[.3,1.5,.2],[.1,.2,1.2]])
    ref=_StubRef(gcontra)
    q=_random_points(15,u_range=(1.0,1.0))
    a=fields.normal(ref,q)
    expected_a=gcontra[0,:]/np.sqrt(gcontra[0,0])
    np.testing.assert_allclose(a,np.broadcast_to(expected_a,a.shape))
    g=fields.evaluate(ref,q,name,PERIOD)[1]
    expected=np.einsum('qa,qa->q',a,g)
    np.testing.assert_allclose(fields.normal_data(ref,q,name,PERIOD),expected,atol=1e-12)
    if name=='constant':
        np.testing.assert_allclose(expected,0,atol=1e-12)


def test_configuration_field_list_matches_module_names():
    assert CONFIG['fields']==list(fields.NAMES)


def test_configuration_gate_threshold_unchanged():
    assert CONFIG['global_l2_order_minimum']==1.8
