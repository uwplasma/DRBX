"""Literal legacy-versus-shared host arithmetic replay, including P/Q rows."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from drbx.geometry import _reconstruction_primitives as shared
from drbx.geometry import _fci_perpendicular_point_primitives as p
from drbx.stencils import q_parallel_primitives as q

# Literal pre-extraction P implementation. Keep this independent of shared code:
# importing the compatibility modules twice would not establish legacy replay.
_LEGACY_SOURCE = r'''
"""Frozen host algebra for the qualified perpendicular structured point rows.

Extracted without campaign imports or module-global input paths. Numerical policy
comes from scripts/p07_combined_global/kernels.py at the accepted P05 revision.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

EXP4 = np.array([(a, d-a) for d in range(5) for a in range(d+1)], int)

def wrap(x,period):return x-period*np.floor(x/period+.5)


def nearest(nodes,target,count,period):
 distance=np.round(abs(wrap(nodes-target,period))/(period/len(nodes)),12)
 return np.lexsort((np.arange(len(nodes)),distance))[:count]


# Vectorized 28 September 2026: same elementwise operations and reduction order as the frozen
# formulas (bitwise-identical outputs, verified against the loop versions); only Python loops removed.
def _leave_one_out(m):
    others = np.array([[k for k in range(m) if k != j] for j in range(m)], dtype=np.intp)
    drop = np.array([[i for i in range(m - 1) if i != k] for k in range(m - 1)], dtype=np.intp)
    return others, drop


def _loo_products(f, drop):
    # f[..., k] over the m-1 "others"; returns P[..., k] = prod of f without entry k, in original order.
    return np.prod(f[..., drop], axis=-1)


def theta_rows(nodes, target):
    nn = np.asarray(nodes, dtype=np.longdouble); t = np.longdouble(target); m = len(nn)
    others_idx, drop = _leave_one_out(m)
    others = nn[others_idx]
    den = np.sin((nn[:, None] - others) / 2); f = np.sin((t - others) / 2) / den; df = .5 * np.cos((t - others) / 2) / den
    v = np.prod(f, axis=1); P = _loo_products(f, drop)
    d = 0
    for k in range(m - 1): d = d + df[:, k] * P[:, k]
    hit = np.flatnonzero(abs(wrap(target - nodes, 2 * np.pi)) < 32 * np.finfo(float).eps * 2 * np.pi)
    if len(hit): v = np.eye(m)[hit[0]]
    return np.array(v, float), np.array(d, float)


def eta_rows(nodes, target, period, step):
    x = np.asarray(wrap(nodes - target, period) / step, dtype=np.longdouble); m = len(x)
    others_idx, drop = _leave_one_out(m)
    xo = x[others_idx]
    den = x[:, None] - xo; f = -xo / den; df = 1 / den
    v = np.prod(f, axis=1); P = _loo_products(f, drop)
    d = 0
    for k in range(m - 1): d = d + df[:, k] * P[:, k]
    d = d / step
    hit = np.flatnonzero(abs(x) * step < 32 * np.finfo(float).eps * period)
    if len(hit): v = np.eye(len(x))[hit[0]]
    return np.array(v, float), np.array(d, float)


def members(t,o):return t.order[t.starts[o]:t.starts[o+1]]


def cardinal(nodes, targets):
    """Vectorized literal replay of frozen trigonometric cardinal formula."""
    nodes = np.asarray(nodes, np.longdouble); targets = np.asarray(targets, np.longdouble)
    others_idx, drop = _leave_one_out(7)
    others = nodes[others_idx]                                   # (7, 6)
    den = np.sin((nodes[:, None] - others) / 2)                  # (7, 6)
    arg = (targets[:, None, None] - others[None]) / 2            # (Q, 7, 6)
    f = np.sin(arg) / den; df = .5 * np.cos(arg) / den
    v = np.prod(f, axis=2)
    P = _loo_products(f, drop)
    d = 0
    for k in range(6): d = d + df[:, :, k] * P[:, :, k]
    hit = np.abs(wrap(targets[:, None] - nodes[None], 2 * np.pi)) < 32 * np.finfo(float).eps * 2 * np.pi
    for i in np.flatnonzero(hit.any(axis=1)): v[i] = hit[i].astype(float)
    return np.asarray(v, float), np.asarray(d, float)


def pinv(A,root):
    u,s,vt=np.linalg.svd(root[:,None]*A,full_matrices=False);keep=s>=1e-4*s[0]
    C=(vt[keep].T/s[keep])@u[:,keep].T*root[None]
    return C,int(sum(keep)),float(s[0]/s[keep][-1])


def resid(target,C,A):
    return float(np.linalg.norm(target@C@A-target)/max(np.linalg.norm(target),1e-300))


def basis(xy,center,scale,exp):return np.prod(((xy-center)/scale)[:,None,:]**exp[None],axis=2)


def planar(points,center,scale,exp):
 xy=np.column_stack((points[:,0]*np.cos(points[:,1]),points[:,0]*np.sin(points[:,1])));z=(xy-center)/scale
 B=np.prod(z[:,None,:]**exp[None],axis=2);der=np.zeros((len(points),len(exp),2))
 for axis in (0,1):
  mask=exp[:,axis]>0;pow=exp[mask].copy();pow[:,axis]-=1
  der[:,mask,axis]=np.prod(z[:,None,:]**pow[None],axis=2)*exp[mask,axis]/scale
 dx,dy=der[:,:,0],der[:,:,1]
 Dr=np.cos(points[:,1,None])*dx+np.sin(points[:,1,None])*dy
 Dt=points[:,0,None]*(-np.sin(points[:,1,None])*dx+np.cos(points[:,1,None])*dy)
 return B,Dr,Dt


def fit(t,don,center,scale,target):
 A=[];U=[]
 for oid in don:
  mm=members(t,int(oid));B=basis(t.xy[mm],center,scale,EXP4)
  A.append((t.rv[mm]/t.vol[oid])@B);U.append(B.mean(axis=0))
 A=np.asarray(A);U=np.asarray(U)
 root=1/(1+np.linalg.norm((t.g.owner_centroid_xy[don]-center)/scale,axis=1)**2)
 C,rank,cond=pinv(A,root);CU,urank,ucond=pinv(U,root)
 return A,U,root,C,rank,urank,cond,ucond,max(residual(target,C,A),residual(target,CU,U))


def eta_plane_rows(t,p):
 ei=[];ev=[];ed=[]
 for et in p[:,2]:
  ids=nearest(t.centers[2],et,4,t.g.eta_period);v,d=eta_rows(t.centers[2][ids],et,t.g.eta_period,t.g.deta)
  ei.append(ids);ev.append(v);ed.append(d)
 return np.asarray(ei),np.asarray(ev),np.asarray(ed)


def rows(nodes, targets):
    x = np.asarray(nodes, float); s = np.asarray(targets, float); m = len(x)
    others_idx, drop = _leave_one_out(m)
    other = x[others_idx]                                        # (m, m-1)
    den = np.prod(x[:, None] - other, axis=1)                    # (m,)
    diff = s[:, None, None] - other[None]                        # (Q, m, m-1)
    v = np.prod(diff, axis=2) / den
    P = _loo_products(diff, drop)
    d = 0
    for j in range(m - 1): d = d + P[:, :, j]
    return v, d / den



residual = resid
'''

legacy = SimpleNamespace()
exec(_LEGACY_SOURCE, legacy.__dict__)


def _equal(actual, expected):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _equal(a, b)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)
        if isinstance(actual, np.ndarray):
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()


def test_compatibility_exports_and_literal_legacy_arithmetic():
    names = ('wrap', 'nearest', '_leave_one_out', '_loo_products', 'theta_rows',
             'eta_rows', 'members', 'cardinal', 'pinv', 'resid', 'basis',
             'planar', 'fit', 'eta_plane_rows', 'rows')
    for name in names:
        assert getattr(p, name) is getattr(q, name) is getattr(shared, name)
    assert p.residual is q.residual is shared.resid is shared.residual
    assert p.EXP4 is q.EXP4 is shared.EXP4
    _equal(shared.EXP4, legacy.EXP4)
    period = 2*np.pi
    nodes = (np.arange(16)+.5)*period/16
    targets = (-period-1e-9, 0., nodes[0], period-.01, period+nodes[0])
    for target in targets:
        for count in (4, 5, 7):
            ids = legacy.nearest(nodes, target, count, period)
            _equal(shared.nearest(nodes, target, count, period), ids)
            _equal(shared.eta_rows(nodes[ids], target, period, period/16),
                   legacy.eta_rows(nodes[ids], target, period, period/16))
        ids = legacy.nearest(nodes, target, 7, period)
        _equal(shared.theta_rows(nodes[ids], target), legacy.theta_rows(nodes[ids], target))
        _equal(shared.cardinal(nodes[ids], np.array(targets)), legacy.cardinal(nodes[ids], np.array(targets)))
    _equal(shared.nearest(nodes, period, 7, period), legacy.nearest(nodes, period, 7, period))
    _equal(shared.wrap(np.array(targets), period), legacy.wrap(np.array(targets), period))
    for m in (4, 5, 7):
        indices, drop = shared._leave_one_out(m)
        _equal((indices, drop), legacy._leave_one_out(m))
        values = np.arange(2*m*(m-1), dtype=float).reshape(2, m, m-1)/17
        _equal(shared._loo_products(values, drop), legacy._loo_products(values, drop))
    for radial in ([1.5, .5, -.5, -1.5], [0., -.5, -1.5, -2.5, -3.5]):
        target = np.array([radial[0], -.37, -2.8, .001])
        _equal(shared.rows(radial, target), legacy.rows(radial, target))
    matrix = np.diag([1., 1e-4, np.nextafter(1e-4, 0)])
    _equal(shared.pinv(matrix, np.ones(3)), legacy.pinv(matrix, np.ones(3)))
    assert shared.pinv(matrix, np.ones(3))[1] == 2
    _equal(shared.resid(np.eye(3), np.eye(3), matrix), legacy.residual(np.eye(3), np.eye(3), matrix))


def test_complete_owner_fit_and_four_plane_helper():
    from tests.test_fci_perpendicular_point_rows import _toy_context
    t = _toy_context()
    ro = np.arange(t.n**3)//2
    rv = 1.+np.arange(t.n**3)%3
    vol = np.bincount(ro, weights=rv)
    centroid = np.column_stack([np.bincount(ro, weights=rv*t.xy[:,a])/vol for a in (0, 1)])
    topo = SimpleNamespace(order=np.argsort(ro, kind='stable'),
        starts=np.r_[0, np.cumsum(np.bincount(ro))], xy=t.xy, rv=rv, vol=vol,
        centers=t.centers, g=SimpleNamespace(owner_centroid_xy=centroid,
        eta_period=t.g.eta_period, deta=t.g.deta))
    center = np.array([.2, .1]); scale = .3; donors = np.arange(64)
    points = t.pts[[0, 127, 300]].copy(); points[:,2] += 2*np.pi
    target = legacy.planar(points, center, scale, legacy.EXP4)[0]
    _equal(shared.basis(t.xy, center, scale, shared.EXP4), legacy.basis(t.xy, center, scale, legacy.EXP4))
    _equal(shared.planar(points, center, scale, shared.EXP4), legacy.planar(points, center, scale, legacy.EXP4))
    _equal(shared.members(topo, 1), legacy.members(topo, 1))
    _equal(shared.fit(topo, donors, center, scale, target), legacy.fit(topo, donors, center, scale, target))
    _equal(shared.eta_plane_rows(topo, points), legacy.eta_plane_rows(topo, points))


@pytest.mark.parametrize('key', [(0,0,7), (4,0,7), (6,0,7), (7,0,0)])
def test_p_rows_legacy_replay_including_wall_and_eta_wrap(monkeypatch, key):
    from tests.test_fci_perpendicular_point_rows import _toy_context
    from drbx.geometry import fci_perpendicular_reconstruction as builder
    t = _toy_context()
    point = t.pts[np.ravel_multi_index(key, (8,8,8))].copy()
    shifted = point.copy(); shifted[1] += 2*np.pi; shifted[2] -= 2*np.pi
    points = np.stack((point, shifted))
    actual = builder.StructuredReconstruction(t).rows(key, points, location='cell')
    with monkeypatch.context() as mp:
        for name in vars(legacy):
            if callable(getattr(legacy, name)) and hasattr(p, name):
                mp.setattr(p, name, getattr(legacy, name))
        expected = builder.StructuredReconstruction(t).rows(key, points, location='cell')
    for field in fields(actual):
        _equal(getattr(actual, field.name), getattr(expected, field.name))


from tests.test_q_parallel_hsx_portable import patch


def test_q_actual_hsx_rows_legacy_replay_both_spans(patch, monkeypatch):
    import json
    from drbx.stencils import q_parallel_support, q_parallel_wall
    from drbx.stencils.q_parallel import prepare_chunk, topology_from_arrays
    arrays, prepared, _, _ = patch
    meta = json.loads(str(arrays['metadata']))
    topology = topology_from_arrays(tuple(arrays[f'centers_{i}'] for i in range(3)),
                                    arrays['active'], arrays['aggregate'], arrays['volume'])
    def geom(points):
        for i in range(meta['geom_calls']):
            if np.array_equal(points, arrays[f'geom_points_{i}']):
                return tuple(arrays[f'geom_{k}_{i}'] for k in ('J','b','B'))
        raise AssertionError('geometry query changed')
    def jacobian(points):
        for i in range(meta['jac_calls']):
            if np.array_equal(points, arrays[f'jac_points_{i}']):
                return arrays[f'jac_J_{i}']
        raise AssertionError('wall query changed')
    monkeypatch.setattr(q_parallel_support, 'r', legacy)
    monkeypatch.setattr(q_parallel_wall, 'r', legacy)
    for span, actual in zip((1/16, 1/32), prepared, strict=True):
        expected = prepare_chunk(topology, arrays['owners'], arrays['ends'], geom, jacobian,
            span=span, source_identity='fixture-traces', geometry_identity='fixture-HSX', raw=arrays['raw'])
        for field in fields(actual):
            if field.name != 'metadata':
                _equal(getattr(actual, field.name), getattr(expected, field.name))
        _equal(actual.metadata['row_diagnostics'], expected.metadata['row_diagnostics'])
