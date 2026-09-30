"""Frozen P07 tensor-contracted, q3 integrated face functionals.

The caller supplies canonical face keys, quadrature points, and the already
weighted normal tensor component.  This host builder never queries an MMS
field or a campaign global.  In particular the coupled fit tests the *whole
integrated functional*, rather than a stronger set of point-gradient targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import _fci_perpendicular_point_primitives as r
from .fci_perpendicular_reconstruction import PointRowContext, StructuredReconstruction

_EXP3 = np.array([(a, degree-a) for degree in range(4) for a in range(degree+1)], dtype=int)


@dataclass(frozen=True)
class IntegratedFaceRow:
    donor_ids: np.ndarray
    weights: np.ndarray
    boundary_conditioned: bool
    trace_donor_points: np.ndarray
    trace_target_points: np.ndarray
    value_loading: np.ndarray
    tangential_loading: np.ndarray
    family: int


def contract_face_tensor(weights, tensor, face_axis):
    """Freeze P07's normal-row convention: ``w_q * T[q, axis, d]``."""
    weights=np.asarray(weights,dtype=np.float64)
    tensor=np.asarray(tensor,dtype=np.float64)
    axis=np.asarray(face_axis,dtype=np.int64)
    if weights.ndim!=2 or tensor.shape!=weights.shape+(3,3) or axis.shape!=(len(weights),):
        raise ValueError('P07 tensor/face quadrature shape mismatch')
    if np.any((axis<0)|(axis>2)):
        raise ValueError('face axis must be 0, 1, or 2')
    return weights[:,:,None]*tensor[np.arange(len(weights))[:,None],
                                     np.arange(weights.shape[1])[None,:],axis[:,None],:]


def _radial(nodes, targets):
    value = []
    derivative = []
    for j, x in enumerate(nodes):
        other = np.delete(nodes, j)
        denominator = np.prod(x-other)
        value.append(np.prod(targets[:, None]-other, axis=1)/denominator)
        derivative.append(sum(np.prod(targets[:, None]-np.delete(other, k), axis=1)
                              for k in range(3))/denominator)
    return np.array(value).T, np.array(derivative).T


def _compact(ids, coefficients):
    ids = np.asarray(ids, dtype=np.int64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if not len(ids):
        return ids, coefficients
    unique, inverse = np.unique(ids, return_inverse=True)
    summed = np.bincount(inverse, weights=coefficients, minlength=len(unique))
    keep = np.abs(summed) > 0
    return unique[keep], summed[keep]


def _singleton(t, key, points, integ):
    n = t.n; axis, i = map(int, key[:2])
    anchor = i if axis == 0 else i+.5
    layers = np.arange(i-2, i+2) if axis == 0 else np.arange(i-1, i+3)
    L, D = _radial(layers+.5-anchor, points[:, 0]*n-anchor)
    D *= n
    radial = np.where(layers < 0, -layers-1, layers)
    ids = []; coefficients = []
    for q, point in enumerate(points):
        ti = r.nearest(t.centers[1], point[1], 7, 2*np.pi)
        tv, td = r.theta_rows(t.centers[1][ti], point[1])
        ei = r.nearest(t.centers[2], point[2], 4, t.g.eta_period)
        ev, ed = r.eta_rows(t.centers[2][ei], point[2], t.g.eta_period, t.g.deta)
        theta = (ti[None, :]+np.where(layers < 0, n//2, 0)[:, None]) % n
        raw = (radial[:, None, None]*n+theta[:, None, :])*n+ei[None, :, None]
        owner = t.ro[raw]
        if not np.all(t.starts[owner+1]-t.starts[owner] == 1):
            raise ValueError("ordinary integrated donor is aggregated")
        weight = (integ[q, 0]*D[q, :, None, None]*ev[None, :, None]*tv[None, None, :]
                  +integ[q, 1]*L[q, :, None, None]*ev[None, :, None]*td[None, None, :]
                  +integ[q, 2]*L[q, :, None, None]*ed[None, :, None]*tv[None, None, :])
        ids.extend(owner.ravel()); coefficients.extend(weight.ravel())
    return _compact(ids, coefficients)


def _ringwise(t, service, key, points, integ):
    n = t.n; axis, i = map(int, key[:2]); anchor = i if axis == 0 else i+.5
    layers = np.arange(i-2, i+2) if axis == 0 else np.arange(i-1, i+3)
    L, D = _radial(layers+.5-anchor, points[:, 0]*n-anchor); D *= n
    radial = np.where(layers < 0, -layers-1, layers)
    ids = []; coefficients = []
    for q, point in enumerate(points):
        ei = r.nearest(t.centers[2], point[2], 4, t.g.eta_period)
        ev, ed = r.eta_rows(t.centers[2][ei], point[2], t.g.eta_period, t.g.deta)
        for l, rr in enumerate(radial):
            theta = float((point[1]+(np.pi if layers[l] < 0 else 0)) % (2*np.pi))
            for e, kk in enumerate(ei):
                owners, angles = service.ring(rr, kk)
                pick = r.nearest(angles, theta, 7, 2*np.pi)
                donor = owners[pick]; nodes = angles[pick]
                tag = ('integrated-ring', int(rr), int(kk), *map(int, donor))
                if tag not in service.fits:
                    A = np.array([(t.rv[t.members(o)]/t.vol[o]) @ r.cardinal(nodes, t.pts[t.members(o), 1])[0] for o in donor])
                    U = np.array([r.cardinal(nodes, t.pts[t.members(o), 1])[0].mean(axis=0) for o in donor])
                    C, rank, _ = r.pinv(A, np.ones(7)); CU, urank, _ = r.pinv(U, np.ones(7))
                    service.fits[tag] = A, U, C, CU, rank, urank
                A, U, C, CU, _, _ = service.fits[tag]
                v, d = r.cardinal(nodes, [theta])
                target = np.vstack((v, d))
                if max(r.resid(target, C, A), r.resid(target, CU, U)) > 1e-9:
                    raise RuntimeError("unsupported integrated ringwise target")
                value = v[0] @ C; derivative = d[0] @ C
                weight = (integ[q, 0]*D[q, l]*ev[e]*value
                          +integ[q, 1]*L[q, l]*ev[e]*derivative
                          +integ[q, 2]*L[q, l]*ed[e]*value)
                ids.extend(donor); coefficients.extend(weight)
    return _compact(ids, coefficients)


def _coupled(t, service, key, points, integ):
    n = t.n; axis, i = map(int, key[:2])
    layers = np.arange(i-2, i+2) if axis == 0 else np.arange(i-1, i+3)
    radial = np.where(layers < 0, -layers-1, layers)
    center = np.array((points[4, 0]*np.cos(points[4, 1]), points[4, 0]*np.sin(points[4, 1])))
    scale = max(t.g.dr, points[4, 0]*t.g.dtheta)
    ei, ev, ed = r.eta_plane_rows(t, points)
    ids = []; coefficients = []

    def donors(lo, hi, plane):
        result = []
        for rr in range(lo, hi+1):
            owners, angles = service.ring(rr, plane)
            result.extend(owners[r.nearest(angles, points[4, 1], min(7, len(owners)), 2*np.pi)])
        return np.asarray(result, dtype=np.int64)

    for plane in np.unique(ei):
        target3 = _target_for_plane(points, integ, ei, ev, ed, plane, center, scale, _EXP3)
        target4 = _target_for_plane(points, integ, ei, ev, ed, plane, center, scale, r.EXP4)
        for level in range(4):
            lo = max(0, int(radial.min())-level); hi = min(n-1, int(radial.max())+level)
            donor = donors(lo, hi, plane)
            A, U, root, _, _, _, _, _, _ = r.fit(t, donor, center, scale, target4)
            C3, _, _ = r.pinv(A[:, :10], root); CU3, _, _ = r.pinv(U[:, :10], root)
            res3 = max(r.resid(target3, C3, A[:, :10]), r.resid(target3, CU3, U[:, :10]))
            if res3 <= 1e-9:
                break
        else:
            raise RuntimeError(("unsupported integrated cubic", key, plane))
        for extra in range(3):
            donor = donors(max(0, lo-extra), min(n-1, hi+extra), plane)
            A, U, _, C4, _, _, _, _, res4 = r.fit(t, donor, center, scale, target4)
            if res4 <= 1e-9:
                break
        else:
            raise RuntimeError(("unsupported integrated quartic", key, plane))
        ids.extend(donor); coefficients.extend(target4 @ C4)
    return _compact(ids, coefficients)


def _target_for_plane(points, integ, ei, ev, ed, plane, center, scale, exp):
    B, Dr, Dt = r.planar(points, center, scale, exp)
    value = np.sum(ev*(ei == plane), axis=1)
    derivative = np.sum(ed*(ei == plane), axis=1)
    return np.sum(integ[:, 0, None]*Dr*value[:, None]
                  +integ[:, 1, None]*Dt*value[:, None]
                  +integ[:, 2, None]*B*derivative[:, None], axis=0)


def prepare_integrated_face_rows(context: PointRowContext, keys, family, points, integrand, *,
                                 inner_support="profile7"):
    """Prepare one P07 row per canonical face with explicit q3 geometry arrays.

    ``inner_support`` (``"profile7"`` default, or ``"last_aggregate"``) is the inner donor-support rule of
    :class:`StructuredReconstruction`. At ``"last_aggregate"`` a singleton or ringwise face (family 5, 6) whose
    anchor ring is at most the last agglomerated ring is built as the coupled quartic (family code 7, the code
    it reports); all other faces are unchanged.

    ``integrand`` has shape ``(faces, 9, 3)`` and is the face quadrature
    weight times the geometry tensor's selected normal row.  Family codes are
    the frozen P07 topology census codes 0..7.
    """
    keys = np.asarray(keys, dtype=np.int64)
    family = np.asarray(family, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    integrand = np.asarray(integrand, dtype=np.float64)
    if keys.shape != (len(family), 4) or points.shape != (len(family), 9, 3) or integrand.shape != points.shape:
        raise ValueError("P07 face keys, q3 points, and weighted integrands have incompatible shapes")
    service = StructuredReconstruction(context, inner_support=inner_support)
    top = service.top
    result = []
    for key, code, p, integ in zip(keys, family, points, integrand, strict=True):
        code = int(code)
        if (code in (5, 6) and service.inner_support == "fixed_radius"
                and service.below_fixed_switch(key[0], key[1])):
            code = 7
        if (code in (5, 6) and service.inner_support == "any_aggregate"
                and service.stencil_min_profile(key[0], key[1]) < service.t.n):
            code = 7
        if (code in (5, 6) and service.inner_support == "last_aggregate"
                and service.anchor_ring(key[0], key[1]) <= service.last):
            code = 7
        if code == 0:
            donor = np.empty(0, np.int64); weight = np.empty(0); conditioned = False
            donor_points = np.empty((0, 3)); target_points = p.copy()
            value_loading = np.empty(0); tangential_loading = np.empty((0, 2))
        elif code in (1, 2, 3, 4):
            z = r.boundary_map(context.n, 'full', key, -1, p, np.ones(9), integ, top, context.g.eta_period)
            donor = z['donor_ids']; weight = z['W']; conditioned = bool(z['boundary_conditioned'])
            donor_points = z['trace_donor_points']; target_points = z['trace_target_points']
            value_loading = z['value_loading']; tangential_loading = z['tangential_loading']
        else:
            builder = {5: _singleton, 6: lambda t,k,p,i: _ringwise(t,service,k,p,i),
                       7: lambda t,k,p,i: _coupled(t,service,k,p,i)}.get(code)
            if builder is None:
                raise ValueError(f"unsupported P07 family {code}")
            donor, weight = builder(context, key, p, integ)
            conditioned = False; donor_points = np.empty((0, 3)); target_points = p.copy()
            value_loading = np.empty(0); tangential_loading = np.empty((0, 2))
        result.append(IntegratedFaceRow(donor, weight, conditioned, donor_points,
                                        target_points, value_loading, tangential_loading, code))
    return tuple(result)
