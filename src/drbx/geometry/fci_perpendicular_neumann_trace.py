"""Host-only structured physical-normal Neumann wall-trace elimination.

The tensor patch is selected from the existing boundary cardinal support.  Its
interior samples are complete singleton owner observations; the wall trace is
eliminated once using the physical contravariant normal at 28 wall nodes.
No manufactured wall values enter these rows.
"""
from dataclasses import dataclass
import numpy as np

from . import _fci_perpendicular_point_primitives as r
from .fci_perpendicular_reconstruction import PointRowContext


@dataclass(frozen=True)
class NeumannPointRows:
    donor_ids: np.ndarray
    value: np.ndarray
    gradient: np.ndarray
    boundary_points: np.ndarray
    boundary_value: np.ndarray
    boundary_gradient: np.ndarray
    condition: float
    constraint_residual: float


def prepare_neumann_point_rows(context: PointRowContext, points, *,
                               normal_coefficients, radial_degree=4,
                               max_condition=1.0e8, support_anchors=None,
                               patch_cache=None):
    """Prepare independent fixed-support wall patches for requested points.

    ``normal_coefficients`` maps wall node coordinates ``(28,3)`` to
    ``g_contra[...,0,:]/sqrt(g_contra[...,0,0])`` in physical length units.
    Degree four uses the qualified quartic radial wall basis; degree three
    uses the qualified boundary-transverse basis.  The returned boundary
    loading multiplies physical-normal samples at ``boundary_points``.  A
    supplied ``patch_cache`` is scoped to one fixed context and normal callback.
    """
    if radial_degree not in (3, 4):
        raise ValueError("radial_degree must be 3 or 4")
    t = context
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (targets, 3)")
    anchors = p[:, 1:] if support_anchors is None else np.asarray(support_anchors, dtype=np.float64)
    if anchors.shape != (len(p), 2):
        raise ValueError("support_anchors must have shape (targets, 2)")
    h = 1.0 / t.n
    radial_nodes = np.r_[0., -np.arange(radial_degree, dtype=float)-.5]
    layers = np.arange(t.n-1, t.n-1-radial_degree, -1)
    result = []
    if patch_cache is None:
        patch_cache = {}
    for target, anchor in zip(p, anchors, strict=True):
        ti = r.nearest(t.centers[1], anchor[0], 7, 2*np.pi)
        ei = r.nearest(t.centers[2], anchor[1], 4, t.g.eta_period)
        patch_key = (id(t), id(normal_coefficients), radial_degree,
                     tuple(map(int, ti)), tuple(map(int, ei)))
        prepared = patch_cache.get(patch_key)
        if prepared is None:
            theta = t.centers[1][ti]
            eta = t.centers[2][ei]
            wall = np.column_stack((np.ones(28),
                                    np.repeat(theta, 4), np.tile(eta, 7)))
            a = np.asarray(normal_coefficients(wall), dtype=np.float64)
            if a.shape != (28, 3) or not np.all(np.isfinite(a)) or np.any(a[:, 0] <= 0):
                raise ValueError("invalid outward physical normal coefficients")
            _, d0 = r.rows(radial_nodes, [0.])
            theta_derivative = np.array([r.theta_rows(theta, x)[1] for x in theta])
            eta_derivative = np.array([r.eta_rows(eta, x, t.g.eta_period, t.g.deta)[1] for x in eta])
            m = np.diag(a[:, 0] * d0[0, 0] / h)
            m += a[:, 1, None] * np.kron(theta_derivative, np.eye(4))
            m += a[:, 2, None] * np.kron(np.eye(7), eta_derivative)
            singular = np.linalg.svd(m, compute_uv=False)
            condition = float(singular[0] / singular[-1]) if singular[-1] else np.inf
            if not np.isfinite(condition) or condition > max_condition:
                raise ValueError(f"Neumann wall trace system condition {condition:.6g} exceeds {max_condition:.6g}")
            radial_constraint = np.hstack([np.diag(a[:, 0]*derivative/h)
                                           for derivative in d0[0, 1:]])
            response = np.linalg.solve(m, -radial_constraint)
            inverse = np.linalg.solve(m, np.eye(28))
            residual = max(float(np.linalg.norm(m@response+radial_constraint, ord=np.inf)),
                           float(np.linalg.norm(m@inverse-np.eye(28), ord=np.inf)))
            raw = ((layers[:, None, None]*t.n+ti[None, :, None])*t.n+ei[None, None, :]).reshape(-1)
            donor = t.ro[raw]
            if np.any(t.starts[donor+1]-t.starts[donor] != 1):
                raise ValueError("Neumann wall patch contains aggregated donor")
            ids, inverse_ids = np.unique(donor, return_inverse=True)
            prepared = (theta, eta, wall, d0, response, inverse, condition,
                        residual, ids, inverse_ids)
            patch_cache[patch_key] = prepared
        theta, eta, wall, d0, response, inverse, condition, residual, ids, inverse_ids = prepared
        if condition > max_condition:
            raise ValueError(f"Neumann wall trace system condition {condition:.6g} exceeds {max_condition:.6g}")
        tv, td = r.theta_rows(theta, target[1])
        ev, ed = r.eta_rows(eta, target[2], t.g.eta_period, t.g.deta)
        l, d = r.rows(radial_nodes, [(target[0]-1.)/h])
        angular = np.kron(tv, ev)
        angular_theta = np.kron(td, ev)
        angular_eta = np.kron(tv, ed)
        wall_target = np.stack((l[0, 0]*angular, d[0, 0]/h*angular,
                                l[0, 0]*angular_theta, l[0, 0]*angular_eta))
        interior_target = np.stack((np.kron(l[0, 1:], angular),
                                    np.kron(d[0, 1:]/h, angular),
                                    np.kron(l[0, 1:], angular_theta),
                                    np.kron(l[0, 1:], angular_eta)))
        owner_map = interior_target + wall_target@response
        boundary_map = wall_target@inverse
        compact = np.zeros((4, len(ids)))
        for row in range(4):
            np.add.at(compact[row], inverse_ids, owner_map[row])
        result.append(NeumannPointRows(ids, compact[0], compact[1:], wall,
                                        boundary_map[0], boundary_map[1:],
                                        condition, residual))
    return tuple(result)
