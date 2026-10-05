"""Analytic testbed of the nodal SBP perpendicular Laplacian: a shaped, eta-dependent disk times a periodic ``eta``.

    X = u cos t + d(eta) u^2 cos 2t,  Y = u sin t - d(eta) u^2 sin 2t,  Z = eta,  d(eta) = D0 (1 + 0.5 sin eta)
    b = (-IOTA Y, IOTA X, 1) / |.|   (Cartesian unit field; ``b^u != 0`` on the shaped logical surfaces)
    A = J (g^ij - b^i b^j)  (logical ``u, theta, eta``),   L f = J^-1 d_i (c A^ij d_j f)

Non-orthogonal (``g^{u theta}, g^{u eta}, g^{theta eta} != 0``), anisotropic and three-dimensional. Exact operators, conormal
fluxes and physical-normal derivatives come from JAX autodiff of the analytic definition. Copy of the geometry of the P09
Laplacian study (``work/p09_laplacian_20261004/tb_geom.py``) with self-contained smooth fields.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drbx.geometry.nodal_families import build_family_a_layout  # noqa: E402
from drbx.geometry.nodal_layout import node_points, wall_points  # noqa: E402
from drbx.geometry.sbp_laplacian import (build_laplacian_plan, laplacian_face_metric_from_callable,  # noqa: E402
                                         nodal_laplacian_metric_from_callable)

D0, IOTA = 0.12, 1.5


def xyz(q):
    u, t, e = q[0], q[1], q[2]
    d = D0 * (1 + 0.5 * jnp.sin(e))
    return jnp.stack([u * jnp.cos(t) + d * u * u * jnp.cos(2 * t), u * jnp.sin(t) - d * u * u * jnp.sin(2 * t), e])


def tensor(q):
    """``(A = J (g^ij - b^i b^j), J, g^ij)`` at the logical point ``q = (u, theta, eta)``."""
    E = jax.jacfwd(xyz)(q)
    J = jnp.linalg.det(E)
    Ei = jnp.linalg.inv(E)
    X = xyz(q)
    bc = jnp.stack([-IOTA * X[1], IOTA * X[0], 1.0])
    bc = bc / jnp.sqrt(bc @ bc)
    b = Ei @ bc
    G = Ei @ Ei.T
    return J * (G - jnp.outer(b, b)), J, G


# ----------------------------------------------------------------------------------------------- fields
def _wave(q, direction, wavelength, phase, part="re"):
    x, y = q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])
    s = (x * jnp.cos(direction) + y * jnp.sin(direction)) * 2 * jnp.pi / wavelength + q[2] + phase
    return jnp.cos(s) if part == "re" else jnp.sin(s)


def _switch(q, u_s=0.21, w2=0.06, phase=0.0):
    x, y = q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])
    s = x * x + y * y
    return jnp.exp(-(((s - u_s ** 2) / w2) ** 2)) * (x ** 4 - 6 * x * x * y * y + y ** 4) / u_s ** 4 * jnp.cos(q[2] + phase)


FIELDS = {
    "n": lambda q: 1.0 + 0.5 * _wave(q, 0.0, 2.0, 0.0),
    "Ti": lambda q: 1.0 + 0.5 * _wave(q, 2.0944, 4.0, 0.3, "im"),
    "omega": lambda q: _switch(q) + 0.5 * _wave(q, 0.5236, 2.0, 0.0),
}
FIELD_NAMES = tuple(FIELDS)


def coefficient(q):
    """A positive, state-like polarization coefficient ``c(u, theta, eta)`` (varies in all three directions)."""
    x, y = q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])
    return 1.0 + 0.5 * jnp.sin(1.5 * x + 0.7 * y + q[2])


def synthetic_geometry(pts):
    """Closed-form NumPy metric ``(A_log (Q, 3, 3), J (Q,), g^{u j} (Q, 3))`` with every entry coupled (parity data).

    ``g = D L L^T D`` with ``D = diag(1, 1/u, 1)`` and a smooth, anisotropic ``L(u, theta, eta)``; ``J = u (1 + 0.2 cos theta cos eta)``.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    u, t, e = pts.T
    L = np.zeros((len(u), 3, 3))
    L[:, 0, 0] = 1.0 + 0.06 * np.sin(t + e)
    L[:, 0, 1] = 0.12 * np.cos(2 * t)
    L[:, 0, 2] = 0.03 * u
    L[:, 1, 0] = 0.09 * u * np.cos(e)
    L[:, 1, 1] = 0.8
    L[:, 1, 2] = 0.15 * np.sin(t)
    L[:, 2, 0] = 0.06
    L[:, 2, 1] = 0.03 * np.cos(t - e)
    L[:, 2, 2] = 0.4 + 0.1 * np.cos(e)
    D = np.zeros((len(u), 3, 3))
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = 1.0, 1.0 / u, 1.0
    G = D @ L @ np.swapaxes(L, 1, 2) @ D
    J = u * (1.0 + 0.2 * np.cos(t) * np.cos(e))
    return J[:, None, None] * G, J, G[:, 0]


def _batched(fn, pts, block=2048):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    outs = []
    jfn = jax.jit(jax.vmap(fn))
    for s in range(0, len(pts), block):
        c = pts[s:s + block]
        pad = block - len(c)
        if pad:
            c = np.concatenate([c, np.repeat(c[-1:], pad, 0)])
        r = jfn(jnp.asarray(c))
        r = r if isinstance(r, tuple) else (r,)
        outs.append([np.asarray(x)[:block - pad] for x in r])
    res = [np.concatenate([o[i] for o in outs]) for i in range(len(outs[0]))]
    return res if len(res) > 1 else res[0]


def _geometry_fn(q):
    A, J, G = tensor(q)
    return A, J, G[0]


def _field_stack(q):
    return jnp.stack([f(q) for f in FIELDS.values()])


def _lap_fn(use_coeff):
    def lap(q):
        out = []
        for f in FIELDS.values():
            def flux(x, f=f):
                A, _J, _G = tensor(x)
                c = coefficient(x) if use_coeff else 1.0
                return c * (A @ jax.grad(f)(x))

            out.append(jnp.trace(jax.jacfwd(flux)(q)) / tensor(q)[1])
        return jnp.stack(out)

    return lap


def _wall_fn(q):
    """Conormal flux ``A^{u j} d_j f`` and physical-normal derivative ``g^{u j} d_j f / sqrt(g^{uu})`` per field."""
    A, _J, G = tensor(q)
    gr = jnp.stack([jax.grad(f)(q) for f in FIELDS.values()])             # (F, 3)
    return gr @ A[0], gr @ G[0] / jnp.sqrt(G[0, 0])


def values(pts):
    return _batched(_field_stack, pts)


def laplacians(pts, with_coeff=False):
    return _batched(_lap_fn(with_coeff), pts)


def coefficients(pts):
    return _batched(coefficient, pts)


def wall_data(pts):
    """``(conormal (Q, F), normal_derivative (Q, F))`` at the wall points."""
    return _batched(_wall_fn, pts)


# ----------------------------------------------------------------------------------------------- the case
class Case:
    """Family-A layout, Laplacian plan and exact data of the testbed at ``(n, n_eta)``."""

    def __init__(self, n: int, n_eta: int, with_exact: bool = True):
        self.layout = lay = build_family_a_layout(n, n_eta=n_eta)
        self._faces = {}
        self.plan = build_laplacian_plan(lay, nodal_laplacian_metric_from_callable(lay, lambda p: _batched(_geometry_fn, p)))
        self.pts = node_points(lay)
        E, P = lay.n_eta, lay.P
        self.E, self.P, self.N = E, P, lay.blocks[1].N
        self.H = np.asarray(self.plan.Hp) * lay.deta
        if with_exact:
            flat = self.pts.reshape(-1, 3)
            self.vals = values(flat).reshape(E, P, -1)
            self.lap = laplacians(flat).reshape(E, P, -1)
            self.lap_c = laplacians(flat, True).reshape(E, P, -1)
            self.coeff = coefficients(flat).reshape(E, P)
            wp = wall_points(lay, lay.walls[0])
            self.wall_pts = wp
            self.wall_val = values(wp.reshape(-1, 3)).reshape(E, self.N, -1)
            con, nor = wall_data(wp.reshape(-1, 3))
            self.wall_conormal = con.reshape(E, self.N, -1)
            self.wall_normal = nor.reshape(E, self.N, -1)

    def faces_plan(self, theta: bool = True, radial: bool = True):
        """Plan with the analytic ``A`` evaluated at the ``eta`` half planes (and the ``theta`` half nodes / radial faces)."""
        key = (theta, radial)
        if key not in self._faces:
            met = nodal_laplacian_metric_from_callable(self.layout, lambda p: _batched(_geometry_fn, p))
            fm = laplacian_face_metric_from_callable(self.layout, lambda p: _batched(_geometry_fn, p)[0], theta=theta, radial=radial)
            self._faces[key] = build_laplacian_plan(self.layout, met, faces=fm)
        return self._faces[key]

    def h_rel_error(self, err, ref, mask=None):
        m = np.ones_like(self.H, bool) if mask is None else mask
        return float(np.sqrt(np.sum((self.H * err ** 2)[m])) / np.sqrt(np.sum(self.H * ref ** 2)))


@lru_cache(maxsize=8)
def case(n: int, n_eta: int, with_exact: bool = True) -> Case:
    return Case(n, n_eta, with_exact)


# ----------------------------------------------------------------------------------------------- flat polynomial case
#: constant Cartesian tensor ``a_ij`` over ``(x, y, eta)`` (anisotropic, ``a_xy != 0``, ``a_x,eta != 0``) of the flat disk
A_CART = np.array([[1.3, 0.4, 0.3], [0.4, 0.8, -0.2], [0.3, -0.2, 0.6]])


def flat_geometry(pts):
    """``A = J Lam^-1 a Lam^-T`` of the flat disk (``J = u``) with the constant Cartesian tensor :data:`A_CART`; ``g^{u j}``."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    u, t = pts[:, 0], pts[:, 1]
    c, s = np.cos(t), np.sin(t)
    Li = np.zeros((len(u), 3, 3))
    Li[:, 0, 0], Li[:, 0, 1], Li[:, 1, 0], Li[:, 1, 1], Li[:, 2, 2] = c, s, -s / u, c / u, 1.0
    G = np.einsum("qij,jk,qlk->qil", Li, A_CART, Li)
    return u[:, None, None] * G, u, np.einsum("qij,qlj->qil", Li, Li)[:, 0]


def poly(coefs, x, y, dx=0, dy=0):
    """``d^dx_x d^dy_y sum c_pq x^p y^q`` for ``coefs = {(p, q): c}``."""
    out = 0.0 * x
    for (p, q), c in coefs.items():
        if p < dx or q < dy:
            continue
        kx = np.prod(np.arange(p, p - dx, -1.0)) if dx else 1.0
        ky = np.prod(np.arange(q, q - dy, -1.0)) if dy else 1.0
        out = out + c * kx * ky * x ** (p - dx) * y ** (q - dy)
    return out


def flat_plan(n: int, n_eta: int):
    layout = build_family_a_layout(n, n_eta=n_eta)
    return layout, build_laplacian_plan(layout, nodal_laplacian_metric_from_callable(layout, flat_geometry))
