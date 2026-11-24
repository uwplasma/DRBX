"""narrow face-flux Laplacian nodal and Laplacian extraction kernels and Laplacian file I/O."""

from __future__ import annotations


import numpy as np

from ...operators.nodal.nodal_plan import NodalMetric

from .nodal_metric import metric_from_geometry_provider


def _nodal_values(provider, points, chunk):
    adapter = metric_from_geometry_provider(provider)
    flat = points.reshape(-1, 3)
    hs, js, Bs, Ks = [], [], [], []
    for s in range(0, len(flat), chunk):
        h, jac, B, Kc = adapter.nodal_metric(flat[s:s + chunk])
        hs.append(h); js.append(jac); Bs.append(B); Ks.append(Kc)
    E, P = points.shape[:2]
    return NodalMetric(np.concatenate(hs).reshape(E, P, 3), np.concatenate(js).reshape(E, P),
                       np.concatenate(Bs).reshape(E, P), np.concatenate(Ks).reshape(E, P, 3))


def _geometry_fn(perp, mev, chunk: int):
    """``points (Q, 3) -> (A signed to |J| P_perp, divA, signed J, g^{u j})``, evaluated in chunks."""

    def geometry(q):
        q = np.asarray(q, dtype=np.float64)
        A, d, J, gu = [], [], [], []
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            T, D = perp(qq)
            m = mev.evaluate(qq, reject_nonpositive_J=False)
            Js = np.asarray(m.signed_J, dtype=np.float64)
            sg = np.sign(Js)
            A.append(T * sg[:, None, None]); d.append(D * sg[:, None]); J.append(Js)
            gu.append(np.asarray(m.contravariant_metric, dtype=np.float64)[:, 0, :])
        return np.concatenate(A), np.concatenate(d), np.concatenate(J), np.concatenate(gu)

    return geometry


def _laplacian_values(reference, points, wp, chunk):
    E, P = points.shape[:2]
    geometry = _geometry_fn(reference.autodiff_perpendicular(), reference.metric_evaluator, chunk)
    A, divA, Js, gu = geometry(points.reshape(-1, 3))
    Js = Js.reshape(E, P)
    if not (np.all(Js > 0) or np.all(Js < 0)):
        raise ValueError("the jacobian changes sign over the nodes")
    sign = float(np.sign(Js.flat[0]))
    Aw, dw, Jw, gw = geometry(wp.reshape(-1, 3))
    Nw = wp.shape[1]
    return dict(points=points, A=A.reshape(E, P, 3, 3), divA=divA.reshape(E, P, 3),
                J=np.abs(Js), ginv_u=gu.reshape(E, P, 3), sign=np.array(sign),
                wall_points=wp, wall_A=Aw.reshape(E, Nw, 3, 3),
                wall_divA=dw.reshape(E, Nw, 3), wall_J=np.abs(Jw).reshape(E, Nw),
                wall_ginv_u=gw.reshape(E, Nw, 3))


