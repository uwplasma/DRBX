"""The midpoint diffusion reference R of every field, in the sign/coefficient convention of the O_q3 reference.

``p_shared.perpendicular_reference_rhs.reference_rhs`` returns ``perpendicular_diffusion = D_f div(P grad f)`` built from the
exact face flux (the O_q3 construction, called O here).  :func:`diffusion_midpoint_reference` is the second continuum
reference of the same term, the generalization of ``p08_step5_combined.references.psi_midpoint_reference`` to every field of
the state: at the raw midpoints ``div(P grad f) = (div(J P) . grad f + J P : Hess f) / |J|`` with the AUTODIFF divergence of
``J P`` (``curvature_reference.perpendicular_geometry(env.ref, points, method="autodiff")``) and the exact gradient and Hessian
of ``f``, projected to owners by raw volume over owner volume, times ``D_f``.  (``psi_midpoint_reference`` returns the
positive operator ``-div(P grad psi)``; here the sign is ``+``, like ``reference_rhs``.)  ``env.ref`` carries the selected
B evaluator, so R is internally consistent with the operator of the same ``bfield_toroidal`` mode.
"""
from __future__ import annotations

import numpy as np

#: state slots of the five-column state ``(n, Te, Ti, omega, phi)`` of the four evolved fields
FIELD_SLOTS = {"density": 0, "Te": 1, "Ti": 2, "vorticity": 3}


def diffusion_midpoint_reference(env, support, state, diffusion: dict, fields=tuple(FIELD_SLOTS)) -> dict:
    """``{field: (n_owners,)}`` owner projection of ``D_f div(P grad f)`` at the raw midpoints of ``support``
    (a ``perpendicular_reference_rhs.OwnerSupport``); ``state`` is a ``P06NState``; ``diffusion`` maps field -> ``D_f``."""
    from p_shared import perpendicular_reference_rhs as prr
    from p_shared import replay_units as ru
    from p_shared.curvature_reference import perpendicular_geometry

    points = support.points
    _v, gradient, hessian = state.values_gradients_hessians(points)
    gradient, hessian = np.asarray(gradient), np.asarray(hessian)
    tensor, divergence = perpendicular_geometry(env.ref, points, method="autodiff")
    tensor, divergence = np.asarray(tensor), np.asarray(divergence)
    jac = np.asarray(support.raw_geometry(env)["J"], dtype=np.float64)
    slots = [FIELD_SLOTS[f] for f in fields]
    numerator = np.stack([np.einsum("qj,qj->q", divergence, gradient[s]) + np.einsum("qij,qij->q", tensor, hessian[s])
                          for s in slots], axis=1)
    pointwise = numerator / np.maximum(np.abs(jac), 1.0e-30)[:, None]                      # +div(P grad f)
    pair = ru._sparse_scatter(pointwise, support.raw_volume, support.raw_owner)
    owner = prr._gather(pair, support.owners) / support.owner_volume[:, None]
    return {f: float(diffusion.get(f, 1.0)) * owner[:, i] for i, f in enumerate(fields)}
