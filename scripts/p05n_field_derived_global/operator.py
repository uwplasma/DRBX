"""P05N owner action: centered raw-midpoint bracket plus the live U-A face jump.

Field-catalogue-agnostic (see ``rows.py``): callers supply a small ``Role``
per generator/transported field (its physical-field name for live-observation
lookup, its boundary condition, an exact ``evaluate`` callable that doubles as
the Dirichlet trace, and, for Neumann roles, a ``normal_data`` callable).

Import this module as ``from p05n_field_derived_global import operator`` (with
``scripts/`` -- not this package's own directory -- on ``sys.path``): naming
this file ``operator.py`` would otherwise shadow the stdlib ``operator``
module for anything imported after it if this directory were placed first on
``sys.path``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from p07_combined_global.kernels import num
from p05_direct_midpoint_global.direct_operator import point_bracket, direct_pair_actions
from drbx.native.fci_perpendicular_face_corrections import p05_scalar_face_jump

from p05n_field_derived_global.rows import cell_rows, face_common_gradient, side_values


@dataclass
class Role:
    name: str                       # physical field name (key into live_values)
    bc: str                         # 'dirichlet' or 'neumann'
    evaluate: Callable               # (q)->(v,g) exact value+gradient; also the Dirichlet trace
    normal_data: Optional[Callable] = None   # (q)->g_N; required iff bc == 'neumann'

    def __post_init__(self):
        if self.bc not in ("dirichlet", "neumann"):
            raise ValueError(f"unknown boundary condition {self.bc!r}")
        if self.bc == "neumann" and self.normal_data is None:
            raise ValueError(f"Neumann role {self.name!r} requires normal_data")


def live_observations(t, values):
    """Owner averages Σ raw_volume·f(midpoint)/owner_volume, over the full grid.

    ``values`` is (n**3, F): the analytic field(s) evaluated at every raw
    midpoint ``t.pts``. Computed live -- never taken from any reuse bundle.
    """
    values = np.asarray(values, dtype=np.float64)
    sums = np.zeros((len(t.vol), values.shape[1]))
    np.add.at(sums, t.ro, t.rv[:, None] * values)
    return sums / t.vol[:, None]


def owner_raw(t, owner):
    raw_ids = np.flatnonzero(t.ro == int(owner))
    if not len(raw_ids):
        raise ValueError(f"owner {owner} has no raw members")
    points = t.pts[raw_ids]
    keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
    return raw_ids, points, keys


def owner_centered(t, S, ref, owner, generator, transported, live_values,
                    normal_coefficients, patch_cache, context):
    """Centered raw-midpoint bracket, its analytic reference R, and diagnostics.

    Returns a dict with the raw-volume-projected owner action (`centered`),
    the analogously projected exact reference (`R`), the swapped-argument
    antisymmetry defect, the exact-input O==R replay (`exact_input_defect`),
    per-role max condition/constraint-residual over conditioned rows, and the
    constant-field gradient/action bound (populated only when the field name
    is 'constant').
    """
    raw_ids, points, keys = owner_raw(t, owner)
    ga_col = live_values[generator.name]
    gf_col = live_values[transported.name]
    va, grad_a, diag_a = cell_rows(t, S, ga_col, keys, points, bc=generator.bc,
                                    dirichlet_trace=generator.evaluate,
                                    normal_coefficients=normal_coefficients, context=context,
                                    patch_cache=patch_cache, normal_data=generator.normal_data)
    vf, grad_f, diag_f = cell_rows(t, S, gf_col, keys, points, bc=transported.bc,
                                    dirichlet_trace=transported.evaluate,
                                    normal_coefficients=normal_coefficients, context=context,
                                    patch_cache=patch_cache, normal_data=transported.normal_data)
    metric = ref._metric(points)
    h = metric["bcov"] / metric["B"][:, None]
    jac = np.abs(metric["J"])
    volume = t.rv[raw_ids]

    stacked_recon = np.stack((grad_a, grad_f), axis=2)  # (raw,3,2)
    recon_action, antisymmetry = direct_pair_actions(h, jac, stacked_recon, ((0, 1),))
    centered = float(np.dot(volume, recon_action[:, 0]) / t.vol[owner])

    exact_va, exact_ga = generator.evaluate(points)
    exact_vf, exact_gf = transported.evaluate(points)
    # R: the plain analytic point_bracket, projected the same way as the candidate.
    R = float(np.dot(volume, point_bracket(h, jac, exact_ga, exact_gf)) / t.vol[owner])
    # Exact-input O==R: feed the exact gradients through the *candidate's own*
    # batched direct_pair_actions path (not the plain point_bracket used for R)
    # and confirm the two independent code paths agree.
    stacked_exact = np.stack((exact_ga, exact_gf), axis=2)
    exact_input_action, _ = direct_pair_actions(h, jac, stacked_exact, ((0, 1),))
    exact_input_defect = abs(R - float(np.dot(volume, exact_input_action[:, 0]) / t.vol[owner]))

    condition_max = 0.0; residual_max = 0.0; boundary_rows = 0
    for diag in diag_a + diag_f:
        if diag.neumann:
            condition_max = max(condition_max, diag.condition)
            residual_max = max(residual_max, diag.constraint_residual)
        if diag.boundary_conditioned:
            boundary_rows += 1

    return dict(centered=centered, R=R, antisymmetry=antisymmetry,
                exact_input_defect=exact_input_defect,
                condition_max=condition_max, residual_max=residual_max,
                boundary_rows=boundary_rows, raw_count=len(raw_ids),
                grad_a=grad_a, grad_f=grad_f, exact_grad_a=exact_ga, exact_grad_f=exact_gf,
                value_a=va, value_f=vf)


def _face_kind(n, key):
    axis, i = int(key[0]), int(key[1])
    if axis == 0:
        if i == n:
            return "physical_wall"
        if i == n - 1:
            return "radial_n_minus_1"
        return None
    if i >= n - 2:
        return "transverse_last_two_layers"
    return None


def owner_boundary_faces(t, owner):
    """Every true owner-boundary face touching ``owner`` (deduplicated).

    Mirrors scripts/p07_combined_global/topology.py::census's lo/hi
    convention exactly (face ``(axis,i,j,k)`` separates the raw cell with
    axis-``axis`` index ``i-1`` (lo) from the one with index ``i`` (hi); a
    face with lo==hi is a purely internal aggregation seam and is dropped).
    The physical wall exterior (axis 0, i==n) has ``hi=None``.
    """
    n = t.n
    grid = t.ro.reshape(n, n, n)
    raw_ids = np.flatnonzero(t.ro == int(owner))
    ijk = np.array(np.unravel_index(raw_ids, (n, n, n))).T
    seen = {}
    for i, j, k in ijk:
        i, j, k = int(i), int(j), int(k)
        for i_face in (i, i + 1):
            if i_face == 0:
                continue  # collapsed r=0 axis face: never touches a near-wall owner
            lo = int(grid[i_face - 1, j, k]) if i_face - 1 >= 0 else None
            hi = int(grid[i_face, j, k]) if i_face < n else None
            seen[(0, i_face, j, k)] = (lo, hi)
        for j_face in (j, (j + 1) % n):
            lo = int(grid[i, (j_face - 1) % n, k]); hi = int(grid[i, j_face, k])
            seen[(1, i, j_face, k)] = (lo, hi)
        for k_face in (k, (k + 1) % n):
            lo = int(grid[i, j, (k_face - 1) % n]); hi = int(grid[i, j, k_face])
            seen[(2, i, j, k_face)] = (lo, hi)
    result = []
    for key, (lo, hi) in seen.items():
        if lo is not None and hi is not None and lo == hi:
            continue
        if lo == owner or hi == owner:
            result.append((key, lo, hi))
    return result


def owner_jump(t, S, ref, owner, generator, transported, live_values,
               normal_coefficients, patch_cache, context, *, order=3, face_report=None):
    """Live scattered U-A jump for owner, plus the per-face-kind zero-check max."""
    faces = owner_boundary_faces(t, owner)
    total = 0.0
    zero_face_max = {}
    ga_col = live_values[generator.name]
    gf_col = live_values[transported.name]
    for key, lo, hi in faces:
        axis = key[0]
        keys_arr = np.array([key])
        points, weights = num.quadrature(t.faces, keys_arr, order, face=True)
        points = points[0]; weights = weights[0]
        grad_a, diag_common = face_common_gradient(t, S, ga_col, key, points, bc=generator.bc,
                                                     dirichlet_trace=generator.evaluate,
                                                     normal_coefficients=normal_coefficients,
                                                     context=context, patch_cache=patch_cache,
                                                     normal_data=generator.normal_data)
        lv, rv, (lb, rb) = side_values(t, S, gf_col, key, points, bc=transported.bc,
                                        dirichlet_trace=transported.evaluate,
                                        normal_coefficients=normal_coefficients,
                                        context=context, patch_cache=patch_cache,
                                        normal_data=transported.normal_data)
        metric = ref._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        common_gradient = np.zeros((1, len(points), 3, 2))
        common_gradient[0, :, :, 0] = grad_a
        lower_value = np.zeros((1, len(points), 2)); upper_value = lower_value.copy()
        lower_value[0, :, 1] = lv; upper_value[0, :, 1] = rv
        face_jump = float(np.asarray(p05_scalar_face_jump(
            common_gradient, lower_value, upper_value, h[None], weights[None],
            np.array([axis]), np.array([[0, 1]])))[0, 0])
        contribution = face_jump if lo == owner else -face_jump
        total += contribution / t.vol[owner]
        kind = _face_kind(t.n, key)
        if kind is not None:
            zero_face_max[kind] = max(zero_face_max.get(kind, 0.0), abs(face_jump))
        if face_report is not None:
            face_report.append(dict(key=[int(v) for v in key], kind=kind, lo=lo, hi=hi,
                                     jump=face_jump, side_defect=float(np.max(np.abs(rv - lv)))))
    return total, zero_face_max
