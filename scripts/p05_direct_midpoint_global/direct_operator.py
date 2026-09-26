"""Algebra and deterministic regional bookkeeping for the P05 midpoint test."""
from __future__ import annotations

import numpy as np


def point_bracket(h, jacobian, grad_a, grad_b):
    """Evaluate the coordinate-correct strong bracket at raw midpoint points."""
    h = np.asarray(h, dtype=np.float64)
    jacobian = np.abs(np.asarray(jacobian, dtype=np.float64))
    grad_a = np.asarray(grad_a, dtype=np.float64)
    grad_b = np.asarray(grad_b, dtype=np.float64)
    if grad_a.shape != grad_b.shape or grad_a.shape[-1] != 3:
        raise ValueError("both argument gradients must have matching (..., 3) shape")
    if h.shape != grad_a.shape or jacobian.shape != grad_a.shape[:-1]:
        raise ValueError("h/J shapes must match the point-gradient shape")
    if np.any(jacobian <= 0.0) or not np.isfinite(jacobian).all():
        raise ValueError("Jacobian must be finite and nonzero")
    result = -np.einsum("...i,...i->...", np.cross(h, grad_a), grad_b) / jacobian
    return result


def direct_pair_actions(h, jacobian, gradients, pairs):
    """Return all oriented pair actions and the maximum swapped-argument defect."""
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 3 or gradients.shape[1] != 3:
        raise ValueError("gradients must have shape (points, 3, fields)")
    actions = np.empty((gradients.shape[0], len(pairs)), dtype=np.float64)
    antisymmetry = 0.0
    for index, (a, b) in enumerate(pairs):
        ab = point_bracket(h, jacobian, gradients[:, :, a], gradients[:, :, b])
        ba = point_bracket(h, jacobian, gradients[:, :, b], gradients[:, :, a])
        actions[:, index] = ab
        antisymmetry = max(antisymmetry, float(np.max(np.abs(ab + ba), initial=0.0)))
    if not np.isfinite(actions).all():
        raise FloatingPointError("nonfinite direct midpoint bracket action")
    return actions, antisymmetry


def project_raw_to_owners(raw_action, raw_volume, raw_owner, owner_volume, owner_count=None):
    """Project complete raw actions by stored physical raw-volume weights."""
    action = np.asarray(raw_action, dtype=np.float64)
    volume = np.asarray(raw_volume, dtype=np.float64)
    owner = np.asarray(raw_owner, dtype=np.int64)
    denominator = np.asarray(owner_volume, dtype=np.float64)
    if action.ndim != 2 or volume.shape != (len(action),) or owner.shape != (len(action),):
        raise ValueError("raw arrays have incompatible dimensions")
    if np.any(owner < 0) or not np.isfinite(action).all() or not np.isfinite(volume).all():
        raise ValueError("raw actions, volumes, and owners must be valid")
    count = len(denominator) if owner_count is None else int(owner_count)
    if denominator.shape != (count,) or np.any(denominator <= 0.0):
        raise ValueError("owner volumes must be positive and match owner_count")
    if np.any(owner >= count):
        raise ValueError("raw owner index is outside the owner array")
    numerator = np.zeros((count, action.shape[1]), dtype=np.float64)
    for pair in range(action.shape[1]):
        np.add.at(numerator[:, pair], owner, volume * action[:, pair])
    return numerator / denominator[:, None]


def candidate_variants(centered_owner_action, saved_u_minus_a):
    """Return the frozen direct-centered and direct-centered-plus-saved-jump actions."""
    centered = np.asarray(centered_owner_action, dtype=np.float64)
    jump = np.asarray(saved_u_minus_a, dtype=np.float64)
    if centered.shape != jump.shape or centered.ndim != 2:
        raise ValueError("centered owner actions and saved jump must have the same (owners, pairs) shape")
    if not np.isfinite(centered).all() or not np.isfinite(jump).all():
        raise ValueError("candidate actions must be finite")
    return np.stack((centered, centered + jump), axis=2)


def regional_owner_masks(raw_owner, raw_ijk, owner_count, n):
    """Return overlapping named masks using complete owner memberships.

    A one-cell layer at r=0 is the axis core; r=1 is the first ring. Wall is
    i=n-1; the adjacent reconstruction band is n-6..n-2. Transition is the
    pair of radial rings straddling the first fully singleton theta row at a
    fixed eta index. Aggregate and ordinary masks classify the remaining
    owners. Named masks may overlap by design; output records each count.
    """
    raw_owner = np.asarray(raw_owner, dtype=np.int64)
    raw_ijk = np.asarray(raw_ijk, dtype=np.int64)
    if raw_ijk.shape != (len(raw_owner), 3):
        raise ValueError("raw coordinates must have shape (n**3, 3)")
    sizes = np.bincount(raw_owner, minlength=owner_count)
    min_i = np.full(owner_count, n, dtype=np.int64)
    max_i = np.full(owner_count, -1, dtype=np.int64)
    has_axis = np.zeros(owner_count, dtype=bool)
    has_first = np.zeros(owner_count, dtype=bool)
    has_wall = np.zeros(owner_count, dtype=bool)
    has_adjacent = np.zeros(owner_count, dtype=bool)
    for i in range(n):
        owners = raw_owner[raw_ijk[:, 0] == i]
        if len(owners):
            unique = np.unique(owners)
            min_i[unique] = np.minimum(min_i[unique], i)
            max_i[unique] = np.maximum(max_i[unique], i)
            if i == 0: has_axis[unique] = True
            if i == 1: has_first[unique] = True
            if i == n - 1: has_wall[unique] = True
            if n - 6 <= i <= n - 2: has_adjacent[unique] = True
    grid = raw_owner.reshape(n, n, n)
    radial_theta_count = np.array([len(np.unique(grid[i, :, 0])) for i in range(n)])
    full = np.flatnonzero((np.arange(n) >= 2) & (radial_theta_count == n))
    transition_i = int(full[0]) if len(full) else n - 1
    transition = (min_i <= transition_i) & (max_i >= max(0, transition_i - 1))
    masks = {
        "axis_core": has_axis,
        "first_ring": has_first,
        "wall": has_wall,
        "adjacent_reconstruction_band": has_adjacent & ~has_wall,
        "transition": transition & ~has_axis & ~has_first & ~has_wall & ~has_adjacent,
        "aggregate": (sizes > 1) & ~has_axis & ~has_first & ~has_wall & ~has_adjacent,
        "ordinary": (sizes == 1) & ~has_axis & ~has_first & ~has_wall & ~has_adjacent & ~transition,
    }
    return masks, {"first_full_singleton_ring": transition_i,
                   "max_owner_member_count": int(sizes.max(initial=0)),
                   "aggregate_owner_count": int(np.count_nonzero(sizes > 1)),
                   "overlap_counts": {name: int(mask.sum()) for name, mask in masks.items()}}


def weighted_stats(error, owner_volume, mask=None):
    """Physical-volume RMS and maximum absolute error for one region."""
    error = np.asarray(error, dtype=np.float64)
    volume = np.asarray(owner_volume, dtype=np.float64)
    if error.shape[0] != len(volume):
        raise ValueError("owner error and volume lengths differ")
    selected = np.ones(len(volume), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != (len(volume),) or not selected.any():
        return {"owner_count": int(selected.sum()), "rms": None, "maximum": None}
    w = volume[selected]
    e = error[selected]
    return {"owner_count": int(selected.sum()),
            "rms": np.sqrt(np.einsum("o,op,op->p", w, e, e) / w.sum()),
            "maximum": np.max(np.abs(e), axis=0)}


def orders_from_rms(rms_by_resolution, resolutions=(32, 48, 64)):
    """Descriptive refinement orders; zero/tiny denominators stay undefined."""
    rms = np.asarray(rms_by_resolution, dtype=np.float64)
    if rms.shape[0] != len(resolutions):
        raise ValueError("one RMS row is required for each resolution")
    out = []
    for k, (n0, n1) in enumerate(zip(resolutions[:-1], resolutions[1:])):
        a, b = rms[k], rms[k + 1]
        value = np.full_like(a, np.nan)
        valid = (a > 1.0e-14) & (b > 1.0e-14)
        value[valid] = np.log(a[valid] / b[valid]) / np.log(n1 / n0)
        out.append(value)
    return np.stack(out)
